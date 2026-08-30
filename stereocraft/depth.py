"""Monocular depth estimation.

Two backends, and the difference between them is the whole point of this module.

**Depth Anything 3** (the default) returns depth in **metres**.  That is what
makes a physically correct stereo pair possible: knowing that the subject is 3 m
away and the treeline 40 m away, the renderer can work out the parallax a real
pair of eyes would actually see, rather than approximating it from a picture of
what is nearer than what.

**Depth-Anything V2** is kept as a fallback, and returns only relative depth.  It
is mapped onto an assumed range of metres so that one renderer serves both, but
that mapping is a guess and the geometry it produces is approximate.  It is here
because these models do fall over on ordinary photographs -- DA3's own 1.4B
variant was tested and disintegrates on portraits -- and a second opinion costs
little.

Both hand back **inverse** depth, in 1/metres.  Inverse because that is the
quantity that behaves: it varies linearly across a slanted plane where depth
itself does not, so the guided upsample interpolates it correctly, and it is
already what the disparity formula wants.
"""

import os
import sys

import numpy as np
import torch

# --- Depth Anything 3, the metric backend -----------------------------------
DA3_MODEL = "depth-anything/DA3METRIC-LARGE"
# DA3METRIC-LARGE does not return depth in metres directly.  Its own
# documentation gives the conversion as `metres = focal_px * output / 300`, where
# the focal length is in pixels at the resolution the network ran at.  Checked
# against the 1.4B model, which does report metres: 2.36-27.99 m against
# 2.55-34.96 m on the same photo, which is close enough to trust the formula.
DA3_SCALE = 300.0
DA3_PATCH = 14
# DA3 resizes the *longest* side, where Depth-Anything V2 took the shortest.
# Bigger gives cleaner subject silhouettes, which is what the warp cares about
# most; `auto` follows the photo up to here.
DA3_MAX_RES = 2048

# --- Depth-Anything V2, the fallback ----------------------------------------
MODELS = {
    "da2-small": "depth-anything/Depth-Anything-V2-Small-hf",
    "da2-base": "depth-anything/Depth-Anything-V2-Base-hf",
    "da2-large": "depth-anything/Depth-Anything-V2-Large-hf",
}
# What the old names meant, so a script written against them still runs.
ALIASES = {"small": "da2-small", "base": "da2-base", "large": "da2-large"}
# Patch size of the ViT backbone; input dimensions must be a multiple of it.
PATCH = 14
# Depth-Anything is trained at a 518px short side.  Feeding it more resolves finer
# structure, but drifting too far from training makes the overall depth wobble, so
# "auto" follows the photo between the trained size and twice it.
MIN_SIZE, MAX_SIZE = 518, 1036
# The range of metres a Depth-Anything V2 map is pretended to span.  It has no
# scale of its own, so one is invented -- a room-to-far-treeline spread that
# suits most photographs and is wrong for any particular one.
DA2_NEAR, DA2_FAR = 1.0, 50.0

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

# A 28mm-equivalent lens, which is what most phones point at the world, used when
# nothing better is known.  Only the reading of "focus at 3 metres" depends on
# it: see `Depth.focal`.
DEFAULT_FOCAL_35MM = 28.0
FILM_WIDTH_MM = 36.0


def _app_dir():
    """Where the app's own files sit: beside the exe once frozen, else the repo."""
    if getattr(sys, "frozen", False):  # the packaged Windows build
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _use_local_cache():
    """Prefer the checkpoints already sitting next to the app over ~/.cache."""
    if os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME"):
        return
    local = os.path.join(_app_dir(), "hf-cache")
    if os.path.isdir(local):
        os.environ["HF_HUB_CACHE"] = local


def checkpoint(model):
    """Where to load `model` from.

    Weights shipped with the app win.  The portable build puts them in a plain
    `models/large` folder beside the exe, and a folder is something
    `from_pretrained` reads without touching the network at all -- so it starts
    the same whether or not the machine it landed on has any internet.
    """
    bundled = os.path.join(_app_dir(), "models", model)
    if os.path.isdir(bundled):
        return bundled
    _use_local_cache()
    return DA3_MODEL if model == "da3" else MODELS[model]


def _one_image_at_a_time(model):
    """Stop DA3 building a thread pool for every single frame.

    Its input processor farms the per-image work out to a `ThreadPool` of eight
    workers, created and torn down on each call.  For a batch of photographs that
    is sensible; this app hands it exactly one image at a time, so the pool never
    has anything to parallelise and exists only to be built and destroyed.

    **And on Windows, building and destroying it is not free.**  The threads are
    joined properly -- the count does not climb -- but the commit charge does:
    measured at 3.69 MB leaked per call, against 0.00 with the same work done
    sequentially.  Per frame that is nothing.  Over the thirty thousand frames of
    a long clip it is eighty-six gigabytes of memory the process has asked the
    system to promise and will never use, and a machine hits its commit limit
    and starts refusing allocations long before the end.  What that looks like
    from the outside is ffmpeg dying on a frame nine hours in, having encoded
    twenty-three thousand of them perfectly well -- which is what sent this
    conversion, and the search for the cause, a very long way in the wrong
    direction.

    Sequential is also simply the right call for one image: it is the same work
    without eight threads being started to watch one of them do it.
    """
    processor = getattr(model, "input_processor", None)
    if processor is None:  # a DA3 that no longer works this way; nothing to do
        return

    def one_at_a_time(*args, **kwargs):
        kwargs.setdefault("sequential", True)
        return processor(*args, **kwargs)

    model.input_processor = one_at_a_time


def dtype_for(device):
    """fp16 halves the memory traffic on a GPU and is indistinguishable here;
    CPU stays fp32.  `budget` prices the other device with this too, so an
    estimate never disagrees with what would actually run."""
    return torch.float16 if device.type == "cuda" else torch.float32


def pick_device(request="auto"):
    if request != "auto":
        return torch.device(request)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def focal_from_35mm(equivalent_mm, width):
    """Focal length in pixels, from the 35mm-equivalent figure cameras record."""
    return float(equivalent_mm) / FILM_WIDTH_MM * width


def exif_focal(path, width):
    """The photo's own focal length in pixels, or None if it does not say.

    Cameras record the 35mm equivalent, which is exactly the number that turns
    into pixels without knowing the sensor size.  Most photographs that have been
    through a download, a screenshot or an editor have lost it.
    """
    try:
        from PIL import ExifTags, Image

        with Image.open(path) as img:
            exif = img.getexif()
            tags = {**exif, **(exif.get_ifd(0x8769) or {})}
    except Exception:  # a photo that will not give up its EXIF is not an error
        return None
    key = next((k for k, v in ExifTags.TAGS.items() if v == "FocalLengthIn35mmFilm"), None)
    value = tags.get(key)
    return focal_from_35mm(value, width) if value else None


class Depth:
    """Inverse depth in 1/metres, and how the scale behind it was arrived at.

    `focal` is in pixels **at the resolution the network ran at**, and matters
    only for reading the convergence distance as a real distance.  It does not
    change the shape of the depth: a focal length that is wrong by some factor
    scales the whole scene by that factor, which the focus distance then absorbs.
    So the geometry is right either way, and only the metre labels are off.
    """

    def __init__(self, inverse, focal, source, working, metric):
        self.inverse = inverse
        self.focal = focal
        self.source = source
        self.working = working
        self.metric = metric

    def metres(self):
        """The depth map the other way up, for reporting and for sanity checks."""
        return 1.0 / self.inverse.clamp_min(1e-6)


class DepthEstimator:
    """Loads one depth checkpoint and keeps it resident."""

    def __init__(self, model="da3", device="auto"):
        model = ALIASES.get(model, model)
        if model != "da3" and model not in MODELS:
            raise ValueError(f"unknown model {model!r}, pick da3 or one of {sorted(MODELS)}")
        self.name = model
        self.device = pick_device(device)
        self.dtype = dtype_for(self.device)
        self.model = self._load_da3() if model == "da3" else self._load_da2()

    # --- loading ------------------------------------------------------------
    def _load_da3(self):
        import types

        # DA3 narrates every inference to stdout, which would sit in the middle of
        # the progress line a clip is drawing.  Its logger reads this once, at
        # import, and has no other switch.
        os.environ.setdefault("DA3_LOG_LEVEL", "WARN")

        # Two of DA3's api-level imports are replaced before they are reached.
        # Neither is used for a depth map, and both cost more than they are worth:
        #
        #   export     -- glb/ply/gaussian-splat writers, which drag in moviepy,
        #                 cv2 and gsplat.  We write images, not meshes.
        #   pose_align -- aligns camera poses across views, using `evo`.  This
        #                 app is monocular: the metric model returns no
        #                 intrinsics and no extrinsics, so there are no poses to
        #                 align.  `evo` is also GPL-3.0, and stubbing it keeps the
        #                 whole distribution permissively licensed.
        #
        # Do not "fix" either of these by installing the missing packages.
        for name, attribute in (("depth_anything_3.utils.export", "export"),
                                ("depth_anything_3.utils.pose_align", "align_poses_umeyama")):
            if name not in sys.modules:
                stub = types.ModuleType(name)
                setattr(stub, attribute, lambda *args, **kwargs: None)
                sys.modules[name] = stub
        from depth_anything_3.api import DepthAnything3  # heavy; import on demand

        model = DepthAnything3.from_pretrained(checkpoint("da3")).to(self.device).eval()
        _one_image_at_a_time(model)
        return model

    def _load_da2(self):
        from transformers import AutoModelForDepthEstimation  # heavy; import on demand

        model = (
            AutoModelForDepthEstimation.from_pretrained(checkpoint(self.name), dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        mean = torch.tensor(_MEAN, device=self.device).view(3, 1, 1)
        std = torch.tensor(_STD, device=self.device).view(3, 1, 1)
        self._mean, self._std = mean, std
        return model

    # --- sizing -------------------------------------------------------------
    @staticmethod
    def resolve_size(size, short_side):
        """Turn a `--depth-size` request into an actual short-side length."""
        if size in (None, 0, "auto"):
            return int(min(MAX_SIZE, max(MIN_SIZE, short_side)))
        return max(PATCH, int(size))

    def working_size(self, height, width, size="auto"):
        """The resolution the network will actually run at for this shape.

        Shared with `budget`, which prices a conversion before one happens, so
        that what is charged for and what is run are never two different sizes.
        """
        if self.name == "da3":
            longest = max(height, width)
            target = min(DA3_MAX_RES, longest) if size in (None, 0, "auto") else int(size)
            scale = target / longest
        else:
            scale = self.resolve_size(size, min(height, width)) / min(height, width)
        patch = DA3_PATCH if self.name == "da3" else PATCH
        return (max(patch, int(round(height * scale / patch)) * patch),
                max(patch, int(round(width * scale / patch)) * patch))

    # --- inference ----------------------------------------------------------
    @torch.inference_mode()
    def __call__(self, image, size="auto", focal=None):
        """Estimate inverse depth in 1/metres.

        `image` is a uint8 HxWx3 array.  `focal` is the caller's best guess at the
        lens, in pixels at the photo's own width -- from EXIF, usually -- and is
        used only where the model cannot say for itself.
        """
        if self.name == "da3":
            return self._run_da3(image, size, focal)
        return self._run_da2(image, size, focal)

    def _run_da3(self, image, size, focal):
        from PIL import Image as PILImage

        height, width = image.shape[:2]
        target = max(self.working_size(height, width, size))
        prediction = self.model.inference([PILImage.fromarray(image)], process_res=target)

        raw = torch.as_tensor(np.asarray(prediction.depth[0]), device=self.device).float()
        work_h, work_w = raw.shape
        focal_work, source = self._focal(prediction, focal, width, work_w)

        if getattr(prediction, "intrinsics", None) is not None:
            metres = raw  # a model that knows the lens reports metres itself
        else:
            metres = raw * focal_work / DA3_SCALE
        inverse = (1.0 / metres.clamp_min(1e-6))[None, None]
        return Depth(inverse, focal_work, source, (work_h, work_w), metric=True)

    @staticmethod
    def _focal(prediction, focal, width, work_w):
        """The lens, in pixels at the working resolution, and where it came from."""
        intrinsics = getattr(prediction, "intrinsics", None)
        if intrinsics is not None:
            return float(np.asarray(intrinsics[0])[0][0]), "model"
        if focal:
            return float(focal) * work_w / width, "exif"
        return focal_from_35mm(DEFAULT_FOCAL_35MM, work_w), "assumed"

    def _run_da2(self, image, size, focal):
        """The relative backend, stretched onto an invented range of metres.

        Everything downstream works in 1/metres, so the fallback has to speak the
        same language.  It cannot: the map says only what is nearer than what.
        Pinning it across `DA2_NEAR`..`DA2_FAR` is the honest minimum -- the
        scene's shape survives, its scale is fiction.
        """
        from . import stereo

        height, width = image.shape[:2]
        work_h, work_w = self.working_size(height, width, size)

        x = torch.from_numpy(np.ascontiguousarray(image)).to(self.device)
        x = x.permute(2, 0, 1).float().div_(255.0)
        x = torch.nn.functional.interpolate(
            x[None], size=(work_h, work_w), mode="bicubic", align_corners=False, antialias=True
        ).clamp_(0, 1)
        x = (x[0] - self._mean) / self._std

        raw = self.model(pixel_values=x[None].to(self.dtype)).predicted_depth
        raw = raw.reshape(1, 1, *raw.shape[-2:]).float()

        near, far = 1.0 / DA2_NEAR, 1.0 / DA2_FAR
        inverse = far + stereo.normalize(raw) * (near - far)
        focal_work = (float(focal) * work_w / width) if focal else focal_from_35mm(
            DEFAULT_FOCAL_35MM, work_w)
        return Depth(inverse, focal_work, "exif" if focal else "assumed",
                     (work_h, work_w), metric=False)
