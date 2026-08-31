"""The whole pipeline: photo in, side-by-side 3D photo out."""

import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pillow_heif  # teaches Pillow to read the HEIC files iPhones produce
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from . import budget, logbook, spherical, stereo, vr180
from .depth import DEFAULT_FOCAL_35MM, DepthEstimator, exif_focal, focal_from_35mm

pillow_heif.register_heif_opener()

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}

# Every name the app writes.  Nothing inside a JPEG or an mp4 announces how it is
# meant to be looked at, so players go by the file name -- which makes these load
# -bearing rather than decorative, and makes the exact spelling matter far more
# than it looks.
#
# Skybox's published rules are a list of tokens and nothing else.  The angle is
# "360" or "180x180"; **a bare "180" is not on the list**.  The layout is "sbs",
# "lr" or "3dh" for a pair, "hsbs" or "half sbs" for one squeezed to half width,
# and "full sbs" for one that kept it -- and where the layout is not spelled out,
# a still is assumed to be half width.  Separators and case do not matter, so
# "full_sbs" and "Full SBS" are the same token.
#
# The old "_180_sbs" therefore said layout and no angle at all, and both things
# that went wrong in a headset follow from that one fact: a clip with no angle is
# shown on a cinema screen rather than wrapped onto the sphere, and a photo is
# shown on that screen at half width, which stretches each eye across twice the
# width it belongs in.  Hence "180x180" for the angle, said the way the rules say
# it, and the width said out loud rather than guessed at.
#
# Every name still ends in one of the four below, which is what `cli.collect`
# reads to avoid converting its own output all over again when it is pointed at a
# folder twice -- "_180x180_full_sbs" ends in "_sbs", and so does a file an
# earlier version left behind as "_180_sbs".
SBS_TAGS = ("_sbs", "_sbs_cross", "_hsbs", "_hsbs_cross")


def full_width(settings):
    """Whether each eye keeps its own full width in the finished frame.

    A photo always does.  A flat clip is squeezed to half width unless `--full`
    says otherwise, that being the size players and their hardware decoders
    expect.  A vr180 clip never is: there the frame size comes from the
    projection rather than from the source, so `video.geometry` gives every eye
    the whole square and `full_width` has nothing left to decide.
    """
    return settings.projection == "vr180" or getattr(settings, "full_width", True)


def tag(settings):
    """What to call the output, which is the only way a player will know what it
    is looking at.  A cross-eyed pair is right|left, and shown to a headset as
    though it were left|right it puts each eye on the other one's view -- that
    last token is for the human reading the folder, no player having a word for
    it.
    """
    parts = []
    if settings.projection == "vr180":
        parts.append("180x180")
    parts.append("full_sbs" if full_width(settings) else "hsbs")
    if settings.cross_eyed:
        parts.append("cross")
    return "_" + "_".join(parts)


@dataclass
class Settings:
    model: str = "da3"
    depth_size: object = "auto"
    # How big the viewer is, and what they are looking at.  "auto" sizes both to
    # the scene, which is what a stereographer does and what literal human eyes
    # measurably do not: 65mm on a close-up asks for five times more separation
    # than anyone can fuse, and on a telephoto shot asks for almost none.  A
    # number here is taken literally -- 65.0 for real eyes, larger for a landscape.
    # See `stereo.auto_geometry`.
    eyes_mm: object = "auto"
    focus_m: object = "auto"
    # What "auto" aims for: near-to-far separation as a percentage of frame width.
    target_pct: float = 2.0
    # A ceiling on parallax, as a percentage of frame width.  Metric depth is
    # honest about how close things get, and the geometry will cheerfully ask for
    # more separation than anyone can fuse.
    limit_pct: float = 3.0
    # The same two questions, asked in degrees of arc for the vr180 path.
    #
    # They are a separate pair rather than a reuse of the two above because a
    # percentage of frame width does not survive the move to a sphere: the frame
    # is 180 degrees, so 2% of it is 3.6 degrees of parallax, several times what
    # an eye can fuse.  Feeding `target_pct` to the spherical renderer would
    # therefore not be a smaller effect or a larger one, it would be a different
    # question answered with the wrong units -- so `target_pct` and `limit_pct`
    # are left meaning exactly what they mean on a plane, and these carry the
    # spherical case.  See `vr180.TARGET_DEG`, which is where the defaults live
    # and which explains why 0.6 is the one number in that file with no
    # measurement behind it.
    target_deg: float = vr180.TARGET_DEG
    limit_deg: float = vr180.LIMIT_DEG
    cross_eyed: bool = False
    max_size: int = 0
    quality: int = 95
    fmt: str = "auto"
    save_depth: bool = False
    device: str = "auto"
    # "flat" writes the rectilinear pair that every 3D-TV and VR player will show
    # on a virtual screen.  "vr180" wraps the same geometry onto a hemisphere at
    # its true angular scale instead, which is more immersive where the photo
    # reaches and simply dark where it does not -- see `vr180`.
    projection: str = "flat"
    # Stored width per eye for the vr180 projection, or auto to keep as much of
    # the source's own detail as `vr180_cap` has room for -- see `vr180`.
    vr180_size: object = "auto"
    vr180_cap: object = vr180.MAX_SIZE
    # How brightly to fill the part of the sphere the picture never reached with
    # a blurred spread of the picture, 0 for not at all -- what a social video
    # site puts behind a clip that does not fill the frame.  A brightness rather
    # than a switch because the right amount is a matter of taste and of the
    # headset: past about half, the wash starts competing with the picture for
    # the eye.  0 by default, this being the one thing here that is not
    # measured.  See `vr180.surround`.
    vr180_surround: float = 0.0
    # Called with anything the user should be told that is not the result --
    # a pass skipped for want of memory, a fallback taken, a quality traded.
    # Left unset it goes to stderr, which suits the command line and reaches
    # nobody at all in the window: the frozen app is built with no console, so
    # a warning printed there is a warning discarded.  See `notice`.
    on_notice: object = None
    # Called when a photo will not fit, with the `TooBig` describing it, and
    # expected to return "resize" or "skip".  Left unset nothing is ever
    # silently downscaled: the photo is skipped and the caller told why.
    on_oversize: object = None


@dataclass
class VideoSettings(Settings):
    """The same settings, carrying the defaults a moving picture wants instead.

    `target_pct` is lower, which comes to the same thing as a narrower baseline:
    any error in the depth map becomes a horizontal position error in proportion
    to the separation, and less separation shrinks it.  A still shows that error
    as a silhouette a pixel out of place, which nobody sees; a clip shows it as
    an edge that shimmers, which everybody does.  The gaps the warp opens up
    scale with it too, and a filled gap that reads as plausible while it holds
    still crawls once the edge it belongs to moves.  A narrower baseline is what
    stepping back from the scene would do -- the geometry stays self-consistent,
    it is simply a smaller pair of eyes.

    `depth_size` is pinned rather than following the frame, because feeding the
    network more pixels buys fine per-frame structure that the temporal smoothing
    then averages away.

    `focus_m` is deliberately not changed: at 3m almost the whole scene sits
    behind the window with only a near subject in front of it, and that is the
    arrangement that stays comfortable.  Things poking out of the screen are what
    break at the frame edge, and motion makes that worse rather than better.
    """

    target_pct: float = 1.3
    depth_size: object = 1400
    # Enhance the source before converting it, where it is short of what the
    # frame can hold -- see `prepass`.  Both only act when they are needed: a
    # clip already at the ceiling is not upscaled and one already at 60fps is
    # not interpolated, so leaving them on costs nothing on footage that has
    # nothing to gain.
    upscale: bool = False
    # How much of the upscaler's own picture to keep, against a plain
    # enlargement of the source: 1.0 for all of it, 0 for an honest resample and
    # nothing invented, None for the model's own default.  The knob for footage
    # that comes back looking painted rather than filmed -- see `upscale.DETAIL`.
    upscale_detail: object = None
    interpolate: bool = False
    # Fill the part of the sphere the lens never reached with the scene the clip
    # was actually shot in, rather than with dark or a blurred wash -- gathered
    # once per shot, which is what keeps it from boiling.  See `plate`.  vr180
    # only: a flat pair is photograph edge to edge and has no periphery to fill.
    outpaint: bool = False
    # How far past the source's own field of view to widen it, in degrees.  A
    # knob rather than a constant because the trade is a matter of taste: more
    # reach fills more of the sphere and leaves the model less to go on.  See
    # `outpaint`.
    outpaint_reach: float = 40.0
    # Which model does the widening, or "auto" for the best one whose weights
    # are to hand -- see `outpaint.PREFERRED`.
    outpaint_model: str = "auto"
    # None rather than a number, because what a clip can afford depends on how
    # fast it runs and a class attribute cannot see the clip.  `video.geometry`
    # asks `vr180.video_cap` once the frame rate is known; a number here is
    # taken as given instead.
    vr180_cap: object = None
    # Squeeze each eye to half width, so the clip comes out the size it went in.
    # That is what players and headsets expect, and what their hardware decoders
    # can actually keep up with -- full width doubles it, and 4K doubled is past
    # what most of them will take.
    full_width: bool = False
    # How much of the previous frame's depth to keep, 0 for none.
    temporal: float = 0.5
    crf: int = 18
    # "auto" writes h264 where a headset will decode it and hevc where it will
    # not -- see `video.codec_for`.  A name here is taken at face value.
    codec: str = "auto"
    audio: bool = True


class TooBig(Exception):
    """A photo that will not fit in memory, and the resize that would fit.

    `target` is the size to convert at instead, or None when no size would help.
    """

    def __init__(self, path, source, target, device, wanted, free, limit=Settings.limit_pct,
                 alternative=None):
        self.path = Path(path)
        self.source = source
        self.target = target
        self.device = device
        self.wanted = wanted
        self.free = free
        self.limit = limit
        self.alternative = alternative
        super().__init__(self.describe())

    def _sbs_width(self, width):
        """Width of the finished pair: two eyes, less the margin `make_pair`
        trims off each end where one eye has no pixels to show.

        Priced at the comfort clamp, which is the most that trim can ever be, so
        the figure quoted is the smallest the output could come out at.
        """
        return 2 * (width - 2 * stereo.max_margin(width, self.limit))

    @staticmethod
    def _mp(size):
        return f"{size[0]}x{size[1]} ({size[0] * size[1] / 1e6:.1f} MP)"

    def describe(self):
        where = "video memory" if self.device.type == "cuda" else "memory"
        lines = [
            f"{self.path.name} is {self._mp(self.source)} and needs about "
            f"{self.wanted / 1e9:.1f} GB, but only {self.free / 1e9:.1f} GB of "
            f"{where} is free."
        ]
        if self.target is None:
            lines.append("No smaller size would fit either: the depth model needs that much"
                         " whatever size the photo is.")
            lines.append(f"The {self.alternative} depth model would fit, and costs little in"
                         f" quality (--model {self.alternative})." if self.alternative
                         else "Free some memory and try again.")
        else:
            keep = 100 * (self.target[0] * self.target[1]) / (self.source[0] * self.source[1])
            lines.append(
                f"Resizing to {self._mp(self.target)} would fit -- "
                f"{self.target[0] / self.source[0]:.0%} of the width, {keep:.0f}% of the pixels."
            )
            lines.append(
                f"The side-by-side image would come out about "
                f"{self._sbs_width(self.target[0])}x{self.target[1]} instead of "
                f"{self._sbs_width(self.source[0])}x{self.source[1]}."
            )
        return "\n".join(lines)


def vr180_frame(settings, width, height, focal=None):
    """Pixels in the finished frame, for a vr180 conversion that has not started.

    The memory check runs before anything is decoded, so the lens here is
    whatever EXIF or the default says rather than what the depth model will
    report later.  Being a little out costs a little accuracy in the estimate;
    not asking at all costs the estimate entirely, a hemisphere being up to
    sixty times the area of the clip that was spread across it.
    """
    if settings.projection != "vr180":
        return None
    spot = vr180.patch(focal or focal_from_35mm(DEFAULT_FOCAL_35MM, width), width, height,
                       settings.vr180_cap or vr180.MAX_SIZE,
                       None if settings.vr180_size in (None, 0, "auto")
                       else int(settings.vr180_size))
    return 2 * spot.width * spot.height


def notice(settings, message):
    """Tell the user something that is not the answer.

    This exists because the alternative had already gone wrong: "not enough
    memory to add detail" went to stderr, the window is frozen with no console,
    and so a conversion quietly did less than it was asked to and said nothing.
    A warning nobody can see is not a warning.
    """
    # Into the log as well as onto the screen.  A notice is the app saying it
    # did less than it was asked to, which is exactly the thing someone reads
    # the log to find out afterwards -- and the dialog it went to is long gone
    # by then.
    logbook.note("notice", said=repr(message))
    handler = getattr(settings, "on_notice", None)
    if handler is None:
        print(message, file=sys.stderr)
    else:
        handler(message)


def photo_size(path):
    """The photo's dimensions without decoding it, EXIF orientation included."""
    with Image.open(path) as img:
        width, height = img.size
        if img.getexif().get(274, 1) in (5, 6, 7, 8):  # 274 is Orientation
            width, height = height, width
        return width, height


def load_image(path, size=None):
    """Read a photo, honouring the EXIF orientation cameras write.

    `size` resizes on the way in.  JPEGs take the shortcut of decoding straight
    to a smaller raster, so an oversized photo never has to exist full size.
    """
    with Image.open(path) as img:
        if size is not None:
            # draft() sees the raster as stored, so a photo the EXIF turns on
            # its side wants the target the same way round.
            sideways = img.getexif().get(274, 1) in (5, 6, 7, 8)
            img.draft("RGB", size[::-1] if sideways else size)  # JPEG only; a no-op elsewhere
            img = ImageOps.exif_transpose(img).convert("RGB")
            return np.array(img.resize(size, Image.LANCZOS) if img.size != size else img)
        return np.array(ImageOps.exif_transpose(img).convert("RGB"))


def output_path(src, dst, fmt, name="_full_sbs"):
    src = Path(src)
    ext = src.suffix.lower() if fmt == "auto" else f".{fmt}"
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    if dst is None:
        return src.with_name(f"{src.stem}{name}{ext}")
    dst = Path(dst)
    if dst.is_dir() or dst.suffix == "":
        return dst / f"{src.stem}{name}{ext}"
    return dst


def save_image(array, path, quality=95, xmp=None):
    """`xmp` is written only into a JPEG, which is the format that reliably
    carries it and the one a VR photo viewer expects to be handed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(array)
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        extra = {"xmp": xmp} if xmp else {}
        img.save(path, quality=quality, subsampling=0, optimize=True, **extra)
    elif suffix == ".png":
        img.save(path, compress_level=6)
    else:
        img.save(path)
    return path


def save_depth_map(inverse, path):
    """Write the depth map as a 16-bit PNG: near white, far black.

    This used to store centimetres, on the reasoning that real units beat a
    pretty picture.  They do, but only if you can see them -- a scene two to
    twenty-five metres away came out between 195 and 2475 of a possible 65535,
    which is a black rectangle.  Nobody could tell a working depth map from a
    broken one.

    So it is scaled across its own range to be looked at, and the two distances
    that scaling used go in the PNG's own metadata, which keeps it measurable:

        metres = far - (value / 65535) * (far - near)
    """
    # The pipeline hands over a plain [H, W]; tolerate the leading dimensions a
    # caller might keep, since a wrong shape here writes a file rather than
    # raising, and a wrong file is harder to notice.
    while inverse.dim() > 2:
        inverse = inverse[0]
    metres = 1.0 / inverse.clamp_min(1e-6)
    near, far = float(metres.min()), float(metres.max())
    if far - near < 1e-6:  # a scene all at one distance has nothing to shade
        scaled = torch.zeros_like(metres)
    else:
        scaled = (far - metres) / (far - near)
    gray = (scaled.clamp(0, 1) * 65535).round().to(torch.int32).cpu().numpy().astype(np.uint16)

    from PIL.PngImagePlugin import PngInfo

    meta = PngInfo()
    meta.add_text("stereocraft:near_m", f"{near:.6f}")
    meta.add_text("stereocraft:far_m", f"{far:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray, mode="I;16").save(path, pnginfo=meta)
    return path


def _area(size):
    """Pixels in a proposed size, and less than nothing for "no size at all",
    so that any real offer beats having none."""
    return size[0] * size[1] if size else -1


# Every way a photo can fail to fit, so that one `except` covers the lot.
OUT_OF_MEMORY = (torch.cuda.OutOfMemoryError, MemoryError, TooBig)


def _to_uint8(tensor):
    return (tensor.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


class Converter:
    """Holds the depth model between conversions so batches pay for it once."""

    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self._depth = None
        # Eye separation and screen plane the last frame was rendered with.  Worth
        # keeping because `auto` picks them per scene, and a result that is too
        # strong or too flat is hard to correct without knowing where it started.
        self.chose = (None, None)
        # Fraction of the last vr180 frame that came from the photograph rather
        # than from nowhere.  None after a flat render, which is all coverage.
        self.covered = None
        # The piece of sphere it was put on, which the metadata is written from.
        self.spot = None
        # Where the lens came from: "model", "exif" or "assumed".  Worth keeping
        # because it means far more to a sphere than to a plane -- see `render`.
        self.lens = None

    @property
    def depth_model(self):
        if self._depth is None or self._depth.name != self.settings.model:
            self._depth = DepthEstimator(self.settings.model, self.settings.device)
        return self._depth

    def let_go(self):
        """Put down everything on the card that can be picked up again.

        For a pause, which is someone asking for their machine back.  A paused
        run is idle, so what it is holding is memory rather than work, and
        almost all of that can be given up and taken again in a couple of
        seconds.

        Measured on a 4080 SUPER, paused mid-render on a 1080p clip with the
        metric model: 5784 MB reserved before, 142 MB after, of which 67 MB is
        the frames in flight.  Three gigabytes of that was the allocator's own
        cache and 1.3 GB the weights.  Paused between the chunks of an upscale
        pass it is starker -- 13848 MB reserved against 47 MB actually live, all
        the rest being cache the pass will ask the pool for again; that one
        settles at 1722 MB rather than 142 because the little that is live is
        scattered through segments the pool cannot then hand back whole.

        There is nothing else worth chasing.  A plate keeps its store off the
        card by design and brings across one shot at a time, so the surround is
        25 MB whether the clip has three cuts in it or a hundred -- see
        `plate.to`, which is where that decision was already made.

        The collection is not housekeeping.  The estimator sits in a reference
        cycle, and measured without this the weights were still on the card
        after `empty_cache` and went back on nobody's timetable.

        What cannot be given back is the CUDA context -- 291 MB on the same
        machine -- which goes when the process does and not before.

        Nothing needs putting back afterwards: `depth_model` builds again for
        the frame that next asks for one, from the same file and to the same
        weights.  Paused and resumed this way, a render and an upscale pass both
        wrote byte for byte the file they wrote running straight through.  And if something else
        has taken the card in the meantime, the rebuild raises where the render
        already catches it and the clip finishes slowly on the CPU -- which is a
        pause costing time, rather than a pause costing the work.
        """
        cuda = torch.cuda.is_available()
        held = torch.cuda.memory_reserved() if cuda else 0
        self._depth = None
        gc.collect()
        if not cuda:
            return 0
        torch.cuda.empty_cache()
        return held - torch.cuda.memory_reserved()

    def convert(self, src, dst=None):
        """Convert one photo.

        A photo that will not fit in video memory is retried on the CPU.  One
        that will not fit there either is put to `Settings.on_oversize`, which
        answers "resize" to convert it smaller or "skip" to leave it alone;
        skipping returns None rather than raising, since it is a choice and not
        a failure.
        """
        try:
            return self._attempt(src, dst)
        except TooBig as oversize:
            # The handler hears about it either way, including the hopeless
            # case, so that a skip is never a silent one.
            if self._decide(oversize) != "resize" or oversize.target is None:
                return None
            return self._attempt(src, dst, oversize.target)

    def _decide(self, oversize):
        ask = self.settings.on_oversize
        # Without someone to ask, nothing is quietly downscaled behind the
        # caller's back -- the photo is skipped and they are told about it.
        if ask is None:
            notice(self.settings, oversize.describe())
            return "skip"
        return ask(oversize)

    def _attempt(self, src, dst, size=None):
        """Run once, falling back from the GPU to the CPU if room runs out.

        Every way of running out -- the pre-flight check, CUDA's own error, a
        failed host allocation -- leaves by the same door, as a `TooBig` for
        whichever device was the last to turn the photo away.
        """
        try:
            return self._run(src, dst, size)
        except OUT_OF_MEMORY as problem:
            gpu = self.depth_model.device
            if gpu.type == "cpu":
                raise self._too_big(problem, src, size, [gpu]) from None
            notice(self.settings, f"{Path(src).name}: too big for GPU memory, trying the CPU")
            requested, self.settings.device, self._depth = self.settings.device, "cpu", None
            torch.cuda.empty_cache()
            try:
                return self._run(src, dst, size)
            except OUT_OF_MEMORY as fallback_problem:
                oversize = self._too_big(fallback_problem, src, size, [gpu, self.depth_model.device])
            finally:  # leave the next photo free to try the GPU again
                self.settings.device, self._depth = requested, None
            raise oversize from None

    def _too_big(self, problem, src, size, devices):
        """Turn a memory failure into a proposal, sized for what is free now."""
        source = problem.source if isinstance(problem, TooBig) else (size or photo_size(src))
        return self._proposal(src, source, devices)

    def _proposal(self, src, source, devices):
        """The best offer among the devices still in play.

        Priced against the model that is really loaded, so a small model on a
        modest card is judged by what it costs rather than by what the large one
        would.  Both devices are asked, because the roomier of the two is not
        always the GPU -- a generous card in a busy machine can have more to
        spare than the system does -- and the better offer is the one to make.
        """
        estimator, size = self.depth_model, self.settings.depth_size
        frame = vr180_frame(self.settings, *source)
        offers = [(budget.plan(estimator, *source, size, device, frame), device)
                  for device in devices]
        target, device = max(offers, key=lambda offer: _area(offer[0]))
        return TooBig(src, source, target, device,
                      budget.needs(estimator, *source, size, device, frame),
                      budget.free_bytes(device) or 0, self.settings.limit_pct,
                      None if target else budget.smaller_model(estimator, *source, size, device))

    def geometry(self, inverse, width, focal, target=None):
        """Eye separation in mm and screen plane in metres, for this scene.

        Either can be pinned to a number, in which case it is taken at face value
        and the other is still chosen to suit.  `target` overrides what auto aims
        for, which is how the vr180 path asks in degrees of arc for what the flat
        one asks in percent of frame width -- the same logic, different units.
        """
        cfg = self.settings
        eyes, focus = stereo.auto_geometry(inverse, width, focal, target or cfg.target_pct)
        if cfg.eyes_mm != "auto":
            eyes = float(cfg.eyes_mm)
        if cfg.focus_m != "auto":
            focus = float(cfg.focus_m)
        return eyes, focus

    def render(self, image, normalizer=None, margin=None, focal=None, spot=None, plate=None):
        """One frame in; the two eye views and the depth map behind them out.

        Composing is left to the caller because a video squeezes each eye to half
        width first, and doing that after they are glued together would blend the
        two across the seam.

        `normalizer` replaces the per-image percentile stretch with something
        that remembers the frames before this one, and `margin` pins the edge
        trim so a clip cannot change size part-way through.  A still passes
        neither and gets exactly what it always did.

        `spot` pins the piece of sphere a vr180 frame is put on, for the same
        reason `margin` exists: left to work itself out from the lens it could
        come out a different size from one frame to the next, and no encoder
        will take that.

        `plate` is this frame's view of the scene around it, from `plate.Plates`,
        and is passed straight through to `vr180.render`.
        """
        cfg = self.settings
        estimator = self.depth_model
        width = image.shape[1]

        depth = estimator(image, cfg.depth_size, focal)
        self.lens = depth.source
        rgb = (torch.from_numpy(np.ascontiguousarray(image)).to(estimator.device)
               .permute(2, 0, 1).float().div_(255.0))
        guide = rgb.mean(0)[None, None]

        # Inverse depth is what gets interpolated, here and in the smoothing:
        # 1/Z is linear across a slanted surface where Z itself is not.
        small = depth.inverse if normalizer is None else normalizer(depth.inverse)
        # Held to the range it came from.  A guided filter is free to overshoot at
        # an edge, and an inverse depth that overshoots towards zero is a pixel
        # claiming to be kilometres away -- which the disparity then takes
        # literally.
        inverse = stereo.guided_upsample(small, guide)[0, 0].clamp_(
            float(small.min()), float(small.max()))

        # The lens, restated in pixels at the width actually being rendered.
        focal_full = depth.focal * width / depth.working[1]
        if cfg.projection == "vr180":
            return self._render_vr180(rgb, inverse, focal_full, spot, plate)

        eyes, focus = self.geometry(inverse, width, focal_full)
        half = stereo.half_disparity(inverse, focal_full, eyes, focus, cfg.limit_pct, width)
        left, right = stereo.make_pair(rgb, half, margin)
        self.chose = (eyes, focus)  # what `auto` settled on, for whoever reports it
        self.covered = None  # a flat frame is photograph edge to edge
        return left, right, inverse

    def _render_vr180(self, rgb, inverse, focal_full, spot=None, plate=None):
        """The same frame, on a hemisphere instead of a plane.

        The depth map is not re-estimated here.  It was measured on the
        photograph, which is the only shape the network understands, and is
        warped across with the colour -- see `vr180` for why that ordering is
        not negotiable.

        **The lens matters here in a way it does not on a plane.**  `Depth` notes
        that a focal length wrong by some factor only rescales the scene, which
        the focus distance then absorbs -- true when the picture stays flat, and
        false the moment it goes on a sphere.  The focal length is what decides
        where each pixel lands in angle, so getting it wrong puts the whole
        picture at the wrong apparent size: right-looking, and not life-sized.
        EXIF supplies it where a photo still has any, and `depth.source` says so
        when it does not, because a guessed angular scale is worth admitting to.
        """
        cfg = self.settings
        if spot is None:
            spot = vr180.patch(focal_full, rgb.shape[2], rgb.shape[1],
                               cfg.vr180_cap or vr180.MAX_SIZE,
                               None if cfg.vr180_size in (None, 0, "auto")
                               else int(cfg.vr180_size))
        # Asked against the projection rather than the photo: a baseline is a
        # distance in millimetres either way, but what counts as enough of one is
        # set by how far the picture is stretched to get onto the sphere.
        eyes, focus = self.geometry(inverse, spot.width, spot.per_radian,
                                    vr180.auto_target(spot, cfg.target_deg))
        left, right, mask = vr180.render(rgb, inverse, focal_full, eyes, focus, spot,
                                         limit_deg=cfg.limit_deg,
                                         wash=cfg.vr180_surround, plate=plate)
        self.chose = (eyes, focus)
        self.covered = vr180.coverage(mask, spot)
        self.spot = spot
        return left, right, inverse

    def _run(self, src, dst, size=None):
        cfg = self.settings
        started = time.perf_counter()
        estimator = self.depth_model
        device = estimator.device

        # Asking the file how big it is costs nothing, so a photo that cannot
        # fit is turned away before it is ever decoded.  That matters most on
        # the CPU path, where the alternative is the kernel's OOM killer taking
        # the process with it rather than an exception anyone can catch.
        if size is None:
            source = photo_size(src)
            frame = vr180_frame(cfg, *source, exif_focal(src, source[0]))
            if not budget.fits(estimator, *source, cfg.depth_size, frame=frame):
                raise self._proposal(src, source, [device])

        image = load_image(src, size)
        height, width = image.shape[:2]

        # The photo's own lens, where it still remembers it.  Only the reading of
        # the focus distance depends on this, not the shape of the depth -- see
        # `depth.Depth`.
        left, right, inverse = self.render(image, focal=exif_focal(src, width))
        sbs = stereo.compose(left, right, cfg.cross_eyed)

        if cfg.max_size and sbs.shape[2] > cfg.max_size:
            scale = cfg.max_size / sbs.shape[2]
            size = (max(1, round(sbs.shape[1] * scale)), cfg.max_size)
            sbs = F.interpolate(sbs[None], size=size, mode="bilinear", align_corners=False, antialias=True)[0]

        # Where on the sphere the picture sits.  A hemisphere is half of one, so
        # there is something to say even when the frame is the whole square.
        marks = spherical.gpano(self.spot) if cfg.projection == "vr180" and self.spot else None
        out = save_image(_to_uint8(sbs), output_path(src, dst, cfg.fmt, tag(cfg)),
                         cfg.quality, marks)
        if cfg.save_depth:
            save_depth_map(inverse, out.with_name(f"{Path(src).stem}_depth.png"))

        return {
            "input": Path(src),
            "output": out,
            "source_size": (width, height),
            "resized_from": photo_size(src) if size else None,
            "output_size": (sbs.shape[2], sbs.shape[1]),
            "eyes_mm": self.chose[0],
            "focus_m": self.chose[1],
            "coverage": self.covered,
            "lens": self.lens,
            "patch": self.spot,
            "seconds": time.perf_counter() - started,
        }


def convert(src, dst=None, **kwargs):
    """One-shot helper for scripts: `stereocraft.convert("photo.jpg", eyes_mm=70)`."""
    return Converter(Settings(**kwargs)).convert(src, dst)
