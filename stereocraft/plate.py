"""The periphery, built once per shot instead of once per frame.

A VR180 frame is mostly not photograph.  `vr180` records the measurement: a 28mm
phone lens fills 15% of a hemisphere and a 49mm lens 5%, so at 4096 a side the
picture lands under a thousand wide with fifteen megapixels of nothing around it.
`vr180.surround` fills that with a dim, coarse wash, and earns its place by being
a fixed function of the frame -- it moves exactly as the picture moves and so
cannot crawl or boil.  That is the bar anything replacing it has to clear, and it
is the bar a per-frame diffusion fill fails: invent the periphery sixty times a
second and it will be a different periphery sixty times a second.

**So it is invented once per shot rather than once per frame.**  There is then
only one answer for it to differ from, and the boil is gone by construction
rather than by tuning.  What the frame gets is a rigid resample of that one
answer, which is the same guarantee the wash already had.

**And most of it need not be invented at all.**  On a pan or a handheld drift the
periphery of frame 1 is real photograph by frame 200.  Registering the shot into
one plate recovers those pixels having invented nothing whatsoever, which is the
standard `vr180.surround` already holds itself to.  So there are three layers
going outwards -- the picture, then real pixels from elsewhere in the same shot,
then a bounded widening that a model paints -- see `outpaint` -- and the
existing wash beyond that.

**What can go wrong is a shimmy rather than a boil.**  It is the one artefact the
wash could not have: the plate is registered, and a rotation track that jitters
frame to frame slides the periphery against the live picture.  It is as
objectionable as boil and arrives by a different door, so the track is smoothed
before it is used -- see `Track.smoothed`.  The plate is soft enough that
low-passing it costs nothing anyone can see.

**Only what holds still belongs in it.**  A plate is built once for a whole shot,
so anything in it that moves is wrong for every frame but one -- a subject
carried out past the picture's edge sits frozen there while the real one moves,
which is a ghost limb rather than a periphery.  So the near field is held out of
both the mosaic and the registration, by depth, and what is left is the room.

That second use is not a refinement.  On a close-up the subject is most of the
frame and therefore most of the features, so an unmasked fit finds *her* motion
and reports it as the camera's: measured on real footage, 64 degrees of yaw where
the camera had turned 21.  Every frame then lands on the sphere at the wrong
bearing and the plate comes out a smear.  See `Track.add` and `far_field`.

**Every way this fails lands on what the app already did.**  No weights, no
memory, or a shot that will not register: the widening is skipped and the mosaic
stands on its own, or the mosaic collapses to the single frame and `vr180.render`
does exactly what it did before.  Nothing here can make a clip worse than not
asking for it.
"""

import math
import os
import subprocess
import tempfile

import numpy as np
import torch

from . import vr180
from .depth import DEFAULT_FOCAL_35MM, focal_from_35mm

# The TorchScript trace `simple-lama-inpainting` publishes, which is why there is
# no architecture written out below the way `upscale` and `interpolate` both have
# to: `torch.jit.load` opens it with the graph already in it.
#
# **The licence is the one loose end in the tree, and it is deliberate.**  This
# mirror declares Apache-2.0; the authors of the weights it is a trace of
# (`advimman/lama`) publish big-lama under CC BY-NC-SA 4.0, which is the more
# restrictive of the two and the one to assume.  So this is the only piece of the
# distribution that is not permissively licensed -- against the standard the rest
# of it keeps, and the `evo` note at the top of `requirements.txt` in particular.
# It is a narrow exception rather than a change of policy: the app runs without
# these weights and simply stops the surround at what the clip itself saw, which
# is the same way it behaves without the super-resolution ones.  Anything built
# to be handed out rather than kept should leave `lama` out of `-Models`.
REPO = "okaris/simple-lama"
FILENAME = "big-lama.pt"
REVISION = "d5706085cdbdd5eb72503fdcd9fa648e952cfa53"
SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"
# The trace was built with a stride-8 pyramid inside it and will not take a size
# that is not a multiple of one.
ALIGN = 8

# The plate is a whole 360 by 180 sphere rather than the output hemisphere, so a
# clip that pans right round needs no special case -- content simply leaves one
# edge and arrives at the other.
PLATE_SPAN_AZ = 360.0
PLATE_SPAN_EL = 180.0
# And it is stored well below the render size.  At 2048 across 360 degrees a
# 65-degree phone frame lands on about 370 pixels, a third of a 1080 source --
# which sounds careless until you remember what it is for: this is peripheral,
# dimmed, and outside the fovea, and the alternative is a median stack four times
# the size.  The painted part is upsampled about four times on the way to a 4096
# frame and is meant to read as soft; the mosaic beside it is real photograph and
# is the part that would show the loss, which is why this is a named number
# rather than a buried one.
PLATE_WIDTH = 2048
PLATE_HEIGHT = 1024
# Frames decoded per shot to build the mosaic from.  The cost of another one is
# a projection and 6 MB of stack, and the gain falls away quickly: what more
# samples buy is a better median, and 24 is well past where a median stops
# changing its mind.
SAMPLES = 24
# Width the clip is decoded at for this pass.  Enough to supersample slightly
# into the plate rather than resample out of it, and small enough that the
# registration below is nearly free.
DECODE_WIDTH = 640
# Matches that have to agree before a fit is believed.  Chaining frame to frame
# drifts and matching to the shot's reference frame does not, so the reference is
# tried first every time and the chain is only for once a pan has taken the
# overlap away -- and this is the number that decides which has happened.  See
# `Track.add`.
MIN_INLIERS = 25
# How much of a shot has to change before it is called a cut.  The same relative
# test `video.TemporalDepth` makes on depth maps, made here on colour instead,
# because that one runs inside the render loop and this pass is over long before
# the render loop starts.
CUT = 0.35
# Frames either side used to smooth the rotation track.  Camera motion is smooth
# and estimation noise is not, so this is nearly free of real motion and takes
# most of the jitter -- see the module docstring on shimmy.
SMOOTH_RADIUS = 3
# One frame in this many is registered, and the rest are interpolated between.
# Registering wants a depth map per frame to know what the room is, and a depth
# map per frame of a ninety-second clip is minutes of model time for a track a
# camera could not possibly have wandered off in a fifteenth of a second.
# Camera motion is smooth; estimation noise is what needed the smoothing, and
# sampling is a cheaper way to the same place.
REGISTER_EVERY = 15
# Where the room stops and the subject starts, as a quantile of the frame's own
# depth.  Relative because metric depth has no fixed idea of "close": a bedroom
# close-up in the test footage runs 0.27 to 0.60 metres end to end, and any
# absolute threshold would take all of it or none.
FAR_QUANTILE = 0.5
# And below this much movement the camera is called still and the last bearing
# is kept exactly.  Smoothing alone does not finish the job: a locked-off tripod
# shot still measured a residual eleven thousandths of a degree of wander, which
# is a thirtieth of a pixel and yet enough to move a hard edge by a few levels
# every frame -- the shimmy this whole arrangement exists to avoid, arriving in
# the one case where there was no motion to track in the first place.  Held
# against the last bearing *emitted* rather than the last one measured, so a slow
# genuine pan is followed with at most this much lag rather than being quietly
# ignored.  A fortieth of a degree is a quarter of a pixel at the largest frame
# the app will ever write.
STILL_DEG = 0.025


def _reason(problem):
    """The short of why a pass was skipped, for `pipeline.notice`."""
    return str(problem) or problem.__class__.__name__


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def intrinsics(width, height, focal_px):
    """The pinhole matrix, with the principal point in the middle where every
    other part of this app already assumes it is."""
    return np.array([[focal_px, 0.0, (width - 1) / 2.0],
                     [0.0, focal_px, (height - 1) / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_from(homography, matrix):
    """The camera rotation a homography implies, given the lens.

    For a camera that only turns, `x_b = K R K^-1 x_a`, so peeling the lens off
    both ends leaves the rotation.  It will not be quite orthonormal -- the
    homography was fitted to noisy matches and knows nothing about being a
    rotation -- so it is put back on the manifold with an SVD, which is the
    closest rotation to it in the least-squares sense.

    Returns None if what came back is not a rotation at all, which is what a fit
    on a moving subject rather than a moving camera produces.
    """
    if homography is None:
        return None
    inner = np.linalg.inv(matrix) @ homography @ matrix
    u, _, vt = np.linalg.svd(inner)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:  # a reflection, which no camera ever performs
        u[:, -1] *= -1
        rotation = u @ vt
    # A genuine pan leaves the singular values near one another; a fit that
    # locked onto something moving through the frame does not, and the rotation
    # squeezed out of it would swing the whole periphery.
    scales = np.linalg.svd(inner, compute_uv=False)
    if scales[0] / max(scales[-1], 1e-9) > 1.5:
        return None
    return rotation


class Track:
    """Where the camera was pointing for each frame of one shot.

    Every frame is matched against the shot's reference frame first, which has no
    drift in it at all, and falls back to chaining off the previous frame only
    once a pan has taken away the overlap the direct match needed.  A frame that
    matches neither keeps the last rotation rather than jumping to identity: a
    single motion-blurred frame is a gap to be bridged, not a reason to swing the
    periphery back to where the shot started.
    """

    def __init__(self, matrix):
        import cv2

        self.matrix = matrix
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.reference = None  # (keypoints, descriptors) of the first frame
        self.previous = None  # and of the one before this
        self.previous_rotation = np.eye(3)
        self.rotations = []
        self.registered = 0  # frames placed by a real match rather than by holding

    def _features(self, gray):
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return (keypoints, descriptors) if descriptors is not None else None

    def _match(self, source, target):
        """The homography taking `source`'s pixels to `target`'s, or None."""
        import cv2

        if source is None or target is None:
            return None
        pairs = self.matcher.match(source[1], target[1])
        if len(pairs) < MIN_INLIERS:
            return None
        src = np.float32([source[0][p.queryIdx].pt for p in pairs]).reshape(-1, 1, 2)
        dst = np.float32([target[0][p.trainIdx].pt for p in pairs]).reshape(-1, 1, 2)
        homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if inliers is None or int(inliers.sum()) < MIN_INLIERS:
            return None
        return homography

    def add(self, gray, keep=None):
        """Place one frame, and return the rotation taking its own directions
        into the reference frame's.

        `keep` marks the pixels worth matching on -- the far field, in practice.
        It is not an optimisation.  On a close-up the subject is most of the
        frame and therefore most of the features, so RANSAC finds *her* motion
        and calls it the camera's: measured on a real clip, 64 degrees of yaw
        where the camera had actually turned 21.  Everything downstream is laid
        on the sphere at those bearings, so the plate comes out a smear.
        """
        if keep is not None:
            gray = np.where(keep, gray, 0).astype(gray.dtype)
        features = self._features(gray)
        if self.reference is None:
            self.reference = features
            self.previous = features
            self.rotations.append(np.eye(3))
            self.registered += 1
            return self.rotations[-1]

        # Straight to the reference where there is still enough of it in view.
        rotation = rotation_from(self._match(features, self.reference), self.matrix)
        if rotation is None and self.previous is not None:
            # And through the previous frame where there is not, which is a pan
            # that has gone past its own starting point.
            step = rotation_from(self._match(features, self.previous), self.matrix)
            if step is not None:
                rotation = self.previous_rotation @ step
        if rotation is None:
            rotation = self.previous_rotation  # hold, rather than snap back
        else:
            self.registered += 1

        self.previous = features or self.previous
        self.previous_rotation = rotation
        self.rotations.append(rotation)
        return rotation

    def smoothed(self):
        """The track with the estimation noise taken out of it.

        Smoothed as rotation vectors rather than as matrices, because the average
        of two rotation matrices is not a rotation and the average of two
        rotation vectors is very nearly one over the angles a hand-held camera
        covers between frames.
        """
        import cv2

        if len(self.rotations) < 2:
            return list(self.rotations)
        vectors = np.array([cv2.Rodrigues(r)[0].ravel() for r in self.rotations])
        out = np.empty_like(vectors)
        for index in range(len(vectors)):
            lo = max(0, index - SMOOTH_RADIUS)
            hi = min(len(vectors), index + SMOOTH_RADIUS + 1)
            out[index] = vectors[lo:hi].mean(axis=0)
        return _held(cv2.Rodrigues(v)[0] for v in out)


def far_field(estimator, frame, quantile=FAR_QUANTILE):
    """Which pixels of a frame are the room rather than what is in front of it.

    A plate holds still for a whole shot, so only what holds still belongs in it.
    On the footage this was built for the split is clean -- the subject fills the
    near half of the depth histogram and the bed, headboard and wall the far half
    -- and it is the same split that keeps the registration honest.
    """
    import torch.nn.functional as F

    height, width = frame.shape[:2]
    depth = estimator(frame, 518, None)
    inverse = depth.inverse.squeeze()
    metres = 1.0 / inverse.clamp_min(1e-6)
    metres = F.interpolate(metres[None, None].float(), size=(height, width),
                           mode="bilinear", align_corners=False)[0, 0]
    return (metres >= float(metres.flatten().quantile(quantile))).cpu()


def _interpolate(rotations, at, total):
    """A rotation for every frame, from rotations for some of them.

    As rotation vectors, which average sensibly over the fraction of a degree a
    camera covers between two samples; as matrices they would not.
    """
    import cv2

    if not rotations:
        return [np.eye(3)] * total
    vectors = [cv2.Rodrigues(r)[0].ravel() for r in rotations]
    out = []
    for index in range(total):
        if index <= at[0]:
            out.append(rotations[0])
            continue
        if index >= at[-1]:
            out.append(rotations[-1])
            continue
        j = int(np.searchsorted(at, index))
        lo, hi = at[j - 1], at[j]
        t = (index - lo) / max(hi - lo, 1)
        out.append(cv2.Rodrigues(vectors[j - 1] * (1 - t) + vectors[j] * t)[0])
    return out


def _held(rotations, still_deg=STILL_DEG):
    """A track with movement too small to see taken out of it entirely.

    Not a smoothing: the previous bearing is *kept*, bit for bit, so a shot that
    did not move produces a periphery that does not move either -- identical
    frame to frame rather than merely close.  That is worth having as an exact
    property rather than a tight tolerance, because it is the one the tests can
    assert and the one a viewer would notice the loss of.
    """
    import cv2

    out, held = [], None
    for rotation in rotations:
        if held is None or math.degrees(np.linalg.norm(
                cv2.Rodrigues(held.T @ rotation)[0])) > still_deg:
            held = rotation
        out.append(held)
    return out


def is_cut(previous, current, threshold=CUT):
    """Whether the picture changed wholesale between two frames.

    Relative, like `video.TemporalDepth`'s test and for the same reason: an
    absolute threshold on brightness would call every cut in a dark scene a
    continuation and every flicker in a bright one a cut.
    """
    if previous is None:
        return False
    scale = float(np.abs(current).mean()) + 1e-9
    return float(np.abs(current.astype(np.float32) - previous.astype(np.float32)).mean()) / scale > threshold


# ---------------------------------------------------------------------------
# The mosaic
# ---------------------------------------------------------------------------


def plate_patch(width=PLATE_WIDTH, height=PLATE_HEIGHT):
    """The whole sphere, as a `vr180.Patch` so the projection code already knows
    how to read it."""
    return vr180.Patch(PLATE_SPAN_AZ, PLATE_SPAN_EL, width, height)


def mosaic(samples, focal_px, spot, band_rows=64):
    """Median-combine projected frames into one plate.

    **Median rather than mean, and that is the whole of it.**  A running mean
    smears anyone who walked through the shot into a ghost stretched across the
    periphery, and a ghost in the corner of your eye is exactly what the eye is
    built to notice.  A median over a couple of dozen samples throws the walker
    away and keeps the wall behind them.

    `samples` is an iterable of `(frame [3, H, W] in [0, 1], rotation)`, or of
    `(frame, rotation, keep)` where `keep` marks the pixels that belong in a
    plate at all -- the far field.  A plate is built once for a whole shot, so
    anything in it that moves will be wrong for every frame but one; holding the
    near field out is what keeps it to the part of the scene that stays put.

    The rotation takes that frame's own directions into the plate's -- the same
    rotation takes that frame's own directions into the plate's -- the same way
    round as everywhere else here, and transposed below because the projection
    asks the opposite question.  It walks the *plate's* pixels and needs to know
    where each one falls in the frame, which is the inverse of knowing where the
    frame's pixels fall on the plate.

    Done in bands over the plate's rows, because the stack is the expensive part
    and nothing needs all of it at once.
    """
    colours, masks = [], []
    for sample in samples:
        frame, rotation = sample[0], sample[1]
        keep = sample[2] if len(sample) > 2 else None
        turned = torch.as_tensor(rotation, dtype=torch.float32).T
        # Colour and "is this the room" go through the projection together, so
        # the second lands on exactly the pixels the first did.
        stacked = frame if keep is None else torch.cat([frame, keep[None].float()])
        projected, mask = vr180.project(stacked, focal_px, spot, rotation=turned)
        if keep is not None:
            mask = mask & (projected[3] > 0.5)
            projected = projected[:3]
        colours.append((projected.clamp(0, 1) * 255).to(torch.uint8))
        masks.append(mask)
    if not colours:
        return (torch.zeros(3, spot.height, spot.width),
                torch.zeros(spot.height, spot.width, dtype=torch.bool))

    stack_colour = torch.stack(colours)  # [n, 3, H, W] uint8
    stack_mask = torch.stack(masks)  # [n, H, W]
    out = torch.zeros(3, spot.height, spot.width)
    covered = stack_mask.any(0)
    for top in range(0, spot.height, band_rows):
        bottom = min(top + band_rows, spot.height)
        band = stack_colour[:, :, top:bottom].float().div_(255.0)
        # Anything that frame never saw is put beyond the median's reach rather
        # than counted as black, which is what a plain median would do with it.
        band[~stack_mask[:, None, top:bottom].expand_as(band)] = float("nan")
        out[:, top:bottom] = torch.nanmedian(band, dim=0).values.nan_to_num_(0.0)
    return out, covered


# ---------------------------------------------------------------------------
# The weights for the small backend, which `outpaint` loads through here
# ---------------------------------------------------------------------------


def checkpoint():
    """Where to load the weights from: beside the app if they were shipped with
    it, else the cache, else Hugging Face -- the same order `upscale.checkpoint`
    uses, and for the same reason."""
    from .depth import _app_dir, _use_local_cache

    bundled = os.path.join(_app_dir(), "models", "lama", FILENAME)
    if os.path.isfile(bundled):
        return bundled
    _use_local_cache()
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO, FILENAME, revision=REVISION)


def verify(path):
    """The weights come from a mirror rather than from the authors, so they are
    checked before anything is built out of them."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise RuntimeError(f"{path} is not the checkpoint this was written against "
                           f"(sha256 {digest.hexdigest()})")
    return path


def load(device="cpu", path=None):
    """Big-LaMa, as a TorchScript trace: the small backend, and the last one.

    It was the first choice here and is no longer, because it was measured and
    found wanting on exactly the footage this pass exists for.  Asked to widen an
    indoor close-up it continues the bed line and the headboard as *flat
    horizontal bands* -- a facet ratio of 22.7 where a photograph sits near 9 and
    where FLUX comes back at 8.5.  No amount of work on the blending moves that;
    it is what a texture-continuation model does when the thing to continue has
    structure rather than texture.

    It stays because it is 200 MB against 13 GB, needs no prompt and no
    quantisation, and runs on a machine with no card at all.  A periphery in
    bands beats a periphery in the dark, and this is what a build without the
    large weights falls back to.  What it is *good* at is unchanged: it is
    deterministic, so what it paints cannot differ between two runs of the same
    shot.

    On what the weights may be used for, see the note beside `REPO`.
    """
    model = torch.jit.load(verify(path or checkpoint()), map_location="cpu")
    return model.eval().to(device)


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


class Plates:
    """The finished plates, and which frame reads which.

    `at` hands back the plate already turned to the frame's own bearing and
    resampled to the render's patch, which is the only thing the renderer wants
    of it -- at the brightness it was filmed at.  How brightly to *show* it is
    `vr180`'s to decide, along with matching it to the frame in front of it,
    both being questions about the finished picture rather than about the plate.
    """

    def __init__(self, shots, rotations, spot):
        # One (colour, covered) per shot, and one (shot, rotation) per frame.
        # Colour is uint8: a clip with fifty shots in it holds fifty plates, and
        # at float that is a gigabyte of periphery waiting to be looked at.
        self.shots = [((c.clamp(0, 1) * 255).to(torch.uint8) if c.is_floating_point() else c, m)
                      for c, m in shots]
        self.rotations = rotations
        self.spot = spot
        self.device = None  # where the render wants them; see `to`

    def __len__(self):
        return len(self.rotations)

    def _ready(self, shot):
        """The shot's plate as float, converted once rather than once a frame.

        A plate is 25 MB in float and the same shot is asked for hundreds of
        frames running, so this holds the one in play and lets the rest stay in
        the uint8 they are stored as.
        """
        if getattr(self, "_current", None) != shot:
            stored, covered = self.shots[shot]
            device = getattr(self, "device", None) or stored.device
            self._current = shot
            self._float = (stored.to(device).float().div_(255.0), covered.to(device))
        return self._float

    def reach(self, spot):
        """How much of the render's patch the plate covers, as a mask, for
        whoever reports what the conversion managed.  Taken at the first frame:
        it moves with the camera, and one number has to stand for the clip."""
        got = self.at(0, spot)
        if got is None:
            return torch.zeros(spot.height, spot.width, dtype=torch.bool)
        return got[1]

    def to(self, device):
        """Say where the render is happening.  Noted rather than acted on: the
        store stays where it is and `_ready` brings across the one shot in play.

        A plate is 6 MB stored, which sounds like nothing until a clip with a
        hundred cuts in it puts 630 MB of periphery on a card that is also
        holding the depth model and a 4096-square render.  Only ever one of them
        is being looked at, so only ever one of them needs to be there.
        """
        self.device = torch.device(device)
        self._current = None  # the cached copy is on the device we have just left
        return self

    def at(self, index, spot):
        """The plate as this frame sees it: `(colour [3, h, w], mask [h, w])` on
        `spot`'s grid, or None where there is nothing to show.

        The rotation goes the way it does because the render always puts the live
        picture dead centre of the patch, so the patch's own axis *is* where the
        camera is pointing this frame.  What the plate must be asked is therefore
        "what was at this bearing, in the frame the plate was built in", which is
        the frame's rotation applied to the patch's directions.
        """
        if not self.rotations:
            return None
        shot, rotation = self.rotations[min(index, len(self.rotations) - 1)]
        colour, covered = self._ready(shot)
        device = colour.device
        turned = torch.as_tensor(rotation, dtype=torch.float32)

        out = torch.zeros(3, spot.height, spot.width, device=device)
        mask = torch.zeros(spot.height, spot.width, dtype=torch.bool, device=device)
        band = max(1, 8_000_000 // max(spot.width, 1))
        for top in range(0, spot.height, band):
            rows = min(band, spot.height - top)
            x, y, z = vr180.directions(top, rows, spot, device, torch.float32)
            x, y, z = vr180.turn(x, y, z, turned)
            grid = self._lookup(x, y, z)
            out[:, top:top + rows] = torch.nn.functional.grid_sample(
                colour[None], grid, mode="bilinear", padding_mode="border",
                align_corners=False)[0]
            seen = torch.nn.functional.grid_sample(
                covered.float()[None, None], grid, mode="bilinear", padding_mode="border",
                align_corners=False)[0, 0]
            # Only where the sample was wholly inside the real plate, so the
            # bilinear tap cannot drag a half-black edge pixel into view.
            mask[top:top + rows] = seen > 0.999
        return out, mask

    def _lookup(self, x, y, z):
        """Directions to the plate's own normalised coordinates.

        The plate is a whole sphere, so azimuth wraps and nothing has to be
        marked invalid -- which is the reason it is stored as one.
        """
        az = torch.atan2(x, z)
        el = torch.asin(y.clamp(-1.0, 1.0))
        gx = az / math.pi  # [-pi, pi] over a 360-degree plate
        gy = -el / (math.pi / 2.0)
        return torch.stack((gx, gy), dim=-1)[None]


def _decoder(src, width, height, stderr):
    from .video import NO_CONSOLE, _tool

    args = [_tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(src),
            "-vf", f"scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=stderr, bufsize=0, **NO_CONSOLE)


def decode_size(clip, width=DECODE_WIDTH):
    """The size this pass reads the clip at, never larger than the clip itself."""
    from .video import _even

    if clip.width <= width:
        return _even(clip.width), _even(clip.height)
    return _even(width), _even(round(width * clip.height / clip.width))


def wanted(clip, settings):
    """Whether this pass has anything to do for these settings."""
    return bool(getattr(settings, "outpaint", False)) and settings.projection == "vr180"


class _Sampler:
    """At most `2 * SAMPLES` frames of one shot, spread evenly over it, in memory
    that does not grow with the shot.

    A shot's length is not known until it ends, and a two-minute one held whole at
    this size is gigabytes.  So frames are taken at a stride that doubles whenever
    the store fills, dropping every other one as it goes -- which leaves what is
    kept evenly spread over everything seen so far, whatever that turns out to be,
    without ever having to know in advance how much there was going to be.
    """

    def __init__(self, most=2 * SAMPLES):
        self.most = max(2, most)
        self.stride = 1
        self.seen = 0
        self.kept = []

    def offer(self, array, room=None):
        position, self.seen = self.seen, self.seen + 1
        if position % self.stride:
            return
        self.kept.append((position, array.copy(), room))
        if len(self.kept) > self.most:
            self.kept = self.kept[::2]
            self.stride *= 2

    def frames(self):
        step = max(1, len(self.kept) // SAMPLES)
        return self.kept[::step][:SAMPLES]


def build(src, clip, settings, on_progress=None, device=None, estimator=None):
    """Read a clip and return its `Plates`, or None if there is nothing to give.

    One decode and no encode: what comes out is a handful of images rather than a
    video.  `estimator` is the depth model the conversion has already loaded --
    borrowed rather than loaded again, and used to tell the room from whatever is
    moving about in front of it.  Without one the whole frame is treated as room,
    which is what a clip with nothing in the foreground actually looks like and
    what a machine short of memory has to settle for.

    Each shot's plate is built the moment its last frame goes by, so what is held
    is one shot's worth of frames and a stack of finished plates -- not the clip.
    """
    from .pipeline import notice
    from .video import _read_exactly, _stop, _log

    if not wanted(clip, settings):
        return None

    import cv2

    width, height = decode_size(clip)
    focal_px = focal_from_35mm(DEFAULT_FOCAL_35MM, width)
    spot = plate_patch()
    matrix = intrinsics(width, height, focal_px)
    reach = float(getattr(settings, "outpaint_reach", None) or 0) or None

    from . import outpaint

    # Chosen now so the user hears about it before the wait, but *loaded* long
    # after -- see the two phases below.
    picked = outpaint.choose(getattr(settings, "outpaint_model", "auto"), device=device)
    if picked is None:
        # No painter, no pass.  The mosaic alone was worth having when this was
        # designed and is not worth having now: on the close-up footage it is
        # for, the subject fills the frame and never leaves it, so what the
        # median accumulates is a smear of her rather than the room behind her.
        # The plain wash is better than that, and is what `vr180.render` does
        # with no plate at all.
        notice(settings, "no weights for the painted surround, so the surround is the "
                         "blurred spread of the picture as usual")
        return None

    frame_bytes = width * height * 3
    buffer = bytearray(frame_bytes)
    view = memoryview(buffer)
    array = np.frombuffer(buffer, np.uint8).reshape(height, width, 3)

    mosaics, turns, shot_of = [], [], []
    kept, track = _Sampler(), Track(matrix)
    at, seen_in_shot = [], 0
    previous_small = None
    index = 0

    def close(kept, track, at, seen):
        """Finish a shot as far as it can be finished without a model: combine
        what was really there, and let go of the frames."""
        steady = _held(_interpolate(track.smoothed(), at, seen) if at else [])
        # A track that placed most of its frames is worth mosaicking from; one
        # that did not was measuring something other than the camera.
        trusted = track.registered >= 0.8 * max(len(at), 1)
        mosaics.append(_stow(_combine(kept, steady, focal_px, spot, trusted)))
        turns.extend(steady if steady else [np.eye(3)] * seen)

    try:
        with tempfile.TemporaryFile() as log:
            decoder = _decoder(src, width, height, log)
            try:
                while _read_exactly(decoder.stdout, view) == frame_bytes:
                    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
                    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
                    if is_cut(previous_small, small):
                        close(kept, track, at, seen_in_shot)
                        kept, track = _Sampler(), Track(matrix)
                        at, seen_in_shot = [], 0
                    previous_small = small

                    room = None
                    if seen_in_shot % REGISTER_EVERY == 0:
                        if estimator is not None:
                            try:
                                room = far_field(estimator, array)
                            except Exception:
                                room = None  # a frame the model would not take
                        track.add(gray, keep=None if room is None else room.numpy())
                        at.append(seen_in_shot)
                    kept.offer(array, room)

                    shot_of.append(len(mosaics))
                    seen_in_shot += 1
                    index += 1
                    if on_progress and on_progress("looking around", index, clip.frames) is False:
                        _stop(decoder)
                        return False
            except BaseException:
                _stop(decoder)
                raise
            decoder.stdout.close()
            if decoder.wait() and not index:
                raise RuntimeError(f"could not read {src}: {_log(log)}")
        if not index:
            return None
        close(kept, track, at, seen_in_shot)
    finally:
        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()

    # **Phase two, and the reason it is a phase.**  The painter is thirteen
    # gigabytes; the depth model that told the room from the subject is two, and
    # it is the conversion's, not this pass's, so it stays loaded either way.
    # Held at the same time inside a WSL VM given half the host's memory, the two
    # of them plus a clip being decoded took the *virtual machine* out rather
    # than the process -- which looks like the distro crashing and says nothing
    # about why.  So nothing is loaded until every frame has been read and every
    # mosaic combined, and what is carried between the phases is one finished
    # plate per shot at 6 MB rather than the frames that made it.
    if picked is None:
        return Plates([(m[2], m[3]) for m in mosaics], list(zip(shot_of, turns)), spot)

    painter = model = None
    try:
        painter, model = outpaint.load(picked, device=device)
    except Exception as problem:
        notice(settings, f"filling the surround from the clip itself, without the "
                         f"painted part: {_reason(problem)}")
    built = []
    try:
        # Drained from the front rather than comprehended, and each result put
        # back to bytes as it arrives.  A comprehension holds the whole of
        # `mosaics` and the whole of what it builds at the same time, which is
        # the tallest this pass ever stands -- and it stands there on a machine
        # that has just been asked to load a thirteen-gigabyte painter.
        for index in range(len(mosaics)):
            parts, mosaics[index] = mosaics[index], None
            colour, covered = _widen_shot(_thaw(parts), painter, settings,
                                          (width, height), focal_px, reach)
            built.append((_u8(colour), covered))
            del parts, colour, covered
    finally:
        del painter, model
        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()

    return Plates(built, list(zip(shot_of, turns)), spot)


def _u8(image):
    """A 0-1 colour plane as bytes, or a mask left exactly as it is."""
    if not image.is_floating_point():
        return image
    return (image.clamp(0, 1) * 255).round().to(torch.uint8)


def _stow(parts):
    """One shot's plate as bytes, for the wait between the two phases.

    `_combine` works in float and `build` then holds what it made until the last
    frame of the clip has been read -- which on a long film is hundreds of shots,
    and at float32 a plate is 57 MB of them.  That is the pass's real cost and it
    grows with the length of the film rather than the size of a frame: a hundred
    shots is 5.7 GB held, before phase two builds a second list beside it.

    As bytes the same plate is 19 MB, and nothing is lost.  Both colour planes
    are medians of 8-bit frames, and `Plates` stores the finished plate as uint8
    regardless -- so this is not a new conversion, only the existing one done at
    the point where it saves something instead of after.
    """
    reference, seen, colour, covered, room = parts
    return _u8(reference), seen, _u8(colour), covered, room


def _thaw(parts):
    """Back to float for the painter, one shot at a time."""
    reference, seen, colour, covered, room = parts
    return (reference.float().div_(255.0), seen,
            colour.float().div_(255.0), covered, room)


def _combine(kept, turns, focal_px, spot, trusted=True):
    """One shot's real pixels, three ways.

    Returns `(reference, seen, mosaic, covered, room)`.

    **The reference is one clean frame, and that is deliberate.**  The obvious
    thing to hand the model is the mosaic -- it is the widest real thing there
    is -- and on a close-up it is the wrong thing.  A mosaic is two dozen frames
    median-combined at whatever bearings the track believed, and when most of
    the frame is a moving subject the track is only ever partly right, so what
    comes out is a smear of skin tones with the room somewhere underneath it.
    Asked to continue *that*, a model produces architecture: panels, frames,
    shapes that belong to no room.  Handed one sharp frame instead it produces
    the room, and the same model on the same footage does one or the other
    depending only on which it was shown.

    So the mosaic is not thrown away -- it is demoted.  It contributes real
    pixels where the camera genuinely turned to see something new, and nothing
    to what the model is asked to carry on from.
    """
    empty = (torch.zeros(3, spot.height, spot.width),
             torch.zeros(spot.height, spot.width, dtype=torch.bool))
    frames = kept.frames()
    if not frames or not turns:
        return empty[0], empty[1], empty[0], empty[1], empty[1]

    samples, room_samples = [], []
    for entry in frames:
        position, array = entry[0], entry[1]
        near = entry[2] if len(entry) > 2 else None
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
        turn = turns[min(position, len(turns) - 1)]
        samples.append((tensor, turn))
        room_samples.append((tensor, turn) if near is None else (tensor, turn, near))

    # The one nearest the shot's own bearing, which is the plate's axis and so
    # the middle of any canvas taken off it.
    reference, seen = mosaic(samples[:1], focal_px, spot)
    # And *its* far field, not the union of everything's.  Taking the room mask
    # from the whole stack was a bug worth naming: only the frames the tracker
    # registers get a depth map, the sampler keeps a different set, and a frame
    # kept without one counts as room from edge to edge.  The union of two dozen
    # such frames is the entire picture, so the pass that was supposed to see
    # only the room saw all of it -- and duly painted the subject a second time,
    # out where she could never move again.  One frame, one depth map, one mask.
    _, room = mosaic(room_samples[:1], focal_px, spot)
    if not trusted or len(samples) < 2:
        return reference, seen, reference, seen, room

    colour, covered = mosaic(samples, focal_px, spot)
    return reference, seen, colour, covered, room


def _widen_shot(parts, painter, settings, source_size, focal_px, reach):
    """The other half: hand the model one clean frame, and put the real pixels
    back over whatever it paints underneath them."""
    from . import outpaint
    from .pipeline import notice

    reference, seen, colour, covered, room = parts
    if painter is None or not bool(seen.any()):
        return colour, covered
    try:
        painted, reached = outpaint.widen(reference, seen, room, source_size, focal_px,
                                          painter, reach_deg=reach or outpaint.REACH_DEG,
                                          on_note=lambda m: notice(settings, m))
        # Whatever the camera really turned to see outranks whatever was painted
        # in its place.  Inside the reference the two are the same picture.
        extra = covered & ~seen
        return torch.where(extra[None], colour, painted), reached | covered
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        notice(settings, "not enough memory to paint the surround, so it stops at what "
                         "the camera saw")
    except Exception as problem:
        notice(settings, f"could not paint the surround, so it stops at what the camera "
                         f"saw: {_reason(problem)}")
    return colour, covered
