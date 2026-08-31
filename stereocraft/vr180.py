"""VR180: the same geometry, wrapped onto a sphere instead of laid on a plane.

A flat side-by-side pair is shown in a headset as a screen hanging in space, and
the player is free to make that screen whatever size it likes.  VR180 is the
other arrangement: the picture is mapped onto the inside of a hemisphere at a
fixed angular scale, so something that subtended thirty degrees to the camera
subtends thirty degrees to the viewer.  That is the whole gain, and it is a real
one -- but a photograph has no periphery to put in the rest of the sphere.

**Most of the frame is dark, and it stays that way.**  A 28mm phone lens fills
15% of a hemisphere and a 49mm lens 5%, so at 4096 pixels a side the picture
comes back under a thousand wide with fifteen megapixels of black around it.
Storing only the piece the lens reached was built, measured, and taken out
again: it saved four fifths of the pixels and recorded where the piece belonged,
in the fields both formats provide for exactly that -- and no player read them.
Skybox assumes every eye is a full 180 by 180, and a 65-by-91-degree patch handed
to that assumption comes out 2.7 times too close and 40% stretched sideways,
which is sharp, convincing and wrong.  So the square is what gets written, and
`Patch` still carries its spans because 180 out of 360 is itself a crop and the
projection bounds have to say so.

**Where the depth comes from.**  The depth model runs on the photograph, before
any of this.  It is trained on ordinary perspective images and an equirectangular
one is not that -- straight lines are not straight in it, and the network has
never seen a picture where they were not.  So the colour and the inverse depth
are estimated flat and warped across together, and the stereo is rendered
afterwards, in equirectangular space.

**Why the stereo is not simply the flat renderer pointed at a warped picture.**
One fixed pair of eyes cannot see a whole hemisphere.  Turn to look ninety
degrees left and eyes that were side by side are now one behind the other, with
no separation left to give.  The standard answer is omnidirectional stereo, where
every column is taken from a different point on a circle the size of the eye
separation, so the pair is always across the line of sight.  It is not a
physically consistent single viewpoint and it cannot be -- but it is what every
VR180 camera records and every player expects, and the disparity it asks for is

    dtheta = B * (1/Z - 1/Zc)

radians, which is `stereo.half_disparity` exactly, with the projection's
pixels-per-radian standing in where the focal length used to be.  The one
addition is a cosine taper towards the poles, where the separation has to fall
to nothing: looking straight up there is no "across the line of sight" left to
put two eyes on, and insisting on some anyway is what makes badly made VR180
hurt to look at.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import stereo

# The container is fixed by the format: 180 degrees each way.
FOV = 180.0
# Near-to-far angular separation `auto` aims for, in degrees.  The flat
# renderer aims at a percentage of frame width, and that number does not survive
# the move: 2% of a 180-degree frame is 3.6 degrees of parallax, several times
# what anyone can fuse.  This projection thinks in angles, so it asks in angles.
TARGET_DEG = 0.6
# The ceiling on separation, for the same reason `Settings.limit_pct` exists and
# in the units that mean something here.  Beyond about a degree the two images
# stop fusing and start doubling.
LIMIT_DEG = 1.2
# How wide the fade at the edge of the real picture is.  In degrees rather than
# pixels, because it is a fact about the sphere and not about the file.
FALLOFF_DEG = 4.0
# How coarse the optional surround is, and how bright.  Coarse enough to read as
# light rather than as a second blurry photograph, and dim because a bright
# periphery is tiring to sit inside and competes with the thing you are meant to
# be looking at -- at full strength the wash measures brighter than the picture
# it came from, the picture having dark parts and the average of it not.
SURROUND_BLUR_DEG = 12.0
SURROUND_DIM = 0.45
# How far the wash stays at the picture's own brightness before easing to
# `SURROUND_DIM`.
#
# Without this there is a cliff exactly where the eye is drawn: the picture ends
# at full strength and the wash begins at 45% of it, which measured as a step of
# 75 of 255 across the boundary.  Easing from one to the other over 25 degrees
# takes that to 53, and what is left is inherent -- a wash is a blur, so it
# cannot match a sharp pillow-against-headboard edge locally however bright it
# is.  Widening `FALLOFF_DEG` instead was tried and does not help: it fades the
# photograph rather than lifting the wash, so the step stays and real picture is
# lost.
SURROUND_RAMP_DEG = 25.0
# How brightly the plate is shown, when there is one.  Full strength, which took
# a measurement to arrive at and is worth writing down: at 0.7 the step across
# the edge of the photograph measures 30% and reads as a seam, at 0.85 it is 15%,
# and at 1.0 it is 0.4% -- invisible.
#
# The reason it can be 1.0 where `SURROUND_DIM` is 0.45 is that the two are not
# the same kind of thing.  A wash is a blur of the picture, and a blur measures
# brighter than what it came from, the picture having dark parts and the average
# of it not; it has to be pulled down to stop it glowing.  A plate is the scene
# itself, at the exposure `_match_gain` has just matched to this very frame, dark
# parts and all -- so dimming it is not restraint, it is a vignette drawn round
# real footage, and it lands exactly where the eye is already looking.  What the
# viewer sees past the plate is still the wash, still governed by
# `Settings.vr180_surround`, and still dim.
PLATE_DIM = 1.0
# And how wide the fade is where the plate runs out.  Wider than the picture's
# own edge: that one is a join between photograph and nothing, where this is a
# join between what was filmed and what was inferred from it, and the eye goes
# straight to a hard circle where a painted ring stops.
PLATE_FALLOFF_DEG = 10.0
# Ceiling on the stored width per eye for a still, which nothing has to decode
# in real time.
MAX_SIZE = 4096
# And for a clip, which does.  A hardware decoder is limited by pixels per
# second rather than by pixels, so what a clip can afford depends on how fast it
# runs: a Quest 3 will take 8192x4096 at 60fps, 6688x3344 at 90 and 5792x2896 at
# 120, and each of those is two eyes side by side.  Half of each, then, and the
# frame rate decides which.  Sitting at 2048 regardless spent half the decoder
# on nothing: 4096 per eye is 22.8 pixels per degree against the headset's own
# 25, where 2048 was 11.4 -- less than half what the display could have shown.
VIDEO_CAPS = ((60.0, 4096), (90.0, 3344), (120.0, 2896))
# Output pixels per band, so peak memory stops tracking the whole projection --
# the same trick `stereo._render_eye` plays, for the same reason.
_BAND_PIXELS = 8_000_000


def even(n):
    """Encoders refuse odd dimensions, and a width that is even stays even when
    it is doubled into a pair."""
    return max(2, int(n) - int(n) % 2)


def fov(focal_px, extent):
    """The angle a lens covers across `extent` pixels, in degrees.  Works for
    either axis, the two differing only in how many pixels there are."""
    return math.degrees(2.0 * math.atan(extent / (2.0 * float(focal_px))))


@dataclass
class Patch:
    """The piece of sphere a picture covers, and the pixels it is stored in.

    `span_az` and `span_el` are its full angular extent in degrees, centred on
    the view axis.  The whole point of the class is that the two are allowed to
    be smaller than 180: everything that reads it -- the sampling grid, the
    disparity, the metadata -- works off the spans rather than assuming the
    picture fills the sphere.  A `full` frame is the special case where they do.
    """

    span_az: float
    span_el: float
    width: int
    height: int

    @property
    def full_width(self):
        """Width of the 360-degree equirectangular frame this is a piece of."""
        return int(round(self.width * 360.0 / self.span_az))

    @property
    def full_height(self):
        return int(round(self.height * FOV / self.span_el))

    @property
    def left(self):
        return int(round((self.full_width - self.width) / 2.0))

    @property
    def top(self):
        return int(round((self.full_height - self.height) / 2.0))

    @property
    def per_radian(self):
        """Pixels per radian, which is what stands in for the focal length once
        the picture is on a sphere."""
        return self.width / math.radians(self.span_az)


def video_cap(fps):
    """The largest per-eye square a clip at this frame rate can be decoded at.

    Rounded down through `VIDEO_CAPS`, and never below the slowest entry -- a
    clip claiming some absurd frame rate gets the most conservative answer
    rather than an exception, having to play on the same hardware as the rest.
    """
    for limit, side in VIDEO_CAPS:
        if fps <= limit + 1e-6:
            return side
    return VIDEO_CAPS[-1][1]


def natural_size(focal_px):
    """The per-eye square that would keep this source's own detail exactly.

    Giving the sphere the pixels per degree the source had, which for a
    65-degree lens is a square nearly two and a half times its width.  Not what
    the frame is written at -- see `patch` -- but what says whether a source has
    the detail to fill one: a 720-wide clip wants 1759 and a ceiling of 4096
    will be well over half empty of anything it can offer.

    **Measured at the centre of the frame, which is the number that used to be
    wrong.**  A pinhole does not spread its pixels evenly over the field it
    covers: the density rises as `sec^2` towards the edges, so the far corners
    of a wide frame carry half again as many pixels per degree as the middle
    does.  This was `width * FOV / fov(focal_px, width)`, which is the density
    *averaged across the width* -- a number no part of the picture actually
    offers, and one the periphery flatters.  A 1280-wide frame through a 28mm
    lens averages 19.6 pixels a degree and offers 17.4 in the middle, so it was
    judged able to fill 3519 of a 4096 ceiling where the part anyone is looking
    at fills 3128.  That is the difference between clearing
    `prepass.WORTH_UPSCALING` and not, and 720p landscape cleared it by 1% --
    so the upscaler declined footage that was being enlarged 1.31x in the centre
    of frame, on the strength of detail sitting out at the edges.

    The centre is the conservative reading and the one that describes where the
    subject is.  It is also what the old expression converges to as the lens
    narrows and the `sec^2` spread flattens out, so nothing changes for the
    long lenses; it is the wide ones that were being over-credited.
    """
    return math.radians(FOV) * float(focal_px)


def patch(focal_px, src_w, src_h, cap=MAX_SIZE, size=None):
    """The piece of sphere a frame covers, and how big to store it.

    Always the hemisphere the format asks for.  Storing only the piece the lens
    actually reached was built and then taken out again: it saved four fifths of
    the pixels and recorded where the piece belonged, in the fields both formats
    provide for exactly that, and no player read them -- Skybox assumes every eye
    is a full 180 by 180, and showed a 65-by-91-degree patch 2.7 times too close
    and 40% stretched sideways.  See the history for it if a player ever catches
    up.

    And always the ceiling, rather than what the source can fill.  A headset
    shows 25 pixels a degree whatever it is handed, so a frame short of that is
    visibly soft however faithfully it matches its source -- and the source is
    the half of that this cannot fix, which is what `natural_size` is for and
    what the upscaler in `prepass` is for.
    """
    side = even(max(64, size or cap))
    return Patch(FOV, FOV, side, side)


def auto_target(spot, target_deg=TARGET_DEG):
    """`target_deg` restated as the percentage of frame width that
    `stereo.auto_geometry` takes, so that one piece of scene-fitting logic serves
    both projections instead of two that can drift apart.

    Asked of the patch rather than of the format, which is not the pedantry it
    looks: the shared routine aims at a share of the frame width, so if the frame
    is ever less than the whole 180 degrees, the same share is a different angle.
    Pinned at 180, a 65-degree frame once came out with under half the separation
    it had asked for and looked merely flat rather than wrong.

    `target_deg` is a parameter rather than the constant it reads by default
    because `TARGET_DEG` is the one number in this file arrived at by reasoning
    rather than by measurement, and the caller has to be able to disagree with
    it -- see `Settings.target_deg`.
    """
    return 100.0 * target_deg / spot.span_az


def _elevation(top, rows, spot, device, dtype):
    """Elevation of each stored row, in radians, counting down from the top of
    the patch -- which is the way round equirectangular has it."""
    row = torch.arange(top, top + rows, device=device, dtype=dtype)
    span = math.radians(spot.span_el)
    return span / 2.0 - (row + 0.5) / spot.height * span


def directions(top, rows, spot, device, dtype):
    """Unit vectors for a band of stored pixels: +x right, +y up, +z forward.

    Split out of `_grid` because the plate needs the same directions without the
    pinhole behind them -- it is looking up a sphere, not a photograph.
    """
    col = torch.arange(spot.width, device=device, dtype=dtype)
    span_az = math.radians(spot.span_az)
    az = (col + 0.5) / spot.width * span_az - span_az / 2.0
    el = _elevation(top, rows, spot, device, dtype)[:, None]

    cos_el, sin_el = torch.cos(el), torch.sin(el)
    return (torch.sin(az)[None, :] * cos_el,
            sin_el.expand(rows, spot.width),
            torch.cos(az)[None, :] * cos_el)


def turn(x, y, z, rotation):
    """Rotate a field of directions.  `rotation` is a 3x3 taking the directions
    given into the frame they should be read in."""
    r = rotation.to(device=x.device, dtype=x.dtype)
    return (r[0, 0] * x + r[0, 1] * y + r[0, 2] * z,
            r[1, 0] * x + r[1, 1] * y + r[1, 2] * z,
            r[2, 0] * x + r[2, 1] * y + r[2, 2] * z)


def _grid(top, rows, spot, focal_px, src_h, src_w, device, dtype, rotation=None):
    """Where each stored pixel in a band of rows reads from in the source
    photograph, and whether it reads from anywhere at all.

    Azimuth and elevation give a direction; the direction is put back through
    the pinhole the depth model reported.  Anything behind the camera, or past
    the edge of the frame, is not in the photograph and is marked as such.

    `rotation` turns the directions first, which is what lets a frame shot with
    the camera pointing somewhere else be laid on the same plate -- see `plate`.
    """
    x, y, z = directions(top, rows, spot, device, dtype)
    if rotation is not None:
        x, y, z = turn(x, y, z, rotation)

    # Behind the camera divides by a negative and folds the picture back on
    # itself, so it is held at 1 for the arithmetic and thrown away by the mask.
    front = z > 1e-6
    depth = torch.where(front, z, torch.ones_like(z))
    u = (src_w - 1) / 2.0 + focal_px * x / depth
    v = (src_h - 1) / 2.0 - focal_px * y / depth
    valid = front & (u >= 0) & (u <= src_w - 1) & (v >= 0) & (v <= src_h - 1)

    # grid_sample's normalised coordinates, align_corners=False: a pixel centre
    # at i sits at (2i + 1) / n - 1.
    gx = (2.0 * u + 1.0) / src_w - 1.0
    gy = (2.0 * v + 1.0) / src_h - 1.0
    return torch.stack((gx, gy), dim=-1)[None], valid


def project(source, focal_px, spot, mode="bicubic", rotation=None):
    """Warp a `[C, H, W]` perspective image onto the patch, and say which of its
    pixels came from anywhere.

    Banded over output rows: a large projection built in one piece is several
    gigabytes, and nothing here needs it to be.

    `mode` is bicubic for colour and wants to be bilinear for depth.  The
    projection is nearly always enlarging now -- the frame is the ceiling rather
    than whatever the source could fill -- so the kernel has started to matter,
    and bicubic costs the same to the millisecond.  But bicubic overshoots at an
    edge, and an overshoot in an inverse-depth map is not a soft halo, it is a
    pixel claiming to be somewhere it is not and a disparity to match.

    `rotation` says where the camera was pointing when this frame was taken,
    relative to whatever the patch is centred on.  A still and every frame of the
    live picture leave it None, which is the camera pointing straight down the
    patch's own axis; `plate` passes one to lay a panned frame in the right place.
    """
    src_h, src_w = source.shape[-2:]
    out = torch.zeros(source.shape[0], spot.height, spot.width,
                      device=source.device, dtype=source.dtype)
    mask = torch.zeros(spot.height, spot.width, device=source.device, dtype=torch.bool)
    band = max(1, _BAND_PIXELS // max(spot.width, 1))
    for top in range(0, spot.height, band):
        rows = min(band, spot.height - top)
        grid, valid = _grid(top, rows, spot, focal_px, src_h, src_w,
                            source.device, source.dtype, rotation)
        out[:, top:top + rows] = F.grid_sample(
            source[None], grid, mode=mode, padding_mode="zeros", align_corners=False)[0]
        mask[top:top + rows] = valid
    return out, mask


def coverage(mask, spot):
    """The share of the hemisphere that is photograph, by solid angle.

    Deliberately not the share of the pixels.  Equirectangular packs far more
    pixels into a degree near the pole than at the equator, so counting them
    would flatter or libel the result depending on nothing but where in the frame
    the picture landed.  What a viewer notices is how much of what they can turn
    to look at is real, and that is solid angle.
    """
    weight = torch.cos(_elevation(0, spot.height, spot, mask.device, torch.float32))[:, None]
    per_pixel = math.radians(spot.span_az) / spot.width * math.radians(spot.span_el) / spot.height
    return float((mask.float() * weight).sum()) * per_pixel / (2.0 * math.pi)


def _falloff(mask, spot, falloff_deg=FALLOFF_DEG):
    """A soft edge at the boundary of the real picture, in place of a cut one.

    The blur of a hard mask runs from one deep inside to zero outside, which is
    the ramp wanted; multiplying by the mask again keeps the fade strictly
    inside, so no pixel that came from nowhere is ever shown at any strength.
    The blur replicates at the frame edge rather than padding it with nothing, so
    a picture that does reach the edge is not darkened there for the crime of
    having no room to spare.
    """
    radius = max(1, round(falloff_deg * spot.width / spot.span_az))
    soft = stereo._box(mask.float()[None, None], radius)[0, 0].clamp_(0, 1)
    return mask.float() * (soft * soft * (3.0 - 2.0 * soft))  # smoothstep


def surround(equirect, mask, spot, dim=SURROUND_DIM, blur_deg=SURROUND_BLUR_DEG,
             ramp_deg=SURROUND_RAMP_DEG):
    """A dim, blurred wash for the part of the sphere the photograph never
    reached -- what every social video site puts behind a clip that does not
    fill the frame.

    It earns its place here for the reason a diffusion fill does not: it is a
    fixed function of the frame, so it moves exactly as the picture moves and
    cannot crawl or boil from one frame to the next.  Nothing is invented that
    was not already on screen.

    Two things differ from the rectangular case, both because this is a sphere.
    The wash is spread outward from the edge rather than scaled up from the
    middle, so what sits beside the viewer is the colour of whatever the camera
    saw in that direction -- a centre copy would put the subject's face in their
    peripheral vision.  And `render` uses one wash for both eyes, because a
    periphery carrying parallax of its own would fight the real picture over
    where the eyes should converge.
    """
    colour, weight = (equirect * mask)[None], mask.float()[None, None]

    # Push-pull: average down until every level has something in it, then come
    # back up, blending the finer detail in wherever there was any.
    stack = []
    while min(colour.shape[-2:]) > 1:
        stack.append((colour, weight))
        colour = F.avg_pool2d(colour, 2, ceil_mode=True)
        weight = F.avg_pool2d(weight, 2, ceil_mode=True)

    # Stopping short of the finest levels is what blurs it.  Those levels carry
    # the picture's own detail, and putting it back would make the surround a
    # second, blurrier photograph rather than the light coming off the first.
    span = max(2.0, blur_deg * spot.width / spot.span_az)
    coarse = max(1, min(len(stack), int(round(math.log2(span)))))

    out = colour / weight.clamp_min(1e-6)
    for colour, weight in reversed(stack[coarse:]):
        out = F.interpolate(out, size=colour.shape[-2:], mode="bilinear", align_corners=False)
        seen = weight.clamp(0, 1)
        out = seen * (colour / weight.clamp_min(1e-6)) + (1 - seen) * out
    out = F.interpolate(out, size=tuple(mask.shape), mode="bilinear", align_corners=False)
    return _ease(out[0], mask, spot, dim, ramp_deg)


def _ease(wash, mask, spot, dim, ramp_deg=SURROUND_RAMP_DEG):
    """The wash at the picture's own strength where it leaves the picture, eased
    to `dim` further out.

    Both ends go through `_expose`, so a brightness above 1 still screens rather
    than clamps and the two halves still meet without a step at exactly 1.  What
    changes is only *where* each applies: the edge of the photograph is the one
    place a viewer's eye is guaranteed to be, and it is the one place the wash
    should be indistinguishable from what it is spreading.
    """
    lit = _expose(wash, 1.0)
    if not ramp_deg or float(dim) == 1.0:
        return _expose(wash, dim)
    radius = max(1, round(ramp_deg * spot.width / spot.span_az))
    near = stereo._box(mask.float()[None, None], radius)[0, 0].clamp(0, 1)
    near = near * near * (3.0 - 2.0 * near)  # smoothstep, out from the edge
    return torch.lerp(_expose(wash, dim), lit, near)


def _expose(wash, gain):
    """Set how bright the surround is.  Below 1 that is a plain multiply; above
    it, the wash is being asked for more light than it measured.

    A dark scene spreads a dark wash, and turning the number up is the only way
    to light one -- but multiplying past 1 and clamping would stop every channel
    at the same place, so a bright corner would lose its colour before it lost
    its shape and land as a flat white blob, exactly where a viewer's eye is
    most easily caught.  So above 1 the wash is screened into itself instead:
    `1 - (1 - w)**gain`, which reaches for white without ever arriving, leaves 0
    at 0, and is the identity at gain 1 -- so the two halves meet without a step
    and a number chosen by eye keeps meaning what it meant.
    """
    gain = float(gain)
    if gain <= 1.0:
        return (wash * gain).clamp_(0, 1)
    return (1.0 - (1.0 - wash.clamp(0, 1)) ** gain).clamp_(0, 1)


def half_disparity(inverse, spot, eyes_mm, focus_m, limit_deg=LIMIT_DEG, elevation=None):
    """Half the omnidirectional-stereo separation, in pixels, for every pixel.

    The formula is `stereo.half_disparity`'s, because it is the same geometry:
    what changes is that a radian of angle rather than a pixel of sensor is what
    the separation is measured against, so pixels-per-radian goes in where the
    focal length was.  The clamp is in degrees for the same reason -- a
    percentage of the frame width means a percentage of however much sphere this
    patch happens to cover, which is not a quantity comfort is described in.
    """
    ppr = spot.per_radian
    half = stereo.half_disparity(inverse, ppr, eyes_mm, focus_m)
    cap = math.radians(limit_deg) * ppr / 2.0
    half = half.clamp(-cap, cap)
    if elevation is None:
        elevation = _elevation(0, spot.height, spot, inverse.device, torch.float32)
    # Towards the poles the two eyes line up along the view direction and the
    # separation has to go to nothing.  Without this the top and bottom of the
    # sphere ask for parallax that no arrangement of two eyes could produce.
    return half * torch.cos(elevation.to(half.dtype))[:, None]


def render(rgb, inverse, focal_px, eyes_mm, focus_m, spot, limit_deg=LIMIT_DEG,
           wash=0.0, plate=None, plate_dim=PLATE_DIM):
    """One flat frame and its depth in; a VR180 eye pair and its coverage out.

    `rgb` is `[3, H, W]` in [0, 1] and `inverse` the matching `[H, W]` inverse
    depth in 1/metres -- both as the flat path has them, because both are warped
    from there rather than estimated here.

    `wash` is how bright to make the `surround` that fills the empty part of the
    sphere, 0 for none -- which is the same thing as saying the dark is a
    surround with no light in it.

    `plate` is `(colour, mask)` on this same grid: the scene this shot was
    actually filmed in, gathered once by `plate.build` and handed here already
    turned to this frame's own bearing.  It goes *behind* the picture and *in
    front of* the wash, so the three of them read outwards as photograph, then
    scene, then light.  Everything about why it is built once per shot rather
    than once per frame is in `plate`; what matters here is only that it arrives
    fixed, so nothing this function does can make it boil.
    """
    equirect, mask = project(rgb, focal_px, spot)
    equirect.clamp_(0, 1)  # bicubic overshoots, and this is colour
    depth, _ = project(inverse[None], focal_px, spot, mode="bilinear")
    # Nothing was ever pointed at the void, so it has no depth either.  Parked at
    # the screen plane it asks for no separation, which keeps it still and stops
    # the splat dragging it sideways over the edge of the real picture.
    depth = torch.where(mask, depth[0], torch.full_like(depth[0], 1.0 / max(focus_m, 1e-6)))

    half = half_disparity(depth, spot, eyes_mm, focus_m, limit_deg)
    # No margin: the flat path trims the sliver at each edge that only one eye
    # reaches, and here that sliver is angle.  Trimming it would quietly narrow
    # the field of view and put every remaining pixel at the wrong bearing.
    left, right = stereo.make_pair(equirect, half, margin=0)

    fade = _falloff(mask, spot)
    behind = _behind(equirect, mask, spot, wash, plate, plate_dim)
    if behind is None:
        return left * fade, right * fade, mask
    # One backdrop, both eyes, from the picture before it was split -- see
    # `surround` for why it must not have a parallax of its own.
    return torch.lerp(behind, left, fade), torch.lerp(behind, right, fade), mask


def _behind(equirect, mask, spot, wash, plate, plate_dim):
    """Everything the photograph is shown in front of, or None for plain dark.

    The wash is spread out of the plate rather than out of the frame wherever
    there is a plate, which is not a detail: it means the light at the edge of
    the sphere is the light of whatever the camera turned to see in that
    direction, and it means the wash and the plate meet without a step because
    one is made from the other.
    """
    if plate is None:
        return surround(equirect, mask, spot, dim=float(wash)) if wash else None

    colour, seen = plate
    colour = _match_gain(colour, equirect, seen & mask) * plate_dim
    colour = colour.clamp_(0, 1)
    # What the wash is spread from: photograph where there is any, and the plate
    # everywhere else it reaches.
    source = torch.where(mask, equirect, colour)
    lit = surround(source, mask | seen, spot, dim=float(wash)) if wash else torch.zeros_like(colour)
    # A wider fade than the picture's, because this is the join between what was
    # filmed and what was inferred from it -- and a hard circle where the painted
    # ring stops is the one thing a viewer's eye goes straight to.
    return torch.lerp(lit, colour, _falloff(seen, spot, PLATE_FALLOFF_DEG))


def _match_gain(colour, equirect, overlap, limit=2.0):
    """Scale the plate to the brightness the live frame is at.

    A phone's automatic exposure walks across a shot, so a plate median-combined
    over the whole of one sits at the average exposure and the frame in front of
    it does not.  Left alone that is a visible step at the edge of the picture,
    on footage where nothing else went wrong.  Clamped, because an overlap of
    almost nothing -- the first frame of a whip pan -- would otherwise produce a
    ratio of almost anything.
    """
    if not bool(overlap.any()):
        return colour
    here = float(colour[:, overlap].mean())
    there = float(equirect[:, overlap].mean())
    if here < 1e-4:
        return colour
    return colour * min(max(there / here, 1.0 / limit), limit)
