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
# How coarse the optional surround is, and how much dimmer than the picture.
# Coarse enough to read as light rather than as a second blurry photograph, and
# dim because a bright periphery is tiring to sit inside and competes with the
# thing you are meant to be looking at.
SURROUND_BLUR_DEG = 12.0
SURROUND_DIM = 0.45
# Ceilings on the stored width per eye.  A still can afford a big one; a clip has
# to come back out of a hardware decoder, and 2048 per eye is 4096 across.
MAX_SIZE = 4096
VIDEO_MAX_SIZE = 2048
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


def per_eye(width, focal_px, cap=MAX_SIZE):
    """The square to store each eye at.

    Keeping the photograph's own detail means giving the sphere the pixels per
    degree the photograph had, which for a 65-degree lens is a square nearly
    three times the source width and for a 49-degree one four times.  It is asked
    for and then capped, so a big photograph is used softer than it arrived --
    the price of a frame that needs nothing read to sit at the right size.
    """
    natural = width * FOV / fov(focal_px, width)
    return even(max(64, min(cap, round(natural))))


def patch(focal_px, src_w, src_h, cap=MAX_SIZE, size=None):
    """The piece of sphere a frame covers, and how big to store it.

    Always the hemisphere the format asks for.  Storing only the piece the lens
    actually reached was built and then taken out again: it saved four fifths of
    the pixels and recorded where the piece belonged, in the fields both formats
    provide for exactly that, and no player read them -- Skybox assumes every eye
    is a full 180 by 180, and showed a 65-by-91-degree patch 2.7 times too close
    and 40% stretched sideways.  See the history for it if a player ever catches
    up.
    """
    side = even(min(cap, size or per_eye(src_w, focal_px, cap)))
    return Patch(FOV, FOV, side, side)


def auto_target(spot):
    """`TARGET_DEG` restated as the percentage of frame width that
    `stereo.auto_geometry` takes, so that one piece of scene-fitting logic serves
    both projections instead of two that can drift apart.

    Asked of the patch rather than of the format, which is not the pedantry it
    looks: the shared routine aims at a share of the frame width, so if the frame
    is ever less than the whole 180 degrees, the same share is a different angle.
    Pinned at 180, a 65-degree frame once came out with under half the separation
    it had asked for and looked merely flat rather than wrong.
    """
    return 100.0 * TARGET_DEG / spot.span_az


def _elevation(top, rows, spot, device, dtype):
    """Elevation of each stored row, in radians, counting down from the top of
    the patch -- which is the way round equirectangular has it."""
    row = torch.arange(top, top + rows, device=device, dtype=dtype)
    span = math.radians(spot.span_el)
    return span / 2.0 - (row + 0.5) / spot.height * span


def _grid(top, rows, spot, focal_px, src_h, src_w, device, dtype):
    """Where each stored pixel in a band of rows reads from in the source
    photograph, and whether it reads from anywhere at all.

    Azimuth and elevation give a direction; the direction is put back through
    the pinhole the depth model reported.  Anything behind the camera, or past
    the edge of the frame, is not in the photograph and is marked as such.
    """
    col = torch.arange(spot.width, device=device, dtype=dtype)
    span_az = math.radians(spot.span_az)
    az = (col + 0.5) / spot.width * span_az - span_az / 2.0
    el = _elevation(top, rows, spot, device, dtype)[:, None]

    cos_el, sin_el = torch.cos(el), torch.sin(el)
    x = torch.sin(az)[None, :] * cos_el
    y = sin_el.expand(rows, spot.width)
    z = torch.cos(az)[None, :] * cos_el

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


def project(source, focal_px, spot):
    """Warp a `[C, H, W]` perspective image onto the patch, and say which of its
    pixels came from anywhere.

    Banded over output rows: a large projection built in one piece is several
    gigabytes, and nothing here needs it to be.
    """
    src_h, src_w = source.shape[-2:]
    out = torch.zeros(source.shape[0], spot.height, spot.width,
                      device=source.device, dtype=source.dtype)
    mask = torch.zeros(spot.height, spot.width, device=source.device, dtype=torch.bool)
    band = max(1, _BAND_PIXELS // max(spot.width, 1))
    for top in range(0, spot.height, band):
        rows = min(band, spot.height - top)
        grid, valid = _grid(top, rows, spot, focal_px, src_h, src_w,
                            source.device, source.dtype)
        out[:, top:top + rows] = F.grid_sample(
            source[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
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


def _falloff(mask, spot):
    """A soft edge at the boundary of the real picture, in place of a cut one.

    The blur of a hard mask runs from one deep inside to zero outside, which is
    the ramp wanted; multiplying by the mask again keeps the fade strictly
    inside, so no pixel that came from nowhere is ever shown at any strength.
    The blur replicates at the frame edge rather than padding it with nothing, so
    a picture that does reach the edge is not darkened there for the crime of
    having no room to spare.
    """
    radius = max(1, round(FALLOFF_DEG * spot.width / spot.span_az))
    soft = stereo._box(mask.float()[None, None], radius)[0, 0].clamp_(0, 1)
    return mask.float() * (soft * soft * (3.0 - 2.0 * soft))  # smoothstep


def surround(equirect, mask, spot, dim=SURROUND_DIM, blur_deg=SURROUND_BLUR_DEG):
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
    return (out[0] * dim).clamp_(0, 1)


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
           wash=False):
    """One flat frame and its depth in; a VR180 eye pair and its coverage out.

    `rgb` is `[3, H, W]` in [0, 1] and `inverse` the matching `[H, W]` inverse
    depth in 1/metres -- both as the flat path has them, because both are warped
    from there rather than estimated here.

    `wash` fills the empty part of the sphere with `surround` instead of leaving
    it black.
    """
    equirect, mask = project(rgb, focal_px, spot)
    depth, _ = project(inverse[None], focal_px, spot)
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
    if not wash:
        return left * fade, right * fade, mask
    # One wash, both eyes, from the picture before it was split -- see
    # `surround` for why it must not have a parallax of its own.
    fill = surround(equirect, mask, spot)
    return torch.lerp(fill, left, fade), torch.lerp(fill, right, fade), mask
