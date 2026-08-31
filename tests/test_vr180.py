"""The spherical geometry, which has to be exactly as right as the flat one.

A projection that is wrong by a few degrees still looks like a picture -- it
just sits at the wrong bearing, and the only symptom is that a headset feels
subtly wrong to be inside.  So the angles are checked against the arithmetic
rather than against how the result looks.
"""

import math

import pytest
import torch

from stereocraft import stereo, vr180


def inv(*metres):
    return torch.tensor([[1.0 / z for z in metres]])


def lens(equivalent_mm, width):
    """Focal length in pixels for a 35mm-equivalent lens at this width."""
    return float(equivalent_mm) / 36.0 * width


def dot(size=64, at=None):
    """A source photo that is dark everywhere but one pixel."""
    image = torch.zeros(1, size, size)
    row, col = at or (size // 2, size // 2)
    image[0, row, col] = 1.0
    return image


class TestLens:
    def test_the_fov_of_a_28mm_phone(self):
        assert vr180.fov(lens(28, 1000), 1000) == pytest.approx(65.47, abs=0.01)

    def test_it_works_on_either_axis(self):
        """A 3:2 frame is taller in pixels than it is in nothing else, and the
        vertical angle falls out of the same arithmetic."""
        focal = lens(28, 3000)
        assert vr180.fov(focal, 2000) < vr180.fov(focal, 3000)

    def test_a_longer_lens_sees_less(self):
        assert vr180.fov(2000.0, 1000) < vr180.fov(1000.0, 1000)


class TestPatch:
    """Where the stored frame sits in the 360-by-180 one it is a piece of.

    A hemisphere is half a sphere, so there is a placement to get right even
    though the frame is always the whole square.
    """

    def test_it_is_always_the_square_the_format_asks_for(self):
        spot = vr180.patch(lens(28, 4000), 4000, 3000)
        assert spot.span_az == spot.span_el == 180.0
        assert spot.width == spot.height

    def test_the_full_frame_it_belongs_to_is_two_to_one(self):
        """Equirectangular is 360 across and 180 down.  If the stored piece does
        not imply that shape, every offset recorded against it is wrong."""
        spot = vr180.patch(lens(28, 4000), 4000, 3000)
        assert spot.full_width == pytest.approx(2 * spot.full_height, rel=0.01)

    def test_the_piece_sits_in_the_middle(self):
        spot = vr180.patch(lens(28, 4000), 4000, 3000)
        assert spot.left == pytest.approx((spot.full_width - spot.width) / 2, abs=1)
        assert spot.top == pytest.approx((spot.full_height - spot.height) / 2, abs=1)

    def test_it_is_half_the_sphere_across_and_all_of_it_down(self):
        """180 of a possible 360, so a quarter falls off each side and nothing
        off the top -- which is the whole of what makes a file VR180 and not
        360, and the reason the placement matters even without a tighter crop."""
        spot = vr180.patch(lens(28, 1000), 1000, 750)
        assert spot.full_width == 2 * spot.width
        assert spot.full_height == spot.height
        assert spot.top == 0
        assert spot.left == spot.width // 2

    def test_it_is_the_ceiling_whatever_the_source(self):
        """A headset shows 25 pixels a degree whatever it is handed, so the frame
        is the ceiling and the source's own size has stopped deciding it."""
        small = vr180.patch(lens(28, 640), 640, 480, cap=4096)
        large = vr180.patch(lens(28, 4000), 4000, 3000, cap=4096)
        assert small.width == large.width == 4096

    def test_the_cap_is_a_cap(self):
        assert vr180.patch(lens(28, 9000), 9000, 6000, cap=4096).width == 4096

    def test_an_explicit_size_is_taken(self):
        assert vr180.patch(lens(28, 3000), 3000, 2000, size=1024).width == 1024

    def test_both_dimensions_come_out_even(self):
        spot = vr180.patch(lens(35, 999), 999, 667)
        assert spot.width % 2 == 0 and spot.height % 2 == 0

    def test_pixels_per_radian_follows_the_span(self):
        """Half the angle in the same pixels is twice the density, and the
        disparity is measured against it."""
        assert vr180.Patch(90.0, 90.0, 1000, 1000).per_radian == pytest.approx(
            2 * vr180.Patch(180.0, 180.0, 1000, 1000).per_radian)


class TestNaturalSize:
    """What a source could fill, as against what the frame is written at."""

    def test_it_is_the_density_at_the_centre_of_frame(self):
        """Pixels per radian on the view axis, spread over the whole 180 degrees
        -- which for a 28mm lens on 1000 pixels is pi times its focal length."""
        assert vr180.natural_size(lens(28, 1000)) == pytest.approx(math.pi * lens(28, 1000))

    def test_it_does_not_credit_a_source_with_its_own_edge_stretch(self):
        """The measurement this replaced, and the reason it was replaced.

        A pinhole packs more pixels into a degree at the edge of the frame than
        on the axis, so averaging density across the width reports a number the
        middle of the picture cannot deliver.  1280 wide through a 28mm lens
        averages out at 3519 and offers 3128 where the subject is -- and 3519 is
        over `prepass.WORTH_UPSCALING` of a 4096 ceiling where 3128 is under it,
        so the difference decided whether the upscaler ran at all.
        """
        averaged = 1280 * vr180.FOV / vr180.fov(lens(28, 1280), 1280)
        assert averaged == pytest.approx(3519, rel=1e-3), "what it used to say"
        assert vr180.natural_size(lens(28, 1280)) == pytest.approx(3128, rel=1e-3)
        assert vr180.natural_size(lens(28, 1280)) < averaged

    def test_the_two_agree_once_the_lens_is_narrow(self):
        """The `sec^2` spread is what separates them, and it flattens out as the
        field narrows -- so nothing changed for a long lens, only for a wide one."""
        gaps = []
        for equivalent in (28, 50, 100, 300):
            focal = lens(equivalent, 1000)
            averaged = 1000 * vr180.FOV / vr180.fov(focal, 1000)
            gaps.append(averaged / vr180.natural_size(focal))
        assert gaps == sorted(gaps, reverse=True), "the gap closes as the lens narrows"
        assert gaps[0] > 1.12, "a 28mm frame was over-credited by an eighth"
        assert gaps[-1] == pytest.approx(1.0, abs=2e-3), "a 300mm one by a tenth of a percent"

    def test_it_is_what_says_a_source_is_short_of_the_ceiling(self):
        """720 wide reaches 1759 of a 4096 ceiling, which is the whole case for
        upscaling; 1920 wide overshoots it and needs nothing."""
        assert vr180.natural_size(lens(28, 720)) < 4096
        assert vr180.natural_size(lens(28, 1920)) > 4096

    def test_a_longer_lens_wants_more(self):
        """The narrower the lens, the more the sphere stretches what it saw."""
        assert vr180.natural_size(lens(50, 1000)) > vr180.natural_size(lens(28, 1000))


class TestProjection:
    def test_the_centre_of_the_patch_is_the_centre_of_the_photo(self):
        # Odd sizes, so that "the centre" is a pixel rather than the crack
        # between two of them.
        spot = vr180.Patch(180.0, 180.0, 33, 33)
        out, mask = vr180.project(dot(65), focal_px=65.0, spot=spot)
        assert bool(mask[16, 16])
        assert divmod(int(torch.argmax(out[0])), 33) == (16, 16)

    @pytest.mark.parametrize("col", [180, 200, 240, 255])
    def test_an_azimuth_lands_where_the_tangent_says(self, col):
        """A ray at theta off-axis meets the sensor at f*tan(theta), which is the
        whole of what "perspective" means and the one thing worth checking."""
        spot, focal, src = vr180.Patch(180.0, 180.0, 360, 360), 500.0, 800
        grid, valid = vr180._grid(180, 1, spot, focal, src, src,
                                  torch.device("cpu"), torch.float32)
        azimuth = math.radians((col + 0.5) / spot.width * spot.span_az - spot.span_az / 2)
        u = ((float(grid[0, 0, col, 0]) + 1.0) * src - 1.0) / 2.0
        assert u == pytest.approx((src - 1) / 2 + focal * math.tan(azimuth), abs=0.01)
        assert bool(valid[0, col])

    def test_a_bearing_reads_the_same_place_at_any_resolution(self):
        """A column at a given bearing has to come from the same place in the
        photograph however many pixels the frame is stored in, or the picture
        ends up at the wrong scale while looking perfectly sharp."""
        focal, src = 500.0, 800
        bearing = math.radians(10.0)

        def sampled(p):
            col = int(round((math.degrees(bearing) + p.span_az / 2) / p.span_az * p.width - 0.5))
            grid, _ = vr180._grid(p.height // 2, 1, p, focal, src, src,
                                  torch.device("cpu"), torch.float32)
            return ((float(grid[0, 0, col, 0]) + 1.0) * src - 1.0) / 2.0

        assert sampled(vr180.Patch(180.0, 180.0, 720, 720)) == pytest.approx(
            sampled(vr180.Patch(180.0, 180.0, 1440, 1440)), abs=1.5)

    def test_colour_is_sampled_bicubic_and_depth_bilinear(self):
        """Bicubic rings, and a ring in an inverse-depth map is not a soft halo
        -- it is a pixel claiming a distance it does not have, and a disparity
        to match.  Colour wants the sharper kernel; depth must not have it."""
        spot = vr180.Patch(180.0, 180.0, 96, 96)
        step = torch.zeros(1, 64, 64)
        step[:, :, 32:] = 1.0  # a hard edge, which is what makes ringing visible
        sharp, _ = vr180.project(step, 64.0, spot)
        soft, _ = vr180.project(step, 64.0, spot, mode="bilinear")
        assert float(sharp.max()) > float(soft.max()) or float(sharp.min()) < float(soft.min()), \
            "bicubic should overshoot where bilinear does not"

    def test_render_clamps_the_overshoot_out_of_the_colour(self):
        rgb = torch.zeros(3, 64, 64)
        rgb[:, :, 32:] = 1.0
        spot = vr180.Patch(180.0, 180.0, 96, 96)
        left, right, _ = vr180.render(rgb, torch.full((64, 64), 1 / 3), 64.0, 65.0, 3.0, spot)
        for eye in (left, right):
            assert float(eye.min()) >= 0.0 and float(eye.max()) <= 1.0

    def test_nothing_behind_the_camera_is_ever_valid(self):
        spot = vr180.Patch(180.0, 180.0, 64, 64)
        _, mask = vr180.project(torch.ones(1, 64, 64), focal_px=64.0, spot=spot)
        assert not bool(mask[:, 0].any()), "the left pole looks backwards"
        assert not bool(mask[:, -1].any()), "and so does the right one"

    @pytest.mark.parametrize("focal", [128.0, 256.0, 512.0])
    def test_coverage_is_the_share_of_the_hemisphere_the_lens_saw(self, focal):
        """The solid angle a rectilinear frame covers is 4*asin(sin(a/2)sin(b/2)),
        and a hemisphere is 2pi of them."""
        spot, src = vr180.Patch(180.0, 180.0, 512, 512), 256
        _, mask = vr180.project(torch.ones(1, src, src), focal, spot)
        a = b = math.radians(vr180.fov(focal, src))
        expected = 4 * math.asin(math.sin(a / 2) * math.sin(b / 2)) / (2 * math.pi)
        assert vr180.coverage(mask, spot) == pytest.approx(expected, rel=0.02)

    def test_resolution_does_not_change_how_much_sphere_is_real(self):
        """More stored pixels, the same picture on the same sphere.  If this
        moves, the sizing has quietly changed the field of view."""
        focal, src = 256.0, 256
        small = vr180.Patch(180.0, 180.0, 256, 256)
        large = vr180.Patch(180.0, 180.0, 768, 768)
        _, small_mask = vr180.project(torch.ones(1, src, src), focal, small)
        _, large_mask = vr180.project(torch.ones(1, src, src), focal, large)
        assert vr180.coverage(small_mask, small) == pytest.approx(
            vr180.coverage(large_mask, large), rel=0.03)

    def test_the_square_is_mostly_dark(self):
        """Which is the cost of the format, and the thing cropping was built to
        avoid before no player turned out to read where the crop belonged."""
        focal, src = 256.0, 256
        spot = vr180.patch(focal, src, src, size=256)
        _, mask = vr180.project(torch.ones(1, src, src), focal, spot)
        assert float(mask.float().mean()) < 0.35


class TestOdsDisparity:
    """dtheta = B*(1/Z - 1/Zc) radians, tapering to nothing at the poles."""

    def spot(self, span=180.0, width=360):
        return vr180.Patch(span, span, width, width)

    def test_matches_the_formula_at_the_equator(self):
        spot, z, focus = self.spot(), 5.0, 3.0
        half = vr180.half_disparity(inv(z), spot, 65.0, focus, elevation=torch.zeros(1))
        radians = float(half) * 2 / spot.per_radian
        assert radians == pytest.approx(0.065 * (1 / z - 1 / focus), abs=1e-6)

    def test_the_angle_is_the_same_whatever_the_patch(self):
        """Pixels change with the crop; the geometry must not."""
        z, focus = 5.0, 3.0
        wide = vr180.half_disparity(inv(z), self.spot(180.0, 360), 65.0, focus,
                                    elevation=torch.zeros(1))
        tight = vr180.half_disparity(inv(z), self.spot(45.0, 360), 65.0, focus,
                                     elevation=torch.zeros(1))
        assert float(wide) * 2 / self.spot(180.0, 360).per_radian == pytest.approx(
            float(tight) * 2 / self.spot(45.0, 360).per_radian, rel=1e-5)

    def test_zero_at_the_screen_plane(self):
        half = vr180.half_disparity(inv(3.0), self.spot(), 65.0, 3.0, elevation=torch.zeros(1))
        assert float(half) == pytest.approx(0.0, abs=1e-9)

    def test_the_poles_have_no_separation_to_give(self):
        """Looking straight up there is no across-the-line-of-sight left to put
        two eyes on, and a projection that asks for parallax there is asking for
        something no pair of eyes could produce."""
        poles = torch.tensor([math.pi / 2, 0.0, -math.pi / 2])
        half = vr180.half_disparity(inv(1.0).expand(3, 1), self.spot(), 65.0, 3.0,
                                    elevation=poles)
        up, level, down = half[:, 0].tolist()
        assert up == pytest.approx(0.0, abs=1e-6)
        assert down == pytest.approx(0.0, abs=1e-6)
        assert abs(level) > 0

    def test_the_taper_is_a_cosine(self):
        at = torch.tensor([0.0, math.pi / 3])  # 60 degrees up, so half the effect
        half = vr180.half_disparity(inv(1.0).expand(2, 1), self.spot(), 65.0, 3.0, elevation=at)
        assert float(half[1, 0]) == pytest.approx(float(half[0, 0]) * 0.5, rel=1e-4)

    def test_the_clamp_is_in_degrees(self):
        spot, limit = self.spot(), 1.2
        half = vr180.half_disparity(inv(0.02), spot, 500.0, 3.0, limit_deg=limit,
                                    elevation=torch.zeros(1))
        assert float(half.abs()) <= math.radians(limit) * spot.per_radian / 2 + 1e-6

    def test_defaults_to_one_elevation_per_row(self):
        spot = vr180.Patch(180.0, 180.0, 8, 8)
        assert vr180.half_disparity(torch.full((8, 8), 0.5), spot, 65.0, 3.0).shape == (8, 8)


class TestRender:
    def spot(self, w=32, h=32, span=180.0):
        return vr180.Patch(span, span * h / w, w, h)

    def test_the_pair_is_the_size_the_patch_asked_for(self):
        rgb = torch.rand(3, 48, 64)
        spot = self.spot(40, 30)
        left, right, mask = vr180.render(rgb, torch.full((48, 64), 0.5), 64.0, 65.0, 3.0, spot)
        assert left.shape == right.shape == (3, 30, 40)
        assert mask.shape == (30, 40)

    def test_nothing_is_trimmed(self):
        """The flat path trims the sliver only one eye reaches.  Here that sliver
        is angle, and trimming it would put every remaining pixel at the wrong
        bearing while looking perfectly fine."""
        rgb = torch.rand(3, 48, 64)
        left, _, _ = vr180.render(rgb, torch.full((48, 64), 0.5), 64.0, 65.0, 3.0, self.spot(40, 40))
        assert left.shape[2] == 40

    def test_the_void_stays_dark(self):
        rgb = torch.ones(3, 64, 64)
        spot = self.spot(64, 64)
        left, right, mask = vr180.render(rgb, torch.full((64, 64), 0.5), 64.0, 65.0, 3.0, spot)
        assert float(left[:, ~mask].max()) == pytest.approx(0.0, abs=1e-6)
        assert float(right[:, ~mask].max()) == pytest.approx(0.0, abs=1e-6)

    def test_the_edge_is_faded_rather_than_cut(self):
        """A hard rectangle floating in a void is what this avoids, so deep
        inside the picture is untouched and the rim is on its way down."""
        spot = vr180.Patch(180.0, 180.0, 96, 96)
        rgb = torch.ones(3, 64, 64)
        left, _, mask = vr180.render(rgb, torch.full((64, 64), 1 / 3), 64.0, 65.0, 3.0, spot)
        inside = left[0][mask]
        assert float(inside.max()) == pytest.approx(1.0, abs=1e-3), "the middle is untouched"
        assert bool((inside < 0.9).any()), "and the rim is on its way down"

    def test_the_fade_only_touches_the_boundary(self):
        """It is measured in degrees of sphere, so it must stay a rim rather than
        eating into the picture -- 4 degrees of a 65-degree lens, not a third."""
        spot = vr180.Patch(180.0, 180.0, 96, 96)
        rgb = torch.ones(3, 64, 64)
        left, _, mask = vr180.render(rgb, torch.full((64, 64), 1 / 3), 64.0, 65.0, 3.0, spot)
        inside = left[0][mask]
        assert float((inside > 0.99).float().mean()) > 0.5, "most of it is full strength"

    def test_a_flat_scene_leaves_the_two_eyes_alike(self):
        """Everything at the screen plane has no separation, so the pair matches."""
        rgb = torch.rand(3, 64, 64)
        left, right, _ = vr180.render(rgb, torch.full((64, 64), 1.0 / 3.0), 64.0, 65.0, 3.0,
                                      self.spot(48, 48))
        assert torch.allclose(left, right, atol=1e-5)

    def test_depth_moves_the_eyes_apart(self):
        rgb = torch.rand(3, 64, 64)
        left, right, _ = vr180.render(rgb, torch.full((64, 64), 1 / 0.8), 64.0, 200.0, 3.0,
                                      self.spot(48, 48))
        assert not torch.allclose(left, right, atol=1e-3)


class TestSurround:
    """The blurred wash, which is optional and is the one thing here that is not
    measured -- so what it must not do matters more than what it does."""

    def spot(self, side=128):
        return vr180.Patch(180.0, 180.0, side, side)

    def scene(self, h=64, w=64):
        """Something with structure, so a blur is distinguishable from a smear."""
        torch.manual_seed(0)
        base = torch.rand(3, 8, 8)
        return torch.nn.functional.interpolate(base[None], size=(h, w), mode="nearest")[0]

    def rendered(self, wash, spot=None, focal=64.0):
        spot = spot or self.spot()
        rgb = self.scene()
        return vr180.render(rgb, torch.full((64, 64), 1 / 3), focal, 65.0, 3.0, spot, wash=wash)

    def test_off_it_is_still_black(self):
        left, _, mask = self.rendered(wash=0.0)
        assert float(left[:, ~mask].max()) == pytest.approx(0.0, abs=1e-6)

    def test_on_the_void_is_filled(self):
        left, _, mask = self.rendered(wash=vr180.SURROUND_DIM)
        assert float(left[:, ~mask].mean()) > 0.02, "the dark is meant to be lit now"

    def test_both_eyes_get_exactly_the_same_wash(self):
        """The one that matters.  A periphery with a parallax of its own fights
        the real picture over where the viewer's eyes should converge, and there
        is no true disparity to give it -- nothing was ever there."""
        left, right, mask = self.rendered(wash=vr180.SURROUND_DIM)
        assert torch.equal(left[:, ~mask], right[:, ~mask])

    def test_the_picture_itself_is_untouched(self):
        """The wash goes behind, not over.  Deep inside the picture the two
        renders have to agree to the bit."""
        plain, _, mask = self.rendered(wash=0.0)
        washed, _, _ = self.rendered(wash=vr180.SURROUND_DIM)
        deep = stereo._box(mask.float()[None, None], 6)[0, 0] > 0.999
        assert bool(deep.any()), "there is a deep interior to check"
        assert torch.allclose(plain[:, deep], washed[:, deep], atol=1e-6)

    def test_it_is_dimmer_than_the_picture(self):
        """A bright periphery is tiring to sit inside and competes with the
        thing you are meant to be looking at."""
        left, _, mask = self.rendered(wash=vr180.SURROUND_DIM)
        assert float(left[:, ~mask].mean()) < float(left[:, mask].mean())

    def test_it_is_smoother_than_the_picture(self):
        """Otherwise it is a second, blurrier photograph rather than the light
        coming off the first."""
        left, _, mask = self.rendered(wash=vr180.SURROUND_DIM)
        rough = (left[:, :, 1:] - left[:, :, :-1]).abs()
        void = (~mask)[:, 1:] & (~mask)[:, :-1]
        inside = mask[:, 1:] & mask[:, :-1]
        assert float(rough[:, void].mean()) < float(rough[:, inside].mean()) / 4

    def test_it_follows_the_edge_it_is_beside(self):
        """Spread outward rather than scaled up from the middle, so what sits
        beside the viewer is the colour the camera saw in that direction."""
        spot = self.spot(160)
        rgb = torch.zeros(3, 64, 64)
        rgb[0] = 1.0  # a red left half and a blue right half
        rgb[:, :, 32:] = torch.tensor([0.0, 0.0, 1.0])[:, None, None]
        eq, mask = vr180.project(rgb, 64.0, spot)
        fill = vr180.surround(eq, mask, spot)
        row = spot.height // 2
        left_of, right_of = fill[:, row, 4], fill[:, row, -5]
        assert float(left_of[0]) > float(left_of[2]), "red side stays red"
        assert float(right_of[2]) > float(right_of[0]), "blue side stays blue"

    def test_brightness_is_a_plain_multiply_below_one(self):
        wash = torch.rand(3, 16, 16)
        assert torch.allclose(vr180._expose(wash.clone(), 0.45), wash * 0.45, atol=1e-6)

    def test_one_is_the_identity(self):
        """So a number picked by eye at 1.0 keeps meaning what it meant when the
        overbright half was added underneath it."""
        wash = torch.rand(3, 16, 16)
        assert torch.allclose(vr180._expose(wash.clone(), 1.0), wash, atol=1e-6)

    def test_the_two_halves_meet_without_a_step(self):
        wash = torch.linspace(0, 1, 128)
        below = vr180._expose(wash.clone(), 1.0)
        above = vr180._expose(wash.clone(), 1.0 + 1e-7)
        assert float((above - below).abs().max()) < 1e-5

    @pytest.mark.parametrize("gain", [1.5, 2.0, 3.0, 8.0])
    def test_overbright_lights_a_dark_scene_without_flattening_it(self, gain):
        """The point of going past 1: a dark scene spreads a dark wash, and a
        multiply-and-clamp would stop every channel at the same place and leave
        a colourless blob where the eye is most easily caught."""
        dark = torch.linspace(0.02, 0.30, 64)
        lit = vr180._expose(dark.clone(), gain)
        assert float(lit.mean()) > float(dark.mean()), "it has to actually brighten"
        assert float(lit.max()) < 1.0, "and never arrive at white"
        assert bool((lit.diff() > 0).all()), "keeping every step it started with"

    def test_more_gain_is_never_less_light(self):
        wash = torch.rand(3, 16, 16)
        seen = [float(vr180._expose(wash.clone(), g).mean()) for g in (0.45, 1.0, 1.5, 3.0, 8.0)]
        assert seen == sorted(seen)

    def test_overbright_survives_the_round_trip_through_render(self):
        left, _, mask = self.rendered(wash=3.0)
        plain, _, _ = self.rendered(wash=vr180.SURROUND_DIM)
        assert float(left[:, ~mask].mean()) > float(plain[:, ~mask].mean())
        assert float(left.max()) <= 1.0

    def test_a_small_move_makes_a_small_change(self):
        """The anti-boil property, and the whole reason this is a blur and not a
        diffusion model: the wash is a fixed function of the frame, so a frame
        that barely moves produces a surround that barely moves."""
        spot = self.spot()
        rgb = self.scene()
        moved = torch.roll(rgb, shifts=1, dims=2)
        washes = []
        for source in (rgb, moved):
            eq, mask = vr180.project(source, 64.0, spot)
            washes.append(vr180.surround(eq, mask, spot))
        drift = float((washes[0] - washes[1]).abs().mean())
        assert drift < 0.02, f"a one-pixel move must not restate the surround ({drift:.3f})"


class TestAutoTarget:
    """The separation `auto` asks for has to be an angle, and stay one however
    much of the sphere the file happens to store."""

    def scene(self):
        return 1.0 / torch.linspace(2.0, 40.0, 4096)[None]

    def chosen(self, spot, target_deg=vr180.TARGET_DEG):
        eyes, _ = stereo.auto_geometry(self.scene(), spot.width, spot.per_radian,
                                       vr180.auto_target(spot, target_deg))
        return eyes

    def test_a_full_frame_asks_for_its_share_of_180(self):
        assert vr180.auto_target(vr180.Patch(180.0, 180.0, 8, 8)) == pytest.approx(
            100.0 * vr180.TARGET_DEG / 180.0)

    def test_a_crop_asks_for_its_share_of_the_crop(self):
        """The share is of the frame, and the frame got smaller."""
        assert vr180.auto_target(vr180.Patch(65.5, 51.5, 8, 8)) == pytest.approx(
            100.0 * vr180.TARGET_DEG / 65.5)

    def test_the_angle_is_the_callers_to_choose(self):
        """`TARGET_DEG` is the default and not the only answer: it is the one
        number in `vr180` reasoned about rather than measured, so it has to be
        possible to disagree with it.  See `Settings.target_deg`."""
        spot = vr180.Patch(180.0, 180.0, 8, 8)
        assert vr180.auto_target(spot, 1.8) == pytest.approx(3 * vr180.auto_target(spot, 0.6))
        assert vr180.auto_target(spot) == vr180.auto_target(spot, vr180.TARGET_DEG)

    def test_a_wider_angle_asks_for_a_wider_baseline(self):
        """And it has to reach the geometry, not merely be stored: twice the
        angle is twice the separation, the scene being the same one."""
        spot = vr180.Patch(180.0, 180.0, 360, 360)
        assert self.chosen(spot, 1.2) == pytest.approx(2 * self.chosen(spot, 0.6), rel=1e-6)

    @pytest.mark.parametrize("span", [180.0, 96.0, 65.5, 40.3])
    def test_the_shared_geometry_hits_that_angle(self, span):
        """Feed `auto_geometry` the patch's pixels-per-radian and it should pick a
        baseline that spreads the scene over `TARGET_DEG` degrees -- whatever the
        patch, because degrees of arc do not care how the file was cut."""
        spot = vr180.Patch(span, span, 360, 360)
        lo, hi = torch.quantile(self.scene().flatten(), torch.tensor([0.02, 0.98]))
        assert (self.chosen(spot) / 1000.0) * float(hi - lo) == pytest.approx(
            math.radians(vr180.TARGET_DEG), rel=1e-3)

    def test_cropping_does_not_change_the_baseline(self):
        """The bug this replaced: pinned at 180, a 65-degree patch came out with
        under half the separation the same scene got in the square."""
        full = self.chosen(vr180.Patch(180.0, 180.0, 1768, 1768))
        crop = self.chosen(vr180.Patch(65.5, 51.5, 642, 505))
        assert crop == pytest.approx(full, rel=0.02)


class TestPlate:
    """The plate as `render` sees it: a backdrop that goes behind the picture and
    in front of the wash.  How it is built is `test_plate`'s business; what
    matters here is that putting one in changes only what it should."""

    def spot(self, w=48, h=48):
        return vr180.Patch(180.0, 180.0, w, h)

    def scene(self, spot):
        """A plate with something recognisable on it, covering more of the sphere
        than the photograph will."""
        colour = torch.rand(3, spot.height, spot.width)
        mask = torch.zeros(spot.height, spot.width, dtype=torch.bool)
        edge = spot.width // 6
        mask[edge:-edge, edge:-edge] = True
        return colour, mask

    def rendered(self, spot, wash=0.0, plate=None, rgb=None):
        rgb = torch.rand(3, 32, 32) if rgb is None else rgb
        return vr180.render(rgb, torch.full((32, 32), 0.5), 32.0, 65.0, 3.0, spot,
                            wash=wash, plate=plate)

    def test_both_eyes_get_exactly_the_same_plate(self):
        """The rule `surround` already sets, and for the same reason: a periphery
        carrying parallax of its own fights the real picture over where the eyes
        should converge."""
        spot = self.spot()
        left, right, mask = self.rendered(spot, plate=self.scene(spot))
        assert torch.equal(left[:, ~mask], right[:, ~mask])

    def test_the_picture_itself_is_untouched(self):
        """Deep inside the photograph the fade is 1, so the backdrop behind it
        cannot show through however bright it is."""
        spot = self.spot(96, 96)
        rgb = torch.rand(3, 32, 32)
        without = self.rendered(spot, rgb=rgb)
        with_plate = self.rendered(spot, plate=self.scene(spot), rgb=rgb)
        deep = vr180._falloff(without[2], spot) > 0.999
        assert bool(deep.any())
        assert torch.allclose(without[0][:, deep], with_plate[0][:, deep], atol=1e-6)

    def test_it_fills_what_the_photograph_did_not_reach(self):
        spot = self.spot()
        rgb = torch.rand(3, 32, 32)
        dark, _, mask = self.rendered(spot, rgb=rgb)
        lit, _, _ = self.rendered(spot, plate=self.scene(spot), rgb=rgb)
        assert float(lit[:, ~mask].mean()) > float(dark[:, ~mask].mean())

    def test_no_plate_and_no_wash_is_still_a_dark_sphere(self):
        """Nothing added means nothing changed: this is the path a clip takes
        when the pass was never asked for or could not run."""
        spot = self.spot()
        left, _, mask = self.rendered(spot)
        assert float(left[:, ~mask].max()) == pytest.approx(0.0, abs=1e-6)

    def test_it_meets_the_picture_without_a_step(self):
        """The measurement that set `PLATE_DIM`.  A plate is the scene itself at
        the exposure `_match_gain` has just matched to this frame, so dimming it
        does not read as restraint -- it draws a vignette round real footage,
        exactly where the eye is already looking.  At 0.7 the step measured 30%."""
        spot = self.spot(96, 96)
        colour = torch.full((3, spot.height, spot.width), 0.5)
        mask = torch.ones(spot.height, spot.width, dtype=torch.bool)
        rgb = torch.full((3, 32, 32), 0.5)
        left, _, seen = vr180.render(rgb, torch.full((32, 32), 0.5), 32.0, 65.0, 3.0, spot,
                                     plate=(colour, mask))
        fade = vr180._falloff(seen, spot)
        inside, outside = fade > 0.999, fade < 0.001
        assert bool(inside.any()) and bool(outside.any())
        here, there = float(left[:, inside].mean()), float(left[:, outside].mean())
        assert abs(here - there) / here < 0.05

    def test_a_dimmer_plate_is_still_possible_for_whoever_wants_one(self):
        spot = self.spot(64, 64)
        colour = torch.ones(3, spot.height, spot.width)
        mask = torch.ones(spot.height, spot.width, dtype=torch.bool)
        rgb = torch.ones(3, 32, 32)
        bright, _, seen = vr180.render(rgb, torch.full((32, 32), 0.5), 32.0, 65.0, 3.0, spot,
                                       plate=(colour, mask), plate_dim=1.0)
        dim, _, _ = vr180.render(rgb, torch.full((32, 32), 0.5), 32.0, 65.0, 3.0, spot,
                                 plate=(colour, mask), plate_dim=0.5)
        far = vr180._falloff(seen, spot) < 0.001
        assert float(dim[:, far].mean()) < float(bright[:, far].mean())

    def test_the_wash_is_spread_out_of_the_plate_rather_than_the_frame(self):
        """So the light at the edge of the sphere is the light of whatever the
        camera turned to see in that direction, and the two meet without a step
        because one is made from the other."""
        spot = self.spot(96, 96)
        colour, mask = self.scene(spot)
        colour[0] = 1.0  # a plate that is redder than any frame will be
        colour[1:] = 0.0
        left, _, seen = self.rendered(spot, wash=0.6, plate=(colour, mask))
        far = ~mask
        assert bool(far.any())
        assert float(left[0][far].mean()) > float(left[1][far].mean())

    def test_a_plate_at_a_different_exposure_is_matched_to_the_frame(self):
        """A phone's automatic exposure walks across a shot, so a plate combined
        over the whole of one sits at the average and the frame in front of it
        does not.  Left alone that is a visible step at the edge of the picture."""
        spot = self.spot(96, 96)
        overlap = torch.ones(spot.height, spot.width, dtype=torch.bool)
        bright = vr180._match_gain(torch.full((3, spot.height, spot.width), 0.2),
                                   torch.full((3, spot.height, spot.width), 0.6), overlap)
        assert float(bright.mean()) == pytest.approx(0.4, abs=1e-3)  # clamped at 2x

    def test_the_gain_cannot_run_away_on_a_sliver_of_overlap(self):
        spot = self.spot(32, 32)
        overlap = torch.zeros(spot.height, spot.width, dtype=torch.bool)
        overlap[0, 0] = True
        colour = torch.full((3, spot.height, spot.width), 1e-9)
        matched = vr180._match_gain(colour, torch.ones(3, spot.height, spot.width), overlap)
        assert torch.equal(matched, colour)  # too dark to divide by, so left alone

    def test_no_overlap_leaves_the_plate_alone(self):
        spot = self.spot(32, 32)
        colour = torch.rand(3, spot.height, spot.width)
        none = torch.zeros(spot.height, spot.width, dtype=torch.bool)
        assert torch.equal(vr180._match_gain(colour, colour, none), colour)


class TestRotatedProjection:
    """`project` learned to turn the camera, for the plate's sake.  A still and
    every live frame leave it None, so the one thing that must stay true is that
    None and an identity rotation are the same picture."""

    def test_identity_changes_nothing(self):
        source = torch.rand(3, 40, 60)
        spot = vr180.Patch(180.0, 180.0, 48, 48)
        plain, plain_mask = vr180.project(source, 50.0, spot)
        turned, turned_mask = vr180.project(source, 50.0, spot, rotation=torch.eye(3))
        assert torch.equal(plain, turned) and torch.equal(plain_mask, turned_mask)

    def test_turning_the_camera_moves_the_picture(self):
        source = torch.rand(3, 40, 60)
        spot = vr180.Patch(180.0, 180.0, 48, 48)
        angle = math.radians(30.0)
        yaw = torch.tensor([[math.cos(angle), 0.0, math.sin(angle)],
                            [0.0, 1.0, 0.0],
                            [-math.sin(angle), 0.0, math.cos(angle)]])
        _, straight = vr180.project(source, 50.0, spot)
        _, turned = vr180.project(source, 50.0, spot, rotation=yaw)
        assert not torch.equal(straight, turned)
        # The same amount of sphere, just somewhere else on it.
        assert abs(int(straight.sum()) - int(turned.sum())) < 0.25 * int(straight.sum())


class TestSurroundRamp:
    """The wash meets the picture at the picture's own brightness.

    Without it the boundary is a cliff -- full strength inside, `SURROUND_DIM`
    outside -- and it sits exactly where a viewer is guaranteed to be looking.
    """

    def spot(self, side=160):
        return vr180.Patch(180.0, 180.0, side, side)

    def parts(self, side=160):
        spot = self.spot(side)
        mask = torch.zeros(side, side, dtype=torch.bool)
        mask[side // 3: 2 * side // 3, side // 3: 2 * side // 3] = True
        return spot, mask, torch.rand(3, side, side)

    def rim(self, mask, width=15):
        import torch.nn.functional as F

        grown = F.max_pool2d(mask.float()[None, None], 2 * width + 1, 1, width)[0, 0] > 0.5
        return grown & ~mask

    def test_it_is_brighter_where_it_leaves_the_picture(self):
        spot, mask, colour = self.parts()
        eased = vr180.surround(colour, mask, spot, dim=0.4)
        flat = vr180.surround(colour, mask, spot, dim=0.4, ramp_deg=0)
        edge = self.rim(mask)
        assert float(eased[:, edge].mean()) > float(flat[:, edge].mean())

    def test_and_settles_to_what_was_asked_for_further_out(self):
        """The knob still means what it meant; the ramp only changes where."""
        spot, mask, colour = self.parts(240)
        eased = vr180.surround(colour, mask, spot, dim=0.3)
        flat = vr180.surround(colour, mask, spot, dim=0.3, ramp_deg=0)
        # Past the ramp's own radius, which is 33px on a 240 patch -- growing by
        # 90 covers the whole thing and leaves nothing to measure.
        far = ~self.rim(mask, 50) & ~mask
        assert bool(far.any())
        assert float(eased[:, far].mean()) == pytest.approx(float(flat[:, far].mean()), rel=0.15)

    def test_full_brightness_needs_no_ramp_at_all(self):
        """At 1.0 there is nothing to ease between, and the two paths agree."""
        spot, mask, colour = self.parts()
        assert torch.allclose(vr180.surround(colour, mask, spot, dim=1.0),
                              vr180.surround(colour, mask, spot, dim=1.0, ramp_deg=0))

    def test_an_overbright_wash_still_screens_rather_than_clamps(self):
        """`_expose` above 1 reaches for white without arriving, and the ramp
        must not turn that back into a multiply-and-clamp."""
        spot, mask, colour = self.parts()
        bright = vr180.surround(colour, mask, spot, dim=2.0)
        assert float(bright.max()) <= 1.0
        assert float(bright[:, ~mask].mean()) > float(
            vr180.surround(colour, mask, spot, dim=1.0)[:, ~mask].mean())
