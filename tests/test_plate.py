"""The periphery pass: what it registers, what it combines, and what it holds still.

Two of these are worth more than the rest.  `TestNoBoil` asserts the property the
whole module exists to provide -- that a shot which did not move produces a
periphery that does not move, identically rather than nearly -- and it is an
exact assertion because the design makes it one.  `TestFallsBack` asserts the
other half of the bargain: every way this can fail lands on what the app did
before it existed, so switching it on can cost a clip nothing.

Nothing here needs the depth model, and only the ring needs the weights.
"""

import math

import cv2
import numpy as np
import pytest
import torch

from stereocraft import plate, vr180


def scene(width=640, height=360, seed=0):
    """A picture with enough structure for ORB to lock onto and for a median to
    have an opinion about."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), np.uint8)
    image[:height // 2] = (200, 170, 120)
    image[height // 2:] = (60, 110, 70)
    for _ in range(90):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        cv2.circle(image, (x, y), int(rng.integers(6, 26)),
                   tuple(int(v) for v in rng.integers(30, 200, 3)), -1)
    return cv2.GaussianBlur(image, (0, 0), 1.2)


def yaw(degrees):
    turn = math.radians(degrees)
    return np.array([[math.cos(turn), 0.0, math.sin(turn)],
                     [0.0, 1.0, 0.0],
                     [-math.sin(turn), 0.0, math.cos(turn)]])


def panned(image, angles, focal_px=None):
    """The same scene seen from a camera yawed by each of `angles`.

    Warped by `K R^-1 K^-1` rather than shifted sideways, which is the whole
    difference between a pan and a slide: a camera that turns does not translate
    its picture, it applies a homography with perspective in it, and a test built
    on a sliding crop measures the wrong thing.
    """
    height, width = image.shape[:2]
    focal_px = focal_px or plate.focal_from_35mm(plate.DEFAULT_FOCAL_35MM, width)
    matrix = plate.intrinsics(width, height, focal_px)
    out = []
    for angle in angles:
        homography = matrix @ np.linalg.inv(yaw(angle)) @ np.linalg.inv(matrix)
        out.append(cv2.warpPerspective(image, homography, (width, height),
                                       borderMode=cv2.BORDER_REFLECT))
    return out


def track_of(frames, focal_px=None):
    height, width = frames[0].shape[:2]
    focal_px = focal_px or plate.focal_from_35mm(plate.DEFAULT_FOCAL_35MM, width)
    track = plate.Track(plate.intrinsics(width, height, focal_px))
    for frame in frames:
        track.add(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    return track


def yaw_of(rotation):
    return math.degrees(math.atan2(rotation[0, 2], rotation[2, 2]))


def as_samples(frames, rotations):
    return [(torch.from_numpy(f).permute(2, 0, 1).float().div(255.0), r)
            for f, r in zip(frames, rotations)]


class TestRegistration:
    def test_recovers_a_known_pan(self):
        angles = list(range(0, 25, 2))
        track = track_of(panned(scene(), angles))
        assert track.registered == len(angles)
        for wanted, got in zip(angles, track.rotations):
            assert yaw_of(got) == pytest.approx(wanted, abs=0.5)

    def test_the_first_frame_is_the_origin(self):
        track = track_of(panned(scene(), [0, 5, 10]))
        assert np.allclose(track.rotations[0], np.eye(3))

    def test_a_frame_it_cannot_place_holds_rather_than_snapping_back(self):
        """A single blurred frame is a gap to bridge.  Jumping to identity would
        swing the whole periphery back to where the shot started, for one frame,
        which is far more visible than leaving it where it was."""
        frames = panned(scene(), [0, 8, 16])
        frames.append(np.zeros_like(frames[0]))  # nothing to match at all
        track = track_of(frames)
        assert track.registered == 3
        assert np.allclose(track.rotations[-1], track.rotations[-2])

    def test_a_rotation_is_refused_when_the_fit_is_not_one(self):
        """A homography fitted to something moving through the frame rather than
        to the frame moving is not a rotation, and the singular values say so."""
        matrix = plate.intrinsics(640, 360, 500.0)
        squashed = matrix @ np.diag([1.0, 0.4, 1.0]) @ np.linalg.inv(matrix)
        assert plate.rotation_from(squashed, matrix) is None

    def test_no_homography_is_no_rotation(self):
        assert plate.rotation_from(None, plate.intrinsics(640, 360, 500.0)) is None


class TestHold:
    def test_movement_too_small_to_see_is_not_movement(self):
        rotations = [yaw(0.0), yaw(0.004), yaw(0.008), yaw(0.012)]
        held = plate._held(rotations)
        assert all(np.array_equal(r, held[0]) for r in held)

    def test_real_movement_is_still_followed(self):
        rotations = [yaw(a) for a in (0.0, 1.0, 2.0, 3.0)]
        held = plate._held(rotations)
        assert [round(yaw_of(r)) for r in held] == [0, 1, 2, 3]

    def test_a_slow_pan_is_followed_rather_than_ignored(self):
        """Held against the last bearing emitted, not the last measured, so
        steps below the threshold accumulate into one that is above it instead
        of being discarded one by one."""
        step = plate.STILL_DEG * 0.6
        held = plate._held([yaw(step * i) for i in range(40)])
        assert yaw_of(held[-1]) == pytest.approx(step * 39, abs=plate.STILL_DEG * 2)


class TestCuts:
    def test_the_first_frame_is_never_a_cut(self):
        assert not plate.is_cut(None, scene())

    def test_a_shot_carrying_on_is_not_a_cut(self):
        frames = panned(scene(), [0, 2])
        small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (64, 64)) for f in frames]
        assert not plate.is_cut(*small)

    def test_a_different_picture_is(self):
        a = cv2.resize(cv2.cvtColor(scene(seed=1), cv2.COLOR_RGB2GRAY), (64, 64))
        b = cv2.resize(cv2.cvtColor(scene(seed=2)[::-1], cv2.COLOR_RGB2GRAY), (64, 64))
        assert plate.is_cut(a, b)


class TestMosaic:
    def test_a_pan_covers_more_sphere_than_the_frame_it_started_from(self):
        """The point of the whole exercise, and measured in solid angle rather
        than in pixels for the reason `vr180.coverage` gives."""
        frames = panned(scene(), list(range(0, 46, 3)))
        track = track_of(frames)
        spot = plate.plate_patch(512, 256)
        _, wide = plate.mosaic(as_samples(frames, track.smoothed()), 400.0, spot)
        _, narrow = plate.mosaic(as_samples(frames[:1], [np.eye(3)]), 400.0, spot)
        assert vr180.coverage(wide, spot) > 1.4 * vr180.coverage(narrow, spot)

    def test_a_still_camera_covers_what_one_frame_covers(self):
        frames = [scene()] * 6
        spot = plate.plate_patch(512, 256)
        _, many = plate.mosaic(as_samples(frames, [np.eye(3)] * 6), 400.0, spot)
        _, one = plate.mosaic(as_samples(frames[:1], [np.eye(3)]), 400.0, spot)
        assert torch.equal(many, one)

    def test_the_median_throws_the_walker_away(self):
        """A running mean would smear anyone crossing the shot into a ghost
        stretched across the periphery, which is exactly where the eye looks."""
        base = scene(seed=5)
        base[:, :, 0] = 20  # a scene with no red in it
        frames = []
        for i in range(20):
            frame = base.copy()
            cv2.circle(frame, (60 + i * 28, 180), 40, (255, 0, 0), -1)
            frames.append(frame)
        spot = plate.plate_patch(512, 256)
        colour, covered = plate.mosaic(as_samples(frames, [np.eye(3)] * len(frames)), 400.0, spot)
        assert float(colour[0][covered].max()) < 0.35

    def test_nothing_at_all_is_an_empty_plate(self):
        spot = plate.plate_patch(64, 32)
        colour, covered = plate.mosaic([], 400.0, spot)
        assert colour.shape == (3, 32, 64) and not bool(covered.any())


class TestRoom:
    """Holding the near field out, which is what keeps a plate to the part of
    the scene that stays put."""

    def test_the_mosaic_can_be_told_what_belongs_in_it(self):
        frames = [scene()] * 4
        keep = torch.zeros(360, 640, dtype=torch.bool)
        keep[:120] = True  # only the top of the frame is "room"
        spot = plate.plate_patch(512, 256)
        _, all_of_it = plate.mosaic(as_samples(frames, [np.eye(3)] * 4), 400.0, spot)
        _, room = plate.mosaic([(f, r, keep) for f, r in
                                as_samples(frames, [np.eye(3)] * 4)], 400.0, spot)
        assert int(room.sum()) < int(all_of_it.sum())
        assert not bool((room & ~all_of_it).any())  # never more than was there

    def test_registration_can_be_told_where_to_look(self):
        """The bug this exists for: on a close-up the subject is most of the
        features, so an unmasked fit measures her motion and calls it the
        camera's -- 64 degrees of yaw on a clip whose camera turned 21."""
        frames = panned(scene(), [0, 6, 12])
        keep = np.zeros((360, 640), bool)
        keep[:150] = True
        track = plate.Track(plate.intrinsics(640, 360,
                                             plate.focal_from_35mm(plate.DEFAULT_FOCAL_35MM, 640)))
        for f in frames:
            track.add(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), keep=keep)
        assert track.registered >= 2


class TestInterpolate:
    def test_frames_between_samples_are_filled_in(self):
        turns = [yaw(0.0), yaw(10.0)]
        out = plate._interpolate(turns, [0, 10], 11)
        assert len(out) == 11
        assert yaw_of(out[5]) == pytest.approx(5.0, abs=0.5)

    def test_before_and_after_the_samples_it_holds(self):
        out = plate._interpolate([yaw(0.0), yaw(8.0)], [2, 6], 10)
        assert yaw_of(out[0]) == pytest.approx(0.0, abs=0.01)
        assert yaw_of(out[9]) == pytest.approx(8.0, abs=0.01)

    def test_nothing_registered_is_a_still_camera(self):
        out = plate._interpolate([], [], 5)
        assert len(out) == 5 and all(np.allclose(r, np.eye(3)) for r in out)


class TestSampler:
    def test_it_keeps_a_bounded_number_however_long_the_shot(self):
        sampler = plate._Sampler()
        for _ in range(5000):
            sampler.offer(np.zeros((4, 4, 3), np.uint8))
        assert len(sampler.kept) <= 2 * plate.SAMPLES
        assert len(sampler.frames()) <= plate.SAMPLES

    def test_what_it_keeps_is_spread_over_the_whole_shot(self):
        sampler = plate._Sampler()
        for index in range(1000):
            sampler.offer(np.full((2, 2, 3), index % 256, np.uint8))
        positions = [entry[0] for entry in sampler.frames()]
        assert positions[0] < 50 and positions[-1] > 700

    def test_a_short_shot_keeps_everything_it_saw(self):
        sampler = plate._Sampler()
        for _ in range(5):
            sampler.offer(np.zeros((2, 2, 3), np.uint8))
        assert len(sampler.frames()) == 5


class TestPlates:
    def _plates(self, rotations, colour=None):
        spot = plate.plate_patch(256, 128)
        if colour is None:
            colour = torch.rand(3, 128, 256)
        covered = torch.zeros(128, 256, dtype=torch.bool)
        covered[40:90, 80:180] = True
        return plate.Plates([(colour, covered)], [(0, r) for r in rotations], spot)

    def test_a_frame_past_the_end_reads_the_last_one(self):
        plates = self._plates([np.eye(3)] * 3)
        spot = vr180.patch(0, 0, 0, size=64)
        assert torch.equal(plates.at(99, spot)[0], plates.at(2, spot)[0])

    def test_no_frames_is_nothing_to_show(self):
        spot = plate.plate_patch(64, 32)
        assert plate.Plates([], [], spot).at(0, spot) is None

    def test_the_mask_excludes_what_only_half_landed_on_the_plate(self):
        """A bilinear tap straddling the plate's edge would drag black into
        view, so a pixel counts as covered only when its whole tap was."""
        plates = self._plates([np.eye(3)])
        spot = vr180.patch(0, 0, 0, size=128)
        colour, mask = plates.at(0, spot)
        assert bool(mask.any()) and not bool(mask.all())

    def test_it_brings_across_only_the_shot_being_looked_at(self):
        """A clip with a hundred cuts holds a hundred plates, and only ever one
        of them is on screen."""
        plates = self._plates([np.eye(3)])
        plates.to(torch.device("cpu"))
        assert plates.device == torch.device("cpu")
        spot = vr180.patch(0, 0, 0, size=64)
        assert plates.at(0, spot)[0].device.type == "cpu"


class TestNoBoil:
    """The property the module exists for, asserted rather than eyeballed."""

    def _plates(self, rotations):
        spot = plate.plate_patch(256, 128)
        colour = torch.rand(3, 128, 256)
        covered = torch.zeros(128, 256, dtype=torch.bool)
        covered[30:100, 60:200] = True
        return plate.Plates([(colour, covered)], [(0, r) for r in rotations], spot)

    def test_a_camera_that_did_not_move_gives_a_periphery_that_does_not_either(self):
        """Identical, not merely close.  `_held` is what makes this exact, and it
        is exact so that a regression in it cannot hide inside a tolerance."""
        plates = self._plates(plate._held([yaw(a) for a in (0.0, 0.003, -0.002, 0.004)]))
        spot = vr180.patch(0, 0, 0, size=96)
        first = plates.at(0, spot)
        for index in range(1, 4):
            later = plates.at(index, spot)
            assert torch.equal(first[0], later[0])
            assert torch.equal(first[1], later[1])

    def test_a_camera_that_did_move_moves_the_periphery_with_it(self):
        """The other half: holding still must not mean refusing to turn."""
        plates = self._plates([yaw(0.0), yaw(12.0)])
        spot = vr180.patch(0, 0, 0, size=96)
        before, after = plates.at(0, spot)[0], plates.at(1, spot)[0]
        assert not torch.equal(before, after)

    def test_the_same_shot_built_twice_comes_out_the_same(self):
        """Nothing here is sampled, so there is no seed to pin and no run to
        differ from another -- which is a stronger guarantee than a pinned seed
        and the reason a deterministic model was chosen over a diffusion one."""
        frames = panned(scene(), [0, 4, 8, 12])
        spot = plate.plate_patch(256, 128)
        samples = as_samples(frames, track_of(frames).smoothed())
        once = plate.mosaic(samples, 400.0, spot)
        twice = plate.mosaic(samples, 400.0, spot)
        assert torch.equal(once[0], twice[0]) and torch.equal(once[1], twice[1])


class TestFallsBack:
    def test_a_flat_projection_wants_nothing_to_do_with_this(self):
        from stereocraft.pipeline import VideoSettings

        clip = type("Clip", (), {"width": 640, "height": 360, "fps": 30.0})()
        assert not plate.wanted(clip, VideoSettings(projection="flat", outpaint=True))

    def test_nor_does_a_clip_that_did_not_ask(self):
        from stereocraft.pipeline import VideoSettings

        clip = type("Clip", (), {"width": 640, "height": 360, "fps": 30.0})()
        assert not plate.wanted(clip, VideoSettings(projection="vr180", outpaint=False))

    def test_the_decode_never_enlarges_a_small_clip(self):
        clip = type("Clip", (), {"width": 320, "height": 180})()
        assert plate.decode_size(clip) == (320, 180)

    def test_a_big_clip_is_read_down_to_the_working_width(self):
        clip = type("Clip", (), {"width": 3840, "height": 2160})()
        assert plate.decode_size(clip) == (plate.DECODE_WIDTH, 360)

    def test_a_shot_with_no_frames_in_it_is_an_empty_plate_not_a_crash(self):
        spot = plate.plate_patch(64, 32)
        reference, seen, colour, covered, room = plate._combine(
            plate._Sampler(), [], 400.0, spot)
        assert reference.shape == colour.shape == (3, 32, 64)
        assert not any(bool(m.any()) for m in (seen, covered, room))


class TestWeights:
    def test_the_checkpoint_is_checked_before_it_is_believed(self, tmp_path):
        impostor = tmp_path / "big-lama.pt"
        impostor.write_bytes(b"not the weights")
        with pytest.raises(RuntimeError, match="not the checkpoint"):
            plate.verify(str(impostor))

    @pytest.mark.slow
    def test_a_widened_plate_reaches_further_than_the_camera_did(self):
        """End to end through the real path: mosaic, canvas, model, sphere.
        Uses LaMa because it is the one backend small enough to be bundled and
        so the one always available; what it paints is not the point here, only
        that the plate comes back covering more sphere than it went in with."""
        from stereocraft import outpaint, vr180

        spot = plate.plate_patch(512, 256)
        image = scene(seed=11)
        colour, covered = plate.mosaic(as_samples([image], [np.eye(3)]), 400.0, spot)
        painter, _ = outpaint.load("lama", device="cpu")
        wider, reached = outpaint.widen(colour, covered, covered, (640, 360), 400.0,
                                        painter, reach_deg=25.0)
        assert vr180.coverage(reached, spot) > 1.4 * vr180.coverage(covered, spot)
        assert torch.equal(wider[:, covered], colour[:, covered])  # the real part is kept


class TestChoosingABackend:
    """Which model gets used, and -- the half that matters on a machine that has
    to live within its means -- which one gets declined."""

    def test_a_backend_too_large_for_the_machine_steps_aside(self, monkeypatch):
        """The failure this prevents is not a slow conversion. FLUX offloaded
        keeps 13.4 GB in host memory, and inside a WSL instance given half a
        32 GB machine that takes the virtual machine down rather than the
        process -- which reads as the distro crashing and explains nothing."""
        from stereocraft import budget, outpaint

        monkeypatch.setattr(outpaint, "available", lambda app_dir=None: ("flux", "sdxl", "lama"))
        # What this machine actually reports free inside WSL, against FLUX's 15.
        monkeypatch.setattr(budget, "_available_ram", lambda: int(14e9))
        assert outpaint.choose("auto") == "sdxl"

    def test_a_roomy_machine_gets_the_best_one(self, monkeypatch):
        from stereocraft import budget, outpaint

        monkeypatch.setattr(outpaint, "available", lambda app_dir=None: ("flux", "sdxl", "lama"))
        monkeypatch.setattr(budget, "_available_ram", lambda: int(64e9))
        assert outpaint.choose("auto") == "flux"

    def test_naming_one_takes_it_at_face_value(self, monkeypatch):
        """Asked for by name it is not second-guessed: the size check is what
        `auto` uses to pick, not a veto over the person running it."""
        from stereocraft import budget, outpaint

        monkeypatch.setattr(outpaint, "available", lambda app_dir=None: ("flux", "sdxl"))
        monkeypatch.setattr(budget, "_available_ram", lambda: int(4e9))
        assert outpaint.choose("flux") == "flux"

    def test_asking_for_one_that_is_not_there_gets_nothing(self, monkeypatch):
        from stereocraft import outpaint

        monkeypatch.setattr(outpaint, "available", lambda app_dir=None: ("lama",))
        assert outpaint.choose("flux") is None

    def test_nothing_at_all_is_nothing(self, monkeypatch):
        from stereocraft import outpaint

        monkeypatch.setattr(outpaint, "available", lambda app_dir=None: ())
        assert outpaint.choose("auto") is None


class TestPlatesWaitAsBytes:
    """A plate is built in float and then held until every frame of the clip has
    been read, because the painter cannot be loaded while the clip is being
    decoded.  On a long film that is hundreds of shots waiting at 57 MB each, and
    it is the one cost in this pass that grows with the length of the film rather
    than the size of a frame.  Held as bytes it is 19 MB and identical: both
    colour planes are medians of 8-bit frames, and the finished plate is stored
    as uint8 regardless."""

    def _parts(self, height=8, width=16):
        colour = torch.rand(3, height, width)
        mask = torch.zeros(height, width, dtype=torch.bool)
        mask[: height // 2] = True
        return torch.rand(3, height, width), mask, colour, mask.clone(), mask.clone()

    def test_the_colour_planes_are_bytes_and_the_masks_untouched(self):
        reference, seen, colour, covered, room = plate._stow(self._parts())
        assert reference.dtype is torch.uint8
        assert colour.dtype is torch.uint8
        assert seen.dtype is torch.bool and covered.dtype is torch.bool
        assert room.dtype is torch.bool

    def test_which_is_a_third_of_the_memory(self):
        parts = self._parts(256, 512)
        before = sum(t.numel() * t.element_size() for t in parts)
        after = sum(t.numel() * t.element_size() for t in plate._stow(parts))
        assert after < before / 2.5

    def test_and_comes_back_where_it_was_to_a_level(self):
        parts = self._parts()
        thawed = plate._thaw(plate._stow(parts))
        for original, round_tripped in zip(parts[:1] + parts[2:3], thawed[:1] + thawed[2:3]):
            assert torch.allclose(original, round_tripped, atol=1.0 / 255)

    def test_stowing_twice_does_not_scale_the_picture_again(self):
        """`_u8` has to pass an already-converted plane straight through: the
        second multiply would take a byte value of 200 and ask for 51000."""
        once = plate._stow(self._parts())
        assert all(torch.equal(a, b) for a, b in zip(once, plate._stow(once)))
