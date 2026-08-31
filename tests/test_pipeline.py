"""End to end, with the depth model loaded.  Slow, and the part that matters."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from stereocraft.pipeline import Settings, save_depth_map
import torch

pytestmark = pytest.mark.slow


class TestDepthMap:
    def test_uses_the_whole_range_so_it_can_be_seen(self, tmp_path):
        """It once stored centimetres, which is correct and unreadable: a scene
        two to twenty-five metres away came out under 4% of full scale, which is
        a black rectangle."""
        inverse = torch.linspace(1 / 25.0, 1 / 2.0, 64).reshape(8, 8)
        path = save_depth_map(inverse, tmp_path / "d.png")
        d = np.array(Image.open(path))
        assert d.max() > 60000, "the nearest thing should be near white"
        assert d.min() < 5000, "the furthest should be near black"

    def test_the_metres_survive_in_the_metadata(self, tmp_path):
        inverse = torch.linspace(1 / 25.0, 1 / 2.0, 64).reshape(8, 8)
        image = Image.open(save_depth_map(inverse, tmp_path / "d.png"))
        near = float(image.text["stereocraft:near_m"])
        far = float(image.text["stereocraft:far_m"])
        assert near == pytest.approx(2.0, rel=1e-3) and far == pytest.approx(25.0, rel=1e-3)

        d = np.array(image).astype(float)
        recovered = far - (d / 65535) * (far - near)
        assert recovered.min() == pytest.approx(2.0, rel=1e-2)
        assert recovered.max() == pytest.approx(25.0, rel=1e-2)

    def test_near_is_brighter_than_far(self, tmp_path):
        inverse = torch.tensor([[1 / 20.0, 1 / 2.0]])
        d = np.array(Image.open(save_depth_map(inverse, tmp_path / "d.png")))
        assert d[0, 1] > d[0, 0]

    def test_a_scene_at_one_distance_does_not_divide_by_zero(self, tmp_path):
        flat = torch.full((4, 4), 0.25)
        d = np.array(Image.open(save_depth_map(flat, tmp_path / "d.png")))
        assert d.shape == (4, 4)


class TestPhoto:
    def test_converts_and_the_eyes_differ(self, converter, photo, tmp_path):
        info = converter.convert(photo, tmp_path / "out.jpg")
        out = np.asarray(Image.open(info["output"])).astype(int)
        w = out.shape[1] // 2
        left, right = out[:, :w], out[:, w:]
        assert not np.array_equal(left, right), "a stereo pair whose eyes match is not one"
        assert left.std() > 5 and right.std() > 5, "neither eye may be blank"

    def test_reports_the_geometry_it_chose(self, converter, photo, tmp_path):
        info = converter.convert(photo, tmp_path / "out.jpg")
        assert info["eyes_mm"] > 0 and info["focus_m"] > 0

    def test_the_pair_is_twice_a_frame_less_the_trim(self, converter, photo, tmp_path):
        info = converter.convert(photo, tmp_path / "out.jpg")
        width, _ = info["output_size"]
        assert width % 2 == 0 and width <= 2 * 320


class TestVr180:
    """The spherical projection, run for real rather than on synthetic tensors."""

    def convert(self, converter, photo, out, **kwargs):
        was = converter.settings
        converter.settings = Settings(projection="vr180", **kwargs)
        try:
            return converter.convert(photo, out)
        finally:
            converter.settings = was

    def test_the_pair_is_the_patch_side_by_side(self, converter, photo, tmp_path):
        info = self.convert(converter, photo, tmp_path / "vr.jpg", vr180_size=192)
        spot = info["patch"]
        assert info["output_size"] == (2 * spot.width, spot.height)
        assert spot.width == spot.height == 192, "the default frame is the square"

    def test_the_frame_is_mostly_dark(self, converter, photo, tmp_path):
        """The price of the format: a rectilinear lens cannot fill a hemisphere,
        and storing only the part it reached needs a player that reads where the
        part belongs.  None does, so the dark is written."""
        info = self.convert(converter, photo, tmp_path / "vr.jpg", vr180_size=192)
        out = np.asarray(Image.open(info["output"])).astype(int)
        assert (out.max(axis=2) > 8).mean() < 0.35

    def test_resolution_does_not_change_how_much_sphere_is_real(
            self, converter, photo, tmp_path):
        """More pixels, same picture, same place.  If this moves, the sizing has
        quietly changed the field of view rather than just the file size."""
        small = self.convert(converter, photo, tmp_path / "a.jpg", vr180_size=192)
        large = self.convert(converter, photo, tmp_path / "b.jpg", vr180_size=384)
        assert small["coverage"] == pytest.approx(large["coverage"], rel=0.05)

    def test_the_eyes_differ_but_the_void_does_not(self, converter, photo, tmp_path):
        """Where there is picture the two eyes must disagree; where there is
        nothing they must agree exactly, both being black."""
        info = self.convert(converter, photo, tmp_path / "vr.jpg", vr180_size=192)
        out = np.asarray(Image.open(info["output"])).astype(int)
        left, right = out[:, :192], out[:, 192:]
        assert not np.array_equal(left, right)
        corner = (slice(0, 20), slice(0, 20))  # a pole, which no lens ever reaches
        assert left[corner].max() < 8 and right[corner].max() < 8

    def test_it_says_where_on_the_sphere_it_belongs(self, converter, photo, tmp_path):
        """A hemisphere is half a sphere, so there is a placement to record even
        when the frame is the whole square the format asks for."""
        info = self.convert(converter, photo, tmp_path / "vr.jpg", vr180_size=192)
        written = Path(info["output"]).read_bytes()
        spot = info["patch"]
        assert b"ns.google.com/photos" in written
        assert f'CroppedAreaLeftPixels="{spot.left}"'.encode() in written
        assert f'FullPanoWidthPixels="{spot.full_width}"'.encode() in written

    def test_a_flat_photo_says_nothing_of_the_kind(self, converter, photo, tmp_path):
        info = converter.convert(photo, tmp_path / "flat.jpg")
        assert b"ns.google.com/photos" not in Path(info["output"]).read_bytes()
        assert info["coverage"] is None

    def test_the_surround_lights_the_dark_without_claiming_it_is_real(
            self, converter, photo, tmp_path):
        """It fills the frame and changes nothing about the measurement: the
        coverage figure counts what the camera saw, not what was painted in."""
        plain = self.convert(converter, photo, tmp_path / "a.jpg", vr180_size=192)
        washed = self.convert(converter, photo, tmp_path / "b.jpg", vr180_size=192,
                              vr180_surround=0.45)
        lit = lambda p: (np.asarray(Image.open(p)).max(axis=2) > 8).mean()
        assert lit(plain["output"]) < 0.35
        assert lit(washed["output"]) > 0.9
        assert washed["coverage"] == pytest.approx(plain["coverage"], rel=1e-6)

    def test_the_surround_is_the_same_in_both_eyes(self, converter, photo, tmp_path):
        """A periphery with a parallax of its own would fight the picture over
        where the viewer's eyes converge, and there is no true disparity to give
        it -- nothing was ever there to have one."""
        info = self.convert(converter, photo, tmp_path / "vr.jpg", vr180_size=192,
                            vr180_surround=0.45, fmt="png")
        out = np.asarray(Image.open(info["output"])).astype(int)
        left, right = out[:, :192], out[:, 192:]
        corner = (slice(0, 24), slice(0, 24))  # a pole, which no lens ever reaches
        assert np.array_equal(left[corner], right[corner])

    def test_reports_how_much_of_the_sphere_is_real(self, converter, photo, tmp_path):
        info = self.convert(converter, photo, tmp_path / "vr.jpg", vr180_size=192)
        assert 0.0 < info["coverage"] < 0.5, "a rectilinear photo cannot fill a hemisphere"

    def test_says_when_the_lens_was_only_assumed(self, converter, photo, tmp_path):
        """A wrong focal length rescales a flat scene harmlessly and puts a
        spherical one at the wrong apparent size, so a guess has to be owned up
        to rather than quietly used."""
        info = self.convert(converter, photo, tmp_path / "vr.jpg", vr180_size=128)
        assert info["lens"] == "assumed", "the fixture photo carries no EXIF"

    def test_auto_sizing_gives_the_ceiling_not_the_photo_s_own_size(
            self, converter, photo, tmp_path):
        """A headset shows 25 pixels a degree whatever it is handed, so a small
        photo gets the same frame a large one does.  What the photo could
        actually fill is `vr180.natural_size`, and the gap is what the upscaler
        exists to close."""
        from stereocraft import vr180
        info = self.convert(converter, photo, tmp_path / "vr.jpg")
        assert info["patch"].width == Settings.vr180_cap
        assert vr180.natural_size(320 * 28 / 36) < info["patch"].width, "and it is short"

    def test_the_depth_setting_reaches_the_sphere(self, converter, photo, tmp_path):
        """`--target` is a percentage of frame width and means nothing here, so
        the spherical path reads `target_deg` -- and used to read neither, taking
        `vr180.TARGET_DEG` whatever the caller asked for.  The separation is what
        proves it arrived: the baseline `auto` settles on has to scale with it.
        """
        gentle = self.convert(converter, photo, tmp_path / "a.jpg", vr180_size=192,
                              target_deg=0.4)
        strong = self.convert(converter, photo, tmp_path / "b.jpg", vr180_size=192,
                              target_deg=1.6)
        assert strong["eyes_mm"] == pytest.approx(4 * gentle["eyes_mm"], rel=1e-3)

    def test_the_ceiling_reaches_it_too(self, converter, photo, tmp_path):
        """The other half of the same fault: `limit_deg` was left at its default
        however the caller set it, so a raised target could not get past it."""
        from stereocraft import vr180
        wide = self.convert(converter, photo, tmp_path / "c.jpg", vr180_size=192,
                            target_deg=6.0, limit_deg=12.0, fmt="png")
        pinched = self.convert(converter, photo, tmp_path / "d.jpg", vr180_size=192,
                               target_deg=6.0, limit_deg=vr180.LIMIT_DEG, fmt="png")
        spread = lambda info: np.abs(
            np.asarray(Image.open(info["output"])).astype(int)[:, :192]
            - np.asarray(Image.open(info["output"])).astype(int)[:, 192:]).mean()
        assert spread(wide) > spread(pinched), "the clamp has to be the caller's to move"


class TestVideo:
    def test_every_frame_survives_with_audio(self, converter, clip, tmp_path):
        """-shortest once cost three frames off the end of a ninety-frame clip,
        and the checks at the time counted frame sizes rather than frames."""
        from stereocraft import video
        from stereocraft.pipeline import VideoSettings
        from conftest import audio_codec, frame_count, frame_sizes

        converter.settings = VideoSettings()
        info = video.convert_video(clip, tmp_path / "out.mp4", converter)
        assert info["frames"] == 60
        assert frame_count(info["output"]) == 60
        assert len(frame_sizes(info["output"])) == 1, "every frame must be the same size"
        assert audio_codec(info["output"]) is not None

    def test_stopping_leaves_nothing_behind(self, converter, clip, tmp_path):
        from stereocraft import video
        from stereocraft.pipeline import VideoSettings

        converter.settings = VideoSettings()
        out = tmp_path / "stopped.mp4"
        result = video.convert_video(clip, out, converter,
                                     on_progress=lambda done, total, secs: done < 5)
        assert result is None and not out.exists()
