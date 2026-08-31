"""What a source is short of, and what gets done about it before conversion.

Nothing here loads a model.  The decisions -- whether a clip needs enhancing at
all, and how wide to write it if it does -- are arithmetic on the clip's shape
and frame rate, and they are the part that has to be right: enhancing a source
that needed nothing costs an hour, and skipping one that needed it costs the
whole point of the feature.
"""

import pytest
import torch

from stereocraft import prepass, video, vr180
from stereocraft.pipeline import VideoSettings


def clip(w=720, h=1280, fps=30.0):
    return video.Clip(w, h, fps, 100, 3.3)


def both():
    return VideoSettings(projection="vr180", upscale=True, interpolate=True)


class TestWanted:
    @pytest.mark.parametrize("w,h,fps,upscale,smooth", [
        (720, 1280, 30.0, True, 60.0),    # portrait 30: short of pixels and of frames
        (1920, 1080, 60.0, False, None),  # already fills the ceiling, already smooth
        (1920, 1080, 30.0, False, 60.0),  # pixels enough, frames not
        (720, 1280, 60.0, True, None),    # frames enough, pixels not
    ])
    def test_it_asks_for_only_what_is_missing(self, w, h, fps, upscale, smooth):
        assert prepass.wanted(clip(w, h, fps), both()) == (upscale, smooth)

    def test_ntsc_sixty_is_left_alone(self):
        """59.94 is what NTSC-descended footage actually reports, and
        interpolating it to 60 would be resampling a whole clip for a rounding
        error."""
        assert prepass.wanted(clip(1920, 1080, 59.94), both())[1] is None
        assert prepass.wanted(clip(1920, 1080, 50.0), both())[1] == 60.0

    def test_a_flat_pair_is_never_enhanced(self):
        """It is written at the source's own size, so there is no ceiling to be
        short of and nothing to gain."""
        flat = VideoSettings(upscale=True, interpolate=True)
        assert prepass.wanted(clip(), flat) == (False, None)

    def test_neither_happens_unless_asked(self):
        assert prepass.wanted(clip(), VideoSettings(projection="vr180")) == (False, None)

    def test_a_source_barely_short_is_left_alone(self):
        """1626 wide reaches 97% of the ceiling on its own, and closing the last
        3% is the most expensive thing the app does -- a source that big leaves
        room for three frames at a time, so every frame is computed three times.
        """
        assert prepass.wanted(clip(1626, 2890), both())[0] is False
        assert prepass.wanted(clip(1920, 1080), both())[0] is False

    def test_720p_landscape_is_short_after_all(self):
        """The case that sent this measurement back to be looked at.

        1280x720 used to clear `WORTH_UPSCALING` by 1%, so `--upscale` was
        accepted and quietly did nothing on the commonest size of clip there is.
        It cleared it on density averaged across the frame width, which the
        `sec^2` spread at the edge of a wide lens inflates; measured on the axis,
        where the subject is, the same clip fills 76% of the ceiling and the
        middle of it is being enlarged 1.3x.  See `vr180.natural_size`.
        """
        assert prepass.wanted(clip(1280, 720), both())[0] is True

    def test_a_source_well_short_is_not(self):
        for width in (640, 720, 1080):
            assert prepass.wanted(clip(width, 1920), both())[0] is True

    def test_upscaling_follows_what_the_source_could_fill(self):
        """`natural_size` is the whole test: a clip that could already fill the
        ceiling has nothing a super-resolution pass can add."""
        from stereocraft.depth import DEFAULT_FOCAL_35MM, focal_from_35mm
        for width in (480, 720, 1080, 1920, 3840):
            spot = clip(width, 1080)
            lens = focal_from_35mm(DEFAULT_FOCAL_35MM, width)
            short = vr180.natural_size(lens) < (vr180.video_cap(30.0)
                                                        * prepass.WORTH_UPSCALING)
            assert prepass.wanted(spot, both())[0] is short


class TestTargetWidth:
    def test_it_does_not_carry_pixels_the_projection_throws_away(self):
        """Four times a 720 source is 2880, and the projection only reads the
        1490 that fills the ceiling.  The rest would be disk spent on nothing."""
        assert prepass.target_width(clip(), both()) < 720 * 4

    def test_it_keeps_a_little_over_what_is_needed(self):
        """Supersampling into the bicubic sample is worth something."""
        cap = vr180.video_cap(30.0)
        lens = 720 * 28 / 36
        fills = cap * vr180.fov(lens, 720) / vr180.FOV
        assert prepass.target_width(clip(), both()) > fills

    def test_it_never_asks_for_more_than_the_upscaler_makes(self):
        wide = prepass.target_width(clip(1920, 1080), both())
        assert wide <= 1920 * 4

    def test_it_comes_out_even(self):
        for width in (639, 721, 1081):
            assert prepass.target_width(clip(width, 1080), both()) % 2 == 0


class TestSmoothedCount:
    """What the interpolated pass will produce, which is what its progress bar
    counts against -- without it the window shows a number and no total."""

    def test_doubling(self):
        assert prepass.smoothed_count(60, 30.0, 60.0) == 119

    def test_rates_that_do_not_divide(self):
        assert prepass.smoothed_count(49, 24.0, 60.0) == 121

    def test_an_unknown_length_stays_unknown(self):
        assert prepass.smoothed_count(None, 30.0, 60.0) is None


class TestDetailReachesTheModel:
    """`--upscale-detail` is the answer to a clip that comes back looking
    painted, and it is worth nothing if it stops at the settings object."""

    @pytest.fixture
    def caught(self, monkeypatch):
        from stereocraft import upscale
        seen = {}

        def run(model, frames, **kwargs):
            seen.update(kwargs)
            return (f for f in frames)

        monkeypatch.setattr(upscale, "load", lambda **kwargs: object())
        monkeypatch.setattr(upscale, "chunk_length", lambda *a, **k: 8)
        monkeypatch.setattr(upscale, "run", run)
        return seen

    def convert(self, clip, tmp_path, **kwargs):
        prepass.run(clip, video.probe(clip),
                    VideoSettings(projection="vr180", upscale=True, **kwargs),
                    work=tmp_path)

    def test_a_number_is_passed_through(self, caught, silent_clip, tmp_path):
        self.convert(silent_clip, tmp_path, upscale_detail=0.2)
        assert caught["detail"] == pytest.approx(0.2)

    def test_unset_takes_the_model_s_own_default(self, caught, silent_clip, tmp_path):
        from stereocraft import upscale
        self.convert(silent_clip, tmp_path)
        assert caught["detail"] == pytest.approx(upscale.DETAIL)


class TestStopping:
    """The Stop button, which before this reached nothing until the conversion
    itself began -- so a clip spent two passes, minutes each, refusing to stop."""

    @pytest.fixture
    def fake_model(self, monkeypatch):
        from stereocraft import upscale
        monkeypatch.setattr(upscale, "load", lambda **kwargs: object())
        monkeypatch.setattr(upscale, "chunk_length", lambda *a, **k: 8)
        monkeypatch.setattr(upscale, "run",
                            lambda model, frames, **kwargs: (f for f in frames))

    def test_it_reports_a_stage_a_count_and_a_total(self, fake_model, silent_clip, tmp_path):
        seen = []
        cfg = VideoSettings(projection="vr180", upscale=True)
        clip = video.probe(silent_clip)
        prepass.run(silent_clip, clip, cfg, on_progress=lambda *a: seen.append(a) or True,
                    work=tmp_path)
        assert seen, "a pass that says nothing is a window that says nothing"
        label, done, total = seen[-1]
        assert label == "adding detail"
        assert done == clip.frames and total == clip.frames

    def test_returning_false_stops_it(self, fake_model, silent_clip, tmp_path):
        out, _ = prepass.run(silent_clip, video.probe(silent_clip),
                             VideoSettings(projection="vr180", upscale=True),
                             on_progress=lambda label, done, total: done < 3, work=tmp_path)
        assert out is False, "False, not None -- None means there was nothing to do"

    def test_stopping_leaves_no_half_written_intermediate(self, fake_model, silent_clip, tmp_path):
        prepass.run(silent_clip, video.probe(silent_clip),
                    VideoSettings(projection="vr180", upscale=True),
                    on_progress=lambda label, done, total: done < 3, work=tmp_path)
        assert list(tmp_path.glob("*.mp4")) == []

    def test_nothing_to_do_is_not_the_same_as_stopped(self, silent_clip, tmp_path):
        out, clip = prepass.run(silent_clip, video.probe(silent_clip),
                                VideoSettings(), work=tmp_path)
        assert out is None
