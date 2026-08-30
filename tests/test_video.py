"""The video plumbing: probing, geometry, encoder choice, temporal smoothing.

None of this needs the depth model.  It does need ffmpeg, which video needs
anyway.
"""

import subprocess

import pytest
import torch

from stereocraft import video, vr180
from stereocraft.pipeline import Settings, VideoSettings


class TestProbe:
    def test_reads_the_shape_of_a_clip(self, clip):
        c = video.probe(clip)
        assert (c.width, c.height) == (320, 240)
        assert c.fps == pytest.approx(30.0)
        assert c.frames == 60 and c.audio == "aac"

    def test_a_silent_clip_has_no_audio(self, silent_clip):
        assert video.probe(silent_clip).audio is None

    def test_rotation_is_reported_the_way_the_decoder_will_hand_it_over(self, clip, tmp_path):
        """Phones record sideways and note the rotation instead of turning the
        pixels; probe has to agree with what ffmpeg actually decodes."""
        turned = tmp_path / "turned.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "90",
                        "-i", str(clip), "-c", "copy", str(turned)], check=True)
        c = video.probe(turned)
        assert (c.width, c.height) == (240, 320)

    def test_something_with_no_picture_in_it_is_refused(self, tmp_path):
        junk = tmp_path / "notes.txt"
        junk.write_text("not a video")
        with pytest.raises(ValueError):
            video.probe(junk)

    def test_a_still_reads_as_a_one_frame_clip(self, photo):
        """ffprobe sees a JPEG as a single-frame mjpeg stream, so probe does not
        refuse it.  Nothing reaches here by that route -- the CLI and the window
        both sort by extension first -- but it is worth knowing which end the
        filtering happens at."""
        assert video.probe(photo).frames <= 1


class TestGeometry:
    def test_half_width_puts_a_clip_out_the_size_it_came_in(self):
        g = video.geometry(video.Clip(1920, 1080, 30.0, 100, 3.3), VideoSettings())
        assert g.width == 1920 and g.height == 1080

    def test_full_width_keeps_every_native_pixel(self):
        g = video.geometry(video.Clip(1920, 1080, 30.0, 100, 3.3), VideoSettings(full_width=True))
        assert g.width > 1920

    @pytest.mark.parametrize("w,h", [(1921, 1081), (641, 361), (100, 99)])
    def test_both_dimensions_come_out_even(self, w, h):
        """yuv420p halves both, so an odd one cannot be encoded."""
        g = video.geometry(video.Clip(w, h, 30.0, 10, 1.0), VideoSettings())
        assert g.width % 2 == 0 and g.height % 2 == 0

    def test_the_margin_is_pinned_rather_than_measured(self):
        """Frames trimmed by different amounts come out different sizes, which
        no encoder will take."""
        clip = video.Clip(1920, 1080, 30.0, 100, 3.3)
        settings = VideoSettings()
        from stereocraft import stereo
        assert video.geometry(clip, settings).margin == stereo.max_margin(1920, settings.limit_pct)


class TestVr180Geometry:
    """A clip on a sphere: a patch of it, no trim, and settled before frame one."""

    def clip(self, w=1920, h=1080):
        return video.Clip(w, h, 30.0, 100, 3.3)

    def test_it_carries_the_patch_the_frames_go_on(self):
        g = video.geometry(self.clip(), VideoSettings(projection="vr180"))
        assert g.patch is not None
        assert (g.eye, g.height) == (g.patch.width, g.patch.height)

    def test_a_flat_clip_has_no_patch(self):
        assert video.geometry(self.clip(), VideoSettings()).patch is None

    def test_nothing_is_trimmed(self):
        """The sliver only one eye reaches is angle here, and cutting it would
        put every remaining pixel at the wrong bearing."""
        assert video.geometry(self.clip(), VideoSettings(projection="vr180")).margin == 0

    def test_the_size_is_the_ceiling_not_the_source(self):
        """A headset shows 25 pixels a degree whatever it is handed, so a small
        clip gets the same frame a large one does and differs only in how much
        of it it can fill -- which is what the upscaler is for."""
        small = video.geometry(self.clip(640, 480), VideoSettings(projection="vr180"))
        large = video.geometry(self.clip(3840, 2160), VideoSettings(projection="vr180"))
        assert small.eye == large.eye == vr180.video_cap(30.0)

    def test_an_explicit_size_is_taken(self):
        g = video.geometry(self.clip(), VideoSettings(projection="vr180", vr180_size=1024))
        assert g.eye == 1024

    @pytest.mark.parametrize("fps,side", [(24.0, 4096), (30.0, 4096), (60.0, 4096),
                                          (72.0, 3344), (90.0, 3344), (120.0, 2896)])
    def test_the_ceiling_follows_the_frame_rate(self, fps, side):
        """A hardware decoder is limited by pixels per second, not by pixels, so
        a fast clip cannot have what a slow one can."""
        g = video.geometry(video.Clip(3840, 2160, fps, 100, 3.3),
                           VideoSettings(projection="vr180"))
        assert g.eye == side

    @pytest.mark.parametrize("fps,limit", [(60.0, 8192 * 4096), (90.0, 6688 * 3344),
                                           (120.0, 5792 * 2896)])
    def test_the_frame_never_passes_what_the_headset_decodes(self, fps, limit):
        """The whole point of the ceiling.  A frame past this is not a slower
        conversion, it is one the headset refuses to play."""
        g = video.geometry(video.Clip(7680, 4320, fps, 100, 3.3),
                           VideoSettings(projection="vr180"))
        assert g.width * g.height <= limit

    def test_a_number_is_taken_as_given(self):
        """Someone who knows their player says so and is believed."""
        g = video.geometry(self.clip(), VideoSettings(projection="vr180", vr180_cap=1024))
        assert g.eye == 1024

    def test_a_1080p_clip_now_reaches_the_headset_s_own_resolution(self):
        """What the ceiling was raised for: 2048 an eye was 11.4 pixels per
        degree against a Quest 3's 25, and 1080p had the pixels all along."""
        g = video.geometry(self.clip(1920, 1080), VideoSettings(projection="vr180"))
        assert g.eye / 180 == pytest.approx(22.8, abs=0.1)

    def test_each_eye_is_square(self):
        """Skybox assumes every eye is a full 180 by 180, which is what the
        frame is, so nothing has to be read for it to come out right."""
        g = video.geometry(self.clip(), VideoSettings(projection="vr180"))
        assert g.eye == g.height
        assert g.patch.span_az == g.patch.span_el == 180.0

    @pytest.mark.parametrize("w,h", [(321, 241), (641, 361), (1001, 667)])
    def test_both_dimensions_come_out_even(self, w, h):
        g = video.geometry(video.Clip(w, h, 30.0, 10, 1.0), VideoSettings(projection="vr180"))
        assert g.width % 2 == 0 and g.height % 2 == 0


class TestCodecChoice:
    """Which codec a frame gets, which is decided by its width and not by the
    projection -- the projection was the first rule and missed a full-width flat
    pair off a 4K source, 7448 across and past what a headset decodes on h264."""

    @pytest.mark.parametrize("width,codec", [
        (1920, "h264"),   # an ordinary flat pair
        (3840, "h264"),   # 4K flat, still fine
        (3724, "h264"),   # full width off 1080p
        (7448, "hevc"),   # full width off 4K
        (8192, "hevc"),   # every vr180 frame
    ])
    def test_it_switches_where_h264_would_not_play(self, width, codec):
        assert video.codec_for(width) == codec

    def test_a_name_is_taken_at_face_value(self):
        assert video.codec_for(8192, "h264") == "h264"
        assert video.codec_for(640, "hevc") == "hevc"

    def test_the_ceiling_is_the_headset_s_and_not_the_level_s(self):
        """h264 level 6.x reaches 8192; a Quest decodes 4K of it.  The lower
        number is the one that matters."""
        assert video.H264_CEILING == 4096


class TestEncoders:
    def test_prefers_x264_when_it_is_there(self):
        video._ENCODERS_SEEN = None
        name, quality, _ = video.pick_encoder("h264")
        assert name == "libx264" and quality == "-crf"

    def test_falls_past_it_when_it_is_not(self, monkeypatch):
        monkeypatch.setattr(video, "available_encoders", lambda: {"h264_nvenc", "libopenh264"})
        name, quality, _ = video.pick_encoder("h264")
        assert name == "h264_nvenc" and quality == "-cq"

    def test_falls_all_the_way_to_software(self, monkeypatch):
        monkeypatch.setattr(video, "available_encoders", lambda: {"libopenh264"})
        assert video.pick_encoder("h264")[0] == "libopenh264"

    def test_says_so_rather_than_writing_nothing(self, monkeypatch):
        monkeypatch.setattr(video, "available_encoders", lambda: {"mjpeg"})
        with pytest.raises(video.MissingFFmpeg):
            video.pick_encoder("h264")


class _Ran:
    """What a stood-on `subprocess.run` gives back: a command that worked."""

    returncode = 0


class TestNoConsole:
    """Every ffmpeg has to be launched with whatever it takes to keep Windows
    from giving it a console window of its own: the window has none, so each
    child would flash a black box up over the screen -- and the encoder's would
    sit there for the length of the clip."""

    def test_the_probe_asks_for_no_window(self, monkeypatch, clip):
        seen = {}
        real = video.subprocess.run

        def watched(args, **kwargs):
            seen.update(kwargs)
            return real(args, **kwargs)

        monkeypatch.setattr(video.subprocess, "run", watched)
        video.probe(clip)
        assert seen.items() >= video.NO_CONSOLE.items()

    def test_and_so_do_the_three_that_do_the_work(self, monkeypatch, clip, tmp_path):
        seen = []
        info = video.probe(clip)  # before Popen is stood on, since it needs it
        settings = video.VideoSettings()
        monkeypatch.setattr(video.subprocess, "Popen", lambda args, **kwargs: seen.append(kwargs))
        monkeypatch.setattr(video.subprocess, "run",
                            lambda args, **kwargs: seen.append(kwargs) or _Ran())
        video._decoder(clip, info, None, None)
        video._encoder(tmp_path / "out.mp4", info, video.geometry(info, settings),
                       settings, None)
        # The remux is the third, and the one most easily forgotten: it runs
        # after the encode rather than beside it.
        video._remux(tmp_path / "picture.mp4", clip, tmp_path / "out.mp4", info,
                     settings, None)
        assert len(seen) == 3
        assert all(kwargs.items() >= video.NO_CONSOLE.items() for kwargs in seen)


class TestTemporalDepth:
    def test_the_first_frame_passes_through(self):
        d = torch.rand(1, 1, 8, 8)
        assert torch.allclose(video.TemporalDepth(0.5)(d), d)

    def test_smoothing_pulls_towards_the_frame_before(self):
        """A modest change, since a large one is a cut and handled differently."""
        smooth = video.TemporalDepth(0.5)
        smooth(torch.full((1, 1, 8, 8), 1.0))
        out = smooth(torch.full((1, 1, 8, 8), 1.2))
        assert float(out.mean()) == pytest.approx(1.1, abs=1e-5)
        assert smooth.cuts == 0

    def test_zero_keeps_nothing(self):
        smooth = video.TemporalDepth(0.0)
        smooth(torch.full((1, 1, 8, 8), 1.0))
        assert float(smooth(torch.full((1, 1, 8, 8), 1.2)).mean()) == pytest.approx(1.2)

    def test_it_never_freezes_the_first_frame_forever(self):
        assert video.TemporalDepth(1.0).keep <= 0.95

    def test_a_cut_starts_the_memory_again(self):
        """Averaging across a cut would drag one scene into the next."""
        smooth = video.TemporalDepth(0.5)
        smooth(torch.full((1, 1, 8, 8), 0.1))
        after = smooth(torch.full((1, 1, 8, 8), 5.0))
        assert smooth.cuts == 1
        assert float(after.mean()) == pytest.approx(5.0), "the new scene arrives unmixed"

    def test_an_ordinary_change_is_not_a_cut(self):
        smooth = video.TemporalDepth(0.5)
        smooth(torch.full((1, 1, 8, 8), 1.0))
        smooth(torch.full((1, 1, 8, 8), 1.05))
        assert smooth.cuts == 0


class TestOutputPath:
    def test_names_the_output_after_the_input(self, tmp_path):
        assert video.output_path(tmp_path / "a.mov").name == "a_full_sbs.mp4"

    def test_a_folder_gets_the_generated_name(self, tmp_path):
        assert video.output_path(tmp_path / "a.mov", tmp_path).name == "a_full_sbs.mp4"

    def test_an_explicit_name_is_kept(self, tmp_path):
        assert video.output_path(tmp_path / "a.mov", tmp_path / "b.mkv").name == "b.mkv"


class TestClock:
    @pytest.mark.parametrize("seconds,expected", [(5, "5s"), (65, "1m05s"), (3600, "1h00m")])
    def test_formats_a_wait(self, seconds, expected):
        assert video.clock(seconds) == expected


class TestTheSoundtrackIsAddedAfterwards:
    """The encode used to take two inputs: frames on a pipe, and the source file
    for its audio.  They run at wildly different speeds -- a frame arrives once
    the depth model and an 8K render are done with it, the file reads at disk
    speed -- and ffmpeg buffers the soundtrack until the picture catches up.  On
    a long clip that is the whole soundtrack held in memory, which is how a
    conversion came to die of an ffmpeg allocation failure rather than of
    anything we did.  So the picture is written alone and the sound put back
    after, where both sides are files."""

    def _args(self, monkeypatch, out, info, settings, **kwargs):
        seen = []
        monkeypatch.setattr(video.subprocess, "Popen", lambda args, **kw: seen.append(args))
        video._encoder(out, info, video.geometry(info, settings), settings, None, **kwargs)
        return seen[0]

    def test_the_encode_reads_one_input_and_it_is_the_pipe(self, monkeypatch, clip, tmp_path):
        info = video.probe(clip)
        args = self._args(monkeypatch, tmp_path / "out.mp4", info, video.VideoSettings())
        assert args.count("-i") == 1
        assert args[args.index("-i") + 1] == "-"

    def test_and_says_nothing_about_audio(self, monkeypatch, clip, tmp_path):
        info = video.probe(clip)
        args = self._args(monkeypatch, tmp_path / "out.mp4", info, video.VideoSettings())
        assert not [a for a in args if a.startswith("-c:a") or a == "-map"]

    def test_the_remux_copies_the_picture_rather_than_encoding_it_again(
            self, monkeypatch, clip, tmp_path):
        seen = []
        info = video.probe(clip)  # before run is stood on, since probe needs it
        monkeypatch.setattr(video.subprocess, "run",
                            lambda args, **kw: seen.append(args) or _Ran())
        video._remux(tmp_path / "picture.mp4", clip, tmp_path / "out.mp4", info,
                     video.VideoSettings(), None)
        args = seen[0]
        assert args[args.index("-c:v") + 1] == "copy"
        # Both sides are files, so neither waits on the other.
        assert args.count("-i") == 2
        # And the index is moved to the front here, since the encode no longer
        # does it -- a big file must not be rewritten twice.
        assert "+faststart" in args

    def test_the_encode_leaves_faststart_to_the_remux(self, monkeypatch, clip, tmp_path):
        info = video.probe(clip)
        args = self._args(monkeypatch, tmp_path / "out.mp4", info, video.VideoSettings(),
                          faststart=False)
        assert "+faststart" not in args

    def test_but_does_it_itself_when_there_is_no_sound_to_add(self, monkeypatch, clip, tmp_path):
        info = video.probe(clip)
        args = self._args(monkeypatch, tmp_path / "out.mp4", info, video.VideoSettings())
        assert "+faststart" in args


class TestHugeFramesGiveUpFrameParallelism:
    """x264 and x265 hold several frames at once -- lookahead, references, and a
    copy per frame thread -- so their memory is a multiple of the frame.  At 8192
    square a yuv420p frame is 100 MB and the default multiple does not fit in
    anything.  Slice threading uses the cores inside one frame instead, which at
    this size costs nothing: a frame that tall is more CTU rows than any desktop
    has threads."""

    def test_an_ordinary_frame_is_left_alone(self):
        assert video._thrift("libx265", 1920, 1080) == []
        assert video._thrift("libx264", 1920, 1080) == []

    def test_a_vr180_frame_holds_one_at_a_time(self):
        args = video._thrift("libx265", 8192, 8192)
        assert "frame-threads=1" in args[args.index("-x265-params") + 1]

    def test_and_x264_slices_instead(self):
        args = video._thrift("libx264", 8192, 8192)
        assert "sliced-threads=1" in args[args.index("-x264-params") + 1]

    def test_a_hardware_encoder_is_not_told_about_x265_options(self):
        assert video._thrift("hevc_nvenc", 8192, 8192) == []


class TestItSaysSoBeforeTheWaitRatherThanAfter:
    """A conversion refused an allocation nine hours in, on a machine that had a
    game left running and 14 GB of commit left out of 72.  Nothing was leaking:
    everything else had got there first.  ffmpeg's complaint at that point --
    "Cannot allocate memory", about one frame -- says nothing about the cause,
    so the check belongs at the start where it can still be acted on."""

    class _Geo:
        def __init__(self, width, height):
            self.width, self.height = width, height

    def _said(self, monkeypatch, free, geo):
        heard = []
        monkeypatch.setattr(video.logbook, "headroom", lambda: free)
        settings = video.VideoSettings()
        settings.on_notice = heard.append
        video._warn_if_full(settings, "clip.mp4", geo)
        return heard

    def test_a_full_machine_is_worth_saying_out_loud(self, monkeypatch):
        said = self._said(monkeypatch, 3_000_000_000, self._Geo(8192, 4096))
        assert said and "GB" in said[0]

    def test_an_empty_one_is_not(self, monkeypatch):
        assert not self._said(monkeypatch, 60_000_000_000, self._Geo(8192, 4096))

    def test_a_small_frame_wants_less_and_so_is_not_warned_about(self, monkeypatch):
        """The same 6 GB that is tight for an 8K sphere is ample for 1080p, so
        the threshold has to follow the frame rather than be one number."""
        assert not self._said(monkeypatch, 6_000_000_000, self._Geo(1920, 1080))
        assert self._said(monkeypatch, 6_000_000_000, self._Geo(8192, 4096))

    def test_not_knowing_is_not_a_warning(self, monkeypatch):
        assert not self._said(monkeypatch, None, self._Geo(8192, 4096))

    def test_an_8k_frame_wants_more_than_a_1080p_one(self):
        assert (video._headroom_wanted(self._Geo(8192, 4096))
                > 2 * video._headroom_wanted(self._Geo(1920, 1080)))
