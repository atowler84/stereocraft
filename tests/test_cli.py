"""Argument handling, none of which needs a depth model."""

import pytest

from stereocraft import cli, vr180
from stereocraft.pipeline import SBS_TAGS, Settings, VideoSettings, output_path, tag


def parse(*argv):
    return cli.build_parser().parse_args(list(argv))


class TestRetiredFlags:
    """--disparity and --convergence described something the renderer no longer
    computes.  Translating them would quietly produce a different picture than
    the one asked for, so they stop instead."""

    @pytest.mark.parametrize("flag,replacement", [("--disparity", "--eyes"), ("--convergence", "--focus")])
    def test_say_what_to_use_instead(self, flag, replacement):
        message = cli.retired(parse(flag, "2", "x.jpg"))
        assert message and replacement in message

    def test_silent_when_neither_is_given(self):
        assert cli.retired(parse("x.jpg")) is None

    def test_main_refuses_rather_than_guessing(self, capsys):
        assert cli.main(["--disparity", "2", "x.jpg"]) == 2
        assert "--eyes" in capsys.readouterr().err


class TestMisdirectedFlags:
    """--target and --limit are percentages of frame width, and the spherical
    path does not think in those.  They used to be accepted and ignored, which
    is the one answer a settings flag must never give."""

    @pytest.mark.parametrize("flag,replacement", [("--target", "--target-deg"),
                                                  ("--limit", "--limit-deg")])
    def test_say_what_to_use_instead(self, flag, replacement):
        message = cli.misdirected(parse("--projection", "vr180", flag, "2", "x.jpg"))
        assert message and replacement in message

    def test_a_flat_pair_is_where_a_percentage_belongs(self):
        assert cli.misdirected(parse("--target", "2", "x.jpg")) is None
        assert cli.misdirected(parse("--limit", "3", "x.jpg")) is None

    def test_silent_when_neither_is_given(self):
        assert cli.misdirected(parse("--projection", "vr180", "x.jpg")) is None

    def test_main_refuses_rather_than_ignoring(self, capsys):
        assert cli.main(["--projection", "vr180", "-t", "2", "x.jpg"]) == 2
        assert "--target-deg" in capsys.readouterr().err


class TestDegreeFlags:
    """The spherical pair, which is what `--target` hands over to."""

    def test_they_default_to_what_vr180_asks_for(self):
        s = cli.settings_for(parse("--projection", "vr180", "x.jpg"), video=False)
        assert s.target_deg == vr180.TARGET_DEG and s.limit_deg == vr180.LIMIT_DEG

    def test_they_reach_the_settings(self):
        s = cli.settings_for(parse("--projection", "vr180", "--target-deg", "1.4",
                                   "--limit-deg", "2.5", "x.jpg"), video=False)
        assert s.target_deg == 1.4 and s.limit_deg == 2.5

    def test_a_clip_gets_them_too(self):
        s = cli.settings_for(parse("--projection", "vr180", "--target-deg", "0.9", "x.mp4"),
                             video=True)
        assert s.target_deg == 0.9

    def test_the_percentage_pair_still_defaults_without_being_given(self):
        """`--limit` lost its literal default so that "was it given?" can be
        asked; the setting it feeds must not have lost it too."""
        assert cli.settings_for(parse("x.jpg"), video=False).limit_pct == Settings.limit_pct
        assert cli.settings_for(parse("--limit", "2", "x.jpg"), video=False).limit_pct == 2.0


class TestNumber:
    @pytest.mark.parametrize("given", [None, "auto", "AUTO"])
    def test_auto_stays_auto(self, given):
        assert cli._number(given) == "auto"

    def test_a_measurement_becomes_a_number(self):
        assert cli._number("65") == 65.0


class TestSettingsFor:
    def test_a_photo_gets_the_photo_defaults(self):
        s = cli.settings_for(parse("x.jpg"), video=False)
        assert isinstance(s, Settings) and not isinstance(s, VideoSettings)
        assert s.eyes_mm == "auto" and s.target_pct == Settings.target_pct

    def test_a_clip_gets_the_gentler_ones(self):
        s = cli.settings_for(parse("x.mp4"), video=True)
        assert isinstance(s, VideoSettings)
        assert s.target_pct == VideoSettings.target_pct < Settings.target_pct

    def test_an_explicit_measurement_wins(self):
        s = cli.settings_for(parse("--eyes", "70", "--focus", "2.5", "x.jpg"), video=False)
        assert s.eyes_mm == 70.0 and s.focus_m == 2.5

    def test_the_codec_is_left_to_the_frame(self):
        """Not to the projection, which was the first thing tried and missed a
        full-width flat pair off a 4K source -- 7448 across and past what a
        headset decodes on h264."""
        assert cli.settings_for(parse("x.mp4"), video=True).codec == "auto"
        assert cli.settings_for(parse("--projection", "vr180", "x.mp4"),
                                video=True).codec == "auto"

    def test_an_explicit_codec_still_wins(self):
        s = cli.settings_for(parse("--projection", "vr180", "--codec", "h264", "x.mp4"),
                             video=True)
        assert s.codec == "h264"

    def test_video_flags_reach_the_settings(self):
        s = cli.settings_for(parse("--codec", "hevc", "--crf", "22", "--temporal", "0.8",
                                   "--full", "--no-audio", "x.mp4"), video=True)
        assert (s.codec, s.crf, s.temporal, s.full_width, s.audio) == ("hevc", 22, 0.8, True, False)


class TestNaming:
    """A player has nothing but the file name to go on -- no JPEG and no mp4 the
    app writes says how it is meant to be looked at -- so the name is the
    setting, and getting it wrong shows the wrong thing rather than nothing."""

    @pytest.mark.parametrize("projection,cross,expected", [
        ("flat", False, "_full_sbs"),
        ("flat", True, "_full_sbs_cross"),
        ("vr180", False, "_180x180_full_sbs"),
        ("vr180", True, "_180x180_full_sbs_cross"),
    ])
    def test_every_combination_gets_its_own_name(self, projection, cross, expected):
        assert tag(Settings(projection=projection, cross_eyed=cross)) == expected

    def test_the_angle_is_spelled_the_way_the_rules_spell_it(self):
        """A bare "180" is not on any player's list of keywords -- 180x180 is.
        The clip that taught us this played on a cinema screen."""
        assert "180x180" in tag(Settings(projection="vr180"))

    def test_vr180_carries_both_tokens_a_player_reads(self):
        """The angle sets the projection and the layout sets the arrangement,
        and players key on the two of them separately -- which is why it is not
        just "_vr180"."""
        name = tag(Settings(projection="vr180"))
        assert "180x180" in name and "sbs" in name

    def test_a_photo_says_its_eyes_are_full_width(self):
        """Because they are, and because a still with nothing said about it is
        assumed to be half width and stretched across twice the width it
        belongs in."""
        assert tag(Settings()) == "_full_sbs"
        assert tag(Settings(projection="vr180")).endswith("_full_sbs")

    @pytest.mark.parametrize("projection,full,expected", [
        ("flat", False, "_hsbs"),
        ("flat", True, "_full_sbs"),
        # vr180 sizes the frame from the projection rather than from the source,
        # so every eye is whole whatever `--full` says.
        ("vr180", False, "_180x180_full_sbs"),
        ("vr180", True, "_180x180_full_sbs"),
    ])
    def test_a_clip_says_which_width_its_eyes_are_at(self, projection, full, expected):
        assert tag(VideoSettings(projection=projection, full_width=full)) == expected

    def test_the_photo_path_uses_it(self, tmp_path):
        out = output_path(tmp_path / "a.jpg", None, "auto", tag(Settings(projection="vr180")))
        assert out.name == "a_180x180_full_sbs.jpg"

    def test_the_video_path_uses_it(self, tmp_path):
        from stereocraft import video
        out = video.output_path(tmp_path / "a.mov", None,
                                tag(VideoSettings(projection="vr180", cross_eyed=True)))
        assert out.name == "a_180x180_full_sbs_cross.mp4"

    @pytest.mark.parametrize("old", ["a_sbs.jpg", "a_sbs_cross.jpg",
                                     "a_180_sbs.jpg", "a_180_sbs_cross.mp4"])
    def test_a_library_an_earlier_version_wrote_is_still_left_alone(self, tmp_path, old):
        """The names changed; a folder full of the old ones must not be
        converted all over again because of it."""
        (tmp_path / old).write_bytes(b"x")
        assert cli.collect([str(tmp_path)]) == []


class TestCollect:
    def test_finds_photos_and_clips_and_skips_its_own_output(self, tmp_path):
        for name in ("a.jpg", "b.mp4", "c_sbs.jpg", "d_depth.png", "notes.txt"):
            (tmp_path / name).write_bytes(b"x")
        found = {p.name for p in cli.collect([str(tmp_path)])}
        assert found == {"a.jpg", "b.mp4"}

    @pytest.mark.parametrize("name", SBS_TAGS)
    def test_skips_every_name_it_can_write(self, tmp_path, name):
        (tmp_path / f"a{name}.jpg").write_bytes(b"x")
        (tmp_path / f"b{name}.mp4").write_bytes(b"x")
        assert cli.collect([str(tmp_path)]) == []

    @pytest.mark.parametrize("projection", ["flat", "vr180"])
    @pytest.mark.parametrize("cross", [False, True])
    def test_what_it_writes_is_what_it_then_ignores(self, tmp_path, projection, cross):
        """The round trip, which is the whole point of the list: run it over a
        folder twice and the second pass must find nothing it made on the first.
        A name that gets written but not skipped converts the conversions."""
        settings = Settings(projection=projection, cross_eyed=cross)
        written = output_path(tmp_path / "a.jpg", None, "auto", tag(settings))
        written.write_bytes(b"x")
        assert cli.collect([str(tmp_path)]) == [], f"{written.name} came back round"

    def test_says_so_when_something_is_missing(self, capsys):
        assert cli.collect(["definitely-not-here-*.jpg"]) == []
        assert "not found" in capsys.readouterr().err


class TestClock:
    @pytest.mark.parametrize("seconds,expected", [(0, "0s"), (45, "45s"), (90, "1m30s"), (3700, "1h01m")])
    def test_reads_the_way_someone_waiting_would_say_it(self, seconds, expected):
        assert cli.clock(seconds) == expected


class TestOutpaint:
    """The flag that fills the space around a vr180 clip with the scene it was
    shot in.  Off unless asked, like the two passes beside it."""

    def test_it_is_off_by_default(self):
        assert parse("clip.mp4").outpaint is False

    def test_asking_turns_it_on(self):
        assert parse("--outpaint", "clip.mp4").outpaint is True

    def test_it_reaches_the_video_settings(self):
        settings = cli.settings_for(parse("--projection", "vr180", "--outpaint", "clip.mp4"),
                                    video=True)
        assert settings.outpaint is True

    def test_a_photo_never_carries_it(self):
        """A still has no other frames to gather a periphery from, so the flag
        is a video one and `Settings` has no field for it at all."""
        settings = cli.settings_for(parse("--outpaint", "photo.jpg"), video=False)
        assert not hasattr(settings, "outpaint")


class TestVersion:
    def test_the_two_copies_of_it_agree(self):
        """It lives in `stereocraft/__init__.py` and in `pyproject.toml`, and the
        Windows build reads the second while `--version` prints the first. Bumped
        singly they disagree silently, and the exe reports the version before the
        one it is."""
        import re
        from pathlib import Path

        from stereocraft import __version__

        pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
        declared = re.search(r'^version\s*=\s*"(.+)"', pyproject, re.MULTILINE).group(1)
        assert declared == __version__
