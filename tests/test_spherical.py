"""Saying where on the sphere a picture belongs.

Two halves that fail in opposite ways.  The metadata half fails quietly -- the
file plays, and is simply shown at the wrong size, which nobody notices until
they are wearing it.  The mp4 half fails loudly: splicing bytes into a container
full of absolute offsets and getting the fixup wrong gives a file that decodes
whatever happens to be where the samples used to be.  Both are checked by taking
the numbers back out rather than by trusting that they went in.
"""

import shutil
import struct
import subprocess

import pytest

from stereocraft import spherical, vr180
from conftest import frame_count


def lens(equivalent_mm, width):
    return float(equivalent_mm) / 36.0 * width


def read_boxes(path):
    """Every box in a file, flattened, as {kind: payload bytes}.

    Deliberately a separate walker from the one that wrote them: a reader that
    shares the writer's bugs agrees with it about everything.
    """
    data = open(path, "rb").read()
    found = {}

    def walk(start, end):
        at = start
        while at + 8 <= end:
            size = struct.unpack_from(">I", data, at)[0]
            kind = data[at + 4:at + 8]
            if size < 8 or at + size > end:
                return
            found.setdefault(kind, data[at + 8:at + size])
            if kind in (b"moov", b"trak", b"mdia", b"minf", b"stbl", b"sv3d", b"proj"):
                walk(at + 8, at + size)
            elif kind == b"stsd":
                walk(at + 16, at + size)
            elif kind in (b"avc1", b"hvc1", b"hev1"):
                walk(at + 8 + 78, at + size)  # past the fixed VisualSampleEntry fields
            at += size

    walk(0, len(data))
    return found


class TestGpano:
    def spot(self):
        return vr180.patch(lens(28, 4000), 4000, 3000)

    def test_says_which_kind_of_panorama_the_numbers_describe(self):
        text = spherical.gpano(self.spot()).decode()
        assert 'GPano:ProjectionType="equirectangular"' in text

    def test_it_does_not_ask_to_be_opened_as_one(self):
        """The fields describe one eye and the file holds two side by side.
        Asking a viewer to wrap the frame it has round a sphere is how a
        monoscopic 360 declares itself, and a 2:1 pair that gets believed comes
        out at twice the width it belongs at."""
        assert 'GPano:UsePanoramaViewer="False"' in spherical.gpano(self.spot()).decode()

    def test_a_caller_that_means_it_can_still_say_so(self):
        """For a single-eye frame, where the claim would be true."""
        assert 'GPano:UsePanoramaViewer="True"' in spherical.gpano(self.spot(),
                                                                   viewer=True).decode()

    def test_the_stored_size_is_what_is_stored(self):
        spot = self.spot()
        text = spherical.gpano(spot).decode()
        assert f'GPano:CroppedAreaImageWidthPixels="{spot.width}"' in text
        assert f'GPano:CroppedAreaImageHeightPixels="{spot.height}"' in text

    def test_the_full_frame_is_the_sphere_it_was_cut_from(self):
        spot = self.spot()
        text = spherical.gpano(spot).decode()
        assert f'GPano:FullPanoWidthPixels="{spot.full_width}"' in text
        assert f'GPano:FullPanoHeightPixels="{spot.full_height}"' in text

    def test_the_piece_is_placed_in_the_middle(self):
        """Off by the offset is off by that many degrees of bearing, which reads
        as a picture that will not sit still where you put your head."""
        spot = self.spot()
        text = spherical.gpano(spot).decode()
        assert f'GPano:CroppedAreaLeftPixels="{spot.left}"' in text
        assert f'GPano:CroppedAreaTopPixels="{spot.top}"' in text

    def test_it_is_a_well_formed_packet(self):
        import xml.etree.ElementTree as ET
        body = spherical.gpano(self.spot()).decode()
        body = body[body.index("<x:xmpmeta"):body.index("<?xpacket end")]
        ET.fromstring(body)  # raises if it is not

    def test_pillow_writes_it_into_a_jpeg(self, tmp_path):
        import numpy as np
        from stereocraft.pipeline import save_image

        out = save_image(np.zeros((8, 16, 3), np.uint8), tmp_path / "a.jpg",
                         xmp=spherical.gpano(self.spot()))
        assert b"ns.google.com/photos" in out.read_bytes()


class TestStereoMode:
    def test_left_right_is_the_ordinary_pair(self):
        assert spherical.st3d(spherical.LEFT_RIGHT)[-1] == 2

    def test_right_left_is_what_cross_eyed_writes(self):
        """A cross-eyed pair really is right|left, and the format has a number
        for exactly that -- so a headset can put the eyes back the right way
        round instead of showing them inside out."""
        assert spherical.st3d(spherical.RIGHT_LEFT)[-1] == 4

    def test_it_is_a_full_box(self):
        box = spherical.st3d()
        assert struct.unpack_from(">I", box)[0] == len(box)
        assert box[4:8] == b"st3d"
        assert box[8] == 0, "version"


class TestProjectionBounds:
    """The bounds say how much of the 360-by-180 frame to crop off each edge, so
    putting them back has to give the angles the patch started with."""

    def bounds(self, spot):
        box = spherical.sv3d(spot)
        at = box.index(b"equi")
        top, bottom, left, right = struct.unpack_from(">IIII", box, at + 4 + 4)
        return [v / 2 ** 32 for v in (top, bottom, left, right)]

    @pytest.mark.parametrize("equivalent_mm", [13, 28, 49])
    def test_the_angles_survive_the_round_trip(self, equivalent_mm):
        """Whatever lens it was shot on, the frame is the hemisphere and the
        bounds have to put it back as one."""
        spot = vr180.patch(lens(equivalent_mm, 2000), 2000, 1500)
        top, bottom, left, right = self.bounds(spot)
        assert (1 - left - right) * 360 == pytest.approx(spot.span_az, abs=0.1)
        assert (1 - top - bottom) * 180 == pytest.approx(spot.span_el, abs=0.1)

    def test_the_patch_stays_centred(self):
        spot = vr180.patch(lens(28, 2000), 2000, 1500)
        top, bottom, left, right = self.bounds(spot)
        assert left == pytest.approx(right, abs=1e-3)
        assert top == pytest.approx(bottom, abs=1e-3)

    def test_it_crops_a_quarter_off_each_side_and_nothing_off_the_top(self):
        """Which is the whole of what makes a file VR180 rather than 360, and
        the reason the bounds are written even though the frame is never
        cropped tighter than the format's own hemisphere."""
        spot = vr180.patch(lens(28, 1000), 1000, 1000)
        top, bottom, left, right = self.bounds(spot)
        assert left == pytest.approx(0.25, abs=1e-3)
        assert right == pytest.approx(0.25, abs=1e-3)
        assert top == pytest.approx(0.0, abs=1e-3)
        assert bottom == pytest.approx(0.0, abs=1e-3)

    def test_the_bounds_do_not_follow_the_lens(self):
        """The frame is the hemisphere whatever was pointed at it, so a long lens
        and a wide one record the same placement and differ only in how much of
        the frame turns out to be lit."""
        wide = self.bounds(vr180.patch(lens(13, 2000), 2000, 1500))
        narrow = self.bounds(vr180.patch(lens(49, 2000), 2000, 1500))
        assert wide == pytest.approx(narrow, abs=1e-3)


class TestAnnotate:
    def spot(self):
        return vr180.patch(lens(28, 320), 320, 240)

    def prepared(self, clip, tmp_path):
        out = tmp_path / "a.mp4"
        shutil.copy(clip, out)
        return out

    def test_the_boxes_go_in(self, silent_clip, tmp_path):
        out = self.prepared(silent_clip, tmp_path)
        assert spherical.annotate(out, self.spot())
        boxes = read_boxes(out)
        assert b"st3d" in boxes and b"sv3d" in boxes
        assert b"equi" in boxes and b"prhd" in boxes

    def test_the_clip_still_decodes_every_frame(self, silent_clip, tmp_path):
        """The one that matters.  `moov` is written first, so making it bigger
        pushes `mdat` down the file and every chunk offset has to move with it.
        Get that wrong and the file does not fail -- it decodes whatever is now
        where the samples used to be."""
        out = self.prepared(silent_clip, tmp_path)
        before = frame_count(out)
        assert spherical.annotate(out, self.spot())
        assert frame_count(out) == before

    def test_a_clip_with_sound_keeps_it(self, clip, tmp_path):
        """The soundtrack is a second track with chunk offsets of its own, and
        they move too."""
        from conftest import audio_codec

        out = self.prepared(clip, tmp_path)
        assert spherical.annotate(out, self.spot())
        assert audio_codec(out) is not None
        assert frame_count(out) == 60

    def probe(self, path):
        return subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream_side_data", "-of", "default", str(path)],
            capture_output=True, text=True, check=True).stdout

    def test_ffmpeg_reads_back_what_was_written(self, silent_clip, tmp_path):
        """A second opinion from something that did not write the file."""
        out = self.prepared(silent_clip, tmp_path)
        spherical.annotate(out, self.spot())
        probed = self.probe(out)
        assert "Stereo 3D" in probed and "side by side" in probed
        assert "Spherical Mapping" in probed

    def test_right_left_is_written_even_though_ffmpeg_drops_it(self, silent_clip, tmp_path):
        """ffmpeg 6.1 knows stereo modes 0 to 2 and discards anything past them,
        so a right-left clip comes back with its projection and no stereo tag.
        The bytes are still right, and writing mode 2 instead would be a file
        that confidently tells a headset to swap the viewer's eyes."""
        out = self.prepared(silent_clip, tmp_path)
        spherical.annotate(out, self.spot(), spherical.RIGHT_LEFT)
        assert read_boxes(out)[b"st3d"][-1] == spherical.RIGHT_LEFT
        assert "Spherical Mapping" in self.probe(out)

    def test_the_file_grows_by_exactly_the_boxes(self, silent_clip, tmp_path):
        out = self.prepared(silent_clip, tmp_path)
        before = out.stat().st_size
        extra = len(spherical.st3d()) + len(spherical.sv3d(self.spot()))
        spherical.annotate(out, self.spot())
        assert out.stat().st_size == before + extra

    def test_something_that_is_not_an_mp4_is_left_alone(self, tmp_path):
        """Reported rather than half-written: a clip that plays is worth more
        than one that is correctly labelled and does not."""
        out = tmp_path / "not.mp4"
        out.write_bytes(b"this is not a container at all, not even slightly")
        before = out.read_bytes()
        assert spherical.annotate(out, self.spot()) is False
        assert out.read_bytes() == before
