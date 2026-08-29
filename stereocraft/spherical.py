"""Telling a player what it is looking at.

The pictures `vr180` makes are pieces of a sphere, and nothing about a JPEG or an
mp4 says so on its own.  Worse, once the empty part of the sphere is cropped away
-- which is the only sane thing to do with it, see `vr180` -- a player that does
not know where the piece belongs will stretch it across the whole 180 degrees and
show it at several times life size.  Cropping and saying so are one change, not
two, and this is the saying-so half.

Both formats have had a place to put it for years:

**Stills** carry Google's GPano XMP, whose `CroppedArea*` and `FullPano*` fields
exist precisely to say "this is a piece of a panorama, and here is where".  The
fields describe one eye, which is the natural reading once a player has split a
side-by-side frame -- and splitting it is what the `_180x180_full_sbs` in the
name is for.  It is the weaker half of this module: GPano has no vocabulary for a
stereo pair in one frame, so the crop offsets are honest and the pairing, the
layout and the width of an eye are all still carried by the file name.  Skybox
reads none of this and goes by the name alone -- see `pipeline.SBS_TAGS` for what
that costs if the name is spelled even slightly wrong.

**Video** carries the boxes from Google's Spherical Video V2, which does have a
vocabulary for both: `st3d` says the frame holds two views side by side, and
`sv3d/proj/equi` says each view is equirectangular and which part of the sphere
it covers.  A VR180 file is not a special projection at all -- it is a 360-degree
equirectangular one whose bounds crop it to the middle 180 -- so the same
`projection_bounds` that record the crop are what make it VR180 in the first
place.

ffmpeg will not write either, so the mp4 boxes are spliced in afterwards.  That
means moving bytes in a file that has offsets pointing into it: `moov` is written
first here (`-movflags +faststart`), so growing it shifts every sample in `mdat`
and every chunk offset in every track has to move with it.  `_shift_chunks` is
that, and it is the part to be suspicious of if a converted clip ever plays as
noise.
"""

import struct

# Spherical Video V2 stereo modes: 0 monoscopic, 1 top-bottom, 2 left-right,
# 3 custom, 4 right-left.  The last is exactly what `--cross` writes, and saying
# so is the difference between a headset swapping the eyes back and a viewer
# wearing the pair inside out.
LEFT_RIGHT = 2
RIGHT_LEFT = 4
# ffmpeg 6.1 knows 0 to 2 and discards the rest, so a right-left clip reads back
# with its projection and no stereo tag at all.  Writing left-right instead would
# read back cleanly and tell the headset to swap the viewer's eyes, which is the
# worse of the two: unlabelled is a question, mislabelled is a wrong answer.
# Where the sample entry hides inside a track.
_STSD_PATH = (b"mdia", b"minf", b"stbl", b"stsd")


# --- stills -----------------------------------------------------------------
def gpano(spot, heading=0.0, viewer=False):
    """GPano XMP describing where one eye's picture sits on the sphere.

    `FullPano*` is the 360-degree frame the piece was cut from, and `CroppedArea*`
    is the piece and its offset.  A viewer that reads none of it still sees an
    ordinary photograph, which is the reason the fields are laid out this way.

    **`UsePanoramaViewer` is False, and that is the whole of the difference
    between a description and a claim.**  The fields above describe one eye; the
    file they are written into holds two, side by side.  Set the flag and what
    the file says is "show the frame you have in a panorama viewer, it is
    equirectangular" -- which of a 2:1 side-by-side pair is exactly how a
    monoscopic 360 panorama declares itself, and a viewer that believes it wraps
    both eyes round the sphere at twice the width they belong at.  Left False the
    same numbers are there for anything that splits the pair first, and nothing
    is being asserted about the frame as a whole.  Reasoned rather than measured,
    unlike the file name, where the rules are published and the symptom was
    reproduced.
    """
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" '
        'xmlns:GPano="http://ns.google.com/photos/1.0/panorama/" '
        f'GPano:UsePanoramaViewer="{"True" if viewer else "False"}" '
        'GPano:ProjectionType="equirectangular" '
        f'GPano:CroppedAreaImageWidthPixels="{spot.width}" '
        f'GPano:CroppedAreaImageHeightPixels="{spot.height}" '
        f'GPano:FullPanoWidthPixels="{spot.full_width}" '
        f'GPano:FullPanoHeightPixels="{spot.full_height}" '
        f'GPano:CroppedAreaLeftPixels="{spot.left}" '
        f'GPano:CroppedAreaTopPixels="{spot.top}" '
        f'GPano:InitialViewHeadingDegrees="{heading:.0f}" '
        'GPano:InitialViewPitchDegrees="0" '
        'GPano:InitialViewRollDegrees="0"/>'
        "</rdf:RDF></x:xmpmeta>"
        '<?xpacket end="w"?>'
    ).encode("utf-8")


# --- boxes ------------------------------------------------------------------
def _box(kind, payload):
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def _full_box(kind, payload, version=0, flags=0):
    return _box(kind, bytes([version]) + flags.to_bytes(3, "big") + payload)


def _fixed32(fraction):
    """A 0.32 fixed-point fraction, which is how the projection bounds are
    written: the whole of a 32-bit word stands for the whole frame."""
    return max(0, min(0xFFFFFFFF, int(round(fraction * (1 << 32)))))


def st3d(mode=LEFT_RIGHT):
    return _full_box(b"st3d", bytes([mode]))


def sv3d(spot, source=b"stereocraft"):
    """The projection boxes: equirectangular, bounded to the part that exists.

    The bounds are how much to crop off each side of the full 360-by-180 frame,
    so a patch covering the middle 65 degrees of azimuth crops away the other
    295 -- and a full VR180 frame still crops away the 180 behind the viewer.
    """
    full_w, full_h = spot.full_width, spot.full_height
    equi = _full_box(b"equi", struct.pack(
        ">IIII",
        _fixed32(spot.top / full_h),
        _fixed32((full_h - spot.top - spot.height) / full_h),
        _fixed32(spot.left / full_w),
        _fixed32((full_w - spot.left - spot.width) / full_w),
    ))
    proj = _box(b"proj", _full_box(b"prhd", struct.pack(">iii", 0, 0, 0)) + equi)
    return _box(b"sv3d", _full_box(b"svhd", source + b"\x00") + proj)


# --- walking an mp4 ---------------------------------------------------------
def _children(data, start, end):
    """Every box directly inside a range: (kind, start, size, payload start)."""
    at = start
    while at + 8 <= end:
        size = struct.unpack_from(">I", data, at)[0]
        kind = data[at + 4:at + 8]
        head = 8
        if size == 1:  # 64-bit size, for a box past 4 GB
            if at + 16 > end:
                return
            size = struct.unpack_from(">Q", data, at + 8)[0]
            head = 16
        elif size == 0:  # runs to the end of its parent
            size = end - at
        if size < head or at + size > end:
            return
        yield kind, at, size, at + head
        at += size


def _child(data, start, end, kind):
    for found, at, size, payload in _children(data, start, end):
        if found == kind:
            return at, size, payload
    return None


def _video_track(data, moov):
    """The `trak` that carries the picture, told apart from the soundtrack by the
    handler its media declares."""
    at, size, payload = moov
    for kind, t_at, t_size, t_payload in _children(data, payload, at + size):
        if kind != b"trak":
            continue
        mdia = _child(data, t_payload, t_at + t_size, b"mdia")
        if mdia is None:
            continue
        hdlr = _child(data, mdia[2], mdia[0] + mdia[1], b"hdlr")
        # handler_type sits 8 bytes into a full box's payload, after the
        # version, flags and a reserved word.
        if hdlr and data[hdlr[2] + 8:hdlr[2] + 12] == b"vide":
            return t_at, t_size, t_payload
    return None


def _sample_entry(data, moov):
    """The chain of boxes down to the first sample entry, which is where the
    projection boxes belong: outermost first, each as (start, size, payload)."""
    track = _video_track(data, moov)
    if track is None:
        return None
    chain, box = [moov, track], track
    for kind in _STSD_PATH:
        box = _child(data, box[2], box[0] + box[1], kind)
        if box is None:
            return None
        chain.append(box)
    # stsd is a full box with an entry count in front of its children, and the
    # first entry is the one the picture is actually coded with.
    entry = next(iter(_children(data, box[2] + 8, box[0] + box[1])), None)
    if entry is None:
        return None
    chain.append((entry[1], entry[2], entry[3]))
    return chain


def _grow(data, chain, delta):
    """Add `delta` to the size of every box on the chain, in place."""
    for start, size, payload in chain:
        if payload - start == 16:  # a 64-bit size, written after the kind
            struct.pack_into(">Q", data, start + 8, size + delta)
        else:
            struct.pack_into(">I", data, start, size + delta)


def _shift_chunks(data, start, end, insert_at, delta):
    """Move every chunk offset that pointed past the splice.

    `moov` comes first in these files, so making it bigger pushes all of `mdat`
    down the file -- and `stco` holds absolute offsets into it.  Left alone, a
    player seeks to where the samples used to be and decodes whatever is there
    now, which does not fail, it just plays as noise.
    """
    for kind, at, size, payload in _children(data, start, end):
        if kind in (b"stco", b"co64"):
            wide = kind == b"co64"
            count = struct.unpack_from(">I", data, payload + 4)[0]
            step, form = (8, ">Q") if wide else (4, ">I")
            for index in range(count):
                spot = payload + 8 + index * step
                if spot + step > at + size:
                    break
                offset = struct.unpack_from(form, data, spot)[0]
                if offset >= insert_at:
                    struct.pack_into(form, data, spot, offset + delta)
        elif kind in (b"trak", b"mdia", b"minf", b"stbl"):
            _shift_chunks(data, payload, at + size, insert_at, delta)


def annotate(path, spot, mode=LEFT_RIGHT):
    """Write the projection boxes into a finished mp4.

    Returns True if they went in.  A file this cannot make sense of is left
    exactly as it was and reported rather than half-written: a clip that plays
    is worth more than one that is correctly labelled and does not.
    """
    data = bytearray(open(path, "rb").read())
    moov = _child(data, 0, len(data), b"moov")
    if moov is None:
        return False
    chain = _sample_entry(data, moov)
    if chain is None:
        return False

    extra = st3d(mode) + sv3d(spot)
    entry_start, entry_size, _ = chain[-1]
    insert_at = entry_start + entry_size

    grown = data[:insert_at] + extra + data[insert_at:]
    _grow(grown, chain, len(extra))
    # Every ancestor has already moved, so the chunk fix reads the new sizes.
    _shift_chunks(grown, moov[2], moov[0] + moov[1] + len(extra), insert_at, len(extra))
    open(path, "wb").write(grown)
    return True
