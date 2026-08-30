"""Side-by-side 3D video: the still pipeline, run over every frame of a clip.

Each frame gets exactly what a photo gets -- the same depth network, the same
guided upsample, the same splat-and-resample renderer.  Two things have to be
added around it, and both come from the picture moving rather than from anything
about video files:

**Depth has to hold still.**  Depth-Anything is a per-frame model, and a
per-frame estimate wobbles.  In a depth map that reads as noise; turned into a
stereo pair it is the *geometry* that wobbles, which is a great deal harder to
look at.  `TemporalDepth` is the answer, and what it smooths and in what order is
the whole of the difference between a clip that is comfortable and one that is
not.

**Frames have to stay the same size.**  `make_pair` trims the sliver at each
edge that only one eye can see, and sizes that trim from the frame it is given.
A still always uses the full depth range so that comes out constant; a smoothed
clip does not, so the trim is pinned for the whole clip up front instead.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import budget, logbook, spherical, stereo, vr180
from .depth import DEFAULT_FOCAL_35MM, _app_dir, focal_from_35mm
from .pipeline import OUT_OF_MEMORY, Converter, VideoSettings, notice, tag

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts", ".m2ts", ".wmv", ".flv"}
# A child of a windowed Windows app is given a console window of its own, so
# every ffmpeg this runs would flash a black box over whatever the screen was
# showing -- one for the probe, and two that stay up for the whole encode.
# Harmless and horrible; the flag says to give it no window at all.
NO_CONSOLE = ({"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
              if sys.platform == "win32" else {})

# Audio an mp4 will carry as it stands.  Anything else is re-encoded, which is a
# generation of loss on the soundtrack but beats refusing the file.
COPYABLE_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac"}
# What to encode with, best first, per codec.  Which of these exists depends on
# how the ffmpeg to hand was built: an LGPL build carries no x264 or x265, both
# being GPL, so a portable app that ships its own cannot assume them.  Each entry
# is (encoder, quality flag, extra arguments) -- they do not agree on how quality
# is expressed, which is the other reason this cannot be one hardcoded name.
ENCODERS = {
    "h264": [("libx264", "-crf", ["-preset", "medium"]),
             ("h264_nvenc", "-cq", ["-rc", "vbr", "-preset", "p6"]),
             ("h264_qsv", "-global_quality", []),
             ("h264_amf", "-qp_i", []),
             ("libopenh264", "-q", []),
             ("h264_mf", "-q", [])],
    "hevc": [("libx265", "-crf", ["-preset", "medium"]),
             ("hevc_nvenc", "-cq", ["-rc", "vbr", "-preset", "p6"]),
             ("hevc_qsv", "-global_quality", []),
             ("hevc_amf", "-qp_i", []),
             ("hevc_mf", "-q", [])],
}
# Frame-to-frame change in depth, relative to the depth itself, that means the
# picture cut rather than moved.  Relative because metres have no fixed scale --
# an ordinary pan sits far below this at any distance.
SCENE_CUT = 0.35
# Frames between memory lines in the log.  Often enough that a climb is visible
# in the trail a killed process leaves behind, rare enough that an hour of
# conversion is a page rather than a book.
HEARTBEAT = 200
# Video memory the caching allocator may sit on unused before it is asked for it
# back.  Some slack is the whole point of a caching allocator -- handing every
# block back would mean asking the driver for it again next frame -- so this is
# well above the churn of one frame and well below the pool a big one leaves.
CUDA_SLACK = 2_000_000_000


class MissingFFmpeg(Exception):
    """ffmpeg is not installed, and nothing can be done about a video without it."""

    def __init__(self, name):
        super().__init__(
            f"{name} was not found. Video needs ffmpeg installed and on the PATH.\n"
            "  Debian/Ubuntu: sudo apt install ffmpeg\n"
            "  macOS:         brew install ffmpeg\n"
            "  Windows:       winget install ffmpeg\n"
            f"A copy of {name} sitting beside the app is used too, if there is one."
        )


def available_encoders():
    """The encoder names this ffmpeg was built with, asked once and remembered."""
    global _ENCODERS_SEEN
    if _ENCODERS_SEEN is None:
        result = subprocess.run([_tool("ffmpeg"), "-hide_banner", "-encoders"],
                                capture_output=True, text=True, **NO_CONSOLE)
        _ENCODERS_SEEN = {line.split()[1] for line in result.stdout.splitlines()
                          if line.startswith(" ") and len(line.split()) > 1}
    return _ENCODERS_SEEN


# Where h264 stops being the safe answer.  Not its own level limit -- 6.x goes
# to 8192 -- but what a headset will actually decode: 4K for h264, 8K only on
# h265 and AV1.  A VR180 frame is 8192 across and a full-width flat pair off a
# 4K source is 7448, so the frame is what decides this rather than the
# projection, which was the first thing tried and missed the second case.
H264_CEILING = 4096
# And where the encoder's own frame-parallelism stops being affordable.  x264 and
# x265 both keep several frames in flight at once -- a lookahead queue, the
# reference list, and one working copy per frame thread -- so their memory is a
# multiple of the frame rather than the frame.  At 8192 square a yuv420p frame is
# 100 MB before any analysis buffer, and the default parallelism on a many-core
# machine asks for tens of gigabytes of it.
#
# Slice threading is the way out and costs almost nothing at this size: it splits
# one frame across the cores instead of running several frames at once, and a
# frame 8192 tall is 128 CTU rows, which is more parallelism than any desktop has
# threads.  The compression loss that makes slice threading a bad default at
# 1080p is therefore not being paid here either.
THREAD_CEILING = 4096 * 4096


def codec_for(width, asked="auto"):
    """The codec to write a frame this wide with.

    `auto` is not a preference for hevc.  h264 plays on more than anything else
    does and CRF does not mean the same number in the two encoders -- 18 in x265
    is nearer 22 in x264 -- so switching quietly changes what a quality setting
    asks for.  It switches where h264 would simply not play.
    """
    if asked and asked != "auto":
        return asked
    return "hevc" if width > H264_CEILING else "h264"


def pick_encoder(codec):
    """The best encoder available for this codec, and how to ask it for quality.

    x264 first where it exists, being the best of them at a given size.  A build
    without it -- an LGPL one, most likely -- falls to the graphics card, and
    failing that to a software encoder that is not GPL.  Something is always
    there, so the choice never has to be explained to whoever is converting.
    """
    have = available_encoders()
    for name, quality, extra in ENCODERS.get(codec, ENCODERS["h264"]):
        if name in have:
            return name, quality, extra
    raise MissingFFmpeg(f"an encoder for {codec}")


_ENCODERS_SEEN = None


def _tool(name):
    """Find ffmpeg or ffprobe: on the PATH, or shipped next to the app."""
    if sys.platform == "win32":
        name += ".exe"
    found = shutil.which(name)
    if found:
        return found
    beside = os.path.join(_app_dir(), name)
    if os.path.exists(beside):
        return beside
    raise MissingFFmpeg(name)


@dataclass
class Clip:
    """What a video is, as far as any of this needs to care."""

    width: int
    height: int
    fps: float
    frames: int
    duration: float
    audio: object = None  # the audio stream's codec, or None if it is silent


def _fraction(text):
    """ffprobe writes frame rates as "30000/1001"; None for the 0/0 it uses to
    mean it does not know."""
    try:
        top, _, bottom = str(text).partition("/")
        rate = float(top) / float(bottom or 1)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def _rotation(stream):
    """How far the player is meant to turn this stream before showing it.

    Phones record sideways and note the rotation rather than rotating the
    pixels, so this is the difference between a portrait clip being understood
    as portrait and being handed to the depth model on its side.
    """
    tag = stream.get("tags", {}).get("rotate")
    if tag is not None:
        return int(round(float(tag)))
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            return int(round(float(side["rotation"])))
    return 0


def probe(path):
    """Everything about a clip that the conversion needs, in one ffprobe call."""
    result = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, **NO_CONSOLE,
    )
    if result.returncode != 0:
        raise ValueError(f"{Path(path).name} could not be read: {result.stderr.strip()}")
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"{Path(path).name} has no video track in it")

    width, height = int(video["width"]), int(video["height"])
    if _rotation(video) % 180 == 90:  # -90 lands on 90 here too, which is the point
        width, height = height, width
    # The average rate rather than the nominal one: a variable-rate phone clip
    # quotes something optimistic as `r_frame_rate`, and the average is what
    # keeps the soundtrack level with the picture.
    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate")) or 30.0
    duration = float(info.get("format", {}).get("duration") or video.get("duration") or 0.0)
    frames = int(video.get("nb_frames") or 0) or int(round(duration * fps))
    audio = next((s.get("codec_name") for s in streams if s.get("codec_type") == "audio"), None)
    return Clip(width, height, fps, frames, duration, audio)


class TemporalDepth:
    """The inverse depth map, with a memory of the frames before this one.

    This used to have two halves.  The larger one smoothed the percentile range
    that mapped raw depth onto 0-1, because measured afresh every frame it
    twitched as the scene moved -- a subject walking toward the camera shifted it
    and the whole map slid to compensate -- and that was the single biggest
    source of the wobble a per-frame model gave a clip.

    Metric depth removes the need for that range -- metres are metres whatever
    else is in the frame -- but it is worth being honest that this is not a free
    win.  Renormalising every frame was also cancelling the model's own global
    scale wobble, and measured against Depth-Anything V2 on a static shot, metric
    depth is about a third *noisier* frame to frame, not quieter.  Smoothing the
    metric scale back out was tried and made it worse: the noise is spread
    through the map rather than sitting in one global factor.

    It does not matter in the end, which is why a plain exponential average is
    all that is left here.  In the units that count -- how far the disparity
    field actually moves between frames -- both models sit around a tenth of a
    pixel before any smoothing at all, against roughly a third of a pixel for the
    smallest movement an eye can pick out.  The default halves that again.

    A cut is the one thing an average cannot be asked to sit through, so a frame
    that differs wholesale from the last is taken on its own and the memory
    starts again from there.  The test is relative, because depth in metres has
    no fixed scale to set an absolute threshold against: a room and a landscape
    are both ordinary, and differ by two orders of magnitude.
    """

    def __init__(self, keep=0.5, cut=SCENE_CUT):
        # 1.0 would freeze the first frame's depth over the whole clip, so the
        # knob stops short of it however far it is turned up.
        self.keep = min(max(float(keep), 0.0), 0.95)
        self.cut = cut
        self.previous = None
        self.cuts = 0

    def reset(self):
        self.previous = None

    def __call__(self, inverse):
        if self.previous is not None and self.previous.shape == inverse.shape:
            scale = float(inverse.abs().mean()) + 1e-9
            if float((inverse - self.previous).abs().mean()) / scale > self.cut:
                self.cuts += 1
                self.reset()

        if self.previous is None:  # the first frame, or the first after a cut
            self.previous = inverse
            return inverse

        depth = torch.lerp(inverse, self.previous, self.keep)
        self.previous = depth
        return depth


def clock(seconds):
    """A duration at the coarseness someone waiting on it actually reads."""
    seconds = int(max(seconds, 0))
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def _even(value):
    """yuv420p halves both dimensions, so both have to be even to start with."""
    return max(2, int(value) - int(value) % 2)


@dataclass
class Geometry:
    """Where every frame of this clip ends up, worked out once for all of them."""

    margin: int  # trimmed off each end of each eye, pinned so no frame differs
    eye: int  # width of one eye view in the finished frame
    height: int
    patch: object = None  # the piece of sphere it sits on, for a vr180 clip

    @property
    def width(self):
        return 2 * self.eye


def geometry(clip, settings):
    """The finished frame's shape.

    Half width per eye by default, which puts the clip out at the size it came
    in.  That is what players and headsets expect and what their hardware
    decoders can keep up with; `full_width` keeps every native pixel instead and
    doubles the frame, which is past what most of them will decode above 1080p.

    A vr180 clip is a piece of a sphere rather than a rectangle, and which piece
    has to be settled before the first frame is decoded: every frame must come
    out the size of the first, and the first has not been through the depth model
    yet to say what lens it was shot on.  So it assumes the lens most cameras
    have -- the same assumption `depth` falls back on for a photo that has lost
    its EXIF, and the only one available here, since no clip carries intrinsics
    and the metric model reports none.

    How big it can be comes from the frame rate, a hardware decoder being
    limited by pixels per second rather than by pixels -- see `vr180.video_cap`.
    """
    if settings.projection == "vr180":
        assumed = focal_from_35mm(DEFAULT_FOCAL_35MM, clip.width)
        cap = settings.vr180_cap or vr180.video_cap(clip.fps)
        spot = vr180.patch(assumed, clip.width, clip.height, cap,
                           None if settings.vr180_size in (None, 0, "auto")
                           else int(settings.vr180_size))
        return Geometry(margin=0, eye=spot.width, height=spot.height, patch=spot)
    margin = stereo.max_margin(clip.width, settings.limit_pct)
    eye = clip.width - 2 * margin if settings.full_width else clip.width // 2
    return Geometry(margin=margin, eye=_even(eye), height=_even(clip.height))


def output_path(src, dst=None, name="_full_sbs"):
    src = Path(src)
    if dst is None:
        return src.with_name(f"{src.stem}{name}.mp4")
    dst = Path(dst)
    if dst.is_dir() or dst.suffix == "":
        return dst / f"{src.stem}{name}.mp4"
    return dst


def _write_all(stream, frame):
    """Push one finished frame down the pipe, the whole of it.

    A raw pipe may take less than it is handed, and at 8K a frame is 100 MB --
    far past what either platform moves in one call -- so the loop is what makes
    a short write a detail rather than a torn frame.  `write` returning None is
    a buffered stream saying it took everything.
    """
    view = memoryview(frame).cast("B")
    while view:
        written = stream.write(view)
        if written is None:
            return
        if not written:
            raise BrokenPipeError("the encoder stopped taking frames")
        view = view[written:]


def _read_exactly(stream, view):
    """Fill `view` from `stream`, or report how far it got at the end of the file.

    A pipe hands over whatever it has rather than what was asked for, so a frame
    arrives in as many pieces as the operating system feels like.
    """
    got = 0
    while got < len(view):
        read = stream.readinto(view[got:])
        if not read:
            break
        got += read
    return got


def _decoder(src, clip, size, stderr):
    """ffmpeg reading the clip out as raw RGB, one frame after another.

    Constant frame rate on the way out even if the file is variable: the raw
    frames carry no timestamps, so anything else would leave the picture and the
    soundtrack drifting apart over the length of the clip.  Rotation is applied
    by ffmpeg itself, so frames arrive the right way up.
    """
    args = [_tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(src)]
    if size is not None:
        args += ["-vf", f"scale={size[0]}:{size[1]}"]
    args += ["-fps_mode", "cfr", "-r", f"{clip.fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=stderr, bufsize=0,
                            **NO_CONSOLE)


def wants_sound(clip, cfg):
    """Whether this conversion should carry the original soundtrack."""
    return bool(cfg.audio and clip.audio)


def _audio_args(clip):
    """Copied when the container will take it as it is, so the soundtrack goes
    through untouched; re-encoded only when it would otherwise be refused."""
    return (["-c:a", "copy"] if clip.audio in COPYABLE_AUDIO
            else ["-c:a", "aac", "-b:a", "192k"])


def _encoder(out, clip, geo, cfg, stderr, faststart=True):
    """ffmpeg taking finished frames on stdin, and writing the picture alone.

    **The soundtrack used to be muxed in right here, and that was the leak.**
    The old command gave ffmpeg two inputs: this pipe, and the source file for
    its audio.  They arrive at wildly different speeds -- a frame comes down the
    pipe once the depth model and an 8K render have finished with it, which is
    the better part of a second, while the source file reads at disk speed -- and
    a muxer cannot write an audio packet until the video packet it interleaves
    with has turned up.  So ffmpeg reads the soundtrack far ahead and holds it,
    and what it holds is proportional to how long the clip is.  An hour of
    audio is tens of thousands of packets sitting in the muxing queue behind a
    picture that is still on frame 200.

    That is why this failed on long clips and not on short ones, and why the
    error came from ffmpeg rather than from us.  The picture is written on its
    own here and `_remux` puts the sound back afterwards, where both sides are
    files and neither has to wait for the other.

    `faststart` is left to whichever pass writes the finished file: doing it here
    as well would rewrite a very large file twice for no gain.
    """
    args = [_tool("ffmpeg"), "-v", "error", "-y", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{geo.width}x{geo.height}", "-r", f"{clip.fps}", "-i", "-"]
    codec = codec_for(geo.width, cfg.codec)
    encoder, quality, extra = pick_encoder(codec)
    if encoder != ENCODERS[codec][0][0]:
        # Worth a word.  The preferred encoder is the best of them at a given
        # size, and falling past it is invisible in the finished file unless
        # someone thinks to ask ffprobe -- so a clip that came out softer than
        # the last one would look like the app's fault rather than the ffmpeg's.
        notice(cfg, f"{Path(out).name}: encoding with {encoder}; this ffmpeg has no "
                    f"{ENCODERS[codec][0][0]}")
    args += ["-c:v", encoder, quality, str(cfg.crf), *extra, "-pix_fmt", "yuv420p"]
    args += _thrift(encoder, geo.width, geo.height)
    if faststart:
        args += ["-movflags", "+faststart"]
    args.append(str(out))
    return subprocess.Popen(args, stdin=subprocess.PIPE, stderr=stderr, bufsize=0,
                            **NO_CONSOLE)


def _thrift(encoder, width, height):
    """Encoder settings that trade frame parallelism for memory, above
    `THREAD_CEILING`.  Nothing at all below it, where the defaults are right."""
    if width * height <= THREAD_CEILING:
        return []
    if encoder == "libx265":
        # Frame threads are what multiply the memory; WPP, which is on by
        # default, keeps the cores busy inside the one frame instead.  The
        # lookahead is halved for the same reason -- every frame in it is another
        # 100 MB held.
        return ["-x265-params", "frame-threads=1:rc-lookahead=10"]
    if encoder == "libx264":
        return ["-x264-params", "sliced-threads=1"]
    return []


def _remux(picture, src, out, clip, cfg, stderr):
    """Put the soundtrack back and move the index to the front, in one pass.

    Both inputs are files, so neither waits on the other and nothing is buffered
    to speak of -- which is the whole point of doing it here rather than during
    the encode.  The picture is copied rather than re-encoded, so this runs at
    the speed of the disk.

    No -shortest.  It looks like the safe option and is not: an AAC track carries
    a little encoder priming, which makes ffmpeg reckon it the shorter stream and
    truncate the picture to match.  On a 90-frame clip that quietly cost three
    frames off the end.  Every frame rendered is a frame worth keeping; a
    fractionally long soundtrack is harmless.
    """
    args = [_tool("ffmpeg"), "-v", "error", "-y", "-nostdin",
            "-i", str(picture), "-i", str(src),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            *_audio_args(clip), "-movflags", "+faststart", str(out)]
    return subprocess.run(args, stderr=stderr, **NO_CONSOLE).returncode


def _headroom_wanted(geo):
    """Roughly what a conversion of this frame size needs the machine to have
    left, in bytes of commit.

    Measured rather than reasoned.  An 8192x4096 clip sat at 10 GB of commit in
    this process with 5 GB in the encoder beside it, and the frame is what drives
    both -- the renderer holds a handful of them in flight and the encoder holds
    a lookahead of them.  A fixed base covers the models and the runtime, which
    are there whatever size the picture is.
    """
    return 4_000_000_000 + 80 * geo.width * geo.height * 3


def _warn_if_full(cfg, name, geo):
    """Say so *before* the wait if the machine is already close to full.

    A conversion at this size runs for hours, and being refused a single frame
    at the end of it loses all of them.  Whether the memory has gone on this
    app or on something else is not knowable from here and does not change the
    advice, so the check is simply whether there is room -- asked once, at the
    start, while there is still something the person can do about it.

    Not a guess at the cause of any particular failure.  It is the cheap
    question that was never asked, and an eleven-hour job deserves it.
    """
    free = logbook.headroom()
    wanted = _headroom_wanted(geo)
    logbook.note("headroom", free=logbook._gb(free), wanted=logbook._gb(wanted))
    if free is None or free >= wanted:
        return
    notice(cfg, f"{name}: this machine has about {free / 1e9:.0f} GB of memory left to "
                f"promise and a {geo.width}x{geo.height} conversion wants nearer "
                f"{wanted / 1e9:.0f} GB. It can run for hours and then be refused a "
                f"single frame; closing whatever else is running is the cure.")


def _decode_size(converter, src, clip):
    """The size to decode at: native, something smaller that fits, or nothing.

    One question for the whole file rather than one per frame, since every frame
    of a clip is the same size -- and asked before a single frame is decoded.
    Returns None for native, a `(width, height)` to scale to on the way in, or
    False when the answer was to skip the clip.
    """
    cfg = converter.settings
    estimator = converter.depth_model
    # A vr180 clip is priced on the frame it comes out as rather than the one it
    # went in as: the projection spreads it across a hemisphere, and a 640x480
    # source priced on itself passes this check and runs out of memory later.
    geo = geometry(clip, cfg)
    frame = geo.width * geo.height if cfg.projection == "vr180" else None
    if budget.fits(estimator, clip.width, clip.height, cfg.depth_size, frame=frame):
        return None
    oversize = converter._proposal(src, (clip.width, clip.height), [estimator.device])
    if converter._decide(oversize) != "resize" or oversize.target is None:
        return False
    return _even(oversize.target[0]), _even(oversize.target[1])


def _fall_back_to_cpu(converter):
    """Move off the GPU mid-clip, for when something else on the machine takes
    the video memory out from under a conversion already forty frames in.

    Worth doing rather than giving up: the frames already encoded stay encoded
    and the clip finishes slowly, instead of the work being thrown away.
    """
    if converter.depth_model.device.type == "cpu":
        return False
    notice(converter.settings,
           "out of video memory part-way through; carrying on with the CPU")
    converter.settings.device = "cpu"
    converter._depth = None
    torch.cuda.empty_cache()
    return True


def _release(device):
    """Hand back the video memory the allocator is holding but not using.

    **On Windows this is not housekeeping; it is the difference between fitting
    and not.**  The display driver keeps a system-memory backing store for every
    byte of video memory a process reserves, so PyTorch's caching pool is
    charged to host commit as well as to the card -- and that pool only ever
    grows, settling at the high-water mark of the largest thing it ever held.

    Measured on the machine this failed on: 8 GB reserved on the card cost
    9.37 GB of host commit with the resident set still at 0.66 GB, and giving it
    back brought the commit to 1.46 GB.  A conversion that had peaked once at
    ten gigabytes of video memory therefore went on asking the system to promise
    about seven it had no further use for -- for the remaining four hours, on a
    machine with 31 GB, beside an encoder wanting five and a half of its own.

    At the heartbeat rather than every frame: the pool refills from the next
    allocation, so paying for it constantly would be a real cost for no further
    gain, while once every couple of hundred frames is nothing.
    """
    if device.type != "cuda":
        return
    if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() > CUDA_SLACK:
        torch.cuda.empty_cache()


def _squeeze(eye, width, height):
    """Resize one eye view to its place in the finished frame.

    Each eye separately, never the finished pair: resampling across the seam
    would blend the two eyes into one another at the join.
    """
    if eye.shape[-2:] == (height, width):
        return eye
    return F.interpolate(eye[None], size=(height, width), mode="bilinear",
                         align_corners=False, antialias=True)[0]


def convert_video(src, dst=None, converter=None, on_progress=None, on_frame=None,
                  on_stage=None, **kwargs):
    """Convert one clip into a side-by-side 3D one.

    `converter` keeps the depth model loaded across several clips; without one,
    `kwargs` are `VideoSettings` fields for a single conversion.  `on_progress`
    is called with `(frames done, frames expected, seconds so far)` and returning
    False from it cancels, leaving no half-written file behind.  `on_stage` is
    called `(label, done, expected)` through the passes that run before the
    conversion -- each of which takes minutes -- and cancels the same way; a
    caller that does not take it sees nothing until the conversion itself
    starts, which is a long time to look at a still bar.  `on_frame` is
    handed each finished frame as it goes by, for showing the work in progress;
    it costs nothing, the array having had to be made for the encoder anyway --
    but it is the *same* array every time, so anything keeping a frame past the
    call has to copy it.

    Returns what was made, or None if it was cancelled or skipped.
    """
    if converter is None:
        converter = Converter(VideoSettings(**kwargs))
    elif kwargs:
        raise TypeError("give convert_video a converter or settings, not both")

    src = Path(src)
    cfg = converter.settings
    started = time.perf_counter()
    clip = probe(src)
    source = (clip.width, clip.height)

    # Enhanced first and separately, because the models that do it will not fit
    # on a card beside the depth model and an 8K render -- see `prepass`.  What
    # comes back describes the intermediate, so everything below sizes itself to
    # what it will actually read; `src` stays the original, for the output name.
    # Imported here rather than at the top: `prepass` reaches back into this
    # module for the decoder and encoder plumbing, and the two would deadlock.
    from . import prepass

    with logbook.stage("prepass", clip=src.name, frames=clip.frames):
        enhanced, clip = prepass.run(src, clip, cfg, on_progress=on_stage,
                                     work=(dst if dst and Path(dst).is_dir() else None))
    if enhanced is False:  # stopped part-way through a pass, and cleaned up after itself
        return None
    reading = enhanced or src

    size = _decode_size(converter, reading, clip)
    if size is False:  # too big, and the answer was to leave it alone
        return None
    if size is not None:
        clip = Clip(size[0], size[1], clip.fps, clip.frames, clip.duration, clip.audio)

    if converter.depth_model.device.type == "cpu":
        # Roughly five and a half seconds per megapixel of network input, measured
        # on an eight-core desktop.  Rough is enough: the point is to say "hours"
        # before someone finds out by waiting, not to be right to the minute.
        work = converter.depth_model.working_size(clip.height, clip.width, cfg.depth_size)
        each = 5.4 * work[0] * work[1] / 1e6
        total = each * (clip.frames or 1)
        if total > 600:
            notice(cfg, f"{src.name}: this is a CPU conversion -- about {each:.0f}s a frame, "
                        f"so {clock(total)} for {clip.frames} frames. A graphics card does it "
                        f"in minutes, and the da2-small depth model is the quickest way "
                        f"through without one.")

    geo = geometry(clip, cfg)

    # The scene around the picture, gathered once per shot before a single frame
    # is rendered -- which is the whole reason it cannot boil.  Cheap next to the
    # two passes above it: one decode at a fraction of the width, and no encode.
    from . import plate as plates_module

    plates = None
    if plates_module.wanted(clip, cfg):
        try:
            with logbook.stage("surround", clip=src.name, frames=clip.frames):
                plates = plates_module.build(reading, clip, cfg, on_progress=on_stage,
                                             device=converter.depth_model.device,
                                             estimator=converter.depth_model)
        except Exception as problem:
            # A surround that could not be built is a smaller picture, not a
            # failed one; the wash is still there behind it.
            notice(cfg, f"{src.name}: could not look around the clip, so the surround is "
                        f"the usual blur: {plates_module._reason(problem)}")
        if plates is False:  # stopped while looking around
            if enhanced:
                enhanced.unlink(missing_ok=True)
            return None
        if plates is not None:
            plates = plates.to(converter.depth_model.device)

    out = output_path(src, dst, tag(cfg))
    out.parent.mkdir(parents=True, exist_ok=True)
    # The picture is written on its own first whenever there is a soundtrack to
    # add, and `_remux` makes `out` from it; see `_encoder` for why the two
    # cannot be one pass.  With no sound there is nothing to add and the encoder
    # writes the finished file directly.
    # The passes above peak far higher than the render does -- the upscaler alone
    # is nine gigabytes of card -- and every byte still reserved is a byte of
    # host commit the render does not need.  See `_release`.
    _release(converter.depth_model.device)
    sound = wants_sound(clip, cfg)
    picture = out.with_name(f"{out.stem}.picture{out.suffix}") if sound else out
    _warn_if_full(cfg, src.name, geo)
    logbook.note("clip", name=src.name, size=f"{clip.width}x{clip.height}",
                 frames=clip.frames, fps=round(clip.fps, 3),
                 out=f"{geo.width}x{geo.height}", codec=codec_for(geo.width, cfg.codec),
                 sound=sound, shots=(len(plates.shots) if plates is not None else 0))
    normalizer = TemporalDepth(cfg.temporal)

    frame_bytes = clip.width * clip.height * 3
    buffer = bytearray(frame_bytes)
    view = memoryview(buffer)
    # Reused rather than reallocated per frame, and writable so that handing it
    # to Torch does not have to copy it first.
    frame = np.frombuffer(buffer, np.uint8).reshape(clip.height, clip.width, 3)

    # One buffer for the finished frame, filled again every time rather than
    # made again every time.  At 8192x4096 a frame is 100 MB, and the old line
    # allocated two of them per frame -- one for the download off the card, one
    # more for the `tobytes` handed to the pipe.  Over the thirty thousand frames
    # of a long clip that is six terabytes of allocation for a picture that is
    # always exactly the same size, and it is asked for on a machine where the
    # encoder next door is trying to reserve 100 MB of its own.
    canvas = torch.empty((geo.height, geo.width, 3), dtype=torch.uint8)
    pixels = canvas.numpy()

    # Pinned for every frame when the projection is a sphere, and meaningless
    # when it is not.
    square = geo.patch
    done, cancelled = 0, False
    # What the caller asked for, to be put back afterwards: a clip that had
    # to finish on the CPU should not decide that for the clips behind it,
    # any more than one oversized photo decides it for the rest of a folder.
    requested = cfg.device
    try:
        with tempfile.TemporaryFile() as decode_log, tempfile.TemporaryFile() as encode_log:
            decoder = _decoder(reading, clip, size, decode_log)
            encoder = _encoder(picture, clip, geo, cfg, encode_log,
                               faststart=picture == out)
            try:
                while _read_exactly(decoder.stdout, view) == frame_bytes:
                    backdrop = plates.at(done, square) if plates is not None else None
                    try:
                        left, right, _ = converter.render(frame, normalizer, geo.margin,
                                                          spot=square, plate=backdrop)
                    except OUT_OF_MEMORY:
                        if not _fall_back_to_cpu(converter):
                            raise
                        normalizer.reset()  # its memory is on a device we have just left
                        if plates is not None:  # and so is the plate's
                            plates = plates.to(converter.depth_model.device)
                            backdrop = plates.at(done, square)
                        left, right, _ = converter.render(frame, normalizer, geo.margin,
                                                          spot=square, plate=backdrop)

                    left = _squeeze(left, geo.eye, geo.height)
                    right = _squeeze(right, geo.eye, geo.height)
                    sbs = stereo.compose(left, right, cfg.cross_eyed)
                    canvas.copy_((sbs.clamp(0, 1) * 255).round().to(torch.uint8)
                                 .permute(1, 2, 0).contiguous())
                    _write_all(encoder.stdin, pixels)
                    if on_frame is not None:
                        on_frame(pixels)

                    done += 1
                    if done % HEARTBEAT == 0:
                        current, peak = logbook.memory()
                        card = logbook.cuda_memory()
                        # The encoder as well as ourselves.  It is a separate
                        # process and it is the one that ran out of memory, so
                        # measuring only this side is measuring the wrong half.
                        _release(converter.depth_model.device)
                        enc_rss, enc_commit = logbook.process_memory(encoder.pid)
                        logbook.note("rendering", frame=done, of=clip.frames,
                                     **logbook.usage(),
                                     rss=logbook._gb(current), peak=logbook._gb(peak),
                                     enc=logbook._gb(enc_rss),
                                     enc_commit=logbook._gb(enc_commit),
                                     **({"cuda": logbook._gb(card[0]),
                                         "cuda_held": logbook._gb(card[1])} if card else {}))
                    if on_progress and on_progress(done, clip.frames, time.perf_counter() - started) is False:
                        cancelled = True
                        break
            except BrokenPipeError:
                # The encoder died, and its own complaint is more use than the broken
                # pipe that is all this end saw of it.
                _stop(decoder, encoder)
                _discard(out, picture)
                raise RuntimeError(f"ffmpeg could not encode {out.name}: {_log(encode_log)}") from None
            except BaseException:  # Ctrl-C, out of memory, a frame that would not render
                _stop(decoder, encoder)
                _discard(out, picture)
                raise

            if cancelled:
                _stop(decoder, encoder)
                _discard(out, picture)
                return None

            decoder.stdout.close()
            encoder.stdin.close()
            decoder.wait()
            encoder.wait()
            if not done:
                _discard(out, picture)
                raise RuntimeError(f"no frames came out of {src.name}: {_log(decode_log)}")
            if encoder.returncode:
                _discard(out, picture)
                raise RuntimeError(f"ffmpeg could not write {out.name}: {_log(encode_log)}")

            if sound:
                with tempfile.TemporaryFile() as mux_log:
                    if _remux(picture, src, out, clip, cfg, mux_log):
                        # The picture is finished and only the soundtrack failed,
                        # so the clip is worth keeping without it -- the whole
                        # conversion is in that file, and it is hours of it.
                        notice(cfg, f"{out.name}: the soundtrack could not be carried "
                                    f"over, so the clip is silent: {_log(mux_log)}")
                        picture.replace(out)
                    else:
                        picture.unlink(missing_ok=True)

        # Said last, into the finished file, because ffmpeg will not say it at
        # all.  A failure here is worth a word rather than an exception: the clip
        # plays, it is only unlabelled.
        marked = False
        if geo.patch is not None:
            marked = spherical.annotate(
                out, geo.patch,
                spherical.RIGHT_LEFT if cfg.cross_eyed else spherical.LEFT_RIGHT)
            if not marked:
                notice(cfg, f"{out.name}: could not write the projection boxes; the clip is "
                            f"fine but a player will have to be told what it is")

        seconds = time.perf_counter() - started
        current, peak = logbook.memory()
        logbook.note("converted", name=out.name, frames=done, cuts=normalizer.cuts,
                     took=f"{seconds:.1f}s", rss=logbook._gb(current),
                     peak=logbook._gb(peak))
        return {
            "input": src,
            "output": out,
            "source_size": (clip.width, clip.height),
            "resized_from": source if size else None,
            "output_size": (geo.width, geo.height),
            "frames": done,
            "cuts": normalizer.cuts,
            "coverage": converter.covered,
            "lens": converter.lens,
            "patch": geo.patch,
            "enhanced": (clip.width, clip.height, clip.fps) if enhanced else None,
            "surround": None if plates is None else vr180.coverage(plates.reach(square), square),
            "marked": marked if geo.patch is not None else None,
            "fps": clip.fps,
            "seconds": seconds,
        }
    finally:
        if converter.settings.device != requested:
            converter.settings.device, converter._depth = requested, None
        if enhanced:  # False when a pass was stopped, and it tidied up itself
            enhanced.unlink(missing_ok=True)
        if picture != out:  # only ever still here if something went wrong late
            picture.unlink(missing_ok=True)


def _stop(*processes):
    for process in processes:
        for pipe in (process.stdin, process.stdout):
            if pipe and not pipe.closed:
                try:
                    pipe.close()
                except OSError:
                    pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def _discard(out, picture):
    """Throw away a conversion, whichever of the two files it had got to."""
    for path in {out, picture}:
        path.unlink(missing_ok=True)


def _log(handle, lines=3):
    """What ffmpeg said: the tail for the message, the whole of it for the log.

    This used to return the last line alone.  That is the right length for an
    error someone reads in a dialog and the wrong length for one they have to
    diagnose -- an encoder running out of memory says so over several lines, and
    the last of them is the least informative.  So the tail is what is shown and
    the rest goes to `logbook`, where it is still there afterwards.
    """
    handle.seek(0)
    text = handle.read().decode("utf-8", "replace").strip()
    if not text:
        return "no reason given"
    logbook.note("ffmpeg said", lines=len(text.splitlines()))
    logbook.log.debug("ffmpeg output:\n%s", text)
    return " / ".join(text.splitlines()[-lines:])
