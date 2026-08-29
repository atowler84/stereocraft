"""Making a source big enough and smooth enough before any of it is converted.

The VR180 frame is the ceiling now, and a source is judged on whether it can
fill one.  Two ways it usually cannot:

**Not enough pixels.**  `vr180.natural_size` says what a source could fill, and
portrait video mostly says no -- 720 across a phone's 65 degrees stretches to
1980 of the 4096 that gets written.  `upscale` puts detail there rather than
resampling into it.

**Not enough frames.**  30fps is markedly worse in a headset than on a flat
screen.  60 is the target and not an arbitrary one: the decode ceiling falls as
frame rate rises -- 4096 an eye at 60, 3344 at 90, 2896 at 120 -- so 60 is the
fastest a clip can run and still be written at full angular resolution.

**Why this is a pass of its own rather than part of the loop.**  Measured: the
upscaler peaks at 9.4 GB on a 720x1280 chunk, and the 8192x4096 render peaks at
4.3 GB with the depth model holding 2 GB behind it.  On a 16 GB card those do
not fit together, and interleaving them would mean loading and unloading a model
per frame.  So the source is enhanced end to end, written once, and converted
from there with the enhancement models gone.

**Two passes and not one.**  The upscaler peaks at 9.4 GB on a 720x1280 chunk
and the interpolator is not free either, so they are run one after the other
with the first unloaded before the second starts, each writing its own
intermediate.  Upscaling goes first: super-resolution should propagate along
genuine motion rather than through invented frames, and interpolating first
would double the work of the more expensive of the two.

**Interpolation falls back to ffmpeg where the weights are missing.**  RIFE is
much the better of the two -- reconstructing held-out frames it scores 35.19 dB
against `minterpolate`'s 29.66, on a floor of 16.31 for doing nothing -- but a
machine without the checkpoint should still be able to smooth a clip, and
`minterpolate` is motion-compensated, already here, and needs nothing fetched.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from . import budget, vr180
from .pipeline import notice
from .depth import DEFAULT_FOCAL_35MM, focal_from_35mm
from .video import (NO_CONSOLE, codec_for, _even, _log, _read_exactly, _stop, _thrift,
                    _tool, pick_encoder)

# The frame rate below which a clip is worth interpolating, and what to take it
# to.  59 rather than 60 so that 59.94 -- which is what NTSC-descended footage
# actually reports -- is left alone rather than resampled for a rounding error.
SMOOTH_BELOW_FPS = 59.0
TARGET_FPS = 60.0
# How short of the ceiling a source has to be before upscaling is worth its
# cost.  A 1440-wide clip reaches 97% of a 4096 ceiling on its own, and the pass
# that would close the last 3% is the most expensive thing in the app -- a big
# source leaves room for only three frames at a time, so every frame is computed
# three times over.  Below this the gain is worth it; above it, plainly not.
WORTH_UPSCALING = 0.85
# Motion-compensated, bidirectional estimation, overlapped block compensation
# with variable block size.  The slow end of the filter's settings, which is the
# right end when the result is going to be looked at from the inside.
MINTERPOLATE = ("minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:"
                "me_mode=bidir:vsbmc=1")


def wanted(clip, settings):
    """What this clip is short of: `(upscale, target fps or None)`.

    Asked before anything is decoded, so the lens is the one `depth` assumes for
    a photo that has lost its EXIF -- no clip carries intrinsics and the metric
    model reports none, so it is the only lens available either way.
    """
    if settings.projection != "vr180":
        return False, None  # a flat pair is written at the source's own size
    cap = settings.vr180_cap or vr180.video_cap(clip.fps)
    lens = focal_from_35mm(DEFAULT_FOCAL_35MM, clip.width)
    short = vr180.natural_size(clip.width, lens) < cap * WORTH_UPSCALING
    return (bool(settings.upscale) and short,
            TARGET_FPS if (settings.interpolate and clip.fps < SMOOTH_BELOW_FPS) else None)


def target_width(clip, settings):
    """How wide to write the intermediate.

    Not the upscaler's full four times.  The projection discards anything past
    the width that fills the ceiling -- 1490 pixels for 4096 an eye -- so
    carrying 2880 to disk would be paying for pixels thrown away on the way in.
    A little over is kept, because supersampling into the bicubic sample is
    worth something and rounding to nothing is not.
    """
    cap = settings.vr180_cap or vr180.video_cap(clip.fps)
    lens = focal_from_35mm(DEFAULT_FOCAL_35MM, clip.width)
    fills = cap * vr180.fov(lens, clip.width) / vr180.FOV
    return _even(min(clip.width * 4, int(fills * 1.4)))


def _decoder(src, stderr):
    args = [_tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(src),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=stderr, bufsize=0, **NO_CONSOLE)


def _encoder(out, src, width, height, fps, clip, cfg, stderr):
    """Raw frames in, one silent intermediate out.

    **No soundtrack, and that is the point.**  This used to take the source file
    as a second input for its audio, alongside the pipe.  The two arrive at
    speeds that are not comparable -- a frame comes down the pipe once
    super-resolution or RIFE has finished with it, and the file reads as fast as
    the disk turns -- and a muxer cannot write an audio packet before the video
    packet it interleaves with exists.  So ffmpeg read the whole soundtrack ahead
    and held it, and what it held grew with the length of the clip.  On a long
    one that is how the pass ran the machine out of memory before the conversion
    proper had started.  See `video._encoder`, which had the same fault.

    Nothing is lost by dropping it here.  This file is an intermediate that is
    read back and deleted; `video._remux` takes the soundtrack from the original
    at the end, which is both correct and one generation of audio better than
    copying it through here first.
    """
    args = [_tool("ffmpeg"), "-v", "error", "-y", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-"]
    encoder, quality, extra = pick_encoder(codec_for(width, cfg.codec))
    # Kinder than the finished file's, because this one is read back and
    # converted rather than looked at, and its errors would be baked in.
    args += ["-c:v", encoder, quality, str(max(1, cfg.crf - 6)), *extra, "-pix_fmt", "yuv420p"]
    # An upscaled intermediate reaches four times the source width, so this is
    # every bit as able to ask for an impossible amount of encoder memory as the
    # finished frame is.
    args += _thrift(encoder, width, height)
    args += ["-an", str(out)]
    return subprocess.Popen(args, stdin=subprocess.PIPE, stderr=stderr, bufsize=0, **NO_CONSOLE)


def _frames(decoder, width, height):
    """Decoded frames as `[3, H, W]` float tensors in [0, 1]."""
    size = width * height * 3
    buffer = bytearray(size)
    view = memoryview(buffer)
    array = np.frombuffer(buffer, np.uint8).reshape(height, width, 3)
    while _read_exactly(decoder.stdout, view) == size:
        yield torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)


class _Stopped(Exception):
    """Raised through the model's own iteration to unwind a stopped pass."""


class _NoRoom(Exception):
    """Not enough memory to run a pass usefully, so it is skipped rather than
    run badly -- the conversion behind it is still worth having."""


def _pass(src, out, clip, settings, produce, width, height, in_fps, out_fps,
          label="working", expected=None, on_progress=None):
    """One decode, one model, one encode.

    Both stages have exactly this shape, which is why they share it: what
    differs is the callable in the middle and the size and rate that come out
    the other end.

    `expected` counts frames going *in* rather than coming out, which is
    deliberate: the upscaler holds a whole chunk before it returns any of it, so
    counting what comes out leaves the window still for the length of a chunk
    and gives a Stop button nothing to act on.  Measured, before this, that was
    four and a half minutes of apparently nothing happening.
    """
    done = 0
    with tempfile.TemporaryFile() as read_log, tempfile.TemporaryFile() as write_log:
        decoder = _decoder(src, read_log)
        encoder = _encoder(out, src, width, height, out_fps, clip, settings, write_log)
        def watched(frames):
            """Count and check on the way in, where frames arrive steadily."""
            for index, frame in enumerate(frames, 1):
                if on_progress and on_progress(label, index, expected) is False:
                    raise _Stopped
                yield frame

        try:
            for frame in produce(watched(_frames(decoder, clip.width, clip.height))):
                pixels = (frame.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
                encoder.stdin.write(pixels.contiguous().cpu().numpy().tobytes())
                done += 1
        except _Stopped:
            _stop(decoder, encoder)
            out.unlink(missing_ok=True)
            return None
        except BaseException:
            _stop(decoder, encoder)
            out.unlink(missing_ok=True)
            raise
        # Closed and waited for rather than stopped: `_stop` terminates, and an
        # encoder terminated between the last frame and its own trailer writes a
        # file that is not one.
        decoder.stdout.close()
        encoder.stdin.close()
        decoder.wait()
        encoder.wait()
        if encoder.returncode or not done:
            out.unlink(missing_ok=True)
            raise RuntimeError(f"could not write {out.name}: {_log(write_log)}")
    return done


def _device(settings):
    if settings.device != "auto":
        return torch.device(settings.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smoothed_count(frames, source_fps, target_fps):
    """How many frames come out of re-timing this many, so a stage that takes
    minutes has something to count against."""
    if not frames:
        return None
    return int(round((frames - 1) * float(target_fps) / float(source_fps))) + 1


def run(src, clip, settings, on_progress=None, work=None):
    """Write an enhanced copy of `src` and return `(path, clip)`, or `(None, clip)`.

    The returned clip describes the intermediate rather than the source, so the
    conversion behind this sizes itself to what it will actually read -- frame
    count included, which is what gives the conversion's own progress bar a
    denominator.

    `on_progress` is called `(label, done, expected)` and stops the run by
    returning False.  The labels are what the window shows instead of sitting on
    "converting" through two passes that each take minutes.

    Returns `False` in place of a path when it was stopped, which is a different
    answer from `None` -- one means there was nothing to do and the conversion
    should carry on with the original, the other means someone asked it to stop.
    """
    upscaling, smooth_to = wanted(clip, settings)
    if not upscaling and not smooth_to:
        return None, clip

    from .video import Clip

    work = Path(work or tempfile.gettempdir())
    stem = Path(src).stem
    reading, made = Path(src), []
    width, height, fps = clip.width, clip.height, clip.fps
    device = _device(settings)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    try:
        if upscaling:
            from . import upscale

            model = upscale.load(device=device, dtype=dtype)
            try:
                width = target_width(clip, settings)
                height = _even(round(width * clip.height / clip.width))
                chunk, overlap = upscale.chunk_plan(budget.free_bytes(device) or 4e9,
                                                    clip.width, clip.height, out_width=width)
                if not chunk:
                    # Doing it badly is worse than not doing it: over three or
                    # four frames there is no sequence left to propagate along,
                    # which was the whole reason for this model over a
                    # per-frame one.
                    notice(settings, f"{Path(src).name}: not enough memory to add detail to "
                                     f"a frame this big; converting without it")
                    raise _NoRoom
                if overlap < upscale.GOOD_OVERLAP:
                    # Worth a word: it is the joins between chunks that suffer,
                    # and a seam nobody was told about reads as a bad model.
                    # Said only below the run-up that measured as converged, and
                    # measured against the longest one the chunking can reach
                    # rather than against `OVERLAP`, which no card reaches -- a
                    # warning that fires on every run says nothing about this one.
                    notice(settings,
                           f"{Path(src).name}: not enough memory for a full run-up between "
                           f"chunks ({overlap} frames rather than {upscale.BEST_OVERLAP}); the "
                           f"picture may shift slightly every {chunk - 2 * overlap} frames")
                out = work / f"{stem}_upscaled.mp4"
                detail = (upscale.DETAIL if settings.upscale_detail is None
                          else float(settings.upscale_detail))
                count = _pass(reading, out, clip, settings,
                              lambda frames: upscale.run(model, frames, out_size=(height, width),
                                                         chunk=chunk, overlap=overlap,
                                                         device=device, dtype=dtype,
                                                         detail=detail),
                              width, height, fps, fps,
                              label="adding detail", expected=clip.frames,
                              on_progress=on_progress)
                skipped = False
            except _NoRoom:
                skipped, count = True, None
            finally:
                # The pass behind this wants the card back, and 9.4 GB of
                # upscaler is what would otherwise be standing in its way.
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if skipped:
                width, height = clip.width, clip.height  # nothing was resized after all
            elif count is None:
                return False, clip  # stopped part-way, and tidied up after itself
            else:
                made.append(out)
                reading = out
                clip = Clip(width, height, fps, count, clip.duration, clip.audio)

        if smooth_to:
            model = _interpolator(device, settings)
            if model is None:
                # No weights: let ffmpeg do it, which needs no frame to come
                # into Python at all.
                out = work / f"{stem}_smooth.mp4"
                if on_progress:
                    on_progress("smoothing", 0, None)
                _ffmpeg_smooth(reading, out, smooth_to, settings, clip)
                count = smoothed_count(clip.frames, fps, smooth_to)
            else:
                from . import interpolate

                try:
                    out = work / f"{stem}_smooth.mp4"
                    count = _pass(reading, out, clip, settings,
                                  lambda frames: interpolate.run(model, frames, fps, smooth_to),
                                  width, height, fps, smooth_to, label="smoothing",
                                  expected=smoothed_count(clip.frames, fps, smooth_to),
                                  on_progress=on_progress)
                    if count is None:
                        return False, clip
                finally:
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            made.append(out)
            reading = out
            fps = smooth_to
            clip = Clip(width, height, fps, count, clip.duration, clip.audio)
    except BaseException:
        for path in made:
            path.unlink(missing_ok=True)
        raise

    # Only the last one is read; the rest were rungs on the way to it.
    for path in made[:-1]:
        path.unlink(missing_ok=True)
    return reading, Clip(width, height, fps, clip.frames, clip.duration, clip.audio)


def _interpolator(device, settings):
    """RIFE if its weights are to hand, else None and ffmpeg does the work."""
    try:
        from . import interpolate

        return interpolate.load(device=device)
    except Exception as problem:  # no checkpoint, no network, a mirror gone
        notice(settings, f"smoothing with ffmpeg rather than RIFE, which is a little "
                         f"softer between frames: {problem}")
        return None


def _ffmpeg_smooth(src, out, fps, settings, clip):
    args = [_tool("ffmpeg"), "-v", "error", "-y", "-nostdin", "-i", str(src),
            "-vf", MINTERPOLATE.format(fps=fps)]
    encoder, quality, extra = pick_encoder(codec_for(clip.width, settings.codec))
    args += ["-c:v", encoder, quality, str(max(1, settings.crf - 6)), *extra,
             "-pix_fmt", "yuv420p"]
    # Silent for the same reason as `_encoder`: the soundtrack is put back at
    # the end, from the original rather than from this.
    args += ["-an", str(out)]
    if subprocess.run(args, capture_output=True, **NO_CONSOLE).returncode:
        raise RuntimeError(f"could not interpolate {Path(src).name}")
