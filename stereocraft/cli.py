"""Command line entry point."""

import argparse
import glob
import sys
import time
from pathlib import Path

from . import __version__
from .pipeline import SBS_TAGS, SUFFIXES, Converter, Settings, VideoSettings
from .video import VIDEO_SUFFIXES, clock, convert_video


MEDIA = SUFFIXES | VIDEO_SUFFIXES


def is_video(path):
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def collect(inputs):
    """Expand files, directories and globs into a sorted list of photos and clips."""
    found = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            found += sorted(p for p in path.iterdir() if p.suffix.lower() in MEDIA)
        elif path.exists():
            found.append(path)
        else:  # let the shell off the hook for unexpanded globs
            matches = sorted(Path(match) for match in glob.glob(item))
            if not matches:
                print(f"skipping {item}: not found", file=sys.stderr)
            found += matches
    # Its own output, left where it was written.  Pointed at the same folder
    # twice this would otherwise convert the conversions, and every projection
    # the app can write has to be listed here or it would do exactly that.
    return [p for p in found if not p.stem.endswith(SBS_TAGS + ("_depth",))]


def oversize_handler(mode):
    """What to do about a photo that will not fit, as `Settings.on_oversize`.

    Asking needs someone there to answer, so a pipe or a cron job quietly gets
    the safe half of the choice rather than blocking forever on a prompt.
    """
    def settled(oversize):
        # The outcome line covers an ordinary resize; a photo no resize can
        # save still needs explaining.
        if oversize.target is None:
            print(f"\n{oversize.describe()}", file=sys.stderr)
        return mode

    if mode in ("resize", "skip"):
        return settled

    def ask(oversize):
        print(f"\n{oversize.describe()}", file=sys.stderr)
        if oversize.target is None:
            return "skip"  # nothing to offer, and describe() has said why
        if not sys.stdin.isatty():
            print("Not running interactively, so skipping it. "
                  "Pass --oversize resize to shrink photos like this instead.", file=sys.stderr)
            return "skip"
        while True:
            answer = input("Resize and convert it, or skip it? [r/s] ").strip().lower()
            if answer in ("r", "resize"):
                return "resize"
            if answer in ("s", "skip", ""):
                return "skip"

    return ask


class Defaults(argparse.ArgumentDefaultsHelpFormatter):
    """The stock formatter, less the "(default: None)" it would otherwise print
    against the two settings whose default depends on whether the input moves."""

    def _get_help_string(self, action):
        return action.help if action.default is None else super()._get_help_string(action)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="stereocraft",
        description="Turn a photo or a video into a side-by-side 3D one.",
        formatter_class=Defaults,
    )
    parser.add_argument("inputs", nargs="*", help="image or video files, or folders")
    parser.add_argument("-o", "--output", help="output file, or folder for several inputs")
    parser.add_argument("-e", "--eyes", default=None, metavar="MM",
                        help="distance between the two eyes, in millimetres, or auto to size it "
                             "to the scene. 65 is the human average, and worth trying, but a "
                             "close-up wants less and a landscape a great deal more")
    parser.add_argument("-f", "--focus", default=None, metavar="METRES",
                        help="how far away the screen plane sits, or auto. Whatever is at this "
                             "distance has no separation; nearer comes forward, further recedes")
    parser.add_argument("-t", "--target", type=float, default=None, metavar="PCT",
                        help="what auto aims for: near-to-far separation as a %% of frame width. "
                             "(default: 2.0 photo, 1.3 video)")
    parser.add_argument("--limit", type=float, default=3.0, metavar="PCT",
                        help="ceiling on separation as a %% of frame width, so something very "
                             "close cannot demand more parallax than an eye can fuse")
    parser.add_argument("-m", "--model", choices=("da3", "da2-small", "da2-base", "da2-large"),
                        default="da3",
                        help="da3 measures depth in metres; the da2 models only rank it, and are "
                             "fitted onto an assumed range")
    parser.add_argument("--depth-size", default=None,
                        help="longest side fed to the depth model (shortest, for the da2 models), "
                             "or auto to follow the frame. (default: auto photo, 1400 video)")
    # The old names meant something this no longer computes.  Failing loudly beats
    # translating them into a number that only looks like what was asked for.
    parser.add_argument("-d", "--disparity", type=float, help=argparse.SUPPRESS)
    parser.add_argument("-c", "--convergence", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--projection", choices=("flat", "vr180"), default="flat",
                        help="flat writes a rectilinear pair, which a player shows on a virtual "
                             "screen. vr180 wraps it onto a 180-degree hemisphere at its true "
                             "angular scale instead -- more immersive where the photo reaches, "
                             "and dark where it does not, which is most of it")
    parser.add_argument("--vr180-size", type=int, default=None, metavar="PX",
                        help="stored width per eye for --projection vr180, each eye being a "
                             "square 180 degrees across. (default: as much of the source's own "
                             "detail as fits, capped at 4096 for a photo and 2048 for a clip)")
    parser.add_argument("--vr180-surround", action="store_true",
                        help="fill the part of the sphere the picture never reached with a dim, "
                             "blurred spread of the picture rather than leaving it black -- what "
                             "a social video site puts behind a clip that does not fill the "
                             "frame. A fixed function of each frame, so a clip cannot crawl")
    parser.add_argument("--cross", action="store_true", help="write right|left for cross-eyed viewing")
    parser.add_argument("--max-size", type=int, default=0, help="cap the output width, 0 for native")
    parser.add_argument("--format", choices=("auto", "jpg", "png"), default="auto", dest="fmt")
    parser.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--save-depth", action="store_true", help="also write a 16-bit depth map")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--oversize", choices=("ask", "skip", "resize"), default="ask",
                        help="a photo too big for memory: ask, skip it, or resize it to fit")

    video = parser.add_argument_group("video")
    video.add_argument("--full", action="store_true",
                       help="keep every native pixel, doubling the frame width, instead of "
                            "squeezing each eye to half width as players expect")
    video.add_argument("--temporal", type=float, default=0.5, metavar="KEEP",
                       help="how much of the previous frame's depth to carry over, 0 to 0.95. "
                            "Steadies a clip that shimmers, at a little edge sharpness; 0 is off")
    video.add_argument("--crf", type=int, default=18, help="encoder quality, lower is better")
    video.add_argument("--codec", choices=("h264", "hevc"), default="h264",
                       help="hevc is worth it above 4K, where h264 runs out of level")
    video.add_argument("--no-audio", action="store_true", help="leave the soundtrack behind")
    parser.add_argument("--gui", action="store_true", help="open the desktop window instead")
    parser.add_argument("-V", "--version", action="version", version=f"StereoCraft {__version__}")
    return parser


def reporter(name, prefix):
    """Progress for a clip, which unlike a photo takes long enough to need it.

    One line that rewrites itself, and only when someone is there to watch it --
    redirected into a file it would leave thousands of lines nobody reads, so a
    pipe gets nothing and the finished line at the end says it all.
    """
    if not sys.stderr.isatty():
        return None
    last = warm = 0.0

    def report(done, total, seconds):
        nonlocal last, warm
        now = time.monotonic()
        # The first frame pays for the graphics driver building its kernels, and
        # counting that against all the rest puts minutes on the first estimate.
        if done == 1:
            warm = seconds
        if done < total and now - last < 0.5:  # a frame can land every few ms
            return True
        last = now
        rate = (done - 1) / (seconds - warm) if done > 1 and seconds > warm else 0
        left = f"  {clock((total - done) / rate)} left" if rate and total > done else ""
        share = f"{done}/{total}" if total else f"{done}"
        print(f"\r{prefix}{name}  frame {share}{left}      ", end="", file=sys.stderr, flush=True)
        return True

    return report


def _number(value):
    """A setting that is either a measurement or the word auto."""
    if value is None or str(value).lower() == "auto":
        return "auto"
    return float(value)


def settings_for(args, video):
    """The settings for one kind of input.

    `--eyes` and `--depth-size` are left unset by default so that each kind can
    bring its own, which is the whole of how a clip ends up gentler than a photo
    without anyone having to ask for it.
    """
    common = dict(
        model=args.model,
        focus_m=_number(args.focus),
        limit_pct=args.limit,
        cross_eyed=args.cross,
        device=args.device,
        projection=args.projection,
        vr180_surround=args.vr180_surround,
        on_oversize=oversize_handler(args.oversize),
    )
    if video:
        settings = VideoSettings(**common, full_width=args.full, temporal=args.temporal,
                                 crf=args.crf, codec=args.codec, audio=not args.no_audio)
    else:
        settings = Settings(**common, max_size=args.max_size, quality=args.quality,
                            fmt=args.fmt, save_depth=args.save_depth)
    if args.eyes is not None:
        settings.eyes_mm = _number(args.eyes)
    if args.target is not None:
        settings.target_pct = args.target
    if args.depth_size is not None:
        settings.depth_size = args.depth_size
    if args.vr180_size is not None:
        settings.vr180_size = args.vr180_size
    return settings


def retired(args):
    """The settings that used to exist, and what to say to someone still using them.

    Depth is measured in metres now, so a percentage of frame width and a
    normalised screen plane no longer describe anything the renderer does.  A
    plausible-looking translation would quietly convert to a different picture
    than the one that was asked for, which is worse than stopping.
    """
    for old, new, why in (
        ("disparity", "--eyes MM",
         "separation is worked out from the eye distance and the scene, not set as a percentage"),
        ("convergence", "--focus METRES",
         "the screen plane is a real distance now, not a position in a normalised range"),
    ):
        if getattr(args, old) is not None:
            return f"--{old} is gone: use {new} instead, because {why}."
    return None


def main(argv=None):
    args = build_parser().parse_args(argv)
    retirement = retired(args)
    if retirement:
        print(retirement, file=sys.stderr)
        return 2
    if args.gui:
        from .gui import main as gui_main

        return gui_main()

    if not args.inputs:
        build_parser().print_help()
        return 1
    photos = collect(args.inputs)
    if not photos:
        print("nothing to convert", file=sys.stderr)
        return 1
    if args.output and len(photos) > 1 and Path(args.output).suffix:
        print("with several inputs, --output must be a folder", file=sys.stderr)
        return 2

    # One converter for the batch either way, so the depth model is loaded once;
    # only the settings on it change as the run moves between stills and clips.
    converter = Converter(settings_for(args, video=False))
    for_photos, for_videos = converter.settings, settings_for(args, video=True)
    failures = skipped = 0
    for index, item in enumerate(photos, 1):
        prefix = f"[{index}/{len(photos)}] " if len(photos) > 1 else ""
        moving = is_video(item)
        converter.settings = for_videos if moving else for_photos
        try:
            if moving:
                info = convert_video(item, args.output, converter, reporter(item.name, prefix))
                print("\r\033[K" if sys.stderr.isatty() else "", end="", file=sys.stderr)
            else:
                info = converter.convert(item, args.output)
        except KeyboardInterrupt:
            print(f"\n{prefix}{item.name}: stopped", file=sys.stderr)
            return 130
        except Exception as error:  # keep a batch going when one file is broken
            failures += 1
            print(f"{prefix}{item.name}: {error}", file=sys.stderr)
            continue
        if info is None:  # too big, and the answer was to skip it
            skipped += 1
            print(f"{prefix}{item.name}: skipped", file=sys.stderr)
            continue
        width, height = info["output_size"]
        note = ""
        if info["resized_from"]:
            was, now = info["resized_from"], info["source_size"]
            note = f"  (resized from {was[0]}x{was[1]} to {now[0]}x{now[1]})"
        # A photo is quick enough to be worth a tenth of a second; a clip is
        # measured in minutes, where that precision would be noise.
        if moving:
            note = f"  {info['frames']} frames{note}"
            taken = clock(info["seconds"])
        else:
            taken = f"{info['seconds']:.1f}s"
        # What the geometry came out as.  `auto` picks it per scene, so without
        # this there is nothing to adjust from when a result wants tuning.
        eyes, focus = info.get("eyes_mm"), info.get("focus_m")
        chose = f"  {eyes:.0f}mm@{focus:.1f}m" if eyes else ""
        # How much of a vr180 frame is photograph and how much is the dark it was
        # never pointed at.  The single most useful number about the result, and
        # the one nobody would think to ask for until they had put it on.
        if info.get("coverage") is not None:
            chose += f"  {info['coverage']:.0%} of a sphere"
            if info.get("marked") is False:
                chose += " (unlabelled)"
            # On a plane a wrong lens only rescales the scene and the focus
            # distance absorbs it.  On a sphere it decides where every pixel
            # lands in angle, so a guessed one means a picture at the wrong
            # apparent size -- which looks entirely fine, and is not.
            if info.get("lens") == "assumed":
                chose += " (28mm assumed)"
        print(f"{prefix}{info['output']}  {width}x{height}{chose}{note}  {taken}")
    if skipped:
        print(f"{skipped} file{'s' if skipped > 1 else ''} skipped as too large", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
