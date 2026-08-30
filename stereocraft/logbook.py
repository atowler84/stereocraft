"""What the app was doing, written down, because the window cannot say it later.

The packaged app is a windowed exe.  It has no console, and `launcher` rebinds
`stdout` and `stderr` to the null device precisely so that a stray progress line
cannot take the window down -- which is right, and which also means that when a
conversion dies there is nothing at all to look at afterwards.  `pipeline.notice`
covers the case where the app has something to *say* to whoever is watching; it
does not cover the case where the app stops saying anything, and that is the case
worth having a file for.

**It is written for the post-mortem, not for the running commentary.**  The thing
that goes wrong on a long clip is that memory climbs until Windows refuses, and
the process is gone before any handler of ours runs -- there is no exception to
catch and no last words to write.  So the useful record is not the crash, which
will never be written, but the trail before it: how big the clip was, which pass
was running, and what the peak working set had reached by the end of each one.
A log that ends mid-pass with the peak already at fifteen gigabytes has said
everything the missing crash would have said.

Peak rather than current, because current is a sample and misses the spike that
actually did it.  Both come from the OS -- `VmHWM` on Linux, `PeakWorkingSetSize`
on Windows -- and neither needs a dependency the portable build would have to
carry.
"""

import functools
import logging
import logging.handlers
import os
import sys
import time
from contextlib import contextmanager

# Kept small: this is a file someone attaches to a bug report, not a telemetry
# stream.  Three runs of a long conversion fit comfortably.
MAX_BYTES = 2_000_000
KEEP = 3

log = logging.getLogger("stereocraft")
_started = False


def _writable(directory):
    """Whether a log file can actually be made here, found out by making one.

    Asked rather than assumed because the two candidate locations differ exactly
    in this: a portable folder the user unzipped is writable, and the same app
    dropped in Program Files is not.
    """
    try:
        os.makedirs(directory, exist_ok=True)
        probe = os.path.join(directory, ".writable")
        with open(probe, "w"):
            pass
        os.unlink(probe)
        return True
    except OSError:
        return False


def directory():
    """Where the log goes: beside the app if that is writable, else per-user.

    Beside the app first because that is where someone looking for it will look
    -- it is the folder they unzipped and the folder the exe is in -- and a
    portable build is the normal case.  The per-user fallback is for an install
    into a directory the user does not own, where the first choice would fail at
    the first write and take the app with it.
    """
    from .depth import _app_dir

    beside = os.path.join(_app_dir(), "logs")
    if _writable(beside):
        return beside
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        fallback = os.path.join(root, "StereoCraft", "logs")
    else:
        root = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
        fallback = os.path.join(root, "stereocraft")
    return fallback if _writable(fallback) else ""


def start(name="stereocraft"):
    """Open the log, and see to it that a crash reaches it.

    Idempotent, and never raises: a log that cannot be opened is a shame, not a
    reason for the app not to run.  Returns the path written to, or "" if
    nowhere would take it.
    """
    global _started
    if _started:
        return getattr(log, "path", "")

    _started = True
    log.setLevel(logging.DEBUG)
    log.propagate = False
    log.path = ""

    where = directory()
    if where:
        try:
            handler = logging.handlers.RotatingFileHandler(
                os.path.join(where, f"{name}.log"), maxBytes=MAX_BYTES,
                backupCount=KEEP, encoding="utf-8", delay=False)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"))
            log.addHandler(handler)
            log.path = handler.baseFilename
        except OSError:
            pass
    if not log.handlers:
        log.addHandler(logging.NullHandler())

    _catch_crashes()
    note("start", version=_version(), python=sys.version.split()[0],
         frozen=bool(getattr(sys, "frozen", False)), ram=_gb(total_memory()))
    return log.path


def _version():
    try:
        from . import __version__
        return __version__
    except Exception:
        return "?"


def _catch_crashes():
    """Send an uncaught exception to the log on its way out.

    This catches the tidy half of going wrong -- a Python exception nobody
    handled.  It cannot catch the other half: an out-of-memory kill by the OS,
    or a native allocation failure inside Torch, takes the process without
    unwinding anything, and the last thing in the log is then whatever `stage`
    wrote when the pass began.  That is the intended reading of a log that just
    stops.
    """
    previous = sys.excepthook

    def hook(kind, value, traceback):
        log.critical("uncaught %s: %s", kind.__name__, value,
                     exc_info=(kind, value, traceback))
        note("died", peak=_gb(memory()[1]))
        previous(kind, value, traceback)

    sys.excepthook = hook

    # A crash on a worker thread -- the GUI runs its conversions on one -- does
    # not go through `sys.excepthook` at all, and would otherwise be silent.
    if hasattr(sys, "excepthook") and hasattr(sys, "unraisablehook"):
        import threading

        def thread_hook(args):
            if args.exc_type is SystemExit:
                return
            log.critical("uncaught %s on %s: %s", args.exc_type.__name__,
                         getattr(args.thread, "name", "?"), args.exc_value,
                         exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
            note("died", peak=_gb(memory()[1]))

        threading.excepthook = thread_hook


def memory():
    """`(current, peak)` bytes of working set for this process, or `(0, 0)`.

    Peak is the number that matters here and neither platform makes us track it
    ourselves, which is the whole reason this is worth a function rather than a
    dependency.
    """
    if sys.platform == "win32":
        return _windows_memory()
    try:
        current = peak = None
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    current = int(line.split()[1]) * 1024
                elif line.startswith("VmHWM:"):
                    peak = int(line.split()[1]) * 1024
        return current, peak
    except (OSError, ValueError, IndexError):
        return None, None


@functools.lru_cache(maxsize=1)
def _psapi():
    """`GetProcessMemoryInfo`, its argument types declared, and our own handle.

    **Declaring the types is the entire reason this is a function.**  A process
    handle is a pointer, and ctypes assumes a C int for anything it has not been
    told about -- so on 64-bit Windows `GetCurrentProcess()` came back truncated
    to 32 bits and every call failed.  It failed the quiet way, too: the API
    reports failure by returning zero rather than by raising, so the log simply
    printed `peak=?` in precisely the situation it was written for.

    `K32GetProcessMemoryInfo` is preferred because it is in kernel32, which is
    loaded already; psapi carries the same call under its older name for
    anything that does not have it.
    """
    import ctypes
    import ctypes.wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.wintypes.DWORD),
                    ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    # The _EX tail.  `GetProcessMemoryInfo` fills it when `cb`
                    # says the struct is this long, and `PrivateUsage` is the
                    # commit charge -- which is the number Windows actually
                    # refuses on, and so the number an allocation failure has to
                    # be read against.
                    ("PrivateUsage", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentProcess.argtypes = []

    ask = None
    for library, name in ((kernel32, "K32GetProcessMemoryInfo"),
                          (ctypes.WinDLL("psapi"), "GetProcessMemoryInfo")):
        ask = getattr(library, name, None)
        if ask is not None:
            break
    if ask is None:
        raise OSError("no GetProcessMemoryInfo on this Windows")
    ask.restype = ctypes.wintypes.BOOL
    ask.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.wintypes.DWORD]
    return ctypes, Counters, ask, kernel32.GetCurrentProcess()


def _windows_memory():
    try:
        ctypes, Counters, ask, process = _psapi()
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not ask(process, ctypes.byref(counters), counters.cb):
            return None, None
        return counters.WorkingSetSize, counters.PeakWorkingSetSize
    except Exception:
        return None, None


def committed():
    """This process's commit charge, which is what an allocation is refused on.

    Resident set is what the process is *touching*; commit is what it has asked
    the system to promise it.  Windows hands out `Cannot allocate memory` when
    the system-wide commit charge reaches the limit, and a process can sit well
    below its commit in resident pages -- so a log that reports only RSS can
    show eight comfortable gigabytes on a machine that is about to refuse a
    hundred-megabyte request.  That is exactly what happened.
    """
    if sys.platform != "win32":
        try:
            with open("/proc/self/status") as handle:
                for line in handle:
                    if line.startswith("VmSize:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        return None
    try:
        ctypes, Counters, ask, process = _psapi()
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not ask(process, ctypes.byref(counters), counters.cb):
            return None
        return counters.PrivateUsage
    except Exception:
        return None


def process_memory(pid):
    """`(rss, commit)` for another process, or `(None, None)`.

    The encoder is a child process, and it is the one that actually ran out of
    memory -- so a log that measures only ourselves is blind to the half of the
    conversion that failed.  Asking costs a handle and two calls, once every two
    hundred frames.
    """
    if pid is None:
        return None, None
    if sys.platform != "win32":
        try:
            rss = commit = None
            with open(f"/proc/{int(pid)}/status") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
                    elif line.startswith("VmSize:"):
                        commit = int(line.split()[1]) * 1024
            return rss, commit
        except (OSError, ValueError, IndexError):
            return None, None
    handle = None
    try:
        ctypes, Counters, ask, _ = _psapi()
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        # QUERY_LIMITED_INFORMATION is the one a child of ours is always allowed;
        # the older pair is for anything that does not honour it.
        for access in (0x1000, 0x0400 | 0x0010):
            handle = kernel32.OpenProcess(access, False, int(pid))
            if handle:
                break
        if not handle:
            return None, None
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not ask(handle, ctypes.byref(counters), counters.cb):
            return None, None
        return counters.WorkingSetSize, counters.PrivateUsage
    except Exception:
        return None, None
    finally:
        if handle:
            try:
                ctypes.WinDLL("kernel32").CloseHandle(ctypes.c_void_p(handle))
            except Exception:
                pass


def headroom():
    """How much the whole machine has left to promise, not just this process.

    The one number that says whether the next big allocation -- ours or
    ffmpeg's, in its own process -- is going to be refused.
    """
    if sys.platform == "win32":
        status = _status()
        return None if status is None else int(status.ullAvailPageFile)
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _status():
    """`GlobalMemoryStatusEx`, freshly asked -- the numbers move."""
    try:
        import ctypes
        import ctypes.wintypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.wintypes.DWORD),
                        ("dwMemoryLoad", ctypes.wintypes.DWORD),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = Status()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status
    except Exception:
        return None


def total_memory():
    """How much RAM the machine has, so a peak can be read against something."""
    if sys.platform == "win32":
        status = _status()
        return None if status is None else int(status.ullTotalPhys)
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _gb(value):
    """Bytes as gigabytes, and `?` only for genuinely not knowing.

    Zero is a real answer -- an idle card has nothing on it -- and printing it
    as `?` made a working measurement look like a broken one."""
    return "?" if value is None else f"{value / 1e9:.2f}G"


def cuda_memory():
    """What Torch is holding on the card, without loading Torch to ask.

    Only reported if Torch is imported already -- which it will be by the time
    anything interesting happens, and importing it from here to find out would
    be a ten-second pause in the log's constructor.
    """
    torch = sys.modules.get("torch")
    if torch is None or not getattr(torch, "cuda", None):
        return None
    try:
        if not torch.cuda.is_available():
            return None
        # Reserved rather than only allocated, because reserved is the one that
        # costs.  On Windows the display driver keeps a system-memory backing
        # store for every byte of video memory a process reserves, so the
        # caching allocator's pool is charged to the host as well as the card --
        # see `video._release`.
        return (torch.cuda.memory_allocated(), torch.cuda.memory_reserved(),
                torch.cuda.max_memory_allocated())
    except Exception:
        return None


def note(event, **fields):
    """One line, `event key=value ...`, which is the form that greps.

    Deliberately flat.  The questions asked of this file are "how far did it
    get" and "how big had it grown", and both are answered by reading down a
    column.
    """
    if not _started:
        return
    parts = " ".join(f"{key}={value}" for key, value in fields.items())
    log.info("%s %s", event, parts) if parts else log.info("%s", event)


@contextmanager
def stage(name, **fields):
    """Bracket a pass, and record what it cost.

    The opening line is written *before* the pass runs and flushed with it, so a
    pass that never returns still leaves its name in the file -- which is the
    only evidence there will be of an out-of-memory kill.
    """
    note(f"{name} begin", **fields)
    started = time.perf_counter()
    try:
        yield
    except BaseException as problem:
        note(f"{name} failed", after=f"{time.perf_counter() - started:.1f}s",
             error=type(problem).__name__, peak=_gb(memory()[1]))
        raise
    else:
        current, peak = memory()
        card = cuda_memory()
        note(f"{name} done", took=f"{time.perf_counter() - started:.1f}s",
             **usage(), rss=_gb(current), peak=_gb(peak),
             **({"cuda": _gb(card[0]), "cuda_held": _gb(card[1]),
                 "cuda_peak": _gb(card[2])} if card else {}))


def usage():
    """The two numbers an allocation failure has to be read against, as fields.

    `commit` is what this process has asked the system to promise; `free` is what
    the machine has left to promise anybody, this process or the ffmpeg beside
    it.  Resident set says neither of those things, which is how a log came to
    show eight comfortable gigabytes at the moment ffmpeg was refused a hundred
    megabytes.
    """
    return {"commit": _gb(committed()), "free": _gb(headroom())}
