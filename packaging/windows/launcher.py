"""Entry point for the packaged Windows build.

One folder holds two exes: `StereoCraft.exe` opens the window, `StereoCraft-cli.exe` is the
command line.  They are the same program -- the analysis PyInstaller does over
Torch is slow and would be identical for both -- so the name the exe was
launched under is what decides which half runs.
"""

import multiprocessing
import os
import sys


def _unscript():
    """Stop `@torch.jit.script` compiling, without turning TorchScript off.

    Depth Anything 3 puts the decorator on one small matrix helper -- a 4x4
    affine inverse, in `utils/geometry.py` -- and TorchScript compiles from
    *source*, which a frozen app does not have, having shipped .pyc instead.
    Running that function as ordinary Python costs nothing.

    This used to be `PYTORCH_JIT=0`, set before Torch was imported, and that was
    too blunt by half.  The flag does not just make the decorator a no-op: at
    import time it swaps `RecursiveScriptModule` for a stub with none of the
    machinery on it, and `torch.jit.load` -- which compiles nothing, and only
    reads a graph that was serialised years ago -- fails with `has no attribute
    '_construct'`.  That is how the surround's painted edge came to be silently
    missing from the frozen build while working perfectly from source.  So only
    the half that needs source is taken out, and loading is left alone.

    Patched rather than stubbed at the import, because the decorator is applied
    while the module is being imported and there is no later moment to catch it.
    """
    import torch.jit

    def unscripted(obj=None, *args, **kwargs):
        # Used bare as `@torch.jit.script` the function arrives here directly;
        # called with arguments it has to hand back a decorator instead.
        if obj is None or not callable(obj):
            return lambda function: function
        return obj

    torch.jit.script = unscripted


def main():
    # A frozen app re-executes itself to make a child process, so anything
    # spawning one has to be told it is already inside the app.
    multiprocessing.freeze_support()

    # Before anything else that can fail.  This is the build with no console and
    # no stderr -- see below -- so the log file is the only place a crash can
    # leave a mark, and it has to be open before there is anything to mark.
    from stereocraft import logbook
    logbook.start()

    _unscript()

    # The windowed exe has no console attached, which leaves stdout and stderr
    # as None.  The odd progress line written to either is worth losing, but
    # not worth an AttributeError taking the window down with it.
    null = None
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            null = null or open(os.devnull, "w")
            setattr(sys, name, null)

    if "cli" in os.path.basename(sys.argv[0]).lower():
        from stereocraft.cli import main as run
    else:
        from stereocraft.gui import main as run
    return run()


if __name__ == "__main__":
    sys.exit(main())
