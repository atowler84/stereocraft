# -*- mode: python ; coding: utf-8 -*-
"""Portable Windows build: one folder, two exes, no installer.

Built by `build.ps1`, which stages a copy of the source next to this file and
drops the weights into the finished folder afterwards.
"""

import os

from PyInstaller.utils.hooks import (collect_data_files, collect_dynamic_libs,
                                     collect_submodules, copy_metadata)

ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir, os.pardir))
ICON = os.path.join(ROOT, "stereocraft", "stereocraft.ico")

hiddenimports = [
    "stereocraft.cli",
    "stereocraft.gui",
    "stereocraft.video",
    # Named here as well as imported, because `launcher` opens the log before it
    # imports either half of the app -- and this is the build whose only account
    # of a crash is that file.
    "stereocraft.logbook",
    # transformers reaches its model classes by name through the auto mappings,
    # so there is no import statement for the analysis to follow to them.
    "transformers.models.depth_anything.configuration_depth_anything",
    "transformers.models.depth_anything.modeling_depth_anything",
    "transformers.models.dinov2.configuration_dinov2",  # Depth-Anything's backbone
    "transformers.models.dinov2.modeling_dinov2",
] + collect_submodules("transformers.models.auto")

# Depth Anything 3 builds its network from a name in a config file rather than
# an import, so nothing here is reachable by following import statements.
#
# The DINOv2 backbone has to be asked for separately.  Its folder has no
# __init__.py, which makes it a namespace package, and collect_submodules walks
# straight past those -- so the encoder, the largest part of the model, goes
# quietly missing and only turns up when the frozen app tries to load weights.
hiddenimports += collect_submodules("depth_anything_3.model")
hiddenimports += collect_submodules("depth_anything_3.model.dinov2")

# transformers decides at import time whether Torch is there by asking the
# installed metadata for its version, so the .dist-info has to come along or
# the whole library concludes it has no backend.
datas = copy_metadata("torch", recursive=True) + copy_metadata("transformers", recursive=True)
# DA3 keeps the architecture of each checkpoint in a .yaml beside its code, and
# reads it at load time.  Without these the weights have nothing to load into.
datas += collect_data_files("depth_anything_3", includes=["**/*.yaml"])
# imageio reads its own version out of its installed metadata the moment it is
# imported, and DA3 imports it on the way to the model.  Without the .dist-info
# the import dies and takes the conversion with it.
datas += copy_metadata("imageio")
# The window sets its own title bar icon at runtime, so the file has to come
# along as data as well as being built into the exe.
datas += [(ICON, "stereocraft")]

# HEIC support is a compiled library that nothing imports by name.
binaries = collect_dynamic_libs("pillow_heif")
# OpenCV was a dependency long before anything imported it -- DA3 wants it and
# the app never called it -- and `plate` is the first module that actually does.
# Its extension module carries compiled libraries of its own that PyInstaller
# does not always find by following the import, and the failure only appears
# after the freeze: the source tree runs perfectly and the exe cannot register a
# single frame.
binaries += collect_dynamic_libs("cv2")
hiddenimports += ["cv2"]

# ffmpeg is deliberately not bundled here.  PyInstaller puts everything it
# collects under _internal, and `video._tool` looks beside the exe -- where a
# user would drop their own copy.  build.ps1 lays it there after the freeze, the
# same way it lays out the weights.

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # None of these are used, and each one that creeps in through an optional
    # import of transformers costs hundreds of megabytes.
    # torchvision and cv2 were excluded until Depth Anything 3 arrived; its input
    # pipeline reaches for both, so they now have to come along.  The rest are
    # still dead weight, and several are things DA3's own api module imports at
    # the top for exporting meshes and gaussian splats -- work this app never
    # asks it to do.  `depth.DepthEstimator._load_da3` puts a stub in front of
    # that import, so leaving them out costs nothing at runtime.
    # Everything here was checked by blocking the import and converting a photo,
    # rather than guessed at.  `evo` matters beyond size: it is GPL-3.0, the only
    # copyleft package DA3 would otherwise pull in, and stubbing the module that
    # imports it takes scipy out with it.
    excludes=[
        "torchaudio", "tensorflow", "jax", "flax", "keras",
        "evo", "scipy", "matplotlib", "IPython", "notebook", "pytest",
        "moviepy", "gsplat", "open3d", "pycolmap", "trimesh", "plyfile",
        "fastapi", "uvicorn", "gradio", "e3nn", "dash", "flask",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

gui = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="StereoCraft",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX on Torch's DLLs is slow to build and a good way to break them
    console=False,
    disable_windowed_traceback=False,
    icon=ICON,
)

cli = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="StereoCraft-cli",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=ICON,
)

coll = COLLECT(
    gui, cli,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="StereoCraft",
)
