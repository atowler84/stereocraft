"""StereoCraft - turn a single photo into a high-resolution side-by-side 3D image."""

__version__ = "2.4.1"

__all__ = ["Settings", "VideoSettings", "Converter", "convert", "convert_video"]

_HOMES = {"Settings": ".pipeline", "VideoSettings": ".pipeline", "Converter": ".pipeline",
          "convert": ".pipeline", "convert_video": ".video"}


def __getattr__(name):
    # Lazy so that `import stereocraft` stays cheap (torch takes a couple of seconds).
    if name in _HOMES:
        from importlib import import_module

        return getattr(import_module(_HOMES[name], __name__), name)
    raise AttributeError(name)
