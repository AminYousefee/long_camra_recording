"""thermalcam -- a small library for TOPDON TC001 / TS001 thermal cameras.

Reads raw radiometric frames over plain UVC (no vendor driver, no vendor app)
and hands you a NumPy array of temperatures in degrees Celsius.

Quick start
-----------
>>> from thermalcam import ThermalCamera, TemperatureSpan
>>> with ThermalCamera() as cam:
...     frame = cam.read()
...     print(frame.stats())
...     print(frame.at(128, 96), "C at the centre pixel")

Manual span -- pin the display range so colours mean the same thing every frame:

>>> span = TemperatureSpan.manual(20.0, 40.0)
>>> image = colorize(frame, span, colormap="inferno", scale=3)

Format decoding is based on the reverse-engineering work of LeoDJ
(P2Pro-Viewer) and Les Wright (PyThermalCamera).
"""

from .calibration import Calibration, ReferencePoint
from .camera import ThermalCamera, ThermalCameraError, find_cameras
from .config import Config
from .decode import (FrameLayoutError, SENSOR_HEIGHT, SENSOR_WIDTH,
                     celsius_to_raw, raw_to_celsius)
from .devices import VideoDevice, find_by_name, list_video_devices
from .frame import Reading, ThermalFrame
from .span import TemperatureSpan

__version__ = "1.1.0"

__all__ = [
    "ThermalCamera",
    "ThermalCameraError",
    "ThermalFrame",
    "TemperatureSpan",
    "Calibration",
    "ReferencePoint",
    "Reading",
    "Config",
    "FrameLayoutError",
    "VideoDevice",
    "list_video_devices",
    "find_by_name",
    "find_cameras",
    "raw_to_celsius",
    "celsius_to_raw",
    "SENSOR_WIDTH",
    "SENSOR_HEIGHT",
]

_RENDER_EXPORTS = {"colorize", "colorize_celsius", "colorbar", "annotate",
                   "draw_marker", "COLORMAPS", "COLORMAP_NAMES"}


def __getattr__(name):
    """Expose rendering helpers lazily so headless installs stay importable."""
    if name in _RENDER_EXPORTS:
        from . import render
        return getattr(render, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
