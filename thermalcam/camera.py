"""Capture interface for TOPDON TC001 / TS001 (InfiRay-based UVC thermal cameras).

The hard part is not reading frames -- it is stopping OpenCV from helpfully
converting them. The camera streams YUY2, and the bottom half of that stream is
16-bit radiometric data masquerading as pixels. Any YUY2->BGR conversion
silently turns it into garbage, and you get a picture that looks fine with
temperatures that are nonsense.

``CAP_PROP_CONVERT_RGB`` is meant to switch that off, but backends honour it
inconsistently and OpenCV versions differ. So :meth:`ThermalCamera.open` does not
trust a backend that merely opens: it works through the configuration
strategies in :mod:`thermalcam.capture_config`, pulls real frames, and checks
that what came back decodes to a plausible thermal scene. Only then does it keep
the combination. If none qualify, it raises -- with the diagnosis -- rather than
returning a camera that lies.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import time
from typing import Iterator, Optional

# Both must be set before cv2 is imported.
# MSMF's hardware transform path can force format conversion in the driver,
# defeating CONVERT_RGB entirely.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
# FFmpeg logs through its own channel, not OpenCV's, so silencing OpenCV does
# not stop "[dshow @ ...] Could not enumerate video devices" during a scan.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # AV_LOG_QUIET
# Some videoio messages are emitted during backend initialisation, before any
# runtime setLogLevel() call can take effect. This catches those.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from . import decode  # noqa: E402
from .capture_config import STRATEGIES, Strategy, convert_rgb_readback  # noqa: E402
from .devices import find_by_name, list_video_devices  # noqa: E402
from .decode import (FRAME_HEIGHT, SENSOR_HEIGHT, SENSOR_WIDTH,  # noqa: E402
                     FrameLayoutError, detect_byte_order, normalize_buffer,
                     raw_from_thermal, raw_to_celsius, split_halves,
                     visual_to_gray)
from .frame import ThermalFrame  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_FPS = 25

#: Minimum :func:`decode.plausibility_score` for a device to be accepted as a
#: thermal camera. A correctly decoded scene lands near 2.5-3.0; a regular
#: webcam's bytes score well under 1.
MIN_PLAUSIBILITY = 2.0

#: Frames to discard before judging a configuration. The first reads after a
#: mode change are routinely stale or the previous size.
SETTLE_FRAMES = 8

#: Pause after releasing a device before re-opening it. Windows does not free
#: the handle synchronously, and hammering open/release makes later attempts
#: spuriously report "would not open".
RELEASE_SETTLE_S = 0.08


class ThermalCameraError(RuntimeError):
    """Raised when the camera cannot be opened in a usable (raw) mode."""


def set_opencv_log_level(verbose: bool) -> None:
    """Silence OpenCV's videoio chatter unless asked for.

    Probing camera indices makes OpenCV print ``[ WARN ]`` and ``[ERROR]`` lines
    for every index that is not a camera -- including alarming-looking
    ``obsensor_uvc_stream_channel.cpp ... Camera index out of range``. They are
    entirely normal for a scan and say nothing about the thermal camera, but
    they bury the output that matters.
    """
    try:
        logging_utils = cv2.utils.logging
        logging_utils.setLogLevel(
            logging_utils.LOG_LEVEL_INFO if verbose
            else logging_utils.LOG_LEVEL_SILENT)
    except AttributeError:  # pragma: no cover - very old/new OpenCV
        pass


def _backend_name(backend: int) -> str:
    for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_V4L2", "CAP_AVFOUNDATION", "CAP_ANY"):
        if getattr(cv2, name, None) == backend:
            return name
    return str(backend)


def _platform_backends() -> list[int]:
    """Backends to try, best-first, for the current OS.

    Windows indices come from DirectShow enumeration, so try DSHOW first to
    keep them stable. This thermal camera's DSHOW path converts the raw payload
    to BGR; MSMF is tried next and returns the untouched stream. Every
    combination is validated before it is accepted.
    """
    if sys.platform.startswith("win"):
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


def has_thermal_geometry(shape: tuple) -> bool:
    """Whether a frame's size is consistent with a 256x384 thermal stream.

    This is what separates "the thermal camera is here but something converted
    the frame" from "this index is an ordinary webcam". Both come back as
    3-channel BGR and are otherwise indistinguishable, but they need opposite
    fixes -- one is a driver problem, the other means look at another index.
    """
    expected_bytes = SENSOR_WIDTH * FRAME_HEIGHT * 2
    if len(shape) == 3 and shape[1] == SENSOR_WIDTH:
        extra_rows = shape[0] - FRAME_HEIGHT
        if 0 <= extra_rows <= 16 and extra_rows % 2 == 0:
            return True                               # raw, converted, or padded
    if len(shape) == 2:
        rows, cols = shape
        size = rows * cols
        if size in (expected_bytes, expected_bytes // 2):
            return True                               # flat, or uint16-packed
        row_bytes = SENSOR_WIDTH * 2
        if rows == 1 and size % row_bytes == 0:
            padded_rows = size // row_bytes
            extra_rows = padded_rows - FRAME_HEIGHT
            if 0 < extra_rows <= 16 and extra_rows % 2 == 0:
                return True                           # symmetric sensor padding
    return False


def describe_shape(shape: tuple) -> str:
    """Human-readable frame size, from the sensor's point of view."""
    if len(shape) == 3:
        return f"{shape[1]}x{shape[0]} {shape[2]}-channel"
    if len(shape) == 2:
        return f"{shape[1]}x{shape[0]} flat"
    return str(shape)


class _Attempt:
    """One (index, backend, strategy) trial and why it did or did not work."""

    __slots__ = ("index", "backend", "strategy", "reason", "convert_rgb",
                 "shape", "thermal_geometry")

    def __init__(self, index, backend, strategy, reason, convert_rgb=None,
                 shape=None):
        self.index = index
        self.backend = _backend_name(backend)
        self.strategy = strategy
        self.reason = reason
        self.convert_rgb = convert_rgb
        self.shape = shape
        self.thermal_geometry = bool(shape) and has_thermal_geometry(shape)

    def __str__(self) -> str:
        readback = ""
        if self.convert_rgb is not None:
            readback = f" [CONVERT_RGB reads back as {self.convert_rgb:g}]"
        where = f"index {self.index} / {self.backend}"
        if self.strategy:
            where += f" / {self.strategy}"
        return f"{where}: {self.reason}{readback}"


class ThermalCamera:
    """A TC001-class thermal camera opened in raw radiometric mode.

    Parameters
    ----------
    index:
        Camera index as OpenCV sees it. ``None`` scans for the first device
        that yields decodable thermal frames -- slower to start, but it means
        you do not have to hardcode an index that changes between machines.
    offset_c, gain:
        Linear correction applied to every decoded frame, as
        ``corrected = measured * gain + offset_c``.
    swap_bytes:
        ``None`` (default) auto-detects byte order from the first frame.
    verbose:
        Let OpenCV's own log messages through. Off by default.

    Example
    -------
    >>> with ThermalCamera() as cam:
    ...     frame = cam.read()
    ...     print(frame.at(128, 96))
    """

    def __init__(self, index: Optional[int] = None, *,
                 name: Optional[str] = None,
                 backend: Optional[int] = None,
                 strategy: Optional[str] = None,
                 offset_c: float = 0.0,
                 gain: float = 1.0,
                 swap_bytes: Optional[bool] = None,
                 max_scan_index: int = 10,
                 verbose: bool = False) -> None:
        self.index = index
        self.name = name
        self.backend = backend
        self.strategy = strategy
        self.offset_c = offset_c
        self.gain = gain
        self.swap_bytes = swap_bytes
        self.max_scan_index = max_scan_index
        self.verbose = verbose

        self._cap: Optional[cv2.VideoCapture] = None
        self._resolved_index: Optional[int] = None
        self._resolved_backend: Optional[int] = None
        self._resolved_strategy: Optional[str] = None
        self.attempts: list[_Attempt] = []
        self.devices: list = []

    # --- Lifecycle -----------------------------------------------------------

    def open(self) -> "ThermalCamera":
        """Open the device and verify it delivers raw thermal frames."""
        if self._cap is not None:
            return self

        set_opencv_log_level(self.verbose)

        # Ask the OS what is actually attached. Two reasons this matters:
        # picking a device by name is stable across replugs where an index is
        # not, and the cheap "does this index open" probe below produces false
        # negatives on some Windows setups -- a device the OS lists must never
        # be skipped on the strength of that probe alone.
        try:
            self.devices = list_video_devices()
        except Exception:                                  # never fatal
            self.devices = []
        index = self.index
        if index is None and self.name:
            index = find_by_name(self.name)
            if index is None:
                listing = "\n".join(f"  - {d}" for d in self.devices) or "  (none)"
                raise ThermalCameraError(
                    f"No video device matched name {self.name!r}.\n"
                    f"Devices the system reports:\n{listing}")

        if index is not None:
            indices = [index]
        else:
            available = [device for device in self.devices if device.index is not None]
            thermal = [device.index for device in available if device.looks_thermal]
            unknown = [device.index for device in available
                       if not device.looks_thermal and "webcam" not in device.name.lower()]
            webcams = [device.index for device in available
                       if not device.looks_thermal and "webcam" in device.name.lower()]
            indices = thermal + unknown + webcams or list(range(self.max_scan_index))
        backends = [self.backend] if self.backend is not None \
            else _platform_backends()
        strategies = [s for s in STRATEGIES
                      if self.strategy is None or s.name == self.strategy]
        if not strategies:
            raise ValueError(f"Unknown strategy {self.strategy!r}. "
                             f"Available: {[s.name for s in STRATEGIES]}")

        self.attempts = []
        for index in indices:
            for backend in backends:
                for strat in strategies:
                    cap, present = self._try_configuration(index, backend, strat)
                    if cap is not None:
                        self._cap = cap
                        self._resolved_index = index
                        self._resolved_backend = backend
                        self._resolved_strategy = strat.name
                        log.info("Opened thermal camera: index %d, %s, strategy %s",
                                 index, _backend_name(backend), strat.name)
                        return self
                    if not present:
                        # Nothing here at all -- no point trying four more
                        # property orderings against a device that will not open.
                        break

        raise ThermalCameraError(self._diagnosis())

    def _try_configuration(self, index: int, backend: int, strat: Strategy
                           ) -> tuple[Optional[cv2.VideoCapture], bool]:
        """Open, apply one strategy, and accept only if frames decode.

        Returns ``(capture_or_None, device_was_present)``. The second value lets
        the caller distinguish "this ordering did not work" from "there is no
        camera here", and stop early in the latter case.

        There is deliberately no separate existence check before this. Probing
        an index by opening and immediately closing it, then opening it again a
        moment later, is a good way to make a perfectly working camera look
        absent on Windows -- the handle is not released synchronously, so the
        second open can fail on a device that is genuinely there. Opening once
        and keeping the handle avoids that window entirely.

        The device is still re-opened between *strategies*, because backend
        property state is sticky and a failed attempt would poison the next one.
        """
        # Retry once. Windows does not free a capture handle synchronously, so
        # an open issued shortly after the same device was released can fail on
        # a camera that is perfectly present -- which then looks identical to
        # "not plugged in". One short retry removes that false negative.
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            time.sleep(RELEASE_SETTLE_S * 4)
            cap = cv2.VideoCapture(index, backend)

        if not cap.isOpened():
            cap.release()
            time.sleep(RELEASE_SETTLE_S)
            self.attempts.append(_Attempt(index, backend, None, "no device here"))
            return None, False

        try:
            # Nothing may touch the capture before or after this call: on DSHOW
            # any property set after CAP_PROP_CONVERT_RGB reverts it. Each
            # strategy owns its own full ordering, buffer size included.
            strat.apply(cap, SENSOR_WIDTH, FRAME_HEIGHT, DEFAULT_FPS)
        except cv2.error as exc:
            cap.release()
            self.attempts.append(
                _Attempt(index, backend, strat.name, f"property error: {exc}"))
            return None, True

        readback = convert_rgb_readback(cap)
        reason, seen_shape = "no frames arrived", None

        for _ in range(SETTLE_FRAMES):
            ok, raw_frame = cap.read()
            if not ok or raw_frame is None:
                continue
            seen_shape = tuple(raw_frame.shape)
            try:
                buffer = normalize_buffer(raw_frame)
            except FrameLayoutError as exc:
                reason = str(exc).split("\n")[0]
                continue

            _, thermal = split_halves(buffer)
            swap = self.swap_bytes
            if swap is None:
                swap = detect_byte_order(thermal)
            celsius = raw_to_celsius(raw_from_thermal(thermal, swap_bytes=swap))

            # An ordinary webcam's bytes also arrive as a decodable buffer, so
            # shape proves nothing -- the values have to survive a strict check
            # before this device is accepted. Handing back a camera that reports
            # confident nonsense is worse than failing to open one.
            rejection = decode.scene_rejection_reason(celsius)
            if rejection is not None:
                reason = f"not a thermal scene: {rejection}"
                continue

            self.swap_bytes = swap
            return cap, True

        cap.release()
        time.sleep(RELEASE_SETTLE_S)
        self.attempts.append(_Attempt(index, backend, strat.name, reason,
                                      readback, seen_shape))
        return None, True

    def _diagnosis(self) -> str:
        """Build an error message that says what to actually do next."""
        lines = ["Could not open a TC001-class thermal camera in raw mode.", ""]

        if self.devices:
            lines.append("Video devices the system reports:")
            lines.extend(f"  {d}" + ("   <-- looks like the thermal camera"
                                     if d.looks_thermal else "")
                         for d in self.devices)
            lines.append("")
        else:
            lines += [
                "Could not enumerate device names. On Windows, installing "
                "pygrabber makes this diagnosis far more precise:",
                "    pip install pygrabber",
                "",
            ]

        real = [a for a in self.attempts if a.reason != "no device here"]
        if not real:
            lines.append("No camera devices responded at any index.")
            lines.append("Check the cable and that the camera's blue LED is lit.")
            return "\n".join(lines)

        lines.append("Tried:")
        lines.extend(f"  - {a}" for a in real[:24])
        if len(real) > 24:
            lines.append(f"  ... and {len(real) - 24} more")
        lines.append("")

        converted = [a for a in real if "converted frame" in a.reason]
        thermal_sized = [a for a in converted if a.thermal_geometry]

        if thermal_sized:
            # A 256x384 stream that came back as BGR: the camera is definitely
            # there, and something in the capture path is converting it.
            lines += [
                "The thermal camera IS present and streaming at 256x384, but "
                "every capture path converted the frame to BGR, which destroys "
                "the radiometric data.",
                "",
                "In order of likelihood:",
                "  1. The TOPDON PC software is installed. Uninstall it completely "
                "-- its driver replaces the generic UVC one and blocks raw capture.",
                "  2. Another program has the camera open (Camera app, Teams, OBS, "
                "Zoom). Close it and retry.",
                f"  3. This OpenCV build converts on every backend. You are on "
                f"OpenCV {cv2.__version__}. Try:  "
                f"pip install \"opencv-python>=4.8,<5\"",
            ]
        elif converted:
            # Everything that responded was some other camera.
            sizes = sorted({describe_shape(a.shape) for a in converted if a.shape})
            lines += [
                "No thermal camera was found. The device(s) that responded are "
                "ordinary cameras -- most likely a built-in webcam.",
                f"  Saw: {', '.join(sizes)}   (the TC001 streams 256x384)",
                "",
                "So the thermal camera is not reaching Windows as a UVC device:",
                "  1. Check the cable. Many USB-C cables are charge-only. Use the "
                "one that came with the camera, straight into the machine, not "
                "through a hub or dock.",
                "  2. Open Device Manager and look under Cameras / Imaging devices "
                "while plugging it in. If nothing appears, it is a cable, port or "
                "power problem, not a software one.",
                "  3. If it appears with a warning triangle, the TOPDON driver is "
                "installed. Uninstall the TOPDON PC software completely.",
                "  4. Settings > Privacy & security > Camera: 'Let desktop apps "
                "access your camera' must be on.",
            ]
        else:
            lines += [
                "Devices responded but none produced decodable thermal frames.",
                "Run with --debug to see the raw frame shapes each backend returned.",
            ]
        return "\n".join(lines)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "ThermalCamera":
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # --- Capture -------------------------------------------------------------

    def read(self) -> ThermalFrame:
        """Grab and decode the next frame. Raises if the device drops out."""
        if self._cap is None:
            self.open()
        assert self._cap is not None

        ok, raw_frame = self._cap.read()
        if not ok or raw_frame is None:
            raise ThermalCameraError(
                "Lost the camera stream. It was probably unplugged, or another "
                "program took the device."
            )

        buffer = normalize_buffer(raw_frame)
        visual, thermal = split_halves(buffer)
        return ThermalFrame(
            raw=raw_from_thermal(thermal, swap_bytes=bool(self.swap_bytes)),
            gray=visual_to_gray(visual),
            offset_c=self.offset_c,
            gain=self.gain,
        )

    def frames(self) -> Iterator[ThermalFrame]:
        """Yield frames until the stream ends or you break out.

        >>> for frame in cam.frames():
        ...     if frame.hottest.celsius > 60:
        ...         break
        """
        while True:
            try:
                yield self.read()
            except ThermalCameraError:
                return

    # --- Introspection -------------------------------------------------------

    def info(self) -> dict:
        """What actually got negotiated -- useful when debugging a new machine."""
        if self._cap is None:
            return {"open": False}
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        return {
            "open": True,
            "index": self._resolved_index,
            "backend": _backend_name(self._resolved_backend or cv2.CAP_ANY),
            "strategy": self._resolved_strategy,
            "fourcc": "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)),
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self._cap.get(cv2.CAP_PROP_FPS),
            "swap_bytes": self.swap_bytes,
            "offset_c": self.offset_c,
            "gain": self.gain,
            "platform": platform.platform(),
            "opencv": cv2.__version__,
        }


def find_cameras(max_index: int = 10, verbose: bool = False) -> list[dict]:
    """Probe indices and report which ones are usable thermal cameras.

    Returns one dict per index that responded, each with ``thermal: bool``.
    Unlike a bare index scan this actually tries every backend and strategy, so
    a "not thermal" verdict here means the data really could not be recovered.
    """
    set_opencv_log_level(verbose)
    results = []

    # Names, where the OS will tell us. Index alone cannot distinguish a
    # built-in webcam from a thermal camera, and with several attached the
    # numbering shifts between reboots.
    try:
        named_devices = {d.index: d for d in list_video_devices()
                         if d.index is not None}
    except Exception:
        named_devices = {}
    named = {index: device.name for index, device in named_devices.items()}

    if named_devices:
        thermal_indices = [i for i, device in named_devices.items()
                           if device.looks_thermal]
        unknown_indices = [i for i, device in named_devices.items()
                           if not device.looks_thermal and "webcam" not in device.name.lower()]
        webcam_indices = [i for i, device in named_devices.items()
                          if not device.looks_thermal and "webcam" in device.name.lower()]
        indices = thermal_indices + unknown_indices + webcam_indices
    else:
        indices = list(range(max_index))

    for index in indices:
        entry = {"index": index, "name": named.get(index, ""),
                 "backend": "-", "strategy": "-",
                 "thermal": False, "note": "responded, but not a thermal camera"}

        # Let ThermalCamera try every backend and strategy for this index.
        cam = ThermalCamera(index=index, max_scan_index=max_index, verbose=verbose)
        try:
            cam.open()
            info = cam.info()
            entry.update(thermal=True, backend=info["backend"],
                         strategy=info["strategy"],
                         note=f"decodes OK (swap_bytes={cam.swap_bytes})")
        except ThermalCameraError:
            real = [a for a in cam.attempts if a.reason != "no device here"]
            if not real:
                continue                      # nothing at this index at all
            entry["backend"] = real[0].backend
            shapes = {describe_shape(a.shape) for a in real if a.shape}
            seen = f" [{', '.join(sorted(shapes))}]" if shapes else ""
            if any(a.thermal_geometry for a in real):
                # 256x384 but unusable: the thermal camera, being converted.
                entry["note"] = f"thermal camera, but converted to BGR{seen}"
            elif any("converted frame" in a.reason for a in real):
                # Some other camera entirely -- almost always the webcam.
                entry["note"] = f"ordinary camera, not thermal{seen}"
            else:
                entry["note"] = real[0].reason[:60] + seen
        finally:
            cam.close()

        results.append(entry)
    return results
