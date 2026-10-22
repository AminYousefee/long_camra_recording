"""Capture interface for TOPDON TC001 / TS001 (InfiRay-based UVC thermal cameras).

Opening the camera means three decisions, and each one has burned us:

* **Which device.** On Windows the TC001 is usually just "UVC Camera" or "USB
  Camera". Asking each camera which *modes* it offers (a driver query, no
  stream opened) identifies it reliably: only the thermal camera offers
  256-px-wide frames or the temperature mode. Ordinary webcams are skipped
  without being opened -- probing a webcam in the thermal camera's format is
  what used to hang for minutes on Media Foundation.
* **Which mode / layout.** Two TC001 revisions exist with different frame
  layouts (see :mod:`thermalcam.layouts`). The camera's mode list says which
  one is attached; the best layout it offers is used.
* **Which capture path.** FFmpeg first on every OS: it opens an exact mode and
  passes the bytes through untouched. OpenCV (Media Foundation on Windows,
  V4L2 on Linux) is a fallback for machines without FFmpeg.

Whatever is chosen, the first frames must decode to a plausible thermal scene
before the camera is accepted -- including a check that the "temperatures"
are not the camera's 8-bit picture, which decodes to a believable ~240 C.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import time
from typing import Callable, Iterable, Iterator, Optional

# Must be set before cv2 is imported (here or by the application).
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # AV_LOG_QUIET
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from . import decode, layouts  # noqa: E402
from .capture_config import STRATEGIES, Strategy  # noqa: E402
from .decode import (FRAME_HEIGHT, SENSOR_HEIGHT, SENSOR_WIDTH,  # noqa: E402
                     FrameLayoutError, normalize_buffer, raw_to_celsius)
from .devices import list_video_devices  # noqa: E402
from .ffmpeg_capture import CAP_FFMPEG_RAW, FFmpegRawCapture  # noqa: E402
from .frame import ThermalFrame  # noqa: E402
from .layouts import CLASSIC_384, GRAY_392, Layout, LayoutError  # noqa: E402
from .modes import DeviceRef, device_refs, list_modes  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_FPS = 25

#: Kept for callers that still import it; validation now uses
#: :func:`decode.scene_rejection_reason` (an absolute veto, not a score).
MIN_PLAUSIBILITY = 2.0

#: Frames read before giving up on a capture path. The first reads after a
#: mode change are routinely stale or empty.
SETTLE_FRAMES = 10

#: Seconds a capture path gets to deliver a valid frame.
OPEN_FRAME_TIMEOUT_S = 4.0

#: Pause after releasing a device before opening it again. Windows frees the
#: handle asynchronously; opening straight away can fail on a present camera.
RELEASE_SETTLE_S = 0.4


class ThermalCameraError(RuntimeError):
    """Raised when the camera cannot be opened in a usable (raw) mode."""


class OpenCancelled(ThermalCameraError):
    """Raised by :meth:`ThermalCamera.open` when ``should_stop`` became true."""


def set_opencv_log_level(verbose: bool) -> None:
    """Silence OpenCV's videoio chatter unless asked for.

    Probing cameras makes OpenCV print ``[ WARN ]`` lines (``can't grab
    frame``, ``Camera index out of range``) that say nothing useful and bury
    the output that matters.
    """
    try:
        logging_utils = cv2.utils.logging
        logging_utils.setLogLevel(
            logging_utils.LOG_LEVEL_INFO if verbose
            else logging_utils.LOG_LEVEL_ERROR)
    except AttributeError:  # pragma: no cover - very old/new OpenCV
        try:
            cv2.setLogLevel(3 if verbose else 2)
        except Exception:
            pass


#: The FFmpeg backend has its format fixed on the command line.
FFMPEG_STRATEGY = Strategy("ffmpeg_raw", lambda cap, w, h, fps: None,
                           "raw bytes piped from FFmpeg")


def _backend_name(backend: int) -> str:
    if backend == CAP_FFMPEG_RAW:
        return "FFMPEG"
    for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_V4L2", "CAP_AVFOUNDATION", "CAP_ANY"):
        if getattr(cv2, name, None) == backend:
            return name
    return str(backend)


def _platform_backends() -> list[int]:
    """OpenCV backends worth trying for raw frames on this OS (fallback only).

    DirectShow is left out on Windows: it converts this camera's stream to BGR
    whatever CAP_PROP_CONVERT_RGB says. AVFoundation (macOS) always converts.
    """
    if sys.platform.startswith("win"):
        return [cv2.CAP_MSMF]
    if sys.platform == "darwin":
        return []
    return [cv2.CAP_V4L2]


def has_thermal_geometry(shape: tuple) -> bool:
    """Whether a frame's size is consistent with a TC001-family stream."""
    if len(shape) == 3 and shape[1] == SENSOR_WIDTH:
        extra_rows = shape[0] - FRAME_HEIGHT
        return 0 <= extra_rows <= 16 and extra_rows % 2 == 0
    if len(shape) == 2:
        size = int(np.prod(shape))
        return size in layouts.BY_BYTES or size * 2 in layouts.BY_BYTES
    return False


def describe_shape(shape: tuple) -> str:
    if len(shape) == 3:
        return f"{shape[1]}x{shape[0]} {shape[2]}-channel"
    if len(shape) == 2:
        return f"{shape[1]}x{shape[0]} flat"
    return str(shape)


class _Attempt:
    """One (device, path, layout) trial and why it did or did not work."""

    __slots__ = ("index", "backend", "strategy", "reason", "shape")

    def __init__(self, index, backend, strategy, reason, shape=None):
        self.index = index
        self.backend = _backend_name(backend) if isinstance(backend, int) else str(backend)
        self.strategy = strategy
        self.reason = reason
        self.shape = shape

    @property
    def thermal_geometry(self) -> bool:
        return bool(self.shape) and has_thermal_geometry(self.shape)

    def __str__(self) -> str:
        where = f"index {self.index} / {self.backend}"
        if self.strategy:
            where += f" / {self.strategy}"
        return f"{where}: {self.reason}"


def frame_rejection(dec: layouts.Decoded) -> Optional[str]:
    """Why a decoded frame cannot be real thermal data (None if it can)."""
    raw = dec.raw64
    if float(np.mean((raw >> 8) == 0x80)) > 0.98 and int(raw.max() - raw.min()) < 256:
        return "the 'temperatures' are the camera's 8-bit picture (all ~240 C)"
    return decode.scene_rejection_reason(raw_to_celsius(raw))


class ThermalCamera:
    """A TC001-class thermal camera delivering per-pixel temperatures.

    Parameters
    ----------
    index:
        Camera index (OpenCV numbering; N of /dev/videoN on Linux). ``None``
        scans for the thermal camera.
    offset_c, gain:
        Linear correction, ``corrected = measured * gain + offset_c``.
    swap_bytes:
        Byte order of the classic layout's samples; ``None`` auto-detects.
    exclude:
        Indices the scan must never touch (the ordinary camera in use).
    preferred:
        ``(index, backend, layout_name)`` that worked last time; tried first.
    should_stop:
        Callable checked between attempts; :meth:`open` then raises
        :class:`OpenCancelled`.
    """

    def __init__(self, index: Optional[int] = None, *,
                 name: Optional[str] = None,
                 backend: Optional[int] = None,
                 strategy: Optional[str] = None,
                 offset_c: float = 0.0,
                 gain: float = 1.0,
                 swap_bytes: Optional[bool] = None,
                 max_scan_index: int = 10,
                 verbose: bool = False,
                 exclude: Iterable[int] = (),
                 preferred: Optional[tuple] = None,
                 should_stop: Optional[Callable[[], bool]] = None) -> None:
        self.index = index
        self.name = name
        self.backend = backend
        self.strategy = strategy
        self.offset_c = offset_c
        self.gain = gain
        self.swap_bytes = swap_bytes
        self.max_scan_index = max_scan_index
        self.verbose = verbose
        self.exclude = {int(i) for i in exclude if i is not None}
        self.preferred = preferred
        self.should_stop = should_stop

        self._cap = None
        self._layout: Optional[Layout] = None
        self._resolved_index: Optional[int] = None
        self._resolved_backend: Optional[int] = None
        self._ref: Optional[DeviceRef] = None
        self.attempts: list[_Attempt] = []
        self.devices: list = []
        self.modes: dict = {}

    # --- Lifecycle -----------------------------------------------------------

    def open(self) -> "ThermalCamera":
        """Find the camera, pick its best mode and verify it delivers
        plausible temperatures. Raises ThermalCameraError with a diagnosis."""
        if self._cap is not None:
            return self
        set_opencv_log_level(self.verbose)
        try:
            self.devices = list_video_devices(include_pnp=False)
        except Exception:
            self.devices = []
        refs = {r.index: r for r in device_refs(self.devices)}

        index = self.index
        if index is None and self.name:
            low = self.name.lower()
            index = next((r.index for r in refs.values() if low in r.name.lower()), None)
            if index is None:
                raise ThermalCameraError(f"No video device matched name {self.name!r}.")

        if index is not None:
            candidates = [index]
        else:
            available = [r for r in refs.values() if r.index not in self.exclude]
            look = {d.index: d for d in self.devices if d.index is not None}
            thermal = [r.index for r in available if look[r.index].looks_thermal]
            webcam = [r.index for r in available
                      if r.index not in thermal and "webcam" in r.name.lower()]
            other = [r.index for r in available if r.index not in thermal + webcam]
            candidates = thermal + other + webcam
            if not candidates and not refs:
                candidates = [i for i in range(self.max_scan_index) if i not in self.exclude]

        self.attempts = []
        plans = []
        for idx in candidates:
            self._check_stop()
            ref = refs.get(idx)
            if ref is None and sys.platform.startswith("linux"):
                ref = DeviceRef(index=idx, name=f"/dev/video{idx}", path=f"/dev/video{idx}")
            modes = list_modes(ref) if ref is not None else []
            self.modes[idx] = modes
            sizes = [m.size for m in modes]
            if modes and not layouts.is_thermal_mode_list(sizes):
                self.attempts.append(_Attempt(idx, "-", None,
                                              "not a thermal camera (offers "
                                              + ", ".join(sorted({m.size for m in modes})[:4]) + ")"))
                continue
            offered = layouts.layout_for_modes(sizes) if modes else [CLASSIC_384, GRAY_392]
            if modes and not offered:
                offered = [CLASSIC_384]       # 256 wide, unknown variant: try the classic one
            for lay in offered:
                if ref is not None and shutil_ffmpeg():
                    plans.append((idx, ref, CAP_FFMPEG_RAW, lay))
            for lay in offered:
                for be in _platform_backends():
                    if self.backend is None or self.backend == be:
                        plans.append((idx, ref, be, lay))
        if self.backend is not None:
            plans = [p for p in plans if p[2] == self.backend] or plans

        if self.preferred is not None:
            p_idx, p_be, p_lay = self.preferred
            pref = [p for p in plans if (p[0], p[2], p[3].name) == (p_idx, p_be, p_lay)]
            plans = pref + [p for p in plans if p not in pref]

        for idx, ref, be, lay in plans:
            self._check_stop()
            if be == CAP_FFMPEG_RAW:
                ok = self._try_ffmpeg(idx, ref, lay)
            else:
                ok = self._try_opencv(idx, ref, be, lay)
            if ok:
                self._resolved_index, self._resolved_backend = idx, be
                self._layout, self._ref = lay, ref
                log.info("Opened thermal camera: index %s, %s, layout %s", idx,
                         _backend_name(be), lay.name)
                return self
        self._check_stop()
        raise ThermalCameraError(self._diagnosis())

    def _check_stop(self) -> None:
        if self.should_stop is not None and self.should_stop():
            raise OpenCancelled("thermal camera open cancelled")

    @property
    def resolved(self) -> Optional[tuple]:
        """``(index, backend, layout_name)`` actually in use, for ``preferred=``."""
        if self._cap is None:
            return None
        return (self._resolved_index, self._resolved_backend, self._layout.name)

    @property
    def layout(self) -> Optional[Layout]:
        return self._layout

    # --- capture paths -----------------------------------------------------------

    def _accept_first_frames(self, cap, idx, backend, lay: Layout, convert,
                             swap_hint: Optional[bool] = None) -> bool:
        """Read frames until one decodes and validates (or time runs out)."""
        reason, shape = "no frames arrived", None
        deadline = time.time() + OPEN_FRAME_TIMEOUT_S
        for _ in range(SETTLE_FRAMES):
            if self.should_stop is not None and self.should_stop():
                cap.release()
                raise OpenCancelled("thermal camera open cancelled")
            if time.time() > deadline:
                break
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                if time.time() - t0 > 3.0:
                    reason = "reads time out (device busy?)"
                    break
                continue
            shape = tuple(frame.shape)
            try:
                flat = convert(frame)
                swap = self.swap_bytes if self.swap_bytes is not None else swap_hint
                if swap is None and lay is CLASSIC_384:
                    swap = self._detect_swap(flat)
                dec = lay.decode(flat, swap_bytes=bool(swap))
            except (LayoutError, FrameLayoutError) as exc:
                reason = str(exc).split("\n")[0][:160]
                continue
            rejection = frame_rejection(dec)
            if rejection is not None:
                reason = f"not a thermal scene: {rejection}"
                continue
            self.swap_bytes = bool(swap)
            self._cap = cap
            return True
        cap.release()
        time.sleep(RELEASE_SETTLE_S)
        self.attempts.append(_Attempt(idx, backend, lay.name, reason, shape))
        return False

    @staticmethod
    def _detect_swap(flat: np.ndarray) -> bool:
        frame = flat[:CLASSIC_384.frame_bytes].reshape(384, 256, 2)
        top, bottom = frame[:192], frame[192:]
        thermal = top if layouts.looks_like_picture(bottom) and not layouts.looks_like_picture(top) else bottom
        return decode.detect_byte_order(thermal)

    def _try_ffmpeg(self, idx, ref: DeviceRef, lay: Layout) -> bool:
        formats = ["yuyv422", "uyvy422"] if sys.platform == "darwin" else ["yuyv422"]
        for pix in formats:
            cap = FFmpegRawCapture(ref, lay.width, lay.height, pixel_format=pix, fps=DEFAULT_FPS)
            if not cap.isOpened():
                self.attempts.append(_Attempt(idx, CAP_FFMPEG_RAW, lay.name,
                                              f"ffmpeg: {cap.error}"[:200]))
                time.sleep(RELEASE_SETTLE_S)
                continue
            # UYVY: every 16-bit sample arrives byte-swapped
            hint = True if (pix == "uyvy422" and lay is not CLASSIC_384) else None
            if self._accept_first_frames(cap, idx, CAP_FFMPEG_RAW, lay, lambda f: f, hint):
                return True
        return False

    def _try_opencv(self, idx, ref, backend, lay: Layout) -> bool:
        src = ref.path if (ref is not None and ref.path and backend == cv2.CAP_V4L2) else idx
        cap = cv2.VideoCapture(src, backend)
        if not cap.isOpened():
            cap.release()
            time.sleep(RELEASE_SETTLE_S)
            cap = cv2.VideoCapture(src, backend)
        if not cap.isOpened():
            cap.release()
            self.attempts.append(_Attempt(idx, backend, None, "would not open"))
            return False
        try:
            STRATEGIES[0].apply(cap, lay.width, lay.height, DEFAULT_FPS)
        except cv2.error as exc:
            cap.release()
            self.attempts.append(_Attempt(idx, backend, lay.name, f"property error: {exc}"))
            return False
        return self._accept_first_frames(cap, idx, backend, lay,
                                         lambda f: self._to_flat(f, lay))

    @staticmethod
    def _to_flat(frame: np.ndarray, lay: Layout) -> np.ndarray:
        """OpenCV frames come in several shapes; give the decoder raw bytes."""
        if frame.ndim == 3 and frame.shape[2] == 3:
            raise FrameLayoutError(f"converted frame {frame.shape}: the backend destroyed the raw data")
        flat = layouts.as_bytes(frame)
        if flat.size == lay.frame_bytes:
            return flat
        if lay is CLASSIC_384:
            return layouts.as_bytes(normalize_buffer(frame))
        raise FrameLayoutError(f"frame has {flat.size} bytes, {lay.name} needs {lay.frame_bytes}")

    def _diagnosis(self) -> str:
        lines = ["Could not open a TC001-class thermal camera.", ""]
        if self.devices:
            lines.append("Cameras the system reports:")
            for d in self.devices:
                modes = self.modes.get(d.index) or []
                tail = (" -- modes: " + ", ".join(sorted({m.size for m in modes})[:6])) if modes else ""
                lines.append(f"  {d}{tail}")
            lines.append("")
        if not shutil_ffmpeg():
            lines += ["FFmpeg is not installed (or not on PATH); it is the main capture path.",
                      ""]
        if self.attempts:
            lines.append("Tried:")
            lines.extend(f"  - {a}" for a in self.attempts[:20])
            lines.append("")
        if not any("not a thermal camera" not in a.reason for a in self.attempts):
            lines += ["No camera offered the thermal camera's modes (256-px-wide frames).",
                      "Check the cable (many USB-C cables are charge-only), plug it straight",
                      "into the computer, and close other programs that use cameras."]
        else:
            lines += ["The thermal camera was found but no capture path gave valid frames.",
                      "Close other programs that may hold it (TOPDON software, Camera app,",
                      "Teams, OBS), unplug and replug it, and try again."]
        return "\n".join(lines)

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
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
        ok, frame = self._cap.read()
        if not ok or frame is None:
            err = getattr(self._cap, "error", "")
            raise ThermalCameraError(
                "Lost the camera stream (unplugged, or another program took the device)"
                + (f": {err}" if err else "."))
        try:
            flat = frame if self._resolved_backend == CAP_FFMPEG_RAW else self._to_flat(frame, self._layout)
            dec = self._layout.decode(flat, swap_bytes=bool(self.swap_bytes))
        except (LayoutError, FrameLayoutError) as exc:
            raise ThermalCameraError(f"bad frame: {exc}") from exc
        return ThermalFrame(raw=dec.raw64, gray=dec.gray, offset_c=self.offset_c,
                            gain=self.gain, approximate=dec.approximate)

    def frames(self) -> Iterator[ThermalFrame]:
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
        lay = self._layout
        return {
            "open": True,
            "index": self._resolved_index,
            "name": self._ref.name if self._ref else "",
            "backend": _backend_name(self._resolved_backend),
            "strategy": lay.name,
            "layout": lay.name,
            "mode": lay.size,
            "approximate": lay.approximate,
            "temp_divisor": layouts.TEMP_MODE_DIVISOR if lay is not CLASSIC_384 else 64.0,
            "width": SENSOR_WIDTH,
            "height": SENSOR_HEIGHT,
            "fps": DEFAULT_FPS,
            "swap_bytes": self.swap_bytes,
            "offset_c": self.offset_c,
            "gain": self.gain,
            "platform": platform.platform(),
            "opencv": cv2.__version__,
        }


def shutil_ffmpeg() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


def find_cameras(max_index: int = 10, verbose: bool = False) -> list[dict]:
    """Report which cameras are usable thermal cameras.

    Returns one dict per camera, each with ``thermal: bool``.
    """
    set_opencv_log_level(verbose)
    results = []
    try:
        devices = list_video_devices(include_pnp=False)
    except Exception:
        devices = []
    indices = [d.index for d in devices if d.index is not None] or list(range(max_index))
    names = {d.index: d.name for d in devices}
    for index in indices:
        entry = {"index": index, "name": names.get(index, ""), "backend": "-",
                 "strategy": "-", "thermal": False, "note": ""}
        cam = ThermalCamera(index=index, verbose=verbose)
        try:
            cam.open()
            info = cam.info()
            entry.update(thermal=True, backend=info["backend"], strategy=info["layout"],
                         note=f"decodes OK ({info['mode']}"
                              + (", approximate" if info["approximate"] else "") + ")")
        except ThermalCameraError:
            real = [a for a in cam.attempts if a.reason != "would not open"]
            if not real:
                continue
            entry["note"] = real[0].reason[:70]
        finally:
            cam.close()
        results.append(entry)
    return results
