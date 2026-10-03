"""Back-end services for Amin Studio: camera owners, thermal pipeline,
thermal video recorder, and the time-lapse schedule maths.

Nothing in here touches Tk. The GUI (amin_studio.py) talks to these objects
from the main thread; each service owns its device from one worker thread.
"""

from __future__ import annotations

import csv
import json
import logging
import logging.handlers
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

import cv2
import numpy as np

from platform_utils import imwrite, no_window_kwargs
from thermalcam import Calibration, TemperatureSpan, ThermalCamera
from thermalcam.camera import OpenCancelled
from thermalcam.decode import SENSOR_HEIGHT, SENSOR_WIDTH, celsius_to_raw
from thermalcam.render import COLORMAP_NAMES, colorbar, colorize_celsius, draw_marker, label

log = logging.getLogger("studio")

SECONDS_PER_DAY = 86400
STAMP_FMT = "%Y-%m-%d_%H-%M-%S"          # sortable, used in every file name


def setup_logging(log_dir: str) -> str:
    """(Re)point the 'studio' log at <log_dir>/studio.log, rotated at midnight.

    Returns the log file path. Safe to call repeatedly (e.g. when the save
    path changes): the previous file handler is replaced.
    """
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "studio.log")
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        if isinstance(h, logging.FileHandler):
            if os.path.abspath(h.baseFilename) == os.path.abspath(path):
                return path
            log.removeHandler(h)
            h.close()
    handler = logging.handlers.TimedRotatingFileHandler(
        path, when="midnight", backupCount=120, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S%z"))
    log.addHandler(handler)
    log.propagate = False
    return path


def ffmpeg_log_file(log_dir: Optional[str], name: str):
    """Append-mode file for an FFmpeg process's stderr (or DEVNULL)."""
    if not log_dir:
        return subprocess.DEVNULL
    try:
        os.makedirs(log_dir, exist_ok=True)
        f = open(os.path.join(log_dir, name), "ab")
        f.write(f"\n===== {datetime.now().astimezone().isoformat(timespec='seconds')} =====\n".encode())
        f.flush()
        return f
    except OSError:
        return subprocess.DEVNULL


def next_clock_boundary(now: datetime, period_s: int) -> datetime:
    """Next multiple of period_s counted from local midnight (e.g. :00 / :30)."""
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = (now - midnight).total_seconds()
    k = int(elapsed // period_s) + 1
    return midnight + timedelta(seconds=k * period_s)
MAX_SHOTS_PER_DAY = 24 * 60          # one per minute

# Colorbar geometry used by thermalcam.render.colorbar (bar + labels).
COLORBAR_WIDTH = 26 + 54


def device_index(path: str) -> Optional[int]:
    """'/dev/video3' -> 3.  Anything else -> None."""
    digits = "".join(ch for ch in os.path.basename(path or "") if ch.isdigit())
    return int(digits) if digits else None


# =============================================================================
# Thermal processing: correction + rendering, shared by preview/video/photo
# =============================================================================
class ThermalPipeline:
    """Holds every user-facing thermal setting and applies it to frames.

    correction modes
        off        factory conversion only
        manual     corrected = measured * gain + offset
        reference  fit from reference points (thermalcam.Calibration)
    """

    SPAN_MODES = ("auto", "manual", "percentile")
    CORRECTIONS = ("off", "manual", "reference")

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.span = TemperatureSpan.auto()
        self.colormap = "inferno"
        self.calib = Calibration()
        self.correction = "off"
        self.offset_c = 0.0
        self.gain = 1.0
        # Remember manual bounds / percentiles so cycling modes restores them.
        self.manual_bounds = (20.0, 40.0)
        self.percentiles = (1.0, 99.0)

    # --- correction ---------------------------------------------------------
    def observe(self, raw_celsius: np.ndarray) -> None:
        """Feed UNcorrected temperatures to the reference points."""
        with self.lock:
            if self.calib.points:
                self.calib.update(raw_celsius)

    def correct(self, raw_celsius: np.ndarray) -> np.ndarray:
        with self.lock:
            if self.correction == "manual":
                if self.gain == 1.0 and self.offset_c == 0.0:
                    return raw_celsius
                return (raw_celsius * np.float32(self.gain)
                        + np.float32(self.offset_c)).astype(np.float32)
            if self.correction == "reference":
                return self.calib.apply(raw_celsius)
            return raw_celsius

    def correct_scalar(self, value: float) -> float:
        with self.lock:
            if self.correction == "manual":
                return value * self.gain + self.offset_c
            if self.correction == "reference":
                return self.calib.apply_scalar(value)
            return value

    def coeffs(self) -> list[float]:
        """Correction as polynomial coefficients, highest order first:
        corrected = polyval(coeffs, sensor_celsius). [1, 0] = no correction."""
        with self.lock:
            if self.correction == "manual":
                return [float(self.gain), float(self.offset_c)]
            if self.correction == "reference" and self.calib.coeffs is not None:
                return [float(c) for c in self.calib.coeffs]
            return [1.0, 0.0]

    def snapshot(self) -> dict:
        """Everything that changes how data is corrected or displayed."""
        with self.lock:
            return {
                "correction": self.correction,
                "coeffs": [round(c, 8) for c in self.coeffs()],
                "span": self.span_text(),
                "colormap": self.colormap,
                "ref_points": [p.to_dict() for p in self.calib.points],
            }

    def describe_correction(self) -> str:
        with self.lock:
            if self.correction == "manual":
                return f"manual  x{self.gain:.4f} {self.offset_c:+.2f}C"
            if self.correction == "reference":
                return self.calib.describe()
            n = len(self.calib.points)
            extra = f"  ({n} ref pt{'s' if n != 1 else ''} placed)" if n else ""
            return "correction off" + extra

    # --- span -----------------------------------------------------------------
    def set_span_mode(self, mode: str, celsius: Optional[np.ndarray] = None) -> None:
        with self.lock:
            if mode == "manual":
                if self.span.mode == "manual":
                    return
                if celsius is not None and celsius.size > 1:
                    lo, hi = self.span.bounds(celsius)
                else:
                    lo, hi = self.manual_bounds
                self.span = TemperatureSpan.manual(round(lo, 1), round(hi, 1))
                self.manual_bounds = (self.span.vmin, self.span.vmax)
            elif mode == "percentile":
                self.span = TemperatureSpan.percentile(*self.percentiles)
            else:
                self.span = TemperatureSpan.auto()

    def set_manual_span(self, lo: float, hi: float) -> None:
        with self.lock:
            self.span = TemperatureSpan.manual(lo, hi)
            self.manual_bounds = (lo, hi)

    def set_percentiles(self, lo: float, hi: float) -> None:
        with self.lock:
            self.span = TemperatureSpan.percentile(lo, hi)
            self.percentiles = (lo, hi)

    def _ensure_manual(self, celsius: Optional[np.ndarray]) -> None:
        if self.span.mode != "manual":
            self.set_span_mode("manual", celsius)

    def shift_span(self, delta: float, celsius=None) -> None:
        with self.lock:
            self._ensure_manual(celsius)
            self.span.shift(delta)
            self.manual_bounds = (self.span.vmin, self.span.vmax)

    def widen_span(self, delta: float, celsius=None) -> None:
        with self.lock:
            self._ensure_manual(celsius)
            self.span.widen(delta)
            self.manual_bounds = (self.span.vmin, self.span.vmax)

    def bracket_points(self) -> bool:
        with self.lock:
            values = [p.known_c for p in self.calib.points if p.known_c is not None]
            if len(values) < 2:
                return False
            lo, hi = min(values), max(values)
            pad = max((hi - lo) * 0.25, 1.0)
            self.set_manual_span(lo - pad, hi + pad)
            return True

    def cycle_colormap(self, step: int = 1) -> str:
        with self.lock:
            i = COLORMAP_NAMES.index(self.colormap) if self.colormap in COLORMAP_NAMES else 0
            self.colormap = COLORMAP_NAMES[(i + step) % len(COLORMAP_NAMES)]
            return self.colormap

    def span_text(self) -> str:
        with self.lock:
            s = self.span
            if s.mode == "manual":
                return f"manual {s.vmin:.1f}..{s.vmax:.1f}C"
            if s.mode == "percentile":
                return f"percentile {s.lower_percentile:g}..{s.upper_percentile:g}%"
            return "auto"

    # --- rendering ------------------------------------------------------------
    def render(self, celsius: np.ndarray, scale: int = 3, *, bar: bool = True,
               markers: bool = False, selected=None, stamp: Optional[str] = None,
               cursor: Optional[tuple[int, int]] = None) -> np.ndarray:
        """BGR image of a corrected Celsius frame, optional scale bar/overlays."""
        with self.lock:
            span = TemperatureSpan.from_dict(self.span.to_dict())
            cmap = self.colormap
            points = list(self.calib.points) if markers else []
        image = colorize_celsius(celsius, span, cmap, scale)

        for i, point in enumerate(points, start=1):
            measured = point.measured_c
            shown = self.correct_scalar(measured) if measured is not None else None
            bits = [str(i)]
            if shown is not None:
                bits.append(f"{shown:.1f}C")
            if point.known_c is not None:
                bits.append(f"={point.known_c:.1f}")
            if not point.is_uniform:
                bits.append("!")
                colour = (80, 180, 255)
            elif point.known_c is not None:
                colour = (120, 255, 120)
            else:
                colour = (200, 200, 200)
            draw_marker(image, point.x, point.y, scale, colour, " ".join(bits),
                        selected=point is selected)

        if cursor is not None:
            cx, cy = cursor
            h, w = celsius.shape
            if 0 <= cx < w and 0 <= cy < h:
                label(image, f"{celsius[cy, cx]:.1f}C",
                      (min(cx * scale + 10, image.shape[1] - 60), max(14, cy * scale - 10)),
                      scale=0.42)

        if stamp:
            label(image, stamp, (6, image.shape[0] - 8), scale=0.45)

        if bar:
            ref_markers = [(p.known_c, str(i)) for i, p in enumerate(points, start=1)
                           if p.known_c is not None]
            cb = colorbar(span, height=image.shape[0], colormap=cmap,
                          celsius=celsius, markers=ref_markers)
            image = np.hstack([image, cb])
        return image

    # --- persistence ------------------------------------------------------------
    def to_dict(self) -> dict:
        with self.lock:
            return {
                "span": self.span.to_dict(),
                "colormap": self.colormap,
                "correction": self.correction,
                "offset_c": self.offset_c,
                "gain": self.gain,
                "manual_bounds": list(self.manual_bounds),
                "percentiles": list(self.percentiles),
            }

    def load_dict(self, data: dict) -> None:
        with self.lock:
            try:
                self.span = TemperatureSpan.from_dict(data.get("span", {}))
            except Exception:
                self.span = TemperatureSpan.auto()
            cm = data.get("colormap", "inferno")
            self.colormap = cm if cm in COLORMAP_NAMES else "inferno"
            corr = data.get("correction", "off")
            self.correction = corr if corr in self.CORRECTIONS else "off"
            self.offset_c = float(data.get("offset_c", 0.0))
            self.gain = float(data.get("gain", 1.0))
            mb = data.get("manual_bounds") or [20.0, 40.0]
            pc = data.get("percentiles") or [1.0, 99.0]
            self.manual_bounds = (float(mb[0]), float(mb[1]))
            self.percentiles = (float(pc[0]), float(pc[1]))


# =============================================================================
# Thermal camera owner
# =============================================================================
class ThermalService:
    """Owns the thermal camera from one worker thread.

    Consumers call acquire(name) / release(name); the camera runs while anybody
    holds it. The latest frame is kept in memory (it is small: 256x192) and
    consumers read it or wait for fresh ones. Unplugs are retried.

    acquire() and release() never block the caller (the GUI thread): stopping
    only signals the worker. A worker started afterwards first waits for its
    predecessor to let go of the device, and a stopped worker can no longer
    change the service state, so a slow open that finishes late cannot
    overwrite a newer one.
    """

    LOG_NAME = "thermal camera"
    #: consecutive failed reads tolerated before the camera is re-opened
    READ_FAILURES_BEFORE_RECONNECT = 8
    #: ...or after reads have been failing for this long
    READ_FAILURE_WINDOW_S = 2.0

    def __init__(self, pipeline: ThermalPipeline) -> None:
        self._status = None
        self.error = ""
        self.pipeline = pipeline
        self.cond = threading.Condition()
        self.users: set[str] = set()
        self.device: Optional[str] = None       # '/dev/videoN', 'N' (Windows) or None = auto scan
        self.avoid: set[int] = set()            # indices never probed by the auto scan
        self.thread: Optional[threading.Thread] = None
        self._retiring: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.status = "OFF"
        self.error = ""
        self.frame_no = 0
        self.latest_raw_c: Optional[np.ndarray] = None   # uncorrected °C
        self.latest_ts = 0.0
        self.fps = 0.0
        self.info: dict = {}
        #: index of the camera currently open (None when closed)
        self.found_index: Optional[int] = None
        #: (index, backend, strategy) that worked, per requested device -> fast re-open
        self._known: dict = {}
        self._known_swap: dict = {}

    # --- lifecycle --------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        if value != getattr(self, "_status", None):
            level = logging.WARNING if value in ("ERROR", "RECONNECTING") else logging.INFO
            if getattr(self, "_status", None) is not None or value != "OFF":
                log.log(level, "%s: %s -> %s%s", self.LOG_NAME, getattr(self, "_status", None),
                        value, f"  ({self.error})" if value in ("ERROR", "RECONNECTING") and self.error else "")
        self._status = value

    def _same_device(self, device: Optional[str]) -> bool:
        """Would `device` be served by the camera that is running now?"""
        if device == self.device or device is None:
            return True
        idx = device_index(device)
        return idx is not None and idx == self.found_index

    def acquire(self, user: str, device: Optional[str], avoid=()) -> None:
        """Start (or join) the thermal camera. Never blocks.

        avoid: camera indices the auto-detect scan must not touch (the
        ordinary camera, which may be open at the same time).
        """
        with self.cond:
            self.users.add(user)
            self.avoid = {i for i in avoid if i is not None}
            if self.running and not self._same_device(device):
                # Different device asked for: restart on the new one.
                self._stop_locked()
            if not self.running:
                self.device = device
                self._stop = threading.Event()
                self.error = ""
                self.status = "OPENING"
                self.thread = threading.Thread(target=self._worker,
                                               args=(self._stop, self._retiring),
                                               name="thermal-camera", daemon=True)
                self.thread.start()

    def release(self, user: str) -> None:
        with self.cond:
            self.users.discard(user)
            if not self.users:
                self._stop_locked()

    def shutdown(self, timeout: float = 4.0) -> None:
        """Stop and wait (briefly) for the device to be closed. For app exit."""
        with self.cond:
            self.users.clear()
            self._stop_locked()
            thread = self._retiring
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def wait_closed(self, timeout: float = 5.0) -> bool:
        thread = self._retiring
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            return not thread.is_alive()
        return True

    def _stop_locked(self) -> None:
        """Signal the worker to stop. Non-blocking; caller holds self.cond."""
        self._stop.set()
        self.cond.notify_all()
        if self.thread is not None:
            self._retiring = self.thread
        self.thread = None
        self.status = "OFF"
        self.latest_raw_c = None
        self.fps = 0.0
        self.found_index = None

    # --- worker -----------------------------------------------------------------------
    def _publish(self, stop: threading.Event, **state) -> bool:
        """Set service state, unless this worker has been stopped meanwhile."""
        with self.cond:
            if stop.is_set():
                return False
            status = state.pop("status", None)
            for key, value in state.items():
                setattr(self, key, value)
            if status is not None:
                self.status = status
            return True

    def _worker(self, stop: threading.Event, predecessor: Optional[threading.Thread]) -> None:
        # The previous worker may still be closing the device (or finishing a
        # slow open). Opening while it holds the device fails on Windows.
        if predecessor is not None:
            deadline = time.time() + 20.0
            while predecessor.is_alive() and not stop.is_set() and time.time() < deadline:
                predecessor.join(timeout=0.2)
        cam = None
        failures = 0
        first_failure = 0.0
        while not stop.is_set():
            if cam is None:
                key = self.device or "auto"
                try:
                    idx = device_index(self.device) if self.device else None
                    cam = ThermalCamera(index=idx, exclude=self.avoid,
                                        preferred=self._known.get(key),
                                        swap_bytes=self._known_swap.get(key),
                                        should_stop=stop.is_set).open()
                    self._known[key] = cam.resolved
                    self._known_swap[key] = cam.swap_bytes
                    if not self._publish(stop, info=cam.info(), error="",
                                         found_index=cam.resolved[0], status="ONLINE"):
                        break
                    failures = 0
                except OpenCancelled:
                    cam = None
                    break
                except Exception as exc:
                    cam = None
                    # a stale preference must not stick: next try does a full search
                    self._known.pop(key, None)
                    self._known_swap.pop(key, None)
                    err = str(exc).splitlines()[0] if str(exc) else repr(exc)
                    self._publish(stop, error=err, status="ERROR")
                    stop.wait(3.0)
                    continue

            try:
                frame = cam.read()
            except Exception as exc:
                # A single failed read is not a lost camera (MSMF in particular
                # drops the odd frame); only give up on a run of failures.
                now = time.time()
                if failures == 0:
                    first_failure = now
                failures += 1
                if (failures < self.READ_FAILURES_BEFORE_RECONNECT
                        and now - first_failure < self.READ_FAILURE_WINDOW_S):
                    stop.wait(0.05)
                    continue
                err = str(exc).splitlines()[0] if str(exc) else repr(exc)
                self._publish(stop, error=err, found_index=None, status="RECONNECTING")
                try:
                    cam.close()
                except Exception:
                    pass
                cam = None
                stop.wait(2.0)
                continue
            failures = 0

            raw_c = frame.celsius
            self.pipeline.observe(raw_c)
            now = time.time()
            with self.cond:
                if stop.is_set():
                    break
                if self.status != "ONLINE":
                    self.error = ""
                    self.status = "ONLINE"
                if self.latest_ts:
                    dt = now - self.latest_ts
                    if dt > 0:
                        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
                self.latest_raw_c = raw_c
                self.latest_ts = now
                self.frame_no += 1
                self.cond.notify_all()

        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass

    # --- consumers ----------------------------------------------------------------------
    def latest(self) -> tuple[Optional[np.ndarray], float, int]:
        with self.cond:
            return self.latest_raw_c, self.latest_ts, self.frame_no

    def collect(self, n: int = 1, timeout: float = 15.0) -> list[np.ndarray]:
        """Wait for n frames that arrive after this call. Uncorrected °C."""
        frames: list[np.ndarray] = []
        deadline = time.time() + timeout
        with self.cond:
            last = self.frame_no
            while len(frames) < n:
                remaining = deadline - time.time()
                if remaining <= 0 or not self.users:
                    break
                self.cond.wait(timeout=min(remaining, 0.5))
                if self.frame_no != last and self.latest_raw_c is not None:
                    frames.append(self.latest_raw_c)
                    last = self.frame_no
        return frames

    def status_text(self) -> str:
        if self.status == "ONLINE":
            where = self.device or f"index {self.info.get('index', '?')}"
            return f"{where}  •  ONLINE  {self.fps:4.1f} fps"
        if self.status in ("ERROR", "RECONNECTING"):
            return f"{self.status}  •  {self.error[:70]}"
        return self.status


# =============================================================================
# Normal (visible) camera owner, used for live preview and photos
# =============================================================================
class CameraIsThermalError(RuntimeError):
    """The device picked as the normal camera is the thermal camera."""

    def __init__(self, index: Optional[int]) -> None:
        super().__init__("this is the thermal camera (256 px wide) -- pick the other camera")
        self.index = index


class CameraService:
    """Owns a V4L2 camera via OpenCV from one worker thread.

    The worker only grab()s continuously (cheap: no JPEG decode) and decodes
    with retrieve() when someone wants a frame -- a photo, or live preview at
    a limited rate. That keeps a 4K MJPEG stream affordable while idling.
    """

    WARMUP_S = 1.5          # let auto-exposure settle after opening
    LOG_NAME = "camera"

    def __init__(self) -> None:
        self._status = None
        self.error = ""
        #: Called from the worker once per open, after warm-up, with the
        #: device path -- used to lock / apply exposure & white balance.
        self.on_ready: Optional[Callable[[str], None]] = None
        self.ready_at = 0.0
        self.cond = threading.Condition()
        self.users: set[str] = set()
        self.config: Optional[tuple[str, int, int]] = None
        self.thread: Optional[threading.Thread] = None
        self._retiring: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.status = "OFF"
        self.error = ""
        self.opened_at = 0.0
        self.latest: Optional[np.ndarray] = None
        self.latest_ts = 0.0
        #: set when the chosen device turned out to be the thermal camera
        self.thermal_index: Optional[int] = None
        self._decode_requests = 0
        self.preview_hz = 8.0
        self.actual_size = (0, 0)
        self.backend: Optional[int] = None
        self._jobs: list = []           # callables run on the camera thread (controls)

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        if value != getattr(self, "_status", None):
            level = logging.WARNING if value in ("ERROR", "RECONNECTING") else logging.INFO
            if getattr(self, "_status", None) is not None or value != "OFF":
                log.log(level, "%s: %s -> %s%s", self.LOG_NAME, getattr(self, "_status", None),
                        value, f"  ({self.error})" if value in ("ERROR", "RECONNECTING") and self.error else "")
        self._status = value

    def acquire(self, user: str, device: str, width: int, height: int) -> None:
        """Start (or join) the camera. Never blocks (see ThermalService)."""
        cfg = (device, int(width), int(height))
        with self.cond:
            self.users.add(user)
            if self.running and cfg != self.config:
                self._stop_locked()
            if not self.running:
                self.config = cfg
                self._stop = threading.Event()
                self.error = ""
                self.status = "OPENING"
                self.thread = threading.Thread(target=self._worker,
                                               args=(self._stop, self._retiring),
                                               name="camera", daemon=True)
                self.thread.start()

    def release(self, user: str) -> None:
        with self.cond:
            self.users.discard(user)
            if not self.users:
                self._stop_locked()

    def shutdown(self, timeout: float = 4.0) -> None:
        with self.cond:
            self.users.clear()
            self._stop_locked()
        self.wait_closed(timeout)

    def wait_closed(self, timeout: float = 5.0) -> bool:
        """Block until a stopped worker has released the device (e.g. before
        FFmpeg opens the same camera)."""
        thread = self._retiring
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            return not thread.is_alive()
        return True

    def _stop_locked(self) -> None:
        """Signal the worker to stop. Non-blocking; caller holds self.cond."""
        self._stop.set()
        self.cond.notify_all()
        if self.thread is not None:
            self._retiring = self.thread
        self.thread = None
        self.status = "OFF"
        self.latest = None

    def _open(self):
        device, w, h = self.config
        if sys.platform.startswith("win") or sys.platform == "darwin":
            index = device_index(device)
            if index is None:
                raise RuntimeError(f"invalid Windows camera index: {device}")
            backends = ([cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                        if sys.platform.startswith("win")
                        else [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY])
            cap = None
            for backend in backends:
                candidate = cv2.VideoCapture(index, backend)
                if candidate.isOpened():
                    cap = candidate
                    self.backend = backend
                    break
                candidate.release()
            if cap is None:
                raise RuntimeError(f"could not open {device}")
        else:
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            self.backend = cv2.CAP_V4L2
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(device)
                self.backend = cv2.CAP_ANY
        if not cap.isOpened():
            raise RuntimeError(f"could not open {device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        self.actual_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self.actual_size[0] == SENSOR_WIDTH:
            # Only the TC001 streams 256 px wide. Holding it here would lock the
            # thermal pipeline out of its own camera.
            cap.release()
            raise CameraIsThermalError(device_index(device))
        return cap

    def _publish(self, stop: threading.Event, **state) -> bool:
        with self.cond:
            if stop.is_set():
                return False
            status = state.pop("status", None)
            for key, value in state.items():
                setattr(self, key, value)
            if status is not None:
                self.status = status
            return True

    def _worker(self, stop: threading.Event, predecessor: Optional[threading.Thread] = None) -> None:
        if predecessor is not None:
            deadline = time.time() + 20.0
            while predecessor.is_alive() and not stop.is_set() and time.time() < deadline:
                predecessor.join(timeout=0.2)
        cap = None
        failures = 0
        last_preview = 0.0
        hook_done = False
        while not stop.is_set():
            if cap is None:
                try:
                    cap = self._open()
                    if stop.is_set():
                        break
                    hook_done = False
                    failures = 0
                    if not self._publish(stop, opened_at=time.time(), ready_at=float("inf"),
                                         error="", status="ONLINE"):
                        break
                except CameraIsThermalError as exc:
                    # Do not retry: every retry would snatch the device from
                    # the thermal service again.
                    self._publish(stop, error=str(exc), thermal_index=exc.index, status="ERROR")
                    break
                except Exception as exc:
                    cap = None
                    self._publish(stop, error=str(exc), status="ERROR")
                    stop.wait(3.0)
                    continue

            if not cap.grab():
                failures += 1
                if failures > 20:
                    self._publish(stop, error="grab() keeps failing", status="RECONNECTING")
                    cap.release()
                    cap = None
                    stop.wait(2.0)
                else:
                    time.sleep(0.05)
                continue
            failures = 0

            now = time.time()
            if not hook_done and now - self.opened_at >= self.WARMUP_S:
                hook_done = True
                if self.on_ready is not None:
                    try:
                        self.on_ready(self.config[0], cap, self.backend)
                        settle = 0.8        # new exposure takes a few frames
                    except Exception as exc:
                        log.warning("camera controls: %s", exc)
                        settle = 0.0
                else:
                    settle = 0.0
                self.ready_at = time.time() + settle
            with self.cond:
                jobs, self._jobs = self._jobs, []
            for job in jobs:
                job(cap)
            with self.cond:
                want = self._decode_requests > 0
                preview = "preview" in self.users and now - last_preview >= 1.0 / self.preview_hz
            if want or preview:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    last_preview = now
                    with self.cond:
                        if stop.is_set():
                            break
                        self.latest = frame
                        self.latest_ts = now
                        self.cond.notify_all()

        if cap is not None:
            cap.release()

    def snapshot(self, timeout: float = 10.0) -> Optional[np.ndarray]:
        """A freshly decoded frame taken after this call (and after warm-up)."""
        deadline = time.time() + timeout
        with self.cond:
            self._decode_requests += 1
            try:
                t_req = time.time()
                while True:
                    ready = (self.latest is not None
                             and self.latest_ts > max(t_req, self.ready_at))
                    if ready:
                        return self.latest
                    remaining = deadline - time.time()
                    if remaining <= 0 or not self.users:
                        return None
                    self.cond.wait(timeout=min(remaining, 0.25))
            finally:
                self._decode_requests -= 1

    def preview_frame(self) -> tuple[Optional[np.ndarray], float]:
        with self.cond:
            return self.latest, self.latest_ts

    def run_on_camera(self, fn: Callable, timeout: float = 5.0):
        """Run fn(cap) on the camera thread (OpenCV captures are not safe to
        use from two threads) and return its result. Raises if not running."""
        if not self.running or threading.current_thread() is self.thread:
            raise RuntimeError("camera is not open")
        done = threading.Event()
        box: dict = {}

        def job(cap):
            try:
                box["value"] = fn(cap)
            except Exception as exc:
                box["error"] = exc
            finally:
                done.set()
        with self.cond:
            self._jobs.append(job)
        if not done.wait(timeout):
            raise RuntimeError("camera did not respond")
        if "error" in box:
            raise box["error"]
        return box.get("value")


# =============================================================================
# Thermal video recorder (FFmpeg fed from a pipe, segmented like the main one)
# =============================================================================
class ThermalVideoRecorder:
    """Writes the thermal stream to chunked files at a fixed frame rate.

    Chunks are cut by this class itself, on wall-clock boundaries counted
    from midnight (e.g. every :00 and :30 for 30-min chunks) -- the camera
    FFmpeg uses -segment_atclocktime with the same period, so camera and
    thermal chunks start together and carry matching names.

    Per chunk:
        <prefix>_<stamp>_thermal.mkv         false colour, H.264   (optional)
        <prefix>_<stamp>_thermal_raw.mkv     FFV1 gray16le, UNCORRECTED sensor
                                             data: °C = value/64 - 273.15
        <prefix>_<stamp>_thermal_frames.csv  wall-clock time of every frame
    Per recording:
        <prefix>_<stamp>_thermal_info.json   format description
        <prefix>_<stamp>_thermal_settings.csv  correction/span at start and at
                                             every change, timestamped
    """

    SCALE = 3

    def __init__(self, service: ThermalService, pipeline: ThermalPipeline, *,
                 out_dir: str, prefix: str, fps: float, chunk_sec: int,
                 outputs: tuple[str, ...], crf: str, preset: str, overlay: bool,
                 log_dir: Optional[str] = None) -> None:
        self.service = service
        self.pipeline = pipeline
        self.out_dir = out_dir
        self.prefix = prefix
        self.fps = max(0.1, float(fps))
        self.chunk_sec = max(10, int(chunk_sec))
        self.outputs = outputs
        self.crf = crf
        self.preset = preset
        self.overlay = overlay
        self.log_dir = log_dir
        self.procs: dict[str, subprocess.Popen] = {}
        self._closing: list[threading.Thread] = []
        self.thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.frames_written = 0
        self.chunk_frames = 0
        self.last_error = ""
        self.restarts = 0
        self.chunk_name = ""
        self._frames_csv = None
        self._frames_writer = None
        self._settings_file = None
        self._settings_writer = None
        self._last_settings = None
        self.color_size = (SENSOR_WIDTH * self.SCALE + COLORBAR_WIDTH,
                           SENSOR_HEIGHT * self.SCALE)

    # --- ffmpeg ---------------------------------------------------------------
    def _cmd(self, kind: str, base: str) -> list[str]:
        fps = f"{self.fps:g}"
        common = ["ffmpeg", "-y", "-nostats", "-loglevel", "warning"]
        if kind == "color":
            w, h = self.color_size
            return [*common, "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                    "-framerate", fps, "-i", "-",
                    "-c:v", "libx264", "-preset", self.preset, "-crf", self.crf,
                    "-pix_fmt", "yuv420p", "-g", str(max(1, int(self.fps * 2))),
                    "-f", "matroska", base + "_thermal.mkv"]
        return [*common, "-f", "rawvideo", "-pix_fmt", "gray16le",
                "-s", f"{SENSOR_WIDTH}x{SENSOR_HEIGHT}", "-framerate", fps, "-i", "-",
                "-c:v", "ffv1", "-level", "3", "-g", "1",
                "-f", "matroska", base + "_thermal_raw.mkv"]

    def _open_chunk(self, start: datetime) -> None:
        """Close the current chunk (in the background) and open a new one."""
        self._close_chunk()
        base = os.path.join(self.out_dir, f"{self.prefix}_{start.strftime(STAMP_FMT)}")
        # Never overwrite: an encoder restart within the same second gets a suffix.
        n = 1
        while any(os.path.exists(base + sfx) for sfx in
                  ("_thermal.mkv", "_thermal_raw.mkv", "_thermal_frames.csv")):
            n += 1
            base = os.path.join(self.out_dir, f"{self.prefix}_{start.strftime(STAMP_FMT)}_part{n}")
        self.chunk_name = os.path.basename(base)
        for kind in self.outputs:
            self.procs[kind] = subprocess.Popen(
                self._cmd(kind, base), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=ffmpeg_log_file(self.log_dir, f"ffmpeg_thermal_{kind}.log"),
                **no_window_kwargs())
        self._frames_csv = open(base + "_thermal_frames.csv", "w", newline="")
        self._frames_writer = csv.writer(self._frames_csv)
        self._frames_writer.writerow(["frame_in_chunk", "local_time", "unix_time",
                                      "sensor_frame_age_s", "valid"])
        self.chunk_frames = 0
        log.info("thermal chunk opened: %s  (%s)", self.chunk_name, "+".join(self.outputs))

    def _close_chunk(self) -> None:
        procs, self.procs = self.procs, {}
        if self._frames_csv is not None:
            try:
                self._frames_csv.close()
            except Exception:
                pass
            self._frames_csv = self._frames_writer = None

        def finish(ps=procs):
            for proc in ps.values():
                try:
                    if proc.stdin:
                        proc.stdin.close()
                except Exception:
                    pass
            for kind, proc in ps.items():
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    log.warning("thermal %s encoder had to be killed at chunk end", kind)
        if procs:
            t = threading.Thread(target=finish, daemon=True)
            t.start()
            self._closing.append(t)

    def start(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        start = datetime.now().replace(microsecond=0)
        rec_base = os.path.join(self.out_dir, f"{self.prefix}_{start.strftime(STAMP_FMT)}")
        self._write_info(rec_base, start)
        self._settings_file = open(rec_base + "_thermal_settings.csv", "w", newline="")
        self._settings_writer = csv.writer(self._settings_file)
        self._settings_writer.writerow(["local_time", "event", "correction", "coeffs_high_order_first",
                                        "span", "colormap", "ref_points"])
        self._stop.clear()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            self.thread = None
        self._close_chunk()
        for t in self._closing:
            t.join(timeout=35)
        self._closing.clear()
        if self._settings_file is not None:
            self._settings_file.close()
            self._settings_file = None
        log.info("thermal recorder stopped: %d frames written", self.frames_written)

    def _write_info(self, rec_base: str, start: datetime) -> None:
        info = {
            "created": start.astimezone().isoformat(timespec="seconds"),
            "fps": self.fps,
            "chunk_seconds": self.chunk_sec,
            "chunks_aligned_to": "local clock, multiples of chunk_seconds from midnight",
            "outputs": list(self.outputs),
            "raw_format": ("FFV1 gray16le 256x192, UNCORRECTED sensor data: "
                           "sensor_C = value / 64 - 273.15. Apply the correction in "
                           "*_thermal_settings.csv: corrected_C = polyval(coeffs, sensor_C). "
                           "value 0 = no signal (camera dropped out)."),
            "colour_video": "false colour of the CORRECTED temperature, with the span/palette in the settings log",
            "frames_csv": "one row per frame of each chunk; frame_in_chunk matches the frame index in both videos",
            "settings_at_start": self.pipeline.snapshot(),
            "calibration": self.pipeline.calib.to_dict(),
        }
        try:
            with open(rec_base + "_thermal_info.json", "w") as f:
                json.dump(info, f, indent=2)
        except Exception as exc:
            self.last_error = f"info file: {exc}"
            log.warning("thermal info file: %s", exc)

    def _log_settings(self, event: str) -> None:
        snap = self.pipeline.snapshot()
        if event == "changed" and snap == self._last_settings:
            return
        self._last_settings = snap
        if self._settings_writer is None:
            return
        self._settings_writer.writerow([
            datetime.now().astimezone().isoformat(timespec="milliseconds"), event,
            snap["correction"], " ".join(f"{c:.8g}" for c in snap["coeffs"]),
            snap["span"], snap["colormap"], json.dumps(snap["ref_points"])])
        self._settings_file.flush()
        if event == "changed":
            log.info("thermal settings changed during recording: %s, coeffs %s, span %s",
                     snap["correction"], snap["coeffs"], snap["span"])

    # --- worker -------------------------------------------------------------------
    def _blank(self, kind: str) -> bytes:
        if kind == "color":
            w, h = self.color_size
            img = np.zeros((h, w, 3), np.uint8)
            label(img, "THERMAL SIGNAL LOST", (20, h // 2), scale=1.0, color=(80, 80, 255), thickness=2)
            label(img, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (20, h // 2 + 36), scale=0.7)
            return img.tobytes()
        return np.zeros((SENSOR_HEIGHT, SENSOR_WIDTH), np.uint16).tobytes()

    def _frame_bytes(self, kind: str, raw_c: np.ndarray) -> bytes:
        if kind == "color":
            celsius = self.pipeline.correct(raw_c)
            stamp = None
            if self.overlay:
                stamp = (datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
                         f"   min {celsius.min():.1f}  max {celsius.max():.1f}  "
                         f"mean {celsius.mean():.1f}C")
            img = self.pipeline.render(celsius, self.SCALE, bar=True, stamp=stamp)
            w, h = self.color_size
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h))
            return np.ascontiguousarray(img).tobytes()
        raw = celsius_to_raw(np.clip(raw_c, -273.0, 750.0))
        return np.ascontiguousarray(raw.astype("<u2")).tobytes()

    def _worker(self) -> None:
        period = 1.0 / self.fps
        # Wait for the first real frame: opening the camera is not "signal lost".
        while not self._stop.is_set():
            raw_c, ts, _ = self.service.latest()
            if raw_c is not None and time.time() - ts < 2.0:
                break
            self._stop.wait(0.1)
        if self._stop.is_set():
            return

        now_dt = datetime.now().replace(microsecond=0)
        try:
            self._open_chunk(now_dt)
        except Exception as exc:
            self.last_error = f"cannot start encoder: {exc}"
            log.error("thermal recorder: %s", self.last_error)
            return
        self._log_settings("start")
        boundary = next_clock_boundary(datetime.now(), self.chunk_sec)
        next_t = time.time()
        last_settings_check = 0.0
        last_flush = time.time()

        while not self._stop.is_set():
            now = time.time()
            if now < next_t:
                self._stop.wait(min(next_t - now, 0.25))
                continue
            next_t += period
            if now - next_t > 5 * period:           # fell far behind: resync
                next_t = now + period

            if datetime.now() >= boundary:
                try:
                    self._open_chunk(boundary)
                except Exception as exc:
                    self.last_error = f"chunk open failed: {exc}"
                    log.error("thermal recorder: %s", self.last_error)
                boundary = next_clock_boundary(datetime.now(), self.chunk_sec)

            if now - last_settings_check > 1.0:
                last_settings_check = now
                self._log_settings("changed")

            raw_c, ts, _ = self.service.latest()
            fresh = raw_c is not None and (now - ts) < 2.0

            for kind in list(self.procs):
                proc = self.procs[kind]
                if proc.poll() is not None:
                    self.last_error = f"{kind} encoder exited (code {proc.returncode})"
                    log.error("thermal %s", self.last_error + "; starting a new chunk")
                    self.restarts += 1
                    try:
                        self._open_chunk(datetime.now().replace(microsecond=0))
                    except Exception as exc:
                        self.last_error = f"encoder restart failed: {exc}"
                        log.error("thermal recorder: %s", self.last_error)
                    break
                try:
                    data = self._frame_bytes(kind, raw_c) if fresh else self._blank(kind)
                    proc.stdin.write(data)
                except (BrokenPipeError, OSError, ValueError) as exc:
                    self.last_error = f"{kind} pipe: {exc}"
            if self._frames_writer is not None:
                self._frames_writer.writerow([
                    self.chunk_frames,
                    datetime.fromtimestamp(now).astimezone().isoformat(timespec="milliseconds"),
                    f"{now:.3f}", f"{now - ts:.3f}" if raw_c is not None else "", 1 if fresh else 0])
                if now - last_flush > 2.0:
                    self._frames_csv.flush()
                    last_flush = now
            self.chunk_frames += 1
            self.frames_written += 1

    def estimate_mbps(self) -> float:
        return estimate_thermal_mbps(self.outputs, self.fps)


def estimate_thermal_mbps(outputs, fps: float) -> float:
    total = 0.0
    if "color" in outputs:
        total += 0.06 * fps          # ~1.5 Mbit/s at 25 fps, 848x576 false colour
    if "raw" in outputs:
        # 256*192*2 bytes, FFV1 ~2x on a thermal scene
        total += (SENSOR_WIDTH * SENSOR_HEIGHT * 2 * 8 / 2.0) * fps / 1e6
    return total


# =============================================================================
# Time-lapse schedule
# =============================================================================
def format_hms(sec: int) -> str:
    sec = int(sec) % SECONDS_PER_DAY
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def parse_hms(text: str) -> int:
    """'7', '07:30', '07:30:15' -> seconds of day. Raises ValueError."""
    parts = [p for p in text.strip().replace(".", ":").split(":") if p != ""]
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"bad time {text!r}")
    nums = [int(p) for p in parts] + [0] * (3 - len(parts))
    h, m, s = nums
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
        raise ValueError(f"time out of range {text!r}")
    return h * 3600 + m * 60 + s


def step_schedule(shots_per_day: int, start_sec: int) -> list[int]:
    n = int(shots_per_day)
    if not 1 <= n <= MAX_SHOTS_PER_DAY:
        raise ValueError(f"frames per day must be 1..{MAX_SHOTS_PER_DAY}")
    step = SECONDS_PER_DAY / n
    return sorted({int(round(start_sec + i * step)) % SECONDS_PER_DAY for i in range(n)})


def parse_schedule_text(text: str) -> list[int]:
    tokens = [t for t in text.replace(",", "\n").replace(";", "\n").split() if t.strip()]
    times = sorted({parse_hms(t) for t in tokens})
    if not times:
        raise ValueError("no times given")
    if len(times) > MAX_SHOTS_PER_DAY:
        raise ValueError(f"at most {MAX_SHOTS_PER_DAY} times per day")
    return times


def next_trigger(after: datetime, times: list[int]) -> datetime:
    """First scheduled datetime strictly after `after` (seconds resolution)."""
    midnight = after.replace(hour=0, minute=0, second=0, microsecond=0)
    for day in (0, 1):
        base = midnight + timedelta(days=day)
        for t in times:
            candidate = base + timedelta(seconds=t)
            if candidate > after:
                return candidate
    return midnight + timedelta(days=1, seconds=times[0])


def describe_step(shots_per_day: int) -> str:
    step = SECONDS_PER_DAY / max(1, shots_per_day)
    h, rem = divmod(int(round(step)), 3600)
    m, s = divmod(rem, 60)
    return f"every {h}h {m:02d}m {s:02d}s" if h else f"every {m}m {s:02d}s"


@dataclass
class PhotoSettings:
    out_dir: str
    prefix: str
    day_folders: bool
    use_camera: bool
    use_thermal: bool
    image_ext: str            # "jpg" | "png"
    jpeg_quality: int
    thermal_image: bool
    thermal_bar: bool
    thermal_data: str         # "none" | "npy" | "csv" | "png16"
    thermal_average: int
    thermal_corrected_copy: bool = False


class PhotoCapturer:
    """Takes one synchronised camera + thermal shot and writes the files.

    Every file of one shot shares the SAME timestamp (the scheduled time, to the
    second), so sorting the folder by name keeps each pair side by side:

        frame_2026-09-28_12-30-00_camera.jpg
        frame_2026-09-28_12-30-00_thermal.png
        frame_2026-09-28_12-30-00_thermal_C.npy
    """

    def __init__(self, camera: CameraService, thermal: ThermalService,
                 pipeline: ThermalPipeline) -> None:
        self.camera = camera
        self.thermal = thermal
        self.pipeline = pipeline
        self.bytes_written = 0
        self.shots = 0

    def capture(self, when: datetime, s: PhotoSettings) -> dict:
        folder = s.out_dir
        if s.day_folders:
            folder = os.path.join(folder, when.strftime("%Y-%m-%d"))
        os.makedirs(folder, exist_ok=True)
        base = os.path.join(folder, f"{s.prefix}_{when.strftime(STAMP_FMT)}")
        result = {"time": when, "files": [], "errors": [], "camera": None,
                  "thermal": None, "thermal_stats": None}

        def grab_camera():
            frame = self.camera.snapshot(timeout=12.0)
            if frame is None:
                result["errors"].append("camera: no frame")
                return
            path = f"{base}_camera.{s.image_ext}"
            params = [cv2.IMWRITE_JPEG_QUALITY, int(s.jpeg_quality)] if s.image_ext == "jpg" else []
            if imwrite(path, frame, params):
                result["files"].append(path)
                result["camera"] = frame
            else:
                result["errors"].append(f"camera: could not write {path}")

        def grab_thermal():
            frames = self.thermal.collect(max(1, s.thermal_average), timeout=15.0)
            if not frames:
                result["errors"].append("thermal: no frame")
                return
            raw_c = frames[0] if len(frames) == 1 else np.mean(np.stack(frames), axis=0).astype(np.float32)
            celsius = self.pipeline.correct(raw_c)
            result["coeffs"] = self.pipeline.coeffs()
            result["thermal_stats"] = (float(celsius.min()), float(celsius.max()),
                                       float(celsius.mean()),
                                       float(celsius[SENSOR_HEIGHT // 2, SENSOR_WIDTH // 2]))
            stamp = when.strftime("%Y-%m-%d %H:%M:%S") if s.thermal_bar else None
            img = self.pipeline.render(celsius, 3, bar=s.thermal_bar, stamp=stamp)
            result["thermal"] = img
            if s.thermal_image:
                path = f"{base}_thermal.png"
                if imwrite(path, img):
                    result["files"].append(path)

            # Temperature data is stored UNCORRECTED (straight from the sensor),
            # so a bad calibration can always be redone later. The correction
            # used is in capture_log.csv. Optionally also save a corrected copy.
            def save(data, suffix):
                if s.thermal_data == "npy":
                    path = f"{base}_thermal_{suffix}.npy"
                    np.save(path, data.astype(np.float32))
                elif s.thermal_data == "csv":
                    path = f"{base}_thermal_{suffix}.csv"
                    np.savetxt(path, data, fmt="%.3f", delimiter=",")
                elif s.thermal_data == "png16":
                    path = f"{base}_thermal_{suffix}_K64.png"
                    if not imwrite(path, celsius_to_raw(np.clip(data, -273.0, 750.0))):
                        raise OSError(f"could not write {path}")
                else:
                    return
                result["files"].append(path)
            try:
                save(raw_c, "sensorC")
                if s.thermal_corrected_copy and self.pipeline.correction != "off":
                    save(celsius, "correctedC")
            except Exception as exc:
                result["errors"].append(f"thermal data: {exc}")

        workers = []
        if s.use_camera:
            workers.append(threading.Thread(target=grab_camera, daemon=True))
        if s.use_thermal:
            workers.append(threading.Thread(target=grab_thermal, daemon=True))
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=30)

        for path in result["files"]:
            try:
                self.bytes_written += os.path.getsize(path)
            except OSError:
                pass
        self.shots += 1
        self._log(s.out_dir, result)
        return result

    def _log(self, out_dir: str, result: dict) -> None:
        path = os.path.join(out_dir, "capture_log.csv")
        new = not os.path.exists(path)
        snap = self.pipeline.snapshot()
        try:
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["timestamp_local", "files", "thermal_min_C", "thermal_max_C",
                                "thermal_mean_C", "thermal_center_C", "correction",
                                "coeffs_high_order_first", "span", "errors"])
                st = result["thermal_stats"]
                w.writerow([
                    # includes the UTC offset, so the DST change-over hour is unambiguous
                    result["time"].astimezone().isoformat(timespec="seconds"),
                    ";".join(os.path.basename(p) for p in result["files"]),
                    *([f"{v:.2f}" for v in st] if st else ["", "", "", ""]),
                    snap["correction"] if st else "",
                    " ".join(f"{c:.8g}" for c in result.get("coeffs", [])) if st else "",
                    snap["span"] if st else "",
                    " | ".join(result["errors"])])
        except Exception as exc:
            log.warning("capture_log.csv: %s", exc)
        if result["errors"]:
            log.warning("shot %s: %s", result["time"].strftime(STAMP_FMT), " | ".join(result["errors"]))
        else:
            log.info("shot %s: %d files", result["time"].strftime(STAMP_FMT), len(result["files"]))

    @property
    def mean_shot_bytes(self) -> Optional[float]:
        return self.bytes_written / self.shots if self.shots and self.bytes_written else None


def estimate_shot_bytes(s: PhotoSettings, width: int, height: int) -> float:
    total = 0.0
    if s.use_camera:
        total += width * height * (0.35 if s.image_ext == "jpg" else 1.6)
    if s.use_thermal:
        if s.thermal_image:
            total += 160e3
        per = {"npy": 196.7e3, "csv": 420e3, "png16": 75e3}.get(s.thermal_data, 0.0)
        total += per * (2 if s.thermal_corrected_copy else 1)
    return total


class TimelapseScheduler:
    """Fires callback(datetime) at each scheduled time of day, every day."""

    def __init__(self, times: list[int], callback: Callable[[datetime], None]) -> None:
        self.times = times
        self.callback = callback
        self._stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.next_at: Optional[datetime] = None

    def start(self) -> None:
        self._stop.clear()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
            self.thread = None
        self.next_at = None

    def _worker(self) -> None:
        last = datetime.now().replace(microsecond=0)
        while not self._stop.is_set():
            nxt = next_trigger(last, self.times)
            self.next_at = nxt
            while not self._stop.is_set():
                remaining = (nxt - datetime.now()).total_seconds()
                if remaining <= 0:
                    break
                self._stop.wait(min(remaining, 0.25))
            if self._stop.is_set():
                break
            late = (datetime.now() - nxt).total_seconds()
            last = nxt
            if late > 30:          # e.g. machine was suspended: skip, don't burst
                last = datetime.now().replace(microsecond=0)
                continue
            threading.Thread(target=self.callback, args=(nxt,), daemon=True).start()


# =============================================================================
# Camera exposure / white balance / focus (V4L2 controls via v4l2-ctl)
# =============================================================================
#: Semantic name -> the control names different kernels / cameras use.
CTRL_NAMES = {
    "auto_exposure": ("auto_exposure", "exposure_auto"),          # menu: 1 = manual
    "exposure": ("exposure_time_absolute", "exposure_absolute"),
    "gain": ("gain",),
    "wb_auto": ("white_balance_automatic", "white_balance_temperature_auto"),
    "wb": ("white_balance_temperature",),
    "focus_auto": ("focus_automatic_continuous", "focus_auto"),
    "focus": ("focus_absolute",),
}
VALUE_CTRLS = ("exposure", "gain", "wb", "focus")
AUTO_FOR = {"exposure": "auto_exposure", "gain": "auto_exposure", "wb": "wb_auto", "focus": "focus_auto"}

_CTRL_RE = re.compile(r"^\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)$")


def _v4l2(args: list[str], timeout: float = 3.0) -> str:
    res = subprocess.run(["v4l2-ctl", *args], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, timeout=timeout, **no_window_kwargs())
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or f"v4l2-ctl {' '.join(args)} failed")
    return res.stdout


def list_controls(device: str) -> dict[str, dict]:
    """{semantic: {name, min, max, default, value}} for the controls present."""
    if not sys.platform.startswith("linux"):
        return {}
    raw: dict[str, dict] = {}
    for line in _v4l2(["-d", device, "--list-ctrls"]).splitlines():
        m = _CTRL_RE.match(line)
        if not m:
            continue
        fields = dict(kv.split("=", 1) for kv in m.group(3).split() if "=" in kv)
        raw[m.group(1)] = {k: int(v) for k, v in fields.items()
                           if k in ("min", "max", "default", "value") and re.fullmatch(r"-?\d+", v)}
        raw[m.group(1)]["name"] = m.group(1)
    found = {}
    for semantic, names in CTRL_NAMES.items():
        for n in names:
            if n in raw:
                found[semantic] = raw[n]
                break
    return found


def _set(device: str, pairs: dict[str, int]) -> None:
    if pairs:
        _v4l2(["-d", device, "--set-ctrl", ",".join(f"{k}={int(v)}" for k, v in pairs.items())])


def camera_controls_manual(device: str, values: dict[str, Optional[int]]) -> dict[str, int]:
    """Switch the given controls to manual and set them. None = leave on auto."""
    ctrls = list_controls(device)
    autos, sets = {}, {}
    for key, val in values.items():
        if val is None or key not in ctrls:
            continue
        auto = AUTO_FOR[key]
        if auto in ctrls:
            autos[ctrls[auto]["name"]] = 1 if auto == "auto_exposure" else 0
        c = ctrls[key]
        v = max(c.get("min", val), min(c.get("max", val), int(val)))
        sets[c["name"]] = v
    _set(device, autos)                 # auto off first, or the values are ignored
    for name, v in sets.items():        # one by one: an unsupported one must not block the rest
        try:
            _set(device, {name: v})
        except RuntimeError as exc:
            log.warning("camera control %s=%s: %s", name, v, exc)
    applied = {k: sets[ctrls[k]["name"]] for k in values if k in ctrls and ctrls[k]["name"] in sets}
    log.info("camera controls set to manual on %s: %s", device, applied)
    return applied


def camera_controls_read(device: str) -> dict[str, int]:
    ctrls = list_controls(device)
    return {k: ctrls[k]["value"] for k in VALUE_CTRLS if k in ctrls and "value" in ctrls[k]}


def camera_controls_lock(device: str) -> dict[str, int]:
    """Freeze whatever auto-exposure / AWB / AF settled on: read, then set manual."""
    current = camera_controls_read(device)
    return camera_controls_manual(device, current)


def camera_controls_auto(device: str) -> None:
    ctrls = list_controls(device)
    pairs = {ctrls[a]["name"]: ctrls[a].get("default", 3 if a == "auto_exposure" else 1)
             for a in ("auto_exposure", "wb_auto", "focus_auto") if a in ctrls}
    # A camera whose default is "manual" would stay manual; prefer the auto value.
    if "auto_exposure" in ctrls and pairs.get(ctrls["auto_exposure"]["name"]) == 1:
        pairs[ctrls["auto_exposure"]["name"]] = 3
    for name, v in pairs.items():
        try:
            _set(device, {name: v})
        except RuntimeError as exc:
            log.warning("camera control %s=%s: %s", name, v, exc)
    log.info("camera controls back to auto on %s", device)




def have_v4l2_ctl() -> bool:
    return sys.platform.startswith("linux") and shutil.which("v4l2-ctl") is not None


class CameraControls:
    """Exposure / gain / white balance / focus on one camera.

    read() -> {key: value}; manual(values) -> applied; lock() -> applied; auto().
    Keys: exposure, gain, wb, focus. Units are the platform's own (V4L2:
    exposure in 100 us; DirectShow: log2 seconds, e.g. -6 = 1/64 s), which is
    why READ fills the fields from the camera rather than expecting typed units.
    """

    def read(self) -> dict[str, int]:
        raise NotImplementedError

    def manual(self, values: dict[str, Optional[int]]) -> dict[str, int]:
        raise NotImplementedError

    def auto(self) -> None:
        raise NotImplementedError

    def lock(self) -> dict[str, int]:
        return self.manual(self.read())


class V4L2Controls(CameraControls):
    """Linux, through v4l2-ctl -- works even while FFmpeg streams the camera."""

    def __init__(self, device: str) -> None:
        self.device = device

    def read(self):
        return camera_controls_read(self.device)

    def manual(self, values):
        return camera_controls_manual(self.device, values)

    def auto(self):
        camera_controls_auto(self.device)


class OpenCVControls(CameraControls):
    """Windows (DirectShow / Media Foundation), macOS and Linux without
    v4l2-ctl, through OpenCV capture properties on an open camera.

    Must be used on the thread that owns `cap`. What a camera supports varies;
    unsupported controls are left out of read() and ignored by manual().
    """

    VALUE_PROPS = {"exposure": cv2.CAP_PROP_EXPOSURE, "gain": cv2.CAP_PROP_GAIN,
                   "wb": cv2.CAP_PROP_WB_TEMPERATURE, "focus": cv2.CAP_PROP_FOCUS}
    AUTO_PROPS = {"exposure": cv2.CAP_PROP_AUTO_EXPOSURE, "gain": cv2.CAP_PROP_AUTO_EXPOSURE,
                  "wb": cv2.CAP_PROP_AUTO_WB, "focus": cv2.CAP_PROP_AUTOFOCUS}

    def __init__(self, cap, backend: Optional[int] = None) -> None:
        self.cap = cap
        self.backend = backend
        # V4L2 takes its menu values (1 = manual, 3 = aperture priority);
        # DirectShow / MSMF / AVFoundation take a flag (0 = manual, 1 = auto).
        self.ae_manual, self.ae_auto = (1, 3) if backend == cv2.CAP_V4L2 else (0, 1)

    def _get(self, prop) -> Optional[float]:
        try:
            v = float(self.cap.get(prop))
        except Exception:
            return None
        return v if np.isfinite(v) and v != -1 else None

    def read(self):
        out = {}
        for key, prop in self.VALUE_PROPS.items():
            v = self._get(prop)
            if v is None or (key == "wb" and v <= 0):
                continue
            if self.backend == cv2.CAP_AVFOUNDATION:
                # AVFoundation reports 0 for controls it does not have; only
                # trust a value the camera also accepts back (no side effect:
                # it is the current value).
                try:
                    if not self.cap.set(prop, v):
                        continue
                except Exception:
                    continue
            out[key] = int(round(v))
        return out

    def manual(self, values):
        applied = {}
        for key, val in values.items():
            if val is None or key not in self.VALUE_PROPS:
                continue
            auto_prop = self.AUTO_PROPS[key]
            off = self.ae_manual if auto_prop == cv2.CAP_PROP_AUTO_EXPOSURE else 0
            try:
                self.cap.set(auto_prop, off)            # auto off first
                if self.cap.set(self.VALUE_PROPS[key], float(val)):
                    applied[key] = int(val)
            except Exception as exc:
                log.warning("camera control %s=%s: %s", key, val, exc)
        log.info("camera controls set to manual: %s", applied)
        return applied

    def auto(self):
        for prop, on in ((cv2.CAP_PROP_AUTO_EXPOSURE, self.ae_auto),
                         (cv2.CAP_PROP_AUTO_WB, 1), (cv2.CAP_PROP_AUTOFOCUS, 1)):
            try:
                self.cap.set(prop, on)
            except Exception:
                pass
        log.info("camera controls back to auto")


class ServiceControls(CameraControls):
    """OpenCV controls on the camera that CameraService currently has open."""

    def __init__(self, service: "CameraService") -> None:
        self.service = service

    def _call(self, name, *args):
        backend = self.service.backend
        return self.service.run_on_camera(
            lambda cap: getattr(OpenCVControls(cap, backend), name)(*args))

    def read(self):
        return self._call("read")

    def manual(self, values):
        return self._call("manual", values)

    def auto(self):
        return self._call("auto")

    def lock(self):
        return self._call("lock")


class TemporaryControls(CameraControls):
    """Opens the camera just to apply settings, then closes it again.

    Used on Windows / macOS when FFmpeg is about to record: FFmpeg opens the
    camera itself and cannot set these, but UVC cameras keep the values while
    they stay powered. `settle_s` lets auto exposure / AWB / AF settle first
    (for "Lock at start")."""

    def __init__(self, device: str, settle_s: float = 0.0) -> None:
        self.device = device
        self.settle_s = settle_s

    def _run(self, name, *args):
        index = device_index(self.device)
        if sys.platform.startswith("win"):
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        elif sys.platform == "darwin":
            backends = [cv2.CAP_AVFOUNDATION]
        else:
            backends = [cv2.CAP_V4L2]
        last = None
        for backend in backends:
            src = self.device if backend == cv2.CAP_V4L2 else index
            cap = cv2.VideoCapture(src, backend)
            try:
                if not cap.isOpened():
                    last = RuntimeError(f"could not open {self.device}")
                    continue
                t_end = time.time() + self.settle_s
                while time.time() < t_end:
                    cap.grab()
                return getattr(OpenCVControls(cap, backend), name)(*args)
            finally:
                cap.release()
                time.sleep(0.3)
        raise last or RuntimeError(f"could not open {self.device}")

    def read(self):
        return self._run("read")

    def manual(self, values):
        return self._run("manual", values)

    def auto(self):
        return self._run("auto")

    def lock(self):
        return self._run("lock")


# =============================================================================
# Keep the machine awake while capturing
# =============================================================================
class SleepInhibitor:
    """Keeps the machine awake during a capture.

    Linux: a systemd-logind block inhibitor for sleep / idle / lid switch.
    Windows: SetThreadExecutionState (system + away mode required). Must be
    called from a long-lived thread -- the GUI thread is.
    macOS: `caffeinate -ims`.
    """

    _ES_CONTINUOUS = 0x80000000
    _ES_SYSTEM_REQUIRED = 0x00000001
    _ES_AWAYMODE_REQUIRED = 0x00000040

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self._win_active = False

    @property
    def active(self) -> bool:
        return self._win_active or (self.proc is not None and self.proc.poll() is None)

    def acquire(self, why: str) -> bool:
        if self.active:
            return True
        if sys.platform.startswith("win"):
            return self._acquire_windows(why)
        if sys.platform == "darwin":
            cmd = ["caffeinate", "-ims"]
            if not shutil.which(cmd[0]):
                log.warning("caffeinate not found: the machine may sleep during capture")
                return False
        else:
            exe = shutil.which("systemd-inhibit")
            if not exe:
                log.warning("systemd-inhibit not found: the machine may suspend during capture")
                return False
            cmd = [exe, "--what=sleep:idle:handle-lid-switch", "--who=Amin Studio",
                   f"--why={why}", "--mode=block", "sleep", "infinity"]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.2)
            if self.proc.poll() is not None:
                log.warning("sleep inhibitor exited immediately (code %s)", self.proc.returncode)
                self.proc = None
                return False
            log.info("sleep / suspend blocked: %s", why)
            return True
        except Exception as exc:
            log.warning("could not block sleep: %s", exc)
            self.proc = None
            return False

    def _acquire_windows(self, why: str) -> bool:
        try:
            import ctypes
            flags = self._ES_CONTINUOUS | self._ES_SYSTEM_REQUIRED | self._ES_AWAYMODE_REQUIRED
            if not ctypes.windll.kernel32.SetThreadExecutionState(flags):
                # away mode is refused on some machines; system-required alone still works
                if not ctypes.windll.kernel32.SetThreadExecutionState(
                        self._ES_CONTINUOUS | self._ES_SYSTEM_REQUIRED):
                    log.warning("SetThreadExecutionState refused: the PC may sleep during capture")
                    return False
            self._win_active = True
            log.info("sleep blocked (Windows): %s", why)
            return True
        except Exception as exc:
            log.warning("could not block sleep: %s", exc)
            return False

    def release(self) -> None:
        if self._win_active:
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(self._ES_CONTINUOUS)
            except Exception:
                pass
            self._win_active = False
            log.info("sleep allowed again")
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
            log.info("sleep / suspend allowed again")
