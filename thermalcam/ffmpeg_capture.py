"""Raw thermal frames through FFmpeg instead of OpenCV.

OpenCV's macOS backend (AVFoundation) always converts frames to BGR and has no
way to switch that off -- which destroys the radiometric half of the TC001
stream. FFmpeg can ask the OS for the camera's native YUYV 4:2:2 format and
hand the bytes over untouched (``-f rawvideo -pix_fmt yuyv422`` with the same
input format is a straight copy). So on macOS this is the primary path, and on
Windows / Linux it is the last resort when no OpenCV backend gives raw frames.

:class:`FFmpegRawCapture` mimics the small part of ``cv2.VideoCapture`` that
:class:`thermalcam.camera.ThermalCamera` uses (isOpened / read / set / get /
release), so the camera code treats it as one more backend.

If the OS delivers UYVY instead of YUYV, every 16-bit sample just has its two
bytes swapped; the byte-order detection in :mod:`thermalcam.decode` handles
that automatically.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from .decode import FRAME_HEIGHT, SENSOR_WIDTH

#: Pseudo backend id, distinct from every cv2.CAP_* constant.
CAP_FFMPEG_RAW = -1001

FRAME_BYTES = SENSOR_WIDTH * FRAME_HEIGHT * 2


def _no_window() -> dict:
    if sys.platform.startswith("win"):
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def input_args(index: int, platform: Optional[str] = None,
               fps: int = 25, pixel_format: str = "yuyv422") -> Optional[list[str]]:
    """FFmpeg input arguments that open camera `index` in its raw format."""
    platform = platform or sys.platform
    size = f"{SENSOR_WIDTH}x{FRAME_HEIGHT}"
    if platform.startswith("win"):
        # DirectShow addresses cameras by name; indices come from the same
        # enumeration the rest of the app uses.
        from .devices import list_video_devices
        devices = [d for d in list_video_devices() if d.index is not None]
        device = next((d for d in devices if d.index == index), None)
        if device is None:
            return None
        same = [d for d in devices if d.name == device.name]
        args = ["-f", "dshow", "-rtbufsize", "64M", "-video_size", size,
                "-pixel_format", pixel_format, "-framerate", str(fps)]
        if same.index(device):
            args += ["-video_device_number", str(same.index(device))]
        return args + ["-i", f"video={device.name}"]
    if platform == "darwin":
        return ["-f", "avfoundation", "-video_size", size, "-pixel_format", pixel_format,
                "-framerate", str(fps), "-i", f"{index}:none"]
    return ["-f", "v4l2", "-input_format", pixel_format, "-video_size", size,
            "-framerate", str(fps), "-i", f"/dev/video{index}"]


class FFmpegRawCapture:
    """A cv2.VideoCapture look-alike reading raw YUYV frames from FFmpeg."""

    def __init__(self, index: int, *, fps: int = 25, open_timeout: float = 8.0,
                 command: Optional[list[str]] = None) -> None:
        self.index = index
        self.proc: Optional[subprocess.Popen] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_no = 0
        self._cond = threading.Condition()
        self._closed = False
        self.error = ""
        self._stderr: deque = deque(maxlen=20)
        ffmpeg = shutil.which("ffmpeg")
        if command is None:
            if not ffmpeg:
                self.error = "ffmpeg not found"
                return
            args = None
            for pix in ("yuyv422", "uyvy422"):
                args = input_args(index, fps=fps, pixel_format=pix)
                if args is None:
                    break
                cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", *args,
                       "-f", "rawvideo", "-pix_fmt", pix, "-"]
                if self._start(cmd, open_timeout):
                    return
            if args is None:
                self.error = f"no device at index {index}"
        else:
            self._start(command, open_timeout)

    # --- process ------------------------------------------------------------
    def _start(self, cmd: list[str], open_timeout: float) -> bool:
        self._close_proc()
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         stdin=subprocess.DEVNULL, bufsize=0, **_no_window())
        except OSError as exc:
            self.error = str(exc)
            self.proc = None
            return False
        self._closed = False
        self._stderr.clear()
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        threading.Thread(target=self._drain_stderr, args=(self.proc,), daemon=True).start()
        with self._cond:
            self._cond.wait_for(lambda: self._frame is not None or self.proc is None
                                or self.proc.poll() is not None, timeout=open_timeout)
            ok = self._frame is not None
        if not ok:
            self._close_proc()
            time.sleep(0.1)
            self.error = self._stderr[-1] if self._stderr else "no frames from ffmpeg"
        return ok

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        """Keep FFmpeg's stderr flowing (a full pipe would stall it) and
        remember the last lines for error messages."""
        try:
            for line in iter(proc.stderr.readline, b""):
                text = line.decode(errors="replace").strip()
                if text:
                    self._stderr.append(text)
        except Exception:
            pass

    def _reader(self, proc: subprocess.Popen) -> None:
        stream = proc.stdout
        buf = bytearray()
        while True:
            try:
                chunk = stream.read(FRAME_BYTES - len(buf))
            except Exception:
                break
            if not chunk:
                break
            buf.extend(chunk)
            if len(buf) == FRAME_BYTES:
                frame = np.frombuffer(bytes(buf), np.uint8).reshape(FRAME_HEIGHT, SENSOR_WIDTH, 2)
                buf.clear()
                with self._cond:
                    self._frame = frame
                    self._frame_no += 1
                    self._cond.notify_all()
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _close_proc(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception:
                pass
            for stream in (proc.stdout, proc.stderr):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass

    # --- cv2.VideoCapture interface -------------------------------------------
    def isOpened(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def read(self, timeout: float = 2.0):
        """Next frame (blocking up to `timeout`), as (ok, (384, 256, 2) uint8)."""
        with self._cond:
            last = self._frame_no
            self._cond.wait_for(lambda: self._frame_no != last or self._closed, timeout=timeout)
            if self._frame_no == last:
                return False, None
            return True, self._frame

    def grab(self) -> bool:
        ok, _ = self.read()
        return ok

    def set(self, prop, value) -> bool:      # the format is fixed on the command line
        return False

    def get(self, prop) -> float:
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(SENSOR_WIDTH)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(FRAME_HEIGHT)
        if prop == cv2.CAP_PROP_FPS:
            return 25.0
        if prop == cv2.CAP_PROP_CONVERT_RGB:
            return 0.0
        return 0.0

    def release(self) -> None:
        self._close_proc()
        time.sleep(0.05)
