"""Raw thermal frames through FFmpeg.

This is the primary capture path on every OS. FFmpeg can ask the driver for an
exact mode (including the newer TC001's "4x12305" temperature mode) and hand
the bytes over untouched (``-f rawvideo`` with the input's own pixel format is
a straight copy), and it behaves the same on DirectShow, V4L2 and
AVFoundation. OpenCV, by comparison, converts the stream on DirectShow and
AVFoundation, and its Media Foundation reads can hang for 10 s per frame when
the camera was just used by another program.

:class:`FFmpegRawCapture` mimics the small part of ``cv2.VideoCapture`` that
:class:`thermalcam.camera.ThermalCamera` uses (isOpened / read / get /
release).
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

#: Pseudo backend id, distinct from every cv2.CAP_* constant.
CAP_FFMPEG_RAW = -1001


def _no_window() -> dict:
    if sys.platform.startswith("win"):
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


class FFmpegRawCapture:
    """A cv2.VideoCapture look-alike reading raw frames from an FFmpeg pipe.

    ref:      a :class:`thermalcam.modes.DeviceRef` (or None with `command`)
    width, height, pixel_format: the capture mode; one frame is width*height*2
              bytes for the 4:2:2 formats used here.
    command:  full FFmpeg command instead of building one (tests).
    """

    def __init__(self, ref=None, width: int = 256, height: int = 384, *,
                 pixel_format: str = "yuyv422", fps: int = 25,
                 open_timeout: float = 6.0, command: Optional[list[str]] = None) -> None:
        self.ref = ref
        self.width, self.height = int(width), int(height)
        self.pixel_format = pixel_format
        self.fps = fps
        self.frame_bytes = self.width * self.height * 2
        self.proc: Optional[subprocess.Popen] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_no = 0
        self._returned_no = 0
        self._cond = threading.Condition()
        self._closed = True
        self._threads: list[threading.Thread] = []
        self.error = ""
        self.stderr_tail: deque = deque(maxlen=30)
        if command is None:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                self.error = "ffmpeg not found on PATH"
                return
            if ref is None:
                self.error = "no device"
                return
            command = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error",
                       *ref.ffmpeg_input(size=f"{self.width}x{self.height}",
                                         fmt=pixel_format, fps=fps, rtbufsize="32M"),
                       "-f", "rawvideo", "-pix_fmt", pixel_format, "-"]
        self.command = command
        self._start(command, open_timeout)

    # --- process ------------------------------------------------------------
    def _start(self, cmd: list[str], open_timeout: float) -> bool:
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         stdin=subprocess.DEVNULL, bufsize=0, **_no_window())
        except OSError as exc:
            self.error = str(exc)
            self.proc = None
            return False
        self._closed = False
        for target in (self._reader, self._drain_stderr):
            t = threading.Thread(target=target, args=(self.proc,), daemon=True)
            t.start()
            self._threads.append(t)
        with self._cond:
            self._cond.wait_for(lambda: self._frame is not None or self._closed, timeout=open_timeout)
            ok = self._frame is not None
        if not ok:
            # Let the stderr thread catch FFmpeg's last words before reporting.
            time.sleep(0.2)
            self.error = self._error_text() or ("no frames from ffmpeg within "
                                                f"{open_timeout:g} s")
            self.release()
        return ok

    def _error_text(self) -> str:
        lines = [ln for ln in self.stderr_tail if ln]
        return " | ".join(lines[-3:])[:300]

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        """Keep FFmpeg's stderr flowing (a full pipe would stall it) and
        remember the last lines for error messages."""
        try:
            for line in iter(proc.stderr.readline, b""):
                text = line.decode(errors="replace").strip()
                if text:
                    self.stderr_tail.append(text)
        except Exception:
            pass

    def _reader(self, proc: subprocess.Popen) -> None:
        stream = proc.stdout
        size = self.frame_bytes
        buf = bytearray(size)
        view = memoryview(buf)
        filled = 0
        while True:
            try:
                n = stream.readinto(view[filled:])
            except Exception:
                break
            if not n:
                break
            filled += n
            if filled == size:
                frame = np.frombuffer(bytes(buf), np.uint8)
                filled = 0
                with self._cond:
                    self._frame = frame
                    self._frame_no += 1
                    self._cond.notify_all()
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # --- cv2.VideoCapture interface -------------------------------------------
    def isOpened(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and not self._closed

    def read(self, timeout: float = 2.0):
        """The newest frame not returned before, waiting up to `timeout`.

        Returns (ok, flat uint8 array of frame_bytes)."""
        with self._cond:
            self._cond.wait_for(lambda: self._frame_no != self._returned_no or self._closed,
                                timeout=timeout)
            if self._frame_no == self._returned_no:
                if self._closed and not self.error:
                    self.error = self._error_text() or "ffmpeg stopped"
                return False, None
            self._returned_no = self._frame_no
            return True, self._frame

    def grab(self) -> bool:
        ok, _ = self.read()
        return ok

    def set(self, prop, value) -> bool:      # the mode is fixed on the command line
        return False

    def get(self, prop) -> float:
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def release(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=3)
            except Exception:
                pass
            for stream in (proc.stdout, proc.stderr):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads = []
        with self._cond:
            self._closed = True
            self._cond.notify_all()
