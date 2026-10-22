"""Small cross-platform helpers shared by the app and its services.

Everything OS-specific that is not about cameras lives here, so the rest of the
code can stay free of ``sys.platform`` branches:

* subprocess flags (no console window flashing up on Windows),
* image files on paths OpenCV cannot open (non-ASCII paths on Windows),
* stopping FFmpeg so the file it writes is finalised on every OS,
* CPU / RAM usage without /proc,
* fonts that exist on each OS.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from typing import Optional

import cv2
import numpy as np

IS_WINDOWS = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


# --- subprocess --------------------------------------------------------------
def no_window_kwargs() -> dict:
    """Keyword arguments that stop a child process from opening a console
    window on Windows (FFmpeg, PowerShell, ...). Empty elsewhere.

    Without this, every FFmpeg chunk / device scan flashes a black window when
    the app runs under pythonw.exe (e.g. launched at login)."""
    if IS_WINDOWS:
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def stop_ffmpeg(proc: Optional[subprocess.Popen], timeout: float = 10.0) -> None:
    """Stop an FFmpeg process so it finalises its output file.

    SIGTERM does that on Linux/macOS, but on Windows terminate() is
    TerminateProcess -- an instant kill that leaves the last file without its
    index. Sending "q" on stdin is FFmpeg's own clean-exit command and works on
    every OS (the process must have been started with stdin=PIPE)."""
    if proc is None:
        return
    if proc.poll() is not None:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        return
    try:
        if proc.stdin is not None:
            proc.stdin.write(b"q")
            proc.stdin.flush()
            proc.stdin.close()
        proc.wait(timeout=timeout)
        return
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# --- image files ---------------------------------------------------------------
def imwrite(path: str, img: np.ndarray, params: Optional[list] = None) -> bool:
    """cv2.imwrite that also works for non-ASCII paths (Windows)."""
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, img, params or [])
        if not ok:
            return False
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except (OSError, cv2.error):
        return False


def imread(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """cv2.imread that also works for non-ASCII paths, and tolerates a file
    that is being rewritten at the same moment (returns None)."""
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except (OSError, cv2.error):
        return None


# --- CPU / RAM -----------------------------------------------------------------
try:
    import psutil  # type: ignore
except Exception:              # optional: the fallbacks below cover each OS
    psutil = None


class SystemMonitor:
    """CPU % (since the previous call) and RAM % on Linux, Windows and macOS."""

    def __init__(self) -> None:
        self._last = None
        if psutil is not None:
            psutil.cpu_percent(None)        # prime the counter

    def ram_percent(self) -> float:
        if psutil is not None:
            return float(psutil.virtual_memory().percent)
        if IS_LINUX:
            total = avail = 0
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        avail = int(line.split()[1])
            return 100.0 - avail / total * 100.0 if total else 0.0
        if IS_WINDOWS:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return float(st.dwMemoryLoad)
        if IS_MAC:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3).stdout
            pages = {}
            page = 4096
            for line in out.splitlines():
                if "page size of" in line:
                    page = int(line.split("page size of")[1].split()[0])
                elif ":" in line:
                    k, v = line.split(":", 1)
                    try:
                        pages[k.strip()] = int(v.strip().rstrip("."))
                    except ValueError:
                        pass
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                       text=True, timeout=3).stdout.strip() or 0)
            free = (pages.get("Pages free", 0) + pages.get("Pages inactive", 0)
                    + pages.get("Pages speculative", 0)) * page
            return 100.0 * (1 - free / total) if total else 0.0
        return 0.0

    def _cpu_times(self):
        """(total, idle) in arbitrary units, or None."""
        if IS_LINUX:
            with open("/proc/stat") as f:
                v = list(map(int, f.readline().split()[1:8]))
            return sum(v), v[3] + v[4]
        if IS_WINDOWS:
            idle, kernel, user = (ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong())
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel),
                                                  ctypes.byref(user))
            # kernel time includes idle time
            return kernel.value + user.value, idle.value
        return None

    def cpu_percent(self) -> float:
        if psutil is not None:
            return float(psutil.cpu_percent(None))
        if IS_MAC:
            # no cheap counter without psutil: 1-min load average scaled by cores
            try:
                return min(100.0, os.getloadavg()[0] / (os.cpu_count() or 1) * 100.0)
            except OSError:
                return 0.0
        cur = self._cpu_times()
        if cur is None:
            return 0.0
        if self._last is None:
            self._last = cur
            return 0.0
        dt, di = cur[0] - self._last[0], cur[1] - self._last[1]
        self._last = cur
        return max(0.0, min(100.0, 100.0 * (dt - di) / dt)) if dt > 0 else 0.0


# --- fonts -------------------------------------------------------------------------
if IS_WINDOWS:
    UI_FAMILY, MONO_FAMILY = "Segoe UI", "Consolas"
elif IS_MAC:
    UI_FAMILY, MONO_FAMILY = "Helvetica Neue", "Menlo"
else:
    UI_FAMILY, MONO_FAMILY = "DejaVu Sans", "DejaVu Sans Mono"


def wait_for(predicate, timeout: float, step: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()
