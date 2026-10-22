"""What capture modes a camera offers, and how to address it with FFmpeg.

Used for both cameras: the thermal camera (to find its temperature mode and to
recognise it among identically named "USB Camera"s) and the normal camera (to
record in a mode it really has, instead of asking for 4K from a 1080p webcam
and having FFmpeg exit).

Nothing here opens a video stream; listing modes is a query to the driver, so
it is safe while another program is using the camera.

    Windows  ffmpeg -f dshow -list_options true -i video=NAME
    Linux    v4l2-ctl --list-formats-ext   (or ffmpeg -f v4l2 -list_formats all)
    macOS    ffmpeg -f avfoundation with an impossible size, which makes it
             print the supported modes
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Optional

from .devices import VideoDevice, list_video_devices

_TIMEOUT = 15


def _no_window() -> dict:
    if sys.platform.startswith("win"):
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def _run(cmd: list[str], timeout: float = _TIMEOUT) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=timeout, **_no_window())
    except (OSError, subprocess.SubprocessError):
        return ""
    return (res.stdout + res.stderr).decode("utf-8", errors="replace")


@dataclass(frozen=True)
class Mode:
    """One capture mode: a pixel format (or codec) at a size and frame rate."""

    fmt: str                 # 'yuyv422', 'nv12', 'mjpeg', 'h264', ...
    width: int
    height: int
    fps: float               # highest frame rate offered at this size (0 = unknown)
    compressed: bool = False

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def __str__(self) -> str:
        rate = f"@{self.fps:g}" if self.fps else ""
        return f"{self.fmt} {self.size}{rate}"


@dataclass(frozen=True)
class DeviceRef:
    """How FFmpeg (and OpenCV) address one camera on this OS."""

    index: Optional[int]          # OpenCV index (Windows / macOS) or N of /dev/videoN
    name: str
    number: int = 0               # Windows: Nth device with the same name
    path: str = ""                # Linux: /dev/videoN

    def ffmpeg_input(self, *, size: Optional[str] = None, fmt: Optional[str] = None,
                     compressed: bool = False, fps: Optional[float] = None,
                     platform: Optional[str] = None, rtbufsize: str = "256M") -> list[str]:
        """FFmpeg input options + -i for this device in the given mode."""
        platform = platform or sys.platform
        rate = [] if not fps else ["-framerate", f"{fps:g}"]
        if platform.startswith("win"):
            args = ["-f", "dshow", "-rtbufsize", rtbufsize]
            if size:
                args += ["-video_size", size]
            if fmt:
                args += ["-vcodec" if compressed else "-pixel_format", fmt]
            args += rate
            if self.number:
                args += ["-video_device_number", str(self.number)]
            return args + ["-i", f"video={self.name}"]
        if platform == "darwin":
            args = ["-f", "avfoundation"]
            if size:
                args += ["-video_size", size]
            if fmt and not compressed:
                args += ["-pixel_format", fmt]
            return args + rate + ["-i", f"{self.index}:none"]
        args = ["-f", "v4l2"]
        if fmt:
            args += ["-input_format", fmt]
        if size:
            args += ["-video_size", size]
        return args + rate + ["-i", self.path or f"/dev/video{self.index}"]


def device_refs(devices: Optional[list[VideoDevice]] = None) -> list[DeviceRef]:
    """DeviceRefs for every addressable camera, in OpenCV index order."""
    if devices is None:
        devices = list_video_devices(include_pnp=False)
    refs = []
    seen: dict[str, int] = {}
    for d in devices:
        if d.index is None:
            continue
        n = seen.get(d.name, 0)
        seen[d.name] = n + 1
        path = f"/dev/video{d.index}" if sys.platform.startswith("linux") else ""
        refs.append(DeviceRef(index=d.index, name=d.name, number=n, path=path))
    return refs


def device_ref(index: int, devices: Optional[list[VideoDevice]] = None) -> Optional[DeviceRef]:
    return next((r for r in device_refs(devices) if r.index == index), None)


# --- parsing ------------------------------------------------------------------

_DSHOW = re.compile(r"(pixel_format|vcodec)=(\w+)\s+min s=(\d+)x(\d+) fps=([\d.]+)\s+max s=(\d+)x(\d+) fps=([\d.]+)")
_V4L2_FMT = re.compile(r"\[\d+\]:\s+'(\w+)'")
_V4L2_SIZE = re.compile(r"Size:\s+Discrete\s+(\d+)x(\d+)")
_V4L2_FPS = re.compile(r"Interval:.*\(([\d.]+) fps\)")
_AVF = re.compile(r"(\d+)x(\d+)@\[([\d.]+)\s+([\d.]+)\]fps")

_V4L2_FOURCC = {"YUYV": "yuyv422", "MJPG": "mjpeg", "UYVY": "uyvy422", "NV12": "nv12",
                "H264": "h264", "GREY": "gray", "Y16": "gray16le", "RGB3": "rgb24", "BGR3": "bgr24"}
_COMPRESSED = {"mjpeg", "h264", "hevc", "mpeg4"}


def parse_dshow(text: str) -> list[Mode]:
    best: dict[tuple, float] = {}
    for kind, fmt, _w0, _h0, _f0, w, h, fmax in _DSHOW.findall(text):
        key = (fmt, int(w), int(h), kind == "vcodec")
        best[key] = max(best.get(key, 0.0), float(fmax))
    return [Mode(f, w, h, fps, comp) for (f, w, h, comp), fps in best.items()]


def parse_v4l2_ctl(text: str) -> list[Mode]:
    modes: dict[tuple, float] = {}
    fmt, size = None, None
    for line in text.splitlines():
        m = _V4L2_FMT.search(line)
        if m:
            fmt = _V4L2_FOURCC.get(m.group(1), m.group(1).lower())
            size = None
            continue
        m = _V4L2_SIZE.search(line)
        if m and fmt:
            size = (int(m.group(1)), int(m.group(2)))
            modes.setdefault((fmt, *size), 0.0)
            continue
        m = _V4L2_FPS.search(line)
        if m and fmt and size:
            key = (fmt, *size)
            modes[key] = max(modes[key], float(m.group(1)))
    return [Mode(f, w, h, fps, f in _COMPRESSED) for (f, w, h), fps in modes.items()]


def parse_ffmpeg_v4l2(text: str) -> list[Mode]:
    out = []
    for line in text.splitlines():
        rest = line.split("]", 1)[-1].strip()
        # "Raw : yuyv422 : YUYV 4:2:2 : 640x480 320x240" -- fields are
        # separated by " : " (the description itself may contain "4:2:2")
        parts = re.split(r"\s*:\s+", rest)
        if len(parts) < 4 or parts[0] not in ("Raw", "Compressed"):
            continue
        for size in parts[-1].split():
            m = re.fullmatch(r"(\d+)x(\d+)", size)
            if m:
                out.append(Mode(parts[1], int(m.group(1)), int(m.group(2)), 0.0,
                                parts[0] == "Compressed"))
    return out


def parse_avfoundation(text: str) -> list[Mode]:
    best: dict[tuple, float] = {}
    for w, h, _lo, hi in _AVF.findall(text):
        key = (int(w), int(h))
        best[key] = max(best.get(key, 0.0), float(hi))
    # AVFoundation converts between pixel formats itself; report the camera's
    # native 4:2:2 ordering, which is what the thermal camera streams.
    return [Mode("uyvy422", w, h, fps) for (w, h), fps in best.items()]


# --- query ----------------------------------------------------------------------

_cache: dict[tuple, list[Mode]] = {}
_lock = threading.Lock()


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def list_modes(ref: DeviceRef, use_cache: bool = True) -> list[Mode]:
    """Every mode the camera advertises ([] if it cannot be queried)."""
    key = (sys.platform, ref.name, ref.number, ref.index, ref.path)
    if use_cache:
        with _lock:
            if key in _cache:
                return list(_cache[key])
    modes = _query(ref)
    if modes:
        with _lock:
            _cache[key] = modes
    return list(modes)


def _query(ref: DeviceRef) -> list[Mode]:
    ffmpeg = shutil.which("ffmpeg")
    if sys.platform.startswith("win"):
        if not ffmpeg:
            return []
        cmd = [ffmpeg, "-hide_banner", "-f", "dshow", "-list_options", "true"]
        if ref.number:
            cmd += ["-video_device_number", str(ref.number)]
        return parse_dshow(_run(cmd + ["-i", f"video={ref.name}"]))
    if sys.platform == "darwin":
        if not ffmpeg:
            return []
        return parse_avfoundation(_run([ffmpeg, "-hide_banner", "-f", "avfoundation",
                                        "-video_size", "1x1", "-framerate", "1",
                                        "-i", f"{ref.index}:none", "-frames:v", "1",
                                        "-f", "null", "-"]))
    path = ref.path or f"/dev/video{ref.index}"
    if shutil.which("v4l2-ctl"):
        modes = parse_v4l2_ctl(_run(["v4l2-ctl", "-d", path, "--list-formats-ext"]))
        if modes:
            return modes
    if ffmpeg:
        return parse_ffmpeg_v4l2(_run([ffmpeg, "-hide_banner", "-f", "v4l2",
                                       "-list_formats", "all", "-i", path]))
    return []


# --- choosing a recording mode for the normal camera ------------------------------

def choose_recording_mode(modes: list[Mode], width: int, height: int, fps: float
                          ) -> Optional[Mode]:
    """The camera mode to record from for a requested size and frame rate.

    Preference: the exact size (MJPEG first -- uncompressed 1080p/4K usually
    only runs at a few fps over USB), then the largest size not above the
    request with the same aspect ratio, then the largest size at all. Within a
    size, the mode whose frame rate reaches the request wins.
    """
    usable = [m for m in modes if m.fmt in ("mjpeg", "yuyv422", "nv12", "uyvy422", "yuv420p",
                                            "h264", "rgb24", "bgr24", "bgr0")]
    if not usable:
        return None

    def rank(m: Mode):
        fps_ok = (m.fps == 0) or (m.fps + 0.5 >= fps)
        fmt_rank = {"mjpeg": 0, "nv12": 1, "yuyv422": 2, "uyvy422": 2, "yuv420p": 2,
                    "h264": 3}.get(m.fmt, 4)
        return (not fps_ok, fmt_rank, -m.fps)

    exact = [m for m in usable if (m.width, m.height) == (width, height)]
    if exact:
        # Even if the exact size only runs at a lower rate, it is what the
        # user asked for; the recorder then uses the rate the mode really has.
        return min(exact, key=rank)
    aspect = width / height
    smaller = [m for m in usable if m.width <= width and m.height <= height
               and abs(m.width / m.height - aspect) < 0.05]
    pool = smaller or [m for m in usable if m.width <= width and m.height <= height] or usable
    top = max(m.pixels for m in pool)
    return min((m for m in pool if m.pixels == top), key=rank)
