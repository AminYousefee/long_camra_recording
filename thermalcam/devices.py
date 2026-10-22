"""Enumerate video capture devices by *name*, not just by index.

OpenCV can only address cameras by integer index, and it will not tell you what
any of them are. With one camera that is tolerable. With three -- a built-in
webcam, a USB webcam and a thermal camera -- it is unworkable: indices are
assigned in enumeration order, so they shuffle when you plug something in,
reboot, or dock the machine. Code that hardcodes index 1 silently starts reading
the wrong camera.

This module asks the operating system what is actually there.

Windows
    ``pygrabber`` (pip install pygrabber) enumerates DirectShow devices in the
    same order OpenCV's CAP_DSHOW backend uses, so position in that list *is*
    the OpenCV index. If it is unavailable, FFmpeg's DirectShow listing gives
    device names and indices; PowerShell is the final names-only fallback.

macOS
    FFmpeg's AVFoundation listing gives device names and indices; OpenCV is a
    fallback when FFmpeg enumeration is unavailable.

Linux
    /sys/class/video4linux/video*/name, where the videoN number is the index.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


def _no_window() -> dict:
    """Do not flash a console window for helper processes on Windows."""
    if sys.platform.startswith("win"):
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}

#: Substrings that identify a TC001-class thermal camera by device name.
#: The TC001 usually enumerates under the InfiRay sensor module's name rather
#: than TOPDON's, since TOPDON rebadge InfiRay hardware. Generic USB-camera
#: names are intentionally not enough to classify a device as thermal.
THERMAL_NAME_HINTS = ("topdon", "tc001", "ts001", "infiray", "thermal",
                      "xtherm", "p2 pro")

#: Names that contain a thermal-sounding word but are definitely not our camera.
#: Windows Hello face-recognition modules enumerate as "IR Webcam" or similar,
#: which is easily mistaken for a thermal camera in Device Manager -- they are
#: near-infrared cameras for face unlock, not radiometric sensors, and Windows
#: reserves them so they never appear to DirectShow at all.
NOT_THERMAL_NAME_HINTS = ("ir webcam", "ir camera", "windows hello",
                          "face", "depth", "intel(r) realsense")


@dataclass
class VideoDevice:
    """A camera the operating system reports."""

    index: Optional[int]        # OpenCV index, when we can determine it
    name: str
    source: str                 # how we found it
    status: str = ""            # driver status, where available

    @property
    def looks_thermal(self) -> bool:
        low = self.name.lower()
        if any(hint in low for hint in NOT_THERMAL_NAME_HINTS):
            return False
        return any(hint in low for hint in THERMAL_NAME_HINTS)

    @property
    def is_hello_ir(self) -> bool:
        """A Windows Hello near-infrared camera, not a thermal one.

        Worth naming explicitly: it shows up in Device Manager as an "IR
        camera", which reads exactly like the thermal camera to anyone
        checking whether theirs is connected.
        """
        low = self.name.lower()
        return any(h in low for h in ("ir webcam", "ir camera", "windows hello"))

    @property
    def addressable(self) -> bool:
        """Whether OpenCV can actually open this device by index."""
        return self.index is not None

    def __str__(self) -> str:
        index = "?" if self.index is None else str(self.index)
        tail = f"  [{self.status}]" if self.status else ""
        return f"index {index:>2}  {self.name}{tail}"


# --- Windows -----------------------------------------------------------------

def _windows_ffmpeg() -> list[VideoDevice]:
    """Enumerate DirectShow cameras with FFmpeg when pygrabber is unavailable."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True, timeout=15, **_no_window())
    except (OSError, subprocess.SubprocessError):
        return []

    devices = []
    for line in (result.stdout + result.stderr).splitlines():
        match = re.search(r'\[dshow @ [^\]]+\]\s+"(.+)" \(video\)\s*$', line)
        if match:
            devices.append(VideoDevice(index=len(devices), name=match.group(1),
                                       source="DirectShow/FFmpeg"))
    return devices


def _windows_directshow() -> list[VideoDevice]:
    """DirectShow devices in OpenCV's index order, via pygrabber or FFmpeg."""
    try:
        from pygrabber.dshow_graph import FilterGraph
    except Exception:
        return _windows_ffmpeg()
    try:
        names = FilterGraph().get_input_devices()
    except Exception:
        return _windows_ffmpeg()
    if not names:
        return _windows_ffmpeg()
    return [VideoDevice(index=i, name=name, source="DirectShow")
            for i, name in enumerate(names)]


def _windows_pnp() -> list[VideoDevice]:
    """Cameras as Windows itself lists them, via PowerShell.

    No index mapping, but it answers the question that matters when a device is
    visible in Device Manager yet invisible to OpenCV: is Windows classifying it
    as a camera, and is its driver actually running?
    """
    # Camera and Image only. The 'Media' class is mostly audio hardware, which
    # buries the two or three entries that actually matter under a list of
    # sound controllers.
    script = (
        "Get-CimInstance Win32_PnPEntity | "
        "Where-Object { $_.PNPClass -in 'Camera','Image' } | "
        "ForEach-Object { \"$($_.Name)|$($_.Status)|$($_.PNPClass)\" }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=25, **_no_window())
    except (OSError, subprocess.SubprocessError):
        return []

    devices = []
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 3 and parts[0]:
            devices.append(VideoDevice(index=None, name=parts[0],
                                       source=f"PnP/{parts[2]}",
                                       status=parts[1]))
    return devices


# --- Hardware-level detection ------------------------------------------------

#: Windows Configuration Manager problem codes worth translating. A device can
#: be physically present and still unusable; the code says which.
PROBLEM_CODES = {
    0: "working normally",
    1: "not configured correctly",
    3: "driver corrupted, or out of memory",
    10: "cannot start -- usually a driver or power problem",
    18: "drivers need reinstalling",
    19: "registry configuration is damaged",
    21: "Windows is removing it",
    22: "disabled",
    24: "not present, not working, or missing drivers",
    28: "no drivers installed for this device",
    31: "Windows cannot load the driver for this device",
    43: "Windows stopped it because it reported a problem",
    45: "not currently connected",
}


#: Generic USB drivers that, once bound to a camera, hide it from the capture
#: stack completely. Zadig installs these to let user-space code talk to a
#: device directly; the side effect is that the device stops being a camera as
#: far as Windows is concerned, so DirectShow and Media Foundation cannot see
#: it. The device looks perfectly healthy in Device Manager throughout.
HIJACK_DRIVER_CLASSES = {
    "libusbk": "libusbK",
    "libusb-win32": "libusb-win32",
    "libusb0": "libusb-win32",
    "winusb": "WinUSB",
    "usbdev": "a generic USB driver",
}


@dataclass(frozen=True)
class PnpEntry:
    """One Windows device node, as the Configuration Manager sees it."""

    instance_id: str
    name: str
    device_class: str
    status: str
    problem: int

    @property
    def problem_text(self) -> str:
        return PROBLEM_CODES.get(self.problem, f"problem code {self.problem}")

    @property
    def is_healthy(self) -> bool:
        return self.problem == 0 and self.status.upper() == "OK"

    @property
    def is_usb(self) -> bool:
        return self.instance_id.upper().startswith("USB")

    @property
    def is_camera_class(self) -> bool:
        return self.device_class in ("Camera", "Image")

    @property
    def hijacked_by(self) -> Optional[str]:
        """Name of the generic USB driver holding this device, if any.

        A device in this state reports ``status: OK`` and "working normally",
        which is true and thoroughly misleading -- it is working normally *as a
        generic USB device*, which is precisely why no camera API can open it.
        """
        low = self.device_class.lower()
        for key, label in HIJACK_DRIVER_CLASSES.items():
            if key in low:
                return label
        return None


def snapshot_pnp_devices() -> dict[str, PnpEntry]:
    """Every device Windows currently knows about, keyed by instance ID.

    Taking one of these with the camera plugged in and one without, then
    diffing, is the only way to answer the question that actually matters when
    a device is invisible to the capture layer: does Windows notice it at all?
    A device that does not appear in the difference never reached the operating
    system, which makes it a cable, port or power fault rather than a software
    one.
    """
    if not sys.platform.startswith("win"):
        return {}

    script = (
        "Get-CimInstance Win32_PnPEntity | ForEach-Object { "
        "\"$($_.PNPDeviceID)`t$($_.Name)`t$($_.PNPClass)`t"
        "$($_.Status)`t$($_.ConfigManagerErrorCode)\" }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=60, **_no_window())
    except (OSError, subprocess.SubprocessError):
        return {}

    devices = {}
    for line in out.stdout.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5 or not parts[0]:
            continue
        try:
            problem = int(parts[4])
        except ValueError:
            problem = 0
        devices[parts[0]] = PnpEntry(instance_id=parts[0], name=parts[1] or "(unnamed)",
                                     device_class=parts[2], status=parts[3],
                                     problem=problem)
    return devices


def diff_snapshots(before: dict[str, PnpEntry], after: dict[str, PnpEntry]
                   ) -> list[PnpEntry]:
    """Devices present in ``before`` but gone from ``after``."""
    return [entry for key, entry in before.items() if key not in after]


# --- Linux -------------------------------------------------------------------

def _linux_v4l2() -> list[VideoDevice]:
    from pathlib import Path

    devices = []
    root = Path("/sys/class/video4linux")
    if not root.exists():
        return devices
    for entry in sorted(root.glob("video*"), key=lambda p: int(
            re.sub(r"\D", "", p.name) or 0)):
        # UVC cameras expose a second node per camera for metadata; only the
        # one with index 0 delivers frames.
        try:
            if (entry / "index").read_text().strip() not in ("", "0"):
                continue
        except OSError:
            pass
        name_file = entry / "name"
        try:
            name = name_file.read_text().strip()
        except OSError:
            name = entry.name
        devices.append(VideoDevice(index=int(re.sub(r"\D", "", entry.name) or 0),
                                   name=name, source="V4L2"))
    return devices


def _macos_avfoundation() -> list[VideoDevice]:
    """Enumerate AVFoundation video inputs using FFmpeg, if available."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "avfoundation", "-i", ""],
                capture_output=True, text=True, timeout=15, **_no_window())
        except (OSError, subprocess.SubprocessError):
            result = None

        if result is not None:
            devices = []
            in_video_section = False
            for line in (result.stdout + result.stderr).splitlines():
                if "AVFoundation video devices:" in line:
                    in_video_section = True
                    continue
                if "AVFoundation audio devices:" in line:
                    in_video_section = False
                if not in_video_section:
                    continue
                match = re.search(r"\[AVFoundation indev @ [^\]]+\]\s+\[(\d+)\]\s+(.+)$", line)
                if match:
                    devices.append(VideoDevice(index=int(match.group(1)),
                                               name=match.group(2).strip(),
                                               source="AVFoundation/FFmpeg"))
            if devices:
                return devices

    try:
        import cv2
    except ImportError:
        return []
    devices = []
    for index in range(10):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            devices.append(VideoDevice(index=index, name=f"Camera {index}",
                                       source="AVFoundation/OpenCV"))
        cap.release()
    return devices


# --- Public ------------------------------------------------------------------

def list_video_devices(include_pnp: bool = True) -> list[VideoDevice]:
    """Every video capture device the OS reports, best effort.

    Windows and macOS use their native capture APIs through FFmpeg (with
    optional platform-specific enumerators preferred when available). Linux
    uses the V4L2 device nodes.

    include_pnp (Windows only): also list cameras Windows knows about but
    DirectShow cannot address (names only, index None). That needs PowerShell,
    which takes seconds, so the capture code leaves it out; it is for
    diagnostics.
    """
    if sys.platform.startswith("win"):
        dshow = _windows_directshow()
        if not include_pnp:
            return dshow
        known = {d.name.lower() for d in dshow}
        extra = [d for d in _windows_pnp() if d.name.lower() not in known]
        return dshow + extra
    if sys.platform == "darwin":
        return _macos_avfoundation()
    return _linux_v4l2()


def find_thermal_index() -> Optional[int]:
    """The OpenCV index of the first device whose name looks thermal.

    Returns None if no device matched, or if we could not enumerate names at
    all -- in which case the caller should fall back to probing indices.
    """
    for device in list_video_devices(include_pnp=False):
        if device.looks_thermal and device.index is not None:
            return device.index
    return None


def find_by_name(fragment: str) -> Optional[int]:
    """The OpenCV index of the first device whose name contains ``fragment``.

    Case-insensitive. This is how you pin a camera when several are attached:

    >>> find_by_name("Logitech")   # the USB webcam, wherever it landed
    2
    """
    low = fragment.lower()
    for device in list_video_devices(include_pnp=False):
        if low in device.name.lower() and device.index is not None:
            return device.index
    return None
