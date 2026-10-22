"""Frame layouts of the TC001 family, and how to turn each into temperatures.

The TOPDON TC001 has shipped with (at least) two different USB interfaces, and
they are *not* interchangeable -- decoding one as the other produces numbers
that look plausible and are completely wrong:

``classic-384``  (original TC001 / TS001 / InfiRay P2 Pro)
    One YUYV mode, 256x384. Rows 0-191 are the camera's own 8-bit picture,
    rows 192-383 are 16-bit radiometric samples, Kelvin x 64.

``temp-12305``   (newer TC001 revision -- no 256x384 mode at all)
    A mode the camera advertises as "4x12305" YUYV. It is really a 136-byte
    header followed by 256x192 uint16 temperatures (unit: ``TEMP_MODE_DIVISOR``
    per Kelvin). This is the mode to use on that camera: it is the camera's
    own per-pixel temperature, nothing derived.

``gray-392``     (newer TC001 revision, its default 256x392 mode)
    Rows 0-191: 16-bit *uncalibrated* sensor counts ("gray"), rows 192-195:
    info lines, rows 196-387: the 8-bit picture, rows 388-391: info lines with
    the camera's max/min/average/centre temperature of the frame. Per-pixel
    temperatures are only *approximate* here (fitted from those four numbers),
    so this layout is a fallback for platforms that cannot select the
    temperature mode.

Every decoder returns temperatures as uint16 Kelvin x 64 -- the classic
camera's own unit -- so the rest of the program (recorder, files, maths) has
one representation whatever camera is attached.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SENSOR_WIDTH = 256
SENSOR_HEIGHT = 192
KELVIN_X64 = 64.0

#: Units per Kelvin of the newer TC001's temperature data (temp-12305 mode and
#: the max/min/avg/centre numbers in the gray-392 info lines). 16 follows the
#: InfiRay "Tiny" convention (value = Kelvin x 16). NOT yet confirmed against
#: a reference temperature on real hardware -- run tests/run_thermal_unit_check
#: and change it (or "thermal_temp_divisor" in settings.json) if a known
#: temperature reads wrong.
TEMP_MODE_DIVISOR = 16.0


def set_temp_mode_divisor(value: float) -> None:
    """Override :data:`TEMP_MODE_DIVISOR` (e.g. from the app's settings)."""
    global TEMP_MODE_DIVISOR
    value = float(value)
    if not 1.0 <= value <= 1000.0:
        raise ValueError(f"temperature divisor out of range: {value}")
    TEMP_MODE_DIVISOR = value


class LayoutError(RuntimeError):
    """The bytes do not form a usable frame of this layout."""


@dataclass
class Decoded:
    raw64: np.ndarray                 # (192, 256) uint16, Kelvin x 64
    gray: np.ndarray                  # (192, 256) uint8, for display only
    approximate: bool = False         # per-pixel values fitted, not measured
    info: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Layout:
    name: str
    width: int                        # capture mode as the camera advertises it
    height: int
    description: str
    approximate: bool = False

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * 2

    def decode(self, data, swap_bytes: bool = False) -> Decoded:
        flat = as_bytes(data)
        if flat.size < self.frame_bytes:
            raise LayoutError(f"{self.name}: frame has {flat.size} bytes, expected {self.frame_bytes}")
        flat = flat[:self.frame_bytes]
        return _DECODERS[self.name](flat, swap_bytes)


CLASSIC_384 = Layout("classic-384", 256, 384,
                     "8-bit picture on top, Kelvin x 64 below (original TC001 / P2 Pro)")
TEMP_12305 = Layout("temp-12305", 4, 12305,
                    "header + 256x192 temperatures (newer TC001)")
GRAY_392 = Layout("gray-392", 256, 392,
                  "sensor counts + picture + info lines (newer TC001 default mode); "
                  "temperatures fitted from the camera's frame statistics",
                  approximate=True)

#: Best first. The capture code picks the first one the camera offers.
LAYOUTS = (TEMP_12305, CLASSIC_384, GRAY_392)
BY_NAME = {lay.name: lay for lay in LAYOUTS}
BY_BYTES = {lay.frame_bytes: lay for lay in LAYOUTS}


# --- helpers ------------------------------------------------------------------

def as_bytes(data) -> np.ndarray:
    """Any frame OpenCV or FFmpeg hands over, as a flat uint8 array (memory order)."""
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data).view(np.uint8).reshape(-1)
    return np.frombuffer(data, np.uint8)


def _u16(pairs: np.ndarray, swap: bool = False) -> np.ndarray:
    lo, hi = (pairs[..., 1], pairs[..., 0]) if swap else (pairs[..., 0], pairs[..., 1])
    return lo.astype(np.uint16) | (hi.astype(np.uint16) << np.uint16(8))


def looks_like_picture(half: np.ndarray) -> bool:
    """True if a (rows, 256, 2) block is the camera's 8-bit YUYV picture.

    The TC001's picture is grey, so every chroma byte is 128. Read as 16-bit
    "temperatures" that becomes 0x80xx -- 239..243 C, which passes every range
    check while being pure nonsense. This is the check that catches it.
    """
    chroma = half[..., 1]
    return chroma.size > 0 and float(np.mean(np.abs(chroma.astype(np.int16) - 128) <= 1)) > 0.98


def _display_gray(raw64: np.ndarray) -> np.ndarray:
    v = raw64.astype(np.float32)
    lo, hi = np.percentile(v, (1, 99))
    if hi - lo < 1:
        hi = lo + 1
    return np.clip((v - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def kelvin_to_raw64(kelvin: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(kelvin, np.float64) * KELVIN_X64), 0, 65535).astype(np.uint16)


# --- classic ------------------------------------------------------------------

def _decode_classic(flat: np.ndarray, swap: bool) -> Decoded:
    frame = flat.reshape(2 * SENSOR_HEIGHT, SENSOR_WIDTH, 2)
    top, bottom = frame[:SENSOR_HEIGHT], frame[SENSOR_HEIGHT:]
    # Normally picture on top, temperatures below. Accept the reverse too,
    # but never a picture in place of the temperatures.
    if looks_like_picture(bottom) and not looks_like_picture(top):
        top, bottom = bottom, top
    elif looks_like_picture(bottom):
        raise LayoutError("both halves are 8-bit pictures: this is not radiometric data")
    return Decoded(raw64=_u16(bottom, swap), gray=np.ascontiguousarray(top[..., 0]))


# --- newer TC001: temperature mode --------------------------------------------

TEMP_HEADER_BYTES = 136
_TEMP_DATA_BYTES = SENSOR_WIDTH * SENSOR_HEIGHT * 2


def _decode_temp(flat: np.ndarray, swap: bool) -> Decoded:
    data = flat[-_TEMP_DATA_BYTES:]
    values = data.view("<u2").reshape(SENSOR_HEIGHT, SENSOR_WIDTH)
    if not values.any():
        raise LayoutError("temperature frame is empty (camera still starting)")
    kelvin = values.astype(np.float32) / np.float32(TEMP_MODE_DIVISOR)
    raw64 = kelvin_to_raw64(kelvin)
    return Decoded(raw64=raw64, gray=_display_gray(raw64),
                   info={"header": flat[:TEMP_HEADER_BYTES].view("<u2")[:32].tolist()})


# --- newer TC001: default 256x392 mode ----------------------------------------

_OSD_TEMP = re.compile(rb"max_temp is (\d+), min_temp is (\d+), avg_temp is (\d+), cen_temp is (\d+)")


def frame_temperature_stats(info_rows: np.ndarray) -> Optional[dict]:
    """max/min/avg/centre temperature (camera units) from the info lines."""
    words = _u16(info_rows[0])
    if words[0] == 0xFFFF and words[1] == 0xEEEE:
        st = {"max": int(words[2]), "min": int(words[5]), "avg": int(words[8]), "cen": int(words[9])}
        if 0 < st["min"] <= st["avg"] <= st["max"]:
            return st
    m = _OSD_TEMP.search(info_rows.tobytes())
    if m:
        mx, mn, avg, cen = (int(x) for x in m.groups())
        return {"max": mx, "min": mn, "avg": avg, "cen": cen}
    return None


class GrayFit:
    """Remembers the last good sensor-count -> temperature slope, for frames
    whose own statistics cannot give one (a perfectly uniform scene)."""

    slope: Optional[float] = None


def _decode_gray392(flat: np.ndarray, swap: bool, _memo=GrayFit()) -> Decoded:
    frame = flat.reshape(392, SENSOR_WIDTH, 2)
    counts_block, picture, info = frame[0:192], frame[196:388], frame[388:392]
    if not looks_like_picture(picture):
        raise LayoutError("rows 196-387 are not the camera picture: not a gray-392 frame")
    counts = _u16(counts_block).astype(np.float64)
    stats = frame_temperature_stats(info)
    if stats is None:
        raise LayoutError("no temperature statistics in the info lines")
    g = {"max": counts.max(), "min": counts.min(), "avg": counts.mean(), "cen": counts[96, 128]}
    xs = np.array([g[k] for k in ("min", "max", "avg", "cen")])
    ys = np.array([stats[k] for k in ("min", "max", "avg", "cen")], dtype=np.float64)
    if xs.max() - xs.min() >= 2:
        slope, intercept = np.polyfit(xs, ys, 1)
        _memo.slope = slope
    elif _memo.slope is not None:
        slope = _memo.slope
        intercept = stats["avg"] - slope * g["avg"]
    else:
        slope, intercept = 0.0, float(stats["avg"])
    temps = counts * slope + intercept
    raw64 = kelvin_to_raw64(temps / TEMP_MODE_DIVISOR)
    return Decoded(raw64=raw64, gray=np.ascontiguousarray(picture[..., 0]), approximate=True,
                   info={"camera_stats": stats, "slope": float(slope)})


_DECODERS = {
    CLASSIC_384.name: _decode_classic,
    TEMP_12305.name: _decode_temp,
    GRAY_392.name: _decode_gray392,
}


def layout_for_modes(modes) -> list[Layout]:
    """Layouts the camera supports, best first, from its advertised modes
    (iterable of (width, height) or 'WxH' strings)."""
    sizes = set()
    for m in modes:
        if isinstance(m, str):
            w, h = m.lower().split("x")
            sizes.add((int(w), int(h)))
        else:
            sizes.add((int(m[0]), int(m[1])))
    return [lay for lay in LAYOUTS if (lay.width, lay.height) in sizes]


def is_thermal_mode_list(modes) -> bool:
    """Does a camera advertising these modes look like a TC001-class camera?"""
    if layout_for_modes(modes):
        return True
    for m in modes:
        w = int(m.lower().split("x")[0]) if isinstance(m, str) else int(m[0])
        if w == SENSOR_WIDTH:
            return True
    return False


def layout_for_frame(flat_size: int) -> Optional[Layout]:
    return BY_BYTES.get(int(flat_size))
