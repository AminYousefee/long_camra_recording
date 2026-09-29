"""Raw frame decoding for InfiRay-based UVC thermal cameras (TOPDON TC001 / TS001).

The camera enumerates as a plain UVC webcam streaming YUY2 at 256x384.
That tall frame is two stacked 256x192 halves:

    rows   0..191  -> "visual" half: a contrast-stretched 8-bit image (YUYV)
    rows 192..383  -> "thermal" half: 16-bit raw radiometric samples

Each thermal pixel is two bytes, little-endian, encoding Kelvin * 64:

    raw     = byte0 + (byte1 << 8)
    kelvin  = raw / 64
    celsius = raw / 64 - 273.15

Credit for the format reverse-engineering: LeoDJ (P2Pro-Viewer) and
Les Wright (PyThermalCamera). This module only re-implements the decode.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# --- Sensor geometry ---------------------------------------------------------

SENSOR_WIDTH = 256
SENSOR_HEIGHT = 192
FRAME_HEIGHT = SENSOR_HEIGHT * 2  # visual half stacked on thermal half

# --- Radiometric constants ---------------------------------------------------

KELVIN_DIVISOR = 64.0
CELSIUS_OFFSET = 273.15

# Datasheet measurement range; a hard sanity bound on any decoded frame.
DEVICE_MIN_C = -20.0
DEVICE_MAX_C = 550.0

# The band an ordinary scene actually lives in. Used only as a soft hint when
# disambiguating byte order -- readings outside it are legal, just less likely.
TYPICAL_MIN_C = -20.0
TYPICAL_MAX_C = 250.0


class FrameLayoutError(RuntimeError):
    """Raised when OpenCV hands back a frame we cannot decode losslessly.

    The usual cause on Windows is a capture backend that ignores
    ``CAP_PROP_CONVERT_RGB=0`` and converts YUY2 to 3-channel BGR. That
    conversion is lossy and permanently destroys the thermal bytes.
    """


def normalize_buffer(frame: np.ndarray, sensor_height: int = SENSOR_HEIGHT,
                     sensor_width: int = SENSOR_WIDTH) -> np.ndarray:
    """Coerce whatever OpenCV returned into a ``(2*H, W, 2)`` uint8 array.

    Different backends present the same YUY2 bytes in different shapes:

    * ``(384, 256, 2)`` uint8  -- the common, already-correct case
    * ``(384, 256)``    uint16 -- V4L2 sometimes packs the pair into one uint16
    * ``(384, 512)``    uint8  -- flattened rows, occasionally with stride padding

    A 3-channel result means the driver already converted to BGR; the thermal
    bytes are gone and we raise instead of returning nonsense temperatures.
    """
    if frame is None or frame.size == 0:
        raise FrameLayoutError("Empty frame from capture device.")

    total_rows = sensor_height * 2

    if frame.ndim == 3 and frame.shape[2] not in (2,):
        raise FrameLayoutError(
            f"Capture backend returned a converted frame (shape={frame.shape}, "
            f"dtype={frame.dtype}). The thermal half has been destroyed by "
            f"YUY2->BGR conversion. Try a different backend, or make sure the "
            f"vendor driver/app is not holding the device."
        )

    # uint16-packed layout -> reinterpret as byte pairs.
    if frame.ndim == 2 and frame.dtype == np.uint16:
        rows, cols = frame.shape
        if rows >= total_rows and cols >= sensor_width:
            extra_rows = rows - total_rows
            start = extra_rows // 2 if extra_rows <= 16 and extra_rows % 2 == 0 else 0
            frame = frame.view(np.uint8).reshape(rows, cols, 2)[
                start:start + total_rows]

    if frame.ndim == 3 and frame.shape[2] == 2:
        extra_rows = frame.shape[0] - total_rows
        if 0 <= extra_rows <= 16 and extra_rows % 2 == 0:
            start = extra_rows // 2
            frame = frame[start:start + total_rows]

    # Flattened / stride-padded layout -> fold back into rows.
    if frame.ndim == 2:
        flat = np.ascontiguousarray(frame).reshape(-1)
        row_bytes = sensor_width * 2
        if frame.shape[0] == 1 and flat.size % row_bytes == 0:
            source_rows = flat.size // row_bytes
            extra_rows = source_rows - total_rows
            if 0 <= extra_rows <= 16 and extra_rows % 2 == 0:
                start = extra_rows // 2
                frame = flat.reshape(source_rows, sensor_width, 2)[
                    start:start + total_rows]

        if frame.ndim == 2:
            flat = np.ascontiguousarray(frame).reshape(-1)
            if flat.size % total_rows != 0:
                raise FrameLayoutError(
                    f"Cannot reshape flat buffer of {flat.size} bytes into "
                    f"{total_rows} rows (shape={frame.shape}, dtype={frame.dtype})."
                )
            bytes_per_row = flat.size // total_rows
            if bytes_per_row % 2 != 0 or bytes_per_row // 2 < sensor_width:
                raise FrameLayoutError(
                    f"Unexpected row stride: {bytes_per_row} bytes/row for a "
                    f"{sensor_width}px-wide sensor."
                )
            frame = flat.reshape(total_rows, bytes_per_row // 2, 2)
            frame = frame[:, :sensor_width, :]

    if frame.ndim != 3 or frame.shape[2] != 2:
        raise FrameLayoutError(
            f"Unsupported frame layout: shape={frame.shape}, dtype={frame.dtype}."
        )

    if frame.shape[0] < total_rows:
        raise FrameLayoutError(
            f"Frame has {frame.shape[0]} rows, expected at least {total_rows}. "
            f"The capture resolution is probably not set to {sensor_width}x{total_rows}."
        )

    return frame[:total_rows, :sensor_width, :]


def split_halves(frame: np.ndarray, sensor_height: int = SENSOR_HEIGHT
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Split a normalized frame into ``(visual_half, thermal_half)``."""
    return frame[:sensor_height], frame[sensor_height:sensor_height * 2]


def raw_from_thermal(thermal_half: np.ndarray, swap_bytes: bool = False) -> np.ndarray:
    """Combine the byte pairs of the thermal half into a uint16 raw array."""
    lo_index, hi_index = (1, 0) if swap_bytes else (0, 1)
    lo = thermal_half[..., lo_index].astype(np.uint16)
    hi = thermal_half[..., hi_index].astype(np.uint16)
    return lo | (hi << np.uint16(8))


def raw_to_celsius(raw: np.ndarray | float) -> np.ndarray | float:
    """Convert raw 16-bit samples (Kelvin * 64) to degrees Celsius."""
    return np.asarray(raw, dtype=np.float32) / KELVIN_DIVISOR - CELSIUS_OFFSET


def celsius_to_raw(celsius: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`raw_to_celsius`, for building synthetic test frames."""
    return np.rint((np.asarray(celsius, dtype=np.float64) + CELSIUS_OFFSET)
                   * KELVIN_DIVISOR).astype(np.uint16)


def visual_to_gray(visual_half: np.ndarray) -> np.ndarray:
    """Extract the 8-bit luma plane from the YUYV visual half.

    In YUY2 the byte layout per pixel pair is ``Y0 U Y1 V``, so channel 0 of
    every pixel is that pixel's luma. This is the camera's own auto-contrast
    image -- convenient to look at, but *not* radiometric. Use the temperature
    array for anything you intend to measure.
    """
    return np.ascontiguousarray(visual_half[..., 0])


def plausibility_score(celsius: np.ndarray) -> float:
    """Score how much a decoded frame looks like a real thermal image.

    Three signals, because no single one is sufficient:

    * *in range* -- pixels inside the device's rated span. Weighted highest;
      a wrong decode usually pushes most pixels far outside it.
    * *typical* -- pixels inside an everyday temperature band. This is what
      breaks ties on flat scenes, where a byte-swap can still land inside the
      device range but at an absurd value (25C decodes to 280C swapped).
    * *smoothness* -- real thermal scenes vary gently between neighbouring
      pixels. Swapping the bytes promotes the LSB to MSB, so adjacent pixels
      jump by hundreds of degrees. This is what breaks ties on textured scenes.

    Higher is better. Roughly 0..3.
    """
    in_range = float(np.mean((celsius >= DEVICE_MIN_C) & (celsius <= DEVICE_MAX_C)))
    typical = float(np.mean((celsius >= TYPICAL_MIN_C) & (celsius <= TYPICAL_MAX_C)))

    if celsius.ndim >= 2 and celsius.shape[1] > 1:
        roughness = float(np.mean(np.abs(np.diff(celsius.astype(np.float32), axis=1))))
    else:
        roughness = 0.0
    smoothness = 1.0 / (1.0 + roughness)

    return in_range * 1.5 + typical + smoothness * 0.5


#: Hard physical bounds. Nothing outside these can be a real reading from this
#: sensor, whatever the scene -- a frame containing such values is misdecoded.
ABSOLUTE_MIN_C = -60.0
ABSOLUTE_MAX_C = 620.0

#: Largest span a single ordinary scene should cover. Misdecoded data spreads
#: across hundreds of degrees because the high byte is being read as noise.
MAX_SCENE_SPREAD_C = 320.0


def scene_rejection_reason(celsius: np.ndarray) -> Optional[str]:
    """Why this frame cannot be a real thermal scene, or None if it could be.

    Stricter than :func:`plausibility_score`, and used for a different job:
    deciding whether a capture path is trustworthy enough to hand to the user.
    Scoring is fine for picking the better of two byte orders, but as an
    accept/reject gate it is too generous -- misdecoded data can score well
    while containing values like -156 C and +750 C in the same frame. Reporting
    that as a working camera is the worst possible outcome, so each check here
    is an absolute veto rather than a contribution to an average.
    """
    finite = np.isfinite(celsius)
    if not finite.all():
        return "frame contains non-finite values"

    low = float(np.min(celsius))
    high = float(np.max(celsius))

    if low < ABSOLUTE_MIN_C or high > ABSOLUTE_MAX_C:
        return (f"values from {low:.1f}C to {high:.1f}C, outside anything this "
                f"sensor can physically produce")

    spread = high - low
    if spread > MAX_SCENE_SPREAD_C:
        return f"scene spans {spread:.0f}C, far wider than a real one"

    outside = float(np.mean((celsius < DEVICE_MIN_C) | (celsius > DEVICE_MAX_C)))
    if outside > 0.005:
        return (f"{outside:.1%} of pixels fall outside the rated "
                f"{DEVICE_MIN_C:.0f}..{DEVICE_MAX_C:.0f}C range")

    median = float(np.median(celsius))
    if not DEVICE_MIN_C <= median <= DEVICE_MAX_C:
        return f"median {median:.1f}C is outside the rated range"

    if spread < 0.02:
        return "frame is perfectly flat, so it is not live sensor data"

    return None


def detect_byte_order(thermal_half: np.ndarray) -> bool:
    """Return ``True`` if the byte pairs appear to be swapped.

    Decodes the frame both ways and keeps whichever scores higher under
    :func:`plausibility_score`. Ties resolve to ``False``, the documented
    little-endian order.

    Some Windows capture paths hand back the two bytes in the opposite order,
    which yields a picture that looks plausible but reads wildly wrong -- worth
    detecting automatically rather than leaving as a config flag people forget.
    """
    normal = plausibility_score(
        raw_to_celsius(raw_from_thermal(thermal_half, swap_bytes=False)))
    swapped = plausibility_score(
        raw_to_celsius(raw_from_thermal(thermal_half, swap_bytes=True)))
    return swapped > normal
