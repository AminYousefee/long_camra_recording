"""The per-frame result object returned by :meth:`ThermalCamera.read`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .decode import raw_to_celsius


@dataclass(frozen=True)
class Reading:
    """A single temperature sample with its pixel location."""

    x: int
    y: int
    celsius: float

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Reading(x={self.x}, y={self.y}, {self.celsius:.2f}C)"


@dataclass
class ThermalFrame:
    """One decoded frame.

    Attributes
    ----------
    raw:
        ``(192, 256)`` uint16 array of raw sensor samples (Kelvin * 64).
    gray:
        ``(192, 256)`` uint8 array -- the camera's own auto-contrast image.
        Nice for previews, not radiometric.
    offset_c, gain:
        Linear correction applied on top of the factory conversion, as
        ``corrected = raw_celsius * gain + offset_c``. Both default to a no-op.
    """

    raw: np.ndarray
    gray: np.ndarray
    offset_c: float = 0.0
    gain: float = 1.0
    #: True when per-pixel values were fitted from the camera's frame
    #: statistics rather than read from its temperature data.
    approximate: bool = False
    _celsius: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    # --- Temperature data ----------------------------------------------------

    @property
    def celsius(self) -> np.ndarray:
        """``(192, 256)`` float32 array of temperatures in degrees Celsius."""
        if self._celsius is None:
            base = raw_to_celsius(self.raw)
            if self.gain != 1.0 or self.offset_c != 0.0:
                base = base * np.float32(self.gain) + np.float32(self.offset_c)
            self._celsius = base
        return self._celsius

    @property
    def fahrenheit(self) -> np.ndarray:
        return self.celsius * 1.8 + 32.0

    @property
    def kelvin(self) -> np.ndarray:
        return self.celsius + 273.15

    @property
    def shape(self) -> tuple[int, int]:
        return self.raw.shape  # type: ignore[return-value]

    # --- Point / region queries ---------------------------------------------

    def at(self, x: int, y: int) -> float:
        """Temperature in Celsius at pixel ``(x, y)``, origin top-left."""
        return float(self.celsius[y, x])

    def region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Celsius sub-array for the rectangle at ``(x, y)`` of size ``w x h``."""
        return self.celsius[y:y + h, x:x + w]

    @property
    def center(self) -> Reading:
        h, w = self.shape
        return Reading(w // 2, h // 2, self.at(w // 2, h // 2))

    @property
    def hottest(self) -> Reading:
        y, x = np.unravel_index(int(np.argmax(self.celsius)), self.shape)
        return Reading(int(x), int(y), float(self.celsius[y, x]))

    @property
    def coldest(self) -> Reading:
        y, x = np.unravel_index(int(np.argmin(self.celsius)), self.shape)
        return Reading(int(x), int(y), float(self.celsius[y, x]))

    @property
    def mean(self) -> float:
        return float(np.mean(self.celsius))

    def stats(self) -> dict:
        """Everything you would normally print in a HUD, in one dict."""
        return {
            "min_c": self.coldest.celsius,
            "max_c": self.hottest.celsius,
            "mean_c": self.mean,
            "center_c": self.center.celsius,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        s = self.stats()
        return (f"ThermalFrame({self.shape[1]}x{self.shape[0]}, "
                f"min={s['min_c']:.1f}C max={s['max_c']:.1f}C "
                f"mean={s['mean_c']:.1f}C)")
