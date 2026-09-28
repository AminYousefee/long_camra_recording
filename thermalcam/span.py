"""Manual and automatic temperature span (level/range) control.

The camera's own 8-bit image is auto-contrast-stretched every frame, which makes
consecutive frames incomparable: the same object shifts colour as soon as
something hotter enters the scene. For any measurement or A/B comparison you
want a *fixed* span -- you pin the low and high Celsius bounds yourself, and a
given temperature then always maps to the same colour.

    span = TemperatureSpan.manual(20.0, 40.0)   # locked, 20C -> 40C
    span = TemperatureSpan.auto()               # per-frame min/max
    span = TemperatureSpan.percentile(1, 99)    # per-frame, outliers clipped
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

Mode = Literal["manual", "auto", "percentile"]


@dataclass
class TemperatureSpan:
    """Maps Celsius values onto the 0-255 display range.

    Construct through the classmethods rather than directly. In ``manual`` mode
    ``vmin``/``vmax`` are authoritative and never change; in the other modes
    they are recomputed from each frame.
    """

    mode: Mode = "auto"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    min_range: float = 0.5  # guards against divide-by-zero on flat scenes

    # --- Constructors --------------------------------------------------------

    @classmethod
    def manual(cls, vmin: float, vmax: float) -> "TemperatureSpan":
        """Lock the span to a fixed Celsius window."""
        if vmax <= vmin:
            raise ValueError(f"vmax ({vmax}) must be greater than vmin ({vmin}).")
        return cls(mode="manual", vmin=float(vmin), vmax=float(vmax))

    @classmethod
    def auto(cls) -> "TemperatureSpan":
        """Rescale to each frame's own min and max."""
        return cls(mode="auto")

    @classmethod
    def percentile(cls, lower: float = 1.0, upper: float = 99.0) -> "TemperatureSpan":
        """Rescale per frame, ignoring the tails (robust against hot specks)."""
        if not 0.0 <= lower < upper <= 100.0:
            raise ValueError("Require 0 <= lower < upper <= 100.")
        return cls(mode="percentile", lower_percentile=lower, upper_percentile=upper)

    # --- Mutation ------------------------------------------------------------

    def set_manual(self, vmin: float, vmax: float) -> "TemperatureSpan":
        """Switch to manual mode with new bounds. Returns self for chaining."""
        if vmax <= vmin:
            raise ValueError(f"vmax ({vmax}) must be greater than vmin ({vmin}).")
        self.mode = "manual"
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        return self

    def set_auto(self) -> "TemperatureSpan":
        self.mode = "auto"
        return self

    def shift(self, delta: float) -> "TemperatureSpan":
        """Move a manual window up or down without changing its width."""
        self._require_manual("shift")
        self.vmin += delta
        self.vmax += delta
        return self

    def widen(self, delta: float) -> "TemperatureSpan":
        """Grow (or with a negative delta, shrink) a manual window symmetrically."""
        self._require_manual("widen")
        centre = (self.vmin + self.vmax) / 2.0
        half = max((self.vmax - self.vmin) / 2.0 + delta, self.min_range / 2.0)
        self.vmin, self.vmax = centre - half, centre + half
        return self

    def center_on(self, celsius: float, width: Optional[float] = None) -> "TemperatureSpan":
        """Re-centre a manual window on a temperature, optionally resizing it."""
        current = (self.vmax - self.vmin) if self.mode == "manual" else 10.0
        w = float(width) if width is not None else current
        return self.set_manual(celsius - w / 2.0, celsius + w / 2.0)

    def _require_manual(self, op: str) -> None:
        if self.mode != "manual" or self.vmin is None or self.vmax is None:
            raise ValueError(f"{op}() needs a manual span; call set_manual() first.")

    # --- Application ---------------------------------------------------------

    def bounds(self, celsius: np.ndarray) -> tuple[float, float]:
        """The (vmin, vmax) this span would use for the given frame."""
        if self.mode == "manual":
            assert self.vmin is not None and self.vmax is not None
            return self.vmin, self.vmax

        if self.mode == "percentile":
            lo = float(np.percentile(celsius, self.lower_percentile))
            hi = float(np.percentile(celsius, self.upper_percentile))
        else:
            lo = float(np.min(celsius))
            hi = float(np.max(celsius))

        if hi - lo < self.min_range:
            centre = (hi + lo) / 2.0
            lo, hi = centre - self.min_range / 2.0, centre + self.min_range / 2.0
        return lo, hi

    def normalize(self, celsius: np.ndarray) -> np.ndarray:
        """Map a Celsius array to uint8 0-255, clipping outside the span."""
        lo, hi = self.bounds(celsius)
        scaled = (np.asarray(celsius, dtype=np.float32) - lo) * (255.0 / (hi - lo))
        return np.clip(scaled, 0.0, 255.0).astype(np.uint8)

    def clipping(self, celsius: np.ndarray) -> tuple[float, float]:
        """Fraction of pixels below and above the span.

        Handy while dialling in a manual window -- if either number is large you
        have chosen bounds that hide most of the scene.
        """
        lo, hi = self.bounds(celsius)
        total = celsius.size
        return (float(np.count_nonzero(celsius < lo)) / total,
                float(np.count_nonzero(celsius > hi)) / total)

    # --- Serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "min_range": self.min_range,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TemperatureSpan":
        return cls(
            mode=data.get("mode", "auto"),
            vmin=data.get("vmin"),
            vmax=data.get("vmax"),
            lower_percentile=data.get("lower_percentile", 1.0),
            upper_percentile=data.get("upper_percentile", 99.0),
            min_range=data.get("min_range", 0.5),
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.mode == "manual":
            return f"TemperatureSpan(manual, {self.vmin:.1f}C..{self.vmax:.1f}C)"
        if self.mode == "percentile":
            return (f"TemperatureSpan(percentile, "
                    f"{self.lower_percentile}..{self.upper_percentile})")
        return "TemperatureSpan(auto)"
