"""Reference-point calibration.

You point the camera at things whose real temperature you already know -- ice
water at 0 C, a thermocouple reading, a stirred hot bath -- mark those spots in
the image, and type in the true value. From those pairs this fits a correction
that is then applied to every pixel.

    calib = Calibration()
    calib.add_point(64, 96, known_c=0.0)      # ice bath
    calib.add_point(180, 96, known_c=55.0)    # hot plate
    calib.update(frame.celsius)               # accumulate measurements
    calib.fit()
    corrected = calib.apply(frame.celsius)

How many points you need
------------------------
* **1 point** -- fits an offset. Corrects the reading at that temperature and
  assumes the error is constant elsewhere. Fine over a narrow range.
* **2 points** -- fits a straight line (gain and offset). This is the useful
  case for most work: bracket your range of interest, one reference near the
  bottom and one near the top.
* **3+ points** -- still fits a straight line, now by least squares, and
  reports how well it fits. More points make the line more trustworthy, not
  more curved.

Quadratic is available with 3 or more points but is not the default, and with
exactly 3 it will pass exactly through them while doing whatever it likes in
between and beyond. Prefer linear unless you have a real reason and enough
points to check it.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np

FitMode = Literal["off", "offset", "linear", "quadratic", "auto"]

#: Frames of history kept per point. At 25 fps this is about a second.
SAMPLE_HISTORY = 25

#: Spatial spread inside a reference box above which the target is too
#: non-uniform to calibrate against reliably. See :meth:`ReferencePoint.spread`.
UNIFORMITY_WARN_C = 1.0


@dataclass
class ReferencePoint:
    """A pixel whose true temperature you know (or will type in shortly).

    The measurement is the median of a small box around the pixel, taken over
    the last :data:`SAMPLE_HISTORY` frames. Both matter: a single pixel on a
    40 mK-noise sensor wanders by a few tenths of a degree, and the median
    shrugs off the shutter recalibration that briefly freezes the stream.
    """

    x: int
    y: int
    known_c: Optional[float] = None
    label: str = ""
    radius: int = 2  # a (2r+1) square box, so 5x5 by default

    #: Spatial spread inside the box on the last frame. See :meth:`spread`.
    last_spread_c: float = 0.0

    _samples: deque = field(default_factory=lambda: deque(maxlen=SAMPLE_HISTORY),
                            repr=False, compare=False)

    # --- Measurement ---------------------------------------------------------

    def sample(self, celsius: np.ndarray) -> float:
        """Mean temperature of the box around this point, this frame."""
        h, w = celsius.shape
        y0, y1 = max(0, self.y - self.radius), min(h, self.y + self.radius + 1)
        x0, x1 = max(0, self.x - self.radius), min(w, self.x + self.radius + 1)
        return float(np.mean(celsius[y0:y1, x0:x1]))

    def spread(self, celsius: np.ndarray) -> float:
        """Spatial standard deviation inside the box.

        This is a uniformity check on the *target*, not sensor noise. Because
        the reading is a box average, aiming at something with a steep gradient
        -- the peak of a hot spot, or an edge -- means the number you get is an
        average over a range of temperatures while the value you type in is for
        one spot. The fit then quietly bakes in that mismatch. A large spread
        here is the warning sign; aim at a flat, uniform patch instead.
        """
        h, w = celsius.shape
        y0, y1 = max(0, self.y - self.radius), min(h, self.y + self.radius + 1)
        x0, x1 = max(0, self.x - self.radius), min(w, self.x + self.radius + 1)
        return float(np.std(celsius[y0:y1, x0:x1]))

    def update(self, celsius: np.ndarray) -> float:
        """Take a sample and add it to the history."""
        value = self.sample(celsius)
        self._samples.append(value)
        self.last_spread_c = self.spread(celsius)
        return value

    @property
    def measured_c(self) -> Optional[float]:
        """Median of the recent samples, or None if nothing sampled yet."""
        if not self._samples:
            return None
        return float(np.median(self._samples))

    @property
    def noise_c(self) -> Optional[float]:
        """Spread of the recent samples -- how settled this point is."""
        if len(self._samples) < 3:
            return None
        return float(np.std(self._samples))

    @property
    def is_usable(self) -> bool:
        return self.known_c is not None and bool(self._samples)

    @property
    def is_uniform(self) -> bool:
        """Whether this point sits on a flat enough patch to trust."""
        return self.last_spread_c <= UNIFORMITY_WARN_C

    def clear_samples(self) -> None:
        self._samples.clear()

    # --- Serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "known_c": self.known_c,
                "label": self.label, "radius": self.radius}

    @classmethod
    def from_dict(cls, data: dict) -> "ReferencePoint":
        return cls(x=int(data["x"]), y=int(data["y"]),
                   known_c=data.get("known_c"), label=data.get("label", ""),
                   radius=int(data.get("radius", 2)))


class Calibration:
    """Fits and applies a correction from a set of reference points."""

    def __init__(self, mode: FitMode = "auto",
                 points: Optional[Iterable[ReferencePoint]] = None) -> None:
        self.mode: FitMode = mode
        self.points: list[ReferencePoint] = list(points or [])
        #: Polynomial coefficients, highest order first (numpy convention).
        #: ``None`` means no correction is being applied.
        self.coeffs: Optional[np.ndarray] = None

    # --- Points --------------------------------------------------------------

    def add_point(self, x: int, y: int, known_c: Optional[float] = None,
                  label: str = "") -> ReferencePoint:
        point = ReferencePoint(x=int(x), y=int(y), known_c=known_c,
                               label=label or f"P{len(self.points) + 1}")
        self.points.append(point)
        return point

    def remove_point(self, point: ReferencePoint) -> None:
        if point in self.points:
            self.points.remove(point)
            self.coeffs = None

    def clear(self) -> None:
        self.points.clear()
        self.coeffs = None

    def nearest(self, x: int, y: int, within: float = 12.0
                ) -> Optional[ReferencePoint]:
        """The reference point nearest to ``(x, y)``, if one is close enough."""
        best, best_d = None, within
        for point in self.points:
            d = float(np.hypot(point.x - x, point.y - y))
            if d <= best_d:
                best, best_d = point, d
        return best

    @property
    def usable_points(self) -> list[ReferencePoint]:
        return [p for p in self.points if p.is_usable]

    def update(self, celsius: np.ndarray) -> None:
        """Sample every point from the current (uncorrected) frame.

        Must be given uncorrected temperatures. Feeding it corrected ones makes
        the fit chase its own output.
        """
        for point in self.points:
            point.update(celsius)

    # --- Fitting -------------------------------------------------------------

    def effective_mode(self) -> str:
        """Which model will actually be fitted, resolving ``auto``."""
        n = len(self.usable_points)
        if self.mode == "off" or n == 0:
            return "off"
        if self.mode != "auto":
            if self.mode == "quadratic" and n < 3:
                return "linear" if n == 2 else "offset"
            if self.mode == "linear" and n < 2:
                return "offset"
            return self.mode
        # auto: the most complex model the data actually supports, capped at
        # linear -- extra points should tighten the line, not bend it.
        return "offset" if n == 1 else "linear"

    def fit(self) -> bool:
        """Fit the correction. Returns True if a correction is now active."""
        mode = self.effective_mode()
        points = self.usable_points
        if mode == "off" or not points:
            self.coeffs = None
            return False

        measured = np.array([p.measured_c for p in points], dtype=np.float64)
        known = np.array([p.known_c for p in points], dtype=np.float64)

        if mode == "offset":
            # Slope pinned at 1; only shift.
            self.coeffs = np.array([1.0, float(np.mean(known - measured))])
        elif mode == "linear":
            if np.ptp(measured) < 1e-6:
                # All references at effectively the same temperature -- a slope
                # would be meaningless, so fall back to a shift.
                self.coeffs = np.array([1.0, float(np.mean(known - measured))])
            else:
                self.coeffs = np.polyfit(measured, known, 1)
        else:  # quadratic
            if len(points) < 3 or np.ptp(measured) < 1e-6:
                return self._fit_as("linear")
            self.coeffs = np.polyfit(measured, known, 2)
        return True

    def _fit_as(self, mode: FitMode) -> bool:
        previous, self.mode = self.mode, mode
        try:
            return self.fit()
        finally:
            self.mode = previous

    @property
    def is_active(self) -> bool:
        return self.coeffs is not None

    # --- Applying ------------------------------------------------------------

    def apply(self, celsius: np.ndarray) -> np.ndarray:
        """Correct a temperature array. A no-op when nothing has been fitted."""
        if self.coeffs is None:
            return celsius
        return np.polyval(self.coeffs, celsius).astype(np.float32)

    def apply_scalar(self, celsius: float) -> float:
        if self.coeffs is None:
            return celsius
        return float(np.polyval(self.coeffs, celsius))

    # --- Quality -------------------------------------------------------------

    def residuals(self) -> list[tuple[ReferencePoint, float, float]]:
        """``(point, corrected_reading, error)`` for each usable point.

        Error is corrected minus known, so positive means reading high.
        """
        out = []
        for point in self.usable_points:
            corrected = self.apply_scalar(point.measured_c)  # type: ignore[arg-type]
            out.append((point, corrected, corrected - point.known_c))
        return out

    @property
    def rms_error_c(self) -> Optional[float]:
        """RMS of the residuals -- how well the model matches your references.

        Note this measures fit quality, not accuracy. A two-point linear fit
        passes exactly through both points and reports zero regardless of how
        good those references were.
        """
        errors = [e for _, _, e in self.residuals()]
        if not errors:
            return None
        return float(np.sqrt(np.mean(np.square(errors))))

    @property
    def pending_points(self) -> list[ReferencePoint]:
        """Points that have been placed but have no known value typed in yet."""
        return [p for p in self.points if p.known_c is None]

    def describe(self) -> str:
        """One-line summary for a HUD."""
        mode = self.effective_mode()
        total, pending = len(self.points), len(self.pending_points)
        # Always mention unfinished points: dropping three markers and filling
        # in two is an easy mistake, and silence about it looks like success.
        note = f"  ({pending} of {total} need a value)" if pending else ""

        if mode == "off" or self.coeffs is None:
            return f"calib: off{note}"

        if len(self.coeffs) == 2:
            gain, offset = self.coeffs
            body = f"x{gain:.4f} {offset:+.2f}C"
        else:
            body = "quadratic " + " ".join(f"{c:+.4g}" for c in self.coeffs)
        rms = self.rms_error_c
        tail = f", rms {rms:.2f}C" if rms is not None else ""
        n = len(self.usable_points)
        return f"calib: {mode} on {n} pt{'s' if n != 1 else ''}, {body}{tail}{note}"

    # --- Serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "points": [p.to_dict() for p in self.points],
            "coeffs": None if self.coeffs is None else list(map(float, self.coeffs)),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        calib = cls(mode=data.get("mode", "auto"),
                    points=[ReferencePoint.from_dict(p)
                            for p in data.get("points", [])])
        coeffs = data.get("coeffs")
        if coeffs:
            calib.coeffs = np.array(coeffs, dtype=np.float64)
        return calib

    def save(self, path: Path | str = "calibration.json") -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str = "calibration.json") -> "Calibration":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Calibration {self.describe()}>"
