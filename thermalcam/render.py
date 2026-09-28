"""Rendering helpers: false colour, scale bar, markers.

Nothing in the capture path depends on this module -- it exists so you can
eyeball a frame or build a viewer without writing your own colormap code. If
you are running headless, ignore it.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .frame import ThermalFrame
from .span import TemperatureSpan

#: Colormap name -> OpenCV constant. ``"gray"`` means no colormap.
COLORMAPS: dict[str, Optional[int]] = {
    "inferno": cv2.COLORMAP_INFERNO,
    "gray": None,
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "bone": cv2.COLORMAP_BONE,
    "rainbow": cv2.COLORMAP_RAINBOW,
    "turbo": cv2.COLORMAP_TURBO,
}

COLORMAP_NAMES = list(COLORMAPS)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _apply_colormap(gray: np.ndarray, colormap: str) -> np.ndarray:
    if colormap not in COLORMAPS:
        raise ValueError(f"Unknown colormap {colormap!r}. "
                         f"Available: {', '.join(COLORMAPS)}")
    cmap = COLORMAPS[colormap]
    if cmap is None:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(gray, cmap)


def label(image: np.ndarray, text: str, org: tuple[int, int],
          scale: float = 0.4, color: tuple[int, int, int] = (255, 255, 255),
          thickness: int = 1) -> None:
    """Draw text with a dark outline so it stays readable on any colormap."""
    cv2.putText(image, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def colorize_celsius(celsius: np.ndarray, span: TemperatureSpan,
                     colormap: str = "inferno", scale: int = 1,
                     interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Render a Celsius array to a BGR image using ``span`` for the levels.

    The span is applied before upscaling, so with a manual span a given
    temperature always produces exactly the same pixel value.
    """
    bgr = _apply_colormap(span.normalize(celsius), colormap)
    if scale != 1:
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (w * scale, h * scale), interpolation=interpolation)
    return bgr


def colorize(frame: ThermalFrame, span: TemperatureSpan,
             colormap: str = "inferno", scale: int = 1,
             interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Render a :class:`ThermalFrame`. See :func:`colorize_celsius`."""
    return colorize_celsius(frame.celsius, span, colormap, scale, interpolation)


def colorbar(span: TemperatureSpan, height: int, width: int = 26,
             colormap: str = "inferno", celsius: Optional[np.ndarray] = None,
             ticks: int = 7, label_width: int = 54,
             markers: Optional[list[tuple[float, str]]] = None) -> np.ndarray:
    """A vertical temperature scale, hot at the top, labelled in Celsius.

    ``markers`` draws a tick for specific temperatures (used by the viewer to
    show where the calibration references sit on the scale). For a non-manual
    span, pass the frame's ``celsius`` so the bounds can be resolved.
    """
    reference = celsius if celsius is not None else np.zeros((1, 1), np.float32)
    lo, hi = span.bounds(reference)

    ramp = np.repeat(np.linspace(255, 0, height, dtype=np.uint8).reshape(-1, 1),
                     width, axis=1)
    bar = _apply_colormap(ramp, colormap)

    canvas = np.zeros((height, width + label_width, 3), np.uint8)
    canvas[:, :width] = bar
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (90, 90, 90), 1)

    def y_for(value: float) -> int:
        frac = (hi - value) / (hi - lo) if hi > lo else 0.0
        return int(round(min(max(frac, 0.0), 1.0) * (height - 1)))

    for i in range(ticks):
        frac = i / (ticks - 1) if ticks > 1 else 0.0
        y = int(frac * (height - 1))
        value = hi - frac * (hi - lo)
        cv2.line(canvas, (width - 5, y), (width - 1, y), (255, 255, 255), 1)
        text_y = min(height - 3, max(9, y + 4))
        label(canvas, f"{value:5.1f}", (width + 3, text_y), scale=0.36)

    for value, text in (markers or []):
        y = y_for(value)
        cv2.line(canvas, (0, y), (width - 1, y), (255, 255, 255), 1)
        cv2.line(canvas, (0, y), (width - 1, y), (0, 0, 0), 1, cv2.LINE_AA)
        label(canvas, text, (2, max(9, y - 3)), scale=0.3)

    return canvas


def draw_marker(image: np.ndarray, x: int, y: int, scale: int = 1,
                color: tuple[int, int, int] = (255, 255, 255),
                text: str = "", selected: bool = False) -> None:
    """Crosshair with an optional caption, in display coordinates."""
    px, py = int(x * scale + scale // 2), int(y * scale + scale // 2)
    size = 9 if not selected else 13
    cv2.drawMarker(image, (px, py), (0, 0, 0), cv2.MARKER_CROSS, size + 2, 3)
    cv2.drawMarker(image, (px, py), color, cv2.MARKER_CROSS, size, 1)
    if selected:
        cv2.circle(image, (px, py), size, color, 1, cv2.LINE_AA)
    if text:
        label(image, text, (px + 8, py - 6), scale=0.38, color=color)


def annotate(image: np.ndarray, frame: ThermalFrame, span: TemperatureSpan,
             scale: int = 1) -> np.ndarray:
    """Draw min/max markers and a stats line. Convenience for simple scripts."""
    out = image.copy()
    stats = frame.stats()
    draw_marker(out, frame.hottest.x, frame.hottest.y, scale, (255, 255, 255))
    draw_marker(out, frame.coldest.x, frame.coldest.y, scale, (255, 160, 60))
    mode = f"{span.vmin:.1f}..{span.vmax:.1f}C" if span.mode == "manual" else span.mode
    label(out, f"min {stats['min_c']:.1f}  max {stats['max_c']:.1f}  "
               f"ctr {stats['center_c']:.1f}  [{mode}]", (6, 16))
    return out
