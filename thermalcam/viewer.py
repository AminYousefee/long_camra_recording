"""Live viewer with reference-point calibration and a temperature scale bar.

    python -m thermalcam.viewer

Click anywhere on the image to drop a reference point, press ``k``, and type the
temperature that spot really is. With two points bracketing your range the
correction is a proper gain-and-offset fit; with one it is a simple shift.
Everything is applied live, and ``s`` saves it so the next run starts calibrated.

Everything here is built on the public API -- :class:`~thermalcam.ThermalCamera`,
:class:`~thermalcam.Calibration`, :class:`~thermalcam.TemperatureSpan` and the
helpers in :mod:`thermalcam.render`. If you want a different interface, this
file is a worked example rather than a dependency.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .calibration import Calibration, ReferencePoint
from .camera import ThermalCamera, ThermalCameraError
from .config import Config
from .render import (COLORMAP_NAMES, colorbar, colorize_celsius, draw_marker,
                     label)
from .span import TemperatureSpan

WINDOW = "thermalcam"
HUD_HEIGHT = 74
PANEL_BG = (26, 26, 28)

HELP_LINES = [
    "left click  add or select a reference point",
    "k           type the true temperature of the selected point",
    "1-9         select point N        x  delete selected     c  clear all",
    "f           cycle fit: auto / offset / linear / quadratic / off",
    "e           TYPE the span min and max exactly",
    "m           manual or auto span   [ ]  shift span   - +  narrow/widen",
    "b           fit span to the calibration points",
    "p           colormap              t  freeze          h  this help",
    "r           refit from current readings",
    "s           save calibration      l  load            q  quit",
]


@dataclass
class _Entry:
    """An in-progress numeric entry in the status bar.

    OpenCV windows have no text fields, so typed input is collected keystroke
    by keystroke. ``target`` says what the number is for once it is committed.
    """

    target: str          # "point" | "span_min" | "span_max"
    prompt: str
    text: str = ""


class Viewer:
    """Owns the window, the interaction state, and the per-frame composition."""

    def __init__(self, camera: ThermalCamera, config: Config,
                 calibration: Calibration, calib_path: Path) -> None:
        self.camera = camera
        self.config = config
        self.calib = calibration
        self.calib_path = calib_path

        self.span = config.span
        self.colormap = config.colormap
        self.scale = max(1, int(config.scale))

        self.selected: Optional[ReferencePoint] = None
        self.entry: Optional[_Entry] = None
        self.pending_span_min: Optional[float] = None
        self.message = ""
        self.message_frames = 0
        self.show_help = False
        self.frozen = False

        self._celsius: Optional[np.ndarray] = None    # corrected, for display
        self._raw_celsius: Optional[np.ndarray] = None  # uncorrected, for fitting
        self._cursor: Optional[tuple[int, int]] = None

    # --- Messaging -----------------------------------------------------------

    def notify(self, text: str, frames: int = 60) -> None:
        self.message = text
        self.message_frames = frames

    # --- Mouse ---------------------------------------------------------------

    def on_mouse(self, event, mx, my, flags, _param) -> None:
        x, y = mx // self.scale, my // self.scale
        if self._raw_celsius is None:
            return
        h, w = self._raw_celsius.shape
        if not (0 <= x < w and 0 <= y < h):
            self._cursor = None
            return

        self._cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            existing = self.calib.nearest(x, y, within=8)
            if existing is not None:
                self.selected = existing
                self.notify(f"selected {existing.label}")
            else:
                self.selected = self.calib.add_point(x, y)
                self.notify(f"added {self.selected.label} at ({x}, {y}) "
                            f"-- press k to set its true temperature")
        elif event == cv2.EVENT_RBUTTONDOWN:
            existing = self.calib.nearest(x, y, within=8)
            if existing is not None:
                self.calib.remove_point(existing)
                if self.selected is existing:
                    self.selected = None
                self.calib.fit()
                self.notify("point removed")

    # --- Keyboard ------------------------------------------------------------

    def handle_key(self, key: int) -> bool:
        """Return False to quit."""
        if key in (-1, 255):
            return True

        if self.entry is not None:
            return self._handle_entry_key(key)

        char = chr(key) if 32 <= key < 127 else ""

        if key in (27, ord("q")):
            return False
        elif char == "k":
            self._begin_point_entry()
        elif char == "e":
            self._begin_span_entry()
        elif char and char.isdigit() and char != "0":
            index = int(char) - 1
            if index < len(self.calib.points):
                self.selected = self.calib.points[index]
                self.notify(f"selected {self.selected.label}")
        elif char == "x":
            if self.selected is not None:
                self.calib.remove_point(self.selected)
                self.selected = None
                self.calib.fit()
                self.notify("point removed")
        elif char == "c":
            self.calib.clear()
            self.selected = None
            self.notify("all reference points cleared")
        elif char == "f":
            self._cycle_fit_mode()
        elif char == "b":
            self._fit_span_to_points()
        elif char == "m":
            if self.span.mode == "manual":
                self.span.set_auto()
                self.notify("span: auto")
            else:
                lo, hi = self.span.bounds(self._display_celsius())
                self.span.set_manual(lo, hi)
                self.notify(f"span: manual {lo:.1f}..{hi:.1f}C")
        elif char == "[":
            self._span_shift(-1.0)
        elif char == "]":
            self._span_shift(+1.0)
        elif char in ("-", "_"):
            self._span_widen(-1.0)
        elif char in ("+", "="):
            self._span_widen(+1.0)
        elif char == "p":
            i = COLORMAP_NAMES.index(self.colormap) if self.colormap in COLORMAP_NAMES else 0
            self.colormap = COLORMAP_NAMES[(i + 1) % len(COLORMAP_NAMES)]
            self.notify(f"colormap: {self.colormap}")
        elif char == "t":
            self.frozen = not self.frozen
            self.notify("frozen" if self.frozen else "running")
        elif char == "h":
            self.show_help = not self.show_help
        elif char == "r":
            self._refit()
        elif char == "s":
            self._save()
        elif char == "l":
            self._load()
        return True

    def _begin_point_entry(self) -> None:
        if self.selected is None:
            self.notify("click a point first, then press k")
            return
        self.entry = _Entry("point",
                            f"true temperature of {self.selected.label}")

    def _begin_span_entry(self) -> None:
        """Type the scale bounds exactly, rather than nudging towards them."""
        lo, hi = self.span.bounds(self._display_celsius())
        self.pending_span_min = None
        self.entry = _Entry("span_min", f"span MINIMUM in C (now {lo:.1f})")

    def _handle_entry_key(self, key: int) -> bool:
        assert self.entry is not None
        char = chr(key) if 32 <= key < 127 else ""
        if key == 27:                                   # escape
            self.entry = None
            self.pending_span_min = None
            self.notify("cancelled")
        elif key in (13, 10):                           # enter
            self._commit_entry()
        elif key in (8, 127):                           # backspace
            self.entry.text = self.entry.text[:-1]
        elif char and (char.isdigit() or char in ".-"):
            self.entry.text += char
        return True

    def _commit_entry(self) -> None:
        entry, self.entry = self.entry, None
        if entry is None:
            return
        text = entry.text.strip()

        if entry.target == "point":
            self._commit_point_value(text)
        elif entry.target == "span_min":
            self._commit_span_min(text)
        else:
            self._commit_span_max(text)

    def _commit_point_value(self, text: str) -> None:
        if self.selected is None:
            return
        if not text:
            self.selected.known_c = None
            self.calib.fit()
            self.notify(f"{self.selected.label}: value cleared")
            return
        try:
            value = float(text)
        except ValueError:
            self.notify(f"'{text}' is not a number")
            return
        self.selected.known_c = value
        self.calib.fit()
        self.notify(f"{self.selected.label} = {value:.2f}C  |  {self.calib.describe()}")

    def _commit_span_min(self, text: str) -> None:
        lo, hi = self.span.bounds(self._display_celsius())
        if text:
            try:
                lo = float(text)
            except ValueError:
                self.notify(f"'{text}' is not a number")
                return
        # An empty entry keeps the current value, so you can change just one end.
        self.pending_span_min = lo
        self.entry = _Entry("span_max", f"span MAXIMUM in C (now {hi:.1f})")

    def _commit_span_max(self, text: str) -> None:
        lo = self.pending_span_min
        self.pending_span_min = None
        if lo is None:
            return
        _, hi = self.span.bounds(self._display_celsius())
        if text:
            try:
                hi = float(text)
            except ValueError:
                self.notify(f"'{text}' is not a number")
                return
        try:
            self.span.set_manual(lo, hi)
        except ValueError as exc:
            self.notify(str(exc))
            return
        self.notify(f"span locked to {lo:.1f}..{hi:.1f}C  "
                    f"(press s to keep it for next time)")

    def _cycle_fit_mode(self) -> None:
        order = ["auto", "offset", "linear", "quadratic", "off"]
        self.calib.mode = order[(order.index(self.calib.mode) + 1) % len(order)]
        self.calib.fit()
        self.notify(self.calib.describe())

    def _fit_span_to_points(self) -> None:
        """Set the manual span to bracket the calibration references."""
        values = [p.known_c for p in self.calib.points if p.known_c is not None]
        if len(values) < 2:
            self.notify("need at least two points with values")
            return
        lo, hi = min(values), max(values)
        pad = max((hi - lo) * 0.25, 1.0)
        self.span.set_manual(lo - pad, hi + pad)
        self.notify(f"span: manual {self.span.vmin:.1f}..{self.span.vmax:.1f}C")

    def _span_shift(self, delta: float) -> None:
        if self.span.mode != "manual":
            lo, hi = self.span.bounds(self._display_celsius())
            self.span.set_manual(lo, hi)
        self.span.shift(delta)
        self.notify(f"span: {self.span.vmin:.1f}..{self.span.vmax:.1f}C")

    def _span_widen(self, delta: float) -> None:
        if self.span.mode != "manual":
            lo, hi = self.span.bounds(self._display_celsius())
            self.span.set_manual(lo, hi)
        self.span.widen(delta)
        self.notify(f"span: {self.span.vmin:.1f}..{self.span.vmax:.1f}C")

    def _save(self) -> None:
        self.calib.save(self.calib_path)
        self.config.span = self.span
        self.config.colormap = self.colormap
        self.config.scale = self.scale
        self.config.save()
        self.notify(f"saved {self.calib_path.name} and thermalcam.json")

    def _load(self) -> None:
        # Deliberately no fit() here. A reloaded point has no sample history
        # yet, so refitting would find nothing usable and wipe the very
        # coefficients we just restored. The saved fit stands until the user
        # explicitly asks for a new one with 'r'.
        self.calib = Calibration.load(self.calib_path)
        self.selected = None
        self.notify(f"loaded {self.calib_path.name}: {self.calib.describe()}")

    def _refit(self) -> None:
        """Re-fit from what the reference points are reading right now."""
        if not self.calib.usable_points:
            self.notify("nothing to fit -- no points with known values")
            return
        self.calib.fit()
        self.notify("refitted: " + self.calib.describe())

    # --- Composition ---------------------------------------------------------

    def _display_celsius(self) -> np.ndarray:
        if self._celsius is None:
            return np.zeros((1, 1), np.float32)
        return self._celsius

    def compose(self) -> np.ndarray:
        celsius = self._display_celsius()
        image = colorize_celsius(celsius, self.span, self.colormap, self.scale)

        # Reference points, with their live corrected reading.
        for i, point in enumerate(self.calib.points, start=1):
            measured = point.measured_c
            shown = self.calib.apply_scalar(measured) if measured is not None else None
            bits = [f"{i}"]
            if shown is not None:
                bits.append(f"{shown:.1f}C")
            if point.known_c is not None:
                bits.append(f"={point.known_c:.1f}")
            if not point.is_uniform:
                bits.append("!")            # target is not flat -- see HUD
                colour = (80, 180, 255)     # amber-ish warning
            elif point.known_c is not None:
                colour = (120, 255, 120)
            else:
                colour = (200, 200, 200)
            draw_marker(image, point.x, point.y, self.scale, colour,
                        " ".join(bits), selected=point is self.selected)

        # Cursor readout, suppressed when it would collide with a marker label.
        if self._cursor is not None:
            cx, cy = self._cursor
            h, w = celsius.shape
            if 0 <= cx < w and 0 <= cy < h and self.calib.nearest(cx, cy, 14) is None:
                label(image, f"{celsius[cy, cx]:.1f}C",
                      (min(cx * self.scale + 10, image.shape[1] - 60),
                       max(14, cy * self.scale - 10)), scale=0.38)

        markers = [(p.known_c, f"{i}") for i, p in
                   enumerate(self.calib.points, start=1) if p.known_c is not None]
        bar = colorbar(self.span, height=image.shape[0], colormap=self.colormap,
                       celsius=celsius, markers=markers)

        body = np.hstack([image, bar])
        return np.vstack([body, self._hud(body.shape[1], celsius)])

    def _hud(self, width: int, celsius: np.ndarray) -> np.ndarray:
        panel = np.full((HUD_HEIGHT, width, 3), PANEL_BG, np.uint8)

        span_text = (f"manual {self.span.vmin:.1f}..{self.span.vmax:.1f}C"
                     if self.span.mode == "manual" else self.span.mode)
        low, high = self.span.clipping(celsius)
        clip = f"  clip {low:.0%}/{high:.0%}" if (low + high) > 0.02 else ""
        label(panel, f"min {celsius.min():6.1f}   max {celsius.max():6.1f}   "
                     f"mean {celsius.mean():6.1f}      span: {span_text}{clip}",
              (8, 17), scale=0.42)

        label(panel, self.calib.describe(), (8, 36), scale=0.42,
              color=(140, 255, 140) if self.calib.is_active else (170, 170, 170))

        rough = [f"{i}" for i, p in enumerate(self.calib.points, start=1)
                 if not p.is_uniform]
        if self.entry is not None:
            label(panel, f"{self.entry.prompt}: {self.entry.text}_   "
                         f"(enter to accept, blank to keep, esc to cancel)",
                  (8, 58), scale=0.44, color=(120, 220, 255))
        elif rough:
            label(panel, f"point {', '.join(rough)} sits on a steep gradient -- "
                         f"aim at a flat, uniform patch or the fit will be off",
                  (8, 58), scale=0.4, color=(80, 180, 255))
        elif self.message_frames > 0:
            self.message_frames -= 1
            label(panel, self.message, (8, 58), scale=0.4, color=(200, 200, 200))
        else:
            label(panel, "h for help   click to add a reference point   "
                         "k to set its true temperature",
                  (8, 58), scale=0.4, color=(130, 130, 130))
        return panel

    def _overlay_help(self, image: np.ndarray) -> np.ndarray:
        pad, line_h = 14, 19
        h = pad * 2 + line_h * len(HELP_LINES)
        w = min(image.shape[1] - 20, 470)
        box = image[10:10 + h, 10:10 + w]
        if box.shape[0] != h or box.shape[1] != w:
            return image
        cv2.addWeighted(box, 0.15, np.zeros_like(box), 0.0, 0, box)
        for i, text in enumerate(HELP_LINES):
            label(image, text, (10 + pad, 10 + pad + line_h * (i + 1) - 6),
                  scale=0.4)
        return image

    # --- Main loop -----------------------------------------------------------

    def run(self) -> int:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        # Only fit at startup if the loaded file had no coefficients of its own.
        if not self.calib.is_active:
            self.calib.fit()

        while True:
            if not self.frozen or self._raw_celsius is None:
                frame = self.camera.read()
                self._raw_celsius = frame.celsius
                # Sample the references from the *uncorrected* frame, then
                # correct for display. Feeding corrected values back into the
                # fit would make it chase its own output.
                self.calib.update(self._raw_celsius)
                self._celsius = self.calib.apply(self._raw_celsius)

            canvas = self.compose()
            if self.show_help:
                canvas = self._overlay_help(canvas)
            cv2.imshow(WINDOW, canvas)

            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if not self.handle_key(cv2.waitKey(1) & 0xFF):
                break

        cv2.destroyAllWindows()
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m thermalcam.viewer",
        description="Live thermal viewer with reference-point calibration.")
    parser.add_argument("-i", "--index", type=int, default=None,
                        help="Camera index (default: from config, else scan).")
    parser.add_argument("-s", "--scale", type=int, default=None,
                        help="Display magnification (default: from config, else 3).")
    parser.add_argument("--calibration", default="calibration.json",
                        help="Calibration file to load and save.")
    parser.add_argument("--span", nargs=2, type=float, metavar=("MIN", "MAX"),
                        default=None,
                        help="Lock the scale to a fixed Celsius range, so a given "
                             "temperature always maps to the same shade.")
    parser.add_argument("--colormap", choices=COLORMAP_NAMES, default=None,
                        help="Palette to start with (default: from config). "
                             "Use 'gray' for a plain greyscale mapping.")
    parser.add_argument("--debug", action="store_true",
                        help="Let OpenCV's own log output through.")
    args = parser.parse_args(argv)

    config = Config.load()
    if args.scale is not None:
        config.scale = args.scale
    if args.colormap is not None:
        config.colormap = args.colormap
    if args.span is not None:
        try:
            config.span = TemperatureSpan.manual(*args.span)
        except ValueError as exc:
            print(f"--span: {exc}")
            return 1

    calib_path = Path(args.calibration)
    calibration = Calibration.load(calib_path)

    index = args.index if args.index is not None else config.device_index
    try:
        camera = ThermalCamera(index=index, verbose=args.debug).open()
    except ThermalCameraError as exc:
        print(f"FAILED\n{exc}")
        print("\nRun  python -m thermalcam.probe  to diagnose.")
        return 1

    print(camera.info())
    if calibration.points:
        print(f"Loaded {calib_path}: {calibration.describe()}")
    print("\nClick the image to add a reference point, then press k to type its "
          "true temperature.\nPress h in the window for the full key list.\n")

    try:
        return Viewer(camera, config, calibration, calib_path).run()
    except ThermalCameraError as exc:
        print(f"\nStream lost: {exc}")
        return 1
    finally:
        camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
