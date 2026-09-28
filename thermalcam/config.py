"""Persist span and correction settings to JSON.

Keeps machine-specific choices (which camera index, which manual span, what
offset) out of your code, so moving to another PC means editing a small text
file rather than hunting through scripts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .span import TemperatureSpan

DEFAULT_PATH = Path("thermalcam.json")


@dataclass
class Config:
    """Serializable settings bundle."""

    device_index: Optional[int] = None      # None = auto-scan
    span: TemperatureSpan = field(default_factory=TemperatureSpan.auto)
    colormap: str = "inferno"
    scale: int = 3
    offset_c: float = 0.0
    gain: float = 1.0
    swap_bytes: Optional[bool] = None       # None = auto-detect

    # --- I/O -----------------------------------------------------------------

    def save(self, path: Path | str = DEFAULT_PATH) -> Path:
        path = Path(path)
        path.write_text(json.dumps({
            "device_index": self.device_index,
            "span": self.span.to_dict(),
            "colormap": self.colormap,
            "scale": self.scale,
            "offset_c": self.offset_c,
            "gain": self.gain,
            "swap_bytes": self.swap_bytes,
        }, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "Config":
        """Load from disk, or return defaults if the file does not exist."""
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            device_index=data.get("device_index"),
            span=TemperatureSpan.from_dict(data.get("span", {})),
            colormap=data.get("colormap", "inferno"),
            scale=data.get("scale", 3),
            offset_c=data.get("offset_c", 0.0),
            gain=data.get("gain", 1.0),
            swap_bytes=data.get("swap_bytes"),
        )

    # --- Convenience ---------------------------------------------------------

    def open_camera(self):
        """Build and open a :class:`ThermalCamera` from these settings."""
        from .camera import ThermalCamera
        return ThermalCamera(
            index=self.device_index,
            offset_c=self.offset_c,
            gain=self.gain,
            swap_bytes=self.swap_bytes,
        ).open()
