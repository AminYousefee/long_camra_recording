"""A stand-in for the TC001: writes raw frames to stdout like FFmpeg would.

    python tests/fake_thermal_stream.py temp-12305 [--fps 25] [--die-after 5]
    python tests/fake_thermal_stream.py gray-392
    python tests/fake_thermal_stream.py classic-384
    python tests/fake_thermal_stream.py picture-384     (the 240 C trap)
    python tests/fake_thermal_stream.py replay --file frames.raw --size 4x12305

The synthetic scene is a 22 C background with a 34 C "hand" that moves, so
tests can check both absolute values and that frames change.
"""

import argparse
import sys
import time

import numpy as np

W, H = 256, 192
DIVISOR = 16.0          # must match thermalcam.layouts.TEMP_MODE_DIVISOR in tests


def scene(t: float) -> np.ndarray:
    """Celsius, (192, 256)."""
    y, x = np.mgrid[0:H, 0:W]
    c = np.full((H, W), 22.0) + 0.01 * y           # gentle gradient
    cx = 128 + 60 * np.sin(t)
    c[(x - cx) ** 2 + (y - 96) ** 2 < 30 ** 2] = 34.0
    c += np.random.default_rng(int(t * 1000)).normal(0, 0.05, c.shape)
    return c


def picture(c: np.ndarray) -> np.ndarray:
    """The camera's 8-bit YUYV picture: luma = contrast-stretched, chroma 128."""
    lo, hi = c.min(), c.max()
    yv = np.clip((c - lo) / max(hi - lo, 1e-3) * 255, 0, 255).astype(np.uint8)
    out = np.empty((H, W, 2), np.uint8)
    out[..., 0] = yv
    out[..., 1] = 128
    return out


def u16_pairs(v: np.ndarray) -> np.ndarray:
    v = v.astype("<u2")
    return v.view(np.uint8).reshape(v.shape[0], v.shape[1], 2)


def frame(kind: str, t: float) -> bytes:
    c = scene(t)
    if kind == "classic-384":
        k64 = np.rint((c + 273.15) * 64).astype(np.uint16)
        return np.concatenate([picture(c), u16_pairs(k64)]).tobytes()
    if kind == "picture-384":
        return np.concatenate([picture(c), picture(c)]).tobytes()
    if kind == "temp-12305":
        header = np.zeros(68, "<u2")
        header[:2] = (30579, 28802)
        header[-2:] = (52445, 43707)
        temps = np.rint((c + 273.15) * DIVISOR).astype("<u2")
        return header.tobytes() + temps.tobytes()
    if kind == "gray-392":
        temps = (c + 273.15) * DIVISOR
        gray = np.rint(2400 + (c - 22.0) * 1.7).astype(np.uint16)  # ~1.7 counts/K
        info1 = np.zeros((4, W, 2), np.uint8)
        info2 = np.zeros((4, W), "<u2")
        info2[0, :10] = (0xFFFF, 0xEEEE, int(temps.max()), 0, 0, int(temps.min()), 0, 0,
                         int(temps.mean()), int(temps[96, 128]))
        info2 = info2.view(np.uint8).reshape(4, W, 2)
        return np.concatenate([u16_pairs(gray), info1, picture(c), info2]).tobytes()
    raise SystemExit(f"unknown kind {kind}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--die-after", type=float, default=0.0)
    ap.add_argument("--file")
    ap.add_argument("--size")
    a = ap.parse_args()
    out = sys.stdout.buffer
    t0 = time.time()
    replay = None
    if a.kind == "replay":
        w, h = (int(v) for v in a.size.split("x"))
        data = np.fromfile(a.file, np.uint8)
        replay = data.reshape(-1, w * h * 2)
    n = 0
    try:
        while True:
            t = time.time() - t0
            if a.die_after and t > a.die_after:
                sys.stderr.write("fake camera: simulated unplug\n")
                return 1
            data = replay[n % len(replay)].tobytes() if replay is not None else frame(a.kind, t)
            out.write(data)
            out.flush()
            n += 1
            time.sleep(max(0.0, t0 + n / a.fps - time.time()))
    except (BrokenPipeError, OSError):
        return 0


if __name__ == "__main__":
    sys.exit(main())
