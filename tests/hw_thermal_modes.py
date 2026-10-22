"""Hardware survey of the thermal camera's capture modes.

Captures a few raw frames of every mode the thermal camera offers (through
FFmpeg, untouched YUYV bytes) into hwtest_output/modes/ so the layout of each
mode can be analysed. Run from the project folder:
    python tests/hw_thermal_modes.py
"""
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "hwtest_output", "modes")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "modes_log.txt"), "w", encoding="utf-8")


def say(*a):
    t = " ".join(str(x) for x in a)
    print(t, flush=True)
    LOG.write(t + "\n")
    LOG.flush()


def run(cmd, timeout=40):
    kw = {"creationflags": 0x08000000} if sys.platform.startswith("win") else {}
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, **kw)
        return r.returncode, r.stdout, r.stderr.decode(errors="replace")
    except subprocess.TimeoutExpired:
        return -999, b"", "TIMEOUT"


def main():
    ff = shutil.which("ffmpeg")
    say(datetime.now().isoformat(), "ffmpeg", ff)
    if sys.platform.startswith("win"):
        rc, out, err = run([ff, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"])
        names = re.findall(r'"(.+)" \(video\)', err)
        say("devices", names)
        thermal = None
        for n in names:
            rc, out, err = run([ff, "-hide_banner", "-f", "dshow", "-list_options", "true", "-i", f"video={n}"])
            if "s=256x" in err:
                thermal = n
                modes = sorted(set(re.findall(r"pixel_format=(\w+)\s+min s=(\d+x\d+) fps=(\d+)", err)))
                break
        if thermal is None:
            say("no 256-px-wide camera found")
            return
        say("thermal camera:", thermal, "modes:", modes)
        for pix, size, fps in modes:
            path = os.path.join(OUT, f"mode_{size}_{pix}.raw")
            cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "dshow", "-rtbufsize", "64M",
                   "-video_size", size, "-pixel_format", pix, "-framerate", fps,
                   "-i", f"video={thermal}", "-frames:v", "6", "-f", "rawvideo",
                   "-pix_fmt", pix, path]
            t0 = time.time()
            rc, out, err = run(cmd)
            sz = os.path.getsize(path) if os.path.exists(path) else 0
            say(f"{size:>10s} {pix}: rc={rc} {time.time() - t0:4.1f}s  {sz} bytes  {err.strip()[:200]}")
            time.sleep(0.7)
    else:
        say("Linux/mac: use v4l2-ctl --list-formats-ext / ffmpeg -f avfoundation -list_devices")

    # Several consecutive MSMF frames of the default mode (to see what changes)
    import cv2
    import numpy as np
    from thermalcam.devices import list_video_devices
    idx = next((d.index for d in list_video_devices() if d.name == thermal), 0)
    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    frames = []
    t0 = time.time()
    while len(frames) < 50 and time.time() - t0 < 15:
        ok, f = cap.read()
        if ok and f is not None:
            frames.append(f.copy())
    cap.release()
    say(f"MSMF default mode: {len(frames)} frames in {time.time() - t0:.1f}s, shape "
        f"{frames[0].shape if frames else None}")
    if frames:
        np.save(os.path.join(OUT, "msmf_default_50frames.npy"), np.stack(frames))
    say("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        say(traceback.format_exc())
    LOG.close()
