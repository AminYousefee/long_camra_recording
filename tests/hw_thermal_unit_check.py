"""Two-point check of the thermal camera's temperature unit.

Streams the camera's temperature-only mode (4x12305 on the newer TC001) through
FFmpeg for ~45 s and logs the raw values every half second, while you point the
camera at a plain wall and then at your open hand. Results go to
hwtest_output/unit_check.csv and hwtest_output/unit_check_*.npy.
"""
import csv
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "hwtest_output")
os.makedirs(OUT, exist_ok=True)
W, H, HDR = 4, 12305, 136
FRAME = W * H * 2


def banner(text):
    print("\n" + "#" * 64 + f"\n#   {text}\n" + "#" * 64 + "\n", flush=True)


def main():
    ff = shutil.which("ffmpeg")
    kw = {"creationflags": 0x08000000} if sys.platform.startswith("win") else {}
    r = subprocess.run([ff, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                       capture_output=True, text=True, errors="replace", **kw)
    names = re.findall(r'"(.+)" \(video\)', r.stderr)
    thermal = None
    for n in names:
        r = subprocess.run([ff, "-hide_banner", "-f", "dshow", "-list_options", "true", "-i", f"video={n}"],
                           capture_output=True, text=True, errors="replace", **kw)
        if "s=4x12305" in r.stderr:
            thermal = n
            break
    if thermal is None:
        print("no camera with the 4x12305 temperature mode found:", names)
        return
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-f", "dshow", "-rtbufsize", "64M",
           "-video_size", f"{W}x{H}", "-pixel_format", "yuyv422", "-framerate", "25",
           "-i", f"video={thermal}", "-f", "rawvideo", "-pix_fmt", "yuyv422", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0, **kw)
    latest = {"frame": None, "n": 0}

    def reader():
        buf = bytearray()
        while True:
            chunk = proc.stdout.read(FRAME - len(buf))
            if not chunk:
                break
            buf += chunk
            if len(buf) == FRAME:
                latest["frame"] = np.frombuffer(bytes(buf[HDR:]), "<u2").reshape(192, 256)
                latest["n"] += 1
                buf.clear()
    threading.Thread(target=reader, daemon=True).start()

    steps = [(0, "Point the THERMAL camera at a plain WALL (room temperature)"),
             (15, "Now hold your OPEN HAND (palm) ~15 cm in front of the thermal lens"),
             (30, "If you have a mug of HOT water, point at it now (else keep the hand)"),
             (45, "Done - thank you!")]
    rows = []
    t0 = time.time()
    k = 0
    with open(os.path.join(OUT, "unit_check.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "phase", "frames", "min", "p5", "median", "p95", "max", "center"])
        while True:
            t = time.time() - t0
            if k < len(steps) and t >= steps[k][0]:
                banner(steps[k][1])
                k += 1
            if t >= 45:
                break
            fr = latest["frame"]
            if fr is not None:
                v = fr.astype(float)
                row = [f"{t:.1f}", k - 1, latest["n"], int(v.min()), int(np.percentile(v, 5)),
                       int(np.median(v)), int(np.percentile(v, 95)), int(v.max()), int(v[96, 128])]
                w.writerow(row)
                rows.append(row)
                print("  t=%5.1fs  median %5d  p95 %5d  max %5d" % (t, row[5], row[6], row[7]), flush=True)
                if int(t * 2) in (20, 50, 80):
                    np.save(os.path.join(OUT, f"unit_check_t{int(t)}.npy"), fr)
            time.sleep(0.5)
    proc.kill()
    print("saved", os.path.join(OUT, "unit_check.csv"))
    time.sleep(3)


if __name__ == "__main__":
    main()
