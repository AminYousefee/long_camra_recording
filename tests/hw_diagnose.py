"""One-off hardware diagnosis: what do the cameras, FFmpeg and the encoders
actually do on this machine?  Writes everything to hwtest_output/diag_*.txt
and the raw thermal frames to hwtest_output/*.npy.

Run from the project folder:   python tests/hw_diagnose.py
"""
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "hwtest_output")
os.makedirs(OUT, exist_ok=True)
STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
REPORT = open(os.path.join(OUT, f"diag_{STAMP}.txt"), "w", encoding="utf-8")


def say(*a):
    text = " ".join(str(x) for x in a)
    print(text, flush=True)
    REPORT.write(text + "\n")
    REPORT.flush()


def run(cmd, timeout=30):
    kw = {}
    if sys.platform.startswith("win"):
        kw["creationflags"] = 0x08000000
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           errors="replace", **kw)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as exc:
        return -999, f"TIMEOUT after {timeout}s: {exc}"
    except Exception as exc:
        return -998, repr(exc)


def section(t):
    say("")
    say("=" * 70)
    say(t)
    say("=" * 70)


def main():
    import numpy as np
    import cv2
    section("ENVIRONMENT")
    say("python", sys.version.replace("\n", " "))
    say("platform", sys.platform)
    say("opencv", cv2.__version__, " numpy", np.__version__)
    ff = shutil.which("ffmpeg")
    say("ffmpeg", ff)
    if ff:
        rc, out = run([ff, "-hide_banner", "-version"])
        say(out.splitlines()[0] if out else out)
        rc, out = run([ff, "-hide_banner", "-encoders"])
        encs = [l.split()[1] for l in out.splitlines() if re.match(r"^\s*V\S*\s+\S", l)]
        say("h264/hevc encoders:", [e for e in encs if "264" in e or "265" in e or "hevc" in e])

    section("DEVICES (thermalcam.devices)")
    from thermalcam.devices import list_video_devices
    t0 = time.time()
    devs = list_video_devices()
    say(f"list_video_devices took {time.time() - t0:.1f}s")
    for d in devs:
        say("  ", d, "| source", d.source, "| looks_thermal", d.looks_thermal)

    names = []
    if sys.platform.startswith("win") and ff:
        section("FFMPEG DSHOW DEVICES + MODES")
        rc, out = run([ff, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"])
        for line in out.splitlines():
            m = re.search(r'"(.+)" \(video\)', line)
            if m:
                names.append(m.group(1))
        say("video devices:", names)
        seen = {}
        for name in names:
            n = seen.get(name, 0)
            seen[name] = n + 1
            cmd = [ff, "-hide_banner", "-f", "dshow", "-list_options", "true"]
            if n:
                cmd += ["-video_device_number", str(n)]
            cmd += ["-i", f"video={name}"]
            rc, out = run(cmd)
            say(f"--- {name} #{n}")
            modes = sorted({re.sub(r"\s+", " ", l.split("]", 1)[-1].strip())
                            for l in out.splitlines() if "pixel_format" in l or "vcodec" in l})
            for m in modes[:60]:
                say("    ", m)

    section("ENCODER SMOKE TEST (2 s of 1920x1080@30 test pattern)")
    if ff:
        for enc, extra in (("libx264", ["-preset", "veryfast", "-crf", "23"]),
                           ("h264_qsv", ["-preset", "veryfast", "-global_quality", "23", "-look_ahead", "0"]),
                           ("h264_mf", ["-rate_control", "quality", "-quality", "60"]),
                           ("h264_nvenc", []), ("h264_amf", [])):
            cmd = [ff, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                   "testsrc2=size=1920x1080:rate=30", "-t", "2", "-vf", "format=nv12",
                   "-c:v", enc, *extra, "-f", "null", "-"]
            t0 = time.time()
            rc, out = run(cmd, timeout=60)
            say(f"  {enc:12s} rc={rc}  {time.time() - t0:5.1f}s  {out.strip()[:300]}")

    section("THERMAL CAMERA: RAW LAYOUT PER BACKEND")
    from thermalcam.capture_config import STRATEGIES
    from thermalcam.ffmpeg_capture import FFmpegRawCapture
    from thermalcam import decode

    def analyse(tag, frame):
        say(f"  [{tag}] shape={frame.shape} dtype={frame.dtype}")
        np.save(os.path.join(OUT, f"thermal_raw_{tag}_{STAMP}.npy"), frame)
        try:
            buf = decode.normalize_buffer(frame)
        except Exception as exc:
            say("     normalize_buffer:", exc)
            return
        top, bottom = buf[:192], buf[192:384]
        for name, half in (("top", top), ("bottom", bottom)):
            b0, b1 = half[..., 0].astype(float), half[..., 1].astype(float)
            raw = decode.raw_from_thermal(half)
            c = decode.raw_to_celsius(raw)
            say(f"     {name:6s} byte0 mean {b0.mean():6.1f} sd {b0.std():5.1f} | byte1 mean {b1.mean():6.1f}"
                f" sd {b1.std():5.1f} | as LE temp {c.min():7.1f}..{c.max():7.1f} median {np.median(c):7.1f}")
        # correlation of visual-Y with thermal, normal and flipped
        for name, vis, th in (("top=vis", top, bottom), ("bottom=vis", bottom, top)):
            y = vis[..., 0].astype(float)
            t = decode.raw_to_celsius(decode.raw_from_thermal(th)).astype(float)
            for flip in (False, True):
                tt = t[::-1] if flip else t
                cc = np.corrcoef(y.ravel(), tt.ravel())[0, 1]
                say(f"     corr({name}, other{' flipped' if flip else ''}) = {cc:+.3f}")

    thermal_idx = [d.index for d in devs if d.index is not None]
    # try every addressable index; the thermal one is recognised by 256 px width
    backends = []
    if sys.platform.startswith("win"):
        backends = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)]
    elif sys.platform == "darwin":
        backends = []
    else:
        backends = [("V4L2", cv2.CAP_V4L2)]
    for idx in thermal_idx:
        for bname, be in backends:
            for strat in STRATEGIES[:2]:
                cap = cv2.VideoCapture(idx, be)
                if not cap.isOpened():
                    say(f"index {idx} {bname}: does not open")
                    cap.release()
                    break
                strat.apply(cap, 256, 384, 25)
                frame = None
                t0 = time.time()
                for _ in range(15):
                    ok, f = cap.read()
                    if ok and f is not None:
                        frame = f
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap.release()
                time.sleep(0.5)
                say(f"index {idx} {bname} {strat.name}: props {w:.0f}x{h:.0f}, "
                    f"frame {None if frame is None else frame.shape}, {time.time() - t0:.1f}s")
                if frame is None:
                    continue
                if frame.ndim == 3 and frame.shape[2] == 3 and frame.shape[1] != 256:
                    say("   (ordinary camera)")
                    break
                analyse(f"idx{idx}_{bname}_{strat.name}", frame)
        if ff:
            cap = FFmpegRawCapture(idx)
            if cap.isOpened():
                frame = None
                for _ in range(10):
                    ok, f = cap.read()
                    if ok:
                        frame = f
                cap.release()
                say(f"index {idx} FFMPEG raw: frame {None if frame is None else frame.shape}")
                if frame is not None:
                    analyse(f"idx{idx}_FFMPEG", frame)
            else:
                say(f"index {idx} FFMPEG raw: {cap.error}")
            time.sleep(0.5)

    section("THERMAL CAMERA: ThermalCamera().open() as the app does it")
    from thermalcam import ThermalCamera
    t0 = time.time()
    cam = ThermalCamera()
    try:
        cam.open()
        say(f"opened in {time.time() - t0:.1f}s: {cam.info()}")
        for _ in range(5):
            fr = cam.read()
        s = fr.stats()
        say("stats:", {k: round(v, 2) for k, v in s.items()})
        cam.close()
    except Exception as exc:
        say(f"FAILED after {time.time() - t0:.1f}s: {exc}")
    for a in cam.attempts:
        say("   attempt:", a)

    say("")
    say("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        say(traceback.format_exc())
    REPORT.close()
