"""Hardware self-test: real cameras, real recording, real files.

Close Amin Studio first (it would hold the cameras), then run from the
project folder:

    python tests/hw_selftest.py                 # ~2.5 min
    python tests/hw_selftest.py --quick         # ~1 min, no chunk-boundary test

Windows: double-click tests\\run_selftest.cmd instead.

It uses the same code as the app (thermalcam + studio_services):
  1. finds the cameras by their modes (nothing is opened to identify them)
  2. opens the thermal camera, checks fps and that the values are temperatures
  3. takes two photo pairs (camera JPG + thermal PNG + NPY)
  4. records camera + thermal across a chunk boundary, kills the camera FFmpeg
     half-way to check it is restarted, then checks every file with ffprobe
Everything lands in hwtest_output/selftest_<time>/ with report.txt.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import numpy as np  # noqa: E402

import studio_services as S  # noqa: E402
from thermalcam import layouts as L  # noqa: E402
from thermalcam import modes as M  # noqa: E402
from thermalcam.devices import list_video_devices  # noqa: E402

STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT = os.path.join(ROOT, "hwtest_output", f"selftest_{STAMP}")
os.makedirs(OUT, exist_ok=True)
REPORT = open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8")
RESULTS = []


def say(*a):
    text = " ".join(str(x) for x in a)
    print(text, flush=True)
    REPORT.write(text + "\n")
    REPORT.flush()


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    say(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return ok


def ffprobe(path):
    exe = shutil.which("ffprobe")
    if not exe:
        return {}
    kw = {"creationflags": 0x08000000} if sys.platform.startswith("win") else {}
    out = subprocess.run([exe, "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames",
                          "-show_entries", "format=duration", "-of", "json", path],
                         capture_output=True, text=True, **kw)
    try:
        return json.loads(out.stdout or "{}")
    except ValueError:
        return {}


def settings():
    path = os.path.expanduser("~/.config/amin_studio/settings.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def find_cameras():
    say("\n=== 1. Cameras ===")
    t0 = time.time()
    devices = [d for d in list_video_devices(include_pnp=False) if d.index is not None]
    refs = {r.index: r for r in M.device_refs(devices)}
    found = []
    for d in devices:
        ref = refs[d.index]
        if sys.platform.startswith("linux"):
            ref = M.DeviceRef(index=d.index, name=d.name, path=f"/dev/video{d.index}")
        modes = M.list_modes(ref, use_cache=False)
        sizes = sorted({m.size for m in modes})
        thermal = L.is_thermal_mode_list(sizes)
        say(f"  index {d.index}: {d.name!r}  thermal={thermal}  modes: {', '.join(sizes[:10])}")
        found.append((d, ref, modes, thermal))
    check("camera list", found, f"{len(found)} camera(s) in {time.time() - t0:.1f} s")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seconds", type=int, default=0)
    args = ap.parse_args()
    S.setup_logging(os.path.join(OUT, "logs"))
    say(f"Amin Studio hardware self-test  {STAMP}  ({sys.platform}, python {sys.version.split()[0]})")
    say(f"ffmpeg: {shutil.which('ffmpeg')}")
    cfg = settings()
    ui = cfg.get("ui", {})
    if cfg.get("thermal_temp_divisor"):
        L.set_temp_mode_divisor(float(cfg["thermal_temp_divisor"]))

    cams = find_cameras()
    thermal_dev = next((c for c in cams if c[3]), None)
    normals = [c for c in cams if not c[3]]
    wanted = str(ui.get("camera_path", ""))
    normal = next((c for c in normals if str(c[0].index) == wanted or
                   f"/dev/video{c[0].index}" == wanted), normals[0] if normals else None)
    check("thermal camera identified", thermal_dev is not None,
          thermal_dev[0].name if thermal_dev else "none offers 256-px modes")
    check("normal camera available", normal is not None, normal[0].name if normal else "")

    # --- thermal --------------------------------------------------------------
    say("\n=== 2. Thermal camera ===")
    pipeline = S.ThermalPipeline()
    thermal = S.ThermalService(pipeline)
    avoid = [normal[0].index] if normal else []
    t0 = time.time()
    thermal.acquire("selftest", None, avoid=avoid)
    while time.time() - t0 < 45 and thermal.status != "ONLINE":
        time.sleep(0.2)
    opened = thermal.status == "ONLINE"
    check("thermal camera opens", opened,
          f"{time.time() - t0:.1f} s, {thermal.info}" if opened else thermal.error)
    if opened:
        frames = thermal.collect(50, timeout=10)
        check("thermal frame rate", len(frames) >= 40, f"{len(frames)} frames in <=10 s, {thermal.fps:.1f} fps")
        if frames:
            c = np.stack(frames)
            lo, hi, mean = float(c.min()), float(c.max()), float(c.mean())
            noise = float(np.median(c.std(axis=0)))
            say(f"  scene: min {lo:.2f}  max {hi:.2f}  mean {mean:.2f} C   temporal noise {noise:.3f} K")
            check("values are temperatures, not the 8-bit picture", not (238 < mean < 244 and hi - lo < 5),
                  f"mean {mean:.1f} C")
            check("values in a plausible range", -30 < lo and hi < 600 and -20 < mean < 100)
            check("layout", True, f"{thermal.info.get('layout')} via {thermal.info.get('backend')}"
                  + (" (APPROXIMATE)" if thermal.info.get("approximate") else ""))
            np.save(os.path.join(OUT, "thermal_10frames.npy"), c[:10])

    # --- photos -----------------------------------------------------------------
    say("\n=== 3. Photo pairs ===")
    camera = S.CameraService()
    if normal:
        path = f"/dev/video{normal[0].index}" if sys.platform.startswith("linux") else str(normal[0].index)
        w, h = (int(v) for v in (ui.get("photo_res") or "1920x1080").split()[0].split("x"))
        camera.acquire("selftest", path, w, h)
    photo_dir = os.path.join(OUT, "photos")
    ps = S.PhotoSettings(out_dir=photo_dir, prefix="frame", day_folders=False,
                         use_camera=normal is not None, use_thermal=opened, image_ext="jpg",
                         jpeg_quality=95, thermal_image=True, thermal_bar=True,
                         thermal_data="npy", thermal_average=5)
    cap = S.PhotoCapturer(camera, thermal, pipeline)
    for i in range(2):
        t0 = time.time()
        res = cap.capture(datetime.now().replace(microsecond=0), ps)
        check(f"photo pair {i + 1}", not res["errors"] and res["files"],
              f"{len(res['files'])} files in {time.time() - t0:.1f} s {res['errors']}")
        if res.get("camera") is not None:
            say(f"    camera image {res['camera'].shape[1]}x{res['camera'].shape[0]}")
        time.sleep(1.1)
    camera.shutdown()
    time.sleep(1.0)            # Windows releases the camera asynchronously

    # --- video ------------------------------------------------------------------
    say("\n=== 4. Video recording (camera + thermal) ===")
    rec_dir = os.path.join(OUT, "video")
    seconds = args.seconds or (35 if args.quick else 80)
    cam_rec = None
    if normal:
        res = (ui.get("res") or "1920x1080").split()[0]
        w, h = (int(v) for v in res.split("x"))
        codec = (ui.get("codec") or "libx264").split()[0]
        req = S.RecordingRequest(device=normal[1], width=w, height=h, fps=float(ui.get("fps") or 30),
                                 codec=codec, preset=ui.get("preset") or "veryfast",
                                 crf=str(int(float(ui.get("crf") or 23))), chunk_sec=60,
                                 out_dir=rec_dir, prefix="selftest",
                                 preview_path=os.path.join(OUT, "preview.jpg"), preview_every_s=1.0,
                                 log_dir=os.path.join(OUT, "logs"))
        cam_rec = S.CameraRecorder(req)
        t0 = time.time()
        cam_rec.plan()
        say(f"  plan ({time.time() - t0:.1f} s): {cam_rec.describe()}  {cam_rec.take_notice()}")
        check("camera ffmpeg starts", cam_rec.start(), cam_rec.last_error)
    th_rec = None
    if opened:
        th_rec = S.ThermalVideoRecorder(thermal, pipeline, out_dir=rec_dir, prefix="selftest",
                                        fps=25, chunk_sec=60, outputs=("color", "raw"), crf="23",
                                        preset="veryfast", overlay=True, log_dir=os.path.join(OUT, "logs"))
        th_rec.start()
    events = []
    t0 = time.time()
    killed = False
    while time.time() - t0 < seconds:
        if cam_rec is not None:
            if not killed and time.time() - t0 > seconds * 0.45 and cam_rec.proc is not None:
                say("  ... killing the camera FFmpeg on purpose (crash-recovery test)")
                cam_rec.proc.kill()
                killed = True
            ev = cam_rec.poll()
            if ev:
                events.append((round(time.time() - t0, 1), ev))
                say(f"  t={time.time() - t0:5.1f}s camera recorder: {ev}  {cam_rec.last_error[-160:] if ev == 'exited' else ''}")
        time.sleep(0.5)
    if cam_rec is not None:
        cam_rec.stop()
        check("camera ffmpeg restarted after the crash", ("restarted" in [e for _, e in events]) or not killed,
              str(events))
        unplanned = [e for e in events if e[1] == "exited"]
        check("no unplanned camera ffmpeg exits", len(unplanned) <= 1, str(unplanned))
    if th_rec is not None:
        th_rec.stop()
    thermal.shutdown()

    say("\n=== 5. Files ===")
    cams_mkv = sorted(glob.glob(os.path.join(rec_dir, "selftest_*_camera.mkv")))
    if cam_rec is not None:
        total = 0
        for p in cams_mkv:
            info = ffprobe(p)
            st = (info.get("streams") or [{}])[0]
            n = int(st.get("nb_read_frames") or 0)
            total += n
            say(f"  {os.path.basename(p)}: {st.get('codec_name')} {st.get('pix_fmt')} "
                f"{st.get('width')}x{st.get('height')} {n} frames, "
                f"{float(info.get('format', {}).get('duration') or 0):.1f} s")
            check(f"camera chunk plays ({os.path.basename(p)})", n > 0 and st.get("pix_fmt") == "yuv420p")
        fps = cam_rec._fps()
        check("camera frames recorded", total > 0.6 * fps * (seconds - 8),
              f"{total} frames, expected ~{fps * seconds:.0f} minus the restart gap")
        if not args.quick:
            check("camera chunk cut on the minute", len(cams_mkv) >= 2, f"{len(cams_mkv)} chunks")
    if th_rec is not None:
        raws = sorted(glob.glob(os.path.join(rec_dir, "selftest_*_thermal_raw.mkv")))
        csvs = sorted(glob.glob(os.path.join(rec_dir, "selftest_*_thermal_frames.csv")))
        colors = sorted(glob.glob(os.path.join(rec_dir, "selftest_*_thermal.mkv")))
        for raw, col, fc in zip(raws, colors, csvs):
            with open(fc) as f:
                rows = list(csv.reader(f))[1:]
            nr = int(((ffprobe(raw).get("streams") or [{}])[0]).get("nb_read_frames") or 0)
            nc = int(((ffprobe(col).get("streams") or [{}])[0]).get("nb_read_frames") or 0)
            valid = sum(1 for r in rows if r[-1] == "1")
            check(f"thermal chunk {os.path.basename(raw)}", nr == len(rows) == nc and nr > 0,
                  f"raw {nr}, colour {nc}, csv {len(rows)} rows ({valid} valid)")
        if raws:
            exe = shutil.which("ffmpeg")
            kw = {"creationflags": 0x08000000} if sys.platform.startswith("win") else {}
            data = subprocess.run([exe, "-v", "error", "-i", raws[-1], "-frames:v", "5", "-f", "rawvideo",
                                   "-pix_fmt", "gray16le", "-"], capture_output=True, **kw).stdout
            c = np.frombuffer(data, "<u2").reshape(-1, 192, 256) / 64 - 273.15
            check("radiometric video holds temperatures", -20 < float(np.median(c)) < 100,
                  f"median {float(np.median(c)):.2f} C")
        if not args.quick:
            check("thermal chunk cut on the minute", len(raws) >= 2, f"{len(raws)} chunks")

    failed = [r for r in RESULTS if not r[1]]
    say("\n" + "=" * 60)
    say(f"RESULT: {len(RESULTS) - len(failed)} passed, {len(failed)} failed")
    for name, _, detail in failed:
        say(f"  FAILED: {name}  {detail}")
    say(f"Files: {OUT}")
    return 0 if not failed else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        say(traceback.format_exc())
        code = 2
    REPORT.close()
    sys.exit(code)
