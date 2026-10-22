"""Drive the real GUI with fake cameras (Linux + Xvfb, or any desktop).

    xvfb-run -s "-screen 0 1600x1000x24" python tests/gui_smoke.py OUTDIR

Live stream -> VIDEO recording across a chunk boundary (camera + thermal) ->
PHOTO "capture now" -> close. Writes screenshots and a result JSON to OUTDIR.
"""

import glob
import json
import os
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="gui_smoke_"))
os.makedirs(OUT, exist_ok=True)
os.environ["HOME"] = os.path.join(OUT, "home")          # keep settings out of the real ~
os.makedirs(os.environ["HOME"], exist_ok=True)

import test_thermal as tt  # noqa: E402
from thermalcam import camera as tcam  # noqa: E402
from thermalcam.devices import VideoDevice  # noqa: E402
from thermalcam import modes as M  # noqa: E402

# --- fake thermal camera behind the real ThermalCamera ----------------------
tcam.list_video_devices = lambda include_pnp=False: [VideoDevice(0, "Fake Webcam", "t"),
                                                       VideoDevice(2, "USB Camera", "t")]
tcam.list_modes = lambda ref, use_cache=True: tt.mode_list(
    *(["256x392", "4x12305"] if ref.index == 2 else ["640x480", "1280x720"]))
tcam.FFmpegRawCapture = tt.FakeCapture
tcam.shutil_ffmpeg = lambda: True
tcam._platform_backends = lambda: []
tt.FakeCapture.kinds = {"temp-12305": "temp-12305"}

import studio_services as S  # noqa: E402
from test_recording import FakeOpenCVCamera  # noqa: E402

S.CameraService._open = lambda self: FakeOpenCVCamera(*self.config[1:])


class TestPatternRecorder(S.CameraRecorder):
    def __init__(self, req, input_override=None):
        super().__init__(req, input_override=["-re", "-f", "lavfi", "-i",
                                              f"testsrc2=size={req.width}x{req.height}:rate=15"])


import amin_studio as A  # noqa: E402

A.CameraRecorder = TestPatternRecorder
A.AdvancedRecorderApp._scan_devices = lambda self: [
    {"path": "/dev/video0", "name": "Fake Webcam", "thermal": False, "sizes": ["1280x720"],
     "label": "/dev/video0 - Fake Webcam  |  max 1280x720"},
    {"path": "/dev/video2", "name": "USB Camera", "thermal": True, "sizes": ["4x12305"],
     "label": "/dev/video2 - USB Camera  |  THERMAL (4x12305)"},
]

result = {"steps": [], "errors": []}


def shot(app, name):
    try:
        from PIL import ImageGrab
        app.root.update()
        ImageGrab.grab().save(os.path.join(OUT, f"{name}.png"))
    except Exception as exc:
        result["errors"].append(f"screenshot {name}: {exc}")


def main():
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    app = A.AdvancedRecorderApp(root)
    rec_dir = os.path.join(OUT, "recordings")
    app.dest_path_var.set(rec_dir)
    app.res_var.set("1280x720 (720p)")
    app.fps_var.set("15")
    app.preset_var.set("ultrafast")
    app.chunk_var.set("1")
    app.thermal_fps_var.set("10")
    app.thermal_video_var.set("Colour + radiometric")

    def step(name, fn, delay_ms):
        def run():
            try:
                fn()
                result["steps"].append({"step": name, "t": time.time(), "alert": app.alert_label.cget("text"),
                                        "status": app.status_text.cget("text"),
                                        "thermal": app.thermal_status_label.cget("text")})
            except Exception:
                result["errors"].append(f"{name}: {traceback.format_exc()}")
        root.after(delay_ms, run)

    t = 0

    def at(seconds, name, fn):
        nonlocal t
        t += int(seconds * 1000)
        step(name, fn, t)

    at(3, "devices", lambda: result.update(cam=app.cam_combo.get(), th=app.thermal_combo.get()))
    at(0.5, "live on", app.toggle_live_stream)
    at(6, "live shot", lambda: shot(app, "1_live"))
    at(0.5, "record start", app.start_recording)
    at(5, "recording shot", lambda: shot(app, "2_recording"))
    # run across at least one clock-aligned chunk boundary (1-min chunks)
    at(70, "recording late shot", lambda: shot(app, "3_recording_late"))
    at(0.5, "record stop", app.stop_recording)
    at(3, "photo tab", lambda: app.set_mode("photo"))
    at(1, "capture now", app.capture_now)
    at(12, "photo shot", lambda: shot(app, "4_photo"))
    at(1, "close", app.on_close)
    root.mainloop()

    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(rec_dir, "*")))
    result["files"] = files
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
