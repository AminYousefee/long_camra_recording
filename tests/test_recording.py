"""Recording and photo capture end to end, without real cameras.

    python -m unittest tests.test_recording -v

* the thermal camera is tests/fake_thermal_stream.py behind the real
  ThermalService / ThermalVideoRecorder / PhotoCapturer code;
* the normal camera is a fake OpenCV capture (photos, preview) and an FFmpeg
  test pattern (video recording), driven through the real CameraRecorder.

Needs FFmpeg on PATH. Takes about two minutes (it records real chunks).
"""

import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import studio_services as S  # noqa: E402
from test_thermal import FakeCapture, PatchedCamera  # noqa: E402
from thermalcam.modes import DeviceRef  # noqa: E402

FFPROBE = shutil.which("ffprobe")


def probe(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames,profile",
                          "-show_entries", "format=duration", "-of", "json", path],
                         capture_output=True, text=True)
    return json.loads(out.stdout or "{}")


class FakeOpenCVCamera:
    """What CameraService needs from cv2.VideoCapture, at 30 fps."""

    def __init__(self, w=1280, h=720):
        self.w, self.h = w, h
        self.t0 = time.time()
        self.n = 0
        self.released = False

    def grab(self):
        due = self.t0 + self.n / 30.0
        time.sleep(max(0.0, due - time.time()))
        self.n += 1
        return not self.released

    def retrieve(self):
        img = np.zeros((self.h, self.w, 3), np.uint8)
        img[:, :, 1] = (self.n * 4) % 256
        return True, img

    def set(self, *a):
        return True

    def get(self, *a):
        return 0.0

    def release(self):
        self.released = True


class ThermalStack(PatchedCamera):
    def setUp(self):
        super().setUp()
        FakeCapture.kinds = {"temp-12305": "temp-12305"}
        self.tmp = tempfile.mkdtemp(prefix="studio_test_")
        S.setup_logging(os.path.join(self.tmp, "logs"))
        self.pipeline = S.ThermalPipeline()
        self.thermal = S.ThermalService(self.pipeline)

    def tearDown(self):
        self.thermal.shutdown()
        super().tearDown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def wait_status(self, want, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.thermal.status == want:
                return True
            time.sleep(0.1)
        return False


class TestThermalService(ThermalStack):
    def test_online_frames_and_reconnect(self):
        FakeCapture.die_after = 4.0              # every "camera" unplugs after 4 s
        self.thermal.acquire("test", None)
        self.assertTrue(self.wait_status("ONLINE"), self.thermal.status_text())
        frames = self.thermal.collect(5, timeout=5)
        self.assertEqual(len(frames), 5)
        self.assertAlmostEqual(float(np.median(frames[-1])), 22.5, delta=1.0)
        self.assertTrue(self.wait_status("RECONNECTING", 10), "unplug not noticed")
        self.assertTrue(self.wait_status("ONLINE", 15), "did not reconnect")

    def test_photo_pair(self):
        cam = S.CameraService()
        cam._open = lambda: FakeOpenCVCamera()
        cam.WARMUP_S = 0.3
        self.thermal.acquire("photo", None)
        cam.acquire("photo", "/dev/video9", 1280, 720)
        settings = S.PhotoSettings(out_dir=self.tmp, prefix="frame", day_folders=False,
                                   use_camera=True, use_thermal=True, image_ext="jpg",
                                   jpeg_quality=90, thermal_image=True, thermal_bar=True,
                                   thermal_data="npy", thermal_average=3)
        cap = S.PhotoCapturer(cam, self.thermal, self.pipeline)
        try:
            for _ in range(3):
                res = cap.capture(datetime.now().replace(microsecond=0), settings)
                self.assertEqual(res["errors"], [])
                time.sleep(1.05)
        finally:
            cam.shutdown()
        jpgs = glob.glob(os.path.join(self.tmp, "frame_*_camera.jpg"))
        pngs = glob.glob(os.path.join(self.tmp, "frame_*_thermal.png"))
        npys = sorted(glob.glob(os.path.join(self.tmp, "frame_*_thermal_sensorC.npy")))
        self.assertEqual((len(jpgs), len(pngs), len(npys)), (3, 3, 3))
        c = np.load(npys[-1])
        self.assertEqual(c.shape, (192, 256))
        self.assertAlmostEqual(float(np.median(c)), 22.5, delta=1.0)
        self.assertAlmostEqual(float(c.max()), 34.0, delta=0.6)
        with open(os.path.join(self.tmp, "capture_log.csv")) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(not r["errors"] for r in rows))


@unittest.skipUnless(FFPROBE, "needs ffmpeg/ffprobe")
class TestThermalRecorder(ThermalStack):
    def test_chunks_frames_and_values(self):
        self.thermal.acquire("record", None)
        self.assertTrue(self.wait_status("ONLINE"))
        rec = S.ThermalVideoRecorder(self.thermal, self.pipeline, out_dir=self.tmp, prefix="t",
                                     fps=10, chunk_sec=10, outputs=("color", "raw"), crf="23",
                                     preset="ultrafast", overlay=True,
                                     log_dir=os.path.join(self.tmp, "logs"))
        rec.start()
        time.sleep(23)
        rec.stop()
        raws = sorted(glob.glob(os.path.join(self.tmp, "t_*_thermal_raw.mkv")))
        colors = sorted(glob.glob(os.path.join(self.tmp, "t_*_thermal.mkv")))
        csvs = sorted(glob.glob(os.path.join(self.tmp, "t_*_thermal_frames.csv")))
        self.assertGreaterEqual(len(raws), 2, raws)
        self.assertEqual(len(raws), len(colors))
        self.assertEqual(len(raws), len(csvs))
        # chunks after the first start on the clock (seconds % 10 == 0)
        for path in raws[1:]:
            stamp = os.path.basename(path).split("_")[2]          # HH-MM-SS
            self.assertEqual(int(stamp.split("-")[2]) % 10, 0, path)
        total = 0
        for raw, color, frames_csv in zip(raws, colors, csvs):
            with open(frames_csv) as f:
                rows = list(csv.reader(f))[1:]
            n_raw = int(probe(raw)["streams"][0]["nb_read_frames"])
            n_col = int(probe(color)["streams"][0]["nb_read_frames"])
            self.assertEqual(n_raw, len(rows), raw)        # CSV row n == video frame n
            self.assertEqual(n_col, len(rows), color)
            total += n_raw
        self.assertGreater(total, 200)                     # ~23 s x 10 fps
        data = subprocess.run(["ffmpeg", "-v", "error", "-i", raws[-1], "-f", "rawvideo",
                               "-pix_fmt", "gray16le", "-"], capture_output=True).stdout
        c = np.frombuffer(data, "<u2").reshape(-1, 192, 256) / 64 - 273.15
        self.assertAlmostEqual(float(np.median(c)), 22.5, delta=1.0)
        self.assertAlmostEqual(float(c[-1].max()), 34.0, delta=0.6)

    def test_colour_video_follows_the_palette(self):
        self.assertEqual(self.pipeline.colormap, "gray")          # default palette
        rec = S.ThermalVideoRecorder(self.thermal, self.pipeline, out_dir=self.tmp, prefix="p",
                                     fps=10, chunk_sec=60, outputs=("color",), crf="23",
                                     preset="ultrafast", overlay=False, log_dir=None)
        raw_c = np.linspace(20, 40, 192 * 256).reshape(192, 256)
        w, h = rec.color_size

        def frame():
            return np.frombuffer(rec._frame_bytes("color", raw_c), np.uint8).reshape(h, w, 3)

        img = frame()
        self.assertTrue((img[..., 0] == img[..., 1]).all() and (img[..., 1] == img[..., 2]).all())
        self.pipeline.colormap = "inferno"                        # changed while recording
        img = frame()
        self.assertFalse((img[..., 0] == img[..., 2]).all())

    def test_broken_encoder_is_throttled(self):
        self.thermal.acquire("record", None)
        self.assertTrue(self.wait_status("ONLINE"))
        rec = S.ThermalVideoRecorder(self.thermal, self.pipeline, out_dir=self.tmp, prefix="b",
                                     fps=10, chunk_sec=60, outputs=("raw",), crf="23",
                                     preset="ultrafast", overlay=False, log_dir=None)
        rec._cmd = lambda kind, base: ["ffmpeg", "-v", "error", "-f", "nosuchformat", "-i", "-",
                                        base + "_thermal_raw.mkv"]
        rec.start()
        time.sleep(12)
        rec.stop()
        # without the back-off this was a new process (and CSV file) per frame
        self.assertLessEqual(rec.restarts, 6, rec.restarts)
        self.assertLessEqual(len(glob.glob(os.path.join(self.tmp, "b_*_frames.csv"))), 7)


@unittest.skipUnless(FFPROBE, "needs ffmpeg/ffprobe")
class TestCameraRecorder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="camrec_test_")
        S.setup_logging(os.path.join(self.tmp, "logs"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def request(self, codec="libx264"):
        return S.RecordingRequest(device=DeviceRef(index=0, name="test", path="/dev/null"),
                                  width=640, height=360, fps=15, codec=codec, preset="ultrafast",
                                  crf="28", chunk_sec=10, out_dir=self.tmp, prefix="cam",
                                  preview_path=os.path.join(self.tmp, "preview.jpg"),
                                  preview_every_s=1.0, log_dir=os.path.join(self.tmp, "logs"))

    TEST_INPUT = ["-re", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=15"]

    def run_for(self, rec, seconds, kill_at=None):
        t0 = time.time()
        events = []
        while time.time() - t0 < seconds:
            if kill_at is not None and time.time() - t0 > kill_at and rec.proc is not None:
                rec.proc.kill()
                kill_at = None
            ev = rec.poll()
            if ev:
                events.append(ev)
            time.sleep(0.25)
        return events

    def test_records_playable_chunks_and_preview(self):
        rec = S.CameraRecorder(self.request(), input_override=self.TEST_INPUT)
        rec.plan()
        self.assertTrue(rec.start())
        self.run_for(rec, 23)
        rec.stop()
        chunks = sorted(glob.glob(os.path.join(self.tmp, "cam_*_camera.mkv")))
        self.assertGreaterEqual(len(chunks), 2, chunks)
        for path in chunks:
            info = probe(path)
            st = info["streams"][0]
            self.assertEqual(st["codec_name"], "h264")
            self.assertEqual(st["pix_fmt"], "yuv420p", "4:2:2 H.264 does not play in most players")
            self.assertGreater(int(st["nb_read_frames"]), 10)
        self.assertTrue(os.path.getsize(os.path.join(self.tmp, "preview.jpg")) > 1000)

    def test_restarts_after_crash(self):
        rec = S.CameraRecorder(self.request(), input_override=self.TEST_INPUT)
        rec.plan()
        rec.start()
        events = self.run_for(rec, 12, kill_at=4)
        rec.stop()
        self.assertIn("exited", events)
        self.assertIn("restarted", events)
        self.assertEqual(rec.restarts, 1)

    def test_gpu_encoder_falls_back(self):
        ok, _ = S.encoder_works("h264_qsv")
        if ok:
            self.skipTest("this machine really has QSV")
        rec = S.CameraRecorder(self.request("h264_qsv"), input_override=self.TEST_INPUT)
        rec.plan()
        self.assertEqual(rec.codec, "libx264")
        self.assertIn("LIBX264", rec.take_notice())

    def test_encoder_failure_at_runtime_switches_to_libx264(self):
        S._encoder_ok["h264_nvenc"] = (True, "")      # pretend the probe passed
        try:
            rec = S.CameraRecorder(self.request("h264_nvenc"), input_override=self.TEST_INPUT)
            rec.plan()
            rec.start()
            events = self.run_for(rec, 6)
            rec.stop()
        finally:
            S._encoder_ok.pop("h264_nvenc", None)
        if "exited" not in events:
            self.skipTest("this machine really has NVENC")
        self.assertEqual(rec.codec, "libx264")
        self.assertIn("restarted", events)

    def test_error_classification(self):
        c = S._classify_ffmpeg_error
        self.assertEqual(c("[dshow @ 0] Could not set video options | video=x: I/O error"), "mode")
        self.assertEqual(c("[h264_qsv @ 0] Error initializing an internal MFX session"), "encoder")
        self.assertEqual(c("[dshow @ 0] Could not run graph (device already in use)"), "device")
        self.assertEqual(c("ioctl(VIDIOC_STREAMON): Device or resource busy"), "device")

    def test_command_uses_real_camera_mode(self):
        from thermalcam.modes import Mode
        rec = S.CameraRecorder(self.request())
        rec.mode = Mode("mjpeg", 1920, 1080, 30.0, True)
        cmd = rec.command()
        self.assertIn("1920x1080", cmd)
        self.assertNotIn("scale=640:360", " ".join(cmd))       # no upscaling of the recording
        rec.use_default_mode = True
        self.assertNotIn("-video_size", rec.command())


if __name__ == "__main__":
    unittest.main(verbosity=2)
