"""The failures seen in the high-voltage room, simulated without cameras.

    python -m unittest tests.test_supervisor -v

* camera + thermal start together, with the same time stamp in the names;
* the camera FFmpeg dies          -> both stop, both restart together;
* the thermal picture is lost     -> the camera stops too, both restart together;
* the camera picture freezes      -> noticed, FFmpeg stopped, a new file started;
* the recording drive is read-only -> nothing started, a clear reason, retries;
* the camera gets a new /dev/videoN after a USB reset -> followed;
* the log keeps working when its drive dies; the retry delay never overflows.

Needs FFmpeg on PATH (records real, short files). Takes about three minutes.
"""

import glob
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import studio_services as S  # noqa: E402
from thermalcam.modes import DeviceRef  # noqa: E402

FFPROBE = shutil.which("ffprobe")


def frame_count(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    try:
        return int(out.split(",")[0])
    except ValueError:
        return 0


def stamps(paths, suffix):
    """{'2026-11-13_15-30-00', ...} from file names like p_<stamp><suffix>."""
    out = set()
    for p in paths:
        name = os.path.basename(p)
        out.add(name[name.index("_") + 1:name.index(suffix)])
    return out


def unpaired(cams, ths, stopped_at):
    """Camera / thermal file names without a partner. A camera chunk that the
    camera opened in the last 2 s before STOP (a clock boundary just before
    the stop) cannot have a thermal partner yet, and is not counted."""
    c, t = stamps(cams, "_camera.mkv"), stamps(ths, "_thermal_raw.mkv")
    late = {x for x in c - t
            if time.mktime(time.strptime(x, "%Y-%m-%d_%H-%M-%S")) >= stopped_at - 2.5}
    return (c - t - late) | (t - c)


class FakeThermal:
    """What the supervisor and ThermalVideoRecorder need from ThermalService:
    status / error / latest(), with a switch to lose the picture."""

    def __init__(self, fps=25.0):
        self.status = "OFF"
        self.error = ""
        self.fps = fps
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._raw = None
        self._ts = 0.0
        self._n = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        base = np.full((192, 256), 22.5)
        while not self._stop.is_set():
            if self.lost.is_set():
                self.status, self.error = "RECONNECTING", "fake camera unplugged"
            else:
                with self._lock:
                    self._raw, self._ts, self._n = base + (self._n % 10) * 0.1, time.time(), self._n + 1
                self.status, self.error = "ONLINE", ""
            time.sleep(1.0 / self.fps)

    def latest(self):
        with self._lock:
            return self._raw, self._ts, self._n

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)


@unittest.skipUnless(FFPROBE, "needs ffmpeg/ffprobe")
class SupervisorBase(unittest.TestCase):
    TEST_INPUT = ["-re", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15"]

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="supervisor_test_")
        S.setup_logging(os.path.join(self.tmp, "logs"))
        self.pipeline = S.ThermalPipeline()
        self.thermal = FakeThermal()
        self.sup = None

    def tearDown(self):
        if self.sup is not None and self.sup.state != self.sup.STOPPED:
            self.sup.stop()
        self.thermal.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def camera(self, chunk_sec=60, input_override=None):
        req = S.RecordingRequest(device=DeviceRef(index=0, name="test", path="/dev/null"),
                                 width=320, height=240, fps=15, codec="libx264", preset="ultrafast",
                                 crf="30", chunk_sec=chunk_sec, out_dir=self.tmp, prefix="p",
                                 preview_path=None, log_dir=os.path.join(self.tmp, "logs"))
        rec = S.CameraRecorder(req, input_override=input_override or self.TEST_INPUT)
        rec.plan()
        return rec

    def make_thermal(self, chunk_sec=60):
        def make():
            return S.ThermalVideoRecorder(self.thermal, self.pipeline, out_dir=self.tmp, prefix="p",
                                          fps=10, chunk_sec=chunk_sec, outputs=("raw",), crf="23",
                                          preset="ultrafast", overlay=False,
                                          log_dir=os.path.join(self.tmp, "logs"))
        return make

    def supervisor(self, camera=True, thermal=True, chunk_sec=60):
        self.sup = S.RecordingSupervisor(
            self.tmp, camera=self.camera(chunk_sec) if camera else None,
            thermal_service=self.thermal if thermal else None,
            make_thermal=self.make_thermal(chunk_sec) if thermal else None)
        return self.sup

    def wait_state(self, want, timeout=30.0, not_=False):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if (self.sup.state == want) != not_:
                return True
            time.sleep(0.1)
        return False

    def events(self):
        out = []
        while not self.sup.events.empty():
            out.append(self.sup.events.get_nowait())
        return out

    def camera_files(self):
        return sorted(glob.glob(os.path.join(self.tmp, "p_*_camera.mkv")))

    def thermal_files(self):
        return sorted(glob.glob(os.path.join(self.tmp, "p_*_thermal_raw.mkv")))


class TestStrictSync(SupervisorBase):
    def test_pair_starts_together_and_camera_crash_restarts_both(self):
        sup = self.supervisor()
        sup.start()
        self.assertTrue(self.wait_state(sup.RECORDING), sup.reason)
        time.sleep(5)
        first_thermal = sup.thermal_rec
        killed_at = time.time()
        sup.camera.proc.kill()                         # "the camera crashed"
        self.assertTrue(self.wait_state(sup.RECORDING, 10, not_=True), "crash not noticed")
        noticed = time.time() - killed_at
        self.assertLess(noticed, 3.0)
        self.assertIsNone(sup.thermal_rec)             # thermal stopped with it
        self.assertIsNot(first_thermal, None)
        self.assertTrue(self.wait_state(sup.RECORDING, 30), sup.reason)
        time.sleep(4)
        stopped_at = time.time()
        sup.stop()
        kinds = [e[0] for e in self.events()]
        self.assertEqual(kinds.count("started"), 1, kinds)
        self.assertIn("interrupted", kinds)
        self.assertIn("resumed", kinds)
        cams, ths = self.camera_files(), self.thermal_files()
        self.assertGreaterEqual(len(cams), 2, cams)
        self.assertGreaterEqual(len(ths), 2, ths)
        # same names: each camera file has its thermal partner
        self.assertEqual(unpaired(cams, ths, stopped_at), set())
        for path in cams[:2] + ths[:2]:
            self.assertGreater(frame_count(path), 15, path)
        # the first thermal file ended when the camera died, not later
        with open(sorted(glob.glob(os.path.join(self.tmp, "p_*_thermal_frames.csv")))[0]) as f:
            last_row_time = float(f.read().strip().splitlines()[-1].split(",")[2])
        self.assertLess(last_row_time, killed_at + 3.0)

    def test_thermal_loss_stops_the_camera_too(self):
        sup = self.supervisor()
        sup.start()
        self.assertTrue(self.wait_state(sup.RECORDING), sup.reason)
        time.sleep(4)
        self.thermal.lost.set()                         # "thermal camera unplugged"
        self.assertTrue(self.wait_state(sup.WAITING, 12), sup.state)
        self.assertIn("THERMAL", sup.reason)
        self.assertIsNone(sup.camera.proc, "camera must stop with the thermal (strict sync)")
        time.sleep(4)
        self.assertEqual(sup.state, sup.WAITING)        # nothing restarts without the thermal
        self.assertIsNone(sup.camera.proc)
        self.thermal.lost.clear()                       # back
        self.assertTrue(self.wait_state(sup.RECORDING, 30), sup.reason)
        time.sleep(4)
        stopped_at = time.time()
        sup.stop()
        cams, ths = self.camera_files(), self.thermal_files()
        self.assertGreaterEqual(len(cams), 2, cams)
        self.assertEqual(unpaired(cams, ths, stopped_at), set())

    def test_clock_boundary_cuts_both_with_the_same_name(self):
        sup = self.supervisor(chunk_sec=10)
        sup.start()
        self.assertTrue(self.wait_state(sup.RECORDING), sup.reason)
        time.sleep(24)                                  # crosses two 10-s boundaries
        stopped_at = time.time()
        sup.stop()
        cams, ths = self.camera_files(), self.thermal_files()
        self.assertGreaterEqual(len(cams), 3, cams)
        self.assertGreaterEqual(len(ths), 3, ths)
        self.assertEqual(unpaired(cams, ths, stopped_at), set())
        self.assertEqual(sup.interruptions, 0)

    def test_thermal_only_and_camera_only(self):
        for cam, th in ((False, True), (True, False)):
            sup = self.supervisor(camera=cam, thermal=th)
            sup.start()
            self.assertTrue(self.wait_state(sup.RECORDING), (cam, th, sup.reason))
            time.sleep(3)
            sup.stop()
        # (2 if a minute boundary fell inside the few seconds)
        self.assertIn(len(self.camera_files()), (1, 2))
        self.assertIn(len(self.thermal_files()), (1, 2))


class TestStorageAndDevices(SupervisorBase):
    def test_unwritable_folder_waits_with_a_clear_reason(self):
        real = S.dir_writable
        S.dir_writable = lambda path: (False, f"Read-only file system ({path})")
        try:
            sup = self.supervisor(thermal=False)
            sup.start()
            time.sleep(3)
            self.assertEqual(sup.state, sup.WAITING)
            self.assertIn("CANNOT BE WRITTEN", sup.reason)
            self.assertIn("Read-only", sup.reason)
            self.assertIsNone(sup.camera.proc)          # no FFmpeg doomed to fail
        finally:
            S.dir_writable = real
        self.assertTrue(self.wait_state(sup.RECORDING, 30), sup.reason)
        sup.stop()
        self.assertIn(len(self.camera_files()), (1, 2))

    def test_dir_writable(self):
        self.assertEqual(S.dir_writable(self.tmp), (True, ""))
        ok, why = S.dir_writable(os.path.join(self.tmp, "gone"))
        self.assertFalse(ok)
        self.assertIn("does not exist", why)
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".amin")], [])

    def test_camera_followed_to_its_new_number_and_unplug_seen(self):
        dev = os.path.join(self.tmp, "dev")
        os.makedirs(dev)
        for n in (3, 5):
            open(os.path.join(dev, f"video{n}"), "w").close()
        link = os.path.join(dev, "usb-AC310_webcam-video-index0")
        os.symlink(os.path.join(dev, "video3"), link)
        req = S.RecordingRequest(device=DeviceRef(index=3, name="AC310", path=os.path.join(dev, "video3")),
                                 width=320, height=240, fps=15, codec="libx264", preset="ultrafast",
                                 crf="30", chunk_sec=60, out_dir=self.tmp, prefix="p")
        rec = S.CameraRecorder(req)
        rec.stable_path = link
        self.assertIsNone(rec.device_problem())
        os.remove(link)                                  # USB reset: gone ...
        self.assertIn("unplugged", rec.device_problem())
        os.symlink(os.path.join(dev, "video5"), link)    # ... back as video5
        self.assertIsNone(rec.device_problem())
        self.assertEqual(rec.req.device.path, os.path.join(dev, "video5"))
        self.assertEqual(rec.req.device.index, 5)
        self.assertIn(os.path.join(dev, "video5"), rec.command())


@unittest.skipUnless(FFPROBE, "needs ffmpeg/ffprobe")
class TestFrozenCamera(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="freeze_test_")
        S.setup_logging(os.path.join(self.tmp, "logs"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "needs a named pipe")
    def test_frozen_picture_is_noticed_and_the_file_is_kept(self):
        fifo = os.path.join(self.tmp, "camera.fifo")
        os.mkfifo(fifo)
        stop = threading.Event()

        def camera():                       # 3 s of pictures, then silence (a USB stall)
            frame = bytes(160 * 120 * 3 // 2)
            with open(fifo, "wb") as f:
                for _ in range(45):
                    f.write(frame)
                    f.flush()
                    time.sleep(1 / 15)
                stop.wait(60)
        threading.Thread(target=camera, daemon=True).start()
        req = S.RecordingRequest(device=DeviceRef(index=0, name="test", path="/dev/null"),
                                 width=160, height=120, fps=15, codec="libx264", preset="ultrafast",
                                 crf="30", chunk_sec=60, out_dir=self.tmp, prefix="f",
                                 log_dir=os.path.join(self.tmp, "logs"))
        rec = S.CameraRecorder(req, input_override=["-f", "rawvideo", "-pix_fmt", "yuv420p",
                                                    "-s", "160x120", "-framerate", "15", "-i", fifo])
        rec.FREEZE_TIMEOUT_S = 3.0
        rec.plan()
        self.assertTrue(rec.start())
        t0 = time.time()
        event = None
        while time.time() - t0 < 30 and event != "exited":
            event = rec.poll()
            time.sleep(0.25)
        stop.set()
        rec.stop()
        self.assertEqual(event, "exited", "freeze not detected")
        self.assertIn("frozen", rec.last_error)
        self.assertLess(time.time() - t0, 25)
        files = glob.glob(os.path.join(self.tmp, "f_*_camera.mkv"))
        self.assertEqual(len(files), 1, files)
        self.assertGreater(frame_count(files[0]), 30)     # the 3 s before the freeze are kept


class TestSmallThings(unittest.TestCase):
    def test_retry_delay_never_overflows(self):
        req = S.RecordingRequest(device=DeviceRef(index=0, name="t", path="/dev/null"), width=1,
                                 height=1, fps=1, codec="libx264", preset="ultrafast", crf="23",
                                 chunk_sec=60, out_dir=tempfile.gettempdir(), prefix="x")
        rec = S.CameraRecorder(req, input_override=[])
        for n in (0, 1, 2, 3, 4, 1025, 10 ** 6):
            rec.failures_in_row = n
            self.assertLessEqual(rec.retry_delay(), 10.0)
        rec.failures_in_row = 1
        self.assertEqual(rec.retry_delay(), 0.5)            # first retry: (almost) at once

    @unittest.skipIf(sys.platform.startswith("win"), "POSIX exit codes")
    def test_exit_code_226_is_read_only_file_system(self):
        self.assertIn("Read-only file system", S.describe_exit_code(226))
        self.assertEqual(S.describe_exit_code(1), "exit code 1")

    def test_log_survives_a_dead_drive(self):
        tmp = tempfile.mkdtemp(prefix="log_test_")
        try:
            drive, local = os.path.join(tmp, "drive", "logs"), os.path.join(tmp, "local")
            os.makedirs(drive)
            lg = logging.getLogger("studio.test_resilient")
            lg.propagate = False
            h = S.ResilientFileHandler(os.path.join(drive, "studio.log"), local)
            lg.addHandler(h)
            lg.warning("before")
            shutil.rmtree(os.path.join(tmp, "drive"))           # the drive goes away
            h._primary.stream.close()
            h._primary.stream = None
            real_stderr, sys.stderr = sys.stderr, open(os.path.join(tmp, "stderr.txt"), "w")
            try:
                for i in range(50):
                    lg.warning("during %d", i)
            finally:
                sys.stderr.close()
                sys.stderr = real_stderr
            with open(os.path.join(local, "studio.log")) as f:
                text = f.read()
            self.assertIn("during 0", text)
            self.assertIn("during 49", text)
            with open(os.path.join(tmp, "stderr.txt")) as f:
                notices = f.read().splitlines()
            self.assertEqual(len(notices), 1, notices)          # one line, not 50 tracebacks
            self.assertNotIn("Traceback", notices[0])
            os.makedirs(drive)                                   # the drive is back
            lg.warning("after")
            with open(os.path.join(drive, "studio.log")) as f:
                self.assertIn("after", f.read())
            lg.removeHandler(h)
            h.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
