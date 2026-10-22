"""Thermal camera: frame layouts, mode parsing, and opening the right camera.

Runs anywhere (no camera needed):  python -m unittest discover -s tests -v
The camera is replaced by tests/fake_thermal_stream.py, and two real frames
from a newer-revision TC001 are in tests/fixtures/.
"""

import os
import sys
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from thermalcam import camera as tcam  # noqa: E402
from thermalcam import layouts as L  # noqa: E402
from thermalcam import modes as M  # noqa: E402
from thermalcam.devices import VideoDevice  # noqa: E402
from thermalcam.ffmpeg_capture import FFmpegRawCapture  # noqa: E402

FIX = os.path.join(HERE, "fixtures")
FAKE = os.path.join(HERE, "fake_thermal_stream.py")


def celsius(raw64):
    return raw64.astype(np.float64) / 64 - 273.15


def fixture_frames(name, lay):
    return np.fromfile(os.path.join(FIX, name), np.uint8).reshape(-1, lay.frame_bytes)


class FakeCapture(FFmpegRawCapture):
    """FFmpegRawCapture fed by the fake camera instead of FFmpeg."""

    kinds = {}            # layout name -> fake stream kind (set per test)
    die_after = 0.0

    def __init__(self, ref=None, width=256, height=384, **kw):
        lay = next(lay for lay in L.LAYOUTS if (lay.width, lay.height) == (width, height))
        kind = self.kinds.get(lay.name)
        cmd = [sys.executable, FAKE, kind or "none", "--fps", "25"]
        if self.die_after:
            cmd += ["--die-after", str(self.die_after)]
        super().__init__(ref, width, height, command=cmd, open_timeout=kw.get("open_timeout", 4.0))


def mode_list(*sizes):
    return [M.Mode("yuyv422", *map(int, s.split("x")), 25.0) for s in sizes]


class PatchedCamera(unittest.TestCase):
    """Base: two cameras, index 0 = 'USB Camera' (thermal), 1 = webcam."""

    thermal_modes = ["256x192", "256x392", "4x12305"]
    webcam_modes = ["640x480", "1920x1080"]

    def setUp(self):
        self._saved = (tcam.list_video_devices, tcam.list_modes, tcam.FFmpegRawCapture,
                       tcam.shutil_ffmpeg, tcam._platform_backends)
        devices = [VideoDevice(0, "USB Camera", "test"), VideoDevice(1, "HD Webcam C960", "test")]
        tcam.list_video_devices = lambda include_pnp=False: devices
        self.queried = []

        def modes(ref, use_cache=True):
            self.queried.append(ref.index)
            return mode_list(*(self.thermal_modes if ref.index == 0 else self.webcam_modes))
        tcam.list_modes = modes
        tcam.FFmpegRawCapture = FakeCapture
        tcam.shutil_ffmpeg = lambda: True
        tcam._platform_backends = lambda: []        # no OpenCV fallback in tests
        FakeCapture.kinds = {}
        FakeCapture.die_after = 0.0

    def tearDown(self):
        (tcam.list_video_devices, tcam.list_modes, tcam.FFmpegRawCapture,
         tcam.shutil_ffmpeg, tcam._platform_backends) = self._saved


class TestLayouts(unittest.TestCase):
    def test_real_temp_mode_frame(self):
        for fr in fixture_frames("tc001b_temp_12305_2frames.raw", L.TEMP_12305):
            dec = L.TEMP_12305.decode(fr)
            c = celsius(dec.raw64)
            self.assertEqual(c.shape, (192, 256))
            self.assertFalse(dec.approximate)
            self.assertIsNone(tcam.frame_rejection(dec))
            # value/16 K: this indoor scene is 5..25 C, never the 240 C artefact
            self.assertTrue(0 < c.mean() < 40, c.mean())

    def test_real_gray_mode_frame_matches_temp_mode(self):
        temp = celsius(L.TEMP_12305.decode(fixture_frames("tc001b_temp_12305_2frames.raw",
                                                           L.TEMP_12305)[0]).raw64)
        dec = L.GRAY_392.decode(fixture_frames("tc001b_gray_392_2frames.raw", L.GRAY_392)[0])
        gray = celsius(dec.raw64)
        self.assertTrue(dec.approximate)
        # same scene a second apart: the fitted values agree with the camera's own
        self.assertLess(abs(gray.mean() - temp.mean()), 1.0)
        self.assertLess(abs(np.percentile(gray, 95) - np.percentile(temp, 95)), 1.5)

    def test_old_decoder_bug_is_caught(self):
        """The 256x392 frame decoded the old way gave ~240 C; that must be refused."""
        fr = fixture_frames("tc001b_gray_392_2frames.raw", L.GRAY_392)[0]
        picture_half = fr[196 * 512:388 * 512].reshape(192, 256, 2)
        fake = L.Decoded(raw64=L._u16(picture_half), gray=picture_half[..., 0])
        self.assertAlmostEqual(celsius(fake.raw64).mean(), 240, delta=3)
        self.assertIn("picture", tcam.frame_rejection(fake))

    def test_synthetic_layouts(self):
        sys.path.insert(0, HERE)
        import fake_thermal_stream as fs
        for kind, lay in (("classic-384", L.CLASSIC_384), ("temp-12305", L.TEMP_12305),
                          ("gray-392", L.GRAY_392)):
            data = np.frombuffer(fs.frame(kind, 0.5), np.uint8)
            self.assertEqual(data.size, lay.frame_bytes, kind)
            dec = lay.decode(data)
            c = celsius(dec.raw64)
            self.assertIsNone(tcam.frame_rejection(dec), kind)
            self.assertAlmostEqual(np.median(c), 22.5, delta=1.0, msg=kind)
            self.assertAlmostEqual(c.max(), 34.0, delta=1.5 if lay.approximate else 0.3, msg=kind)
        with self.assertRaises(L.LayoutError):
            L.CLASSIC_384.decode(np.frombuffer(fs.frame("picture-384", 0), np.uint8))

    def test_mode_helpers(self):
        self.assertEqual([lay.name for lay in L.layout_for_modes(["256x392", "4x12305"])],
                         ["temp-12305", "gray-392"])
        self.assertTrue(L.is_thermal_mode_list(["256x384"]))
        self.assertFalse(L.is_thermal_mode_list(["640x480", "1920x1080"]))


class TestModeParsing(unittest.TestCase):
    def read(self, name):
        with open(os.path.join(FIX, name)) as f:
            return f.read()

    def test_dshow_thermal(self):
        modes = M.parse_dshow(self.read("dshow_list_options_tc001b.txt"))
        sizes = {m.size for m in modes}
        self.assertIn("4x12305", sizes)
        self.assertIn("256x392", sizes)
        self.assertEqual(L.layout_for_modes(sizes)[0], L.TEMP_12305)

    def test_choose_recording_mode(self):
        ac310 = M.parse_dshow(self.read("dshow_list_options_ac310.txt"))
        emeet = M.parse_dshow(self.read("dshow_list_options_emeet.txt"))
        m = M.choose_recording_mode(ac310, 1920, 1080, 30)
        self.assertEqual((m.fmt, m.size, m.fps, m.compressed), ("mjpeg", "1920x1080", 30.0, True))
        m = M.choose_recording_mode(ac310, 3840, 2160, 30)
        self.assertEqual((m.fmt, m.size), ("mjpeg", "3840x2160"))
        # 1280x720: YUYV only at 10 fps, MJPEG at 30 -> MJPEG
        self.assertEqual(M.choose_recording_mode(ac310, 1280, 720, 30).fmt, "mjpeg")
        # the eMeet has no 4K: largest 16:9 mode below it
        m = M.choose_recording_mode(emeet, 3840, 2160, 30)
        self.assertEqual(m.size, "1920x1080")
        # 60 fps asked, camera does 30 -> still its 1080p MJPEG
        self.assertEqual(M.choose_recording_mode(emeet, 1920, 1080, 60).size, "1920x1080")

    def test_v4l2_parsing(self):
        text = """ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture

	[0]: 'MJPG' (Motion-JPEG, compressed)
		Size: Discrete 1920x1080
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 1280x720
			Interval: Discrete 0.017s (60.000 fps)
			Interval: Discrete 0.033s (30.000 fps)
	[1]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 256x392
			Interval: Discrete 0.040s (25.000 fps)
		Size: Discrete 4x12305
			Interval: Discrete 0.040s (25.000 fps)
"""
        modes = {(m.fmt, m.size): m for m in M.parse_v4l2_ctl(text)}
        self.assertEqual(modes[("mjpeg", "1280x720")].fps, 60.0)
        self.assertTrue(modes[("mjpeg", "1920x1080")].compressed)
        self.assertIn(("yuyv422", "4x12305"), modes)
        ff = ("[video4linux2,v4l2 @ 0x5] Raw       :     yuyv422 :           YUYV 4:2:2 : 256x392 4x12305\n"
              "[video4linux2,v4l2 @ 0x5] Compressed:       mjpeg :          Motion-JPEG : 1920x1080 1280x720\n")
        sizes = {(m.fmt, m.size) for m in M.parse_ffmpeg_v4l2(ff)}
        self.assertIn(("yuyv422", "4x12305"), sizes)
        self.assertIn(("mjpeg", "1280x720"), sizes)

    def test_avfoundation_parsing(self):
        text = ("[avfoundation @ 0x1] Selected video size (1x1) is not supported by the device.\n"
                "[avfoundation @ 0x1] Supported modes:\n"
                "[avfoundation @ 0x1]   256x392@[25.000000 25.000000]fps\n"
                "[avfoundation @ 0x1]   1920x1080@[1.000000 30.000000]fps\n")
        sizes = {m.size: m.fps for m in M.parse_avfoundation(text)}
        self.assertEqual(sizes, {"256x392": 25.0, "1920x1080": 30.0})

    def test_ffmpeg_input_args(self):
        ref = M.DeviceRef(index=2, name="USB Camera", number=1)
        win = ref.ffmpeg_input(size="1920x1080", fmt="mjpeg", compressed=True, fps=30, platform="win32")
        self.assertEqual(win[:2], ["-f", "dshow"])
        self.assertIn("-vcodec", win)
        self.assertEqual(win[-2:], ["-i", "video=USB Camera"])
        self.assertEqual(win[win.index("-video_device_number") + 1], "1")
        lin = M.DeviceRef(index=3, name="cam", path="/dev/video3").ffmpeg_input(
            size="4x12305", fmt="yuyv422", fps=25, platform="linux")
        self.assertEqual(lin, ["-f", "v4l2", "-input_format", "yuyv422", "-video_size", "4x12305",
                               "-framerate", "25", "-i", "/dev/video3"])
        mac = M.DeviceRef(index=1, name="cam").ffmpeg_input(size="256x392", fmt="yuyv422",
                                                             fps=25, platform="darwin")
        self.assertEqual(mac[-2:], ["-i", "1:none"])


class TestOpenCamera(PatchedCamera):
    def test_prefers_temperature_mode_and_never_touches_the_webcam(self):
        FakeCapture.kinds = {"temp-12305": "temp-12305", "gray-392": "gray-392"}
        cam = tcam.ThermalCamera().open()
        try:
            self.assertEqual(cam.info()["layout"], "temp-12305")
            self.assertEqual(cam.resolved, (0, tcam.CAP_FFMPEG_RAW, "temp-12305"))
            fr = cam.read()
            self.assertFalse(fr.approximate)
            self.assertAlmostEqual(float(np.median(fr.celsius)), 22.5, delta=1.0)
            self.assertAlmostEqual(float(fr.celsius.max()), 34.0, delta=0.5)
        finally:
            cam.close()
        # the webcam was asked for its modes, but never opened
        self.assertTrue(all(a.index != 1 or "not a thermal camera" in a.reason for a in cam.attempts))

    def test_falls_back_to_gray_mode(self):
        FakeCapture.kinds = {"gray-392": "gray-392"}        # temperature mode "fails"
        cam = tcam.ThermalCamera().open()
        try:
            self.assertEqual(cam.info()["layout"], "gray-392")
            self.assertTrue(cam.read().approximate)
        finally:
            cam.close()

    def test_refuses_picture_instead_of_temperatures(self):
        self.thermal_modes = ["256x384"]
        FakeCapture.kinds = {"classic-384": "picture-384"}
        with self.assertRaises(tcam.ThermalCameraError) as ctx:
            tcam.ThermalCamera().open()
        self.assertIn("picture", str(ctx.exception))

    def test_classic_camera(self):
        self.thermal_modes = ["256x192", "256x384"]
        FakeCapture.kinds = {"classic-384": "classic-384"}
        cam = tcam.ThermalCamera().open()
        try:
            self.assertEqual(cam.info()["layout"], "classic-384")
            c = cam.read().celsius
            self.assertAlmostEqual(float(c.max()), 34.0, delta=0.3)
        finally:
            cam.close()

    def test_exclude_and_cancel(self):
        FakeCapture.kinds = {"temp-12305": "temp-12305"}
        with self.assertRaises(tcam.ThermalCameraError):
            tcam.ThermalCamera(exclude=[0]).open()
        with self.assertRaises(tcam.OpenCancelled):
            tcam.ThermalCamera(should_stop=lambda: True).open()

    def test_unplug_raises_on_read(self):
        FakeCapture.kinds = {"temp-12305": "temp-12305"}
        FakeCapture.die_after = 1.5
        cam = tcam.ThermalCamera().open()
        try:
            deadline = time.time() + 6
            with self.assertRaises(tcam.ThermalCameraError):
                while time.time() < deadline:
                    cam.read()
        finally:
            cam.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
