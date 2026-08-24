# Cyber Video Data Logger

A Linux desktop application for long-duration camera recording and monitoring. The program provides a modern CustomTkinter GUI for selecting a V4L2 camera, configuring resolution/FPS/codec/quality, recording segmented video files with FFmpeg, previewing the camera, estimating storage requirements, monitoring remaining recording capacity, and optionally displaying CPU/RAM telemetry.

This guide explains how to set up the project from a fresh **Ubuntu** machine by cloning it from GitHub and installing all required system and Python dependencies.

---

## Features

- V4L2 camera discovery on Linux
- Configurable recording resolution and frame rate
- H.264 and H.265 CPU encoding
- Optional Intel/VAAPI H.264 hardware encoding
- Segmented/continuous recording for long experiments
- Configurable chunk duration
- Camera image preview and live preview
- Recording directly through FFmpeg
- Storage-used progress bar
- Estimated remaining recording time
- Storage requirement calculator
- Storage statistics updated during recording
- Optional CPU and RAM monitoring/plots
- CPU/RAM measurement remains disabled until requested
- Automatic detection of FFmpeg recording-process failure
- Low-disk-space protection
- Dark cyber-style GUI

---

# 1. Requirements

## Operating system

The application is designed for Linux and is intended to run on distributions such as:

- Ubuntu 22.04 LTS
- Ubuntu 24.04 LTS
- Newer Ubuntu releases should normally work as well

The instructions below assume Ubuntu or another Debian-based distribution.

## Hardware

You need:

- A computer running Ubuntu with a graphical desktop environment
- A Linux-compatible USB/web camera or other V4L2 video device
- Enough disk space for the desired recording duration
- A display, keyboard, and mouse, or a working remote graphical desktop

For 4K recording, CPU performance, USB bandwidth, camera capability, and storage speed can all matter.

---

# 2. Install Git and system prerequisites

Open a terminal and first update the Ubuntu package index:

```bash
sudo apt update
```

Install the required system packages:

```bash
sudo apt install -y \
    git \
    python3 \
    python3-pip \
    python3-venv \
    python3-tk \
    ffmpeg \
    v4l-utils \
    libgl1 \
    libglib2.0-0
```

These packages are used for the following purposes:

| Package | Purpose |
|---|---|
| `git` | Clone and update the project from GitHub |
| `python3` | Run the application |
| `python3-pip` | Install Python dependencies |
| `python3-venv` | Create an isolated Python environment |
| `python3-tk` | Tk/Tkinter GUI support |
| `ffmpeg` | Actual video capture, encoding, and segmented recording |
| `v4l-utils` | Provides `v4l2-ctl` for camera discovery/capability detection |
| `libgl1` | OpenCV runtime dependency on many Ubuntu systems |
| `libglib2.0-0` | OpenCV/runtime support |

Verify FFmpeg:

```bash
ffmpeg -version
```

Verify V4L2 tools:

```bash
v4l2-ctl --version
```

Verify Python:

```bash
python3 --version
```

---

# 3. Clone the project from GitHub

Go to the directory where you want to keep the project. For example:

```bash
cd ~
```

Clone your repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

For example, when your repository URL looks like:

```text
https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

run:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Then enter the project directory:

```bash
cd YOUR_REPOSITORY
```

You should normally have at least these files:

```text
YOUR_REPOSITORY/
├── cyber_video_data_logger.py
├── requirements_cyber_video_logger.txt
└── README.md
```

Check with:

```bash
ls
```

---

# 4. Install Python dependencies

The Python requirements for the application are stored in:

```text
requirements_cyber_video_logger.txt
```

They include the GUI, Pillow image support, and OpenCV.

## Recommended method: virtual environment

Using a virtual environment is the safest option on modern Ubuntu/Debian systems because it prevents `pip` packages from interfering with Python packages managed by `apt`.

Create the virtual environment inside the project directory:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Your shell prompt will normally show `(.venv)`.

Upgrade pip:

```bash
python3 -m pip install --upgrade pip
```

Install the project's Python packages:

```bash
python3 -m pip install -r requirements_cyber_video_logger.txt
```

After this, leave the virtual environment active while running the application.

### Activate the environment again later

Every time you open a new terminal and want to run the application:

```bash
cd ~/YOUR_REPOSITORY
source .venv/bin/activate
python3 cyber_video_data_logger.py
```

To leave the virtual environment:

```bash
deactivate
```

---

# 5. Alternative: install without a virtual environment

If you intentionally do not want to use a virtual environment, newer Ubuntu/Debian versions may block normal global `pip` installation because the system Python installation is externally managed.

Install the packages for your user with:

```bash
python3 -m pip install --user --break-system-packages -r requirements_cyber_video_logger.txt
```

Do **not** use `sudo pip install ...` unless you fully understand the consequences. It can overwrite packages managed by Ubuntu and cause Python/package-manager conflicts.

The virtual-environment installation in the previous section is recommended for most installations.

---

# 6. Connect and verify the camera

Connect the camera before launching the application.

List detected video devices:

```bash
ls -l /dev/video*
```

You may see devices such as:

```text
/dev/video0
/dev/video1
```

List cameras with V4L2:

```bash
v4l2-ctl --list-devices
```

For example:

```text
USB Camera:
    /dev/video0
    /dev/video1
```

Inspect one device:

```bash
v4l2-ctl -d /dev/video0 --all
```

List the resolutions, pixel formats, and frame rates supported by the camera:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

This is particularly important if you want to record at 4K or 60 FPS.

---

# 7. Camera permissions

Normally Ubuntu gives logged-in desktop users access to camera devices automatically. If the application cannot open `/dev/video0`, check its permissions:

```bash
ls -l /dev/video0
```

Check whether your account belongs to the `video` group:

```bash
groups
```

If `video` is missing, add your user to it:

```bash
sudo usermod -aG video "$USER"
```

Then **log out of Ubuntu and log back in** for the new group membership to take effect.

You can verify afterward with:

```bash
groups
```

Do not solve camera permissions by permanently running the whole GUI with `sudo`.

---

# 8. Run the application

Enter the cloned repository:

```bash
cd ~/YOUR_REPOSITORY
```

If you used the recommended virtual environment:

```bash
source .venv/bin/activate
```

Run:

```bash
python3 cyber_video_data_logger.py
```

The graphical application should open.

---

# 9. First-time setup inside the application

When the program starts:

1. Wait for camera scanning to finish.
2. Select the desired camera.
3. Select the recording resolution.
4. Select the frame rate.
5. Select a video codec.
6. Select the encoder preset when using a CPU encoder.
7. Set the CRF/quality value.
8. Set the desired chunk duration.
9. Select the recording destination directory.
10. Review the estimated storage usage and available recording time.
11. Start recording.

The application writes recording chunks to the destination folder you selected.

---

# 10. Recording settings

## Resolution

Typical options include:

- `3840x2160` — 4K
- `2560x1440` — 1440p / 2K
- `1920x1080` — Full HD
- `1280x720` — 720p

The selected combination must actually be supported by your camera.

## Frame rate

Typical options include:

- 15 FPS
- 24 FPS
- 30 FPS
- 60 FPS

Higher FPS normally increases CPU/GPU workload, USB traffic, and storage requirements.

## Codec

The application supports selections such as:

### `libx264`

H.264 software encoding using the CPU.

This is generally the safest default.

### `libx265`

H.265/HEVC software encoding using the CPU.

It can produce smaller files at comparable visual quality but requires significantly more CPU than H.264, especially at high resolution.

### `h264_vaapi`

Hardware-accelerated H.264 using VAAPI.

This requires compatible graphics hardware and Linux VAAPI support. See the VAAPI section below.

---

# 11. CRF / quality

For software encoders such as `libx264` and `libx265`, CRF controls the quality/file-size tradeoff.

General interpretation:

```text
Lower CRF  -> higher quality -> larger files
Higher CRF -> lower quality  -> smaller files
```

A value around `23` is a common H.264 starting point.

The exact bitrate cannot be known beforehand when CRF encoding is used because the bitrate depends on the content being recorded. A static scene is usually easier to compress than a scene with significant motion, noise, or fine detail.

The application's initial capacity calculation is therefore an estimate. During recording, measured disk usage provides a more meaningful indication of actual recording capacity.

---

# 12. Chunk duration

Long recordings are divided into separate video files rather than one enormous file.

For example, with:

```text
Chunk = 30 minutes
```

a long experiment produces files representing approximately 30-minute sections.

Advantages include:

- Easier file management
- Less risk from one extremely large file
- Easier copying and processing
- Better recovery if recording is unexpectedly interrupted
- Convenient periodic storage updates

Choose the chunk size based on your experiment/workflow.

---

# 13. Storage monitoring

The storage progress bar represents **used disk capacity**.

Conceptually:

```text
Empty bar  -> little disk space used
Full bar   -> drive nearly full
```

The GUI also displays free disk space and estimated remaining recording time.

Storage does not need to be polled every second. During recording, it can be refreshed on the recording/chunk schedule so the program performs unnecessary filesystem work less frequently.

The program also contains protection against critically low free disk space so that recording can be stopped cleanly before the drive is completely full.

---

# 14. Storage and recording-time calculations

Before enough real recording data exists, storage requirements must be estimated from the chosen recording settings, including factors such as:

- Resolution
- Frame rate
- Codec
- CRF/quality
- Encoder preset

Because CRF is variable bitrate, two videos recorded with identical settings can still have different sizes.

For example:

- A stationary laboratory setup with an almost unchanged background may compress very efficiently.
- A noisy image, moving equipment, people walking through the scene, or constantly changing detail may require substantially more bitrate.

Therefore, pre-recording storage calculations should be treated as planning estimates rather than guaranteed file sizes.

The most reliable long-duration estimate is based on the actual amount of data produced by the real recording setup.

---

# 15. CPU and RAM monitor

CPU/RAM monitoring is optional.

It is intentionally not supposed to continuously measure system load when you have not requested telemetry.

When CPU/RAM monitoring is disabled:

- CPU usage is not sampled for the plot
- RAM usage is not sampled for the plot
- Telemetry history is not continuously collected
- Telemetry plot updates do not run

When you enable the CPU/RAM display, monitoring begins and live telemetry is shown.

Disable/hide it again when you do not need it.

This design keeps unnecessary monitoring overhead away from the recording process.

---

# 16. Preview modes

The application provides camera-preview functionality so you can inspect framing/focus without making the preview the main recording pipeline.

When recording, FFmpeg creates a current preview image in shared memory:

```text
/dev/shm/current_snapshot.jpg
```

`/dev/shm` is memory-backed temporary storage on Linux, so the preview image does not continuously write small snapshot files to the recording disk.

The GUI can read this snapshot for preview purposes while FFmpeg continues handling the video recording.

For long unattended recordings, avoid running a high-rate preview unless it is actually needed. Preview image decoding and GUI updates consume CPU resources, especially on older machines.

---

# 17. Optional Intel/VAAPI hardware encoding

If you want to use:

```text
h264_vaapi
```

you need compatible Linux graphics hardware and drivers.

Install VAAPI diagnostic tools:

```bash
sudo apt install -y vainfo
```

Run:

```bash
vainfo
```

Check whether the render device exists:

```bash
ls -l /dev/dri/
```

The application expects a render device such as:

```text
/dev/dri/renderD128
```

Check your groups:

```bash
groups
```

If necessary, add yourself to the `render` and `video` groups:

```bash
sudo usermod -aG render,video "$USER"
```

Then log out and back in.

Test whether FFmpeg sees VAAPI encoders:

```bash
ffmpeg -encoders | grep vaapi
```

You should see an encoder such as:

```text
h264_vaapi
```

If VAAPI does not work, select `libx264` instead. Software H.264 does not depend on `/dev/dri/renderD128`.

---

# 18. Test the camera with FFmpeg directly

If the GUI detects the camera but recording fails, first test the camera outside the application.

List camera formats:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

The recorder uses an MJPEG camera input path, so verify that your camera supports MJPEG for the requested resolution/frame rate.

A simple test is:

```bash
ffmpeg \
    -f v4l2 \
    -input_format mjpeg \
    -framerate 30 \
    -video_size 1920x1080 \
    -i /dev/video0 \
    -t 10 \
    -c:v libx264 \
    test_recording.mkv
```

Stop after the test and play it with a video player.

If your camera does not support `1920x1080` MJPEG at 30 FPS, replace those values with a combination shown by:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

---

# 19. Check camera bandwidth and supported modes

Do not assume that a camera marketed as 4K can provide every combination such as:

```text
4K + MJPEG + 60 FPS
```

Check the actual modes exposed to Linux:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

Use a resolution/FPS combination explicitly shown in the output.

If a recording is laggy or frames are missing, test lower settings such as:

```text
3840x2160 @ 30 FPS
1920x1080 @ 30 FPS
1920x1080 @ 60 FPS
```

and compare results.

---

# 20. Updating the project from GitHub

Once the project has already been cloned, you do not need to clone it again each time.

Enter the repository:

```bash
cd ~/YOUR_REPOSITORY
```

Pull the newest version:

```bash
git pull
```

If `requirements_cyber_video_logger.txt` changed, update the Python packages too.

With a virtual environment:

```bash
source .venv/bin/activate
python3 -m pip install -r requirements_cyber_video_logger.txt
```

Then run:

```bash
python3 cyber_video_data_logger.py
```

---

# 21. Complete installation example

For a new Ubuntu computer, the complete recommended procedure is approximately:

```bash
# Update Ubuntu package metadata
sudo apt update

# Install system dependencies
sudo apt install -y \
    git \
    python3 \
    python3-pip \
    python3-venv \
    python3-tk \
    ffmpeg \
    v4l-utils \
    libgl1 \
    libglib2.0-0

# Go home
cd ~

# Clone the project
# Replace this URL with the real GitHub repository URL
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

# Enter the repository
cd YOUR_REPOSITORY

# Create Python virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Upgrade pip
python3 -m pip install --upgrade pip

# Install Python dependencies
python3 -m pip install -r requirements_cyber_video_logger.txt

# Confirm that a camera is detected
v4l2-ctl --list-devices

# Start the program
python3 cyber_video_data_logger.py
```

After the first installation, starting the program is much shorter:

```bash
cd ~/YOUR_REPOSITORY
source .venv/bin/activate
python3 cyber_video_data_logger.py
```

---

# 22. Troubleshooting

## `ModuleNotFoundError: No module named 'customtkinter'`

You probably installed the dependencies into a different Python environment.

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Then:

```bash
python3 -m pip install -r requirements_cyber_video_logger.txt
```

Check:

```bash
python3 -m pip show customtkinter
```

---

## `externally-managed-environment`

This commonly occurs on newer Ubuntu/Debian versions when installing packages into the system Python environment.

Recommended solution:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements_cyber_video_logger.txt
```

If you deliberately do not want a virtual environment:

```bash
python3 -m pip install --user --break-system-packages -r requirements_cyber_video_logger.txt
```

---

## `v4l2-ctl: command not found`

Install:

```bash
sudo apt install v4l-utils
```

---

## `ffmpeg: command not found`

Install:

```bash
sudo apt install ffmpeg
```

---

## No cameras appear in the application

Check:

```bash
v4l2-ctl --list-devices
```

and:

```bash
ls -l /dev/video*
```

If no `/dev/video*` devices exist, Ubuntu is not currently exposing the camera as a V4L2 device.

If a device exists but access is denied, check your `video` group membership.

---

## Camera is detected but recording immediately stops

Check the camera's supported modes:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

Make sure the selected resolution and FPS are actually supported with MJPEG.

Then test FFmpeg directly using the example earlier in this README.

---

## `Permission denied: /dev/video0`

Add your account to the video group:

```bash
sudo usermod -aG video "$USER"
```

Log out and log back in.

---

## OpenCV error mentioning `libGL.so.1`

Install:

```bash
sudo apt install libgl1
```

Then try again.

---

## GUI does not open over SSH

This is a desktop GUI application. A normal SSH terminal does not provide a graphical display.

Run it directly from the Ubuntu desktop, or configure an appropriate graphical remote desktop/X environment if remote GUI access is required.

---

## VAAPI encoder fails

Test:

```bash
vainfo
```

and:

```bash
ffmpeg -encoders | grep vaapi
```

Also verify:

```bash
ls -l /dev/dri/renderD128
```

If VAAPI is unavailable, use:

```text
libx264
```

instead.

---

## Recorded video appears laggy

Possible causes include:

- Camera cannot sustain the selected resolution/FPS
- CPU cannot encode the selected resolution/FPS fast enough
- `libx265` is too demanding for the computer
- USB bandwidth limitations
- High preview workload on a weak machine
- Storage device cannot sustain the write rate
- Incorrect camera format/FPS combination

Useful checks:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

and inspect a recorded file with:

```bash
ffprobe -v error \
    -select_streams v:0 \
    -show_entries stream=codec_name,r_frame_rate,avg_frame_rate,nb_frames,duration \
    -of default=noprint_wrappers=1 \
    your_recording.mkv
```

Try recording with preview disabled and compare the result. Also try `libx264` with the `veryfast` or `ultrafast` preset if CPU usage is very high.

---

# 23. Useful diagnostic commands

Camera devices:

```bash
v4l2-ctl --list-devices
```

Camera modes:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

Device permissions:

```bash
ls -l /dev/video*
```

User groups:

```bash
groups
```

FFmpeg version:

```bash
ffmpeg -version
```

Available H.264/H.265 encoders:

```bash
ffmpeg -encoders | grep -E '264|265|hevc'
```

VAAPI encoders:

```bash
ffmpeg -encoders | grep vaapi
```

Disk usage:

```bash
df -h
```

Python dependencies:

```bash
python3 -m pip list
```

---

# 24. Suggested GitHub repository layout

A clean repository can look like this:

```text
cyber-video-data-logger/
├── cyber_video_data_logger.py
├── requirements_cyber_video_logger.txt
├── README.md
├── .gitignore
└── LICENSE
```

A useful `.gitignore` would normally exclude:

```text
.venv/
__pycache__/
*.pyc
*.mkv
*.mp4
```

Large recorded video files should generally not be committed to GitHub.

---

# 25. Quick start

After the project has already been installed:

```bash
cd ~/YOUR_REPOSITORY
source .venv/bin/activate
python3 cyber_video_data_logger.py
```

Then select your camera/settings and start recording from the GUI.

---

# Notes

- This application is Linux-specific because camera discovery and recording use V4L2 (`/dev/video*`).
- The application is intended for video-only recording; ensure your workflow does not require an audio track unless audio support is added separately.
- Always test the chosen camera/resolution/FPS/encoder combination before starting a long unattended experiment.
- For critical recordings, perform a short test recording and play the resulting file before beginning the real experiment.
- Keep sufficient free disk space and verify the destination path before starting a long recording.
