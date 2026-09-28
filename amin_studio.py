import os
import sys
import glob
import json
import time
import shutil
import subprocess
import threading
import re
from collections import deque
from datetime import datetime

import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image, ImageTk
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from thermalcam import Calibration  # noqa: E402
from thermalcam.decode import SENSOR_HEIGHT, SENSOR_WIDTH  # noqa: E402
from thermalcam.devices import NOT_THERMAL_NAME_HINTS, THERMAL_NAME_HINTS  # noqa: E402
from thermalcam.render import COLORMAP_NAMES  # noqa: E402

from studio_services import (  # noqa: E402
    COLORBAR_WIDTH,
    SleepInhibitor,
    camera_controls_auto,
    camera_controls_lock,
    camera_controls_manual,
    camera_controls_read,
    ffmpeg_log_file,
    log,
    setup_logging,
    CameraService,
    PhotoCapturer,
    PhotoSettings,
    ThermalPipeline,
    ThermalService,
    ThermalVideoRecorder,
    TimelapseScheduler,
    describe_step,
    estimate_shot_bytes,
    estimate_thermal_mbps,
    format_hms,
    parse_hms,
    parse_schedule_text,
    step_schedule,
)


# -----------------------------------------------------------------------------
# Visual system
# -----------------------------------------------------------------------------
COLORS = {
    "bg": "#070A0D",
    "panel": "#0D1117",
    "panel_alt": "#111820",
    "border": "#1F2933",
    "text": "#E6EDF3",
    "text_dim": "#8B949E",
    "cyan": "#22D3EE",
    "cyan_dark": "#0E7490",
    "green": "#3FB950",
    "amber": "#D29922",
    "red": "#F85149",
    "orange": "#F0883E",
    "bar_bg": "#1A222C",
    "black": "#000000",
}

FONT_UI = ("Segoe UI", 12)
FONT_UI_SMALL = ("Segoe UI", 10)
FONT_UI_BOLD = ("Segoe UI", 12, "bold")
FONT_SECTION = ("Segoe UI", 11, "bold")
FONT_MONO = ("DejaVu Sans Mono", 11)
FONT_MONO_BIG = ("DejaVu Sans Mono", 20, "bold")
FONT_MONO_SMALL = ("DejaVu Sans Mono", 9)

CONFIG_DIR = os.path.expanduser("~/.config/amin_studio")
SETTINGS_PATH = os.path.join(CONFIG_DIR, "settings.json")
CALIB_PATH = os.path.join(CONFIG_DIR, "calibration.json")
ACTIVE_RUN_PATH = os.path.join(CONFIG_DIR, "active_run.json")
AUTOSTART_PATH = os.path.expanduser("~/.config/autostart/amin-studio.desktop")
RESUME_DELAY_S = 20

RESOLUTIONS = [
    "3840x2160 (4K)",
    "2560x1440 (2K)",
    "1920x1080 (1080p)",
    "1280x720 (720p)",
]
SHOTS_PRESETS = ["1", "2", "3", "4", "6", "8", "12", "24", "48", "96", "144", "288", "720", "1440"]
THERMAL_DATA_CHOICES = {
    "NPY  (°C float32)": "npy",
    "CSV  (°C text)": "csv",
    "PNG16  (raw K×64)": "png16",
    "None": "none",
}
THERMAL_VIDEO_CHOICES = {
    "Colour video (H.264)": ("color",),
    "Radiometric (FFV1 16-bit)": ("raw",),
    "Colour + radiometric": ("color", "raw"),
}
CORRECTION_CHOICES = {
    "Off": "off",
    "Offset / gain": "manual",
    "Reference points": "reference",
}
SPAN_CHOICES = {"Auto": "auto", "Manual": "manual", "Percentile": "percentile"}

SHORTCUTS = [
    ("GENERAL", [
        ("H", "show / hide this help"),
        ("V  /  I", "VIDEO tab  /  PHOTO (imaging) tab"),
        ("SPACE", "start recording / time-lapse   (press twice to stop)"),
        ("L", "live stream on / off"),
        ("C", "capture one photo pair now (PHOTO tab)"),
        ("G", "CPU / RAM telemetry"),
        ("R", "refresh devices"),
        ("ESC", "cancel a pending auto-resume"),
        ("ESC", "close popup / leave a text field"),
    ]),
    ("THERMAL DISPLAY", [
        ("P  /  Shift+P", "next / previous palette"),
        ("M", "span mode: auto → manual → percentile"),
        ("E", "type the span min and max"),
        ("[  /  ]", "shift manual span down / up 1 °C"),
        ("-  /  +", "narrow / widen manual span"),
        ("B", "fit span to the reference points"),
        ("T", "freeze / unfreeze thermal preview"),
    ]),
    ("THERMAL CALIBRATION", [
        ("click", "add a reference point on the thermal preview (or select one)"),
        ("right-click", "remove the point under the cursor"),
        ("K  /  double-click", "type the true temperature of the selected point"),
        ("1 … 9", "select point N"),
        ("X  /  Shift+X", "delete selected point  /  clear all points"),
        ("F", "cycle fit: auto / offset / linear / quadratic / off"),
        ("U", "refit from current readings"),
        ("S  /  O", "save  /  load thermal settings + calibration"),
    ]),
]


class AdvancedRecorderApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("4K + Thermal Data Logger")
        self.root.geometry("1320x900")
        self.root.minsize(1120, 740)
        self.root.configure(fg_color=COLORS["bg"])

        self.mode = "video"

        # Recording state
        self.ffmpeg_proc = None
        self.start_time = None
        self.is_recording = False
        self.thermal_recorder = None

        # Time-lapse state
        self.is_timelapse = False
        self.scheduler = None
        self.timelapse_started = None
        self.last_capture = None
        self.timelapse_shots = 0
        self._capture_lock = threading.Lock()

        # Preview state
        self.is_streaming = False
        self.stream_thread = None
        self.is_image_previewing = False
        self.image_preview_thread = None
        self.thermal_frozen = False
        self._frozen_celsius = None
        self._thermal_view_celsius = None
        self._thermal_geom = None
        self._thermal_cursor = None
        self._last_thermal_render_no = -1
        self.selected_point = None

        # RAM snapshot used by FFmpeg while recording
        self.shm_snapshot = "/dev/shm/current_snapshot.jpg"

        # Devices found by the last scan: list of dicts
        self.devices = []

        # Services
        self.pipeline = ThermalPipeline()
        self.thermal = ThermalService(self.pipeline)
        self.camera = CameraService()
        self.capturer = PhotoCapturer(self.camera, self.thermal, self.pipeline)
        self.camera.on_ready = self._camera_ready_hook

        # System telemetry is completely opt-in.
        self.is_telemetry_active = False
        self._telemetry_after_id = None
        self.cpu_history = deque(maxlen=60)
        self.ram_history = deque(maxlen=60)
        self._last_cpu_sample = None

        self._storage_after_id = None
        self._camera_retry_id = None
        self._camera_restarts = 0
        self._rec_params = None
        self.inhibitor = SleepInhibitor()
        self._resume_after_id = None
        self._resume_attempts = 0
        self._last_heartbeat = 0.0
        self._last_thermal_snapshot = None
        # camera exposure / WB / focus control state (read by worker threads)
        self._ctrl_mode = "auto"
        self._ctrl_manual = {}
        self._ctrl_touched = False
        self._locked_values = None
        self._log_dir = os.path.join(CONFIG_DIR, "logs")
        try:
            setup_logging(self._log_dir)
        except Exception:
            pass
        log.info("===== Amin Studio started (pid %d) =====", os.getpid())
        self._space_armed_until = 0.0
        self._alert_after_id = None
        self.help_window = None

        self._load_settings_file()
        self.setup_ui()
        self._apply_loaded_settings()
        self.refresh_cameras()
        self.set_mode("video", force=True)
        self.update_storage_stats(schedule_next=True)
        self.update_timer_loop()
        self._thermal_preview_loop()

        self.root.bind_all("<Key>", self.on_key, add="+")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(1500, self._check_resume)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def section_label(self, parent, text):
        return ctk.CTkLabel(
            parent,
            text=text.upper(),
            font=FONT_SECTION,
            text_color=COLORS["text_dim"],
            anchor="w",
        )

    def make_card(self, parent):
        return ctk.CTkFrame(
            parent,
            fg_color=COLORS["panel"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"],
        )

    def make_entry(self, parent, **kwargs):
        defaults = dict(
            fg_color=COLORS["panel_alt"],
            border_color=COLORS["border"],
            text_color=COLORS["text"],
            corner_radius=7,
            height=34,
            font=FONT_UI,
        )
        defaults.update(kwargs)
        entry = ctk.CTkEntry(parent, **defaults)
        entry.bind("<Escape>", lambda _e: self.root.focus_set())
        return entry

    def make_combo(self, parent, values, variable=None, command=None, state="normal", **kwargs):
        options = dict(
            values=values,
            variable=variable,
            fg_color=COLORS["panel_alt"],
            button_color=COLORS["border"],
            button_hover_color=COLORS["cyan_dark"],
            border_color=COLORS["border"],
            text_color=COLORS["text"],
            dropdown_fg_color=COLORS["panel_alt"],
            dropdown_hover_color=COLORS["border"],
            dropdown_text_color=COLORS["text"],
            corner_radius=7,
            height=34,
            font=FONT_UI,
            state=state,
        )
        if command is not None:
            options["command"] = command
        options.update(kwargs)
        return ctk.CTkComboBox(parent, **options)

    def make_button(self, parent, text, command, accent=False, danger=False, **kwargs):
        if danger:
            fg = COLORS["red"]
            hover = "#C93C37"
            text_color = "white"
        elif accent:
            fg = COLORS["cyan_dark"]
            hover = "#155E75"
            text_color = COLORS["text"]
        else:
            fg = COLORS["panel_alt"]
            hover = COLORS["border"]
            text_color = COLORS["text"]

        options = dict(
            text=text,
            command=command,
            fg_color=fg,
            hover_color=hover,
            text_color=text_color,
            border_width=1,
            border_color=COLORS["border"],
            corner_radius=7,
            height=36,
            font=FONT_UI_BOLD,
        )
        options.update(kwargs)
        return ctk.CTkButton(parent, **options)

    def make_switch(self, parent, text, variable, command=None):
        return ctk.CTkSwitch(
            parent,
            text=text,
            variable=variable,
            onvalue=True,
            offvalue=False,
            command=command,
            font=FONT_UI_SMALL,
            text_color=COLORS["text"],
            progress_color=COLORS["cyan_dark"],
            button_color=COLORS["text"],
            button_hover_color=COLORS["cyan"],
            fg_color=COLORS["border"],
        )

    def make_check(self, parent, text, variable, command=None):
        return ctk.CTkCheckBox(
            parent,
            text=text,
            variable=variable,
            onvalue=True,
            offvalue=False,
            command=command,
            font=FONT_UI_SMALL,
            text_color=COLORS["text"],
            fg_color=COLORS["cyan_dark"],
            hover_color="#155E75",
            border_color=COLORS["border"],
            checkmark_color=COLORS["text"],
            corner_radius=5,
            border_width=2,
            checkbox_width=18,
            checkbox_height=18,
        )

    def small_label(self, parent, text, color=None, **kwargs):
        return ctk.CTkLabel(
            parent,
            text=text,
            font=FONT_MONO_SMALL,
            text_color=color or COLORS["text_dim"],
            **kwargs,
        )

    def add_labeled_control(self, parent, row, label, widget):
        lbl = ctk.CTkLabel(
            parent,
            text=label,
            font=FONT_UI_SMALL,
            text_color=COLORS["text_dim"],
            anchor="w",
        )
        lbl.grid(row=row, column=0, sticky="w", padx=(14, 8), pady=5)
        widget.grid(row=row, column=1, sticky="ew", padx=(0, 14), pady=5)
        return lbl

    def divider(self, parent, row, columnspan=2):
        ctk.CTkFrame(parent, height=1, fg_color=COLORS["border"]).grid(
            row=row, column=0, columnspan=columnspan, sticky="ew", padx=14, pady=8
        )

    # ------------------------------------------------------------------
    # Main UI
    # ------------------------------------------------------------------
    def setup_ui(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # --------------------------- Header ---------------------------
        self.top_bar = ctk.CTkFrame(
            self.root,
            fg_color=COLORS["bg"],
            corner_radius=0,
            height=74,
        )
        self.top_bar.grid(row=0, column=0, sticky="ew", padx=18, pady=(12, 6))
        self.top_bar.grid_columnconfigure(1, weight=1)

        status_box = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        status_box.grid(row=0, column=0, sticky="w")

        self.status_dot = ctk.CTkLabel(
            status_box,
            text="●",
            font=("Segoe UI", 22),
            text_color=COLORS["text_dim"],
            width=20,
        )
        self.status_dot.grid(row=0, column=0, rowspan=3, padx=(0, 8))

        self.status_text = ctk.CTkLabel(
            status_box,
            text="SYSTEM READY",
            font=("Segoe UI", 14, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        )
        self.status_text.grid(row=0, column=1, sticky="w")

        self.camera_status_label = ctk.CTkLabel(
            status_box,
            text="CAMERA: scanning...",
            font=FONT_MONO_SMALL,
            text_color=COLORS["text_dim"],
            anchor="w",
        )
        self.camera_status_label.grid(row=1, column=1, sticky="w")

        self.thermal_status_label = ctk.CTkLabel(
            status_box,
            text="THERMAL: off",
            font=FONT_MONO_SMALL,
            text_color=COLORS["text_dim"],
            anchor="w",
        )
        self.thermal_status_label.grid(row=2, column=1, sticky="w")

        title_box = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        title_box.grid(row=0, column=1)
        ctk.CTkLabel(
            title_box,
            text="4K + THERMAL DATA LOGGER",
            font=("Segoe UI", 18, "bold"),
            text_color=COLORS["text"],
        ).pack()
        self.subtitle_label = ctk.CTkLabel(
            title_box,
            text="CAPTURE / RECORD / TELEMETRY",
            font=FONT_MONO_SMALL,
            text_color=COLORS["cyan"],
        )
        self.subtitle_label.pack(pady=(1, 4))

        self.mode_tabs = ctk.CTkSegmentedButton(
            title_box,
            values=["VIDEO", "PHOTO"],
            command=lambda value: self.set_mode(value.lower()),
            font=FONT_UI_BOLD,
            height=30,
            width=240,
            corner_radius=8,
            fg_color=COLORS["panel_alt"],
            selected_color=COLORS["cyan_dark"],
            selected_hover_color="#155E75",
            unselected_color=COLORS["panel_alt"],
            unselected_hover_color=COLORS["border"],
            text_color=COLORS["text"],
            border_width=0,
        )
        self.mode_tabs.pack()
        self.mode_tabs.set("VIDEO")

        right_status = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        right_status.grid(row=0, column=2, sticky="e")
        self.clock_label = ctk.CTkLabel(
            right_status,
            text="--:--:--",
            font=FONT_MONO,
            text_color=COLORS["text"],
        )
        self.clock_label.pack(anchor="e")
        self.audio_sign = ctk.CTkLabel(
            right_status,
            text="AUDIO OFF  •  VIDEO ONLY",
            font=FONT_MONO_SMALL,
            text_color=COLORS["amber"],
        )
        self.audio_sign.pack(anchor="e", pady=(2, 0))
        ctk.CTkLabel(
            right_status,
            text="PRESS H FOR SHORTCUTS",
            font=FONT_MONO_SMALL,
            text_color=COLORS["text_dim"],
        ).pack(anchor="e", pady=(2, 0))

        # --------------------------- Main -----------------------------
        self.main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_container.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
        self.main_container.grid_columnconfigure(0, weight=0, minsize=400)
        self.main_container.grid_columnconfigure(1, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        self.left_panel = ctk.CTkScrollableFrame(
            self.main_container,
            fg_color="transparent",
            width=384,
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["cyan_dark"],
        )
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.left_panel.grid_columnconfigure(0, weight=1)

        self.right_panel = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=3)
        self.right_panel.grid_rowconfigure(1, weight=0)

        self._build_devices_card()       # row 0 (shared)
        self._build_capture_card()       # row 1 (video)
        self._build_photo_card()         # row 1 (photo)
        self._build_thermal_card()       # row 2 (shared)
        self._build_storage_card()       # row 3 (shared)
        self._build_recording_card()     # row 4 (video)
        self._build_timelapse_card()     # row 4 (photo)
        self._build_preview_card()
        self._build_telemetry_card()
        self._build_alert_banner()

    # ------------------------------------------------------------------
    # Devices & output card (shared by both tabs)
    # ------------------------------------------------------------------
    def _build_devices_card(self):
        card = self.make_card(self.left_panel)
        card.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        card.grid_columnconfigure(1, weight=1)
        self.devices_card = card

        self.section_label(card, "Devices & Output").grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(12, 8)
        )

        self.use_camera_var = tk.BooleanVar(value=True)
        self.use_thermal_var = tk.BooleanVar(value=True)

        self.use_camera_switch = self.make_switch(card, "CAMERA", self.use_camera_var, self.on_device_toggle)
        self.use_camera_switch.grid(row=1, column=0, sticky="w", padx=(14, 8), pady=5)
        self.cam_combo = self.make_combo(card, ["Scanning cameras..."], command=lambda _v: self.on_device_change())
        self.cam_combo.grid(row=1, column=1, sticky="ew", padx=(0, 14), pady=5)

        self.use_thermal_switch = self.make_switch(card, "THERMAL", self.use_thermal_var, self.on_device_toggle)
        self.use_thermal_switch.grid(row=2, column=0, sticky="w", padx=(14, 8), pady=5)
        self.thermal_combo = self.make_combo(card, ["Auto-detect"], command=lambda _v: self.on_device_change())
        self.thermal_combo.grid(row=2, column=1, sticky="ew", padx=(0, 14), pady=5)

        # Camera exposure / white balance / focus
        self.cam_ctrl_var = tk.StringVar(value="Auto")
        self.cam_ctrl_seg = ctk.CTkSegmentedButton(
            card,
            values=["Auto", "Lock at start", "Manual"],
            variable=self.cam_ctrl_var,
            command=lambda _v: self._on_ctrl_mode_change(),
            font=FONT_UI_SMALL,
            height=30,
            corner_radius=7,
            fg_color=COLORS["panel_alt"],
            selected_color=COLORS["cyan_dark"],
            selected_hover_color="#155E75",
            unselected_color=COLORS["panel_alt"],
            unselected_hover_color=COLORS["border"],
            text_color=COLORS["text"],
        )
        self.add_labeled_control(card, 3, "Exposure/WB", self.cam_ctrl_seg)

        self.ctrl_manual_frame = ctk.CTkFrame(card, fg_color="transparent")
        mf = self.ctrl_manual_frame
        self.ctrl_vars = {}
        self.ctrl_entries = []
        for i, (key, text) in enumerate((("exposure", "EXP"), ("gain", "GAIN"), ("wb", "WB K"), ("focus", "FOCUS"))):
            mf.grid_columnconfigure(i, weight=1)
            self.small_label(mf, text).grid(row=0, column=i, sticky="w", padx=(0 if i == 0 else 4, 0))
            var = tk.StringVar(value="")
            entry = self.make_entry(mf, textvariable=var, width=60, height=30)
            entry.grid(row=1, column=i, sticky="ew", padx=(0 if i == 0 else 4, 0))
            entry.bind("<FocusOut>", lambda _e: self._sync_ctrl_settings())
            entry.bind("<Return>", lambda _e: self._sync_ctrl_settings())
            self.ctrl_vars[key] = var
            self.ctrl_entries.append(entry)
        self.ctrl_read_btn = self.make_button(
            mf, "READ", self.read_camera_controls, width=54, height=30, font=FONT_UI_SMALL)
        self.ctrl_read_btn.grid(row=1, column=4, padx=(6, 0))
        self.ctrl_status_label = self.small_label(card, "", anchor="w", justify="left", wraplength=340)

        camera_actions = ctk.CTkFrame(card, fg_color="transparent")
        camera_actions.grid(row=6, column=1, sticky="ew", padx=(0, 14), pady=(0, 5))
        camera_actions.grid_columnconfigure(0, weight=1)
        self.refresh_cam_btn = self.make_button(
            camera_actions,
            "REFRESH DEVICES",
            self.refresh_cameras,
            height=30,
            font=FONT_UI_SMALL,
        )
        self.refresh_cam_btn.grid(row=0, column=0, sticky="ew")

        path_frame = ctk.CTkFrame(card, fg_color="transparent")
        path_frame.grid(row=7, column=1, sticky="ew", padx=(0, 14), pady=(5, 5))
        path_frame.grid_columnconfigure(0, weight=1)

        self.dest_path_var = tk.StringVar(value=os.path.expanduser("~/recordings"))
        self.dest_entry = self.make_entry(path_frame, textvariable=self.dest_path_var)
        self.dest_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.browse_btn = self.make_button(
            path_frame,
            "BROWSE",
            self.browse_destination,
            width=76,
            height=34,
            font=FONT_UI_SMALL,
        )
        self.browse_btn.grid(row=0, column=1)

        ctk.CTkLabel(
            card,
            text="Save path",
            font=FONT_UI_SMALL,
            text_color=COLORS["text_dim"],
        ).grid(row=7, column=0, sticky="w", padx=(14, 8), pady=(5, 5))

        opts = ctk.CTkFrame(card, fg_color="transparent")
        opts.grid(row=8, column=0, columnspan=2, sticky="ew", padx=14, pady=(4, 14))
        self.resume_var = tk.BooleanVar(value=True)
        self.autostart_var = tk.BooleanVar(value=os.path.exists(AUTOSTART_PATH))
        self.resume_check = self.make_check(opts, "Resume unfinished capture on launch", self.resume_var)
        self.resume_check.grid(row=0, column=0, sticky="w")
        self.autostart_check = self.make_check(opts, "Launch at login", self.autostart_var, self._toggle_autostart)
        self.autostart_check.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._on_ctrl_mode_change(initial=True)

    # ------------------------------------------------------------------
    # Video capture configuration card
    # ------------------------------------------------------------------
    def _build_capture_card(self):
        self.settings_card = self.make_card(self.left_panel)
        self.settings_card.grid_columnconfigure(1, weight=1)

        self.section_label(self.settings_card, "Video Configuration").grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(12, 8)
        )

        self.res_var = tk.StringVar(value="3840x2160 (4K)")
        self.res_combo = self.make_combo(self.settings_card, RESOLUTIONS, self.res_var)
        self.add_labeled_control(self.settings_card, 1, "Resolution", self.res_combo)
        self.res_combo.configure(command=lambda _: self.update_storage_stats(schedule_next=False))

        self.fps_var = tk.StringVar(value="30")
        self.fps_combo = self.make_combo(self.settings_card, ["30", "60", "24", "15"], self.fps_var)
        self.add_labeled_control(self.settings_card, 2, "Framerate", self.fps_combo)

        self.codec_var = tk.StringVar(value="libx264 (Standard CPU)")
        self.codec_combo = self.make_combo(
            self.settings_card,
            [
                "libx264 (Standard CPU)",
                "libx265 (High Compression)",
                "h264_vaapi (Intel GPU)",
            ],
            self.codec_var,
        )
        self.add_labeled_control(self.settings_card, 3, "Codec", self.codec_combo)

        self.preset_var = tk.StringVar(value="veryfast")
        self.preset_combo = self.make_combo(
            self.settings_card,
            ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"],
            self.preset_var,
        )
        self.add_labeled_control(self.settings_card, 4, "Preset", self.preset_combo)

        self.crf_var = tk.StringVar(value="23")
        self.crf_spin = self.make_entry(self.settings_card, textvariable=self.crf_var)
        self.add_labeled_control(self.settings_card, 5, "Quality / CRF", self.crf_spin)

        self.chunk_var = tk.StringVar(value="30")
        self.chunk_spin = self.make_entry(self.settings_card, textvariable=self.chunk_var)
        self.add_labeled_control(self.settings_card, 6, "Chunk / min", self.chunk_spin)

        self.video_prefix_var = tk.StringVar(value="trackingWheel")
        self.video_prefix_entry = self.make_entry(self.settings_card, textvariable=self.video_prefix_var)
        self.add_labeled_control(self.settings_card, 7, "File prefix", self.video_prefix_entry)
        self.video_name_label = self.small_label(self.settings_card, "", anchor="w", justify="left")
        self.video_name_label.grid(row=8, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 2))
        self.video_prefix_var.trace_add("write", lambda *_: self._update_video_name_example())
        self.chunk_var.trace_add("write", lambda *_: self._update_video_name_example())

        self.divider(self.settings_card, 9)
        self.section_label(self.settings_card, "Thermal video").grid(
            row=10, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 4)
        )

        self.thermal_fps_var = tk.StringVar(value="25")
        self.thermal_fps_combo = self.make_combo(
            self.settings_card, ["25", "15", "10", "5", "2", "1"], self.thermal_fps_var,
            command=lambda _v: self.update_storage_stats(schedule_next=False),
        )
        self.add_labeled_control(self.settings_card, 11, "Thermal fps", self.thermal_fps_combo)

        self.thermal_video_var = tk.StringVar(value="Colour video (H.264)")
        self.thermal_video_combo = self.make_combo(
            self.settings_card, list(THERMAL_VIDEO_CHOICES), self.thermal_video_var, state="readonly",
            command=lambda _v: self.update_storage_stats(schedule_next=False),
        )
        self.add_labeled_control(self.settings_card, 12, "Thermal output", self.thermal_video_combo)

        self.thermal_overlay_var = tk.BooleanVar(value=True)
        self.thermal_overlay_check = self.make_check(
            self.settings_card, "Burn in scale bar, time & min/max", self.thermal_overlay_var
        )
        self.thermal_overlay_check.grid(row=13, column=1, sticky="w", padx=(0, 14), pady=(2, 14))
        self._update_video_name_example()

    # ------------------------------------------------------------------
    # Photo (time-lapse) configuration card
    # ------------------------------------------------------------------
    def _build_photo_card(self):
        card = self.make_card(self.left_panel)
        card.grid_columnconfigure(1, weight=1)
        self.photo_card = card

        self.section_label(card, "Photo Configuration").grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(12, 8)
        )

        self.photo_res_var = tk.StringVar(value="3840x2160 (4K)")
        self.photo_res_combo = self.make_combo(
            card, RESOLUTIONS, self.photo_res_var,
            command=lambda _v: self.update_storage_stats(schedule_next=False),
        )
        self.add_labeled_control(card, 1, "Resolution", self.photo_res_combo)

        fmt_row = ctk.CTkFrame(card, fg_color="transparent")
        fmt_row.grid_columnconfigure(0, weight=1)
        self.photo_fmt_var = tk.StringVar(value="JPG")
        self.photo_fmt_combo = self.make_combo(
            fmt_row, ["JPG", "PNG"], self.photo_fmt_var, state="readonly", width=90,
            command=lambda _v: self.update_storage_stats(schedule_next=False),
        )
        self.photo_fmt_combo.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.small_label(fmt_row, "Q").grid(row=0, column=1, padx=(0, 4))
        self.jpeg_q_var = tk.StringVar(value="95")
        self.jpeg_q_entry = self.make_entry(fmt_row, textvariable=self.jpeg_q_var, width=52)
        self.jpeg_q_entry.grid(row=0, column=2)
        self.add_labeled_control(card, 2, "Format", fmt_row)

        self.prefix_var = tk.StringVar(value="frame")
        self.prefix_entry = self.make_entry(card, textvariable=self.prefix_var)
        self.add_labeled_control(card, 3, "File prefix", self.prefix_entry)

        self.day_folders_var = tk.BooleanVar(value=False)
        self.day_folders_check = self.make_check(card, "One sub-folder per day", self.day_folders_var)
        self.day_folders_check.grid(row=4, column=1, sticky="w", padx=(0, 14), pady=(2, 2))

        self.name_example_label = self.small_label(card, "", anchor="w", justify="left")
        self.name_example_label.grid(row=5, column=0, columnspan=2, sticky="ew", padx=14, pady=(2, 0))
        self.prefix_var.trace_add("write", lambda *_: self._update_name_example())
        self.photo_fmt_var.trace_add("write", lambda *_: self._update_name_example())

        self.divider(card, 6)
        self.section_label(card, "Schedule").grid(
            row=7, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 4)
        )

        self.shots_var = tk.StringVar(value="2")
        self.shots_combo = self.make_combo(
            card, SHOTS_PRESETS, self.shots_var,
            command=lambda _v: self.on_schedule_change(),
        )
        self.shots_combo.bind("<Return>", lambda _e: self.on_schedule_change())
        self.shots_combo.bind("<FocusOut>", lambda _e: self.on_schedule_change())
        self.add_labeled_control(card, 8, "Frames / day", self.shots_combo)

        self.sched_mode_var = tk.StringVar(value="Constant step")
        self.sched_mode_seg = ctk.CTkSegmentedButton(
            card,
            values=["Constant step", "Manual times"],
            variable=self.sched_mode_var,
            command=lambda _v: self.on_schedule_change(),
            font=FONT_UI_SMALL,
            height=30,
            corner_radius=7,
            fg_color=COLORS["panel_alt"],
            selected_color=COLORS["cyan_dark"],
            selected_hover_color="#155E75",
            unselected_color=COLORS["panel_alt"],
            unselected_hover_color=COLORS["border"],
            text_color=COLORS["text"],
        )
        self.add_labeled_control(card, 9, "Mode", self.sched_mode_seg)

        self.start_time_var = tk.StringVar(value="08:00:00")
        self.start_time_entry = self.make_entry(card, textvariable=self.start_time_var)
        self.start_time_entry.bind("<Return>", lambda _e: self.on_schedule_change())
        self.start_time_entry.bind("<FocusOut>", lambda _e: self.on_schedule_change())
        self.start_time_label = self.add_labeled_control(card, 10, "First shot", self.start_time_entry)

        self.times_box = ctk.CTkTextbox(
            card,
            height=120,
            fg_color=COLORS["panel_alt"],
            border_color=COLORS["border"],
            border_width=1,
            text_color=COLORS["text"],
            font=FONT_MONO,
            corner_radius=7,
            wrap="none",
        )
        self.times_box.bind("<Escape>", lambda _e: self.root.focus_set())
        self.times_box.bind("<KeyRelease>", lambda _e: self._validate_manual_times())
        self.times_label = ctk.CTkLabel(
            card, text="Times\n(HH:MM[:SS],\none per line)", font=FONT_UI_SMALL,
            text_color=COLORS["text_dim"], anchor="nw", justify="left",
        )
        self.fill_times_btn = self.make_button(
            card, "FILL FROM FRAMES / DAY", self.fill_manual_times, height=28, font=FONT_UI_SMALL,
        )

        self.schedule_info_label = self.small_label(card, "", color=COLORS["cyan"], anchor="w", justify="left")
        self.schedule_info_label.grid(row=14, column=0, columnspan=2, sticky="ew", padx=14, pady=(4, 14))

        self._update_name_example()

    def _update_name_example(self):
        prefix = self._safe_prefix()
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ext = self.photo_fmt_var.get().lower() if hasattr(self, "photo_fmt_var") else "jpg"
        self.name_example_label.configure(
            text=f"{prefix}_{stamp}_camera.{ext}\n{prefix}_{stamp}_thermal.png\n"
                 f"{prefix}_{stamp}_thermal_sensorC.npy   (sort by name)"
        )

    def _safe_prefix(self):
        prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", self.prefix_var.get().strip()) or "frame"
        return prefix

    # ------------------------------------------------------------------
    # Thermal camera card (shared)
    # ------------------------------------------------------------------
    def _build_thermal_card(self):
        card = self.make_card(self.left_panel)
        card.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        card.grid_columnconfigure(1, weight=1)
        self.thermal_card = card

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(12, 8))
        header.grid_columnconfigure(0, weight=1)
        self.section_label(header, "Thermal Camera").grid(row=0, column=0, sticky="w")
        self.small_label(header, "TC001 / TS001", color=COLORS["orange"]).grid(row=0, column=1, sticky="e")

        self.palette_var = tk.StringVar(value="inferno")
        self.palette_combo = self.make_combo(
            card, COLORMAP_NAMES, self.palette_var, state="readonly",
            command=self.on_palette_change,
        )
        self.add_labeled_control(card, 1, "Palette", self.palette_combo)

        self.span_mode_var = tk.StringVar(value="Auto")
        self.span_mode_seg = ctk.CTkSegmentedButton(
            card,
            values=list(SPAN_CHOICES),
            variable=self.span_mode_var,
            command=self.on_span_mode_change,
            font=FONT_UI_SMALL,
            height=30,
            corner_radius=7,
            fg_color=COLORS["panel_alt"],
            selected_color=COLORS["cyan_dark"],
            selected_hover_color="#155E75",
            unselected_color=COLORS["panel_alt"],
            unselected_hover_color=COLORS["border"],
            text_color=COLORS["text"],
        )
        self.add_labeled_control(card, 2, "Span mode", self.span_mode_seg)

        span_row = ctk.CTkFrame(card, fg_color="transparent")
        span_row.grid_columnconfigure(0, weight=1)
        span_row.grid_columnconfigure(2, weight=1)
        self.span_lo_var = tk.StringVar(value="20.0")
        self.span_hi_var = tk.StringVar(value="40.0")
        self.span_lo_entry = self.make_entry(span_row, textvariable=self.span_lo_var, width=64)
        self.span_lo_entry.grid(row=0, column=0, sticky="ew")
        self.small_label(span_row, "→").grid(row=0, column=1, padx=4)
        self.span_hi_entry = self.make_entry(span_row, textvariable=self.span_hi_var, width=64)
        self.span_hi_entry.grid(row=0, column=2, sticky="ew")
        self.span_apply_btn = self.make_button(
            span_row, "SET", self.apply_span_entries, width=50, height=34, font=FONT_UI_SMALL
        )
        self.span_apply_btn.grid(row=0, column=3, padx=(6, 0))
        for entry in (self.span_lo_entry, self.span_hi_entry):
            entry.bind("<Return>", lambda _e: self.apply_span_entries())
        self.span_row_label = self.add_labeled_control(card, 3, "Span °C", span_row)

        self.divider(card, 4)

        self.correction_var = tk.StringVar(value="Off")
        self.correction_combo = self.make_combo(
            card, list(CORRECTION_CHOICES), self.correction_var, state="readonly",
            command=self.on_correction_change,
        )
        self.add_labeled_control(card, 5, "Correction", self.correction_combo)

        og_row = ctk.CTkFrame(card, fg_color="transparent")
        og_row.grid_columnconfigure(1, weight=1)
        og_row.grid_columnconfigure(3, weight=1)
        self.small_label(og_row, "OFFSET").grid(row=0, column=0, padx=(0, 4))
        self.offset_var = tk.StringVar(value="0.0")
        self.offset_entry = self.make_entry(og_row, textvariable=self.offset_var, width=60)
        self.offset_entry.grid(row=0, column=1, sticky="ew")
        self.small_label(og_row, "GAIN").grid(row=0, column=2, padx=(8, 4))
        self.gain_var = tk.StringVar(value="1.0")
        self.gain_entry = self.make_entry(og_row, textvariable=self.gain_var, width=60)
        self.gain_entry.grid(row=0, column=3, sticky="ew")
        for entry in (self.offset_entry, self.gain_entry):
            entry.bind("<Return>", lambda _e: self.apply_offset_gain())
            entry.bind("<FocusOut>", lambda _e: self.apply_offset_gain())
        self.offset_row = og_row
        self.offset_row_label = self.add_labeled_control(card, 6, "°C", og_row)

        self.calib_label = self.small_label(card, "correction off", anchor="w", justify="left", wraplength=340)
        self.calib_label.grid(row=7, column=0, columnspan=2, sticky="ew", padx=14, pady=(2, 4))

        calib_btns = ctk.CTkFrame(card, fg_color="transparent")
        calib_btns.grid(row=8, column=0, columnspan=2, sticky="ew", padx=14, pady=(2, 4))
        for i in range(4):
            calib_btns.grid_columnconfigure(i, weight=1)
        self.make_button(calib_btns, "FIT MODE", self.cycle_fit_mode, height=30, font=FONT_UI_SMALL).grid(
            row=0, column=0, sticky="ew", padx=(0, 3))
        self.make_button(calib_btns, "CLEAR PTS", self.clear_points, height=30, font=FONT_UI_SMALL).grid(
            row=0, column=1, sticky="ew", padx=3)
        self.make_button(calib_btns, "LOAD", self.load_thermal_settings, height=30, font=FONT_UI_SMALL).grid(
            row=0, column=2, sticky="ew", padx=3)
        self.make_button(calib_btns, "SAVE", self.save_thermal_settings, height=30, font=FONT_UI_SMALL).grid(
            row=0, column=3, sticky="ew", padx=(3, 0))

        self.small_label(
            card,
            "Click the thermal preview to add a reference point,\n"
            "K to type its true temperature. H for all shortcuts.",
            anchor="w", justify="left",
        ).grid(row=9, column=0, columnspan=2, sticky="ew", padx=14, pady=(2, 8))

        # Photo-only thermal outputs
        self.thermal_photo_frame = ctk.CTkFrame(card, fg_color="transparent")
        self.thermal_photo_frame.grid_columnconfigure(1, weight=1)
        tp = self.thermal_photo_frame
        ctk.CTkFrame(tp, height=1, fg_color=COLORS["border"]).grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 8))
        self.section_label(tp, "Thermal photo output").grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 4))
        self.thermal_image_var = tk.BooleanVar(value=True)
        self.thermal_bar_var = tk.BooleanVar(value=True)
        checks = ctk.CTkFrame(tp, fg_color="transparent")
        self.make_check(checks, "Colour PNG", self.thermal_image_var,
                        lambda: self.update_storage_stats(schedule_next=False)).grid(row=0, column=0, sticky="w")
        self.make_check(checks, "Scale bar + time", self.thermal_bar_var).grid(row=0, column=1, sticky="w", padx=(12, 0))
        self.add_labeled_control(tp, 2, "Image", checks)
        self.thermal_data_var = tk.StringVar(value="NPY  (°C float32)")
        self.thermal_data_combo = self.make_combo(
            tp, list(THERMAL_DATA_CHOICES), self.thermal_data_var, state="readonly",
            command=lambda _v: self.update_storage_stats(schedule_next=False),
        )
        self.add_labeled_control(tp, 3, "Temp. data", self.thermal_data_combo)
        self.thermal_avg_var = tk.StringVar(value="5")
        avg_row = ctk.CTkFrame(tp, fg_color="transparent")
        self.thermal_avg_entry = self.make_entry(avg_row, textvariable=self.thermal_avg_var, width=60)
        self.thermal_avg_entry.grid(row=0, column=0)
        self.small_label(avg_row, "frames averaged per shot (noise ↓)").grid(row=0, column=1, padx=(6, 0))
        self.add_labeled_control(tp, 4, "Average", avg_row)
        self.corrected_copy_var = tk.BooleanVar(value=False)
        self.make_check(
            tp, "Also save a corrected copy", self.corrected_copy_var,
            lambda: self.update_storage_stats(schedule_next=False),
        ).grid(row=5, column=1, sticky="w", padx=(0, 14), pady=(2, 0))
        self.small_label(
            tp, "Data files hold the UNCORRECTED sensor °C; the correction\n"
                "used for every shot is in capture_log.csv.",
            anchor="w", justify="left",
        ).grid(row=6, column=0, columnspan=2, sticky="ew", padx=14, pady=(4, 10))

        self._refresh_thermal_controls()

    # ------------------------------------------------------------------
    # Storage card
    # ------------------------------------------------------------------
    def _build_storage_card(self):
        self.storage_card = self.make_card(self.left_panel)
        self.storage_card.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.storage_card.grid_columnconfigure(0, weight=1)

        self.section_label(self.storage_card, "Storage").grid(
            row=0, column=0, sticky="ew", padx=14, pady=(12, 6)
        )

        stats_row = ctk.CTkFrame(self.storage_card, fg_color="transparent")
        stats_row.grid(row=1, column=0, sticky="ew", padx=14)
        stats_row.grid_columnconfigure(0, weight=1)
        stats_row.grid_columnconfigure(1, weight=1)

        self.storage_free_big = ctk.CTkLabel(
            stats_row,
            text="-- GB",
            font=FONT_MONO_BIG,
            text_color=COLORS["text"],
            anchor="w",
        )
        self.storage_free_big.grid(row=0, column=0, sticky="w")

        self.time_left_big = ctk.CTkLabel(
            stats_row,
            text="-- DAYS",
            font=FONT_MONO_BIG,
            text_color=COLORS["cyan"],
            anchor="e",
        )
        self.time_left_big.grid(row=0, column=1, sticky="e")

        self.storage_label = ctk.CTkLabel(
            stats_row,
            text="FREE SPACE",
            font=FONT_MONO_SMALL,
            text_color=COLORS["text_dim"],
            anchor="w",
        )
        self.storage_label.grid(row=1, column=0, sticky="w")

        self.time_left_label = ctk.CTkLabel(
            stats_row,
            text="EST. RECORDING CAPACITY",
            font=FONT_MONO_SMALL,
            text_color=COLORS["text_dim"],
            anchor="e",
        )
        self.time_left_label.grid(row=1, column=1, sticky="e")

        self.storage_bar = ctk.CTkProgressBar(
            self.storage_card,
            height=9,
            corner_radius=5,
            fg_color=COLORS["bar_bg"],
            progress_color=COLORS["cyan"],
        )
        self.storage_bar.grid(row=2, column=0, sticky="ew", padx=14, pady=(10, 4))
        self.storage_bar.set(0)

        self.storage_detail_label = ctk.CTkLabel(
            self.storage_card,
            text="Drive status unavailable",
            font=FONT_MONO_SMALL,
            text_color=COLORS["text_dim"],
            anchor="w",
        )
        self.storage_detail_label.grid(row=3, column=0, sticky="ew", padx=14)

        divider = ctk.CTkFrame(self.storage_card, height=1, fg_color=COLORS["border"])
        divider.grid(row=4, column=0, sticky="ew", padx=14, pady=10)

        calc = ctk.CTkFrame(self.storage_card, fg_color="transparent")
        calc.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 12))
        calc.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(
            calc,
            text="Estimate",
            font=FONT_UI_SMALL,
            text_color=COLORS["text_dim"],
        ).grid(row=0, column=0, padx=(0, 7))

        self.days_input = self.make_entry(calc, width=62)
        self.days_input.insert(0, "42")
        self.days_input.grid(row=0, column=1, padx=(0, 7))

        self.calc_btn = self.make_button(
            calc,
            "CALCULATE",
            self.calculate_required_storage,
            height=34,
            font=FONT_UI_SMALL,
        )
        self.calc_btn.grid(row=0, column=2, sticky="ew")

        self.calc_result_label = ctk.CTkLabel(
            calc,
            text="",
            font=FONT_MONO_SMALL,
            text_color=COLORS["cyan"],
            anchor="w",
        )
        self.calc_result_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

    # ------------------------------------------------------------------
    # Recording card (video tab)
    # ------------------------------------------------------------------
    def _build_recording_card(self):
        self.action_card = self.make_card(self.left_panel)
        self.action_card.grid_columnconfigure(0, weight=1)

        self.section_label(self.action_card, "Recording").grid(
            row=0, column=0, sticky="ew", padx=14, pady=(12, 4)
        )

        self.uptime_label = ctk.CTkLabel(
            self.action_card,
            text="00:00:00",
            font=("DejaVu Sans Mono", 28, "bold"),
            text_color=COLORS["text"],
        )
        self.uptime_label.grid(row=1, column=0, padx=14, pady=(4, 0))

        self.uptime_sub_label = ctk.CTkLabel(
            self.action_card,
            text="STANDBY",
            font=FONT_MONO_SMALL,
            text_color=COLORS["text_dim"],
        )
        self.uptime_sub_label.grid(row=2, column=0, padx=14, pady=(0, 2))

        self.thermal_rec_label = self.small_label(self.action_card, "")
        self.thermal_rec_label.grid(row=3, column=0, padx=14, pady=(0, 7))

        self.record_btn = self.make_button(
            self.action_card,
            "●  START RECORDING",
            self.toggle_recording,
            accent=True,
            height=48,
        )
        self.record_btn.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 8))

        self.telemetry_btn = self.make_button(
            self.action_card,
            "SHOW CPU / RAM",
            self.toggle_system_telemetry,
            height=38,
            font=FONT_UI_SMALL,
        )
        self.telemetry_btn.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 14))

    # ------------------------------------------------------------------
    # Time-lapse card (photo tab)
    # ------------------------------------------------------------------
    def _build_timelapse_card(self):
        card = self.make_card(self.left_panel)
        card.grid_columnconfigure(0, weight=1)
        card.grid_columnconfigure(1, weight=1)
        self.timelapse_card = card

        self.section_label(card, "Time-lapse").grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(12, 4)
        )

        self.countdown_label = ctk.CTkLabel(
            card,
            text="--:--:--",
            font=("DejaVu Sans Mono", 28, "bold"),
            text_color=COLORS["text"],
        )
        self.countdown_label.grid(row=1, column=0, columnspan=2, padx=14, pady=(4, 0))

        self.countdown_sub_label = ctk.CTkLabel(
            card,
            text="STANDBY",
            font=FONT_MONO_SMALL,
            text_color=COLORS["text_dim"],
        )
        self.countdown_sub_label.grid(row=2, column=0, columnspan=2, padx=14, pady=(0, 2))

        self.last_shot_label = self.small_label(card, "NO SHOTS YET", wraplength=340, justify="center")
        self.last_shot_label.grid(row=3, column=0, columnspan=2, padx=14, pady=(0, 8))

        self.timelapse_btn = self.make_button(
            card,
            "●  START TIME-LAPSE",
            self.toggle_timelapse,
            accent=True,
            height=48,
        )
        self.timelapse_btn.grid(row=4, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 8))

        self.capture_now_btn = self.make_button(
            card, "CAPTURE NOW", self.capture_now, height=38, font=FONT_UI_SMALL,
        )
        self.capture_now_btn.grid(row=5, column=0, sticky="ew", padx=(14, 4), pady=(0, 14))

        self.telemetry_btn_photo = self.make_button(
            card, "SHOW CPU / RAM", self.toggle_system_telemetry, height=38, font=FONT_UI_SMALL,
        )
        self.telemetry_btn_photo.grid(row=5, column=1, sticky="ew", padx=(4, 14), pady=(0, 14))

    # ------------------------------------------------------------------
    # Camera preview card (dual: camera | thermal)
    # ------------------------------------------------------------------
    def _build_preview_card(self):
        self.preview_card = self.make_card(self.right_panel)
        self.preview_card.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        self.preview_card.grid_columnconfigure(0, weight=1)
        self.preview_card.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(self.preview_card, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 8))
        header.grid_columnconfigure(0, weight=1)

        self.section_label(header, "Camera Preview").grid(row=0, column=0, sticky="w")
        self.preview_mode_label = ctk.CTkLabel(
            header,
            text="OFFLINE",
            font=FONT_MONO_SMALL,
            text_color=COLORS["text_dim"],
        )
        self.preview_mode_label.grid(row=0, column=1, sticky="e")

        self.preview_controls = ctk.CTkFrame(self.preview_card, fg_color="transparent")
        self.preview_controls.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        controls = self.preview_controls
        controls.grid_columnconfigure(5, weight=1)

        ctk.CTkLabel(controls, text="RATE", font=FONT_MONO_SMALL, text_color=COLORS["text_dim"]).grid(
            row=0, column=0, padx=(0, 5)
        )
        self.snap_rate_var = tk.StringVar(value="60")
        self.snap_rate_spin = self.make_entry(controls, textvariable=self.snap_rate_var, width=60, height=30)
        self.snap_rate_spin.grid(row=0, column=1, padx=(0, 5))
        ctk.CTkLabel(controls, text="img/min", font=FONT_MONO_SMALL, text_color=COLORS["text_dim"]).grid(
            row=0, column=2, padx=(0, 16)
        )

        ctk.CTkLabel(controls, text="DURATION", font=FONT_MONO_SMALL, text_color=COLORS["text_dim"]).grid(
            row=0, column=3, padx=(0, 5)
        )
        self.snap_duration_var = tk.StringVar(value="15")
        self.snap_duration_spin = self.make_entry(controls, textvariable=self.snap_duration_var, width=60, height=30)
        self.snap_duration_spin.grid(row=0, column=4, padx=(0, 5))
        ctk.CTkLabel(controls, text="sec", font=FONT_MONO_SMALL, text_color=COLORS["text_dim"]).grid(
            row=0, column=5, sticky="w"
        )

        self.photo_preview_info = self.small_label(self.preview_card, "", anchor="w")

        # Dual display area
        self.display_area = tk.Frame(self.preview_card, bg=COLORS["panel"], bd=0, highlightthickness=0)
        self.display_area.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 10))
        self.display_area.grid_rowconfigure(1, weight=1)

        self.cam_pane_title = tk.Label(
            self.display_area, text="CAMERA", bg=COLORS["panel"], fg=COLORS["text_dim"],
            font=FONT_MONO_SMALL, anchor="w",
        )
        self.preview_display = tk.Label(
            self.display_area,
            bg=COLORS["black"],
            fg=COLORS["text_dim"],
            text="CAMERA PREVIEW OFFLINE\n\nSelect IMAGE PREVIEW or LIVE STREAM",
            font=FONT_MONO,
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        self.thermal_pane_title = tk.Label(
            self.display_area, text="THERMAL", bg=COLORS["panel"], fg=COLORS["orange"],
            font=FONT_MONO_SMALL, anchor="w",
        )
        self.thermal_display = tk.Label(
            self.display_area,
            bg=COLORS["black"],
            fg=COLORS["text_dim"],
            text="THERMAL PREVIEW OFFLINE",
            font=FONT_MONO,
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            cursor="crosshair",
        )
        self.thermal_display.bind("<Button-1>", self.on_thermal_click)
        self.thermal_display.bind("<Double-Button-1>", lambda _e: self.prompt_point_value())
        self.thermal_display.bind("<Button-3>", self.on_thermal_right_click)
        self.thermal_display.bind("<Motion>", self.on_thermal_motion)
        self.thermal_display.bind("<Leave>", lambda _e: setattr(self, "_thermal_cursor", None))
        self.preview_display.bind("<Button-1>", lambda _e: self.root.focus_set())

        preview_ctrls = ctk.CTkFrame(self.preview_card, fg_color="transparent")
        preview_ctrls.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 14))
        preview_ctrls.grid_columnconfigure(0, weight=1)
        preview_ctrls.grid_columnconfigure(1, weight=1)

        self.snap_btn = self.make_button(
            preview_ctrls,
            "IMAGE PREVIEW",
            self.toggle_image_preview,
            height=38,
        )
        self.snap_btn.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.stream_btn = self.make_button(
            preview_ctrls,
            "LIVE STREAM",
            self.toggle_live_stream,
            accent=True,
            height=38,
        )
        self.stream_btn.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        self._layout_preview_panes()

    def _layout_preview_panes(self):
        """Show the camera and/or thermal pane depending on what is enabled."""
        cam = self.use_camera_var.get()
        th = self.use_thermal_var.get()
        for w in (self.cam_pane_title, self.preview_display, self.thermal_pane_title, self.thermal_display):
            w.grid_forget()
        for c in range(2):
            self.display_area.grid_columnconfigure(c, weight=0, uniform="")
        col = 0
        panes = []
        if cam:
            panes.append((self.cam_pane_title, self.preview_display, 1))
        if th:
            panes.append((self.thermal_pane_title, self.thermal_display, 1))
        if not panes:
            panes.append((self.cam_pane_title, self.preview_display, 1))
        for title, display, weight in panes:
            self.display_area.grid_columnconfigure(col, weight=weight, uniform="panes")
            title.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 6, 0), pady=(0, 3))
            display.grid(row=1, column=col, sticky="nsew", padx=(0 if col == 0 else 6, 0))
            col += 1

    # ------------------------------------------------------------------
    # Telemetry card
    # ------------------------------------------------------------------
    def _build_telemetry_card(self):
        self.telemetry_card = self.make_card(self.right_panel)
        self.telemetry_card.grid_columnconfigure(0, weight=1)
        self.telemetry_card.grid_columnconfigure(1, weight=1)
        self.telemetry_card.grid_rowconfigure(1, weight=1)

        title_row = ctk.CTkFrame(self.telemetry_card, fg_color="transparent")
        title_row.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(12, 6))
        title_row.grid_columnconfigure(0, weight=1)
        self.section_label(title_row, "System Telemetry").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            title_row,
            text="● ON DEMAND  /  60 SAMPLES",
            font=FONT_MONO_SMALL,
            text_color=COLORS["green"],
        ).grid(row=0, column=1, sticky="e")

        self.cpu_panel = self._build_metric_panel(
            self.telemetry_card,
            column=0,
            title="CPU LOAD",
            color=COLORS["cyan"],
        )
        self.ram_panel = self._build_metric_panel(
            self.telemetry_card,
            column=1,
            title="RAM USED",
            color=COLORS["green"],
        )

    def _build_metric_panel(self, parent, column, title, color):
        panel = ctk.CTkFrame(
            parent,
            fg_color=COLORS["panel_alt"],
            corner_radius=9,
            border_width=1,
            border_color=COLORS["border"],
        )
        panel.grid(
            row=1,
            column=column,
            sticky="nsew",
            padx=(14 if column == 0 else 5, 5 if column == 0 else 14),
            pady=(0, 14),
        )
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(3, weight=1)

        header = ctk.CTkFrame(panel, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text=title,
            font=FONT_MONO_SMALL,
            text_color=COLORS["text_dim"],
        ).grid(row=0, column=0, sticky="w")

        value_label = ctk.CTkLabel(
            header,
            text="0.0%",
            font=("DejaVu Sans Mono", 16, "bold"),
            text_color=color,
        )
        value_label.grid(row=0, column=1, sticky="e")

        bar = ctk.CTkProgressBar(
            panel,
            height=8,
            corner_radius=4,
            fg_color=COLORS["bar_bg"],
            progress_color=color,
        )
        bar.grid(row=1, column=0, sticky="ew", padx=12, pady=(2, 8))
        bar.set(0)

        summary_label = ctk.CTkLabel(
            panel,
            text="MIN 0.0   AVG 0.0   MAX 0.0",
            font=FONT_MONO_SMALL,
            text_color=COLORS["text_dim"],
            anchor="w",
        )
        summary_label.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 4))

        canvas = tk.Canvas(
            panel,
            bg=COLORS["panel_alt"],
            highlightthickness=0,
            bd=0,
            height=86,
        )
        canvas.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 9))
        canvas.bind("<Configure>", lambda _e: self.draw_telemetry_charts())

        return {
            "frame": panel,
            "value": value_label,
            "bar": bar,
            "summary": summary_label,
            "canvas": canvas,
            "color": color,
        }

    # ------------------------------------------------------------------
    # Inline alert banner
    # ------------------------------------------------------------------
    def _build_alert_banner(self):
        self.alert_banner = ctk.CTkFrame(
            self.root,
            fg_color="#321416",
            border_width=1,
            border_color=COLORS["red"],
            corner_radius=8,
        )
        self.alert_label = ctk.CTkLabel(
            self.alert_banner,
            text="",
            font=FONT_UI_BOLD,
            text_color="#FFD7D5",
            anchor="w",
            justify="left",
        )
        self.alert_label.pack(fill="x", padx=12, pady=8)

    def show_alert(self, message, kind="error", auto_hide_ms=None, quiet=False):
        colors = {
            "error": ("#321416", COLORS["red"], "#FFD7D5"),
            "warning": ("#332A12", COLORS["amber"], "#FFE7A3"),
            "info": ("#0B2A33", COLORS["cyan"], "#C6F6FF"),
        }
        fg, border, text = colors.get(kind, colors["error"])
        if kind in ("error", "warning") and not quiet:
            log.log(40 if kind == "error" else 30, "alert: %s", " ".join(str(message).split()))
        self.alert_banner.configure(fg_color=fg, border_color=border)
        self.alert_label.configure(text=message, text_color=text)
        self.alert_banner.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 10))
        self.alert_banner.lift()
        if self._alert_after_id is not None:
            try:
                self.root.after_cancel(self._alert_after_id)
            except Exception:
                pass
            self._alert_after_id = None
        if auto_hide_ms:
            self._alert_after_id = self.root.after(auto_hide_ms, self.hide_alert)

    def hide_alert(self):
        self._alert_after_id = None
        self.alert_banner.grid_forget()

    def notify(self, message, ms=2500):
        self.show_alert(message, "info", ms)

    # ------------------------------------------------------------------
    # Tabs: VIDEO / PHOTO
    # ------------------------------------------------------------------
    def set_mode(self, mode, force=False):
        if mode == self.mode and not force:
            return
        if self.is_recording or self.is_timelapse:
            self.mode_tabs.set(self.mode.upper())
            self.show_alert("STOP THE CURRENT CAPTURE BEFORE SWITCHING TABS", "warning", 3500)
            return
        if self.is_streaming:
            self.toggle_live_stream()
        if self.is_image_previewing:
            self.toggle_image_preview()

        self.mode = mode
        self.mode_tabs.set(mode.upper())

        if mode == "video":
            self.photo_card.grid_forget()
            self.timelapse_card.grid_forget()
            self.thermal_photo_frame.grid_forget()
            self.settings_card.grid(row=1, column=0, sticky="ew", pady=(0, 10))
            self.action_card.grid(row=4, column=0, sticky="ew", pady=(0, 10))
            self.subtitle_label.configure(text="CAPTURE / RECORD / TELEMETRY")
            self.audio_sign.configure(text="AUDIO OFF  •  VIDEO ONLY")
            self.time_left_label.configure(text="EST. RECORDING CAPACITY")
            self.photo_preview_info.grid_forget()
            self.preview_controls.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
            self.snap_btn.configure(text="IMAGE PREVIEW", command=self.toggle_image_preview)
        else:
            self.settings_card.grid_forget()
            self.action_card.grid_forget()
            self.photo_card.grid(row=1, column=0, sticky="ew", pady=(0, 10))
            self.timelapse_card.grid(row=4, column=0, sticky="ew", pady=(0, 10))
            self.thermal_photo_frame.grid(row=10, column=0, columnspan=2, sticky="ew")
            self.subtitle_label.configure(text="TIME-LAPSE / DUAL CAMERA STILLS")
            self.time_left_label.configure(text="EST. TIME-LAPSE CAPACITY")
            self.preview_controls.grid_forget()
            self.photo_preview_info.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
            self.snap_btn.configure(text="SHOW LAST CAPTURE", command=self.show_last_capture)
            self.on_schedule_change()
        self._update_mode_sign()
        self.update_storage_stats(schedule_next=False)

    def _update_mode_sign(self):
        if self.mode == "photo":
            try:
                n = len(self.current_schedule())
                self.audio_sign.configure(text=f"STILLS  •  {n} / DAY")
            except ValueError:
                self.audio_sign.configure(text="STILLS  •  SCHEDULE INVALID")

    # ------------------------------------------------------------------
    # System telemetry
    # ------------------------------------------------------------------
    def _read_ram_percent(self):
        with open("/proc/meminfo", "r") as f:
            mem_total = 0
            mem_available = 0
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1])
                if mem_total and mem_available:
                    break
        if not mem_total:
            return 0.0
        return 100.0 - (mem_available / mem_total * 100.0)

    def _read_cpu_percent(self):
        with open("/proc/stat", "r") as f:
            values = list(map(int, f.readline().split()[1:8]))

        idle = values[3] + values[4]
        total = sum(values)
        current = (total, idle)

        if self._last_cpu_sample is None:
            self._last_cpu_sample = current
            return 0.0

        prev_total, prev_idle = self._last_cpu_sample
        self._last_cpu_sample = current

        total_diff = total - prev_total
        idle_diff = idle - prev_idle
        if total_diff <= 0:
            return 0.0

        return max(0.0, min(100.0, 100.0 * (total_diff - idle_diff) / total_diff))

    def toggle_system_telemetry(self):
        if self.is_telemetry_active:
            self.stop_system_telemetry()
        else:
            self.start_system_telemetry()

    def _telemetry_buttons(self):
        return (self.telemetry_btn, self.telemetry_btn_photo)

    def start_system_telemetry(self):
        if self.is_telemetry_active:
            return

        self.is_telemetry_active = True
        self.cpu_history.clear()
        self.ram_history.clear()
        self._last_cpu_sample = None

        self.right_panel.grid_rowconfigure(1, weight=2)
        self.telemetry_card.grid(row=1, column=0, sticky="nsew")
        for btn in self._telemetry_buttons():
            btn.configure(
                text="HIDE CPU / RAM",
                fg_color=COLORS["cyan_dark"],
                hover_color="#155E75",
            )

        try:
            self._read_cpu_percent()
            ram_pct = self._read_ram_percent()
            self.ram_history.append(ram_pct)
            self._update_metric_panel(self.ram_panel, self.ram_history, ram_pct)
            self.ram_panel["canvas"].after_idle(self.draw_telemetry_charts)
        except Exception as exc:
            self.show_alert(f"SYSTEM TELEMETRY ERROR  •  {exc}", "warning", 5000)

        self._telemetry_after_id = self.root.after(1000, self.update_system_metrics)

    def stop_system_telemetry(self):
        self.is_telemetry_active = False

        if self._telemetry_after_id is not None:
            try:
                self.root.after_cancel(self._telemetry_after_id)
            except Exception:
                pass
            self._telemetry_after_id = None

        self.telemetry_card.grid_forget()
        self.right_panel.grid_rowconfigure(1, weight=0)
        for btn in self._telemetry_buttons():
            btn.configure(
                text="SHOW CPU / RAM",
                fg_color=COLORS["panel_alt"],
                hover_color=COLORS["border"],
            )

        self.cpu_history.clear()
        self.ram_history.clear()
        self._last_cpu_sample = None
        self._reset_metric_panel(self.cpu_panel)
        self._reset_metric_panel(self.ram_panel)

    def update_system_metrics(self):
        if not self.is_telemetry_active:
            self._telemetry_after_id = None
            return

        try:
            cpu_pct = self._read_cpu_percent()
            ram_pct = self._read_ram_percent()

            self.cpu_history.append(cpu_pct)
            self.ram_history.append(ram_pct)

            self._update_metric_panel(self.cpu_panel, self.cpu_history, cpu_pct)
            self._update_metric_panel(self.ram_panel, self.ram_history, ram_pct)
            self.draw_telemetry_charts()
        except Exception as exc:
            self.show_alert(f"SYSTEM TELEMETRY ERROR  •  {exc}", "warning", 5000)

        if self.is_telemetry_active:
            self._telemetry_after_id = self.root.after(1000, self.update_system_metrics)
        else:
            self._telemetry_after_id = None

    @staticmethod
    def _reset_metric_panel(panel):
        panel["value"].configure(text="--.-%")
        panel["bar"].set(0)
        panel["summary"].configure(text="MIN --.-   AVG --.-   MAX --.-")
        panel["canvas"].delete("all")

    def _update_metric_panel(self, panel, history, current):
        values = list(history)
        if not values:
            return
        panel["value"].configure(text=f"{current:5.1f}%")
        panel["bar"].set(current / 100.0)
        panel["summary"].configure(
            text=f"MIN {min(values):4.1f}   AVG {sum(values)/len(values):4.1f}   MAX {max(values):4.1f}"
        )

    def draw_telemetry_charts(self):
        self._draw_chart(self.cpu_panel["canvas"], self.cpu_history, self.cpu_panel["color"])
        self._draw_chart(self.ram_panel["canvas"], self.ram_history, self.ram_panel["color"])

    @staticmethod
    def _draw_chart(canvas, history, line_color):
        canvas.delete("all")
        width = max(canvas.winfo_width(), 20)
        height = max(canvas.winfo_height(), 20)

        for pct in (25, 50, 75):
            y = height - (pct / 100.0) * height
            canvas.create_line(0, y, width, y, fill=COLORS["border"], dash=(2, 6))

        values = list(history)
        if len(values) < 2:
            return

        step = width / (len(values) - 1)
        points = []
        for i, value in enumerate(values):
            x = i * step
            y = height - (max(0.0, min(100.0, value)) / 100.0) * (height - 6) - 3
            points.extend([x, y])

        canvas.create_line(
            *points,
            fill=line_color,
            width=2,
            smooth=True,
            splinesteps=12,
        )

        x = width - 2
        y = height - (max(0.0, min(100.0, values[-1])) / 100.0) * (height - 6) - 3
        canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=line_color, outline=line_color)

    # ------------------------------------------------------------------
    # Camera discovery
    # ------------------------------------------------------------------
    @staticmethod
    def _looks_thermal(name, sizes):
        if "256x384" in sizes:
            return True
        low = name.lower()
        if any(h in low for h in NOT_THERMAL_NAME_HINTS):
            return False
        # "usb camera" alone is too generic for a normal webcam; trust it only
        # together with the tell-tale sensor geometry.
        hints = [h for h in THERMAL_NAME_HINTS if h != "usb camera"]
        return any(h in low for h in hints)

    def refresh_cameras(self):
        if self.is_recording or self.is_timelapse:
            self.show_alert("CANNOT RESCAN DEVICES WHILE CAPTURING", "warning", 3000)
            return
        self.cam_combo.configure(values=["Scanning cameras..."])
        self.cam_combo.set("Scanning cameras...")
        self.camera_status_label.configure(text="CAMERA: scanning...", text_color=COLORS["text_dim"])

        def worker():
            video_nodes = sorted(glob.glob("/dev/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0))
            found = []

            for dev in video_nodes:
                try:
                    res = subprocess.run(
                        ["v4l2-ctl", "-d", dev, "--all"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=3,
                    )
                    if "Video Capture" not in res.stdout:
                        continue

                    name = "Unknown Camera"
                    for line in res.stdout.splitlines():
                        if "Card type" in line:
                            name = line.split(":", 1)[1].strip()
                            break

                    fmt_out = subprocess.run(
                        ["v4l2-ctl", "-d", dev, "--list-formats-ext"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=3,
                    )
                    sizes = []
                    for line in fmt_out.stdout.splitlines():
                        match = re.search(r"Size: Discrete (\d+x\d+)", line)
                        if match:
                            sizes.append(match.group(1))
                    if not sizes:
                        continue          # metadata-only node
                    max_res = max(sizes, key=lambda s: int(s.split("x")[0]) * int(s.split("x")[1]))
                    thermal = self._looks_thermal(name, sizes)
                    tag = "  |  THERMAL 256x192" if thermal else f"  |  max {max_res}"
                    found.append({"path": dev, "name": name, "thermal": thermal,
                                  "label": f"{dev} - {name}{tag}"})
                except Exception:
                    continue

            self.root.after(0, lambda: self._apply_camera_results(found))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_camera_results(self, found):
        self.devices = found
        labels = [d["label"] for d in found]
        normal = [d for d in found if not d["thermal"]]
        thermals = [d for d in found if d["thermal"]]

        wanted_cam = getattr(self, "_wanted_cam", None)
        wanted_th = getattr(self, "_wanted_thermal", None)

        if labels:
            self.cam_combo.configure(values=labels)
            pick = next((d["label"] for d in found if d["path"] == wanted_cam), None)
            pick = pick or (normal[0]["label"] if normal else labels[0])
            self.cam_combo.set(pick)
        else:
            self.cam_combo.configure(values=["No Camera Found"])
            self.cam_combo.set("No Camera Found")

        th_values = ["Auto-detect"] + labels
        self.thermal_combo.configure(values=th_values)
        pick = next((d["label"] for d in found if d["path"] == wanted_th), None)
        pick = pick or (thermals[0]["label"] if thermals else "Auto-detect")
        self.thermal_combo.set(pick)
        self._update_device_status()

    def _update_device_status(self):
        cam = self.get_selected_camera_path()
        if not self.use_camera_var.get():
            self.camera_status_label.configure(text="CAMERA: disabled", text_color=COLORS["text_dim"])
        elif cam and cam.startswith("/dev/"):
            self.camera_status_label.configure(text=f"CAMERA: {cam}  •  ONLINE", text_color=COLORS["green"])
        else:
            self.camera_status_label.configure(text="CAMERA: NOT FOUND", text_color=COLORS["red"])

    def get_selected_camera_path(self):
        full_str = self.cam_combo.get()
        return full_str.split(" - ")[0] if full_str else ""

    def get_selected_thermal_path(self):
        """'/dev/videoN', or None for auto-detect."""
        full_str = self.thermal_combo.get()
        if not full_str or full_str.startswith("Auto"):
            return None
        return full_str.split(" - ")[0]

    def camera_is_valid(self):
        cam = self.get_selected_camera_path()
        return bool(cam) and cam.startswith("/dev/")

    def on_device_toggle(self):
        if self.is_recording or self.is_timelapse:
            return
        if not self.use_thermal_var.get():
            self.thermal.release("preview")
        if not self.use_camera_var.get():
            self.camera.release("preview")
        self._layout_preview_panes()
        self._update_device_status()
        self.update_storage_stats(schedule_next=False)

    def on_device_change(self):
        self._update_device_status()
        if self.is_streaming:
            # Restart the preview on the newly chosen device(s).
            self.toggle_live_stream()
            self.toggle_live_stream()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def browse_destination(self):
        folder = filedialog.askdirectory(initialdir=self.dest_path_var.get())
        if folder:
            self.dest_path_var.set(folder)
            self.update_storage_stats(schedule_next=False)

    def _thermal_video_outputs(self):
        return THERMAL_VIDEO_CHOICES.get(self.thermal_video_var.get(), ("color",))

    def _thermal_fps(self):
        try:
            return max(0.1, float(self.thermal_fps_var.get()))
        except ValueError:
            return 25.0

    def estimate_bitrate_mbps(self):
        total = 0.0
        if self.use_camera_var.get():
            res_str = self.res_var.get().split()[0]
            w, h = map(int, res_str.split("x"))
            crf = float(self.crf_var.get())
            base_pixels = 1920 * 1080
            scale = (w * h) / base_pixels
            base_mbps = 2.0 * (2 ** ((23 - crf) / 6))
            total += max(0.5, base_mbps * scale)
        if self.use_thermal_var.get():
            total += estimate_thermal_mbps(self._thermal_video_outputs(), self._thermal_fps())
        return max(total, 0.05)

    def _photo_resolution(self):
        w, h = map(int, self.photo_res_var.get().split()[0].split("x"))
        return w, h

    def photo_bytes_per_day(self):
        s = self.collect_photo_settings(validate=False)
        per_shot = self.capturer.mean_shot_bytes if self.capturer.shots >= 1 else None
        if per_shot is None:
            w, h = self._photo_resolution()
            per_shot = estimate_shot_bytes(s, w, h)
        n = len(self.current_schedule())
        return per_shot * n, per_shot, n

    def update_storage_stats(self, schedule_next=False):
        path = self.dest_path_var.get()
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
            except Exception:
                path = "/"

        try:
            total, used, free = shutil.disk_usage(path)
            free_gb = free / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            used_pct = (used / total) * 100 if total else 0

            if (self.is_recording or self.is_timelapse) and free_gb < 5.0:
                if self.is_recording:
                    self.stop_recording(reason=f"low disk: {free_gb:.1f} GB free")
                if self.is_timelapse:
                    self.stop_timelapse(reason=f"low disk: {free_gb:.1f} GB free")
                self.show_alert(
                    f"CRITICAL STORAGE  •  Capture stopped cleanly  •  Only {free_gb:.1f} GB remains",
                    "error",
                )

            if self.mode == "photo":
                try:
                    bytes_day, per_shot, n = self.photo_bytes_per_day()
                    gb_day = bytes_day / (1024 ** 3)
                    days_left = free_gb / gb_day if gb_day else float("inf")
                    self.time_left_big.configure(
                        text=f"{days_left:.0f} DAYS" if days_left < 1e5 else "∞ DAYS"
                    )
                    self.storage_detail_label.configure(
                        text=f"{used_pct:4.1f}% USED  •  {total_gb:.0f} GB  •  "
                             f"~{per_shot / 1e6:.2f} MB/SHOT  •  {gb_day * 1024:.0f} MB/DAY"
                    )
                except ValueError:
                    self.time_left_big.configure(text="-- DAYS")
                    self.storage_detail_label.configure(text=f"{used_pct:4.1f}% USED  •  SCHEDULE INVALID")
            else:
                est_mbps = self.estimate_bitrate_mbps()
                gb_per_hour = (est_mbps * 3600) / (8 * 1024)
                hours_left = free_gb / gb_per_hour if gb_per_hour else 0
                days_left = hours_left / 24
                self.time_left_big.configure(text=f"{days_left:.1f} DAYS")
                self.storage_detail_label.configure(
                    text=f"{used_pct:4.1f}% USED  •  {total_gb:.0f} GB TOTAL  •  ~{hours_left:.0f} HOURS"
                )

            self.storage_bar.set(used_pct / 100.0)
            self.storage_free_big.configure(text=f"{free_gb:.0f} GB")

            if used_pct > 90:
                self.storage_bar.configure(progress_color=COLORS["red"])
            elif used_pct > 80:
                self.storage_bar.configure(progress_color=COLORS["amber"])
            else:
                self.storage_bar.configure(progress_color=COLORS["cyan"])

        except Exception as exc:
            self.storage_free_big.configure(text="-- GB")
            self.time_left_big.configure(text="-- DAYS")
            self.storage_detail_label.configure(text=f"DRIVE ERROR  •  {exc}")

        if schedule_next:
            if self._storage_after_id is not None:
                try:
                    self.root.after_cancel(self._storage_after_id)
                except Exception:
                    pass
            self._storage_after_id = self.root.after(
                30000, lambda: self.update_storage_stats(schedule_next=True)
            )

    def calculate_required_storage(self):
        try:
            days = float(self.days_input.get())
            if days <= 0:
                raise ValueError
            if self.mode == "photo":
                bytes_day, _, _ = self.photo_bytes_per_day()
                req_gb = days * bytes_day / (1024 ** 3)
            else:
                est_mbps = self.estimate_bitrate_mbps()
                gb_per_day = (est_mbps * 3600 * 24) / (8 * 1024)
                req_gb = days * gb_per_day
            self.calc_result_label.configure(
                text=f"{days:g} DAYS  →  {req_gb:.1f} GB  /  {req_gb / 1024:.2f} TB"
            )
        except ValueError:
            self.calc_result_label.configure(text="ENTER A VALID POSITIVE NUMBER OF DAYS")

    # ------------------------------------------------------------------
    # Recording control (video tab)
    # ------------------------------------------------------------------
    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def _recording_params(self):
        res = self.res_var.get().split()[0]
        fps = self.fps_var.get()
        codec = self.codec_var.get().split()[0]
        preset = self.preset_var.get()
        crf = str(int(float(self.crf_var.get())))
        chunk_sec = int(round(float(self.chunk_var.get()) * 60))
        out_dir = self.dest_path_var.get()
        if chunk_sec < 60:
            raise ValueError
        rate_ipm = max(1, int(float(self.snap_rate_var.get())))
        return dict(res=res, fps=fps, codec=codec, preset=preset, crf=crf,
                    chunk_sec=chunk_sec, out_dir=out_dir, rate_ipm=rate_ipm,
                    prefix=self._safe_video_prefix())

    def start_recording(self, resumed=False):
        self.hide_alert()

        if self.is_streaming:
            self.toggle_live_stream()
        if self.is_image_previewing:
            self.toggle_image_preview()

        use_cam = self.use_camera_var.get()
        use_th = self.use_thermal_var.get()
        if not use_cam and not use_th:
            self.show_alert("ENABLE THE CAMERA, THE THERMAL CAMERA, OR BOTH", "error", 5000)
            return

        camera = self.get_selected_camera_path()
        if use_cam and not self.camera_is_valid():
            self.show_alert("NO VALID CAMERA SELECTED", "error", 5000)
            return
        if use_cam and use_th and camera == self.get_selected_thermal_path():
            self.show_alert("CAMERA AND THERMAL CAMERA ARE THE SAME DEVICE", "error", 5000)
            return

        try:
            p = self._recording_params()
            os.makedirs(p["out_dir"], exist_ok=True)
        except ValueError:
            self.show_alert("INVALID RECORDING PARAMETER  •  CHUNK MUST BE ≥ 1 MIN", "error", 5000)
            return
        except Exception as exc:
            self.show_alert(f"CONFIGURATION ERROR  •  {exc}", "error", 5000)
            return

        self._attach_log(p["out_dir"])
        self._rec_params = p

        if use_cam and not self._start_camera_ffmpeg():
            return

        if use_th:
            self._start_thermal_recorder(p)

        self.is_recording = True
        self.start_time = time.time()
        self._begin_capture_session("video", resumed)
        what = " + ".join(x for x, on in (("CAMERA", use_cam), ("THERMAL", use_th)) if on)
        log.info("RECORDING STARTED%s  %s  -> %s  (%s %s fps %s crf %s, chunks %d min, prefix %s)",
                 " (resumed)" if resumed else "", what, p["out_dir"], p["res"], p["fps"],
                 p["codec"], p["crf"], p["chunk_sec"] // 60, p["prefix"])
        if use_cam:
            # Once the stream runs, lock / set exposure etc. if requested.
            self.root.after(4000, lambda: self._apply_camera_controls_async(camera, fresh=True))

        self.record_btn.configure(
            text="■  STOP RECORDING",
            fg_color=COLORS["red"],
            hover_color="#C93C37",
        )
        self.status_text.configure(text="RECORDING ACTIVE", text_color=COLORS["red"])
        self.status_dot.configure(text_color=COLORS["red"])
        self.uptime_sub_label.configure(text="CAPTURE ACTIVE", text_color=COLORS["red"])
        self.set_inputs_state("disabled")
        self.show_alert(f"RECORDING STARTED  •  {what}  •  CAPTURE PIPELINE ACTIVE", "info", 2500)

    def _start_camera_ffmpeg(self):
        """Start (or restart) the camera FFmpeg. Returns True on success."""
        p = self._rec_params
        camera = self.get_selected_camera_path()
        # Chunks cut on the wall clock (every :00 / :30 for 30 min), so camera
        # and thermal chunks start together; numeric names sort by date.
        out_template = os.path.join(p["out_dir"], p["prefix"] + "_%Y-%m-%d_%H-%M-%S_camera.mkv")
        segment = ["-f", "segment", "-segment_time", str(p["chunk_sec"]),
                   "-segment_atclocktime", "1", "-reset_timestamps", "1", "-strftime", "1",
                   out_template]
        prev_fps = f"{p['rate_ipm']}/60"
        preview = ["-map", "[prev_out]", "-update", "1", "-y", self.shm_snapshot]

        cmd = ["ffmpeg", "-y", "-nostats", "-loglevel", "warning",
               "-f", "v4l2", "-framerate", p["fps"], "-video_size", p["res"],
               "-input_format", "mjpeg", "-i", camera]
        if p["codec"] == "h264_vaapi":
            cmd += ["-vaapi_device", "/dev/dri/renderD128",
                    "-filter_complex",
                    f"[0:v]split=2[rec_in][prev_in];"
                    f"[rec_in]format=nv12,hwupload[rec_out];"
                    f"[prev_in]fps={prev_fps},scale=640:-1[prev_out]",
                    "-map", "[rec_out]", "-c:v", "h264_vaapi", "-qp", p["crf"], "-g", "60",
                    *segment, *preview]
        else:
            cmd += ["-filter_complex",
                    f"[0:v]split=2[rec_out][prev_in];"
                    f"[prev_in]fps={prev_fps},scale=640:-1[prev_out]",
                    "-map", "[rec_out]", "-c:v", p["codec"], "-preset", p["preset"],
                    "-crf", p["crf"], "-g", "60",
                    *segment, *preview]
        try:
            self.ffmpeg_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL,
                stderr=ffmpeg_log_file(self._log_dir, "ffmpeg_camera.log"),
            )
            return True
        except Exception as exc:
            log.error("camera ffmpeg failed to start: %s", exc)
            self.show_alert(f"FFMPEG START FAILURE  •  {exc}", "error")
            return False

    def _start_thermal_recorder(self, p):
        self.thermal.acquire("record", self.get_selected_thermal_path())
        preset = p["preset"] if p["preset"] in ("ultrafast", "superfast", "veryfast", "faster",
                                                "fast", "medium") else "veryfast"
        self.thermal_recorder = ThermalVideoRecorder(
            self.thermal,
            self.pipeline,
            out_dir=p["out_dir"],
            prefix=p["prefix"],
            fps=self._thermal_fps(),
            chunk_sec=p["chunk_sec"],
            outputs=self._thermal_video_outputs(),
            crf=p["crf"],
            preset=preset,
            overlay=self.thermal_overlay_var.get(),
            log_dir=self._log_dir,
        )
        try:
            self.thermal_recorder.start()
        except Exception as exc:
            self.thermal_recorder = None
            self.thermal.release("record")
            log.error("thermal recorder failed to start: %s", exc)
            self.show_alert(f"THERMAL RECORDER FAILED  •  {exc}", "warning", 6000)

    def _stop_thermal_recorder(self):
        if self.thermal_recorder is not None:
            try:
                self.thermal_recorder.stop()
            except Exception as exc:
                log.warning("thermal recorder stop: %s", exc)
            self.thermal_recorder = None
        self.thermal.release("record")

    def stop_recording(self, reason="stopped by user"):
        if self.is_streaming:
            self.toggle_live_stream()
        if self.is_image_previewing:
            self.toggle_image_preview()

        if self._camera_retry_id is not None:
            try:
                self.root.after_cancel(self._camera_retry_id)
            except Exception:
                pass
            self._camera_retry_id = None

        if self.ffmpeg_proc:
            self.ffmpeg_proc.terminate()
            try:
                self.ffmpeg_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_proc.kill()
            self.ffmpeg_proc = None

        self._stop_thermal_recorder()

        if self.is_recording:
            elapsed = time.time() - (self.start_time or time.time())
            log.info("RECORDING STOPPED (%s) after %.2f h", reason, elapsed / 3600)
        self.is_recording = False
        self.start_time = None
        self._end_capture_session()

        self.record_btn.configure(
            text="●  START RECORDING",
            fg_color=COLORS["cyan_dark"],
            hover_color="#155E75",
        )
        self._set_ready_status()
        self.uptime_label.configure(text="00:00:00")
        self.uptime_sub_label.configure(text="STANDBY", text_color=COLORS["text_dim"])
        self.thermal_rec_label.configure(text="")
        self.set_inputs_state("normal")
        self.update_storage_stats(schedule_next=False)

    def _set_ready_status(self):
        self.status_text.configure(text="SYSTEM READY", text_color=COLORS["text"])
        self.status_dot.configure(text_color=COLORS["green"])

    def handle_ffmpeg_crash(self):
        """The camera FFmpeg died: restart only it. Thermal keeps recording."""
        if not self.is_recording or self.ffmpeg_proc is None:
            return
        code = self.ffmpeg_proc.returncode
        self.ffmpeg_proc = None
        self._camera_restarts += 1
        log.error("camera ffmpeg exited (code %s); restart #%d in 2 s -- see logs/ffmpeg_camera.log",
                  code, self._camera_restarts)
        self.status_text.configure(text="RECOVERING", text_color=COLORS["amber"])
        self.status_dot.configure(text_color=COLORS["amber"])
        self.show_alert("CAMERA FFMPEG EXITED  •  RESTARTING (THERMAL CONTINUES)", "warning")
        self._camera_retry_id = self.root.after(2000, self._retry_camera_ffmpeg)

    def _retry_camera_ffmpeg(self):
        self._camera_retry_id = None
        if not self.is_recording:
            return
        if self._start_camera_ffmpeg():
            self.status_text.configure(text="RECORDING ACTIVE", text_color=COLORS["red"])
            self.status_dot.configure(text_color=COLORS["red"])
            log.info("camera ffmpeg restarted")
            self.root.after(4000, lambda: self._apply_camera_controls_async(
                self.get_selected_camera_path(), fresh=False))
        else:
            self._camera_retry_id = self.root.after(10000, self._retry_camera_ffmpeg)

    def set_inputs_state(self, state):
        widgets = [
            self.cam_combo,
            self.thermal_combo,
            self.use_camera_switch,
            self.use_thermal_switch,
            self.res_combo,
            self.fps_combo,
            self.codec_combo,
            self.preset_combo,
            self.crf_spin,
            self.chunk_spin,
            self.dest_entry,
            self.browse_btn,
            self.snap_rate_spin,
            self.snap_duration_spin,
            self.refresh_cam_btn,
            self.thermal_fps_combo,
            self.thermal_video_combo,
            self.thermal_overlay_check,
            self.photo_res_combo,
            self.photo_fmt_combo,
            self.jpeg_q_entry,
            self.prefix_entry,
            self.day_folders_check,
            self.shots_combo,
            self.sched_mode_seg,
            self.start_time_entry,
            self.times_box,
            self.fill_times_btn,
            self.thermal_data_combo,
            self.thermal_avg_entry,
            self.video_prefix_entry,
            self.cam_ctrl_seg,
            self.ctrl_read_btn,
            *self.ctrl_entries,
        ]
        for widget in widgets:
            try:
                if state == "normal" and widget in (self.thermal_video_combo, self.photo_fmt_combo,
                                                     self.thermal_data_combo):
                    widget.configure(state="readonly")
                else:
                    widget.configure(state=state)
            except Exception:
                pass

    def update_timer_loop(self):
        self.clock_label.configure(text=time.strftime("%H:%M:%S"))

        if self.is_recording and self.ffmpeg_proc:
            poll_result = self.ffmpeg_proc.poll()
            if poll_result is not None:
                self.handle_ffmpeg_crash()

        if self.is_recording and self.start_time:
            elapsed = time.time() - self.start_time
            days = elapsed / 86400
            hrs = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            secs = int(elapsed % 60)
            self.uptime_label.configure(text=f"{hrs:02d}:{mins:02d}:{secs:02d}")
            self.uptime_sub_label.configure(text=f"{days:.2f} DAYS  •  ACTIVE", text_color=COLORS["red"])
            rec = self.thermal_recorder
            if rec is not None:
                text = f"THERMAL  •  {rec.frames_written} FRAMES @ {rec.fps:g} FPS"
                color = COLORS["orange"]
                if self.thermal.status != "ONLINE":
                    text = f"THERMAL  •  {self.thermal.status}  (blank frames written)"
                    color = COLORS["amber"]
                elif rec.last_error:
                    text += f"  •  {rec.last_error[:40]}"
                    color = COLORS["amber"]
                self.thermal_rec_label.configure(text=text, text_color=color)
            else:
                self.thermal_rec_label.configure(text="")

        if self.is_timelapse:
            self._update_countdown()

        self._update_thermal_status()
        self._log_thermal_settings_changes()
        if self.is_recording or self.is_timelapse:
            self._log_heartbeat()
        self.root.after(1000, self.update_timer_loop)

    def _update_thermal_status(self):
        if not self.use_thermal_var.get():
            self.thermal_status_label.configure(text="THERMAL: disabled", text_color=COLORS["text_dim"])
            return
        st = self.thermal.status
        color = {"ONLINE": COLORS["green"], "OPENING": COLORS["amber"], "ERROR": COLORS["red"],
                 "RECONNECTING": COLORS["amber"]}.get(st, COLORS["text_dim"])
        if st == "OFF":
            dev = self.get_selected_thermal_path() or "auto-detect"
            text = f"THERMAL: {dev}  •  STANDBY"
        else:
            text = f"THERMAL: {self.thermal.status_text()}"
        self.thermal_status_label.configure(text=text, text_color=color)


    # ------------------------------------------------------------------
    # File naming (video)
    # ------------------------------------------------------------------
    def _safe_video_prefix(self):
        return re.sub(r"[^A-Za-z0-9._-]+", "_", self.video_prefix_var.get().strip()) or "trackingWheel"

    def _update_video_name_example(self):
        if not hasattr(self, "video_name_label"):
            return
        prefix = self._safe_video_prefix()
        try:
            chunk_s = int(round(float(self.chunk_var.get()) * 60))
            base = datetime.now().replace(second=0, microsecond=0)
            nxt = base.strftime("%Y-%m-%d_%H-%M-%S")
            when = f"new chunk every {chunk_s // 60} min, on the clock"
        except ValueError:
            nxt, when = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "chunk length invalid"
        self.video_name_label.configure(
            text=f"{prefix}_{nxt}_camera.mkv\n{prefix}_{nxt}_thermal.mkv   ({when})"
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _attach_log(self, out_dir):
        """Keep the run log next to the data: <save path>/logs/studio.log."""
        log_dir = os.path.join(out_dir, "logs")
        try:
            path = setup_logging(log_dir)
            self._log_dir = log_dir
            log.info("logging to %s", path)
        except Exception as exc:
            log.warning("cannot log to %s (%s); keeping %s", log_dir, exc, self._log_dir)

    def _log_heartbeat(self):
        """Once an hour while capturing: a line proving the run is alive."""
        now = time.time()
        if now - self._last_heartbeat < 3600:
            return
        self._last_heartbeat = now
        try:
            free_gb = shutil.disk_usage(self.dest_path_var.get()).free / 1024 ** 3
        except Exception:
            free_gb = float("nan")
        if self.is_recording:
            rec = self.thermal_recorder
            log.info("heartbeat: recording %.2f h, camera restarts %d, thermal frames %s (%s), free %.1f GB",
                     (now - (self.start_time or now)) / 3600, self._camera_restarts,
                     rec.frames_written if rec else "-", self.thermal.status, free_gb)
        elif self.is_timelapse:
            log.info("heartbeat: time-lapse, %d shots so far, next %s, thermal %s, free %.1f GB",
                     self.timelapse_shots,
                     self.scheduler.next_at.strftime("%H:%M:%S") if self.scheduler and self.scheduler.next_at else "-",
                     self.thermal.status, free_gb)

    def _log_thermal_settings_changes(self):
        """During a capture, log every change of correction / span / palette."""
        # (a video recording logs this itself, in *_thermal_settings.csv too)
        if not self.is_timelapse:
            self._last_thermal_snapshot = None
            return
        snap = self.pipeline.snapshot()
        if self._last_thermal_snapshot is not None and snap != self._last_thermal_snapshot:
            log.info("thermal settings changed: correction %s, coeffs %s, span %s, palette %s, %d ref pts",
                     snap["correction"], snap["coeffs"], snap["span"], snap["colormap"],
                     len(snap["ref_points"]))
        self._last_thermal_snapshot = snap

    # ------------------------------------------------------------------
    # Capture session: sleep blocking + crash / reboot resume
    # ------------------------------------------------------------------
    def _begin_capture_session(self, kind, resumed=False):
        self._last_heartbeat = time.time()
        self._last_thermal_snapshot = self.pipeline.snapshot()
        if not self.inhibitor.acquire(f"{kind} capture in progress"):
            self.show_alert("COULD NOT BLOCK SLEEP  •  DISABLE SUSPEND IN SYSTEM SETTINGS FOR LONG RUNS",
                            "warning", 8000)
        try:
            self._save_settings_file()
            run = {"kind": kind, "started": datetime.now().astimezone().isoformat(timespec="seconds"),
                   "pid": os.getpid(), "out_dir": self.dest_path_var.get()}
            os.makedirs(CONFIG_DIR, exist_ok=True)
            with open(ACTIVE_RUN_PATH, "w") as f:
                json.dump(run, f, indent=2)
        except Exception as exc:
            log.warning("could not write %s: %s", ACTIVE_RUN_PATH, exc)
        self._resume_attempts = 0

    def _end_capture_session(self):
        self.inhibitor.release()
        # "Lock at start" is per run: hand exposure / WB / focus back to the camera.
        if self._ctrl_mode == "lock" and self._ctrl_touched and self.camera_is_valid():
            dev = self.get_selected_camera_path()
            self._locked_values = None
            self._ctrl_touched = False

            def unlock():
                try:
                    camera_controls_auto(dev)
                except Exception as exc:
                    log.warning("camera controls back to auto on %s: %s", dev, exc)
            threading.Thread(target=unlock, daemon=True).start()
        try:
            if os.path.exists(ACTIVE_RUN_PATH):
                os.remove(ACTIVE_RUN_PATH)
        except Exception as exc:
            log.warning("could not remove %s: %s", ACTIVE_RUN_PATH, exc)

    def _check_resume(self):
        """A capture was running when the app last died: offer to resume it."""
        if not os.path.exists(ACTIVE_RUN_PATH):
            return
        try:
            with open(ACTIVE_RUN_PATH) as f:
                run = json.load(f)
        except Exception:
            run = {}
        kind = run.get("kind")
        if run.get("out_dir"):
            self._attach_log(run["out_dir"])
        if kind not in ("video", "photo"):
            self._end_capture_session()
            return
        log.warning("found unfinished %s capture from %s (app or machine stopped without STOP)",
                    kind, run.get("started", "?"))
        if not self.resume_var.get():
            self.show_alert(f"THE LAST {kind.upper()} CAPTURE ENDED UNEXPECTEDLY ({run.get('started', '?')})  •  "
                            f"AUTO-RESUME IS OFF", "warning", 12000)
            self._end_capture_session()
            return
        self._resume_kind = kind
        self._resume_deadline = time.time() + RESUME_DELAY_S
        self._resume_tick()

    def _resume_tick(self):
        left = int(round(self._resume_deadline - time.time()))
        if left > 0:
            self.show_alert(
                f"UNFINISHED {self._resume_kind.upper()} CAPTURE FOUND  •  RESUMING IN {left} s  •  ESC TO CANCEL",
                "warning", quiet=True)
            self._resume_after_id = self.root.after(1000, self._resume_tick)
            return
        self._resume_after_id = None
        self.hide_alert()
        self._do_resume()

    def cancel_resume(self):
        if self._resume_after_id is None:
            return False
        try:
            self.root.after_cancel(self._resume_after_id)
        except Exception:
            pass
        self._resume_after_id = None
        self._end_capture_session()
        log.info("auto-resume cancelled by user")
        self.show_alert("AUTO-RESUME CANCELLED", "info", 3000)
        return True

    def _do_resume(self):
        kind = self._resume_kind
        # Devices may still be enumerating right after a reboot: wait for the scan.
        needs_cam = self.use_camera_var.get()
        if needs_cam and not self.camera_is_valid() and self._resume_attempts < 12:
            self._resume_attempts += 1
            if self._resume_attempts % 3 == 1:
                self.refresh_cameras()
            self.show_alert(f"RESUME  •  WAITING FOR THE CAMERA ({self._resume_attempts}/12)", "warning", quiet=True)
            self._resume_after_id = self.root.after(5000, self._do_resume)
            return
        self._resume_after_id = None
        log.info("auto-resuming %s capture", kind)
        self.set_mode(kind)
        if kind == "video":
            self.start_recording(resumed=True)
            ok = self.is_recording
        else:
            self.start_timelapse(resumed=True)
            ok = self.is_timelapse
        if not ok:
            log.error("auto-resume of %s capture failed", kind)
            self._end_capture_session()

    def _toggle_autostart(self):
        try:
            if self.autostart_var.get():
                os.makedirs(os.path.dirname(AUTOSTART_PATH), exist_ok=True)
                script = os.path.abspath(__file__)
                with open(AUTOSTART_PATH, "w") as f:
                    f.write("[Desktop Entry]\nType=Application\nName=Amin Studio\n"
                            f"Exec=\"{sys.executable}\" \"{script}\"\n"
                            f"Path={os.path.dirname(script)}\n"
                            "X-GNOME-Autostart-enabled=true\nX-GNOME-Autostart-Delay=15\n")
                log.info("launch at login enabled (%s)", AUTOSTART_PATH)
                self.notify("WILL LAUNCH AT LOGIN  •  COMBINE WITH AUTOMATIC LOGIN TO SURVIVE A REBOOT", 5000)
            elif os.path.exists(AUTOSTART_PATH):
                os.remove(AUTOSTART_PATH)
                log.info("launch at login disabled")
        except Exception as exc:
            self.show_alert(f"AUTOSTART  •  {exc}", "error", 5000)

    # ------------------------------------------------------------------
    # Camera exposure / white balance / focus
    # ------------------------------------------------------------------
    CTRL_MODES = {"Auto": "auto", "Lock at start": "lock", "Manual": "manual"}

    def _on_ctrl_mode_change(self, initial=False):
        mode = self.CTRL_MODES.get(self.cam_ctrl_var.get(), "auto")
        self._ctrl_mode = mode
        if mode == "manual":
            self.ctrl_manual_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 4))
        else:
            self.ctrl_manual_frame.grid_forget()
        self.ctrl_status_label.grid(row=5, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 4))
        hints = {
            "auto": "Camera adjusts exposure, white balance and focus itself.",
            "lock": "At the start of each run the settled values are read and\nfrozen, so every image is taken with the same settings.",
            "manual": "Your values are applied every time the camera opens.\nREAD fills in what the camera is using now.",
        }
        self.ctrl_status_label.configure(text=hints[mode], text_color=COLORS["text_dim"])
        self._sync_ctrl_settings()
        if not initial and mode != "lock":
            # Apply straight away if the camera is running (preview / recording).
            dev = self.get_selected_camera_path()
            if self.camera_is_valid() and (self.camera.running or self.is_recording):
                self._apply_camera_controls_async(dev, fresh=False)

    def _sync_ctrl_settings(self):
        """Copy the manual entries into plain state that worker threads read."""
        values = {}
        for key, var in self.ctrl_vars.items():
            text = var.get().strip()
            if not text:
                continue
            try:
                values[key] = int(float(text))
            except ValueError:
                self.show_alert(f"CAMERA {key.upper()} MUST BE A WHOLE NUMBER", "error", 3000)
        self._ctrl_manual = values

    def read_camera_controls(self):
        dev = self.get_selected_camera_path()
        if not self.camera_is_valid():
            self.show_alert("NO VALID CAMERA SELECTED", "error", 3000)
            return

        def worker():
            try:
                vals = camera_controls_read(dev)
            except Exception as exc:
                msg = f"CANNOT READ CAMERA CONTROLS  •  {exc}"
                self.root.after(0, lambda: self.show_alert(msg, "error", 5000))
                return

            def apply():
                for key, var in self.ctrl_vars.items():
                    var.set(str(vals[key]) if key in vals else "")
                self._sync_ctrl_settings()
                missing = [k for k in self.ctrl_vars if k not in vals]
                txt = "Read from camera: " + ", ".join(f"{k} {v}" for k, v in vals.items())
                if missing:
                    txt += f"\n(not supported by this camera: {', '.join(missing)})"
                self.ctrl_status_label.configure(text=txt, text_color=COLORS["cyan"])
            self.root.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def _camera_ready_hook(self, device):
        """Called from a camera worker thread once the camera has warmed up.

        Raises on failure; the caller logs it and carries on.
        """
        mode = self._ctrl_mode
        if mode == "auto":
            if self._ctrl_touched:
                camera_controls_auto(device)
                self._ctrl_touched = False
            return
        if mode == "manual":
            applied = camera_controls_manual(device, dict(self._ctrl_manual))
            self._ctrl_touched = True
            msg = "Manual: " + (", ".join(f"{k} {v}" for k, v in applied.items()) or "nothing applied")
        else:  # lock
            if self._locked_values is None:
                self._locked_values = camera_controls_lock(device)
                msg = "Locked at: " + (", ".join(f"{k} {v}" for k, v in self._locked_values.items())
                                       or "nothing (no controls)")
            else:
                # Camera was re-opened mid-run: re-apply the same frozen values.
                camera_controls_manual(device, dict(self._locked_values))
                msg = "Re-applied lock: " + ", ".join(f"{k} {v}" for k, v in self._locked_values.items())
            self._ctrl_touched = True
        self.root.after(0, lambda: self.ctrl_status_label.configure(text=msg, text_color=COLORS["green"]))

    def _apply_camera_controls_async(self, device, fresh=False):
        """For when FFmpeg (not CameraService) owns the camera, or on demand."""
        if fresh:
            self._locked_values = None

        def worker():
            try:
                self._camera_ready_hook(device)
            except Exception as exc:
                log.warning("camera controls on %s: %s", device, exc)
                msg = f"CAMERA CONTROLS  •  {exc}"
                self.root.after(0, lambda: self.show_alert(msg, "warning", 5000))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Time-lapse (photo tab)
    # ------------------------------------------------------------------
    def current_schedule(self):
        """Sorted seconds-of-day. Raises ValueError with a readable message."""
        if self.sched_mode_var.get() == "Manual times":
            return parse_schedule_text(self.times_box.get("1.0", "end"))
        try:
            n = int(float(self.shots_var.get()))
        except ValueError:
            raise ValueError("frames per day must be a whole number")
        return step_schedule(n, parse_hms(self.start_time_var.get()))

    def on_schedule_change(self):
        manual = self.sched_mode_var.get() == "Manual times"
        card = self.photo_card
        if manual:
            self.start_time_entry.grid_forget()
            self.start_time_label.grid_forget()
            self.times_label.grid(row=11, column=0, sticky="nw", padx=(14, 8), pady=5)
            self.times_box.grid(row=11, column=1, sticky="ew", padx=(0, 14), pady=5)
            self.fill_times_btn.grid(row=12, column=1, sticky="ew", padx=(0, 14), pady=(0, 4))
            if not self.times_box.get("1.0", "end").strip():
                self.fill_manual_times()
            self._validate_manual_times()
        else:
            self.times_label.grid_forget()
            self.times_box.grid_forget()
            self.fill_times_btn.grid_forget()
            self.start_time_label.grid(row=10, column=0, sticky="w", padx=(14, 8), pady=5)
            self.start_time_entry.grid(row=10, column=1, sticky="ew", padx=(0, 14), pady=5)
            try:
                times = self.current_schedule()
                n = len(times)
                preview = ", ".join(format_hms(t)[:5 if n > 4 else 8] for t in times[:6])
                more = f", … (+{n - 6})" if n > 6 else ""
                self.schedule_info_label.configure(
                    text=f"{n} / DAY  •  {describe_step(n)}\n{preview}{more}",
                    text_color=COLORS["cyan"],
                )
            except ValueError as exc:
                self.schedule_info_label.configure(text=f"INVALID  •  {exc}", text_color=COLORS["red"])
        _ = card
        self._update_mode_sign()
        self.update_storage_stats(schedule_next=False)

    def fill_manual_times(self):
        try:
            n = int(float(self.shots_var.get()))
            start = parse_hms(self.start_time_var.get())
            times = step_schedule(n, start)
        except ValueError as exc:
            self.show_alert(f"CANNOT FILL TIMES  •  {exc}", "error", 4000)
            return
        self.times_box.delete("1.0", "end")
        self.times_box.insert("1.0", "\n".join(format_hms(t) for t in times))
        self._validate_manual_times()

    def _validate_manual_times(self):
        if self.sched_mode_var.get() != "Manual times":
            return
        try:
            times = parse_schedule_text(self.times_box.get("1.0", "end"))
            self.schedule_info_label.configure(
                text=f"{len(times)} / DAY  •  manual times  •  first {format_hms(times[0])}, "
                     f"last {format_hms(times[-1])}",
                text_color=COLORS["cyan"],
            )
            self.shots_var.set(str(len(times)))
        except ValueError as exc:
            self.schedule_info_label.configure(text=f"INVALID  •  {exc}", text_color=COLORS["red"])
        self._update_mode_sign()

    def collect_photo_settings(self, validate=True):
        ext = "png" if self.photo_fmt_var.get().upper() == "PNG" else "jpg"
        try:
            q = int(float(self.jpeg_q_var.get()))
        except ValueError:
            if validate:
                raise ValueError("JPEG quality must be 1-100")
            q = 95
        try:
            avg = int(float(self.thermal_avg_var.get()))
        except ValueError:
            if validate:
                raise ValueError("thermal averaging must be a whole number")
            avg = 1
        if validate and not (1 <= q <= 100):
            raise ValueError("JPEG quality must be 1-100")
        if validate and not (1 <= avg <= 100):
            raise ValueError("thermal averaging must be 1-100 frames")
        return PhotoSettings(
            out_dir=self.dest_path_var.get(),
            prefix=self._safe_prefix(),
            day_folders=self.day_folders_var.get(),
            use_camera=self.use_camera_var.get(),
            use_thermal=self.use_thermal_var.get(),
            image_ext=ext,
            jpeg_quality=max(1, min(100, q)),
            thermal_image=self.thermal_image_var.get(),
            thermal_bar=self.thermal_bar_var.get(),
            thermal_data=THERMAL_DATA_CHOICES.get(self.thermal_data_var.get(), "npy"),
            thermal_average=max(1, min(100, avg)),
            thermal_corrected_copy=self.corrected_copy_var.get(),
        )

    def _validate_photo_devices(self, s):
        if not s.use_camera and not s.use_thermal:
            raise ValueError("enable the camera, the thermal camera, or both")
        if s.use_camera and not self.camera_is_valid():
            raise ValueError("no valid camera selected")
        if s.use_camera and s.use_thermal and self.get_selected_camera_path() == self.get_selected_thermal_path():
            raise ValueError("camera and thermal camera are the same device")
        if s.use_thermal and not s.thermal_image and s.thermal_data == "none":
            raise ValueError("thermal is enabled but no thermal output is selected")

    def _photo_interval_s(self, times):
        if len(times) < 2:
            return 86400
        gaps = [(b - a) for a, b in zip(times, times[1:])] + [86400 - times[-1] + times[0]]
        return min(gaps)

    def _acquire_photo_devices(self, user, s, keep_camera=True):
        if s.use_thermal:
            self.thermal.acquire(user, self.get_selected_thermal_path())
        if s.use_camera and keep_camera:
            w, h = self._photo_resolution()
            self.camera.acquire(user, self.get_selected_camera_path(), w, h)

    def toggle_timelapse(self):
        if self.is_timelapse:
            self.stop_timelapse()
        else:
            self.start_timelapse()

    def start_timelapse(self, resumed=False):
        self.hide_alert()
        self.root.focus_set()
        try:
            times = self.current_schedule()
            s = self.collect_photo_settings()
            self._validate_photo_devices(s)
            os.makedirs(s.out_dir, exist_ok=True)
        except ValueError as exc:
            self.show_alert(f"CANNOT START TIME-LAPSE  •  {str(exc).upper()}", "error", 5000)
            return
        except Exception as exc:
            self.show_alert(f"CONFIGURATION ERROR  •  {exc}", "error", 5000)
            return

        # Keep the visible camera open between shots when they come often;
        # for sparse schedules release it and re-open (with warm-up) per shot.
        self._tl_keep_camera = self._photo_interval_s(times) <= 15 * 60
        self._prepare_timelapse_cache()
        self._tl_settings = s
        self._tl_times = times
        self._attach_log(s.out_dir)
        self._locked_values = None                  # "Lock at start": lock afresh
        camera_was_running = self.camera.running
        self._acquire_photo_devices("timelapse", s, keep_camera=self._tl_keep_camera)
        if s.use_camera and camera_was_running and self._tl_keep_camera:
            # Already open (live preview), so the ready hook won't fire again.
            self._apply_camera_controls_async(self.get_selected_camera_path(), fresh=True)

        self.capturer.shots = 0
        self.capturer.bytes_written = 0
        self.timelapse_shots = 0
        self.is_timelapse = True
        self.timelapse_started = time.time()
        self.scheduler = TimelapseScheduler(times, self._on_scheduled_shot)
        self.scheduler.start()
        self._begin_capture_session("photo", resumed)
        log.info("TIME-LAPSE STARTED%s  %d shots/day (%s)  camera=%s thermal=%s  -> %s  "
                 "prefix %s, %s, thermal data %s x%d avg, camera controls %s",
                 " (resumed)" if resumed else "", len(times), self.sched_mode_var.get(),
                 self.get_selected_camera_path() if s.use_camera else "off",
                 (self.get_selected_thermal_path() or "auto") if s.use_thermal else "off",
                 s.out_dir, s.prefix, s.image_ext, s.thermal_data, s.thermal_average, self._ctrl_mode)

        self.timelapse_btn.configure(text="■  STOP TIME-LAPSE", fg_color=COLORS["red"], hover_color="#C93C37")
        self.status_text.configure(text="TIME-LAPSE ACTIVE", text_color=COLORS["red"])
        self.status_dot.configure(text_color=COLORS["red"])
        self.countdown_sub_label.configure(text="WAITING FOR NEXT SHOT", text_color=COLORS["red"])
        self.set_inputs_state("disabled")
        self.show_alert(f"TIME-LAPSE STARTED  •  {len(times)} SHOTS / DAY", "info", 3000)
        self._update_countdown()

    def stop_timelapse(self, reason="stopped by user"):
        if self.scheduler is not None:
            self.scheduler.stop()
            self.scheduler = None
        if self.is_timelapse:
            log.info("TIME-LAPSE STOPPED (%s) after %d shots", reason, self.timelapse_shots)
        self.is_timelapse = False
        self._end_capture_session()
        self.thermal.release("timelapse")
        self.camera.release("timelapse")
        self.timelapse_btn.configure(
            text="●  START TIME-LAPSE", fg_color=COLORS["cyan_dark"], hover_color="#155E75"
        )
        self._set_ready_status()
        self.countdown_label.configure(text="--:--:--")
        self.countdown_sub_label.configure(text="STANDBY", text_color=COLORS["text_dim"])
        self.set_inputs_state("normal")
        self.update_storage_stats(schedule_next=False)

    def _update_countdown(self):
        sched = self.scheduler
        if sched is None or sched.next_at is None:
            return
        remaining = max(0, int((sched.next_at - datetime.now()).total_seconds()))
        h, rem = divmod(remaining, 3600)
        m, s = divmod(rem, 60)
        self.countdown_label.configure(text=f"{h:02d}:{m:02d}:{s:02d}")
        self.countdown_sub_label.configure(
            text=f"NEXT {sched.next_at.strftime('%a %H:%M:%S')}  •  {self.timelapse_shots} TAKEN",
            text_color=COLORS["red"],
        )

    def _on_scheduled_shot(self, when):
        """Runs in a scheduler thread."""
        s = self._tl_settings
        temp_user = None
        if s.use_camera and not self._tl_keep_camera:
            temp_user = "timelapse-shot"
            w, h = self._photo_resolution_cached
            self.camera.acquire(temp_user, self._tl_camera_path, w, h)
        try:
            self._do_capture(when, s)
        finally:
            if temp_user:
                self.camera.release(temp_user)

    def _do_capture(self, when, s):
        if not self._capture_lock.acquire(timeout=60):
            log.warning("shot %s skipped: previous capture still running", when.strftime("%Y-%m-%d_%H-%M-%S"))
            self.root.after(0, lambda: self.show_alert("PREVIOUS CAPTURE STILL RUNNING  •  SHOT SKIPPED", "warning", 4000))
            return
        try:
            result = self.capturer.capture(when, s)
        except Exception as exc:
            log.exception("capture %s failed", when)
            msg = f"CAPTURE FAILED  •  {exc}"
            self.root.after(0, lambda: self.show_alert(msg, "error", 6000))
            return
        finally:
            self._capture_lock.release()
        self.root.after(0, lambda: self._after_capture(result))

    def _after_capture(self, result):
        self.last_capture = result
        if self.is_timelapse:
            self.timelapse_shots += 1
        names = [os.path.basename(p) for p in result["files"]]
        stamp = result["time"].strftime("%Y-%m-%d %H:%M:%S")
        text = f"LAST  {stamp}  •  {len(names)} FILE{'S' if len(names) != 1 else ''}"
        if result["thermal_stats"]:
            lo, hi, mean, ctr = result["thermal_stats"]
            text += f"\nTHERMAL  min {lo:.1f}  max {hi:.1f}  mean {mean:.1f} °C"
        self.last_shot_label.configure(text=text, text_color=COLORS["text_dim"])
        if result["errors"]:
            self.show_alert("CAPTURE PROBLEM  •  " + "  |  ".join(result["errors"]).upper(), "warning", 8000)
            self.last_shot_label.configure(text_color=COLORS["amber"])
        if self.mode == "photo":
            self.photo_preview_info.configure(text=f"LAST CAPTURE  {stamp}   " + "   ".join(names))
            if not self.is_streaming:
                self.show_last_capture()
        self.update_storage_stats(schedule_next=False)

    def capture_now(self):
        if self.mode != "photo":
            return
        try:
            s = self.collect_photo_settings()
            self._validate_photo_devices(s)
            os.makedirs(s.out_dir, exist_ok=True)
        except ValueError as exc:
            self.show_alert(f"CANNOT CAPTURE  •  {str(exc).upper()}", "error", 5000)
            return
        if self._capture_lock.locked():
            self.show_alert("A CAPTURE IS ALREADY IN PROGRESS", "warning", 3000)
            return

        when = datetime.now().replace(microsecond=0)
        self._acquire_photo_devices("manual-shot", s)
        self.capture_now_btn.configure(text="CAPTURING…", state="disabled")
        self.notify("CAPTURING PHOTO PAIR…  (thermal warm-up can take a few seconds)", 4000)

        def worker():
            try:
                self._do_capture(when, s)
            finally:
                self.thermal.release("manual-shot")
                self.camera.release("manual-shot")
                self.root.after(0, lambda: self.capture_now_btn.configure(text="CAPTURE NOW", state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    @property
    def _photo_resolution_cached(self):
        return self._tl_resolution

    def _prepare_timelapse_cache(self):
        self._tl_resolution = self._photo_resolution()
        self._tl_camera_path = self.get_selected_camera_path()

    def show_last_capture(self):
        result = self.last_capture
        if not result:
            self.show_alert("NO CAPTURE YET  •  PRESS CAPTURE NOW OR START THE TIME-LAPSE", "info", 3000)
            return
        if self.is_streaming:
            self.toggle_live_stream()
        if result.get("camera") is not None:
            self.render_frame_to_preview(result["camera"])
        if result.get("thermal") is not None:
            self._show_image_on(self.thermal_display, result["thermal"])
            self._thermal_geom = None       # not clickable: it's a stored image
        self.preview_mode_label.configure(
            text=f"LAST CAPTURE  {result['time'].strftime('%H:%M:%S')}", text_color=COLORS["cyan"]
        )

    # ------------------------------------------------------------------
    # Image preview burst (video tab)
    # ------------------------------------------------------------------
    def toggle_image_preview(self):
        if self.is_image_previewing:
            self.is_image_previewing = False
            self.snap_btn.configure(text="IMAGE PREVIEW")
            self.preview_mode_label.configure(text="OFFLINE", text_color=COLORS["text_dim"])
            self.thermal.release("preview")
            return

        if self.is_streaming:
            self.toggle_live_stream()

        try:
            rate_ipm = int(float(self.snap_rate_var.get()))
            duration_sec = int(float(self.snap_duration_var.get()))
            if rate_ipm <= 0 or duration_sec <= 0:
                raise ValueError
        except ValueError:
            self.show_alert("RATE AND DURATION MUST BE POSITIVE NUMBERS", "error", 5000)
            return

        self.is_image_previewing = True
        self.snap_btn.configure(
            text="STOP IMAGE PREVIEW",
            fg_color=COLORS["red"],
            hover_color="#C93C37",
        )
        self.preview_mode_label.configure(text="BURST PREVIEW", text_color=COLORS["cyan"])
        if self.use_thermal_var.get():
            self.thermal.acquire("preview", self.get_selected_thermal_path())

        self.image_preview_thread = threading.Thread(
            target=self._image_preview_worker,
            daemon=True,
        )
        self.image_preview_thread.start()

    def _image_preview_worker(self):
        try:
            rate_ipm = int(float(self.snap_rate_var.get()))
            duration_sec = int(float(self.snap_duration_var.get()))
        except ValueError:
            return

        interval = 60.0 / rate_ipm
        start_time = time.time()
        last_frame_time = 0.0
        cap = None
        use_cam = self.use_camera_var.get()

        if use_cam and not self.is_recording:
            camera = self.get_selected_camera_path()
            if not self.camera_is_valid():
                self.root.after(0, lambda: self.show_alert("NO VALID CAMERA FOUND", "error", 5000))
                use_cam = False
            else:
                cap = cv2.VideoCapture(camera)
                time.sleep(0.5)

        while self.is_image_previewing:
            current_time = time.time()
            if current_time - start_time > duration_sec:
                break

            if current_time - last_frame_time >= interval:
                if use_cam and self.is_recording:
                    if os.path.exists(self.shm_snapshot):
                        try:
                            frame = cv2.imread(self.shm_snapshot)
                            if frame is not None:
                                self.root.after(0, self.render_frame_to_preview, frame)
                        except Exception:
                            pass
                elif cap and cap.isOpened():
                    for _ in range(4):
                        cap.grab()
                    ret, frame = cap.read()
                    if ret:
                        self.root.after(0, self.render_frame_to_preview, frame)
                self._burst_thermal_due = True
                last_frame_time = current_time

            time.sleep(0.05)

        if cap:
            cap.release()

        self.is_image_previewing = False
        self.root.after(0, self._reset_image_preview_ui)

    def _reset_image_preview_ui(self):
        self.thermal.release("preview")
        self.snap_btn.configure(
            text="IMAGE PREVIEW" if self.mode == "video" else "SHOW LAST CAPTURE",
            fg_color=COLORS["panel_alt"],
            hover_color=COLORS["border"],
        )
        self.preview_mode_label.configure(text="OFFLINE", text_color=COLORS["text_dim"])

    # ------------------------------------------------------------------
    # Live stream (both tabs)
    # ------------------------------------------------------------------
    def toggle_live_stream(self):
        if self.is_streaming:
            self.is_streaming = False
            self.stream_btn.configure(
                text="LIVE STREAM",
                fg_color=COLORS["cyan_dark"],
                hover_color="#155E75",
            )
            self.preview_mode_label.configure(text="OFFLINE", text_color=COLORS["text_dim"])
            self.preview_display.configure(image="", text="CAMERA PREVIEW OFFLINE")
            self.preview_display.imgtk = None
            self.thermal.release("preview")
            self.camera.release("preview")
            if self.thermal.status == "OFF" or not self.use_thermal_var.get():
                self.thermal_display.configure(image="", text="THERMAL PREVIEW OFFLINE")
                self.thermal_display.imgtk = None
                self.thermal_pane_title.configure(text="THERMAL")
            return

        if self.is_image_previewing:
            self.toggle_image_preview()

        use_cam = self.use_camera_var.get()
        use_th = self.use_thermal_var.get()
        if not use_cam and not use_th:
            self.show_alert("ENABLE THE CAMERA OR THE THERMAL CAMERA FIRST", "error", 4000)
            return
        if use_cam and not self.is_recording and not self.camera_is_valid():
            self.show_alert("NO VALID CAMERA FOUND", "error", 5000)
            use_cam = False
            if not use_th:
                return

        self.is_streaming = True
        self.stream_btn.configure(
            text="STOP STREAM",
            fg_color=COLORS["red"],
            hover_color="#C93C37",
        )
        self.preview_mode_label.configure(text="LIVE", text_color=COLORS["green"])

        if use_th:
            self.thermal.acquire("preview", self.get_selected_thermal_path())
            self.thermal_display.configure(text="OPENING THERMAL CAMERA…")
        if use_cam:
            if self.is_recording:
                self.stream_thread = threading.Thread(target=self._stream_worker_recording, daemon=True)
                self.stream_thread.start()
            else:
                if self.mode == "photo":
                    w, h = self._photo_resolution()
                else:
                    w, h = map(int, self.res_var.get().split()[0].split("x"))
                self.camera.acquire("preview", self.get_selected_camera_path(), w, h)
                self.preview_display.configure(text="OPENING CAMERA…")
                self._camera_preview_loop()

    def _stream_worker_recording(self):
        while self.is_streaming and self.is_recording:
            if os.path.exists(self.shm_snapshot):
                try:
                    frame = cv2.imread(self.shm_snapshot)
                    if frame is not None:
                        self.root.after(0, self.render_frame_to_preview, frame)
                except Exception:
                    pass
            time.sleep(0.5)
        self.root.after(0, self._reset_stream_ui)

    def _camera_preview_loop(self):
        if not self.is_streaming or self.is_recording:
            return
        frame, _ts = self.camera.preview_frame()
        if frame is not None:
            self.render_frame_to_preview(frame)
        elif self.camera.status == "ERROR":
            self.preview_display.configure(image="", text=f"CAMERA ERROR\n{self.camera.error[:60]}")
        self.root.after(120, self._camera_preview_loop)

    def _reset_stream_ui(self):
        if not self.is_streaming:
            self.stream_btn.configure(
                text="LIVE STREAM",
                fg_color=COLORS["cyan_dark"],
                hover_color="#155E75",
            )
            self.preview_mode_label.configure(text="OFFLINE", text_color=COLORS["text_dim"])

    # ------------------------------------------------------------------
    # Thermal preview + interaction
    # ------------------------------------------------------------------
    def _thermal_preview_active(self):
        return self.use_thermal_var.get() and (self.is_streaming or self.is_image_previewing)

    def _thermal_preview_loop(self):
        try:
            if self._thermal_preview_active():
                self._render_thermal_preview()
        except Exception as exc:
            self.thermal_display.configure(image="", text=f"THERMAL RENDER ERROR\n{exc}")
        finally:
            self.root.after(150, self._thermal_preview_loop)

    def _current_display_celsius(self):
        if self.thermal_frozen and self._frozen_celsius is not None:
            return self._frozen_celsius
        raw_c, ts, _no = self.thermal.latest()
        if raw_c is None:
            return None
        celsius = self.pipeline.correct(raw_c)
        if self.thermal_frozen:
            self._frozen_celsius = celsius
        return celsius

    def _render_thermal_preview(self, force=False):
        if self.is_image_previewing and not force:
            # In burst mode, refresh at the burst rate only.
            if not getattr(self, "_burst_thermal_due", False):
                return
            self._burst_thermal_due = False

        celsius = self._current_display_celsius()
        if celsius is None:
            st = self.thermal.status
            msg = "OPENING THERMAL CAMERA…" if st == "OPENING" else f"THERMAL {st}\n{self.thermal.error[:70]}"
            self.thermal_display.configure(image="", text=msg)
            self.thermal_display.imgtk = None
            self._thermal_view_celsius = None
            return
        self._thermal_view_celsius = celsius

        avail_w = max(self.thermal_display.winfo_width() - 4, 200)
        avail_h = max(self.thermal_display.winfo_height() - 4, 150)
        # Render at an integer scale (exact markers), then fit to the pane.
        k = int(min((avail_w - COLORBAR_WIDTH) / SENSOR_WIDTH, avail_h / SENSOR_HEIGHT))
        k = max(2, min(k, 5))
        img = self.pipeline.render(
            celsius, k, bar=True, markers=True, selected=self.selected_point,
            cursor=self._thermal_cursor,
        )
        h, w = img.shape[:2]
        f = min(avail_w / w, avail_h / h)
        if abs(f - 1.0) > 0.02:
            img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))),
                             interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
        else:
            f = 1.0
        frozen = "  •  FROZEN" if self.thermal_frozen else ""
        self.thermal_pane_title.configure(
            text=f"THERMAL  {celsius.min():.1f} – {celsius.max():.1f} °C  •  "
                 f"span {self.pipeline.span_text()}{frozen}"
        )
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.thermal_display.imgtk = imgtk
        self.thermal_display.configure(image=imgtk, text="")
        lw = self.thermal_display.winfo_width()
        lh = self.thermal_display.winfo_height()
        self._thermal_geom = ((lw - w) / 2.0, (lh - h) / 2.0, k * f)

    def _event_to_sensor(self, event):
        if self._thermal_geom is None or self._thermal_view_celsius is None:
            return None
        ox, oy, k = self._thermal_geom
        x = int((event.x - ox) // k)
        y = int((event.y - oy) // k)
        if 0 <= x < SENSOR_WIDTH and 0 <= y < SENSOR_HEIGHT:
            return x, y
        return None

    def on_thermal_motion(self, event):
        self._thermal_cursor = self._event_to_sensor(event)

    def on_thermal_click(self, event):
        self.root.focus_set()
        pos = self._event_to_sensor(event)
        if pos is None:
            return
        x, y = pos
        with self.pipeline.lock:
            existing = self.pipeline.calib.nearest(x, y, within=8)
            if existing is not None:
                self.selected_point = existing
                msg = f"SELECTED {existing.label}  •  PRESS K TO SET ITS TRUE TEMPERATURE"
            else:
                self.selected_point = self.pipeline.calib.add_point(x, y)
                msg = (f"ADDED {self.selected_point.label} AT ({x}, {y})  •  "
                       f"PRESS K (OR DOUBLE-CLICK) TO TYPE ITS TRUE TEMPERATURE")
        self.notify(msg, 4000)
        self._refresh_calib_label()

    def on_thermal_right_click(self, event):
        pos = self._event_to_sensor(event)
        if pos is None:
            return
        with self.pipeline.lock:
            existing = self.pipeline.calib.nearest(pos[0], pos[1], within=8)
            if existing is None:
                return
            self.pipeline.calib.remove_point(existing)
            if self.selected_point is existing:
                self.selected_point = None
            self.pipeline.calib.fit()
        self.notify("REFERENCE POINT REMOVED")
        self._refresh_calib_label()

    def prompt_point_value(self):
        point = self.selected_point
        if point is None:
            self.show_alert("CLICK A POINT ON THE THERMAL PREVIEW FIRST, THEN PRESS K", "warning", 3500)
            return
        measured = point.measured_c
        now = f"  (reads {measured:.2f} °C uncorrected)" if measured is not None else ""
        dialog = ctk.CTkInputDialog(
            title="Reference temperature",
            text=f"True temperature of {point.label} in °C{now}\nLeave empty to clear its value.",
            fg_color=COLORS["panel"],
            button_fg_color=COLORS["cyan_dark"],
            button_hover_color="#155E75",
            entry_fg_color=COLORS["panel_alt"],
            entry_border_color=COLORS["border"],
            entry_text_color=COLORS["text"],
            text_color=COLORS["text"],
            font=FONT_UI,
        )
        value = dialog.get_input()
        if value is None:
            return
        value = value.strip()
        with self.pipeline.lock:
            if not value:
                point.known_c = None
                self.pipeline.calib.fit()
                msg = f"{point.label}: VALUE CLEARED"
            else:
                try:
                    point.known_c = float(value.replace(",", "."))
                except ValueError:
                    self.show_alert(f"'{value}' IS NOT A NUMBER", "error", 3000)
                    return
                self.pipeline.calib.fit()
                if self.pipeline.correction != "reference":
                    self.pipeline.correction = "reference"
                    self.correction_var.set("Reference points")
                    self._refresh_thermal_controls()
                msg = f"{point.label} = {point.known_c:.2f} °C  •  {self.pipeline.calib.describe()}"
        self.notify(msg.upper(), 5000)
        self._refresh_calib_label()

    def _refresh_calib_label(self):
        text = self.pipeline.describe_correction()
        with self.pipeline.lock:
            rough = [str(i) for i, p in enumerate(self.pipeline.calib.points, start=1) if not p.is_uniform]
        color = COLORS["green"] if (self.pipeline.correction != "off" and
                                    (self.pipeline.correction == "manual" or self.pipeline.calib.is_active)) \
            else COLORS["text_dim"]
        if rough:
            text += f"\n⚠ point {', '.join(rough)} on a steep gradient — aim at a flat patch"
            color = COLORS["amber"]
        self.calib_label.configure(text=text, text_color=color)

    # --- thermal settings callbacks ----------------------------------------
    def on_palette_change(self, value):
        with self.pipeline.lock:
            self.pipeline.colormap = value

    def on_span_mode_change(self, value):
        mode = SPAN_CHOICES.get(value, "auto")
        self.pipeline.set_span_mode(mode, self._thermal_view_celsius)
        self._refresh_thermal_controls()

    def apply_span_entries(self):
        try:
            lo = float(self.span_lo_var.get().replace(",", "."))
            hi = float(self.span_hi_var.get().replace(",", "."))
            mode = SPAN_CHOICES.get(self.span_mode_var.get(), "auto")
            if mode == "percentile":
                self.pipeline.set_percentiles(lo, hi)
            else:
                self.pipeline.set_manual_span(lo, hi)
        except ValueError as exc:
            self.show_alert(f"INVALID SPAN  •  {str(exc).upper()}", "error", 4000)
            return
        self.root.focus_set()
        self._refresh_thermal_controls()
        self.notify(f"SPAN  •  {self.pipeline.span_text().upper()}")

    def on_correction_change(self, value):
        with self.pipeline.lock:
            self.pipeline.correction = CORRECTION_CHOICES.get(value, "off")
            if self.pipeline.correction == "reference":
                self.pipeline.calib.fit()
        if self.pipeline.correction == "manual":
            self.apply_offset_gain()
        self._refresh_thermal_controls()

    def apply_offset_gain(self):
        try:
            off = float(self.offset_var.get().replace(",", "."))
            gain = float(self.gain_var.get().replace(",", "."))
        except ValueError:
            self.show_alert("OFFSET AND GAIN MUST BE NUMBERS", "error", 3500)
            return
        with self.pipeline.lock:
            self.pipeline.offset_c = off
            self.pipeline.gain = gain
        self._refresh_calib_label()

    def _refresh_thermal_controls(self):
        """Sync widgets to the pipeline (after keyboard changes, loads, etc.)."""
        p = self.pipeline
        with p.lock:
            span = p.span
            mode = span.mode
            self.palette_var.set(p.colormap)
            self.span_mode_var.set({v: k for k, v in SPAN_CHOICES.items()}[mode])
            if mode == "manual":
                self.span_lo_var.set(f"{span.vmin:.1f}")
                self.span_hi_var.set(f"{span.vmax:.1f}")
            elif mode == "percentile":
                self.span_lo_var.set(f"{span.lower_percentile:g}")
                self.span_hi_var.set(f"{span.upper_percentile:g}")
            else:
                lo, hi = p.manual_bounds
                self.span_lo_var.set(f"{lo:.1f}")
                self.span_hi_var.set(f"{hi:.1f}")
            self.correction_var.set({v: k for k, v in CORRECTION_CHOICES.items()}[p.correction])
            self.offset_var.set(f"{p.offset_c:g}")
            self.gain_var.set(f"{p.gain:g}")
            correction = p.correction
        self.span_row_label.configure(text="Percentile %" if mode == "percentile" else "Span °C")
        entry_state = "disabled" if mode == "auto" else "normal"
        for w in (self.span_lo_entry, self.span_hi_entry, self.span_apply_btn):
            w.configure(state=entry_state)
        og_state = "normal" if correction == "manual" else "disabled"
        for w in (self.offset_entry, self.gain_entry):
            w.configure(state=og_state)
        self._refresh_calib_label()

    def cycle_fit_mode(self):
        order = ["auto", "offset", "linear", "quadratic", "off"]
        with self.pipeline.lock:
            calib = self.pipeline.calib
            calib.mode = order[(order.index(calib.mode) + 1) % len(order)]
            calib.fit()
            msg = calib.describe()
        self.notify(f"FIT MODE {calib.mode.upper()}  •  {msg}", 4000)
        self._refresh_calib_label()

    def clear_points(self):
        with self.pipeline.lock:
            self.pipeline.calib.clear()
        self.selected_point = None
        self.notify("ALL REFERENCE POINTS CLEARED")
        self._refresh_calib_label()

    def refit(self):
        with self.pipeline.lock:
            if not self.pipeline.calib.usable_points:
                msg = None
            else:
                self.pipeline.calib.fit()
                msg = self.pipeline.calib.describe()
        if msg is None:
            self.show_alert("NOTHING TO FIT  •  NO POINTS WITH KNOWN VALUES", "warning", 3000)
        else:
            self.notify(f"REFITTED  •  {msg}", 4000)
        self._refresh_calib_label()

    def save_thermal_settings(self):
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            with self.pipeline.lock:
                self.pipeline.calib.save(CALIB_PATH)
            self._save_settings_file()
            self.notify(f"SAVED THERMAL SETTINGS + CALIBRATION  •  {CONFIG_DIR}", 3500)
        except Exception as exc:
            self.show_alert(f"SAVE FAILED  •  {exc}", "error", 5000)

    def load_thermal_settings(self):
        try:
            data = self._read_settings_file()
            if data.get("thermal"):
                self.pipeline.load_dict(data["thermal"])
            with self.pipeline.lock:
                # No refit on load: reloaded points have no samples yet, so a
                # refit would wipe the very coefficients we just restored.
                self.pipeline.calib = Calibration.load(CALIB_PATH)
            self.selected_point = None
            self._refresh_thermal_controls()
            self.notify(f"LOADED  •  {self.pipeline.describe_correction()}", 3500)
        except Exception as exc:
            self.show_alert(f"LOAD FAILED  •  {exc}", "error", 5000)

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------
    def _read_settings_file(self):
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH) as f:
                return json.load(f)
        return {}

    def _load_settings_file(self):
        try:
            self._settings = self._read_settings_file()
        except Exception:
            self._settings = {}
        try:
            if self._settings.get("thermal"):
                self.pipeline.load_dict(self._settings["thermal"])
            if os.path.exists(CALIB_PATH):
                self.pipeline.calib = Calibration.load(CALIB_PATH)
        except Exception:
            pass

    def _apply_loaded_settings(self):
        ui = self._settings.get("ui", {})
        simple = {
            "dest": self.dest_path_var, "res": self.res_var, "fps": self.fps_var, "codec": self.codec_var,
            "preset": self.preset_var, "crf": self.crf_var, "chunk": self.chunk_var,
            "thermal_fps": self.thermal_fps_var, "thermal_video": self.thermal_video_var,
            "photo_res": self.photo_res_var, "photo_fmt": self.photo_fmt_var, "jpeg_q": self.jpeg_q_var,
            "prefix": self.prefix_var, "shots": self.shots_var, "sched_mode": self.sched_mode_var,
            "start_time": self.start_time_var, "thermal_data": self.thermal_data_var,
            "thermal_avg": self.thermal_avg_var, "snap_rate": self.snap_rate_var,
            "snap_duration": self.snap_duration_var,
            "video_prefix": self.video_prefix_var, "cam_ctrl_mode": self.cam_ctrl_var,
        }
        for key, var in simple.items():
            if key in ui:
                try:
                    var.set(ui[key])
                except Exception:
                    pass
        for key, var in {"use_camera": self.use_camera_var, "use_thermal": self.use_thermal_var,
                         "overlay": self.thermal_overlay_var, "day_folders": self.day_folders_var,
                         "thermal_image": self.thermal_image_var, "thermal_bar": self.thermal_bar_var,
                         "resume": self.resume_var, "corrected_copy": self.corrected_copy_var}.items():
            if key in ui:
                var.set(bool(ui[key]))
        if ui.get("manual_times"):
            self.times_box.insert("1.0", ui["manual_times"])
        for key, value in (ui.get("cam_ctrl_values") or {}).items():
            if key in self.ctrl_vars and value is not None:
                self.ctrl_vars[key].set(str(value))
        self._on_ctrl_mode_change(initial=True)
        self._update_video_name_example()
        self._wanted_cam = ui.get("camera_path")
        self._wanted_thermal = ui.get("thermal_path")
        self._refresh_thermal_controls()
        self._layout_preview_panes()

    def _save_settings_file(self):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        ui = {
            "dest": self.dest_path_var.get(), "res": self.res_var.get(), "fps": self.fps_var.get(),
            "codec": self.codec_var.get(), "preset": self.preset_var.get(), "crf": self.crf_var.get(),
            "chunk": self.chunk_var.get(), "thermal_fps": self.thermal_fps_var.get(),
            "thermal_video": self.thermal_video_var.get(), "photo_res": self.photo_res_var.get(),
            "photo_fmt": self.photo_fmt_var.get(), "jpeg_q": self.jpeg_q_var.get(),
            "prefix": self.prefix_var.get(), "shots": self.shots_var.get(),
            "sched_mode": self.sched_mode_var.get(), "start_time": self.start_time_var.get(),
            "thermal_data": self.thermal_data_var.get(), "thermal_avg": self.thermal_avg_var.get(),
            "snap_rate": self.snap_rate_var.get(), "snap_duration": self.snap_duration_var.get(),
            "use_camera": self.use_camera_var.get(), "use_thermal": self.use_thermal_var.get(),
            "overlay": self.thermal_overlay_var.get(), "day_folders": self.day_folders_var.get(),
            "thermal_image": self.thermal_image_var.get(), "thermal_bar": self.thermal_bar_var.get(),
            "manual_times": self.times_box.get("1.0", "end").strip(),
            "camera_path": self.get_selected_camera_path(),
            "thermal_path": self.get_selected_thermal_path(),
            "video_prefix": self.video_prefix_var.get(), "cam_ctrl_mode": self.cam_ctrl_var.get(),
            "cam_ctrl_values": {k: v.get() for k, v in self.ctrl_vars.items()},
            "resume": self.resume_var.get(), "corrected_copy": self.corrected_copy_var.get(),
        }
        data = {"ui": ui, "thermal": self.pipeline.to_dict()}
        tmp = SETTINGS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, SETTINGS_PATH)

    # ------------------------------------------------------------------
    # Keyboard shortcuts + help popup
    # ------------------------------------------------------------------
    def _typing(self):
        widget = self.root.focus_get()
        return isinstance(widget, (tk.Entry, tk.Text))

    def on_key(self, event):
        key = event.keysym
        if key == "Escape":
            if self.cancel_resume():
                return
            if self.help_window is not None:
                self.close_help()
            else:
                self.root.focus_set()
            return
        if self._typing():
            return
        if event.widget is not None and self.help_window is not None:
            try:
                if str(event.widget).startswith(str(self.help_window)) and key.lower() != "h":
                    return
            except Exception:
                pass
        if event.state & 0x4:                      # Ctrl held: leave to the toolkit
            return

        char = event.char
        cel = self._thermal_view_celsius

        if key.lower() == "h":
            self.toggle_help()
        elif key == "v":
            self.set_mode("video")
        elif key == "i":
            self.set_mode("photo")
        elif key == "space":
            self._space_pressed()
        elif key == "l":
            self.toggle_live_stream()
        elif key == "c":
            self.capture_now()
        elif key == "g":
            self.toggle_system_telemetry()
        elif key == "r":
            self.refresh_cameras()
        elif key == "p":
            self.notify(f"PALETTE  •  {self.pipeline.cycle_colormap(+1).upper()}", 1500)
            self._refresh_thermal_controls()
        elif key == "P":
            self.notify(f"PALETTE  •  {self.pipeline.cycle_colormap(-1).upper()}", 1500)
            self._refresh_thermal_controls()
        elif key == "m":
            order = ["auto", "manual", "percentile"]
            nxt = order[(order.index(self.pipeline.span.mode) + 1) % 3]
            self.pipeline.set_span_mode(nxt, cel)
            self._refresh_thermal_controls()
            self.notify(f"SPAN  •  {self.pipeline.span_text().upper()}", 2000)
        elif key == "e":
            if self.pipeline.span.mode == "auto":
                self.pipeline.set_span_mode("manual", cel)
                self._refresh_thermal_controls()
            self._reveal(self.thermal_card)
            self.span_lo_entry.focus_set()
            self.span_lo_entry.select_range(0, "end")
            self.notify("TYPE THE SPAN MINIMUM, TAB, MAXIMUM, THEN ENTER", 4000)
        elif key == "bracketleft":
            self._span_nudge(lambda: self.pipeline.shift_span(-1.0, cel))
        elif key == "bracketright":
            self._span_nudge(lambda: self.pipeline.shift_span(+1.0, cel))
        elif key in ("minus", "underscore", "KP_Subtract"):
            self._span_nudge(lambda: self.pipeline.widen_span(-1.0, cel))
        elif key in ("plus", "equal", "KP_Add"):
            self._span_nudge(lambda: self.pipeline.widen_span(+1.0, cel))
        elif key == "b":
            if self.pipeline.bracket_points():
                self._refresh_thermal_controls()
                self.notify(f"SPAN  •  {self.pipeline.span_text().upper()}")
            else:
                self.show_alert("NEED AT LEAST TWO REFERENCE POINTS WITH VALUES", "warning", 3000)
        elif key == "t":
            self.thermal_frozen = not self.thermal_frozen
            self._frozen_celsius = None
            self.notify("THERMAL PREVIEW FROZEN" if self.thermal_frozen else "THERMAL PREVIEW RUNNING", 1500)
        elif key == "k":
            self.prompt_point_value()
        elif char and char in "123456789":
            idx = int(char) - 1
            with self.pipeline.lock:
                pts = self.pipeline.calib.points
                if idx < len(pts):
                    self.selected_point = pts[idx]
                    self.notify(f"SELECTED {pts[idx].label}", 1500)
        elif key == "x":
            if self.selected_point is not None:
                with self.pipeline.lock:
                    self.pipeline.calib.remove_point(self.selected_point)
                    self.pipeline.calib.fit()
                self.selected_point = None
                self.notify("REFERENCE POINT REMOVED")
                self._refresh_calib_label()
        elif key == "X":
            self.clear_points()
        elif key == "f":
            self.cycle_fit_mode()
        elif key == "u":
            self.refit()
        elif key == "s":
            self.save_thermal_settings()
        elif key == "o":
            self.load_thermal_settings()

    def _span_nudge(self, action):
        try:
            action()
        except ValueError as exc:
            self.show_alert(str(exc).upper(), "warning", 2500)
            return
        self._refresh_thermal_controls()
        self.notify(f"SPAN  •  {self.pipeline.span_text().upper()}", 1500)

    def _reveal(self, widget):
        """Scroll the left panel so a card is visible."""
        try:
            canvas = self.left_panel._parent_canvas
            total = max(1, self.left_panel.winfo_height())
            canvas.yview_moveto(max(0.0, widget.winfo_y() / total - 0.02))
        except Exception:
            pass

    def _space_pressed(self):
        active = self.is_recording if self.mode == "video" else self.is_timelapse
        if not active:
            if self.mode == "video":
                self.start_recording()
            else:
                self.start_timelapse()
            return
        now = time.time()
        if now <= self._space_armed_until:
            self._space_armed_until = 0.0
            if self.mode == "video":
                self.stop_recording()
            else:
                self.stop_timelapse()
        else:
            self._space_armed_until = now + 2.0
            self.show_alert("PRESS SPACE AGAIN WITHIN 2 s TO STOP", "warning", 2000)

    def toggle_help(self):
        if self.help_window is not None:
            self.close_help()
        else:
            self.open_help()

    def close_help(self):
        if self.help_window is not None:
            try:
                self.help_window.destroy()
            except Exception:
                pass
            self.help_window = None
            self.root.focus_set()

    def open_help(self):
        win = ctk.CTkToplevel(self.root)
        win.title("Keyboard shortcuts")
        win.configure(fg_color=COLORS["bg"])
        win.resizable(False, False)
        win.transient(self.root)
        self.help_window = win

        card = self.make_card(win)
        card.pack(fill="both", expand=True, padx=14, pady=14)

        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=18, pady=(14, 4))
        ctk.CTkLabel(head, text="KEYBOARD SHORTCUTS", font=("Segoe UI", 16, "bold"),
                     text_color=COLORS["text"]).pack(side="left")
        ctk.CTkLabel(head, text="H / ESC TO CLOSE", font=FONT_MONO_SMALL,
                     text_color=COLORS["cyan"]).pack(side="right")

        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=18, pady=(4, 16))
        for col, groups in enumerate((SHORTCUTS[:2], SHORTCUTS[2:])):
            column = ctk.CTkFrame(body, fg_color="transparent")
            column.grid(row=0, column=col, sticky="nw", padx=(0 if col == 0 else 24, 0))
            r = 0
            for title, rows in groups:
                ctk.CTkLabel(column, text=title, font=FONT_SECTION, text_color=COLORS["text_dim"],
                             anchor="w").grid(row=r, column=0, columnspan=2, sticky="w", pady=(8 if r else 0, 4))
                r += 1
                for keys, desc in rows:
                    ctk.CTkLabel(
                        column, text=keys, font=("DejaVu Sans Mono", 10, "bold"),
                        text_color=COLORS["cyan"], fg_color=COLORS["panel_alt"], corner_radius=5,
                        width=120, anchor="center",
                    ).grid(row=r, column=0, sticky="w", pady=2)
                    ctk.CTkLabel(column, text=desc, font=FONT_UI_SMALL, text_color=COLORS["text"],
                                 anchor="w").grid(row=r, column=1, sticky="w", padx=(10, 0), pady=2)
                    r += 1

        ctk.CTkLabel(
            card,
            text="Shortcuts are ignored while you are typing in a field — press ESC to leave the field.",
            font=FONT_MONO_SMALL, text_color=COLORS["text_dim"],
        ).pack(padx=18, pady=(0, 14))

        win.bind("<Escape>", lambda _e: self.close_help())
        win.bind("<Key-h>", lambda _e: self.close_help())
        win.bind("<Key-H>", lambda _e: self.close_help())
        win.protocol("WM_DELETE_WINDOW", self.close_help)
        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_height()) // 3
        win.geometry(f"+{max(0, x)}+{max(0, y)}")
        win.after(50, win.focus_force)

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------
    def _show_image_on(self, widget, bgr):
        target_w = max(widget.winfo_width(), 320)
        target_h = max(widget.winfo_height(), 240)
        h, w = bgr.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        widget.imgtk = imgtk
        widget.configure(image=imgtk, text="")

    def render_frame_to_preview(self, cv2_frame):
        try:
            self._show_image_on(self.preview_display, cv2_frame)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def on_close(self):
        try:
            self._save_settings_file()
        except Exception:
            pass
        try:
            if self.is_recording:
                self.stop_recording(reason="window closed")
            if self.is_timelapse:
                self.stop_timelapse(reason="window closed")
            self._end_capture_session()
            self.is_streaming = False
            self.is_image_previewing = False
            self.thermal.shutdown()
            self.camera.shutdown()
            log.info("===== Amin Studio closed =====")
        finally:
            self.root.destroy()


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    app = AdvancedRecorderApp(root)
    root.mainloop()
