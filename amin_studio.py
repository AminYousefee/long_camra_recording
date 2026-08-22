import os
import glob
import time
import shutil
import subprocess
import threading
import re
from collections import deque

import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image, ImageTk
import cv2


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


class AdvancedRecorderApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("4K Video Data Logger")
        self.root.geometry("1200x820")
        self.root.minsize(1040, 700)
        self.root.configure(fg_color=COLORS["bg"])

        # Recording state
        self.ffmpeg_proc = None
        self.start_time = None
        self.is_recording = False

        # Preview state
        self.is_streaming = False
        self.stream_thread = None
        self.is_image_previewing = False
        self.image_preview_thread = None

        # RAM snapshot used by FFmpeg while recording
        self.shm_snapshot = "/dev/shm/current_snapshot.jpg"

        # System telemetry is completely opt-in. Nothing is sampled until
        # the user presses the CPU / RAM monitor button.
        self.is_telemetry_active = False
        self._telemetry_after_id = None
        self.cpu_history = deque(maxlen=60)
        self.ram_history = deque(maxlen=60)
        self._last_cpu_sample = None

        # Prevent duplicate storage timers
        self._storage_after_id = None

        self.setup_ui()
        self.refresh_cameras()
        self.update_storage_stats(schedule_next=True)
        self.update_timer_loop()

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
        return ctk.CTkEntry(parent, **defaults)

    def make_combo(self, parent, values, variable=None):
        return ctk.CTkComboBox(
            parent,
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
        )

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
        self.status_dot.grid(row=0, column=0, rowspan=2, padx=(0, 8))

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

        title_box = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        title_box.grid(row=0, column=1)
        ctk.CTkLabel(
            title_box,
            text="4K VIDEO DATA LOGGER",
            font=("Segoe UI", 18, "bold"),
            text_color=COLORS["text"],
        ).pack()
        ctk.CTkLabel(
            title_box,
            text="CAPTURE / RECORD / TELEMETRY",
            font=FONT_MONO_SMALL,
            text_color=COLORS["cyan"],
        ).pack(pady=(1, 0))

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

        # --------------------------- Main -----------------------------
        self.main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_container.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
        self.main_container.grid_columnconfigure(0, weight=0, minsize=360)
        self.main_container.grid_columnconfigure(1, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        self.left_panel = ctk.CTkFrame(self.main_container, fg_color="transparent", width=360)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(3, weight=1)

        self.right_panel = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=3)
        self.right_panel.grid_rowconfigure(1, weight=2)

        self._build_capture_card()
        self._build_storage_card()
        self._build_recording_card()
        self._build_preview_card()
        self._build_telemetry_card()
        self._build_alert_banner()

    # ------------------------------------------------------------------
    # Capture configuration card
    # ------------------------------------------------------------------
    def _build_capture_card(self):
        self.settings_card = self.make_card(self.left_panel)
        self.settings_card.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.settings_card.grid_columnconfigure(1, weight=1)

        self.section_label(self.settings_card, "Capture Configuration").grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(12, 8)
        )

        self.cam_combo = self.make_combo(self.settings_card, ["Scanning cameras..."])
        self.add_labeled_control(self.settings_card, 1, "Camera", self.cam_combo)

        camera_actions = ctk.CTkFrame(self.settings_card, fg_color="transparent")
        camera_actions.grid(row=2, column=1, sticky="ew", padx=(0, 14), pady=(0, 5))
        camera_actions.grid_columnconfigure(0, weight=1)
        self.refresh_cam_btn = self.make_button(
            camera_actions,
            "REFRESH DEVICES",
            self.refresh_cameras,
            height=30,
            font=FONT_UI_SMALL,
        )
        self.refresh_cam_btn.grid(row=0, column=0, sticky="ew")

        self.res_var = tk.StringVar(value="3840x2160 (4K)")
        self.res_combo = self.make_combo(
            self.settings_card,
            [
                "3840x2160 (4K)",
                "2560x1440 (2K)",
                "1920x1080 (1080p)",
                "1280x720 (720p)",
            ],
            self.res_var,
        )
        self.add_labeled_control(self.settings_card, 3, "Resolution", self.res_combo)
        self.res_combo.configure(command=lambda _: self.update_storage_stats(schedule_next=False))

        self.fps_var = tk.StringVar(value="30")
        self.fps_combo = self.make_combo(self.settings_card, ["30", "60", "24", "15"], self.fps_var)
        self.add_labeled_control(self.settings_card, 4, "Framerate", self.fps_combo)

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
        self.add_labeled_control(self.settings_card, 5, "Codec", self.codec_combo)

        self.preset_var = tk.StringVar(value="veryfast")
        self.preset_combo = self.make_combo(
            self.settings_card,
            ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"],
            self.preset_var,
        )
        self.add_labeled_control(self.settings_card, 6, "Preset", self.preset_combo)

        self.crf_var = tk.StringVar(value="23")
        self.crf_spin = self.make_entry(self.settings_card, textvariable=self.crf_var)
        self.add_labeled_control(self.settings_card, 7, "Quality / CRF", self.crf_spin)

        self.chunk_var = tk.StringVar(value="30")
        self.chunk_spin = self.make_entry(self.settings_card, textvariable=self.chunk_var)
        self.add_labeled_control(self.settings_card, 8, "Chunk / min", self.chunk_spin)

        path_frame = ctk.CTkFrame(self.settings_card, fg_color="transparent")
        path_frame.grid(row=9, column=1, sticky="ew", padx=(0, 14), pady=5)
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
            self.settings_card,
            text="Save path",
            font=FONT_UI_SMALL,
            text_color=COLORS["text_dim"],
        ).grid(row=9, column=0, sticky="w", padx=(14, 8), pady=5)

        ctk.CTkFrame(self.settings_card, fg_color="transparent", height=8).grid(
            row=10, column=0, columnspan=2
        )

    # ------------------------------------------------------------------
    # Storage card
    # ------------------------------------------------------------------
    def _build_storage_card(self):
        self.storage_card = self.make_card(self.left_panel)
        self.storage_card.grid(row=1, column=0, sticky="ew", pady=(0, 10))
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
    # Recording card
    # ------------------------------------------------------------------
    def _build_recording_card(self):
        self.action_card = self.make_card(self.left_panel)
        self.action_card.grid(row=2, column=0, sticky="ew", pady=(0, 10))
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
        self.uptime_sub_label.grid(row=2, column=0, padx=14, pady=(0, 9))

        self.record_btn = self.make_button(
            self.action_card,
            "●  START RECORDING",
            self.toggle_recording,
            accent=True,
            height=48,
        )
        self.record_btn.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 8))

        self.telemetry_btn = self.make_button(
            self.action_card,
            "SHOW CPU / RAM",
            self.toggle_system_telemetry,
            height=38,
            font=FONT_UI_SMALL,
        )
        self.telemetry_btn.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 14))

    # ------------------------------------------------------------------
    # Camera preview card
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

        controls = ctk.CTkFrame(self.preview_card, fg_color="transparent")
        controls.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
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

        self.preview_display = tk.Label(
            self.preview_card,
            bg=COLORS["black"],
            fg=COLORS["text_dim"],
            text="CAMERA PREVIEW OFFLINE\n\nSelect IMAGE PREVIEW or LIVE STREAM",
            font=FONT_MONO,
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        self.preview_display.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 10))

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

    # ------------------------------------------------------------------
    # Telemetry card
    # ------------------------------------------------------------------
    def _build_telemetry_card(self):
        self.telemetry_card = self.make_card(self.right_panel)
        # Do not grid the telemetry card here. It stays hidden until requested.
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
        )
        self.alert_label.pack(fill="x", padx=12, pady=8)

    def show_alert(self, message, kind="error", auto_hide_ms=None):
        colors = {
            "error": ("#321416", COLORS["red"], "#FFD7D5"),
            "warning": ("#332A12", COLORS["amber"], "#FFE7A3"),
            "info": ("#0B2A33", COLORS["cyan"], "#C6F6FF"),
        }
        fg, border, text = colors.get(kind, colors["error"])
        self.alert_banner.configure(fg_color=fg, border_color=border)
        self.alert_label.configure(text=message, text_color=text)
        self.alert_banner.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 10))
        self.alert_banner.lift()
        if auto_hide_ms:
            self.root.after(auto_hide_ms, self.hide_alert)

    def hide_alert(self):
        self.alert_banner.grid_forget()

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

    def start_system_telemetry(self):
        if self.is_telemetry_active:
            return

        self.is_telemetry_active = True
        self.cpu_history.clear()
        self.ram_history.clear()
        self._last_cpu_sample = None

        # Show the panel only after the user explicitly requests monitoring.
        self.right_panel.grid_rowconfigure(1, weight=2)
        self.telemetry_card.grid(row=1, column=0, sticky="nsew")
        self.telemetry_btn.configure(
            text="HIDE CPU / RAM",
            fg_color=COLORS["cyan_dark"],
            hover_color="#155E75",
        )

        # Establish a CPU baseline now. The first displayed CPU percentage is
        # calculated one second later, so no artificial 0% sample is graphed.
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

        # Hide the entire telemetry card and let the preview reclaim the space.
        self.telemetry_card.grid_forget()
        self.right_panel.grid_rowconfigure(1, weight=0)
        self.telemetry_btn.configure(
            text="SHOW CPU / RAM",
            fg_color=COLORS["panel_alt"],
            hover_color=COLORS["border"],
        )

        # Do not retain old measurements while monitoring is off.
        self.cpu_history.clear()
        self.ram_history.clear()
        self._last_cpu_sample = None
        self._reset_metric_panel(self.cpu_panel)
        self._reset_metric_panel(self.ram_panel)

    def update_system_metrics(self):
        # This guard is important: no /proc CPU or RAM reads occur while the
        # monitor is disabled, even if a previously queued callback fires.
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

        # Horizontal reference lines at 25, 50, 75%
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

        # Current sample marker
        x = width - 2
        y = height - (max(0.0, min(100.0, values[-1])) / 100.0) * (height - 6) - 3
        canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=line_color, outline=line_color)

    # ------------------------------------------------------------------
    # Camera discovery
    # ------------------------------------------------------------------
    def refresh_cameras(self):
        self.cam_combo.configure(values=["Scanning cameras..."])
        self.cam_combo.set("Scanning cameras...")
        self.camera_status_label.configure(text="CAMERA: scanning...")

        def worker():
            video_nodes = sorted(glob.glob("/dev/video*"))
            valid_cams = []

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

                    max_res_str = ""
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
                            w, h = map(int, match.group(1).split("x"))
                            sizes.append((w * h, match.group(1)))
                    if sizes:
                        sizes.sort(reverse=True)
                        max_res_str = f"  |  max {sizes[0][1]}"

                    valid_cams.append(f"{dev} - {name}{max_res_str}")
                except Exception:
                    continue

            self.root.after(0, lambda: self._apply_camera_results(valid_cams))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_camera_results(self, valid_cams):
        if valid_cams:
            self.cam_combo.configure(values=valid_cams)
            self.cam_combo.set(valid_cams[0])
            self.camera_status_label.configure(
                text=f"CAMERA: {valid_cams[0].split(' - ')[0]}  •  ONLINE",
                text_color=COLORS["green"],
            )
        else:
            self.cam_combo.configure(values=["No Camera Found"])
            self.cam_combo.set("No Camera Found")
            self.camera_status_label.configure(
                text="CAMERA: NOT FOUND",
                text_color=COLORS["red"],
            )

    def get_selected_camera_path(self):
        full_str = self.cam_combo.get()
        return full_str.split(" - ")[0] if full_str else ""

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def browse_destination(self):
        folder = filedialog.askdirectory(initialdir=self.dest_path_var.get())
        if folder:
            self.dest_path_var.set(folder)
            self.update_storage_stats(schedule_next=False)

    def estimate_bitrate_mbps(self):
        res_str = self.res_var.get().split()[0]
        w, h = map(int, res_str.split("x"))
        crf = float(self.crf_var.get())
        base_pixels = 1920 * 1080
        scale = (w * h) / base_pixels
        base_mbps = 2.0 * (2 ** ((23 - crf) / 6))
        return max(0.5, base_mbps * scale)

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

            if self.is_recording and free_gb < 5.0:
                self.stop_recording()
                self.show_alert(
                    f"CRITICAL STORAGE  •  Recording stopped cleanly  •  Only {free_gb:.1f} GB remains",
                    "error",
                )

            est_mbps = self.estimate_bitrate_mbps()
            gb_per_hour = (est_mbps * 3600) / (8 * 1024)
            hours_left = free_gb / gb_per_hour if gb_per_hour else 0
            days_left = hours_left / 24

            self.storage_bar.set(used_pct / 100.0)
            self.storage_free_big.configure(text=f"{free_gb:.0f} GB")
            self.time_left_big.configure(text=f"{days_left:.1f} DAYS")
            self.storage_detail_label.configure(
                text=f"{used_pct:4.1f}% USED  •  {total_gb:.0f} GB TOTAL  •  ~{hours_left:.0f} HOURS"
            )

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
            est_mbps = self.estimate_bitrate_mbps()
            gb_per_day = (est_mbps * 3600 * 24) / (8 * 1024)
            req_gb = days * gb_per_day
            self.calc_result_label.configure(
                text=f"{days:g} DAYS  →  {req_gb:.1f} GB  /  {req_gb / 1024:.2f} TB"
            )
        except ValueError:
            self.calc_result_label.configure(text="ENTER A VALID POSITIVE NUMBER OF DAYS")

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------
    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.hide_alert()

        if self.is_streaming:
            self.toggle_live_stream()
        if self.is_image_previewing:
            self.toggle_image_preview()

        camera = self.get_selected_camera_path()
        if not camera or "No Camera" in camera or "Scanning" in camera:
            self.show_alert("NO VALID CAMERA SELECTED", "error", 5000)
            return

        try:
            res = self.res_var.get().split()[0]
            fps = self.fps_var.get()
            codec = self.codec_var.get().split()[0]
            preset = self.preset_var.get()
            crf = str(int(float(self.crf_var.get())))
            chunk_sec = int(float(self.chunk_var.get())) * 60
            out_dir = self.dest_path_var.get()

            rate_ipm = max(1, int(float(self.snap_rate_var.get())))
            os.makedirs(out_dir, exist_ok=True)
        except ValueError:
            self.show_alert("INVALID RECORDING PARAMETER", "error", 5000)
            return
        except Exception as exc:
            self.show_alert(f"CONFIGURATION ERROR  •  {exc}", "error", 5000)
            return

        out_template = os.path.join(out_dir, "rec_%Y-%m-%d_%H-%M-%S.mkv")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "v4l2",
            "-framerate",
            fps,
            "-video_size",
            res,
            "-input_format",
            "mjpeg",
            "-i",
            camera,
        ]

        prev_fps = f"{rate_ipm}/60"

        if codec == "h264_vaapi":
            cmd += [
                "-vaapi_device",
                "/dev/dri/renderD128",
                "-filter_complex",
                f"[0:v]split=2[rec_in][prev_in];"
                f"[rec_in]format=nv12,hwupload[rec_out];"
                f"[prev_in]fps={prev_fps},scale=640:-1[prev_out]",
                "-map",
                "[rec_out]",
                "-c:v",
                "h264_vaapi",
                "-qp",
                crf,
                "-g",
                "60",
                "-f",
                "segment",
                "-segment_time",
                str(chunk_sec),
                "-reset_timestamps",
                "1",
                "-strftime",
                "1",
                out_template,
                "-map",
                "[prev_out]",
                "-update",
                "1",
                "-y",
                self.shm_snapshot,
            ]
        else:
            cmd += [
                "-filter_complex",
                f"[0:v]split=2[rec_out][prev_in];"
                f"[prev_in]fps={prev_fps},scale=640:-1[prev_out]",
                "-map",
                "[rec_out]",
                "-c:v",
                codec,
                "-preset",
                preset,
                "-crf",
                crf,
                "-g",
                "60",
                "-f",
                "segment",
                "-segment_time",
                str(chunk_sec),
                "-reset_timestamps",
                "1",
                "-strftime",
                "1",
                out_template,
                "-map",
                "[prev_out]",
                "-update",
                "1",
                "-y",
                self.shm_snapshot,
            ]

        try:
            self.ffmpeg_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.is_recording = True
            self.start_time = time.time()

            self.record_btn.configure(
                text="■  STOP RECORDING",
                fg_color=COLORS["red"],
                hover_color="#C93C37",
            )
            self.status_text.configure(text="RECORDING ACTIVE", text_color=COLORS["red"])
            self.status_dot.configure(text_color=COLORS["red"])
            self.uptime_sub_label.configure(text="CAPTURE ACTIVE", text_color=COLORS["red"])
            self.set_inputs_state("disabled")
            self.show_alert("RECORDING STARTED  •  CAPTURE PIPELINE ACTIVE", "info", 2500)
        except Exception as exc:
            self.show_alert(f"FFMPEG START FAILURE  •  {exc}", "error")

    def stop_recording(self):
        if self.is_streaming:
            self.toggle_live_stream()
        if self.is_image_previewing:
            self.toggle_image_preview()

        if self.ffmpeg_proc:
            self.ffmpeg_proc.terminate()
            try:
                self.ffmpeg_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_proc.kill()
            self.ffmpeg_proc = None

        self.is_recording = False
        self.start_time = None

        self.record_btn.configure(
            text="●  START RECORDING",
            fg_color=COLORS["cyan_dark"],
            hover_color="#155E75",
        )
        self.status_text.configure(text="SYSTEM READY", text_color=COLORS["text"])
        self.status_dot.configure(text_color=COLORS["green"])
        self.uptime_label.configure(text="00:00:00")
        self.uptime_sub_label.configure(text="STANDBY", text_color=COLORS["text_dim"])
        self.set_inputs_state("normal")
        self.update_storage_stats(schedule_next=False)

    def handle_ffmpeg_crash(self):
        if not self.is_recording:
            return

        self.status_text.configure(text="RECOVERING", text_color=COLORS["amber"])
        self.status_dot.configure(text_color=COLORS["amber"])
        self.show_alert("FFMPEG PROCESS EXITED  •  ATTEMPTING RECOVERY", "warning")

        # Avoid blocking the GUI with sleep: recover after 2 seconds.
        self.is_recording = False
        self.ffmpeg_proc = None
        self.set_inputs_state("normal")
        self.root.after(2000, self.start_recording)

    def set_inputs_state(self, state):
        # CustomTkinter uses normal/disabled for these controls.
        for widget in [
            self.cam_combo,
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
        ]:
            try:
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

        self.root.after(1000, self.update_timer_loop)

    # ------------------------------------------------------------------
    # Image preview burst
    # ------------------------------------------------------------------
    def toggle_image_preview(self):
        if self.is_image_previewing:
            self.is_image_previewing = False
            self.snap_btn.configure(text="IMAGE PREVIEW")
            self.preview_mode_label.configure(text="OFFLINE", text_color=COLORS["text_dim"])
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

        if not self.is_recording:
            camera = self.get_selected_camera_path()
            if not camera or "No Camera" in camera or "Scanning" in camera:
                self.root.after(0, lambda: self.show_alert("NO VALID CAMERA FOUND", "error", 5000))
                self.is_image_previewing = False
                self.root.after(0, self._reset_image_preview_ui)
                return

            cap = cv2.VideoCapture(camera)
            time.sleep(0.5)

        while self.is_image_previewing:
            current_time = time.time()
            if current_time - start_time > duration_sec:
                break

            if current_time - last_frame_time >= interval:
                if self.is_recording:
                    if os.path.exists(self.shm_snapshot):
                        try:
                            frame = cv2.imread(self.shm_snapshot)
                            if frame is not None:
                                self.root.after(0, self.render_frame_to_preview, frame)
                        except Exception:
                            pass
                else:
                    if cap and cap.isOpened():
                        for _ in range(4):
                            cap.grab()
                        ret, frame = cap.read()
                        if ret:
                            self.root.after(0, self.render_frame_to_preview, frame)

                last_frame_time = current_time

            time.sleep(0.05)

        if cap:
            cap.release()

        self.is_image_previewing = False
        self.root.after(0, self._reset_image_preview_ui)

    def _reset_image_preview_ui(self):
        self.snap_btn.configure(
            text="IMAGE PREVIEW",
            fg_color=COLORS["panel_alt"],
            hover_color=COLORS["border"],
        )
        self.preview_mode_label.configure(text="OFFLINE", text_color=COLORS["text_dim"])

    # ------------------------------------------------------------------
    # Live stream
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
            return

        if self.is_image_previewing:
            self.toggle_image_preview()

        self.is_streaming = True
        self.stream_btn.configure(
            text="STOP STREAM",
            fg_color=COLORS["red"],
            hover_color="#C93C37",
        )
        self.preview_mode_label.configure(text="LIVE", text_color=COLORS["green"])

        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()

    def _stream_worker(self):
        if self.is_recording:
            while self.is_streaming and self.is_recording:
                if os.path.exists(self.shm_snapshot):
                    try:
                        frame = cv2.imread(self.shm_snapshot)
                        if frame is not None:
                            self.root.after(0, self.render_frame_to_preview, frame)
                    except Exception:
                        pass
                time.sleep(0.5)
        else:
            camera = self.get_selected_camera_path()
            if not camera or "No Camera" in camera or "Scanning" in camera:
                self.root.after(0, lambda: self.show_alert("NO VALID CAMERA FOUND", "error", 5000))
                self.is_streaming = False
                self.root.after(0, self._reset_stream_ui)
                return

            cap = cv2.VideoCapture(camera)
            while self.is_streaming and not self.is_recording:
                ret, frame = cap.read()
                if ret:
                    self.root.after(0, self.render_frame_to_preview, frame)
                else:
                    time.sleep(0.1)
                time.sleep(0.03)
            cap.release()

        self.root.after(0, self._reset_stream_ui)

    def _reset_stream_ui(self):
        if not self.is_streaming:
            self.stream_btn.configure(
                text="LIVE STREAM",
                fg_color=COLORS["cyan_dark"],
                hover_color="#155E75",
            )
            self.preview_mode_label.configure(text="OFFLINE", text_color=COLORS["text_dim"])

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------
    def render_frame_to_preview(self, cv2_frame):
        try:
            h, w = cv2_frame.shape[:2]
            target_w = max(self.preview_display.winfo_width(), 320)
            target_h_available = max(self.preview_display.winfo_height(), 240)

            scale = min(target_w / w, target_h_available / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            resized = cv2.resize(cv2_frame, (new_w, new_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.preview_display.imgtk = imgtk
            self.preview_display.configure(image=imgtk, text="")
        except Exception:
            pass


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    app = AdvancedRecorderApp(root)
    root.mainloop()
