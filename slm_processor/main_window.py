"""Main window for the Scanning Laser Microscopy (SLM) Image Processor."""

import os
import numpy as np
from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QSplitter, QGroupBox, QLabel, QComboBox, QSlider,
    QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QRadioButton,
    QButtonGroup, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QStatusBar, QToolBar, QScrollArea, QFrame, QAbstractScrollArea,
    QApplication, QMenu, QProgressDialog
)
from PyQt6.QtCore import Qt, pyqtSlot, QObject, QEvent, QTimer, QSettings
from PyQt6.QtGui import QAction, QIcon, QFont, QShortcut, QKeySequence

from slm_processor.data_model import SLMDataset, SLMScan, OBJECTIVE_CALIBRATION
from slm_processor import processing
from slm_processor.widgets.image_canvas import SLMImageCanvas
from slm_processor.widgets.profile_canvas import SLMProfileCanvas
from slm_processor.widgets.series_canvas import SLMSeriesCanvas
from slm_processor.widgets.dual_view import SLMDualView
from slm_processor.widgets.overlay_view import SLMOverlayView
from slm_processor.widgets.channel_manager_dialog import SLMChannelManagerDialog
from slm_processor.widgets.histogram_widget import SLMHistogramWidget


DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 11px;
}
QGroupBox {
    border: 1px solid #3c3c3c;
    border-radius: 5px;
    margin-top: 8px;
    padding-top: 10px;
    font-weight: bold;
    color: #4da6ff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QComboBox, QSpinBox, QDoubleSpinBox, QPushButton {
    background-color: #2d2d2d;
    color: #ffffff;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 3px 6px;
    min-height: 18px;
}
QComboBox:hover, QPushButton:hover {
    background-color: #3d3d3d;
    border-color: #007acc;
}
QComboBox::drop-down {
    border: none;
}
QPushButton:pressed {
    background-color: #005a9e;
}
QPushButton#primaryBtn {
    background-color: #007acc;
    color: white;
    font-weight: bold;
}
QPushButton#primaryBtn:hover {
    background-color: #1a8ad6;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #404040;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #007acc;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: 1px solid #777777;
    width: 12px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 6px;
}
QTabWidget::pane {
    border: 1px solid #3c3c3c;
    background: #1e1e1e;
}
QTabBar::tab {
    background: #252525;
    color: #aaaaaa;
    padding: 6px 14px;
    border: 1px solid #333333;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: #1e1e1e;
    color: #4da6ff;
    border-color: #3c3c3c;
    font-weight: bold;
}
QTableWidget {
    background-color: #252525;
    gridline-color: #3c3c3c;
    border: 1px solid #3c3c3c;
    color: #e0e0e0;
}
QHeaderView::section {
    background-color: #2d2d2d;
    color: #cccccc;
    padding: 4px;
    border: 1px solid #3c3c3c;
    font-weight: bold;
}
QStatusBar {
    background: #181818;
    color: #888888;
    border-top: 1px solid #333333;
}
QScrollBar:vertical {
    border: none;
    background: #1e1e1e;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #444444;
    min-height: 20px;
    border-radius: 5px;
}
QSplitter::handle:horizontal {
    background-color: #2b2b2b;
    border-left: 1px solid #3c3c3c;
    border-right: 1px solid #3c3c3c;
    width: 6px;
}
QSplitter::handle:horizontal:hover {
    background-color: #007acc;
}
QSplitter::handle:vertical {
    background-color: #2b2b2b;
    border-top: 1px solid #3c3c3c;
    border-bottom: 1px solid #3c3c3c;
    height: 6px;
}
QSplitter::handle:vertical:hover {
    background-color: #007acc;
}
"""


class NoScrollComboFilter(QObject):
    """Event filter that prevents mouse wheel scrolling from changing QComboBox values

    when the popup list is closed, while smoothly forwarding the scroll event
    to any parent QAbstractScrollArea (such as the sidebar scroll area).
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QComboBox):
            event.ignore()
            parent = obj.parentWidget()
            while parent:
                if isinstance(parent, QAbstractScrollArea) and parent.viewport():
                    QApplication.sendEvent(parent.viewport(), event)
                    break
                parent = parent.parentWidget()
            return True
        return super().eventFilter(obj, event)


class SLMMainWindow(QMainWindow):
    """Main application window for Scanning Laser Microscopy."""

    def __init__(self, initial_dir: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Scanning Laser Microscopy Image Processor")
        self.resize(1600, 900)
        self.setStyleSheet(DARK_STYLE)

        # Install application-wide event filter to disable mouse-wheel scrolling on dropdown lists
        app = QApplication.instance()
        if app is not None and not getattr(app, "_no_scroll_combo_filter_installed", False):
            self._no_scroll_filter = NoScrollComboFilter(app)
            app.installEventFilter(self._no_scroll_filter)
            app._no_scroll_combo_filter_installed = True

        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Dataset state
        self.dataset: Optional[SLMDataset] = None
        self.dataset_harmonics: Optional[SLMDataset] = None
        self.dataset_timedomain: Optional[SLMDataset] = None
        self.current_scan_idx: int = 0
        self.current_raw_channel: str = "PR_R"
        self.current_processed_data: Optional[np.ndarray] = None
        self.channel_manager_dlg: Optional[SLMChannelManagerDialog] = None

        # Coordinates calibration (Default 50x objective from LSM.py)
        self.calib_x_scale = 62.166693  # um per Volt
        self.calib_y_scale = 64.654161  # um per Volt
        has_calib = self.calib_x_scale > 0 and self.calib_y_scale > 0
        self.calib_unit = "um" if has_calib else "V"

        # Leveling & filters state
        self.plane_level_mode = "None"
        self.line_sub_mode = "None"
        self.filter_mode = "None"
        self.filter_param = 1.0

        # Contrast state
        self.contrast_mode = "Min-Max"

        # Active tool
        self.active_tool_mode = "crosshair"

        # Cached series data
        self.series_metrics: List[float] = []

        # Animation player state
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._on_animation_step)
        self.is_anim_playing = False

        self._build_ui()
        self._build_menus()
        self._setup_shortcuts()

        # If directory specified or default exists, load it
        if initial_dir and os.path.exists(initial_dir):
            self.load_directory(initial_dir)
        elif os.path.exists("SLM_images"):
            self.load_directory("SLM_images")

    def _build_menus(self):
        menubar = self.menuBar()

        # File Menu
        menu_file = menubar.addMenu("&File")
        
        act_open_dataset = QAction("Open Dataset Folder (Series)...", self)
        act_open_dataset.setShortcut("Ctrl+O")
        act_open_dataset.triggered.connect(self._on_open_dataset_dialog)
        menu_file.addAction(act_open_dataset)

        act_open_single = QAction("Open Single Scan Folder...", self)
        act_open_single.setShortcut("Ctrl+Shift+O")
        act_open_single.triggered.connect(self._on_open_single_dialog)
        menu_file.addAction(act_open_single)

        self.menu_recent = menu_file.addMenu("Open &Recent Dataset")
        self._rebuild_recent_menu()

        menu_file.addSeparator()

        act_export_img = QAction("Export Current Image (TIFF / PNG)...", self)
        act_export_img.setShortcut("Ctrl+S")
        act_export_img.triggered.connect(self._on_export_image)
        menu_file.addAction(act_export_img)

        act_export_gif = QAction("Export Series as Animated GIF...", self)
        act_export_gif.setShortcut("Ctrl+Shift+G")
        act_export_gif.setToolTip("Export the current scan series / time-domain frames into an animated GIF movie")
        act_export_gif.triggered.connect(self._on_export_gif)
        menu_file.addAction(act_export_gif)

        act_export_batch = QAction("Export Batch Summary CSV...", self)
        act_export_batch.triggered.connect(self._on_export_batch_summary)
        menu_file.addAction(act_export_batch)

        menu_file.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        menu_file.addAction(act_exit)

        # Tools Menu
        menu_tools = menubar.addMenu("&Tools")
        act_chan_mgr = QAction("Open Channel Manager...", self)
        act_chan_mgr.setShortcut("Ctrl+M")
        act_chan_mgr.triggered.connect(self._on_open_channel_manager)
        menu_tools.addAction(act_chan_mgr)

        menu_tools.addSeparator()

        act_hermitian = QAction("Inverse Hermitian Transform (Time Domain)...", self)
        act_hermitian.setShortcut("Ctrl+T")
        act_hermitian.triggered.connect(self._on_open_hermitian_dialog)
        menu_tools.addAction(act_hermitian)

        act_auto_phase = QAction("Auto-Phase Current Scan", self)
        act_auto_phase.setShortcut("Ctrl+P")
        act_auto_phase.triggered.connect(self._on_auto_phase_clicked)
        menu_tools.addAction(act_auto_phase)

        act_auto_phase_all = QAction("Auto-Phase All Scans", self)
        act_auto_phase_all.triggered.connect(self._on_auto_phase_all_clicked)
        menu_tools.addAction(act_auto_phase_all)

        menu_tools.addSeparator()
        act_reset_filters = QAction("Reset Preprocessing Filters", self)
        act_reset_filters.triggered.connect(self._on_reset_filters)
        menu_tools.addAction(act_reset_filters)

        # Help Menu
        menu_help = menubar.addMenu("&Help")
        act_about = QAction("About SLM Processor...", self)
        act_about.triggered.connect(self._show_about)
        menu_help.addAction(act_about)

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(0)

        # Main horizontal splitter making the left panel resizable
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.setChildrenCollapsible(False)

        # Left control sidebar in a scroll area with adjustable width
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setMinimumWidth(300)
        sidebar_scroll.setMaximumWidth(1200)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(4, 4, 4, 4)
        sidebar_layout.setSpacing(8)

        # --- 1. Dataset & Step Navigation ---
        group_nav = QGroupBox("Dataset & Run Steps")
        nav_layout = QVBoxLayout(group_nav)
        
        row_ds_header = QHBoxLayout()
        self.lbl_dataset_info = QLabel("No dataset loaded")
        self.lbl_dataset_info.setStyleSheet("color: #aaaaaa; font-style: italic;")
        row_ds_header.addWidget(self.lbl_dataset_info, 1)

        self.btn_recent_menu = QPushButton("📂 Recent ▾")
        self.btn_recent_menu.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        self.btn_recent_menu.setToolTip("Quickly select and open recently accessed dataset folders")
        self.btn_recent_menu.clicked.connect(self._show_recent_popup)
        row_ds_header.addWidget(self.btn_recent_menu)
        nav_layout.addLayout(row_ds_header)

        row_step = QHBoxLayout()
        row_step.addWidget(QLabel("Step:"))
        self.combo_steps = QComboBox()
        self.combo_steps.currentIndexChanged.connect(self._on_step_changed)
        row_step.addWidget(self.combo_steps, 1)

        self.btn_prev_step = QPushButton("◀")
        self.btn_prev_step.setFixedWidth(28)
        self.btn_prev_step.clicked.connect(self._on_prev_step)
        self.btn_next_step = QPushButton("▶")
        self.btn_next_step.setFixedWidth(28)
        self.btn_next_step.clicked.connect(self._on_next_step)
        row_step.addWidget(self.btn_prev_step)
        row_step.addWidget(self.btn_next_step)
        nav_layout.addLayout(row_step)

        self.slider_steps = QSlider(Qt.Orientation.Horizontal)
        self.slider_steps.valueChanged.connect(self._on_step_slider_changed)
        nav_layout.addWidget(self.slider_steps)

        # Series Movie / Animation Player Controls
        row_player = QHBoxLayout()
        row_player.setContentsMargins(0, 2, 0, 2)
        row_player.setSpacing(6)

        self.btn_play_pause = QPushButton("▶ Play")
        self.btn_play_pause.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                font-weight: bold;
                padding: 3px 10px;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover { background-color: #1a8ad6; }
        """)
        self.btn_play_pause.setToolTip("Play or pause automatic step playback across the series")
        self.btn_play_pause.clicked.connect(self._toggle_animation_playback)
        row_player.addWidget(self.btn_play_pause)

        lbl_speed = QLabel("Speed:")
        lbl_speed.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        row_player.addWidget(lbl_speed)

        self.spin_anim_fps = QSpinBox()
        self.spin_anim_fps.setRange(1, 60)
        self.spin_anim_fps.setValue(10)
        self.spin_anim_fps.setSuffix(" fps")
        self.spin_anim_fps.setFixedWidth(65)
        self.spin_anim_fps.setStyleSheet("font-size: 10px; padding: 1px 3px;")
        self.spin_anim_fps.valueChanged.connect(self._on_anim_speed_changed)
        row_player.addWidget(self.spin_anim_fps)

        self.chk_anim_loop = QCheckBox("🔁 Loop")
        self.chk_anim_loop.setChecked(True)
        self.chk_anim_loop.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        row_player.addWidget(self.chk_anim_loop)
        row_player.addStretch(1)
        nav_layout.addLayout(row_player)

        # Harmonic -> Time Domain Transform Buttons
        self.btn_hermitian_transform = QPushButton("🔄 Transform to Time Domain (IFFT)")
        self.btn_hermitian_transform.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 10px;
            }
            QPushButton:hover { background-color: #1a8ad6; }
        """)
        self.btn_hermitian_transform.setToolTip("Convert harmonic series into time-domain oscillations via Inverse Hermitian FFT")
        self.btn_hermitian_transform.clicked.connect(self._on_open_hermitian_dialog)
        self.btn_hermitian_transform.setVisible(False)
        nav_layout.addWidget(self.btn_hermitian_transform)

        self.btn_switch_domain = QPushButton("↩ Switch to Harmonics Domain")
        self.btn_switch_domain.setStyleSheet("""
            QPushButton {
                background-color: #383838;
                color: #4da6ff;
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 10px;
            }
            QPushButton:hover { background-color: #484848; }
        """)
        self.btn_switch_domain.setToolTip("Toggle between time-domain and harmonic frequency-domain datasets")
        self.btn_switch_domain.clicked.connect(self._on_toggle_domain)
        self.btn_switch_domain.setVisible(False)
        nav_layout.addWidget(self.btn_switch_domain)

        sidebar_layout.addWidget(group_nav)

        # --- 2. Channel Selection ---
        group_chan = QGroupBox("Channel Selection")
        chan_layout = QVBoxLayout(group_chan)

        self.combo_channel = QComboBox()
        self.combo_channel.currentIndexChanged.connect(self._on_channel_changed)
        chan_layout.addWidget(self.combo_channel)

        self.btn_channel_manager = QPushButton("⚙ Manage / Add Channels...")
        self.btn_channel_manager.clicked.connect(self._on_open_channel_manager)
        chan_layout.addWidget(self.btn_channel_manager)
        sidebar_layout.addWidget(group_chan)

        # --- 3. Lock-in Phase Rotation ---
        group_phase = QGroupBox("Lock-in Phase Rotation")
        phase_layout = QVBoxLayout(group_phase)

        row_phase_input = QHBoxLayout()
        row_phase_input.addWidget(QLabel("Phase:"))
        self.spin_phase = QDoubleSpinBox()
        self.spin_phase.setRange(-180.0, 180.0)
        self.spin_phase.setSingleStep(1.0)
        self.spin_phase.setDecimals(1)
        self.spin_phase.setSuffix(" °")
        self.spin_phase.valueChanged.connect(self._on_phase_spin_changed)
        row_phase_input.addWidget(self.spin_phase, 1)

        self.btn_auto_phase = QPushButton("Auto Phase")
        self.btn_auto_phase.setObjectName("primaryBtn")
        self.btn_auto_phase.clicked.connect(self._on_auto_phase_clicked)
        row_phase_input.addWidget(self.btn_auto_phase)
        phase_layout.addLayout(row_phase_input)

        self.slider_phase = QSlider(Qt.Orientation.Horizontal)
        self.slider_phase.setRange(-180, 180)
        self.slider_phase.setValue(0)
        self.slider_phase.valueChanged.connect(self._on_phase_slider_changed)
        phase_layout.addWidget(self.slider_phase)
        sidebar_layout.addWidget(group_phase)

        # --- 4. Leveling & Preprocessing ---
        group_proc = QGroupBox("Leveling & Preprocessing")
        proc_layout = QGridLayout(group_proc)
        proc_layout.setContentsMargins(6, 8, 6, 8)
        proc_layout.setSpacing(6)

        proc_layout.addWidget(QLabel("Plane Leveling:"), 0, 0)
        self.combo_plane = QComboBox()
        self.combo_plane.addItems(["None", "Subtract Plane Tilt (1st Order)", "Subtract Polynomial (2nd Order)"])
        self.combo_plane.currentIndexChanged.connect(self._update_display)
        proc_layout.addWidget(self.combo_plane, 0, 1)

        proc_layout.addWidget(QLabel("Line Subtraction:"), 1, 0)
        self.combo_line_sub = QComboBox()
        self.combo_line_sub.addItems([
            "None",
            "Subtract Row Mean",
            "Subtract Row Median",
            "Subtract Row Linear Tilt",
            "Subtract Column Mean"
        ])
        self.combo_line_sub.currentIndexChanged.connect(self._update_display)
        proc_layout.addWidget(self.combo_line_sub, 1, 1)

        proc_layout.addWidget(QLabel("Spatial Filter:"), 2, 0)
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["None", "Gaussian Filter", "Median Filter"])
        self.combo_filter.currentIndexChanged.connect(self._update_display)
        proc_layout.addWidget(self.combo_filter, 2, 1)

        proc_layout.addWidget(QLabel("Filter Param (σ / k):"), 3, 0)
        self.spin_filter_param = QDoubleSpinBox()
        self.spin_filter_param.setRange(0.1, 15.0)
        self.spin_filter_param.setValue(1.0)
        self.spin_filter_param.setSingleStep(0.5)
        self.spin_filter_param.valueChanged.connect(self._update_display)
        proc_layout.addWidget(self.spin_filter_param, 3, 1)
        sidebar_layout.addWidget(group_proc)

        # --- 5. Display, Units & Colormap ---
        group_display = QGroupBox("Display & Calibration")
        disp_layout = QGridLayout(group_display)
        disp_layout.setContentsMargins(6, 8, 6, 8)
        disp_layout.setSpacing(6)

        disp_layout.addWidget(QLabel("Colormap:"), 0, 0)
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["cividis", "viridis", "inferno", "plasma", "magma", "turbo", "coolwarm", "bwr", "gray"])
        self.combo_cmap.setCurrentText("cividis")
        self.combo_cmap.currentTextChanged.connect(self._on_cmap_changed)
        disp_layout.addWidget(self.combo_cmap, 0, 1)

        disp_layout.addWidget(QLabel("Contrast:"), 1, 0)
        self.combo_contrast = QComboBox()
        self.combo_contrast.addItems(["Min-Max", "1%-99%", "2%-98%", "5%-95%", "Manual"])
        self.combo_contrast.setCurrentText("Min-Max")
        self.combo_contrast.currentIndexChanged.connect(self._on_contrast_changed)
        disp_layout.addWidget(self.combo_contrast, 1, 1)

        # Interactive Histogram & Threshold Sliders
        self.histogram_widget = SLMHistogramWidget(self)
        self.histogram_widget.clim_changed.connect(self._on_histogram_clim_changed)
        disp_layout.addWidget(self.histogram_widget, 2, 0, 1, 2)

        disp_layout.addWidget(QLabel("Axis Units:"), 3, 0)
        self.combo_units = QComboBox()
        self.combo_units.addItems(["Mirror Volts (V)", "Physical Distance (µm)"])
        # Default to Physical Distance (µm) if calibration available (Index 1), else Volts (Index 0)
        default_unit_idx = 1 if (self.calib_x_scale > 0 and self.calib_y_scale > 0) else 0
        self.combo_units.setCurrentIndex(default_unit_idx)
        self.combo_units.currentIndexChanged.connect(self._on_units_changed)
        disp_layout.addWidget(self.combo_units, 3, 1)

        disp_layout.addWidget(QLabel("Objective:"), 4, 0)
        self.combo_objective = QComboBox()
        self.combo_objective.addItems(["50x", "100x", "20x", "10x", "5x", "Custom"])
        self.combo_objective.currentTextChanged.connect(self._on_objective_changed)
        disp_layout.addWidget(self.combo_objective, 4, 1)

        disp_layout.addWidget(QLabel("Calibration (µm/V):"), 5, 0)
        row_calib = QHBoxLayout()
        self.spin_calib_x = QDoubleSpinBox()
        self.spin_calib_x.setRange(0.001, 100000.0)
        self.spin_calib_x.setDecimals(3)
        self.spin_calib_x.setValue(self.calib_x_scale)
        self.spin_calib_x.setPrefix("X: ")
        self.spin_calib_x.valueChanged.connect(self._on_calib_changed)
        self.spin_calib_y = QDoubleSpinBox()
        self.spin_calib_y.setRange(0.001, 100000.0)
        self.spin_calib_y.setDecimals(3)
        self.spin_calib_y.setValue(self.calib_y_scale)
        self.spin_calib_y.setPrefix("Y: ")
        self.spin_calib_y.valueChanged.connect(self._on_calib_changed)
        row_calib.addWidget(self.spin_calib_x)
        row_calib.addWidget(self.spin_calib_y)
        disp_layout.addLayout(row_calib, 5, 1)

        row_toggles = QHBoxLayout()
        self.chk_invert_y = QCheckBox("Invert Y-Axis")
        self.chk_invert_y.toggled.connect(self._on_invert_y_toggled)
        row_toggles.addWidget(self.chk_invert_y)

        self.chk_show_scale_bar = QCheckBox("Show Scale Bar")
        self.chk_show_scale_bar.setChecked(True)
        self.chk_show_scale_bar.setToolTip("Show / hide calibrated scale bar in bottom-right corner")
        self.chk_show_scale_bar.toggled.connect(self._on_scale_bar_toggled)
        row_toggles.addWidget(self.chk_show_scale_bar)
        disp_layout.addLayout(row_toggles, 6, 0, 1, 2)

        sidebar_layout.addWidget(group_display)

        # --- 6. Interactive Analysis Tools ---
        group_tools = QGroupBox("Interactive Tools")
        tools_layout = QVBoxLayout(group_tools)

        self.btn_group_tools = QButtonGroup(self)
        self.rb_tool_crosshair = QRadioButton("Crosshair Readout")
        self.rb_tool_line = QRadioButton("Line Profile Cut (drag or select)")
        self.rb_tool_roi = QRadioButton("ROI Rectangle (drag)")
        self.rb_tool_caliper = QRadioButton("Distance Caliper (click & drag)")
        self.rb_tool_navigate = QRadioButton("Pan / Zoom Canvas")
        self.rb_tool_crosshair.setChecked(True)

        for i, rb in enumerate([self.rb_tool_crosshair, self.rb_tool_line, self.rb_tool_roi, self.rb_tool_caliper, self.rb_tool_navigate]):
            self.btn_group_tools.addButton(rb, i)
            tools_layout.addWidget(rb)
        self.btn_group_tools.buttonClicked.connect(self._on_tool_mode_changed)

        # Quick line cut selectors
        row_cuts = QHBoxLayout()
        btn_hcut = QPushButton("Horizontal Mid-Cut")
        btn_hcut.clicked.connect(self._on_quick_hcut)
        btn_vcut = QPushButton("Vertical Mid-Cut")
        btn_vcut.clicked.connect(self._on_quick_vcut)
        row_cuts.addWidget(btn_hcut)
        row_cuts.addWidget(btn_vcut)
        tools_layout.addLayout(row_cuts)

        sidebar_layout.addWidget(group_tools)

        # --- 7. Live Readout & ROI Stats ---
        group_stats = QGroupBox("Live Readout & ROI Stats")
        stats_layout = QVBoxLayout(group_stats)
        self.lbl_cursor_val = QLabel("Cursor: (X=-, Y=-) Val: -")
        self.lbl_cursor_val.setStyleSheet("color: #00ffcc; font-weight: bold; font-size: 10px;")
        stats_layout.addWidget(self.lbl_cursor_val)

        self.table_roi = QTableWidget(6, 2)
        self.table_roi.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table_roi.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_roi.verticalHeader().setVisible(False)
        self.table_roi.setFixedHeight(140)
        
        metrics = ["ROI Mean", "ROI Std Dev", "ROI Min", "ROI Max", "Peak-to-Peak", "Pixel Count"]
        for row, m in enumerate(metrics):
            item_m = QTableWidgetItem(m)
            item_m.setFlags(Qt.ItemFlag.ItemIsEnabled)
            item_v = QTableWidgetItem("-")
            item_v.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table_roi.setItem(row, 0, item_m)
            self.table_roi.setItem(row, 1, item_v)
        stats_layout.addWidget(self.table_roi)
        sidebar_layout.addWidget(group_stats)

        sidebar_layout.addStretch(1)
        sidebar_scroll.setWidget(sidebar_content)

        # --- Center Central Tabs ---
        self.tabs = QTabWidget()
        
        # Tab 1: Single Scan View & 1D Profile Cut (with Single / PR on PD Overlay suboptions)
        tab_single = QWidget()
        layout_single = QVBoxLayout(tab_single)
        layout_single.setContentsMargins(2, 2, 2, 2)
        layout_single.setSpacing(4)

        # Sub-option control bar for 2D Image Panel (Single Channel vs PR on PD Overlay)
        self.frame_subopt = QFrame()
        self.frame_subopt.setObjectName("suboptFrame")
        self.frame_subopt.setStyleSheet("""
            QFrame#suboptFrame {
                background-color: #242424;
                border: 1px solid #383838;
                border-radius: 5px;
                padding: 2px;
            }
        """)
        subopt_vbox = QVBoxLayout(self.frame_subopt)
        subopt_vbox.setContentsMargins(6, 4, 6, 4)
        subopt_vbox.setSpacing(4)

        # Row 1: Mode Switch & Quick-Switch Pill Bar
        row_sub_mode = QHBoxLayout()
        row_sub_mode.setContentsMargins(0, 0, 0, 0)
        row_sub_mode.setSpacing(6)

        lbl_sub_title = QLabel("Quick Switch:")
        lbl_sub_title.setStyleSheet("font-weight: bold; color: #aaaaaa; font-size: 11px;")
        row_sub_mode.addWidget(lbl_sub_title)

        pill_style_base = """
            QPushButton {
                background-color: #2b2b2b;
                color: #cccccc;
                font-weight: bold;
                padding: 4px 10px;
                border-radius: 4px;
                border: 1px solid #444444;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #383838; }
            QPushButton:checked {
                background-color: #007acc;
                color: #ffffff;
                border: 1px solid #0098ff;
            }
        """

        self.btn_quick_pd = QPushButton("🔬 PD (1)")
        self.btn_quick_pd.setCheckable(True)
        self.btn_quick_pd.setStyleSheet(pill_style_base)
        self.btn_quick_pd.setToolTip("Quick switch to Photodiode reflection baseline (Shortcut: Key 1)")
        self.btn_quick_pd.clicked.connect(self._quick_switch_pd)
        row_sub_mode.addWidget(self.btn_quick_pd)

        self.btn_quick_pr = QPushButton("⚡ PR (2)")
        self.btn_quick_pr.setCheckable(True)
        self.btn_quick_pr.setStyleSheet(pill_style_base)
        self.btn_quick_pr.setToolTip("Quick switch to Photoresponse signal (Shortcut: Key 2)")
        self.btn_quick_pr.clicked.connect(self._quick_switch_pr)
        row_sub_mode.addWidget(self.btn_quick_pr)

        self.btn_quick_overlay = QPushButton("🔀 Overlay (3)")
        self.btn_quick_overlay.setCheckable(True)
        self.btn_quick_overlay.setStyleSheet(pill_style_base.replace("#007acc", "#9c27b0").replace("#0098ff", "#ba68c8"))
        self.btn_quick_overlay.setToolTip("Quick switch to PR on PD blended overlay (Shortcut: Key 3)")
        self.btn_quick_overlay.clicked.connect(self._quick_switch_overlay)
        row_sub_mode.addWidget(self.btn_quick_overlay)

        self.btn_quick_dual = QPushButton("◫ Dual View (4)")
        self.btn_quick_dual.setCheckable(True)
        self.btn_quick_dual.setStyleSheet(pill_style_base.replace("#007acc", "#d84315").replace("#0098ff", "#ff5722"))
        self.btn_quick_dual.setToolTip("Toggle Dual side-by-side comparison view (Shortcut: Key 4)")
        self.btn_quick_dual.clicked.connect(self._quick_switch_dual)
        row_sub_mode.addWidget(self.btn_quick_dual)

        # Retain aliases for internal button group logic
        self.btn_sub_single = self.btn_quick_pr
        self.btn_sub_overlay = self.btn_quick_overlay

        self.btn_group_submode = QButtonGroup(self)
        self.btn_group_submode.addButton(self.btn_quick_pd, 0)
        self.btn_group_submode.addButton(self.btn_quick_pr, 1)
        self.btn_group_submode.addButton(self.btn_quick_overlay, 2)
        self.btn_group_submode.addButton(self.btn_quick_dual, 3)

        row_sub_mode.addSpacing(15)

        self.lbl_sub_single_info = QLabel("Active Channel: PR_R")
        self.lbl_sub_single_info.setStyleSheet("color: #4da6ff; font-weight: bold; font-size: 11px;")
        row_sub_mode.addWidget(self.lbl_sub_single_info)
        row_sub_mode.addStretch(1)
        subopt_vbox.addLayout(row_sub_mode)

        # Row 2: Inline PR on PD Overlay Controls
        self.widget_overlay_inline = QWidget()
        row_over_ctrl = QHBoxLayout(self.widget_overlay_inline)
        row_over_ctrl.setContentsMargins(0, 2, 0, 2)
        row_over_ctrl.setSpacing(6)

        lbl_top = QLabel("Top (PR):")
        lbl_top.setStyleSheet("color: #ff3366; font-weight: bold; font-size: 11px;")
        row_over_ctrl.addWidget(lbl_top)

        self.combo_overlay_pr_chan = QComboBox()
        self.combo_overlay_pr_chan.currentIndexChanged.connect(self._on_overlay_param_changed)
        row_over_ctrl.addWidget(self.combo_overlay_pr_chan)

        self.combo_overlay_pr_cmap = QComboBox()
        self.combo_overlay_pr_cmap.addItems(["inferno", "turbo", "plasma", "viridis", "coolwarm", "hot", "magma"])
        self.combo_overlay_pr_cmap.currentIndexChanged.connect(self._on_overlay_param_changed)
        row_over_ctrl.addWidget(self.combo_overlay_pr_cmap)

        row_over_ctrl.addSpacing(10)

        lbl_base = QLabel("Base (PD):")
        lbl_base.setStyleSheet("color: #00ffcc; font-weight: bold; font-size: 11px;")
        row_over_ctrl.addWidget(lbl_base)

        self.combo_overlay_pd_chan = QComboBox()
        self.combo_overlay_pd_chan.currentIndexChanged.connect(self._on_overlay_param_changed)
        row_over_ctrl.addWidget(self.combo_overlay_pd_chan)

        self.combo_overlay_pd_cmap = QComboBox()
        self.combo_overlay_pd_cmap.addItems(["gray", "bone", "cividis", "Greys_r", "viridis", "coolwarm"])
        self.combo_overlay_pd_cmap.currentIndexChanged.connect(self._on_overlay_param_changed)
        row_over_ctrl.addWidget(self.combo_overlay_pd_cmap)

        row_over_ctrl.addSpacing(10)

        lbl_alpha = QLabel("Alpha:")
        lbl_alpha.setStyleSheet("color: #cccccc; font-weight: bold; font-size: 11px;")
        row_over_ctrl.addWidget(lbl_alpha)

        self.slider_overlay_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_overlay_alpha.setRange(0, 100)
        self.slider_overlay_alpha.setValue(60)
        self.slider_overlay_alpha.setFixedWidth(100)
        self.slider_overlay_alpha.valueChanged.connect(self._on_sidebar_alpha_slider_changed)
        row_over_ctrl.addWidget(self.slider_overlay_alpha)

        self.spin_overlay_alpha = QSpinBox()
        self.spin_overlay_alpha.setRange(0, 100)
        self.spin_overlay_alpha.setValue(60)
        self.spin_overlay_alpha.setSuffix(" %")
        self.spin_overlay_alpha.setFixedWidth(55)
        self.spin_overlay_alpha.valueChanged.connect(self._on_sidebar_alpha_spin_changed)
        row_over_ctrl.addWidget(self.spin_overlay_alpha)

        row_over_ctrl.addSpacing(10)

        self.chk_overlay_cbar_alpha = QCheckBox("Cbar Min α=0 → Max α=1")
        self.chk_overlay_cbar_alpha.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 11px;")
        self.chk_overlay_cbar_alpha.setToolTip(
            "Sets overlay opacity based on colorbar values: min value is completely transparent (alpha=0) "
            "and max value is opaque (alpha=1). Shows only meaningful signal on top of base layer."
        )
        self.chk_overlay_cbar_alpha.toggled.connect(self._on_overlay_cbar_alpha_toggled)
        self.chk_overlay_gradient = self.chk_overlay_cbar_alpha
        row_over_ctrl.addWidget(self.chk_overlay_cbar_alpha)

        row_over_ctrl.addStretch(1)
        self.widget_overlay_inline.setVisible(False)
        subopt_vbox.addWidget(self.widget_overlay_inline)

        layout_single.addWidget(self.frame_subopt)

        splitter_v = QSplitter(Qt.Orientation.Vertical)
        self.canvas_2d = SLMImageCanvas(self, title="Scanning Laser Microscopy Image")
        self.canvas_2d.pixel_hovered.connect(self._on_pixel_hovered)
        self.canvas_2d.line_cut_changed.connect(self._on_line_cut_changed)
        self.canvas_2d.roi_changed.connect(self._on_roi_changed)
        self.canvas_2d.caliper_measured.connect(self._on_caliper_measured)

        self.canvas_profile = SLMProfileCanvas(self)

        splitter_v.addWidget(self.canvas_2d)
        splitter_v.addWidget(self.canvas_profile)
        splitter_v.setSizes([550, 220])
        layout_single.addWidget(splitter_v)
        self.tabs.addTab(tab_single, "2D Image & 1D Cut")

        # Tab 2: Dual Side-by-Side Comparison View (with synchronized interactive tools)
        self.dual_view = SLMDualView(self)
        self.dual_view.channel_changed.connect(self._update_dual_view)
        self.dual_view.line_cut_changed.connect(self._on_dual_line_cut_changed)
        self.dual_view.roi_changed.connect(self._on_dual_roi_changed)
        self.tabs.addTab(self.dual_view, "Dual View (Side-by-Side)")

        # Standalone overlay view instance kept alive for compatibility
        self.overlay_view = SLMOverlayView(self)
        self.overlay_view.alpha_changed.connect(self._sync_alpha_from_view)
        self.overlay_view.gradient_changed.connect(self._sync_gradient_from_view)

        # Tab 3: Series Analysis
        tab_series = QWidget()
        layout_series = QVBoxLayout(tab_series)
        layout_series.setContentsMargins(4, 4, 4, 4)
        layout_series.setSpacing(4)
        self.canvas_series = SLMSeriesCanvas(self)
        self.canvas_series.step_selected.connect(self._on_series_step_clicked)
        self.canvas_series.export_gif_requested.connect(self._on_export_gif)
        layout_series.addWidget(self.canvas_series)
        self.tabs.addTab(self.canvas_series, "Run Series Trend")

        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Ensure tabs and children don't block left panel resizing
        self.tabs.setMinimumWidth(400)
        for i in range(self.tabs.count()):
            tab_w = self.tabs.widget(i)
            if tab_w is not None:
                tab_w.setMinimumWidth(400)

        # Resizable horizontal splitter layout: left panel is 1.7x larger (578px vs 340px)
        self.main_splitter.addWidget(sidebar_scroll)
        self.main_splitter.addWidget(self.tabs)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([578, 1022])
        main_layout.addWidget(self.main_splitter)

        # Status Bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready. Load a dataset folder or single scan folder to begin.")

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_splitter_initialized", False):
            self._splitter_initialized = True
            left_w = 578
            right_w = max(500, self.main_splitter.width() - left_w)
            self.main_splitter.setSizes([left_w, right_w])

    # --- Loading Logic ---

    def load_directory(self, dir_path: str):
        """Loads a dataset from directory (supports multi-folder and flat harmonic series)."""
        try:
            self.dataset = SLMDataset(dir_path)
            if len(self.dataset) == 0:
                QMessageBox.warning(self, "No Scans Found", f"No valid SLM scan files found in:\n{dir_path}")
                return

            if self.dataset.is_harmonic_series:
                self.dataset_harmonics = self.dataset
                self.dataset_timedomain = None
                self.btn_hermitian_transform.setVisible(True)
                self.btn_switch_domain.setVisible(False)
            else:
                self.dataset_harmonics = None
                self.dataset_timedomain = None
                self.btn_hermitian_transform.setVisible(False)
                self.btn_switch_domain.setVisible(False)

            self.combo_contrast.setCurrentText("Min-Max")
            self._populate_dataset_ui()
            self._save_recent_dataset(dir_path)
            if hasattr(self, "channel_manager_dlg") and self.channel_manager_dlg and self.channel_manager_dlg.isVisible():
                self.channel_manager_dlg.set_active_dataset(self.dataset)
            self.statusBar.showMessage(f"Loaded {len(self.dataset)} scans from {dir_path}", 4000)

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load dataset: {str(e)}")

    def _populate_dataset_ui(self):
        if not self.dataset or len(self.dataset) == 0:
            return

        self.combo_steps.blockSignals(True)
        self.slider_steps.blockSignals(True)
        self.combo_steps.clear()

        is_time = (getattr(self.dataset, "series_variable_name", "") == "Time (ms)")
        for i, scan in enumerate(self.dataset.scans):
            if is_time:
                self.combo_steps.addItem(f"t = {scan.step_name} (frame {i})")
            elif "Harm" in scan.step_name or "Step" in scan.step_name:
                self.combo_steps.addItem(f"{scan.step_name}")
            else:
                self.combo_steps.addItem(f"Step {scan.step_name} (Run {scan.step_value:g})")

        self.slider_steps.setRange(0, len(self.dataset) - 1)
        self.slider_steps.setValue(0)
        self.combo_steps.setCurrentIndex(0)
        self.combo_steps.blockSignals(False)
        self.slider_steps.blockSignals(False)

        self.current_scan_idx = 0
        if is_time:
            self.lbl_dataset_info.setText(f"Time Domain: {len(self.dataset)} slices (0 to {self.dataset.scans[-1].step_value:.2f} ms)")
        elif self.dataset.is_harmonic_series:
            self.lbl_dataset_info.setText(f"Harmonic Series: {len(self.dataset)} steps (Harm 1 to {len(self.dataset)})")
        else:
            self.lbl_dataset_info.setText(f"{len(self.dataset)} scans loaded from {os.path.basename(self.dataset.root_dir)}")

        self._refresh_channel_dropdowns()
        self._update_display()
        self._update_series_trend()

    def _on_open_hermitian_dialog(self):
        target_ds = self.dataset_harmonics if self.dataset_harmonics else self.dataset
        if not target_ds:
            QMessageBox.information(self, "No Dataset", "Please open a dataset first.")
            return

        from slm_processor.widgets.hermitian_dialog import SLMHermitianDialog
        dlg = SLMHermitianDialog(target_ds, self)
        dlg.transformed.connect(self._on_time_domain_transformed)
        dlg.exec()

    def _on_time_domain_transformed(self, time_ds: SLMDataset):
        if not self.dataset_harmonics and self.dataset.is_harmonic_series:
            self.dataset_harmonics = self.dataset
        self.dataset_timedomain = time_ds
        self._set_active_dataset(time_ds)

    def _on_toggle_domain(self):
        if self.dataset == self.dataset_timedomain and self.dataset_harmonics:
            self._set_active_dataset(self.dataset_harmonics)
        elif self.dataset == self.dataset_harmonics and self.dataset_timedomain:
            self._set_active_dataset(self.dataset_timedomain)

    def _set_active_dataset(self, ds: SLMDataset):
        self.dataset = ds
        is_time = (getattr(ds, "series_variable_name", "") == "Time (ms)")

        self.btn_hermitian_transform.setVisible(not is_time and ds.is_harmonic_series)
        self.btn_switch_domain.setVisible(is_time and self.dataset_harmonics is not None)

        self._populate_dataset_ui()

    def _on_open_channel_manager(self):
        if not self.dataset:
            QMessageBox.information(self, "No Dataset", "Please open a dataset first.")
            return

        all_ds = {}
        if self.dataset_harmonics:
            all_ds[f"Harmonics Domain ({len(self.dataset_harmonics)} harmonics)"] = self.dataset_harmonics
        if self.dataset_timedomain:
            all_ds[f"Time Domain ({len(self.dataset_timedomain)} time slices)"] = self.dataset_timedomain

        if not all_ds and self.dataset:
            name = os.path.basename(self.dataset.root_dir) if self.dataset.root_dir else "Active Dataset"
            all_ds[f"{name} ({len(self.dataset)} scans)"] = self.dataset
        elif self.dataset and self.dataset not in all_ds.values():
            name = os.path.basename(self.dataset.root_dir) if self.dataset.root_dir else "Active Dataset"
            all_ds[f"{name} ({len(self.dataset)} scans)"] = self.dataset

        if self.channel_manager_dlg is None or not self.channel_manager_dlg.isVisible():
            self.channel_manager_dlg = SLMChannelManagerDialog(self.dataset, self, all_datasets=all_ds)
            self.channel_manager_dlg.channel_selected.connect(self._on_channel_manager_select_channel)
            self.channel_manager_dlg.send_to_dual.connect(self._on_channel_manager_send_dual)
            self.channel_manager_dlg.send_to_overlay.connect(self._on_channel_manager_send_overlay)
            self.channel_manager_dlg.channels_updated.connect(self._on_channels_updated)
        else:
            self.channel_manager_dlg.set_active_dataset(self.dataset, all_datasets=all_ds)

        self.channel_manager_dlg.show()
        self.channel_manager_dlg.raise_()
        self.channel_manager_dlg.activateWindow()

    def _on_channel_manager_select_channel(self, chan_name: str):
        self.tabs.setCurrentIndex(0)
        if hasattr(self, "btn_sub_single") and hasattr(self, "btn_sub_overlay"):
            self.btn_sub_single.setChecked(True)
            self._on_submode_toggled(self.btn_sub_single)
        chans = [self.combo_channel.itemText(i) for i in range(self.combo_channel.count())]
        if chan_name in chans:
            self.combo_channel.setCurrentText(chan_name)
        self._update_display()
        self._update_quick_buttons_state()
        self.statusBar.showMessage(f"Channel '{chan_name}' loaded in 2D View", 3000)

    def _on_channel_manager_send_dual(self, chan_name: str, target: str):
        self.tabs.setCurrentIndex(1)
        if target == "left":
            self.dual_view.combo_left.setCurrentText(chan_name)
        else:
            self.dual_view.combo_right.setCurrentText(chan_name)
        self._update_dual_view()
        self._update_quick_buttons_state()
        self.statusBar.showMessage(f"Channel '{chan_name}' sent to Dual View ({target})", 3000)

    def _on_channel_manager_send_overlay(self, chan_name: str, target: str):
        self.tabs.setCurrentIndex(0)
        if hasattr(self, "btn_sub_overlay"):
            self.btn_sub_overlay.setChecked(True)
            self._on_submode_toggled(self.btn_sub_overlay)
        if target == "top":
            self.combo_overlay_pr_chan.setCurrentText(chan_name)
        else:
            self.combo_overlay_pd_chan.setCurrentText(chan_name)
        self._update_display()
        self._update_quick_buttons_state()
        self.statusBar.showMessage(f"Channel '{chan_name}' set as Overlay {target}", 3000)

    def _on_channels_updated(self):
        self._refresh_channel_dropdowns()
        self._update_display()
        self._update_series_trend()

    def _refresh_channel_dropdowns(self):
        if not self.dataset or len(self.dataset) == 0:
            return

        scan = self.dataset[self.current_scan_idx]
        raw_chans = scan.available_channels
        favs = getattr(self.dataset, "favorite_channels", set())
        chans = sorted(raw_chans, key=lambda c: (0 if c in favs else 1, c))

        # 1. Main channel selector
        curr_main = self.combo_channel.currentText()
        self.combo_channel.blockSignals(True)
        self.combo_channel.clear()
        for ch in chans:
            self.combo_channel.addItem(ch)
        self.combo_channel.addItem("Overlay: Blend Any 2 Channels")

        if curr_main in chans or "Overlay" in curr_main:
            self.combo_channel.setCurrentText(curr_main)
        else:
            if "PR_R" in chans:
                self.combo_channel.setCurrentText("PR_R")
            elif len(chans) > 0:
                self.combo_channel.setCurrentIndex(0)
        self.combo_channel.blockSignals(False)

        # 2. Sidebar Overlay selectors
        curr_top = self.combo_overlay_pr_chan.currentText()
        curr_base = self.combo_overlay_pd_chan.currentText()

        self.combo_overlay_pr_chan.blockSignals(True)
        self.combo_overlay_pd_chan.blockSignals(True)
        self.combo_overlay_pr_chan.clear()
        self.combo_overlay_pd_chan.clear()

        self.combo_overlay_pr_chan.addItems(chans)
        self.combo_overlay_pd_chan.addItems(chans)

        if curr_top in chans:
            self.combo_overlay_pr_chan.setCurrentText(curr_top)
        else:
            for c in ["PR_R", "X_PR", "PR_X", "PR", "P4_R", "P4_X", "P4"]:
                if c in chans:
                    self.combo_overlay_pr_chan.setCurrentText(c)
                    break
            else:
                if len(chans) > 1:
                    self.combo_overlay_pr_chan.setCurrentIndex(1)

        if curr_base in chans:
            self.combo_overlay_pd_chan.setCurrentText(curr_base)
        else:
            for c in ["PD_R", "X_PD", "PD_X", "PD"]:
                if c in chans:
                    self.combo_overlay_pd_chan.setCurrentText(c)
                    break
            else:
                if len(chans) > 0:
                    self.combo_overlay_pd_chan.setCurrentIndex(0)

        self.combo_overlay_pr_chan.blockSignals(False)
        self.combo_overlay_pd_chan.blockSignals(False)

        # 3. Dual view selectors
        self.dual_view.set_channels_list(chans)

        # 4. Overlay view tab
        self.overlay_view.set_channels_list(chans)

    def _on_open_dataset_dialog(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Root Dataset Folder (containing step folders)")
        if dir_path:
            self.load_directory(dir_path)

    def _on_open_single_dialog(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Single Scan Folder (containing TIFF and LOG files)")
        if dir_path:
            self.load_directory(dir_path)

    # --- Data Retrieval & Processing Pipeline ---

    def _get_current_channel_key(self) -> str:
        if hasattr(self, "btn_sub_overlay") and self.btn_sub_overlay.isChecked():
            return "OVERLAY"
        text = self.combo_channel.currentText()
        if "Overlay" in text or "OVERLAY" in text:
            return "OVERLAY"
        return text if text else "PR_R"

    def _get_processed_image(self, scan: SLMScan, channel_key: str, phase_deg: float) -> np.ndarray:
        """Retrieves raw/derived channel and applies user-selected preprocessing."""
        if channel_key == "OVERLAY":
            channel_key = self.combo_overlay_pr_chan.currentText()

        raw = scan.get_channel(channel_key, phase_deg=phase_deg)
        img = raw.copy()

        # 1. Plane leveling
        plane_mode = self.combo_plane.currentText()
        if "1st Order" in plane_mode:
            img, _ = processing.fit_and_subtract_plane(img)
        elif "2nd Order" in plane_mode:
            img, _ = processing.fit_and_subtract_poly2(img)

        # 2. Line subtraction
        line_mode = self.combo_line_sub.currentText()
        if "Row Mean" in line_mode:
            img = processing.line_by_line_subtraction(img, "row_mean")
        elif "Row Median" in line_mode:
            img = processing.line_by_line_subtraction(img, "row_median")
        elif "Row Linear" in line_mode:
            img = processing.line_by_line_subtraction(img, "row_linear")
        elif "Column Mean" in line_mode:
            img = processing.line_by_line_subtraction(img, "col_mean")

        # 3. Spatial filter
        filt_mode = self.combo_filter.currentText()
        param = self.spin_filter_param.value()
        if "Gaussian" in filt_mode:
            img = processing.apply_spatial_filter(img, "gaussian", param)
        elif "Median" in filt_mode:
            img = processing.apply_spatial_filter(img, "median", param)

        return img

    def _get_coordinates(self, scan: SLMScan) -> Tuple[np.ndarray, np.ndarray, str]:
        """Calculates X and Y coordinates according to selected unit and calibration."""
        if self.calib_unit == "um":
            x_coords = scan.x_volts * self.calib_x_scale
            y_coords = scan.y_volts * self.calib_y_scale
            unit = "µm"
        else:
            x_coords = scan.x_volts
            y_coords = scan.y_volts
            unit = "V"
        return x_coords, y_coords, unit

    # --- UI Event Handlers ---

    def _update_display(self):
        """Refreshes the current 2D image and associated 1D profile."""
        if not self.dataset or self.current_scan_idx >= len(self.dataset):
            return

        scan = self.dataset[self.current_scan_idx]
        channel_key = self._get_current_channel_key()
        phase_deg = self.spin_phase.value()
        x_coords, y_coords, unit = self._get_coordinates(scan)

        if channel_key == "OVERLAY":
            top_name = self.combo_overlay_pr_chan.currentText()
            base_name = self.combo_overlay_pd_chan.currentText()

            over_img = self._get_processed_image(scan, top_name, phase_deg)
            base_img = self._get_processed_image(scan, base_name, phase_deg)
            self.current_processed_data = over_img
            self.current_base_data = base_img

            alpha = self.slider_overlay_alpha.value() / 100.0
            base_cmap = self.combo_overlay_pd_cmap.currentText()
            over_cmap = self.combo_overlay_pr_cmap.currentText()

            is_cbar = self.chk_overlay_cbar_alpha.isChecked()
            if is_cbar:
                title = f"Scan: {scan.step_name} | {top_name} on {base_name} (Cbar Min α=0 → Max α={alpha*100:.0f}%)"
            else:
                title = f"Scan: {scan.step_name} | {top_name} on {base_name} (α={alpha*100:.0f}%)"

            clim_b = self._calc_contrast_limits(base_img)
            clim_o = self._calc_contrast_limits(over_img)

            self.canvas_2d.set_overlay_data(
                base_image=base_img,
                overlay_image=over_img,
                alpha=alpha,
                base_cmap=base_cmap,
                overlay_cmap=over_cmap,
                x_coords=x_coords,
                y_coords=y_coords,
                unit=unit,
                title=title,
                clim_overlay=clim_o,
                clim_base=clim_b,
                cbar_alpha=is_cbar,
                base_channel_name=base_name,
                overlay_channel_name=top_name
            )
            self._apply_contrast()
        else:
            processed = self._get_processed_image(scan, channel_key, phase_deg)
            self.current_processed_data = processed
            self.current_base_data = None
            title = f"Scan: {scan.step_name} | {self.combo_channel.currentText()}"
            self.canvas_2d.set_data(processed, x_coords, y_coords, unit=unit, title=title)
            self._apply_contrast(processed)
            if hasattr(self, "lbl_sub_single_info"):
                self.lbl_sub_single_info.setText(f"Active Channel: {self.combo_channel.currentText()}")

        # Update 1D line cut
        if self.canvas_2d.line_points is not None:
            self._update_line_profile(*self.canvas_2d.line_points)

        # Update ROI stats
        if self.canvas_2d.roi_box is not None:
            self._update_roi_stats(*self.canvas_2d.roi_box)

        # Update dual view if active
        if self.tabs.currentIndex() == 1:
            self._update_dual_view()

        # Update standalone overlay view if visible
        if hasattr(self, "overlay_view") and self.overlay_view.isVisible():
            self.overlay_view.set_scan_data(
                scan=scan,
                x_coords=x_coords,
                y_coords=y_coords,
                unit=unit,
                phase_deg=phase_deg,
                proc_fct=self._get_processed_image,
                calib_um_per_v=self.calib_x_scale
            )
        self._update_quick_buttons_state()

    def _calc_contrast_limits(self, image: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
        if image is None or image.size == 0:
            return None
        mode = self.combo_contrast.currentText()
        if mode == "1%-99%":
            vmin = float(np.percentile(image, 1.0))
            vmax = float(np.percentile(image, 99.0))
        elif mode == "2%-98%":
            vmin = float(np.percentile(image, 2.0))
            vmax = float(np.percentile(image, 98.0))
        elif mode == "5%-95%":
            vmin = float(np.percentile(image, 5.0))
            vmax = float(np.percentile(image, 95.0))
        elif mode == "Manual":
            return None
        else:  # "Min-Max" or default fallback
            vmin = float(np.min(image))
            vmax = float(np.max(image))
        if vmin >= vmax:
            vmax = vmin + 1e-6
        return (vmin, vmax)

    def _apply_contrast(self, image: Optional[np.ndarray] = None):
        if self._get_current_channel_key() == "OVERLAY":
            if self.current_base_data is not None:
                clim_b = self._calc_contrast_limits(self.current_base_data)
                if clim_b is not None:
                    self.canvas_2d.set_clim(clim_b[0], clim_b[1])
            if self.current_processed_data is not None:
                clim_o = self._calc_contrast_limits(self.current_processed_data)
                if clim_o is not None:
                    self.canvas_2d.set_clim_overlay(clim_o[0], clim_o[1])
                if hasattr(self, "histogram_widget"):
                    vo_min = clim_o[0] if clim_o else None
                    vo_max = clim_o[1] if clim_o else None
                    self.histogram_widget.set_data(self.current_processed_data, vo_min, vo_max)
        else:
            target = image if image is not None else self.current_processed_data
            if target is not None:
                clim = self._calc_contrast_limits(target)
                if clim is not None:
                    self.canvas_2d.set_clim(clim[0], clim[1])
                else:
                    self.canvas_2d.set_clim(None, None)
                if hasattr(self, "histogram_widget"):
                    c_min = clim[0] if clim else None
                    c_max = clim[1] if clim else None
                    self.histogram_widget.set_data(target, c_min, c_max)

    def _on_histogram_clim_changed(self, vmin: float, vmax: float):
        """Called when user drags histogram threshold lines or edits spinboxes."""
        self.combo_contrast.blockSignals(True)
        self.combo_contrast.setCurrentText("Manual")
        self.combo_contrast.blockSignals(False)
        if self._get_current_channel_key() == "OVERLAY":
            self.canvas_2d.set_clim_overlay(vmin, vmax)
        else:
            self.canvas_2d.set_clim(vmin, vmax)

    def _on_scale_bar_toggled(self, checked: bool):
        self.canvas_2d.set_show_scale_bar(checked)

    def _on_caliper_measured(self, x0: float, y0: float, x1: float, y1: float, length: float):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        unit = self.calib_unit
        self.lbl_cursor_val.setText(f"Caliper: L = {length:.4f} {unit}  (ΔX = {dx:.4f}, ΔY = {dy:.4f})")
        self.statusBar.showMessage(
            f"Caliper: Distance = {length:.4f} {unit} | ΔX: {dx:.4f} {unit} | ΔY: {dy:.4f} {unit} | Start: ({x0:.3f}, {y0:.3f}) End: ({x1:.3f}, {y1:.3f})"
        )

    def _update_line_profile(self, c0: float, r0: float, c1: float, r1: float):
        if self.current_processed_data is None or not self.dataset:
            return

        scan = self.dataset[self.current_scan_idx]
        x_coords, y_coords, unit = self._get_coordinates(scan)

        distances_pix, values, _ = processing.extract_line_profile(
            self.current_processed_data, (c0, r0), (c1, r1)
        )

        # Convert pixel distance to physical unit
        cols = self.current_processed_data.shape[1]
        x_span = abs(float(x_coords[-1] - x_coords[0]))
        dx_per_pix = x_span / max(1, cols - 1)
        distances_phys = distances_pix * dx_per_pix

        is_overlay = (self._get_current_channel_key() == "OVERLAY")
        if is_overlay and hasattr(self, "current_base_data") and self.current_base_data is not None:
            _, values_base, _ = processing.extract_line_profile(
                self.current_base_data, (c0, r0), (c1, r1)
            )
            top_name = self.combo_overlay_pr_chan.currentText() or "PR"
            base_name = self.combo_overlay_pd_chan.currentText() or "PD"
            self.canvas_profile.set_dual_profile(
                distances_phys, values, values_base, unit=unit,
                title=f"Cut: {top_name} & {base_name} from ({c0:.0f}, {r0:.0f}) to ({c1:.0f}, {r1:.0f})",
                calib_um_per_v=self.calib_x_scale,
                label_pr=top_name,
                label_pd=base_name,
                is_overlay=True
            )
        else:
            self.canvas_profile.set_profile(
                distances_phys, values, unit=unit,
                title=f"Cut from ({c0:.0f}, {r0:.0f}) to ({c1:.0f}, {r1:.0f})",
                calib_um_per_v=self.calib_x_scale
            )

    def _update_roi_stats(self, c0: int, r0: int, c1: int, r1: int):
        if self.current_processed_data is None:
            return
        self._update_roi_stats_with_data(self.current_processed_data, (c0, r0, c1, r1))

    def _update_roi_stats_with_data(self, data: np.ndarray, bbox: Tuple[int, int, int, int]):
        stats = processing.calculate_roi_stats(data, bbox)
        
        self.table_roi.item(0, 1).setText(f"{stats['mean']:.4e}")
        self.table_roi.item(1, 1).setText(f"{stats['std']:.4e}")
        self.table_roi.item(2, 1).setText(f"{stats['min']:.4e}")
        self.table_roi.item(3, 1).setText(f"{stats['max']:.4e}")
        self.table_roi.item(4, 1).setText(f"{stats['peak_to_peak']:.4e}")
        self.table_roi.item(5, 1).setText(f"{stats['count']}")

    def _update_series_trend(self):
        """Computes ROI mean across all loaded scans in the dataset."""
        if not self.dataset or len(self.dataset) == 0:
            return

        channel_key = self._get_current_channel_key()
        phase_deg = self.spin_phase.value()
        
        # Get active ROI box or full image if not set
        bbox = self.canvas_2d.roi_box
        if bbox is None:
            shape = self.dataset[0].shape
            bbox = (0, 0, shape[1] - 1, shape[0] - 1)

        metric_vals = []
        for scan in self.dataset.scans:
            proc = self._get_processed_image(scan, channel_key, phase_deg)
            stats = processing.calculate_roi_stats(proc, bbox)
            metric_vals.append(stats["mean"])

        self.series_metrics = metric_vals
        var_name = getattr(self.dataset, "series_variable_name", "Step")
        x_label = "Time (ms)" if var_name == "Time (ms)" else ("Harmonic Number" if var_name == "Harmonic" else "Step / Run Number")
        self.canvas_series.set_series_data(
            step_names=self.dataset.step_names,
            step_vals=self.dataset.step_values,
            metric_vals=metric_vals,
            current_step_idx=self.current_scan_idx,
            metric_name=f"ROI Mean ({self.combo_channel.currentText()})",
            x_label=x_label
        )

    def _update_dual_view(self):
        """Updates side-by-side view canvases based on user-selected channels in dual view."""
        if not self.dataset or self.current_scan_idx >= len(self.dataset):
            return

        scan = self.dataset[self.current_scan_idx]
        phase_deg = self.spin_phase.value()
        x_coords, y_coords, unit = self._get_coordinates(scan)

        # Left channel (View 1)
        key_l = self.dual_view.combo_left.currentText()
        if not key_l:
            key_l = self.dual_view.combo_left.itemText(0) if self.dual_view.combo_left.count() > 0 else "PD_R"
        img_l = self._get_processed_image(scan, key_l, phase_deg)
        self.dual_view.canvas_left.set_data(
            img_l, x_coords, y_coords, unit=unit,
            title=f"View 1: {key_l}"
        )
        self.dual_view.canvas_left.set_colormap(self.dual_view.combo_left_cmap.currentText())

        # Right channel (View 2)
        key_r = self.dual_view.combo_right.currentText()
        if not key_r:
            key_r = self.dual_view.combo_right.itemText(1) if self.dual_view.combo_right.count() > 1 else key_l
        img_r = self._get_processed_image(scan, key_r, phase_deg)
        self.dual_view.canvas_right.set_data(
            img_r, x_coords, y_coords, unit=unit,
            title=f"View 2: {key_r}"
        )
        self.dual_view.canvas_right.set_colormap(self.dual_view.combo_right_cmap.currentText())
        self.dual_view.set_calibration(self.calib_x_scale)
        self.dual_view.refresh_profile()

    # --- Navigation & Control Callbacks ---

    def _on_step_changed(self, idx: int):
        if idx >= 0 and self.dataset and idx < len(self.dataset):
            self.current_scan_idx = idx
            self.slider_steps.blockSignals(True)
            self.slider_steps.setValue(idx)
            self.slider_steps.blockSignals(False)
            self._refresh_channel_dropdowns()
            self._update_display()
            # Update series highlight
            if self.series_metrics:
                self.canvas_series.set_series_data(
                    self.dataset.step_names, self.dataset.step_values,
                    self.series_metrics, current_step_idx=idx,
                    metric_name=f"ROI Mean ({self.combo_channel.currentText()})"
                )

    def _on_step_slider_changed(self, val: int):
        self.combo_steps.setCurrentIndex(val)

    def _on_prev_step(self):
        if self.current_scan_idx > 0:
            self.combo_steps.setCurrentIndex(self.current_scan_idx - 1)

    def _on_next_step(self):
        if self.dataset and self.current_scan_idx < len(self.dataset) - 1:
            self.combo_steps.setCurrentIndex(self.current_scan_idx + 1)

    def _on_series_step_clicked(self, step_idx: int):
        if 0 <= step_idx < len(self.dataset):
            self.combo_steps.setCurrentIndex(step_idx)

    def _on_channel_changed(self, _):
        txt = self.combo_channel.currentText()
        if "Overlay" in txt:
            if hasattr(self, "btn_sub_overlay") and not self.btn_sub_overlay.isChecked():
                self.btn_sub_overlay.setChecked(True)
                self.widget_overlay_inline.setVisible(True)
                self.lbl_sub_single_info.setVisible(False)
        else:
            if hasattr(self, "btn_sub_single") and not self.btn_sub_single.isChecked():
                self.btn_sub_single.setChecked(True)
                self.widget_overlay_inline.setVisible(False)
                self.lbl_sub_single_info.setVisible(True)
            if hasattr(self, "lbl_sub_single_info"):
                self.lbl_sub_single_info.setText(f"Active Channel: {txt}")

        self._update_display()
        self._update_series_trend()

    def _on_submode_toggled(self, button):
        is_overlay = (button == self.btn_sub_overlay)
        self.widget_overlay_inline.setVisible(is_overlay)
        self.lbl_sub_single_info.setVisible(not is_overlay)

        if is_overlay:
            self._update_display()
        else:
            txt = self.combo_channel.currentText()
            if "Overlay" in txt:
                if self.combo_channel.count() > 0:
                    self.combo_channel.setCurrentIndex(0)
            self._update_display()
            self._update_series_trend()

    def _on_phase_spin_changed(self, val: float):
        self.slider_phase.blockSignals(True)
        self.slider_phase.setValue(int(np.round(val)))
        self.slider_phase.blockSignals(False)
        self._update_display()

    def _on_phase_slider_changed(self, val: int):
        self.spin_phase.blockSignals(True)
        self.spin_phase.setValue(float(val))
        self.spin_phase.blockSignals(False)
        self._update_display()

    def _on_auto_phase_clicked(self):
        if not self.dataset or self.current_scan_idx >= len(self.dataset):
            return
        scan = self.dataset[self.current_scan_idx]
        optimal_deg = scan.calculate_auto_phase()
        self.spin_phase.setValue(optimal_deg)
        # If not already on a rotated channel, switch to PR_X_Rot
        if "Rotated" not in self.combo_channel.currentText():
            self.combo_channel.setCurrentIndex(4)  # PR_X_Rot
        self.statusBar.showMessage(f"Auto-phase calculated for scan {scan.step_name}: {optimal_deg:.2f}°", 4000)

    def _on_auto_phase_all_clicked(self):
        if not self.dataset:
            return
        msg = "Auto-Phase calculated per step:\n"
        for scan in self.dataset.scans:
            p = scan.calculate_auto_phase()
            msg += f"• Step {scan.step_name}: {p:.2f}°\n"
        QMessageBox.information(self, "Auto-Phase Results", msg)

    def _on_cmap_changed(self, cmap_name: str):
        self.canvas_2d.set_colormap(cmap_name)

    def _on_contrast_changed(self, _):
        self._apply_contrast()

    def _on_units_changed(self, idx: int):
        self.calib_unit = "um" if idx == 1 else "V"
        self._update_display()

    def _on_objective_changed(self, obj_name: str):
        if obj_name in OBJECTIVE_CALIBRATION:
            cx, cy = OBJECTIVE_CALIBRATION[obj_name]
            self.spin_calib_x.blockSignals(True)
            self.spin_calib_y.blockSignals(True)
            self.spin_calib_x.setValue(cx)
            self.spin_calib_y.setValue(cy)
            self.calib_x_scale = cx
            self.calib_y_scale = cy
            self.spin_calib_x.blockSignals(False)
            self.spin_calib_y.blockSignals(False)
            if hasattr(self, "dual_view"):
                self.dual_view.set_calibration(self.calib_x_scale)
            if self.calib_unit == "um":
                self._update_display()

    def _on_calib_changed(self):
        self.calib_x_scale = self.spin_calib_x.value()
        self.calib_y_scale = self.spin_calib_y.value()
        self.combo_objective.blockSignals(True)
        self.combo_objective.setCurrentText("Custom")
        self.combo_objective.blockSignals(False)
        if hasattr(self, "dual_view"):
            self.dual_view.set_calibration(self.calib_x_scale)
        if self.calib_unit == "um":
            self._update_display()

    def _on_invert_y_toggled(self, invert: bool):
        self.canvas_2d.set_invert_y(invert)
        if hasattr(self, "dual_view"):
            self.dual_view.set_invert_y(invert)
        if hasattr(self, "overlay_view") and self.overlay_view is not None:
            self.overlay_view.set_invert_y(invert)

    def _on_tool_mode_changed(self, button):
        mode_idx = self.btn_group_tools.id(button)
        modes = ["crosshair", "line", "roi", "caliper", "navigate"]
        if 0 <= mode_idx < len(modes):
            self.active_tool_mode = modes[mode_idx]
            self.canvas_2d.set_mode(self.active_tool_mode)
            if hasattr(self, "dual_view"):
                self.dual_view.set_mode(self.active_tool_mode)

    def _on_quick_hcut(self):
        self.rb_tool_line.setChecked(True)
        self._on_tool_mode_changed(self.rb_tool_line)
        if self.tabs.currentIndex() == 1:
            if self.dual_view.canvas_left.image_data is not None:
                rows = self.dual_view.canvas_left.image_data.shape[0]
                self.dual_view.set_horizontal_cut(rows // 2)
        else:
            if self.canvas_2d.image_data is not None:
                rows = self.canvas_2d.image_data.shape[0]
                self.canvas_2d.set_horizontal_cut(rows // 2)

    def _on_quick_vcut(self):
        self.rb_tool_line.setChecked(True)
        self._on_tool_mode_changed(self.rb_tool_line)
        if self.tabs.currentIndex() == 1:
            if self.dual_view.canvas_left.image_data is not None:
                cols = self.dual_view.canvas_left.image_data.shape[1]
                self.dual_view.set_vertical_cut(cols // 2)
        else:
            if self.canvas_2d.image_data is not None:
                cols = self.canvas_2d.image_data.shape[1]
                self.canvas_2d.set_vertical_cut(cols // 2)

    def _on_overlay_cbar_alpha_toggled(self, checked: bool):
        if checked and self.slider_overlay_alpha.value() != 100:
            self.slider_overlay_alpha.setValue(100)
        self.canvas_2d.set_overlay_cbar_alpha(checked)
        self.overlay_view.chk_cbar_alpha.blockSignals(True)
        self.overlay_view.chk_cbar_alpha.setChecked(checked)
        self.overlay_view.chk_cbar_alpha.blockSignals(False)
        self.overlay_view.canvas_2d.set_overlay_cbar_alpha(checked)
        if self._get_current_channel_key() == "OVERLAY":
            self._update_display()

    # Backwards-compatible alias
    _on_overlay_gradient_toggled = _on_overlay_cbar_alpha_toggled

    def _sync_gradient_from_view(self, checked: bool):
        self.chk_overlay_cbar_alpha.blockSignals(True)
        self.chk_overlay_cbar_alpha.setChecked(checked)
        self.chk_overlay_cbar_alpha.blockSignals(False)
        self.canvas_2d.set_overlay_cbar_alpha(checked)
        if self._get_current_channel_key() == "OVERLAY":
            self._update_display()

    def _on_sidebar_alpha_slider_changed(self, val: int):
        self.spin_overlay_alpha.blockSignals(True)
        self.spin_overlay_alpha.setValue(val)
        self.spin_overlay_alpha.blockSignals(False)
        alpha = val / 100.0
        self.canvas_2d.set_overlay_alpha(alpha)
        self.overlay_view.set_alpha(alpha)

    def _on_sidebar_alpha_spin_changed(self, val: int):
        self.slider_overlay_alpha.blockSignals(True)
        self.slider_overlay_alpha.setValue(val)
        self.slider_overlay_alpha.blockSignals(False)
        alpha = val / 100.0
        self.canvas_2d.set_overlay_alpha(alpha)
        self.overlay_view.set_alpha(alpha)

    def _sync_alpha_from_view(self, alpha: float):
        pct = int(np.round(alpha * 100))
        self.slider_overlay_alpha.blockSignals(True)
        self.spin_overlay_alpha.blockSignals(True)
        self.slider_overlay_alpha.setValue(pct)
        self.spin_overlay_alpha.setValue(pct)
        self.slider_overlay_alpha.blockSignals(False)
        self.spin_overlay_alpha.blockSignals(False)
        self.canvas_2d.set_overlay_alpha(alpha)

    def _on_overlay_param_changed(self):
        if self._get_current_channel_key() == "OVERLAY":
            self._update_display()
        self.overlay_view.refresh_display()

    def _on_pixel_hovered(self, col: int, row: int, x: float, y: float, val: float):
        if hasattr(self, "current_base_data") and self.current_base_data is not None:
            top_name = self.combo_overlay_pr_chan.currentText()
            base_name = self.combo_overlay_pd_chan.currentText()
            val_base = float(self.current_base_data[row, col])
            val_top = float(self.current_processed_data[row, col]) if self.current_processed_data is not None else val
            if (
                self.chk_overlay_cbar_alpha.isChecked()
                and hasattr(self, "current_processed_data")
                and self.current_processed_data is not None
            ):
                if self.canvas_2d.clim_overlay is not None:
                    v0, v1 = self.canvas_2d.clim_overlay
                else:
                    v0 = float(np.min(self.current_processed_data))
                    v1 = float(np.max(self.current_processed_data))
                span = float(v1 - v0)
                norm = float(np.clip((val_top - v0) / span, 0.0, 1.0)) if abs(span) > 1e-15 else 0.0
                local_pct = norm * self.slider_overlay_alpha.value()
                alpha_str = f"α={local_pct:.0f}% (Cbar Min 0% → Max {self.slider_overlay_alpha.value()}%)"
            else:
                alpha_str = f"α={self.slider_overlay_alpha.value()}%"

            self.lbl_cursor_val.setText(
                f"Cursor: ({x:.3f} {self.calib_unit}, {y:.3f} {self.calib_unit}) [c={col}, r={row}] "
                f"{top_name}: {val_top:.4e} | {base_name}: {val_base:.4e} ({alpha_str})"
            )
            self.statusBar.showMessage(
                f"X: {x:.4f} {self.calib_unit} | Y: {y:.4f} {self.calib_unit} | {top_name}: {val_top:.4e} | {base_name}: {val_base:.4e}"
            )
        else:
            self.lbl_cursor_val.setText(
                f"Cursor: ({x:.3f} {self.calib_unit}, {y:.3f} {self.calib_unit}) [col={col}, row={row}]  Val: {val:.4e}"
            )
            self.statusBar.showMessage(
                f"X: {x:.4f} {self.calib_unit} | Y: {y:.4f} {self.calib_unit} | Value: {val:.4e}"
            )

    def _on_line_cut_changed(self, c0: float, r0: float, c1: float, r1: float):
        self._update_line_profile(c0, r0, c1, r1)
        if hasattr(self, "dual_view"):
            self.dual_view.line_points = (c0, r0, c1, r1)
            self.dual_view.canvas_left.set_line_cut(c0, r0, c1, r1, emit_signal=False)
            self.dual_view.canvas_right.set_line_cut(c0, r0, c1, r1, emit_signal=False)
            self.dual_view.refresh_profile()

    def _on_roi_changed(self, c0: int, r0: int, c1: int, r1: int):
        self._update_roi_stats(c0, r0, c1, r1)
        self._update_series_trend()
        if hasattr(self, "dual_view"):
            self.dual_view.roi_box = (c0, r0, c1, r1)
            self.dual_view.canvas_left.set_roi(c0, r0, c1, r1, emit_signal=False)
            self.dual_view.canvas_right.set_roi(c0, r0, c1, r1, emit_signal=False)

    def _on_dual_line_cut_changed(self, c0: float, r0: float, c1: float, r1: float):
        self.canvas_2d.set_line_cut(c0, r0, c1, r1, emit_signal=False)

    def _on_dual_roi_changed(self, c0: int, r0: int, c1: int, r1: int):
        self.canvas_2d.set_roi(c0, r0, c1, r1, emit_signal=False)
        if self.dual_view.canvas_left.image_data is not None:
            self._update_roi_stats_with_data(self.dual_view.canvas_left.image_data, (c0, r0, c1, r1))
            self._update_series_trend()

    def _on_tab_changed(self, index: int):
        if index == 1:
            self._update_dual_view()
        elif index == 2:
            self._update_series_trend()
        self._update_quick_buttons_state()

    def _on_reset_filters(self):
        self.combo_plane.setCurrentIndex(0)
        self.combo_line_sub.setCurrentIndex(0)
        self.combo_filter.setCurrentIndex(0)
        self.spin_phase.setValue(0.0)
        self.combo_contrast.setCurrentText("Min-Max")
        self._update_display()

    # --- Export Features ---

    def _on_export_image(self):
        if self.current_processed_data is None:
            return
        path, filt = QFileDialog.getSaveFileName(
            self, "Export Current Image", "",
            "TIFF Image (*.tiff *.tif);;PNG Image (*.png);;PDF Publication Figure (*.pdf)"
        )
        if not path:
            return

        try:
            if path.lower().endswith((".tiff", ".tif")):
                from PIL import Image
                im = Image.fromarray(self.current_processed_data.astype(np.float32))
                im.save(path)
            else:
                self.canvas_2d.fig.savefig(path, dpi=300, bbox_inches="tight")
            self.statusBar.showMessage(f"Image successfully exported to {path}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export image: {e}")

    def _on_export_batch_summary(self):
        if not self.dataset:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Batch Summary CSV", "", "CSV Files (*.csv)")
        if not path:
            return

        try:
            channel_key = self._get_current_channel_key()
            phase_deg = self.spin_phase.value()
            bbox = self.canvas_2d.roi_box
            if bbox is None:
                shape = self.dataset[0].shape
                bbox = (0, 0, shape[1] - 1, shape[0] - 1)

            with open(path, "w") as f:
                f.write("Step_Name,Step_Value,Auto_Phase_Deg,ROI_Mean,ROI_Std,ROI_Min,ROI_Max,Peak_To_Peak\n")
                for scan in self.dataset.scans:
                    auto_p = scan.calculate_auto_phase()
                    proc = self._get_processed_image(scan, channel_key, phase_deg)
                    st = processing.calculate_roi_stats(proc, bbox)
                    f.write(f"{scan.step_name},{scan.step_value},{auto_p:.2f},{st['mean']:.6e},{st['std']:.6e},{st['min']:.6e},{st['max']:.6e},{st['peak_to_peak']:.6e}\n")

            self.statusBar.showMessage(f"Batch summary exported to {path}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export summary: {e}")

    # --- Recent Datasets Management ---

    def _get_recent_datasets(self) -> List[str]:
        settings = QSettings("SLMProcessor", "RecentDatasets")
        val = settings.value("history", [])
        if isinstance(val, str):
            return [val] if val and os.path.exists(val) else []
        elif isinstance(val, list):
            return [str(v) for v in val if v and os.path.exists(str(v))]
        return []

    def _save_recent_dataset(self, path: str):
        if not path or not os.path.exists(path):
            return
        abs_path = os.path.abspath(path)
        recent = self._get_recent_datasets()
        if abs_path in recent:
            recent.remove(abs_path)
        recent.insert(0, abs_path)
        recent = recent[:10]

        settings = QSettings("SLMProcessor", "RecentDatasets")
        settings.setValue("history", recent)
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        if not hasattr(self, "menu_recent"):
            return
        self.menu_recent.clear()
        recent = self._get_recent_datasets()
        if not recent:
            act_none = QAction("No Recent Datasets", self)
            act_none.setEnabled(False)
            self.menu_recent.addAction(act_none)
            return

        for i, p in enumerate(recent):
            base = os.path.basename(p) or p
            parent = os.path.basename(os.path.dirname(p))
            label = f"&{i+1}: {base} ({parent})" if i < 9 else f"{base} ({parent})"
            act = QAction(label, self)
            act.setToolTip(p)
            act.triggered.connect(lambda _, path=p: self.load_directory(path))
            self.menu_recent.addAction(act)

        self.menu_recent.addSeparator()
        act_clear = QAction("Clear Recent History", self)
        act_clear.triggered.connect(self._clear_recent_history)
        self.menu_recent.addAction(act_clear)

    def _clear_recent_history(self):
        settings = QSettings("SLMProcessor", "RecentDatasets")
        settings.setValue("history", [])
        self._rebuild_recent_menu()

    def _show_recent_popup(self):
        menu = QMenu(self)
        menu.setStyleSheet(DARK_STYLE)
        recent = self._get_recent_datasets()
        if not recent:
            act_none = QAction("No Recent Datasets", self)
            act_none.setEnabled(False)
            menu.addAction(act_none)
        else:
            for p in recent:
                base = os.path.basename(p) or p
                act = QAction(f"📂 {base}  [{p}]", self)
                act.triggered.connect(lambda _, path=p: self.load_directory(path))
                menu.addAction(act)
        menu.exec(self.btn_recent_menu.mapToGlobal(self.btn_recent_menu.rect().bottomLeft()))

    # --- Animation Playback & Series GIF Movie ---

    def _toggle_animation_playback(self):
        if not self.dataset or len(self.dataset) == 0:
            return
        if getattr(self, "is_anim_playing", False):
            self._stop_animation()
        else:
            self._start_animation()

    def _start_animation(self):
        self.is_anim_playing = True
        self.btn_play_pause.setText("⏸ Pause")
        self.btn_play_pause.setStyleSheet("font-weight: bold; background-color: #d32f2f; color: white; padding: 3px 10px; border-radius: 3px; font-size: 10px;")
        fps = self.spin_anim_fps.value()
        interval_ms = int(max(10, 1000 / max(1, fps)))
        self.anim_timer.start(interval_ms)

    def _stop_animation(self):
        self.is_anim_playing = False
        self.anim_timer.stop()
        if hasattr(self, "btn_play_pause"):
            self.btn_play_pause.setText("▶ Play")
            self.btn_play_pause.setStyleSheet("font-weight: bold; background-color: #007acc; color: white; padding: 3px 10px; border-radius: 3px; font-size: 10px;")

    def _on_anim_speed_changed(self, fps: int):
        if getattr(self, "is_anim_playing", False):
            interval_ms = int(max(10, 1000 / max(1, fps)))
            self.anim_timer.setInterval(interval_ms)

    def _on_animation_step(self):
        if not self.dataset or len(self.dataset) == 0:
            self._stop_animation()
            return
        next_idx = self.current_scan_idx + 1
        if next_idx >= len(self.dataset):
            if self.chk_anim_loop.isChecked():
                next_idx = 0
            else:
                self._stop_animation()
                return
        self.combo_steps.setCurrentIndex(next_idx)

    def _on_export_gif(self):
        if not self.dataset or len(self.dataset) == 0:
            QMessageBox.information(self, "No Dataset", "Please load a dataset first.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Series as Animated GIF", "", "GIF Animation (*.gif)")
        if not path:
            return
        if not path.lower().endswith(".gif"):
            path += ".gif"

        from PIL import Image
        import matplotlib.cm as cm

        fps = self.spin_anim_fps.value() if hasattr(self, "spin_anim_fps") else 10
        duration_ms = int(max(10, 1000 / max(1, fps)))

        channel_key = self._get_current_channel_key()
        phase_deg = self.spin_phase.value()
        cmap_name = self.combo_cmap.currentText()
        cmap = cm.get_cmap(cmap_name)

        progress = QProgressDialog("Rendering series frames for GIF...", "Cancel", 0, len(self.dataset), self)
        progress.setWindowTitle("Exporting Animated GIF")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        frames = []
        try:
            vmin, vmax = None, None
            if self.canvas_2d.clim is not None:
                vmin, vmax = self.canvas_2d.clim

            for i, scan in enumerate(self.dataset.scans):
                if progress.wasCanceled():
                    return
                progress.setValue(i)
                progress.setLabelText(f"Rendering frame {i+1} of {len(self.dataset)} (Step: {scan.step_name})...")
                QApplication.processEvents()

                img = self._get_processed_image(scan, channel_key, phase_deg)
                if vmin is None or vmax is None:
                    c_min, c_max = float(np.min(img)), float(np.max(img))
                else:
                    c_min, c_max = vmin, vmax

                span = max(1e-12, c_max - c_min)
                norm = np.clip((img - c_min) / span, 0.0, 1.0)
                rgba = (cmap(norm) * 255).astype(np.uint8)
                frame = Image.fromarray(rgba).convert("RGB")
                frames.append(frame)

            progress.setValue(len(self.dataset))
            if frames:
                frames[0].save(
                    path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=duration_ms,
                    loop=0
                )
                self.statusBar.showMessage(f"Series animation exported successfully to {path}", 5000)
                QMessageBox.information(self, "Export Complete", f"Successfully exported animated GIF to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export animated GIF:\n{e}")

    # --- Quick-Switch Shortcuts & Favorite Channels ---

    def _setup_shortcuts(self):
        self.sc_pd = QShortcut(QKeySequence("1"), self)
        self.sc_pd.activated.connect(self._quick_switch_pd)

        self.sc_pr = QShortcut(QKeySequence("2"), self)
        self.sc_pr.activated.connect(self._quick_switch_pr)

        self.sc_over = QShortcut(QKeySequence("3"), self)
        self.sc_over.activated.connect(self._quick_switch_overlay)

        self.sc_dual = QShortcut(QKeySequence("4"), self)
        self.sc_dual.activated.connect(self._quick_switch_dual)

    def _quick_switch_pd(self):
        """Switch to Photodiode (PD) channel in Single Channel view."""
        self.tabs.setCurrentIndex(0)
        if hasattr(self, "btn_sub_single") and hasattr(self, "btn_sub_overlay"):
            self.btn_sub_single.setChecked(True)
            self._on_submode_toggled(self.btn_sub_single)

        chans = [self.combo_channel.itemText(i) for i in range(self.combo_channel.count())]
        for c in ["PD_R", "X_PD", "PD_X", "PD"]:
            if c in chans:
                self.combo_channel.setCurrentText(c)
                self.statusBar.showMessage(f"Switched to Photodiode channel ({c}) [Hotkey: 1]", 3000)
                self._update_quick_buttons_state()
                return
        if chans:
            self.combo_channel.setCurrentIndex(0)
            self._update_quick_buttons_state()

    def _quick_switch_pr(self):
        """Switch to Photoresponse (PR) channel in Single Channel view."""
        self.tabs.setCurrentIndex(0)
        if hasattr(self, "btn_sub_single") and hasattr(self, "btn_sub_overlay"):
            self.btn_sub_single.setChecked(True)
            self._on_submode_toggled(self.btn_sub_single)

        chans = [self.combo_channel.itemText(i) for i in range(self.combo_channel.count())]
        for c in ["PR_R", "X_PR", "PR_X", "PR", "P4_R", "P4_X", "P4"]:
            if c in chans:
                self.combo_channel.setCurrentText(c)
                self.statusBar.showMessage(f"Switched to Photoresponse channel ({c}) [Hotkey: 2]", 3000)
                self._update_quick_buttons_state()
                return
        if len(chans) > 1:
            self.combo_channel.setCurrentIndex(1)
            self._update_quick_buttons_state()

    def _quick_switch_overlay(self):
        """Switch to PR on PD Overlay view."""
        self.tabs.setCurrentIndex(0)
        if hasattr(self, "btn_sub_overlay"):
            self.btn_sub_overlay.setChecked(True)
            self._on_submode_toggled(self.btn_sub_overlay)
            self.statusBar.showMessage("Switched to PR on PD Overlay View [Hotkey: 3]", 3000)
            self._update_quick_buttons_state()

    def _quick_switch_dual(self):
        """Toggle to Dual Side-by-Side View (or back to Single)."""
        if self.tabs.currentIndex() == 1:
            self.tabs.setCurrentIndex(0)
            self.statusBar.showMessage("Returned to 2D Single Image & Cut [Hotkey: 4]", 3000)
        else:
            self.tabs.setCurrentIndex(1)
            self.statusBar.showMessage("Switched to Dual Side-by-Side View [Hotkey: 4]", 3000)
        self._update_quick_buttons_state()

    def _update_quick_buttons_state(self):
        if not hasattr(self, "btn_quick_pd"):
            return
        is_dual = (self.tabs.currentIndex() == 1)
        is_over = (self.tabs.currentIndex() == 0 and hasattr(self, "btn_sub_overlay") and self.btn_sub_overlay.isChecked())
        curr_chan = self.combo_channel.currentText()

        self.btn_quick_dual.blockSignals(True)
        self.btn_quick_overlay.blockSignals(True)
        self.btn_quick_pd.blockSignals(True)
        self.btn_quick_pr.blockSignals(True)

        self.btn_quick_dual.setChecked(is_dual)
        self.btn_quick_overlay.setChecked(is_over and not is_dual)
        self.btn_quick_pd.setChecked(not is_dual and not is_over and ("PD" in curr_chan))
        self.btn_quick_pr.setChecked(not is_dual and not is_over and ("PR" in curr_chan or "P4" in curr_chan))

        self.btn_quick_dual.blockSignals(False)
        self.btn_quick_overlay.blockSignals(False)
        self.btn_quick_pd.blockSignals(False)
        self.btn_quick_pr.blockSignals(False)

    def _show_about(self):
        QMessageBox.about(
            self,
            "About Scanning Laser Microscopy Processor",
            "<b>Scanning Laser Microscopy (SLM) Image Processor</b><br>"
            "An application for processing lock-in laser microscopy images.<br><br>"
            "<b>Features:</b><br>"
            "• Photoresponse (PR) and Photodiode (PD) in-phase and quadrature processing<br>"
            "• Lock-in Amplitude, Phase, and Normalized signals (PR/PD)<br>"
            "• Interactive phase rotation & auto-phase optimization<br>"
            "• Plane tilt and polynomial surface leveling<br>"
            "• Line-by-line background subtraction & spatial filtering<br>"
            "• Real-time crosshairs, 1D line cuts, and ROI statistics<br>"
            "• Dual comparison view & step series trend plots<br>"
            "• Mirror voltage to physical distance calibration (µm/V)"
        )
