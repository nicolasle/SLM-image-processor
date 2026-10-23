"""Interactive Channel Manager and Signal Workbench for SLM microscopy.
Provides live channel previews, statistical health metrics, NumPy formula engine,
direct canvas dispatching, and multi-channel dataset management.
"""

from typing import Dict, List, Optional, Any, Tuple
import os
import re
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QTreeWidget,
    QTreeWidgetItem, QGroupBox, QMessageBox, QTabWidget,
    QHeaderView, QFrame, QSplitter, QTableWidget, QTableWidgetItem,
    QMenu, QFileDialog, QProgressDialog, QApplication
)
from PyQt6.QtCore import pyqtSignal, Qt, QPoint
from PyQt6.QtGui import QColor, QFont, QAction, QIcon
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PIL import Image

from slm_processor.data_model import SLMDataset


class SLMChannelManagerDialog(QDialog):
    """Signal Workbench & Multi-Channel Manager for Scanning Laser Microscopy.

    Includes live channel thumbnail preview, statistical health metrics,
    free-form NumPy mathematical channel calculator, and instant canvas dispatch.
    """
    channels_updated = pyqtSignal()
    channel_selected = pyqtSignal(str)          # Emits channel name to load in 2D view
    send_to_dual = pyqtSignal(str, str)         # channel_name, "left" or "right"
    send_to_overlay = pyqtSignal(str, str)      # channel_name, "top" or "base"

    def __init__(
        self,
        dataset: SLMDataset,
        parent: Optional[QWidget] = None,
        all_datasets: Optional[Dict[str, SLMDataset]] = None
    ):
        super().__init__(parent)
        self.dataset = dataset
        self.all_datasets: Dict[str, SLMDataset] = all_datasets or {}

        if not self.all_datasets and self.dataset:
            ds_name = os.path.basename(self.dataset.root_dir) if self.dataset.root_dir else "Active Dataset"
            if getattr(self.dataset, "is_harmonic_series", False):
                ds_name += f" (Harmonics, {len(self.dataset)} steps)"
            elif getattr(self.dataset, "series_variable_name", "") == "Time (ms)":
                ds_name += f" (Time Domain, {len(self.dataset)} slices)"
            else:
                ds_name += f" ({len(self.dataset)} scans)"
            self.all_datasets[ds_name] = self.dataset

        self.selected_channel_name: Optional[str] = None
        self.selected_dataset: Optional[SLMDataset] = self.dataset
        self.preview_step_idx: int = 0
        self.preview_cbar = None

        self.setWindowTitle("Multi-Channel Manager & Signal Workbench")
        self.resize(1080, 720)
        self.setStyleSheet("""
            QDialog { background-color: #1a1a1a; color: #e0e0e0; font-size: 11px; }
            QGroupBox { border: 1px solid #383838; border-radius: 6px; margin-top: 8px; font-weight: bold; color: #00e5ff; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QTreeWidget { background-color: #222222; border: 1px solid #383838; color: #ffffff; font-size: 11px; }
            QTreeWidget::item { padding: 3px 0; }
            QTreeWidget::item:selected { background-color: #0277bd; color: #ffffff; }
            QHeaderView::section { background-color: #2a2a2a; color: #00e5ff; font-weight: bold; padding: 4px; border: 1px solid #383838; font-size: 11px; }
            QComboBox, QLineEdit, QPushButton { background-color: #2d2d2d; color: #ffffff; border: 1px solid #4a4a4a; border-radius: 4px; padding: 4px 8px; }
            QPushButton:hover { background-color: #3d3d3d; border-color: #007acc; }
            QPushButton#primary { background-color: #007acc; font-weight: bold; }
            QPushButton#primary:hover { background-color: #1a8ad6; }
            QPushButton#dispatch { background-color: #1e3a5f; color: #00e5ff; border: 1px solid #00bcd4; font-weight: bold; padding: 4px 8px; }
            QPushButton#dispatch:hover { background-color: #0288d1; color: white; }
            QTabWidget::pane { border: 1px solid #383838; background-color: #222222; }
            QTabBar::tab { background: #2b2b2b; color: #cccccc; padding: 6px 12px; margin-right: 2px; }
            QTabBar::tab:selected { background: #007acc; color: #ffffff; font-weight: bold; }
            QTableWidget { background-color: #202020; gridline-color: #333333; color: #e0e0e0; }
            QSplitter::handle:horizontal { background-color: #2e2e2e; width: 6px; }
            QSplitter::handle:horizontal:hover { background-color: #007acc; }
        """)

        self._build_ui()
        self._refresh_lists()
        self._select_first_channel()

    def set_active_dataset(self, dataset: SLMDataset, all_datasets: Optional[Dict[str, SLMDataset]] = None):
        """Allows modeless updating when the active dataset changes in the parent window."""
        self.dataset = dataset
        self.all_datasets = all_datasets or {os.path.basename(dataset.root_dir): dataset}
        self.selected_dataset = dataset
        self._refresh_lists()
        self._select_first_channel()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Header Info Banner
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #222b35; border: 1px solid #007acc; border-radius: 4px; padding: 2px;")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(8, 4, 8, 4)
        lbl_head = QLabel("⚡ <b>SLM Signal Workbench</b> — Inspect channel health, compose NumPy expressions, and dispatch signals live to views.")
        lbl_head.setStyleSheet("color: #4fc3f7; font-size: 11px;")
        header_layout.addWidget(lbl_head)
        header_layout.addStretch(1)
        main_layout.addWidget(header_frame)

        # Main Horizontal Splitter: Left (Tree + Formula Creator) & Right (Inspector + Previews)
        self.splitter_main = QSplitter(Qt.Orientation.Horizontal)
        self.splitter_main.setHandleWidth(6)

        # ==============================================================
        # LEFT PANE: Hierarchy, Search, and Channel Creator
        # ==============================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # 1. Channels Tree Group
        group_tree = QGroupBox("Channel Hierarchy & Tags")
        tree_layout = QVBoxLayout(group_tree)
        tree_layout.setContentsMargins(6, 6, 6, 6)
        tree_layout.setSpacing(4)

        # Search bar and collapse/expand
        row_filter = QHBoxLayout()
        self.edit_filter = QLineEdit()
        self.edit_filter.setPlaceholderText("🔍 Filter channels by name...")
        self.edit_filter.textChanged.connect(self._filter_tree)
        row_filter.addWidget(self.edit_filter, 1)

        self.btn_fav_toggle = QPushButton("⭐ Star")
        self.btn_fav_toggle.setToolTip("Toggle favorite star on selected channel (pins it to top of dropdowns)")
        self.btn_fav_toggle.clicked.connect(self._on_toggle_favorite)
        row_filter.addWidget(self.btn_fav_toggle)

        btn_expand = QPushButton("▼")
        btn_expand.setFixedWidth(26)
        btn_expand.setToolTip("Expand all folders")
        btn_expand.clicked.connect(self.tree_channels_expand_all)
        btn_collapse = QPushButton("▲")
        btn_collapse.setFixedWidth(26)
        btn_collapse.setToolTip("Collapse all folders")
        btn_collapse.clicked.connect(self.tree_channels_collapse_all)
        row_filter.addWidget(btn_expand)
        row_filter.addWidget(btn_collapse)
        tree_layout.addLayout(row_filter)

        # Hierarchical Tree
        self.tree_channels = QTreeWidget()
        self.tree_channels.setColumnCount(3)
        self.tree_channels.setHeaderLabels(["Channel / Subfolder", "Type", "Source / Formula"])
        self.tree_channels.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree_channels.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree_channels.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tree_channels.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self.tree_channels.itemDoubleClicked.connect(self._on_tree_double_clicked)
        self.tree_channels.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_channels.customContextMenuRequested.connect(self._on_tree_context_menu)
        tree_layout.addWidget(self.tree_channels, 1)

        # Delete & Counter Row
        row_tree_bottom = QHBoxLayout()
        self.lbl_channel_count = QLabel("Active Channels: -")
        self.lbl_channel_count.setStyleSheet("color: #888888; font-size: 10px;")
        row_tree_bottom.addWidget(self.lbl_channel_count)
        row_tree_bottom.addStretch(1)

        self.btn_remove = QPushButton("Remove Derived Channel")
        self.btn_remove.setStyleSheet("background-color: #8a1f1f; color: #ffffff; padding: 2px 8px; font-size: 10px;")
        self.btn_remove.setEnabled(False)
        self.btn_remove.clicked.connect(self._on_remove_channel)
        row_tree_bottom.addWidget(self.btn_remove)
        tree_layout.addLayout(row_tree_bottom)
        left_layout.addWidget(group_tree, 3)

        # 2. Lower: Creation Tools Tabs
        self.tabs_create = QTabWidget()

        # Target Dataset Selector (if multi-dataset)
        self.combo_target_ds = QComboBox()
        if len(self.all_datasets) > 1:
            ds_frame = QFrame()
            ds_lay = QHBoxLayout(ds_frame)
            ds_lay.setContentsMargins(2, 2, 2, 2)
            ds_lay.addWidget(QLabel("Target Dataset:"))
            self.combo_target_ds.addItem("All Active Datasets", "all")
            for name, ds in self.all_datasets.items():
                self.combo_target_ds.addItem(name, ds)
            ds_lay.addWidget(self.combo_target_ds, 1)
            left_layout.addWidget(ds_frame)

        # Tab A: Arbitrary NumPy Math Expression
        tab_expr = QWidget()
        lay_expr = QVBoxLayout(tab_expr)
        lay_expr.setContentsMargins(6, 6, 6, 6)
        lay_expr.setSpacing(4)

        row_expr_name = QHBoxLayout()
        row_expr_name.addWidget(QLabel("Channel Name:"))
        self.edit_expr_name = QLineEdit("Norm_PR")
        row_expr_name.addWidget(self.edit_expr_name, 1)

        row_expr_name.addWidget(QLabel("Preset:"))
        self.combo_expr_presets = QComboBox()
        self.combo_expr_presets.addItems([
            "Select Preset...",
            "Laser Normalization: PR_R / np.maximum(1e-5, PD_R)",
            "Differential Contrast: (PD_R - np.mean(PD_R)) / np.mean(PD_R)",
            "Asymmetry Ratio: (PR_X - PD_X) / np.maximum(1e-5, PR_X + PD_X)",
            "Decibel Scale: 20 * np.log10(np.maximum(1e-9, PR_R))",
            "Microvolt Conversion: PR_X * 1e6"
        ])
        self.combo_expr_presets.currentIndexChanged.connect(self._on_preset_selected)
        row_expr_name.addWidget(self.combo_expr_presets, 1)
        lay_expr.addLayout(row_expr_name)

        row_code = QHBoxLayout()
        row_code.addWidget(QLabel("Formula:"))
        self.edit_expr_code = QLineEdit("PR_R / np.maximum(1e-5, PD_R)")
        self.edit_expr_code.setPlaceholderText("e.g., (PR_R - 0.002) / np.maximum(1e-5, PD_R) or PR_X * 1e6")
        self.edit_expr_code.textChanged.connect(self._validate_expression_preview)
        row_code.addWidget(self.edit_expr_code, 1)
        lay_expr.addLayout(row_code)

        # Token Helper Chips
        row_chips = QHBoxLayout()
        row_chips.setSpacing(2)
        row_chips.addWidget(QLabel("Insert:"))
        for token in ["PR_R", "PD_R", "PR_X", "PR_Y", "PD_X", "np.maximum", "np.abs", "np.log10"]:
            btn_tok = QPushButton(token)
            btn_tok.setStyleSheet("font-size: 9px; padding: 1px 4px; background-color: #333333;")
            btn_tok.clicked.connect(lambda _, t=token: self.edit_expr_code.insert(t))
            row_chips.addWidget(btn_tok)
        row_chips.addStretch(1)
        lay_expr.addLayout(row_chips)

        # Validation status label
        self.lbl_expr_status = QLabel("Enter expression to evaluate")
        self.lbl_expr_status.setStyleSheet("color: #aaaaaa; font-size: 10px; font-style: italic;")
        lay_expr.addWidget(self.lbl_expr_status)

        self.btn_create_expr = QPushButton("🧮 Compute & Add Custom Expression Channel")
        self.btn_create_expr.setObjectName("primary")
        self.btn_create_expr.clicked.connect(self._on_create_expression_channel)
        lay_expr.addWidget(self.btn_create_expr)
        self.tabs_create.addTab(tab_expr, "🧮 Custom Expression (NumPy)")

        # Tab B: Lock-in Pair
        tab_lockin = QWidget()
        lockin_layout = QGridLayout(tab_lockin)
        lockin_layout.setContentsMargins(6, 6, 6, 6)
        lockin_layout.addWidget(QLabel("Pair Name / Prefix:"), 0, 0)
        self.edit_pair_name = QLineEdit("LockIn")
        lockin_layout.addWidget(self.edit_pair_name, 0, 1)

        lockin_layout.addWidget(QLabel("In-Phase (X):"), 1, 0)
        self.combo_x_chan = QComboBox()
        lockin_layout.addWidget(self.combo_x_chan, 1, 1)

        lockin_layout.addWidget(QLabel("Quadrature (Y):"), 2, 0)
        self.combo_y_chan = QComboBox()
        lockin_layout.addWidget(self.combo_y_chan, 2, 1)

        self.btn_add_lockin = QPushButton("Create Lock-in Channels (R, θ, X_Rot, Y_Rot)")
        self.btn_add_lockin.setObjectName("primary")
        self.btn_add_lockin.clicked.connect(self._on_create_lockin_pair)
        lockin_layout.addWidget(self.btn_add_lockin, 3, 0, 1, 2)
        self.tabs_create.addTab(tab_lockin, "⚡ Lock-in Pair")

        # Tab C: Ratio (A / B)
        tab_ratio = QWidget()
        ratio_layout = QGridLayout(tab_ratio)
        ratio_layout.setContentsMargins(6, 6, 6, 6)
        ratio_layout.addWidget(QLabel("Channel Name:"), 0, 0)
        self.edit_ratio_name = QLineEdit("Ratio_A_B")
        ratio_layout.addWidget(self.edit_ratio_name, 0, 1)

        ratio_layout.addWidget(QLabel("Numerator (A):"), 1, 0)
        self.combo_num_chan = QComboBox()
        ratio_layout.addWidget(self.combo_num_chan, 1, 1)

        ratio_layout.addWidget(QLabel("Denominator (B):"), 2, 0)
        self.combo_denom_chan = QComboBox()
        ratio_layout.addWidget(self.combo_denom_chan, 2, 1)

        self.btn_add_ratio = QPushButton("Create Ratio Channel (A / B)")
        self.btn_add_ratio.setObjectName("primary")
        self.btn_add_ratio.clicked.connect(self._on_create_ratio)
        ratio_layout.addWidget(self.btn_add_ratio, 3, 0, 1, 2)
        self.tabs_create.addTab(tab_ratio, "➗ Ratio (A / B)")

        # Tab D: Difference (A - B)
        tab_diff = QWidget()
        diff_layout = QGridLayout(tab_diff)
        diff_layout.setContentsMargins(6, 6, 6, 6)
        diff_layout.addWidget(QLabel("Channel Name:"), 0, 0)
        self.edit_diff_name = QLineEdit("Diff_A_B")
        diff_layout.addWidget(self.edit_diff_name, 0, 1)

        diff_layout.addWidget(QLabel("Channel A:"), 1, 0)
        self.combo_diff_a = QComboBox()
        diff_layout.addWidget(self.combo_diff_a, 1, 1)

        diff_layout.addWidget(QLabel("Channel B:"), 2, 0)
        self.combo_diff_b = QComboBox()
        diff_layout.addWidget(self.combo_diff_b, 2, 1)

        self.btn_add_diff = QPushButton("Create Difference Channel (A - B)")
        self.btn_add_diff.setObjectName("primary")
        self.btn_add_diff.clicked.connect(self._on_create_diff)
        diff_layout.addWidget(self.btn_add_diff, 3, 0, 1, 2)
        self.tabs_create.addTab(tab_diff, "➖ Difference (A - B)")

        left_layout.addWidget(self.tabs_create, 2)
        self.splitter_main.addWidget(left_widget)

        # ==============================================================
        # RIGHT PANE: Live Channel Preview, Quality Health & Dispatch
        # ==============================================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        group_inspector = QGroupBox("Channel Inspector & Live Preview")
        insp_layout = QVBoxLayout(group_inspector)
        insp_layout.setContentsMargins(8, 8, 8, 8)
        insp_layout.setSpacing(6)

        # Header controls for preview
        row_prev_ctrl = QHBoxLayout()
        self.lbl_selected_title = QLabel("Selected: None")
        self.lbl_selected_title.setStyleSheet("font-weight: bold; color: #00e5ff; font-size: 12px;")
        row_prev_ctrl.addWidget(self.lbl_selected_title, 1)

        row_prev_ctrl.addWidget(QLabel("Colormap:"))
        self.combo_preview_cmap = QComboBox()
        self.combo_preview_cmap.addItems(["cividis", "inferno", "viridis", "gray", "bone", "coolwarm", "turbo", "twilight"])
        self.combo_preview_cmap.setCurrentText("cividis")
        self.combo_preview_cmap.currentTextChanged.connect(self._update_preview)
        row_prev_ctrl.addWidget(self.combo_preview_cmap)

        row_prev_ctrl.addWidget(QLabel("Scan Step:"))
        self.combo_preview_step = QComboBox()
        self.combo_preview_step.currentIndexChanged.connect(self._on_preview_step_changed)
        row_prev_ctrl.addWidget(self.combo_preview_step)
        insp_layout.addLayout(row_prev_ctrl)

        # Matplotlib Live Preview Canvas
        self.fig_preview = Figure(figsize=(4.0, 3.2), dpi=95, layout="compressed")
        self.fig_preview.patch.set_facecolor("#1e1e1e")
        self.ax_preview = self.fig_preview.add_subplot(111)
        self.ax_preview.set_facecolor("#141414")
        self.canvas_preview = FigureCanvasQTAgg(self.fig_preview)
        self.canvas_preview.setMinimumHeight(240)
        insp_layout.addWidget(self.canvas_preview, 1)

        # Channel Statistical Health Table
        self.table_stats = QTableWidget(4, 2)
        self.table_stats.setHorizontalHeaderLabels(["Statistical Metric", "Value"])
        self.table_stats.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_stats.verticalHeader().setVisible(False)
        self.table_stats.setFixedHeight(115)

        stat_metrics = ["Dynamic Range [Min .. Max]", "Mean ± Std Dev (RMS)", "Peak-to-Peak & SNR (dB)", "Dimensions & Data Type"]
        for r, m in enumerate(stat_metrics):
            item_m = QTableWidgetItem(m)
            item_m.setFlags(Qt.ItemFlag.ItemIsEnabled)
            item_v = QTableWidgetItem("-")
            item_v.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table_stats.setItem(r, 0, item_m)
            self.table_stats.setItem(r, 1, item_v)
        insp_layout.addWidget(self.table_stats)

        # Dispatch Action Buttons
        dispatch_box = QGroupBox("Signal Dispatch (Send to Main Views)")
        dispatch_layout = QGridLayout(dispatch_box)
        dispatch_layout.setContentsMargins(6, 6, 6, 6)
        dispatch_layout.setSpacing(4)

        self.btn_view_2d = QPushButton("👁 View in 2D Image Canvas")
        self.btn_view_2d.setObjectName("dispatch")
        self.btn_view_2d.setToolTip("Sets this channel as active single channel in the main window 2D view")
        self.btn_view_2d.clicked.connect(self._on_dispatch_view_2d)
        dispatch_layout.addWidget(self.btn_view_2d, 0, 0, 1, 2)

        self.btn_send_dual_l = QPushButton("◫ Send to Dual (Left)")
        self.btn_send_dual_l.clicked.connect(self._on_dispatch_dual_left)
        dispatch_layout.addWidget(self.btn_send_dual_l, 1, 0)

        self.btn_send_dual_r = QPushButton("◫ Send to Dual (Right)")
        self.btn_send_dual_r.clicked.connect(self._on_dispatch_dual_right)
        dispatch_layout.addWidget(self.btn_send_dual_r, 1, 1)

        self.btn_send_overlay_top = QPushButton("🔀 Set as Overlay Top (PR)")
        self.btn_send_overlay_top.clicked.connect(self._on_dispatch_overlay_top)
        dispatch_layout.addWidget(self.btn_send_overlay_top, 2, 0)

        self.btn_send_overlay_base = QPushButton("🔀 Set as Overlay Base (PD)")
        self.btn_send_overlay_base.clicked.connect(self._on_dispatch_overlay_base)
        dispatch_layout.addWidget(self.btn_send_overlay_base, 2, 1)

        self.btn_export_channel = QPushButton("💾 Export Channel (Active / Entire Series)...")
        self.btn_export_channel.setStyleSheet("font-weight: bold; color: #ffd54f;")
        self.btn_export_channel.clicked.connect(self._on_export_channel)
        dispatch_layout.addWidget(self.btn_export_channel, 3, 0, 1, 2)

        insp_layout.addWidget(dispatch_box)
        right_layout.addWidget(group_inspector)
        self.splitter_main.addWidget(right_widget)

        self.splitter_main.setSizes([480, 560])
        main_layout.addWidget(self.splitter_main, 1)

        # Bottom Bar: Close button
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch(1)
        btn_close = QPushButton("Close Workbench")
        btn_close.setStyleSheet("padding: 5px 16px; font-weight: bold;")
        btn_close.clicked.connect(self.close)
        bottom_bar.addWidget(btn_close)
        main_layout.addLayout(bottom_bar)

    def tree_channels_expand_all(self):
        self.tree_channels.expandAll()

    def tree_channels_collapse_all(self):
        self.tree_channels.collapseAll()

    @property
    def list_channels(self):
        """Backward compatibility adapter for list_channels.count()."""
        dialog = self
        class _ListChannelsCompat:
            def count(self_inner):
                total = 0
                def count_leaves(item):
                    nonlocal total
                    data = item.data(0, Qt.ItemDataRole.UserRole)
                    if data and data.get("is_channel"):
                        total += 1
                    for i in range(item.childCount()):
                        count_leaves(item.child(i))
                for i in range(dialog.tree_channels.topLevelItemCount()):
                    count_leaves(dialog.tree_channels.topLevelItem(i))
                return total
        return _ListChannelsCompat()

    def _select_first_channel(self):
        """Selects the first leaf channel in the tree on startup."""
        for i in range(self.tree_channels.topLevelItemCount()):
            ds_item = self.tree_channels.topLevelItem(i)
            for j in range(ds_item.childCount()):
                folder = ds_item.child(j)
                for k in range(folder.childCount()):
                    ch_item = folder.child(k)
                    data = ch_item.data(0, Qt.ItemDataRole.UserRole)
                    if data and data.get("is_channel"):
                        self.tree_channels.setCurrentItem(ch_item)
                        return

    def _refresh_lists(self):
        """Rebuilds the hierarchical tree widget with subfolders and favorite stars."""
        self.tree_channels.clear()

        total_raw = 0
        total_derived = 0

        # Build tree for each dataset
        for ds_name, ds in self.all_datasets.items():
            if not ds or len(ds) == 0:
                continue

            scan0 = ds[0]
            raw_keys = list(set(scan0.raw_channels.keys()) | set(scan0.file_paths.keys()))
            derived_defs = scan0.derived_defs
            favorites = getattr(ds, "favorite_channels", set())

            total_raw += len(raw_keys)
            total_derived += len(derived_defs)

            # 1. Dataset Root Item
            ds_item = QTreeWidgetItem(self.tree_channels)
            ds_item.setText(0, f"🗂 {ds_name}")
            ds_item.setText(1, "Dataset")
            ds_item.setText(2, f"{len(ds)} scans | Root: {ds.root_dir or 'Memory'}")
            ds_item.setForeground(0, QColor("#00e5ff"))
            font_ds = QFont()
            font_ds.setBold(True)
            ds_item.setFont(0, font_ds)
            ds_item.setData(0, Qt.ItemDataRole.UserRole, {"is_folder": True, "dataset": ds, "name": ds_name})

            # 2. Raw Channels Subfolder
            raw_folder = QTreeWidgetItem(ds_item)
            raw_folder.setText(0, f"📁 Raw Channels ({len(raw_keys)})")
            raw_folder.setText(1, "Folder")
            raw_folder.setText(2, "Direct hardware scan images on disk")
            raw_folder.setForeground(0, QColor("#ffd54f"))
            font_folder = QFont()
            font_folder.setBold(True)
            raw_folder.setFont(0, font_folder)
            raw_folder.setData(0, Qt.ItemDataRole.UserRole, {"is_folder": True, "dataset": ds})

            shape_str = f"{scan0.shape[0]}×{scan0.shape[1]}" if hasattr(scan0, "shape") else "2D"
            for raw_ch in sorted(raw_keys, key=lambda c: (0 if c in favorites else 1, c)):
                item = QTreeWidgetItem(raw_folder)
                fav_badge = "⭐ " if raw_ch in favorites else "  "
                item.setText(0, f"{fav_badge}🖼 {raw_ch}")
                item.setText(1, "Raw Image")
                item.setText(2, f"Shape: {shape_str} | Original TIFF File")
                item.setForeground(0, QColor("#ffd700") if raw_ch in favorites else QColor("#ffffff"))
                item.setData(0, Qt.ItemDataRole.UserRole, {
                    "is_channel": True,
                    "is_raw": True,
                    "name": raw_ch,
                    "dataset": ds
                })

            # 3. Derived Channels Subfolder
            derived_folder = QTreeWidgetItem(ds_item)
            derived_folder.setText(0, f"📁 Derived Channels ({len(derived_defs)})")
            derived_folder.setText(1, "Folder")
            derived_folder.setText(2, "Computed signals, expressions & polar components")
            derived_folder.setForeground(0, QColor("#ffd54f"))
            derived_folder.setFont(0, font_folder)
            derived_folder.setData(0, Qt.ItemDataRole.UserRole, {"is_folder": True, "dataset": ds})

            lockin_items: Dict[str, List[Tuple[str, Any]]] = {}
            ratio_items = []
            diff_items = []
            expr_items = []
            other_items = []

            for ch_name, d in derived_defs.items():
                if d.kind in ("lockin_r", "lockin_theta", "lockin_x_rot", "lockin_y_rot"):
                    prefix = ch_name.split("_")[0]
                    if prefix not in lockin_items:
                        lockin_items[prefix] = []
                    lockin_items[prefix].append((ch_name, d))
                elif d.kind == "ratio":
                    ratio_items.append((ch_name, d))
                elif d.kind == "diff":
                    diff_items.append((ch_name, d))
                elif d.kind == "expression":
                    expr_items.append((ch_name, d))
                else:
                    other_items.append((ch_name, d))

            # A. Lock-in Pairs Subfolder
            if lockin_items:
                lockin_folder = QTreeWidgetItem(derived_folder)
                total_lockin_chans = sum(len(v) for v in lockin_items.values())
                lockin_folder.setText(0, f"📁 Lock-in Pairs ({total_lockin_chans})")
                lockin_folder.setText(1, "Category")
                lockin_folder.setText(2, "Polar amplitude (R), phase (θ), and rotated components")
                lockin_folder.setForeground(0, QColor("#e0e0e0"))
                lockin_folder.setFont(0, font_folder)

                for prefix, pair_chans in sorted(lockin_items.items()):
                    pair_folder = QTreeWidgetItem(lockin_folder)
                    pair_folder.setText(0, f"📁 Pair '{prefix}'")
                    pair_folder.setText(1, "Lock-in Suite")
                    src_str = f"X: {pair_chans[0][1].source_a}, Y: {pair_chans[0][1].source_b}" if pair_chans else ""
                    pair_folder.setText(2, src_str)
                    pair_folder.setForeground(0, QColor("#81d4fa"))

                    for ch_name, d in pair_chans:
                        item = QTreeWidgetItem(pair_folder)
                        kind_label = {
                            "lockin_r": "Amplitude (R)",
                            "lockin_theta": "Phase (θ)",
                            "lockin_x_rot": "Rotated In-Phase (X')",
                            "lockin_y_rot": "Rotated Quadrature (Y')"
                        }.get(d.kind, d.kind)
                        formula_label = {
                            "lockin_r": f"√({d.source_a}² + {d.source_b}²)",
                            "lockin_theta": f"atan2({d.source_b}, {d.source_a})",
                            "lockin_x_rot": f"{d.source_a}·cos(φ) + {d.source_b}·sin(φ)",
                            "lockin_y_rot": f"-{d.source_a}·sin(φ) + {d.source_b}·cos(φ)"
                        }.get(d.kind, f"{d.source_a} & {d.source_b}")

                        fav_badge = "⭐ " if ch_name in favorites else "  "
                        item.setText(0, f"{fav_badge}⚡ {ch_name}")
                        item.setText(1, kind_label)
                        item.setText(2, formula_label)
                        item.setForeground(0, QColor("#ffd700") if ch_name in favorites else QColor("#4fc3f7"))
                        item.setData(0, Qt.ItemDataRole.UserRole, {
                            "is_channel": True,
                            "is_raw": False,
                            "name": ch_name,
                            "dataset": ds
                        })
                    pair_folder.setExpanded(True)
                lockin_folder.setExpanded(True)

            # B. Custom Expressions Subfolder
            if expr_items:
                expr_folder = QTreeWidgetItem(derived_folder)
                expr_folder.setText(0, f"📁 Custom Expressions ({len(expr_items)})")
                expr_folder.setText(1, "Category")
                expr_folder.setText(2, "NumPy mathematical expressions")
                expr_folder.setForeground(0, QColor("#e0e0e0"))
                expr_folder.setFont(0, font_folder)

                for ch_name, d in sorted(expr_items, key=lambda x: (0 if x[0] in favorites else 1, x[0])):
                    item = QTreeWidgetItem(expr_folder)
                    fav_badge = "⭐ " if ch_name in favorites else "  "
                    item.setText(0, f"{fav_badge}🧮 {ch_name}")
                    item.setText(1, "Custom Math")
                    item.setText(2, f"{d.expression}")
                    item.setForeground(0, QColor("#ffd700") if ch_name in favorites else QColor("#80cbc4"))
                    item.setData(0, Qt.ItemDataRole.UserRole, {
                        "is_channel": True,
                        "is_raw": False,
                        "name": ch_name,
                        "dataset": ds
                    })
                expr_folder.setExpanded(True)

            # C. Ratio Channels Subfolder
            if ratio_items:
                ratio_folder = QTreeWidgetItem(derived_folder)
                ratio_folder.setText(0, f"📁 Ratios ({len(ratio_items)})")
                ratio_folder.setText(1, "Category")
                ratio_folder.setText(2, "Normalized ratio channels (A / B)")
                ratio_folder.setForeground(0, QColor("#e0e0e0"))
                ratio_folder.setFont(0, font_folder)

                for ch_name, d in sorted(ratio_items, key=lambda x: (0 if x[0] in favorites else 1, x[0])):
                    item = QTreeWidgetItem(ratio_folder)
                    fav_badge = "⭐ " if ch_name in favorites else "  "
                    item.setText(0, f"{fav_badge}➗ {ch_name}")
                    item.setText(1, "Ratio")
                    item.setText(2, f"{d.source_a} / {d.source_b}")
                    item.setForeground(0, QColor("#ffd700") if ch_name in favorites else QColor("#b388ff"))
                    item.setData(0, Qt.ItemDataRole.UserRole, {
                        "is_channel": True,
                        "is_raw": False,
                        "name": ch_name,
                        "dataset": ds
                    })
                ratio_folder.setExpanded(True)

            # D. Difference Channels Subfolder
            if diff_items:
                diff_folder = QTreeWidgetItem(derived_folder)
                diff_folder.setText(0, f"📁 Differences ({len(diff_items)})")
                diff_folder.setText(1, "Category")
                diff_folder.setText(2, "Differential background subtraction (A - B)")
                diff_folder.setForeground(0, QColor("#e0e0e0"))
                diff_folder.setFont(0, font_folder)

                for ch_name, d in sorted(diff_items, key=lambda x: (0 if x[0] in favorites else 1, x[0])):
                    item = QTreeWidgetItem(diff_folder)
                    fav_badge = "⭐ " if ch_name in favorites else "  "
                    item.setText(0, f"{fav_badge}➖ {ch_name}")
                    item.setText(1, "Difference")
                    item.setText(2, f"{d.source_a} - {d.source_b}")
                    item.setForeground(0, QColor("#ffd700") if ch_name in favorites else QColor("#ff80ab"))
                    item.setData(0, Qt.ItemDataRole.UserRole, {
                        "is_channel": True,
                        "is_raw": False,
                        "name": ch_name,
                        "dataset": ds
                    })
                diff_folder.setExpanded(True)

            ds_item.setExpanded(True)
            raw_folder.setExpanded(True)
            derived_folder.setExpanded(True)

        self.lbl_channel_count.setText(
            f"Active Channels: {total_raw + total_derived} total ({total_raw} raw, {total_derived} derived)"
        )

        # Update creation dropdowns with channels from active dataset
        active_scan = self.dataset[0] if (self.dataset and len(self.dataset) > 0) else None
        all_chans = active_scan.available_channels if active_scan else []

        for combo in [
            self.combo_x_chan, self.combo_y_chan,
            self.combo_num_chan, self.combo_denom_chan,
            self.combo_diff_a, self.combo_diff_b
        ]:
            curr = combo.currentText()
            combo.clear()
            combo.addItems(all_chans)
            if curr in all_chans:
                combo.setCurrentText(curr)

        # Populate scan steps selector for preview
        self.combo_preview_step.blockSignals(True)
        self.combo_preview_step.clear()
        if self.dataset:
            for s in self.dataset.scans:
                self.combo_preview_step.addItem(f"Step {s.step_name}")
            self.combo_preview_step.setCurrentIndex(min(self.preview_step_idx, len(self.dataset) - 1))
        self.combo_preview_step.blockSignals(False)

    def _filter_tree(self, text: str):
        search = text.strip().lower()

        def match_and_filter(item: QTreeWidgetItem) -> bool:
            item_text = (item.text(0) + " " + item.text(1) + " " + item.text(2)).lower()
            direct_match = search in item_text if search else True

            child_matched = False
            for i in range(item.childCount()):
                child = item.child(i)
                if match_and_filter(child):
                    child_matched = True

            should_show = direct_match or child_matched
            item.setHidden(not should_show)
            if should_show and search:
                item.setExpanded(True)
            return should_show

        for i in range(self.tree_channels.topLevelItemCount()):
            match_and_filter(self.tree_channels.topLevelItem(i))

    def _on_tree_selection_changed(self):
        item = self.tree_channels.currentItem()
        if not item:
            self._clear_inspector()
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data or not data.get("is_channel"):
            self._clear_inspector()
            return

        ch_name = data.get("name", "")
        is_raw = data.get("is_raw", False)
        ds = data.get("dataset", self.dataset)

        self.selected_channel_name = ch_name
        self.selected_dataset = ds

        self.btn_remove.setEnabled(not is_raw)
        self.btn_remove.setText(f"Remove '{ch_name}'" if not is_raw else "Cannot Remove Raw Disk Channel")

        # Update Live Preview & Health Inspector
        self._update_preview()

    def _clear_inspector(self):
        self.selected_channel_name = None
        self.lbl_selected_title.setText("Selected: None")
        self.btn_remove.setEnabled(False)
        self.btn_remove.setText("Remove Derived Channel")
        self.ax_preview.clear()
        self.ax_preview.set_facecolor("#141414")
        self.canvas_preview.draw_idle()

    def _on_preview_step_changed(self, idx: int):
        if idx >= 0:
            self.preview_step_idx = idx
            self._update_preview()

    def _update_preview(self):
        """Fetches 2D data for the selected channel and updates preview canvas and metrics table."""
        if not self.selected_channel_name or not self.selected_dataset or len(self.selected_dataset) == 0:
            return

        step_idx = min(self.preview_step_idx, len(self.selected_dataset) - 1)
        scan = self.selected_dataset[step_idx]

        try:
            arr = scan.get_channel(self.selected_channel_name)
        except Exception as e:
            self.lbl_selected_title.setText(f"Error loading '{self.selected_channel_name}': {e}")
            return

        if arr is None or arr.size == 0:
            return

        self.lbl_selected_title.setText(f"Selected: {self.selected_channel_name} (Step: {scan.step_name})")

        # 1. Update Preview Plot
        self.ax_preview.clear()
        self.ax_preview.set_facecolor("#141414")
        cmap_name = self.combo_preview_cmap.currentText()

        v0 = float(np.min(arr))
        v1 = float(np.max(arr))
        im = self.ax_preview.imshow(arr, cmap=cmap_name, origin="lower", vmin=v0, vmax=v1, interpolation="nearest")

        if self.preview_cbar is None:
            self.preview_cbar = self.fig_preview.colorbar(im, ax=self.ax_preview, fraction=0.046, pad=0.04)
        else:
            self.preview_cbar.update_normal(im)

        self.preview_cbar.ax.yaxis.set_tick_params(color="#aaaaaa", labelsize=7)
        for label in self.preview_cbar.ax.yaxis.get_ticklabels():
            label.set_color("#aaaaaa")

        self.ax_preview.tick_params(colors="#777777", labelsize=7)
        for spine in self.ax_preview.spines.values():
            spine.set_color("#3a3a3a")
        self.canvas_preview.draw_idle()

        # 2. Update Health & Statistical Metrics
        v_min = float(np.min(arr))
        v_max = float(np.max(arr))
        v_p2p = v_max - v_min
        v_mean = float(np.mean(arr))
        v_std = float(np.std(arr))

        # Estimate SNR from corner background vs central signal
        rows, cols = arr.shape
        c_r = max(1, rows // 8)
        c_c = max(1, cols // 8)
        corners = np.concatenate([
            arr[:c_r, :c_c].flatten(),
            arr[:c_r, -c_c:].flatten(),
            arr[-c_r:, :c_c].flatten(),
            arr[-c_r:, -c_c:].flatten()
        ])
        noise_floor = float(np.std(corners)) if corners.size > 0 else v_std
        snr_ratio = (v_p2p / max(1e-12, noise_floor))
        snr_db = 20.0 * np.log10(max(1e-6, snr_ratio))

        self.table_stats.item(0, 1).setText(f"{v_min:.4e}  to  {v_max:.4e}")
        self.table_stats.item(1, 1).setText(f"{v_mean:.4e}  ±  {v_std:.4e}")
        self.table_stats.item(2, 1).setText(f"Peak-to-Peak: {v_p2p:.4e} | SNR: {snr_db:.1f} dB")
        self.table_stats.item(3, 1).setText(f"{cols} × {rows} pixels ({arr.dtype})")

    def _on_tree_double_clicked(self, item: QTreeWidgetItem, col: int):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data.get("is_channel"):
            self._on_dispatch_view_2d()

    def _on_tree_context_menu(self, pos: QPoint):
        item = self.tree_channels.itemAt(pos)
        if not item:
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data or not data.get("is_channel"):
            return

        ch_name = data.get("name", "")
        is_raw = data.get("is_raw", False)

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #262626; color: #ffffff; border: 1px solid #444444; }
            QMenu::item:selected { background-color: #007acc; }
        """)

        act_2d = menu.addAction(f"👁 View '{ch_name}' in Main 2D Canvas")
        act_2d.triggered.connect(self._on_dispatch_view_2d)

        menu.addSeparator()
        act_dl = menu.addAction(f"◫ Send '{ch_name}' to Dual View 1 (Left)")
        act_dl.triggered.connect(self._on_dispatch_dual_left)
        act_dr = menu.addAction(f"◫ Send '{ch_name}' to Dual View 2 (Right)")
        act_dr.triggered.connect(self._on_dispatch_dual_right)

        menu.addSeparator()
        act_ot = menu.addAction(f"🔀 Set '{ch_name}' as Overlay Top (PR)")
        act_ot.triggered.connect(self._on_dispatch_overlay_top)
        act_ob = menu.addAction(f"🔀 Set '{ch_name}' as Overlay Base (PD)")
        act_ob.triggered.connect(self._on_dispatch_overlay_base)

        menu.addSeparator()
        act_star = menu.addAction("⭐ Toggle Favorite (Pin to Top)")
        act_star.triggered.connect(self._on_toggle_favorite)

        act_exp = menu.addAction(f"💾 Export '{ch_name}'...")
        act_exp.triggered.connect(self._on_export_channel)

        if not is_raw:
            menu.addSeparator()
            act_del = menu.addAction(f"🗑 Remove Derived Channel '{ch_name}'")
            act_del.triggered.connect(self._on_remove_channel)

        menu.exec(self.tree_channels.viewport().mapToGlobal(pos))

    def _on_toggle_favorite(self):
        if not self.selected_channel_name or not self.selected_dataset:
            return
        is_fav = self.selected_dataset.toggle_favorite_channel(self.selected_channel_name)
        self._refresh_lists()
        self.channels_updated.emit()

    def _on_dispatch_view_2d(self):
        if self.selected_channel_name:
            self.channel_selected.emit(self.selected_channel_name)

    def _on_dispatch_dual_left(self):
        if self.selected_channel_name:
            self.send_to_dual.emit(self.selected_channel_name, "left")

    def _on_dispatch_dual_right(self):
        if self.selected_channel_name:
            self.send_to_dual.emit(self.selected_channel_name, "right")

    def _on_dispatch_overlay_top(self):
        if self.selected_channel_name:
            self.send_to_overlay.emit(self.selected_channel_name, "top")

    def _on_dispatch_overlay_base(self):
        if self.selected_channel_name:
            self.send_to_overlay.emit(self.selected_channel_name, "base")

    def _on_export_channel(self):
        if not self.selected_channel_name or not self.selected_dataset:
            return

        ch = self.selected_channel_name
        ds = self.selected_dataset

        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background-color: #2b2b2b; color: white; border: 1px solid #444; }")
        act_single_tif = menu.addAction("Export Active Scan as 32-bit TIFF...")
        act_single_csv = menu.addAction("Export Active Scan as Matrix CSV...")
        act_all_tif = menu.addAction(f"Export Entire Series ({len(ds)} scans) to Folder as TIFFs...")

        chosen = menu.exec(self.btn_export_channel.mapToGlobal(QPoint(0, self.btn_export_channel.height())))
        if not chosen:
            return

        step_idx = min(self.preview_step_idx, len(ds) - 1)
        scan = ds[step_idx]

        if chosen == act_single_tif:
            path, _ = QFileDialog.getSaveFileName(self, f"Export Channel '{ch}' as TIFF", f"{scan.step_name}_{ch}.tiff", "TIFF (*.tiff *.tif)")
            if path:
                arr = scan.get_channel(ch)
                im = Image.fromarray(arr.astype(np.float32))
                im.save(path)
                QMessageBox.information(self, "Export Complete", f"Successfully exported to {path}")

        elif chosen == act_single_csv:
            path, _ = QFileDialog.getSaveFileName(self, f"Export Channel '{ch}' as CSV", f"{scan.step_name}_{ch}.csv", "CSV (*.csv)")
            if path:
                arr = scan.get_channel(ch)
                np.savetxt(path, arr, delimiter=",", fmt="%.6e")
                QMessageBox.information(self, "Export Complete", f"Successfully exported to {path}")

        elif chosen == act_all_tif:
            folder = QFileDialog.getExistingDirectory(self, f"Select Output Folder for Series Export of '{ch}'")
            if not folder:
                return

            progress = QProgressDialog(f"Exporting '{ch}' series to TIFFs...", "Cancel", 0, len(ds), self)
            progress.setWindowTitle("Batch Series Export")
            progress.show()

            for i, s in enumerate(ds.scans):
                if progress.wasCanceled():
                    return
                progress.setValue(i)
                progress.setLabelText(f"Exporting scan {i+1} of {len(ds)} ({s.step_name})...")
                QApplication.processEvents()

                arr = s.get_channel(ch)
                safe_step = re.sub(r'\W+', '_', s.step_name)
                out_name = f"scan_{safe_step}_{ch}.tiff"
                out_path = os.path.join(folder, out_name)
                im = Image.fromarray(arr.astype(np.float32))
                im.save(out_path)

            progress.setValue(len(ds))
            QMessageBox.information(self, "Export Complete", f"Successfully exported {len(ds)} TIFF scans of '{ch}' to:\n{folder}")

    # ==============================================================
    # MATHEMATICAL EXPRESSION ENGINE
    # ==============================================================

    def _on_preset_selected(self, idx: int):
        presets = [
            ("", ""),
            ("Norm_PR", "PR_R / np.maximum(1e-5, PD_R)"),
            ("Diff_Contrast", "(PD_R - np.mean(PD_R)) / np.mean(PD_R)"),
            ("Asymmetry", "(PR_X - PD_X) / np.maximum(1e-5, PR_X + PD_X)"),
            ("Power_dB", "20 * np.log10(np.maximum(1e-9, PR_R))"),
            ("PR_uV", "PR_X * 1e6"),
        ]
        if 0 < idx < len(presets):
            name, code = presets[idx]
            self.edit_expr_name.setText(name)
            self.edit_expr_code.setText(code)

    def _validate_expression_preview(self):
        code = self.edit_expr_code.text().strip()
        if not code or not self.dataset or len(self.dataset) == 0:
            self.lbl_expr_status.setText("Enter expression to evaluate")
            self.lbl_expr_status.setStyleSheet("color: #aaaaaa;")
            return

        scan = self.dataset[0]
        try:
            res = scan.evaluate_expression(code)
            v0 = float(np.min(res))
            v1 = float(np.max(res))
            self.lbl_expr_status.setText(f"✓ Valid: Output shape {res.shape[1]}×{res.shape[0]}, Range: [{v0:.3e} .. {v1:.3e}]")
            self.lbl_expr_status.setStyleSheet("color: #00e676; font-weight: bold;")
        except Exception as e:
            self.lbl_expr_status.setText(f"✗ Error: {e}")
            self.lbl_expr_status.setStyleSheet("color: #ff5252;")

    def _get_target_datasets(self) -> List[SLMDataset]:
        if len(self.all_datasets) > 1 and self.combo_target_ds.count() > 0:
            target_data = self.combo_target_ds.currentData()
            if target_data == "all":
                return list(self.all_datasets.values())
            elif isinstance(target_data, SLMDataset):
                return [target_data]
        return [self.dataset]

    def _on_create_expression_channel(self):
        name = self.edit_expr_name.text().strip()
        code = self.edit_expr_code.text().strip()
        if not name or not code:
            QMessageBox.warning(self, "Invalid Input", "Please provide both a channel name and a valid mathematical formula.")
            return

        targets = self._get_target_datasets()
        try:
            # Test on first scan of first target dataset
            targets[0][0].evaluate_expression(code)
        except Exception as e:
            QMessageBox.critical(self, "Formula Evaluation Error", f"Failed to evaluate expression on scan data:\n{e}")
            return

        for ds in targets:
            ds.add_expression_to_all(name, code)

        self._refresh_lists()
        self.channels_updated.emit()
        QMessageBox.information(
            self,
            "Channel Created",
            f"Successfully created custom expression channel '{name}':\nFormula: {code}\nApplied to {len(targets)} dataset(s)."
        )

    def _on_create_lockin_pair(self):
        pair = self.edit_pair_name.text().strip()
        x_ch = self.combo_x_chan.currentText()
        y_ch = self.combo_y_chan.currentText()
        if not pair:
            QMessageBox.warning(self, "Invalid Name", "Please specify a pair name prefix.")
            return

        targets = self._get_target_datasets()
        for ds in targets:
            ds.add_lockin_pair_to_all(pair, x_ch, y_ch)

        self._refresh_lists()
        self.channels_updated.emit()
        QMessageBox.information(
            self,
            "Channels Created",
            f"Successfully created lock-in suite for '{pair}':\n• {pair}_R\n• {pair}_Theta\n• {pair}_X_Rot\n• {pair}_Y_Rot"
        )

    def _on_create_ratio(self):
        name = self.edit_ratio_name.text().strip()
        num_ch = self.combo_num_chan.currentText()
        denom_ch = self.combo_denom_chan.currentText()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please specify a channel name.")
            return

        targets = self._get_target_datasets()
        for ds in targets:
            ds.add_ratio_to_all(name, num_ch, denom_ch)

        self._refresh_lists()
        self.channels_updated.emit()

    def _on_create_diff(self):
        name = self.edit_diff_name.text().strip()
        ch_a = self.combo_diff_a.currentText()
        ch_b = self.combo_diff_b.currentText()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please specify a channel name.")
            return

        targets = self._get_target_datasets()
        for ds in targets:
            ds.add_diff_to_all(name, ch_a, ch_b)

        self._refresh_lists()
        self.channels_updated.emit()

    def _on_remove_channel(self):
        item = self.tree_channels.currentItem()
        if not item:
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data or not data.get("is_channel") or data.get("is_raw"):
            return

        target_ch = data.get("name")
        target_ds = data.get("dataset", self.dataset)

        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove derived channel '{target_ch}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        target_ds.remove_derived_from_all(target_ch)
        self._refresh_lists()
        self.channels_updated.emit()
