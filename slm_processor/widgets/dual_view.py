"""Dual side-by-side view widget for comparing arbitrary channels (e.g. PD vs PR) with synchronized interactive tools."""

import numpy as np
from typing import Optional, List, Tuple
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from slm_processor.widgets.image_canvas import SLMImageCanvas
from slm_processor.widgets.profile_canvas import SLMProfileCanvas
from slm_processor import processing


class SLMDualView(QWidget):
    """Side-by-side comparison view with independent channel selectors,

    independent colormaps, synchronized pan/zoom navigation, and fully synchronized
    interactive tools (crosshair readout, line profile cuts, and ROI boxes).
    """
    channel_changed = pyqtSignal()
    line_cut_changed = pyqtSignal(float, float, float, float)
    roi_changed = pyqtSignal(int, int, int, int)
    pixel_hovered = pyqtSignal(int, int, float, float, float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._sync_enabled = True
        self.active_mode = "crosshair"
        self.calib_um_per_v: Optional[float] = None
        self.line_points: Optional[Tuple[float, float, float, float]] = None
        self.roi_box: Optional[Tuple[int, int, int, int]] = None
        self._updating = False

        # Left view (View 1)
        self.canvas_left = SLMImageCanvas(self, title="View 1")
        self.combo_left = QComboBox()
        self.combo_left.currentIndexChanged.connect(self._on_channel_selection_changed)
        
        self.combo_left_cmap = QComboBox()
        self.combo_left_cmap.addItems(["cividis", "gray", "bone", "Greys_r", "viridis", "inferno", "coolwarm"])
        self.combo_left_cmap.setCurrentText("cividis")
        self.combo_left_cmap.currentTextChanged.connect(self._on_left_cmap_changed)

        # Right view (View 2)
        self.canvas_right = SLMImageCanvas(self, title="View 2")
        self.combo_right = QComboBox()
        self.combo_right.currentIndexChanged.connect(self._on_channel_selection_changed)

        self.combo_right_cmap = QComboBox()
        self.combo_right_cmap.addItems(["inferno", "turbo", "plasma", "viridis", "coolwarm", "hot", "magma", "gray"])
        self.combo_right_cmap.currentTextChanged.connect(self._on_right_cmap_changed)

        # Sync checkbox
        self.chk_sync = QCheckBox("Synchronize Pan / Zoom & Tools")
        self.chk_sync.setChecked(True)
        self.chk_sync.toggled.connect(self._toggle_sync)

        # Header bar
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #252525; border-radius: 4px; padding: 2px;")
        header = QHBoxLayout(header_frame)
        header.setContentsMargins(6, 4, 6, 4)
        header.setSpacing(8)

        lbl_v1 = QLabel("View 1 (Left):")
        lbl_v1.setStyleSheet("color: #4da6ff; font-weight: bold;")
        header.addWidget(lbl_v1)
        header.addWidget(self.combo_left)
        header.addWidget(self.combo_left_cmap)

        header.addSpacing(25)

        lbl_v2 = QLabel("View 2 (Right):")
        lbl_v2.setStyleSheet("color: #ff3366; font-weight: bold;")
        header.addWidget(lbl_v2)
        header.addWidget(self.combo_right)
        header.addWidget(self.combo_right_cmap)

        header.addStretch(1)
        header.addWidget(self.chk_sync)

        # Readout label
        self.lbl_readout = QLabel("Cursor: Hover over either image to inspect coordinates and both channel values")
        self.lbl_readout.setStyleSheet("color: #00ffcc; font-size: 10px; padding: 2px 4px;")

        # 1D Profile Cut canvas
        self.canvas_profile = SLMProfileCanvas(self)

        # Horizontal Splitter for the two 2D canvases
        splitter_h = QSplitter(Qt.Orientation.Horizontal)
        splitter_h.addWidget(self.canvas_left)
        splitter_h.addWidget(self.canvas_right)
        splitter_h.setSizes([500, 500])

        # Vertical Splitter between 2D images and bottom 1D profile cut
        splitter_v = QSplitter(Qt.Orientation.Vertical)
        splitter_v.addWidget(splitter_h)
        splitter_v.addWidget(self.canvas_profile)
        splitter_v.setSizes([550, 220])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(header_frame)
        layout.addWidget(self.lbl_readout)
        layout.addWidget(splitter_v, 1)

        # Sync canvas axis changes via canvas limits_changed signals
        self.canvas_left.limits_changed.connect(self._on_left_limits_changed)
        self.canvas_right.limits_changed.connect(self._on_right_limits_changed)

        # Sync aspect ratio when sync is enabled
        self.canvas_left.combo_aspect.currentIndexChanged.connect(self._sync_aspect_from_left)
        self.canvas_right.combo_aspect.currentIndexChanged.connect(self._sync_aspect_from_right)

        # Connect interactive tool signals
        self.canvas_left.line_cut_changed.connect(self._on_left_line_cut_changed)
        self.canvas_right.line_cut_changed.connect(self._on_right_line_cut_changed)

        self.canvas_left.roi_changed.connect(self._on_left_roi_changed)
        self.canvas_right.roi_changed.connect(self._on_right_roi_changed)

        self.canvas_left.pixel_hovered.connect(self._on_left_pixel_hovered)
        self.canvas_right.pixel_hovered.connect(self._on_right_pixel_hovered)

    def set_channels_list(self, channels: List[str]):
        """Populates the channel dropdowns with current channels in dataset."""
        curr_l = self.combo_left.currentText()
        curr_r = self.combo_right.currentText()

        self.combo_left.blockSignals(True)
        self.combo_right.blockSignals(True)

        self.combo_left.clear()
        self.combo_right.clear()

        self.combo_left.addItems(channels)
        self.combo_right.addItems(channels)

        # Set default selection
        if curr_l in channels:
            self.combo_left.setCurrentText(curr_l)
        else:
            for c in ["PD_R", "X_PD", "PD_X", "PD"]:
                if c in channels:
                    self.combo_left.setCurrentText(c)
                    break
            else:
                if len(channels) > 0:
                    self.combo_left.setCurrentIndex(0)

        if curr_r in channels:
            self.combo_right.setCurrentText(curr_r)
        else:
            for c in ["PR_R", "X_PR", "PR_X", "PR", "P4_R", "P4_X", "P4"]:
                if c in channels:
                    self.combo_right.setCurrentText(c)
                    break
            else:
                if len(channels) > 1:
                    self.combo_right.setCurrentIndex(1)

        self.combo_left.blockSignals(False)
        self.combo_right.blockSignals(False)

    def _on_channel_selection_changed(self):
        self.channel_changed.emit()
        self.refresh_profile()

    def _on_left_cmap_changed(self, cmap: str):
        self.canvas_left.set_colormap(cmap)

    def _on_right_cmap_changed(self, cmap: str):
        self.canvas_right.set_colormap(cmap)

    def _toggle_sync(self, enabled: bool):
        self._sync_enabled = enabled

    def _sync_aspect_from_left(self, idx: int):
        if self._sync_enabled and not self._updating:
            self._updating = True
            if self.canvas_right.combo_aspect.currentIndex() != idx:
                self.canvas_right.combo_aspect.setCurrentIndex(idx)
            self._updating = False

    def _sync_aspect_from_right(self, idx: int):
        if self._sync_enabled and not self._updating:
            self._updating = True
            if self.canvas_left.combo_aspect.currentIndex() != idx:
                self.canvas_left.combo_aspect.setCurrentIndex(idx)
            self._updating = False

    def _on_left_limits_changed(self, xlim: Tuple[float, float], ylim: Tuple[float, float]):
        if not self._sync_enabled or self._updating:
            return
        self._updating = True
        self.canvas_right._updating_limits = True
        self.canvas_right.ax.set_xlim(xlim)
        self.canvas_right.ax.set_ylim(ylim)
        self.canvas_right.canvas.draw_idle()
        self.canvas_right._updating_limits = False
        self._updating = False

    def _on_right_limits_changed(self, xlim: Tuple[float, float], ylim: Tuple[float, float]):
        if not self._sync_enabled or self._updating:
            return
        self._updating = True
        self.canvas_left._updating_limits = True
        self.canvas_left.ax.set_xlim(xlim)
        self.canvas_left.ax.set_ylim(ylim)
        self.canvas_left.canvas.draw_idle()
        self.canvas_left._updating_limits = False
        self._updating = False

    # --- Interactive Tools Synchronization ---

    def set_mode(self, mode: str):
        """Sets interactive tool mode ('crosshair', 'line', 'roi', 'navigate') across both canvases."""
        self.active_mode = mode
        self.canvas_left.set_mode(mode)
        self.canvas_right.set_mode(mode)

    def set_invert_y(self, invert: bool):
        """Sets Y-axis inversion across both canvases."""
        self._updating = True
        self.canvas_left.set_invert_y(invert)
        self.canvas_right.set_invert_y(invert)
        self._updating = False

    def set_aspect_mode(self, mode: str):
        """Sets aspect ratio mode across both canvases."""
        self.canvas_left.set_aspect_mode(mode)
        self.canvas_right.set_aspect_mode(mode)

    def reset_view(self):
        """Resets view extent and centering across both canvases."""
        self.canvas_left.reset_view()
        self.canvas_right.reset_view()

    def set_horizontal_cut(self, row: int):
        """Sets horizontal cut at row across both canvases."""
        if self.canvas_left.image_data is None:
            return
        self.canvas_left.set_horizontal_cut(row)
        if self.canvas_right.image_data is not None:
            self.canvas_right.set_horizontal_cut(row)
        if self.canvas_left.line_points:
            self.line_points = self.canvas_left.line_points
            self._update_dual_line_cut(*self.line_points)
            self.line_cut_changed.emit(*self.line_points)

    def set_vertical_cut(self, col: int):
        """Sets vertical cut at col across both canvases."""
        if self.canvas_left.image_data is None:
            return
        self.canvas_left.set_vertical_cut(col)
        if self.canvas_right.image_data is not None:
            self.canvas_right.set_vertical_cut(col)
        if self.canvas_left.line_points:
            self.line_points = self.canvas_left.line_points
            self._update_dual_line_cut(*self.line_points)
            self.line_cut_changed.emit(*self.line_points)

    def set_calibration(self, calib_um_per_v: Optional[float]):
        """Sets calibration for 1D profile physical metric conversion."""
        self.calib_um_per_v = calib_um_per_v
        if self.line_points:
            self.refresh_profile()

    def _on_left_line_cut_changed(self, c0: float, r0: float, c1: float, r1: float):
        self.line_points = (c0, r0, c1, r1)
        if self._sync_enabled:
            self.canvas_right.set_line_cut(c0, r0, c1, r1, emit_signal=False)
        self._update_dual_line_cut(c0, r0, c1, r1)
        self.line_cut_changed.emit(c0, r0, c1, r1)

    def _on_right_line_cut_changed(self, c0: float, r0: float, c1: float, r1: float):
        self.line_points = (c0, r0, c1, r1)
        if self._sync_enabled:
            self.canvas_left.set_line_cut(c0, r0, c1, r1, emit_signal=False)
        self._update_dual_line_cut(c0, r0, c1, r1)
        self.line_cut_changed.emit(c0, r0, c1, r1)

    def _on_left_roi_changed(self, c0: int, r0: int, c1: int, r1: int):
        self.roi_box = (c0, r0, c1, r1)
        if self._sync_enabled:
            self.canvas_right.set_roi(c0, r0, c1, r1, emit_signal=False)
        self.roi_changed.emit(c0, r0, c1, r1)

    def _on_right_roi_changed(self, c0: int, r0: int, c1: int, r1: int):
        self.roi_box = (c0, r0, c1, r1)
        if self._sync_enabled:
            self.canvas_left.set_roi(c0, r0, c1, r1, emit_signal=False)
        self.roi_changed.emit(c0, r0, c1, r1)

    def _on_left_pixel_hovered(self, col: int, row: int, x: float, y: float, val: float):
        if self.active_mode == "crosshair" and self._sync_enabled:
            self.canvas_right.set_crosshair(col, row)

        val_r = 0.0
        if self.canvas_right.image_data is not None:
            rows, cols = self.canvas_right.image_data.shape
            if 0 <= row < rows and 0 <= col < cols:
                val_r = float(self.canvas_right.image_data[row, col])

        name_l = self.combo_left.currentText() or "View 1"
        name_r = self.combo_right.currentText() or "View 2"
        unit = self.canvas_left.unit_label

        self.lbl_readout.setText(
            f"Cursor: ({x:.3f} {unit}, {y:.3f} {unit}) [col={col}, row={row}]  |  "
            f"{name_l}: {val:.4e}  |  {name_r}: {val_r:.4e}"
        )
        self.pixel_hovered.emit(col, row, x, y, val)

    def _on_right_pixel_hovered(self, col: int, row: int, x: float, y: float, val: float):
        if self.active_mode == "crosshair" and self._sync_enabled:
            self.canvas_left.set_crosshair(col, row)

        val_l = 0.0
        if self.canvas_left.image_data is not None:
            rows, cols = self.canvas_left.image_data.shape
            if 0 <= row < rows and 0 <= col < cols:
                val_l = float(self.canvas_left.image_data[row, col])

        name_l = self.combo_left.currentText() or "View 1"
        name_r = self.combo_right.currentText() or "View 2"
        unit = self.canvas_right.unit_label

        self.lbl_readout.setText(
            f"Cursor: ({x:.3f} {unit}, {y:.3f} {unit}) [col={col}, row={row}]  |  "
            f"{name_l}: {val_l:.4e}  |  {name_r}: {val:.4e}"
        )
        self.pixel_hovered.emit(col, row, x, y, val)

    def refresh_profile(self):
        """Refreshes 1D profile cut using current channels and dataset data."""
        if self.line_points is not None:
            self._update_dual_line_cut(*self.line_points)

    def _update_dual_line_cut(self, c0: float, r0: float, c1: float, r1: float):
        if self.canvas_left.image_data is None or self.canvas_right.image_data is None:
            return

        distances_pix, vals_left, _ = processing.extract_line_profile(
            self.canvas_left.image_data, (c0, r0), (c1, r1)
        )
        _, vals_right, _ = processing.extract_line_profile(
            self.canvas_right.image_data, (c0, r0), (c1, r1)
        )

        cols = self.canvas_left.image_data.shape[1]
        x_coords = self.canvas_left.x_coords
        if x_coords is not None and len(x_coords) > 1:
            x_span = abs(float(x_coords[-1] - x_coords[0]))
            dx_per_pix = x_span / max(1, cols - 1)
        else:
            dx_per_pix = 1.0
        distances_phys = distances_pix * dx_per_pix

        name_l = self.combo_left.currentText() or "View 1"
        name_r = self.combo_right.currentText() or "View 2"
        unit = self.canvas_left.unit_label

        self.canvas_profile.set_dual_profile(
            distances=distances_phys,
            values_pr=vals_right,
            values_pd=vals_left,
            unit=unit,
            title=f"Dual Cut: {name_l} & {name_r} from ({c0:.0f}, {r0:.0f}) to ({c1:.0f}, {r1:.0f})",
            calib_um_per_v=self.calib_um_per_v,
            label_pr=name_r,
            label_pd=name_l
        )
