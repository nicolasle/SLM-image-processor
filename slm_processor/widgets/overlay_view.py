"""Dedicated Multi-Channel Overlay View widget for blending any two channels with interactive alpha."""

import numpy as np
from typing import Optional, Tuple, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider,
    QSpinBox, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from slm_processor.widgets.image_canvas import SLMImageCanvas
from slm_processor.widgets.profile_canvas import SLMProfileCanvas
from slm_processor import processing


class SLMOverlayView(QWidget):
    """Dedicated workspace for overlapping ANY two channels with interactive alpha."""

    alpha_changed = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.current_scan = None
        self.x_coords = None
        self.y_coords = None
        self.unit_label = "V"
        self.phase_deg = 0.0
        self.alpha_val = 0.60
        self.proc_fct = None

        # Sub-widgets
        self.canvas_2d = SLMImageCanvas(self, title="Multi-Channel Overlap")
        self.canvas_profile = SLMProfileCanvas(self)

        self.canvas_2d.line_cut_changed.connect(self._on_line_cut_changed)
        self.canvas_2d.overlay_hovered.connect(self._on_overlay_hovered)

        # Header controls
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #252525; border-radius: 4px; padding: 4px;")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(6, 4, 6, 4)
        header_layout.setSpacing(10)

        # Alpha control
        lbl_alpha = QLabel("Overlay Alpha (Opacity):")
        lbl_alpha.setStyleSheet("font-weight: bold; color: #4da6ff;")
        self.slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(60)
        self.slider_alpha.setFixedWidth(140)
        self.slider_alpha.valueChanged.connect(self._on_slider_alpha_changed)

        self.spin_alpha = QSpinBox()
        self.spin_alpha.setRange(0, 100)
        self.spin_alpha.setValue(60)
        self.spin_alpha.setSuffix(" %")
        self.spin_alpha.setFixedWidth(65)
        self.spin_alpha.valueChanged.connect(self._on_spin_alpha_changed)

        # Overlay layer (Top)
        lbl_over = QLabel("Overlay Layer (Top):")
        self.combo_over_channel = QComboBox()
        self.combo_over_channel.currentIndexChanged.connect(self.refresh_display)

        self.combo_over_cmap = QComboBox()
        self.combo_over_cmap.addItems(["inferno", "turbo", "plasma", "viridis", "coolwarm", "hot", "magma"])
        self.combo_over_cmap.currentIndexChanged.connect(self.refresh_display)

        # Base layer (Bottom)
        lbl_base = QLabel("Base Layer (Bottom):")
        self.combo_base_channel = QComboBox()
        self.combo_base_channel.currentIndexChanged.connect(self.refresh_display)

        self.combo_base_cmap = QComboBox()
        self.combo_base_cmap.addItems(["gray", "bone", "cividis", "Greys_r", "viridis", "coolwarm"])
        self.combo_base_cmap.currentIndexChanged.connect(self.refresh_display)

        header_layout.addWidget(lbl_alpha)
        header_layout.addWidget(self.slider_alpha)
        header_layout.addWidget(self.spin_alpha)
        header_layout.addSpacing(10)
        header_layout.addWidget(lbl_over)
        header_layout.addWidget(self.combo_over_channel)
        header_layout.addWidget(self.combo_over_cmap)
        header_layout.addSpacing(10)
        header_layout.addWidget(lbl_base)
        header_layout.addWidget(self.combo_base_channel)
        header_layout.addWidget(self.combo_base_cmap)
        header_layout.addStretch(1)

        # Readout label
        self.lbl_readout = QLabel("Cursor: Hover over image to inspect blended layer values")
        self.lbl_readout.setStyleSheet("color: #00ffcc; font-size: 10px; padding: 2px 4px;")

        # Splitter between 2D canvas and 1D profile
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.canvas_2d)
        splitter.addWidget(self.canvas_profile)
        splitter.setSizes([550, 220])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(header_frame)
        layout.addWidget(self.lbl_readout)
        layout.addWidget(splitter, 1)

    def set_channels_list(self, channels: List[str]):
        """Populates the channel dropdowns with available channels."""
        curr_over = self.combo_over_channel.currentText()
        curr_base = self.combo_base_channel.currentText()

        self.combo_over_channel.blockSignals(True)
        self.combo_base_channel.blockSignals(True)

        self.combo_over_channel.clear()
        self.combo_base_channel.clear()

        self.combo_over_channel.addItems(channels)
        self.combo_base_channel.addItems(channels)

        # Set intelligent defaults
        if curr_over in channels:
            self.combo_over_channel.setCurrentText(curr_over)
        else:
            # Default to PR_R or X_PR or 2nd channel
            for ch in ["PR_R", "X_PR", "PR_X"]:
                if ch in channels:
                    self.combo_over_channel.setCurrentText(ch)
                    break
            else:
                if len(channels) > 1:
                    self.combo_over_channel.setCurrentIndex(1)

        if curr_base in channels:
            self.combo_base_channel.setCurrentText(curr_base)
        else:
            # Default to PD_R or X_PD or 1st channel
            for ch in ["PD_R", "X_PD", "PD_X"]:
                if ch in channels:
                    self.combo_base_channel.setCurrentText(ch)
                    break
            else:
                if len(channels) > 0:
                    self.combo_base_channel.setCurrentIndex(0)

        self.combo_over_channel.blockSignals(False)
        self.combo_base_channel.blockSignals(False)

    def set_scan_data(
        self,
        scan,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        unit: str = "V",
        phase_deg: float = 0.0,
        proc_fct=None,
        calib_um_per_v: Optional[float] = None
    ):
        self.current_scan = scan
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.unit_label = unit
        self.phase_deg = phase_deg
        self.proc_fct = proc_fct
        self.calib_um_per_v = calib_um_per_v

        # Ensure channel dropdowns match the scan
        if self.combo_over_channel.count() != len(scan.available_channels):
            self.set_channels_list(scan.available_channels)

        self.refresh_display()

    def set_alpha(self, alpha: float):
        self.alpha_val = float(np.clip(alpha, 0.0, 1.0))
        pct = int(np.round(self.alpha_val * 100))
        self.slider_alpha.blockSignals(True)
        self.spin_alpha.blockSignals(True)
        self.slider_alpha.setValue(pct)
        self.spin_alpha.setValue(pct)
        self.slider_alpha.blockSignals(False)
        self.spin_alpha.blockSignals(False)
        self.canvas_2d.set_overlay_alpha(self.alpha_val)

    def refresh_display(self):
        if self.current_scan is None or self.x_coords is None or self.y_coords is None:
            return

        over_name = self.combo_over_channel.currentText()
        base_name = self.combo_base_channel.currentText()

        if not over_name or not base_name:
            return

        if self.proc_fct:
            over_img = self.proc_fct(self.current_scan, over_name, self.phase_deg)
            base_img = self.proc_fct(self.current_scan, base_name, self.phase_deg)
        else:
            over_img = self.current_scan.get_channel(over_name, phase_deg=self.phase_deg)
            base_img = self.current_scan.get_channel(base_name, phase_deg=self.phase_deg)

        self.current_over_img = over_img
        self.current_base_img = base_img

        title = f"Scan: {self.current_scan.step_name} | {over_name} on {base_name} (α={self.alpha_val*100:.0f}%)"
        self.canvas_2d.set_overlay_data(
            base_image=base_img,
            overlay_image=over_img,
            alpha=self.alpha_val,
            base_cmap=self.combo_base_cmap.currentText(),
            overlay_cmap=self.combo_over_cmap.currentText(),
            x_coords=self.x_coords,
            y_coords=self.y_coords,
            unit=self.unit_label,
            title=title
        )

        if self.canvas_2d.line_points:
            self._update_dual_line_cut(*self.canvas_2d.line_points)

    def _on_slider_alpha_changed(self, val: int):
        self.alpha_val = val / 100.0
        self.spin_alpha.blockSignals(True)
        self.spin_alpha.setValue(val)
        self.spin_alpha.blockSignals(False)
        self.canvas_2d.set_overlay_alpha(self.alpha_val)
        self.alpha_changed.emit(self.alpha_val)

    def _on_spin_alpha_changed(self, val: int):
        self.alpha_val = val / 100.0
        self.slider_alpha.blockSignals(True)
        self.slider_alpha.setValue(val)
        self.slider_alpha.blockSignals(False)
        self.canvas_2d.set_overlay_alpha(self.alpha_val)
        self.alpha_changed.emit(self.alpha_val)

    def _on_line_cut_changed(self, c0: float, r0: float, c1: float, r1: float):
        self._update_dual_line_cut(c0, r0, c1, r1)

    def _update_dual_line_cut(self, c0: float, r0: float, c1: float, r1: float):
        if not hasattr(self, "current_over_img") or not hasattr(self, "current_base_img"):
            return

        distances_pix, vals_over, _ = processing.extract_line_profile(
            self.current_over_img, (c0, r0), (c1, r1)
        )
        _, vals_base, _ = processing.extract_line_profile(
            self.current_base_img, (c0, r0), (c1, r1)
        )

        cols = self.current_over_img.shape[1]
        x_span = abs(float(self.x_coords[-1] - self.x_coords[0]))
        dx_per_pix = x_span / max(1, cols - 1)
        distances_phys = distances_pix * dx_per_pix

        over_name = self.combo_over_channel.currentText()
        base_name = self.combo_base_channel.currentText()

        self.canvas_profile.set_dual_profile(
            distances=distances_phys,
            values_pr=vals_over,
            values_pd=vals_base,
            unit=self.unit_label,
            title=f"Cut: {over_name} & {base_name} from ({c0:.0f}, {r0:.0f}) to ({c1:.0f}, {r1:.0f})",
            calib_um_per_v=getattr(self, "calib_um_per_v", None)
        )

    def _on_overlay_hovered(self, col: int, row: int, x: float, y: float, val_over: float, val_base: float):
        over_name = self.combo_over_channel.currentText()
        base_name = self.combo_base_channel.currentText()
        self.lbl_readout.setText(
            f"Cursor: ({x:.3f} {self.unit_label}, {y:.3f} {self.unit_label}) [col={col}, row={row}]  |  "
            f"{over_name}: {val_over:.4e}  |  {base_name}: {val_base:.4e}  |  Alpha: {self.alpha_val*100:.0f}%"
        )
