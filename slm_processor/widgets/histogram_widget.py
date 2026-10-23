"""Interactive Histogram and Real-Time Dual-Threshold Contrast Widget for SLM image analysis."""

import numpy as np
from typing import Optional, Tuple
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QCheckBox, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class SLMHistogramWidget(QWidget):
    """Compact interactive histogram widget with draggable low/high threshold lines,

    percentile presets, and numerical spinboxes for real-time contrast adjustment.
    """
    clim_changed = pyqtSignal(float, float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.current_data: Optional[np.ndarray] = None
        self.vmin: float = 0.0
        self.vmax: float = 1.0
        self.data_min: float = 0.0
        self.data_max: float = 1.0

        # Dragging state
        self._dragging_target: Optional[str] = None  # "min", "max", or None

        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(3)

        # 1. Figure canvas for histogram
        self.fig = Figure(figsize=(3.5, 1.2), dpi=100, layout="tight")
        self.fig.patch.set_facecolor("#222222")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#181818")

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setFixedHeight(95)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Connect mouse events
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        main_layout.addWidget(self.canvas)

        # 2. Threshold spinboxes row
        row_inputs = QHBoxLayout()
        row_inputs.setContentsMargins(0, 0, 0, 0)
        row_inputs.setSpacing(4)

        lbl_min = QLabel("Min:")
        lbl_min.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 10px;")
        row_inputs.addWidget(lbl_min)

        self.spin_vmin = QDoubleSpinBox()
        self.spin_vmin.setRange(-1e9, 1e9)
        self.spin_vmin.setDecimals(4)
        self.spin_vmin.setStyleSheet("font-size: 10px; padding: 1px 3px;")
        self.spin_vmin.valueChanged.connect(self._on_spin_changed)
        row_inputs.addWidget(self.spin_vmin, 1)

        lbl_max = QLabel("Max:")
        lbl_max.setStyleSheet("color: #ff3366; font-weight: bold; font-size: 10px;")
        row_inputs.addWidget(lbl_max)

        self.spin_vmax = QDoubleSpinBox()
        self.spin_vmax.setRange(-1e9, 1e9)
        self.spin_vmax.setDecimals(4)
        self.spin_vmax.setStyleSheet("font-size: 10px; padding: 1px 3px;")
        self.spin_vmax.valueChanged.connect(self._on_spin_changed)
        row_inputs.addWidget(self.spin_vmax, 1)

        self.chk_log = QCheckBox("Log Y")
        self.chk_log.setStyleSheet("color: #aaaaaa; font-size: 9px;")
        self.chk_log.toggled.connect(self._replot_histogram)
        row_inputs.addWidget(self.chk_log)

        main_layout.addLayout(row_inputs)

        # 3. Quick preset buttons
        row_presets = QHBoxLayout()
        row_presets.setContentsMargins(0, 0, 0, 0)
        row_presets.setSpacing(3)

        btn_style = """
            QPushButton {
                background-color: #2b2b2b;
                color: #cccccc;
                font-size: 9px;
                padding: 2px 4px;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: #383838; color: #ffffff; }
        """

        presets = [
            ("Min-Max", 0.0, 100.0),
            ("1%-99%", 1.0, 99.0),
            ("2%-98%", 2.0, 98.0),
            ("5%-95%", 5.0, 95.0),
        ]

        for text, p_low, p_high in presets:
            btn = QPushButton(text)
            btn.setStyleSheet(btn_style)
            btn.clicked.connect(lambda _, pl=p_low, ph=p_high: self.set_percentile_limits(pl, ph))
            row_presets.addWidget(btn)

        main_layout.addLayout(row_presets)

        # Style initial empty plot
        self._format_axes()
        self.canvas.draw_idle()

    def _format_axes(self):
        self.ax.set_facecolor("#181818")
        self.ax.tick_params(colors="#777777", labelsize=7, pad=2)
        for spine in self.ax.spines.values():
            spine.set_color("#3c3c3c")
        self.ax.yaxis.set_visible(False)  # keep clean, counts unneeded

    def set_data(self, data: Optional[np.ndarray], vmin: Optional[float] = None, vmax: Optional[float] = None):
        """Updates the histogram with new image data and sets threshold limits."""
        if data is None or data.size == 0:
            self.current_data = None
            self.ax.clear()
            self._format_axes()
            self.canvas.draw_idle()
            return

        valid = data[np.isfinite(data)]
        if valid.size == 0:
            return

        self.current_data = valid
        self.data_min = float(np.min(valid))
        self.data_max = float(np.max(valid))

        if vmin is not None and vmax is not None and vmin < vmax:
            self.vmin = float(vmin)
            self.vmax = float(vmax)
        else:
            self.vmin = self.data_min
            self.vmax = self.data_max

        # Update spinboxes without triggering signals
        self.spin_vmin.blockSignals(True)
        self.spin_vmax.blockSignals(True)
        self.spin_vmin.setValue(self.vmin)
        self.spin_vmax.setValue(self.vmax)
        self.spin_vmin.blockSignals(False)
        self.spin_vmax.blockSignals(False)

        self._replot_histogram()

    def set_percentile_limits(self, p_low: float, p_high: float):
        """Sets vmin and vmax according to percentiles of current data."""
        if self.current_data is None or self.current_data.size == 0:
            return

        if p_low == 0.0 and p_high == 100.0:
            v0 = float(np.min(self.current_data))
            v1 = float(np.max(self.current_data))
        else:
            v0 = float(np.percentile(self.current_data, p_low))
            v1 = float(np.percentile(self.current_data, p_high))
        if v0 >= v1:
            v1 = v0 + 1e-6

        self.set_clim(v0, v1, emit=True)

    def set_clim(self, vmin: float, vmax: float, emit: bool = False):
        """Sets display color limits and updates visuals."""
        if vmin >= vmax:
            vmax = vmin + 1e-6

        self.vmin = float(vmin)
        self.vmax = float(vmax)

        self.spin_vmin.blockSignals(True)
        self.spin_vmax.blockSignals(True)
        self.spin_vmin.setValue(self.vmin)
        self.spin_vmax.setValue(self.vmax)
        self.spin_vmin.blockSignals(False)
        self.spin_vmax.blockSignals(False)

        self._update_threshold_lines()
        self.canvas.draw_idle()

        if emit:
            self.clim_changed.emit(self.vmin, self.vmax)

    def _replot_histogram(self):
        if self.current_data is None or self.current_data.size == 0:
            return

        self.ax.clear()
        self._format_axes()

        # Compute histogram with 80 bins
        counts, bin_edges = np.histogram(self.current_data, bins=80)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        y_vals = counts.astype(np.float32)
        if self.chk_log.isChecked():
            y_vals = np.log10(np.maximum(1, y_vals))

        # Filled area plot for histogram
        self.ax.fill_between(bin_centers, 0, y_vals, color="#007acc", alpha=0.55)
        self.ax.plot(bin_centers, y_vals, color="#4da6ff", linewidth=1.0)

        # Visible range slightly padded
        span = self.data_max - self.data_min
        pad = span * 0.05 if span > 1e-12 else 1.0
        self.ax.set_xlim(self.data_min - pad, self.data_max + pad)
        y_max = np.max(y_vals) if len(y_vals) > 0 else 1.0
        self.ax.set_ylim(0, y_max * 1.1)

        # Draw draggable threshold lines
        self.line_min = self.ax.axvline(self.vmin, color="#00e5ff", linewidth=1.8, linestyle="--", alpha=0.95)
        self.line_max = self.ax.axvline(self.vmax, color="#ff3366", linewidth=1.8, linestyle="--", alpha=0.95)

        self.canvas.draw_idle()

    def _update_threshold_lines(self):
        if hasattr(self, "line_min") and self.line_min is not None:
            self.line_min.set_xdata([self.vmin, self.vmin])
        if hasattr(self, "line_max") and self.line_max is not None:
            self.line_max.set_xdata([self.vmax, self.vmax])

    def _on_spin_changed(self):
        vmin = self.spin_vmin.value()
        vmax = self.spin_vmax.value()
        if vmin < vmax:
            self.vmin = vmin
            self.vmax = vmax
            self._update_threshold_lines()
            self.canvas.draw_idle()
            self.clim_changed.emit(self.vmin, self.vmax)

    def _on_press(self, event):
        if event.inaxes != self.ax or self.current_data is None:
            return

        x_click = event.xdata
        if x_click is None:
            return

        span = max(1e-12, self.data_max - self.data_min)
        dist_min = abs(x_click - self.vmin) / span
        dist_max = abs(x_click - self.vmax) / span

        # Grab threshold if within 8% of total width
        if dist_min < 0.08 and dist_min <= dist_max:
            self._dragging_target = "min"
        elif dist_max < 0.08:
            self._dragging_target = "max"
        elif x_click < (self.vmin + self.vmax) / 2:
            self._dragging_target = "min"
            self._drag_to(x_click)
        else:
            self._dragging_target = "max"
            self._drag_to(x_click)

    def _on_motion(self, event):
        if event.inaxes != self.ax or self.current_data is None:
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            return

        x = event.xdata
        if x is None:
            return

        span = max(1e-12, self.data_max - self.data_min)
        dist_min = abs(x - self.vmin) / span
        dist_max = abs(x - self.vmax) / span

        if dist_min < 0.06 or dist_max < 0.06 or self._dragging_target is not None:
            self.canvas.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

        if self._dragging_target is not None:
            self._drag_to(x)

    def _drag_to(self, x: float):
        if self._dragging_target == "min":
            new_min = min(x, self.vmax - 1e-6)
            self.vmin = float(new_min)
            self.spin_vmin.blockSignals(True)
            self.spin_vmin.setValue(self.vmin)
            self.spin_vmin.blockSignals(False)
        elif self._dragging_target == "max":
            new_max = max(x, self.vmin + 1e-6)
            self.vmax = float(new_max)
            self.spin_vmax.blockSignals(True)
            self.spin_vmax.setValue(self.vmax)
            self.spin_vmax.blockSignals(False)

        self._update_threshold_lines()
        self.canvas.draw_idle()
        self.clim_changed.emit(self.vmin, self.vmax)

    def _on_release(self, _):
        if self._dragging_target is not None:
            self._dragging_target = None
            self.clim_changed.emit(self.vmin, self.vmax)
