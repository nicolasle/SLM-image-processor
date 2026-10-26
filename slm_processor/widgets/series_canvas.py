"""Step / Run Series Trend Canvas for SLM image analysis."""

import numpy as np
from typing import Optional, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QComboBox
)
from PyQt6.QtCore import pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class SLMSeriesCanvas(QWidget):
    """Plots signal trends across multi-run / multi-step datasets."""
    step_selected = pyqtSignal(int)  # emits index of the clicked scan step
    export_gif_requested = pyqtSignal()
    metric_changed = pyqtSignal(str)  # emits selected metric key ("mean", "max", etc.)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.step_names: List[str] = []
        self.step_vals: List[float] = []
        self.metric_vals: List[float] = []
        self.current_step_idx: int = 0
        self.metric_name: str = "ROI Mean"

        # Figure setup
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.fig.patch.set_facecolor("#2b2b2b")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1e1e1e")

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # Header controls
        header = QHBoxLayout()
        self.title_label = QLabel("Run / Step Series Trend")
        self.title_label.setStyleSheet("color: #e0e0e0; font-weight: bold; font-size: 11px;")
        header.addWidget(self.title_label)

        self.lbl_metric = QLabel("Metric:")
        self.lbl_metric.setStyleSheet("color: #aaaaaa; font-size: 10px; margin-left: 8px;")
        header.addWidget(self.lbl_metric)

        self.combo_metric = QComboBox()
        self.combo_metric.setStyleSheet("font-size: 10px; padding: 1px 4px;")
        self.combo_metric.addItem("Mean", "mean")
        self.combo_metric.addItem("Max (Peak)", "max")
        self.combo_metric.addItem("Min", "min")
        self.combo_metric.addItem("Std Dev", "std")
        self.combo_metric.addItem("Peak-to-Peak", "peak_to_peak")
        self.combo_metric.addItem("Sum", "sum")
        self.combo_metric.currentIndexChanged.connect(self._on_metric_changed)
        header.addWidget(self.combo_metric)

        header.addStretch(1)
        self.btn_export_gif = QPushButton("Export Movie GIF...")
        self.btn_export_gif.setStyleSheet("font-size: 10px; padding: 2px 8px; color: #00e5ff;")
        self.btn_export_gif.clicked.connect(self.export_gif_requested.emit)

        self.btn_export = QPushButton("Export Series CSV")
        self.btn_export.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        self.btn_export.clicked.connect(self._export_csv)

        header.addWidget(self.btn_export_gif)
        header.addWidget(self.btn_export)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.addLayout(header)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.canvas.mpl_connect("button_press_event", self._on_click)

    def set_series_data(
        self,
        step_names: List[str],
        step_vals: List[float],
        metric_vals: List[float],
        current_step_idx: int = 0,
        metric_name: str = "ROI Mean",
        x_label: str = "Step / Run Number"
    ):
        self.step_names = step_names
        self.step_vals = step_vals
        self.metric_vals = metric_vals
        self.current_step_idx = current_step_idx
        self.metric_name = metric_name

        self.ax.clear()
        self.ax.set_facecolor("#1e1e1e")
        self.ax.set_xlabel(x_label, color="#cccccc", fontsize=9)
        self.ax.set_ylabel(metric_name, color="#cccccc", fontsize=9)
        self.ax.tick_params(colors="#aaaaaa", labelsize=8)
        self.ax.grid(True, color="#444444", linestyle="--", alpha=0.5)
        for spine in self.ax.spines.values():
            spine.set_color("#555555")

        if len(step_vals) > 0 and len(metric_vals) == len(step_vals):
            x = np.array(step_vals)
            y = np.array(metric_vals)

            # Trend line with markers
            self.ax.plot(x, y, "-o", color="#3399ff", linewidth=1.8, markersize=6, label="Trend")

            # Highlight current step
            if 0 <= current_step_idx < len(x):
                self.ax.plot(
                    [x[current_step_idx]], [y[current_step_idx]],
                    "o", color="#ffcc00", markersize=10, markeredgecolor="#ffffff",
                    label=f"Current: Step {step_names[current_step_idx]}"
                )

            self.ax.legend(facecolor="#2b2b2b", edgecolor="#555555", labelcolor="#cccccc", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def get_selected_metric_key(self) -> str:
        """Returns the internal data key for the selected metric (e.g. 'mean', 'max')."""
        return self.combo_metric.currentData() or "mean"

    def get_selected_metric_label(self) -> str:
        """Returns the user-visible label for the selected metric (e.g. 'Max (Peak)')."""
        return self.combo_metric.currentText()

    def _on_metric_changed(self):
        self.metric_changed.emit(self.get_selected_metric_key())

    def _on_click(self, event):
        if event.inaxes != self.ax or not self.step_vals:
            return
        if self.toolbar.mode != "":
            return

        # Find closest step
        x_click = event.xdata
        closest_idx = int(np.argmin(np.abs(np.array(self.step_vals) - x_click)))
        self.step_selected.emit(closest_idx)

    def _export_csv(self):
        if not self.step_vals or not self.metric_vals:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Series CSV", "", "CSV Files (*.csv)")
        if not path:
            return

        try:
            with open(path, "w") as f:
                f.write(f"Step_Name,Step_Value,{self.metric_name}\n")
                for name, val, metric in zip(self.step_names, self.step_vals, self.metric_vals):
                    f.write(f"{name},{val},{metric:.6e}\n")
        except Exception as e:
            print(f"Error exporting series CSV: {e}")
