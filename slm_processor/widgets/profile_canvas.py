"""1D Line Profile Cut Canvas for SLM image analysis."""

import numpy as np
from typing import Optional, Tuple
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class SLMProfileCanvas(QWidget):
    """Displays 1D line profiles extracted from 2D images,

    supporting single channel or dual PR & PD curves.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.distances: Optional[np.ndarray] = None
        self.values: Optional[np.ndarray] = None
        self.values_pd: Optional[np.ndarray] = None
        self.unit_label: str = "V"

        # Figure setup
        self.fig = Figure(figsize=(5, 2.5), dpi=100)
        self.fig.patch.set_facecolor("#2b2b2b")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1e1e1e")
        self.ax_twin = None

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # Header controls
        header = QHBoxLayout()
        self.title_label = QLabel("1D Line Profile")
        self.title_label.setStyleSheet("color: #e0e0e0; font-weight: bold; font-size: 11px;")
        
        self.stats_label = QLabel("Peak: - | Min: - | Mean: -")
        self.stats_label.setStyleSheet("color: #00ffcc; font-size: 10px;")

        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        self.btn_export.clicked.connect(self._export_csv)

        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.stats_label)
        header.addWidget(self.btn_export)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.addLayout(header)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.line_artist, = self.ax.plot([], [], color="#ff3366", linewidth=1.5, marker=".", markersize=3)
        self._setup_axes()

    def _setup_axes(self):
        self.ax.set_xlabel(f"Distance along cut ({self.unit_label})", color="#cccccc", fontsize=9)
        self.ax.set_ylabel("Signal", color="#cccccc", fontsize=9)
        self.ax.tick_params(colors="#aaaaaa", labelsize=8)
        self.ax.grid(True, color="#444444", linestyle="--", alpha=0.5)
        for spine in self.ax.spines.values():
            spine.set_color("#555555")

    def set_profile(
        self,
        distances: np.ndarray,
        values: np.ndarray,
        unit: str = "V",
        title: str = "1D Line Profile"
    ):
        self.distances = distances
        self.values = values
        self.values_pd = None
        self.unit_label = unit
        self.title_label.setText(title)

        if self.ax_twin is not None:
            self.ax_twin.remove()
            self.ax_twin = None

        if len(distances) == 0 or len(values) == 0:
            self.ax.clear()
            self.ax.set_facecolor("#1e1e1e")
            self._setup_axes()
            self.canvas.draw_idle()
            return

        self.ax.clear()
        self.ax.set_facecolor("#1e1e1e")
        self._setup_axes()

        self.line_artist, = self.ax.plot(
            distances, values, color="#ff3366", linewidth=1.5, marker=".", markersize=3, label="Signal"
        )

        v_max = float(np.max(values))
        v_min = float(np.min(values))
        v_mean = float(np.mean(values))
        max_idx = int(np.argmax(values))
        dist_at_max = distances[max_idx]

        self.stats_label.setText(
            f"Peak: {v_max:.3e} (at {dist_at_max:.3f} {unit}) | Min: {v_min:.3e} | Mean: {v_mean:.3e}"
        )

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def set_dual_profile(
        self,
        distances: np.ndarray,
        values_pr: np.ndarray,
        values_pd: np.ndarray,
        unit: str = "V",
        title: str = "1D Profile: PR on PD"
    ):
        self.distances = distances
        self.values = values_pr
        self.values_pd = values_pd
        self.unit_label = unit
        self.title_label.setText(title)

        if self.ax_twin is not None:
            self.ax_twin.remove()
            self.ax_twin = None

        self.ax.clear()
        self.ax.set_facecolor("#1e1e1e")
        self._setup_axes()
        self.ax.set_ylabel("PR Signal (V)", color="#ff3366", fontsize=9)
        self.ax.tick_params(axis="y", colors="#ff3366", labelsize=8)

        line_pr, = self.ax.plot(
            distances, values_pr, color="#ff3366", linewidth=1.8, marker=".", markersize=3, label="PR Signal"
        )

        self.ax_twin = self.ax.twinx()
        self.ax_twin.set_ylabel("PD Baseline (V)", color="#00ffcc", fontsize=9)
        self.ax_twin.tick_params(axis="y", colors="#00ffcc", labelsize=8)
        for spine in self.ax_twin.spines.values():
            spine.set_color("#555555")

        line_pd, = self.ax_twin.plot(
            distances, values_pd, color="#00ffcc", linewidth=1.5, linestyle="--", label="PD Baseline"
        )

        # Joint legend
        lines = [line_pr, line_pd]
        labels = [l.get_label() for l in lines]
        self.ax.legend(lines, labels, loc="upper right", facecolor="#2b2b2b", edgecolor="#555555", labelcolor="#cccccc", fontsize=8)

        pr_max = float(np.max(values_pr))
        pd_mean = float(np.mean(values_pd))
        self.stats_label.setText(f"PR Peak: {pr_max:.3e} | PD Mean: {pd_mean:.3e}")

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax_twin.relim()
        self.ax_twin.autoscale_view()
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _export_csv(self):
        if self.distances is None or self.values is None or len(self.distances) == 0:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Line Profile CSV", "", "CSV Files (*.csv)")
        if not path:
            return

        try:
            if self.values_pd is not None:
                data = np.column_stack([self.distances, self.values, self.values_pd])
                header = f"Distance_{self.unit_label},Photoresponse,Photodiode"
            else:
                data = np.column_stack([self.distances, self.values])
                header = f"Distance_{self.unit_label},Signal_Value"
            np.savetxt(path, data, delimiter=",", header=header, comments="")
        except Exception as e:
            print(f"Error exporting profile CSV: {e}")
