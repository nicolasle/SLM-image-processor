"""1D Line Profile Cut Canvas for SLM image analysis with Erf beam fitting."""

from typing import Optional, Tuple
import numpy as np
from scipy.special import erf
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QCheckBox, QFrame
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from slm_processor import processing
from slm_processor.processing import ErfFitResult


class SLMProfileCanvas(QWidget):
    """Displays 1D line profiles extracted from 2D images,
    supporting single channel or dual PR & PD curves,
    and interactive Error Function (Erf) knife-edge beam fitting.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.distances: Optional[np.ndarray] = None
        self.values: Optional[np.ndarray] = None
        self.values_pd: Optional[np.ndarray] = None
        self.unit_label: str = "V"
        self.calib_um_per_v: Optional[float] = None

        self.current_fit: Optional[ErfFitResult] = None
        self.fit_artist = None
        self.fit_center_artist = None
        self.fit_span_artist = None
        self.line_artist = None
        self.line_pr = None
        self.line_pd = None

        # Figure setup
        self.fig = Figure(figsize=(5, 2.7), dpi=100)
        self.fig.patch.set_facecolor("#2b2b2b")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1e1e1e")
        self.ax_twin = None

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # Header controls - Row 1
        header = QHBoxLayout()
        header.setSpacing(4)
        self.title_label = QLabel("1D Line Profile")
        self.title_label.setStyleSheet("color: #e0e0e0; font-weight: bold; font-size: 11px;")

        self.stats_label = QLabel("Peak: - | Min: - | Mean: -")
        self.stats_label.setStyleSheet("color: #00ffcc; font-size: 10px;")

        header.addWidget(self.title_label)
        header.addWidget(self.stats_label)
        header.addStretch(1)

        # Channel Selector for Erf fit (only visible if dual curves)
        self.combo_fit_channel = QComboBox()
        self.combo_fit_channel.setStyleSheet("font-size: 10px; padding: 1px 4px;")
        self.combo_fit_channel.addItem("Fit: Signal", "signal")
        self.combo_fit_channel.addItem("Fit: Baseline (PD)", "pd")
        self.combo_fit_channel.setVisible(False)
        self.combo_fit_channel.currentIndexChanged.connect(self._on_channel_selection_changed)
        header.addWidget(self.combo_fit_channel)

        # Fit Erf Button
        self.btn_fit_erf = QPushButton("⚡ Fit Erf (Beam 1/e²)")
        self.btn_fit_erf.setStyleSheet(
            "font-size: 10px; font-weight: bold; background-color: #0288d1; color: white; "
            "padding: 3px 8px; border-radius: 3px;"
        )
        self.btn_fit_erf.setToolTip("Fit error function to determine 1/e² laser beam radius and diameter")
        self.btn_fit_erf.clicked.connect(self._run_fit)
        header.addWidget(self.btn_fit_erf)

        # Clear Fit Button
        self.btn_clear_fit = QPushButton("✕")
        self.btn_clear_fit.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        self.btn_clear_fit.setToolTip("Clear Erf fit overlay")
        self.btn_clear_fit.setEnabled(False)
        self.btn_clear_fit.clicked.connect(self._clear_fit)
        header.addWidget(self.btn_clear_fit)

        # Fit Details Button
        self.btn_fit_details = QPushButton("ℹ Details...")
        self.btn_fit_details.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        self.btn_fit_details.setToolTip("Open comprehensive Erf fit dialog & Gaussian beam intensity profile")
        self.btn_fit_details.setEnabled(False)
        self.btn_fit_details.clicked.connect(self._open_fit_dialog)
        header.addWidget(self.btn_fit_details)

        # Auto-fit Checkbox
        self.chk_auto_fit = QCheckBox("Auto-fit")
        self.chk_auto_fit.setStyleSheet("font-size: 10px; color: #bbbbbb;")
        self.chk_auto_fit.setToolTip("Automatically fit Erf on newly drawn cut lines")
        header.addWidget(self.chk_auto_fit)

        # Export CSV Button
        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        self.btn_export.clicked.connect(self._export_csv)
        header.addWidget(self.btn_export)

        # Header Row 2 - Erf Results Banner (hidden until fitted)
        self.fit_banner = QFrame()
        self.fit_banner.setStyleSheet(
            "QFrame { background-color: #0d2738; border: 1px solid #0288d1; border-radius: 3px; }"
        )
        banner_layout = QHBoxLayout(self.fit_banner)
        banner_layout.setContentsMargins(6, 3, 6, 3)

        self.lbl_fit_badge = QLabel("⚡ Laser Beam 1/e²: -")
        self.lbl_fit_badge.setStyleSheet("color: #00e5ff; font-size: 11px; font-weight: bold;")
        banner_layout.addWidget(self.lbl_fit_badge, 1)

        self.btn_badge_report = QPushButton("View Full Report")
        self.btn_badge_report.setStyleSheet(
            "font-size: 9px; padding: 1px 6px; background-color: #01579b; color: white;"
        )
        self.btn_badge_report.clicked.connect(self._open_fit_dialog)
        banner_layout.addWidget(self.btn_badge_report)

        self.fit_banner.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.addLayout(header)
        layout.addWidget(self.fit_banner)
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
        title: str = "1D Line Profile",
        calib_um_per_v: Optional[float] = None
    ):
        self.distances = distances
        self.values = values
        self.values_pd = None
        self.unit_label = unit
        self.calib_um_per_v = calib_um_per_v
        self.title_label.setText(title)
        self.combo_fit_channel.setVisible(False)

        self._clear_fit_artists()

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
        self.line_pr = self.line_artist
        self.line_pd = None

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

        # Handle auto-fit
        if self.chk_auto_fit.isChecked():
            self._run_fit()
        elif self.current_fit is not None:
            # Re-run fit with new units/data if previously fitted
            self._run_fit()

    def set_dual_profile(
        self,
        distances: np.ndarray,
        values_pr: np.ndarray,
        values_pd: np.ndarray,
        unit: str = "V",
        title: str = "1D Profile: PR on PD",
        calib_um_per_v: Optional[float] = None
    ):
        self.distances = distances
        self.values = values_pr
        self.values_pd = values_pd
        self.unit_label = unit
        self.calib_um_per_v = calib_um_per_v
        self.title_label.setText(title)
        self.combo_fit_channel.setVisible(True)

        self._clear_fit_artists()

        if self.ax_twin is not None:
            self.ax_twin.remove()
            self.ax_twin = None

        self.ax.clear()
        self.ax.set_facecolor("#1e1e1e")
        self._setup_axes()
        self.ax.set_ylabel("PR Signal (V)", color="#ff3366", fontsize=9)
        self.ax.tick_params(axis="y", colors="#ff3366", labelsize=8)

        self.line_pr, = self.ax.plot(
            distances, values_pr, color="#ff3366", linewidth=1.8, marker=".", markersize=3, label="PR Signal"
        )
        self.line_artist = self.line_pr

        self.ax_twin = self.ax.twinx()
        self.ax_twin.set_ylabel("PD Baseline (V)", color="#00ffcc", fontsize=9)
        self.ax_twin.tick_params(axis="y", colors="#00ffcc", labelsize=8)
        for spine in self.ax_twin.spines.values():
            spine.set_color("#555555")

        self.line_pd, = self.ax_twin.plot(
            distances, values_pd, color="#00ffcc", linewidth=1.5, linestyle="--", label="PD Baseline"
        )

        lines = [self.line_pr, self.line_pd]
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

        # Handle auto-fit
        if self.chk_auto_fit.isChecked():
            self._run_fit()
        elif self.current_fit is not None:
            self._run_fit()

    def _on_channel_selection_changed(self):
        if self.current_fit is not None or self.chk_auto_fit.isChecked():
            self._run_fit()

    def _run_fit(self):
        """Executes the Erf knife-edge beam fit and updates the canvas."""
        if self.distances is None or len(self.distances) < 4:
            return

        if self.values_pd is not None and self.combo_fit_channel.currentIndex() == 1:
            y_target = self.values_pd
        else:
            y_target = self.values

        if y_target is None or len(y_target) < 4:
            return

        res = processing.fit_erf_beam(
            x=self.distances,
            y=y_target,
            unit=self.unit_label,
            with_linear_drift=True
        )
        self.apply_fit_result(res)

    def apply_fit_result(self, res: ErfFitResult):
        """Applies fit results to the canvas and updates the summary badge."""
        self.current_fit = res

        if not res.success:
            self.lbl_fit_badge.setText(f"Fit Notice: {res.message}")
            self.lbl_fit_badge.setStyleSheet("color: #ffaa66; font-size: 10px; font-weight: bold;")
            self.fit_banner.show()
            return

        # Format informative badge string
        badge = (
            f"⚡ Laser Beam 1/e²:  Radius w = {res.w:.4g} {self.unit_label}  |  "
            f"Diameter 2w = {res.diameter:.4g} {self.unit_label}  |  "
            f"FWHM = {res.fwhm:.4g} {self.unit_label}  |  R² = {res.r_squared:.4f}"
        )
        if self.unit_label == "V" and self.calib_um_per_v:
            w_um = res.w * self.calib_um_per_v
            d_um = res.diameter * self.calib_um_per_v
            badge += f"  (≈ {w_um:.3g} µm radius, {d_um:.3g} µm diameter)"

        self.lbl_fit_badge.setText(badge)
        self.lbl_fit_badge.setStyleSheet("color: #00e5ff; font-size: 11px; font-weight: bold;")
        self.fit_banner.show()
        self.btn_clear_fit.setEnabled(True)
        self.btn_fit_details.setEnabled(True)

        self._draw_fit_artists(res)

    def _draw_fit_artists(self, res: ErfFitResult):
        self._clear_fit_artists()

        if res.x_fit is None or res.y_fit is None:
            return

        target_ax = self.ax_twin if (
            self.values_pd is not None
            and self.combo_fit_channel.currentIndex() == 1
            and self.ax_twin is not None
        ) else self.ax

        self.fit_artist, = target_ax.plot(
            res.x_fit, res.y_fit,
            color="#00e5ff", linewidth=2.0, linestyle="--",
            label=f"Erf Fit (w = {res.w:.3g} {self.unit_label})"
        )
        self.fit_center_artist = target_ax.axvline(
            res.x0, color="#00e5ff", linestyle=":", linewidth=1.3,
            label=f"Center x₀ = {res.x0:.3g}"
        )
        self.fit_span_artist = target_ax.axvspan(
            res.x0 - res.w, res.x0 + res.w,
            color="#00e5ff", alpha=0.15
        )

        # Refresh legend
        if self.ax_twin is not None and self.line_pd is not None:
            lines = [self.line_pr, self.line_pd, self.fit_artist]
            labels = [l.get_label() for l in lines]
            self.ax.legend(
                lines, labels, loc="upper right",
                facecolor="#2b2b2b", edgecolor="#555555",
                labelcolor="#cccccc", fontsize=8
            )
        else:
            self.ax.legend(
                loc="upper right",
                facecolor="#2b2b2b", edgecolor="#555555",
                labelcolor="#cccccc", fontsize=8
            )

        self.canvas.draw_idle()

    def _clear_fit_artists(self):
        if self.fit_artist is not None:
            try:
                self.fit_artist.remove()
            except Exception:
                pass
            self.fit_artist = None

        if self.fit_center_artist is not None:
            try:
                self.fit_center_artist.remove()
            except Exception:
                pass
            self.fit_center_artist = None

        if self.fit_span_artist is not None:
            try:
                self.fit_span_artist.remove()
            except Exception:
                pass
            self.fit_span_artist = None

    def _clear_fit(self):
        self.current_fit = None
        self._clear_fit_artists()
        self.fit_banner.hide()
        self.btn_clear_fit.setEnabled(False)
        self.btn_fit_details.setEnabled(False)

        # Restore clean legend
        if self.ax_twin is not None and self.line_pr is not None and self.line_pd is not None:
            lines = [self.line_pr, self.line_pd]
            labels = [l.get_label() for l in lines]
            self.ax.legend(lines, labels, loc="upper right", facecolor="#2b2b2b", edgecolor="#555555", labelcolor="#cccccc", fontsize=8)
        elif self.line_artist is not None:
            self.ax.legend([self.line_artist], ["Signal"], loc="upper right", facecolor="#2b2b2b", edgecolor="#555555", labelcolor="#cccccc", fontsize=8)

        self.canvas.draw_idle()

    def _open_fit_dialog(self):
        if self.distances is None or self.values is None or len(self.distances) < 4:
            return

        from slm_processor.widgets.erf_fit_dialog import SLMErfFitDialog
        dlg = SLMErfFitDialog(
            distances=self.distances,
            values=self.values,
            values_pd=self.values_pd,
            unit=self.unit_label,
            calib_um_per_v=self.calib_um_per_v,
            parent=self
        )
        dlg.fitApplied.connect(self.apply_fit_result)
        dlg.exec()

    def _export_csv(self):
        if self.distances is None or self.values is None or len(self.distances) == 0:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Line Profile CSV", "", "CSV Files (*.csv)")
        if not path:
            return

        try:
            cols = [self.distances, self.values]
            header_items = [f"Distance_{self.unit_label}", "Signal_Value"]

            if self.values_pd is not None:
                cols.append(self.values_pd)
                header_items.append("PD_Baseline")

            if self.current_fit is not None and self.current_fit.success and self.current_fit.w > 0:
                res = self.current_fit
                y_fit_at_pts = (
                    res.y0
                    + res.m * (self.distances - res.x0)
                    + res.A * erf(np.sqrt(2.0) * (self.distances - res.x0) / max(1e-12, res.w))
                )
                cols.append(y_fit_at_pts)
                header_items.append(f"Erf_Fit_w={res.w:.4g}{self.unit_label}")

            data = np.column_stack(cols)
            header = ",".join(header_items)
            np.savetxt(path, data, delimiter=",", header=header, comments="")
        except Exception as e:
            print(f"Error exporting profile CSV: {e}")
