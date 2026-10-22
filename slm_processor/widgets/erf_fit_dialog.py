"""Dialog for detailed inspection and configuration of Error Function (Erf) knife-edge beam fitting."""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QApplication,
    QTabWidget, QWidget, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from slm_processor import processing
from slm_processor.processing import ErfFitResult


class SLMErfFitDialog(QDialog):
    """Detailed inspection dialog for Erf beam fits with dual-view (step response + Gaussian intensity)."""

    fitApplied = pyqtSignal(object)  # Emits ErfFitResult

    def __init__(
        self,
        distances: np.ndarray,
        values: np.ndarray,
        values_pd: Optional[np.ndarray] = None,
        unit: str = "V",
        calib_um_per_v: Optional[float] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Laser Beam 1/e² Size - Erf Knife-Edge Fit")
        self.resize(850, 680)

        self.distances = distances
        self.values = values
        self.values_pd = values_pd
        self.unit = unit
        self.calib_um_per_v = calib_um_per_v

        self.current_fit: Optional[ErfFitResult] = None

        self._build_ui()
        self._run_fit()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(6)

        # 1. Header Banner
        self.lbl_banner = QLabel("Laser Beam Profile & 1/e² Size Estimation")
        self.lbl_banner.setStyleSheet(
            "background-color: #1e3a5f; color: #00e5ff; font-size: 13px; "
            "font-weight: bold; padding: 8px; border-radius: 4px; border: 1px solid #00bcd4;"
        )
        self.lbl_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.lbl_banner)

        # 2. Controls Panel (Top)
        ctrl_group = QGroupBox("Fitting Configuration")
        ctrl_layout = QHBoxLayout(ctrl_group)
        ctrl_layout.setContentsMargins(6, 6, 6, 6)

        # Channel Selector
        ctrl_layout.addWidget(QLabel("Target Channel:"))
        self.combo_channel = QComboBox()
        self.combo_channel.addItem("Primary Signal", "signal")
        if self.values_pd is not None:
            self.combo_channel.addItem("PD Baseline", "pd")
        self.combo_channel.currentIndexChanged.connect(self._run_fit)
        ctrl_layout.addWidget(self.combo_channel)

        # Linear Background
        self.chk_linear = QCheckBox("Fit Linear Drift (m·x)")
        self.chk_linear.setChecked(True)
        self.chk_linear.toggled.connect(self._run_fit)
        ctrl_layout.addWidget(self.chk_linear)

        # Sub-range fitting
        self.chk_subrange = QCheckBox("Fit Sub-range:")
        self.chk_subrange.toggled.connect(self._on_subrange_toggled)
        ctrl_layout.addWidget(self.chk_subrange)

        d_min = float(np.min(self.distances)) if len(self.distances) > 0 else 0.0
        d_max = float(np.max(self.distances)) if len(self.distances) > 0 else 1.0

        ctrl_layout.addWidget(QLabel("Min:"))
        self.spin_min = QDoubleSpinBox()
        self.spin_min.setRange(-1e6, 1e6)
        self.spin_min.setDecimals(4)
        self.spin_min.setValue(d_min)
        self.spin_min.setEnabled(False)
        ctrl_layout.addWidget(self.spin_min)

        ctrl_layout.addWidget(QLabel("Max:"))
        self.spin_max = QDoubleSpinBox()
        self.spin_max.setRange(-1e6, 1e6)
        self.spin_max.setDecimals(4)
        self.spin_max.setValue(d_max)
        self.spin_max.setEnabled(False)
        ctrl_layout.addWidget(self.spin_max)

        self.btn_refit = QPushButton("↻ Refit")
        self.btn_refit.setStyleSheet("font-weight: bold; background-color: #0288d1; color: white; padding: 4px 12px;")
        self.btn_refit.clicked.connect(self._run_fit)
        ctrl_layout.addWidget(self.btn_refit)

        ctrl_layout.addStretch(1)
        main_layout.addWidget(ctrl_group)

        # 3. Splitter: Plots (left/center) and Parameters Table (right)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Tabs for plots
        plot_tabs = QTabWidget()

        # Tab 1: Erf Fit on Cross-section
        tab_erf = QWidget()
        layout_erf = QVBoxLayout(tab_erf)
        layout_erf.setContentsMargins(2, 2, 2, 2)
        self.fig_erf = Figure(figsize=(5, 4), dpi=100)
        self.fig_erf.patch.set_facecolor("#2b2b2b")
        self.ax_erf = self.fig_erf.add_subplot(111)
        self.ax_erf.set_facecolor("#1e1e1e")
        self.canvas_erf = FigureCanvasQTAgg(self.fig_erf)
        self.toolbar_erf = NavigationToolbar2QT(self.canvas_erf, self)
        layout_erf.addWidget(self.toolbar_erf)
        layout_erf.addWidget(self.canvas_erf)
        plot_tabs.addTab(tab_erf, "Cross-Section Erf Fit")

        # Tab 2: Derived Gaussian Laser Beam Profile
        tab_gauss = QWidget()
        layout_gauss = QVBoxLayout(tab_gauss)
        layout_gauss.setContentsMargins(2, 2, 2, 2)
        self.fig_gauss = Figure(figsize=(5, 4), dpi=100)
        self.fig_gauss.patch.set_facecolor("#2b2b2b")
        self.ax_gauss = self.fig_gauss.add_subplot(111)
        self.ax_gauss.set_facecolor("#1e1e1e")
        self.canvas_gauss = FigureCanvasQTAgg(self.fig_gauss)
        self.toolbar_gauss = NavigationToolbar2QT(self.canvas_gauss, self)
        layout_gauss.addWidget(self.toolbar_gauss)
        layout_gauss.addWidget(self.canvas_gauss)
        plot_tabs.addTab(tab_gauss, "Gaussian Beam Intensity (d/dx Erf)")

        splitter.addWidget(plot_tabs)

        # Table: Fit Metrics
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(2, 2, 2, 2)

        lbl_tbl = QLabel("Quantitative Laser Beam Parameters")
        lbl_tbl.setStyleSheet("font-weight: bold; color: #e0e0e0; font-size: 11px;")
        table_layout.addWidget(lbl_tbl)

        self.table_params = QTableWidget(10, 2)
        self.table_params.setHorizontalHeaderLabels(["Metric", "Fitted Value"])
        self.table_params.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_params.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_params.verticalHeader().setVisible(False)
        self.table_params.setStyleSheet(
            "QTableWidget { background-color: #242424; color: #ffffff; gridline-color: #3d3d3d; font-size: 11px; }"
            "QHeaderView::section { background-color: #333333; color: #00e5ff; font-weight: bold; }"
        )
        table_layout.addWidget(self.table_params)

        splitter.addWidget(table_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter, stretch=1)

        # 4. Bottom Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_apply = QPushButton("✓ Apply Fit to Canvas")
        self.btn_apply.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 5px 14px;")
        self.btn_apply.clicked.connect(self._apply_and_close)
        btn_layout.addWidget(self.btn_apply)

        self.btn_copy = QPushButton("📋 Copy Report")
        self.btn_copy.clicked.connect(self._copy_report)
        btn_layout.addWidget(self.btn_copy)

        self.btn_export_csv = QPushButton("Export Fit CSV...")
        self.btn_export_csv.clicked.connect(self._export_csv)
        btn_layout.addWidget(self.btn_export_csv)

        btn_layout.addStretch(1)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        btn_layout.addWidget(self.btn_close)

        main_layout.addLayout(btn_layout)

    def _on_subrange_toggled(self, checked: bool):
        self.spin_min.setEnabled(checked)
        self.spin_max.setEnabled(checked)
        self._run_fit()

    def _get_active_y(self) -> np.ndarray:
        channel = self.combo_channel.currentData()
        if channel == "pd" and self.values_pd is not None:
            return self.values_pd
        return self.values

    def _run_fit(self):
        y = self._get_active_y()
        with_linear = self.chk_linear.isChecked()

        x_range = None
        if self.chk_subrange.isChecked():
            x_range = (self.spin_min.value(), self.spin_max.value())

        result = processing.fit_erf_beam(
            x=self.distances,
            y=y,
            unit=self.unit,
            with_linear_drift=with_linear,
            x_range=x_range
        )
        self.current_fit = result

        if not result.success:
            self.lbl_banner.setText(f"Fit Failed: {result.message}")
            self.lbl_banner.setStyleSheet(
                "background-color: #5a1e1e; color: #ff8888; font-size: 12px; "
                "font-weight: bold; padding: 8px; border-radius: 4px; border: 1px solid #d32f2f;"
            )
            return

        # Update banner
        self.lbl_banner.setStyleSheet(
            "background-color: #1e3a5f; color: #00e5ff; font-size: 13px; "
            "font-weight: bold; padding: 8px; border-radius: 4px; border: 1px solid #00bcd4;"
        )
        banner_text = (
            f"1/e² Beam Radius w = {result.w:.4g} {self.unit}  |  "
            f"Diameter 2w = {result.diameter:.4g} {self.unit}  |  "
            f"FWHM = {result.fwhm:.4g} {self.unit}  |  R² = {result.r_squared:.4f}"
        )
        if self.unit == "V" and self.calib_um_per_v:
            w_um = result.w * self.calib_um_per_v
            banner_text += f"  (≈ {w_um:.3f} µm)"
        self.lbl_banner.setText(banner_text)

        # Update Table
        self._populate_table(result)

        # Update Plots
        self._plot_erf(result, y)
        self._plot_gaussian(result)

    def _populate_table(self, res: ErfFitResult):
        unit = self.unit
        params = [
            ("1/e² Beam Radius (w)", f"{res.w:.4g} ± {res.w_error:.3g} {unit}"),
            ("1/e² Beam Diameter (2w)", f"{res.diameter:.4g} ± {res.diameter_error:.3g} {unit}"),
            ("FWHM (1.1774 w)", f"{res.fwhm:.4g} ± {res.fwhm_error:.3g} {unit}"),
            ("Center Position (x₀)", f"{res.x0:.4g} ± {res.x0_error:.3g} {unit}"),
            ("Transition Height (2|A|)", f"{2.0 * abs(res.A):.4e} ± {2.0 * res.A_error:.3e}"),
            ("Baseline Level (y₀)", f"{res.y0:.4e} ± {res.y0_error:.3e}"),
            ("Linear Drift Slope (m)", f"{res.m:.4e} ± {res.m_error:.3e} / {unit}"),
            ("10%–90% Knife-Edge Width", f"{res.d10_90:.4g} {unit}"),
            ("Goodness of Fit (R²)", f"{res.r_squared:.5f}"),
            ("Residual RMSE", f"{res.rmse:.4e}")
        ]

        if unit == "V" and self.calib_um_per_v:
            w_um = res.w * self.calib_um_per_v
            d_um = res.diameter * self.calib_um_per_v
            fwhm_um = res.fwhm * self.calib_um_per_v
            params.insert(1, ("1/e² Radius in µm", f"{w_um:.4g} µm"))
            params.insert(3, ("1/e² Diameter in µm", f"{d_um:.4g} µm"))
            params.insert(5, ("FWHM in µm", f"{fwhm_um:.4g} µm"))

        self.table_params.setRowCount(len(params))
        for row, (name, val) in enumerate(params):
            item_name = QTableWidgetItem(name)
            item_name.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            item_val = QTableWidgetItem(val)
            item_val.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            if "1/e²" in name or "FWHM" in name:
                item_name.setForeground(Qt.GlobalColor.cyan)
                item_val.setForeground(Qt.GlobalColor.cyan)
            self.table_params.setItem(row, 0, item_name)
            self.table_params.setItem(row, 1, item_val)

    def _plot_erf(self, res: ErfFitResult, y: np.ndarray):
        self.ax_erf.clear()
        self.ax_erf.set_facecolor("#1e1e1e")
        self.ax_erf.grid(True, color="#444444", linestyle="--", alpha=0.5)

        # Plot data
        self.ax_erf.plot(
            self.distances, y,
            color="#ff3366" if self.combo_channel.currentData() == "signal" else "#00ffcc",
            marker="o", markersize=3, linewidth=1.0, alpha=0.8, label="Measured Cut"
        )

        # Plot Erf fit
        if res.x_fit is not None and res.y_fit is not None:
            self.ax_erf.plot(
                res.x_fit, res.y_fit,
                color="#00e5ff", linewidth=2.2, linestyle="-",
                label=f"Erf Fit (w = {res.w:.3g} {self.unit})"
            )

            # Center line
            self.ax_erf.axvline(
                res.x0, color="#00e5ff", linestyle=":", linewidth=1.5,
                label=f"Center x₀ = {res.x0:.3g}"
            )

            # Beam span shaded region (diameter 2w)
            self.ax_erf.axvspan(
                res.x0 - res.w, res.x0 + res.w,
                color="#00e5ff", alpha=0.15,
                label=f"1/e² Diameter = {res.diameter:.3g} {self.unit}"
            )

        self.ax_erf.set_xlabel(f"Distance ({self.unit})", color="#cccccc", fontsize=9)
        self.ax_erf.set_ylabel("Signal", color="#cccccc", fontsize=9)
        self.ax_erf.tick_params(colors="#aaaaaa", labelsize=8)
        self.ax_erf.legend(loc="best", facecolor="#2b2b2b", edgecolor="#555555", labelcolor="#cccccc", fontsize=8)
        for spine in self.ax_erf.spines.values():
            spine.set_color("#555555")

        self.fig_erf.tight_layout()
        self.canvas_erf.draw_idle()

    def _plot_gaussian(self, res: ErfFitResult):
        self.ax_gauss.clear()
        self.ax_gauss.set_facecolor("#1e1e1e")
        self.ax_gauss.grid(True, color="#444444", linestyle="--", alpha=0.5)

        if res.x_fit is not None and res.y_deriv is not None:
            max_i = float(np.max(res.y_deriv)) if len(res.y_deriv) > 0 else 1.0

            # Normalized intensity
            norm_i = res.y_deriv / max(1e-15, max_i)

            self.ax_gauss.plot(
                res.x_fit, norm_i,
                color="#ffd700", linewidth=2.2,
                label=f"Beam Profile I(x) ∝ d/dx Erf"
            )

            # 1/e^2 line (at 1/e^2 ≈ 0.1353)
            self.ax_gauss.axhline(
                1.0 / np.e**2, color="#ff9800", linestyle="--", linewidth=1.2,
                label="1/e² Level (13.53%)"
            )

            # Half-maximum line (at 0.5)
            self.ax_gauss.axhline(
                0.5, color="#4caf50", linestyle=":", linewidth=1.2,
                label="50% Level (FWHM)"
            )

            # Center line
            self.ax_gauss.axvline(
                res.x0, color="#ffd700", linestyle=":", linewidth=1.5
            )

            # Shaded span for 1/e^2 radius
            self.ax_gauss.axvspan(
                res.x0 - res.w, res.x0 + res.w,
                color="#ff9800", alpha=0.18,
                label=f"1/e² Radius w = {res.w:.3g} {self.unit}"
            )

        self.ax_gauss.set_xlabel(f"Position ({self.unit})", color="#cccccc", fontsize=9)
        self.ax_gauss.set_ylabel("Normalized Intensity I(x)", color="#cccccc", fontsize=9)
        self.ax_gauss.tick_params(colors="#aaaaaa", labelsize=8)
        self.ax_gauss.legend(loc="upper right", facecolor="#2b2b2b", edgecolor="#555555", labelcolor="#cccccc", fontsize=8)
        for spine in self.ax_gauss.spines.values():
            spine.set_color("#555555")

        self.fig_gauss.tight_layout()
        self.canvas_gauss.draw_idle()

    def _apply_and_close(self):
        if self.current_fit is not None and self.current_fit.success:
            self.fitApplied.emit(self.current_fit)
        self.accept()

    def _copy_report(self):
        if self.current_fit is None or not self.current_fit.success:
            return
        res = self.current_fit
        report = [
            "=== Laser Beam 1/e² Size Fit Report (Erf Knife-Edge) ===",
            f"1/e² Beam Radius (w):     {res.w:.4g} ± {res.w_error:.3g} {self.unit}",
            f"1/e² Beam Diameter (2w):   {res.diameter:.4g} ± {res.diameter_error:.3g} {self.unit}",
            f"Beam FWHM (1.1774 w):      {res.fwhm:.4g} ± {res.fwhm_error:.3g} {self.unit}",
            f"Beam Center Position (x₀): {res.x0:.4g} ± {res.x0_error:.3g} {self.unit}",
            f"Transition Height (2|A|):  {2.0 * abs(res.A):.4e} ± {2.0 * res.A_error:.3e}",
            f"Baseline Level (y₀):       {res.y0:.4e} ± {res.y0_error:.3e}",
            f"Linear Drift Slope (m):    {res.m:.4e} ± {res.m_error:.3e} / {self.unit}",
            f"10%-90% Knife-Edge Width:  {res.d10_90:.4g} {self.unit}",
            f"Coefficient of Det. (R²):  {res.r_squared:.5f}",
            f"Residual RMSE:             {res.rmse:.4e}",
        ]
        if self.unit == "V" and self.calib_um_per_v:
            report.append(f"Calibrated 1/e² Radius:    {res.w * self.calib_um_per_v:.4g} µm")
            report.append(f"Calibrated 1/e² Diameter:  {res.diameter * self.calib_um_per_v:.4g} µm")

        text = "\n".join(report)
        QApplication.clipboard().setText(text)
        self.lbl_banner.setText("✓ Report copied to clipboard!")

    def _export_csv(self):
        if self.current_fit is None or not self.current_fit.success:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Erf Fit Data", "", "CSV Files (*.csv)")
        if not path:
            return

        res = self.current_fit
        try:
            # Interpolate raw values at smooth x or save discrete pairs
            data = np.column_stack([res.x_fit, res.y_fit, res.y_deriv])
            header = f"Distance_{self.unit},Fitted_Erf,Gaussian_Intensity_Derivative"
            np.savetxt(path, data, delimiter=",", header=header, comments="")
            self.lbl_banner.setText(f"✓ Saved fit data to {path}")
        except Exception as e:
            self.lbl_banner.setText(f"Export error: {e}")
