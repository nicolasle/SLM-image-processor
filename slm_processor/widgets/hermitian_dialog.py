"""Dialog for configuring and executing Inverse Hermitian Transform (IFFT) to fall back into the time domain."""

import numpy as np
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QCheckBox, QComboBox, QPushButton, QGroupBox,
    QProgressBar, QMessageBox, QFrame, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from slm_processor.data_model import SLMDataset


class HermitianWorker(QThread):
    """Background worker to perform IFFT without freezing the GUI."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        dataset: SLMDataset,
        lia_freq: float,
        phasor_conv: str,
        n_time_points: int,
        time_shift_ms: float
    ):
        super().__init__()
        self.dataset = dataset
        self.lia_freq = lia_freq
        self.phasor_conv = phasor_conv
        self.n_time_points = n_time_points
        self.time_shift_ms = time_shift_ms

    def run(self):
        try:
            self.progress.emit(20)
            time_ds = self.dataset.convert_harmonics_to_timedomain(
                lia_freq_hz=self.lia_freq,
                phasor_convention=self.phasor_conv,
                n_time_points=self.n_time_points,
                time_shift_ms=self.time_shift_ms
            )
            self.progress.emit(100)
            self.finished.emit(time_ds)
        except Exception as e:
            self.failed.emit(str(e))


class SLMHermitianDialog(QDialog):
    """Interactive dialog for converting a harmonic series into a time-domain dataset."""

    transformed = pyqtSignal(object)  # Emits new time-domain SLMDataset

    def __init__(self, dataset: SLMDataset, parent: Optional[QDialog] = None):
        super().__init__(parent)
        self.dataset = dataset
        self.result_dataset: Optional[SLMDataset] = None

        self.setWindowTitle("Inverse Hermitian Transform (Time Domain Reconstitution)")
        self.setMinimumWidth(500)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #e0e0e0; }
            QGroupBox { border: 1px solid #3c3c3c; border-radius: 6px; margin-top: 10px; font-weight: bold; color: #4da6ff; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 2px 8px; }
            QLabel { color: #cccccc; }
            QDoubleSpinBox, QSpinBox, QComboBox { background-color: #2b2b2b; color: #ffffff; border: 1px solid #4a4a4a; border-radius: 4px; padding: 3px; }
        """)

        self._build_ui()
        self._update_timing_labels()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        # Header Info Banner
        banner = QFrame()
        banner.setStyleSheet("background-color: #252b33; border: 1px solid #007acc; border-radius: 6px; padding: 8px;")
        banner_layout = QVBoxLayout(banner)
        banner_layout.setContentsMargins(8, 6, 8, 6)

        title_lbl = QLabel("🔄 Inverse Hermitian Fourier Transform")
        title_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #4da6ff;")
        desc_lbl = QLabel(
            r"Converts discrete harmonic lock-in components ($X_n, Y_n$ at frequencies $n \cdot f_{LIA}$) "
            r"back into the real-time physical domain $S(t)$ over one modulation period $T = 1/f_{LIA}$."
        )
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("font-size: 10px; color: #aaaaaa;")
        banner_layout.addWidget(title_lbl)
        banner_layout.addWidget(desc_lbl)
        layout.addWidget(banner)

        # 1. Dataset Harmonics Summary
        grp_summary = QGroupBox("Harmonics in Dataset")
        lay_summary = QVBoxLayout(grp_summary)
        lay_summary.setContentsMargins(10, 10, 10, 10)

        harm_nums = [int(np.round(s.step_value)) for s in self.dataset.scans if s.step_value > 0]
        max_h = max(harm_nums) if harm_nums else 0
        img_shape = self.dataset[0].shape if len(self.dataset) > 0 else (0, 0)

        lbl_h_info = QLabel(
            f"• Detected Harmonics: <b>{len(harm_nums)} steps</b> (Harm 1 to Harm {max_h})<br>"
            f"• Image Dimensions: <b>{img_shape[1]} × {img_shape[0]} pixels</b>"
        )
        lay_summary.addWidget(lbl_h_info)
        layout.addWidget(grp_summary)

        # 2. Lock-in Amplifier (LIA) Parameters
        grp_lia = QGroupBox("Lock-in Reference Parameters")
        lay_lia = QVBoxLayout(grp_lia)
        lay_lia.setContentsMargins(10, 10, 10, 10)
        lay_lia.setSpacing(8)

        row_freq = QHBoxLayout()
        lbl_f = QLabel("LIA Reference Frequency (f_LIA):")
        lbl_f.setStyleSheet("font-weight: bold;")
        self.spin_freq = QDoubleSpinBox()
        self.spin_freq.setRange(0.001, 10000000.0)
        self.spin_freq.setDecimals(3)
        self.spin_freq.setSuffix(" Hz")
        self.spin_freq.setFixedWidth(130)

        # Default frequency: auto-detected from filename (e.g. fchop180Hz) or 180 Hz
        def_freq = self.dataset.detected_lia_freq if self.dataset.detected_lia_freq else 180.0
        self.spin_freq.setValue(def_freq)
        self.spin_freq.valueChanged.connect(self._update_timing_labels)

        row_freq.addWidget(lbl_f)
        row_freq.addStretch(1)
        row_freq.addWidget(self.spin_freq)
        lay_lia.addLayout(row_freq)

        self.lbl_period = QLabel("Modulation Period T: - ms")
        self.lbl_period.setStyleSheet("color: #00ffcc; font-size: 11px;")
        lay_lia.addWidget(self.lbl_period)

        layout.addWidget(grp_lia)

        # 3. Transform Options
        grp_opt = QGroupBox("Reconstitution Options")
        lay_opt = QVBoxLayout(grp_opt)
        lay_opt.setContentsMargins(10, 10, 10, 10)
        lay_opt.setSpacing(8)

        # Output points
        row_pts = QHBoxLayout()
        lbl_pts = QLabel("Time Points (Frames):")
        self.spin_pts = QSpinBox()
        self.spin_pts.setRange(10, 5000)
        def_pts = max(20, 2 * max_h)
        self.spin_pts.setValue(def_pts)
        self.spin_pts.setFixedWidth(130)
        self.spin_pts.valueChanged.connect(self._update_timing_labels)
        row_pts.addWidget(lbl_pts)
        row_pts.addStretch(1)
        row_pts.addWidget(self.spin_pts)
        lay_opt.addLayout(row_pts)

        self.lbl_res = QLabel("Time Step Δt: - µs")
        self.lbl_res.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        lay_opt.addWidget(self.lbl_res)

        # Phasor convention
        row_phasor = QHBoxLayout()
        lbl_conv = QLabel("Phasor Convention:")
        self.combo_conv = QComboBox()
        self.combo_conv.addItem("Standard (Y - iX) [process_30K formula]", "process_30K")
        self.combo_conv.addItem("Direct (X + iY)", "direct")
        self.combo_conv.addItem("Negative Quadrature (-Y + iX)", "negative_y")
        row_phasor.addWidget(lbl_conv)
        row_phasor.addStretch(1)
        row_phasor.addWidget(self.combo_conv)
        lay_opt.addLayout(row_phasor)

        # Time shift / Roll
        row_shift = QHBoxLayout()
        lbl_shift = QLabel("Time Offset / Trigger Shift:")
        self.spin_shift = QDoubleSpinBox()
        self.spin_shift.setRange(-1000.0, 1000.0)
        self.spin_shift.setDecimals(3)
        self.spin_shift.setSuffix(" ms")
        self.spin_shift.setValue(0.0)
        self.spin_shift.setFixedWidth(130)
        row_shift.addWidget(lbl_shift)
        row_shift.addStretch(1)
        row_shift.addWidget(self.spin_shift)
        lay_opt.addLayout(row_shift)

        layout.addWidget(grp_opt)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #4a4a4a; border-radius: 4px; text-align: center; color: white; }
            QProgressBar::chunk { background-color: #007acc; }
        """)
        layout.addWidget(self.progress_bar)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_transform = QPushButton("🚀 Transform to Time Domain")
        self.btn_transform.setStyleSheet("""
            QPushButton { background-color: #007acc; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px; font-size: 11px; }
            QPushButton:hover { background-color: #1a8ad6; }
        """)
        self.btn_transform.clicked.connect(self._run_transform)

        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_transform)
        layout.addLayout(btn_layout)

    def _update_timing_labels(self):
        f = self.spin_freq.value()
        if f > 0:
            T_s = 1.0 / f
            T_ms = T_s * 1000.0
            n_pts = self.spin_pts.value()
            dt_us = (T_s / n_pts) * 1e6 if n_pts > 0 else 0
            self.lbl_period.setText(f"Modulation Period T = <b>{T_ms:.3f} ms</b> (1 / {f:g} Hz)")
            self.lbl_res.setText(f"Temporal Step Δt = <b>{dt_us:.2f} µs</b> per frame")

    def _run_transform(self):
        f = self.spin_freq.value()
        if f <= 0:
            QMessageBox.warning(self, "Invalid Frequency", "LIA frequency must be positive.")
            return

        conv = self.combo_conv.currentData()
        pts = self.spin_pts.value()
        shift = self.spin_shift.value()

        self.btn_transform.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)

        # Run background thread
        self.worker = HermitianWorker(self.dataset, f, conv, pts, shift)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.start()

    def _on_worker_finished(self, time_ds: SLMDataset):
        self.progress_bar.setValue(100)
        self.result_dataset = time_ds
        self.transformed.emit(time_ds)
        self.accept()

    def _on_worker_failed(self, error_msg: str):
        self.btn_transform.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Transform Failed", f"Inverse Hermitian Transform error:\n{error_msg}")
