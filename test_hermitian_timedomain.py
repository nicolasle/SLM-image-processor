"""Automated test suite for Flat Harmonic Series loading and Inverse Hermitian Transform (Time Domain Reconstitution)."""

import os
import sys
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

QMessageBox.information = lambda *args, **kwargs: QMessageBox.StandardButton.Ok
QMessageBox.warning = lambda *args, **kwargs: QMessageBox.StandardButton.Ok

from slm_processor.data_model import SLMDataset
from slm_processor.main_window import SLMMainWindow
from slm_processor.widgets.hermitian_dialog import SLMHermitianDialog


def test_harmonics_and_timedomain():
    print("=======================================================")
    print("  TESTING HARMONIC AUTO-DETECTION & IFFT TIME DOMAIN   ")
    print("=======================================================")

    app = QApplication.instance() or QApplication(sys.argv)
    path_30k = r"C:\Users\nicol\OneDrive - Universite de Liege\PhD\91 - Simon TFE\data\280426_harmonics\data_30K"

    if not os.path.exists(path_30k):
        print(f"Skipping: {path_30k} not found.")
        return

    # 1. Test Dataset Ingestion
    print("\n--- TEST 1: Smart Ingestion of Flat Harmonic Series (data_30K) ---")
    ds = SLMDataset(path_30k)
    assert ds.is_harmonic_series is True, "Dataset should be recognized as harmonic series!"
    assert ds.detected_lia_freq == 180.0, f"Expected LIA freq 180.0, got {ds.detected_lia_freq}"
    assert len(ds) == 300, f"Expected 300 harmonic steps, got {len(ds)}"
    print(f"[OK] data_30K recognized: {len(ds)} harmonics, LIA freq = {ds.detected_lia_freq} Hz")

    # Check channels on step 1 (odd harmonic: has PR and PD) and step 2 (even harmonic: PR only)
    scan1 = ds[0]
    scan2 = ds[1]
    assert scan1.has_channel("PR_X") and scan1.has_channel("PR_Y")
    assert scan1.has_channel("PD_X") and scan1.has_channel("PD_Y")
    assert scan1.has_channel("PR_R") and scan1.has_channel("PD_R")
    assert scan2.has_channel("PR_X") and scan2.has_channel("PR_Y")
    print("[OK] Channel normalization & auto lock-in pairing passed on flat series")

    # 2. Test Inverse Hermitian Transform (Time Domain)
    print("\n--- TEST 2: Inverse Hermitian Transform Execution ---")
    time_ds = ds.convert_harmonics_to_timedomain(
        lia_freq_hz=180.0,
        phasor_convention="process_30K",
        n_time_points=40
    )

    assert len(time_ds) == 40, f"Expected 40 time frames, got {len(time_ds)}"
    assert time_ds.series_variable_name == "Time (ms)"
    assert np.isclose(time_ds[0].step_value, 0.0)
    T_expected = (1.0 / 180.0) * 1000.0 # ~5.556 ms
    assert np.isclose(time_ds[-1].step_value, T_expected * (39 / 40.0), atol=1e-2)
    assert "PR" in time_ds[0].available_channels
    assert "PD" in time_ds[0].available_channels
    print(f"[OK] Time domain reconstituted: 40 frames spanning 0 to {T_expected:.3f} ms")

    # 3. Test Full GUI Integration with Time Domain
    print("\n--- TEST 3: GUI Navigation in Harmonic & Time Domain ---")
    window = SLMMainWindow(initial_dir=path_30k)
    window.show()
    QTest.qWait(100)

    assert window.btn_hermitian_transform.isVisible() is True
    print("[OK] 'Transform to Time Domain' button is visible for harmonic dataset")

    # Execute Transform via dialog
    dlg = SLMHermitianDialog(window.dataset, window)
    dlg.transformed.connect(window._on_time_domain_transformed)
    dlg.spin_freq.setValue(180.0)
    dlg.spin_pts.setValue(30)
    dlg._run_transform()
    dlg.worker.wait(5000)
    QTest.qWait(100)

    # Check GUI is now in Time Domain
    assert window.dataset.series_variable_name == "Time (ms)"
    assert len(window.dataset) == 30
    assert window.btn_switch_domain.isVisible() is True
    assert "PR" in window.dataset[0].available_channels
    print(f"[OK] GUI successfully transformed into Time Domain ({len(window.dataset)} time slices)")

    # Test scrubbing through time
    window.combo_steps.setCurrentIndex(15) # middle of time period
    QTest.qWait(50)
    assert window.current_scan_idx == 15
    print("[OK] Time scrubbing working smoothly in GUI")

    # Test toggling back to Harmonics Domain
    window.btn_switch_domain.click()
    QTest.qWait(50)
    assert window.dataset.is_harmonic_series is True
    assert len(window.dataset) == 300
    print("[OK] Successfully toggled back to Harmonic Frequency Domain (300 steps)")

    # Toggle again back to Time Domain
    window.btn_switch_domain.click()
    QTest.qWait(50)
    assert window.dataset.series_variable_name == "Time (ms)"
    print("[OK] Successfully toggled back to Time Domain")

    print("\n=======================================================")
    print("  ALL HARMONIC & TIME DOMAIN TESTS PASSED PERFECTLY!  ")
    print("=======================================================")


if __name__ == "__main__":
    test_harmonics_and_timedomain()
