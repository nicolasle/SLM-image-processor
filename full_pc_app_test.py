"""Full end-to-end PC simulation test suite for the SLM Image Processor application.
Simulates real user actions across all tabs, views, channel selections, and tools.
"""

import os
import sys
import numpy as np

# Use offscreen platform for headless execution
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

# Bypass modal message boxes during automated test execution
QMessageBox.information = lambda *args, **kwargs: QMessageBox.StandardButton.Ok
QMessageBox.warning = lambda *args, **kwargs: QMessageBox.StandardButton.Ok

from slm_processor.main_window import SLMMainWindow
from slm_processor.widgets.channel_manager_dialog import SLMChannelManagerDialog


def run_full_suite():
    print("=======================================================")
    print("      SLM IMAGE PROCESSOR - FULL PC TEST SUITE        ")
    print("=======================================================")

    app = QApplication.instance() or QApplication(sys.argv)
    window = SLMMainWindow(initial_dir="SLM_images")
    window.show()
    QTest.qWait(100)

    assert window.dataset is not None, "Dataset failed to load!"
    assert len(window.dataset) == 5, f"Expected 5 scan steps, found {len(window.dataset)}"
    print(f"[OK] Dataset loaded: {len(window.dataset)} scans ({[s.step_name for s in window.dataset.scans]})")

    # ----------------------------------------------------
    # TEST 1: Resizable Splitter
    # ----------------------------------------------------
    print("\n--- TEST 1: Resizable Left Panel ---")
    orig_sizes = window.main_splitter.sizes()
    print(f"Original splitter sizes: {orig_sizes}")
    window.main_splitter.setSizes([450, 910])
    QTest.qWait(50)
    new_sizes = window.main_splitter.sizes()
    assert new_sizes[0] > 0 and new_sizes[1] > 0
    print(f"[OK] Resized splitter sizes: {new_sizes}")

    # ----------------------------------------------------
    # TEST 2: Dual Side-by-Side View (The reported issue)
    # ----------------------------------------------------
    print("\n--- TEST 2: Dual Side-by-Side View (Left & Right Channels) ---")
    window.tabs.setCurrentIndex(1)  # Switch to Dual View
    QTest.qWait(50)

    # Check channels available in dual view combos
    chans = [window.dual_view.combo_left.itemText(i) for i in range(window.dual_view.combo_left.count())]
    print(f"Dual View channel options ({len(chans)}): {chans[:6]}...")

    # Test selecting View 1 = X_PD, View 2 = Y_PR
    print("Selecting View 1 = 'X_PD' and View 2 = 'Y_PR'...")
    window.dual_view.combo_left.setCurrentText("X_PD")
    window.dual_view.combo_right.setCurrentText("Y_PR")
    QTest.qWait(50)

    scan0 = window.dataset[0]
    expected_left_x_pd = scan0.get_channel("X_PD")
    expected_right_y_pr = scan0.get_channel("Y_PR")

    assert np.allclose(window.dual_view.canvas_left.image_data, expected_left_x_pd), "Left canvas does not match X_PD!"
    assert np.allclose(window.dual_view.canvas_right.image_data, expected_right_y_pr), "Right canvas does not match Y_PR!"
    print("[OK] View 1 correctly renders X_PD, View 2 correctly renders Y_PR")

    # Now change Right View to PR_R and check it updates!
    print("Switching View 2 to 'PR_R'...")
    window.dual_view.combo_right.setCurrentText("PR_R")
    QTest.qWait(50)
    expected_right_pr_r = scan0.get_channel("PR_R")
    assert np.allclose(window.dual_view.canvas_right.image_data, expected_right_pr_r), "Right canvas did NOT update to PR_R!"
    print("[OK] View 2 successfully changed to PR_R!")

    # Switch View 2 to X_PR and check it updates!
    print("Switching View 2 to 'X_PR'...")
    window.dual_view.combo_right.setCurrentText("X_PR")
    QTest.qWait(50)
    expected_right_x_pr = scan0.get_channel("X_PR")
    assert np.allclose(window.dual_view.canvas_right.image_data, expected_right_x_pr), "Right canvas did NOT update to X_PR!"
    print("[OK] View 2 successfully changed to X_PR!")

    # Switch View 1 to PD_R
    window.dual_view.combo_left.setCurrentText("PD_R")
    QTest.qWait(50)
    assert np.allclose(window.dual_view.canvas_left.image_data, scan0.get_channel("PD_R"))
    print("[OK] View 1 successfully changed to PD_R!")

    # Test colormap changes in Dual View
    window.dual_view.combo_left_cmap.setCurrentText("cividis")
    window.dual_view.combo_right_cmap.setCurrentText("turbo")
    QTest.qWait(50)
    assert window.dual_view.canvas_left.cmap_name == "cividis"
    assert window.dual_view.canvas_right.cmap_name == "turbo"
    print("[OK] Independent colormaps on Dual View passed")

    # Test step switching while in Dual View
    print("Switching to scan step 114 while in Dual View...")
    window.combo_steps.setCurrentIndex(3) # step 114
    QTest.qWait(50)
    scan_114 = window.dataset[3]
    assert np.allclose(window.dual_view.canvas_left.image_data, scan_114.get_channel("PD_R"))
    assert np.allclose(window.dual_view.canvas_right.image_data, scan_114.get_channel("X_PR"))
    print("[OK] Step switching in Dual View passed")
    window.combo_steps.setCurrentIndex(0) # back to 0

    # ----------------------------------------------------
    # TEST 3: Multi-Channel Manager (Creating Custom Channels)
    # ----------------------------------------------------
    print("\n--- TEST 3: Multi-Channel Manager (Add/Remove Custom Channels) ---")
    dlg = SLMChannelManagerDialog(window.dataset, window)
    QTest.qWait(50)
    dlg.edit_pair_name.setText("TestLockIn")
    dlg.combo_x_chan.setCurrentText("X_PR")
    dlg.combo_y_chan.setCurrentText("Y_PR")
    dlg._on_create_lockin_pair()
    QTest.qWait(50)

    # Check that new channels were created across dataset
    for scan in window.dataset.scans:
        assert "TestLockIn_R" in scan.available_channels
        assert "TestLockIn_Theta" in scan.available_channels
        assert "TestLockIn_X_Rot" in scan.available_channels

    # Create custom ratio: TestRatio = PR_R / PD_R
    dlg.edit_ratio_name.setText("CustomRatio_PR_PD")
    dlg.combo_num_chan.setCurrentText("PR_R")
    dlg.combo_denom_chan.setCurrentText("PD_R")
    dlg._on_create_ratio()
    QTest.qWait(50)

    for scan in window.dataset.scans:
        assert "CustomRatio_PR_PD" in scan.available_channels

    dlg.accept()
    window._on_channels_updated()
    QTest.qWait(50)

    # Verify that the new custom channels appear in Dual View and can be selected on the Right canvas
    print("Selecting newly created 'CustomRatio_PR_PD' in Right Dual View...")
    window.dual_view.combo_right.setCurrentText("CustomRatio_PR_PD")
    QTest.qWait(50)
    expected_ratio = scan0.get_channel("CustomRatio_PR_PD")
    assert np.allclose(window.dual_view.canvas_right.image_data, expected_ratio)
    print("[OK] Custom derived channel rendered on Right Dual View successfully!")

    # ----------------------------------------------------
    # TEST 4: PR on PD Overlay & Dedicated Overlay View
    # ----------------------------------------------------
    print("\n--- TEST 4: Dedicated Overlay View & Real-Time Alpha ---")
    window.tabs.setCurrentIndex(2)  # Switch to Overlay View
    QTest.qWait(50)

    # Configure overlay layers: Top = PR_R, Base = PD_R
    window.overlay_view.combo_over_channel.setCurrentText("PR_R")
    window.overlay_view.combo_base_channel.setCurrentText("PD_R")
    QTest.qWait(50)

    assert window.overlay_view.canvas_2d.is_overlay_mode is True
    assert np.allclose(window.overlay_view.canvas_2d.image_data, scan0.get_channel("PD_R"))
    assert np.allclose(window.overlay_view.canvas_2d.overlay_data, scan0.get_channel("PR_R"))

    # Test changing Alpha across various values
    for alpha_pct in [0, 25, 50, 75, 100]:
        window.overlay_view.slider_alpha.setValue(alpha_pct)
        QTest.qWait(20)
        assert np.isclose(window.overlay_view.canvas_2d.overlay_alpha, alpha_pct / 100.0)
    print("[OK] Interactive alpha slider test passed (0% to 100%)")

    # Test dual line cut in Overlay View
    window.overlay_view.canvas_2d.set_horizontal_cut(62)
    QTest.qWait(50)
    assert window.overlay_view.canvas_profile.values is not None
    assert window.overlay_view.canvas_profile.values_pd is not None
    print(f"[OK] Overlay Dual 1D cut points: PR={len(window.overlay_view.canvas_profile.values)}, PD={len(window.overlay_view.canvas_profile.values_pd)}")

    # ----------------------------------------------------
    # TEST 5: Single Scan View & 1D Cut
    # ----------------------------------------------------
    print("\n--- TEST 5: Single Scan View & Preprocessing Tools ---")
    window.tabs.setCurrentIndex(0) # Main view
    QTest.qWait(50)

    # Test plane tilt subtraction
    window.combo_plane.setCurrentText("Subtract Plane Tilt (1st Order)")
    QTest.qWait(50)
    plane_data = window.current_processed_data.copy()
    window.combo_plane.setCurrentText("None")
    QTest.qWait(50)
    raw_data = window.current_processed_data.copy()
    assert not np.allclose(plane_data, raw_data), "Plane tilt subtraction had no effect!"
    print("[OK] Plane tilt subtraction verified")

    # Test auto phase button
    window._on_auto_phase_clicked()
    QTest.qWait(50)
    opt_phase = window.spin_phase.value()
    assert np.isclose(opt_phase, scan0.calculate_auto_phase(), atol=0.1)
    print(f"[OK] Auto-phase button verified (Phase = {opt_phase:.2f}°)")

    # Test objective calibration factors
    for obj_name, (cx, cy) in [("50x", (62.166693, 64.654161)), ("100x", (29.721644, 31.549683)), ("10x", (308.087566, 320.422395))]:
        window.combo_objective.setCurrentText(obj_name)
        QTest.qWait(20)
        assert np.isclose(window.calib_x_scale, cx, atol=1e-4)
        assert np.isclose(window.calib_y_scale, cy, atol=1e-4)
    print("[OK] Objective calibration presets verified")

    # ----------------------------------------------------
    # TEST 6: Step Series Trend
    # ----------------------------------------------------
    print("\n--- TEST 6: Run Step Series Trend ---")
    window.tabs.setCurrentIndex(3) # Series trend
    QTest.qWait(50)
    assert len(window.canvas_series.step_vals) == 5
    assert len(window.canvas_series.metric_vals) == 5
    print(f"[OK] Series trend computed across all 5 steps: {window.canvas_series.metric_vals}")

    print("\n=======================================================")
    print("  ALL FULL PC APPLICATION TESTS PASSED PERFECTLY!     ")
    print("=======================================================")

if __name__ == "__main__":
    run_full_suite()
