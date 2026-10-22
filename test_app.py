"""Comprehensive automated test for SLM Image Processor."""

import os
import sys
import numpy as np

def run_tests():
    print("=== Testing SLM Data Model ===")
    from slm_processor.data_model import SLMDataset, SLMScan
    from slm_processor import processing

    ds = SLMDataset("SLM_images")
    assert len(ds) == 5, f"Expected 5 scans, got {len(ds)}"
    print(f"Loaded {len(ds)} scans: {[s.step_name for s in ds.scans]}")

    scan0 = ds[0]
    assert scan0.shape == (125, 125), f"Expected shape (125, 125), got {scan0.shape}"
    assert len(scan0.x_volts) == 125, f"Expected 125 X coords, got {len(scan0.x_volts)}"
    assert len(scan0.y_volts) == 125, f"Expected 125 Y coords, got {len(scan0.y_volts)}"
    assert np.isclose(scan0.x_volts[0], -0.4, atol=1e-3)
    assert np.isclose(scan0.x_volts[-1], 0.4, atol=1e-3)
    assert np.isclose(scan0.y_volts[0], -0.7, atol=1e-3)
    assert np.isclose(scan0.y_volts[-1], 0.7, atol=1e-3)
    print("Coordinates check: PASSED (Line 1 = X [-0.4, 0.4] V, Line 2 = Y [-0.7, 0.7] V)")

    channels = [
        "PR_R", "PR_X", "PR_Y", "PR_THETA", "PR_X_ROT", "PR_Y_ROT",
        "PD_R", "PD_X", "PD_Y", "PD_THETA",
        "NORM_R", "NORM_X", "NORM_ROT"
    ]
    for ch in channels:
        arr = scan0.get_channel(ch, phase_deg=45.0)
        assert arr.shape == (125, 125)
        assert not np.isnan(arr).any(), f"NaN found in channel {ch}"
    print(f"Channel calculation ({len(channels)} channels): PASSED")

    auto_p = scan0.calculate_auto_phase()
    print(f"Scan 0 Auto-Phase: {auto_p:.2f} deg")
    assert -180.0 <= auto_p <= 180.0
    print("Auto-phase calculation: PASSED")

    print("\n=== Testing Preprocessing Algorithms ===")
    img = scan0.r_pr

    # Plane fit
    p_img, (a, b, c) = processing.fit_and_subtract_plane(img)
    assert p_img.shape == img.shape
    print("Plane tilt subtraction: PASSED")

    # Poly2 fit
    poly_img, coeffs = processing.fit_and_subtract_poly2(img)
    assert poly_img.shape == img.shape
    print("Polynomial 2nd order subtraction: PASSED")

    # Line subtractions
    for mode in ["row_mean", "row_median", "row_linear", "col_mean", "col_median"]:
        l_img = processing.line_by_line_subtraction(img, mode=mode)
        assert l_img.shape == img.shape
    print("Line-by-line subtractions: PASSED")

    # Filters
    g_img = processing.apply_spatial_filter(img, "gaussian", 1.5)
    m_img = processing.apply_spatial_filter(img, "median", 3)
    assert g_img.shape == img.shape and m_img.shape == img.shape
    print("Spatial filters (Gaussian, Median): PASSED")

    # 1D line profile
    dist, val, _ = processing.extract_line_profile(img, (0, 62), (124, 62))
    assert len(dist) == len(val) and len(dist) > 0
    print(f"1D line cut ({len(dist)} points): PASSED")

    # ROI stats
    roi_stats = processing.calculate_roi_stats(img, (20, 20, 80, 80))
    assert "mean" in roi_stats and "std" in roi_stats and roi_stats["count"] == 61 * 61
    print("ROI calculation: PASSED")

    print("\n=== Testing PyQt6 GUI Components ===")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt6.QtWidgets import QApplication
    from slm_processor.main_window import SLMMainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = SLMMainWindow(initial_dir="SLM_images")
    assert window.dataset is not None
    assert len(window.dataset) == 5

    # Switch steps
    window._on_step_changed(2)
    assert window.current_scan_idx == 2

    # Switch channel
    window.combo_channel.setCurrentIndex(6) # PD_R
    window._on_channel_changed(6)

    # Test auto phase button
    window._on_auto_phase_clicked()
    
    # Test line cuts
    window._on_quick_hcut()
    window._on_quick_vcut()

    # Test units toggle and objective switching
    window.combo_units.setCurrentIndex(1) # um
    assert window.calib_unit == "um"
    window.combo_objective.setCurrentText("100x")
    assert np.isclose(window.calib_x_scale, 29.721644, atol=1e-4)
    assert np.isclose(window.calib_y_scale, 31.549683, atol=1e-4)
    window.combo_objective.setCurrentText("50x")
    assert np.isclose(window.calib_x_scale, 62.166693, atol=1e-4)
    assert np.isclose(window.calib_y_scale, 64.654161, atol=1e-4)
    window.combo_units.setCurrentIndex(0) # V
    assert window.calib_unit == "V"

    # Test resizable splitter (left panel width adjustment)
    assert hasattr(window, "main_splitter")
    sizes = window.main_splitter.sizes()
    assert len(sizes) == 2
    window.main_splitter.setSizes([400, 960])
    assert window.main_splitter.sizes()[0] > 0
    print("Resizable panel splitter: PASSED")

    # Test overlay channel and alpha adjustment
    overlay_idx = window.combo_channel.count() - 1
    window.combo_channel.setCurrentIndex(overlay_idx)
    assert window._get_current_channel_key() == "OVERLAY"
    window.slider_overlay_alpha.setValue(45)
    assert window.canvas_2d.is_overlay_mode is True
    assert np.isclose(window.canvas_2d.overlay_alpha, 0.45)
    assert np.isclose(window.overlay_view.alpha_val, 0.45)
    print("PR on PD overlay & dynamic alpha: PASSED")

    # Test multi-channel handler & arbitrary derived channels
    from slm_processor.widgets.channel_manager_dialog import SLMChannelManagerDialog
    dlg = SLMChannelManagerDialog(window.dataset)
    assert dlg.list_channels.count() > 0
    dlg.close()

    # Add custom lock-in pair, ratio, and diff
    window.dataset.add_lockin_pair_to_all("CustomLI", "X_PR", "Y_PR")
    window.dataset.add_ratio_to_all("CustomRatio", "PR_R", "PD_R")
    window.dataset.add_diff_to_all("CustomDiff", "X_PR", "Y_PR")
    window._refresh_channel_dropdowns()

    assert "CustomLI_R" in window.dataset[0].available_channels
    assert "CustomRatio" in window.dataset[0].available_channels
    assert "CustomDiff" in window.dataset[0].available_channels

    # Test arbitrary channel overlap: overlap CustomRatio on CustomDiff
    window.combo_channel.setCurrentText("Overlay: Blend Any 2 Channels")
    window.combo_overlay_pr_chan.setCurrentText("CustomRatio")
    window.combo_overlay_pd_chan.setCurrentText("CustomDiff")
    window.slider_overlay_alpha.setValue(55)
    window._update_display()

    assert window.canvas_2d.is_overlay_mode is True
    assert np.isclose(window.canvas_2d.overlay_alpha, 0.55)
    print("Multi-channel creation & arbitrary layer overlap: PASSED")

    # Test tabs including dedicated overlay tab
    window.tabs.setCurrentIndex(1) # Dual view
    window._update_dual_view()
    window.tabs.setCurrentIndex(2) # Dedicated Overlay view
    window.overlay_view.combo_over_channel.setCurrentText("CustomRatio")
    window.overlay_view.combo_base_channel.setCurrentText("CustomDiff")
    window.overlay_view.refresh_display()
    window.tabs.setCurrentIndex(3) # Series
    window._update_series_trend()

    print("PyQt6 GUI initialization & callbacks: ALL PASSED")
    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
