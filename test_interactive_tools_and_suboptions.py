"""Comprehensive verification of interactive tools across all windows and the PR on PD suboption."""

import os
import sys
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from slm_processor.main_window import SLMMainWindow
from slm_processor.data_model import SLMDataset


def test_interactive_tools_and_suboptions():
    print("=================================================================")
    print("   TESTING INTERACTIVE TOOLS & PR ON PD MAIN PANEL SUBOPTION     ")
    print("=================================================================")

    app = QApplication.instance() or QApplication(sys.argv)
    window = SLMMainWindow(initial_dir="SLM_images")
    window.resize(1200, 800)
    window.show()
    QTest.qWait(100)

    # -------------------------------------------------------------
    # PART 1: Main Panel PR on PD Suboption
    # -------------------------------------------------------------
    print("\n--- TEST 1: Main Panel Suboption (Single Channel vs PR on PD Overlay) ---")
    window.tabs.setCurrentIndex(0)
    assert window.btn_sub_single.isChecked() is True
    assert window.btn_sub_overlay.isChecked() is False
    assert window.canvas_2d.is_overlay_mode is False
    assert window.widget_overlay_inline.isHidden() is True
    print("  [OK] Default state is Single Channel mode")

    # Toggle to PR on PD Overlay
    window.btn_sub_overlay.click()
    QTest.qWait(50)
    assert window.btn_sub_overlay.isChecked() is True
    assert window.btn_sub_single.isChecked() is False
    assert window.canvas_2d.is_overlay_mode is True
    assert window.widget_overlay_inline.isHidden() is False
    print("  [OK] Successfully toggled to PR on PD (Overlay) suboption")
    print("  [OK] Inline overlay controls are visible right above 2D image")

    # Adjust inline overlay controls
    window.combo_overlay_pr_chan.setCurrentText("PR_R")
    window.combo_overlay_pd_chan.setCurrentText("PD_R")
    window.slider_overlay_alpha.setValue(75)
    assert window.spin_overlay_alpha.value() == 75
    assert np.isclose(window.canvas_2d.overlay_alpha, 0.75)
    print("  [OK] Inline overlay Alpha adjustment verified (75%)")

    # Test Cbar Min alpha=0 to Max alpha=1 checkbox
    window.chk_overlay_cbar_alpha.setChecked(True)
    assert window.canvas_2d.overlay_cbar_alpha is True
    assert window.slider_overlay_alpha.value() == 100
    print("  [OK] Inline Cbar Min alpha=0 to Max alpha=1 checkbox verified")
    window.chk_overlay_cbar_alpha.setChecked(False)

    # Line profile cut on PR on PD Overlay
    window.rb_tool_line.setChecked(True)
    window._on_tool_mode_changed(window.rb_tool_line)
    window._on_quick_hcut()
    assert window.canvas_profile.values is not None
    assert window.canvas_profile.values_pd is not None
    assert "PR_R" in window.canvas_profile.title_label.text()
    assert "PD_R" in window.canvas_profile.title_label.text()
    print("  [OK] Line cut on overlay displays dual profile (PR_R & PD_R)")

    # Toggle back to Single Channel
    window.btn_sub_single.click()
    QTest.qWait(50)
    assert window.canvas_2d.is_overlay_mode is False
    assert window.widget_overlay_inline.isHidden() is True
    print("  [OK] Successfully switched back to Single Channel mode")

    # -------------------------------------------------------------
    # PART 2: Dual View Interactive Tools
    # -------------------------------------------------------------
    print("\n--- TEST 2: Dual View (Side-by-Side) Interactive Tools ---")
    window.tabs.setCurrentIndex(1)
    window._update_dual_view()
    QTest.qWait(50)

    # 1. Mode propagation
    for mode in ["crosshair", "line", "roi", "navigate"]:
        rb = {
            "crosshair": window.rb_tool_crosshair,
            "line": window.rb_tool_line,
            "roi": window.rb_tool_roi,
            "navigate": window.rb_tool_navigate
        }[mode]
        rb.setChecked(True)
        window._on_tool_mode_changed(rb)
        assert window.dual_view.active_mode == mode
        assert window.dual_view.canvas_left.interaction_mode == mode
        assert window.dual_view.canvas_right.interaction_mode == mode
    print("  [OK] Tool mode changes propagate synchronously to both Dual View canvases")

    # 2. Line Profile Cut in Dual View
    window.rb_tool_line.setChecked(True)
    window._on_tool_mode_changed(window.rb_tool_line)

    # Draw cut on Left canvas: should sync to Right canvas and update profile
    window.dual_view.canvas_left.set_line_cut(5.0, 20.0, 55.0, 20.0, emit_signal=True)
    assert window.dual_view.canvas_right.line_points == (5.0, 20.0, 55.0, 20.0)
    assert window.dual_view.line_points == (5.0, 20.0, 55.0, 20.0)
    assert window.dual_view.canvas_profile.values is not None
    assert window.dual_view.canvas_profile.values_pd is not None
    assert len(window.dual_view.canvas_profile.distances) > 0
    print("  [OK] Drawing cut on Left canvas synchronizes to Right and plots dual 1D profile")

    # Draw cut on Right canvas: should sync to Left canvas
    window.dual_view.canvas_right.set_line_cut(10.0, 35.0, 50.0, 35.0, emit_signal=True)
    assert window.dual_view.canvas_left.line_points == (10.0, 35.0, 50.0, 35.0)
    assert window.dual_view.line_points == (10.0, 35.0, 50.0, 35.0)
    print("  [OK] Drawing cut on Right canvas synchronizes to Left and updates 1D profile")

    # Quick Horizontal Mid-Cut in Dual View
    window._on_quick_hcut()
    rows = window.dual_view.canvas_left.image_data.shape[0]
    expected_row = float(rows // 2)
    assert window.dual_view.line_points[1] == expected_row
    assert window.dual_view.canvas_right.line_points[1] == expected_row
    print("  [OK] Quick Horizontal Mid-Cut works on Dual View")

    # Quick Vertical Mid-Cut in Dual View
    window._on_quick_vcut()
    cols = window.dual_view.canvas_left.image_data.shape[1]
    expected_col = float(cols // 2)
    assert window.dual_view.line_points[0] == expected_col
    assert window.dual_view.canvas_right.line_points[0] == expected_col
    print("  [OK] Quick Vertical Mid-Cut works on Dual View")

    # 3. ROI Rectangle Box in Dual View
    window.rb_tool_roi.setChecked(True)
    window._on_tool_mode_changed(window.rb_tool_roi)

    window.dual_view.canvas_left.set_roi(8, 8, 32, 32, emit_signal=True)
    assert window.dual_view.canvas_right.roi_box == (8, 8, 32, 32)
    assert window.canvas_2d.roi_box == (8, 8, 32, 32)
    mean_val = float(window.table_roi.item(0, 1).text())
    assert not np.isnan(mean_val)
    print(f"  [OK] Drawing ROI in Dual View updates both canvases and sidebar ROI table (Mean={mean_val:.4e})")

    # 4. Crosshair Readout in Dual View
    window.rb_tool_crosshair.setChecked(True)
    window._on_tool_mode_changed(window.rb_tool_crosshair)
    window.dual_view.canvas_left.pixel_hovered.emit(25, 25, 0.05, -0.05, 1.23e-3)
    assert "Cursor:" in window.dual_view.lbl_readout.text()
    assert window.dual_view.canvas_right.crosshair_pos == (25, 25)
    print("  [OK] Crosshair inspection and readout synchronized across Dual View")

    # 5. Invert Y-Axis in Dual View and Single View
    window.chk_invert_y.setChecked(True)
    assert window.dual_view.canvas_left.invert_y is True
    assert window.dual_view.canvas_right.invert_y is True
    assert window.canvas_2d.invert_y is True
    assert window.dual_view.canvas_left.ax.get_ylim()[0] > window.dual_view.canvas_left.ax.get_ylim()[1]
    assert window.dual_view.canvas_right.ax.get_ylim()[0] > window.dual_view.canvas_right.ax.get_ylim()[1]
    assert window.canvas_2d.ax.get_ylim()[0] > window.canvas_2d.ax.get_ylim()[1]

    window.chk_invert_y.setChecked(False)
    assert window.dual_view.canvas_left.invert_y is False
    assert window.dual_view.canvas_right.invert_y is False
    assert window.canvas_2d.invert_y is False
    assert window.dual_view.canvas_left.ax.get_ylim()[0] < window.dual_view.canvas_left.ax.get_ylim()[1]
    assert window.dual_view.canvas_right.ax.get_ylim()[0] < window.dual_view.canvas_right.ax.get_ylim()[1]
    assert window.canvas_2d.ax.get_ylim()[0] < window.canvas_2d.ax.get_ylim()[1]
    print("  [OK] Invert Y-Axis toggles both Dual View and Single View canvases with axis inversion verified")

    # -------------------------------------------------------------
    # PART 3: Tab Navigation & Trend Analysis
    # -------------------------------------------------------------
    print("\n--- TEST 3: App Tab Structure ---")
    assert window.tabs.count() == 3
    assert window.tabs.tabText(0) == "2D Image & 1D Cut"
    assert window.tabs.tabText(1) == "Dual View (Side-by-Side)"
    assert window.tabs.tabText(2) == "Run Series Trend"
    print("  [OK] Tab structure verified: [2D Image & 1D Cut], [Dual View], [Run Series Trend]")

    # Switch to Series Trend Tab
    window.tabs.setCurrentIndex(2)
    window._update_series_trend()
    assert window.series_metrics is not None
    assert len(window.series_metrics) == 5
    print(f"  [OK] Run Series Trend tab displays {len(window.series_metrics)} scan points based on active ROI")

    print("\n=================================================================")
    print("   ALL INTERACTIVE TOOLS & SUBOPTION TESTS PASSED PERFECTLY!     ")
    print("=================================================================")


if __name__ == "__main__":
    test_interactive_tools_and_suboptions()
