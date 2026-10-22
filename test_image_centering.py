"""Verification test specifically for image centering and aspect ratio behavior."""

import os
import sys
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from slm_processor.main_window import SLMMainWindow
from slm_processor.widgets.image_canvas import SLMImageCanvas


def test_centering():
    print("=======================================================")
    print("          TESTING IMAGE CENTERING & ALIGNMENT          ")
    print("=======================================================")

    app = QApplication.instance() or QApplication(sys.argv)
    window = SLMMainWindow(initial_dir="SLM_images")
    window.resize(1200, 800)
    window.show()
    QTest.qWait(100)

    # 1. Single View (Tab 0) Centering
    print("\n--- TEST 1: Single View Centering ---")
    window.tabs.setCurrentIndex(0)
    QTest.qWait(50)
    canvas = window.canvas_2d
    canvas.canvas.draw()
    
    ax_pos = canvas.ax.get_position()
    cb_pos = canvas.cbar.ax.get_position()
    left_m = ax_pos.x0
    right_m = 1.0 - cb_pos.x1
    diff = abs(left_m - right_m)
    print(f"Single View (Physical): Left={left_m:.3f}, Right={right_m:.3f}, Diff={diff:.3f}")
    assert diff < 0.05, f"Image is not centered horizontally! Left={left_m}, Right={right_m}"
    print("[OK] Single View image is properly centered!")

    # 2. Aspect Ratio Modes Centering (Square and Fill)
    print("\n--- TEST 2: Aspect Ratio Modes Centering ---")
    for mode_idx, mode_name in [(1, "Square (1:1)"), (2, "Fill"), (0, "Physical")]:
        canvas.combo_aspect.setCurrentIndex(mode_idx)
        QTest.qWait(50)
        ax_pos = canvas.ax.get_position()
        cb_pos = canvas.cbar.ax.get_position()
        left_m = ax_pos.x0
        right_m = 1.0 - cb_pos.x1
        diff = abs(left_m - right_m)
        print(f"Mode '{mode_name}': Left={left_m:.3f}, Right={right_m:.3f}, Diff={diff:.3f}")
        assert diff < 0.05, f"Mode {mode_name} not centered! Diff={diff}"
    print("[OK] All aspect ratio modes stay centered!")

    # 3. Dual View (Tab 1) Centering on Both Left and Right Canvases
    print("\n--- TEST 3: Dual View Both Canvases Centering ---")
    window.tabs.setCurrentIndex(1)
    QTest.qWait(50)

    c_left = window.dual_view.canvas_left
    c_right = window.dual_view.canvas_right
    c_left.canvas.draw()
    c_right.canvas.draw()

    pos_l = c_left.ax.get_position()
    cb_l = c_left.cbar.ax.get_position()
    diff_l = abs(pos_l.x0 - (1.0 - cb_l.x1))

    pos_r = c_right.ax.get_position()
    cb_r = c_right.cbar.ax.get_position()
    diff_r = abs(pos_r.x0 - (1.0 - cb_r.x1))

    print(f"Dual View Left Canvas: Left={pos_l.x0:.3f}, Right={1.0-cb_l.x1:.3f}, Diff={diff_l:.3f}")
    print(f"Dual View Right Canvas: Left={pos_r.x0:.3f}, Right={1.0-cb_r.x1:.3f}, Diff={diff_r:.3f}")
    assert diff_l < 0.05, f"Dual Left Canvas not centered! Diff={diff_l}"
    assert diff_r < 0.05, f"Dual Right Canvas not centered! Diff={diff_r}"
    print("[OK] Both Dual View canvases are symmetrically centered!")

    # 4. PR on PD Overlay Centering (Main Panel Suboption)
    print("\n--- TEST 4: PR on PD Overlay Centering ---")
    window.tabs.setCurrentIndex(0)
    window.btn_sub_overlay.click()
    QTest.qWait(50)
    c_over = window.canvas_2d
    c_over.canvas.draw()
    pos_o = c_over.ax.get_position()
    cb_o = c_over.cbar.ax.get_position()
    diff_o = abs(pos_o.x0 - (1.0 - cb_o.x1))
    print(f"Overlay Canvas: Left={pos_o.x0:.3f}, Right={1.0-cb_o.x1:.3f}, Diff={diff_o:.3f}")
    assert diff_o < 0.05, f"Overlay Canvas not centered! Diff={diff_o}"
    print("[OK] PR on PD Overlay canvas is properly centered!")
    window.btn_sub_single.click()

    # 5. Dynamic Window Resizing
    print("\n--- TEST 5: Window Resizing Centering ---")
    window.tabs.setCurrentIndex(0)
    for w, h in [(900, 600), (1400, 750), (1050, 900)]:
        window.resize(w, h)
        QTest.qWait(50)
        canvas.canvas.draw()
        pos = canvas.ax.get_position()
        cb = canvas.cbar.ax.get_position()
        diff = abs(pos.x0 - (1.0 - cb.x1))
        print(f"Window {w}x{h}: Left={pos.x0:.3f}, Right={1.0-cb.x1:.3f}, Diff={diff:.3f}")
        assert diff < 0.05, f"Resized {w}x{h} not centered! Diff={diff}"
    print("[OK] Centering holds perfectly across dynamic window resizing!")

    # 6. Reset View / Center Button
    print("\n--- TEST 6: Center / Reset View Button ---")
    canvas.ax.set_xlim(-0.1, 0.1)
    canvas.ax.set_ylim(-0.1, 0.1)
    canvas.canvas.draw()
    assert abs(canvas.ax.get_xlim()[1] - canvas.ax.get_xlim()[0] - 0.2) < 1e-4

    canvas.btn_center.click()
    QTest.qWait(50)
    full_xlim = canvas.ax.get_xlim()
    assert np.isclose(full_xlim[0], -0.4, atol=1e-2)
    assert np.isclose(full_xlim[1], 0.4, atol=1e-2)
    print(f"[OK] Center button restored full extent: xlim={full_xlim}")

    print("\n=======================================================")
    print("      ALL IMAGE CENTERING TESTS PASSED PERFECTLY!      ")
    print("=======================================================")


if __name__ == "__main__":
    test_centering()
