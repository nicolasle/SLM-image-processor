"""Test suite for Colorbar min alpha=0 to max alpha=1 in overlay view."""

import sys
import numpy as np
from PyQt6.QtWidgets import QApplication

from slm_processor.widgets.image_canvas import SLMImageCanvas
from slm_processor.widgets.overlay_view import SLMOverlayView
from slm_processor.data_model import SLMDataset


def test_image_canvas_cbar_alpha():
    print("Testing SLMImageCanvas colorbar-based alpha mapping...")
    app = QApplication.instance() or QApplication(sys.argv)

    canvas = SLMImageCanvas(title="Test Cbar Alpha")
    base = np.zeros((100, 100), dtype=np.float32)
    # Gradient of overlay values from -2.0 to +8.0
    over = np.linspace(-2.0, 8.0, 100)[:, None] * np.ones((1, 100), dtype=np.float32)

    # 1. Standard uniform alpha
    canvas.set_overlay_data(base, over, alpha=0.6, clim_overlay=(-2.0, 8.0), cbar_alpha=False)
    assert not canvas.overlay_cbar_alpha
    assert canvas.overlay_alpha == 0.6
    eff = canvas._compute_effective_alpha()
    assert isinstance(eff, float) and eff == 0.6
    print("  [OK] Default uniform alpha matches 0.6")

    # 2. Enable cbar alpha mapping
    canvas.set_overlay_cbar_alpha(True)
    assert canvas.overlay_cbar_alpha
    eff_map = canvas._compute_effective_alpha()
    assert isinstance(eff_map, np.ndarray)
    assert eff_map.shape == (100, 100)

    # Min value (-2.0) has alpha = 0.0
    assert np.isclose(eff_map[0, :], 0.0).all(), f"Min cbar alpha is not 0: {eff_map[0, 0]}"
    # Max value (8.0) has alpha = 0.6
    assert np.isclose(eff_map[-1, :], 0.6).all(), f"Max cbar alpha is not 0.6: {eff_map[-1, 0]}"
    # Midpoint (3.0) has alpha = 0.3
    mid_row = 50
    assert np.isclose(eff_map[mid_row, 0], 0.3, atol=0.02)
    print("  [OK] Cbar Min (-2.0) has alpha=0.0, Cbar Max (8.0) has alpha=0.6, Mid has alpha=0.3")

    # Set slider to 1.0 (100%)
    canvas.set_overlay_alpha(1.0)
    eff_map_1 = canvas._compute_effective_alpha()
    assert np.isclose(eff_map_1[0, :], 0.0).all()
    assert np.isclose(eff_map_1[-1, :], 1.0).all()
    print("  [OK] When slider=1.0: Cbar Min has alpha=0.0 and Cbar Max has alpha=1.0")

    # 3. Toggle back to uniform
    canvas.set_overlay_cbar_alpha(False)
    assert not canvas.overlay_cbar_alpha
    eff_back = canvas._compute_effective_alpha()
    assert isinstance(eff_back, float) and eff_back == 1.0
    print("  [OK] Successfully toggled back to uniform alpha without error")


def test_overlay_view_widget():
    print("Testing SLMOverlayView checkbox and interaction...")
    app = QApplication.instance() or QApplication(sys.argv)

    view = SLMOverlayView()
    assert hasattr(view, "chk_cbar_alpha")
    assert not view.chk_cbar_alpha.isChecked()

    # Load real scan
    ds = SLMDataset("SLM_images")
    scan = ds.scans[0]
    img = scan.get_channel("X_PD")
    cols = img.shape[1]
    rows = img.shape[0]
    view.set_scan_data(
        scan=scan,
        x_coords=np.linspace(-0.4, 0.4, cols),
        y_coords=np.linspace(-0.7, 0.7, rows),
        unit="V"
    )

    # Check the checkbox
    signal_received = []
    view.gradient_changed.connect(lambda val: signal_received.append(val))

    view.chk_cbar_alpha.setChecked(True)
    assert view.canvas_2d.overlay_cbar_alpha
    assert len(signal_received) == 1 and signal_received[0] is True
    assert view.slider_alpha.value() == 100 # Auto-set to 100% so top is 1.0
    print("  [OK] Checkbox toggles cbar alpha and sets slider to 100%")

    # Hover inspection test
    view._on_overlay_hovered(col=50, row=0, x=0.0, y=0.0, val_over=-1e-3, val_base=2e-3)
    assert "Cbar Min" in view.lbl_readout.text()
    print("  [OK] Hover readout displays cbar local alpha correctly")

    view.chk_cbar_alpha.setChecked(False)
    assert not view.canvas_2d.overlay_cbar_alpha
    assert len(signal_received) == 2 and signal_received[1] is False
    print("  [OK] Checkbox uncheck restores uniform overlay")


if __name__ == "__main__":
    test_image_canvas_cbar_alpha()
    test_overlay_view_widget()
    print("\nALL CBAR ALPHA MAPPING TESTS PASSED SUCCESSFULLY! (100%)")
