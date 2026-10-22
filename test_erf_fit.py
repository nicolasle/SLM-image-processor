"""Test suite for Error Function (Erf) knife-edge beam fitting and UI integration."""

import sys
import numpy as np
from scipy.special import erf
from PyQt6.QtWidgets import QApplication

from slm_processor import processing
from slm_processor.widgets.profile_canvas import SLMProfileCanvas
from slm_processor.widgets.erf_fit_dialog import SLMErfFitDialog


def test_erf_algorithm_um():
    print("Testing Erf fitting algorithm in microns...")
    x = np.linspace(-10, 10, 200) # um
    w_true = 1.35 # um
    x0_true = 1.2 # um
    y0_true = 0.5
    A_true = 1.8

    y_clean = y0_true + A_true * erf(np.sqrt(2.0) * (x - x0_true) / w_true)
    rng = np.random.default_rng(42)
    y_noisy = y_clean + rng.normal(0, 0.015, size=x.shape)

    res = processing.fit_erf_beam(x, y_noisy, unit="µm", with_linear_drift=True)
    assert res.success, f"Fit failed: {res.message}"
    assert abs(res.w - w_true) < 0.05, f"w mismatch: got {res.w}, expected {w_true}"
    assert abs(res.x0 - x0_true) < 0.05, f"x0 mismatch: got {res.x0}, expected {x0_true}"
    assert res.r_squared > 0.995, f"R2 too low: {res.r_squared}"
    assert abs(res.diameter - 2.0 * res.w) < 1e-9
    assert abs(res.fwhm - np.sqrt(2.0 * np.log(2.0)) * res.w) < 1e-9
    print(f"  [OK] Fitted w: {res.w:.4f} um (true: {w_true}), 2w: {res.diameter:.4f} um, R2: {res.r_squared:.5f}")


def test_erf_algorithm_volts():
    print("Testing Erf fitting algorithm in mirror Volts with linear drift...")
    x = np.linspace(-0.25, 0.25, 300) # V
    w_true = 0.022 # V
    x0_true = -0.015 # V
    y0_true = 1.1
    A_true = -0.75 # Falling edge!
    m_true = 0.15 # Linear background slope

    y_clean = y0_true + m_true * (x - x0_true) + A_true * erf(np.sqrt(2.0) * (x - x0_true) / w_true)
    rng = np.random.default_rng(123)
    y_noisy = y_clean + rng.normal(0, 0.008, size=x.shape)

    res = processing.fit_erf_beam(x, y_noisy, unit="V", with_linear_drift=True)
    assert res.success, f"Fit failed: {res.message}"
    assert abs(res.w - w_true) < 0.002, f"w mismatch in V: got {res.w}, expected {w_true}"
    assert abs(res.x0 - x0_true) < 0.002, f"x0 mismatch in V: got {res.x0}, expected {x0_true}"
    assert res.r_squared > 0.99, f"R2 too low: {res.r_squared}"
    print(f"  [OK] Fitted w: {res.w:.5f} V (true: {w_true}), 2w: {res.diameter:.5f} V, R2: {res.r_squared:.5f}")


def test_profile_canvas_ui():
    print("Testing SLMProfileCanvas widget & UI actions...")
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    canvas = SLMProfileCanvas()

    # 1. Single profile in Volts with 50x calibration (62.167 um/V)
    x_v = np.linspace(-0.2, 0.2, 150)
    w_v_true = 0.025 # V -> 1.554 um
    y_v = 0.2 + 0.9 * erf(np.sqrt(2.0) * (x_v - 0.01) / w_v_true)
    canvas.set_profile(x_v, y_v, unit="V", title="Test Cut V", calib_um_per_v=62.1667)

    # Click fit
    canvas._run_fit()
    assert canvas.current_fit is not None
    assert canvas.current_fit.success
    assert abs(canvas.current_fit.w - w_v_true) < 0.002
    assert "V" in canvas.lbl_fit_badge.text()
    assert "µm" in canvas.lbl_fit_badge.text() # Calibrated conversion shown
    print("  [OK] Canvas badge (V) verified")

    # 2. Update to microns
    x_um = x_v * 62.1667
    w_um_true = w_v_true * 62.1667
    canvas.set_profile(x_um, y_v, unit="um", title="Test Cut um")
    assert canvas.current_fit is not None
    assert abs(canvas.current_fit.w - w_um_true) < 0.1
    assert "um" in canvas.lbl_fit_badge.text()
    print("  [OK] Canvas badge (um) verified")

    # 3. Dual profile with PR and PD
    y_pr = 0.1 + 0.8 * erf(np.sqrt(2.0) * (x_um - 0.5) / 1.6)
    y_pd = 1.0 - 0.5 * erf(np.sqrt(2.0) * (x_um + 1.0) / 2.2)
    canvas.set_dual_profile(x_um, y_pr, y_pd, unit="um", title="Dual Cut")
    assert not canvas.combo_fit_channel.isHidden()

    # Fit PR
    canvas.combo_fit_channel.setCurrentIndex(0)
    canvas._run_fit()
    assert abs(canvas.current_fit.w - 1.6) < 0.1

    # Fit PD
    canvas.combo_fit_channel.setCurrentIndex(1)
    canvas._run_fit()
    assert abs(canvas.current_fit.w - 2.2) < 0.1
    print("  [OK] Dual profile channel switching works (w_PR: 1.6 um, w_PD: 2.2 um)")

    # 4. Clear fit
    canvas._clear_fit()
    assert canvas.current_fit is None
    assert canvas.fit_banner.isHidden()
    print("  [OK] Clear fit successfully reset artists and banner")

    # 5. Dialog test
    dlg = SLMErfFitDialog(
        distances=x_um,
        values=y_pr,
        values_pd=y_pd,
        unit="um",
        calib_um_per_v=None
    )
    assert dlg.current_fit is not None
    assert dlg.current_fit.success
    print(f"  [OK] SLMErfFitDialog verified with {dlg.table_params.rowCount()} parameter rows")


if __name__ == "__main__":
    test_erf_algorithm_um()
    test_erf_algorithm_volts()
    test_profile_canvas_ui()
    print("\nALL ERF BEAM SIZE FITTING TESTS PASSED SUCCESSFULLY! (100%)")
