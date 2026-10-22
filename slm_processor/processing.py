"""Image processing and leveling algorithms for Scanning Laser Microscopy images."""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from scipy import ndimage
from scipy.special import erf, erfinv
from scipy.optimize import curve_fit


def fit_and_subtract_plane(image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Fits a 2D first-order plane (z = a*x + b*y + c) and subtracts it.
    Useful for removing overall sample tilt or optical misalignment.
    """
    rows, cols = image.shape
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]
    
    # Flatten arrays
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    z_flat = image.flatten()
    
    # Design matrix: [x, y, 1]
    A = np.column_stack([x_flat, y_flat, np.ones_like(x_flat)])
    # Least squares fit
    coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
    a, b, c = coeffs
    
    plane = a * x_coords + b * y_coords + c
    return (image - plane).astype(np.float32), (float(a), float(b), float(c))


def fit_and_subtract_poly2(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fits a 2nd order polynomial surface:
    z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    and subtracts it.
    """
    rows, cols = image.shape
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]
    
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    z_flat = image.flatten()
    
    A = np.column_stack([
        x_flat**2,
        y_flat**2,
        x_flat * y_flat,
        x_flat,
        y_flat,
        np.ones_like(x_flat)
    ])
    coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
    
    surface = (
        coeffs[0] * x_coords**2 +
        coeffs[1] * y_coords**2 +
        coeffs[2] * x_coords * y_coords +
        coeffs[3] * x_coords +
        coeffs[4] * y_coords +
        coeffs[5]
    )
    return (image - surface).astype(np.float32), coeffs


def line_by_line_subtraction(image: np.ndarray, mode: str = "row_mean") -> np.ndarray:
    """Performs line-by-line background subtraction.
    Modes:
    - 'row_mean': subtract mean of each row
    - 'row_median': subtract median of each row
    - 'col_mean': subtract mean of each col
    - 'col_median': subtract median of each col
    - 'row_linear': subtract linear fit from each row
    """
    out = image.copy()
    rows, cols = image.shape
    
    if mode == "row_mean":
        row_means = np.mean(image, axis=1, keepdims=True)
        out = image - row_means
    elif mode == "row_median":
        row_meds = np.median(image, axis=1, keepdims=True)
        out = image - row_meds
    elif mode == "col_mean":
        col_means = np.mean(image, axis=0, keepdims=True)
        out = image - col_means
    elif mode == "col_median":
        col_meds = np.median(image, axis=0, keepdims=True)
        out = image - col_meds
    elif mode == "row_linear":
        x = np.arange(cols)
        A = np.column_stack([x, np.ones_like(x)])
        for r in range(rows):
            slope, intercept = np.linalg.lstsq(A, image[r, :], rcond=None)[0]
            out[r, :] = image[r, :] - (slope * x + intercept)
    
    return out.astype(np.float32)


def apply_spatial_filter(image: np.ndarray, filter_type: str = "none", param: float = 1.0) -> np.ndarray:
    """Applies spatial filter (gaussian, median, etc.)."""
    if filter_type.lower() == "gaussian":
        return ndimage.gaussian_filter(image, sigma=max(0.1, param)).astype(np.float32)
    elif filter_type.lower() == "median":
        size = int(param)
        if size % 2 == 0:
            size += 1
        size = max(3, size)
        return ndimage.median_filter(image, size=size).astype(np.float32)
    return image


def extract_line_profile(
    image: np.ndarray,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    num_points: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts an intensity profile along a line segment from p0=(x0, y0) to p1=(x1, y1).
    x and y coordinates are given in pixel units (0 to cols-1, 0 to rows-1).
    Returns (distances_in_pixels, values, (x_coords, y_coords)).
    """
    x0, y0 = p0
    x1, y1 = p1
    length = float(np.hypot(x1 - x0, y1 - y0))
    if num_points is None:
        num_points = max(2, int(np.round(length)))

    t = np.linspace(0.0, 1.0, num_points)
    x_coords = x0 + t * (x1 - x0)
    y_coords = y0 + t * (y1 - y0)

    # map_coordinates requires coordinates ordered as (rows/y, cols/x)
    coords = np.vstack([y_coords, x_coords])
    values = ndimage.map_coordinates(image, coords, order=1, mode="nearest")
    distances = t * length

    return distances, values, (x_coords, y_coords)


def calculate_roi_stats(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> Dict[str, Any]:
    """Computes statistics for a rectangular bounding box: (col_min, row_min, col_max, row_max)."""
    c0, r0, c1, r1 = bbox
    rows, cols = image.shape
    
    c_start = max(0, min(c0, c1))
    c_end = min(cols, max(c0, c1) + 1)
    r_start = max(0, min(r0, r1))
    r_end = min(rows, max(r0, r1) + 1)
    
    if c_end <= c_start or r_end <= r_start:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "peak_to_peak": 0.0,
            "sum": 0.0,
            "count": 0,
        }
    
    roi = image[r_start:r_end, c_start:c_end]
    return {
        "mean": float(np.mean(roi)),
        "std": float(np.std(roi)),
        "min": float(np.min(roi)),
        "max": float(np.max(roi)),
        "peak_to_peak": float(np.ptp(roi)),
        "sum": float(np.sum(roi)),
        "count": int(roi.size),
        "shape": roi.shape,
    }


@dataclass
class ErfFitResult:
    """Contains results of an error function (erf) knife-edge fit to a 1D cross-section."""
    success: bool
    message: str = ""
    # Physical laser beam parameters
    w: float = 0.0          # 1/e^2 beam radius (intensity drops to 1/e^2 ≈ 13.53% of peak)
    w_error: float = 0.0
    diameter: float = 0.0   # 1/e^2 beam diameter (2*w)
    diameter_error: float = 0.0
    fwhm: float = 0.0       # Full Width at Half Maximum = sqrt(2*ln(2)) * w ≈ 1.17741 * w
    fwhm_error: float = 0.0
    x0: float = 0.0         # Beam center / knife-edge inflection point
    x0_error: float = 0.0
    y0: float = 0.0         # Baseline signal level at center
    y0_error: float = 0.0
    A: float = 0.0          # Step transition amplitude (total step height = 2*|A|)
    A_error: float = 0.0
    m: float = 0.0          # Linear background drift slope
    m_error: float = 0.0
    # Goodness of fit
    r_squared: float = 0.0
    rmse: float = 0.0
    # Standard knife-edge transition widths
    d10_90: float = 0.0     # 10% to 90% edge transition width (~1.28155 * w)
    d12_88: float = 0.0     # 12% to 88% edge transition width (~1.00000 * w)
    unit: str = "V"
    # Interpolated vectors for plotting
    x_fit: Optional[np.ndarray] = None
    y_fit: Optional[np.ndarray] = None
    y_deriv: Optional[np.ndarray] = None  # Gaussian beam intensity profile d/dx erf
    x_min_fit: float = 0.0
    x_max_fit: float = 0.0


def fit_erf_beam(
    x: np.ndarray,
    y: np.ndarray,
    unit: str = "V",
    with_linear_drift: bool = True,
    x_range: Optional[Tuple[float, float]] = None
) -> ErfFitResult:
    """Fits an error function (erf) to a 1D knife-edge/step response cross-section
    to accurately determine the 1/e^2 laser beam radius (w) and diameter (2w).

    Physical model:
        f(x) = y0 + m * (x - x0) + A * erf(sqrt(2) * (x - x0) / w)

    where:
        w: 1/e^2 beam radius (intensity I(r) = I0 * exp(-2*r^2 / w^2))
        2w: 1/e^2 beam diameter
        FWHM = sqrt(2 * ln(2)) * w ≈ 1.17741 * w
        x0: step inflection point / beam center
        A: step amplitude (transition height = 2*|A|)
        y0: baseline value at center x0
        m: optional linear background drift slope
    """
    if x is None or y is None or len(x) < 4 or len(y) < 4:
        return ErfFitResult(
            success=False,
            message="At least 4 data points are required to fit.",
            unit=unit
        )

    # Convert to float numpy arrays
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    # Filter NaN / Inf
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_valid = x_arr[valid]
    y_valid = y_arr[valid]

    if len(x_valid) < 4:
        return ErfFitResult(
            success=False,
            message="Not enough valid numerical points in profile.",
            unit=unit
        )

    # Sort strictly by x ascending
    sort_idx = np.argsort(x_valid)
    x_s = x_valid[sort_idx]
    y_s = y_valid[sort_idx]

    # Optional sub-range filtering
    if x_range is not None:
        xmin, xmax = min(x_range), max(x_range)
        sub = (x_s >= xmin) & (x_s <= xmax)
        if np.sum(sub) >= 4:
            x_s = x_s[sub]
            y_s = y_s[sub]

    span_x = abs(float(x_s[-1] - x_s[0]))
    if span_x <= 1e-15:
        return ErfFitResult(
            success=False,
            message="Spatial range along cut is zero.",
            unit=unit
        )

    # Robust initial guesses using Gaussian-smoothed derivative
    sigma_filt = max(1.0, float(len(x_s)) * 0.02)
    y_smooth = ndimage.gaussian_filter1d(y_s, sigma=sigma_filt)
    dy = np.gradient(y_smooth, x_s)
    idx_max = int(np.argmax(np.abs(dy)))

    x0_g = float(x_s[idx_max])
    y0_g = float(np.mean(y_s))
    A_g = float(y_s[-1] - y_s[0]) / 2.0
    if abs(A_g) < 1e-12:
        A_g = float(np.ptp(y_s)) / 2.0 * (1.0 if dy[idx_max] >= 0 else -1.0)

    max_slope = abs(float(dy[idx_max]))
    if max_slope > 1e-12 and abs(A_g) > 1e-12:
        w_g = (2.0 * np.sqrt(2.0) / np.sqrt(np.pi)) * abs(A_g) / max_slope
    else:
        w_g = span_x / 5.0
    w_g = float(np.clip(w_g, span_x * 0.005, span_x * 0.9))

    # Curve fitting with bounds to guarantee physical validity (w > 0)
    try:
        if with_linear_drift:
            def model(x_in, y0, A, x0, w, m):
                return y0 + m * (x_in - x0) + A * erf(np.sqrt(2.0) * (x_in - x0) / np.maximum(1e-12, np.abs(w)))

            p0 = [y0_g, A_g, x0_g, w_g, 0.0]
            bounds = (
                [-np.inf, -np.inf, x_s[0] - span_x, 1e-12, -np.inf],
                [np.inf, np.inf, x_s[-1] + span_x, span_x * 5.0, np.inf]
            )
            popt, pcov = curve_fit(model, x_s, y_s, p0=p0, bounds=bounds, maxfev=10000)
            y0_fit, A_fit, x0_fit, w_fit, m_fit = popt
            w_fit = abs(float(w_fit))
            m_fit = float(m_fit)
        else:
            def model(x_in, y0, A, x0, w):
                return y0 + A * erf(np.sqrt(2.0) * (x_in - x0) / np.maximum(1e-12, np.abs(w)))

            p0 = [y0_g, A_g, x0_g, w_g]
            bounds = (
                [-np.inf, -np.inf, x_s[0] - span_x, 1e-12],
                [np.inf, np.inf, x_s[-1] + span_x, span_x * 5.0]
            )
            popt, pcov = curve_fit(model, x_s, y_s, p0=p0, bounds=bounds, maxfev=10000)
            y0_fit, A_fit, x0_fit, w_fit = popt
            w_fit = abs(float(w_fit))
            m_fit = 0.0

        perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.zeros_like(popt)
        y0_err = float(perr[0])
        A_err = float(perr[1])
        x0_err = float(perr[2])
        w_err = float(perr[3])
        m_err = float(perr[4]) if with_linear_drift else 0.0

    except Exception as e:
        return ErfFitResult(
            success=False,
            message=f"Fitting did not converge: {str(e)}",
            unit=unit
        )

    # Goodness of fit
    y_pred = model(x_s, *popt)
    residuals = y_s - y_pred
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_s - np.mean(y_s))**2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # Derived laser beam parameters
    diameter = 2.0 * w_fit
    diameter_err = 2.0 * w_err
    fwhm = np.sqrt(2.0 * np.log(2.0)) * w_fit  # ≈ 1.17741 * w
    fwhm_err = np.sqrt(2.0 * np.log(2.0)) * w_err

    # Knife-edge transition widths
    d10_90 = 2.0 * erfinv(0.8) / np.sqrt(2.0) * w_fit  # ≈ 1.28155 * w
    d12_88 = 1.0 * w_fit

    # Dense interpolated curves for display
    x_dense = np.linspace(x_s[0], x_s[-1], 500)
    y_dense = model(x_dense, *popt)

    # Beam Gaussian intensity profile: I(x) = d/dx f(x)
    arg = np.sqrt(2.0) * (x_dense - x0_fit) / w_fit
    gaussian_deriv = abs(A_fit) * (2.0 * np.sqrt(2.0) / (np.sqrt(np.pi) * w_fit)) * np.exp(-(arg**2))

    return ErfFitResult(
        success=True,
        message="Fit converged successfully",
        w=w_fit,
        w_error=w_err,
        diameter=diameter,
        diameter_error=diameter_err,
        fwhm=fwhm,
        fwhm_error=fwhm_err,
        x0=x0_fit,
        x0_error=x0_err,
        y0=float(y0_fit),
        y0_error=y0_err,
        A=float(A_fit),
        A_error=A_err,
        m=m_fit,
        m_error=m_err,
        r_squared=r_squared,
        rmse=rmse,
        d10_90=d10_90,
        d12_88=d12_88,
        unit=unit,
        x_fit=x_dense,
        y_fit=y_dense,
        y_deriv=gaussian_deriv,
        x_min_fit=float(x_s[0]),
        x_max_fit=float(x_s[-1])
    )

