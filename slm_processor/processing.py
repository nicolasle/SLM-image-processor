"""Image processing and leveling algorithms for Scanning Laser Microscopy images."""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional, Dict, Any


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
