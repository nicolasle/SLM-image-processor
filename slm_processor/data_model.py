"""Flexible Multi-Channel Data Model for Scanning Laser Microscopy (SLM).

Supports dynamic discovery of arbitrary TIFF images, automatic and user-defined
lock-in pairs (X & Y -> R, Phase, Rotated X'), arithmetic channels (Ratio, Diff),
and arbitrary multi-layer overlap.
"""

import os
import re
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


def extract_clean_channel_name(filename: str) -> str:
    """Extracts standardized channel identifier (PR_X, PR_Y, PD_X, PD_Y, etc.) from filename."""
    base, _ = os.path.splitext(filename) if "." in filename else (filename, "")

    # Match X or Xs, Y or Ys as discrete tokens (e.g., Xs_P4, _Xs_, _X_, X_PD, Harm1_X)
    is_x = bool(re.search(r'(?:^|[_\W])X(?:s)?(?:[_\W]|$)', base, re.IGNORECASE))
    is_y = bool(re.search(r'(?:^|[_\W])Y(?:s)?(?:[_\W]|$)', base, re.IGNORECASE))

    # Match Photodiode (PD, Photodiode, Diode)
    is_pd = bool(re.search(r'(?:^|[_\W])(?:PD|Photodiode|Diode)(?:[_\W]|$)', base, re.IGNORECASE))

    # Match Photoresponse (PR, P4 4-point, P2 2-point, Photoresponse, RF, Harm)
    is_p4 = bool(re.search(r'(?:^|[_\W])P4(?:[_\W]|$)', base, re.IGNORECASE))
    is_p2 = bool(re.search(r'(?:^|[_\W])P2(?:[_\W]|$)', base, re.IGNORECASE))
    is_pr = bool(re.search(r'(?:^|[_\W])(?:PR|Photoresponse|RF)(?:[_\W]|$)', base, re.IGNORECASE))

    # Harm without PD indicates RF / Photoresponse
    if not is_pd and not is_pr and not is_p4 and not is_p2:
        if bool(re.search(r'(?:^|[_\W])Harm\d*(?:[_\W]|$)', base, re.IGNORECASE)):
            is_pr = True

    if is_pd:
        if is_x and not is_y:
            return "PD_X"
        elif is_y and not is_x:
            return "PD_Y"
        return "PD"
    elif is_p4 or is_pr or (is_x != is_y and not is_pd):
        # Maps 4-point or general lock-in response to standardized PR channel
        if is_x and not is_y:
            return "PR_X"
        elif is_y and not is_x:
            return "PR_Y"
        return "PR"
    elif is_p2:
        if is_x and not is_y:
            return "P2_X"
        elif is_y and not is_x:
            return "P2_Y"
        return "P2"
    else:
        if is_x and not is_y:
            return "PR_X"
        elif is_y and not is_x:
            return "PR_Y"
        parts = re.split(r'[_\W]+', base)
        return parts[-1] if parts else base


# Optical objective calibration factors from LSM.py (micrometers per Volt)
OBJECTIVE_CALIBRATION: Dict[str, Tuple[float, float]] = {
    "50x": (62.166693, 64.654161),
    "100x": (29.721644, 31.549683),
    "20x": (155.393470, 163.000598),
    "10x": (308.087566, 320.422395),
    "5x": (621.576367, 643.970523),
}


@dataclass
class DerivedChannelDef:
    """Definition of a user-defined or auto-generated derived channel."""
    name: str
    kind: str  # "lockin_r", "lockin_theta", "lockin_x_rot", "lockin_y_rot", "ratio", "diff", "linear", "expression"
    source_a: str = ""
    source_b: Optional[str] = None
    scale_a: float = 1.0
    scale_b: float = 1.0
    expression: Optional[str] = None


class SLMScan:
    """Represents a single scan step containing arbitrary image channels and coordinates."""

    def __init__(self, folder_path: str, step_name: str, step_value: float = 0.0):
        self.folder_path = os.path.abspath(folder_path)
        self.step_name = step_name
        self.step_value = step_value

        # Raw image channels in memory: {channel_name: 2D numpy float32 array}
        self.raw_channels: Dict[str, np.ndarray] = {}

        # Lazy loading paths: {channel_name: filepath}
        self.file_paths: Dict[str, str] = {}

        # Derived channel definitions: {channel_name: DerivedChannelDef}
        self.derived_defs: Dict[str, DerivedChannelDef] = {}

        # Coordinates vectors (Volts)
        self.x_volts: np.ndarray = np.zeros(0, dtype=np.float32)
        self.y_volts: np.ndarray = np.zeros(0, dtype=np.float32)

        # Active lock-in phase rotation in degrees
        self.phase_offset_deg: float = 0.0

    @property
    def shape(self) -> Tuple[int, int]:
        if self.raw_channels:
            first = next(iter(self.raw_channels.values()))
            return first.shape
        if self.file_paths:
            first_k = next(iter(self.file_paths.keys()))
            arr = self.get_channel(first_k)
            return arr.shape
        return (0, 0)

    def has_channel(self, channel_name: str) -> bool:
        """Returns True if the channel is available raw, lazy-loaded, or derived."""
        if channel_name in self.raw_channels or channel_name in self.file_paths or channel_name in self.derived_defs:
            return True
        for k in list(self.raw_channels.keys()) + list(self.file_paths.keys()):
            if k.upper() == channel_name.upper():
                return True
        alias_map = {
            "X_PR": "PR_X", "PR_X": "X_PR",
            "Y_PR": "PR_Y", "PR_Y": "Y_PR",
            "X_PD": "PD_X", "PD_X": "X_PD",
            "Y_PD": "PD_Y", "PD_Y": "Y_PD",
            "P4_X": "PR_X", "PR_X": "P4_X",
            "P4_Y": "PR_Y", "PR_Y": "P4_Y",
            "P4_R": "PR_R", "PR_R": "P4_R",
            "P2_X": "PR_X", "PR_X": "P2_X",
            "P2_Y": "PR_Y", "PR_Y": "P2_Y",
            "P2_R": "PR_R", "PR_R": "P2_R",
            "PR": "PR_R",
            "PD": "PD_R",
        }
        if channel_name in alias_map:
            alt = alias_map[channel_name]
            return alt in self.raw_channels or alt in self.file_paths
        return False

    @property
    def x_pr(self) -> np.ndarray:
        return self.get_channel("X_PR")

    @property
    def y_pr(self) -> np.ndarray:
        return self.get_channel("Y_PR")

    @property
    def x_pd(self) -> np.ndarray:
        return self.get_channel("X_PD")

    @property
    def y_pd(self) -> np.ndarray:
        return self.get_channel("Y_PD")

    @property
    def r_pr(self) -> np.ndarray:
        return self.get_channel("PR_R")

    @property
    def r_pd(self) -> np.ndarray:
        return self.get_channel("PD_R")

    @classmethod
    def load_from_folder(cls, folder_path: str, step_name: Optional[str] = None) -> "SLMScan":
        folder_path = os.path.abspath(folder_path)
        if step_name is None:
            step_name = os.path.basename(folder_path)

        try:
            step_val = float(step_name)
        except ValueError:
            match = re.search(r"[-+]?\d*\.?\d+", step_name)
            step_val = float(match.group()) if match else 0.0

        scan = cls(folder_path=folder_path, step_name=step_name, step_value=step_val)

        # 1. Discover ALL TIFF files in the folder
        all_files = os.listdir(folder_path)
        tiff_files = [f for f in all_files if f.lower().endswith((".tiff", ".tif"))]
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF image files found in {folder_path}")

        for fname in tiff_files:
            base_name, _ = os.path.splitext(fname)
            p = os.path.join(folder_path, fname)
            try:
                chan_name = extract_clean_channel_name(fname)
                with Image.open(p) as img:
                    arr = np.array(img, dtype=np.float32)
                    scan.raw_channels[chan_name] = arr
                    if base_name != chan_name:
                        scan.raw_channels[base_name] = arr
            except Exception as e:
                print(f"Warning: could not read {p}: {e}")

        if not scan.raw_channels:
            raise ValueError(f"Failed to load any valid images from {folder_path}")

        # 2. Discover coordinates from .log file
        log_files = [f for f in all_files if f.lower().endswith(".log")]
        first_img = next(iter(scan.raw_channels.values()))
        rows, cols = first_img.shape

        if log_files:
            log_path = os.path.join(folder_path, log_files[0])
            with open(log_path, "r") as f:
                lines = [l.strip().split() for l in f if l.strip()]
            if len(lines) >= 2:
                scan.x_volts = np.array([float(x) for x in lines[0]], dtype=np.float32)
                scan.y_volts = np.array([float(y) for y in lines[1]], dtype=np.float32)
            elif len(lines) == 1:
                scan.x_volts = np.array([float(x) for x in lines[0]], dtype=np.float32)
                scan.y_volts = np.linspace(-1.0, 1.0, rows, dtype=np.float32)

        if len(scan.x_volts) != cols:
            scan.x_volts = np.linspace(-1.0, 1.0, cols, dtype=np.float32)
        if len(scan.y_volts) != rows:
            scan.y_volts = np.linspace(-1.0, 1.0, rows, dtype=np.float32)

        # 3. Auto-detect lock-in pairs
        scan._auto_detect_lockin_pairs()
        return scan

    def _auto_detect_lockin_pairs(self):
        """Heuristics to pair X and Y lock-in components automatically."""
        raw_keys = list(set(self.raw_channels.keys()) | set(self.file_paths.keys()))

        # Check explicit PR and PD pairs
        if ("PR_X" in raw_keys or "X_PR" in raw_keys) and ("PR_Y" in raw_keys or "Y_PR" in raw_keys):
            x_k = "PR_X" if "PR_X" in raw_keys else "X_PR"
            y_k = "PR_Y" if "PR_Y" in raw_keys else "Y_PR"
            self.add_lockin_pair("PR", x_chan=x_k, y_chan=y_k)

        if ("PD_X" in raw_keys or "X_PD" in raw_keys) and ("PD_Y" in raw_keys or "Y_PD" in raw_keys):
            x_k = "PD_X" if "PD_X" in raw_keys else "X_PD"
            y_k = "PD_Y" if "PD_Y" in raw_keys else "Y_PD"
            self.add_lockin_pair("PD", x_chan=x_k, y_chan=y_k)

        # Pattern 1: X_<tag> and Y_<tag> (e.g. X_PD, Y_PD, X_PR, Y_PR)
        for k in raw_keys:
            if k.upper().startswith("X_"):
                tag = k[2:]
                y_key = None
                for candidate in raw_keys:
                    if candidate.upper() == f"Y_{tag.upper()}":
                        y_key = candidate
                        break
                if y_key:
                    self.add_lockin_pair(tag, x_chan=k, y_chan=y_key)

            # Pattern 2: <tag>_X and <tag>_Y (e.g. PR_X, PR_Y, PD_X, PD_Y, P4_X, P4_Y)
            elif k.upper().endswith("_X"):
                tag = k[:-2]
                y_key = None
                for candidate in raw_keys:
                    if candidate.upper() == f"{tag.upper()}_Y":
                        y_key = candidate
                        break
                if y_key:
                    self.add_lockin_pair(tag, x_chan=k, y_chan=y_key)

            # Pattern 3: Embedded Xs and Ys (e.g. 3.22K_Xs_P4 and 3.22K_Ys_P4)
            elif re.search(r'(?:^|[_\W])Xs(?:[_\W]|$)', k, re.IGNORECASE):
                expected_y = re.sub(r'([_\W]|^)Xs([_\W]|$)', r'\1Ys\2', k, flags=re.IGNORECASE)
                if expected_y in raw_keys and expected_y != k:
                    tag_match = re.search(r'Xs[_\W]+([a-zA-Z0-9]+)', k, re.IGNORECASE)
                    tag = tag_match.group(1) if tag_match else "Lockin"
                    if tag.upper() in ("P4", "PR"):
                        tag = "PR"
                    elif tag.upper() == "PD":
                        tag = "PD"
                    if f"{tag}_R" not in self.derived_defs:
                        self.add_lockin_pair(tag, x_chan=k, y_chan=expected_y)

        # Auto-create common ratio if PR and PD pairs exist
        avail = self.available_channels
        has_pr_r = "PR_R" in avail or "P4_R" in avail
        pr_r_name = "PR_R" if "PR_R" in avail else "P4_R"
        has_pd_r = "PD_R" in avail
        if has_pr_r and has_pd_r:
            self.add_ratio_channel("Norm_R (PR_R / PD_R)", pr_r_name, "PD_R")

        has_pr_x = "PR_X" in avail or "P4_X" in avail or "X_PR" in avail
        pr_x_name = "PR_X" if "PR_X" in avail else ("P4_X" if "P4_X" in avail else "X_PR")
        has_pd_x = "PD_X" in avail or "X_PD" in avail
        pd_x_name = "PD_X" if "PD_X" in avail else "X_PD"
        if has_pr_x and has_pd_x:
            self.add_ratio_channel("Norm_X (PR_X / |PD_X|)", pr_x_name, pd_x_name)

    # --- Channel Definition Methods ---

    def add_lockin_pair(self, pair_name: str, x_chan: str, y_chan: str):
        """Adds a lock-in pair and generates Amplitude, Phase, and Rotated channels."""
        self.derived_defs[f"{pair_name}_R"] = DerivedChannelDef(
            name=f"{pair_name}_R", kind="lockin_r", source_a=x_chan, source_b=y_chan
        )
        self.derived_defs[f"{pair_name}_Theta"] = DerivedChannelDef(
            name=f"{pair_name}_Theta", kind="lockin_theta", source_a=x_chan, source_b=y_chan
        )
        self.derived_defs[f"{pair_name}_X_Rot"] = DerivedChannelDef(
            name=f"{pair_name}_X_Rot", kind="lockin_x_rot", source_a=x_chan, source_b=y_chan
        )
        self.derived_defs[f"{pair_name}_Y_Rot"] = DerivedChannelDef(
            name=f"{pair_name}_Y_Rot", kind="lockin_y_rot", source_a=x_chan, source_b=y_chan
        )

    def add_ratio_channel(self, name: str, num_chan: str, denom_chan: str):
        """Adds a derived ratio channel: num_chan / denom_chan."""
        self.derived_defs[name] = DerivedChannelDef(
            name=name, kind="ratio", source_a=num_chan, source_b=denom_chan
        )

    def add_diff_channel(self, name: str, chan_a: str, chan_b: str):
        """Adds a derived difference channel: chan_a - chan_b."""
        self.derived_defs[name] = DerivedChannelDef(
            name=name, kind="diff", source_a=chan_a, source_b=chan_b
        )

    def add_expression_channel(self, name: str, expression: str):
        """Adds a derived channel computed from an arbitrary NumPy expression."""
        self.derived_defs[name] = DerivedChannelDef(
            name=name, kind="expression", source_a="", expression=expression
        )

    def evaluate_expression(self, expr: str, phase_deg: float = 0.0) -> np.ndarray:
        """Safely evaluates a NumPy mathematical expression using available channels."""
        safe_globals = {
            "__builtins__": {},
            "np": np,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "arcsin": np.arcsin,
            "arccos": np.arccos,
            "arctan": np.arctan,
            "arctan2": np.arctan2,
            "deg2rad": np.deg2rad,
            "rad2deg": np.rad2deg,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "log": np.log,
            "log10": np.log10,
            "log2": np.log2,
            "exp": np.exp,
            "maximum": np.maximum,
            "minimum": np.minimum,
            "clip": np.clip,
            "mean": np.mean,
            "median": np.median,
            "std": np.std,
            "pi": np.pi,
            "e": np.e,
        }

        local_vars = {}
        for ch in self.available_channels:
            safe_name = re.sub(r'\W+', '_', ch)
            if ch in expr or safe_name in expr:
                try:
                    arr = self.get_channel(ch, phase_deg=phase_deg)
                    local_vars[ch] = arr
                    if safe_name != ch:
                        local_vars[safe_name] = arr
                except Exception:
                    pass

        if "PR" in expr and "PR" not in local_vars:
            local_vars["PR"] = self.get_channel("PR_R", phase_deg=phase_deg)
        if "PD" in expr and "PD" not in local_vars:
            local_vars["PD"] = self.get_channel("PD_R", phase_deg=phase_deg)

        result = eval(expr, safe_globals, local_vars)
        if isinstance(result, (int, float)):
            rows, cols = self.shape
            result = np.full((rows, cols), result, dtype=np.float32)
        elif isinstance(result, np.ndarray):
            result = result.astype(np.float32)
        else:
            raise ValueError(f"Expression returned unsupported type: {type(result)}")

        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    def remove_derived_channel(self, name: str):
        """Removes a user-defined derived channel."""
        self.derived_defs.pop(name, None)

    @property
    def available_channels(self) -> List[str]:
        """Returns sorted list of all available channel names (raw + lazy + derived)."""
        keys = set(self.raw_channels.keys()) | set(self.file_paths.keys()) | set(self.derived_defs.keys())
        return sorted(list(keys))

    def get_channel(self, channel_name: str, phase_deg: float = 0.0) -> np.ndarray:
        """Retrieves or calculates any raw or derived channel by name with lazy loading."""
        eps = 1e-15

        # Check raw channels in memory
        if channel_name in self.raw_channels:
            return self.raw_channels[channel_name]

        # Check lazy-loaded file paths
        if channel_name in self.file_paths:
            p = self.file_paths[channel_name]
            try:
                with Image.open(p) as img:
                    arr = np.array(img, dtype=np.float32)
                    self.raw_channels[channel_name] = arr
                    return arr
            except Exception as e:
                print(f"Error loading {p}: {e}")

        # Check case-insensitive raw channels and file paths
        for k in self.raw_channels:
            if k.upper() == channel_name.upper():
                return self.raw_channels[k]

        for k in self.file_paths:
            if k.upper() == channel_name.upper():
                p = self.file_paths[k]
                try:
                    with Image.open(p) as img:
                        arr = np.array(img, dtype=np.float32)
                        self.raw_channels[k] = arr
                        return arr
                except Exception as e:
                    print(f"Error loading {p}: {e}")

        # Check aliases: X_PR <-> PR_X, Y_PR <-> PR_Y, X_PD <-> PD_X, Y_PD <-> PD_Y
        alias_map = {
            "X_PR": "PR_X", "PR_X": "X_PR",
            "Y_PR": "PR_Y", "PR_Y": "Y_PR",
            "X_PD": "PD_X", "PD_X": "X_PD",
            "Y_PD": "PD_Y", "PD_Y": "Y_PD",
            "P4_X": "PR_X", "PR_X": "P4_X",
            "P4_Y": "PR_Y", "PR_Y": "P4_Y",
            "P4_R": "PR_R", "PR_R": "P4_R",
            "P2_X": "PR_X", "PR_X": "P2_X",
            "P2_Y": "PR_Y", "PR_Y": "P2_Y",
            "P2_R": "PR_R", "PR_R": "P2_R",
            "PR": "PR_R",
            "PD": "PD_R",
        }
        if channel_name in alias_map:
            alt = alias_map[channel_name]
            if alt in self.raw_channels:
                return self.raw_channels[alt]
            if alt in self.file_paths:
                p = self.file_paths[alt]
                try:
                    with Image.open(p) as img:
                        arr = np.array(img, dtype=np.float32)
                        self.raw_channels[alt] = arr
                        return arr
                except Exception as e:
                    print(f"Error loading {p}: {e}")

        # Check derived channels
        if channel_name in self.derived_defs:
            d = self.derived_defs[channel_name]
            arr_a = self.get_channel(d.source_a, phase_deg)
            arr_b = self.get_channel(d.source_b, phase_deg) if d.source_b else None

            if d.kind == "lockin_r":
                return np.hypot(arr_a, arr_b)
            elif d.kind == "lockin_theta":
                return np.rad2deg(np.arctan2(arr_b, arr_a))
            elif d.kind == "lockin_x_rot":
                rad = np.deg2rad(phase_deg)
                return arr_a * np.cos(rad) + arr_b * np.sin(rad)
            elif d.kind == "lockin_y_rot":
                rad = np.deg2rad(phase_deg)
                return -arr_a * np.sin(rad) + arr_b * np.cos(rad)
            elif d.kind == "ratio":
                denom = np.where(np.abs(arr_b) > eps, np.abs(arr_b), np.nan)
                return np.nan_to_num(arr_a / denom, nan=0.0)
            elif d.kind == "diff":
                return arr_a - arr_b
            elif d.kind == "linear":
                return d.scale_a * arr_a + (d.scale_b * arr_b if arr_b is not None else 0.0)
            elif d.kind == "expression" and d.expression:
                return self.evaluate_expression(d.expression, phase_deg)

        # Fallbacks for backwards compatibility
        ch = channel_name.upper()
        if ch in ("PR_R", "OVERLAY") and self.has_channel("PR_X") and self.has_channel("PR_Y"):
            return np.hypot(self.get_channel("PR_X"), self.get_channel("PR_Y"))
        elif ch == "PD_R" and self.has_channel("PD_X") and self.has_channel("PD_Y"):
            return np.hypot(self.get_channel("PD_X"), self.get_channel("PD_Y"))

        # Fallback for missing channels: return zeros with correct shape instead of arbitrary first channel
        rows, cols = self.shape
        if rows > 0 and cols > 0:
            return np.zeros((rows, cols), dtype=np.float32)
        return np.zeros((1, 1), dtype=np.float32)

    def calculate_auto_phase(self, x_chan: Optional[str] = None, y_chan: Optional[str] = None) -> float:
        """Computes optimal lock-in phase angle that aligns signal variance into X'."""
        if x_chan is None or y_chan is None:
            # Default to PR or first detected pair
            if "X_PR" in self.raw_channels and "Y_PR" in self.raw_channels:
                x_chan, y_chan = "X_PR", "Y_PR"
            else:
                for k, d in self.derived_defs.items():
                    if d.kind == "lockin_r":
                        x_chan, y_chan = d.source_a, d.source_b
                        break

        if not x_chan or not y_chan:
            return 0.0

        x = self.get_channel(x_chan)
        y = self.get_channel(y_chan)
        s_xy = 2.0 * np.sum(x * y)
        s_diff = np.sum(x**2 - y**2)
        angle_rad = 0.5 * np.arctan2(s_xy, s_diff)

        rad1 = angle_rad
        rad2 = angle_rad + np.pi / 2.0
        p1 = np.sum((x * np.cos(rad1) + y * np.sin(rad1))**2)
        p2 = np.sum((x * np.cos(rad2) + y * np.sin(rad2))**2)
        best_rad = rad1 if p1 >= p2 else rad2

        deg = float(np.rad2deg(best_rad))
        while deg > 180.0:
            deg -= 360.0
        while deg < -180.0:
            deg += 360.0
        return deg


class SLMDataset:
    """Manages a collection of SLM scans with dynamic channel management across steps."""

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.scans: List[SLMScan] = []
        self.is_harmonic_series: bool = False
        self.detected_lia_freq: Optional[float] = None
        self.series_variable_name: str = "Step"  # "Harmonic", "Time (ms)", "Step", etc.
        self.favorite_channels: set[str] = set()
        self._load_dataset()

    def _load_dataset(self):
        # 1. Check if subdirectories contain TIFFs (Hierarchical Multi-Folder structure)
        entries = [
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]

        subdirs_with_tiffs = []
        for entry in entries:
            subfolder = os.path.join(self.root_dir, entry)
            sub_tiffs = [f for f in os.listdir(subfolder) if f.lower().endswith((".tiff", ".tif"))]
            if sub_tiffs:
                subdirs_with_tiffs.append((entry, subfolder))

        if subdirs_with_tiffs:
            self._load_hierarchical_dataset(subdirs_with_tiffs)
            return

        # 2. Check if root folder itself contains TIFFs directly
        all_root_files = os.listdir(self.root_dir)
        root_tiffs = [f for f in all_root_files if f.lower().endswith((".tiff", ".tif"))]
        if not root_tiffs:
            return

        # Check for repeating series patterns in filenames (Flat Folder Series)
        series_info = self._detect_flat_series(root_tiffs)
        if series_info:
            self._load_flat_series_dataset(series_info, all_root_files)
        else:
            # Single scan acquisition
            scan = SLMScan.load_from_folder(self.root_dir)
            self.scans.append(scan)

    def _load_hierarchical_dataset(self, subdirs_with_tiffs: List[Tuple[str, str]]):
        loaded_scans = []
        for entry, subfolder in subdirs_with_tiffs:
            try:
                scan = SLMScan.load_from_folder(subfolder, step_name=entry)
                loaded_scans.append(scan)
            except Exception as e:
                print(f"Warning: Failed to load scan from {subfolder}: {e}")

        loaded_scans.sort(key=lambda s: s.step_value)
        self.scans = loaded_scans

        # Propagate reference PD to any scans in the series that lack PD channels
        scans_with_pd = [s for s in self.scans if s.has_channel("PD_X") and s.has_channel("PD_Y")]
        if scans_with_pd and len(scans_with_pd) < len(self.scans):
            for s in self.scans:
                if not s.has_channel("PD_X") or not s.has_channel("PD_Y"):
                    nearest = min(scans_with_pd, key=lambda other: abs(other.step_value - s.step_value))
                    for ch in ["PD_X", "PD_Y"]:
                        if ch in nearest.file_paths and ch not in s.file_paths:
                            s.file_paths[ch] = nearest.file_paths[ch]
                        elif ch in nearest.raw_channels and ch not in s.raw_channels:
                            s.raw_channels[ch] = nearest.raw_channels[ch]
                    s._auto_detect_lockin_pairs()

    @staticmethod
    def _extract_controlled_parameter(stems: List[str]) -> Optional[Tuple[str, str, List[Tuple[str, float, str, str]]]]:
        """Analyzes a list of scan stems to detect the changing controlled parameter.

        Returns:
            (param_name, common_unit, [(stem, val_num, unit, extra), ...])
        """
        if not stems or len(stems) <= 1:
            return None

        # Check for explicit tokens like Harm, Step, Run, Scan first if present in all stems
        for token_pat, token_name in [
            (r'Harm(\d+)', 'Harmonic'),
            (r'Step[_-]?(\d+(?:\.\d+)?)', 'Step'),
            (r'Run[_-]?(\d+(?:\.\d+)?)', 'Run'),
            (r'Scan[_-]?(\d+(?:\.\d+)?)', 'Scan'),
        ]:
            matches = []
            for s in stems:
                m = re.search(token_pat, s, re.IGNORECASE)
                if m:
                    matches.append((s, float(m.group(1)), "", ""))
            if len(matches) == len(stems) and len(set(x[1] for x in matches)) > 1:
                return token_name, "", matches

        # Strategy 1: Common Prefix Analysis
        prefix = os.path.commonprefix(stems)
        p_clean = prefix.rstrip("_ -")
        m_param = re.search(r'([a-zA-Z0-9]+)$', p_clean)
        param_name = m_param.group(1) if m_param else "Param"

        num_pat = re.compile(r'^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-ZµμΩ%]*)(?:[_-]?(.*))?$')

        parsed = []
        units = []
        all_matched = True

        for s in stems:
            rem = s[len(prefix):].lstrip('_')
            m = num_pat.match(rem)
            if m:
                v = float(m.group(1))
                u = m.group(2) or ""
                extra = m.group(3) or ""
                parsed.append((s, v, u, extra))
                if u:
                    units.append(u)
            else:
                all_matched = False
                break

        if all_matched and len(set(p[1] for p in parsed)) > 1:
            common_u = units[0] if units else ""
            return param_name, common_u, parsed

        # Strategy 2: Token Column Analysis (delimited by _ or -)
        tokens_list = [re.split(r'[_\-]+', s) for s in stems]
        min_len = min(len(t) for t in tokens_list)
        best_col = None
        best_unique = 0
        best_parsed = None
        best_name = "Param"
        best_unit = ""

        token_num_pat = re.compile(r'^([a-zA-Z]*)([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-ZµμΩ%]*)$')

        for col in range(min_len):
            col_toks = [t[col] for t in tokens_list]
            col_parsed = []
            col_matched = True
            col_units = []
            names = []
            for i, tok in enumerate(col_toks):
                m = token_num_pat.match(tok)
                if m:
                    p_prefix = m.group(1)
                    num = float(m.group(2))
                    u = m.group(3) or ""
                    extra = "_".join(tokens_list[i][col+1:])
                    col_parsed.append((stems[i], num, u, extra))
                    if p_prefix:
                        names.append(p_prefix)
                    if u:
                        col_units.append(u)
                else:
                    col_matched = False
                    break
            if col_matched:
                unique_nums = len(set(p[1] for p in col_parsed))
                if unique_nums > best_unique:
                    best_unique = unique_nums
                    best_col = col
                    best_parsed = col_parsed
                    best_name = names[0] if names else (tokens_list[0][col-1] if col > 0 else "Param")
                    best_unit = col_units[0] if col_units else ""

        if best_parsed and best_unique > 1:
            return best_name, best_unit, best_parsed

        # Strategy 3: General physical unit search across all stems
        units_pat = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(uA|mA|nA|pA|A|V|mV|uV|nV|kV|W|mW|uW|nW|pW|kW|dBm|dB|K|mK|GHz|MHz|kHz|Hz|THz|um|nm|pm|mm|cm|m|deg|rad|ms|us|ns|s|min|h|T|mT|uT|G|kG|Oe|bar|mbar|Torr|Pa|kPa|%)(?:_|[^\w]|$)'
        unit_candidates = defaultdict(dict)
        for s in stems:
            for m in re.finditer(units_pat, s, re.IGNORECASE):
                val = float(m.group(1))
                u = m.group(2)
                unit_candidates[u.upper()][s] = val

        best_u = None
        best_cnt = 1
        for u, s_dict in unit_candidates.items():
            unique_v = len(set(s_dict.values()))
            if unique_v > best_cnt and len(s_dict) == len(stems):
                best_cnt = unique_v
                best_u = u

        if best_u is not None:
            parsed_u = []
            for s in stems:
                val = unit_candidates[best_u][s]
                parsed_u.append((s, val, best_u, ""))
            return "Param", best_u, parsed_u

        # Strategy 4: Fallback to any changing number in the stem
        num_search = re.compile(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))')
        fallback_parsed = []
        for s in stems:
            m = num_search.search(s[len(prefix):] if prefix else s)
            if not m:
                m = num_search.search(s)
            val = float(m.group(1)) if m else 0.0
            fallback_parsed.append((s, val, "", ""))

        if len(set(x[1] for x in fallback_parsed)) > 1:
            return param_name, "", fallback_parsed

        return None

    def _detect_flat_series(self, tiffs: List[str]) -> Optional[Dict[str, Any]]:
        """Automatically detects repeating acquisition series patterns in flat folders.

        Identifies repeating names appended with channel identifiers (e.g. PD, PR),
        extracts the controlled parameter that changes across scan steps, and sorts
        them numerically.
        """
        if len(tiffs) <= 1:
            return None

        # Auto-detect chopper / LIA reference frequency if encoded in filenames (e.g. fchop180Hz or fmod3kHz)
        for f in tiffs[:50]:
            m_f = re.search(r'(?:fchop|fmod)(\d+(?:\.\d+)?)\s*(kHz|Hz|MHz)?', f, re.IGNORECASE)
            if m_f:
                val = float(m_f.group(1))
                u = (m_f.group(2) or "Hz").lower()
                scale = 1000.0 if u == "khz" else (1e6 if u == "mhz" else 1.0)
                self.detected_lia_freq = val * scale
                break

        # 1. Pattern detection: Group files by channel suffix stripping
        chan_suffix_regex = re.compile(
            r'(?P<sep>[_-])'
            r'(?P<chan>'
                r'(?:(?:Xs|Ys|X|Y)[_-]?(?:PD|PR|P4|P2|Photodiode|Photoresponse|Diode|RF))'
                r'|(?:(?:PD|PR|P4|P2|Photodiode|Photoresponse|Diode|RF)[_-]?(?:Xs|Ys|X|Y)?)'
                r'|(?:(?:Xs|Ys|X|Y)\b)'
            r')'
            r'(?P<tag>[_-](?:ovld|ovl|sat|saturated))?'
            r'\.(?:tiff?|tif)$',
            re.IGNORECASE
        )

        stem_groups: Dict[str, List[str]] = defaultdict(list)
        for f in tiffs:
            m = chan_suffix_regex.search(f)
            if m:
                stem = f[:m.start()]
            else:
                base, _ = os.path.splitext(f)
                stem = base
            stem_groups[stem].append(f)

        if len(stem_groups) > 1:
            stems = list(stem_groups.keys())
            param_info = self._extract_controlled_parameter(stems)
            if param_info is not None:
                param_name, common_unit, parsed_stems = param_info

                parsed_scans = []
                for s_name, val_num, unit, extra in parsed_stems:
                    parsed_scans.append({
                        'stem': s_name,
                        'val_num': val_num,
                        'unit': unit or common_unit,
                        'extra': extra,
                        'files': stem_groups[s_name]
                    })

                parsed_scans.sort(key=lambda s: (s['val_num'], s['extra'], s['stem']))

                is_harm = bool(re.search(r'Harm', param_name, re.IGNORECASE))
                if is_harm:
                    self.is_harmonic_series = True
                    self.series_variable_name = "Harmonic"
                    if self.detected_lia_freq is None:
                        self.detected_lia_freq = 180.0
                    var_name = "Harmonic"
                else:
                    var_name = f"{param_name} ({common_unit})" if common_unit else param_name
                    self.series_variable_name = var_name

                groups_dict = defaultdict(list)
                for s in parsed_scans:
                    groups_dict[s['val_num']].extend(s['files'])

                return {
                    'var_name': var_name,
                    'param_name': param_name,
                    'unit': common_unit,
                    'scans': parsed_scans,
                    'groups': groups_dict
                }

        # 2. Legacy fallback: Standard series tokens (Harm, Step, Run, Scan)
        patterns = [
            (r'Harm(\d+)', 'Harmonic'),
            (r'Step[_-]?(\d+(?:\.\d+)?)', 'Step'),
            (r'Run[_-]?(\d+(?:\.\d+)?)', 'Run'),
            (r'Scan[_-]?(\d+(?:\.\d+)?)', 'Scan'),
        ]

        for pat, var_name in patterns:
            groups = defaultdict(list)
            for f in tiffs:
                m = re.search(pat, f, re.IGNORECASE)
                if m:
                    val_str = m.group(1)
                    try:
                        val_num = float(val_str)
                        groups[val_num].append(f)
                    except ValueError:
                        pass
            if len(groups) > 1:
                if var_name == "Harmonic":
                    self.is_harmonic_series = True
                    self.series_variable_name = "Harmonic"
                    if self.detected_lia_freq is None:
                        self.detected_lia_freq = 180.0
                else:
                    self.series_variable_name = var_name

                return {
                    'var_name': var_name,
                    'regex': pat,
                    'groups': groups
                }

        # 3. Fallback: Generic numeric index
        groups = defaultdict(list)
        for f in tiffs:
            m = re.search(r'_(\d{1,5})[_\.]', f)
            if m:
                groups[float(m.group(1))].append(f)
        if len(groups) > 1:
            return {
                'var_name': 'Index',
                'regex': r'_(\d{1,5})[_\.]',
                'groups': groups
            }

        return None

    def _load_flat_series_dataset(self, series_info: Dict[str, Any], all_files: List[str]):
        var_name = series_info['var_name']
        log_files = [f for f in all_files if f.lower().endswith(".log")]

        # Pre-cache common coordinate log if available
        default_x_volts = None
        default_y_volts = None
        if log_files:
            try:
                sample_log = os.path.join(self.root_dir, log_files[0])
                with open(sample_log, "r") as lf:
                    lines = [l.strip().split() for l in lf if l.strip()]
                if len(lines) >= 2:
                    default_x_volts = np.array([float(x) for x in lines[0]], dtype=np.float32)
                    default_y_volts = np.array([float(y) for y in lines[1]], dtype=np.float32)
            except Exception as e:
                print(f"Warning reading sample log: {e}")

        # Check if structured scans were extracted via pattern detection
        if "scans" in series_info:
            scans_data = series_info["scans"]
            param_name = series_info.get("param_name", "Step")
            common_unit = series_info.get("unit", "")

            for s_entry in scans_data:
                val_num = s_entry["val_num"]
                stem = s_entry["stem"]
                step_files = s_entry["files"]
                u = s_entry.get("unit") or common_unit
                unit_str = f" {u}" if u else ""
                extra_str = f" ({s_entry['extra']})" if s_entry.get("extra") else ""
                step_name = f"{param_name} {val_num:g}{unit_str}{extra_str}"

                scan = SLMScan(folder_path=self.root_dir, step_name=step_name, step_value=val_num)

                # Prioritize non-overloaded channels for standard channel keys
                sorted_files = sorted(step_files, key=lambda f: 1 if bool(re.search(r'(?:ovld|ovl)', f, re.IGNORECASE)) else 0)
                for f in sorted_files:
                    chan_name = extract_clean_channel_name(f)
                    is_ovld = bool(re.search(r'(?:ovld|ovl)', f, re.IGNORECASE))
                    if chan_name not in scan.file_paths or not is_ovld:
                        scan.file_paths[chan_name] = os.path.join(self.root_dir, f)
                    if is_ovld:
                        scan.file_paths[f"{chan_name}_ovld"] = os.path.join(self.root_dir, f)

                # Match per-step log file from step files or stem
                matched_log = None
                for f in step_files:
                    cand_log = os.path.join(self.root_dir, os.path.splitext(f)[0] + ".log")
                    if os.path.isfile(cand_log):
                        matched_log = cand_log
                        break
                if not matched_log:
                    for lf in log_files:
                        if stem.lower() in lf.lower():
                            matched_log = os.path.join(self.root_dir, lf)
                            break

                if matched_log:
                    try:
                        with open(matched_log, "r") as lf:
                            lines = [l.strip().split() for l in lf if l.strip()]
                        if len(lines) >= 2:
                            scan.x_volts = np.array([float(x) for x in lines[0]], dtype=np.float32)
                            scan.y_volts = np.array([float(y) for y in lines[1]], dtype=np.float32)
                    except Exception:
                        pass

                if len(scan.x_volts) == 0 and default_x_volts is not None:
                    scan.x_volts = default_x_volts.copy()
                    scan.y_volts = default_y_volts.copy()

                scan._auto_detect_lockin_pairs()
                self.scans.append(scan)
        else:
            # Fallback legacy groups loading
            groups = series_info['groups']
            for val_num in sorted(groups.keys()):
                step_files = groups[val_num]
                step_name = f"{var_name} {int(val_num) if val_num.is_integer() else val_num}"
                scan = SLMScan(folder_path=self.root_dir, step_name=step_name, step_value=val_num)

                for f in step_files:
                    chan_name = extract_clean_channel_name(f)
                    scan.file_paths[chan_name] = os.path.join(self.root_dir, f)

                matched_log = None
                for f in step_files:
                    cand_log = os.path.join(self.root_dir, os.path.splitext(f)[0] + ".log")
                    if os.path.isfile(cand_log):
                        matched_log = cand_log
                        break

                if matched_log:
                    try:
                        with open(matched_log, "r") as lf:
                            lines = [l.strip().split() for l in lf if l.strip()]
                        if len(lines) >= 2:
                            scan.x_volts = np.array([float(x) for x in lines[0]], dtype=np.float32)
                            scan.y_volts = np.array([float(y) for y in lines[1]], dtype=np.float32)
                    except Exception:
                        pass

                if len(scan.x_volts) == 0 and default_x_volts is not None:
                    scan.x_volts = default_x_volts.copy()
                    scan.y_volts = default_y_volts.copy()

                scan._auto_detect_lockin_pairs()
                self.scans.append(scan)

        # Propagate reference PD to any scans in the series that lack PD channels
        scans_with_pd = [s for s in self.scans if s.has_channel("PD_X") and s.has_channel("PD_Y")]
        if scans_with_pd and len(scans_with_pd) < len(self.scans):
            for s in self.scans:
                if not s.has_channel("PD_X") or not s.has_channel("PD_Y"):
                    nearest = min(scans_with_pd, key=lambda other: abs(other.step_value - s.step_value))
                    for ch in ["PD_X", "PD_Y"]:
                        if ch in nearest.file_paths and ch not in s.file_paths:
                            s.file_paths[ch] = nearest.file_paths[ch]
                        elif ch in nearest.raw_channels and ch not in s.raw_channels:
                            s.raw_channels[ch] = nearest.raw_channels[ch]
                    s._auto_detect_lockin_pairs()

    def convert_harmonics_to_timedomain(
        self,
        lia_freq_hz: float = 180.0,
        phasor_convention: str = "process_30K",
        n_time_points: Optional[int] = None,
        time_shift_ms: float = 0.0
    ) -> "SLMDataset":
        """Executes Inverse Hermitian Fourier Transform on harmonic series to reconstruct time domain."""
        if not self.scans:
            raise ValueError("No scans in dataset.")

        # Map available harmonics
        harm_map: Dict[int, SLMScan] = {}
        for s in self.scans:
            try:
                h = int(np.round(s.step_value))
                if h > 0:
                    harm_map[h] = s
            except Exception:
                pass

        if not harm_map:
            raise ValueError("No positive integer harmonic steps detected.")

        max_harm = max(harm_map.keys())
        first_scan = next(iter(harm_map.values()))
        rows, cols = first_scan.shape

        has_pr = any(s.has_channel("PR_X") or s.has_channel("X_PR") for s in harm_map.values())
        has_pd = any(s.has_channel("PD_X") or s.has_channel("X_PD") for s in harm_map.values())

        if not has_pr and not has_pd:
            raise ValueError("No PR or PD lock-in channels found to transform.")

        N_time = n_time_points if n_time_points is not None and n_time_points > 0 else 2 * max_harm

        # 1. Transform Photoresponse (PR)
        ifft_pr = None
        if has_pr:
            Xs_pr = np.zeros((max_harm + 1, rows, cols), dtype=np.float32)
            Ys_pr = np.zeros((max_harm + 1, rows, cols), dtype=np.float32)
            for h, scan in harm_map.items():
                if scan.has_channel("PR_X"):
                    Xs_pr[h] = scan.get_channel("PR_X")
                elif scan.has_channel("X_PR"):
                    Xs_pr[h] = scan.get_channel("X_PR")

                if scan.has_channel("PR_Y"):
                    Ys_pr[h] = scan.get_channel("PR_Y")
                elif scan.has_channel("Y_PR"):
                    Ys_pr[h] = scan.get_channel("Y_PR")

            if phasor_convention == "direct":
                f_pr = Xs_pr + 1j * Ys_pr
            elif phasor_convention == "negative_y":
                f_pr = -Ys_pr + 1j * Xs_pr
            else:  # "process_30K" standard: -(-Y + iX) = Y - iX
                f_pr = Ys_pr - 1j * Xs_pr

            ifft_pr = np.fft.irfft(f_pr, axis=0, n=N_time)

        # 2. Transform Photodiode (PD)
        ifft_pd = None
        if has_pd:
            Xs_pd = np.zeros((max_harm + 1, rows, cols), dtype=np.float32)
            Ys_pd = np.zeros((max_harm + 1, rows, cols), dtype=np.float32)
            for h, scan in harm_map.items():
                if scan.has_channel("PD_X"):
                    Xs_pd[h] = scan.get_channel("PD_X")
                elif scan.has_channel("X_PD"):
                    Xs_pd[h] = scan.get_channel("X_PD")

                if scan.has_channel("PD_Y"):
                    Ys_pd[h] = scan.get_channel("PD_Y")
                elif scan.has_channel("Y_PD"):
                    Ys_pd[h] = scan.get_channel("Y_PD")

            if phasor_convention == "direct":
                f_pd = Xs_pd + 1j * Ys_pd
            else:
                f_pd = -Ys_pd + 1j * Xs_pd

            ifft_pd = np.fft.irfft(f_pd, axis=0, n=N_time)

        # 3. Construct time grid
        T_period_s = 1.0 / lia_freq_hz
        T_period_ms = T_period_s * 1000.0
        time_vals_ms = np.linspace(0.0, T_period_ms, N_time, endpoint=False)

        # Optional time shift / roll
        if abs(time_shift_ms) > 1e-9:
            shift_idx = int(np.round((time_shift_ms / T_period_ms) * N_time))
            if ifft_pr is not None:
                ifft_pr = np.roll(ifft_pr, shift_idx, axis=0)
            if ifft_pd is not None:
                ifft_pd = np.roll(ifft_pd, shift_idx, axis=0)

        # 4. Construct time-domain dataset
        time_ds = SLMDataset.__new__(SLMDataset)
        time_ds.root_dir = self.root_dir
        time_ds.is_harmonic_series = False
        time_ds.series_variable_name = "Time (ms)"
        time_ds.detected_lia_freq = lia_freq_hz
        time_ds.scans = []

        x_coords = first_scan.x_volts.copy()
        y_coords = first_scan.y_volts.copy()

        for t_idx in range(N_time):
            t_ms = float(time_vals_ms[t_idx])
            scan_t = SLMScan(folder_path=self.root_dir, step_name=f"{t_ms:.3f} ms", step_value=t_ms)
            scan_t.x_volts = x_coords.copy()
            scan_t.y_volts = y_coords.copy()

            if ifft_pr is not None:
                scan_t.raw_channels["PR"] = ifft_pr[t_idx].astype(np.float32)
            if ifft_pd is not None:
                scan_t.raw_channels["PD"] = ifft_pd[t_idx].astype(np.float32)

            if ifft_pr is not None and ifft_pd is not None:
                scan_t.add_ratio_channel("Norm (PR / PD)", "PR", "PD")

            time_ds.scans.append(scan_t)

        return time_ds

    def __len__(self) -> int:
        return len(self.scans)

    def __getitem__(self, idx: int) -> SLMScan:
        return self.scans[idx]

    @property
    def step_names(self) -> List[str]:
        return [s.step_name for s in self.scans]

    @property
    def step_values(self) -> List[float]:
        return [s.step_value for s in self.scans]

    @property
    def common_channels(self) -> List[str]:
        if not self.scans:
            return []
        return self.scans[0].available_channels

    def add_lockin_pair_to_all(self, pair_name: str, x_chan: str, y_chan: str):
        for scan in self.scans:
            scan.add_lockin_pair(pair_name, x_chan, y_chan)

    def add_ratio_to_all(self, name: str, num_chan: str, denom_chan: str):
        for scan in self.scans:
            scan.add_ratio_channel(name, num_chan, denom_chan)

    def add_diff_to_all(self, name: str, chan_a: str, chan_b: str):
        for scan in self.scans:
            scan.add_diff_channel(name, chan_a, chan_b)

    def add_expression_to_all(self, name: str, expression: str):
        for scan in self.scans:
            scan.add_expression_channel(name, expression)

    def toggle_favorite_channel(self, channel_name: str) -> bool:
        if channel_name in self.favorite_channels:
            self.favorite_channels.remove(channel_name)
            return False
        else:
            self.favorite_channels.add(channel_name)
            return True

    def remove_derived_from_all(self, name: str):
        for scan in self.scans:
            scan.remove_derived_channel(name)
