"""Interactive Matplotlib 2D Image Canvas for SLM image inspection.
Supports single-channel display and PR on PD multi-layer overlay with dynamic alpha.
"""

import numpy as np
from typing import Optional, Tuple, Callable
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QCheckBox, QPushButton, QToolButton
)
from PyQt6.QtCore import pyqtSignal, Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.patches as patches


class SLMImageCanvas(QWidget):
    """Matplotlib canvas widget specialized for 2D SLM image display,

    crosshair inspection, 1D line cut selection, and ROI rectangle selection.
    Supports both single channel mode and PR on PD multi-layer overlay.
    """
    # Signals
    pixel_hovered = pyqtSignal(int, int, float, float, float)  # col, row, x_coord, y_coord, val
    overlay_hovered = pyqtSignal(int, int, float, float, float, float) # col, row, x, y, val_pr, val_pd
    line_cut_changed = pyqtSignal(float, float, float, float)  # col0, row0, col1, row1 (pixels)
    roi_changed = pyqtSignal(int, int, int, int)              # c0, r0, c1, r1 (pixels)
    limits_changed = pyqtSignal(tuple, tuple)                 # (xlim_min, xlim_max), (ylim_min, ylim_max)
    caliper_measured = pyqtSignal(float, float, float, float, float)  # x0, y0, x1, y1, length (physical)

    def __init__(self, parent: Optional[QWidget] = None, title: str = ""):
        super().__init__(parent)
        self.title_text = title

        # Data state
        self.image_data: Optional[np.ndarray] = None
        self.overlay_data: Optional[np.ndarray] = None
        self.is_overlay_mode: bool = False
        self.overlay_alpha: float = 0.6
        self.overlay_cbar_alpha: bool = False
        self.base_cmap: str = "gray"
        self.overlay_cmap: str = "inferno"

        self.x_coords: Optional[np.ndarray] = None  # columns (volts or um)
        self.y_coords: Optional[np.ndarray] = None  # rows (volts or um)
        self.unit_label: str = "µm"

        # Interactive mode: "navigate", "crosshair", "line", "roi", "caliper"
        self.interaction_mode = "crosshair"
        self._updating_limits = False
        
        # Selection states
        self.crosshair_pos: Optional[Tuple[int, int]] = None
        self.line_points: Optional[Tuple[float, float, float, float]] = None # (c0, r0, c1, r1)
        self.roi_box: Optional[Tuple[int, int, int, int]] = None             # (c0, r0, c1, r1)
        self.caliper_points: Optional[Tuple[float, float, float, float]] = None # (x0, y0, x1, y1)
        self._drag_start: Optional[Tuple[float, float]] = None
        self._is_dragging = False

        # Visual settings
        self.cmap_name = "cividis"
        self.origin = "lower"
        self.invert_y = False
        self.show_scale_bar = True
        self.clim: Optional[Tuple[float, float]] = None
        self.clim_overlay: Optional[Tuple[float, float]] = None
        self.aspect_mode: str = "physical"  # "physical", "square", "fill"
        self.base_channel_name: str = "Photodiode"
        self.overlay_channel_name: str = "Photoresponse"

        # Setup figure with compressed layout: guarantees mathematical centering with colorbar
        self.fig = Figure(figsize=(5, 5), dpi=100, layout="compressed")
        self.fig.patch.set_facecolor("#2b2b2b")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1e1e1e")
        self.ax.set_anchor('C')
        self._connect_axis_callbacks()

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # Matplotlib artists
        self.im_artist = None
        self.overlay_artist = None
        self.cbar = None
        self.crosshair_v = None
        self.crosshair_h = None
        self.line_artist = None
        self.roi_patch = None
        self.caliper_line = None
        self.caliper_text = None
        self.scalebar_bg = None
        self.scalebar_fg = None
        self.scalebar_text = None

        # Add aspect ratio selector and center button directly on toolbar
        self.toolbar.addSeparator()
        lbl_aspect = QLabel(" Aspect: ")
        lbl_aspect.setStyleSheet("color: #cccccc; font-size: 10px; font-weight: bold;")
        self.toolbar.addWidget(lbl_aspect)

        self.combo_aspect = QComboBox()
        self.combo_aspect.addItems(["Physical", "Square (1:1)", "Fill"])
        self.combo_aspect.setToolTip("Switch between Physical aspect ratio, 1:1 Square scan raster, or Fill window (always centered)")
        self.combo_aspect.setStyleSheet("font-size: 10px; padding: 1px 4px;")
        self.combo_aspect.currentIndexChanged.connect(self._on_aspect_combo_changed)
        self.toolbar.addWidget(self.combo_aspect)

        self.btn_center = QPushButton(" ⌖ Center ")
        self.btn_center.setToolTip("Fit and center image in canvas")
        self.btn_center.setStyleSheet("font-size: 10px; padding: 2px 6px; font-weight: bold;")
        self.btn_center.clicked.connect(self.reset_view)
        self.toolbar.addWidget(self.btn_center)

        self.chk_scalebar = QCheckBox(" Scale Bar ")
        self.chk_scalebar.setChecked(True)
        self.chk_scalebar.setToolTip("Show / hide calibrated physical scale bar in the bottom-right corner")
        self.chk_scalebar.setStyleSheet("color: #00e5ff; font-size: 10px; font-weight: bold;")
        self.chk_scalebar.toggled.connect(self.set_show_scale_bar)
        self.toolbar.addWidget(self.chk_scalebar)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Connect canvas events
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)

    def set_data(
        self,
        image: np.ndarray,
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
        unit: str = "V",
        title: Optional[str] = None
    ):
        """Sets single-layer image data and updates the view."""
        self.is_overlay_mode = False
        self.overlay_data = None
        self.image_data = image
        rows, cols = image.shape

        if x_coords is None or len(x_coords) != cols:
            self.x_coords = np.arange(cols, dtype=np.float32)
        else:
            self.x_coords = x_coords

        if y_coords is None or len(y_coords) != rows:
            self.y_coords = np.arange(rows, dtype=np.float32)
        else:
            self.y_coords = y_coords

        self.unit_label = unit
        if title is not None:
            self.title_text = title

        # Initialize crosshair to center if not set
        if self.crosshair_pos is None:
            self.crosshair_pos = (cols // 2, rows // 2)

        # Initialize line cut to horizontal center line if not set
        if self.line_points is None:
            self.line_points = (0.0, float(rows // 2), float(cols - 1), float(rows // 2))

        # Initialize ROI to central 50% if not set
        if self.roi_box is None:
            self.roi_box = (cols // 4, rows // 4, 3 * cols // 4, 3 * rows // 4)

        self.redraw_image()

    def set_overlay_data(
        self,
        base_image: np.ndarray,
        overlay_image: np.ndarray,
        alpha: float = 0.6,
        base_cmap: str = "gray",
        overlay_cmap: str = "inferno",
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
        unit: str = "V",
        title: Optional[str] = None,
        clim_overlay: Optional[Tuple[float, float]] = None,
        clim_base: Optional[Tuple[float, float]] = None,
        cbar_alpha: bool = False,
        base_channel_name: str = "Photodiode",
        overlay_channel_name: str = "Photoresponse"
    ):
        """Sets two-layer overlay data: base_image (PD) and overlay_image (PR)."""
        self.is_overlay_mode = True
        self.image_data = base_image
        self.overlay_data = overlay_image
        self.overlay_alpha = float(np.clip(alpha, 0.0, 1.0))
        self.overlay_cbar_alpha = bool(cbar_alpha)
        self.base_cmap = base_cmap
        self.overlay_cmap = overlay_cmap
        self.clim_overlay = clim_overlay
        if clim_base is not None:
            self.clim = clim_base
        self.base_channel_name = base_channel_name
        self.overlay_channel_name = overlay_channel_name

        rows, cols = base_image.shape
        if x_coords is None or len(x_coords) != cols:
            self.x_coords = np.arange(cols, dtype=np.float32)
        else:
            self.x_coords = x_coords

        if y_coords is None or len(y_coords) != rows:
            self.y_coords = np.arange(rows, dtype=np.float32)
        else:
            self.y_coords = y_coords

        self.unit_label = unit
        if title is not None:
            self.title_text = title

        if self.crosshair_pos is None:
            self.crosshair_pos = (cols // 2, rows // 2)

        if self.line_points is None:
            self.line_points = (0.0, float(rows // 2), float(cols - 1), float(rows // 2))

        if self.roi_box is None:
            self.roi_box = (cols // 4, rows // 4, 3 * cols // 4, 3 * rows // 4)

        self.redraw_image()

    def _compute_effective_alpha(self):
        """Returns either a scalar alpha or a 2D alpha map where cbar min is alpha=0 and cbar max is alpha=1."""
        if self.overlay_cbar_alpha and self.overlay_data is not None:
            if self.clim_overlay is not None:
                v0, v1 = self.clim_overlay
            else:
                v0 = float(np.min(self.overlay_data))
                v1 = float(np.max(self.overlay_data))

            span = float(v1 - v0)
            if abs(span) > 1e-15:
                norm = np.clip((self.overlay_data - v0) / span, 0.0, 1.0).astype(np.float32)
            else:
                norm = np.zeros_like(self.overlay_data, dtype=np.float32)

            top_alpha = max(1e-4, self.overlay_alpha)
            return norm * top_alpha

        return self.overlay_alpha

    def set_overlay_alpha(self, alpha: float):
        """Dynamically adjusts the overlay transparency without full redraw."""
        self.overlay_alpha = float(np.clip(alpha, 0.0, 1.0))
        if self.overlay_artist is not None:
            eff_alpha = self._compute_effective_alpha()
            # Reset _alpha to avoid Matplotlib ValueError when changing from array to scalar
            self.overlay_artist._alpha = None
            self.overlay_artist.set_alpha(eff_alpha)

            # Update colorbar to match visible dominant layer
            if self.cbar is not None and self.is_overlay_mode:
                if self.overlay_alpha <= 1e-4 and self.im_artist is not None:
                    self.cbar.update_normal(self.im_artist)
                    base_lbl = getattr(self, "base_channel_name", "Base")
                    self.cbar.set_label(f"Base: {base_lbl}", color="#aaaaaa", fontsize=8)
                elif self.overlay_alpha > 1e-4:
                    self.cbar.update_normal(self.overlay_artist)
                    over_lbl = getattr(self, "overlay_channel_name", "Overlay")
                    self.cbar.set_label(f"Overlay: {over_lbl}", color="#aaaaaa", fontsize=8)

            self.canvas.draw_idle()

    def set_overlay_cbar_alpha(self, enabled: bool):
        """Toggles colorbar-based alpha mapping: min value alpha=0 to max value alpha=1."""
        self.overlay_cbar_alpha = bool(enabled)
        if self.overlay_artist is not None:
            eff_alpha = self._compute_effective_alpha()
            self.overlay_artist._alpha = None
            self.overlay_artist.set_alpha(eff_alpha)
            self.canvas.draw_idle()

    # Backwards-compatible alias
    set_overlay_alpha_gradient = set_overlay_cbar_alpha

    def set_clim(self, vmin: Optional[float], vmax: Optional[float]):
        """Sets display color limits for base/active image."""
        if vmin is not None and vmax is not None and vmin < vmax:
            self.clim = (vmin, vmax)
        else:
            self.clim = None
        if self.im_artist is not None:
            if self.clim is not None:
                self.im_artist.set_clim(self.clim[0], self.clim[1])
            else:
                self.im_artist.autoscale()
            if self.cbar is not None:
                if not self.is_overlay_mode or self.overlay_alpha <= 1e-4:
                    self.cbar.update_normal(self.im_artist)
            self.canvas.draw_idle()

    def set_clim_overlay(self, vmin: Optional[float], vmax: Optional[float]):
        """Sets display color limits for overlay image layer."""
        if vmin is not None and vmax is not None and vmin < vmax:
            self.clim_overlay = (vmin, vmax)
        else:
            self.clim_overlay = None
        if self.overlay_artist is not None:
            if self.clim_overlay is not None:
                self.overlay_artist.set_clim(self.clim_overlay[0], self.clim_overlay[1])
            else:
                self.overlay_artist.autoscale()
            if self.cbar is not None and self.is_overlay_mode and self.overlay_alpha > 1e-4:
                self.cbar.update_normal(self.overlay_artist)
            if self.overlay_cbar_alpha:
                eff_alpha = self._compute_effective_alpha()
                self.overlay_artist._alpha = None
                self.overlay_artist.set_alpha(eff_alpha)
            self.canvas.draw_idle()

    def set_colormap(self, cmap_name: str):
        self.cmap_name = cmap_name
        if self.im_artist is not None:
            self.im_artist.set_cmap(cmap_name)
            if self.cbar is not None and not self.is_overlay_mode:
                self.cbar.update_normal(self.im_artist)
            self.canvas.draw_idle()

    def set_invert_y(self, invert: bool):
        """Inverts or restores Y-axis orientation (e.g. Origin Top vs. Origin Bottom)."""
        self.invert_y = bool(invert)
        if self.ax is not None and self.image_data is not None:
            self._updating_limits = True
            y0, y1 = self.ax.get_ylim()
            y_low = min(y0, y1)
            y_high = max(y0, y1)
            if self.invert_y:
                self.ax.set_ylim(y_high, y_low)
            else:
                self.ax.set_ylim(y_low, y_high)
            self._updating_limits = False
            self.limits_changed.emit(self.ax.get_xlim(), self.ax.get_ylim())
            self.canvas.draw_idle()

    def set_mode(self, mode: str):
        """Sets interaction mode: 'navigate', 'crosshair', 'line', 'roi'."""
        self.interaction_mode = mode
        if mode == "navigate":
            pass
        else:
            if self.toolbar.mode != "":
                self.toolbar.mode = ""
        self._update_overlays()
        self.canvas.draw_idle()

    def _connect_axis_callbacks(self):
        """Connects axis limit callbacks for synchronized navigation."""
        if self.ax is not None:
            self.ax.callbacks.connect("xlim_changed", self._on_axis_limits_changed)
            self.ax.callbacks.connect("ylim_changed", self._on_axis_limits_changed)

    def _on_axis_limits_changed(self, ax):
        if not self._updating_limits:
            self._update_scale_bar()
            self.limits_changed.emit(ax.get_xlim(), ax.get_ylim())

    def set_show_scale_bar(self, show: bool):
        """Enables or disables on-canvas physical scale bar."""
        self.show_scale_bar = bool(show)
        if hasattr(self, "chk_scalebar"):
            self.chk_scalebar.blockSignals(True)
            self.chk_scalebar.setChecked(self.show_scale_bar)
            self.chk_scalebar.blockSignals(False)
        self._update_scale_bar()
        self.canvas.draw_idle()

    def _on_aspect_combo_changed(self, idx: int):
        modes = ["physical", "square", "fill"]
        if 0 <= idx < len(modes):
            self.set_aspect_mode(modes[idx])

    def set_aspect_mode(self, mode: str):
        """Sets aspect ratio mode: 'physical', 'square', or 'fill'."""
        self.aspect_mode = mode
        idx_map = {"physical": 0, "square": 1, "fill": 2}
        if mode in idx_map:
            target_idx = idx_map[mode]
            if self.combo_aspect.currentIndex() != target_idx:
                self.combo_aspect.blockSignals(True)
                self.combo_aspect.setCurrentIndex(target_idx)
                self.combo_aspect.blockSignals(False)

        if self.ax is not None and self.image_data is not None:
            self.redraw_image(preserve_limits=True)

    def reset_view(self):
        """Resets axis limits to the full extent of the current image and centers it."""
        if self.x_coords is None or self.y_coords is None or self.ax is None:
            return
        x0, x1 = float(self.x_coords[0]), float(self.x_coords[-1])
        y0, y1 = float(self.y_coords[0]), float(self.y_coords[-1])
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)

        self._updating_limits = True
        self.ax.set_xlim(xmin, xmax)
        if self.invert_y:
            self.ax.set_ylim(ymax, ymin)
        else:
            self.ax.set_ylim(ymin, ymax)
        self._updating_limits = False
        self.toolbar.update()
        self.limits_changed.emit(self.ax.get_xlim(), self.ax.get_ylim())
        self.canvas.draw_idle()

    def redraw_image(self, preserve_limits: bool = False):
        """Full redraw of the image(s) and coordinates, centered mathematically."""
        if self.image_data is None or self.x_coords is None or self.y_coords is None:
            return

        rows, cols = self.image_data.shape
        x0, x1 = float(self.x_coords[0]), float(self.x_coords[-1])
        y0, y1 = float(self.y_coords[0]), float(self.y_coords[-1])
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)
        extent = [xmin, xmax, ymin, ymax]

        prev_xlim = None
        prev_ylim = None
        if preserve_limits and self.ax is not None:
            try:
                prev_xlim = self.ax.get_xlim()
                prev_ylim = self.ax.get_ylim()
            except Exception:
                pass

        self._updating_limits = True
        self.ax.clear()
        self.ax.set_facecolor("#1e1e1e")

        # Apply aspect ratio mode
        if self.aspect_mode == "square":
            self.ax.set_aspect("auto")
            self.ax.set_box_aspect(1.0)
        elif self.aspect_mode == "fill":
            self.ax.set_aspect("auto")
            self.ax.set_box_aspect(None)
        else:  # "physical"
            self.ax.set_box_aspect(None)
            self.ax.set_aspect("equal", adjustable="box")

        self.ax.set_anchor('C')

        if self.is_overlay_mode and self.overlay_data is not None:
            # Base layer (Photodiode)
            if self.clim is not None:
                v_b0, v_b1 = self.clim
            else:
                v_b0 = float(np.min(self.image_data))
                v_b1 = float(np.max(self.image_data))
            self.im_artist = self.ax.imshow(
                self.image_data,
                extent=extent,
                origin=self.origin,
                cmap=self.base_cmap,
                vmin=v_b0,
                vmax=v_b1,
                interpolation="nearest"
            )

            # Overlay layer (Photoresponse)
            if self.clim_overlay is not None:
                v_o0, v_o1 = self.clim_overlay
            else:
                v_o0 = float(np.min(self.overlay_data))
                v_o1 = float(np.max(self.overlay_data))

            eff_alpha = self._compute_effective_alpha()

            self.overlay_artist = self.ax.imshow(
                self.overlay_data,
                extent=extent,
                origin=self.origin,
                cmap=self.overlay_cmap,
                alpha=eff_alpha,
                vmin=v_o0,
                vmax=v_o1,
                interpolation="nearest"
            )
            if self.overlay_alpha <= 1e-4:
                cbar_target = self.im_artist
                cbar_label = f"Base: {getattr(self, 'base_channel_name', 'Photodiode')}"
            else:
                cbar_target = self.overlay_artist
                cbar_label = f"Overlay: {getattr(self, 'overlay_channel_name', 'Photoresponse')}"
        else:
            self.overlay_artist = None
            vmin = self.clim[0] if self.clim is not None else float(np.min(self.image_data))
            vmax = self.clim[1] if self.clim is not None else float(np.max(self.image_data))

            self.im_artist = self.ax.imshow(
                self.image_data,
                extent=extent,
                origin=self.origin,
                cmap=self.cmap_name,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest"
            )
            cbar_target = self.im_artist
            cbar_label = "Signal"

        # Styling
        self.ax.set_title(self.title_text, color="#e0e0e0", fontsize=11, fontweight="bold", pad=8)
        self.ax.set_xlabel(f"X Mirror ({self.unit_label})", color="#cccccc", fontsize=9)
        self.ax.set_ylabel(f"Y Mirror ({self.unit_label})", color="#cccccc", fontsize=9)
        self.ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#555555")

        # Colorbar: create once or update existing mappable
        if self.cbar is None:
            self.cbar = self.fig.colorbar(cbar_target, ax=self.ax, fraction=0.046, pad=0.04)
        else:
            self.cbar.update_normal(cbar_target)

        self.cbar.set_label(cbar_label, color="#aaaaaa", fontsize=8)
        self.cbar.ax.yaxis.set_tick_params(color="#aaaaaa", labelsize=8)
        for label in self.cbar.ax.yaxis.get_ticklabels():
            label.set_color("#aaaaaa")

        # Overlays & Callbacks
        self._create_overlays()
        self._connect_axis_callbacks()

        # Restore limits if requested and valid, respecting invert_y
        if prev_xlim is not None and prev_ylim is not None:
            self.ax.set_xlim(prev_xlim)
            y_low = min(prev_ylim)
            y_high = max(prev_ylim)
            if self.invert_y:
                self.ax.set_ylim(y_high, y_low)
            else:
                self.ax.set_ylim(y_low, y_high)
        else:
            if self.invert_y:
                self.ax.set_ylim(ymax, ymin)
            else:
                self.ax.set_ylim(ymin, ymax)

        if not preserve_limits:
            self.toolbar.update()

        self._updating_limits = False
        self.canvas.draw_idle()

    def _create_overlays(self):
        """Creates or recreates line, caliper, crosshair, and scale bar artists."""
        try:
            self.crosshair_v = self.ax.axvline(x=0, color="#00ffcc", linestyle="--", linewidth=1.0, alpha=0.8)
            self.crosshair_h = self.ax.axhline(y=0, color="#00ffcc", linestyle="--", linewidth=1.0, alpha=0.8)
            self.line_artist, = self.ax.plot([], [], color="#ff3366", marker="o", markersize=3, linewidth=1.5, alpha=0.9)
            self.roi_patch = patches.Rectangle((0, 0), 0, 0, linewidth=1.5, edgecolor="#ffff00", facecolor="none", linestyle="--", alpha=0.9)
            self.ax.add_patch(self.roi_patch)

            # Caliper tool artists
            self.caliper_line, = self.ax.plot([], [], color="#00e5ff", linewidth=2.0, marker="o", markersize=4, alpha=0.95)
            self.caliper_text = self.ax.text(
                0, 0, "", color="#00e5ff", fontsize=9, fontweight="bold",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#181818", edgecolor="#00e5ff", alpha=0.85)
            )

            # Calibrated physical scale bar artists
            self.scalebar_bg, = self.ax.plot([], [], color="#000000", linewidth=5.5, alpha=0.85, solid_capstyle="butt")
            self.scalebar_fg, = self.ax.plot([], [], color="#ffffff", linewidth=3.5, alpha=0.95, solid_capstyle="butt")
            self.scalebar_text = self.ax.text(
                0, 0, "", color="#ffffff", fontsize=8.5, fontweight="bold",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e1e1e", edgecolor="none", alpha=0.75)
            )

            self._update_overlays()
            self._update_scale_bar()
        except (np.linalg.LinAlgError, ValueError):
            pass

    def _pixel_to_coord(self, col: float, row: float) -> Tuple[float, float]:
        if self.x_coords is None or self.y_coords is None or self.image_data is None:
            return 0.0, 0.0
        rows, cols = self.image_data.shape
        c_idx = max(0, min(cols - 1, int(np.round(col))))
        r_idx = max(0, min(rows - 1, int(np.round(row))))
        return float(self.x_coords[c_idx]), float(self.y_coords[r_idx])

    def _coord_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        if self.x_coords is None or self.y_coords is None or self.image_data is None:
            return 0, 0
        cols = len(self.x_coords)
        rows = len(self.y_coords)
        c_idx = int(np.clip(np.argmin(np.abs(self.x_coords - x)), 0, cols - 1))
        r_idx = int(np.clip(np.argmin(np.abs(self.y_coords - y)), 0, rows - 1))
        return c_idx, r_idx

    def _update_scale_bar(self):
        """Calculates and draws calibrated scale bar in physical or voltage units."""
        if not hasattr(self, "scalebar_fg") or self.scalebar_fg is None or self.ax is None:
            return
        if not self.show_scale_bar or self.image_data is None:
            if self.scalebar_bg:
                self.scalebar_bg.set_visible(False)
            if self.scalebar_fg:
                self.scalebar_fg.set_visible(False)
            if self.scalebar_text:
                self.scalebar_text.set_visible(False)
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_min, x_max = min(xlim), max(xlim)
        y_min, y_max = min(ylim), max(ylim)
        dx_vis = abs(x_max - x_min)
        dy_vis = abs(y_max - y_min)

        if dx_vis <= 1e-9 or dy_vis <= 1e-9:
            return

        # Target scale bar size is ~20% of visible width
        target_len = dx_vis * 0.20
        exponent = np.floor(np.log10(target_len))
        fraction = target_len / (10 ** exponent)
        if fraction < 1.5:
            nice_mult = 1.0
        elif fraction < 3.5:
            nice_mult = 2.0
        elif fraction < 7.5:
            nice_mult = 5.0
        else:
            nice_mult = 10.0
        scale_len = float(nice_mult * (10 ** exponent))

        # Bottom-right placement with 5% margin
        x_end = x_max - 0.05 * dx_vis
        x_start = x_end - scale_len
        if x_start < x_min:
            x_start = x_min + 0.05 * dx_vis

        # In y: account for orientation
        if self.invert_y or ylim[0] > ylim[1]:
            # Origin top: visually bottom is high y
            y_bar = y_max - 0.05 * dy_vis
            y_txt = y_bar - 0.02 * dy_vis
            va = "bottom"
        else:
            # Origin bottom: visually bottom is low y
            y_bar = y_min + 0.05 * dy_vis
            y_txt = y_bar + 0.02 * dy_vis
            va = "bottom"

        self.scalebar_bg.set_data([x_start, x_end], [y_bar, y_bar])
        self.scalebar_fg.set_data([x_start, x_end], [y_bar, y_bar])
        self.scalebar_bg.set_visible(True)
        self.scalebar_fg.set_visible(True)

        if scale_len >= 1.0:
            lbl = f"{int(scale_len) if scale_len.is_integer() else scale_len:g} {self.unit_label}"
        else:
            lbl = f"{scale_len:g} {self.unit_label}"

        x_mid = 0.5 * (x_start + x_end)
        self.scalebar_text.set_position((x_mid, y_txt))
        self.scalebar_text.set_text(lbl)
        self.scalebar_text.set_va(va)
        self.scalebar_text.set_visible(True)

    def _update_overlays(self):
        if self.crosshair_pos and self.crosshair_v and self.crosshair_h:
            col, row = self.crosshair_pos
            x_pos, y_pos = self._pixel_to_coord(col, row)
            self.crosshair_v.set_xdata([x_pos, x_pos])
            self.crosshair_h.set_ydata([y_pos, y_pos])
            self.crosshair_v.set_visible(self.interaction_mode in ("crosshair", "navigate"))
            self.crosshair_h.set_visible(self.interaction_mode in ("crosshair", "navigate"))

        if self.line_points and self.line_artist:
            c0, r0, c1, r1 = self.line_points
            x0, y0 = self._pixel_to_coord(c0, r0)
            x1, y1 = self._pixel_to_coord(c1, r1)
            self.line_artist.set_data([x0, x1], [y0, y1])
            self.line_artist.set_visible(self.interaction_mode in ("line", "navigate"))

        if self.roi_box and self.roi_patch:
            c0, r0, c1, r1 = self.roi_box
            x0, y0 = self._pixel_to_coord(min(c0, c1), min(r0, r1))
            x1, y1 = self._pixel_to_coord(max(c0, c1), max(r0, r1))
            self.roi_patch.set_xy((min(x0, x1), min(y0, y1)))
            self.roi_patch.set_width(abs(x1 - x0))
            self.roi_patch.set_height(abs(y1 - y0))
            self.roi_patch.set_visible(self.interaction_mode in ("roi", "navigate"))

        if hasattr(self, "caliper_line") and self.caliper_line:
            is_cal = (self.interaction_mode == "caliper" and self.caliper_points is not None)
            self.caliper_line.set_visible(is_cal)
            if self.caliper_text:
                self.caliper_text.set_visible(is_cal)
            if is_cal and self.caliper_points:
                x0, y0, x1, y1 = self.caliper_points
                self.caliper_line.set_data([x0, x1], [y0, y1])
                dx = x1 - x0
                dy = y1 - y0
                dist = float(np.hypot(dx, dy))
                xm = 0.5 * (x0 + x1)
                ym = 0.5 * (y0 + y1)
                txt = f"L = {dist:.3f} {self.unit_label}\n(Δx={abs(dx):.3f}, Δy={abs(dy):.3f})"
                self.caliper_text.set_position((xm, ym))
                self.caliper_text.set_text(txt)

        self._update_scale_bar()

    def set_horizontal_cut(self, row: int):
        if self.image_data is None:
            return
        rows, cols = self.image_data.shape
        row = max(0, min(rows - 1, row))
        self.line_points = (0.0, float(row), float(cols - 1), float(row))
        self._update_overlays()
        self.canvas.draw_idle()
        self.line_cut_changed.emit(*self.line_points)

    def set_vertical_cut(self, col: int):
        if self.image_data is None:
            return
        rows, cols = self.image_data.shape
        col = max(0, min(cols - 1, col))
        self.line_points = (float(col), 0.0, float(col), float(rows - 1))
        self._update_overlays()
        self.canvas.draw_idle()
        self.line_cut_changed.emit(*self.line_points)

    def set_line_cut(self, c0: float, r0: float, c1: float, r1: float, emit_signal: bool = False):
        """Programmatically set line cut coordinates and update visuals."""
        self.line_points = (float(c0), float(r0), float(c1), float(r1))
        self._update_overlays()
        self.canvas.draw_idle()
        if emit_signal:
            self.line_cut_changed.emit(*self.line_points)

    def set_roi(self, c0: int, r0: int, c1: int, r1: int, emit_signal: bool = False):
        """Programmatically set ROI bounding box and update visuals."""
        self.roi_box = (int(c0), int(r0), int(c1), int(r1))
        self._update_overlays()
        self.canvas.draw_idle()
        if emit_signal:
            self.roi_changed.emit(*self.roi_box)

    def set_crosshair(self, col: int, row: int):
        """Programmatically set crosshair position and update visuals."""
        self.crosshair_pos = (int(col), int(row))
        self._update_overlays()
        self.canvas.draw_idle()

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax or self.image_data is None:
            return

        col, row = self._coord_to_pixel(event.xdata, event.ydata)
        rows, cols = self.image_data.shape
        if 0 <= col < cols and 0 <= row < rows:
            val = float(self.image_data[row, col])
            self.pixel_hovered.emit(col, row, event.xdata, event.ydata, val)
            if self.is_overlay_mode and self.overlay_data is not None:
                val_over = float(self.overlay_data[row, col])
                self.overlay_hovered.emit(col, row, event.xdata, event.ydata, val_over, val)

        if self._is_dragging and self._drag_start is not None:
            x_start, y_start = self._drag_start
            c_start, r_start = self._coord_to_pixel(x_start, y_start)
            c_curr, r_curr = col, row

            if self.interaction_mode == "line":
                self.line_points = (float(c_start), float(r_start), float(c_curr), float(r_curr))
                self._update_overlays()
                self.canvas.draw_idle()
            elif self.interaction_mode == "roi":
                self.roi_box = (c_start, r_start, c_curr, r_curr)
                self._update_overlays()
                self.canvas.draw_idle()
            elif self.interaction_mode == "caliper":
                self.caliper_points = (float(x_start), float(y_start), float(event.xdata), float(event.ydata))
                self._update_overlays()
                self.canvas.draw_idle()
                dx = event.xdata - x_start
                dy = event.ydata - y_start
                dist = float(np.hypot(dx, dy))
                self.caliper_measured.emit(float(x_start), float(y_start), float(event.xdata), float(event.ydata), dist)

    def _on_mouse_press(self, event):
        if event.inaxes != self.ax or self.image_data is None:
            return
        if self.toolbar.mode != "":
            return

        col, row = self._coord_to_pixel(event.xdata, event.ydata)

        if self.interaction_mode == "crosshair":
            self.crosshair_pos = (col, row)
            self._update_overlays()
            self.canvas.draw_idle()
            val = float(self.image_data[row, col])
            self.pixel_hovered.emit(col, row, event.xdata, event.ydata, val)
            if self.is_overlay_mode and self.overlay_data is not None:
                val_over = float(self.overlay_data[row, col])
                self.overlay_hovered.emit(col, row, event.xdata, event.ydata, val_over, val)
        elif self.interaction_mode in ("line", "roi", "caliper"):
            self._is_dragging = True
            self._drag_start = (event.xdata, event.ydata)
            if self.interaction_mode == "caliper":
                self.caliper_points = (float(event.xdata), float(event.ydata), float(event.xdata), float(event.ydata))
                self._update_overlays()
                self.canvas.draw_idle()

    def _on_mouse_release(self, event):
        if self._is_dragging:
            self._is_dragging = False
            if self.interaction_mode == "line" and self.line_points:
                self.line_cut_changed.emit(*self.line_points)
            elif self.interaction_mode == "roi" and self.roi_box:
                self.roi_changed.emit(*self.roi_box)
            elif self.interaction_mode == "caliper" and self.caliper_points:
                x0, y0, x1, y1 = self.caliper_points
                dx = x1 - x0
                dy = y1 - y0
                dist = float(np.hypot(dx, dy))
                self.caliper_measured.emit(float(x0), float(y0), float(x1), float(y1), dist)
