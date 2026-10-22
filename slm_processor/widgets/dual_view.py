"""Dual side-by-side view widget for comparing arbitrary channels (e.g. PD vs PR)."""

import numpy as np
from typing import Optional, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from slm_processor.widgets.image_canvas import SLMImageCanvas


class SLMDualView(QWidget):
    """Side-by-side comparison view with independent channel selectors,

    independent colormaps, and synchronized zoom/pan navigation.
    """
    channel_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._sync_enabled = True

        # Left view (View 1)
        self.canvas_left = SLMImageCanvas(self, title="View 1")
        self.combo_left = QComboBox()
        self.combo_left.currentIndexChanged.connect(self._on_channel_selection_changed)
        
        self.combo_left_cmap = QComboBox()
        self.combo_left_cmap.addItems(["gray", "cividis", "bone", "Greys_r", "viridis", "inferno", "coolwarm"])
        self.combo_left_cmap.currentTextChanged.connect(self._on_left_cmap_changed)

        # Right view (View 2)
        self.canvas_right = SLMImageCanvas(self, title="View 2")
        self.combo_right = QComboBox()
        self.combo_right.currentIndexChanged.connect(self._on_channel_selection_changed)

        self.combo_right_cmap = QComboBox()
        self.combo_right_cmap.addItems(["inferno", "turbo", "plasma", "viridis", "coolwarm", "hot", "magma", "gray"])
        self.combo_right_cmap.currentTextChanged.connect(self._on_right_cmap_changed)

        # Sync checkbox
        self.chk_sync = QCheckBox("Synchronize Pan / Zoom")
        self.chk_sync.setChecked(True)
        self.chk_sync.toggled.connect(self._toggle_sync)

        # Header bar
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #252525; border-radius: 4px; padding: 2px;")
        header = QHBoxLayout(header_frame)
        header.setContentsMargins(6, 4, 6, 4)
        header.setSpacing(8)

        lbl_v1 = QLabel("View 1 (Left):")
        lbl_v1.setStyleSheet("color: #4da6ff; font-weight: bold;")
        header.addWidget(lbl_v1)
        header.addWidget(self.combo_left)
        header.addWidget(self.combo_left_cmap)

        header.addSpacing(25)

        lbl_v2 = QLabel("View 2 (Right):")
        lbl_v2.setStyleSheet("color: #ff3366; font-weight: bold;")
        header.addWidget(lbl_v2)
        header.addWidget(self.combo_right)
        header.addWidget(self.combo_right_cmap)

        header.addStretch(1)
        header.addWidget(self.chk_sync)

        # Splitter for the two canvases
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.canvas_left)
        splitter.addWidget(self.canvas_right)
        splitter.setSizes([500, 500])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(header_frame)
        layout.addWidget(splitter, 1)

        # Sync canvas axis changes via canvas limits_changed signals
        self.canvas_left.limits_changed.connect(self._on_left_limits_changed)
        self.canvas_right.limits_changed.connect(self._on_right_limits_changed)

        # Sync aspect ratio when sync is enabled
        self.canvas_left.combo_aspect.currentIndexChanged.connect(self._sync_aspect_from_left)
        self.canvas_right.combo_aspect.currentIndexChanged.connect(self._sync_aspect_from_right)

        self._updating = False

    def set_channels_list(self, channels: List[str]):
        """Populates the channel dropdowns with current channels in dataset."""
        curr_l = self.combo_left.currentText()
        curr_r = self.combo_right.currentText()

        self.combo_left.blockSignals(True)
        self.combo_right.blockSignals(True)

        self.combo_left.clear()
        self.combo_right.clear()

        self.combo_left.addItems(channels)
        self.combo_right.addItems(channels)

        # Set default selection
        if curr_l in channels:
            self.combo_left.setCurrentText(curr_l)
        else:
            for c in ["PD_R", "X_PD", "PD_X"]:
                if c in channels:
                    self.combo_left.setCurrentText(c)
                    break
            else:
                if len(channels) > 0:
                    self.combo_left.setCurrentIndex(0)

        if curr_r in channels:
            self.combo_right.setCurrentText(curr_r)
        else:
            for c in ["PR_R", "X_PR", "PR_X"]:
                if c in channels:
                    self.combo_right.setCurrentText(c)
                    break
            else:
                if len(channels) > 1:
                    self.combo_right.setCurrentIndex(1)

        self.combo_left.blockSignals(False)
        self.combo_right.blockSignals(False)

    def _on_channel_selection_changed(self):
        self.channel_changed.emit()

    def _on_left_cmap_changed(self, cmap: str):
        self.canvas_left.set_colormap(cmap)

    def _on_right_cmap_changed(self, cmap: str):
        self.canvas_right.set_colormap(cmap)

    def _toggle_sync(self, enabled: bool):
        self._sync_enabled = enabled

    def _sync_aspect_from_left(self, idx: int):
        if self._sync_enabled and not self._updating:
            self._updating = True
            if self.canvas_right.combo_aspect.currentIndex() != idx:
                self.canvas_right.combo_aspect.setCurrentIndex(idx)
            self._updating = False

    def _sync_aspect_from_right(self, idx: int):
        if self._sync_enabled and not self._updating:
            self._updating = True
            if self.canvas_left.combo_aspect.currentIndex() != idx:
                self.canvas_left.combo_aspect.setCurrentIndex(idx)
            self._updating = False

    def _on_left_limits_changed(self, xlim: Tuple[float, float], ylim: Tuple[float, float]):
        if not self._sync_enabled or self._updating:
            return
        self._updating = True
        self.canvas_right._updating_limits = True
        self.canvas_right.ax.set_xlim(xlim)
        self.canvas_right.ax.set_ylim(ylim)
        self.canvas_right.canvas.draw_idle()
        self.canvas_right._updating_limits = False
        self._updating = False

    def _on_right_limits_changed(self, xlim: Tuple[float, float], ylim: Tuple[float, float]):
        if not self._sync_enabled or self._updating:
            return
        self._updating = True
        self.canvas_left._updating_limits = True
        self.canvas_left.ax.set_xlim(xlim)
        self.canvas_left.ax.set_ylim(ylim)
        self.canvas_left.canvas.draw_idle()
        self.canvas_left._updating_limits = False
        self._updating = False
