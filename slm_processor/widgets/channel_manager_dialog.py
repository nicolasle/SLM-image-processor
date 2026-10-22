"""Interactive Channel Manager dialog for creating, pairing, and managing channels."""

from typing import List, Optional
from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QListWidget,
    QListWidgetItem, QGroupBox, QMessageBox, QTabWidget
)
from PyQt6.QtCore import pyqtSignal, Qt
from slm_processor.data_model import SLMDataset


class SLMChannelManagerDialog(QDialog):
    """Dialog allowing users to inspect active channels, create lock-in pairs,

    build ratio or difference channels, and manage derived quantities.
    """
    channels_updated = pyqtSignal()

    def __init__(self, dataset: SLMDataset, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.dataset = dataset
        self.setWindowTitle("Multi-Channel Manager & Derived Channels")
        self.resize(580, 480)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #e0e0e0; }
            QGroupBox { border: 1px solid #3c3c3c; border-radius: 5px; margin-top: 8px; font-weight: bold; color: #4da6ff; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QListWidget { background-color: #252525; border: 1px solid #3c3c3c; color: #ffffff; }
            QComboBox, QLineEdit, QPushButton { background-color: #2d2d2d; color: #ffffff; border: 1px solid #4a4a4a; border-radius: 4px; padding: 4px 8px; }
            QPushButton:hover { background-color: #3d3d3d; border-color: #007acc; }
            QPushButton#primary { background-color: #007acc; font-weight: bold; }
        """)

        self._build_ui()
        self._refresh_lists()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Upper: Current channels list
        group_current = QGroupBox("Active Channels in Dataset")
        curr_layout = QVBoxLayout(group_current)
        
        self.list_channels = QListWidget()
        curr_layout.addWidget(self.list_channels)

        row_rem = QHBoxLayout()
        self.btn_remove = QPushButton("Remove Selected Derived Channel")
        self.btn_remove.clicked.connect(self._on_remove_channel)
        row_rem.addStretch(1)
        row_rem.addWidget(self.btn_remove)
        curr_layout.addLayout(row_rem)
        layout.addWidget(group_current)

        # Lower: Tabbed creation tools
        self.tabs_create = QTabWidget()

        # Tab 1: Lock-in Pair (X & Y -> R, Phase, Rotated)
        tab_lockin = QWidget()
        lockin_layout = QGridLayout(tab_lockin)
        lockin_layout.addWidget(QLabel("Pair Name / Prefix:"), 0, 0)
        self.edit_pair_name = QLineEdit("LockIn")
        lockin_layout.addWidget(self.edit_pair_name, 0, 1)

        lockin_layout.addWidget(QLabel("In-Phase Channel (X):"), 1, 0)
        self.combo_x_chan = QComboBox()
        lockin_layout.addWidget(self.combo_x_chan, 1, 1)

        lockin_layout.addWidget(QLabel("Quadrature Channel (Y):"), 2, 0)
        self.combo_y_chan = QComboBox()
        lockin_layout.addWidget(self.combo_y_chan, 2, 1)

        self.btn_add_lockin = QPushButton("Create Lock-in Channels (R, Phase, Rotated)")
        self.btn_add_lockin.setObjectName("primary")
        self.btn_add_lockin.clicked.connect(self._on_create_lockin_pair)
        lockin_layout.addWidget(self.btn_add_lockin, 3, 0, 1, 2)
        self.tabs_create.addTab(tab_lockin, "Create Lock-in Pair (X & Y)")

        # Tab 2: Ratio Channel (A / B)
        tab_ratio = QWidget()
        ratio_layout = QGridLayout(tab_ratio)
        ratio_layout.addWidget(QLabel("Ratio Channel Name:"), 0, 0)
        self.edit_ratio_name = QLineEdit("Ratio_A_B")
        ratio_layout.addWidget(self.edit_ratio_name, 0, 1)

        ratio_layout.addWidget(QLabel("Numerator Channel (A):"), 1, 0)
        self.combo_num_chan = QComboBox()
        ratio_layout.addWidget(self.combo_num_chan, 1, 1)

        ratio_layout.addWidget(QLabel("Denominator Channel (B):"), 2, 0)
        self.combo_denom_chan = QComboBox()
        ratio_layout.addWidget(self.combo_denom_chan, 2, 1)

        self.btn_add_ratio = QPushButton("Create Ratio Channel (A / B)")
        self.btn_add_ratio.setObjectName("primary")
        self.btn_add_ratio.clicked.connect(self._on_create_ratio)
        ratio_layout.addWidget(self.btn_add_ratio, 3, 0, 1, 2)
        self.tabs_create.addTab(tab_ratio, "Create Ratio (A / B)")

        # Tab 3: Difference Channel (A - B)
        tab_diff = QWidget()
        diff_layout = QGridLayout(tab_diff)
        diff_layout.addWidget(QLabel("Difference Channel Name:"), 0, 0)
        self.edit_diff_name = QLineEdit("Diff_A_B")
        diff_layout.addWidget(self.edit_diff_name, 0, 1)

        diff_layout.addWidget(QLabel("Channel A:"), 1, 0)
        self.combo_diff_a = QComboBox()
        diff_layout.addWidget(self.combo_diff_a, 1, 1)

        diff_layout.addWidget(QLabel("Channel B:"), 2, 0)
        self.combo_diff_b = QComboBox()
        diff_layout.addWidget(self.combo_diff_b, 2, 1)

        self.btn_add_diff = QPushButton("Create Difference Channel (A - B)")
        self.btn_add_diff.setObjectName("primary")
        self.btn_add_diff.clicked.connect(self._on_create_diff)
        diff_layout.addWidget(self.btn_add_diff, 3, 0, 1, 2)
        self.tabs_create.addTab(tab_diff, "Create Difference (A - B)")

        layout.addWidget(self.tabs_create)

        # Close button
        btn_close = QPushButton("Done")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

    def _refresh_lists(self):
        if not self.dataset or len(self.dataset) == 0:
            return

        scan0 = self.dataset[0]
        all_chans = scan0.available_channels
        raw_keys = set(scan0.raw_channels.keys())

        self.list_channels.clear()
        for ch in all_chans:
            if ch in raw_keys:
                item = QListWidgetItem(f"• [Raw Image]  {ch}")
            else:
                d = scan0.derived_defs.get(ch)
                detail = f" ({d.kind}: {d.source_a} & {d.source_b})" if d else ""
                item = QListWidgetItem(f"★ [Derived]    {ch}{detail}")
            self.list_channels.addItem(item)

        # Update comboboxes
        for combo in [self.combo_x_chan, self.combo_y_chan, self.combo_num_chan, self.combo_denom_chan, self.combo_diff_a, self.combo_diff_b]:
            combo.clear()
            combo.addItems(all_chans)

    def _on_create_lockin_pair(self):
        pair = self.edit_pair_name.text().strip()
        x_ch = self.combo_x_chan.currentText()
        y_ch = self.combo_y_chan.currentText()
        if not pair:
            QMessageBox.warning(self, "Invalid Name", "Please specify a pair name prefix.")
            return

        self.dataset.add_lockin_pair_to_all(pair, x_ch, y_ch)
        self._refresh_lists()
        self.channels_updated.emit()
        QMessageBox.information(self, "Channels Created", f"Successfully created lock-in suite for '{pair}':\n• {pair}_R\n• {pair}_Theta\n• {pair}_X_Rot\n• {pair}_Y_Rot")

    def _on_create_ratio(self):
        name = self.edit_ratio_name.text().strip()
        num_ch = self.combo_num_chan.currentText()
        denom_ch = self.combo_denom_chan.currentText()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please specify a channel name.")
            return

        self.dataset.add_ratio_to_all(name, num_ch, denom_ch)
        self._refresh_lists()
        self.channels_updated.emit()

    def _on_create_diff(self):
        name = self.edit_diff_name.text().strip()
        ch_a = self.combo_diff_a.currentText()
        ch_b = self.combo_diff_b.currentText()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please specify a channel name.")
            return

        self.dataset.add_diff_to_all(name, ch_a, ch_b)
        self._refresh_lists()
        self.channels_updated.emit()

    def _on_remove_channel(self):
        curr_row = self.list_channels.currentRow()
        if curr_row < 0 or not self.dataset:
            return

        scan0 = self.dataset[0]
        all_chans = scan0.available_channels
        if curr_row < len(all_chans):
            target = all_chans[curr_row]
            if target in scan0.raw_channels:
                QMessageBox.warning(self, "Cannot Remove", f"'{target}' is a raw image file from disk and cannot be removed.")
                return
            self.dataset.remove_derived_from_all(target)
            self._refresh_lists()
            self.channels_updated.emit()
