"""Interactive Channel Manager dialog for organizing, creating, and managing channels in subfolders."""

from typing import Dict, List, Optional, Any
import os
from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QTreeWidget,
    QTreeWidgetItem, QGroupBox, QMessageBox, QTabWidget,
    QHeaderView, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor, QFont
from slm_processor.data_model import SLMDataset


class SLMChannelManagerDialog(QDialog):
    """Dialog allowing users to inspect active channels sorted in subfolders
    (for each dataset, separated into Raw and Derived channels), create lock-in pairs,
    build ratio or difference channels, and manage derived quantities.
    """
    channels_updated = pyqtSignal()

    def __init__(
        self,
        dataset: SLMDataset,
        parent: Optional[QWidget] = None,
        all_datasets: Optional[Dict[str, SLMDataset]] = None
    ):
        super().__init__(parent)
        self.dataset = dataset
        self.all_datasets: Dict[str, SLMDataset] = all_datasets or {}

        # If all_datasets is not provided, populate from the primary dataset
        if not self.all_datasets and self.dataset:
            ds_name = os.path.basename(self.dataset.root_dir) if self.dataset.root_dir else "Active Dataset"
            if getattr(self.dataset, "is_harmonic_series", False):
                ds_name += f" (Harmonic Series, {len(self.dataset)} harmonics)"
            elif getattr(self.dataset, "series_variable_name", "") == "Time (ms)":
                ds_name += f" (Time Domain, {len(self.dataset)} slices)"
            else:
                ds_name += f" ({len(self.dataset)} scans)"
            self.all_datasets[ds_name] = self.dataset

        self.setWindowTitle("Multi-Channel Manager & Hierarchy")
        self.resize(700, 600)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #e0e0e0; }
            QGroupBox { border: 1px solid #3c3c3c; border-radius: 5px; margin-top: 8px; font-weight: bold; color: #4da6ff; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QTreeWidget { background-color: #242424; border: 1px solid #3c3c3c; color: #ffffff; font-size: 11px; }
            QTreeWidget::item { padding: 3px 0; }
            QTreeWidget::item:selected { background-color: #0d47a1; color: #ffffff; }
            QHeaderView::section { background-color: #2e2e2e; color: #00e5ff; font-weight: bold; padding: 4px; border: 1px solid #3a3a3a; font-size: 11px; }
            QComboBox, QLineEdit, QPushButton { background-color: #2d2d2d; color: #ffffff; border: 1px solid #4a4a4a; border-radius: 4px; padding: 4px 8px; }
            QPushButton:hover { background-color: #3d3d3d; border-color: #007acc; }
            QPushButton#primary { background-color: #007acc; font-weight: bold; }
            QTabWidget::pane { border: 1px solid #3c3c3c; background-color: #222222; }
            QTabBar::tab { background: #2b2b2b; color: #cccccc; padding: 6px 12px; margin-right: 2px; }
            QTabBar::tab:selected { background: #007acc; color: #ffffff; font-weight: bold; }
        """)

        self._build_ui()
        self._refresh_lists()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Upper: Channels Tree Group
        group_tree = QGroupBox("Channel Hierarchy (Sorted by Dataset, Raw & Derived Subfolders)")
        tree_layout = QVBoxLayout(group_tree)
        tree_layout.setContentsMargins(8, 8, 8, 8)
        tree_layout.setSpacing(6)

        # Toolbar above tree: Search Filter + Expand/Collapse
        toolbar = QHBoxLayout()
        self.edit_filter = QLineEdit()
        self.edit_filter.setPlaceholderText("🔍 Filter channels by name...")
        self.edit_filter.textChanged.connect(self._filter_tree)
        toolbar.addWidget(self.edit_filter, 1)

        self.btn_expand_all = QPushButton("▼ Expand All")
        self.btn_expand_all.setStyleSheet("font-size: 10px; padding: 3px 8px;")
        self.btn_expand_all.clicked.connect(self.tree_channels_expand_all)
        toolbar.addWidget(self.btn_expand_all)

        self.btn_collapse_all = QPushButton("▲ Collapse All")
        self.btn_collapse_all.setStyleSheet("font-size: 10px; padding: 3px 8px;")
        self.btn_collapse_all.clicked.connect(self.tree_channels_collapse_all)
        toolbar.addWidget(self.btn_collapse_all)

        tree_layout.addLayout(toolbar)

        # Hierarchical Tree
        self.tree_channels = QTreeWidget()
        self.tree_channels.setColumnCount(3)
        self.tree_channels.setHeaderLabels(["Channel / Subfolder", "Type", "Details / Source Formula"])
        self.tree_channels.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree_channels.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree_channels.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tree_channels.itemSelectionChanged.connect(self._on_tree_selection_changed)
        tree_layout.addWidget(self.tree_channels, 1)

        # Bottom row of tree group: Status and Delete action
        row_rem = QHBoxLayout()
        self.lbl_channel_count = QLabel("Total: -")
        self.lbl_channel_count.setStyleSheet("color: #888888; font-size: 10px;")
        row_rem.addWidget(self.lbl_channel_count)
        row_rem.addStretch(1)

        self.btn_remove = QPushButton("Remove Selected Derived Channel")
        self.btn_remove.setStyleSheet("background-color: #8a1f1f; color: #ffffff; padding: 4px 12px;")
        self.btn_remove.setEnabled(False)
        self.btn_remove.clicked.connect(self._on_remove_channel)
        row_rem.addWidget(self.btn_remove)

        tree_layout.addLayout(row_rem)
        layout.addWidget(group_tree, 3)

        # Lower: Tabbed creation tools
        self.tabs_create = QTabWidget()

        # Optional Target Dataset selector if multiple datasets
        self.combo_target_ds = QComboBox()
        if len(self.all_datasets) > 1:
            ds_select_frame = QFrame()
            ds_sel_layout = QHBoxLayout(ds_select_frame)
            ds_sel_layout.setContentsMargins(4, 2, 4, 2)
            ds_sel_layout.addWidget(QLabel("Target Dataset for New Channels:"))
            self.combo_target_ds.addItem("All Active Datasets", "all")
            for name, ds in self.all_datasets.items():
                self.combo_target_ds.addItem(name, ds)
            ds_sel_layout.addWidget(self.combo_target_ds, 1)
            layout.addWidget(ds_select_frame)

        # Tab 1: Lock-in Pair (X & Y -> R, Phase, Rotated)
        tab_lockin = QWidget()
        lockin_layout = QGridLayout(tab_lockin)
        lockin_layout.setContentsMargins(8, 8, 8, 8)
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
        ratio_layout.setContentsMargins(8, 8, 8, 8)
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
        diff_layout.setContentsMargins(8, 8, 8, 8)
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

        layout.addWidget(self.tabs_create, 2)

        # Done button
        btn_close = QPushButton("Done")
        btn_close.setStyleSheet("padding: 6px 14px; font-weight: bold;")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

    def tree_channels_expand_all(self):
        self.tree_channels.expandAll()

    def tree_channels_collapse_all(self):
        self.tree_channels.collapseAll()

    @property
    def list_channels(self):
        """Backward compatibility adapter for list_channels.count()."""
        dialog = self
        class _ListChannelsCompat:
            def count(self_inner):
                total = 0
                def count_leaves(item):
                    nonlocal total
                    data = item.data(0, Qt.ItemDataRole.UserRole)
                    if data and data.get("is_channel"):
                        total += 1
                    for i in range(item.childCount()):
                        count_leaves(item.child(i))
                for i in range(dialog.tree_channels.topLevelItemCount()):
                    count_leaves(dialog.tree_channels.topLevelItem(i))
                return total
        return _ListChannelsCompat()

    def _refresh_lists(self):
        """Rebuilds the hierarchical tree widget with subfolders for each dataset (Raw and Derived)."""
        self.tree_channels.clear()

        total_raw = 0
        total_derived = 0

        # Build tree for each dataset
        for ds_name, ds in self.all_datasets.items():
            if not ds or len(ds) == 0:
                continue

            scan0 = ds[0]
            raw_keys = list(set(scan0.raw_channels.keys()) | set(scan0.file_paths.keys()))
            derived_defs = scan0.derived_defs

            total_raw += len(raw_keys)
            total_derived += len(derived_defs)

            # 1. Dataset Root Item
            ds_item = QTreeWidgetItem(self.tree_channels)
            ds_item.setText(0, f"🗂 {ds_name}")
            ds_item.setText(1, "Dataset")
            ds_item.setText(2, f"{len(ds)} scans | Root: {ds.root_dir or 'Memory'}")
            ds_item.setForeground(0, QColor("#00e5ff"))
            font_ds = QFont()
            font_ds.setBold(True)
            ds_item.setFont(0, font_ds)
            ds_item.setData(0, Qt.ItemDataRole.UserRole, {"is_folder": True, "dataset": ds, "name": ds_name})

            # 2. Raw Channels Subfolder
            raw_folder = QTreeWidgetItem(ds_item)
            raw_folder.setText(0, f"📁 Raw Channels ({len(raw_keys)})")
            raw_folder.setText(1, "Folder")
            raw_folder.setText(2, "Direct hardware scan images on disk")
            raw_folder.setForeground(0, QColor("#ffd54f"))
            font_folder = QFont()
            font_folder.setBold(True)
            raw_folder.setFont(0, font_folder)
            raw_folder.setData(0, Qt.ItemDataRole.UserRole, {"is_folder": True, "dataset": ds})

            shape_str = f"{scan0.shape[0]}×{scan0.shape[1]}" if hasattr(scan0, "shape") else "2D"
            for raw_ch in sorted(raw_keys):
                item = QTreeWidgetItem(raw_folder)
                item.setText(0, f"  🖼 {raw_ch}")
                item.setText(1, "Raw Image")
                item.setText(2, f"Shape: {shape_str} | Original TIFF File")
                item.setForeground(0, QColor("#ffffff"))
                item.setData(0, Qt.ItemDataRole.UserRole, {
                    "is_channel": True,
                    "is_raw": True,
                    "name": raw_ch,
                    "dataset": ds
                })

            # 3. Derived Channels Subfolder
            derived_folder = QTreeWidgetItem(ds_item)
            derived_folder.setText(0, f"📁 Derived Channels ({len(derived_defs)})")
            derived_folder.setText(1, "Folder")
            derived_folder.setText(2, "Computed signals, lock-in polar components & operations")
            derived_folder.setForeground(0, QColor("#ffd54f"))
            derived_folder.setFont(0, font_folder)
            derived_folder.setData(0, Qt.ItemDataRole.UserRole, {"is_folder": True, "dataset": ds})

            # Organize Derived channels into subcategories: Lock-in, Ratios, Differences, Other
            lockin_items: Dict[str, List[Tuple[str, Any]]] = {}
            ratio_items = []
            diff_items = []
            other_items = []

            for ch_name, d in derived_defs.items():
                if d.kind in ("lockin_r", "lockin_theta", "lockin_xrot", "lockin_yrot"):
                    # Find prefix (e.g. "PR" or "PD" or custom)
                    prefix = ch_name.split("_")[0]
                    if prefix not in lockin_items:
                        lockin_items[prefix] = []
                    lockin_items[prefix].append((ch_name, d))
                elif d.kind == "ratio":
                    ratio_items.append((ch_name, d))
                elif d.kind == "diff":
                    diff_items.append((ch_name, d))
                else:
                    other_items.append((ch_name, d))

            # A. Lock-in Pairs Subfolder
            if lockin_items:
                lockin_folder = QTreeWidgetItem(derived_folder)
                total_lockin_chans = sum(len(v) for v in lockin_items.values())
                lockin_folder.setText(0, f"📁 Lock-in Pairs ({total_lockin_chans})")
                lockin_folder.setText(1, "Category")
                lockin_folder.setText(2, "Polar amplitude (R), phase (θ), and rotated components")
                lockin_folder.setForeground(0, QColor("#e0e0e0"))
                lockin_folder.setFont(0, font_folder)

                for prefix, pair_chans in sorted(lockin_items.items()):
                    # Pair subfolder
                    pair_folder = QTreeWidgetItem(lockin_folder)
                    pair_folder.setText(0, f"📁 Pair '{prefix}'")
                    pair_folder.setText(1, "Lock-in Suite")
                    src_str = f"X: {pair_chans[0][1].source_a}, Y: {pair_chans[0][1].source_b}" if pair_chans else ""
                    pair_folder.setText(2, src_str)
                    pair_folder.setForeground(0, QColor("#81d4fa"))

                    for ch_name, d in pair_chans:
                        item = QTreeWidgetItem(pair_folder)
                        kind_label = {
                            "lockin_r": "Amplitude (R)",
                            "lockin_theta": "Phase (θ)",
                            "lockin_xrot": "Rotated In-Phase (X')",
                            "lockin_yrot": "Rotated Quadrature (Y')"
                        }.get(d.kind, d.kind)
                        formula_label = {
                            "lockin_r": f"√({d.source_a}² + {d.source_b}²)",
                            "lockin_theta": f"atan2({d.source_b}, {d.source_a})",
                            "lockin_xrot": f"{d.source_a}·cos(φ) + {d.source_b}·sin(φ)",
                            "lockin_yrot": f"-{d.source_a}·sin(φ) + {d.source_b}·cos(φ)"
                        }.get(d.kind, f"{d.source_a} & {d.source_b}")

                        item.setText(0, f"  ⚡ {ch_name}")
                        item.setText(1, kind_label)
                        item.setText(2, formula_label)
                        item.setForeground(0, QColor("#4fc3f7"))
                        item.setData(0, Qt.ItemDataRole.UserRole, {
                            "is_channel": True,
                            "is_raw": False,
                            "name": ch_name,
                            "dataset": ds
                        })
                    pair_folder.setExpanded(True)

                lockin_folder.setExpanded(True)

            # B. Ratio Channels Subfolder
            if ratio_items:
                ratio_folder = QTreeWidgetItem(derived_folder)
                ratio_folder.setText(0, f"📁 Ratios ({len(ratio_items)})")
                ratio_folder.setText(1, "Category")
                ratio_folder.setText(2, "Normalized ratio channels (A / B)")
                ratio_folder.setForeground(0, QColor("#e0e0e0"))
                ratio_folder.setFont(0, font_folder)

                for ch_name, d in sorted(ratio_items, key=lambda x: x[0]):
                    item = QTreeWidgetItem(ratio_folder)
                    item.setText(0, f"  ➗ {ch_name}")
                    item.setText(1, "Ratio")
                    item.setText(2, f"{d.source_a} / {d.source_b}")
                    item.setForeground(0, QColor("#b388ff"))
                    item.setData(0, Qt.ItemDataRole.UserRole, {
                        "is_channel": True,
                        "is_raw": False,
                        "name": ch_name,
                        "dataset": ds
                    })
                ratio_folder.setExpanded(True)

            # C. Difference Channels Subfolder
            if diff_items:
                diff_folder = QTreeWidgetItem(derived_folder)
                diff_folder.setText(0, f"📁 Differences ({len(diff_items)})")
                diff_folder.setText(1, "Category")
                diff_folder.setText(2, "Differential background subtraction (A - B)")
                diff_folder.setForeground(0, QColor("#e0e0e0"))
                diff_folder.setFont(0, font_folder)

                for ch_name, d in sorted(diff_items, key=lambda x: x[0]):
                    item = QTreeWidgetItem(diff_folder)
                    item.setText(0, f"  ➖ {ch_name}")
                    item.setText(1, "Difference")
                    item.setText(2, f"{d.source_a} - {d.source_b}")
                    item.setForeground(0, QColor("#ff80ab"))
                    item.setData(0, Qt.ItemDataRole.UserRole, {
                        "is_channel": True,
                        "is_raw": False,
                        "name": ch_name,
                        "dataset": ds
                    })
                diff_folder.setExpanded(True)

            # D. Other Derived Subfolder
            if other_items:
                other_folder = QTreeWidgetItem(derived_folder)
                other_folder.setText(0, f"📁 Custom / Other ({len(other_items)})")
                other_folder.setText(1, "Category")
                other_folder.setText(2, "Custom derived quantities")
                other_folder.setForeground(0, QColor("#e0e0e0"))
                other_folder.setFont(0, font_folder)

                for ch_name, d in other_items:
                    item = QTreeWidgetItem(other_folder)
                    item.setText(0, f"  ★ {ch_name}")
                    item.setText(1, d.kind)
                    item.setText(2, f"{d.source_a} & {d.source_b}")
                    item.setForeground(0, QColor("#80cbc4"))
                    item.setData(0, Qt.ItemDataRole.UserRole, {
                        "is_channel": True,
                        "is_raw": False,
                        "name": ch_name,
                        "dataset": ds
                    })
                other_folder.setExpanded(True)

            ds_item.setExpanded(True)
            raw_folder.setExpanded(True)
            derived_folder.setExpanded(True)

        self.lbl_channel_count.setText(
            f"Active Channels: {total_raw + total_derived} total ({total_raw} raw on disk, {total_derived} derived)"
        )

        # Update creation comboboxes with channels from active dataset
        active_scan = self.dataset[0] if (self.dataset and len(self.dataset) > 0) else None
        all_chans = active_scan.available_channels if active_scan else []

        for combo in [
            self.combo_x_chan, self.combo_y_chan,
            self.combo_num_chan, self.combo_denom_chan,
            self.combo_diff_a, self.combo_diff_b
        ]:
            curr = combo.currentText()
            combo.clear()
            combo.addItems(all_chans)
            if curr in all_chans:
                combo.setCurrentText(curr)

    def _filter_tree(self, text: str):
        """Filters tree items in real-time matching the search string."""
        search = text.strip().lower()

        def match_and_filter(item: QTreeWidgetItem) -> bool:
            # Check if this item matches
            item_text = (item.text(0) + " " + item.text(1) + " " + item.text(2)).lower()
            direct_match = search in item_text if search else True

            child_matched = False
            for i in range(item.childCount()):
                child = item.child(i)
                if match_and_filter(child):
                    child_matched = True

            should_show = direct_match or child_matched
            item.setHidden(not should_show)
            if should_show and search:
                item.setExpanded(True)
            return should_show

        for i in range(self.tree_channels.topLevelItemCount()):
            match_and_filter(self.tree_channels.topLevelItem(i))

    def _on_tree_selection_changed(self):
        item = self.tree_channels.currentItem()
        if not item:
            self.btn_remove.setEnabled(False)
            self.btn_remove.setText("Remove Selected Derived Channel")
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data.get("is_channel") and not data.get("is_raw"):
            ch_name = data.get("name", "")
            self.btn_remove.setEnabled(True)
            self.btn_remove.setText(f"Remove Derived Channel '{ch_name}'")
        else:
            self.btn_remove.setEnabled(False)
            if data and data.get("is_raw"):
                self.btn_remove.setText("Cannot Remove Raw Disk Image")
            else:
                self.btn_remove.setText("Remove Selected Derived Channel")

    def _get_target_datasets(self) -> List[SLMDataset]:
        """Returns list of datasets to which new channels should be added."""
        if len(self.all_datasets) > 1 and self.combo_target_ds.count() > 0:
            target_data = self.combo_target_ds.currentData()
            if target_data == "all":
                return list(self.all_datasets.values())
            elif isinstance(target_data, SLMDataset):
                return [target_data]
        return [self.dataset]

    def _on_create_lockin_pair(self):
        pair = self.edit_pair_name.text().strip()
        x_ch = self.combo_x_chan.currentText()
        y_ch = self.combo_y_chan.currentText()
        if not pair:
            QMessageBox.warning(self, "Invalid Name", "Please specify a pair name prefix.")
            return

        targets = self._get_target_datasets()
        for ds in targets:
            ds.add_lockin_pair_to_all(pair, x_ch, y_ch)

        self._refresh_lists()
        self.channels_updated.emit()
        QMessageBox.information(
            self,
            "Channels Created",
            f"Successfully created lock-in suite for '{pair}':\n• {pair}_R\n• {pair}_Theta\n• {pair}_X_Rot\n• {pair}_Y_Rot"
        )

    def _on_create_ratio(self):
        name = self.edit_ratio_name.text().strip()
        num_ch = self.combo_num_chan.currentText()
        denom_ch = self.combo_denom_chan.currentText()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please specify a channel name.")
            return

        targets = self._get_target_datasets()
        for ds in targets:
            ds.add_ratio_to_all(name, num_ch, denom_ch)

        self._refresh_lists()
        self.channels_updated.emit()

    def _on_create_diff(self):
        name = self.edit_diff_name.text().strip()
        ch_a = self.combo_diff_a.currentText()
        ch_b = self.combo_diff_b.currentText()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please specify a channel name.")
            return

        targets = self._get_target_datasets()
        for ds in targets:
            ds.add_diff_to_all(name, ch_a, ch_b)

        self._refresh_lists()
        self.channels_updated.emit()

    def _on_remove_channel(self):
        item = self.tree_channels.currentItem()
        if not item:
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data or not data.get("is_channel") or data.get("is_raw"):
            return

        target_ch = data.get("name")
        target_ds = data.get("dataset", self.dataset)

        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove derived channel '{target_ch}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        target_ds.remove_derived_from_all(target_ch)
        self._refresh_lists()
        self.channels_updated.emit()
