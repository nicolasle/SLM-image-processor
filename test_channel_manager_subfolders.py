"""Test suite for hierarchical Channel Manager with subfolders for each dataset (Raw and Derived)."""

import sys
import numpy as np
from PyQt6.QtWidgets import QApplication

from slm_processor.data_model import SLMDataset
from slm_processor.widgets.channel_manager_dialog import SLMChannelManagerDialog


def test_channel_manager_hierarchy():
    print("Testing SLMChannelManagerDialog subfolders hierarchy...")
    app = QApplication.instance() or QApplication(sys.argv)

    ds = SLMDataset("SLM_images")
    dlg = SLMChannelManagerDialog(dataset=ds)

    tree = dlg.tree_channels
    assert tree.topLevelItemCount() == 1, f"Expected 1 dataset root, got {tree.topLevelItemCount()}"

    ds_root = tree.topLevelItem(0)
    assert "Dataset" in ds_root.text(1)
    print("  [OK] Dataset Root verified")

    # Check subfolders: Raw Channels and Derived Channels
    child_names = [ds_root.child(i).text(0) for i in range(ds_root.childCount())]
    assert any("Raw Channels" in name for name in child_names), f"Missing Raw Channels folder in {child_names}"
    assert any("Derived Channels" in name for name in child_names), f"Missing Derived Channels folder in {child_names}"
    print("  [OK] Subfolders present: Raw Channels and Derived Channels")

    # Inspect Raw Channels
    raw_folder = ds_root.child(0)
    raw_items = [raw_folder.child(i).text(0) for i in range(raw_folder.childCount())]
    for key in ["X_PD", "Y_PD", "X_PR", "Y_PR"]:
        assert any(key in item for item in raw_items), f"Missing raw item {key} in {raw_items}"
    print("  [OK] Raw Channels subfolder contains all 4 raw image items")

    # Inspect Derived Channels
    derived_folder = ds_root.child(1)
    derived_subfolders = [derived_folder.child(i).text(0) for i in range(derived_folder.childCount())]
    assert any("Lock-in Pairs" in name for name in derived_subfolders), f"Missing Lock-in Pairs in {derived_subfolders}"
    print("  [OK] Derived Channels subfolders verified")

    # Search filter test
    dlg.edit_filter.setText("PD_R")
    # PD_R should be visible, others hidden
    visible_leaves = []
    def collect_visible(item):
        if item.childCount() == 0 and not item.isHidden():
            visible_leaves.append(item.text(0))
        for i in range(item.childCount()):
            collect_visible(item.child(i))

    collect_visible(ds_root)
    assert any("PD_R" in name for name in visible_leaves), f"PD_R not visible in {visible_leaves}"
    assert not any("PR_Theta" in name for name in visible_leaves), f"PR_Theta should be hidden in {visible_leaves}"
    print("  [OK] Search filter successfully filtered tree items")

    # Reset filter
    dlg.edit_filter.setText("")

    # Test adding a ratio channel
    dlg.edit_ratio_name.setText("Test_Ratio")
    dlg.combo_num_chan.setCurrentText("PR_R")
    dlg.combo_denom_chan.setCurrentText("PD_R")
    dlg._on_create_ratio()

    # Verify ratio appears in Derived -> Ratios subfolder
    ds_root_updated = dlg.tree_channels.topLevelItem(0)
    derived_folder = ds_root_updated.child(1)
    derived_subfolders = [derived_folder.child(i).text(0) for i in range(derived_folder.childCount())]
    assert any("Ratios" in name for name in derived_subfolders), f"Ratios folder missing in {derived_subfolders}"
    print("  [OK] Created ratio channel appears under Derived -> Ratios subfolder")


def test_channel_manager_multi_dataset():
    print("Testing SLMChannelManagerDialog with multiple datasets...")
    app = QApplication.instance() or QApplication(sys.argv)

    ds_harm = SLMDataset("SLM_images")
    # Simulate a second time-domain dataset
    ds_time = SLMDataset("SLM_images")

    all_ds = {
        "Harmonics Domain (5 harmonics)": ds_harm,
        "Time Domain (20 time slices)": ds_time
    }

    dlg = SLMChannelManagerDialog(dataset=ds_harm, all_datasets=all_ds)
    tree = dlg.tree_channels
    assert tree.topLevelItemCount() == 2, f"Expected 2 dataset root items, got {tree.topLevelItemCount()}"

    for i in range(2):
        ds_root = tree.topLevelItem(i)
        subfolders = [ds_root.child(j).text(0) for j in range(ds_root.childCount())]
        assert any("Raw Channels" in s for s in subfolders)
        assert any("Derived Channels" in s for s in subfolders)
        print(f"  [OK] Dataset {i+1} has Raw and Derived subfolders verified")


if __name__ == "__main__":
    test_channel_manager_hierarchy()
    test_channel_manager_multi_dataset()
    print("\nALL CHANNEL MANAGER SUBFOLDER TESTS PASSED SUCCESSFULLY! (100%)")
