"""Scanning Laser Microscopy (SLM) Image Processor.

Launch the GUI application to process, visualize, and analyze lock-in
microscopy images (Photoresponse and Photodiode channels).
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from slm_processor.main_window import SLMMainWindow


def main():
    # Enable High DPI scaling
    if hasattr(Qt.ApplicationAttribute, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("SLM Image Processor")

    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Determine initial directory
    initial_dir = None
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        initial_dir = sys.argv[1]
    elif os.path.exists("SLM_images"):
        initial_dir = "SLM_images"

    window = SLMMainWindow(initial_dir=initial_dir)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
