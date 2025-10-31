# ─────────────────────────────────────────────────────────────────────────────
# File: __main__.py
# (Allows `python -m <your_package_or_folder>` to start the GUI.)
# ─────────────────────────────────────────────────────────────────────────────
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from .gui.main_window import launch
from .io_interface.paths import resource_path


icon_path = resource_path("resources/icons/app_icon.ico")

app = QApplication.instance() or QApplication([])
app.setWindowIcon(QIcon(icon_path))

def main() -> int:
	launch()

if __name__ == "__main__":
	launch()
