from __future__ import annotations

import logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)

from pathlib import Path
from typing import Any, Dict, List

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QMessageBox,
)

# --- 追加：分離した描画ウィジェット ---
from gui.sl_plot import SeaLevelPlot
from gui.sim_plot import SimPlot

# 絶対インポート（既存）
from simulator.simulator import Simulator, SimParams
from io_interface import read_sea_level_csv  # moved here


# ---------------- Bottom Settings Panel ----------------
class SettingsPanel(QWidget):
    """Bottom: interactive settings editor. Emits parametersChanged."""

    parametersChanged = Signal(SimParams)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.growth = QDoubleSpinBox()
        self.growth.setRange(0.0, 50.0)
        self.growth.setDecimals(2)
        self.growth.setValue(8.0)
        self.growth.setSuffix(" mm/yr")

        self.tidal = QDoubleSpinBox()
        self.tidal.setRange(0.0, 10.0)
        self.tidal.setDecimals(2)
        self.tidal.setValue(2.7)
        self.tidal.setSuffix(" m")

        self.max_elev = QDoubleSpinBox()
        self.max_elev.setRange(0.0, 10.0)
        self.max_elev.setDecimals(3)
        self.max_elev.setValue(1.5)
        self.max_elev.setSuffix(" m")

        self.dt = QDoubleSpinBox()
        self.dt.setRange(0.01, 10.0)
        self.dt.setDecimals(2)
        self.dt.setValue(0.1)
        self.dt.setSuffix(" yr")

        self.steps = QSpinBox()
        self.steps.setRange(1, 10_000)
        self.steps.setValue(50)

        self.apply_btn = QPushButton("Apply")
        self.run_btn = QPushButton("Run")

        form = QFormLayout()
        form.addRow("Growth rate:", self.growth)
        form.addRow("Tidal range:", self.tidal)
        form.addRow("Max elevation:", self.max_elev)
        form.addRow("Δt:", self.dt)
        form.addRow("Steps:", self.steps)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.apply_btn)
        buttons.addWidget(self.run_btn)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addLayout(buttons)

        self.apply_btn.clicked.connect(self._emit_params)
        for w in (self.growth, self.tidal, self.max_elev, self.dt, self.steps):
            w.editingFinished.connect(self._emit_params)

    def current_params(self) -> SimParams:
        return SimParams(
            growth_rate_mm_yr=float(self.growth.value()),
            tidal_range_m=float(self.tidal.value()),
            max_elevation_m=float(self.max_elev.value()),
            dt_years=float(self.dt.value()),
            n_steps=int(self.steps.value()),
        )

    @Slot()
    def _emit_params(self) -> None:
        self.parametersChanged.emit(self.current_params())


# ---------------- Main Window ----------------
APP_QSS = """
QMainWindow { background: #121212; }
QLabel[class='panelHeader'] { font-weight: 600; padding: 6px 8px; }
QLabel[class='panelTitle'] { font-weight: 600; }
QFrame#panelFrame { border: 1px solid #2a2a2a; border-radius: 6px; }
QToolBar { border: none; }
QStatusBar { color: #bbb; }
"""


class MainWindow(QMainWindow):
    """
    Layout:
      - Top: horizontal splitter (left: SimPlot, right: SeaLevelPlot)
      - Bottom: interactive settings panel
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MicroAtoll Growth Simulator")
        self.resize(1200, 800)

        self.sim = Simulator()
        self._last_dir: Path | None = None  # remember last directory for file dialog

        self._build_menu_toolbar_status()
        self._build_central_splitters()
        self._connect_yaxis_sharing()

    # ---- Menu/Toolbar/Status ---------------------------------
    def _build_menu_toolbar_status(self) -> None:
        run_action = QAction("Run", self)
        run_action.setShortcut("Ctrl+R")
        run_action.triggered.connect(self._run_sim)

        import_csv_action = QAction("Import CSV…", self)
        import_csv_action.setShortcut("Ctrl+I")
        import_csv_action.triggered.connect(self._import_csv)
        
        debug_circles_action = QAction("Draw Concentric Circles", self)
        debug_circles_action.triggered.connect(self._draw_debug_circles)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(run_action)
        file_menu.addAction(import_csv_action)
        file_menu.addSeparator()
        file_menu.addAction(debug_circles_action)
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        tb = QToolBar("Main", self)
        tb.addAction(run_action)
        tb.addAction(import_csv_action)
        tb.addAction(debug_circles_action)
        self.addToolBar(tb)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")

    # ---- Central Layout --------------------------------------
    def _build_central_splitters(self) -> None:
        # 左＝距離-高度（正方スケール）、右＝CSV（年-高度）
        self.sim_plot = SimPlot()
        self.sl_plot = SeaLevelPlot()
        self.settings_panel = SettingsPanel()

        self.settings_panel.parametersChanged.connect(self._on_params_changed)
        self.settings_panel.run_btn.clicked.connect(self._run_sim)

        top_split = QSplitter(Qt.Horizontal)
        top_split.addWidget(self._wrap_panel(self.sim_plot, "Simulation (Distance–Elevation)"))
        top_split.addWidget(self._wrap_panel(self.sl_plot, "Sea-level (CSV)"))
        top_split.setSizes([3, 2])

        bottom_frame = self._wrap_panel(self.settings_panel, "Interactive Settings")

        vert_split = QSplitter(Qt.Vertical)
        vert_split.addWidget(top_split)
        vert_split.addWidget(bottom_frame)
        vert_split.setSizes([2, 1])

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addWidget(vert_split)
        self.setCentralWidget(central)

    def _wrap_panel(self, w: QWidget, title: str) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setObjectName("panelFrame")
        v = QVBoxLayout(frame)
        header = QLabel(title)
        header.setProperty("class", "panelHeader")
        header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        v.addWidget(header)
        v.addWidget(w, 1)
        return frame

    def _connect_yaxis_sharing(self) -> None:
        """
        右（CSV）→ 左（Sim）へ:
          - yRangeChanged: 数値範囲の共有
          - yMappingChanged: 上下余白（ピクセル）も含めた描画マッピング共有
        """
        self.sl_plot.yRangeChanged.connect(self.sim_plot.set_y_range)
        self.sl_plot.yGeometryChanged.connect(self.sim_plot.apply_right_geometry)

        # 左→右（新規：双方向操作対応）
        self.sim_plot.yRangeEdited.connect(self._apply_y_from_left)

    @Slot(float, float)
    def _apply_y_from_left(self, ymin: float, ymax: float) -> None:
        # 右パネルに外部適用（xlimはロック、幾何再送を内部で行う）
        self.sl_plot.apply_external_y_range(ymin, ymax)

    # ---- Handlers --------------------------------------------
    @Slot(SimParams)
    def _on_params_changed(self, params: SimParams) -> None:
        self.sim.set_params(params)
        self.statusBar().showMessage("Parameters updated", 1500)

    @Slot()
    def _run_sim(self) -> None:
        """
        ここでは例として、距離-高度のダミー曲線を描画。
        後でSimulatorの距離プロファイル出力に差し替えてください。
        """
        results = self.sim.run()
        # 仮のx（距離）を生成：ステップ数に合わせて0..N
        n = max(2, len(results.get("height_m", [])))
        xs = list(range(n))
        zs = results.get("height_m", [0.0] * n)

        self.sim_plot.plot_sim_profile(xs, zs, label="latest run", clear=True)
        # 右側CSVが未読の場合でも、すでに共有yが決まっていればSimPlotはそれを保持
        self.statusBar().showMessage("Simulation finished", 2000)

    @Slot()
    def _import_csv(self) -> None:
        caption = "Import Sea-level CSV"
        start_dir = str(self._last_dir) if self._last_dir else ""
        path, _ = QFileDialog.getOpenFileName(
            self, caption, start_dir, "CSV files (*.csv);;All files (*.*)"
        )
        if not path:
            return
        try:
            xs, ys, meta = read_sea_level_csv(path)
            if not xs:
                raise ValueError("No valid numeric rows were found.")
            self.sl_plot.plot_curve(xs, ys, meta)
            self._last_dir = Path(path).parent
            self.statusBar().showMessage(f"Imported: {Path(path).name}", 2500)

            # CSVのy範囲を左へ同期（signalでも飛ぶが明示的にもう一度適用しておく）
            yr = self.sl_plot.current_y_range()
            if yr:
                self.sim_plot.set_y_range(*yr)
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to read CSV:\n{e}")
            self.statusBar().showMessage("Import failed", 2500)


    # NEW: handler
    def _draw_debug_circles(self) -> None:
        """
        Draw concentric circles on the left panel.
        max_radius is inferred from shared y-range if available.
        """
        self.sim_plot.draw_concentric_circles(max_radius=None, center=(0.0, 0.0), clear=True)
        self.statusBar().showMessage("Drew concentric circles (debug)", 2000)

# ---- Entrypoint ---------------------------------------------
def launch() -> None:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("MicroAtoll Growth Simulator")
    app.setStyleSheet(APP_QSS)
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    launch()
