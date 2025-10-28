from __future__ import annotations

import logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)

from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QSplitter, QStatusBar, QToolBar, QVBoxLayout, QWidget,
    QFileDialog, QMessageBox,
)

# --- 描画ウィジェット ---
from gui.sl_plot import SeaLevelPlot
from gui.sim_plot import SimPlot

# --- 設定パネル（分離ファイル） ---
from gui.setting_panel import SettingsPanel

# --- シミュレータとパラメータ ---
from simulator.simulator import Simulator, SimParams

# CSV I/O
from io_interface import read_sea_level_csv


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
      - Bottom: interactive settings panel (SettingsPanel)
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

        init_action = QAction("Initialize", self)
        init_action.setShortcut("Ctrl+Shift+R")
        init_action.triggered.connect(self._initialize_sim)

        import_csv_action = QAction("Import CSV…", self)
        import_csv_action.setShortcut("Ctrl+I")
        import_csv_action.triggered.connect(self._import_csv)

        debug_circles_action = QAction("Draw Concentric Circles", self)
        debug_circles_action.triggered.connect(self._draw_debug_circles)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(init_action)
        file_menu.addAction(run_action)
        file_menu.addAction(import_csv_action)
        file_menu.addSeparator()
        file_menu.addAction(debug_circles_action)
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        tb = QToolBar("Main", self)
        tb.addAction(init_action)
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
        self.settings_panel.init_btn.clicked.connect(self._initialize_sim) 

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
          - yGeometryChanged: 上下余白（ピクセル）も含めた描画マッピング共有
        """
        self.sl_plot.yRangeChanged.connect(self.sim_plot.set_y_range)
        self.sl_plot.yGeometryChanged.connect(self.sim_plot.apply_right_geometry)

        # 左→右（双方向操作対応）
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
        """Advance exactly one time step and draw old/new polylines together."""
        # 1) パラメータ反映（T0, Δt, gr, BH など）
        params = self.settings_panel.current_params()
        self.sim.set_params(params)
        bh = getattr(params, "base_height", 0.0)

        # 2) 1ステップ進める
        results = self.sim.step_once()
        old = results["old"]; new = results["new"]

        # 3) 描画（旧=細線、新=φ色分け + y<BHをシェード）
        if hasattr(self.sim_plot, "plot_polyline_step"):
            self.sim_plot.plot_polyline_step(
                (old["x"], old["y"]),
                (new["x"], new["y"]),
                new_phi=new["phi"],
                block_level=bh,
                show_vertices=True,
                clear=True,
            )
        else:
            # フォールバック（新だけプロット）
            self.sim_plot.plot_polyline_with_phi(new["x"], new["y"], new["phi"], clear=True, shade_block_region=True, block_level=bh)

        # 進行状況表示（t と τ）
        self.statusBar().showMessage(f"Step τ={new['tau']}  |  t={new['t_years']:.3f} yr", 2500)


    @Slot()
    def _initialize_sim(self) -> None:
        """
        現在のパラメータで τ=0 の初期ポリラインを生成して描画（ステップは進めない）。
        """
        # 1) パラメータ反映
        params = self.settings_panel.current_params()
        self.sim.set_params(params)
        bh = getattr(params, "base_height", 0.0)

        # 2) 初期化（tau=0）
        cur = self.sim.initialize()  # {"x","y","phi","tau","t_years"}

        # 3) 描画（初期のみ：細線不要／φ色分け + y<BHシェード）
        if hasattr(self.sim_plot, "plot_polyline_with_phi"):
            self.sim_plot.plot_polyline_with_phi(
                cur["x"], cur["y"], cur["phi"],
                clear=True, shade_block_region=True, block_level=bh
            )
        else:
            self.sim_plot.plot_sim_profile(cur["x"], cur["y"], label="initial polyline", clear=True)

        self.statusBar().showMessage(f"Initialized: τ={cur['tau']} | t={cur['t_years']:.3f} yr", 2500)

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
            # NEW: シミュレーションへ海水準曲線を登録
            try:
                self.sim.set_sea_level_curve(xs, ys)  # xs=year, ys=sea-level[m]
            except Exception:
                pass
            self._last_dir = Path(path).parent
            self.statusBar().showMessage(f"Imported: {Path(path).name}", 2500)

            # CSVのy範囲を左へ同期（signalでも飛ぶが明示的にもう一度適用）
            yr = self.sl_plot.current_y_range()
            if yr:
                self.sim_plot.set_y_range(*yr)
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to read CSV:\n{e}")
            self.statusBar().showMessage("Import failed", 2500)

    @Slot()
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
