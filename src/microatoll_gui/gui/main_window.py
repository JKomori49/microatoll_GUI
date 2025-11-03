from __future__ import annotations

import logging
import math
logging.getLogger("matplotlib").setLevel(logging.ERROR)

from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QRegion
from PySide6.QtSvg import QSvgGenerator
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QSizePolicy,
    QSplitter, QStatusBar, QToolBar, QVBoxLayout, QWidget,
    QFileDialog, QMessageBox,
)

# --- Plot widgets ---
from .sl_plot import SeaLevelPlot
from .sim_plot import SimPlot

# --- Settings panel (separate file) ---
from .setting_panel import SettingsPanel

# --- Simulator and parameters ---
from microatoll_gui.simulator.simulator import Simulator, SimParams
from microatoll_gui.simulator.iteration import IterativeRunner

# CSV I/O
from microatoll_gui.io_interface import read_sea_level_csv


# ---------------- Main Window ----------------
APP_QSS = """
QMainWindow { background: palette(Window); } 
QLabel[class="panelHeader"] { font-weight: 600; padding: 6px 8px; color: palette(WindowText); }
QLabel[class="panelTitle"]  { font-weight: 600; color: palette(WindowText); }
QFrame#panelFrame { border: 1px solid palette(Mid); border-radius: 6px; } 
QToolBar  { border: none; background: palette(Window); }
QStatusBar{ color: palette(WindowText); background: palette(Window); }
"""


class MainWindow(QMainWindow):
    """
    Layout:
      - Top: horizontal splitter (left: SimPlot, right: SeaLevelPlot)
      - Bottom: interactive settings panel (SettingsPanel)
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Microatoll Growth Simulator")
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

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(init_action)
        file_menu.addAction(run_action)
        file_menu.addAction(import_csv_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        export_menu = self.menuBar().addMenu("&Export")
        self.act_export_png = QAction("Export as png...", self)
        self.act_export_png.setStatusTip("Export as PNG image")
        self.act_export_png.setShortcut("Ctrl+E")
        self.act_export_png.triggered.connect(self._export_image_png)
        export_menu.addAction(self.act_export_png)

        self.act_export_svg_sep = QAction("Export as svg...", self)
        self.act_export_svg_sep.setStatusTip("Export Simulation and Sea-level panels as separate SVG files (vector)")
        self.act_export_svg_sep.triggered.connect(self._export_image_svg_separate)
        export_menu.addAction(self.act_export_svg_sep)

        tb = QToolBar("Main", self)
        tb.addAction(init_action)
        tb.addAction(run_action)
        tb.addAction(import_csv_action)
        #tb.addAction(self.act_export_png)
        self.addToolBar(tb)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")

    # ---- Central Layout --------------------------------------
    def _build_central_splitters(self) -> None:
        # Left: distance–elevation (square scale), Right: CSV (year–elevation)
        self.sim_plot = SimPlot()
        self.sl_plot = SeaLevelPlot()
        self.settings_panel = SettingsPanel()

        self.settings_panel.parametersChanged.connect(self._on_params_changed)
        self.settings_panel.run_btn.clicked.connect(self._run_sim)
        self.settings_panel.init_btn.clicked.connect(self._initialize_sim)

        # --- Top row (left/right) ---
        top_split = QSplitter(Qt.Horizontal)

        left_frame  = self._wrap_panel(self.sim_plot, None)
        right_frame = self._wrap_panel(self.sl_plot, None)

        # Ensure a minimum height (avoid collapsing at startup)
        MIN_TOP_H = 260
        left_frame.setMinimumHeight(MIN_TOP_H)
        right_frame.setMinimumHeight(MIN_TOP_H)

        # Expanding size policy (both directions)
        for fr in (left_frame, right_frame):
            fr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        top_split.addWidget(left_frame)
        top_split.addWidget(right_frame)
        top_split.setSizes([3, 2])

        top_split.setChildrenCollapsible(False)

        # --- Bottom row (settings) ---
        bottom_frame = self._wrap_panel(self.settings_panel, None)
        bottom_frame.setMinimumHeight(160)  # Minimum to prevent the settings panel from dominating

        # --- Vertical splitter ---
        vert_split = QSplitter(Qt.Vertical)
        vert_split.addWidget(top_split)
        vert_split.addWidget(bottom_frame)

        # Initial size ratio top:bottom = 3:1
        vert_split.setStretchFactor(0, 3)
        vert_split.setStretchFactor(1, 1)
        # Prevent collapsing here as well
        vert_split.setChildrenCollapsible(False)

        # Initial sizes based on the window height
        vert_split.setSizes([int(self.height() * 0.66), int(self.height() * 0.34)])

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addWidget(vert_split)
        self.setCentralWidget(central)

        # Keep references for later use (optional)
        self._top_split = top_split
        self._vert_split = vert_split

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_did_initial_split_sizes", False):
            self._did_initial_split_sizes = True
            # Recalculate based on actual displayed height
            h = max(1, self.centralWidget().height())
            self._vert_split.setSizes([int(h * 0.66), int(h * 0.34)])

    def _wrap_panel(self, w: QWidget, title: str | None = None) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setObjectName("panelFrame")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if title:
            header = QLabel(title)
            header.setProperty("class", "panelHeader")
            header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            layout.addWidget(header)

        layout.addWidget(w, 1)
        return frame

    def _connect_yaxis_sharing(self) -> None:
        """
        Right (CSV) -> Left (Sim):
          - yRangeChanged: share numerical y-range
          - yGeometryChanged: share drawing mapping including top/bottom pixel margins
        """
        self.sl_plot.yRangeChanged.connect(self.sim_plot.set_y_range)
        self.sl_plot.yGeometryChanged.connect(self.sim_plot.apply_right_geometry)

        # Left -> Right (bidirectional interaction)
        self.sim_plot.yRangeEdited.connect(self._apply_y_from_left)

    @Slot(float, float)
    def _apply_y_from_left(self, ymin: float, ymax: float) -> None:
        # Apply to right panel externally (xlim locked; geometry rebroadcast internally)
        self.sl_plot.apply_external_y_range(ymin, ymax)

    # ---- Handlers --------------------------------------------
    @Slot(SimParams)
    def _on_params_changed(self, params: SimParams) -> None:
        self.sim.set_params(params)
        self.statusBar().showMessage("Parameters updated", 1500)

    @Slot()
    def _run_sim(self) -> None:
        params = self.settings_panel.current_params()
        self.sim.set_params(params)
        self.sim.initialize()

        try:
            runner = IterativeRunner(self.sim)
            results = runner.run_until_end()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"Simulation error: {e}", 5000)
            return

        # 左パネル：途中灰線 + 最終
        final = results["final"]["new"]
        xs, ys, phi = final["x"], final["y"], final["phi"]
        bh = getattr(params, "base_height", 0.0)

        self.sim_plot.ax.clear()
        for rec in results.get("records", []):
            self.sim_plot.ax.plot(rec["x"], rec["y"], color="0.6", linewidth=0.8, alpha=0.6, zorder=1)
        self.sim_plot.plot_polyline_with_phi(
            xs, ys, phi, clear=False, shade_block_region=True, block_level=bh, show_vertices=False
        )

        # 右パネル：HLG（あれば）を重ねる／無ければクリア
        hlg = results.get("hlg", {}) or {}
        times = hlg.get("t", []) or []
        vals  = hlg.get("y", []) or []
        if times and vals:
            self.sl_plot.plot_hlg_series(times, vals)
        else:
            self.sl_plot.clear_hlg()

        #t_final = final.get("t_years", None)
        #if t_final is not None:
        #    self.sim_plot.ax.set_title(f"Final time = {t_final:.2f} yr")
        self.sim_plot.canvas.draw_idle()

        self.statusBar().showMessage(
            f"Simulation finished. Recorded {len(results.get('records', []))} steps; HLG points: {len(times)}.",
            4000
        )

    @Slot()
    def _initialize_sim(self) -> None:
        """
        Generate and draw the initial polyline at τ=0 with current parameters (no stepping).
        """
        # 1) Apply parameters
        params = self.settings_panel.current_params()
        self.sim.set_params(params)
        bh = getattr(params, "base_height", 0.0)

        # 2) Initialize (tau=0)
        cur = self.sim.initialize()  # {"x","y","phi","tau","t_years"}

        # 3) Draw (initial only: color by φ + shade y<BH)
        if hasattr(self.sim_plot, "plot_polyline_with_phi"):
            self.sim_plot.plot_polyline_with_phi(
                cur["x"], cur["y"], cur["phi"],
                clear=True, shade_block_region=True, block_level=bh
            )
        else:
            self.sim_plot.plot_sim_profile(cur["x"], cur["y"], label="initial polyline", clear=True)

        self.statusBar().showMessage(f"Initialized: τ={cur['tau']} | t={cur['t_years']:.3f} yr", 2500)
        
    # ---- I/O helpers --------------------------------------------
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
            # Right panel: draw only the sea-level curve first
            self.sl_plot.plot_curve(xs, ys, meta)
            # No HLG yet, clear it to avoid clutter
            self.sl_plot.clear_hlg()

            # Set T0/T1 from the first/last time values (floored)
            t0 = math.floor(xs[0])
            t1 = math.floor(xs[-1])
            self.settings_panel.set_time_window(t0, t1)

            # Register sea-level curve to the simulator
            try:
                self.sim.set_sea_level_curve(xs, ys)  # xs=year, ys=sea-level[m]
            except Exception:
                pass
            self._last_dir = Path(path).parent
            self.statusBar().showMessage(f"Imported: {Path(path).name}", 2500)

            # Sync CSV y-range to the left (explicit apply in addition to signal)
            yr = self.sl_plot.current_y_range()
            if yr:
                self.sim_plot.set_y_range(*yr)
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to read CSV:\n{e}")
            self.statusBar().showMessage("Import failed", 2500)

    def _export_image_png(self) -> None:
        """
        Save the two top panels (sim_plot, sl_plot) side by side as a single PNG as displayed.
        """
        # Save dialog
        default_dir = self._last_dir or Path.home()
        suggested = Path(default_dir) / "microatoll_export.png"
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image (PNG)",
            str(suggested),
            "PNG Image (*.png)"
        )
        if not path_str:
            return
        out_path = Path(path_str)
        if out_path.suffix.lower() != ".png":
            out_path = out_path.with_suffix(".png")

        # Check required widgets exist
        if not hasattr(self, "sim_plot") or not hasattr(self, "sl_plot"):
            QMessageBox.critical(self, "Export failed", "Panels are not ready (sim_plot / sl_plot).")
            return

        try:
            # Grab pixmaps of each panel (as displayed)
            pix1 = self.sim_plot.grab()
            pix2 = self.sl_plot.grab()
            w1, h1 = pix1.width(), pix1.height()
            w2, h2 = pix2.width(), pix2.height()

            W = w1 + w2
            H = max(h1, h2)

            # Use white background instead of transparent (adjust color if needed)
            image = QImage(W, H, QImage.Format_ARGB32)
            image.fill(QColor(Qt.white))

            # Paint composite
            painter = QPainter(image)
            # Align to top (adjust y to center if desired)
            painter.drawPixmap(0, 0, pix1)
            painter.drawPixmap(w1, 0, pix2)
            painter.end()

            # Save
            if not image.save(str(out_path), "PNG"):
                raise RuntimeError("Failed to save PNG image.")

            self._last_dir = out_path.parent
            self.statusBar().showMessage(f"Exported: {out_path.name}", 4000)

        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def _export_image_svg_separate(self) -> None:
        """
        Save Matplotlib figures of sim_plot and sl_plot as separate SVG vector files.
        Output filenames: <chosen>_sim.svg and <chosen>_sl.svg.
        """
        import io, sys, traceback
        from pathlib import Path
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        # Base filename for saving
        default_dir = self._last_dir or Path.home()
        suggested = Path(default_dir) / "microatoll_export.svg"
        try:
            path_str, _ = QFileDialog.getSaveFileName(
                self, "Export Images (SVG, vector, separate)", str(suggested), "SVG Image (*.svg)"
            )
            if not path_str:
                return

            base = Path(path_str)
            if base.suffix.lower() != ".svg":
                base = base.with_suffix(".svg")
            sim_path = base.with_name(base.stem + "_sim.svg")
            sl_path  = base.with_name(base.stem + "_sl.svg")

            # Get figures
            fig1 = getattr(getattr(self.sim_plot, "canvas", None), "figure", None)
            fig2 = getattr(getattr(self.sl_plot, "canvas", None), "figure", None)
            if fig1 is None or fig2 is None:
                raise RuntimeError("Matplotlib figures are not available (sim_plot / sl_plot).")

            # Vector output (keep text; do not outline fonts)
            import matplotlib as mpl
            old_fonttype = mpl.rcParams.get("svg.fonttype", "path")
            try:
                mpl.rcParams["svg.fonttype"] = "none"  # keep text as text
                fig1.savefig(sim_path, format="svg", bbox_inches="tight", metadata={"Date": None})
                fig2.savefig(sl_path,  format="svg", bbox_inches="tight", metadata={"Date": None})
            finally:
                mpl.rcParams["svg.fonttype"] = old_fonttype

            self._last_dir = sim_path.parent
            self.statusBar().showMessage(f"Exported: {sim_path.name}, {sl_path.name}", 5000)

        except Exception as e:
            print("[Export SVG Separate] Failed:", file=sys.stderr)
            traceback.print_exc()
            QMessageBox.critical(self, "Export failed", str(e))



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
