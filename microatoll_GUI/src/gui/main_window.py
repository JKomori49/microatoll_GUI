from __future__ import annotations

import logging
import math
import sys
import traceback
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
        self.act_export_png = QAction("Image (png)...", self)
        self.act_export_png.setStatusTip("Export as PNG image")
        self.act_export_png.setShortcut("Ctrl+E")
        self.act_export_png.triggered.connect(self._export_image_png)
        export_menu.addAction(self.act_export_png)

        self.act_export_svg = QAction("Image (svg)...", self)
        self.act_export_svg.setStatusTip("Export as SVG image")
        self.act_export_svg.triggered.connect(self._export_image_svg)
        #export_menu.addAction(self.act_export_svg)     # Export svg is not yet ready. Will be implemented once bugs are removed.

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
        # 左＝距離-高度（正方スケール）、右＝CSV（年-高度）
        self.sim_plot = SimPlot()
        self.sl_plot = SeaLevelPlot()
        self.settings_panel = SettingsPanel()

        self.settings_panel.parametersChanged.connect(self._on_params_changed)
        self.settings_panel.run_btn.clicked.connect(self._run_sim)
        self.settings_panel.init_btn.clicked.connect(self._initialize_sim)

        # --- 上段（左右） ---
        top_split = QSplitter(Qt.Horizontal)

        left_frame  = self._wrap_panel(self.sim_plot, None)
        right_frame = self._wrap_panel(self.sl_plot, None)

        # ★ 最低高さを確保（起動直後に潰れない）
        MIN_TOP_H = 260
        left_frame.setMinimumHeight(MIN_TOP_H)
        right_frame.setMinimumHeight(MIN_TOP_H)

        # 念のため拡張ポリシー（縦横ともに広がる）
        for fr in (left_frame, right_frame):
            fr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        top_split.addWidget(left_frame)
        top_split.addWidget(right_frame)
        top_split.setSizes([3, 2])

        top_split.setChildrenCollapsible(False)

        # --- 下段（設定） ---
        bottom_frame = self._wrap_panel(self.settings_panel, None)
        bottom_frame.setMinimumHeight(160)  # 設定パネルが大きくなりすぎないよう最低限だけ

        # --- 縦割りスプリッタ ---
        vert_split = QSplitter(Qt.Vertical)
        vert_split.addWidget(top_split)
        vert_split.addWidget(bottom_frame)

        # ★ 比率は上段:下段 = 3:1（初期サイズ）
        vert_split.setStretchFactor(0, 3)
        vert_split.setStretchFactor(1, 1)
        # ★ こちらもつぶれ防止
        vert_split.setChildrenCollapsible(False)

        # 初期サイズ（ウィンドウの想定サイズに基づく目安）
        vert_split.setSizes([int(self.height() * 0.66), int(self.height() * 0.34)])

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addWidget(vert_split)
        self.setCentralWidget(central)

        # 後で参照できるよう保持（任意）
        self._top_split = top_split
        self._vert_split = vert_split

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_did_initial_split_sizes", False):
            self._did_initial_split_sizes = True
            # 実際の表示高さに基づいて再配分
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
        params = self.settings_panel.current_params()
        self.sim.set_params(params)
        self.sim.initialize()

        try:
            from simulator.iteration import IterativeRunner
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
            # 右パネル：まず海水準だけ描く
            self.sl_plot.plot_curve(xs, ys, meta)
            # HLGはまだ無いのでクリア（描画を邪魔しない）
            self.sl_plot.clear_hlg()

            # Set T0/T1 from the first/last time values (floored)
            t0 = math.floor(xs[0])
            t1 = math.floor(xs[-1])
            self.settings_panel.set_time_window(t0, t1)

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

    def _export_image_png(self) -> None:
        """
        上段2パネル（sim_plot, sl_plot）を画面見た目のまま横に結合して1枚のPNGに保存する。
        """
        # 保存先ダイアログ
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

        # 必要ウィジェットの存在チェック
        if not hasattr(self, "sim_plot") or not hasattr(self, "sl_plot"):
            QMessageBox.critical(self, "Export failed", "Panels are not ready (sim_plot / sl_plot).")
            return

        try:
            # 各パネルをキャプチャ（見た目どおり）
            pix1 = self.sim_plot.grab()
            pix2 = self.sl_plot.grab()
            w1, h1 = pix1.width(), pix1.height()
            w2, h2 = pix2.width(), pix2.height()

            W = w1 + w2
            H = max(h1, h2)

            # 透明→白背景で出力（必要に応じて色を変更可）
            image = QImage(W, H, QImage.Format_ARGB32)
            image.fill(QColor(Qt.white))

            # 描画合成
            painter = QPainter(image)
            # 上寄せで配置（高さが異なる場合は上に揃える。中央にしたいなら y を調整）
            painter.drawPixmap(0, 0, pix1)
            painter.drawPixmap(w1, 0, pix2)
            painter.end()

            # 保存
            if not image.save(str(out_path), "PNG"):
                raise RuntimeError("Failed to save PNG image.")

            self._last_dir = out_path.parent
            self.statusBar().showMessage(f"Exported: {out_path.name}", 4000)

        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def _export_image_svg(self) -> None:
        """
        ベクターSVGでエクスポート：sim_plot と sl_plot の Matplotlib 図を
        個別に SVG savefig し、1つのSVGに横並びで結合して保存する。
        """
        import sys, io, re, traceback, tempfile
        from pathlib import Path
        from xml.etree import ElementTree as ET
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        # --- 保存先選択 ---
        default_dir = self._last_dir or Path.home()
        suggested = Path(default_dir) / "microatoll_export.svg"
        try:
            out_path_str, _ = QFileDialog.getSaveFileName(
                self, "Export Image (SVG, vector)", str(suggested), "SVG Image (*.svg)"
            )
            if not out_path_str:
                return
            out_path = Path(out_path_str)
            if out_path.suffix.lower() != ".svg":
                out_path = out_path.with_suffix(".svg")

            # 必須チェック
            if not hasattr(self, "sim_plot") or not hasattr(self, "sl_plot"):
                raise RuntimeError("Panels are not ready (sim_plot / sl_plot).")

            # --- 1) 各 Matplotlib Figure を SVG にベクター保存（メモリ or 一時ファイル） ---
            fig1 = getattr(getattr(self.sim_plot, "canvas", None), "figure", None)
            fig2 = getattr(getattr(self.sl_plot, "canvas", None), "figure", None)
            if fig1 is None or fig2 is None:
                raise RuntimeError("Matplotlib figures are not available.")

            # tight=True で余白を最小化（見た目に近づける）
            buf1 = io.BytesIO()
            buf2 = io.BytesIO()
            fig1.savefig(buf1, format="svg", bbox_inches="tight")
            fig2.savefig(buf2, format="svg", bbox_inches="tight")
            svg1 = buf1.getvalue().decode("utf-8")
            svg2 = buf2.getvalue().decode("utf-8")

            # --- 2) SVG を解析してサイズ取得 ---
            def _parse_svg(svg_text: str):
                root = ET.fromstring(svg_text)
                w_attr = root.get("width", "")
                h_attr = root.get("height", "")
                vb_attr = root.get("viewBox", "")
                def _to_float(s):
                    # "800pt" / "800px" / "800" → 800.0
                    m = re.match(r"^\s*([0-9.+-eE]+)", s or "")
                    return float(m.group(1)) if m else None
                w = _to_float(w_attr)
                h = _to_float(h_attr)
                if (w is None or h is None) and vb_attr:
                    parts = [p for p in vb_attr.replace(",", " ").split() if p]
                    if len(parts) == 4:
                        w = w or float(parts[2])
                        h = h or float(parts[3])
                # ルート直下の <defs> と 残り（描画本体）を分離
                defs = []
                body = []
                for child in list(root):
                    if child.tag.endswith("defs"):
                        defs.append(child)
                    else:
                        body.append(child)
                return root, w or 0.0, h or 0.0, defs, body

            r1, w1, h1, defs1, body1 = _parse_svg(svg1)
            r2, w2, h2, defs2, body2 = _parse_svg(svg2)
            if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
                raise RuntimeError("Failed to read SVG sizes.")

            # --- 3) id 衝突回避のため、2枚目の defs/要素の id へプレフィックス付与 ---
            def _prefix_ids(elem: ET.Element, prefix: str):
                # id属性の付け替え + 参照側（url(#id)）の書き換え
                id_map = {}
                for e in elem.iter():
                    idv = e.get("id")
                    if idv:
                        new_id = f"{prefix}{idv}"
                        id_map[idv] = new_id
                        e.set("id", new_id)
                # url(#id) 置換（最小限・単純版）
                url_re = re.compile(r"url\(#([A-Za-z0-9_\-:.]+)\)")
                for e in elem.iter():
                    for k, v in list(e.attrib.items()):
                        def repl(m):
                            old = m.group(1)
                            return f"url(#{id_map.get(old, old)})"
                        e.set(k, url_re.sub(repl, v))

            # 2枚目にだけプレフィックス
            wrap2 = ET.Element("g")
            for x in body2:
                wrap2.append(x)
            defs_wrap2 = ET.Element("defs")
            for d in defs2:
                defs_wrap2.extend(list(d))
            _prefix_ids(wrap2, "S2_")
            _prefix_ids(defs_wrap2, "S2_")

            # --- 4) 出力 SVG を組み立て（横並び） ---
            W = w1 + w2
            H = max(h1, h2)

            svg_ns = "http://www.w3.org/2000/svg"
            xlink_ns = "http://www.w3.org/1999/xlink"
            XMLNS_NS = "http://www.w3.org/2000/xmlns/"

            # ① ルートの名前空間は register_namespace に任せる（xmlns を手動で付けない）
            ET.register_namespace("", svg_ns)
            ET.register_namespace("xlink", xlink_ns)

            # ② ルート作成（xmlns は付けない！）
            root = ET.Element(f"{{{svg_ns}}}svg", {
                "version": "1.1",
                "width": str(W),
                "height": str(H),
                "viewBox": f"0 0 {W} {H}",
                # 必要なら metadata などもここに
            })
            # ※ 必要に応じて xlink も明示したい場合は下記のように XMLNS 名前空間で宣言する
            # root.set(f"{{{XMLNS_NS}}}xlink", xlink_ns)

            # ③ 子要素に紛れ込んだ xmlns 再定義を除去するクリーナ
            def _strip_xmlns(elem: ET.Element, keep_on_root: bool = True) -> None:
                """子孫の不要な xmlns / xmlns:* を除去（ElementTree は稀に継承を再出力する）"""
                for e in elem.iter():
                    # ルートはスキップ（自動定義に任せる）
                    if e is root and keep_on_root:
                        continue
                    # 明示 'xmlns' 属性を除去
                    if "xmlns" in e.attrib:
                        e.attrib.pop("xmlns", None)
                    # xmlns:*（XMLNS 名前空間の属性）を除去
                    for k in list(e.attrib.keys()):
                        if k.startswith(f"{{{XMLNS_NS}}}"):
                            # xlink を子で再宣言している場合も削除
                            e.attrib.pop(k, None)

            # defs 結合（1→2 の順）: まず空 defs を作ってから要素を拡張
            out_defs = ET.SubElement(root, f"{{{svg_ns}}}defs")
            for d in defs1:
                for child in list(d):
                    _strip_xmlns(child, keep_on_root=False)
                    out_defs.append(child)
            for d in list(defs_wrap2):
                for child in list(d):
                    _strip_xmlns(child, keep_on_root=False)
                    out_defs.append(child)

            # 左（図1）本体：そのまま
            g1 = ET.SubElement(root, f"{{{svg_ns}}}g", {"transform": "translate(0,0)"})
            for x in body1:
                _strip_xmlns(x, keep_on_root=False)
                g1.append(x)

            # 右（図2）本体：w1 だけ平行移動
            g2 = ET.SubElement(root, f"{{{svg_ns}}}g", {"transform": f"translate({w1},0)"})
            for x in wrap2:
                _strip_xmlns(x, keep_on_root=False)
                g2.append(x)

            # --- 5) 保存 ---
            try:
                ET.indent(root)  # Py3.9+
            except Exception:
                pass
            tree = ET.ElementTree(root)
            tree.write(out_path, encoding="utf-8", xml_declaration=True)
            
            self._last_dir = out_path.parent
            self.statusBar().showMessage(f"Exported: {out_path.name}", 4000)

        except Exception as e:
            print("[Export SVG Vector] Failed:", file=sys.stderr)
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
