from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure

from gui.plot_interactions import SeaLevelInteractor

class SeaLevelPlot(QWidget):
    """
    右パネル：CSVからインポートした曲線を表示。
    - y軸の実範囲を検出し、共有yスケールとして通知（yRangeChanged）
    - ユーザー操作や新規データでyが変われば emit で左へ伝達
    """

    yRangeChanged = Signal(float, float)  # (ymin, ymax)
    yGeometryChanged = Signal(float, float, int, int, int)  # (ymin, ymax, top_px, bottom_px, canvas_h_px)


    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._y_range: Optional[Tuple[float, float]] = None

        self.fig = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        
        self._hlg_line = None          # 全体を結ぶ細線（任意）
        self._hlg_tri = None           # ▲マーカー（同じ/上昇）
        self._hlg_circ = None          # ●マーカー（下降）
        self._has_curve = False
        self._y_range = None

        self._init_empty()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas, 1)
        
        self._interact = SeaLevelInteractor(self)
        self._interact.connect()

    # --- public API -------------------------------------------------
    def plot_curve(self, xs, ys, meta: dict) -> None:
        xs = list(xs); ys = list(ys)
        self.ax.clear()
        self._hlg_line = None
        self._has_curve = True  # ← 追加

        self.ax.plot(xs, ys, linewidth=0.8, label="Sea level")
        self.ax.grid(True, alpha=0.3)
        name = meta.get("name") or meta.get("filename") or "Sea-level CSV"
        #self.ax.set_title(f"{name}  (rows: {len(xs)}, skipped: {meta.get('skipped_rows', 0)})")
        self.ax.legend(loc="best")

        ymin = min(ys) if ys else -1.0
        ymax = max(ys) if ys else 1.0
        pad = 0.05 * (ymax - ymin + 1e-12)
        self._y_range = (ymin - pad, ymax + pad)
        self.ax.set_ylim(*self._y_range)
        self.canvas.draw_idle()
        self._broadcast_geometry()
        self.yRangeChanged.emit(*self._y_range)

    def clear_hlg(self) -> None:
        """HLGオーバーレイ（線・マーカー）だけを消す（CSVは残す）。"""
        changed = False
        if self._hlg_line is not None:
            self._hlg_line.remove()
            self._hlg_line = None
            changed = True
        if self._hlg_tri is not None:
            self._hlg_tri.remove()
            self._hlg_tri = None
            changed = True
        if self._hlg_circ is not None:
            self._hlg_circ.remove()
            self._hlg_circ = None
            changed = True
        if changed:
            # 凡例を描き直し（Sea level のみ or 他の凡例と整合）
            self.ax.legend(loc="best")
            self.canvas.draw_idle()
            self._broadcast_geometry()

    def plot_hlg_series(self, times, values) -> None:
        """
        記録済みHLG (t, y) をオーバーレイ表示。
        前回より同じ/高い点は ▲、低い点は ● で描画する。
        入力が空・不整合ならHLGのみクリア（CSVはそのまま）。
        """
        ts = list(times or [])
        ys = list(values or [])
        if (not ts) or (not ys) or len(ts) != len(ys):
            self.clear_hlg()
            return

        # 線（全体の接続）を用意（初回のみ作成、以降は更新）
        if self._hlg_line is None:
            (self._hlg_line,) = self.ax.plot(
                ts, ys, linestyle="-", linewidth=1.0,
                color="0.35", alpha=0.9, label="_nolegend_"
            )
        else:
            self._hlg_line.set_data(ts, ys)

        # 前回値との比較でマーカー分類（初点は▲扱い）
        tri_t, tri_y = [], []   # ▲：同じ/上昇（>=）
        circ_t, circ_y = [], [] # ●：下降（<）
        prev = None
        for t, v in zip(ts, ys):
            if prev is None or v >= prev:
                tri_t.append(t); tri_y.append(v)
            else:
                circ_t.append(t); circ_y.append(v)
            prev = v

        # 既存マーカーを消して描き直し
        if self._hlg_tri is not None:
            self._hlg_tri.remove()
            self._hlg_tri = None
        if self._hlg_circ is not None:
            self._hlg_circ.remove()
            self._hlg_circ = None

        if tri_t:
            self._hlg_tri = self.ax.scatter(
                tri_t, tri_y, marker="^", s=28, linewidths=0.6,
                edgecolors="0.2", facecolors="0.2", alpha=0.95, label="HLG"
            )
        if circ_t:
            self._hlg_circ = self.ax.scatter(
                circ_t, circ_y, marker="o", s=26, linewidths=0.6,
                edgecolors="0.25", facecolors="0.65", alpha=0.95, label="HLS after diedown"
            )

        # 凡例更新（Sea level + HLG系）
        self.ax.legend(loc="best")

        # yレンジはCSV基準を維持（必要ならHLGも含めて拡張する以下を有効化）
        # if self._y_range:
        #     ymin = min(self._y_range[0], min(ys))
        #     ymax = max(self._y_range[1], max(ys))
        #     self._y_range = (ymin, ymax)
        #     self.ax.set_ylim(ymin, ymax)

        self.canvas.draw_idle()
        self._broadcast_geometry()

    def current_y_range(self) -> Optional[Tuple[float, float]]:
        return self._y_range

    # --- QWidget overrides ------------------------------------------
    def resizeEvent(self, event) -> None:  # noqa: N802
        # リサイズで軸範囲は保持。必要ならここで将来的に通知も可能。
        # （現状は右→左へ一方向共有）
        super().resizeEvent(event)
        self._broadcast_geometry()

    # --- internals ---------------------------------------------------
    def _init_empty(self) -> None:
        self.ax.clear()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("Time / Age / Year")
        self.ax.set_ylabel("Sea level / Elevation (m)")
        self.ax.set_title("No Sea level curve loaded")
        self.canvas.draw_idle()

    def _broadcast_geometry(self) -> None:
        """Axesの上下位置（ピクセル）を測って左へ通知。"""
        try:
            self.fig.canvas.draw()  # レイアウト確定
            bbox = self.ax.get_window_extent()  # ピクセル座標（FigureCanvas上）
            canvas_h = self.canvas.height() 
            top_px = int(max(0, canvas_h - bbox.y1))   # 上端の余白
            bottom_px = int(max(0, bbox.y0))           # 下端の余白
            if self._y_range:
                ymin, ymax = self._y_range
                # （幅方向は左で xlim をトリミングして対応する方針）
                self.yGeometryChanged.emit(ymin, ymax, top_px, bottom_px, canvas_h)
        except Exception:
            pass

    @staticmethod
    def _compute_y_range(ys: Iterable[float]) -> Tuple[float, float]:
        ys = list(ys)
        if not ys:
            return (0.0, 1.0)
        ymin, ymax = min(ys), max(ys)
        if ymin == ymax:
            pad = 1.0 if ymin == 0 else abs(ymin) * 0.1
            ymin, ymax = ymin - pad, ymax + pad
        else:
            # 5%のマージンで見やすく
            span = ymax - ymin
            ymin -= 0.05 * span
            ymax += 0.05 * span
        return (ymin, ymax)

    def apply_external_y_range(self, ymin: float, ymax: float) -> None:
        """外部から y 範囲の指示を受け取る（x は不変）。幾何を再送して左へ同期。"""
        if ymin == ymax:
            pad = 1.0 if ymin == 0 else abs(ymin) * 0.1
            ymin, ymax = ymin - pad, ymax + pad
        self._y_range = (ymin, ymax)
        self.ax.set_ylim(ymin, ymax)
        self.canvas.draw_idle()
        # 自パネルの標準イベントと同じく通知
        self.yRangeChanged.emit(ymin, ymax)
        self._broadcast_geometry()