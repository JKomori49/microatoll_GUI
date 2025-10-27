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

        self._init_empty()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas, 1)
        
        self._interact = SeaLevelInteractor(self)
        self._interact.connect()

    # --- public API -------------------------------------------------
    def plot_curve(self, xs: Iterable[float], ys: Iterable[float], meta: Dict[str, Any]) -> None:
        xs = list(xs)
        ys = list(ys)

        self.ax.clear()
        self.ax.plot(xs, ys, linewidth=1.5)
        self.ax.grid(True, alpha=0.3)

        # ラベル決定
        xlabel = "X"
        ylabel = "Y"
        header = [h.strip() for h in (meta.get("header") or [])]
        xi = meta.get("x_index")
        yi = meta.get("y_index")
        if isinstance(xi, int) and header and xi < len(header):
            xlabel = header[xi] or xlabel
        if isinstance(yi, int) and header and yi < len(header):
            ylabel = header[yi] or ylabel

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        name = Path(meta.get("path", "CSV")).name
        self.ax.set_title(f"{name}  (rows: {len(xs)}, skipped: {meta.get('skipped_rows', 0)})")

        # y共有のために範囲を決めて発信
        ymin, ymax = self._compute_y_range(ys)
        self._y_range = (ymin, ymax)
        self.ax.set_ylim(ymin, ymax)

        self.canvas.draw_idle()
        self._broadcast_geometry() 
        self.yRangeChanged.emit(ymin, ymax)

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
        self.ax.set_title("No CSV loaded")
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