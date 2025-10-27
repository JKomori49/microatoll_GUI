from __future__ import annotations

from typing import Iterable, Optional, Tuple

from PySide6.QtCore import Signal
from matplotlib.backend_bases import MouseButton
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from gui.plot_interactions import SimInteractor

class SimPlot(QWidget):
    """
    Left panel: distance–elevation plot that mirrors the right panel's vertical geometry.

    Policy:
      - Always keep aspect ratio 1:1 (no vertical shrinking).
      - y-limits are identical to the right panel.
      - The axes box's TOP pixel position and HEIGHT (in pixels) match the right panel.
      - If frame width is insufficient, adjust xlim (trim horizontally) instead of shrinking.
      - On resize, recompute xlim only (preserving the last-applied vertical geometry).

    Public API expected to be called from MainWindow:
      - set_y_range(ymin, ymax): share y-range from the right panel.
      - apply_right_geometry(ymin, ymax, top_px, bottom_px, canvas_h_px):
            pass the right panel's measured pixel geometry:
               * top_px, bottom_px: pixel margins above/below the axes box
               * canvas_h_px: pixel height of the right canvas
            SimPlot will replicate the same top position and axes height in pixels.
    """
    yRangeEdited = Signal(float, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._y_range: Optional[Tuple[float, float]] = None

        # Last geometry received from the right panel
        self._tgt_top_px: Optional[int] = None
        self._tgt_axes_h_px: Optional[int] = None  # = right_canvas_h - top_px - bottom_px

        self.fig = Figure(figsize=(5, 3), tight_layout=False)  # tight_layout disabled for manual margins
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        self._init_empty()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas, 1)
        
        self._interact = SimInteractor(self)
        self._interact.connect()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def set_y_range(self, ymin: float, ymax: float) -> None:
        """
        Apply shared y-range (from right panel). This does not yet change margins.
        Margins (top/bottom) are controlled by apply_right_geometry().
        """
        if ymin == ymax:
            pad = 1.0 if ymin == 0 else abs(ymin) * 0.1
            ymin, ymax = ymin - pad, ymax + pad
        self._y_range = (ymin, ymax)
        self.ax.set_ylim(ymin, ymax)
        self._enforce_equal_aspect()
        # Recompute xlim if vertical geometry already known
        if self._tgt_top_px is not None and self._tgt_axes_h_px is not None:
            self._recompute_xlim_to_fill_height()
        self.canvas.draw_idle()

    def apply_right_geometry(
        self,
        ymin: float,
        ymax: float,
        top_px: int,
        bottom_px: int,
        canvas_h_px: int,
    ) -> None:
        """
        Mirror the right panel's vertical geometry.

        Parameters
        ----------
        ymin, ymax : shared y-range (should match right panel)
        top_px     : pixel margin above axes box in right panel
        bottom_px  : pixel margin below axes box in right panel
        canvas_h_px: pixel height of the right panel's canvas
        """
        # 1) Fix y-range
        self.set_y_range(ymin, ymax)

        # 2) Compute target axes height in pixels (from the right panel)
        axes_h = max(1, int(canvas_h_px) - int(top_px) - int(bottom_px))
        self._tgt_top_px = int(top_px)
        self._tgt_axes_h_px = int(axes_h)

        # 3) Apply margins on *this* canvas so that:
        #       (top margin px) == right.top_px
        #       (axes height px) == right.axes_h
        #    => bottom_frac = 1 - top_frac - axes_h_frac
        H_left = max(1, self.canvas.height())
        top_frac = 1.0 - (float(self._tgt_top_px) / float(H_left))
        axes_frac = float(self._tgt_axes_h_px) / float(H_left)
        bottom_frac = top_frac - axes_frac

        # Clamp and maintain ordering: 0 <= bottom < top <= 1
        # If bottom would go negative (left canvas too short), keep top fixed and clamp bottom to 0.
        # (We still never shrink vertically; aspect=1 and y-lims are preserved.
        #  Horizontal trimming will compensate.)
        eps = 1e-4
        if bottom_frac < 0.0:
            bottom_frac = 0.0
            top_frac = max(eps, top_frac)  # keep top at requested position fraction

        top_frac = max(bottom_frac + eps, min(0.999, top_frac))

        # Disable auto layout; set manual margins
        self.fig.set_tight_layout(False)
        try:
            self.fig.subplots_adjust(top=top_frac, bottom=bottom_frac)
        except Exception:
            # In case of edge rounding issues, try a tiny relaxation
            try:
                self.fig.subplots_adjust(top=min(0.999, top_frac), bottom=max(0.0, bottom_frac))
            except Exception:
                pass

        # 4) Keep aspect 1:1 and recompute xlim to fill vertical extent
        self._enforce_equal_aspect()
        self._recompute_xlim_to_fill_height()
        self.canvas.draw_idle()

    def plot_sim_profile(
        self,
        xs: Iterable[float],
        zs: Iterable[float],
        *,
        label: str | None = None,
        clear: bool = True,
    ) -> None:
        """
        Plot simulation results. Vertical geometry is preserved exactly as previously applied.
        Horizontal span (xlim) is adapted to frame width (no shrinking).
        """
        if clear:
            self.ax.clear()
            self._decorate_axes()

        self.ax.plot(list(xs), list(zs), linewidth=1.8, label=label or "sim")
        if label:
            self.ax.legend(loc="best")

        if self._y_range:
            self.ax.set_ylim(*self._y_range)

        self._enforce_equal_aspect()
        self._recompute_xlim_to_fill_height()
        self.canvas.draw_idle()

    # ---------------------------------------------------------------------
    # Debug helper (concentric circles)
    # ---------------------------------------------------------------------
    def draw_concentric_circles(
        self,
        max_radius: int | None = None,
        center: tuple[float, float] = (0.0, 0.0),
        *,
        clear: bool = True,
    ) -> None:
        cx, cy = center

        # Decide max radius
        if max_radius is None:
            if self._y_range:
                ymin, ymax = self._y_range
                max_radius = max(1, int(np.floor(0.5 * abs(ymax - ymin))))
            else:
                max_radius = 10

        if clear:
            self.ax.clear()
            self._decorate_axes()

        # Honor shared y-range for vertical alignment
        if self._y_range:
            ymin, ymax = self._y_range
            self.ax.set_ylim(ymin, ymax)
            vspan = ymax - ymin
            half = 0.5 * vspan
            self.ax.set_xlim(cx - half, cx + half)
        else:
            extent = float(max_radius) + 1.0
            self.ax.set_xlim(cx - extent, cx + extent)
            self.ax.set_ylim(cy - extent, cy + extent)

        t = np.linspace(0.0, 2.0 * np.pi, 720)
        for r in range(1, max_radius + 1):
            x = cx + r * np.cos(t)
            y = cy + r * np.sin(t)
            self.ax.plot(x, y, linewidth=0.9, alpha=0.9)

        self._enforce_equal_aspect()
        self._recompute_xlim_to_fill_height()
        self.canvas.draw_idle()

    # ---------------------------------------------------------------------
    # Qt overrides
    # ---------------------------------------------------------------------
    def resizeEvent(self, event) -> None:  # noqa: N802
        """
        On resize, keep the last-applied vertical geometry (top position & axes height)
        and recompute xlim only to satisfy aspect=1 without shrinking.
        """
        self._apply_last_vertical_geometry()
        self._recompute_xlim_to_fill_height()
        self.canvas.draw_idle()
        super().resizeEvent(event)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def _init_empty(self) -> None:
        self.ax.clear()
        self._decorate_axes()
        self.ax.set_xlim(0.0, 10.0)
        self.ax.set_ylim(0.0, 10.0)
        self._enforce_equal_aspect()
        self.canvas.draw_idle()

    def _decorate_axes(self) -> None:
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("Horizontal distance")
        self.ax.set_ylabel("Elevation (m)")

    def _enforce_equal_aspect(self) -> None:
        # Keep strict 1:1 aspect; with 'box' we preserve the axes box,
        # while we control horizontal fit via xlim (no vertical shrinking).
        self.ax.set_aspect("equal", adjustable="datalim")

    def _apply_last_vertical_geometry(self) -> None:
        """
        Re-apply the most recently received vertical geometry (top position & axes height)
        on this canvas, expressed as figure fractions.
        """
        if self._tgt_top_px is None or self._tgt_axes_h_px is None:
            return

        H_left = max(1, self.canvas.height())
        top_frac = 1.0 - (float(self._tgt_top_px) / float(H_left))
        axes_frac = float(self._tgt_axes_h_px) / float(H_left)
        bottom_frac = top_frac - axes_frac

        eps = 1e-4
        if bottom_frac < 0.0:
            bottom_frac = 0.0
            top_frac = max(eps, top_frac)
        top_frac = max(bottom_frac + eps, min(0.999, top_frac))

        self.fig.set_tight_layout(False)
        try:
            self.fig.subplots_adjust(top=top_frac, bottom=bottom_frac)
        except Exception:
            try:
                self.fig.subplots_adjust(top=min(0.999, top_frac), bottom=max(0.0, bottom_frac))
            except Exception:
                pass

        self._enforce_equal_aspect()

    def _recompute_xlim_to_fill_height(self, center_x: Optional[float] = None) -> None:
        """
        With fixed y-range and fixed vertical geometry (top position & axes height),
        choose xlim so that aspect=1 holds and the axes fills the vertical extent.
        If center_x is provided, use it as horizontal zoom center.
        """
        if not self._y_range:
            return

        # 現在のレイアウト・ピクセル寸法
        self.fig.canvas.draw()
        bbox = self.ax.get_window_extent()
        dpr = float(getattr(self.canvas, "devicePixelRatioF", lambda: 1.0)())
        w_px = max(1.0, bbox.width / dpr)
        h_px = max(1.0, bbox.height / dpr)

        ymin, ymax = self._y_range
        yspan = max(1e-12, ymax - ymin)
        xspan_required = yspan * (w_px / h_px)

        cur_xlim = self.ax.get_xlim()
        if center_x is None or not np.isfinite(center_x):
            x_center = 0.5 * (cur_xlim[0] + cur_xlim[1])
        else:
            x_center = float(center_x)

        x0 = x_center - 0.5 * xspan_required
        x1 = x_center + 0.5 * xspan_required
        self.ax.set_xlim(x0, x1)
        self._enforce_equal_aspect()

