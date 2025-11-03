from __future__ import annotations

from typing import Iterable, Optional, Tuple

from PySide6.QtCore import Signal
from matplotlib.backend_bases import MouseButton
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .plot_interactions import SimInteractor

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

    def plot_polyline_with_phi(
        self,
        xs,
        ys,
        phi=None,
        *,
        clear: bool = True,
        shade_block_region: bool = True,
        block_level: float = 0.0,
        show_vertices: bool = False,    # show vertices
        vertex_size: int = 18,          # vertex marker size
        vertex_alpha: float = 0.9,      # vertex marker alpha
    ) -> None:
        import numpy as np

        x = np.asarray(list(xs), dtype=float)
        y = np.asarray(list(ys), dtype=float)
        n = len(x)
        if n == 0:
            return

        if clear:
            self.ax.clear()
            self._decorate_axes()

        # Decide plotting range (prioritize y-range)
        pad = 0.1
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        # Add a small margin to the y-range
        yr = ymax - ymin
        if yr <= 0:
            yr = 1.0
        ymin_plot = ymin - pad * yr
        ymax_plot = ymax + pad * yr
        self.set_y_range(ymin_plot, ymax_plot)

        # Shade y < block_level (light gray)
        if shade_block_region:
            # Shade the half-plane within current plot range
            self.ax.axhspan(ymin_plot, float(block_level), facecolor=(0.85, 0.85, 0.85), alpha=0.6, zorder=0)

        # Color by phi
        if phi is None:
            # If phi is None, draw single color (black)
            xy = np.column_stack([x, y])
            xy = np.vstack([xy, xy[0]])  # close polygon
            self.ax.plot(xy[:, 0], xy[:, 1], linewidth=1.8, color="black", label="polyline")
        else:
            phi = np.asarray(list(phi), dtype=int)
            assert len(phi) == n, "phi length must match x,y length"

            # Color segments by run-length (consider wrap from end to start)
            def seg_color(val: int) -> str:
                return "red" if int(val) == 1 else "blue"

            # Determine segment endpoints
            runs = []
            start = 0
            cur = phi[0]
            for i in range(1, n):
                if phi[i] != cur:
                    runs.append((start, i - 1, int(cur)))
                    start = i
                    cur = phi[i]
            runs.append((start, n - 1, int(cur)))  # final run

            # Wrap-around (merge if last and first are the same color)
            if len(runs) > 1 and runs[0][2] == runs[-1][2]:
                s0, e0, c0 = runs[0]
                s1, e1, c1 = runs[-1]
                # Merge and rotate to the front
                runs = [(s1, e0, c0)] + runs[1:-1]

            # Draw each run (closed polyline aware)
            for (s, e, c) in runs:
                if s <= e:
                    xs_seg = x[s:e+1]
                    ys_seg = y[s:e+1]
                    self.ax.plot(xs_seg, ys_seg, linewidth=2.0, color=seg_color(c))
                # segment junction
            # If last and first colors match, draw closing line
            if runs:
                last_e = runs[-1][1]
                first_s = runs[0][0]
                last_c = runs[-1][2]
                first_c = runs[0][2]
                if last_c == first_c:
                    self.ax.plot([x[last_e], x[first_s]], [y[last_e], y[first_s]], linewidth=2.0, color=seg_color(last_c))
                else:
                    # Colors differ; no connecting line needed
                    pass

        # Draw vertex dots
        if show_vertices:
            self._plot_vertices(xs, ys, phi, size=vertex_size, alpha=vertex_alpha, zorder=5)

        # Adjust display
        self._enforce_equal_aspect()
        self._recompute_xlim_to_fill_height()

        bh_text = f"{block_level:.3f}".rstrip("0").rstrip(".")
        self.ax.legend(handles=[
            self.ax.plot([], [], color="red",  linewidth=2.0, label="living")[0],
            self.ax.plot([], [], color="blue", linewidth=2.0, label="dead")[0],
        ], loc="upper right")
        self.canvas.draw_idle()


    def plot_polyline_step(
        self,
        old_xy: tuple,    # (xs_old, ys_old)
        new_xy: tuple,    # (xs_new, ys_new)
        new_phi=None,     # 0/1
        *,
        block_level: float = 0.0,
        clear: bool = True,
        show_vertices: bool = False,     # show vertex markers
        vertex_size_new: int = 18,       # new vertex size
        vertex_size_old: int = 10,       # old vertex size
        vertex_alpha: float = 0.9,       # vertex marker alpha
    ) -> None:
        """
        Draw old polyline as a thin line, and new polyline colored by φ. Shade y < BH in light gray.
        """
        import numpy as np

        x0 = np.asarray(old_xy[0], dtype=float)
        y0 = np.asarray(old_xy[1], dtype=float)
        x1 = np.asarray(new_xy[0], dtype=float)
        y1 = np.asarray(new_xy[1], dtype=float)

        if clear:
            self.ax.clear()
            self._decorate_axes()

        # Display range
        xmin = float(min(x0.min(), x1.min())); xmax = float(max(x0.max(), x1.max()))
        ymin = float(min(y0.min(), y1.min())); ymax = float(max(y0.max(), y1.max()))
        pad = 0.1 * max(1e-6, ymax - ymin)
        ymin_plot, ymax_plot = ymin - pad, ymax + pad
        self.set_y_range(ymin_plot, ymax_plot)

        # Shade blocked region y < BH
        self.ax.axhspan(ymin_plot, float(block_level), facecolor=(0.85, 0.85, 0.85), alpha=0.6, zorder=0)

        # Old polyline (thin gray)
        xy0 = np.column_stack([x0, y0])
        xy0 = np.vstack([xy0, xy0[0]])
        self.ax.plot(xy0[:, 0], xy0[:, 1], linewidth=1.0, color="0.5", label="previous")

        # Old vertices (small gray dots)
        if show_vertices:
            self._plot_vertices(x0, y0, phi=None, size=vertex_size_old, alpha=vertex_alpha, zorder=4, color="0.5")


        # New polyline (color by φ)
        if new_phi is None:
            xy1 = np.column_stack([x1, y1])
            xy1 = np.vstack([xy1, xy1[0]])
            self.ax.plot(xy1[:, 0], xy1[:, 1], linewidth=2.0, color="red", label="current")
        else:
            phi = np.asarray(new_phi, dtype=int)
            n = len(phi)
            def seg_color(val: int) -> str:
                return "red" if int(val) == 1 else "blue"
            # Split into runs and draw (wrap-aware)
            runs = []
            s = 0; cur = phi[0]
            for i in range(1, n):
                if phi[i] != cur:
                    runs.append((s, i - 1, int(cur))); s = i; cur = phi[i]
            runs.append((s, n - 1, int(cur)))
            if len(runs) > 1 and runs[0][2] == runs[-1][2]:
                s0, e0, c0 = runs[0]; s1, e1, c1 = runs[-1]
                runs = [(s1, e0, c0)] + runs[1:-1]
            for (s, e, c) in runs:
                xs_seg = x1[s:e+1]; ys_seg = y1[s:e+1]
                self.ax.plot(xs_seg, ys_seg, linewidth=2.0, color=seg_color(c))
            # Close connecting line
            if runs:
                e_last = runs[-1][1]; s_first = runs[0][0]
                if runs[-1][2] == runs[0][2]:
                    self.ax.plot([x1[e_last], x1[s_first]], [y1[e_last], y1[s_first]], linewidth=2.0, color=seg_color(runs[-1][2]))

        # New vertices (red/blue according to φ)
        if show_vertices:
            self._plot_vertices(x1, y1, phi=new_phi, size=vertex_size_new, alpha=vertex_alpha, zorder=6)

        self._enforce_equal_aspect()
        self._recompute_xlim_to_fill_height()
        self.ax.legend(loc="upper right")
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

        # Current layout pixel dimensions
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


    # Added within SimPlot class
    def _plot_vertices(
        self,
        xs,
        ys,
        phi=None,
        *,
        size: int = 18,
        alpha: float = 0.9,
        zorder: int = 4,
        color=None,
    ) -> None:
        """
        Render vertices as scatter points.
        - If phi is None: single color (given color or axis default)
        - If phi is 0/1: color points 1=red, 0=blue
        """
        import numpy as np
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)

        if phi is None:
            self.ax.scatter(x, y, s=size, alpha=alpha, zorder=zorder, color=color)
            return

        phi = np.asarray(phi, dtype=int)
        mask1 = (phi == 1)
        mask0 = ~mask1
        if mask1.any():
            self.ax.scatter(x[mask1], y[mask1], s=size, alpha=alpha, zorder=zorder, color="red")
        if mask0.any():
            self.ax.scatter(x[mask0], y[mask0], s=size, alpha=alpha, zorder=zorder, color="blue")
