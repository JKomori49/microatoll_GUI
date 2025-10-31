# gui/plot_interactions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib.backend_bases import MouseButton


@dataclass
class _State:
    is_panning: bool = False
    last_x: Optional[float] = None
    last_y: Optional[float] = None
    zoom_factor_per_notch: float = 1.2


class _BaseInteractor:
    """
    Common glue for mpl <-> QWidget owners that expose:
      - owner.ax, owner.fig, owner.canvas  (matplotlib objects)
      - owner._y_range: Optional[tuple[float, float]]
    """

    def __init__(self, owner) -> None:
        self.o = owner
        self.state = _State()
        self._cids: list[int] = []

    def connect(self) -> None:
        c = self.o.canvas.mpl_connect
        self._cids = [
            c("scroll_event", self._on_scroll),
            c("button_press_event", self._on_press),
            c("motion_notify_event", self._on_move),
            c("button_release_event", self._on_release),
        ]

    def disconnect(self) -> None:
        for cid in self._cids:
            try:
                self.o.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._cids.clear()

    # ---- utilities ----
    def _dpi_ratio(self) -> float:
        return float(getattr(self.o.canvas, "devicePixelRatioF", lambda: 1.0)())

    def _axes_bbox_logical_px(self):
        # Matplotlib returns device px; convert to logical px to match Qt
        self.o.fig.canvas.draw()
        dpr = self._dpi_ratio()
        bbox = self.o.ax.get_window_extent()
        return bbox.width / dpr, bbox.height / dpr

    # ---- event stubs to override ----
    def _on_scroll(self, event): ...
    def _on_press(self, event): ...
    def _on_move(self, event): ...
    def _on_release(self, event): ...


class SeaLevelInteractor(_BaseInteractor):
    """
    Right panel behavior:
      - Wheel: zoom Y only (xlim locked)
      - Right-drag: pan Y only (xlim locked)
      - Always emit yRangeChanged and broadcast geometry to left
    """

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.o.ax:
            return
        ymin, ymax = self.o.ax.get_ylim()
        yspan = ymax - ymin
        if yspan <= 0:
            return
        s = self.state.zoom_factor_per_notch if event.step > 0 else 1.0 / self.state.zoom_factor_per_notch

        # Zoom around cursor height; spec only required Y zoom
        y0 = float(event.ydata) if event.ydata is not None else 0.5 * (ymin + ymax)
        ymin_new = y0 - (y0 - ymin) / s
        ymax_new = y0 + (ymax - y0) / s
        if (ymax_new - ymin_new) <= 1e-9:
            return

        self.o.ax.set_ylim(ymin_new, ymax_new)
        self.o.canvas.draw_idle()
        self.o._y_range = (ymin_new, ymax_new)
        # Notify left: keep y in sync & geometry aligned
        self.o.yRangeChanged.emit(ymin_new, ymax_new)
        self.o._broadcast_geometry()

    def _on_press(self, event) -> None:
        if event.inaxes is not self.o.ax:
            return
        if event.button == MouseButton.RIGHT:
            self.state.is_panning = True
            self.state.last_x, self.state.last_y = event.x, event.y

    def _on_move(self, event) -> None:
        if not self.state.is_panning or event.inaxes is not self.o.ax:
            return
        if self.state.last_x is None or self.state.last_y is None:
            return

        dx_px = event.x - self.state.last_x
        dy_px = event.y - self.state.last_y
        self.state.last_x, self.state.last_y = event.x, event.y

        # pan vertically only
        _, h_px = self._axes_bbox_logical_px()
        ymin, ymax = self.o.ax.get_ylim()
        yspan = ymax - ymin
        if yspan <= 0 or h_px <= 0:
            return

        dy_data = -dy_px * (yspan / h_px)
        self.o.ax.set_ylim(ymin + dy_data, ymax + dy_data)
        self.o.canvas.draw_idle()

        self.o._y_range = (ymin + dy_data, ymax + dy_data)
        self.o.yRangeChanged.emit(*self.o._y_range)
        self.o._broadcast_geometry()

    def _on_release(self, event) -> None:
        if event.button == MouseButton.RIGHT:
            self.state.is_panning = False
            self.state.last_x = self.state.last_y = None


class SimInteractor(_BaseInteractor):
    """
    Left panel behavior:
      - Wheel: zoom Y around view center (ignore mouse), keep aspect=1 and recompute xlim
      - Right-drag: pan both X and Y; then refit x for aspect=1
      - Emit yRangeEdited so right can sync Y; left keeps x independently
      - Requires owner._recompute_xlim_to_fill_height(center_x: Optional[float])
    """

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.o.ax or not self.o._y_range:
            return
        ymin, ymax = self.o._y_range
        yspan = ymax - ymin
        if yspan <= 0:
            return
        s = self.state.zoom_factor_per_notch if event.step > 0 else 1.0 / self.state.zoom_factor_per_notch

        # Center-based zoom (ignore cursor)
        y_center = 0.5 * (ymin + ymax)
        ymin_new = y_center - (y_center - ymin) / s
        ymax_new = y_center + (ymax - y_center) / s
        if (ymax_new - ymin_new) <= 1e-9:
            return

        # Apply Y then fit X for aspect=1
        self.o._y_range = (ymin_new, ymax_new)
        self.o.ax.set_ylim(ymin_new, ymax_new)
        self.o._recompute_xlim_to_fill_height(center_x=None)
        self.o.canvas.draw_idle()

        # Notify right Y
        if hasattr(self.o, "yRangeEdited"):
            self.o.yRangeEdited.emit(ymin_new, ymax_new)

    def _on_press(self, event) -> None:
        if event.inaxes is not self.o.ax:
            return
        if event.button == MouseButton.RIGHT:
            self.state.is_panning = True
            self.state.last_x, self.state.last_y = event.x, event.y

    def _on_move(self, event) -> None:
        if not self.state.is_panning or event.inaxes is not self.o.ax:
            return
        if self.state.last_x is None or self.state.last_y is None:
            return

        dx_px = event.x - self.state.last_x
        dy_px = event.y - self.state.last_y
        self.state.last_x, self.state.last_y = event.x, event.y

        w_px, h_px = self._axes_bbox_logical_px()
        ymin, ymax = self.o.ax.get_ylim()
        x0, x1 = self.o.ax.get_xlim()
        yspan = ymax - ymin
        xspan = x1 - x0
        if yspan <= 0 or xspan <= 0 or w_px <= 0 or h_px <= 0:
            return

        dx_data = -dx_px * (xspan / w_px)
        dy_data = -dy_px * (yspan / h_px)

        self.o.ax.set_ylim(ymin + dy_data, ymax + dy_data)
        self.o.ax.set_xlim(x0 + dx_data, x1 + dx_data)
        self.o._y_range = (ymin + dy_data, ymax + dy_data)

        # Refit x to maintain aspect=1
        self.o._recompute_xlim_to_fill_height(center_x=None)
        self.o.canvas.draw_idle()

        if hasattr(self.o, "yRangeEdited"):
            self.o.yRangeEdited.emit(*self.o._y_range)

    def _on_release(self, event) -> None:
        if event.button == MouseButton.RIGHT:
            self.state.is_panning = False
            self.state.last_x = self.state.last_y = None
