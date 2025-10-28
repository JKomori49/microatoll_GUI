# src/simulator.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import numpy as np

from .polyline_utils import (
    circle_by_spacing,
    phi_by_block_level,
    outward_normals_closed,
    resample_polyline_movable_regions,
)


@dataclass
class SimParams:
    growth_rate_mm_yr: float = 8.0
    tidal_range_m: float = 2.7
    max_elevation_m: float = 1.5
    dt_years: float = 0.1         # Δt
    n_steps: int = 50
    base_height: float = 0.0      # BH
    t0_years: float = 0.0         # T0
    vertex_spacing_m: float = 0.05
    resample_each_step: bool = True  # 追加: 毎ステップのリサンプリングON/OFF

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Simulator:
    """
    - 形状操作は polyline_utils に集約
    - ここはパラメータ管理と時間発展（τ、t）のみを担当
    """
    def __init__(self, params: Optional[SimParams] = None) -> None:
        self.params = params or SimParams()
        self._state: Dict[str, Any] = {}  # {"x","y","phi","tau"}

    def set_params(self, params: SimParams) -> None:
        self.params = params

    # ---- init / state ----
    def _ensure_initialized(self) -> None:
        if self._state:
            return
        p = self.params
        x, y = circle_by_spacing(p.vertex_spacing_m, radius=1.0)
        y = y + float(p.base_height)
        phi = phi_by_block_level(y, p.base_height)
        self._state = {"x": x, "y": y, "phi": phi, "tau": 0}

    def initialize(self) -> Dict[str, Any]:
        p = self.params
        x, y = circle_by_spacing(p.vertex_spacing_m, radius=1.0)
        y = y + float(p.base_height)
        phi = phi_by_block_level(y, p.base_height)
        self._state = {"x": x, "y": y, "phi": phi, "tau": 0}
        return self.current_polyline()

    def reset(self) -> Dict[str, Any]:
        return self.initialize()

    def current_polyline(self) -> Dict[str, Any]:
        self._ensure_initialized()
        x = self._state["x"]; y = self._state["y"]; phi = self._state["phi"]; tau = int(self._state["tau"])
        t = float(self.params.t0_years) + float(self.params.dt_years) * tau
        return {"x": x.copy(), "y": y.copy(), "phi": phi.copy(), "tau": tau, "t_years": t}

    # ---- time stepping ----
    def step_once(self) -> Dict[str, Any]:
        self._ensure_initialized()
        p = self.params

        old = self.current_polyline()
        x = old["x"]; y = old["y"]; phi = old["phi"]

        nx, ny = outward_normals_closed(x, y)

        V = float(p.growth_rate_mm_yr) * float(p.dt_years) / 1000.0  # mm/yr → m/step
        mask = (phi == 1)
        x_new = x.copy(); y_new = y.copy()
        x_new[mask] += V * nx[mask]
        y_new[mask] += V * ny[mask]

        # φの更新（運用ポリシーに応じて切替）
        phi_new = phi_by_block_level(y_new, p.base_height)

        # リサンプル（可動区間のみ）
        if p.resample_each_step:
            x_new, y_new, phi_new = resample_polyline_movable_regions(
                x_new, y_new, phi_new, spacing_m=p.vertex_spacing_m
            )

        self._state = {"x": x_new, "y": y_new, "phi": phi_new, "tau": int(self._state["tau"]) + 1}

        new = self.current_polyline()
        return {"old": old, "new": new, "params": p.to_dict()}

    # 互換: 現在の状態のみを返す
    def run(self) -> Dict[str, Any]:
        cur = self.current_polyline()
        return {
            "polyline": {"x": cur["x"].tolist(), "y": cur["y"].tolist(), "phi": cur["phi"].tolist()},
            "params": self.params.to_dict(),
            "t_years": [cur["t_years"]],
            "height_m": cur["y"].tolist(),
        }
