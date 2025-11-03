# src/simulator.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import numpy as np
from numpy.typing import ArrayLike

from .polyline_utils import (
    circle_by_spacing,
    phi_by_block_level,
    outward_normals_closed,
    resample_polyline_movable_regions,
    allowed_mask,
    inside_closed_polyline_mask,   
)


@dataclass
class SimParams:
    growth_rate_mm_yr: float = 8.0
    tidal_range_m: float = 2.7
    max_elevation_m: float = 1.5
    dt_years: float = 0.1         # Δt
    base_height: float = 0.0      # BH
    t0_years: float = 0.0         # T0
    t1_years: float = 100.0       # end time
    record_every_years: float = 0.0  # 0 = do not record
    vertex_spacing_m: float = 0.05
    resample_each_step: bool = True  # resampling ON/OFF each step
    initial_size_m: float = 0.2 

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Simulator:
    """
    - Shape manipulations are in polyline_utils
    - This class manages parameters and time evolution (τ, t) only
    """
    def __init__(self, params: Optional[SimParams] = None) -> None:
        self.params = params or SimParams()
        self._state: Dict[str, Any] = {}  # {"x","y","phi","tau"}
        self._sl_t: Optional[np.ndarray] = None   # sea-level times (years)
        self._sl_h: Optional[np.ndarray] = None   # sea-level heights (m)

    def set_params(self, params: SimParams) -> None:
        self.params = params


    # ---- Sea-level curve API ----
    def set_sea_level_curve(self, times_years: ArrayLike, heights_m: ArrayLike) -> None:
        """
        Set sea-level curve H(t) using linear interpolation. Times should be monotonically increasing.
        """
        t = np.asarray(times_years, dtype=float).ravel()
        h = np.asarray(heights_m, dtype=float).ravel()
        if t.size < 2 or h.size != t.size:
            # Invalid -> discard
            self._sl_t, self._sl_h = None, None
            return
        # Sort by time and store
        order = np.argsort(t)
        self._sl_t = t[order]
        self._sl_h = h[order]

    def _sea_level_at(self, t_years: float) -> float | None:
        """
        Return H(t) via linear interpolation. If curve unset, return None.
        Out-of-range values extrapolate to end values (np.interp behavior).
        """
        if self._sl_t is None or self._sl_h is None:
            return None
        return float(np.interp(float(t_years), self._sl_t, self._sl_h))

    def _sea_level_min_between(self, t_prev: float, t_now: float) -> float | None:
        """
        Return the minimum sea level within [t_prev, t_now].
        - Returns None if self._sl_t/_sl_h are unset.
        - Returns None if the interval has no data points (i.e., do not apply blocking).
        - If t_prev > t_now, they are swapped.
        """
        if self._sl_t is None or self._sl_h is None:
            return None

        t1, t2 = sorted((float(t_prev), float(t_now)))

        # Extract observed data within the interval
        mask = (self._sl_t >= t1) & (self._sl_t <= t2)
        if not np.any(mask):
            return None  # No data points → do not block

        return float(np.min(self._sl_h[mask]))
    
    # ---- init shape helper: upper semicircle closed with chord ----
    @staticmethod
    def _upper_semicircle_closed(spacing: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate the upper semicircle (0..π) as an arc and close it with a chord between endpoints.
        Returns (x, y). Center is (0, 0); caller shifts by adding BH as needed.
        """
        r = float(radius)
        s = max(float(spacing), 1e-6)

        # Arc length: πr; choose number of points for approx. equal spacing
        n_arc = max(16, int(np.ceil((np.pi * r) / s)))
        # Angle 0..π (upper half), CCW
        theta = np.linspace(0.0, np.pi, n_arc, endpoint=True)
        x_arc = r * np.cos(theta)         # x: +R → -R
        y_arc = r * np.sin(theta)         # y: 0 → +R

        # Connect right end (-R,0) → left end (+R,0) with a straight chord
        # Chord length: 2R; number of points:
        n_chord = max(4, int(np.ceil((2.0 * r) / s)))
        x_chord = np.linspace(-r, r, n_chord, endpoint=False)  # endpoint=False to avoid duplicate with arc start (-r,0)
        y_chord = np.zeros_like(x_chord)

        x = np.concatenate([x_arc, x_chord])
        y = np.concatenate([y_arc, y_chord])
        return x, y

    # ---- init / state ----
    def _ensure_initialized(self) -> None:
        if self._state:
            return
        p = self.params
        x, y = self._upper_semicircle_closed(p.vertex_spacing_m, p.initial_size_m)
        y = y + float(p.base_height)
        # Get sea level at τ=0 time
        t_init = float(p.t0_years) + float(p.dt_years) * 0
        H = self._sea_level_at(t_init)
        phi = phi_by_block_level(y, p.base_height, sea_level=H)
        self._state = {"x": x, "y": y, "phi": phi, "tau": 0}

    def initialize(self) -> Dict[str, Any]:
        p = self.params
        x, y = self._upper_semicircle_closed(p.vertex_spacing_m, p.initial_size_m)
        y = y + float(p.base_height)
        t_init = float(p.t0_years)
        H = self._sea_level_at(t_init)
        phi0 = allowed_mask(y, p.base_height, H).astype(int)
        self._state = {"x": x, "y": y, "phi": phi0, "tau": 0}
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
        x = old["x"]; y = old["y"]; phi_old = old["phi"]

        # 1) Move only where φ==1 along outward normals
        nx, ny = outward_normals_closed(x, y)
        V = float(p.growth_rate_mm_yr) * float(p.dt_years) / 1000.0
        mov = (phi_old == 1)
        x_new = x.copy(); y_new = y.copy()
        x_new[mov] += V * nx[mov]
        y_new[mov] += V * ny[mov]

        # 2) Resample only φ==1 regions (keep endpoints φ=0; interior=1)
        if p.resample_each_step:
            x_tmp, y_tmp, phi_tmp = resample_polyline_movable_regions(
                x_new, y_new, phi_old, spacing_m=p.vertex_spacing_m
            )
        else:
            x_tmp, y_tmp, phi_tmp = x_new, y_new, phi_old

        # 3) Irreversible φ update (optimize: evaluate only φ==1 candidates)
        tau_prev = int(self._state["tau"])
        t_prev = float(p.t0_years) + float(p.dt_years) * tau_prev
        tau_new = tau_prev + 1
        t_new = float(p.t0_years) + float(p.dt_years) * tau_new

        # Get minimum sea level within [t_prev, t_new]
        H_min = self._sea_level_min_between(t_prev, t_new)

        # Indices to update (only points with phi_tmp==1)
        cand_idx = np.flatnonzero(phi_tmp == 1)

        # Initialize final φ to zero (irreversible: zeros remain zero)
        phi_final = np.zeros_like(phi_tmp, dtype=int)

        if cand_idx.size > 0:
            # Subset coordinates
            xs = x_tmp[cand_idx]
            ys = y_tmp[cand_idx]

            # Allowed range [BH, H(t)] check (subset only)
            allow_cand = allowed_mask(ys, p.base_height, H_min)  # True/False (len = |cand|)

            # Block points inside the old polyline (subset only)
            inside_old_cand = inside_closed_polyline_mask(
                px=xs, py=ys, poly_x=x, poly_y=y
            )

            # Keep 1 only for points that are allowed and outside(old)
            keep = allow_cand & (~inside_old_cand)
            if np.any(keep):
                phi_final[cand_idx[keep]] = 1

        # 状態更新
        self._state = {"x": x_tmp, "y": y_tmp, "phi": phi_final, "tau": tau_new}

        new = self.current_polyline()
        return {"old": old, "new": new, "params": p.to_dict()}

    # Compatibility: return current state only
    def run(self) -> Dict[str, Any]:
        cur = self.current_polyline()
        return {
            "polyline": {"x": cur["x"].tolist(), "y": cur["y"].tolist(), "phi": cur["phi"].tolist()},
            "params": self.params.to_dict(),
            "t_years": [cur["t_years"]],
            "height_m": cur["y"].tolist(),
        }
