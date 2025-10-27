from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class SimParams:
    """Container for simulation parameters.

    Keep it simple for now. Replace/extend fields as your model matures.
    """
    growth_rate_mm_yr: float = 8.0
    tidal_range_m: float = 2.7
    max_elevation_m: float = 1.5
    dt_years: float = 0.1
    n_steps: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Simulator:
    """A small, synchronous simulator placeholder.

    Replace the internals with your forward model for microatoll growth.
    For now it only returns a monotonically increasing dummy series so the
    GUI has something to draw.
    """

    def __init__(self, params: SimParams | None = None) -> None:
        self.params = params or SimParams()

    def set_params(self, params: SimParams) -> None:
        self.params = params

    def run(self) -> Dict[str, Any]:
        p = self.params
        # Dummy time series: height evolves with a capped linear rule.
        t = [i * p.dt_years for i in range(p.n_steps + 1)]
        h = []
        height = 0.0
        for _ in t:
            # Extremely naive growth law; clamp to max_elevation_m.
            height = min(height + (p.growth_rate_mm_yr / 1000.0) * p.dt_years, p.max_elevation_m)
            h.append(height)
        return {"t_years": t, "height_m": h, "params": p.to_dict()}
