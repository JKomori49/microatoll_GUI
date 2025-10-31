# src/simulator/iteration.py
from __future__ import annotations
from typing import Dict, Any, List
import math
import numpy as np

from .simulator import Simulator

class IterativeRunner:
    """
    Run simulation until T1 and record intermediate polylines/metrics
    every `record_every_years`.
    Records include polyline snapshot and HLG (max y where phi==1).
    """

    def __init__(self, sim: Simulator) -> None:
        self.sim = sim
        self.records: List[Dict[str, Any]] = []
        self.hlg_times: List[float] = []
        self.hlg_values: List[float] = []

    def run_until_end(self) -> Dict[str, Any]:
        p = self.sim.params
        if p.dt_years <= 0.0:
            raise ValueError("Δt must be > 0.")
        if p.t0_years >= p.t1_years:
            raise ValueError("T1 must be greater than T0.")

        n_steps = int(math.ceil((p.t1_years - p.t0_years) / p.dt_years))
        if n_steps <= 0:
            n_steps = 1

        record_every = float(getattr(p, "record_every_years", 0.0))
        next_record_time = (p.t0_years + record_every) if record_every > 0 else None

        result = None
        for _ in range(n_steps):
            result = self.sim.step_once()
            cur = result["new"]
            cur_t = float(cur["t_years"])

            if record_every > 0 and next_record_time is not None and cur_t >= next_record_time:
                x = np.asarray(cur["x"], dtype=float)
                y = np.asarray(cur["y"], dtype=float)
                phi = np.asarray(cur.get("phi"), dtype=int)

                # HLG: φ==1 に限定したときの y 最大値（存在しなければ None）
                hlg_val = None
                if phi.size and np.any(phi == 1):
                    hlg_val = float(y[phi == 1].max())

                self.records.append({
                    "x": x.copy(),
                    "y": y.copy(),
                    "phi": phi.copy(),
                    "t_years": cur_t,
                    "hlg": hlg_val,
                })
                if hlg_val is not None:
                    self.hlg_times.append(cur_t)
                    self.hlg_values.append(hlg_val)

                next_record_time += record_every

        return {
            "final": result,                     # 最後の step_once() 結果
            "records": self.records,             # 途中ポリラインのスナップショット
            "hlg": {"t": self.hlg_times, "y": self.hlg_values},  # 時系列HLG
        }
