from __future__ import annotations
from typing import Dict, Any, List
import math

from simulator.simulator import Simulator, SimParams


class IterativeRunner:
    """
    Run simulation until T1 and optionally record intermediate polylines
    every `record_every_years`.
    """

    def __init__(self, sim: Simulator) -> None:
        self.sim = sim
        self.records: List[Dict[str, Any]] = []

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
        next_record_time = p.t0_years + record_every if record_every > 0 else None

        result = None
        for _ in range(n_steps):
            result = self.sim.step_once()
            cur_t = result["new"]["t_years"]

            # --- 記録条件 ---
            if record_every > 0 and next_record_time is not None and cur_t >= next_record_time:
                self.records.append({
                    "x": result["new"]["x"],
                    "y": result["new"]["y"],
                    "t_years": cur_t
                })
                # 次の記録タイミングを更新
                next_record_time += record_every

        return {
            "final": result,
            "records": self.records,
        }
