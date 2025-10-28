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
    t1_years: float = 100.0   # end time
    record_every_years: float = 0.0  # 0 = 記録しない
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
        self._sl_t: Optional[np.ndarray] = None   # sea-level times (years)
        self._sl_h: Optional[np.ndarray] = None   # sea-level heights (m)

    def set_params(self, params: SimParams) -> None:
        self.params = params


    # ---- Sea-level curve API ----
    def set_sea_level_curve(self, times_years: ArrayLike, heights_m: ArrayLike) -> None:
        """
        海水準曲線 H(t) を設定（線形内挿）。times は単調増加が望ましい。
        """
        t = np.asarray(times_years, dtype=float).ravel()
        h = np.asarray(heights_m, dtype=float).ravel()
        if t.size < 2 or h.size != t.size:
            # 無効なら破棄
            self._sl_t, self._sl_h = None, None
            return
        # 時間でソートして保持
        order = np.argsort(t)
        self._sl_t = t[order]
        self._sl_h = h[order]

    def _sea_level_at(self, t_years: float) -> float | None:
        """
        線形内挿で H(t) を返す。曲線未設定なら None。
        範囲外は端の値で外挿（np.interp の仕様）。
        """
        if self._sl_t is None or self._sl_h is None:
            return None
        return float(np.interp(float(t_years), self._sl_t, self._sl_h))

    def _sea_level_min_between(self, t_prev: float, t_now: float) -> float | None:
        """
        指定した期間 [t_prev, t_now] 内の海水準の最小値を返す。
        - self._sl_t, self._sl_h が未設定なら None。
        - 区間にデータ点が存在しない場合も None を返す（＝海水準ブロックを適用しない）。
        - t_prev > t_now の場合は自動的に入れ替える。
        """
        if self._sl_t is None or self._sl_h is None:
            return None

        t1, t2 = sorted((float(t_prev), float(t_now)))

        # 区間に含まれる観測データを抽出
        mask = (self._sl_t >= t1) & (self._sl_t <= t2)
        if not np.any(mask):
            return None  # データ点なし → ブロックしない

        return float(np.min(self._sl_h[mask]))

    # ---- init / state ----
    def _ensure_initialized(self) -> None:
        if self._state:
            return
        p = self.params
        x, y = circle_by_spacing(p.vertex_spacing_m, radius=0.2)
        y = y + float(p.base_height)
        # τ=0 時刻での海水準を取得
        t_init = float(p.t0_years) + float(p.dt_years) * 0
        H = self._sea_level_at(t_init)
        phi = phi_by_block_level(y, p.base_height, sea_level=H)
        self._state = {"x": x, "y": y, "phi": phi, "tau": 0}

    def initialize(self) -> Dict[str, Any]:
        p = self.params
        x, y = circle_by_spacing(p.vertex_spacing_m, radius=0.2)
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

        # 1) φ=1のみ外向き法線で移動
        nx, ny = outward_normals_closed(x, y)
        V = float(p.growth_rate_mm_yr) * float(p.dt_years) / 1000.0
        mov = (phi_old == 1)
        x_new = x.copy(); y_new = y.copy()
        x_new[mov] += V * nx[mov]
        y_new[mov] += V * ny[mov]

        # 2) φ=1領域のみリサンプリング（端点φ0保持・内部=1）
        if p.resample_each_step:
            x_tmp, y_tmp, phi_tmp = resample_polyline_movable_regions(
                x_new, y_new, phi_old, spacing_m=p.vertex_spacing_m
            )
        else:
            x_tmp, y_tmp, phi_tmp = x_new, y_new, phi_old

        # 3) 不可逆 φ 更新（★軽量化：φ=1の候補だけ判定）
        tau_prev = int(self._state["tau"])
        t_prev = float(p.t0_years) + float(p.dt_years) * tau_prev
        tau_new = tau_prev + 1
        t_new = float(p.t0_years) + float(p.dt_years) * tau_new

        # 区間 [t_prev, t_new] の最小海水準を取得
        H_min = self._sea_level_min_between(t_prev, t_new)

        # 更新対象のインデックス（phi_tmp==1 の点だけ）
        cand_idx = np.flatnonzero(phi_tmp == 1)

        # 最終φをゼロで初期化（不可逆：0は必ず0のまま）
        phi_final = np.zeros_like(phi_tmp, dtype=int)

        if cand_idx.size > 0:
            # サブセット座標
            xs = x_tmp[cand_idx]
            ys = y_tmp[cand_idx]

            # 可動域 [BH, H(t)] の判定（サブセットのみ）
            allow_cand = allowed_mask(ys, p.base_height, H_min)  # True/False (len = |cand|)

            # 旧ポリライン内部ブロック（サブセットのみ）
            inside_old_cand = inside_closed_polyline_mask(
                px=xs, py=ys, poly_x=x, poly_y=y
            )

            # allowed かつ outside(old) の点だけ 1 を残す
            keep = allow_cand & (~inside_old_cand)
            if np.any(keep):
                phi_final[cand_idx[keep]] = 1

        # 状態更新
        self._state = {"x": x_tmp, "y": y_tmp, "phi": phi_final, "tau": tau_new}

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
