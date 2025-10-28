# src/polyline_utils.py
from __future__ import annotations
from typing import Tuple, List
import numpy as np
import math


# -------- 基本生成 --------
def unit_circle(num_points: int = 200, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    n = max(4, int(num_points))
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    return x.astype(float), y.astype(float)


def circle_by_spacing(spacing_m: float, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """周長 / spacing から点数を決めて円を離散化（初期形状向け）。"""
    spacing = max(1e-6, float(spacing_m))
    circumference = 2.0 * math.pi * float(radius)
    n = max(4, int(round(circumference / spacing)))
    return unit_circle(n, radius)


# -------- φ/ブロック判定 --------
def phi_by_block_level(y: np.ndarray, bh: float, sea_level: float | None = None) -> np.ndarray:
    """
    ブロック条件:
      - y < BH  → 0（ブロック）
      - sea_level が与えられる場合、y > sea_level → 0（ブロック）
      - それ以外 → 1（可動）
    """
    yy = np.asarray(y, dtype=float)
    bhf = float(bh)
    if sea_level is None:
        return (yy >= bhf).astype(int)
    # 可動域は [BH, H(t)] の区間
    return ((yy >= bhf) & (yy <= float(sea_level))).astype(int)


def _close_ring(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x,y) を終点=始点でクローズ。"""
    if x[0] == x[-1] and y[0] == y[-1]:
        return x, y
    return np.r_[x, x[0]], np.r_[y, y[0]]

def _point_segment_dist2(px, py, x1, y1, x2, y2):
    """
    点列 p=(px,py) と線分 (x1,y1)-(x2,y2) の二乗距離をベクトル化して返す。
    px,py は配列可。戻り値は px と同形の ndarray。
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    vx, vy = float(x2 - x1), float(y2 - y1)
    wx, wy = px - float(x1), py - float(y1)
    vv = vx * vx + vy * vy
    if vv == 0.0:
        return (wx * wx + wy * wy)
    t = (wx * vx + wy * vy) / vv
    t = np.clip(t, 0.0, 1.0)
    dx = wx - t * vx
    dy = wy - t * vy
    return dx * dx + dy * dy

def inside_closed_polyline_mask(
    px: np.ndarray, py: np.ndarray, poly_x: np.ndarray, poly_y: np.ndarray, *, boundary_eps: float = 1e-9
) -> np.ndarray:
    """
    点列 (px,py) が閉曲線 poly の内部にあるかを even-odd ルールで判定。
    - 境界上（線分からの距離 <= eps）は True とする（ブロック扱い）。
    - poly は単純閉曲線（自己交差なし）を前提。
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    vx, vy = np.asarray(poly_x, dtype=float), np.asarray(poly_y, dtype=float)
    vx, vy = _close_ring(vx, vy)
    n = len(vx) - 1

    # 1) 境界判定（最小距離）
    eps2 = (float(boundary_eps) ** 2)
    min_d2 = np.full(px.shape, np.inf, dtype=float)
    for i in range(n):
        d2 = _point_segment_dist2(px, py, vx[i], vy[i], vx[i+1], vy[i+1])
        min_d2 = np.minimum(min_d2, d2)
    on_boundary = (min_d2 <= eps2)

    # 2) 射線法（x 正方向に半直線）
    inside = np.zeros_like(px, dtype=bool)
    for i in range(n):
        x1, y1, x2, y2 = vx[i], vy[i], vx[i+1], vy[i+1]
        # py が [min(y1,y2), max(y1,y2)) にある辺のみ対象（半開区間で二重カウント防止）
        cond_y = ((y1 <= py) & (py < y2)) | ((y2 <= py) & (py < y1))
        # 交点の x 座標（分母0は cond_y が偽になるので気にしない）
        x_int = x1 + (py - y1) * (x2 - x1) / (y2 - y1 + 1e-300)
        cross = cond_y & (px < x_int)
        inside ^= cross

    return on_boundary | inside


# 許容域マスク（可動域）
def allowed_mask(y: np.ndarray, bh: float, sea_level: float | None) -> np.ndarray:
    """
    可動域 [BH, H(t)] を True、それ以外 False にするブールマスク。
    sea_level が None の時は [BH, +∞)。
    """
    yy = np.asarray(y, dtype=float)
    if sea_level is None:
        return yy >= float(bh)
    return (yy >= float(bh)) & (yy <= float(sea_level))

# -------- 法線（外向き） --------
def outward_normals_closed(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    閉ポリラインの頂点法線（単位ベクトル）を“常に外向き”に揃えて返す。
    - 前後接線の平均→左法線 n=(-ty, tx)
    - 重心ベクトル r と n の平均内積で内外判定し、内向きなら一括反転
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        nx = np.zeros_like(x); ny = np.ones_like(y)
        return nx, ny

    dx_f = np.roll(x, -1) - x
    dy_f = np.roll(y, -1) - y
    dx_b = x - np.roll(x, 1)
    dy_b = y - np.roll(y, 1)

    tx = 0.5 * (dx_f + dx_b)
    ty = 0.5 * (dy_f + dy_b)

    nx = -ty
    ny = tx

    lens = np.hypot(nx, ny)
    lens[lens == 0.0] = 1.0
    nx /= lens
    ny /= lens

    cx = float(np.mean(x)); cy = float(np.mean(y))
    rx = x - cx; ry = y - cy
    mean_dot = float(np.mean(rx * nx + ry * ny))
    if mean_dot < 0.0:
        nx = -nx; ny = -ny
    return nx, ny


# -------- リサンプリング（開区間ユーティリティ） --------
def _segment_cumlen(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    dx = np.diff(xs, prepend=xs[0])
    dy = np.diff(ys, prepend=ys[0])
    d = np.hypot(dx, dy); d[0] = 0.0
    return np.cumsum(d)


def resample_open_segment_by_spacing(xs: np.ndarray, ys: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """
    開区間の折れ線を端点固定で等間隔化（線形補間）。端点を必ず含む。
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    assert xs.size == ys.size and xs.size >= 2
    spacing = float(max(spacing, 1e-9))

    s = _segment_cumlen(xs, ys)
    L = float(s[-1])
    if L == 0.0:
        return xs.copy(), ys.copy()

    nint = max(0, int(round(L / spacing)) - 1)  # 内点数
    samples = np.linspace(0.0, L, num=nint + 2)  # [0, L] 端点含む
    xi = np.interp(samples, s, xs)
    yi = np.interp(samples, s, ys)
    return xi, yi


# -------- リサンプリング（φ=1領域だけ） --------
def resample_polyline_movable_regions(
    x: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    クローズド円周状のポリラインで、φ=1 の連続区間のみを spacing で等間隔再サンプル。
    - φ=0（不動点）は位置・ラベルとも必ず保持
    - 区間端（隣接するφ=0）はアンカーとして含め、内部のみ等間隔化（端点はφ=0、内部はφ=1）
    - すべて φ=1 の場合はループ全体を spacing 等間隔化（φ=1）
    - 出力は開表現（先頭重複なし）
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    phi = np.asarray(phi, dtype=int)
    assert x.ndim == y.ndim == phi.ndim == 1 and len(x) == len(y) == len(phi) >= 3

    spacing = float(max(spacing_m, 1e-9))

    # 全部 φ=1 → 全体を再サンプル
    if np.all(phi == 1):
        xx = np.r_[x, x[0]]; yy = np.r_[y, y[0]]
        d = np.hypot(np.diff(xx), np.diff(yy))
        s = np.r_[0.0, np.cumsum(d)]
        L = float(s[-1])
        if L == 0.0:
            return x.copy(), y.copy(), phi.copy()
        k = max(4, int(round(L / spacing)))
        samples = np.linspace(0.0, L, num=k, endpoint=False)
        xi = np.interp(samples, s, xx)
        yi = np.interp(samples, s, yy)
        return xi, yi, np.ones_like(xi, dtype=int)

    anchors = np.flatnonzero(phi == 0)
    out_x: List[float] = []; out_y: List[float] = []; out_p: List[int] = []

    def slice_cyclic(i: int, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if i <= j:
            return x[i:j+1], y[i:j+1], phi[i:j+1]
        return np.r_[x[i:], x[:j+1]], np.r_[y[i:], y[:j+1]], np.r_[phi[i:], phi[:j+1]]

    for idx, a in enumerate(anchors):
        b = anchors[(idx + 1) % len(anchors)]
        xs, ys, ps = slice_cyclic(a, b)

        if xs.size <= 2:
            if not (len(out_x) and out_x[-1] == x[a] and out_y[-1] == y[a]):
                out_x.append(float(x[a])); out_y.append(float(y[a])); out_p.append(0)
            continue

        inner_phi = ps[1:-1]
        if np.any(inner_phi == 1):
            xi, yi = resample_open_segment_by_spacing(xs, ys, spacing)
            pi = np.ones_like(xi, dtype=int)
            pi[0] = 0; pi[-1] = 0
            if len(out_x):
                if np.isclose(out_x[-1], xi[0]) and np.isclose(out_y[-1], yi[0]):
                    xi = xi[1:]; yi = yi[1:]; pi = pi[1:]
            out_x.extend(map(float, xi)); out_y.extend(map(float, yi)); out_p.extend(map(int, pi))
        else:
            xx, yy, pp = xs, ys, ps
            if len(out_x) and np.isclose(out_x[-1], xx[0]) and np.isclose(out_y[-1], yy[0]):
                xx = xx[1:]; yy = yy[1:]; pp = pp[1:]
            out_x.extend(map(float, xx)); out_y.extend(map(float, yy)); out_p.extend(map(int, pp))

    return np.asarray(out_x), np.asarray(out_y), np.asarray(out_p)
