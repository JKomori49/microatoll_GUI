from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, Any, List

@dataclass
class Polyline:
    """
    Stores a polyline as three aligned 1D arrays: x, y, phi (0/1).
    Internally operates on numpy arrays; export helpers return lists for JSON/CSV.
    """
    x: np.ndarray
    y: np.ndarray
    phi: np.ndarray  # 0/1 ints

    @staticmethod
    def from_ndarray(arr: np.ndarray) -> "Polyline":
        """
        arr shape: (N, 3) -> columns [x, y, phi]
        """
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Input array must have shape (N, 3) with columns [x, y, phi].")
        x = arr[:, 0].astype(float)
        y = arr[:, 1].astype(float)
        phi = arr[:, 2].astype(int)
        return Polyline(x=x, y=y, phi=phi)

    def to_rows(self) -> List[Dict[str, Any]]:
        """Return list of dict rows suitable for JSON serialization."""
        return [
            {"x": float(x), "y": float(y), "phi": int(p)}
            for x, y, p in zip(self.x.tolist(), self.y.tolist(), self.phi.tolist())
        ]

    def as_closed_xy(self) -> np.ndarray:
        """
        Return (M, 2) float array for plotting a closed polyline (append first point at end).
        """
        xy = np.column_stack([self.x, self.y]).astype(float)
        if len(xy) == 0:
            return xy
        return np.vstack([xy, xy[0]])
