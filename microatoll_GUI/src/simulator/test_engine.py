import numpy as np

def generate_circle_polyline(num_points: int = 200, radius: float = 1.0) -> np.ndarray:
    """
    Generate a unit circle polyline sampled by num_points.
    Output ndarray shape: (N, 3) columns = [x, y, phi]
    - x = r*cos(t), y = r*sin(t)
    - phi = 1 for all vertices (alive). You can later modify per-vertex states.
    """
    num_points = max(4, int(num_points))
    t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    x = radius * np.cos(t)
    y = radius * np.sin(t)

    # Example: set lower half as not alive (phi=0). Comment out if not desired.
    phi = np.ones_like(x, dtype=int)
    # phi[y < 0.0] = 0

    arr = np.column_stack([x, y, phi]).astype(float)
    # keep phi integral in last column
    arr[:, 2] = phi
    return arr
