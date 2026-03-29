import math
import numpy as np

try:
    from scipy.interpolate import splprep, splev
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

def smooth_path(
    path:         list[tuple[int, int]],
    obstacle_map: np.ndarray,
    smoothing:    float = 1.5,
    oversample:   int   = 4,
) -> list[tuple[int, int]]:
    if not _SCIPY_OK or len(path) < 4:
        return path

    H, W = obstacle_map.shape
    xs = np.array([p[0] for p in path], dtype=np.float64)
    ys = np.array([p[1] for p in path], dtype=np.float64)

    k_deg = min(3, len(path) - 1)
    s_val = smoothing * len(path)   
    n_out = len(path) * oversample

    try:
        tck, _ = splprep([xs, ys], s=s_val, k=k_deg)
        u_new  = np.linspace(0.0, 1.0, n_out)
        xs_s, ys_s = splev(u_new, tck)
    except Exception as exc:
        return path

    result: list[tuple[int, int]] = []
    prev   = None

    for x_f, y_f in zip(xs_s, ys_s):
        xi = int(round(x_f))
        yi = int(round(y_f))
        xi = max(0, min(W - 1, xi))
        yi = max(0, min(H - 1, yi))

        if obstacle_map[yi, xi] > 0.5:
            continue   

        pt = (xi, yi)
        if pt != prev:
            result.append(pt)
            prev = pt

    if not result:
        return path

    if result[0]  != path[0]:  result.insert(0, path[0])
    if result[-1] != path[-1]: result.append(path[-1])

    return result

def path_length(path: list[tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(
        math.hypot(path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        for i in range(1, len(path))
    )
