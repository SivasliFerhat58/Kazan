import math
import numpy as np
from typing import Optional

ENERGY_BASE:            float = 1.0   
SLOPE_FACTOR:           float = 2.5   
RECOVERY_RATIO:         float = 0.40  
ROUGHNESS_FACTOR:       float = 0.8   
DIAGONAL_MULTIPLIER:    float = math.sqrt(2)   

def compute_energy(
    path:          list[tuple[int, int]],
    image_norm:    np.ndarray,
    slope_map:     np.ndarray,
    roughness_map: Optional[np.ndarray] = None,
) -> dict:
    if len(path) < 2:
        return {
            "total": 0.0, "mean_step": 0.0,
            "per_step": [], "cumulative": [0.0],
            "n_uphill": 0, "n_downhill": 0, "n_flat": 0,
        }

    per_step:  list[float] = []
    cumul:     list[float] = [0.0]
    n_up = n_down = n_flat = 0

    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]

        dist = math.hypot(x1 - x0, y1 - y0)   

        e = dist * ENERGY_BASE

        elev0 = float(image_norm[y0, x0])
        elev1 = float(image_norm[y1, x1])
        delta  = elev1 - elev0

        local_slope = float(slope_map[y1, x1])

        if delta > 1e-4:       
            e   += local_slope * SLOPE_FACTOR * dist
            n_up += 1
        elif delta < -1e-4:    
            e   -= local_slope * SLOPE_FACTOR * dist * RECOVERY_RATIO
            e    = max(e, 0.01 * dist)   
            n_down += 1
        else:
            n_flat += 1

        if roughness_map is not None:
            rough = float(roughness_map[y1, x1])
            e += rough * ROUGHNESS_FACTOR * dist

        step_e = round(max(0.0, e), 5)
        per_step.append(step_e)
        cumul.append(round(cumul[-1] + step_e, 5))

    total     = cumul[-1]
    mean_step = round(total / len(per_step), 5) if per_step else 0.0

    return {
        "total":       round(total, 4),
        "mean_step":   mean_step,
        "per_step":    per_step,
        "cumulative":  cumul,
        "n_uphill":    n_up,
        "n_downhill":  n_down,
        "n_flat":      n_flat,
    }
