import numpy as np

def build_risk_map(
    slope_map:     np.ndarray,
    obstacle_map:  np.ndarray,
    crater_map:    np.ndarray,
    roughness_map: np.ndarray | None = None,
    w_slope: float = 0.5,
    w_obs:   float = 0.3,
    w_crat:  float = 0.2,
    w_rough: float = 0.0,
) -> np.ndarray:

    assert slope_map.shape == obstacle_map.shape == crater_map.shape, (
        "All hazard maps must have identical shapes. "
        f"Got: slope={slope_map.shape}, obs={obstacle_map.shape}, "
        f"crater={crater_map.shape}"
    )

    risk = (
        w_slope * slope_map
      + w_obs   * obstacle_map
      + w_crat  * crater_map
    )

    if roughness_map is not None:
        assert roughness_map.shape == slope_map.shape, (
            f"roughness_map shape {roughness_map.shape} "
            f"!= slope_map shape {slope_map.shape}"
        )
        risk = risk + w_rough * roughness_map

    risk = np.clip(risk, 0.0, 1.0)

    r_min, r_max = risk.min(), risk.max()
    if r_max > r_min:
        risk = (risk - r_min) / (r_max - r_min)

    risk = risk.astype(np.float32)

    layers = "slope+obstacle+crater"
    if roughness_map is not None:
        layers += "+roughness"
    return risk
