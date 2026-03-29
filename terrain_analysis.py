import cv2
import numpy as np

def compute_slope_map(image_norm: np.ndarray) -> np.ndarray:

    img_f64 = (image_norm * 255).astype(np.float64)

    img_blurred = cv2.GaussianBlur(img_f64, (11, 11), 3.0)

    sobelx = cv2.Sobel(img_blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_blurred, cv2.CV_64F, 0, 1, ksize=5)

    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    mag_99th = np.percentile(magnitude, 99)
    if mag_99th > 0:
        slope_map = np.clip(magnitude / mag_99th, 0.0, 1.0).astype(np.float32)
    else:
        slope_map = np.zeros_like(magnitude, dtype=np.float32)

    slope_map = cv2.GaussianBlur(slope_map, (5, 5), 1.0)

    return slope_map

def compute_obstacle_map(
    image_norm: np.ndarray,
    dark_threshold: float = 0.15,
    bright_threshold: float = 0.90,
    morph_kernel_size: int = 3,
) -> np.ndarray:

    dark_mask   = (image_norm < dark_threshold).astype(np.uint8)
    bright_mask = (image_norm > bright_threshold).astype(np.uint8)
    binary = cv2.bitwise_or(dark_mask, bright_mask)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (morph_kernel_size, morph_kernel_size),
    )
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    obstacle_map = opened.astype(np.float32)

    n_obs = int(obstacle_map.sum())
    pct   = 100.0 * n_obs / obstacle_map.size
    return obstacle_map
