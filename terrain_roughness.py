import cv2
import numpy as np

def compute_roughness_map(
    image_norm:  np.ndarray,
    kernel_size: int   = 9,
    blur_sigma:  float = 1.5,
) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1

    img = image_norm.astype(np.float64)
    k   = (kernel_size, kernel_size)

    mean    = cv2.blur(img,    k)
    sq_mean = cv2.blur(img**2, k)

    variance  = np.maximum(sq_mean - mean**2, 0.0)
    roughness = np.sqrt(variance).astype(np.float32)

    if blur_sigma > 0:
        roughness = cv2.GaussianBlur(roughness, (0, 0), blur_sigma)

    rmax = roughness.max()
    if rmax > 1e-9:
        roughness = (roughness / rmax).astype(np.float32)

    n_rough = int((roughness > 0.5).sum())
    pct     = 100.0 * n_rough / roughness.size
    return roughness
