import cv2
import numpy as np

def load_and_preprocess(
    image_path: str,
    target_size: tuple[int, int] = (200, 200),
) -> np.ndarray:
    raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(
            f"[preprocessing] Cannot read image: '{image_path}'\n"
            "  → Check the path and file format."
        )

    resized = cv2.resize(raw, target_size, interpolation=cv2.INTER_LINEAR)

    image_norm = resized.astype(np.float32) / 255.0

    return image_norm
