import os
import numpy as np
import cv2

_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "Veri", "best.pt")
_IMAGE_PATH  = os.path.join(os.path.dirname(__file__), "Veri", "moon.png")
_SAM_PATH    = "mobile_sam.pt"

_CONF_THRESH = 0.05
_IOU_THRESH  = 0.5
_INFER_SIZE  = 1280

_yolo_model = None
_sam_model  = None

def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        import torch
        _yolo_model = torch.hub.load(
            "ultralytics/yolov5", "custom",
            path=_MODEL_PATH, force_reload=False,
            trust_repo=True, verbose=False,
        )
        _yolo_model.conf = _CONF_THRESH
        _yolo_model.iou  = _IOU_THRESH
    return _yolo_model

def _load_sam():
    global _sam_model
    if _sam_model is None:
        from ultralytics import SAM
        _sam_model = SAM(_SAM_PATH)
    return _sam_model

def detect_craters(image: np.ndarray, image_path: str | None = None) -> np.ndarray:
    grid_h, grid_w = image.shape[:2]
    crater_map = np.zeros((grid_h, grid_w), dtype=np.float32)

    if os.path.exists(_MODEL_PATH):
        try:

            effective_path = image_path or (_IMAGE_PATH if os.path.exists(_IMAGE_PATH) else None)
            crater_map = _detect_fullres(image, grid_h, grid_w, effective_path)
            n_pixels = int((crater_map > 0.5).sum())
            return crater_map
        except Exception as exc:
            pass
    else:
        pass

    return _detect_with_hough(image)

def _detect_fullres(image_np: np.ndarray, grid_h: int, grid_w: int,
                    image_path: str | None = None) -> np.ndarray:

    if image_path and os.path.exists(image_path):
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Görüntü okunamadı: {image_path}")
        yolo_input = [image_path]   
    else:

        img_gray = image_np if image_np.ndim == 2 else cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        yolo_input = [cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)]  

    orig_h, orig_w = img_gray.shape
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    model = _load_yolo()
    results = model(yolo_input, size=_INFER_SIZE)
    detected_boxes = results.pandas().xyxy[0]

    if detected_boxes.empty:
        return np.zeros((grid_h, grid_w), dtype=np.float32)

    valid_boxes    = []   
    krater_listesi = []

    for idx, row in detected_boxes.iterrows():
        xmin = int(row["xmin"]); ymin = int(row["ymin"])
        xmax = int(row["xmax"]); ymax = int(row["ymax"])

        x = (xmin + xmax) // 2
        y = (ymin + ymax) // 2
        r = ((xmax - xmin) + (ymax - ymin)) // 4

        if x >= orig_w or y >= orig_h or r <= 0:
            continue

        merkez_yukseklik = int(img_gray[y, x])
        rim_mask = np.zeros(img_gray.shape, dtype=np.uint8)
        cv2.circle(rim_mask, (x, y), max(1, int(r * 0.9)), 255, 2)
        ceper_pikselleri = img_gray[rim_mask == 255]

        if len(ceper_pikselleri) == 0:
            continue

        ceper_ort        = int(np.mean(ceper_pikselleri))
        derinlik         = ceper_ort - merkez_yukseklik

        krater_listesi.append({"id": idx, "x": x, "y": y, "r": r,
                                "derinlik": derinlik})
        valid_boxes.append([xmin, ymin, xmax, ymax])

    if not valid_boxes:
        return np.zeros((grid_h, grid_w), dtype=np.float32)

    sam   = _load_sam()
    sam_results = sam(img_bgr, bboxes=valid_boxes, verbose=False)

    fullres_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    poly_count = 0
    for r_sam in sam_results:
        if r_sam.masks is None:
            continue
        for poly in r_sam.masks.xy:
            if len(poly) > 2:
                cv2.fillPoly(fullres_mask, [poly.astype(np.int32)], 255)
                poly_count += 1

    grid_mask = cv2.resize(fullres_mask, (grid_w, grid_h),
                           interpolation=cv2.INTER_NEAREST)
    crater_map = (grid_mask > 127).astype(np.float32)
    return crater_map

def _detect_with_hough(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    blurred = cv2.GaussianBlur(image, (9, 9), 2)
    min_r = max(5, int(min(h, w) * 0.01))
    max_r = max(min_r + 1, int(min(h, w) * 0.15))

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=min_r * 2,
        param1=60, param2=28,
        minRadius=min_r, maxRadius=max_r,
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for (cx, cy, r) in circles:
            cv2.circle(mask, (cx, cy), r, 1.0, thickness=-1)

    n = 0 if circles is None else len(circles)
    return mask
