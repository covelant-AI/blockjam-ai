import numpy as np
from .data import SimplifiedResults

def filter_yolo_results(results, target_class, topn=None, names=None):
    """
    Filter YOLO detections by class with optional top-N by confidence.

    Parameters
    ----------
    results : ultralytics Results | yolov5 results | np.ndarray | torch.Tensor | list-like
        - Ultralytics YOLOv8: a single `Results` with `.boxes.xyxy`, `.boxes.conf`, `.boxes.cls`.
        - YOLOv5: an object with `.xyxy[0]` -> [N,6] array [x1,y1,x2,y2,conf,cls].
        - Raw: array/tensor shaped [N, >=6] where the first 6 columns are [x1,y1,x2,y2,conf,cls].
    target_class : int | str
        Class id or class name to keep.
    topn : int | None
        If given, return only the top-N detections by confidence.
    names : dict | list | tuple | None
        Optional class-name map. Accepts:
        - dict id->name or name->id
        - list/tuple of names (index = id)
        If omitted, tries `results.names` when available.

    Returns
    -------
    list[list[float]]
        [[x1,y1,x2,y2,conf], ...] sorted by descending confidence.

    Notes
    -----
    - Assumes boxes are XYXY in absolute image coordinates. If you have normalized XYWH,
      convert before calling.
    - There’s no single YOLO “results” schema in the wild; if your object differs,
      you may need to adapt the column mapping.
    """
    import numpy as np
    try:
        import torch
    except Exception:
        torch = None

    def to_np(x):
        if x is None:
            return None
        if torch is not None and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # Build name->id mapping if the user passed names or if results provides `names`
    name_to_id = None
    if isinstance(names, dict):
        # Allow id->name or name->id
        if all(isinstance(k, int) for k in names):
            name_to_id = {v: k for k, v in names.items()}
        else:
            name_to_id = names
    elif isinstance(names, (list, tuple)):
        name_to_id = {n: i for i, n in enumerate(names)}
    else:
        res_names = getattr(results, "names", None)
        if isinstance(res_names, dict):
            if all(isinstance(k, int) for k in res_names):
                name_to_id = {v: k for k, v in res_names.items()}
            else:
                name_to_id = res_names
        elif isinstance(res_names, (list, tuple)):
            name_to_id = {n: i for i, n in enumerate(res_names)}

    # Normalize target_class to an integer class id
    if isinstance(target_class, str):
        if not name_to_id or target_class not in name_to_id:
            raise ValueError(
                f"Can't resolve class name '{target_class}'. "
                "Pass `names=` (id->name, name->id, or list of names) or use a class id."
            )
        cls_id = int(name_to_id[target_class])
    else:
        cls_id = int(target_class)

    detections = None

    # Case A: Ultralytics v8-style Results
    boxes = getattr(results, "boxes", None)
    if boxes is not None and all(hasattr(boxes, a) for a in ("xyxy", "conf", "cls")):
        xyxy = to_np(boxes.xyxy)
        conf = to_np(boxes.conf).reshape(-1, 1)
        cls = to_np(boxes.cls).reshape(-1, 1)
        detections = np.concatenate([xyxy, conf, cls], axis=1)

    # Case B: YOLOv5 results object with .xyxy list
    if detections is None and hasattr(results, "xyxy"):
        arrs = results.xyxy
        if isinstance(arrs, (list, tuple)) and len(arrs):
            detections = to_np(arrs[0])

    # Case C: raw array/tensor/list
    if detections is None:
        arr = to_np(results)
        if arr is not None and arr.ndim == 2 and arr.shape[1] >= 6:
            detections = arr

    if detections is None:
        raise TypeError(
            "Unsupported `results` format. Expected Ultralytics Results, "
            "YOLOv5 results with .xyxy[0], or array shaped [N,>=6] with [x1,y1,x2,y2,conf,cls,...]."
        )

    det = detections
    if det.size == 0:
        return []
    # Heuristic to locate conf/cls columns. Default [x1,y1,x2,y2,conf,cls].
    conf_col, cls_col = 4, 5
    # If the presumed confidence column is not in [0,1], try the last two columns.
    if det.shape[1] >= 6 and (det[:, conf_col].max() > 1.0 or det[:, conf_col].min() < 0.0):
        conf_col, cls_col = det.shape[1] - 2, det.shape[1] - 1

    # Filter by class id
    keep_mask = det[:, cls_col] == cls_id
    det = det[keep_mask]

    if det.size == 0:
        return []

    # Sort by confidence descending
    order = np.argsort(-det[:, conf_col])
    det = det[order]

    # Top-N selection if requested
    if topn is not None:
        det = det[: int(topn)]

    # Return [x1,y1,x2,y2,conf]
    return det[:, [0, 1, 2, 3, conf_col]].tolist()


def build_results_from_xyxy_conf(dets, class_id):
    """
    dets: list/array of shape [N,5] with [x1,y1,x2,y2,conf]
    class_id: int — applied to all detections

    returns: YOLOLite(conf, xywh, cls)
    """
    arr = np.asarray(dets, dtype=np.float32)
    if arr.size == 0:
        # empty, but valid YOLOLite
        return SimplifiedResults(
            conf=np.zeros((0,), dtype=np.float32),
            xywh=np.zeros((0, 4), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.int64),
        )
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError("Expected dets with shape [N,5]: [x1,y1,x2,y2,conf].")

    x1, y1, x2, y2, conf = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

    # Ensure proper corner ordering (swap if needed)
    x1c = np.minimum(x1, x2)
    x2c = np.maximum(x1, x2)
    y1c = np.minimum(y1, y2)
    y2c = np.maximum(y1, y2)

    w = x2c - x1c
    h = y2c - y1c
    xc = x1c + 0.5 * w
    yc = y1c + 0.5 * h

    xywh = np.stack([xc, yc, w, h], axis=1).astype(np.float32)
    cls = np.full((arr.shape[0],), int(class_id), dtype=np.int64)
    conf = conf.astype(np.float32).reshape(-1)

    return SimplifiedResults(conf=conf, xywh=xywh, cls=cls)
