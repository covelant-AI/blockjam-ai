# playertracker/homography.py
import numpy as np
import cv2
from typing import Dict, List, Optional, Sequence, Tuple

# ------- Court constants (meters) -------
SINGLES_W = 8.23
DOUBLES_W = 10.97
COURT_L   = 23.77
SERVICE_D = 5.485

def make_spec(use_doubles: bool = True) -> Dict[str, float]:
    return {
        "width": DOUBLES_W if use_doubles else SINGLES_W,
        "length": COURT_L,
        "singles_width": SINGLES_W,
        "doubles_width": DOUBLES_W,
        "service_from_baseline": SERVICE_D,
    }

# ------- Robust quad ordering (TL,TR,BR,BL) -------
def order_quad_tl_tr_br_bl(poly_xy: np.ndarray) -> np.ndarray:
    P = np.asarray(poly_xy, dtype=np.float32)
    if P.shape != (4, 2):
        raise ValueError("polygon must be (4,2)")
    idx_by_y = np.argsort(P[:, 1])
    top2, bot2 = P[idx_by_y[:2]], P[idx_by_y[2:]]
    tl, tr = top2[np.argsort(top2[:, 0])]
    bl, br = bot2[np.argsort(bot2[:, 0])]
    Q = np.stack([tl, tr, br, bl]).astype(np.float32)
    # keep CCW
    area2 = (Q[0,0]*Q[1,1]-Q[1,0]*Q[0,1]) + (Q[1,0]*Q[2,1]-Q[2,0]*Q[1,1]) + (Q[2,0]*Q[3,1]-Q[3,0]*Q[2,1]) + (Q[3,0]*Q[0,1]-Q[0,0]*Q[3,1])
    if area2 <= 0:
        Q = np.stack([Q[0], Q[3], Q[2], Q[1]]).astype(np.float32)
    return Q

# ------- Homography build + validation -------
def estimate_homography_from_polygon(poly_xy: np.ndarray, *, use_doubles_polygon: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str,float], np.ndarray, np.ndarray]:
    img_quad = order_quad_tl_tr_br_bl(poly_xy)
    spec = make_spec(use_doubles_polygon)
    W, L = spec["width"], spec["length"]
    world_quad = np.array([[0,0],[W,0],[W,L],[0,L]], dtype=np.float32)
    H_img2court = cv2.getPerspectiveTransform(img_quad, world_quad)
    H_court2img = cv2.getPerspectiveTransform(world_quad, img_quad)
    if not np.isfinite(H_img2court).all():
        raise ValueError("Homography has non-finite values; check polygon.")
    return H_img2court, H_court2img, spec, img_quad, world_quad

def validate_homography(H_img2court: np.ndarray, H_court2img: np.ndarray,
                        img_quad: np.ndarray, world_quad: np.ndarray) -> Tuple[float,float]:
    uv = cv2.perspectiveTransform(img_quad.reshape(-1,1,2), H_img2court).reshape(-1,2)
    xy = cv2.perspectiveTransform(world_quad.reshape(-1,1,2), H_court2img).reshape(-1,2)
    err1 = float(np.sqrt(np.mean(np.sum((uv - world_quad)**2, axis=1))))   # meters
    err2 = float(np.sqrt(np.mean(np.sum((xy - img_quad)**2, axis=1))))     # pixels
    return err1, err2

# ------- Warpers -------
def warp_img_to_court(H_img2court: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(pts_xy, np.float32).reshape(-1,1,2), H_img2court).reshape(-1,2)

def warp_court_to_img(H_court2img: np.ndarray, pts_uv: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(pts_uv, np.float32).reshape(-1,1,2), H_court2img).reshape(-1,2)

def warp_segment_court_to_img(H_court2img: np.ndarray, p0_uv: Tuple[float,float], p1_uv: Tuple[float,float]
) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    pts_xy = warp_court_to_img(H_court2img, np.array([p0_uv, p1_uv], np.float32))
    def _cvpt(a: np.ndarray) -> Tuple[int,int]:
        x, y = float(a[0]), float(a[1])
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("non-finite warp result; bad H?")
        return (int(round(x)), int(round(y)))
    return _cvpt(pts_xy[0]), _cvpt(pts_xy[1])

# ------- BBox bottom-center -------
def bbox_bottom_center(xywh_or_tlwh: Sequence[float], fmt: str = "xywh") -> Tuple[float,float]:
    cx, cy, w, h = map(float, xywh_or_tlwh)
    if fmt == "xywh":
        return (cx, cy + 0.5*h)
    elif fmt == "tlwh":
        return (cx + 0.5*w, cy + h)
    raise ValueError("fmt must be 'xywh' or 'tlwh'")

# ------- Canonical court segments in court coords -------
def court_segments_uv(spec: Dict[str,float], include_singles: bool = True
) -> Dict[str, List[Tuple[Tuple[float,float], Tuple[float,float]]]]:
    W, L, SW, srv = spec["width"], spec["length"], spec["singles_width"], spec["service_from_baseline"]
    segs: Dict[str, List[Tuple[Tuple[float,float], Tuple[float,float]]]] = {
        "baselines":   [((0,0),(W,0)), ((0,L),(W,L))],
        "doubles":     [((0,0),(0,L)), ((W,0),(W,L))],
        "service":     [((0,srv),(W,srv)), ((0,L-srv),(W,L-srv))],
        "center_serv": [((0.5*W, srv),(0.5*W, L - srv))],
        "net":         [((0, 0.5*L),(W, 0.5*L))],
        "singles":     [],
    }
    if include_singles:
        u_left = 0.5*(W - SW); u_right = u_left + SW
        segs["singles"] = [((u_left,0),(u_left,L)), ((u_right,0),(u_right,L))]
    return segs

# ------- Predicates in court coords -------
def in_court_uv(uv: Tuple[float,float], spec: Dict[str,float]) -> bool:
    u, v = float(uv[0]), float(uv[1])
    return (0 <= u <= spec["width"]) and (0 <= v <= spec["length"])

def near_net_uv(uv: Tuple[float,float], spec: Dict[str,float], band_m: float = 1.0) -> bool:
    v = float(uv[1]); return abs(v - 0.5*spec["length"]) <= float(band_m)

# ------- (Optional) CV2 overlay for debugging -------
def draw_court_overlay_cv2(image: np.ndarray, poly_xy: np.ndarray, *,
                           use_doubles_polygon: bool = True, draw_singles: bool = True,
                           thickness: int = 2, net_thickness: int = 3, alpha: Optional[float] = 0.6
) -> np.ndarray:
    out = image.copy(); overlay = out.copy() if alpha is not None else out
    H_img2court, H_court2img, spec, img_quad, world_quad = estimate_homography_from_polygon(poly_xy, use_doubles_polygon=use_doubles_polygon)
    # outer quad
    cv2.polylines(overlay, [img_quad.astype(np.int32).reshape(-1,1,2)], True, (0,255,255), thickness, cv2.LINE_AA)
    segs = court_segments_uv(spec, include_singles=draw_singles)
    def draw_family(name: str, col, thick: int):
        for p0_uv, p1_uv in segs[name]:
            p0, p1 = warp_segment_court_to_img(H_court2img, p0_uv, p1_uv)
            cv2.line(overlay, p0, p1, col, thick, cv2.LINE_AA)
    draw_family("baselines", (60,220,60), thickness)
    draw_family("doubles",   (255,120,120), thickness)
    if draw_singles: draw_family("singles", (120,200,255), thickness)
    draw_family("service",   (0,165,255), thickness)
    draw_family("center_serv",(200,200,200), thickness)
    draw_family("net",       (0,0,255), net_thickness)
    if alpha is not None: cv2.addWeighted(overlay, alpha, out, 1-alpha, 0, out)
    return out

__all__ = [
    "SINGLES_W","DOUBLES_W","COURT_L","SERVICE_D",
    "make_spec","order_quad_tl_tr_br_bl",
    "estimate_homography_from_polygon","validate_homography",
    "warp_img_to_court","warp_court_to_img","warp_segment_court_to_img",
    "bbox_bottom_center","court_segments_uv",
    "in_court_uv","near_net_uv",
    "draw_court_overlay_cv2",
]
