# debug_helpers.py
import numpy as np
import cv2

def _hom_warp(H: np.ndarray, pts_uv: np.ndarray) -> np.ndarray:
    """Apply 3x3 homography H to Nx2 points. Returns Nx2 image coords."""
    pts_uv = np.asarray(pts_uv, dtype=np.float32).reshape(-1, 2)
    pts_h = np.concatenate([pts_uv, np.ones((len(pts_uv), 1), dtype=np.float32)], axis=1)
    q = (H @ pts_h.T).T
    q = q[:, :2] / q[:, 2:3]
    return q

def debug_dump_player_uv(apf, tracks, frame_id: int):
    """
    Print a table of (tid, u, v, in_court, near_net_in, near_net_any, half).
    Uses apf._to_uv and apf._near_net_flags (intended for debugging).
    """
    if apf.H_img2court is None or apf.spec is None:
        print("[debug] Homography/spec not initialized.")
        return

    L = float(apf.spec["length"])
    header = f"{'tid':>6} | {'u':>7} {'v':>7} | in_c near_in near_any | half"
    print(header)
    print("-" * len(header))
    for t in tracks:
        if not getattr(t, "is_activated", True):
            continue
        bxby = getattr(t, "xywh")
        bx, by = apf.bbox_bottom_center(bxby, fmt=apf.bbox_format) if hasattr(apf, 'bbox_bottom_center') else None, None
        # use the project utility directly to avoid attribute lookup issues
        from playertracker.homography import bbox_bottom_center as _bbc
        bx, by = _bbc(bxby, fmt=apf.bbox_format)
        uv = apf._to_uv((bx, by))
        if uv is None:
            print(f"{int(t.track_id):6d} | {'None':>7} {'None':>7} |  -     -       -      |  -")
            continue
        in_c, near_in, near_any = apf._near_net_flags(uv)
        half = "near" if uv[1] <= 0.5 * L else "far "
        print(f"{int(t.track_id):6d} | {uv[0]:7.2f} {uv[1]:7.2f} |  {str(in_c)[0]:>1}     {str(near_in)[0]:>1}       {str(near_any)[0]:>1}      | {half}")

def draw_court_guides(frame_bgr: np.ndarray, apf) -> None:
    """
    Draw sidelines, baselines, net line, net band, and half-split in the image.
    """
    if apf.H_court2img is None or apf.spec is None:
        return
    W, L = float(apf.spec["width"]), float(apf.spec["length"])
    m = apf.near_net_lateral_margin_m
    H = apf.H_court2img

    # Lines in (u,v) space
    lines_uv = [
        # baselines (v=0, v=L)
        ((-m, 0.0), (W + m, 0.0)),
        ((-m, L), (W + m, L)),
        # sidelines (u=0, u=W)
        ((0.0, 0.0), (0.0, L)),
        ((W, 0.0), (W, L)),
        # net (v=L/2) and net band
        ((-m, 0.5 * L), (W + m, 0.5 * L)),
        ((-m, 0.5 * L - apf.net_band_m), (W + m, 0.5 * L - apf.net_band_m)),
        ((-m, 0.5 * L + apf.net_band_m), (W + m, 0.5 * L + apf.net_band_m)),
    ]

    for (p0, p1) in lines_uv:
        pts_img = _hom_warp(H, np.array([p0, p1], dtype=np.float32)).astype(int)
        cv2.line(frame_bgr, tuple(pts_img[0]), tuple(pts_img[1]), (0, 255, 255), 1, cv2.LINE_AA)

def draw_player_uv_overlay(frame_bgr: np.ndarray, apf, tracks, *, frame_id: int) -> np.ndarray:
    """
    Draw each track's bottom-center, uv text, and flags.
    Color code: green=in-court, red=off-court; yellow ring if near_net_any.
    """
    if apf.H_img2court is None or apf.spec is None:
        return frame_bgr

    out = frame_bgr.copy()
    draw_court_guides(out, apf)

    W, L = float(apf.spec["width"]), float(apf.spec["length"])
    for t in tracks:
        if not getattr(t, "is_activated", True):
            continue

        from playertracker.homography import bbox_bottom_center as _bbc
        bx, by = _bbc(getattr(t, "xywh"), fmt=apf.bbox_format)
        uv = apf._to_uv((bx, by))
        color = (0, 0, 255)  # default red = off-court / unknown

        if uv is not None:
            in_c, near_in, near_any = apf._near_net_flags(uv)
            color = (0, 200, 0) if in_c else (0, 0, 255)
            # draw bottom-center
            cv2.circle(out, (int(bx), int(by)), 4, color, -1, cv2.LINE_AA)
            if near_any:
                cv2.circle(out, (int(bx), int(by)), 8, (0, 255, 255), 1, cv2.LINE_AA)  # yellow ring
            # annotate text
            half = "N" if uv[1] <= 0.5 * L else "F"
            label = f"id={int(t.track_id)} uv=({uv[0]:.2f},{uv[1]:.2f}) in={int(in_c)} near_in={int(near_in)} any={int(near_any)} half={half}"
            cv2.putText(out, label, (int(bx) + 6, int(by) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(out, (int(bx), int(by)), 4, color, -1, cv2.LINE_AA)
            cv2.putText(out, f"id={int(t.track_id)} uv=None", (int(bx) + 6, int(by) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # frame id stamp
    cv2.putText(out, f"frame={frame_id}", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    return out
