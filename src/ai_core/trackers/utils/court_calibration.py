#!/usr/bin/env python3
"""
Tennis Court Homography Picker (small GUI)

What it does
------------
- Loads a single frame from a video.
- Lets you click labeled court landmarks (default: 4 singles-court corners).
- Computes image->world homography H (pixels -> meters on court plane).
- Overlays a verification grid (singles rectangle, service lines, net) to sanity-check your clicks.
- Saves H and metadata to a .npz file you can load in NumPy.

Why this approach
-----------------
A single global "px-to-m" scale is wrong under perspective. A planar homography fixes that for
points lying on the court. This is the minimum required to convert your tracked ball center
(assumed on/near court plane) from pixels to meters.

Dependencies
------------
- Python 3.8+
- numpy
- opencv-python (cv2)

Usage
-----
python court_calibration_gui.py --video path/to/match.mp4 --frame 500 --output court_homography.npz
  [--mode 4corners | 8points]  [--no-grid]  [--winsize 1280 720]

Controls
--------
- Left click: add point at cursor (in the prompted order).
- Right click: undo last point.
- 'r' : reset all points.
- 'g' : toggle overlay grid on/off.
- 's' : save H + metadata to output file (only available after enough points).
- 'q' or ESC : quit without saving.

Point orders
------------
mode=4corners (default): click these in order (clockwise, starting near camera left):
  1) near-left  singles corner    -> world (0.00,  0.00)
  2) near-right singles corner    -> world (8.23,  0.00)
  3) far-right  singles corner    -> world (8.23, 23.77)
  4) far-left   singles corner    -> world (0.00, 23.77)

mode=8points: click these in order:
  1) near-left  singles corner      (0.00,   0.00)
  2) near-right singles corner      (8.23,   0.00)
  3) far-right  singles corner      (8.23,  23.77)
  4) far-left   singles corner      (0.00,  23.77)
  5) near service line - left sideline   (0.00,   5.485)
  6) near service line - right sideline  (8.23,   5.485)
  7) far  service line - right sideline  (8.23,  18.285)
  8) far  service line - left sideline   (0.00,  18.285)

Notes
-----
- World frame: origin at near-left singles corner; x across width (0..8.23 m), y toward far baseline (0..23.77 m).
- If your near/far assumption is inverted, swap the click order accordingly.
- Good practice: pick crisp line intersections; avoid blurry edges; zoom window if needed.
- You can add undistortion in your own pipeline; this tool does not undistort.

The saved .npz contains:
  - H: (3,3) homography mapping [u,v,1]^T (image px) -> [x_m, y_m, 1]^T (meters on plane)
  - img_pts: (N,2) clicked image points (pixels)
  - world_pts: (N,2) corresponding world points (meters)
  - labels: (N,) labels for clicked points
  - video_path, frame_index, mode, singles_width_m, singles_length_m
"""

import argparse
import sys
import os
import numpy as np

# Delay importing cv2 so this file can at least print a helpful error if it's missing.
try:
    import cv2
except Exception as e:
    print("ImportError: OpenCV (cv2) is required. Install with 'pip install opencv-python'.")
    raise

SINGLES_WIDTH = 8.23
SINGLES_LENGTH = 23.77
NET_Y = SINGLES_LENGTH / 2.0
SERVICE_FROM_NET = 6.40
SERVICE_Y_NEAR = NET_Y - SERVICE_FROM_NET   # 11.885 - 6.40 = 5.485
SERVICE_Y_FAR  = NET_Y + SERVICE_FROM_NET   # 11.885 + 6.40 = 18.285
CENTER_X = SINGLES_WIDTH / 2.0

def world_points_for_mode(mode: str):
    if mode == "4corners":
        labels = [
            "near-left singles corner",
            "near-right singles corner",
            "far-right singles corner",
            "far-left singles corner",
        ]
        world = np.array([
            [0.0, 0.0],
            [SINGLES_WIDTH, 0.0],
            [SINGLES_WIDTH, SINGLES_LENGTH],
            [0.0, SINGLES_LENGTH],
        ], dtype=np.float32)
        return labels, world

    if mode == "8points":
        labels = [
            "near-left singles corner",
            "near-right singles corner",
            "far-right singles corner",
            "far-left singles corner",
            "near service line @ left sideline",
            "near service line @ right sideline",
            "far service line @ right sideline",
            "far service line @ left sideline",
        ]
        world = np.array([
            [0.0, 0.0],
            [SINGLES_WIDTH, 0.0],
            [SINGLES_WIDTH, SINGLES_LENGTH],
            [0.0, SINGLES_LENGTH],
            [0.0, SERVICE_Y_NEAR],
            [SINGLES_WIDTH, SERVICE_Y_NEAR],
            [SINGLES_WIDTH, SERVICE_Y_FAR],
            [0.0, SERVICE_Y_FAR],
        ], dtype=np.float32)
        return labels, world

    raise ValueError(f"Unknown mode: {mode}")

def draw_text_multi(img, lines, org=(10, 22), line_h=22, color=(255,255,255), bg=(0,0,0)):
    """Draw multiple lines with a simple background box for readability."""
    x, y = org
    pad = 6
    w = 0
    h = line_h * len(lines)
    for ln in lines:
        size, _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        w = max(w, size[0])
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - line_h + 4 - pad), (x + w + pad, y + h - pad), bg, -1)
    img[:] = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (x, y + i*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def _project_world_to_image(H, pts_world):
    """Project Nx2 world points (meters) to image pixels using inverse homography."""
    H = np.asarray(H, dtype=float)
    Hi = np.linalg.inv(H)
    pts = np.hstack([pts_world, np.ones((pts_world.shape[0], 1), dtype=float)])
    pix = (Hi @ pts.T).T
    w = pix[:, 2:3]
    ok = (np.abs(w) > 1e-9)
    pix[ok[:,0], 0:2] /= w[ok[:,0]]
    return pix[:, :2]

def draw_verification_grid(img, H, color=(0,255,0)):
    """Draw singles rectangle, service lines, center service line, and net projected to the image."""
    if H is None:
        return img
    # polylines in world (meters)
    lines_world = [
        # singles rectangle (closed loop)
        np.array([[0,0],[SINGLES_WIDTH,0],[SINGLES_WIDTH,SINGLES_LENGTH],[0,SINGLES_LENGTH],[0,0]], dtype=float),
        # near service line
        np.array([[0, SERVICE_Y_NEAR],[SINGLES_WIDTH, SERVICE_Y_NEAR]], dtype=float),
        # far service line
        np.array([[0, SERVICE_Y_FAR],[SINGLES_WIDTH, SERVICE_Y_FAR]], dtype=float),
        # center service line (between service lines only)
        np.array([[CENTER_X, SERVICE_Y_NEAR],[CENTER_X, SERVICE_Y_FAR]], dtype=float),
        # net line
        np.array([[0, NET_Y],[SINGLES_WIDTH, NET_Y]], dtype=float),
    ]
    out = img.copy()
    for lw in lines_world:
        pts_img = _project_world_to_image(H, lw)
        pts = []
        for p in pts_img:
            if np.any(~np.isfinite(p)):
                continue
            pts.append([int(round(p[0])), int(round(p[1]))])
        if len(pts) >= 2:
            for i in range(len(pts)-1):
                cv2.line(out, tuple(pts[i]), tuple(pts[i+1]), color, 2, cv2.LINE_AA)
    return out

class CourtPointPicker:
    def __init__(self, frame_bgr, labels, world_pts, show_grid=True, window_size=None):
        self.base = frame_bgr.copy()
        self.frame = frame_bgr.copy()
        self.h, self.w = self.frame.shape[:2]
        self.labels = labels
        self.world = world_pts.astype(np.float32)
        self.img_pts = []  # list of (x,y) float
        self.H = None
        self.show_grid = bool(show_grid)
        self.window_name = "Court Homography Picker"
        self.window_size = window_size  # (W,H) or None

    def _update_view(self):
        vis = self.base.copy()
        # draw existing points
        for i, (u,v) in enumerate(self.img_pts):
            cv2.circle(vis, (int(u), int(v)), 5, (0,140,255), -1, cv2.LINE_AA)
            cv2.putText(vis, f"{i+1}", (int(u)+6, int(v)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,140,255), 2, cv2.LINE_AA)

        # draw polygon if >=2 points
        if len(self.img_pts) >= 2:
            for i in range(len(self.img_pts)-1):
                p1 = tuple(map(lambda x:int(round(x)), self.img_pts[i]))
                p2 = tuple(map(lambda x:int(round(x)), self.img_pts[i+1]))
                cv2.line(vis, p1, p2, (0,140,255), 2, cv2.LINE_AA)

        # compute H if enough points
        if len(self.img_pts) >= 4:
            img_arr = np.array(self.img_pts[:len(self.world)], dtype=np.float32)
            world_arr = self.world[:len(img_arr)]
            try:
                H, _mask = cv2.findHomography(img_arr, world_arr, method=0)  # direct (DLT); enough points -> exact/LS
                if H is not None and np.isfinite(H).all():
                    self.H = H
                    if self.show_grid:
                        vis = draw_verification_grid(vis, self.H, color=(0,255,0))
            except Exception:
                self.H = None

        # HUD text
        remaining = len(self.labels) - len(self.img_pts)
        next_label = self.labels[len(self.img_pts)] if remaining > 0 else "(all points picked)"
        lines = [
            "Left-click: add point | Right-click: undo | r: reset | g: toggle grid | s: save | q/ESC: quit",
            f"Pick order: {len(self.labels)} points | Next: {next_label}",
            f"Picked: {len(self.img_pts)}/{len(self.labels)}  |  Grid: {'ON' if self.show_grid else 'OFF'}",
            f"H status: {'OK' if self.H is not None else 'N/A (need >=4 pts; ensure non-collinear)'}",
        ]
        draw_text_multi(vis, lines, org=(10, 24))

        # Resize to window if requested
        if self.window_size is not None:
            vis = cv2.resize(vis, self.window_size, interpolation=cv2.INTER_AREA)

        self.frame = vis

    def _mouse(self, event, x, y, flags, param):
        if self.window_size is not None:
            # map window coords back to image coords
            sx = self.w / float(self.window_size[0])
            sy = self.h / float(self.window_size[1])
            x = int(round(x * sx))
            y = int(round(y * sy))

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.img_pts) < len(self.labels):
                self.img_pts.append((float(x), float(y)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.img_pts:
                self.img_pts.pop()

        self._update_view()

    def run(self):
        self._update_view()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        if self.window_size is not None:
            cv2.resizeWindow(self.window_name, self.window_size[0], self.window_size[1])
        cv2.setMouseCallback(self.window_name, self._mouse)

        saved = False
        while True:
            cv2.imshow(self.window_name, self.frame)
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q
                break
            elif key == ord('r'):
                self.img_pts = []
                self.H = None
                self._update_view()
            elif key == ord('g'):
                self.show_grid = not self.show_grid
                self._update_view()
            elif key == ord('s'):
                if len(self.img_pts) >= 4 and self.H is not None:
                    saved = True
                    break
                else:
                    # flash a message: need >=4 pts
                    tmp = self.frame.copy()
                    draw_text_multi(tmp, ["Need >=4 points and a valid H to save."], org=(10, 80), color=(0,255,255), bg=(60,60,60))
                    cv2.imshow(self.window_name, tmp)
                    cv2.waitKey(700)
                    self._update_view()

        cv2.destroyWindow(self.window_name)
        return saved, self.H, np.array(self.img_pts, dtype=np.float32)

def grab_frame(video_path: str, frame_index: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_index < 0:
        frame_index = 0
    if total > 0 and frame_index >= total:
        frame_index = total - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        # try from start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Could not read a frame from the video.")
        frame_index = 0
    cap.release()
    return frame, frame_index, total

def save_npz(output_path, H, img_pts, world_pts, labels, video_path, frame_index, mode):
    np.savez_compressed(output_path,
                        H=H.astype(np.float64),
                        img_pts=img_pts.astype(np.float32),
                        world_pts=world_pts.astype(np.float32),
                        labels=np.array(labels),
                        video_path=str(video_path),
                        frame_index=int(frame_index),
                        mode=str(mode),
                        singles_width_m=float(SINGLES_WIDTH),
                        singles_length_m=float(SINGLES_LENGTH))
    print(f"[OK] Saved: {output_path}")
    print("You can load it in Python via:")
    print("  data = np.load('"+os.path.basename(output_path)+"', allow_pickle=True)")
    print("  H = data['H']")

def main():
    ap = argparse.ArgumentParser(description="Pick tennis court points and compute homography (pixels->meters).")
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--frame", type=int, default=0, help="Frame index to sample (default 0)")
    ap.add_argument("--mode", type=str, default="4corners", choices=["4corners","8points"],
                    help="Which labeled point set to use (default 4corners)")
    ap.add_argument("--output", type=str, default="court_homography.npz", help="Output .npz path")
    ap.add_argument("--no-grid", action="store_true", help="Disable verification grid overlay")
    ap.add_argument("--winsize", type=int, nargs=2, metavar=('W','H'),
                    help="Resize display window to WxH (does not affect saved coordinates)")
    args = ap.parse_args()

    frame, idx, total = grab_frame(args.video, args.frame)
    labels, world_pts = world_points_for_mode(args.mode)

    title = f"Court Homography Picker - frame {idx}/{total-1 if total>0 else '?'} | {os.path.basename(args.video)}"
    print(title)
    print("Click order:")
    for i, lbl in enumerate(labels, 1):
        print(f"  {i}) {lbl}")

    picker = CourtPointPicker(frame, labels, world_pts, show_grid=(not args.no_grid), window_size=tuple(args.winsize) if args.winsize else None)
    saved, H, img_pts = picker.run()

    if not saved:
        print("Exited without saving.")
        sys.exit(1)

    if H is None or not np.isfinite(H).all():
        print("Homography invalid. Check point order/quality and try again.")
        sys.exit(2)

    # Save results
    save_npz(args.output, H, img_pts, world_pts, labels, args.video, idx, args.mode)

if __name__ == "__main__":
    main()
