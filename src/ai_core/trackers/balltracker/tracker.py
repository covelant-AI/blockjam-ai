from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterable
import numpy as np

# ---------- small helpers ----------

def _center(b: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = b[:4]
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

def _area(b: np.ndarray) -> float:
    x1, y1, x2, y2 = b[:4]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def _point_in_poly(pt: np.ndarray, poly_xy: np.ndarray) -> bool:
    # ray-casting; poly_xy shape: (N,2)
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = poly_xy.shape[0]
    for i in range(n):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside

def _apply_H_point(H: np.ndarray, pt_px: np.ndarray) -> np.ndarray:
    p = np.array([pt_px[0], pt_px[1], 1.0], dtype=np.float32)
    q = H @ p
    q /= (q[2] + 1e-12)
    return q[:2].astype(np.float32)

# ---------- track struct ----------

@dataclass
class _TrackH:
    track_id: int
    bbox_px: np.ndarray              # [x1,y1,x2,y2] in pixels
    conf: float
    center_px: np.ndarray            # [cx, cy] in pixels
    center_wrld: np.ndarray          # [ux, uy] in court units (e.g., meters)
    vel_wrld: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))  # m/s
    hits: int = 1
    age: int = 1
    misses: int = 0
    speed_ema: float = 0.0           # EMA of speed (m/s)
    still_streak: int = 0            # consecutive "low-motion" frames
    bounce_grace_left: int = 0
    last_matched: int = 0

# ---------- tracker ----------

class BallTracker:
    """
    Single-ball selector with *required* homography and court polygon.
    All matching & motion thresholds are in court units (e.g., meters).

    Input per frame: Iterable[[x1,y1,x2,y2,conf]] in image pixels.
    Output per frame: at most one dict:
        {"track_id": int, "bbox": [x1,y1,x2,y2], "conf": float}
    """

    def __init__(
        self,
        H_img2court: np.ndarray,         # 3x3 homography mapping image px -> court coords (meters recommended)
        court_poly_px: np.ndarray,       # Nx2 polygon in image pixels
        fps: float,                      # video frame rate
        *,
        # detection filtering
        min_conf: float = 0.25,
        min_box_area: Optional[float] = None,
        max_box_area: Optional[float] = None,
        court_margin_px: float = 6.0,
        # gating (world)
        base_gate_m: float = 1.0,
        speed_gate_gain: float = 0.30,   # extra meters per (m/s)
        max_gate_m: float = 3.0,
        # motion smoothing
        vel_alpha: float = 0.5,          # EMA for velocity
        speed_alpha: float = 0.5,        # EMA for speed magnitude
        # static suppression / bounce awareness
        suppress_static_output: bool = True,
        static_speed_thresh_mps: float = 1.0,  # below this treated as "still"
        static_frames: int = 4,          # consecutive low-motion frames to call static
        bounce_grace: int = 2,           # allow N low-motion frames (apex/bounce) without suppression
        # active selection behavior
        switch_patience: int = 3,        # frames to wait before switching active when misses accrue
        min_hits_to_emit: int = 2,       # debounce output until track is seen twice
        # lifecycle
        max_age: int = 10,               # drop after this many consecutive misses
        next_id_start: int = 1,
    ):
        # --- validate inputs ---
        H_img2court = np.asarray(H_img2court, dtype=np.float32)
        assert H_img2court.shape == (3, 3), "H_img2court must be 3x3"
        det = np.linalg.det(H_img2court[:2, :2])
        if abs(det) < 1e-8:
            raise ValueError("H_img2court appears singular in-plane; check your calibration.")
        court_poly_px = np.asarray(court_poly_px, dtype=np.float32)
        assert court_poly_px.ndim == 2 and court_poly_px.shape[1] == 2, "court_poly_px must be Nx2"

        # --- store config ---
        self.H = H_img2court
        self.court_poly_px = court_poly_px
        self.court_margin_px = float(court_margin_px)
        self.fps = float(max(1e-6, fps))
        self.dt = 1.0 / self.fps

        self.min_conf = float(min_conf)
        self.min_box_area = float(min_box_area) if min_box_area is not None else None
        self.max_box_area = float(max_box_area) if max_box_area is not None else None

        self.base_gate_m = float(base_gate_m)
        self.speed_gate_gain = float(speed_gate_gain)
        self.max_gate_m = float(max_gate_m)

        self.vel_alpha = float(vel_alpha)
        self.speed_alpha = float(speed_alpha)

        self.suppress_static_output = bool(suppress_static_output)
        self.static_speed_thresh_mps = float(static_speed_thresh_mps)
        self.static_frames = int(static_frames)
        self.bounce_grace = int(bounce_grace)

        self.switch_patience = int(switch_patience)
        self.min_hits_to_emit = int(min_hits_to_emit)
        self.max_age = int(max_age)

        # --- state ---
        self._tracks: Dict[int, _TrackH] = {}
        self._next_id = int(next_id_start)
        self.frame_idx = 0
        self._active_id: Optional[int] = None
        self._active_misses = 0
        self._mode = "SEARCH"  # or "LOCKED"

    # ---------- public ----------

    def update_polygon(self, court_poly_px):
        court_poly_px = np.asarray(court_poly_px, dtype=np.float32)
        assert court_poly_px.ndim == 2 and court_poly_px.shape[1] == 2, "court_poly_px must be Nx2"
        self.court_poly_px = court_poly_px

    def reset(self):
        self._tracks.clear()
        self._next_id = 1
        self.frame_idx = 0
        self._active_id = None
        self._active_misses = 0
        self._mode = "SEARCH"

    def update(self, detections: Iterable[Iterable[float]]) -> List[Dict]:
        """
        Update the tracker with YOLO-like detections: [x1,y1,x2,y2,conf]
        Returns 0 or 1 track (the active ball).
        """
        self.frame_idx += 1
        dets = self._filter_and_gate_dets(detections)

        # centers in px and world
        if dets.shape[0]:
            det_centers_px = np.stack([_center(d[:4]) for d in dets], axis=0)
            det_centers_wrld = np.stack([_apply_H_point(self.H, c) for c in det_centers_px], axis=0)
        else:
            det_centers_px = np.zeros((0, 2), np.float32)
            det_centers_wrld = np.zeros((0, 2), np.float32)

        # predict centers in world for all tracks
        track_ids = list(self._tracks.keys())
        pred_wrld: Dict[int, np.ndarray] = {}
        for tid in track_ids:
            t = self._tracks[tid]
            pred_wrld[tid] = t.center_wrld + t.vel_wrld * self.dt

        # associate via greedy on world distance with dynamic gate
        matches, unmatched_tids, unmatched_dets = self._greedy_match_world(
            track_ids, pred_wrld, det_centers_wrld
        )

        # update matched
        for tid, di in matches:
            self._update_track(self._tracks[tid], dets[di, :4], float(dets[di, 4]))

        # spawn policy
        spawnable = self._spawnable_indices(unmatched_dets, det_centers_wrld, pred_wrld)
        for di in sorted(spawnable, key=lambda i: float(dets[i, 4]), reverse=True):
            self._spawn_from_det(dets, di)

        # age & prune
        self._age_and_prune(unmatched_tids)

        # active selection
        self._update_active(matches, det_centers_wrld)

        # emit
        return self._emit()

    # ---------- internals ----------

    def _filter_and_gate_dets(self, dets_in: Iterable[Iterable[float]]) -> np.ndarray:
        if dets_in is None:
            return np.zeros((0, 5), dtype=np.float32)
        arr = np.array(list(dets_in), dtype=np.float32).reshape(-1, 5)
        if arr.size == 0:
            return arr

        keep = arr[:, 4] >= self.min_conf
        if self.min_box_area is not None or self.max_box_area is not None:
            areas = np.array([_area(b) for b in arr[:, :4]])
            if self.min_box_area is not None:
                keep &= areas >= self.min_box_area
            if self.max_box_area is not None:
                keep &= areas <= self.max_box_area

        # court polygon with small outward margin
        if self.court_poly_px is not None:
            poly = self._grow_poly(self.court_poly_px, self.court_margin_px)
            centers = np.stack([_center(b) for b in arr[:, :4]], axis=0)
            mask = np.array([_point_in_poly(centers[i], poly) for i in range(centers.shape[0])])
            keep &= mask

        return arr[keep]

    def _grow_poly(self, poly: np.ndarray, margin_px: float) -> np.ndarray:
        # simple radial growth around centroid (approximate)
        c = np.mean(poly, axis=0, keepdims=True)
        vec = poly - c
        norms = np.linalg.norm(vec, axis=1) + 1e-6
        # per-vertex scaling factor to achieve ~margin increase
        scale = (norms + margin_px) / norms
        return (c + vec * scale[:, None]).astype(np.float32)

    def _gate_radius(self, t: Optional[_TrackH]) -> float:
        if t is None:
            return min(self.base_gate_m, self.max_gate_m)
        speed = float(np.linalg.norm(t.vel_wrld))  # m/s
        g = self.base_gate_m + self.speed_gate_gain * speed
        return float(min(g, self.max_gate_m))

    def _greedy_match_world(
        self,
        track_ids: List[int],
        pred_wrld: Dict[int, np.ndarray],
        det_centers_wrld: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not track_ids or det_centers_wrld.shape[0] == 0:
            return [], track_ids, list(range(det_centers_wrld.shape[0]))

        pairs = []
        for ti, tid in enumerate(track_ids):
            t = self._tracks[tid]
            gate = self._gate_radius(t)
            diffs = det_centers_wrld - pred_wrld[tid][None, :]
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))  # meters
            for di, d in enumerate(dists):
                if d <= gate:
                    pairs.append((ti, di, float(d)))

        # no candidate pairs -> everything unmatched
        if not pairs:
            return [], track_ids, list(range(det_centers_wrld.shape[0]))

        # sort by distance ascending
        pairs.sort(key=lambda x: x[2])
        assigned_t = set()
        assigned_d = set()
        matches: List[Tuple[int, int]] = []

        for ti, di, _ in pairs:
            if ti in assigned_t or di in assigned_d:
                continue
            assigned_t.add(ti)
            assigned_d.add(di)
            matches.append((track_ids[ti], di))

        unmatched_tids = [tid for i, tid in enumerate(track_ids) if i not in assigned_t]
        unmatched_dets = [di for di in range(det_centers_wrld.shape[0]) if di not in assigned_d]
        return matches, unmatched_tids, unmatched_dets

    def _spawnable_indices(
        self,
        unmatched_dets: List[int],
        det_centers_wrld: np.ndarray,
        pred_wrld: Dict[int, np.ndarray],
    ) -> List[int]:
        if self._mode == "LOCKED" and self._active_id in self._tracks:
            # only spawn near the active prediction
            t = self._tracks[self._active_id]
            gate = self._gate_radius(t)
            ref = pred_wrld.get(self._active_id, t.center_wrld)
            return [di for di in unmatched_dets if np.linalg.norm(det_centers_wrld[di] - ref) <= gate]
        else:
            # in SEARCH, allow any (already court-filtered)
            return unmatched_dets

    def _spawn_from_det(self, dets: np.ndarray, di: int):
        b = dets[di, :4].astype(np.float32)
        c = float(dets[di, 4])
        cx = _center(b)
        cw = _apply_H_point(self.H, cx)
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = _TrackH(
            track_id=tid,
            bbox_px=b.copy(),
            conf=c,
            center_px=cx,
            center_wrld=cw,
            hits=1,
            age=1,
            misses=0,
            last_matched=self.frame_idx,
            bounce_grace_left=self.bounce_grace
        )
        if self._active_id is None:
            self._active_id = tid
            self._active_misses = 0
            self._mode = "LOCKED"

    def _update_track(self, t: _TrackH, det_bbox_px: np.ndarray, det_conf: float):
        prev_wrld = t.center_wrld.copy()

        new_center_px = _center(det_bbox_px)
        new_wrld = _apply_H_point(self.H, new_center_px)
        disp_wrld = new_wrld - prev_wrld  # meters per frame
        inst_speed = float(np.linalg.norm(disp_wrld)) / self.dt  # m/s

        # EMA velocity (m/s) and speed
        t.vel_wrld = (1.0 - self.vel_alpha) * t.vel_wrld + self.vel_alpha * (disp_wrld / self.dt)
        t.speed_ema = (1.0 - self.speed_alpha) * t.speed_ema + self.speed_alpha * inst_speed

        # stillness & bounce grace
        if inst_speed < self.static_speed_thresh_mps:
            t.still_streak += 1
            t.bounce_grace_left = max(0, t.bounce_grace_left - 1)
        else:
            t.still_streak = 0
            t.bounce_grace_left = self.bounce_grace

        # bookkeeping
        t.bbox_px = det_bbox_px.astype(np.float32)
        t.center_px = new_center_px.astype(np.float32)
        t.center_wrld = new_wrld.astype(np.float32)
        t.conf = float(det_conf)
        t.hits += 1
        t.misses = 0
        t.age += 1
        t.last_matched = self.frame_idx

    def _age_and_prune(self, tids: List[int]):
        to_drop = []
        for tid in tids:
            tr = self._tracks.get(tid)
            if tr is None:
                continue
            tr.misses += 1
            tr.age += 1
            tr.bounce_grace_left = max(0, tr.bounce_grace_left - 1)
            if tr.misses > self.max_age:
                to_drop.append(tid)
        for tid in to_drop:
            if tid == self._active_id:
                self._active_id = None
                self._active_misses = 0
                self._mode = "SEARCH"
            self._tracks.pop(tid, None)

    def _update_active(self, matches: List[Tuple[int, int]], det_centers_wrld: np.ndarray):
        matched_ids = {tid for tid, _ in matches}

        # keep current active if matched or within patience
        if self._active_id is not None and self._active_id in self._tracks:
            if self._active_id in matched_ids:
                self._active_misses = 0
                self._mode = "LOCKED"
                return
            self._active_misses += 1
            if self._active_misses <= self.switch_patience:
                self._mode = "LOCKED"
                return

        # choose a new active if needed
        if self._tracks:
            def score(t: _TrackH) -> Tuple[float, float, float]:
                # lower distance to nearest current det is better
                if det_centers_wrld.shape[0]:
                    pred = t.center_wrld + t.vel_wrld * self.dt
                    dists = np.linalg.norm(det_centers_wrld - pred[None, :], axis=1)
                    d = float(np.min(dists))
                else:
                    d = 1e9
                return (-d, float(t.hits), float(t.conf))
            cand = max(self._tracks.values(), key=score)
            self._active_id = cand.track_id
            self._active_misses = 0
            self._mode = "LOCKED"
        else:
            self._active_id = None
            self._active_misses = 0
            self._mode = "SEARCH"

    def _is_static_for_output(self, t: _TrackH) -> bool:
        if t.bounce_grace_left > 0:
            return False
        if t.still_streak >= self.static_frames:
            return True
        return t.speed_ema < self.static_speed_thresh_mps

    def _emit(self) -> List[Dict]:
        if self._active_id is None or self._active_id not in self._tracks:
            return []
        t = self._tracks[self._active_id]
        if t.hits < self.min_hits_to_emit:
            return []
        if self.suppress_static_output and self._is_static_for_output(t):
            return []
        return [{
            "track_id": int(t.track_id),
            "bbox": [float(t.bbox_px[0]), float(t.bbox_px[1]),
                     float(t.bbox_px[2]), float(t.bbox_px[3])],
            "conf": float(t.conf),
            "label": "ball",
        }]
