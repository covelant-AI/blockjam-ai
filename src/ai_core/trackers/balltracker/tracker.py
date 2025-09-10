from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple, Optional, Dict, Any, Deque, Union
import numpy as np


class TennisBallActiveTracker:
    """
    Single-target tennis ball tracker (detection-only; no prediction).

    Core:
      - Court gating in PLANE space (image→court homography).
      - Local association in IMAGE space (nearest to last center) with speed/curvature heuristics.
      - No extrapolation; on miss, hold last bbox and decay confidence.
      - ID-stable unless too many misses.

    Guarantees added:
      - Never pick slow-moving balls OUTSIDE the court.
      - Never pick static balls INSIDE the court.
      - Prefer the globally fastest moving ball (estimated against prior frame),
        with a controlled override over local association.

    Notes:
      - All *speed* thresholds are in pixels/second.
      - On the first frame (no history), per-detection speed is unknown and treated as 0.
    """

    # ------------------------ lightweight structs ------------------------

    @dataclass
    class Detection:
        bbox_img: Tuple[float, float, float, float]   # (x1,y1,x2,y2) in IMAGE pixels
        cx_img: float
        cy_img: float
        conf: float
        cls: Optional[int] = None
        cx_plane: Optional[float] = None              # PLANE coords (for court gating only)
        cy_plane: Optional[float] = None
        # annotations used by selection/suppression
        speed_img: Optional[float] = None             # px/s vs previous frame
        in_court: Optional[bool] = None               # cached court membership

    @dataclass
    class TrackState:
        track_id: int
        # histories
        img_trace: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=24))
        trace_plane: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=64))
        # last knowns
        last_plane: Optional[Tuple[float, float]] = None
        last_bbox_img: Optional[Tuple[float, float, float, float]] = None
        last_wh_img: Optional[Tuple[float, float]] = None
        last_conf: float = 0.0
        # counters
        missed: int = 0
        static_streak: int = 0
        floor_streak_frames: int = 0
        age_frames: int = 0

    # ------------------------ factories ------------------------

    @classmethod
    def from_image_polygon(
        cls,
        H_img2court: np.ndarray,
        court_polygon_image: List[Tuple[float, float]],
        court_gate_margin: float = 0.0,
        **kwargs
    ) -> "TennisBallActiveTracker":
        """Build from an IMAGE-space court polygon; warp once to PLANE space."""
        court_polygon_plane = cls._warp_poly_img2plane(court_polygon_image, H_img2court)
        if court_gate_margin != 0.0:
            court_polygon_plane = cls._inflate_polygon(court_polygon_plane, court_gate_margin)
        return cls(H_img2court, court_polygon_plane, **kwargs)

    # ------------------------ init ------------------------

    def __init__(
        self,
        H_img2court: np.ndarray,
        court_polygon_plane: List[Tuple[float, float]],
        fps: float = 30.0,
        *,
        # ---- inputs / parsing ----
        det_format: str = "xyxy",               # "xyxy" or "xywh"
        normalized: bool = False,               # True if YOLO coords are 0..1
        image_size: Tuple[int, int] = (1280, 720),
        only_class: Optional[int] = None,       # set to your ball class id, or None to accept all
        strict_court_gate: bool = False,        # gate in plane space (False lets outside candidates in)
        court_gate_margin: float = 0.05,        # inflate plane polygon by this fraction

        # ---- association (IMAGE space) ----
        max_missed: int = 12,
        max_speed_plane: float = 3000.0,        # px/s cap for near-track implied speed
        base_search_radius_px: Optional[float] = None,  # if None, 2% of image diagonal
        static_patience: int = 6,

        # ---- static / floor heuristics (IMAGE space) ----
        min_speed_plane: float = 20.0,          # px/s under which considered "static" for streaks
        rolling_speed_plane: float = 120.0,     # px/s: "rolling" threshold in image space
        airborne_window: int = 7,               # frames window for curvature
        a_min_px2: float = 0.45,                # curvature threshold (px/frame^2)
        floor_penalty_seconds: float = 1.0,     # after this long floor-like, bias to faster dets
        speed_bonus_beta: float = 0.45,         # cost divisor factor for high-speed dets when penalty active
        floor_penalty_warmup_frames: Optional[int] = None,

        # ---- bbox & confidence handling ----
        wh_ema_alpha: float = 0,              # smoothing for w,h
        conf_decay_per_miss: float = 0.90,      # confidence decay factor^missed

        # ---- suppression ----
        suppress_slow_outside: bool = True,
        outside_slow_speed_px: float = 40.0,    # if outside court AND speed < this -> suppress
        suppress_static_inside: bool = True,
        inside_static_speed_px: float = 12.0,   # if inside court AND speed < this -> suppress

        # ---- global fastest override ----
        always_return_fastest: bool = True,     # prefer global-fastest (post-suppression)
        fast_swap_margin: float = 0.15,         # require >=15% faster than local pick to override
        keep_id_on_swap: bool = True,           # keep same track_id when swapping target
        prev_match_radius_px: Optional[float] = None,  # NN radius for speed estimation; default 6% of diag

        # ---- misc ----
        verbose: bool = False,
    ):
        assert H_img2court.shape == (3, 3), "H_img2court must be 3x3"
        self.H = H_img2court.astype(float)
        self.H_inv = np.linalg.inv(self.H)       # optional for overlays / debug
        self.poly = list(court_polygon_plane)
        if court_gate_margin != 0.0:
            self.poly = self._inflate_polygon(self.poly, court_gate_margin)

        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.verbose = bool(verbose)

        # parsing
        self.det_format = det_format.lower()
        self.normalized = bool(normalized)
        self.image_size = tuple(image_size)
        self.only_class = only_class
        self.strict_court_gate = bool(strict_court_gate)

        # image geometry
        diag = float(np.hypot(*self.image_size))
        self.image_diag = diag

        # association thresholds
        self.max_missed = int(max_missed)
        self.max_speed_px = float(max_speed_plane)
        self.base_search_radius_px = float(base_search_radius_px) if base_search_radius_px is not None else max(12.0, 0.02 * diag)
        self.static_patience = int(static_patience)

        # static / floor heuristics
        self.min_speed_px = float(min_speed_plane)
        self.rolling_speed_px = float(rolling_speed_plane)
        self.airborne_window = int(airborne_window)
        self.a_min_px2 = float(a_min_px2)
        self.floor_penalty_frames = int(round(self.fps * float(floor_penalty_seconds)))
        self.speed_bonus_beta = float(speed_bonus_beta)
        self.floor_penalty_warmup_frames = (
            int(floor_penalty_warmup_frames)
            if floor_penalty_warmup_frames is not None
            else int(self.airborne_window)
        )

        # suppression flags/thresholds
        self.suppress_slow_outside = bool(suppress_slow_outside)
        self.outside_slow_speed_px = float(outside_slow_speed_px)
        self.suppress_static_inside = bool(suppress_static_inside)
        self.inside_static_speed_px = float(inside_static_speed_px)

        # bbox/conf handling
        self.wh_ema_alpha = float(np.clip(wh_ema_alpha, 0.0, 1.0))
        self.conf_decay_per_miss = float(np.clip(conf_decay_per_miss, 0.5, 0.99))

        # global-fastest override config
        self.always_return_fastest = bool(always_return_fastest)
        self.fast_swap_margin = float(max(0.0, fast_swap_margin))
        self.keep_id_on_swap = bool(keep_id_on_swap)
        self._prev_match_radius_px = (
            float(prev_match_radius_px) if prev_match_radius_px is not None else max(16.0, 0.06 * diag)
        )

        # track
        self.track: Optional[TennisBallActiveTracker.TrackState] = None
        self._next_tid = 1

        # last assoc debug
        self._last_assoc_dbg: Optional[Dict[str, Any]] = None

        # prev-frame detections (for per-detection speed estimation)
        self._prev_dets_img: List[Tuple[float, float]] = []

    # ------------------------ public API ------------------------

    def update_polygon(self, court_poly_px):
        court_poly_px = np.asarray(court_poly_px, dtype=np.float32)
        assert court_poly_px.ndim == 2 and court_poly_px.shape[1] == 2, "court_poly_px must be Nx2"
        self.court_poly_px = court_poly_px

    def update(self, yolo_dets: List[Union[Dict[str, Any], Tuple]]) -> Optional[Dict[str, Any]]:
        """
        yolo_dets items can be:
          - dict: {'bbox':[x1,y1,x2,y2], 'conf':0.87, 'class':0 (optional)}
          - tuple: (x1,y1,x2,y2, conf[, class])  or (cx,cy,w,h, conf[, class]) if det_format='xywh'
        Returns dict with keys: track_id, bbox [x1,y1,x2,y2], conf, plus extras & debug.
        """
        dbg = {
            "n_input": len(yolo_dets),
            "n_after_class": 0,
            "n_projected": 0,
            "n_in_court": 0,
            "assoc_cost": None,
            "used_source": None,
            "warning": None,
            "assoc_dbg": None,
        }

        # Parse and project to plane (for court gating)
        dets_all = self._parse_and_project(yolo_dets)
        dbg["n_after_class"] = len(dets_all)
        dbg["n_projected"] = sum(1 for d in dets_all if (d.cx_plane is not None and np.isfinite(d.cx_plane)))
        if self.verbose:
            for d in dets_all:
                print(f"IMG({d.cx_img:.1f},{d.cy_img:.1f}) -> PLANE({d.cx_plane},{d.cy_plane}) conf={d.conf:.3f}")

        # plane-space court gating (hard or soft)
        if self.strict_court_gate:
            dets = [d for d in dets_all if (d.cx_plane is not None and self._point_in_polygon((d.cx_plane, d.cy_plane), self.poly))]
        else:
            dets = [d for d in dets_all if d.cx_plane is not None]
        dbg["n_in_court"] = len(dets)

        # Annotate in_court + per-detection speed vs previous frame
        self._annotate_in_court(dets)
        self._annotate_speeds(dets)

        chosen = None
        source = "hold"

        if self.track is None:
            # Prefer the globally fastest valid detection
            if self.always_return_fastest:
                chosen = self._pick_fastest_global(dets)
                source = "global-fastest" if chosen is not None else "hold"
            # Fallback: initial pick still enforces suppression with unknown speed treated as 0
            if chosen is None:
                chosen = self._select_initial(dets)
            if chosen is not None:
                self._initialize_from_detection(chosen)
                if source == "hold":
                    source = "measurement"
            else:
                dbg["warning"] = "No viable detections after plane gating. Check homography and polygon warp."
        else:
            # Local (stable) association near last center
            chosen_local, cost = self._associate_by_image_last(dets, return_cost=True)
            dbg["assoc_cost"] = cost
            dbg["assoc_dbg"] = self._last_assoc_dbg
            chosen = chosen_local
            source = "measurement" if chosen_local is not None else "hold"

            # Global-fastest override (post-suppression)
            if self.always_return_fastest:
                fastest = self._pick_fastest_global(dets)
                if fastest is not None:
                    sp_fast = fastest.speed_img if (fastest.speed_img is not None and np.isfinite(fastest.speed_img)) else 0.0
                    sp_loc = (
                        chosen_local.speed_img if (chosen_local is not None and chosen_local.speed_img is not None and np.isfinite(chosen_local.speed_img))
                        else (self._img_speed_pxps() or 0.0)
                    )
                    if sp_fast > (1.0 + self.fast_swap_margin) * max(sp_loc, 1e-6):
                        chosen = fastest
                        source = "global-fastest-override"

            if chosen is not None:
                if source.startswith("global-fastest") and not self.keep_id_on_swap:
                    self._initialize_from_detection(chosen)  # resets ID
                else:
                    self._ingest_measurement(chosen)
            else:
                self.track.missed += 1

        # housekeeping / kill
        if self.track is not None:
            t = self.track
            t.age_frames += 1

            img_speed = self._img_speed_pxps()
            if self._is_airborne():
                t.static_streak = 0
            else:
                if (img_speed is not None) and (img_speed < self.min_speed_px):
                    t.static_streak += 1
                else:
                    t.static_streak = 0

            if self._is_floor_like(img_speed if img_speed is not None else 0.0):
                t.floor_streak_frames += 1
            else:
                t.floor_streak_frames = max(0, t.floor_streak_frames - 1)

            if t.missed > self.max_missed:
                self.track = None

        # store prev detections for next-frame speed estimation regardless of outcome
        self._prev_dets_img = [(d.cx_img, d.cy_img) for d in dets_all]

        if self.track is None:
            return None

        # compose bbox/conf (hold last measurement on misses)
        out_bbox: Optional[Tuple[float, float, float, float]] = self.track.last_bbox_img
        out_conf: float = float(self.track.last_conf)
        if self.track.missed > 0:
            out_conf *= self.conf_decay_per_miss ** max(1, self.track.missed)

        # Clip bbox to image bounds
        if out_bbox is not None:
            out_bbox = self._clip_bbox(out_bbox)
        if out_bbox is None:
            return None

        # FINAL OUTPUT SUPPRESSION: never emit slow-outside or static-inside
        last_plane = self.track.last_plane
        in_court_now = (last_plane is not None) and self._point_in_polygon(last_plane, self.poly)
        cur_spd = self._img_speed_pxps() or 0.0
        if in_court_now and self.suppress_static_inside and (cur_spd < self.inside_static_speed_px):
            return None
        if (not in_court_now) and self.suppress_slow_outside and (cur_spd < self.outside_slow_speed_px):
            return None

        cx, cy = self._bbox_center_xyxy(out_bbox)
        plane_speed = self._plane_speed_from_trace()

        return {
            "track_id": int(self.track.track_id),
            "bbox": [float(out_bbox[0]), float(out_bbox[1]), float(out_bbox[2]), float(out_bbox[3])],
            "conf": float(out_conf),
            # extras
            "source": "measurement",  # or "global-fastest"/"global-fastest-override" in debug
            "image_center": (float(cx), float(cy)),
            "image_speed_pxps": float(cur_spd),
            "plane_xy": self.track.last_plane if (self.track.last_plane is not None) else (None, None),
            "plane_speed": plane_speed if plane_speed is not None else None,
            "missed": int(self.track.missed),
            "is_static_warning": bool(self.track.static_streak >= max(1, self.static_patience // 2)),
            "floor_penalty_active": self._floor_penalty_active(),
            "floor_streak_frames": int(self.track.floor_streak_frames),
            # diagnostics
            "debug": {
                **dbg,
                "used_source": dbg.get("used_source"),
            },
        }

    def reset(self):
        self.track = None
        self._prev_dets_img = []

    # ------------------------ internals ------------------------

    def _parse_and_project(self, yolo_dets: List[Union[Dict[str, Any], Tuple]]) -> List[Detection]:
        dets: List[TennisBallActiveTracker.Detection] = []
        W, H = self.image_size

        for obj in yolo_dets:
            # unpack
            if isinstance(obj, dict):
                box = obj.get("bbox")
                conf = float(obj.get("conf", 1.0))
                cls = obj.get("class", None)
            else:
                box = obj[:4]
                conf = float(obj[4])
                cls = obj[5] if len(obj) > 5 else None

            # class filtering (strict)
            if self.only_class is not None:
                if cls is None or int(cls) != int(self.only_class):
                    continue

            # normalized?
            if self.normalized:
                bx = list(box)
                bx = [bx[0] * W, bx[1] * H, bx[2] * W, bx[3] * H]
                box = tuple(bx)

            # format conversion
            if self.det_format == "xywh":
                cx, cy, w, h = box
                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = cx + w / 2.0
                y2 = cy + h / 2.0
            else:
                x1, y1, x2, y2 = box

            # clip
            x1 = float(max(0.0, min(W - 1.0, x1)))
            y1 = float(max(0.0, min(H - 1.0, y1)))
            x2 = float(max(0.0, min(W - 1.0, x2)))
            y2 = float(max(0.0, min(H - 1.0, y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            cx_img, cy_img = self._bbox_center_xyxy((x1, y1, x2, y2))
            plane = self._apply_homography_img2plane((cx_img, cy_img), self.H)
            dets.append(self.Detection(
                bbox_img=(x1, y1, x2, y2),
                cx_img=cx_img, cy_img=cy_img,
                conf=conf, cls=cls,
                cx_plane=float(plane[0]) if plane is not None else None,
                cy_plane=float(plane[1]) if plane is not None else None,
            ))

        dets.sort(key=lambda d: d.conf, reverse=True)  # bias initial pick
        return dets

    def _initialize_from_detection(self, d: Detection):
        t = self.TrackState(track_id=self._next_tid)
        self._next_tid += 1

        # histories
        t.img_trace.append((d.cx_img, d.cy_img))
        if d.cx_plane is not None and d.cy_plane is not None:
            t.last_plane = (d.cx_plane, d.cy_plane)
            t.trace_plane.append(t.last_plane)

        # bbox & conf
        t.last_bbox_img = d.bbox_img
        t.last_wh_img = (d.bbox_img[2]-d.bbox_img[0], d.bbox_img[3]-d.bbox_img[1])
        t.last_conf = float(d.conf)

        self.track = t

    def _ingest_measurement(self, d: Detection):
        t = self.track
        # histories
        t.img_trace.append((d.cx_img, d.cy_img))
        # smooth w,h (EMA)
        w = d.bbox_img[2] - d.bbox_img[0]
        h = d.bbox_img[3] - d.bbox_img[1]
        if t.last_wh_img is None:
            t.last_wh_img = (float(w), float(h))
        else:
            a = self.wh_ema_alpha
            t.last_wh_img = (a*float(w) + (1-a)*t.last_wh_img[0],
                             a*float(h) + (1-a)*t.last_wh_img[1])
        # update bbox/conf
        t.last_bbox_img = d.bbox_img
        t.last_conf = float(d.conf)
        t.missed = 0
        # plane trace (debug)
        if d.cx_plane is not None and d.cy_plane is not None:
            t.last_plane = (d.cx_plane, d.cy_plane)
            t.trace_plane.append(t.last_plane)

    def _select_initial(self, dets: List[Detection]) -> Optional[Detection]:
        """Pick an initial detection enforcing slow/static suppression. Unknown speed treated as 0."""
        if not dets:
            return None
        kept = []
        for d in dets:
            spd = d.speed_img if (d.speed_img is not None and np.isfinite(d.speed_img)) else 0.0
            if (d.in_court is True) and self.suppress_static_inside and (spd < self.inside_static_speed_px):
                continue
            if (d.in_court is False) and self.suppress_slow_outside and (spd < self.outside_slow_speed_px):
                continue
            kept.append(d)
        if not kept:
            return None
        # Prefer in-court; tie-break by confidence
        in_court = [d for d in kept if d.in_court]
        pool = in_court if in_court else kept
        return max(pool, key=lambda d: d.conf)

    # --------- association: NN to last measured center (IMAGE space) + suppression ---------

    def _associate_by_image_last(self, dets: List[Detection], return_cost: bool = False):
        """
        Associate by IMAGE space to the last measured center.
        Priority: implied speed first, then confidence, then proximity.
        Suppress slow outside-court and static inside-court per thresholds.
        """
        if self.track is None or not dets or len(self.track.img_trace) == 0:
            self._last_assoc_dbg = {"n_candidates": 0}
            return (None, None) if return_cost else None

        px, py = self.track.img_trace[-1]  # last measured center
        missed = max(0, self.track.missed)
        recent_speed = self._img_speed_pxps() or 0.0

        # Adaptive search radius grows with speed & misses
        gate_r = self.base_search_radius_px * (1.0 + 0.002 * recent_speed) * (1.0 + 0.30 * missed)
        gate_r = float(np.clip(gate_r, self.base_search_radius_px, 0.25 * float(np.hypot(*self.image_size))))

        rej = {"radius": 0, "speed_cap": 0, "nan": 0, "outside_slow": 0, "inside_static": 0}
        kept = 0

        best = None
        best_speed = -np.inf
        best_conf = -np.inf
        best_dist = np.inf

        speed_cap = self.max_speed_px * (1.0 + 0.15 * missed)

        for d in dets:
            x, y = d.cx_img, d.cy_img
            if not np.isfinite(x) or not np.isfinite(y):
                rej["nan"] += 1
                continue

            dist = float(np.hypot(x - px, y - py))
            if dist > gate_r:
                rej["radius"] += 1
                continue

            spd = dist / max(self.dt, 1e-9)  # px/s relative to last center (association proxy)
            if spd > speed_cap:
                rej["speed_cap"] += 1
                continue

            # Soft court check even if strict gate already applied
            in_court = (d.cx_plane is not None) and self._point_in_polygon((d.cx_plane, d.cy_plane), self.poly)

            # Suppress slow outside-court
            if (not in_court) and self.suppress_slow_outside and (spd < self.outside_slow_speed_px):
                rej["outside_slow"] += 1
                continue

            # Suppress static inside-court
            if in_court and self.suppress_static_inside and (spd < self.inside_static_speed_px):
                rej["inside_static"] += 1
                continue

            kept += 1

            # Selection: higher speed, then higher confidence, then smaller distance
            if (spd > best_speed or
                (np.isclose(spd, best_speed) and (float(d.conf) > best_conf or
                                                  (np.isclose(float(d.conf), best_conf) and dist < best_dist)))):
                best = d
                best_speed = spd
                best_conf = float(d.conf)
                best_dist = dist

        # Debug
        self._last_assoc_dbg = {
            "gate_r": gate_r,
            "speed_cap": speed_cap,
            "missed": missed,
            "recent_speed": recent_speed,
            "rejections": rej,
            "kept": kept,
            "n_candidates": len(dets),
            "winner_speed_pxps": best_speed if best is not None else None,
        }

        best_cost = (-best_speed) if best is not None and np.isfinite(best_speed) else float("inf")
        return (best, best_cost) if return_cost else best

    # --------- telemetry ---------

    def _img_speed_pxps(self) -> Optional[float]:
        """Image-space speed (px/s) from last two measurements of the current track."""
        if self.track is None or len(self.track.img_trace) < 2:
            return None
        (x1, y1) = self.track.img_trace[-1]
        (x0, y0) = self.track.img_trace[-2]
        return float(np.hypot(x1 - x0, y1 - y0) / max(self.dt, 1e-9))

    def _plane_speed_from_trace(self) -> Optional[float]:
        """Approx plane speed from last two plane points (diagnostics only)."""
        if self.track is None or len(self.track.trace_plane) < 2:
            return None
        (x1, y1) = self.track.trace_plane[-1]
        (x0, y0) = self.track.trace_plane[-2]
        if x1 is None or y1 is None or x0 is None or y0 is None:
            return None
        dist = float(np.hypot(x1 - x0, y1 - y0))
        return dist / max(self.dt, 1e-9)

    # ------------------------ floor & airborne heuristics (IMAGE space) ------------------------

    def _floor_penalty_active(self) -> bool:
        return (
            self.track is not None
            and self.track.age_frames >= self.floor_penalty_warmup_frames
            and self.track.floor_streak_frames >= self.floor_penalty_frames
        )

    def _is_airborne(self) -> bool:
        """Treat as airborne when image-space vertical curvature is present. Needs enough points."""
        a = self._parabolic_ay()
        return (a is not None) and (abs(a) >= self.a_min_px2)

    def _is_floor_like(self, img_speed_pxps: float) -> bool:
        slow = (img_speed_pxps < self.rolling_speed_px)
        a = self._parabolic_ay()
        low_curvature = (abs(a) < self.a_min_px2) if a is not None else False
        return bool(slow and low_curvature)

    def _parabolic_ay(self) -> Optional[float]:
        if self.track is None:
            return None
        pts = list(self.track.img_trace)
        if len(pts) < max(4, self.airborne_window):
            return None
        y = np.array([p[1] for p in pts[-self.airborne_window:]], dtype=float)
        n = len(y)
        t = np.arange(n, dtype=float) - (n - 1) * 0.5  # frames, centered
        A = np.column_stack([t**2, t, np.ones(n)])
        try:
            coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
            return float(coeffs[0])  # px / frame^2
        except np.linalg.LinAlgError:
            return None

    # ------------------------ helpers ------------------------

    @staticmethod
    def _warp_poly_img2plane(poly_img: List[Tuple[float, float]], H_img2court: np.ndarray) -> List[Tuple[float, float]]:
        out = []
        for (x, y) in poly_img:
            v = np.array([x, y, 1.0], dtype=float)
            w = H_img2court @ v
            if (not np.isfinite(w).all()) or abs(w[2]) < 1e-8:
                raise ValueError("Invalid homography projection for court vertex; check H_img2court and input points.")
            out.append((float(w[0] / w[2]), float(w[1] / w[2])))
        if len(out) < 3:
            raise ValueError("Warped court polygon has <3 vertices; check H_img2court.")
        return out

    @staticmethod
    def _inflate_polygon(poly: List[Tuple[float, float]], margin: float) -> List[Tuple[float, float]]:
        """Inflate polygon by moving each vertex away from centroid by (1+margin)."""
        P = np.array(poly, dtype=float)
        c = P.mean(axis=0)
        return [tuple((c + (p - c) * (1.0 + margin)).tolist()) for p in P]

    @staticmethod
    def _apply_homography_img2plane(pt_xy: Tuple[float, float], H_img2court: np.ndarray) -> Optional[Tuple[float, float]]:
        x, y = pt_xy
        v = np.array([x, y, 1.0], dtype=float)
        w = H_img2court @ v
        if (not np.isfinite(w).all()) or abs(w[2]) < 1e-8:
            return None
        return (float(w[0] / w[2]), float(w[1] / w[2]))

    @staticmethod
    def _apply_homography_plane2img(pt_xy: Tuple[float, float], H_plane2img: np.ndarray) -> Optional[Tuple[float, float]]:
        x, y = pt_xy
        v = np.array([x, y, 1.0], dtype=float)
        w = H_plane2img @ v
        if (not np.isfinite(w).all()) or abs(w[2]) < 1e-8:
            return None
        return (float(w[0] / w[2]), float(w[1] / w[2]))

    @staticmethod
    def _point_in_polygon(pt: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
        """Even-odd rule with simple on-edge handling treated as inside."""
        x, y = pt
        n = len(poly)
        if n < 3:
            return False
        eps = 1e-9
        inside = False
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]

            # On-edge check (bounding-box + cross-product collinearity)
            if min(x1, x2) - eps <= x <= max(x1, x2) + eps and min(y1, y2) - eps <= y <= max(y1, y2) + eps:
                dx, dy = x2 - x1, y2 - y1
                if abs(dx * (y - y1) - dy * (x - x1)) <= eps:
                    return True

            # Ray casting (skip horizontal edges consistently)
            if (y1 > y) != (y2 > y):
                x_int = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
                if x_int >= x:
                    inside = not inside
        return inside

    @staticmethod
    def _bbox_center_xyxy(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = b
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    def _clip_bbox(self, b: Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float, float]]:
        W, H = self.image_size
        x1, y1, x2, y2 = b
        x1 = max(0.0, min(W - 1.0, x1))
        x2 = max(0.0, min(W - 1.0, x2))
        y1 = max(0.0, min(H - 1.0, y1))
        y2 = max(0.0, min(H - 1.0, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    # ------------------------ speed & global pick ------------------------

    def _annotate_in_court(self, dets: List[Detection]) -> None:
        for d in dets:
            d.in_court = (d.cx_plane is not None) and self._point_in_polygon((d.cx_plane, d.cy_plane), self.poly)

    def _annotate_speeds(self, dets: List[Detection]) -> None:
        """
        Estimate per-detection speed via nearest-neighbor to previous frame centers.
        After the first frame, always assign a finite value; clamp to a sanity cap.
        """
        if not self._prev_dets_img:
            for d in dets:
                d.speed_img = None  # first frame → unknown
            return

        cap = self.max_speed_px * 3.0  # sanity cap against wild matches
        r = self._prev_match_radius_px
        r2 = r * r

        for d in dets:
            best_d2 = None
            # simple NN; if within radius, accept; else still compute but we'll clamp
            for (px, py) in self._prev_dets_img:
                dx, dy = d.cx_img - px, d.cy_img - py
                d2 = dx*dx + dy*dy
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
            if best_d2 is None:
                d.speed_img = 0.0
                continue
            # If the best match is absurdly far, still produce a number but clamp
            dist = float(np.sqrt(best_d2))
            spd = dist / max(self.dt, 1e-9)
            if best_d2 > r2:
                # outside NN radius → likely new/ambiguous; treat as small but nonzero
                spd = min(spd, self.outside_slow_speed_px * 0.9)
            d.speed_img = float(min(spd, cap))

    def _pick_fastest_global(self, dets: List[Detection]) -> Optional[Detection]:
        """Return the fastest detection after applying slow-outside/static-inside suppression."""
        best = None
        best_spd = -np.inf
        for d in dets:
            spd = d.speed_img if (d.speed_img is not None and np.isfinite(d.speed_img)) else 0.0

            # Suppression rules
            if (d.in_court is False) and self.suppress_slow_outside and (spd < self.outside_slow_speed_px):
                continue
            if (d.in_court is True) and self.suppress_static_inside and (spd < self.inside_static_speed_px):
                continue

            # Hard sanity cap
            if spd > self.max_speed_px * 3.0:
                continue

            if (spd > best_spd) or (np.isclose(spd, best_spd) and float(d.conf) > float(best.conf if best else -np.inf)):
                best = d
                best_spd = spd
        return best
