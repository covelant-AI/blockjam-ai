from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple, Optional, Dict, Any, Deque, Union
import numpy as np


class TennisBallActiveMultiTracker:
    """
    Multi-hypothesis tennis ball tracker (detection-only; no prediction).
      - Parallel pool of tracks → no switch lag.
      - Robust association:
          * Mutual-nearest matching first
          * Direction gating (velocity cosine)
          * IoU rescue gate
          * Second-chance expansion when a track fails
      - Returns ONLY the fastest valid ball after hard suppression:
          * drop slow-outside
          * drop static-inside
      - Provides debug snapshots of inactive tracks and association rejections.

    All speed thresholds are in pixels/second (px/s).
    """

    # ------------------------ structs ------------------------

    @dataclass
    class Detection:
        bbox_img: Tuple[float, float, float, float]   # (x1,y1,x2,y2) in IMAGE px
        cx_img: float
        cy_img: float
        conf: float
        cls: Optional[int] = None
        cx_plane: Optional[float] = None              # PLANE coords (for court gating only)
        cy_plane: Optional[float] = None
        in_court: Optional[bool] = None
        speed_img_seed: Optional[float] = None        # px/s vs prev frame (for spawning)

    @dataclass
    class Track:
        track_id: int
        img_trace: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=24))
        plane_trace: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=64))
        last_bbox: Optional[Tuple[float, float, float, float]] = None
        last_wh: Optional[Tuple[float, float]] = None
        last_conf: float = 0.0
        last_plane: Optional[Tuple[float, float]] = None
        last_speed_pxps: float = 0.0
        missed: int = 0
        age_frames: int = 0

    # ------------------------ factories ------------------------

    @classmethod
    def from_image_polygon(
        cls,
        H_img2court: np.ndarray,
        court_polygon_image: List[Tuple[float, float]],
        court_gate_margin: float = 0.0,
        **kwargs
    ) -> "TennisBallActiveMultiTracker":
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
        det_format: str = "xyxy",               # "xyxy" or "xywh"
        normalized: bool = False,               # True if YOLO coords are 0..1
        image_size: Tuple[int, int] = (1280, 720),
        only_class: Optional[int] = None,
        strict_court_gate: bool = False,
        court_gate_margin: float = 0.05,

        # Association / gating
        max_missed: int = 3,
        max_speed_px: float = 4000.0,           # cap on implied assoc speed
        base_search_radius_px: Optional[float] = 0.1,  # if None, 2% of diag
        dir_cos_min: float = -1,              # min cosine(track_vel, to_det) to accept (if velocity known)
        iou_gate: float = 0.10,                 # accept if IoU ≥ this even if cosine fails
        second_chance_expand: float = 2.0,      # expand radius multiplier on second attempt
        speed_cap_expand: float = 2.5,          # expand speed cap multiplier on second attempt

        # Suppression (hard)
        suppress_slow_outside: bool = True,
        outside_slow_speed_px: float = 40.0,    # outside court & speed < this -> invalid
        suppress_static_inside: bool = True,
        inside_static_speed_px: float = 12.0,   # inside court & speed < this -> invalid

        # Track pool
        max_tracks: int = 2,
        spawn_min_conf: float = 0.10,
        replace_margin_pxps: float = 200.0,

        # Pre-association per-frame NMS (helps with dup boxes)
        do_nms: bool = False,
        nms_iou: float = 0.7,

        # Active selection hysteresis (prevents flicker between close speeds)
        active_min_hold_frames: int = 3,
        active_win_margin: float = 0.10,        # challenger must be 10% faster
        active_win_streak: int = 2,             # …for N consecutive frames

        # Debug
        include_inactive_snapshot: bool = True,
        debug_max_inactive: int = 8,

        # Misc
        verbose: bool = False,
    ):
        assert H_img2court.shape == (3, 3), "H_img2court must be 3x3"
        self.H = H_img2court.astype(float)
        self.poly = list(court_polygon_plane)
        if court_gate_margin != 0.0:
            self.poly = self._inflate_polygon(self.poly, court_gate_margin)

        self.fps = float(fps)
        self.dt = 1.0 / self.fps

        self.det_format = det_format.lower()
        self.normalized = bool(normalized)
        self.image_size = tuple(image_size)
        self.only_class = only_class
        self.strict_court_gate = bool(strict_court_gate)

        diag = float(np.hypot(*self.image_size))
        self.diag = diag
        self.base_search_radius_px = float(base_search_radius_px) if base_search_radius_px is not None else max(12.0, 0.02 * diag)

        # thresholds
        self.max_missed = int(max_missed)
        self.max_speed_px = float(max_speed_px)
        self.dir_cos_min = float(dir_cos_min)
        self.iou_gate = float(iou_gate)
        self.second_chance_expand = float(second_chance_expand)
        self.speed_cap_expand = float(speed_cap_expand)

        self.suppress_slow_outside = bool(suppress_slow_outside)
        self.outside_slow_speed_px = float(outside_slow_speed_px)
        self.suppress_static_inside = bool(suppress_static_inside)
        self.inside_static_speed_px = float(inside_static_speed_px)

        # pool config
        self.max_tracks = int(max_tracks)
        self.spawn_min_conf = float(spawn_min_conf)
        self.replace_margin_pxps = float(replace_margin_pxps)

        # NMS
        self.do_nms = bool(do_nms)
        self.nms_iou = float(nms_iou)

        # hysteresis
        self.active_min_hold_frames = int(active_min_hold_frames)
        self.active_win_margin = float(active_win_margin)
        self.active_win_streak = int(active_win_streak)
        self._active_id: Optional[int] = None
        self._active_hold: int = 0
        self._challenger_id: Optional[int] = None
        self._challenger_streak: int = 0

        # debug
        self.include_inactive_snapshot = bool(include_inactive_snapshot)
        self.debug_max_inactive = int(debug_max_inactive)
        self.verbose = bool(verbose)

        # state
        self.tracks: Dict[int, TennisBallActiveMultiTracker.Track] = {}
        self._next_tid = 1

        # previous frame detection centers (for speed seeding)
        self._prev_det_centers: List[Tuple[float, float]] = []
        self._prev_match_radius = 0.06 * diag  # for speed seeding

    # ------------------------ public API ------------------------

    def update(self, yolo_dets: List[Union[Dict[str, Any], Tuple]]) -> Optional[Dict[str, Any]]:
        """
        Input detections: dicts or tuples (xyxy/xywh + conf [+class]).
        Returns a dict for the fastest valid ball (or None), with debug['inactive_tracks'] and debug['assoc'].
        """
        # Parse & project
        dets_all = self._parse_and_project(yolo_dets)
        if self.do_nms and dets_all:
            dets_all = self._nms(dets_all, self.nms_iou)

        # Court gating
        if self.strict_court_gate:
            dets = [d for d in dets_all if (d.cx_plane is not None and self._point_in_polygon((d.cx_plane, d.cy_plane), self.poly))]
        else:
            dets = [d for d in dets_all if d.cx_plane is not None]

        # Annotate
        self._annotate_in_court(dets)
        self._seed_speeds_from_prev(dets)

        # Association (mutual-nearest + direction + IoU rescue)
        assoc_dbg = []
        T = list(self.tracks.values())
        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()

        # Build all candidate pairs with gating
        candidates = []  # (ti, di, dist, implied_speed, cos, iou)
        for ti, t in enumerate(T):
            if t.last_bbox is None or len(t.img_trace) == 0:
                continue
            (px, py) = t.img_trace[-1]
            v = None
            if len(t.img_trace) >= 2:
                (x0, y0) = t.img_trace[-2]
                vx, vy = px - x0, py - y0
                n = np.hypot(vx, vy)
                v = (vx / n, vy / n) if n > 1e-6 else None

            recent_speed = self._track_speed(t) or 0.0
            gate_r = self.base_search_radius_px * (1.0 + 0.002 * recent_speed) * (1.0 + 0.30 * t.missed)
            gate_r = float(np.clip(gate_r, self.base_search_radius_px, 0.25 * self.diag))
            speed_cap = self.max_speed_px * (1.0 + 0.15 * t.missed)

            for di, d in enumerate(dets):
                dx, dy = d.cx_img - px, d.cy_img - py
                dist = float(np.hypot(dx, dy))
                if dist > gate_r:
                    continue
                implied_speed = dist / max(self.dt, 1e-9)
                if implied_speed > speed_cap:
                    continue

                cos_ok = True
                cos_val = None
                if v is not None:
                    dot = v[0] * (dx / (dist + 1e-9)) + v[1] * (dy / (dist + 1e-9))
                    cos_val = float(dot)
                    cos_ok = (cos_val >= self.dir_cos_min)

                iou_val = self._bbox_iou(t.last_bbox, d.bbox_img)
                if not cos_ok and (iou_val is None or iou_val < self.iou_gate):
                    # fail both direction and IoU rescue
                    continue

                candidates.append((ti, di, dist, implied_speed, cos_val, iou_val))

        # Mutual-nearest first
        if candidates:
            # nearest det for each track
            by_track: Dict[int, Tuple[int, float]] = {}
            # nearest track for each det
            by_det: Dict[int, Tuple[int, float]] = {}
            for ti, di, dist, *_ in candidates:
                if (ti not in by_track) or (dist < by_track[ti][1]):
                    by_track[ti] = (di, dist)
                if (di not in by_det) or (dist < by_det[di][1]):
                    by_det[di] = (ti, dist)

            mut_pairs = []
            for ti, (di, _) in by_track.items():
                if di in by_det and by_det[di][0] == ti:
                    # find the full candidate to recover implied_speed etc.
                    for c in candidates:
                        if c[0] == ti and c[1] == di:
                            mut_pairs.append(c); break

            # Assign mutuals
            mut_pairs.sort(key=lambda x: x[2])  # closest first
            for ti, di, dist, implied_speed, cos_val, iou_val in mut_pairs:
                if (T[ti].track_id in assigned_tracks) or (di in assigned_dets):
                    continue
                self._ingest(T[ti], dets[di], implied_speed)
                assigned_tracks.add(T[ti].track_id)
                assigned_dets.add(di)
                assoc_dbg.append({
                    "track_id": T[ti].track_id, "det_idx": di, "type": "mutual",
                    "dist": float(dist), "implied_speed": float(implied_speed),
                    "cos": cos_val, "iou": iou_val
                })

        # Second pass: greedy on remaining candidates
        rem = [c for c in candidates if (T[c[0]].track_id not in assigned_tracks and c[1] not in assigned_dets)]
        rem.sort(key=lambda x: x[2])
        for ti, di, dist, implied_speed, cos_val, iou_val in rem:
            if (T[ti].track_id in assigned_tracks) or (di in assigned_dets):
                continue
            self._ingest(T[ti], dets[di], implied_speed)
            assigned_tracks.add(T[ti].track_id)
            assigned_dets.add(di)
            assoc_dbg.append({
                "track_id": T[ti].track_id, "det_idx": di, "type": "greedy",
                "dist": float(dist), "implied_speed": float(implied_speed),
                "cos": cos_val, "iou": iou_val
            })

        # Second-chance expansion for still-unmatched tracks
        expanded_pairs = []
        for ti, t in enumerate(T):
            if t.track_id in assigned_tracks or t.last_bbox is None or len(t.img_trace) == 0:
                continue
            (px, py) = t.img_trace[-1]
            gate_r = self.base_search_radius_px * self.second_chance_expand * (1.0 + 0.30 * t.missed)
            gate_r = float(np.clip(gate_r, self.base_search_radius_px, 0.45 * self.diag))
            speed_cap = self.max_speed_px * self.speed_cap_expand * (1.0 + 0.15 * t.missed)
            best = None
            for di, d in enumerate(dets):
                if di in assigned_dets:
                    continue
                dist = float(np.hypot(d.cx_img - px, d.cy_img - py))
                if dist > gate_r:
                    continue
                implied_speed = dist / max(self.dt, 1e-9)
                if implied_speed > speed_cap:
                    continue
                # allow without direction here but keep IoU rescue
                iou_val = self._bbox_iou(t.last_bbox, d.bbox_img)
                if iou_val is not None and iou_val >= self.iou_gate:
                    best = (di, dist, implied_speed, None, iou_val) if (best is None or dist < best[1]) else best
                else:
                    best = (di, dist, implied_speed, None, iou_val) if (best is None or dist < best[1]) else best
            if best is not None:
                di, dist, implied_speed, cos_val, iou_val = best
                self._ingest(t, dets[di], implied_speed)
                assigned_tracks.add(t.track_id)
                assigned_dets.add(di)
                expanded_pairs.append({
                    "track_id": t.track_id, "det_idx": di, "type": "second_chance",
                    "dist": float(dist), "implied_speed": float(implied_speed),
                    "cos": cos_val, "iou": iou_val
                })
        assoc_dbg.extend(expanded_pairs)

        # Age & miss unassigned tracks
        for t in T:
            if t.track_id not in assigned_tracks:
                t.missed += 1
                t.age_frames += 1

        # Spawn new tracks from unmatched detections
        unmatched_dets = [dets[i] for i in range(len(dets)) if i not in assigned_dets and float(dets[i].conf) >= self.spawn_min_conf]
        self._spawn_from(unmatched_dets)

        # Kill dead tracks
        dead_ids = [tid for tid, t in self.tracks.items() if t.missed > self.max_missed]
        for tid in dead_ids:
            del self.tracks[tid]

        # Select fastest valid active track (hard suppression + hysteresis)
        active = self._select_active_hysteresis()

        # Update prev detection centers for next-frame speed seeding
        self._prev_det_centers = [(d.cx_img, d.cy_img) for d in dets_all]

        # Prepare debug
        debug = {
            "assoc": assoc_dbg,
            "n_tracks": len(self.tracks),
        }
        if self.include_inactive_snapshot:
            debug["inactive_tracks"] = self._inactive_snapshot(active_id=active.track_id if active else None)

        if active is None:
            return None

        # Output active only
        out_bbox = self._clip_bbox(active.last_bbox) if active.last_bbox is not None else None
        if out_bbox is None:
            return None

        cx, cy = self._bbox_center_xyxy(out_bbox)
        in_court_now = (active.last_plane is not None) and self._point_in_polygon(active.last_plane, self.poly)
        return {
            "track_id": int(active.track_id),
            "bbox": [float(out_bbox[0]), float(out_bbox[1]), float(out_bbox[2]), float(out_bbox[3])],
            "conf": float(active.last_conf),
            "image_center": (float(cx), float(cy)),
            "image_speed_pxps": float(active.last_speed_pxps),
            "in_court": bool(in_court_now),
            "plane_xy": active.last_plane if (active.last_plane is not None) else (None, None),
            "missed": int(active.missed),
            "debug": debug,
        }

    def reset(self):
        self.tracks.clear()
        self._next_tid = 1
        self._prev_det_centers = []
        self._active_id = None
        self._active_hold = 0
        self._challenger_id = None
        self._challenger_streak = 0

    # ------------------------ internals ------------------------

    def _parse_and_project(self, yolo_dets: List[Union[Dict[str, Any], Tuple]]) -> List[Detection]:
        dets: List[TennisBallActiveMultiTracker.Detection] = []
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

            # class filtering
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
            v = np.array([cx_img, cy_img, 1.0], dtype=float)
            w = self.H @ v
            if (not np.isfinite(w).all()) or abs(w[2]) < 1e-8:
                continue
            cx_plane, cy_plane = float(w[0] / w[2]), float(w[1] / w[2])

            dets.append(self.Detection(
                bbox_img=(x1, y1, x2, y2),
                cx_img=cx_img, cy_img=cy_img,
                conf=conf, cls=cls,
                cx_plane=cx_plane, cy_plane=cy_plane,
            ))

        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets

    def _annotate_in_court(self, dets: List[Detection]) -> None:
        for d in dets:
            d.in_court = self._point_in_polygon((d.cx_plane, d.cy_plane), self.poly)

    def _seed_speeds_from_prev(self, dets: List[Detection]) -> None:
        """Seed per-detection speeds vs previous frame centers (for new tracks)."""
        if not self._prev_det_centers:
            for d in dets:
                d.speed_img_seed = None
            return
        r2 = (self._prev_match_radius ** 2)
        cap = self.max_speed_px * 3.0
        for d in dets:
            best = None
            for (px, py) in self._prev_det_centers:
                dx, dy = d.cx_img - px, d.cy_img - py
                d2 = dx*dx + dy*dy
                if best is None or d2 < best:
                    best = d2
            if best is None:
                d.speed_img_seed = 0.0
                continue
            dist = float(np.sqrt(best))
            spd = dist / max(self.dt, 1e-9)
            if best > r2:
                spd = min(spd, self.outside_slow_speed_px * 0.9)
            d.speed_img_seed = float(min(spd, cap))

    # ---- track lifecycle ----

    def _ingest(self, t: Track, d: Detection, implied_speed: float):
        # histories
        t.img_trace.append((d.cx_img, d.cy_img))
        t.age_frames += 1
        t.missed = 0

        # bbox/size smoothing (EMA on size)
        w = d.bbox_img[2] - d.bbox_img[0]
        h = d.bbox_img[3] - d.bbox_img[1]
        if t.last_wh is None:
            t.last_wh = (float(w), float(h))
        else:
            a = 0.6
            t.last_wh = (a*float(w) + (1-a)*t.last_wh[0],
                         a*float(h) + (1-a)*t.last_wh[1])

        # For assignment robustness we keep raw detection bbox; rendering can be smoothed client-side
        t.last_bbox = d.bbox_img
        t.last_conf = float(d.conf)
        t.last_speed_pxps = float(np.clip(implied_speed, 0.0, self.max_speed_px * 3.0))

        if d.cx_plane is not None and d.cy_plane is not None:
            t.last_plane = (d.cx_plane, d.cy_plane)
            t.plane_trace.append(t.last_plane)

    def _spawn_from(self, dets: List[Detection]) -> None:
        if not dets:
            return
        free_slots = max(0, self.max_tracks - len(self.tracks))
        to_spawn = []
        dets_sorted = sorted(dets, key=lambda d: ((d.speed_img_seed or 0.0), float(d.conf)), reverse=True)
        for d in dets_sorted:
            if free_slots > 0:
                to_spawn.append(d); free_slots -= 1
            else:
                worst_tid = self._pick_replacement_target()
                if worst_tid is None:
                    break
                worst = self.tracks[worst_tid]
                new_speed = (d.speed_img_seed or 0.0)
                if new_speed > worst.last_speed_pxps + self.replace_margin_pxps:
                    del self.tracks[worst_tid]
                    to_spawn.append(d)
        for d in to_spawn:
            t = self.Track(track_id=self._next_tid)
            self._next_tid += 1
            t.img_trace.append((d.cx_img, d.cy_img))
            t.last_bbox = d.bbox_img
            t.last_wh = (d.bbox_img[2] - d.bbox_img[0], d.bbox_img[3] - d.bbox_img[1])
            t.last_conf = float(d.conf)
            if d.cx_plane is not None and d.cy_plane is not None:
                t.last_plane = (d.cx_plane, d.cy_plane)
                t.plane_trace.append(t.last_plane)
            t.last_speed_pxps = float(d.speed_img_seed or 0.0)
            self.tracks[t.track_id] = t

    def _pick_replacement_target(self) -> Optional[int]:
        if not self.tracks:
            return None
        # Worst = most missed, then slowest, then oldest
        worst_tid = None
        worst_key = None
        for tid, t in self.tracks.items():
            key = (t.missed, -t.last_speed_pxps, -t.age_frames)
            if worst_key is None or key > worst_key:
                worst_key = key
                worst_tid = tid
        return worst_tid

    # ---- active selection with hysteresis ----

    def _select_active_hysteresis(self) -> Optional[Track]:
        ranked = []
        for t in self.tracks.items():
            pass
        # rank after suppression
        for t in self.tracks.values():
            if t.last_bbox is None:
                continue
            spd = t.last_speed_pxps if np.isfinite(t.last_speed_pxps) else 0.0
            in_court = (t.last_plane is not None) and self._point_in_polygon(t.last_plane, self.poly)
            if in_court and self.suppress_static_inside and (spd < self.inside_static_speed_px):
                continue
            if (not in_court) and self.suppress_slow_outside and (spd < self.outside_slow_speed_px):
                continue
            ranked.append((spd, t))
        if not ranked:
            self._active_id = None
            self._active_hold = 0
            self._challenger_id = None
            self._challenger_streak = 0
            return None
        ranked.sort(key=lambda x: x[0], reverse=True)
        best_spd, best = ranked[0]

        if self._active_id is None or self._active_id not in self.tracks:
            self._active_id = best.track_id
            self._active_hold = self.active_min_hold_frames
            self._challenger_id = None
            self._challenger_streak = 0
            return best

        cur = self.tracks[self._active_id]
        cur_spd = cur.last_speed_pxps if np.isfinite(cur.last_speed_pxps) else 0.0

        if best.track_id == self._active_id:
            self._active_hold = max(self._active_hold - 1, 0)
            self._challenger_id = None
            self._challenger_streak = 0
            return cur

        faster = best_spd > (1.0 + self.active_win_margin) * max(cur_spd, 1e-6)
        if faster:
            if self._challenger_id == best.track_id:
                self._challenger_streak += 1
            else:
                self._challenger_id = best.track_id
                self._challenger_streak = 1
        else:
            self._challenger_id = None
            self._challenger_streak = 0

        if (self._active_hold == 0) and faster and (self._challenger_streak >= self.active_win_streak):
            self._active_id = best.track_id
            self._active_hold = self.active_min_hold_frames
            self._challenger_id = None
            self._challenger_streak = 0
            return best
        else:
            self._active_hold = max(self._active_hold - 1, 0)
            return cur

    # ------------------------ debug helpers ------------------------

    def _inactive_snapshot(self, active_id: Optional[int]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for tid, t in self.tracks.items():
            if active_id is not None and tid == active_id:
                continue
            bbox = self._clip_bbox(t.last_bbox) if t.last_bbox is not None else None
            if bbox is None:
                continue
            cx, cy = self._bbox_center_xyxy(bbox)
            in_court_now = (t.last_plane is not None) and self._point_in_polygon(t.last_plane, self.poly)
            items.append({
                "track_id": int(tid),
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "image_center": (float(cx), float(cy)),
                "image_speed_pxps": float(t.last_speed_pxps if np.isfinite(t.last_speed_pxps) else 0.0),
                "in_court": bool(in_court_now),
                "missed": int(t.missed),
                "age_frames": int(t.age_frames),
                "conf": float(t.last_conf),
            })
            if len(items) >= self.debug_max_inactive:
                break
        items.sort(key=lambda x: x["image_speed_pxps"], reverse=True)
        return items

    # ------------------------ helpers ------------------------

    def _track_speed(self, t: Track) -> Optional[float]:
        if len(t.img_trace) < 2:
            return None
        (x1, y1) = t.img_trace[-1]
        (x0, y0) = t.img_trace[-2]
        return float(np.hypot(x1 - x0, y1 - y0) / max(self.dt, 1e-9))

    @staticmethod
    def _warp_poly_img2plane(poly_img: List[Tuple[float, float]], H_img2court: np.ndarray) -> List[Tuple[float, float]]:
        out = []
        for (x, y) in poly_img:
            v = np.array([x, y, 1.0], dtype=float)
            w = H_img2court @ v
            if (not np.isfinite(w).all()) or abs(w[2]) < 1e-8:
                raise ValueError("Invalid homography projection for court vertex.")
            out.append((float(w[0] / w[2]), float(w[1] / w[2])))
        if len(out) < 3:
            raise ValueError("Warped court polygon has <3 vertices.")
        return out

    @staticmethod
    def _inflate_polygon(poly: List[Tuple[float, float]], margin: float) -> List[Tuple[float, float]]:
        P = np.array(poly, dtype=float)
        c = P.mean(axis=0)
        return [tuple((c + (p - c) * (1.0 + margin)).tolist()) for p in P]

    @staticmethod
    def _point_in_polygon(pt: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
        x, y = pt
        n = len(poly)
        if n < 3:
            return False
        eps = 1e-9
        inside = False
        for i in range(n):
            x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
            if min(x1, x2) - eps <= x <= max(x1, x2) + eps and min(y1, y2) - eps <= y <= max(y1, y2) + eps:
                dx, dy = x2 - x1, y2 - y1
                if abs(dx * (y - y1) - dy * (x - x1)) <= eps:
                    return True
            if (y1 > y) != (y2 > y):
                x_int = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
                if x_int >= x:
                    inside = not inside
        return inside

    @staticmethod
    def _bbox_center_xyxy(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = b
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @staticmethod
    def _bbox_iou(a: Optional[Tuple[float, float, float, float]],
                  b: Optional[Tuple[float, float, float, float]]) -> Optional[float]:
        if a is None or b is None:
            return None
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = a_area + b_area - inter
        return float(inter / max(union, 1e-9))

    def _clip_bbox(self, b: Optional[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
        if b is None:
            return None
        W, H = self.image_size
        x1, y1, x2, y2 = b
        x1 = max(0.0, min(W - 1.0, x1))
        x2 = max(0.0, min(W - 1.0, x2))
        y1 = max(0.0, min(H - 1.0, y1))
        y2 = max(0.0, min(H - 1.0, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    # ------------------------ NMS ------------------------

    def _nms(self, dets: List[Detection], iou_thr: float) -> List[Detection]:
        if not dets:
            return dets
        dets_sorted = sorted(dets, key=lambda d: d.conf, reverse=True)
        keep: List[TennisBallActiveMultiTracker.Detection] = []
        while dets_sorted:
            d = dets_sorted.pop(0)
            keep.append(d)
            rest = []
            for e in dets_sorted:
                iou = self._bbox_iou(d.bbox_img, e.bbox_img) or 0.0
                if iou < iou_thr:
                    rest.append(e)
            dets_sorted = rest
        return keep
