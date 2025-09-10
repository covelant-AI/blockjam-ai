from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple, Optional, Dict, Any, Deque, Union
import numpy as np
import cv2


class TennisBallActiveMultiTracker:
    """
    Multi-hypothesis tennis ball tracker (detection-only; no motion prediction).

    Key points (unchanged basics):
      - Mutual-nearest association with expected-step ranking and direction gating.
      - Additive speed-based search radius (EMA + decay).
      - Per-frame NMS and anti-duplication.
      - Lag-aware instantaneous pixel speed using effective Δt across misses.

    New robustness for world speed:
      - Compute plane-difference speed (court meters) only when projection is trustworthy.
      - Guard with local homography Jacobian singular values; otherwise fall back to Jacobian-bound estimate.
      - Clamp to a physical ceiling. Export speed_mps / speed_kmh / speed_mph for visualizers.
    """

    # ------------------------ structs ------------------------

    @dataclass
    class Detection:
        bbox_img: Tuple[float, float, float, float]
        cx_img: float
        cy_img: float
        conf: float
        cls: Optional[int] = None
        cx_plane: Optional[float] = None
        cy_plane: Optional[float] = None
        in_court: Optional[bool] = None

    @dataclass
    class Track:
        track_id: int
        img_trace: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=24))
        plane_trace: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=64))
        last_bbox: Optional[Tuple[float, float, float, float]] = None
        last_wh: Optional[Tuple[float, float]] = None
        last_conf: float = 0.0
        last_plane: Optional[Tuple[float, float]] = None
        last_speed_pxps: float = 0.0      # instantaneous pixel speed
        smooth_speed_pxps: float = 0.0    # EMA of pixel speed
        last_step_dt_s: float = 0.0
        missed: int = 0
        age_frames: int = 0
        bounce_cooldown: int = 0
        # world speeds (m/s)
        last_speed_mps: float = 0.0
        smooth_speed_mps: float = 0.0

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
        det_format: str = "xyxy",
        normalized: bool = False,
        image_size: Tuple[int, int] = (1280, 720),
        only_class: Optional[int] = None,
        strict_court_gate: bool = False,
        court_gate_margin: float = 0,

        # Association / gating
        max_missed: int = 10,
        max_speed_px: float = 8000,
        base_search_radius_px: Optional[float] = 0.04,  # None→2% diag; <1→frac of diag; ≥1→px
        dir_cos_min: float = -1.0,
        iou_gate: float = 0.10,
        iou_bypass_speed_gate: bool = False,
        second_chance_expand: float = 2.0,
        speed_cap_expand: float = 2.5,

        # Bounce / robustness
        gate_size_mult: float = 1.5,
        bounce_cooldown_frames: int = 5,
        bounce_gate_boost: float = 1.0,
        bounce_speed_cap_boost: float = 3.0,
        dir_cos_min_bounce: float = -1.0,
        vy_flip_min_pxps: float = 200.0,
        airborne_window: int = 7,
        a_min_px2: float = 0.45,

        # Suppression
        suppress_slow_outside: bool = True,
        outside_slow_speed_px: float = 30,
        suppress_static_inside: bool = True,
        inside_static_speed_px: float = 10,
        static_inside_multi_only: bool = True,

        # Track pool
        max_tracks: int = 10,
        spawn_min_conf: float = 0.10,
        replace_margin_pxps: float = 200.0,

        # Per-frame NMS
        do_nms: bool = True,
        nms_iou: float = 0.7,

        # Active selection
        active_min_hold_frames: int = 0,
        active_win_margin: float = 0.10,
        active_win_streak: int = 2,

        # Cold-start policy
        allow_unknown_inside: bool = True,
        allow_unknown_outside: bool = True,
        unknown_outside_near_margin: float = 0.08,
        unknown_outside_min_conf: float = 0.25,

        # Anti-duplication thresholds
        spawn_block_iou: float = 0.70,
        spawn_block_center_frac: float = 0.02,
        merge_tracks_iou: float = 0.80,
        merge_tracks_center_frac: float = 0.015,

        verbose: bool = False,

        # Debug visualizer
        enable_debug_vis: bool = False,
        debug_window_name: str = "TBTracker",
        debug_font_scale: float = 0.5,
        debug_thickness: int = 1,

        # Additive search radius knobs
        speed_gate_add_tau_s: float = 0.1,
        speed_gate_add_cap_px: Optional[float] = None,
        missed_gate_scale: float = 0.0,

        # Smoothing / decay knobs
        use_smoothed_speed_for_radius: bool = True,
        expected_step_use_smoothed: bool = True,
        speed_ema_alpha: float = 0.35,
        speed_ema_delta_cap_pxps: Optional[float] = None,
        speed_missed_decay: float = 0.95,
        use_expected_step_cost: bool = True,

        # --- NEW world-speed guards ---
        world_speed_max_kmh: float = 300.0,   # hard cap
        world_speed_guard_ratio: float = 2.0, # acceptance band vs Jacobian bounds
        world_speed_trust_inflate: float = 0.05,  # trust region margin for plane points
        jacobian_cond_max: float = 50.0,      # reject plane-diff if local conditioning is awful
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
        if base_search_radius_px is None:
            self.base_search_radius_px = max(12.0, 0.02 * diag)
        else:
            self.base_search_radius_px = (
                max(8.0, float(base_search_radius_px) * diag)
                if float(base_search_radius_px) < 1.0 else float(base_search_radius_px)
            )

        # thresholds
        self.max_missed = int(max_missed)
        self.max_speed_px = float(max_speed_px)
        self.dir_cos_min = float(dir_cos_min)
        self.iou_gate = float(iou_gate)
        self.iou_bypass_speed_gate = bool(iou_bypass_speed_gate)
        self.second_chance_expand = float(second_chance_expand)
        self.speed_cap_expand = float(speed_cap_expand)

        # bounce / robustness
        self.gate_size_mult = float(gate_size_mult)
        self.bounce_cooldown_frames = int(bounce_cooldown_frames)
        self.bounce_gate_boost = float(bounce_gate_boost)
        self.bounce_speed_cap_boost = float(bounce_speed_cap_boost)
        self.dir_cos_min_bounce = float(dir_cos_min_bounce)
        self.vy_flip_min_pxps = float(vy_flip_min_pxps)
        self.airborne_window = int(airborne_window)
        self.a_min_px2 = float(a_min_px2)

        # suppression
        self.suppress_slow_outside = bool(suppress_slow_outside)
        self.outside_slow_speed_px = float(outside_slow_speed_px)
        self.suppress_static_inside = bool(suppress_static_inside)
        self.inside_static_speed_px = float(inside_static_speed_px)
        self.static_inside_multi_only = bool(static_inside_multi_only)

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

        # cold-start
        self.allow_unknown_inside = bool(allow_unknown_inside)
        self.allow_unknown_outside = bool(allow_unknown_outside)
        self.unknown_outside_near_margin = float(max(0.0, unknown_outside_near_margin))
        self.unknown_outside_min_conf = float(np.clip(unknown_outside_min_conf, 0.0, 1.0))

        # anti-duplication
        self.spawn_block_iou = float(spawn_block_iou)
        self.spawn_block_center_px = float(spawn_block_center_frac) * self.diag
        self.merge_tracks_iou = float(merge_tracks_iou)
        self.merge_tracks_center_px = float(merge_tracks_center_frac) * self.diag

        # state
        self.tracks: Dict[int, TennisBallActiveMultiTracker.Track] = {}
        self._next_tid = 1
        self._prev_det_centers: List[Tuple[float, float]] = []
        self.verbose = bool(verbose)

        # debug-vis
        self.enable_debug_vis = bool(enable_debug_vis)
        self.debug_window_name = str(debug_window_name)
        self.debug_font_scale = float(debug_font_scale)
        self.debug_thickness = int(debug_thickness)
        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            self.H_inv = None
        self._last_debug = None  # (vis_img, dbg)

        # additive radius
        self.speed_gate_add_tau_s = float(speed_gate_add_tau_s)
        self.speed_gate_add_cap_px = (None if speed_gate_add_cap_px is None else float(speed_gate_add_cap_px))
        self.missed_gate_scale = float(missed_gate_scale)

        # smoothing/decay
        self.use_smoothed_speed_for_radius = bool(use_smoothed_speed_for_radius)
        self.expected_step_use_smoothed = bool(expected_step_use_smoothed)
        self.speed_ema_alpha = float(np.clip(speed_ema_alpha, 0.0, 1.0))
        self.speed_ema_delta_cap_pxps = (None if speed_ema_delta_cap_pxps is None else float(speed_ema_delta_cap_pxps))
        self.speed_missed_decay = float(np.clip(speed_missed_decay, 0.0, 1.0))
        self.use_expected_step_cost = bool(use_expected_step_cost)

        # world speed guard cfg
        self.world_speed_max_kmh = float(world_speed_max_kmh)
        self.world_speed_guard_ratio = float(max(1.0, world_speed_guard_ratio))
        self.world_speed_trust_inflate = float(max(0.0, world_speed_trust_inflate))
        self.jacobian_cond_max = float(max(1.0, jacobian_cond_max))

    # ------------------------ public API ------------------------

    def update(
        self,
        yolo_dets: List[Union[Dict[str, Any], Tuple]],
        debug_image: Optional[np.ndarray] = None,
        return_debug: bool = False,
        show_debug: Optional[bool] = None
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Optional[np.ndarray], Dict[str, Any]]]:
        # --- debug accumulators ---
        dbg: Dict[str, Any] = {
            "raw_dets": [], "gated_dets": [], "tracks_snapshot": {}, "candidates": [],
            "assigned_pairs_mutual": [], "assigned_pairs_greedy": [], "assigned_pairs_second": [],
            "unmatched_det_indices": [], "spawned_tids": [], "merged": [], "killed": [], "active_id": None,
        }

        # Parse & project
        dets_all = self._parse_and_project(yolo_dets)
        if self.do_nms and dets_all:
            dets_all = self._nms(dets_all, self.nms_iou)
        for d in dets_all:
            dbg["raw_dets"].append((d.bbox_img, float(d.conf)))

        # Court gating
        if self.strict_court_gate:
            dets = [d for d in dets_all if (d.cx_plane is not None and self._point_in_polygon((d.cx_plane, d.cy_plane), self.poly))]
        else:
            dets = [d for d in dets_all if d.cx_plane is not None]

        raw_map = { (tuple(map(float,bb)), float(c)): i for i,(bb,c) in enumerate(dbg["raw_dets"]) }
        for d in dets:
            key = (tuple(map(float, d.bbox_img)), float(d.conf))
            if key in raw_map:
                dbg["gated_dets"].append(raw_map[key])

        # Annotate inside
        self._annotate_in_court(dets)

        # cooldowns
        for t in self.tracks.values():
            if t.bounce_cooldown > 0:
                t.bounce_cooldown -= 1

        # Association candidates
        T = list(self.tracks.values())
        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()

        # Snapshot
        for t in self.tracks.values():
            if t.last_bbox is None:
                continue
            cx, cy = self._bbox_center_xyxy(t.last_bbox)
            inst_spd = self._track_speed(t)
            if inst_spd is None or not np.isfinite(inst_spd):
                inst_spd = t.last_speed_pxps if np.isfinite(t.last_speed_pxps) else 0.0
            dbg["tracks_snapshot"][t.track_id] = {
                "center": (float(cx), float(cy)),
                "bbox": tuple(map(float, t.last_bbox)),
                "missed": int(t.missed),
                "age": int(t.age_frames),
                "inst_speed_pxps": float(inst_spd),
                "smooth_speed_pxps": float(t.smooth_speed_pxps),
                "last_step_dt_s": float(t.last_step_dt_s),
                "bounce_cooldown": int(t.bounce_cooldown),
                "in_court": bool(t.last_plane is not None and self._point_in_polygon(t.last_plane, self.poly)),
                "inst_speed_mps": float(t.last_speed_mps),
                "smooth_speed_mps": float(t.smooth_speed_mps),
            }

        candidates = []  # (ti, di, dist, implied_speed, cos, rank_cost)
        per_track_gate_r: Dict[int, float] = {}

        for ti, t in enumerate(T):
            if t.last_bbox is None or len(t.img_trace) == 0:
                continue

            (px, py) = t.img_trace[-1]
            v_unit = None
            if len(t.img_trace) >= 2:
                (x0, y0) = t.img_trace[-2]
                vx, vy = px - x0, py - y0
                n = np.hypot(vx, vy)
                v_unit = (vx / n, vy / n) if n > 1e-6 else None

            inst_speed = self._track_speed(t) or 0.0
            speed_for_radius = (t.smooth_speed_pxps if (self.use_smoothed_speed_for_radius and t.smooth_speed_pxps > 0.0)
                                else inst_speed)

            # additive radius
            speed_add = self.speed_gate_add_tau_s * max(0.0, speed_for_radius)
            if self.speed_gate_add_cap_px is not None:
                speed_add = min(speed_add, self.speed_gate_add_cap_px)

            gate_r = self.base_search_radius_px + speed_add
            gate_r *= (1.0 + self.missed_gate_scale * t.missed)
            if t.last_wh is not None:
                gate_r = max(gate_r, self.gate_size_mult * float(np.hypot(*t.last_wh)))
            if t.bounce_cooldown > 0:
                gate_r *= self.bounce_gate_boost
            gate_r = float(np.clip(gate_r, self.base_search_radius_px, 0.25 * self.diag))
            per_track_gate_r[t.track_id] = gate_r

            # speed cap for this pass
            speed_cap = self.max_speed_px * (1.0 + 0.15 * t.missed)
            if t.bounce_cooldown > 0:
                speed_cap *= self.bounce_speed_cap_boost

            dt_eff = max((t.missed + 1) * self.dt, 1e-9)
            speed_for_expected = (t.smooth_speed_pxps if (self.expected_step_use_smoothed and t.smooth_speed_pxps > 0.0)
                                  else inst_speed)
            expected_step = speed_for_expected * dt_eff

            dir_cos_min_eff = self.dir_cos_min_bounce if t.bounce_cooldown > 0 else self.dir_cos_min

            for di, d in enumerate(dets):
                dx, dy = d.cx_img - px, d.cy_img - py
                dist = float(np.hypot(dx, dy))
                implied_speed = dist / dt_eff

                verdict = "pass"
                cos_val = None
                if dist > gate_r:
                    verdict = f"fail:dist>{gate_r:.1f}"
                elif implied_speed > speed_cap:
                    verdict = f"fail:speed>{speed_cap:.0f}"
                else:
                    if v_unit is not None:
                        cos_val = float(v_unit[0] * (dx / (dist + 1e-9)) + v_unit[1] * (dy / (dist + 1e-9)))
                        if cos_val < dir_cos_min_eff:
                            verdict = f"fail:dir<{dir_cos_min_eff:.2f}"

                rank_cost = abs(dist - expected_step)
                if verdict == "pass":
                    candidates.append((ti, di, dist, implied_speed, cos_val, rank_cost))
                    dbg["candidates"].append({
                        "tid": int(T[ti].track_id), "det_index": int(di),
                        "dist": float(dist), "expected_step": float(expected_step),
                        "rank_cost": float(rank_cost), "implied_speed": float(implied_speed),
                        "cos": None if cos_val is None else float(cos_val),
                        "gate_r": float(gate_r), "speed_cap": float(speed_cap),
                        "inst_speed_pxps": float(inst_speed), "smooth_speed_pxps": float(t.smooth_speed_pxps),
                        "speed_for_radius": float(speed_for_radius), "speed_for_expected": float(speed_for_expected),
                        "verdict": "pass",
                    })
                else:
                    dbg["candidates"].append({
                        "tid": int(T[ti].track_id), "det_index": int(di),
                        "dist": float(dist), "expected_step": float(expected_step),
                        "rank_cost": None, "implied_speed": float(implied_speed),
                        "cos": None if cos_val is None else float(cos_val),
                        "gate_r": float(gate_r), "speed_cap": float(speed_cap),
                        "inst_speed_pxps": float(inst_speed), "smooth_speed_pxps": float(t.smooth_speed_pxps),
                        "speed_for_radius": float(speed_for_radius), "speed_for_expected": float(speed_for_expected),
                        "verdict": verdict,
                    })

        # Mutual-nearest
        if candidates:
            by_track: Dict[int, Tuple[int, float]] = {}
            by_det: Dict[int, Tuple[int, float]] = {}
            for (ti, di, dist, _, _, rank_cost) in candidates:
                metric = float(rank_cost) if self.use_expected_step_cost else float(dist)
                if (ti not in by_track) or (metric < by_track[ti][1]): by_track[ti] = (di, metric)
                if (di not in by_det) or (metric < by_det[di][1]):   by_det[di] = (ti, metric)

            mut_pairs = []
            for ti, (di, _) in by_track.items():
                if di in by_det and by_det[di][0] == ti:
                    for c in candidates:
                        if c[0] == ti and c[1] == di: mut_pairs.append(c); break

            mut_pairs.sort(key=lambda x: (x[5] if self.use_expected_step_cost else x[2]))
            for ti, di, dist, implied_speed, cos_val, rank_cost in mut_pairs:
                Tti = T[ti]
                if (Tti.track_id in assigned_tracks) or (di in assigned_dets): continue
                self._ingest(Tti, dets[di], implied_speed)
                assigned_tracks.add(Tti.track_id); assigned_dets.add(di)
                self._maybe_flag_bounce(Tti)
                dbg["assigned_pairs_mutual"].append((int(Tti.track_id), int(di)))

        # Greedy remainder
        rem = [c for c in candidates if (T[c[0]].track_id not in assigned_tracks and c[1] not in assigned_dets)]
        rem.sort(key=lambda x: (x[5] if self.use_expected_step_cost else x[2]))
        for ti, di, dist, implied_speed, cos_val, rank_cost in rem:
            Tti = T[ti]
            if (Tti.track_id in assigned_tracks) or (di in assigned_dets): continue
            self._ingest(Tti, dets[di], implied_speed)
            assigned_tracks.add(Tti.track_id); assigned_dets.add(di)
            self._maybe_flag_bounce(Tti)
            dbg["assigned_pairs_greedy"].append((int(Tti.track_id), int(di)))

        # Second chance (speed-only)
        for ti, t in enumerate(T):
            if t.track_id in assigned_tracks or t.last_bbox is None or len(t.img_trace) == 0: continue
            gate_r = per_track_gate_r.get(t.track_id, self.base_search_radius_px)
            speed_cap = self.max_speed_px * self.speed_cap_expand * (1.0 + 0.15 * t.missed)
            if t.bounce_cooldown > 0: speed_cap *= self.bounce_speed_cap_boost

            dt_eff = max((t.missed + 1) * self.dt, 1e-9)
            inst_speed = self._track_speed(t) or 0.0
            speed_for_expected = (t.smooth_speed_pxps if (self.expected_step_use_smoothed and t.smooth_speed_pxps > 0.0)
                                  else inst_speed)
            expected_step = speed_for_expected * dt_eff

            best = None
            (px, py) = t.img_trace[-1]
            for di, d in enumerate(dets):
                if di in assigned_dets: continue
                dx, dy = d.cx_img - px, d.cy_img - py
                dist = float(np.hypot(dx, dy)); implied_speed = dist / dt_eff
                if dist > gate_r or implied_speed > speed_cap: continue
                rank_cost = abs(dist - expected_step)
                if (best is None) or ((rank_cost if self.use_expected_step_cost else dist) <
                                      (best[2] if self.use_expected_step_cost else best[1])):
                    best = (di, dist, rank_cost, implied_speed)

            if best is not None:
                di, dist, rank_cost, implied_speed = best
                self._ingest(t, dets[di], implied_speed)
                assigned_tracks.add(t.track_id); assigned_dets.add(di)
                self._maybe_flag_bounce(t)
                dbg["assigned_pairs_second"].append((int(t.track_id), int(di)))

        # Age/decay
        for t in T:
            if t.track_id not in assigned_tracks:
                t.missed += 1; t.age_frames += 1
                if self.speed_missed_decay < 1.0:
                    if t.smooth_speed_pxps > 0.0:
                        t.smooth_speed_pxps = float(max(0.0, self.speed_missed_decay * t.smooth_speed_pxps))
                    if t.smooth_speed_mps > 0.0:
                        t.smooth_speed_mps = float(max(0.0, self.speed_missed_decay * t.smooth_speed_mps))

        # Spawn / merge / kill
        unmatched_dets = [dets[i] for i in range(len(dets))
                          if i not in assigned_dets and float(dets[i].conf) >= self.spawn_min_conf]
        before = set(self.tracks.keys())
        self._spawn_from(unmatched_dets)
        dbg["spawned_tids"] = [int(tid) for tid in sorted(set(self.tracks.keys()) - before)]

        merged_pairs: List[Tuple[int, int]] = []
        if len(self.tracks) >= 2:
            merged_pairs = self._merge_overlapping_tracks()
            dbg["merged"].extend([(int(l), int(s)) for (l, s) in merged_pairs])

        dead_ids = [tid for tid, t in self.tracks.items() if t.missed > self.max_missed]
        for tid in dead_ids:
            del self.tracks[tid]
        dbg["killed"].extend(int(tid) for tid in dead_ids)

        active = self._select_active_hysteresis()
        active_id = active.track_id if active else None
        dbg["active_id"] = int(active_id) if active_id is not None else None

        self._prev_det_centers = [(d.cx_img, d.cy_img) for d in dets_all]

        # Output
        out: List[Dict[str, Any]] = []
        for tid, t in self.tracks.items():
            bbox = self._clip_bbox(t.last_bbox) if t.last_bbox is not None else None
            if bbox is None: continue
            cx, cy = self._bbox_center_xyxy(bbox)
            in_court_now = (t.last_plane is not None) and self._point_in_polygon(t.last_plane, self.poly)
            spd_known = (len(t.img_trace) >= 2)
            inst_spd_px = self._track_speed(t)
            if inst_spd_px is None or not np.isfinite(inst_spd_px):
                inst_spd_px = t.last_speed_pxps if np.isfinite(t.last_speed_pxps) else 0.0

            # choose world speed (prefer smoothed)
            mps_value = t.smooth_speed_mps if t.smooth_speed_mps > 0.0 else (t.last_speed_mps if t.last_speed_mps > 0.0 else None)
            kmh = (mps_value * 3.6) if (mps_value is not None) else None
            mph = (mps_value * 2.2369362921) if (mps_value is not None) else None

            out.append({
                "track_id": int(tid),
                "is_active": bool(tid == active_id),
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "image_center": (float(cx), float(cy)),
                "image_speed_pxps": float(inst_spd_px),
                "smooth_speed_pxps": float(t.smooth_speed_pxps),
                "last_step_dt_s": float(t.last_step_dt_s),
                "speed_known": bool(spd_known),
                "in_court": bool(in_court_now),
                "plane_xy": t.last_plane if (t.last_plane is not None) else (None, None),

                # world speeds for your visualizers
                # "speed_mps": float(mps_value) if mps_value is not None else None,
                # "speed_mps_instant": float(t.last_speed_mps) if t.last_speed_mps > 0.0 else None,
                # "speed_kmh": float(kmh) if kmh is not None else None,
                # "speed_mph": float(mph) if mph is not None else None,

                "missed": int(t.missed),
                "age": int(t.age_frames),
                "conf": float(t.last_conf),
            })

        out.sort(key=lambda x: (-int(x["is_active"]), -x["image_speed_pxps"]))

        # Debug render
        vis_img: Optional[np.ndarray] = None
        do_show = self.enable_debug_vis if show_debug is None else bool(show_debug)
        if debug_image is not None:
            try:
                vis_img = self._render_debug(debug_image.copy(), dets_all, dets, dbg)
                if do_show:
                    cv2.imshow(self.debug_window_name, vis_img)
                    cv2.waitKey(1)
            except Exception:
                vis_img = None
        self._last_debug = (vis_img, dbg)

        if return_debug:
            return out, vis_img, dbg
        return out

    def get_active_ball(self,
        yolo_dets: List[Union[Dict[str, Any], Tuple]],
        debug_image: Optional[np.ndarray] = None,
        return_debug: bool = False,
        show_debug: Optional[bool] = None
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Optional[np.ndarray], Dict[str, Any]]]:
        tracks = self.update(yolo_dets, debug_image, return_debug, show_debug)
        active_track = None
        for track in tracks:
            if track["is_active"]:
                active_track = track
        return active_track

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
            if isinstance(obj, dict):
                box = obj.get("bbox")
                conf = float(obj.get("conf", 1.0))
                cls = obj.get("class", None)
            else:
                box = obj[:4]
                conf = float(obj[4])
                cls = obj[5] if len(obj) > 5 else None

            if self.only_class is not None:
                if cls is None or int(cls) != int(self.only_class):
                    continue

            if self.normalized:
                bx = list(box)
                bx = [bx[0] * W, bx[1] * H, bx[2] * W, bx[3] * H]
                box = tuple(bx)

            if self.det_format == "xywh":
                cx, cy, w, h = box
                x1 = cx - w / 2.0; y1 = cy - h / 2.0
                x2 = cx + w / 2.0; y2 = cy + h / 2.0
            else:
                x1, y1, x2, y2 = box

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

    # ---- track lifecycle ----

    def _ingest(self, t: Track, d: Detection, implied_speed: float):
        # effective dt
        frames_spanned = int(t.missed) + 1
        dt_eff = max(frames_spanned * self.dt, 1e-9)
        t.last_step_dt_s = float(dt_eff)

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

        # keep raw bbox + conf
        t.last_bbox = d.bbox_img
        t.last_conf = float(d.conf)

        # instantaneous pixel speed (px/s)
        t.last_speed_pxps = float(np.clip(implied_speed, 0.0, self.max_speed_px * 3.0))

        # --- WORLD SPEED (guarded) ---
        pxps = float(t.last_speed_pxps)

        # Jacobian singular values at previous pixel (for bounds)
        smin, smax = (None, None)
        if len(t.img_trace) >= 2:
            px_prev, py_prev = t.img_trace[-2]
            smin, smax = self._jacobian_svals(self.H, float(px_prev), float(py_prev))
        lo_est = smin * pxps if (smin is not None) else None
        hi_est = smax * pxps if (smax is not None) else None

        # plane-difference candidate (meters/s)
        mps_plane = None
        if (t.last_plane is not None) and (d.cx_plane is not None) and (d.cy_plane is not None):
            du = float(d.cx_plane - t.last_plane[0])
            dv = float(d.cy_plane - t.last_plane[1])
            mps_plane = float(np.hypot(du, dv) / dt_eff)

        # trust test: polygon + Jacobian band + conditioning
        accept_plane = False
        if (mps_plane is not None) and (lo_est is not None) and (hi_est is not None) and np.isfinite(mps_plane):
            # inflated trust polygon
            poly_trust = self._inflate_polygon(self.poly, self.world_speed_trust_inflate) if self.world_speed_trust_inflate > 0 else self.poly
            prev_in = self._point_in_polygon(t.last_plane, poly_trust) if (t.last_plane is not None) else False
            curr_in = self._point_in_polygon((d.cx_plane, d.cy_plane), poly_trust)
            # local conditioning
            kappa_ok = True
            if len(t.img_trace) >= 2:
                px_prev, py_prev = t.img_trace[-2]
                smin2, smax2 = self._jacobian_svals(self.H, float(px_prev), float(py_prev))
                if (smin2 is None) or (smax2 is None) or (smax2 / max(smin2, 1e-12) > self.jacobian_cond_max):
                    kappa_ok = False
            # guard band
            R = self.world_speed_guard_ratio
            lo_guard = max(0.0, lo_est / R)
            hi_guard = hi_est * R
            accept_plane = prev_in and curr_in and kappa_ok and (lo_guard <= mps_plane <= hi_guard)

        # choose world speed
        if accept_plane:
            inst_mps = mps_plane
        elif (lo_est is not None) and (hi_est is not None) and np.isfinite(pxps):
            # bounded fallback: conservative midpoint in [lo, hi]
            inst_mps = float(np.clip(0.5 * (lo_est + hi_est), lo_est, hi_est))
        else:
            inst_mps = 0.0

        # physical clamp
        max_mps = self.world_speed_max_kmh / 3.6
        if not np.isfinite(inst_mps) or inst_mps < 0.0:
            inst_mps = 0.0
        elif inst_mps > max_mps:
            inst_mps = max_mps

        t.last_speed_mps = float(inst_mps)

        # smooth pixel speed
        prev_px = float(t.smooth_speed_pxps)
        if prev_px <= 0.0:
            sm_px = t.last_speed_pxps
        else:
            if self.speed_ema_delta_cap_pxps is not None:
                delta = np.clip(t.last_speed_pxps - prev_px,
                                -self.speed_ema_delta_cap_pxps, self.speed_ema_delta_cap_pxps)
                target_px = prev_px + delta
            else:
                target_px = t.last_speed_pxps
            sm_px = (1.0 - self.speed_ema_alpha) * prev_px + self.speed_ema_alpha * target_px
        t.smooth_speed_pxps = float(np.clip(sm_px, 0.0, self.max_speed_px * 3.0))

        # smooth world speed
        prev_m = float(t.smooth_speed_mps)
        if prev_m <= 0.0:
            sm_m = t.last_speed_mps
        else:
            sm_m = (1.0 - self.speed_ema_alpha) * prev_m + self.speed_ema_alpha * t.last_speed_mps
        t.smooth_speed_mps = float(max(0.0, sm_m))
        # --- end world speed ---

        # update plane AFTER computing speed
        if d.cx_plane is not None and d.cy_plane is not None:
            t.last_plane = (d.cx_plane, d.cy_plane)
            t.plane_trace.append(t.last_plane)

    def _maybe_flag_bounce(self, t: Track) -> None:
        n = len(t.img_trace)
        if n < 3:
            return
        (x2, y2) = t.img_trace[-1]
        (x1, y1) = t.img_trace[-2]
        (x0, y0) = t.img_trace[-3]
        vy1 = (y2 - y1) / max(self.dt, 1e-9)
        vy0 = (y1 - y0) / max(self.dt, 1e-9)
        flip = (vy1 * vy0 < 0.0) and (abs(vy1) >= self.vy_flip_min_pxps) and (abs(vy0) >= self.vy_flip_min_pxps)
        ay = self._parabolic_ay_y(list(t.img_trace), window=self.airborne_window)
        if flip or (ay is not None and abs(ay) >= self.a_min_px2):
            t.bounce_cooldown = max(t.bounce_cooldown, self.bounce_cooldown_frames)

    # ------------------------ spawn & merge ------------------------

    def _spawn_from(self, dets: List["TennisBallActiveMultiTracker.Detection"]) -> None:
        if not dets:
            return

        def min_dist_to_tracks(d):
            best = float("inf")
            for t in self.tracks.values():
                if t.last_bbox is None:
                    continue
                tx, ty = self._bbox_center_xyxy(t.last_bbox)
                best = min(best, float(np.hypot(d.cx_img - tx, d.cy_img - ty)))
            return best

        def min_dist_to_prev(d):
            if not self._prev_det_centers:
                return float("inf")
            best = float("inf")
            for (px, py) in self._prev_det_centers:
                best = min(best, float(np.hypot(d.cx_img - px, d.cy_img - py)))
            return best

        def is_dup_of_any_track(d):
            for t in self.tracks.values():
                if t.last_bbox is None:
                    continue
                iou = self._bbox_iou(t.last_bbox, d.bbox_img) or 0.0
                if iou >= self.spawn_block_iou:
                    return True
                tx, ty = self._bbox_center_xyxy(t.last_bbox)
                if np.hypot(d.cx_img - tx, d.cy_img - ty) <= self.spawn_block_center_px:
                    return True
            return False

        cand: List[Tuple[float, float, "TennisBallActiveMultiTracker.Detection"]] = []
        for d in dets:
            if is_dup_of_any_track(d):
                continue
            dist_tracks = min_dist_to_tracks(d)
            dist_prev   = min_dist_to_prev(d)
            novelty = float(min(dist_tracks, dist_prev))
            cand.append((novelty, float(d.conf), d))

        if not cand:
            return

        cand.sort(key=lambda x: (x[0], x[1]), reverse=True)

        free = max(0, self.max_tracks - len(self.tracks))
        to_spawn: List["TennisBallActiveMultiTracker.Detection"] = []

        for novelty, conf, d in cand:
            if free > 0:
                to_spawn.append(d); free -= 1
            else:
                worst_tid = self._pick_replacement_target()
                if worst_tid is None:
                    continue
                wt = self.tracks[worst_tid]
                if wt.last_bbox is not None:
                    tx, ty = self._bbox_center_xyxy(wt.last_bbox)
                    if np.hypot(d.cx_img - tx, d.cy_img - ty) <= self.spawn_block_center_px:
                        continue
                del self.tracks[worst_tid]
                to_spawn.append(d)

        for d in to_spawn:
            t = self.Track(track_id=self._next_tid); self._next_tid += 1
            t.img_trace.append((d.cx_img, d.cy_img))
            t.last_bbox = d.bbox_img
            t.last_wh = (d.bbox_img[2] - d.bbox_img[0], d.bbox_img[3] - d.bbox_img[1])
            t.last_conf = float(d.conf)
            if d.cx_plane is not None and d.cy_plane is not None:
                t.last_plane = (d.cx_plane, d.cy_plane)
                t.plane_trace.append(t.last_plane)
            t.last_speed_pxps = 0.0
            t.smooth_speed_pxps = 0.0
            t.last_speed_mps = 0.0
            t.smooth_speed_mps = 0.0
            t.last_step_dt_s = float(self.dt)
            self.tracks[t.track_id] = t

    def _merge_overlapping_tracks(self) -> List[Tuple[int, int]]:
        tids = list(self.tracks.keys())
        to_drop: set[int] = set()
        merged_pairs: List[Tuple[int, int]] = []
        for i in range(len(tids)):
            ti = tids[i]
            if ti in to_drop or ti not in self.tracks:
                continue
            ti_track = self.tracks[ti]
            bi = ti_track.last_bbox
            if bi is None:
                continue
            ci = self._bbox_center_xyxy(bi)
            for j in range(i + 1, len(tids)):
                tj = tids[j]
                if tj in to_drop or tj not in self.tracks:
                    continue
                tj_track = self.tracks[tj]
                bj = tj_track.last_bbox
                if bj is None:
                    continue
                iou = self._bbox_iou(bi, bj) or 0.0
                cj = self._bbox_center_xyxy(bj)
                dist = float(np.hypot(ci[0] - cj[0], ci[1] - cj[1]))
                if (iou >= self.merge_tracks_iou) or (dist <= self.merge_tracks_center_px):
                    key_i = (ti_track.missed, -ti_track.age_frames, -ti_track.last_conf, -ti_track.last_speed_pxps)
                    key_j = (tj_track.missed, -tj_track.age_frames, -tj_track.last_conf, -tj_track.last_speed_pxps)
                    survivor, loser = (ti, tj) if key_i < key_j else (tj, ti)
                    if loser not in to_drop:
                        to_drop.add(loser)
                        merged_pairs.append((int(loser), int(survivor)))
        for tid in to_drop:
            if tid in self.tracks:
                del self.tracks[tid]
        return merged_pairs

    def _pick_replacement_target(self) -> Optional[int]:
        if not self.tracks:
            return None
        worst_tid, worst_key = None, None
        for tid, t in self.tracks.items():
            key = (t.missed, -t.last_speed_pxps, -t.age_frames)
            if worst_key is None or key > worst_key:
                worst_key = key; worst_tid = tid
        return worst_tid

    # ---- active selection with hysteresis (unchanged logic) ----

    def _select_active_hysteresis(self) -> Optional[Track]:
        if not self.tracks:
            self._active_id = None
            self._challenger_id = None
            self._challenger_streak = 0
            self._active_hold = 0
            return None

        near_poly = (
            self._inflate_polygon(self.poly, self.unknown_outside_near_margin)
            if self.unknown_outside_near_margin > 0 else None
        )

        inside_count = sum(
            1 for t in self.tracks.values()
            if t.last_plane is not None and self._point_in_polygon(t.last_plane, self.poly)
        )
        apply_static_inside = self.suppress_slow_outside and (
                not self.static_inside_multi_only or inside_count > 1
        )

        def raw_speed(t):
            s = self._track_speed(t)
            if s is None or not np.isfinite(s):
                s = t.last_speed_pxps if np.isfinite(t.last_speed_pxps) else 0.0
            return float(max(s, 0.0))

        def eligible(t, spd):
            in_court = (t.last_plane is not None) and self._point_in_polygon(t.last_plane, self.poly)
            known = (len(t.img_trace) >= 2)
            if known:
                if in_court and apply_static_inside and (spd < self.inside_static_speed_px):
                    return False
                if (not in_court) and self.suppress_slow_outside and (spd < self.outside_slow_speed_px):
                    return False
                return True
            else:
                if in_court and self.allow_unknown_inside:
                    return True
                if (not in_court) and self.allow_unknown_outside and near_poly is not None and t.last_plane is not None:
                    if self._point_in_polygon(t.last_plane, near_poly) and float(t.last_conf) >= self.unknown_outside_min_conf:
                        return True
                return False

        observed = []
        for t in self.tracks.values():
            if t.last_bbox is None or t.missed > 0:
                continue
            spd = raw_speed(t)
            if eligible(t, spd):
                observed.append((t.track_id, spd, (len(t.img_trace) >= 2)))

        if self._active_id is not None:
            inc = self.tracks.get(self._active_id)
            if inc is not None and inc.missed == 0 and inc.last_bbox is not None:
                spd_inc = raw_speed(inc)
                if eligible(inc, spd_inc):
                    return inc

        strong = [c for c in observed if c[2]]
        if strong:
            strong.sort(key=lambda x: x[1], reverse=True)
            self._active_id = strong[0][0]
            self._active_hold = 0
            self._challenger_id = None
            self._challenger_streak = 0
            return self.tracks[self._active_id]

        if self._active_id is not None and self._active_id in self.tracks:
            return self.tracks[self._active_id]

        self._active_hold = 0
        self._challenger_id = None
        self._challenger_streak = 0
        return None

    # ------------------------ helpers ------------------------

    def _track_speed(self, t: Track) -> Optional[float]:
        if len(t.img_trace) < 2:
            return None
        (x1, y1) = t.img_trace[-1]
        (x0, y0) = t.img_trace[-2]
        dist = float(np.hypot(x1 - x0, y1 - y0))
        dt_eff = float(t.last_step_dt_s) if getattr(t, "last_step_dt_s", 0.0) > 0.0 else self.dt
        return float(dist / max(dt_eff, 1e-9))

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
        x1 = max(0.0, min(W - 1.0, x1)); x2 = max(0.0, min(W - 1.0, x2))
        y1 = max(0.0, min(H - 1.0, y1)); y2 = max(0.0, min(H - 1.0, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _parabolic_ay_y(self, img_trace: List[Tuple[float, float]], window: int) -> Optional[float]:
        if len(img_trace) < max(4, window):
            return None
        y = np.array([p[1] for p in img_trace[-window:]], dtype=float)
        n = len(y)
        t = np.arange(n, dtype=float) - (n - 1) * 0.5
        A = np.column_stack([t**2, t, np.ones(n)])
        try:
            coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
            return float(coeffs[0])
        except np.linalg.LinAlgError:
            return None

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

    def _render_debug(self, img: np.ndarray,
                      dets_all: List["TennisBallActiveMultiTracker.Detection"],
                      dets_gated: List["TennisBallActiveMultiTracker.Detection"],
                      dbg: Dict[str, Any]) -> np.ndarray:
        H, W = img.shape[:2]
        if (W, H) != self.image_size:
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
            H, W = img.shape[:2]

        if getattr(self, "H_inv", None) is not None and self.H_inv is not None:
            poly_pts = self._warp_poly_plane2img(self.poly, self.H_inv)
            if len(poly_pts) >= 3:
                cv2.polylines(img, [np.int32(poly_pts)], True, (200, 200, 200), 1, cv2.LINE_AA)

        raw_set = { (tuple(map(float, bb)), float(c)) for (bb, c) in dbg["raw_dets"] }
        gated_keys = { (tuple(map(float, d.bbox_img)), float(d.conf)) for d in dets_gated }
        for (bb, conf) in raw_set:
            x1, y1, x2, y2 = map(int, bb)
            col = (160, 160, 160) if (bb, conf) not in gated_keys else (230, 230, 230)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)

        for tid, t in self.tracks.items():
            color = self._color_for_id(tid)
            if len(t.img_trace) >= 2:
                pts = np.int32([(int(x), int(y)) for (x, y) in t.img_trace])
                cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
            if t.last_bbox is not None:
                x1, y1, x2, y2 = map(int, t.last_bbox)
                thick = 3 if (dbg["active_id"] is not None and tid == dbg["active_id"]) else 2
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

            inst_spd_px = self._track_speed(t)
            if inst_spd_px is None or not np.isfinite(inst_spd_px):
                inst_spd_px = t.last_speed_pxps if np.isfinite(t.last_speed_pxps) else 0.0
            kmh_dbg = t.smooth_speed_mps * 3.6 if t.smooth_speed_mps > 0.0 else (t.last_speed_mps * 3.6 if t.last_speed_mps > 0.0 else 0.0)
            label = f"T{tid} vpx={inst_spd_px:.1f} spx={t.smooth_speed_pxps:.1f} mps={t.last_speed_mps:.2f} sm_mps={t.smooth_speed_mps:.2f} ({kmh_dbg:.1f} km/h) dt={t.last_step_dt_s:.2f}s"
            pos = (int(t.img_trace[-1][0]) + 6, int(t.img_trace[-1][1]) - 6) if len(t.img_trace) else (x1, y1 - 6)
            self._text(img, label, pos, color)

            # visualize gate radius
            speed_for_radius = (t.smooth_speed_pxps if (self.use_smoothed_speed_for_radius and t.smooth_speed_pxps > 0.0)
                                else inst_spd_px)
            speed_add = self.speed_gate_add_tau_s * max(0.0, speed_for_radius)
            if self.speed_gate_add_cap_px is not None:
                speed_add = min(speed_add, self.speed_gate_add_cap_px)
            gate_r = self.base_search_radius_px + speed_add
            gate_r *= (1.0 + self.missed_gate_scale * t.missed)
            if t.last_wh is not None:
                gate_r = max(gate_r, self.gate_size_mult * float(np.hypot(*t.last_wh)))
            if t.bounce_cooldown > 0:
                gate_r *= self.bounce_gate_boost
            gate_r = float(np.clip(gate_r, self.base_search_radius_px, 0.25 * self.diag))
            if len(t.img_trace):
                cx, cy = t.img_trace[-1]
                cv2.circle(img, (int(cx), int(cy)), int(gate_r), color, 1, cv2.LINE_AA)

        # assignment arrows
        for (pair_list, tag) in [
            (dbg["assigned_pairs_mutual"], "mutual"),
            (dbg["assigned_pairs_greedy"], "greedy"),
            (dbg["assigned_pairs_second"], "second"),
        ]:
            for tid, di in pair_list:
                if di < 0 or di >= len(dets_gated):
                    continue
                d = dets_gated[di]; cx, cy = d.cx_img, d.cy_img
                t = self.tracks.get(tid, None)
                if t is None or not len(t.img_trace):
                    continue
                tx, ty = t.img_trace[-1]
                color = self._color_for_id(tid)
                cv2.arrowedLine(img, (int(tx), int(ty)), (int(cx), int(cy)), color, 2, cv2.LINE_AA, tipLength=0.2)
                self._text(img, f"T{tid} {tag}", (int(cx) + 8, int(cy) + 14), color)

        for gi in dbg["unmatched_det_indices"]:
            if gi < 0 or gi >= len(dets_gated):
                continue
            d = dets_gated[gi]; x1, y1, x2, y2 = map(int, d.bbox_img)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 2, cv2.LINE_AA)
            self._text(img, "UNMATCHED", (x1, max(12, y1 - 6)), (0, 140, 255))
        return img

    def _text(self, img: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int] = (255, 255, 255)):
        fs = self.debug_font_scale; th = max(1, self.debug_thickness)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)

    def _color_for_id(self, tid: int) -> Tuple[int, int, int]:
        rng = np.random.RandomState(tid * 9973 + 123)
        return tuple(int(x) for x in rng.randint(64, 256, size=3))

    @staticmethod
    def _warp_poly_plane2img(poly_plane: List[Tuple[float, float]], H_inv: np.ndarray) -> List[Tuple[int, int]]:
        out = []
        for (x, y) in poly_plane:
            v = np.array([x, y, 1.0], dtype=float)
            w = H_inv @ v
            if np.isfinite(w).all() and abs(w[2]) > 1e-8:
                xi, yi = float(w[0] / w[2]), float(w[1] / w[2])
                out.append((int(round(xi)), int(round(yi))))
        return out

    # --- Jacobian helper (meters per pixel singular values at (x,y)) ---
    @staticmethod
    def _jacobian_svals(H: np.ndarray, x: float, y: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Return (smin, smax) singular values of d(u,v)/d(x,y) at pixel (x,y).
        Units: meters per pixel. Use to bound px→meter velocity.
        """
        h11,h12,h13, h21,h22,h23, h31,h32,h33 = H.flatten()
        w  = h31*x + h32*y + h33
        if not np.isfinite(w) or abs(w) < 1e-9:
            return None, None
        u  = (h11*x + h12*y + h13) / w
        v  = (h21*x + h22*y + h23) / w
        J = (1.0 / w) * np.array([
            [h11 - h31*u,  h12 - h32*u],
            [h21 - h31*v,  h22 - h32*v],
        ], dtype=float)
        try:
            svals = np.linalg.svd(J, compute_uv=False)
            smin, smax = float(np.min(svals)), float(np.max(svals))
            return (smin, smax)
        except np.linalg.LinAlgError:
            return None, None
