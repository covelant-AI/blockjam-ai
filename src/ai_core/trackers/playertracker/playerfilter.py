# player_filter_h.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

# Utilities you exposed (keep this path consistent with your project)
from .homography import (
    estimate_homography_from_polygon,  # -> H_img2court, H_court2img, spec, ...
    warp_img_to_court,                 # Nx2 image -> Nx2 court (m)
    in_court_uv,                       # predicate in court coords (unused here but kept for parity)
    bbox_bottom_center,                # (xywh|tlwh) -> bottom-center px
    near_net_uv,                       # net-band predicate in court coords (unused here but kept for parity)
)

# ----------------------------- internal state -----------------------------

@dataclass
class _TrackState:
    last_uv: Optional[Tuple[float, float]] = None
    ema_speed_m: float = 0.0
    frames_seen: int = 0
    frames_in_court: int = 0
    frames_near_net: int = 0
    last_seen_frame: int = -1
    last_significant_move_frame: int = -10**9
    # Fixes/additions:
    net_camp_ema: float = 0.0
    near_net_stationary_frames: int = 0


# --------------------------- public filter class --------------------------

@dataclass
class ActivePlayerFilter:
    """
    Homography-based active player selector for tennis.

    Key clarifications/fixes vs your original:
    - Proper 'recency' computed from the *previous* seen frame, not the current one.
    - 'Net camping' uses an EMA instead of a lifetime ratio (avoids long-term stickiness).
    - Optional consecutive-stationary near-net hard drop (more robust vs tiny jitter).
    - Lowered default motion threshold; added explicit 'stationary epsilon'.
    - Lightweight state GC to prevent unbounded growth.
    """
    # IO / formats
    bbox_format: str = "xywh"  # 'xywh' (cx,cy,w,h) or 'tlwh' (x,y,w,h)

    # Geometry (populated after init_from_polygon)
    H_img2court: Optional[np.ndarray] = None
    H_court2img: Optional[np.ndarray] = None
    spec: Optional[dict] = None  # {'width','length',...}

    # Near-net band (meters)
    net_band_m: float = 1.2
    # Off-court handling
    near_net_lateral_margin_m: float = 1.0   # allow ± this outside sidelines
    count_off_court_near_net_for_camp: bool = True
    hard_drop_uses_off_court_near_net: bool = True

    # Scoring weights
    w_in_court: float = 2.5
    w_speed: float = 1.2
    w_recency: float = 0.8
    w_net_camp: float = 1.0

    # Motion gates (meters / frame, ~30 fps)
    speed_beta: float = 0.8
    motion_thresh_m_per_frame: float = 0.08        # was 0.25 (≈7.5 m/s) – now realistic
    stationary_speed_eps_m_per_frame: float = 0.02 # <= this counts as stationary near net
    use_consecutive_stationary_drop: bool = True   # use consecutive frames near net for hard drop
    require_recent_motion: bool = True
    recent_motion_window: int = 150
    near_net_static_drop: int = 60                 # frames stationary near net → drop

    # Speed normalization scale (as a fraction of court length per frame)
    speed_norm_frac_of_L: float = 0.05

    # Selection / hygiene
    enforce_half_split: bool = True
    keep_hysteresis: int = 12
    max_absent: int = 45

    # Off-court distance penalty (meters)
    offcourt_soft_margin_m: float = 0.75   # tolerance around the court before penalizing
    offcourt_hard_drop_m: float = 6.0      # if farther than this from the court -> hard drop
    w_offcourt_dist: float = 0.9           # score penalty per meter outside the soft margin

    # Half-split policy
    strict_half_split: bool = True         # if True, never backfill 2nd from the same half


    # State
    _state: Dict[int, _TrackState] = field(default_factory=dict)
    _last_active: List[int] = field(default_factory=list)

    # ---- setup ----
    def init_from_polygon(self, court_polygon: np.ndarray, *, use_doubles_polygon: bool = True) -> None:
        H_img2court, H_court2img, spec, _, _ = estimate_homography_from_polygon(
            court_polygon, use_doubles_polygon=use_doubles_polygon
        )
        self.H_img2court = H_img2court
        self.H_court2img = H_court2img
        self.spec = spec

    # ---- helpers ----
    def _to_uv(self, xy: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if self.H_img2court is None:
            return None
        uv = warp_img_to_court(self.H_img2court, np.array([xy], dtype=np.float32))[0]
        u, v = float(uv[0]), float(uv[1])
        if not np.isfinite(u) or not np.isfinite(v):
            return None
        return u, v

    def _dist_outside_court(self, uv: Tuple[float, float]) -> float:
        """
        Euclidean distance (meters) from point uv to the court rectangle,
        after expanding the rectangle by offcourt_soft_margin_m on all sides.
        Returns 0 if inside the expanded rectangle.
        """
        assert self.spec is not None
        W, L = float(self.spec["width"]), float(self.spec["length"])
        u, v = float(uv[0]), float(uv[1])

        pad = float(self.offcourt_soft_margin_m)
        u0, u1 = -pad, W + pad
        v0, v1 = -pad, L + pad

        dx = (u0 - u) if u < u0 else ((u - u1) if u > u1 else 0.0)
        dy = (v0 - v) if v < v0 else ((v - v1) if v > v1 else 0.0)
        return float(np.hypot(dx, dy))

    def _near_net_flags(self, uv: Tuple[float, float]) -> Tuple[bool, bool, bool]:
        """
        Returns (in_c, near_net_in, near_net_any).

        near_net_in  = inside court & within band.
        near_net_any = within band and within a small lateral margin outside sidelines
                       (v must still be within [0, L]).
        """
        assert self.spec is not None
        W, L = float(self.spec["width"]), float(self.spec["length"])
        u, v = float(uv[0]), float(uv[1])

        in_c = (0.0 <= u <= W) and (0.0 <= v <= L)
        in_band_v = (abs(v - 0.5 * L) <= self.net_band_m) and (0.0 <= v <= L)
        lateral_ok = (-self.near_net_lateral_margin_m <= u <= W + self.near_net_lateral_margin_m)

        near_net_in = in_c and in_band_v
        near_net_any = in_band_v and lateral_ok
        return in_c, near_net_in, near_net_any

    def _update_track(
        self,
        tid: int,
        uv: Optional[Tuple[float, float]],
        frame_id: int
    ) -> Tuple[bool, bool, bool, float, int]:
        """
        Update EMA speed, counters, and near-net stationary streak.
        Returns (in_c, near_net_in, near_net_any, inst_speed_m, prev_seen_frame).
        """
        st = self._state.get(tid, _TrackState())
        prev_seen = st.last_seen_frame

        in_c = near_in = near_any = False
        inst_speed = 0.0

        if uv is not None and self.spec is not None:
            in_c, near_in, near_any = self._near_net_flags(uv)

            if st.last_uv is not None:
                du = uv[0] - st.last_uv[0]
                dv = uv[1] - st.last_uv[1]
                inst_speed = float(np.hypot(du, dv))

        # EMA speed
        st.ema_speed_m = self.speed_beta * st.ema_speed_m + (1 - self.speed_beta) * inst_speed

        st.last_uv = uv
        st.frames_seen += 1
        if in_c:
            st.frames_in_court += 1

        # Decide which near-net condition increments the camping counter
        near_flag = 0.0
        if near_in or (self.count_off_court_near_net_for_camp and near_any):
            st.frames_near_net += 1
            near_flag = 1.0

        # EMA net-camp (short-memory vs lifetime ratio)
        st.net_camp_ema = 0.9 * st.net_camp_ema + 0.1 * near_flag

        # Consecutive stationary near-net frames (for robust hard drop)
        if (near_any if self.hard_drop_uses_off_court_near_net else near_in) and \
           (inst_speed <= self.stationary_speed_eps_m_per_frame):
            st.near_net_stationary_frames += 1
        else:
            st.near_net_stationary_frames = 0

        st.last_seen_frame = frame_id
        if inst_speed >= self.motion_thresh_m_per_frame:
            st.last_significant_move_frame = frame_id

        self._state[tid] = st
        return in_c, near_in, near_any, inst_speed, prev_seen

    def _score(self, tid: int, uv: Optional[Tuple[float, float]], frame_id: int) -> float:
        # No geometry/uv? nuke score so these never win (and don't refresh recency).
        if uv is None or self.spec is None:
            _ = self._state.get(tid, _TrackState())
            return -1e6

        in_c, near_in, near_any, _inst_speed, prev_seen = self._update_track(tid, uv, frame_id)
        st = self._state[tid]
        # --- Off-court distance penalty / hard drop ---
        d_out = self._dist_outside_court(uv)  # meters outside expanded court
        if d_out >= self.offcourt_hard_drop_m:
            return -1e6  # too far to be a player on this court

        # Motion gating / hard drops
        if self.require_recent_motion:
            idle = frame_id - st.last_significant_move_frame

            # Hard drop: stationary near net (two policies)
            if self.use_consecutive_stationary_drop:
                if (near_any if self.hard_drop_uses_off_court_near_net else near_in) and \
                   st.near_net_stationary_frames >= self.near_net_static_drop:
                    return -1e6
            else:
                if (near_any if self.hard_drop_uses_off_court_near_net else near_in) and \
                   idle > self.near_net_static_drop:
                    return -1e6

            # Generic "no recent motion" penalty
            if idle > self.recent_motion_window:
                return -1e4 if in_c else -1e6

        # Proper recency: compare to *previous* seen frame
        if prev_seen < 0:
            recency = 1.0
        else:
            gap = frame_id - prev_seen
            recency = 1.0 if gap <= 1 else float(np.exp(-gap / 30.0))

        # Features
        L = float(self.spec["length"])
        denom = max(self.speed_norm_frac_of_L * L, 1e-6)  # normalized speed in m/frame
        speed_norm = min(1.0, st.ema_speed_m / denom)

        score = (
                self.w_in_court * (1.0 if in_c else 0.0) +
                self.w_speed * speed_norm +
                self.w_recency * recency -
                self.w_net_camp * st.net_camp_ema
                - self.w_offcourt_dist * d_out  # << add this line
        )

        # Hysteresis bump if it was previously selected and not long absent
        if tid in self._last_active and (frame_id - st.last_seen_frame) <= self.keep_hysteresis:
            score += 0.5

        return float(score)

    def _gc_state(self, current_ids: set, frame_id: int) -> None:
        """Trim stale track states to prevent unbounded growth."""
        if not self._state:
            return
        horizon = 5 * self.max_absent
        to_del = []
        for tid, st in list(self._state.items()):
            if tid in current_ids or tid in self._last_active:
                continue
            if st.last_seen_frame >= 0 and (frame_id - st.last_seen_frame) > horizon:
                to_del.append(tid)
        for tid in to_del:
            self._state.pop(tid, None)

    # ---- public API ----
    def select_active_ids(
        self,
        tracks: Sequence,   # objects with .track_id, .xywh (or .tlwh), optional .is_activated
        frame_id: int,
        court_polygon: Optional[np.ndarray] = None,
        use_doubles_polygon: bool = True
    ) -> List[int]:
        # Initialize H lazily if needed
        if self.H_img2court is None:
            if court_polygon is None:
                return []  # refuse to guess without geometry
            self.init_from_polygon(court_polygon, use_doubles_polygon=use_doubles_polygon)

        persons = [t for t in tracks if getattr(t, "is_activated", True)]
        if not persons:
            self._last_active = []
            return []

        scored: List[Tuple[float, int, object, Optional[Tuple[float, float]]]] = []
        for t in persons:
            tid = int(getattr(t, "track_id"))
            bx, by = bbox_bottom_center(getattr(t, "xywh"), fmt=self.bbox_format)
            uv = self._to_uv((bx, by))
            s = self._score(tid, uv, frame_id)
            scored.append((s, tid, t, uv))
        scored.sort(key=lambda z: z[0], reverse=True)

        # Half-split by v (meters) if we have geometry
        chosen: List[int] = []
        if self.enforce_half_split and self.spec is not None:
            L = float(self.spec["length"])
            near = [item for item in scored if (item[3] is not None and item[3][1] <= 0.5 * L)]
            far = [item for item in scored if (item[3] is not None and item[3][1] > 0.5 * L)]

            # Always take the best from each half if available
            if near:
                chosen.append(near[0][1])
            if far:
                chosen.append(far[0][1])

            if not self.strict_half_split:
                # Backfill from the global list if one half is missing
                if len(chosen) < 2:
                    for s, tid, *_ in scored:
                        if tid not in chosen:
                            chosen.append(tid)
                        if len(chosen) == 2:
                            break
            # else: strict → do NOT backfill; may return only 0–1 IDs here
        else:
            chosen = [tid for s, tid, *_ in scored[:2]]

        # Absence pruning
        pruned: List[int] = []
        for tid in chosen:
            st = self._state.get(tid)
            if not st:
                continue
            if frame_id - st.last_seen_frame > self.max_absent:
                continue
            pruned.append(tid)

        self._last_active = pruned[:2]

        # State GC
        current_ids = {int(getattr(t, "track_id")) for t in persons}
        self._gc_state(current_ids, frame_id)

        return self._last_active
