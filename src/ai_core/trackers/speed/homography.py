import math
import os
from collections import deque
from typing import List, Dict, Tuple, Optional

import numpy as np


class Pixel2WorldConverter:
    """
    Convert image-space tracks (bbox px) into court-plane world metrics (meters),
    and compute per-track velocity/speed.

    EXPECTED H:
        H must map image pixel homogeneous coords to court-plane meters:
            [x_m, y_m, w]^T  ~  H @ [u, v, 1]^T

    update() returns a list of dicts, preserving existing keys and adding:
        'pos_world_m', 'pos_world_filtered_m', 'vel_world_mps', 'speed_mps', 'speed_kmh', 'speed_mph', ...
    Additionally, each returned dict now contains:
        '_original' : a verbatim copy of the input dict for that track.
    """

    def __init__(self,
                 H: np.ndarray,
                 # --- original knobs
                 ema_alpha: Optional[float] = 0.25,
                 reject_over_kmh: Optional[float] = 300.0,
                 max_position_jump_m: Optional[float] = None,
                 purge_after_frames: int = 60,
                 position_mode: str = "center",   # default if not overridden on update() or per track
                 meta: Optional[Dict] = None,
                 # --- new knobs (all optional)
                 pos_ema_alpha: Optional[float] = 0.35,  # position smoothing before velocity estimation
                 velocity_window: int = 7,               # samples for regression (per track)
                 min_samples_for_velocity: int = 3,
                 reg_weight_decay: float = 0.7,          # 0<gamma<=1. Newer samples weigh more (gamma**age)
                 hampel_window: int = 7,                 # samples for Hampel outlier on step distance
                 hampel_k: float = 3.0,                  # sensitivity of Hampel rejection
                 max_accel_mps2: Optional[float] = None, # clamp change in |v| per second (None disables)
                 min_report_speed_kmh: float = 0.0       # zero-out tiny speeds after smoothing/reporting
                 ) -> None:

        H = np.asarray(H, dtype=float)
        if H.shape != (3, 3):
            raise ValueError("H must be 3x3")
        if abs(np.linalg.det(H)) < 1e-12 or not np.isfinite(H).all():
            raise ValueError("H appears singular/invalid; check your calibration.")
        self.H = H

        # original behavior
        self.ema_alpha = None if ema_alpha is None else float(ema_alpha)
        self.reject_over_mps = None if reject_over_kmh is None else float(reject_over_kmh) / 3.6
        self.max_jump_m = None if max_position_jump_m is None else float(max_position_jump_m)
        self.purge_after = int(purge_after_frames)
        self.position_mode = str(position_mode)

        # per-track state (original)
        self._prev_pos: Dict[int, Tuple[float, float]] = {}
        self._prev_seen_frame: Dict[int, int] = {}
        self._ema_speed: Dict[int, float] = {}
        self._frame_idx: int = 0

        # new per-track state
        self.pos_ema_alpha = None if pos_ema_alpha is None else float(pos_ema_alpha)
        self.velocity_window = int(max(2, velocity_window))
        self.min_samples_for_velocity = int(max(2, min_samples_for_velocity))
        self.reg_weight_decay = float(reg_weight_decay)
        self.hampel_window = int(max(3, hampel_window))
        self.hampel_k = float(hampel_k)
        self.max_accel_mps2 = None if max_accel_mps2 is None else float(max_accel_mps2)
        self.min_report_speed_mps = float(min_report_speed_kmh) / 3.6

        # history buffers: per track -> deque[(t_s, x_m, y_m)] and step distances
        self._hist: Dict[int, deque] = {}
        self._step_hist: Dict[int, deque] = {}
        self._pos_ema: Dict[int, Tuple[float, float]] = {}
        self._prev_vel: Dict[int, Tuple[float, float]] = {}

        # global time accumulator
        self._time_s: float = 0.0

        # optional calibration metadata
        self.meta: Dict = meta or {}

    # ---------- NEW: load from .npz produced by the GUI ----------
    @classmethod
    def from_file(cls,
                  npz_path: str,
                  *,
                  ema_alpha: Optional[float] = 0.25,
                  reject_over_kmh: Optional[float] = 300.0,
                  max_position_jump_m: Optional[float] = None,
                  purge_after_frames: int = 60,
                  position_mode: str = "center",
                  strict: bool = True,
                  **kwargs) -> "HomographyTrackProcessor":
        """
        Load court calibration (.npz) and create a processor.
        Other kwargs map to __init__ (including new knobs).
        """
        if not os.path.exists(npz_path):
            raise FileNotFoundError(npz_path)

        data = np.load(npz_path, allow_pickle=True)
        if "H" not in data:
            raise KeyError("File does not contain 'H'.")

        H = np.asarray(data["H"], dtype=float)
        if H.shape != (3, 3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
            msg = "Invalid H in file (wrong shape, non-finite, or near-singular)."
            if strict:
                raise ValueError(msg)
            H = np.eye(3, dtype=float)

        def itemize(x, default=None):
            if x is None:
                return default
            try:
                return x.item()
            except Exception:
                return x

        meta = {
            "source_file": os.path.abspath(npz_path),
            "video_path": itemize(data.get("video_path")),
            "frame_index": int(itemize(data.get("frame_index"), -1)) if "frame_index" in data else -1,
            "mode": itemize(data.get("mode")),
            "labels": data.get("labels"),
            "img_pts": data.get("img_pts"),
            "world_pts": data.get("world_pts"),
            "singles_width_m": float(itemize(data.get("singles_width_m"), np.nan))
            if "singles_width_m" in data else np.nan,
            "singles_length_m": float(itemize(data.get("singles_length_m"), np.nan))
            if "singles_length_m" in data else np.nan,
        }

        return cls(H,
                   ema_alpha=ema_alpha,
                   reject_over_kmh=reject_over_kmh,
                   max_position_jump_m=max_position_jump_m,
                   purge_after_frames=purge_after_frames,
                   position_mode=position_mode,
                   meta=meta,
                   **kwargs)

    # ---------- existing helpers ----------
    def set_homography(self, H_new: np.ndarray) -> None:
        H_new = np.asarray(H_new, dtype=float)
        if H_new.shape != (3, 3):
            raise ValueError("H must be 3x3")
        if abs(np.linalg.det(H_new)) < 1e-12 or not np.isfinite(H_new).all():
            raise ValueError("H appears singular/invalid.")
        self.H = H_new
        self._ema_speed.clear()
        # keep histories; speeds will re-stabilize within a handful of frames

    def reset(self) -> None:
        self._prev_pos.clear()
        self._prev_seen_frame.clear()
        self._ema_speed.clear()
        self._hist.clear()
        self._step_hist.clear()
        self._pos_ema.clear()
        self._prev_vel.clear()
        self._frame_idx = 0
        self._time_s = 0.0

    def update(self,
               tracks: List[Dict],
               dt_sec: float,
               position_mode: Optional[str] = None) -> List[Dict]:
        """
        Update with a batch of tracks.

        position_mode (optional): "center" | "bottom_center"
          Precedence per track:
            t.get("position_mode")  >  position_mode (arg)  >  self.position_mode
        Returns per-track dicts that:
          - contain all original keys,
          - add world/velocity fields,
          - and include a verbatim copy under '_original'.
        """
        # advance global time
        dt = float(dt_sec)
        valid_dt = (dt > 0.0 and math.isfinite(dt))
        if valid_dt:
            self._time_s += dt

        self._frame_idx += 1
        out: List[Dict] = []

        # purge stale tracks
        if self.purge_after > 0:
            to_del = [tid for tid, f in self._prev_seen_frame.items()
                      if (self._frame_idx - f) > self.purge_after]
            for tid in to_del:
                self._prev_pos.pop(tid, None)
                self._prev_seen_frame.pop(tid, None)
                self._ema_speed.pop(tid, None)
                self._hist.pop(tid, None)
                self._step_hist.pop(tid, None)
                self._pos_ema.pop(tid, None)
                self._prev_vel.pop(tid, None)

        for t in tracks:
            tid = int(t["track_id"])
            x1, y1, x2, y2 = map(float, t["bbox"])

            # resolve position mode for this track
            mode_eff = (t.get("position_mode")
                        or position_mode
                        or self.position_mode
                        or "center")
            if mode_eff not in ("center", "bottom_center"):
                mode_eff = "center"

            cx, cy = self._bbox_point(x1, y1, x2, y2, mode_eff)

            raw_pos = self._img_to_world(cx, cy)
            x_m, y_m = (None, None) if raw_pos is None else raw_pos

            vx = vy = None
            speed = None
            outlier_rejected = False
            samples_used = 0
            pos_filtered = None

            if raw_pos is not None:
                # initialize per-track buffers
                if tid not in self._hist:
                    self._hist[tid] = deque(maxlen=max(self.velocity_window * 3, 32))
                if tid not in self._step_hist:
                    self._step_hist[tid] = deque(maxlen=max(self.hampel_window * 2, 24))

                # --- adaptive outlier rejection (Hampel on step distances) ---
                prev = self._prev_pos.get(tid)
                if prev is not None:
                    step_dist = math.hypot(x_m - prev[0], y_m - prev[1])
                    self._step_hist[tid].append(step_dist)
                    # hard jump gate (if configured)
                    hard_jump = (self.max_jump_m is not None and step_dist > self.max_jump_m)
                    # adaptive gate
                    adaptive_jump = False
                    if len(self._step_hist[tid]) >= self.hampel_window:
                        arr = np.asarray(self._step_hist[tid], dtype=float)
                        med = np.median(arr)
                        mad = np.median(np.abs(arr - med))
                        sigma = 1.4826 * mad + 1e-9
                        adaptive_jump = (step_dist > med + self.hampel_k * sigma)
                    if hard_jump or adaptive_jump:
                        outlier_rejected = True
                    else:
                        self._prev_pos[tid] = (x_m, y_m)
                else:
                    # first sample for this track
                    self._prev_pos[tid] = (x_m, y_m)

                # update "seen" regardless (so we don't purge active tracks)
                self._prev_seen_frame[tid] = self._frame_idx

                # position EMA (for velocity estimation robustness)
                if self.pos_ema_alpha is not None:
                    pprev = self._pos_ema.get(tid)
                    if pprev is None:
                        pos_filtered = (x_m, y_m)
                    else:
                        a = self.pos_ema_alpha
                        pos_filtered = ((1.0 - a) * pprev[0] + a * x_m,
                                        (1.0 - a) * pprev[1] + a * y_m)
                    self._pos_ema[tid] = pos_filtered
                else:
                    pos_filtered = (x_m, y_m)

                # append to history only if dt is valid AND not outlier
                if valid_dt and not outlier_rejected:
                    self._hist[tid].append((self._time_s, pos_filtered[0], pos_filtered[1]))

                # --- velocity via weighted regression over a short window ---
                vx_est = vy_est = None
                hist = self._hist.get(tid, None)
                if hist and len(hist) >= self.min_samples_for_velocity:
                    # take the most recent K samples
                    K = min(self.velocity_window, len(hist))
                    seg = list(hist)[-K:]
                    t_arr = np.array([s[0] for s in seg], dtype=float)
                    x_arr = np.array([s[1] for s in seg], dtype=float)
                    y_arr = np.array([s[2] for s in seg], dtype=float)
                    # if time span ~0, skip (bad dt)
                    tspan = float(t_arr[-1] - t_arr[0])
                    if tspan > 1e-6 and np.isfinite(t_arr).all():
                        # exponential weights: newer samples get higher weight
                        # age: 0 for newest, up to K-1 for oldest
                        ages = np.arange(K - 1, -1, -1, dtype=float)
                        w = np.power(self.reg_weight_decay, ages)
                        # center time to reduce collinearity
                        t0 = t_arr.mean()
                        t_cent = t_arr - t0
                        try:
                            vx_est = np.polyfit(t_cent, x_arr, deg=1, w=w)[0]
                            vy_est = np.polyfit(t_cent, y_arr, deg=1, w=w)[0]
                            samples_used = K
                        except np.linalg.LinAlgError:
                            vx_est = vy_est = None

                # acceleration clamp (optional)
                if vx_est is not None and vy_est is not None and self.max_accel_mps2 is not None and valid_dt:
                    pv = self._prev_vel.get(tid)
                    if pv is not None:
                        dvx = vx_est - pv[0]
                        dvy = vy_est - pv[1]
                        dv = math.hypot(dvx, dvy)
                        max_dv = self.max_accel_mps2 * dt  # per frame budget
                        if dv > max_dv and max_dv > 0.0:
                            scale = max_dv / dv
                            vx_est = pv[0] + dvx * scale
                            vy_est = pv[1] + dvy * scale

                # finalize velocity
                if vx_est is not None and vy_est is not None:
                    vx, vy = float(vx_est), float(vy_est)
                    speed = math.hypot(vx, vy)
                    if self.reject_over_mps is not None and speed > self.reject_over_mps:
                        vx = vy = speed = None
                    else:
                        self._prev_vel[tid] = (vx, vy)

            # --- speed smoothing (existing EMA on top of the better estimator) ---
            speed_smoothed = speed
            if self.ema_alpha is not None:
                ema_prev = self._ema_speed.get(tid)
                if speed is not None:
                    ema_now = (1.0 - self.ema_alpha) * (0.0 if ema_prev is None else ema_prev) + self.ema_alpha * speed
                    self._ema_speed[tid] = ema_now
                    speed_smoothed = ema_now

            # zero-out tiny speeds if requested
            if speed_smoothed is not None and speed_smoothed < self.min_report_speed_mps:
                speed_smoothed = 0.0

            # assemble row (preserve original keys and embed a verbatim copy)
            row = dict(t)
            row["_original"] = dict(t)  # <-- added: keep the exact inbound dict alongside enriched fields
            row["pos_world_m"] = (x_m, y_m)
            row["pos_world_filtered_m"] = pos_filtered if pos_filtered is not None else (x_m, y_m)
            row["vel_world_mps"] = (vx, vy) if (vx is not None and vy is not None) else None
            row["speed_mps"] = speed_smoothed if speed_smoothed is not None else speed
            if row["speed_mps"] is None:
                row["speed_kmh"] = None
                row["speed_mph"] = None
            else:
                row["speed_kmh"] = row["speed_mps"] * 3.6
                row["speed_mph"] = row["speed_mps"] * 2.2369362920544

            # diagnostics (new fields)
            row["velocity_src"] = "regression"
            row["samples_used"] = samples_used
            row["outlier_rejected"] = bool(outlier_rejected)
            row["position_mode_eff"] = mode_eff  # helpful when mixing balls/players

            out.append(row)

        return out

    # ----------------- helpers -----------------
    def _bbox_point(self, x1: float, y1: float, x2: float, y2: float, mode: str) -> Tuple[float, float]:
        if mode == "bottom_center":
            return (0.5 * (x1 + x2), y2)
        # default: "center"
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def _img_to_world(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        vec = self.H @ np.array([u, v, 1.0], dtype=float)
        w = vec[2]
        if abs(w) < 1e-9 or not np.isfinite(w):
            return None
        x = vec[0] / w
        y = vec[1] / w
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (float(x), float(y))
