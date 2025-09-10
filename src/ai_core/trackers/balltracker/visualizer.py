from __future__ import annotations

from collections import deque
from typing import Dict, Deque, Tuple, Union, Sequence, List, Optional

import cv2
import numpy as np


class TennisBallVisualizer:
    def __init__(
        self,
        trail_maxlen: int = 40,            # crisp, short trail
        stale_ttl_frames: int = 60,        # GC tracks not seen for this many frames
        ball_radius_px: int = 5,           # small solid dot
        trail_thickness_px_max: int = 5,   # head thickness; tail tapers to 1
        color: Tuple[int, int, int] = (0, 0, 255),  # BGR — red for BOTH ball and trail

        # --- NEW: speed readout options ---
        draw_speed: bool = True,
        speed_unit: str = "kmh",           # one of: "kmh" (default), "mph", "mps"
        speed_decimals: int = 1,           # rounding for display
        font_scale: float = 0.45,          # text size
        font_thickness: int = 1,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        text_bg: bool = True,
        text_bg_alpha: float = 0.5,        # 0..1 (used if text_bg is True)
        text_pad_px: int = 3,
        text_offset_px: Tuple[int, int] = (8, -10),  # relative to the ball center (x,y)
    ):
        self.trail_maxlen = int(trail_maxlen)
        self.stale_ttl_frames = int(stale_ttl_frames)
        self._trails: Dict[Union[int, str], Deque[Tuple[int, int]]] = {}
        self._last_seen: Dict[Union[int, str], int] = {}
        self._frame_idx: int = 0

        self.ball_radius_px = int(ball_radius_px)
        self.trail_thickness_px_max = int(trail_thickness_px_max)
        self.color = tuple(int(c) for c in color)

        # speed rendering
        self.draw_speed = bool(draw_speed)
        self.speed_unit = speed_unit.lower().strip()
        assert self.speed_unit in {"kmh", "mph", "mps"}, "speed_unit must be 'kmh', 'mph', or 'mps'"
        self.speed_decimals = int(speed_decimals)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = float(font_scale)
        self.font_thickness = int(font_thickness)
        self.text_color = tuple(int(c) for c in text_color)
        self.text_bg = bool(text_bg)
        self.text_bg_alpha = float(text_bg_alpha)
        self.text_pad_px = int(text_pad_px)
        self.text_offset_px = (int(text_offset_px[0]), int(text_offset_px[1]))

    # ---------- public API ----------
    def draw(self, frame: np.ndarray, objs: List[dict]) -> np.ndarray:
        """Draw all balls + tapered line trails (+ optional speed) for this frame (in-place)."""
        self._frame_idx += 1
        self._gc_stale()

        H, W = frame.shape[:2]

        for obj in objs or ():
            if "track_id" not in obj:
                continue
            tid = obj["track_id"]
            age = int(obj.get("age", 0))
            missed = int(obj.get("missed", 0))

            # (Re)initialize trail when a track is new
            self._ensure_track(tid, reset=(age == 0))
            trail = self._trails[tid]
            self._last_seen[tid] = self._frame_idx

            # Determine center
            center = self._get_center(obj, H, W)
            if center is not None and missed == 0:
                trail.append(center)

            # Draw single-color tapered trail
            self._draw_tapered_trail(frame, trail)

            # Draw ball at head (same color)
            if len(trail) > 0:
                head = trail[-1]
                cv2.circle(frame, head, self.ball_radius_px, self.color, -1, cv2.LINE_AA)

                # NEW: speed readout
                if self.draw_speed:
                    label = self._format_speed_label(obj)
                    if label is not None:
                        self._draw_label_near_point(frame, head, label)

        return frame

    # ---------- helpers ----------
    @staticmethod
    def _clip_box_xyxy(box_xyxy: Sequence[float], H: int, W: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return x1, y1, x2, y2

    @staticmethod
    def _is_xyxy(b: Sequence[float]) -> bool:
        return (len(b) == 4) and (b[2] > b[0]) and (b[3] > b[1])

    def _get_center(self, obj: dict, H: int, W: int) -> Optional[Tuple[int, int]]:
        c = obj.get("image_center", None)
        if c is not None and c[0] is not None and c[1] is not None:
            return int(round(float(c[0]))), int(round(float(c[1])))
        b = obj.get("bbox", None)
        if b is None or len(b) != 4:
            return None
        if self._is_xyxy(b):
            x1, y1, x2, y2 = self._clip_box_xyxy(b, H, W)
        else:
            x, y, w, h = [float(v) for v in b]
            x1, y1, x2, y2 = self._clip_box_xyxy([x, y, x + w, y + h], H, W)
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _ensure_track(self, tid: Union[int, str], reset: bool):
        if reset or tid not in self._trails:
            self._trails[tid] = deque(maxlen=self.trail_maxlen)

    def _gc_stale(self):
        if self.stale_ttl_frames <= 0:
            return
        dead = [tid for tid, last in self._last_seen.items()
                if (self._frame_idx - last) > self.stale_ttl_frames]
        for tid in dead:
            self._trails.pop(tid, None)
            self._last_seen.pop(tid, None)

    def _draw_tapered_trail(self, frame: np.ndarray, trail: Deque[Tuple[int, int]]):
        n = len(trail)
        if n < 2:
            return
        # Oldest -> newest so newer segments render on top
        for i in range(1, n):
            p0 = trail[i - 1]; p1 = trail[i]
            t = i / max(1, n)  # older -> thinner
            thickness = max(1, int(round(self.trail_thickness_px_max * t)))
            cv2.line(frame, p0, p1, self.color, thickness, lineType=cv2.LINE_AA)

    # ---------- NEW: speed extraction + label drawing ----------
    def _format_speed_label(self, obj: dict) -> Optional[str]:
        """
        Choose and format a speed string based on self.speed_unit.
        Expected keys in obj: speed_mps, speed_kmh, speed_mph (may be None).
        Falls back to converting from m/s if the chosen unit isn't present.
        """
        mps = obj.get("speed_mps", None)
        kmh = obj.get("speed_kmh", None)
        mph = obj.get("speed_mph", None)

        unit = self.speed_unit
        value = None

        if unit == "kmh":
            value = kmh if kmh is not None else (mps * 3.6 if mps is not None else None)
            suffix = "km/h"
        elif unit == "mph":
            value = mph if mph is not None else (mps * 2.2369362921 if mps is not None else None)
            suffix = "mph"
        else:  # "mps"
            value = mps
            suffix = "m/s"

        if value is None:
            return None

        return f"{value:.{self.speed_decimals}f} {suffix}"

    def _draw_label_near_point(self, frame: np.ndarray, pt: Tuple[int, int], text: str):
        """
        Draws a text label near (x,y) with optional semi-opaque background.
        Keeps the label inside the frame if possible.
        """
        x, y = int(pt[0]), int(pt[1])
        H, W = frame.shape[:2]

        (tw, th), baseline = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)
        pad = self.text_pad_px
        box_w = tw + 2 * pad
        box_h = th + baseline + 2 * pad

        # Preferred top-left corner relative to the ball
        ox, oy = self.text_offset_px
        tl_x = x + ox
        tl_y = y + oy - box_h  # place box above the offset anchor

        # Clamp inside frame
        tl_x = max(0, min(W - box_w, tl_x))
        tl_y = max(0, min(H - box_h, tl_y))

        br_x = tl_x + box_w
        br_y = tl_y + box_h

        if self.text_bg:
            overlay = frame.copy()
            cv2.rectangle(overlay, (tl_x, tl_y), (br_x, br_y), (0, 0, 0), thickness=-1)
            cv2.addWeighted(overlay, self.text_bg_alpha, frame, 1 - self.text_bg_alpha, 0, frame)

        # Put text
        text_org = (tl_x + pad, br_y - pad - baseline)
        cv2.putText(frame, text, text_org, self.font, self.font_scale, self.text_color,
                    self.font_thickness, cv2.LINE_AA)
