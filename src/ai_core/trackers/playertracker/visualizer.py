from collections import deque
from typing import Dict, List, Tuple, Union, Sequence, Optional
import numpy as np
import cv2


class PlayerVisualizer:
    def __init__(
        self,
        trail_maxlen: int = 40,
        stale_ttl_frames: int = 60,
        seed: int = 1337,
        team_colors: Optional[Dict[Union[int, str], Tuple[int, int, int]]] = None,  # BGR per team label

        # --- NEW: speed readout options ---
        draw_speed: bool = True,
        speed_unit: str = "kmh",        # one of: "kmh" (default), "mph", "mps"
        speed_decimals: int = 1,
        speed_text_color: Tuple[int, int, int] = (255, 255, 255),
        speed_bg_color: Tuple[int, int, int] = (0, 0, 0),
        speed_bg_alpha: float = 0.5,    # 0..1
        speed_text_pad_px: int = 3,
        speed_font_scale: float = 0.5,
        speed_font_thickness: int = 1,
    ):
        self.trail_maxlen = trail_maxlen
        self.stale_ttl_frames = stale_ttl_frames
        self._trails: Dict[Union[int, str], deque] = {}
        self._last_seen: Dict[Union[int, str], int] = {}
        self._frame_idx: int = 0
        self._seed = seed
        self._team_colors = dict(team_colors) if team_colors else {}

        # speed render config
        self.draw_speed = bool(draw_speed)
        self.speed_unit = str(speed_unit).lower().strip()
        assert self.speed_unit in {"kmh", "mph", "mps"}, "speed_unit must be 'kmh', 'mph', or 'mps'"
        self.speed_decimals = int(speed_decimals)
        self.speed_text_color = tuple(int(c) for c in speed_text_color)
        self.speed_bg_color = tuple(int(c) for c in speed_bg_color)
        self.speed_bg_alpha = float(speed_bg_alpha)
        self.speed_text_pad_px = int(speed_text_pad_px)
        self.speed_font_scale = float(speed_font_scale)
        self.speed_font_thickness = int(speed_font_thickness)
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def _id_color(self, track_id: Union[int, str]) -> Tuple[int, int, int]:
        h = abs(hash((self._seed, str(track_id)))) % (2 ** 32)
        rng = np.random.default_rng(h)
        b, g, r = rng.integers(64, 256, size=3).tolist()
        return int(b), int(g), int(r)

    @staticmethod
    def _clip_box(box_xyxy: Sequence[float], H: int, W: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return x1, y1, x2, y2

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

    # ------------------------ Player visualizer ------------------------

    def draw_players(
        self,
        frame: np.ndarray,
        players: List[dict],
        *,
        show_track_id: bool = False,
        show_trail: bool = False,
    ) -> np.ndarray:
        """
        Renders players with classic orange box and small pill.
        Expects per player:
          - "bbox": [x1,y1,x2,y2]   (required)
          - "label": str            (required)
          - "track_id": int|str     (optional; appended if show_track_id=True)
          - "age": int              (optional; for trail reset)
          - NEW speed fields (optional): "speed_mps", "speed_kmh", "speed_mph"
        """
        self._frame_idx += 1
        H, W = frame.shape[:2]
        ORANGE = (0, 165, 255)
        FONT = self._font
        FONT_SCALE = 0.5
        THICKNESS = 1

        def _role_short(label: Optional[str]) -> str:
            if not label:
                return "P?"
            s = str(label).lower().strip()
            if s.startswith("player1") or s == "p1" or s == "1":
                return "P1"
            if s.startswith("player2") or s == "p2" or s == "2":
                return "P2"
            return label

        for obj in players:
            bbox = obj.get("bbox", None)
            label = obj.get("label", None)
            if bbox is None or label is None:
                continue

            tid = obj.get("track_id", None)
            key = tid if tid is not None else f"__label__:{label}"
            is_new = bool(obj.get("age", 0) == 0) if "age" in obj else (key not in self._trails)
            self._ensure_track(key, reset=is_new)
            self._last_seen[key] = self._frame_idx
            trail = self._trails[key]

            x1, y1, x2, y2 = self._clip_box(bbox, H, W)

            # 1) Orange bounding box (classic)
            cv2.rectangle(frame, (x1, y1), (x2, y2), ORANGE, THICKNESS)

            # 2) Small label pill: "P1" / "P2" [+ optional id]
            text = _role_short(label)
            if show_track_id and (tid is not None):
                text = f"{text} | {tid}"
            (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
            pad = 3
            y_top = max(0, y1 - th - 6)
            cv2.rectangle(frame, (x1, y_top), (x1 + tw + 2 * pad, y1), ORANGE, -1)
            cv2.putText(frame, text, (x1 + pad, y1 - 4), FONT, FONT_SCALE, (0, 0, 0), THICKNESS, cv2.LINE_AA)

            # 3) NEW: speed readout (draw exactly one unit, default km/h)
            if self.draw_speed:
                speed_label = self._format_speed_label(obj)
                if speed_label is not None:
                    self._draw_speed_label(frame, (x1, y1, x2, y2), speed_label)

            # 4) Optional trail (per-id color)
            if show_trail:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trail.append((cx, cy))
                color_id = self._id_color(key)
                for i in range(1, len(trail)):
                    p0, p1 = trail[i - 1], trail[i]
                    if p0 is None or p1 is None:
                        continue
                    alpha = i / max(1, len(trail))
                    rad = max(1, int(6 * alpha))
                    cv2.line(frame, p0, p1, color_id, rad)
                if len(trail) > 0 and trail[-1] is not None:
                    cv2.circle(frame, trail[-1], 3, (0, 0, 255), -1)

        self._gc_stale()
        return frame

    # ---------- Legacy wrapper kept for convenience ----------
    def draw_tracked_boxes_from_tracker(
        self,
        frame: np.ndarray,
        tracker_outputs: List[dict],
        show_ids: bool = True,
    ) -> np.ndarray:
        """
        Accepts tracker outputs and renders classic player boxes.
        Expected fields in each dict:
          - "bbox" (required)
          - "label" (recommended: "player1"/"player2" or similar)
          - "track_id" (optional)
          - "age" (optional)
          - Optional: speed_mps / speed_kmh / speed_mph
        """
        players = []
        for obj in tracker_outputs:
            if "bbox" not in obj:
                continue
            players.append({
                "bbox": obj["bbox"],
                "label": obj.get("label", "Player"),
                "track_id": obj.get("track_id", None) if show_ids else None,
                "age": obj.get("age", 0),
                "speed_mps": obj.get("speed_mps", None),
                "speed_kmh": obj.get("speed_kmh", None),
                "speed_mph": obj.get("speed_mph", None),
            })
        return self.draw_players(frame, players, show_track_id=show_ids, show_trail=True)

    # ------------------------ NEW: helpers ------------------------

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

    def _draw_speed_label(self, frame: np.ndarray, box_xyxy: Tuple[int, int, int, int], text: str):
        """
        Draw a semi-opaque label near the box. Prefers inside top-left of the box;
        if it doesn't fit there, it clamps within the frame.
        """
        x1, y1, x2, y2 = box_xyxy
        H, W = frame.shape[:2]

        (tw, th), baseline = cv2.getTextSize(
            text, self._font, self.speed_font_scale, self.speed_font_thickness
        )
        pad = self.speed_text_pad_px
        box_w = tw + 2 * pad
        box_h = th + baseline + 2 * pad

        # Preferred position: inside the box, slightly below the top edge
        tl_x = x1 + 2
        tl_y = y1 + 2

        # If that would overflow bottom/right, clamp inside frame
        tl_x = max(0, min(W - box_w, tl_x))
        tl_y = max(0, min(H - box_h, tl_y))

        br_x = tl_x + box_w
        br_y = tl_y + box_h

        # Semi-opaque background via overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (tl_x, tl_y), (br_x, br_y), self.speed_bg_color, thickness=-1)
        cv2.addWeighted(overlay, self.speed_bg_alpha, frame, 1 - self.speed_bg_alpha, 0, frame)

        # Text
        text_org = (tl_x + pad, br_y - pad - baseline)
        cv2.putText(
            frame, text, text_org, self._font,
            self.speed_font_scale, self.speed_text_color,
            self.speed_font_thickness, cv2.LINE_AA
        )
