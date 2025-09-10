from __future__ import annotations

from collections import deque
from typing import Dict, Sequence, Tuple, Union, List

import cv2
import numpy as np


class TrackedBoxDrawer:
    def __init__(
            self,
            trail_maxlen: int = 40,
            stale_ttl_frames: int = 60,
            seed: int = 1337,
    ):
        self.trail_maxlen = trail_maxlen
        self.stale_ttl_frames = stale_ttl_frames
        self._trails: Dict[Union[int, str], deque] = {}
        self._last_seen: Dict[Union[int, str], int] = {}
        self._frame_idx: int = 0
        self._seed = seed

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
        # force proper ordering if tracker ever emits swapped corners
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

    def draw_tracked_boxes_from_tracker(
            self,
            frame: np.ndarray,
            tracker_outputs: List[dict],
            show_ids: bool = True,
    ) -> np.ndarray:
        """
        Accepts tracker outputs in the format:
        {
            "track_id": int|str,
            "bbox": [x1,y1,x2,y2],           # may be float
            "age": int,                      # 0 on first frame of this ID
            "missed": int,                   # consecutive misses
            "updated": bool,                 # True if matched this frame

            # Optional extras (drawn if present):
            "label": "player1"|"player2"|str,
        }
        """
        self._frame_idx += 1
        H, W = frame.shape[:2]

        for obj in tracker_outputs:
            # Defensive parsing
            if "track_id" not in obj:
                continue
            tid = obj["track_id"]
            bbox = obj.get("bbox", None)
            age = int(obj.get("age", 0))
            missed = int(obj.get("missed", 0))
            updated = bool(obj.get("updated", False))
            is_active = bool(obj.get("is_active", False))

            # Policy: new track when age == 0
            is_new = (age == 0)
            self._ensure_track(tid, reset=is_new)
            trail = self._trails[tid]
            self._last_seen[tid] = self._frame_idx

            # Style: green for updated; orange for predicted
            confident = bool(updated and missed == 0)
            box_color = (0, 255, 0) if (confident or is_active) else (0, 165, 255)
            thickness = 2 if confident else 1

            # Compose display label (now includes optional role/position)
            role = obj.get("label", None)  # e.g., "player1"/"player2"
            if isinstance(role, str):
                role_short = "P1" if role.lower().startswith("player1") else \
                    "P2" if role.lower().startswith("player2") else role
            else:
                role_short = None

            parts = []
            if role_short:
                parts.append(role_short)
            if show_ids:
                parts.append(str(tid))
            overlay_text = " | ".join(parts)

            # Draw if we have a bbox for this frame
            if bbox is not None:
                x1, y1, x2, y2 = self._clip_box(bbox, H, W)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

                (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y_top = max(0, y1 - th - 6)
                cv2.rectangle(frame, (x1, y_top), (x1 + tw + 6, y1), box_color, -1)
                cv2.putText(frame, overlay_text, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # Update trail with center if this is a valid box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trail.append((cx, cy))
            else:
                # If no bbox this frame, don't append a point.
                pass

            # Draw trail using a per-ID consistent color to separate tracks
            color_id = self._id_color(tid)
            for i in range(1, len(trail)):
                p0, p1 = trail[i - 1], trail[i]
                if p0 is None or p1 is None:
                    continue
                alpha = i / max(1, len(trail))  # older → thinner
                rad = max(1, int(6 * alpha))
                cv2.line(frame, p0, p1, color_id, rad)

            if len(trail) > 0 and trail[-1] is not None:
                cv2.circle(frame, trail[-1], 3, (0, 0, 255), -1)

        self._gc_stale()
        return frame

    def draw_world_speeds(
            self,
            frame: np.ndarray,
            world_tracks: List[dict],
            unit: str = "kmh",  # "kmh" | "mph" | "mps"
            decimals: int = 1,
            place: str = "bottom_left_inside",
            # "top_left_outside" | "top_right_outside" | "bottom_left_inside" | "bottom_right_inside"
            min_speed_to_show: float = 0.1,  # in chosen unit; hide tiny jitter
            show_missing: bool = False,  # draw "—" if no speed
    ) -> np.ndarray:
        """
        Overlay speeds for world tracks onto the frame. Expects each item to contain:
          - "track_id": id
          - "bbox": [x1,y1,x2,y2]   (image px)
          - EITHER "speed_mps" OR "speed_kmh"/"speed_mph"
        Optionally uses "updated" and "missed" to color like draw_tracked_boxes_from_tracker.
        """
        H, W = frame.shape[:2]

        def _get_speed(obj) -> Union[float, None]:
            # Prefer explicit field for requested unit; else convert from m/s if present.
            if unit == "kmh" and obj.get("speed_kmh") is not None:
                return float(obj["speed_kmh"])
            if unit == "mph" and obj.get("speed_mph") is not None:
                return float(obj["speed_mph"])
            if unit == "mps" and obj.get("speed_mps") is not None:
                return float(obj["speed_mps"])
            # conversions from m/s
            v = obj.get("speed_mps", None)
            if v is None:
                return None
            v = float(v)
            if unit == "kmh":
                return v * 3.6
            if unit == "mph":
                return v * 2.2369362920544
            return v  # mps

        def _anchor(x1, y1, x2, y2, tw, th):
            pad = 4
            if place == "top_left_outside":
                xa = x1
                ya = max(0, y1 - th - 6)
            elif place == "top_right_outside":
                xa = max(0, x2 - tw - 6)
                ya = max(0, y1 - th - 6)
            elif place == "bottom_right_inside":
                xa = max(0, min(W - tw - 6, x2 - tw - pad))
                ya = min(H - th - 2, y2 - pad)
            else:  # "bottom_left_inside"
                xa = max(0, x1 + pad)
                ya = min(H - 2, y2 - pad)
            return int(xa), int(ya)

        for obj in world_tracks:
            if "bbox" not in obj or obj["bbox"] is None:
                continue
            tid = obj.get("track_id", "?")
            x1, y1, x2, y2 = self._clip_box(obj["bbox"], H, W)

            v = _get_speed(obj)
            if v is None:
                if not show_missing:
                    continue
                label = "—"
            else:
                if v < float(min_speed_to_show):
                    continue
                label = f"{v:.{decimals}f} {('km/h' if unit == 'kmh' else 'mph' if unit == 'mph' else 'm/s')}"

            # Color logic: green if updated+not missed, else orange; else fall back to per-ID color
            updated = bool(obj.get("updated", False))
            missed = int(obj.get("missed", 0))
            if updated and missed == 0:
                bg_color = (0, 255, 0)
                fg_color = (0, 0, 0)
            else:
                bg_color = (0, 165, 255)
                fg_color = (0, 0, 0)

            # text size
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            xa, ya = _anchor(x1, y1, x2, y2, tw, th)

            # background box
            cv2.rectangle(frame, (xa - 3, ya - th - 3), (xa + tw + 3, ya + 3), bg_color, -1)
            # text
            cv2.putText(frame, label, (xa, ya), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg_color, 2, cv2.LINE_AA)

        return frame


def plot_yolo_boxes(frame, result, color=(0, 0, 255), thickness=2):
    """
    Draw YOLO bounding boxes with confidence on an image/frame.

    Args:
        frame (numpy.ndarray): The image or video frame.
        result: YOLO model result containing result.boxes.xyxy and result.boxes.conf.
        color (tuple): BGR color for the boxes (default green).
        thickness (int): Thickness of rectangle borders.

    Returns:
        numpy.ndarray: Frame with bounding boxes drawn.
    """
    boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
    confs = result.boxes.conf.cpu().numpy()  # (N,)

    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)
        label = f"{conf:.2f}"  # Confidence with 2 decimals

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)

        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame
