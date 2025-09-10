from typing import List, Dict, Any, Tuple, Optional
import numpy as np

def _center_from_track(t: Any) -> Tuple[int, float, float]:
    """Return (track_id, cx, cy) from a dict or an STrack-like object."""
    if isinstance(t, dict):
        tid = int(t["track_id"])
        if "bbox" in t:  # xyxy
            x1, y1, x2, y2 = t["bbox"]
            return tid, 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if "xywh" in t:  # center x,y,w,h
            cx, cy, _, _ = t["xywh"]
            return tid, float(cx), float(cy)
        raise KeyError("Track dict must contain 'bbox' (xyxy) or 'xywh'.")
    tid = int(getattr(t, "track_id"))
    if hasattr(t, "xywh"):
        cx, cy, _, _ = list(getattr(t, "xywh"))[:4]
        return tid, float(cx), float(cy)
    if hasattr(t, "xyxy"):
        x1, y1, x2, y2 = list(getattr(t, "xyxy"))[:4]
        return tid, 0.5 * (float(x1) + float(x2)), 0.5 * (float(y1) + float(y2))
    if hasattr(t, "tlwh"):
        x, y, w, h = list(getattr(t, "tlwh"))[:4]
        return tid, float(x) + float(w) * 0.5, float(y) + float(h) * 0.5
    raise AttributeError("Unsupported track type for center extraction.")

class PlayerTopBottomLabeler:
    """
    Stable top/bottom labeling with:
      - 'patience' frames to keep previous labels when one disappears
      - small hysteresis margin to avoid swaps on near-ties
    """
    def __init__(self,
                 patience_frames: int = 45,
                 swap_hysteresis_px_at_1080p: float = 18.0):
        self.patience = int(patience_frames)
        self.swap_hysteresis_1080 = float(swap_hysteresis_px_at_1080p)
        self.last_ids = {"player1": None, "player2": None}  # player1=top, player2=bottom
        self.last_cy  = {}          # tid -> cy
        self.last_seen = {}         # tid -> frame_id

    def reset(self):
        self.last_ids = {"player1": None, "player2": None}
        self.last_cy.clear()
        self.last_seen.clear()

    def label(self,
              active_tracks: List[Any],
              frame_id: int,
              img_shape: Optional[Tuple[int, int]] = None,
              court_polygon: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Args:
            active_tracks: 0..2 tracks (dicts or STrack-like).
            frame_id: current frame number.
            img_shape: (H,W) for scaling hysteresis; if None, assume 1080p scale.
            court_polygon: optional; ignored here (we label purely by y).
        Returns:
            List of dicts with {track_id, label('player1'|'player2'), position('top'|'bottom')}
            player1 always corresponds to the visually upper (smaller y) track when both are present,
            with hysteresis to avoid flip-flopping.
        """
        if not active_tracks:
            return []

        # Scale hysteresis to image height
        H = float(img_shape[0]) if img_shape is not None else 1080.0
        swap_hyst = self.swap_hysteresis_1080 * (H / 1080.0)

        # Extract centers
        items = []
        for t in active_tracks:
            tid, cx, cy = _center_from_track(t)
            self.last_cy[tid] = cy
            self.last_seen[tid] = frame_id
            items.append((tid, cx, cy, t))

        # If we have two, sort by y, but apply a small hysteresis if they are almost tied and we had labels.
        if len(items) == 2:
            # Sort by cy (top first)
            items.sort(key=lambda z: z[2])
            (tid_top, _, cy_top, _), (tid_bot, _, cy_bot, _) = items

            # If previous labels exist for these two IDs, and separation is tiny,
            # keep previous mapping to avoid oscillation on borderline cases.
            prev_p1, prev_p2 = self.last_ids["player1"], self.last_ids["player2"]
            if (prev_p1 in (tid_top, tid_bot)) and (prev_p2 in (tid_top, tid_bot)):
                if abs(cy_bot - cy_top) < swap_hyst:
                    # Keep previous assignment regardless of who is marginally higher
                    mapping = {prev_p1: "player1", prev_p2: "player2"}
                else:
                    mapping = {tid_top: "player1", tid_bot: "player2"}
            else:
                mapping = {tid_top: "player1", tid_bot: "player2"}

            self.last_ids["player1"] = next((tid for tid, lbl in mapping.items() if lbl == "player1"), None)
            self.last_ids["player2"] = next((tid for tid, lbl in mapping.items() if lbl == "player2"), None)

            out = []
            for tid, _, cy, _ in items:
                label = mapping[tid]
                out.append({"track_id": int(tid),
                            "label": label,
                            "position": "top" if label == "player1" else "bottom"})
            # ensure player1 first
            out.sort(key=lambda d: 0 if d["label"] == "player1" else 1)
            return out

        # If we have only one, keep its previous label if possible; otherwise assign by y vs midline.
        tid, cx, cy, _ = items[0]

        if self.last_ids["player1"] == tid:
            label = "player1"
        elif self.last_ids["player2"] == tid:
            label = "player2"
        else:
            # Unknown ID; see if the *other* label's previous ID is still within patience.
            now = frame_id
            other_label = None
            # Prefer to keep the label of the ID that is still within patience and closer in y
            candidates = []
            for lbl in ("player1", "player2"):
                prev_tid = self.last_ids[lbl]
                if prev_tid is None:
                    continue
                last_seen = self.last_seen.get(prev_tid, -10**9)
                if (now - last_seen) <= self.patience:
                    prev_cy = self.last_cy.get(prev_tid, None)
                    if prev_cy is not None:
                        candidates.append((abs(cy - prev_cy), lbl))
            if candidates:
                candidates.sort(key=lambda z: z[0])
                label = candidates[0][1]  # stick to the closer label in y
            else:
                # Fall back to absolute y position versus midline
                if court_polygon is not None:
                    miny, maxy = float(court_polygon[:,1].min()), float(court_polygon[:,1].max())
                    mid_y = 0.5 * (miny + maxy)
                else:
                    mid_y = H / 2.0
                label = "player1" if cy <= mid_y else "player2"

        # Update last_ids for the single-visible case while keeping the other side reserved within patience.
        self.last_ids[label] = tid
        out = [{"track_id": int(tid),
                "label": label,
                "position": "top" if label == "player1" else "bottom"}]
        return out
