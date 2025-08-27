from typing import Optional, List

import numpy as np
from .debug_helpers import debug_dump_player_uv, draw_player_uv_overlay
from .labeler import PlayerTopBottomLabeler
from .playerfilter import ActivePlayerFilter
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import IterableSimpleNamespace

CFG = {
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 300,
    "match_thresh": 0.7,
    "fuse_score": True,
    "gmc_method": "sparseOptFlow",
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.8,
    "with_reid": False,
    "model": "auto",
}


def _xyxy_from_track(t) -> np.ndarray:
    """
    Robustly get [x1,y1,x2,y2] from an Ultralytics STrack-like object.
    Prefers .xyxy if present; falls back to tlwh -> xyxy.
    """
    # some builds expose xyxy as property/array; normalize to 1D
    if hasattr(t, "xyxy"):
        arr = np.asarray(getattr(t, "xyxy")).reshape(-1)
        if arr.size >= 4:
            return arr[:4].astype(float)

    # else use tlwh
    if hasattr(t, "tlwh"):
        tlwh = np.asarray(t.tlwh).reshape(-1)
        x, y, w, h = tlwh[:4].astype(float)
        return np.array([x, y, x + w, y + h], dtype=float)

    # absolute worst case, try bbox attribute names sometimes used downstream
    if hasattr(t, "bbox"):
        b = np.asarray(t.bbox).reshape(-1)
        if b.size >= 4:
            return b[:4].astype(float)

    raise AttributeError("Track does not expose xyxy/tlwh/bbox")


class PlayerTracker(BOTSORT):
    def __init__(self, args=CFG, frame_rate=30, img_shape=None, court_polygon=None):
        super().__init__(IterableSimpleNamespace(**args), frame_rate)
        assert court_polygon is not None or img_shape is not None, \
            "court polygon or image shape must be provided"

        # Active player selector keeps its own temporal state
        self.active_filter = ActivePlayerFilter()
        self.active_filter.init_from_polygon(court_polygon, use_doubles_polygon=True)

        # Keep the polygon here as well in case you want to update it later
        self.court_polygon = court_polygon
        # player labeler
        self.labeler = PlayerTopBottomLabeler(patience_frames=45, swap_hysteresis_px_at_1080p=18.0)


    def update_polygon(self, court_polygon):
        self.active_filter.init_from_polygon(court_polygon, use_doubles_polygon=True)
        self.court_polygon = court_polygon

    def formatted_update(
            self,
            results,  # detector outputs consumed by BOTSORT.update
            img: Optional[np.ndarray] = None,
            feats: Optional[np.ndarray] = None
    ) -> List[dict]:
        """
        Runs BoT-SORT, selects the 2 active players, and returns a compact list:
        [{track_id, bbox[x1,y1,x2,y2], conf}]
        """
        # Run the tracker (we don't use the returned array for filtering)
        _ = self.update(results, img, feats)

        # Use real track objects for selection
        tracks = [t for t in self.tracked_stracks if getattr(t, "is_activated", True)]

        active_ids = self.active_filter.select_active_ids(
            tracks=tracks,
            frame_id=self.frame_id
        )
        active_set = set(int(i) for i in active_ids)

        # Format only the two selected players
        formatted_result: List[dict] = []
        for t in tracks:
            tid = int(t.track_id)
            if tid not in active_set:
                continue
            x1, y1, x2, y2 = _xyxy_from_track(t)
            conf = float(getattr(t, "score", getattr(t, "conf", 1.0)))
            formatted_result.append({
                "track_id": tid,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": conf,
            })

        # add player labels
        labels = self.labeler.label(formatted_result, frame_id=self.frame_id,
                                    img_shape=tuple(img.shape[:2]) if img is not None else None,
                                    court_polygon=self.court_polygon)
        # merge labels back
        by_id = {d["track_id"]: d for d in formatted_result}
        for lab in labels:
            if lab["track_id"] in by_id:
                by_id[lab["track_id"]].update({"label": lab["label"], "position": lab["position"]})
        formatted_result = list(by_id.values())
        # If fewer than two found, you'll get 0/1 items—caller should handle that.
        return formatted_result
