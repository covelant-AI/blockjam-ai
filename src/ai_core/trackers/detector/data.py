from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SimplifiedResults:
    """
        Minimal container for YOLO-style detections.

        Attributes
        ----------
        conf  : (N,) float32  -- per-detection confidence scores in [0,1].
        xywh  : (N,4) float32 -- axis-aligned boxes [xc, yc, w, h].
        cls   : (N,) int64    -- per-detection class ids.
        """
    conf: np.ndarray
    xywh: np.ndarray
    cls: np.ndarray

    def __post_init__(self):
        # Coerce types and shapes
        self.conf = np.asarray(self.conf, dtype=np.float32).reshape(-1)
        self.xywh = np.asarray(self.xywh, dtype=np.float32).reshape(-1, 4)
        self.cls = np.asarray(self.cls, dtype=np.int64).reshape(-1)

        N = self.conf.shape[0]

        # Basic sanity checks; relax/remove if your pipeline tolerates wider ranges
        if N > 0:
            if np.any(self.conf < 0.0) or np.any(self.conf > 1.0):
                raise ValueError("conf values should be in [0, 1].")
            if np.any(self.xywh[:, 2:] < 0):
                raise ValueError("xywh widths/heights must be non-negative.")

    @property
    def N(self) -> int:
        return self.conf.shape[0]

    def topn(self, k: int) -> "SimplifiedResults":
        """Return a new YOLOLite with only the top-k by confidence (no mutation)."""
        if k is None or k >= self.N:
            return self
        idx = np.argsort(-self.conf)[: int(k)]
        return SimplifiedResults(
            conf=self.conf[idx],
            xywh=self.xywh[idx],
            cls=self.cls[idx],
        )

    def __getitem__(self, indices):
        """
        Enable numpy-style indexing on SimplifiedResults.
        Returns a new SimplifiedResults with the selected indices.
        """
        return SimplifiedResults(
            conf=self.conf[indices],
            xywh=self.xywh[indices],
            cls=self.cls[indices],
        )

    def to_list(self):
        """
        Return [[xc,yc,w,h,conf,cls], ...]. Rotations (if any) are not included here
        to avoid mixing formats; handle xywhr separately if you need it.
        """
        return np.hstack([
            self.xywh,
            self.conf.reshape(-1, 1),
            self.cls.reshape(-1, 1).astype(np.float32),
        ]).tolist()
