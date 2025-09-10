from .model import CourtLineDetectorModel
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from .postprocess import postprocess, refine_kps
from .homography import get_trans_matrix, refer_kps
from utils.conversions import scale_points_to_size
class CourtLineDetector:
    def __init__(self, model_path, device):
        self.device = torch.device(device)
        self.model = CourtLineDetectorModel(out_channels=15)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def detect(self, frame, use_refine_kps=True, use_homography=True):
        original_height, original_width = frame.shape[:2]
        image = cv2.resize(frame, (1280, 720)) 
        img = cv2.resize(image, (640, 360)) # model input resolution
        inp = (img.astype(np.float32) / 255.)
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)

        out = self.model(inp.float().to(self.device))[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num]*255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
            if use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
                x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
            points.append((x_pred, y_pred))

        if use_homography:
            matrix_trans = get_trans_matrix(points)
            if matrix_trans is not None:
                points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                points = [np.squeeze(x) for x in points]

        try:
            keypoints = scale_points_to_size(points, (1280, 720), (original_width, original_height))
        except TypeError as e:
            raise TypeError(f"Could not find all keypoints: {e}")
        finally:
            return keypoints
    
    def draw_keypoints(self, image, keypoints):
        height, width = image.shape[:2]
        for j in range(len(keypoints)):
            if keypoints[j][0] is not None and keypoints[j][0] > 0 and keypoints[j][0] < width and keypoints[j][1] > 0 and keypoints[j][1] < height:
                image = cv2.circle(image, (int(keypoints[j][0]), int(keypoints[j][1])),
                                radius=0, color=(0, 0, 255), thickness=10)
                image = cv2.putText(image, str(j), (int(keypoints[j][0]), int(keypoints[j][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return image