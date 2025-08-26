from ultralytics import YOLO
from .tracker import BTracker
import torch
from .top_n_rows import top_n_rows
from ai_core.mini_court.core import MiniCourt
from .homography import HomographyTrackProcessor
from .interpolate import interpolate_ball_track
from tqdm import tqdm

class BallTracker:
    def __init__(self, model_path, device, N_TRACKS=5, CONF_THRESH=0.1):
        self.device = device
        self.N_TRACKS = N_TRACKS
        self.CONF_THRESH = CONF_THRESH
        self.converter = None

        self.model = YOLO(model_path)
        self.model.model = self.model.model.to(self.device)
        self.btracker = BTracker(
            max_tracks=N_TRACKS,  # tennis: keep the single best ball
            min_conf=CONF_THRESH,
            max_center_dist=150,  # tune per resolution/FPS
            max_age=10,  # how many missed frames before dropping
            smooth_alpha=0,  # small EMA to reduce jitter
            switch_patience=2,

            # anti-static knobs (matching)
            min_motion_px = 2,  # per-frame displacement below this counts as "still"
            static_frames= 2,  # frames of stillness before penalizing
            static_penalty_px= 60.0,  # additive penalty to matching cost when static
            motion_alpha= 1,  # EMA for speed_ema (0..1)

            # output suppression of static tracks
            suppress_static_output= True,
            output_min_speed= None,  # if None -> defaults to min_motion_px
            output_static_frames= None,  # if None -> defaults to static_frames
            )
                
    def detect_frames(self, frames, fps=30, reset=False):
        if self.converter is None:
            raise ValueError("Converter not set. Call set_keypoints() first.")
        
        detections = []
        ball_speeds = []
        for frame_idx, frame in tqdm(enumerate(frames), total=len(frames), desc="Processing frames"):
            results = self.model(frame, conf=self.CONF_THRESH, imgsz=(736, 1280))
            # yolo_boxes: list of [x1,y1,x2,y2,conf] for detections labeled "tennis ball"
            boxes = results[0].boxes.xyxy
            conf = results[0].boxes.conf

            # merge boxes with confidence
            merged = torch.cat((boxes, conf.unsqueeze(1)), dim=1).cpu().numpy()

            # filter top n boxes
            merged = top_n_rows(merged, self.N_TRACKS)

            # track ball
            tracks = self.btracker.update(merged)

            world_tracks = self.converter.update(tracks, 1 / fps)

            if world_tracks != []:
                #order world tracks by speed, handling None values
                world_tracks.sort(key=lambda x: x.get("speed_kmh") if x.get("speed_kmh") is not None else 0, reverse=True)
                x1,y1,x2,y2 = world_tracks[0]["bbox"]
                center_x = float((x1 + x2) / 2)
                center_y = float((y1 + y2) / 2)

                detections.append([center_x, center_y])
                ball_speeds.append(world_tracks[0].get("speed_kmh"))
            else:
                detections.append([None,None])
                ball_speeds.append(None)

            
        detections.pop(0)
        detections.append([None,None])
        ball_speeds.pop(0)
        ball_speeds.append(None)

        if reset:
            self.reset()

        return detections, ball_speeds
    
    def interpolate_ball_track(self, ball_track):
        return interpolate_ball_track(ball_track)
        
    def set_keypoints(self, court_keypoints):
        H = MiniCourt().compute_court_homography_matrix(court_keypoints)
        self.converter = HomographyTrackProcessor(
            H=H,
            ema_alpha=0.3,
            reject_over_kmh=250,
            position_mode="center"
        )

    def update_keypoints(self, court_keypoints):
        H = MiniCourt().compute_court_homography_matrix(court_keypoints)
        self.converter.set_homography(H)

    def reset(self):
        self.btracker.reset()
        self.converter.reset()
            

    