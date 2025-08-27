from ai_core.ball_tracker import BallTracker
from ai_core.tracknet_ball_tracker import TrackNetBallTracker
from utils.video_utils import read_video_range, read_video, get_video_info, save_video
import cv2
import numpy as np
import time
from ai_core.letr_court_line_detector import LETRCourtDetector
from ai_core.trackers.core import BallAndPlayerTracker

def compare_ball_trackers():
    video_path = "input_videos/raketlon_2025_1.mov"
    ball_and_player_tracker = BallAndPlayerTracker(model_path="models/yolo_ball_and_player_best.pt", device="cuda")
    letr_court_line_detector = LETRCourtDetector(model_path="models/letr_best_checkpoint.pth", device="cuda")

    frames = read_video_range(video_path, 1000, 1500)
    # Create deep copies of the actual frame data, not just the list
    # frames_copy = [frame.copy() for frame in frames]
    video_info = get_video_info(video_path)

    court_keypoints = letr_court_line_detector.detect(frames[300])
    polygon = np.array([
        court_keypoints[0], # TL
        court_keypoints[1], # TR
        court_keypoints[3], # BR
        court_keypoints[2], # BL
    ],np.int32)
    ball_and_player_tracker.define_trackers_using_polygons(polygon, video_info["fps"])
    
    yolo_start_time = time.time()
    world_tracks = ball_and_player_tracker.detect_frames(frames)
    yolo_end_time = time.time()
    print(f"YOLO time: {yolo_end_time - yolo_start_time}")
    
    
    #Draw detections on YOLO frames
    for frame_idx, frame in enumerate(frames):
        #First draw bbox detection
        world_track = world_tracks[frame_idx]
        for track in world_track:
            if track["bbox"] is None:
                continue
            x1, y1, x2, y2 = track["bbox"]
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            speed = track["speed_kmh"]
            if speed is not None:
                frame = cv2.putText(frame, f"{speed:.2f} km/h", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        #Draw court polygon
        frame = cv2.polylines(frame, [polygon], True, (0, 0, 255), 2)


    save_video(frames, "output_videos/compare_ball_trackers_yolo.avi", video_info["fps"])

        