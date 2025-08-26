from ai_core.ball_tracker import BallTracker
from ai_core.tracknet_ball_tracker import TrackNetBallTracker
from utils.video_utils import read_video_range, read_video, get_video_info, save_video
import cv2
import numpy as np
import time
from ai_core.letr_court_line_detector import LETRCourtDetector

def compare_ball_trackers():
    video_path = "input_videos/short_clip.mp4"
    ball_tracker = BallTracker(model_path="models/yolo_ball_tracker_best.pt", device="cuda")
    tracknet_ball_tracker = TrackNetBallTracker(model_path="models/tracknet_model_best.pt", device="cuda")
    letr_court_line_detector = LETRCourtDetector(model_path="models/letr_best_checkpoint.pth", device="cuda")

    frames = read_video(video_path)
    # Create deep copies of the actual frame data, not just the list
    # frames_copy = [frame.copy() for frame in frames]
    video_info = get_video_info(video_path)

    court_keypoints = letr_court_line_detector.detect(frames[100])
    ball_tracker.set_keypoints(court_keypoints)
    
    yolo_start_time = time.time()
    detections, ball_speeds = ball_tracker.detect_frames(frames, video_info["fps"])
    detections = ball_tracker.interpolate_ball_track(detections)
    yolo_end_time = time.time()
    print(f"YOLO time: {yolo_end_time - yolo_start_time}")

    # tracknet_start_time = time.time()
    # tracknet_detections = tracknet_ball_tracker.detect_frames(frames, video_info)
    # tracknet_end_time = time.time()
    # print(f"TrackNet time: {tracknet_end_time - tracknet_start_time}")
    
    
    #Draw detections on YOLO frames
    for frame_idx, frame in enumerate(frames):
        #First draw bbox detection
        detection = detections[frame_idx]
        if detection is not None and detection[0] is not None:
            x1, y1 = detection
            frame = cv2.circle(frame, (int(x1), int(y1)), 5, (0, 255, 0), -1)

            if ball_speeds[frame_idx] is not None:
                frame = cv2.putText(frame, f"{ball_speeds[frame_idx]:.2f} km/h", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # #Draw detections on TrackNet frames (separate frame objects)
    # for frame_idx, frame in enumerate(frames_copy):
    #     tracknet_detection = tracknet_detections[frame_idx]
    #     if tracknet_detection is not None and tracknet_detection[0] is not None:
    #         x, y = tracknet_detection
    #         frame = cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    save_video(frames, "output_videos/compare_ball_trackers_yolo.avi", video_info["fps"])
    # save_video(frames_copy, "output_videos/compare_ball_trackers_tracknet.avi", video_info["fps"])

        