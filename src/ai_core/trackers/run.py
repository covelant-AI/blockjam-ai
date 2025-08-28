import cv2
import numpy as np
from ultralytics import YOLO

from balltracker.tracker import TennisBallActiveTracker
from detector.utils import filter_yolo_results, build_results_from_xyxy_conf
from playertracker.homography import draw_court_overlay_cv2
from playertracker.homography import estimate_homography_from_polygon
from playertracker.tracker import PlayerTracker
from speed.homography import HomographyTrackProcessor
from utils.visualizer import TrackedBoxDrawer, plot_yolo_boxes

# Load a model
model = YOLO("detector/weights/best2class.pt")  # pretrained YOLO11n model

CONF_THRESH = 0.1
N_TRACKS = 3
FPS = 30

BALL_CLASS = 0
PLAYER_CLASS = 1
# TL/TR/BR/BL
COURT1_POLYGON = np.array([(483, 99), (810, 105), (1235, 539), (30, 540)])
COURT2_POLYGON = np.array([(459, 246), (844, 246), (1278, 473), (1, 476)])


SELECTED_COURT_POLYGON = COURT1_POLYGON
H_img2court, H_court2img, spec, _, _ = estimate_homography_from_polygon(
    SELECTED_COURT_POLYGON, use_doubles_polygon=True
)

# ball tracker
btracker = TennisBallActiveTracker.from_image_polygon(H_img2court, SELECTED_COURT_POLYGON, fps=30.0)

# player tracker
ptracker = PlayerTracker(frame_rate=FPS, court_polygon=SELECTED_COURT_POLYGON)

# visualizer
visualizer = TrackedBoxDrawer(trail_maxlen=40, stale_ttl_frames=90)

# image to real world converter
converter = HomographyTrackProcessor.from_file(
    "court_homography.npz",
    ema_alpha=0.3,
    reject_over_kmh=300,
    position_mode="center"
)

if __name__ == '__main__':
    # Run batched inference on a list of images
    results = model("/home/marco/Downloads/PTTKrakow10m.mp4", stream=True, conf=CONF_THRESH,
                    imgsz=(720, 1280))  # return a generator of Results objects

    for frame_idx, result in enumerate(results):
        # yolo_boxes: list of [x1,y1,x2,y2,conf] for detections labeled "tennis ball" and player
        img = result.orig_img

        players_detections = filter_yolo_results(result, PLAYER_CLASS)
        ball_detections = filter_yolo_results(result, BALL_CLASS, topn=3)
        print(len(ball_detections))
        # track ball
        ball_track = btracker.update(ball_detections)
        if not ball_track:
            ball_tracks = []
        else:
            ball_tracks = [ball_track]

        # track player
        player_tracks = ptracker.formatted_update(build_results_from_xyxy_conf(players_detections, PLAYER_CLASS))

        all_tracks = player_tracks + ball_tracks
        # convert to real world speeds
        world_tracks = converter.update(all_tracks, 1 / FPS)

        # img = plot_yolo_boxes(img, result)
        img = visualizer.draw_tracked_boxes_from_tracker(img, all_tracks, show_ids=True)
        # img = visualizer.draw_world_speeds(
        #     img,
        #     world_tracks,
        #     unit="mps",  # "kmh" | "mph" | "mps"
        #     decimals=1,
        #     place="bottom_left_inside",  # avoids the existing top-left ID label
        #     min_speed_to_show=0.5,  # suppress micro-jitter
        # )

        img = draw_court_overlay_cv2(img, SELECTED_COURT_POLYGON)
        # show or write out
        cv2.imshow("track", img)
        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break
