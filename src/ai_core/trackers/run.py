from time import time

import cv2
import numpy as np
from ultralytics import YOLO

from balltracker.multitracker import TennisBallActiveMultiTracker
from balltracker.visualizer import TennisBallVisualizer
from detector.utils import filter_yolo_results, build_results_from_xyxy_conf
from playertracker.homography import estimate_homography_from_polygon
from playertracker.tracker import PlayerTracker
from playertracker.visualizer import PlayerVisualizer
from speed.homography import Pixel2WorldConverter
from playertracker.homography import draw_court_overlay_cv2

# config
FPS = 30
VIDEO_PATH = "/home/marco/Downloads/clip.mp4"

# TL/TR/BR/BL
COURT1_POLYGON = np.array([(483, 99), (810, 105), (1235, 539), (30, 540)])
COURT2_POLYGON = np.array([(459, 246), (844, 246), (1615, 650), (-345, 650)])

SELECTED_COURT_POLYGON = COURT1_POLYGON

# ball initialization
BALL_CONF = 0.1
BALL_CLASS = 0

ball_model = YOLO("detector/weights/best-balls.engine")  # pretrained YOLO11n model
H_img2court, H_court2img, spec, _, _ = estimate_homography_from_polygon(
    SELECTED_COURT_POLYGON, use_doubles_polygon=True
)
btracker = TennisBallActiveMultiTracker.from_image_polygon(H_img2court, SELECTED_COURT_POLYGON, fps=FPS)
ball_visualizer = TennisBallVisualizer()

# player initialization
PLAYER_CONF = 0.1
PLAYER_CLASS = 0

player_model = YOLO("detector/weights/best-player.engine")
ptracker = PlayerTracker(frame_rate=FPS, court_polygon=SELECTED_COURT_POLYGON)
player_visualizer = PlayerVisualizer()

# image to real world converter
converter = Pixel2WorldConverter(H=H_img2court,
                                 ema_alpha=1,
                                 reject_over_kmh=2000,
                                min_samples_for_velocity=1
                                 )

if __name__ == '__main__':
    # Run batched inference on a list of images
    ball_results = ball_model(VIDEO_PATH, stream=True, conf=BALL_CONF, imgsz=(720, 1280),
                              verbose=False)  # return a generator of Results objects

    player_results = player_model(VIDEO_PATH, stream=True, conf=PLAYER_CONF, imgsz=640,
                                  classes=[PLAYER_CLASS], verbose=False)
    t0 = time()
    for frame_idx, (player_results, ball_results) in enumerate(zip(player_results, ball_results)):
        # yolo_boxes: list of [x1,y1,x2,y2,conf] for detections labeled "tennis ball" and player
        img = ball_results.orig_img

        players_detections = filter_yolo_results(player_results, PLAYER_CLASS, min_conf=PLAYER_CONF)
        ball_detections = filter_yolo_results(ball_results, BALL_CLASS, min_conf=BALL_CONF)
        # track ball
        ball_track = btracker.get_active_ball(ball_detections, debug_image=img, show_debug=False)

        ball_tracks = []
        if ball_track:
            ball_tracks.append(ball_track)

        ball_tracks = converter.update(ball_tracks, 1 / FPS, position_mode="center")

        # track player
        player_tracks = ptracker.formatted_update(build_results_from_xyxy_conf(players_detections, PLAYER_CLASS))
        player_tracks = converter.update(player_tracks, 1 / FPS, position_mode="bottom_center")


        # # visualization
        ball_visualizer.draw(img, ball_tracks)

        player_visualizer.draw_players(img, player_tracks)
        img = draw_court_overlay_cv2(img,SELECTED_COURT_POLYGON)
        # # show or write out
        cv2.imshow("track", img)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break
        print(f"Processed frame in {(time() - t0) * 1000} milli seconds")
        t0 = time()
