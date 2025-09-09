from ultralytics import YOLO
from .playertracker.homography import estimate_homography_from_polygon
from .balltracker.multitracker import TennisBallActiveMultiTracker
from .playertracker.tracker import PlayerTracker
from .detector.utils import filter_yolo_results, build_results_from_xyxy_conf
from .speed.homography import Pixel2WorldConverter

class BallAndPlayerTracker:
    def __init__(
            self, 
            ball_model_path, 
            player_model_path,
            device, 
            CONF_THRESH=0.1, 
            BALL_CLASS=0, 
            PLAYER_CLASS=0,
            ):
        self.CONF_THRESH = CONF_THRESH
        self.BALL_CLASS = BALL_CLASS
        self.PLAYER_CLASS = PLAYER_CLASS

        self.device = device
        self.ball_model = YOLO(ball_model_path)
        self.ball_model = self.ball_model.to(self.device)

        self.player_model = YOLO(player_model_path)
        self.player_model = self.player_model.to(self.device)

    def define_homography(self, court_poly_px):
        H_img2court, H_court2img, spec, _, _ = estimate_homography_from_polygon(
            court_poly_px, use_doubles_polygon=True
        )
        self.H_img2court = H_img2court
        self.H_court2img = H_court2img
        self.spec = spec

    def define_trackers_using_polygons(self, court_poly_px, fps):
        """
        court_poly_px: list of 4 points in the order of top left, top right, bottom right, bottom left
        fps: frame rate of the video
        """
        self.define_homography(court_poly_px)

        self.fps = fps

        self.btracker = TennisBallActiveMultiTracker.from_image_polygon(self.H_img2court, court_poly_px, fps=fps)

        self.ptracker = PlayerTracker(frame_rate=fps, court_polygon=court_poly_px)

        self.converter = Pixel2WorldConverter(H=self.H_img2court,
                                 ema_alpha=1,
                                 reject_over_kmh=200,
                                min_samples_for_velocity=1
                                 )

    def update_court_polygon(self, court_polygon):
        self.define_homography(court_polygon)
        self.converter.set_homography(self.H_img2court)
        # self.btracker.update_polygon(court_polygon)
        self.ptracker.update_polygon(court_polygon)
            
    def detect_frames(self, frames, global_start_index=0, all_polygons=None, update_court_polygon_interval=None):
        if self.btracker is None or self.ptracker is None or self.converter is None:
            raise ValueError("Trackers not defined. Call define_trackers_using_polygons() first.")
        
        if update_court_polygon_interval is not None:
            assert all_polygons is not None, "all_polygons must be provided if update_court_polygon_interval is not None"
        
        all_player_tracks = []
        all_ball_tracks = []
        for frame_index, frame in enumerate(frames):
            ball_results = self.ball_model(frame, conf=self.CONF_THRESH, imgsz=(720, 1280))[0]
            player_results = self.player_model(frame, conf=self.CONF_THRESH, imgsz=640,
                                  classes=[self.PLAYER_CLASS])[0]

            global_frame_index = global_start_index + frame_index
            # if update_court_polygon_interval is not None and global_frame_index % update_court_polygon_interval == 0:
            #     polygon_idx = min(global_frame_index // update_court_polygon_interval, len(all_polygons) - 1)
            #     self.update_court_polygon(all_polygons[polygon_idx])

            players_detections = filter_yolo_results(player_results, self.PLAYER_CLASS, min_conf=self.CONF_THRESH)
            ball_detections = filter_yolo_results(ball_results, self.BALL_CLASS, min_conf=self.CONF_THRESH)

            ball_track = self.btracker.get_active_ball(ball_detections, debug_image=frame, show_debug=False)

            ball_tracks = []
            if ball_track:
                ball_tracks.append(ball_track)
            ball_tracks = self.converter.update(ball_tracks, 1 / self.fps, position_mode="center")

            # track player
            player_tracks = self.ptracker.formatted_update(build_results_from_xyxy_conf(players_detections, self.PLAYER_CLASS))
            player_tracks = self.converter.update(player_tracks, 1 / self.fps, position_mode="bottom_center")

            all_player_tracks.append(player_tracks)
            all_ball_tracks.append(ball_tracks)

        return all_ball_tracks, all_player_tracks
    
    def extract_from_world_tracks(self, req_ball_tracks, req_player_tracks) -> tuple[list[tuple[float, float]], list[float], list[tuple[float, float]], list[float], list[tuple[float, float]], list[float]]:
        ball_detections, ball_speeds = [], []
        p1_detections, p1_speeds = [], []
        p2_detections, p2_speeds = [], []

        for idx in range(len(req_ball_tracks)):
            # Get first detection for each label if it exists
            if len(req_ball_tracks[idx]) > 0:
                ball_det = req_ball_tracks[idx][0]
            else:
                ball_det = None
                
            p1_det   = next((t for t in req_player_tracks[idx] if t['label'] == "player1"), None)
            p2_det   = next((t for t in req_player_tracks[idx] if t['label'] == "player2"), None)

            for det, detections, speeds in [
                (ball_det, ball_detections, ball_speeds),
                (p1_det, p1_detections, p1_speeds),
                (p2_det, p2_detections, p2_speeds),
            ]:
                if det:
                    x1, y1, x2, y2 = det["bbox"]
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    detections.append((x_center, y_center))
                    speeds.append(det.get("speed_kmh"))
                else:
                    detections.append((None, None))
                    speeds.append(None)

        return ball_detections, ball_speeds, p1_detections, p1_speeds, p2_detections, p2_speeds