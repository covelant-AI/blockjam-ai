from ultralytics import YOLO
from .playertracker.homography import estimate_homography_from_polygon
from .balltracker.tracker import BallTracker
from .playertracker.tracker import PlayerTracker
from .detector.utils import filter_yolo_results, build_results_from_xyxy_conf
from .speed.homography import HomographyTrackProcessor

class BallAndPlayerTracker:
    def __init__(
            self, 
            model_path, 
            device, 
            CONF_THRESH=0.1, 
            N_TRACKS=3, 
            BALL_CLASS=0, 
            PLAYER_CLASS=1,
            ):
        self.CONF_THRESH = CONF_THRESH
        self.N_TRACKS = N_TRACKS
        self.BALL_CLASS = BALL_CLASS
        self.PLAYER_CLASS = PLAYER_CLASS

        self.device = device
        self.model = YOLO(model_path)
        self.model = self.model.to(self.device)

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

        self.btracker = BallTracker(
            min_conf=self.CONF_THRESH,
            max_age=10,  # how many missed frames before dropping
            switch_patience=2,

            # anti-static knobs (matching)
            static_frames=2,  # frames of stillness before penalizing

            # output suppression of static tracks
            suppress_static_output=False,
            fps=fps,
            court_poly_px=court_poly_px,
            H_img2court=self.H_img2court,
        )

        self.ptracker = PlayerTracker(frame_rate=fps, court_polygon=court_poly_px)

        self.converter = HomographyTrackProcessor(
            H=self.H_img2court,
            ema_alpha=0.3,
            reject_over_kmh=300,
            position_mode="center",
        )

    def update_court_polygon(self, court_polygon):
        self.define_homography(court_polygon)
        self.converter.set_homography(self.H_img2court)
        self.btracker.update_polygon(court_polygon)
        self.ptracker.update_polygon(court_polygon)
            
    def detect_frames(self, frames, global_start_index=0, all_polygons=None, update_court_polygon_interval=None):
        if self.btracker is None or self.ptracker is None or self.converter is None:
            raise ValueError("Trackers not defined. Call define_trackers_using_polygons() first.")
        
        if update_court_polygon_interval is not None:
            assert all_polygons is not None, "all_polygons must be provided if update_court_polygon_interval is not None"
        
        all_world_tracks = []
        for frame_index, frame in enumerate(frames):
            global_frame_index = global_start_index + frame_index
            if update_court_polygon_interval is not None and global_frame_index % update_court_polygon_interval == 0:
                polygon_idx = min(global_frame_index // update_court_polygon_interval, len(all_polygons) - 1)
                self.update_court_polygon(all_polygons[polygon_idx])

            results = self.model(frame, conf=0.1, imgsz=(736, 1280))[0]
            #print all results ids
            print(results.boxes.cls)

            players_detections = filter_yolo_results(results, self.PLAYER_CLASS)
            ball_detections = filter_yolo_results(results, self.BALL_CLASS, topn=3)

            # track ball
            ball_tracks = self.btracker.update(ball_detections)

            # track player
            player_formatted = build_results_from_xyxy_conf(players_detections, self.PLAYER_CLASS)
            player_tracks = self.ptracker.formatted_update(player_formatted)

            all_tracks = player_tracks + ball_tracks
            world_tracks = self.converter.update(all_tracks, 1 / self.fps)

            all_world_tracks.append(world_tracks)

        return all_world_tracks
    
    def extract_from_world_tracks(self, world_tracks) -> tuple[list[tuple[float, float]], list[float], list[tuple[float, float]], list[float], list[tuple[float, float]], list[float]]:
        ball_detections, ball_speeds = [], []
        p1_detections, p1_speeds = [], []
        p2_detections, p2_speeds = [], []

        for world_track in world_tracks:
            # Get first detection for each label if it exists
            ball_det = next((t for t in world_track if t['label'] == "ball"), None)
            p1_det   = next((t for t in world_track if t['label'] == "player1"), None)
            p2_det   = next((t for t in world_track if t['label'] == "player2"), None)

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