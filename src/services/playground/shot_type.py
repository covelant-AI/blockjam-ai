from src.ai_core.shot_type_classifier.core import ShotTypeClassifier
from src.ai_core.player_tracker.core import PlayerTracker
from src.utils.video_utils import read_video
from src.ai_core.letr_court_line_detector import LETRCourtDetector

def test_shot_type():
    shot_type_classifier = ShotTypeClassifier(
        model_path="models/tennis_rnn_rafa.pth",
        pose_extractor_model_path="models/movenet.tflite",
        device="cuda"
    )

    player_tracker = PlayerTracker(
        model_path="models/yolo11s.pt",
        device="cuda"  # Use GPU for player tracking
    )

    court_detector = LETRCourtDetector(
        model_path="models/letr_best_checkpoint.pth",
        device="cuda"
    )


    frames = read_video("input_videos/short_clip.mp4")
    middle_frame = frames[len(frames) // 2]
    court_keypoints = court_detector.detect(middle_frame)


    player_detections = player_tracker.detect_frames(frames)
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    p1_bboxes = [pd.get("1", None) for pd in player_detections]
    p2_bboxes = [pd.get("2", None) for pd in player_detections]

    # shot_type_classifier.classify(frames, p1_bboxes, flip=True)
    # print(shot_type_classifier.shot_counter.results)

    # shot_type_classifier.human_pose_extractor.cleanup()

    # shot_type_classifier.shot_counter.results = []
    shot_type_classifier.classify(frames, p2_bboxes, flip=False)
    print(shot_type_classifier.shot_counter.results)
    