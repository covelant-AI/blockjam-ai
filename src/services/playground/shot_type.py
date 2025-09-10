from src.ai_core.shot_type_classifier.core import ShotTypeClassifier
from src.ai_core.player_tracker.core import PlayerTracker
from src.utils.video_utils import read_video, get_video_info, read_video_range, save_video
from src.ai_core.letr_court_line_detector import LETRCourtDetector
from src.utils.simple import load_json_file
import cv2

def test_shot_type():
    # shot_type_classifier = ShotTypeClassifier(
    #     model_path="models/tennis_rnn_rafa.pth",
    #     pose_extractor_model_path="models/movenet.tflite",
    #     device="cuda"
    # )

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

    models = ["models/tennis_rnn_rafa.pth", "models/tennis_rnn_rafa2.pth", "models/tennis_rnn_rafa3.pth"]

    all_p1_shots = {}
    all_p2_shots = {}

    for model in models:
        shot_type_classifier = ShotTypeClassifier(
            model_path=model,
            pose_extractor_model_path="models/movenet.tflite",
            device="cuda"
        )
        p1_shots = shot_type_classifier.classify(frames, p1_bboxes, flip=True)
        shot_type_classifier.shot_counter.results = []
        p2_shots = shot_type_classifier.classify(frames, p2_bboxes, flip=False)
        shot_type_classifier.shot_counter.results = []

        all_p1_shots[model] = p1_shots
        all_p2_shots[model] = p2_shots

    print(all_p1_shots)
    print(all_p2_shots)


def show_results():
    json_data = load_json_file('src/services/playground/classify_shots.json')["data"]
    video_path = "input_videos/PTTKrakow.mp4"
    video_info = get_video_info(video_path)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs("output_videos/shot_type", exist_ok=True)
    
    for _s in json_data:
        section = _s["section"]
        section_data = _s["data"]
        frames = read_video_range(video_path, section["start"]["index"], section["end"]["index"])
        p1_shots = section_data["p1"]
        p2_shots = section_data["p2"] if "p2" in section_data else []

        # Initialize shot type tracking for this section
        p1_current_shot = "No shot yet"
        p2_current_shot = "No shot yet"
        
        # Process each frame in the section
        for frame_idx, frame in enumerate(frames):
            # Check if this frame has a shot event
            p1_shot_this_frame = None
            p2_shot_this_frame = None
            
            # Check P1 shots
            for shot in p1_shots:
                if shot["index"] == frame_idx:
                    p1_shot_this_frame = shot["shot_type"]
                    p1_current_shot = shot["shot_type"]
                    break
            
            # Check P2 shots
            for shot in p2_shots:
                if shot["index"] == frame_idx:
                    p2_shot_this_frame = shot["shot_type"]
                    p2_current_shot = shot["shot_type"]
                    break
            
            # Add text overlay to the frame
            # Section info with background
            cv2.rectangle(frame, (5, 10), (350, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"Section: {section['start']['time']:.1f}s - {section['end']['time']:.1f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Current frame info with background
            cv2.rectangle(frame, (5, 45), (200, 70), (0, 0, 0), -1)
            cv2.putText(frame, f"Frame: {frame_idx} / {len(frames)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Player 1 shot info (left side)
            if p1_shot_this_frame:
                # Show shot event with highlight
                shot_time = section['start']['time'] + (frame_idx / len(frames)) * (section['end']['time'] - section['start']['time'])
                # Add background rectangle for better visibility
                cv2.rectangle(frame, (5, 75), (200, 140), (0, 0, 0), -1)
                cv2.putText(frame, f"P1: {p1_shot_this_frame.upper()}!", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                cv2.putText(frame, f"Time: {shot_time:.2f}s", 
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Show current shot type with background
                cv2.rectangle(frame, (5, 75), (200, 100), (0, 0, 0), -1)
                cv2.putText(frame, f"P1: {p1_current_shot}", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Player 2 shot info (right side)
            if p2_shot_this_frame:
                # Show shot event with highlight
                shot_time = section['start']['time'] + (frame_idx / len(frames)) * (section['end']['time'] - section['start']['time'])
                # Add background rectangle for better visibility
                cv2.rectangle(frame, (frame.shape[1] - 205, 75), (frame.shape[1] - 5, 140), (0, 0, 0), -1)
                cv2.putText(frame, f"P2: {p2_shot_this_frame.upper()}!", 
                           (frame.shape[1] - 200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                cv2.putText(frame, f"Time: {shot_time:.2f}s", 
                           (frame.shape[1] - 200, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # Show current shot type with background
                cv2.rectangle(frame, (frame.shape[1] - 205, 75), (frame.shape[1] - 5, 100), (0, 0, 0), -1)
                cv2.putText(frame, f"P2: {p2_current_shot}", 
                           (frame.shape[1] - 200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add shot counter info with background
            cv2.rectangle(frame, (5, frame.shape[0] - 35), (300, frame.shape[0] - 5), (0, 0, 0), -1)
            cv2.putText(frame, f"P1 shots: {len(p1_shots)} | P2 shots: {len(p2_shots)}", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Save the processed section with shot overlays
        output_path = f"output_videos/shot_type/section_{section['start']['time']:.1f}s_to_{section['end']['time']:.1f}s.avi"
        save_video(frames, output_path, video_info["fps"])
        
        # Print section summary
        print(f"\n=== Section Summary: {section['start']['time']:.1f}s - {section['end']['time']:.1f}s ===")
        print(f"P1 shots: {len(p1_shots)}")
        if p1_shots:
            p1_shot_types = [shot['shot_type'] for shot in p1_shots]
            print(f"P1 shot types: {', '.join(p1_shot_types)}")
        print(f"P2 shots: {len(p2_shots)}")
        if p2_shots:
            p2_shot_types = [shot['shot_type'] for shot in p2_shots]
            print(f"P2 shot types: {', '.join(p2_shot_types)}")
        print(f"Output saved to: {output_path}")
        print("=" * 60)