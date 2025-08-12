import runpod
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import ai_core
from src.routers import analysis
from dotenv import load_dotenv
load_dotenv()

def handler(job):
    route = job['input']['route'].split('/')
    path_suffix = '/'.join(route[1:])

    # Get the root directory path
    root_dir = os.path.dirname(__file__)
    models_path = os.getenv("MODELS_PATH", "models")
    models_path = os.path.join(root_dir, models_path)
    
    ai_models = ai_core.initialize_models(
    player_tracker_model_path=os.path.join(models_path, "yolo11s.pt"),
    court_line_detector_model_path=os.path.join(models_path, "model_tennis_court_det.pt"),
    letr_court_line_detector_model_path=os.path.join(models_path, "letr_best_checkpoint.pth"),
    ball_bounce_model_path=os.path.join(models_path, "bounce_detection_model_dict_w10.pth"),
    scoreboard_model_path=os.path.join(models_path, "scoreboard_yolov8s_best.pt"),
    tracknet_ball_tracker_model_path=os.path.join(models_path, "tracknet_model_best.pt"),
    racket_player_ball_detector_model_path=os.path.join(models_path, "racket_player_ball_yolo11n.pt"),
    shot_type_classifier_model_path=os.path.join(models_path, "tennis_rnn_rafa.pth"),
    movenet_pose_extractor_model_path=os.path.join(models_path, "movenet.tflite")
    )

    if route[0] == 'analysis':
        return analysis.handler(job, path_suffix, ai_models)
    else:
        return {"error": f"Unsupported task: {route[0]}"}
    

runpod.serverless.start({"handler":handler})