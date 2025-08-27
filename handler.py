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
        ball_and_player_tracker_model_path=os.path.join(models_path, "yolo_ball_and_player_best.pt"),
        court_line_detector_model_path=os.path.join(models_path, "model_tennis_court_det.pt"),
        letr_court_line_detector_model_path=os.path.join(models_path, "letr_best_checkpoint.pth"),
    )

    if route[0] == 'analysis':
        return analysis.handler(job, path_suffix, ai_models)
    else:
        return {"error": f"Unsupported task: {route[0]}"}
    

runpod.serverless.start({"handler":handler})