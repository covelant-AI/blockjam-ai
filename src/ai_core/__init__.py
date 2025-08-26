import torch
import os
from .player_tracker import PlayerTracker
from .ball_bounce_model import BallBounceModel
from .mini_court import MiniCourt
from .scoreboard import Scoreboard
from .tracknet_ball_tracker import TrackNetBallTracker
from .court_line_detector import CourtLineDetector
from .letr_court_line_detector import LETRCourtDetector
from .racket_player_ball_detector import RacketPlayerBallDetector
from .shot_type_classifier import ShotTypeClassifier
from .ball_tracker import BallTracker

__all__ = ["PlayerTracker", "CourtLineDetector", "BallBounceModel", "MiniCourt", "BallTracker", "LETRCourtDetector", "RacketPlayerBallDetector", "ShotTypeClassifier"]

def initialize_models(player_tracker_model_path, court_line_detector_model_path, letr_court_line_detector_model_path, ball_bounce_model_path, scoreboard_model_path, ball_tracker_model_path, racket_player_ball_detector_model_path, shot_type_classifier_model_path, movenet_pose_extractor_model_path):
    # Import here to avoid circular import
    from schemas.ai import AIModels
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cpu':
        # Enhanced error message with diagnostic information
        cuda_info = {
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A',
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        error_msg = f"CPU is not supported for AI models. Please use a GPU.\n"
        error_msg += f"Diagnostic info: {cuda_info}\n"
        error_msg += f"PyTorch version: {cuda_info['torch_version']}\n"
        error_msg += f"CUDA available: {cuda_info['cuda_available']}\n"
        
        if cuda_info['cuda_available']:
            error_msg += f"CUDA version: {cuda_info['cuda_version']}\n"
            error_msg += f"GPU count: {cuda_info['gpu_count']}\n"
            for i in range(cuda_info['gpu_count']):
                gpu_name = torch.cuda.get_device_name(i)
                error_msg += f"GPU {i}: {gpu_name}\n"
        else:
            error_msg += "CUDA is not available. This may be due to:\n"
            error_msg += "1. Incompatible CUDA version (PyTorch 2.0+ requires CUDA 11.8, 12.1, 12.4, or 12.5)\n"
            error_msg += "2. Missing CUDA drivers\n"
            error_msg += "3. PyTorch was installed without CUDA support\n"
        
        raise Exception(error_msg)

    player_tracker = PlayerTracker(model_path=player_tracker_model_path, device=device)
    ball_tracker = BallTracker(model_path=ball_tracker_model_path, device=device)
    racket_player_ball_detector = RacketPlayerBallDetector(model_path=racket_player_ball_detector_model_path, device=device)
    court_line_detector = CourtLineDetector(court_line_detector_model_path, device=device)
    letr_court_line_detector = LETRCourtDetector(letr_court_line_detector_model_path, device=device)
    ball_bounce_model = BallBounceModel(model_path=ball_bounce_model_path, device=device, window_size=10)
    mini_court = MiniCourt(output_width=1280, output_height=720)
    scoreboard_model = Scoreboard(model_path=scoreboard_model_path, device=device)
    shot_type_classifier = ShotTypeClassifier(model_path=shot_type_classifier_model_path, device=device, pose_extractor_model_path=movenet_pose_extractor_model_path)
    return AIModels(
        player_tracker=player_tracker,
        ball_tracker=ball_tracker,
        racket_player_ball_detector=racket_player_ball_detector,
        court_line_detector=court_line_detector,
        letr_court_line_detector=letr_court_line_detector,
        ball_bounce_model=ball_bounce_model,
        mini_court=mini_court,
        scoreboard_model=scoreboard_model,
        shot_type_classifier=shot_type_classifier
        )