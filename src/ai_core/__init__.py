import torch
from .court_line_detector import CourtLineDetector
from .letr_court_line_detector import LETRCourtDetector
from .trackers.core import BallAndPlayerTracker

__all__ = ["BallAndPlayerTracker", "CourtLineDetector", "LETRCourtDetector"]

def initialize_models(ball_and_player_tracker_model_path, court_line_detector_model_path, letr_court_line_detector_model_path):
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

    ball_and_player_tracker = BallAndPlayerTracker(model_path=ball_and_player_tracker_model_path, device=device)
    court_line_detector = CourtLineDetector(court_line_detector_model_path, device=device)
    letr_court_line_detector = LETRCourtDetector(letr_court_line_detector_model_path, device=device)
    return AIModels(
        ball_and_player_tracker=ball_and_player_tracker,
        court_line_detector=court_line_detector,
        letr_court_line_detector=letr_court_line_detector,
        )