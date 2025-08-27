# handler.py

import os
from services.analysis.core import analyze_video
from utils.bucket import download_file_from_firebase
from marshmallow import ValidationError
from schemas.ai import AIModels
from schemas.video import VideoRequestSchema

def handle_process_video(event, ai_models: AIModels):
    video_path = 'input_videos/input_video.mp4'
    try:        
        # Validate input
        schema = VideoRequestSchema()
        data = event['input']['data']
        errors = schema.validate(data)
        if errors:
            raise ValidationError({"Failed to validate process video request": errors})

        # Download and process
        download_file_from_firebase(data['video_url'], video_path)
        print("analysing video")
        timing_results = analyze_video(
            video_id=data['video_id'],
            video_path=video_path,
            ball_and_player_tracker=ai_models.ball_and_player_tracker,
            court_line_detector=ai_models.court_line_detector,
            letr_court_line_detector=ai_models.letr_court_line_detector,
        )

        return {"message": "Video processed successfully", "timing_results": timing_results}

    except Exception as e:
        raise Exception({"Failed process video analysis": str(e)})

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


def handler(event, path, ai_models: AIModels):
    if path == 'process_video':
        return handle_process_video(event, ai_models)
    else:
        return {"error": f"Unsupported task: {path}"}

        
