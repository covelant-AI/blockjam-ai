# handler.py

import os
from services.analysis.core import analyze_video
from services.analysis.ball_based_sectioning import ball_based_sectioning
from utils.bucket import download_file_from_firebase
from marshmallow import Schema, fields, ValidationError
from services.analysis.court_keypoints import court_keypoints_by_time_step
from schemas.stubs import PlayerStubsRequest
from services.analysis.player_stubs import player_stubs_for_sections
from services.analysis.player_speed import player_speed_for_section
from schemas.player_speed import PlayerSpeedsRequest
from utils.conversions import convert_detections_to_mini_court_coordinates
from schemas.ball_speed import BallSpeedsRequest
from services.analysis.ball_speed import ball_speed_for_section
from services.analysis.scoreboard import scoreboard_service
from schemas.ai import AIModels
from schemas.video import VideoRequestSchema, Feature

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
            features= [Feature(feature) for feature in data['features']],
            video_path=video_path,
            player_tracker=ai_models.player_tracker,
            ball_tracker=ai_models.ball_tracker,
            racket_player_ball_detector=ai_models.racket_player_ball_detector,
            court_line_detector=ai_models.court_line_detector,
            letr_court_line_detector=ai_models.letr_court_line_detector,
            ball_bounce_model=ai_models.ball_bounce_model,
            mini_court=ai_models.mini_court,
            scoreboard_model=ai_models.scoreboard_model,
            shot_type_classifier=ai_models.shot_type_classifier,
        )

        return {"message": "Video processed successfully", "timing_results": timing_results}

    except Exception as e:
        raise Exception({"Failed process video analysis": str(e)})

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


def handle_ball_based_sectioning(event, ai_models: AIModels):
    video_path = 'input_videos/input_video.mp4'
    try:        
        # Validate input
        schema = VideoRequestSchema()
        data = event['input']['data']
        errors = schema.validate(data)
        if errors:
            raise ValidationError({"Failed to validate ball based sectioning request": errors})
        # Download and process
        download_file_from_firebase(data['video_url'], video_path)
        print("analysing video")
        sections, ball_detections_for_sections = ball_based_sectioning(
            video_id=data['video_id'],
            video_path=video_path,
            tracknet_ball_tracker=ai_models.tracknet_ball_tracker,
            chunk_size=1000, 
            make_request=True,
        )

        return ball_detections_for_sections

    except Exception as e:
        raise Exception({"Failed ball based sectioning analysis": str(e)})

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

def handle_court_keypoints_for_sections(event, ai_models: AIModels):
    video_path = 'input_videos/input_video.mp4'
    try:        
        # Validate input
        schema = VideoRequestSchema()
        data = event['input']['data']
        errors = schema.validate(data)
        if errors:
            raise ValidationError({"Failed to validate court keypoints request": errors})

        # Download and process
        download_file_from_firebase(data['video_url'], video_path)
        print("analysing video")
        court_keypoints_for_sections = court_keypoints_by_time_step(
            video_id=data['video_id'],
            video_path=video_path,
            court_line_detector=ai_models.court_line_detector,
            letr_court_line_detector=ai_models.letr_court_line_detector,
            make_request=True,
        )

        return court_keypoints_for_sections

    except Exception as e:
        raise Exception({"Failed court keypoints for sections analysis": str(e)})

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

def handle_player_stubs_for_sections(event, ai_models: AIModels):
    video_path = 'input_videos/input_video.mp4'
    try:        
        # Validate input
        schema = PlayerStubsRequest()
        data = event['input']['data']
        errors = schema.validate(data)
        if errors:
            raise ValidationError({"Failed to validate player stubs request": errors})

        # Download and process
        download_file_from_firebase(data['video_url'], video_path)
        print("analysing video")
        player_detections_for_sections = player_stubs_for_sections(
            video_id=data['video_id'],
            video_path=video_path,
            player_tracker=ai_models.player_tracker,
            all_court_keypoints=data['all_court_keypoints'],
            sections=data['sections'],
            make_request=True,
        )
        return player_detections_for_sections

    except Exception as e:
        raise Exception({"Failed player stubs for sections analysis": str(e)})

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

def handle_player_speeds_for_sections(event, ai_models: AIModels):
    try:        
        # Validate input
        schema = PlayerSpeedsRequest()
        data = event['input']['data']
        errors = schema.validate(data)
        if errors:
            raise ValidationError({"Failed to validate player speeds request": errors})
        mini_court_detections = convert_detections_to_mini_court_coordinates(
            mini_court=ai_models.mini_court,
            sectioned_data=data['sectioned_data'],
        )
    
        player_speeds_for_sections = player_speed_for_section(
            video_id=data['video_id'],
            player_mini_court_detections_for_sections=mini_court_detections['player_mini_court_detections_for_sections'],
            mini_court=ai_models.mini_court,
            video_info=data['video_info'],
            make_request=True,
        )
        return player_speeds_for_sections

    except Exception as e:
        raise Exception({"Failed player speeds for sections analysis": str(e)})

def handle_ball_speeds_for_sections(event, ai_models: AIModels):
    try:        
        # Validate input
        schema = BallSpeedsRequest()
        data = event['input']['data']
        errors = schema.validate(data)
        if errors:
            raise ValidationError({"Failed to validate ball speeds request": errors})
        
        mini_court_detections = convert_detections_to_mini_court_coordinates(
            mini_court=ai_models.mini_court,
            sectioned_data=data['sectioned_data'],
        )
        ball_speeds_for_sections = ball_speed_for_section(
            video_id=data['video_id'],
            ball_mini_court_detections_for_sections=mini_court_detections['ball_mini_court_detections_for_sections'],
            sections=data['sections'],
            mini_court=ai_models.mini_court,
            video_info=data['video_info'],
            make_request=True,
        )

        return ball_speeds_for_sections

    except Exception as e:
        raise Exception({"Failed ball speeds for sections analysis": str(e)})


def handle_scoreboard_changes(event, ai_models: AIModels):
    video_path = 'input_videos/input_video.mp4'
    try:        
        # Validate input
        schema = VideoRequestSchema()
        data = event['input']['data']
        errors = schema.validate(data)
        if errors:
            raise ValidationError({"Failed to validate scoreboard changes request": errors})

        # Download and process
        download_file_from_firebase(data['video_url'], video_path)
        
        scoreboard_changes = scoreboard_service(
            video_id=data['video_id'],
            video_path=video_path,
            scoreboard_model=ai_models.scoreboard_model,
            chunk_size=1000,
            read_from_stub=False,
            save_to_stub=False,
            make_request=True)

        return scoreboard_changes

    except Exception as e:
        raise Exception({"Failed scoreboard changes analysis": str(e)})

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


def handler(event, path, ai_models: AIModels):
    if path == 'process_video':
        return handle_process_video(event, ai_models)
    # elif path == 'ball_based_sectioning':
    #     return handle_ball_based_sectioning(event, ai_models)
    # elif path == 'court_keypoints_for_sections':
    #     return handle_court_keypoints_for_sections(event, ai_models)
    # elif path == 'player_stubs_for_sections':
    #     return handle_player_stubs_for_sections(event, ai_models)
    # elif path == 'player_speeds_for_sections':
    #     return handle_player_speeds_for_sections(event, ai_models)
    # elif path == 'ball_speeds_for_sections':
    #     return handle_ball_speeds_for_sections(event, ai_models)
    # elif path == 'scoreboard_changes':
    #     return handle_scoreboard_changes(event, ai_models)
    else:
        return {"error": f"Unsupported task: {path}"}

        
