from schemas.stubs import SectionedStubsData
from ai_core.ball_bounce_model import BallBounceModel
from ai_core.mini_court import MiniCourt
import os
import requests
from ai_core.tracknet_ball_tracker import TrackNetBallTracker

def ball_bounces_for_sections(
        video_id,
        sectioned_data:list[SectionedStubsData], 
        ball_bounce_model:BallBounceModel,
        tracknet_ball_tracker:TrackNetBallTracker,
        mini_court:MiniCourt,
        video_info:dict,
        make_request:bool=True,
        webhook_path='/ai_analysis/ball_bounces',
        ):
    ball_bounces_for_sections = []
    for _s in sectioned_data:
        section = _s['section']
        ball_detections = _s['ball_detections']
        court_keypoints = _s['court_keypoints']
        interpolated_ball_detections = tracknet_ball_tracker.interpolate_ball_track(ball_detections)
        ball_bounce_df = ball_bounce_model.detections_to_df(interpolated_ball_detections, video_info)
        predictions = ball_bounce_model.predict_df(ball_bounce_df)
        ball_bounce_indexes = ball_bounce_model.detect_bounce_frames(predictions, video_info)

        ball_bounces = []
        for ball_bounce_index in ball_bounce_indexes:
            coord = interpolated_ball_detections[ball_bounce_index]
            ball_bounce_mini_court_coordinates = mini_court.convert_screen_to_mini_court_coordinates(coord, court_keypoints)
            ball_bounces.append({
                'index': int(ball_bounce_index),
                "seconds": float(ball_bounce_index / video_info['fps']),
                'mini_court_coordinates': ball_bounce_mini_court_coordinates,
            })

        _s['ball_bounces'] = ball_bounces

        ball_bounces_for_sections.append({
            'section': section,
            'data': ball_bounces,
        })

    if make_request:
        print(f"Making request to backend")
        response = requests.post(os.getenv('BACKEND_URL')+webhook_path, json={
            'video_id': video_id,
            'data': ball_bounces_for_sections,
        })
        if response.status_code != 200:
            raise Exception(f"Failed to make request 'ball_bounces': {response.text}")
    return ball_bounces
