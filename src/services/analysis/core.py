from utils.video_utils import get_video_info
from utils.conversions import frame_to_time
import time
from .court_keypoints import court_keypoints_by_time_step
from ai_core.court_line_detector import CourtLineDetector
from ai_core.letr_court_line_detector import LETRCourtDetector
from .tracking import tracking
from ai_core.trackers.core import BallAndPlayerTracker
from .ball_based_sectioning import get_sections_fram_ball_detections
import requests
import os
from ai_core.player_stats.core import smooth_speed_data

def analyze_video(
        video_id,
        video_path,
        ball_and_player_tracker: BallAndPlayerTracker,
        court_line_detector: CourtLineDetector,
        letr_court_line_detector: LETRCourtDetector,
        speed_time_step=0.25,
        chunk_size=1000, 
        ):
    video_info = get_video_info(video_path)
    timing_results = {}

    start_time = time.time()
    all_court_keypoints = court_keypoints_by_time_step(video_id, video_path, court_line_detector, letr_court_line_detector, video_info=video_info, make_request=False) #TODO: Make Request when Next Server is ready
    timing_results['court_keypoints'] = time.time() - start_time

    start_time = time.time()
    ball_detections, ball_speeds, p1_detections, p1_speeds, p2_detections, p2_speeds = tracking(video_path, ball_and_player_tracker, all_court_keypoints, video_info, chunk_size=chunk_size)
    timing_results['tracking'] = time.time() - start_time    
    
    sections = get_sections_fram_ball_detections(ball_detections, ball_speeds, video_info)
    base_url = os.getenv('BACKEND_URL')
    section_response = requests.post(base_url+'/ai_analysis/sections', json={
        'video_id': video_id,
        'data': sections,
    })
    if section_response.status_code != 200:
        raise Exception(f"Failed to make request 'sections': {section_response.text}")
    
    whole_video_section = {
        "start": {
            "index": 0,
            "time": frame_to_time(0, video_info['fps'])
        },
        "end": {
            "index": video_info['total_frames'],
            "time": frame_to_time(video_info['total_frames'], video_info['fps'])
        },
    }
    
    ball_speeds = smooth_speed_data(ball_speeds, window_size=3, alpha=0.3, fps=video_info['fps'], time_step=speed_time_step)
    ball_speed_response = requests.post(base_url+'/ai_analysis/ball_speeds', json={
        'video_id': video_id,
        'time_step': speed_time_step,
        'data': [
                {
                'section': whole_video_section,
                'speeds': ball_speeds,
            }
        ],
    })
    if ball_speed_response.status_code != 200:
        raise Exception(f"Failed to make request 'ball_speeds': {ball_speed_response.text}")
    
    p1_speeds = smooth_speed_data(p1_speeds, window_size=3, alpha=0.3, fps=video_info['fps'], time_step=speed_time_step)
    p2_speeds = smooth_speed_data(p2_speeds, window_size=3, alpha=0.3, fps=video_info['fps'], time_step=speed_time_step)
    player_speed_response = requests.post(base_url+'/ai_analysis/player_speeds', json={
        'video_id': video_id,
        'time_step': speed_time_step,
        'data': [{
            'section': whole_video_section,
            'data': {
                'p1_speeds': p1_speeds,
                'p2_speeds': p2_speeds
            },
        }],
    })
    if player_speed_response.status_code != 200:
        raise Exception(f"Failed to make request 'player_speeds': {player_speed_response.text}")

    return timing_results