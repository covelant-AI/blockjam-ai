from ai_core.trackers.core import BallAndPlayerTracker
from utils.video_utils import read_video_range, get_video_info, total_chunks
from services.analysis.court_keypoints import default_time_step as court_keypoints_time_step
from utils.conversions import frame_to_time
from ai_core.player_stats.core import smooth_speed_data
import requests
import os
import numpy as np

def make_speeds_request(video_id, ball_and_player_tracker: BallAndPlayerTracker, req_world_tracks, start_index, end_index, fps, speed_time_step):
    ball_detections, ball_speeds, p1_detections, p1_speeds, p2_detections, p2_speeds = ball_and_player_tracker.extract_from_world_tracks(req_world_tracks)
    section = {
        "start": {
            "index": start_index,
            "time": frame_to_time(start_index, fps)
        },
        "end": {
            "index": end_index,
            "time": frame_to_time(end_index, fps)
        },
    }

    base_url = os.getenv('BACKEND_URL')
    ball_speeds = smooth_speed_data(ball_speeds, alpha=0.3, fps=fps, time_step=speed_time_step)
    if ball_speeds is not None:
        ball_speed_response = requests.post(base_url+'/ai_analysis/ball_speeds', json={
            'video_id': video_id,
            'time_step': speed_time_step,
            'data': [
                    {
                    'section': section,
                    'speeds': ball_speeds,
                }
            ],
            "error_sections": []
        })
        if ball_speed_response.status_code != 200:
            raise Exception(f"Failed to make request 'ball_speeds': {ball_speed_response.text}")
        
    p1_speeds = smooth_speed_data(p1_speeds, alpha=0.3, fps=fps, time_step=speed_time_step)
    p2_speeds = smooth_speed_data(p2_speeds, alpha=0.3, fps=fps, time_step=speed_time_step)
    if p1_speeds is not None and p2_speeds is not None:
        player_speed_response = requests.post(base_url+'/ai_analysis/player_speeds', json={
            'video_id': video_id,
            'time_step': speed_time_step,
            'data': [{
                'section': section,
                'p1_speeds': p1_speeds,
                'p2_speeds': p2_speeds
            }],
            "error_sections": []
        })
        if player_speed_response.status_code != 200:
            raise Exception(f"Failed to make request 'player_speeds': {player_speed_response.text}")

def tracking(
        video_id,
        video_path,
        ball_and_player_tracker: BallAndPlayerTracker,
        all_court_keypoints: list[dict],
        video_info,
        chunk_size=1000,
        make_request_every_n_chunks=10,
        speed_time_step=0.5
    ) -> tuple[list[tuple[float, float]], list[float], list[tuple[float, float]], list[float], list[tuple[float, float]], list[float]]:

    def _get_polygon(court_keypoints):
        return np.array([
            court_keypoints[0], # TL
            court_keypoints[1], # TR
            court_keypoints[3], # BR
            court_keypoints[2], # BL
        ])

    world_tracks = []
    req_world_tracks = []
    all_polygons = [_get_polygon(court_keypoints) for court_keypoints in all_court_keypoints]
    ball_and_player_tracker.define_trackers_using_polygons(all_polygons[0], video_info["fps"])

    total_num_chunks = total_chunks(video_path, chunk_size)
    for chunk_index in range(total_num_chunks):
        start_index = chunk_index * chunk_size
        end_index = start_index + chunk_size

        frames = read_video_range(video_path, start_index, end_index)
        wt = ball_and_player_tracker.detect_frames(frames, start_index, all_polygons, update_court_polygon_interval=int(court_keypoints_time_step*video_info["fps"])) 
        world_tracks.extend(wt)
        req_world_tracks.extend(wt)
        if chunk_index % make_request_every_n_chunks == 0 and chunk_index != 0:
            end_index = start_index + len(frames)
            start_index = end_index-len(req_world_tracks)
            make_speeds_request(video_id, ball_and_player_tracker, req_world_tracks, start_index, end_index, video_info["fps"], speed_time_step)
            req_world_tracks = []
    
    if len(req_world_tracks) > 0:
        end_index = video_info["total_frames"]
        start_index = end_index - len(req_world_tracks)
        make_speeds_request(video_id, ball_and_player_tracker, req_world_tracks, start_index, end_index, video_info["fps"], speed_time_step)

    return ball_and_player_tracker.extract_from_world_tracks(world_tracks)



    