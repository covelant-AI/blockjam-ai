from utils.video_utils import get_video_info
import requests
from ai_core.player_stats.core import get_ball_speed_data
import os
from schemas.ball_speed import BallSpeedsForSection
from ai_core.mini_court import MiniCourt
from utils.bbox_utils import measure_distance, get_center_of_bbox, get_foot_position
import numpy as np
from scipy.signal import find_peaks
from typing import Literal
from utils.ball_speed import find_initial_speed_and_angle
from schemas.stubs import SectionedStubsData
from ai_core.tracknet_ball_tracker import TrackNetBallTracker
from utils.simple import find_object_by_key_value
import json
from services.analysis.estiamte_ball_3d import estimate_3d_position

def ball_hit_detection(ball_detections, player_detections, court_keypoints, distance_threshold=100):
    np_array = np.array(ball_detections)
    minimas, _ = find_peaks(-np_array[:, 1], distance=30, prominence=10)
    maximas, _ = find_peaks(np_array[:, 1], distance=30, prominence=10)
    middle_of_screen = int((court_keypoints[12][1] + court_keypoints[13][1]) / 2)

    def is_hit(data, player_type: Literal["1", "2"], player_detections, court_keypoints):
        hits = []
        for i in data:
            try:
                if (player_type == "1" and np_array[i][1] < (middle_of_screen + 200)) or (player_type == "2" and np_array[i][1] > (middle_of_screen - 200)):
                    player = player_detections[i][player_type]
                    distance = measure_distance(get_center_of_bbox(player), ball_detections[i])
                    if distance < distance_threshold:
                        hits.append(int(i))
            except:
                pass
        return hits

    p1_hits = is_hit(minimas, "1", player_detections, court_keypoints)
    p2_hits = is_hit(maximas, "2", player_detections, court_keypoints)
    return p1_hits, p2_hits



def ball_bounce_hit_speed(hit_indexes, ball_detections, selected_player_detections, court_keypoints, mini_court: MiniCourt, ball_bounce_indexes: list[int], video_info, min_speed=70, max_speed=250, max_time_between_bounces=1.5, min_time_between_bounces=0.25):
    
    hits_speeds = []
    min_frames_between_bounces = int(min_time_between_bounces * video_info['fps'])
    max_frames_between_bounces = int(max_time_between_bounces * video_info['fps'])

    for i in hit_indexes:
        coord = ball_detections[i]
        if selected_player_detections[i] is None:
            continue
        foot_coord = get_foot_position(selected_player_detections[i])
        ball_floor_coord = [coord[0], foot_coord[1]]
        ball_floor_coord = mini_court.convert_screen_to_mini_court_coordinates(ball_floor_coord, court_keypoints)

        def isBounceTimeValid(bounce_index):
            return bounce_index > i + min_frames_between_bounces and bounce_index < i + max_frames_between_bounces
        
        next_bounce = next((bounce for bounce in ball_bounce_indexes if isBounceTimeValid(bounce)), None)
        
        if next_bounce is not None:
            next_bounce_coord = ball_detections[next_bounce]
            next_bounce_coord = mini_court.convert_screen_to_mini_court_coordinates(next_bounce_coord, court_keypoints)

            distance = mini_court.get_distance_in_meters(ball_floor_coord, next_bounce_coord)
            time = (next_bounce - i) / video_info['fps'] # in seconds
            speed, angle = find_initial_speed_and_angle(1.5, float(time), float(distance))
            speed = float(speed * 3.6) # kmph - convert to regular Python float
            if speed > min_speed and speed < max_speed:
                hits_speeds.append({
                    "index": int(i),
                    "seconds": float(i / video_info['fps']),
                    "speed": float(speed),
                })

    return hits_speeds


def instant_hit_speed(hit_indexes, ball_detections, selected_player_detections, mini_court: MiniCourt, court_keypoints, video_info, min_speed=70, max_speed=250):
    hits_speeds = []
    for i in hit_indexes:
        coord = ball_detections[i]
        if selected_player_detections[i] is None:
            continue
        foot_coord = get_foot_position(selected_player_detections[i])
        ball_floor_coord = [coord[0], foot_coord[1]]
        ball_floor_mini_court_coord = mini_court.convert_screen_to_mini_court_coordinates(ball_floor_coord, court_keypoints)

        dx = mini_court.convert_screen_to_mini_court_coordinates([ball_floor_coord[0] + 1, ball_floor_coord[1]], court_keypoints)
        dx_distance = mini_court.get_distance_in_meters(ball_floor_mini_court_coord, dx)
        dy = mini_court.convert_screen_to_mini_court_coordinates([ball_floor_coord[0], ball_floor_coord[1] + 1], court_keypoints)
        dy_distance = mini_court.get_distance_in_meters(ball_floor_mini_court_coord, dy)
        delta_dist = (dx_distance + dy_distance)/2
        
        pixel_distances = []
        for j in range(1, 5):
            try:
                dist = measure_distance(ball_detections[i + j], ball_detections[i + j + 1])
                pixel_distances.append(dist)
            except:
                pass
        
        median_distance = float(np.median(pixel_distances))  # convert to regular Python float

        
        #calculate speed
        time_difference = 1 / video_info['fps']
        x_speed = float((median_distance * delta_dist * 3.6) / time_difference)  # kmph - convert to regular Python float

        if x_speed > min_speed and x_speed < max_speed:
            hits_speeds.append({
                "index": int(i),
                "seconds": float(i / video_info['fps']),
                "speed": float(x_speed),
            })

    return hits_speeds


def combine_hits_speeds(hit_indexes, bounce_speeds, instant_speeds):
    hits_speeds = []
    for hit_idx in hit_indexes:
        if find_object_by_key_value(bounce_speeds, "index", hit_idx) is not None:
            hits_speeds.append(find_object_by_key_value(bounce_speeds, "index", hit_idx))
        elif find_object_by_key_value(instant_speeds, "index", hit_idx) is not None:
            hits_speeds.append(find_object_by_key_value(instant_speeds, "index", hit_idx))
    return hits_speeds

def player_hits_ball_speed(
        video_id,
        sectioned_data:list[SectionedStubsData],
        mini_court:MiniCourt,
        tracknet_ball_tracker:TrackNetBallTracker,
        video_info:dict,
        webhook_path='/ai_analysis/player_hits_ball_speeds',
        make_request=True,
        ):
    all_hits_speeds = []

    # # # #Convert sectioned_data to json #TODO: I hate that works. Fix sectioned_data so it works without json parsing
    sectioned_data_json = json.dumps(sectioned_data)
    sectioned_data = json.loads(sectioned_data_json)

    for _s in sectioned_data:
        section = _s['section']
        court_keypoints = _s['court_keypoints']
        ball_detections = _s['ball_detections']
        player_detections = _s['player_detections']
        ball_bounces = _s['ball_bounces']
        ball_bounce_indexes = [bounce['index'] for bounce in ball_bounces]

        interpolated_ball_detections = tracknet_ball_tracker.interpolate_ball_track(ball_detections)

        p1_hits, p2_hits = ball_hit_detection(interpolated_ball_detections, player_detections, court_keypoints)
        if player_detections is not None:
            p1_detections = [player_detection["1"] if "1" in player_detection else None for player_detection in player_detections]
            p1_hits_bounce_speeds = ball_bounce_hit_speed(p1_hits, interpolated_ball_detections, p1_detections, court_keypoints, mini_court, ball_bounce_indexes, video_info)
            p1_instant_speeds = instant_hit_speed(p1_hits, interpolated_ball_detections, p1_detections, mini_court, court_keypoints, video_info)
            p1_hits_speeds = combine_hits_speeds(p1_hits, p1_hits_bounce_speeds, p1_instant_speeds)

            p2_detections = [player_detection["2"] if "2" in player_detection else None for player_detection in player_detections]
            p2_hits_bounce_speeds = ball_bounce_hit_speed(p2_hits, interpolated_ball_detections, p2_detections, court_keypoints, mini_court, ball_bounce_indexes, video_info)
            p2_instant_speeds = instant_hit_speed(p2_hits, interpolated_ball_detections, p2_detections, mini_court, court_keypoints, video_info)
            p2_hits_speeds = combine_hits_speeds(p2_hits, p2_hits_bounce_speeds, p2_instant_speeds)

            all_hits_speeds.append({
                'section': section,
                'p1': p1_hits_speeds,
                'p2': p2_hits_speeds,
            })
        else:
            all_hits_speeds.append({
                'section': section,
                'p1': p1_hits_speeds,
                'p2': p2_hits_speeds,
            })

    if make_request:
        response = requests.post(os.getenv('BACKEND_URL')+webhook_path, json={
            'video_id': video_id,
            'data': all_hits_speeds,
        })
        if response.status_code != 200:
            raise Exception(f"Failed to handle player hits ball speed: {response.text}")
    return all_hits_speeds

def ball_speed_for_section(
        video_id,
        sectioned_data: list[SectionedStubsData],
        video_info: dict,
        time_step=0.5,
        make_request=False,
        webhook_path='/ai_analysis/ball_speeds'
        ) -> list[BallSpeedsForSection]:
    
    ball_speeds_for_sections = []
    error_sections = []
    for _s in sectioned_data:
        section = _s['section']
        ball_detections = _s['ball_detections']
        court_keypoints = _s['court_keypoints']

        ball_3d_positions = estimate_3d_position(ball_detections, court_keypoints)

        try:
            print(f"Generating ball speed for frames {section['start']['index']} to {section['end']['index']}")
            ball_speeds = get_ball_speed_data(ball_3d_positions, video_info['fps'], time_step)
            ball_speeds_for_sections.append({
                'section': section,
                'speeds': ball_speeds
            })
        except Exception as e:
            print(f"Error generating ball speed for section starting at {section['start']['index']}: {e}")
            error_sections.append({
                'section': section,
                'message': str(e)
            })

    if make_request:
        response = requests.post(os.getenv('BACKEND_URL')+webhook_path, json={
            'video_id': video_id,
            'time_step': time_step,
            'data': ball_speeds_for_sections,
            'error_sections': error_sections,
        })
        if response.status_code != 200:
            raise Exception(f"Failed to handle player speed: {response.text}")

        return ball_speeds_for_sections
