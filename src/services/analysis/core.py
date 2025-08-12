from utils.video_utils import get_video_info
import time
from .ball_based_sectioning import ball_based_sectioning
from .player_stubs import player_stubs_for_sections
from .player_speed import player_speed_for_section
from .ball_speed import ball_speed_for_section
from utils.conversions import convert_detections_to_mini_court_coordinates
from .court_keypoints import court_keypoints_by_time_step, get_keypoint_closest_to_section
from utils.simple import find_object_by_key_value
from ai_core.racket_player_ball_detector import RacketPlayerBallDetector
from ai_core.player_tracker import PlayerTracker
from ai_core.tracknet_ball_tracker import TrackNetBallTracker
from ai_core.court_line_detector import CourtLineDetector
from ai_core.letr_court_line_detector import LETRCourtDetector
from ai_core.ball_bounce_model import BallBounceModel
from ai_core.scoreboard import Scoreboard
from ai_core.mini_court import MiniCourt
from .shot_type import classify_player_shots
from .ball_bounce import ball_bounces_for_sections
from .deadtime import deadtime_removal
from schemas.video import Feature
from typing import List
from ai_core.shot_type_classifier import ShotTypeClassifier

def analyze_video(
        video_id,
        video_path,
        features: List[Feature],  
        player_tracker: PlayerTracker,
        tracknet_ball_tracker: TrackNetBallTracker,
        racket_player_ball_detector: RacketPlayerBallDetector,
        court_line_detector: CourtLineDetector,
        letr_court_line_detector: LETRCourtDetector,
        shot_type_classifier: ShotTypeClassifier,
        ball_bounce_model: BallBounceModel,
        scoreboard_model: Scoreboard,
        mini_court: MiniCourt,
        chunk_size=1000, 
        read_from_stub=False,
        save_to_stub=False,
        make_request=True,
        ):
    video_info = get_video_info(video_path)
    timing_results = {}

    active_sequences = []
    if Feature.DEAD_TIME_DETECTION in features:
    # Step 2: Deadtime removal
        start_time = time.time()
        active_sequences = deadtime_removal(video_id, video_path, racket_player_ball_detector, step_time=2, deadtime_min_seconds=10, non_deadtime_min_seconds=10, make_request=make_request)
        timing_results['deadtime_removal'] = time.time() - start_time

    sections = []
    ball_detections_for_sections = []
    if Feature.MATCH_SECTIONING in features:
        # Step 1: Court keypoints
        start_time = time.time()
        all_court_keypoints = court_keypoints_by_time_step(video_id, video_path, court_line_detector, letr_court_line_detector, video_info=video_info, make_request=False) #TODO: Make Request when Next Server is ready
        timing_results['court_keypoints'] = time.time() - start_time

        # Step 3: Ball-based sectioning
        start_time = time.time()
        sections, ball_detections_for_sections = ball_based_sectioning(video_id, video_path, tracknet_ball_tracker, active_sequences, chunk_size=chunk_size, video_info=video_info, make_video_info_request=make_request, make_sections_request=make_request, make_ball_detections_request=False, read_from_stub=False, save_to_stub=save_to_stub) #TODO: Make Request when Next Server is ready
        timing_results['ball_based_sectioning'] = time.time() - start_time

    player_detections_for_sections = []
    if Feature.TRACK_PLAYER_SPEED in features:
        # Step 4: Player detection for sections
        start_time = time.time()
        player_detections_for_sections = player_stubs_for_sections(video_id, video_path, player_tracker, all_court_keypoints, sections, video_info=video_info, make_request=False, read_from_stub=False, save_to_stub=save_to_stub) #TODO: Make Request when Next Server is ready
        timing_results['player_detection'] = time.time() - start_time

    # Step 5: Mini court coordinate conversion
    start_time = time.time()
    sectioned_data = []
    for section in sections:
        player_detections_section = find_object_by_key_value(player_detections_for_sections, 'section', section)
        ball_detections_section = find_object_by_key_value(ball_detections_for_sections, 'section', section)
        
        sectioned_data.append({
            'section': section,
            'court_keypoints': get_keypoint_closest_to_section(all_court_keypoints, section),
            'player_detections': player_detections_section['data'] if player_detections_section is not None else None,
            'ball_detections': ball_detections_section['data'] if ball_detections_section is not None else None
        })
    

    mini_court_detections = convert_detections_to_mini_court_coordinates(mini_court, sectioned_data)
    player_mini_court_detections_for_sections = mini_court_detections['player_mini_court_detections_for_sections']
    ball_mini_court_detections_for_sections = mini_court_detections['ball_mini_court_detections_for_sections']
    timing_results['mini_court_conversion'] = time.time() - start_time

    if Feature.TRACK_BALL_SPEED in features:
        # Step 7: Ball speed analysis
        start_time = time.time()
        ball_speed_for_section(video_id, sectioned_data, video_info, make_request=make_request)
        timing_results['ball_speed_analysis'] = time.time() - start_time

    

    if Feature.TRACK_PLAYER_SPEED in features:
        # Step 8: Player speed analysis
        start_time = time.time()
        player_speeds_for_sections = player_speed_for_section(video_id, player_mini_court_detections_for_sections, mini_court, video_info=video_info, make_request=make_request)
        timing_results['player_speed_analysis'] = time.time() - start_time

    if Feature.CLASSIFY_SHOT_TYPE in features:
        # Step 9: Shot type classification
        start_time = time.time()
        classify_player_shots(video_id, video_path, sectioned_data, shot_type_classifier, make_request=make_request)
        timing_results['shot_type_classification'] = time.time() - start_time

    # Calculate total time
    timing_results['total_time'] = sum(timing_results.values())
    
    return timing_results
