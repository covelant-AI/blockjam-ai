from ai_core.racket_player_ball_detector.core import RacketPlayerBallDetector
from utils.video_utils import get_video_info, get_every_nth_frame_in_range
import requests
import os
import math

def suppress_short_true_sequences(data, min_true_time, time_per_value):
    """
    Replace sequences of True values shorter than min_true_length with False.

    Parameters:
    - data: list of bools
    - min_true_time: int, minimum time of contiguous True values to keep
    - time_per_value: int, number of seconds each data point represents

    Returns:
    - List of bools with short True sequences suppressed
    """
    modified_data = data.copy()
    i = 0
    min_true_length = min_true_time // time_per_value
    while i < len(modified_data):
        if modified_data[i]:
            start = i
            while i < len(modified_data) and modified_data[i]:
                i += 1
            end = i  # exclusive
            if end - start < min_true_length:
                for j in range(start, end):
                    modified_data[j] = False
        else:
            i += 1
    return modified_data

def find_false_sequences(data, min_seconds, time_per_value, fps):
    """
    Identify sequences of False values longer than min_seconds,
    given each data point represents time_per_value seconds.

    Parameters:
    - data: list of bools
    - min_seconds: int, minimum length in seconds of False sequence to consider
    - time_per_value: int, number of seconds each data point represents

    Returns:
    - List of tuples (start_index, end_index) for False sequences
    - Prints readable time ranges and total time saved
    """
    false_sequences = []
    in_sequence = False
    start_index = -1

    min_length = min_seconds // time_per_value

    for i, value in enumerate(data):
        if not value and not in_sequence:
            in_sequence = True
            start_index = i
        elif value and in_sequence:
            in_sequence = False
            if i - start_index >= min_length:
                false_sequences.append((start_index, i - 1))
            start_index = -1

    # Handle case where sequence goes to end of list
    if in_sequence and len(data) - start_index >= min_length:
        false_sequences.append((start_index, len(data) - 1))

    return false_sequences

def get_active_sequences(total_frames, deadtime_tuples, time_per_value, fps):
    """
    Given the length of the original data and a list of deadtime tuples (start, end),
    return a list of active time tuples (start, end).

    Parameters:
    - data_length: int, total length of the data list
    - total_frames: int, total number of frames in the video

    Returns:
    - List of dicts with start and end frames and times
    """
    active_sequences = []
    current_index = 0

    deadtime_tuples = [(start * time_per_value * fps, end * time_per_value * fps) for start, end in deadtime_tuples]

    def format_sequence(start, end):
        return {
            'start': {
                'index': start,
                'time': start / fps
            },
            'end': {
                'index': end,
                'time': end / fps
            }
        }

    for start, end in deadtime_tuples:
        if current_index < start:
            active_sequences.append(format_sequence(current_index, start - 1))
        current_index = end + 1
    
    if current_index < total_frames:
        active_sequences.append(format_sequence(current_index, total_frames))
    return active_sequences



def deadtime_removal(video_id, video_path, racket_player_ball_detector:RacketPlayerBallDetector, step_time=2, deadtime_min_seconds=10, non_deadtime_min_seconds=10, chunk_size=500, make_request=True, webhook_path='/ai_analysis/deadtime_removal'):
    video_info = get_video_info(video_path)
    
    frame_step_size = int(video_info["fps"] * step_time)
    total_frames_processed = math.ceil(video_info['total_frames'] / frame_step_size)
    chunks = math.ceil(total_frames_processed / chunk_size)

    data = []
    for chunk_idx in range(chunks):
        start_frame = int(chunk_idx * chunk_size * frame_step_size)
        end_frame = int((chunk_idx + 1) * chunk_size * frame_step_size)
        chunk_frames = get_every_nth_frame_in_range(video_path, start_frame, end_frame, frame_step_size)
        chunk_data = racket_player_ball_detector.detect_frames(chunk_frames)
        data.extend(chunk_data)
        
    is_valid_frames = []
    for frame_data in data:
        is_valid = len(frame_data['players']) > 0 or len(frame_data['rackets']) > 0
        is_valid_frames.append(is_valid)

    filtered_is_valid_frames = suppress_short_true_sequences(is_valid_frames, non_deadtime_min_seconds, step_time)
    deadtimes = find_false_sequences(filtered_is_valid_frames, deadtime_min_seconds, step_time, video_info['fps'])
    active_sequences = get_active_sequences(video_info['total_frames'], deadtimes, step_time, video_info['fps'])
    

    if make_request:
        request_data = {
            'video_id': video_id,
            'data': active_sequences
        }
        response = requests.post(os.getenv('BACKEND_URL')+ webhook_path, json=request_data)
        if response.status_code != 200:
            raise Exception(f"Failed to send request to {webhook_path}: {response.text}")

    return active_sequences