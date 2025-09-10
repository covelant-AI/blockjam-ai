from ai_core.racket_player_ball_detector.core import RacketPlayerBallDetector
import cv2
from utils.bucket import download_file_from_firebase
from utils.video_utils import get_video_info, total_chunks, get_frame_at_index


def suppress_short_true_sequences(data, min_true_length=3):
    """
    Replace sequences of True values shorter than min_true_length with False.

    Parameters:
    - data: list of bools
    - min_true_length: int, minimum length of contiguous True values to keep

    Returns:
    - List of bools with short True sequences suppressed
    """
    modified_data = data.copy()
    i = 0
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

def find_false_sequences(data, min_seconds, time_per_value):
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

def racket_player_ball_detector_playground(event, step_time=2, deadtime_min=30, non_deadtime_min=5):
    video_path = 'input_videos/PTTKrakow.mp4'
    # data = event['input']['data']
    # download_file_from_firebase(data['video_url'], video_path)
    video_info = get_video_info(video_path)

    detector = RacketPlayerBallDetector("models/racket_player_ball_yolo11m.pt", "cuda")
    
    frame_step_size = video_info["fps"] * step_time
    iters = total_chunks(video_path, frame_step_size)
    
    is_valid_frames = []
    for i in range(iters):
        frame_idx = i * frame_step_size
        frame = get_frame_at_index(video_path, frame_idx)
        players, rackets, balls = detector.detect_frame(frame)
        is_valid = len(players) > 0 or len(rackets) > 0
        is_valid_frames.append(is_valid)

    is_valid_frames = suppress_short_true_sequences(is_valid_frames, non_deadtime_min)
    deadtimes = find_false_sequences(is_valid_frames, deadtime_min, step_time)
    return deadtimes