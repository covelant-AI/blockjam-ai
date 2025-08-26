from utils.video_utils import total_chunks, read_video_chunk, get_video_info
import requests
import os
from utils.stubs import load_detections_from_stub, save_detections_to_stub
from utils.bbox_utils import measure_distance
import time
from utils.conversions import frame_to_time
from services.analysis.ball_stubs import ball_stubs_for_whole_video
from ai_core.ball_tracker import BallTracker
import numpy as np

def get_sections_fram_ball_detections(
        ball_detections,
        ball_speeds, 
        video_info,
        min_section_time=1.5,
        section_null_threshold=0.5,
        min_time_between_sections=1,
        buffer_time_before_section=3,
        buffer_time_after_section=1,
        max_gap_time=1,
        min_ball_speed_kmh=20,
        ):
    print(f"Getting sections from ball detections")
    max_gap_size = max_gap_time * video_info['fps']
    min_section_size = min_section_time * video_info['fps']
    min_time_between_sections_size = min_time_between_sections * video_info['fps']
    section_buffer_start_size = buffer_time_before_section * video_info['fps']
    section_buffer_end_size = buffer_time_after_section * video_info['fps']
    sections = []
    start = None
    empty_frames = 0

    def create_or_extend_section(start_idx, end_idx):
        """Helper function to create a new section or extend the previous one"""
        last_section_end = sections[-1]['end']['index'] if sections else None
        if last_section_end and start_idx - last_section_end < min_time_between_sections_size:
            # Extend previous section
            end_index = int(min(end_idx + section_buffer_end_size, len(ball_detections)))
            sections[-1]['end'] = {
                "index": end_index,
                "time": frame_to_time(end_index, video_info['fps'])
            }
        else:
            # Create new section
            start_index = int(max(start_idx - section_buffer_start_size, 0))
            end_index = int(min(end_idx + section_buffer_end_size, len(ball_detections)))
            sections.append({
                "start": {
                    "index": start_index,
                    "time": frame_to_time(start_index, video_info['fps'])
                },
                "end": {
                    "index": end_index,
                    "time": frame_to_time(end_index, video_info['fps'])
                }
            })

    for i in range(len(ball_detections)):
        current_detection = tuple(ball_detections[i])
        is_empty = current_detection == (None, None)
        
        # Check pixel distance from previous detection if not empty
        if not is_empty and i > 0:
            if ball_speeds[i] is not None and ball_speeds[i] < min_ball_speed_kmh:
                is_empty = True
        
        if not is_empty and start is None:
            start = i
        elif is_empty and start is not None:
            empty_frames += 1
            frame_diff = i - start
            null_frame_ratio = empty_frames / frame_diff
            if empty_frames > max_gap_size:
                if frame_diff > min_section_size and null_frame_ratio < section_null_threshold:
                    create_or_extend_section(start, i)
                start = None
                empty_frames = 0

    # Handle the case where a section extends to the last frame
    if start is not None:
        frame_diff = len(ball_detections) - start
        if frame_diff > min_section_size:
            create_or_extend_section(start, len(ball_detections))

    return sections

def combine_ball_detections_sections(ball_detections_sections, video_info):
    ball_detections = []
    if ball_detections_sections[0]['section']['start']['index'] > 0:
        ball_detections.extend((None, None) for _ in range(int(ball_detections_sections[0]['section']['start']['index'])))
    for i, sectioned_data in enumerate(ball_detections_sections):
        ball_detections.extend(sectioned_data['data'])
        if i + 1 < len(ball_detections_sections):
            gap_frames = int(ball_detections_sections[i+1]['section']['start']['index'] - sectioned_data['section']['end']['index'])
            ball_detections.extend((None, None) for _ in range(gap_frames))
    if ball_detections_sections[-1]['section']['end']['index'] < video_info['total_frames']:
        ball_detections.extend((None, None) for _ in range(int(video_info['total_frames'] - ball_detections_sections[-1]['section']['end']['index'])))
    return ball_detections

def ball_based_sectioning(
        video_id,
        video_path,
        ball_tracker: BallTracker,
        all_court_keypoints: list[dict],
        video_info=None,
        chunk_size=1000, 
        make_video_info_request=False,
        make_sections_request=False,
        make_ball_detections_request=False,
        read_from_stub=False,
        save_to_stub=False,
        video_info_webhook_path='/ai_analysis/video_info',
        sections_webhook_path='/ai_analysis/sections',
        ball_detections_webhook_path='/ai_analysis/ball_detections',
        ball_detections_stub_path='tracker_stubs/ball_detections_sections.pkl',
        sections_stub_path='tracker_stubs/sections.pkl'
        ):
    print("Starting ball based sectioning")

    # Get video_info if not provided
    if video_info is None:
        video_info = get_video_info(video_path)

    if read_from_stub:
        sections = load_detections_from_stub(sections_stub_path)
        ball_detections_for_sections = load_detections_from_stub(ball_detections_stub_path)
    else:
        ball_detections, ball_speeds = ball_stubs_for_whole_video(video_id, video_path, ball_tracker, all_court_keypoints, video_info=video_info)
        sections = get_sections_fram_ball_detections(ball_detections, ball_speeds, video_info)
        
        ball_detections_for_sections = []
        for section in sections:
            stub = ball_detections[section['start']['index']:section['end']['index']]
            ball_detections_for_sections.append({
                'section': section,
                'data': stub
            })

    if save_to_stub:
        save_detections_to_stub(sections, sections_stub_path)
        save_detections_to_stub(ball_detections_for_sections, ball_detections_stub_path)
        print(f"Saved detections to stubs")

    if make_video_info_request:
        print(f"Making video info request")
        base_url = os.getenv('BACKEND_URL')
        video_info_request = requests.post(base_url+video_info_webhook_path, json={
            'video_id': video_id,
            'data': video_info
        })
        if video_info_request.status_code != 200:
            raise Exception(f"Failed to make request 'video_info': {video_info_request.text}")

    if make_sections_request:
        print(f"Making sections request")
        base_url = os.getenv('BACKEND_URL')
        section_response = requests.post(base_url+sections_webhook_path, json={
            'video_id': video_id,
            'data': sections,
        })
        if section_response.status_code != 200:
            raise Exception(f"Failed to make request 'sections': {section_response.text}")

    if make_ball_detections_request:
        print(f"Making ball detections request")
        base_url = os.getenv('BACKEND_URL')
        ball_detections_for_sections_response = requests.post(base_url+ball_detections_webhook_path, json={
            'video_id': video_id,
            'data': ball_detections_for_sections
        })
        if ball_detections_for_sections_response.status_code != 200:
            raise Exception(f"Failed to make request 'ball_detections_for_sections': {ball_detections_for_sections_response.text}")

    return sections, ball_detections_for_sections, ball_speeds