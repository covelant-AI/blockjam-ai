from utils.video_utils import read_video_range, get_video_info, total_chunks
import requests
import os
from utils.stubs import load_detections_from_stub, save_detections_to_stub
from schemas.section import SectionSchema
from schemas.stubs import StubsData
from ai_core.ball_tracker import BallTracker
import math
from services.analysis.court_keypoints import default_time_step as court_keypoints_time_step

def ball_stubs_for_whole_video(
        video_id,
        video_path,
        ball_tracker: BallTracker,
        all_court_keypoints: list[dict],
        video_info=None,
        make_request=False,
        read_from_stub=False,
        save_to_stub=False,
        chunk_size=1000,
        webhook_path='/ai_analysis/ball_detections_full_video',
        ball_detections_stub_path='tracker_stubs/ball_detections.pkl'
        ) -> list[StubsData]:
    
    if read_from_stub:
        ball_detections = load_detections_from_stub(ball_detections_stub_path)

    else:
        if video_info is None:
                video_info = get_video_info(video_path)
                

        ball_detections = []
        ball_speeds = []
        total_num_chunks = total_chunks(video_path, chunk_size)
        ball_tracker.set_keypoints(all_court_keypoints[0])
        for chunk_index in range(total_num_chunks):
            start_index = chunk_index * chunk_size
            end_index = start_index + chunk_size

            overall_frame_index = start_index + chunk_index * chunk_size
            if overall_frame_index % court_keypoints_time_step == 0:
                court_keypoints_index = int(min(overall_frame_index // (court_keypoints_time_step * video_info['fps']), len(all_court_keypoints) - 1))
                court_keypoints = all_court_keypoints[court_keypoints_index]
                ball_tracker.update_keypoints(court_keypoints)

            frames = read_video_range(video_path, start_index, end_index)
            bd, speeds = ball_tracker.detect_frames(frames, video_info["fps"])
            ball_detections.extend(bd)
            ball_speeds.extend(speeds)
            print(f"Ball Detection: Processed chunk {chunk_index} of {total_num_chunks}")

    if save_to_stub:
        print(f"Saving ball detections to stubs")
        save_detections_to_stub(ball_detections, ball_detections_stub_path)
    
    if make_request:
        print(f"Making request to backend")
        response = requests.post(os.getenv('BACKEND_URL')+webhook_path, json={
            'video_id': video_id,
            'data': ball_detections,
        })
        if response.status_code != 200:
            raise Exception(f"Failed to make request 'ball_detections': {response.text}")

    return ball_detections, ball_speeds


def ball_stubs_for_sections(
        video_id,
        video_path,
        ball_tracker: BallTracker,
        sections: list[SectionSchema],
        video_info=None,
        make_request=False,
        read_from_stub=False,
        save_to_stub=True,
        chunk_size=1000,
        webhook_path='/ai_analysis/ball_detections_sections',
        ball_detections_stub_path='tracker_stubs/ball_detections_sections.pkl'
        ) -> list[StubsData]:
    
    ball_detections_for_sections = []
    error_sections = []

    if video_info is None:
        video_info = get_video_info(video_path)
    
    for section in sections:
        section_start = section['start']['index']
        section_end = section['end']['index']
        ball_detections_for_section = {'section': section, 'data': []}
        for chunk_index in range(math.ceil((section_end - section_start) / chunk_size)):
            try:
                start_index = section_start + (chunk_index * chunk_size)
                end_index = min(section_end, start_index + chunk_size)
                print(f"Generating ball stubs for frames {start_index} to {end_index}")

                frames = read_video_range(video_path, start_index, end_index)

                ball_detections, speeds = ball_tracker.detect_frames(frames, video_info["fps"])
                ball_detections_for_section['data'].extend(ball_detections)
            except Exception as e:
                print(f"Error generating ball stubs for section starting at {section['start']['index']}: {e}")
                error_sections.append({
                    'section': section,
                    'message': str(e)
                })
        ball_detections_for_sections.append(ball_detections_for_section)

    if save_to_stub:
        save_detections_to_stub(ball_detections_for_sections, ball_detections_stub_path)
        print(f"Saved detections to stubs")

    if make_request:
        response = requests.post(os.getenv('BACKEND_URL')+webhook_path, json={
            'video_id': video_id,
            'data': ball_detections_for_sections,
            'error_sections': error_sections,
        })
        if response.status_code != 200:
            raise Exception(f"Failed to make request 'ball_detections_for_sections': {response.text}")

    return ball_detections_for_sections
