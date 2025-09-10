from utils.video_utils import read_video_range, get_video_info
import requests
import os
from utils.stubs import load_detections_from_stub, save_detections_to_stub
from schemas.section import SectionSchema
from schemas.stubs import StubsData
from ai_core.player_tracker import PlayerTracker
from .court_keypoints import get_keypoint_closest_to_section

def player_stubs_for_sections(
        video_id,
        video_path,
        player_tracker: PlayerTracker,
        all_court_keypoints: list,
        sections: list[SectionSchema],
        video_info=None,
        make_request=False,
        read_from_stub=False,
        save_to_stub=True,
        webhook_path='/ai_analysis/player_detections',
        player_detections_stub_path='tracker_stubs/player_detections_sections.pkl'
        ) -> list[StubsData]:
    
    player_detections_for_sections = []
    error_sections = []

    if video_info is None:
        video_info = get_video_info(video_path)
            
    if read_from_stub:
        player_detections_for_sections = load_detections_from_stub(player_detections_stub_path)

    else:
        for section in sections:
            start_index = section['start']['index']
            end_index = section['end']['index']
            try:
                print(f"Generating player stubs for frames {start_index} to {end_index}")

                frames = read_video_range(video_path, start_index, end_index)
                court_keypoints = get_keypoint_closest_to_section(all_court_keypoints, section)

                pd = player_tracker.detect_frames(frames)
                pd = player_tracker.choose_and_filter_players(court_keypoints, pd)
                player_detections_for_sections.append({
                    'section': section,
                    'data': pd
                })
            except Exception as e:
                print(f"Error generating player stubs for section starting at {section['start']['index']}: {e}")
                error_sections.append({
                    'section': section,
                    'message': str(e)
                })

    if save_to_stub:
        save_detections_to_stub(player_detections_for_sections, player_detections_stub_path)
        print(f"Saved detections to stubs")
    
    if make_request:
        response = requests.post(os.getenv('BACKEND_URL')+webhook_path, json={
            'video_id': video_id,
            'data': player_detections_for_sections,
            'error_sections': error_sections,
        })
        if response.status_code != 200:
            raise Exception(f"Failed to make request 'player_detections_for_sections': {response.text}")

    return player_detections_for_sections