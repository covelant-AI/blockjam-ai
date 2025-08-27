from utils.video_utils import get_video_info
import requests
from ai_core.player_stats.core import get_player_speed_data
import os
from schemas.section import SectionSchema
from schemas.player_speed import PlayerSpeedsForSections
from utils.simple import find_object_by_key_value
from ai_core.mini_court import MiniCourt

def player_speed_for_section(
        video_id,
        player_mini_court_detections_for_sections: list[dict],
        mini_court: MiniCourt,
        video_path=None,
        video_info=None,
        time_step=0.5,
        make_request=False,
        webhook_path='/ai_analysis/player_speeds'
        ) -> tuple[list[PlayerSpeedsForSections], list[PlayerSpeedsForSections]]:
    player_speeds_for_sections = []
    error_sections = []

    for player_mini_court_detection_for_sections in player_mini_court_detections_for_sections:
        section = player_mini_court_detection_for_sections['section']
        try:
            print(f"Generating player speed for frames {section['start']['index']} to {section['end']['index']}")
            if video_info is None:
                video_info = get_video_info(video_path)
            p1_speeds, p2_speeds = get_player_speed_data(player_mini_court_detection_for_sections['data'], video_info['fps'], time_step, mini_court)
            player_speeds_for_sections.append({
                'section': section,
                'p1_speeds': p1_speeds,
                'p2_speeds': p2_speeds
            })
        except Exception as e:
            print(f"Error generating player speed for section starting at {section['start']['index']}: {e}")
            error_sections.append({
                'section': section,
                'message': str(e)
            })

    if make_request: 
        response = requests.post(os.getenv('BACKEND_URL')+webhook_path, json={
            'video_id': video_id,
            'time_step': time_step,
            'data': player_speeds_for_sections,
            'error_sections': error_sections,
        })
        if response.status_code != 200:
            raise Exception(f"Failed to handle player speed: {response.text}")

        return player_speeds_for_sections
