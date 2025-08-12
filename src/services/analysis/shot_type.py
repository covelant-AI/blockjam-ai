from ai_core.shot_type_classifier import ShotTypeClassifier
import requests
import os
from dotenv import load_dotenv
from schemas.stubs import SectionedStubsData
from utils.video_utils import get_video_info, read_video_range

load_dotenv()

def classify_player_shots(video_id, video_path, sectioned_data:list[SectionedStubsData], shot_type_classifier:ShotTypeClassifier, make_request=False, webhook_path='/ai_analysis/player_detections'):
    video_info = get_video_info(video_path)
    results = []
    error_sections = []
    for _s in sectioned_data:
        try:
            section = _s['section']
            player_detections = _s['player_detections']
            ball_detections = _s['ball_detections']
            court_keypoints = _s['court_keypoints']   

            frames = read_video_range(video_path, section['start']['index'], section['end']['index'])
            p1_bboxes = [pd.get("1", None) for pd in player_detections]
            p2_bboxes = [pd.get("2", None) for pd in player_detections]
            p1_shots = shot_type_classifier.classify(frames, p1_bboxes, flip=True)
            p1_shots = shot_type_classifier.shot_counter.format_results(video_info['fps'])
            shot_type_classifier.shot_counter.results = []
            p2_shots = shot_type_classifier.classify(frames, p2_bboxes, flip=False)
            p2_shots = shot_type_classifier.shot_counter.format_results(video_info['fps'])
            results.append({
                'section': section,
                'data': {
                    'p1': p1_shots,
                    'p2': p2_shots
                }
            })
        except Exception as e:
            error_sections.append({
                'section': section,
                'error': str(e)
            })


    if make_request:
        response = requests.post(os.getenv('BACKEND_URL')+webhook_path, json={
            'video_id': video_id,
            'data': results,
            'error_sections': error_sections
        })
        if response.status_code != 200:
            raise Exception(f"Failed to make request 'player_detections_for_sections': {response.text}")

    return results