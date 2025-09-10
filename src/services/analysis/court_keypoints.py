from utils.video_utils import get_video_info, get_frame_at_index
import requests
import os
from schemas.section import SectionSchema
from ai_core.court_line_detector import CourtLineDetector
from utils.video_utils import total_chunks
from ai_core.letr_court_line_detector import LETRCourtDetector

default_time_step =5*60

def fill_none_with_closest(all_court_keypoints):
    """
    Fill None values with the closest non-None value.
    
    Args:
        all_court_keypoints: List of court keypoints where some may be None
        
    Returns:
        List of court keypoints with None values filled with closest non-None values
    """
    if not all_court_keypoints:
        return all_court_keypoints
    
    # Find all non-None indices
    non_none_indices = [i for i, keypoint in enumerate(all_court_keypoints) if keypoint is not None]
    
    if not non_none_indices:
        return all_court_keypoints  # All are None, return as is
    
    filled_keypoints = []
    
    for i, keypoint in enumerate(all_court_keypoints):
        if keypoint is not None:
            # Keep original value if not None
            filled_keypoints.append(keypoint)
        else:
            # Find the closest non-None index
            closest_index = min(non_none_indices, key=lambda x: abs(x - i))
            filled_keypoints.append(all_court_keypoints[closest_index])
    
    return filled_keypoints

def get_keypoint_closest_to_section(court_keypoints, section: SectionSchema, time_step=default_time_step):
    middle_time = section['start']['time'] + ((section['end']['time'] - section['start']['time']) // 2)
    middle_time = int(middle_time)

    idx = min(round(middle_time / time_step), len(court_keypoints) - 1)
    return court_keypoints[idx]

def court_keypoints_by_time_step(
        video_id,
        video_path,
        court_line_detector: CourtLineDetector,
        letr_court_line_detector: LETRCourtDetector,
        video_info=None,
        make_request=False,
        webhook_path='/ai_analysis/court_keypoints',
        time_step=default_time_step,
        ) -> list[dict]:
    
    video_info = get_video_info(video_path)
    time_step_frames = time_step * video_info['fps']

    all_court_keypoints = []
    for i in range(total_chunks(video_path, time_step_frames)):
        if i == 0 and video_info['total_frames'] > 10*video_info['fps']:
            frame = get_frame_at_index(video_path, 10*video_info['fps'])
        else:
            frame = get_frame_at_index(video_path, i * time_step_frames)
        try:
            keypoints = court_line_detector.detect(frame)
        except Exception as e:
            print(f"Base court line detector failed. Next trying LETR court line detector for frame {i}: {e}")
            try:
                keypoints = letr_court_line_detector.detect(frame)
            except Exception as e:
                print(f"LETR court line detector failed for frame {i}: {e}")
                keypoints = None
        finally:
            all_court_keypoints.append(keypoints)

    #are all detections None?
    if not any(keypoint is not None for keypoint in all_court_keypoints):
        raise Exception("No court keypoints detections found")
    
    #fill all None values with the closest non-None value
    all_court_keypoints = fill_none_with_closest(all_court_keypoints)

    if make_request:
        request_data = {
            "video_id": video_id,
            "time_step": time_step,
            "data": all_court_keypoints,
        }
        response = requests.post(os.getenv('BACKEND_URL')+webhook_path, json=request_data)
        if response.status_code != 200:
            raise Exception(f"Failed to handle court keypoints: {response.text}")

    return all_court_keypoints
