from ai_core.trackers.core import BallAndPlayerTracker
from utils.video_utils import read_video_range, get_video_info, total_chunks
from services.analysis.court_keypoints import default_time_step as court_keypoints_time_step

def tracking(
        video_path,
        ball_and_player_tracker: BallAndPlayerTracker,
        all_court_keypoints: list[dict],
        video_info,
        chunk_size=1000
    ) -> tuple[list[tuple[float, float]], list[float], list[tuple[float, float]], list[float], list[tuple[float, float]], list[float]]:

    def _get_polygon(court_keypoints):
        return [
            court_keypoints[0], # TL
            court_keypoints[1], # TR
            court_keypoints[3], # BR
            court_keypoints[2], # BL
        ]

    world_tracks = []
    all_polygons = [_get_polygon(court_keypoints) for court_keypoints in all_court_keypoints]
    ball_and_player_tracker.define_trackers_using_polygons(all_polygons[0], video_info["fps"])

    total_num_chunks = total_chunks(video_path, chunk_size)
    for chunk_index in range(total_num_chunks):
        start_index = chunk_index * chunk_size
        end_index = start_index + chunk_size

        frames = read_video_range(video_path, start_index, end_index)
        wt = ball_and_player_tracker.detect_frames(frames, start_index, all_polygons, update_court_polygon_interval=int(court_keypoints_time_step*video_info["fps"])) 
        world_tracks.extend(wt)

    return ball_and_player_tracker.extract_from_world_tracks(world_tracks)



    