from utils.video_utils import total_chunks, read_video_chunk, get_video_info
import requests
import os
from utils.stubs import load_detections_from_stub, save_detections_to_stub
import time

def scoreboard_service(
        video_id,
        video_path,
        scoreboard_model,
        video_info=None,
        chunk_size=1000, 
        read_from_stub=False,
        save_to_stub=True,
        make_request=False,
        ):
    if video_info is None:
        video_info = get_video_info(video_path)

    scoreboard_readings = []

    if read_from_stub:
        scoreboard_readings = load_detections_from_stub("tracker_stubs/scoreboard_readings.pkl")
    else:
        processing_time_start = time.time()
        for chunk_index in range(total_chunks(video_path, chunk_size)):
            time_start = time.time()
            print(f"Processing scoreboard for chunk {chunk_index} of {total_chunks(video_path, chunk_size)}")
            frames = read_video_chunk(video_path, chunk_index, chunk_size)

            frames_elapsed = chunk_size * chunk_index
            sd = scoreboard_model.detect_scoreboard(frames, video_info['fps'], frames_elapsed)
            sr = scoreboard_model.read_scoreboard_detections(frames, sd, video_info['fps'], frames_elapsed)
            scoreboard_readings.extend(sr)

            time_elapsed = time.time() - time_start
            print(f"Processed scoreboard for chunk {chunk_index} in {time_elapsed} seconds")

        if save_to_stub:
            save_detections_to_stub(scoreboard_readings, "tracker_stubs/scoreboard_readings.pkl")
            print(f"Saved scoreboard to stubs")
        
        print(f"Total processing time for scoreboard: {time.time() - processing_time_start} seconds")



    scoreboard_changes = scoreboard_model.get_score_changes(scoreboard_readings)
    formatted_scoreboard_changes = scoreboard_model.get_formatted_scoreboard_changes(scoreboard_changes)

    if make_request:
        base_url = os.getenv('BACKEND_URL')+'/ai_analysis'
        response = requests.post(base_url+'/scoreboard', json={
            'video_id': video_id,
            'scoreboard_changes': formatted_scoreboard_changes
        })
        if response.status_code != 200:
            raise Exception(f"Failed to handle scoreboard: {response.text}")

    return formatted_scoreboard_changes