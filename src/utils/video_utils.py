import cv2
import math
import os

def get_frame_at_index(video_path, index):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    cap.release()
    return frame

def get_every_nth_frame_in_range(video_path, start_frame, end_frame, n):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in range(start_frame, end_frame, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame

def total_chunks(video_path, chunk_size, total_frames=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return math.ceil(total_frames / chunk_size)

def read_video_range(video_path, start_frame, end_frame):
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def read_video_chunk(video_path, chunk_index, chunk_size):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_index * chunk_size)
    frames = []
    for _ in range(chunk_size):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# I want to get some information about the video
def get_video_info(video_path) -> dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    return {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }

def save_video(output_video_frames, output_video_path, video_fps):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def convert_frames_to_lower_fps(video_frames, fps, target_fps, offset=0):
    if offset >= len(video_frames):
        return []
    if fps <= target_fps:
        return video_frames[offset:]
    else:
        step = int(fps/target_fps)
        return video_frames[offset::step]