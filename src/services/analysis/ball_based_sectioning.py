from utils.conversions import frame_to_time

def get_sections_fram_ball_detections(
        ball_detections,
        ball_speeds, 
        video_info,
        min_section_time=1.5,
        section_null_threshold=0.75,
        min_time_between_sections=1,
        buffer_time_before_section=2,
        buffer_time_after_section=1,
        max_gap_time=1,
        min_ball_speed_kmh=5,
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