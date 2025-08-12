from schemas.stubs import StubsData, SectionedStubsData

def convert_pixel_distance_to_meters(pixel_distance, refrence_height_in_meters, refrence_height_in_pixels):
    return (pixel_distance * refrence_height_in_meters) / refrence_height_in_pixels

def convert_meters_to_pixel_distance(meters, refrence_height_in_meters, refrence_height_in_pixels):
    return (meters * refrence_height_in_pixels) / refrence_height_in_meters

def frame_to_time(frame, fps):
    return int(frame) / int(fps)

def convert_detections_to_mini_court_coordinates(
        mini_court,
        sectioned_data: list[SectionedStubsData],
        ) -> dict[str, list[StubsData]]:
    player_mini_court_detections_for_sections = []
    ball_mini_court_detections_for_sections = []
    for _s in sectioned_data:
        section = _s['section']
        court_keypoints = _s['court_keypoints']
        if 'player_detections' in _s and _s['player_detections'] is not None:
            player_detections = _s['player_detections']
            player_mini_court_detections = mini_court.convert_player_boxes_to_mini_court_coordinates(player_detections, court_keypoints)
            player_mini_court_detections_for_sections.append({
                'section': section,
                'data': player_mini_court_detections,
            })
        if 'ball_detections' in _s and _s['ball_detections'] is not None:
            ball_detections = _s['ball_detections']
            ball_mini_court_detections = mini_court.convert_ball_detections_to_mini_court_coordinates(ball_detections, court_keypoints)
            ball_mini_court_detections_for_sections.append({
                'section': section,
                'data': ball_mini_court_detections,
            })
    
    return {
        'player_mini_court_detections_for_sections': player_mini_court_detections_for_sections,
        'ball_mini_court_detections_for_sections': ball_mini_court_detections_for_sections,
    }


def scale_points_to_size(points, original_dimensions, target_dimensions):
    return  [(float(point[0] * (target_dimensions[0] / original_dimensions[0])), float(point[1] * (target_dimensions[1] / original_dimensions[1]))) for point in points]