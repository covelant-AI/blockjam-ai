import cv2
import numpy as np
from utils import get_foot_position

def convert_screen_to_mini_court_coordinates(screen_position: tuple[float, float], original_court_key_points: list[tuple[float, float]]) -> tuple[float, float]:
    """
    Convert a point from screen coordinates to mini court coordinates.
    
    Args:
        screen_position (tuple): (x, y) coordinates of the point on screen
        court_keypoints (list): List of original court keypoints
        
    Returns:
        tuple: (x, y) coordinates on the mini court. x and y are values between 0 and 1.
    """
    # Get the four corners of the original court (keypoints 0,1,2,3)
    original_corners = np.array([
        original_court_key_points[0],  # Top left
        original_court_key_points[1],  # Top right
        original_court_key_points[2],  # Bottom left
        original_court_key_points[3]   # Bottom right
    ], dtype=np.float32)

    # Define the corresponding points on the mini court (normalized coordinates)
    mini_court_corners = np.array([
        [-1, -1],         # Top left
        [1, -1],          # Top right
        [-1, 1],          # Bottom left
        [1, 1]            # Bottom right
    ], dtype=np.float32)

    # Compute homography matrix
    H, _ = cv2.findHomography(original_corners, mini_court_corners)

    # Convert screen position to numpy array
    point = np.array([[screen_position[0], screen_position[1]]], dtype=np.float32)
    point = np.array([point])

    # Transform the point using the homography matrix
    transformed_point = cv2.perspectiveTransform(point, H)
    
    # Extract x and y coordinates
    x, y = transformed_point[0][0]

    return (float(x), float(y))

def get_distance_in_meters(p1, p2, mini_court):
    """
    Get distance between two points in meters.
    
    Args:
        p1 (tuple): (x, y) coordinates of point 1 in mini court coordinates (0-1)
        p2 (tuple): (x, y) coordinates of point 2 in mini court coordinates (0-1)
        mini_court (MiniCourt): MiniCourt instance containing meter ratios
        
    Returns:
        float: Distance between the two points in meters
    """
    dx_meters = (p1[0] - p2[0]) * mini_court.mini_court_width_to_meters_ratio
    dy_meters = (p1[1] - p2[1]) * mini_court.mini_court_length_to_meters_ratio
    distance_in_meters = float(np.sqrt(dx_meters**2 + dy_meters**2))  # Convert to regular Python float
    return distance_in_meters

def convert_player_boxes_to_mini_court_coordinates(player_boxes, original_court_key_points):
    """
    Convert player bounding boxes to mini court coordinates.
    
    Args:
        player_boxes (list): List of dictionaries containing player bounding boxes for each frame
        original_court_key_points (list): List of original court keypoints
        
    Returns:
        list: List of dictionaries containing player positions in mini court coordinates
    """
    output_player_boxes = []

    for frame_num, player_bbox in enumerate(player_boxes):
        output_player_bboxes_dict = {}
        for player_id, bbox in player_bbox.items():
            foot_position = get_foot_position(bbox)
            x, y = convert_screen_to_mini_court_coordinates(foot_position, original_court_key_points)
            output_player_bboxes_dict[player_id] = (x, y)
        output_player_boxes.append(output_player_bboxes_dict)

    return output_player_boxes

def convert_ball_detections_to_mini_court_coordinates(ball_detections, original_court_key_points):
    """
    Convert ball centers to mini court coordinates.
    
    Args:
        ball_detections (list): List of ball detections (x,y) coordinates
        original_court_key_points (list): List of original court keypoints
        
    Returns:
        list: List of ball centers in mini court coordinates
    """
    output_ball_centers = [convert_screen_to_mini_court_coordinates(ball_detection, original_court_key_points) for ball_detection in ball_detections]
    return output_ball_centers