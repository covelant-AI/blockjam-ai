import cv2
import numpy as np

def compute_court_homography_matrix(court_keypoints, court_length, court_width):
    """
    mini_court: MiniCourt object
    court_keypoints: list of court keypoints 

    return H:
        H must map image pixel homogeneous coords to court-plane meters:
            [x_m, y_m, w]^T  ~  H @ [u, v, 1]^T
    """
    # Extract the four corner keypoints (assuming they are in order: top-left, top-right, bottom-left, bottom-right)
    # These represent the detected court corners in the image
    src_points = np.array([
        court_keypoints[0],  # Top left
        court_keypoints[1],  # Top right  
        court_keypoints[2],  # Bottom left
        court_keypoints[3]   # Bottom right
    ], dtype=np.float32)
    
    # Define the corresponding destination points in court-plane meters
    # Using standard tennis court dimensions from constants
    # Origin is at the center of the court, positive x is right, positive y is down
    half_court_length = court_length / 2  # 11.885 meters
    half_court_width = court_width / 2  # 5.485 meters
    
    dst_points = np.array([
        [-half_court_width, -half_court_length],  # Top left: (-5.485, -11.885)
        [half_court_width, -half_court_length],   # Top right: (5.485, -11.885)
        [-half_court_width, half_court_length],   # Bottom left: (-5.485, 11.885)
        [half_court_width, half_court_length]     # Bottom right: (5.485, 11.885)
    ], dtype=np.float32)
    
    # Compute the homography matrix using OpenCV
    # This will give us H such that dst_points ~ H @ src_points
    H, _ = cv2.findHomography(src_points, dst_points)
    
    return H