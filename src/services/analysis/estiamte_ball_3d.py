import numpy as np
import cv2

court_3d_points = np.array([
    [0, 0, 0],           # Top-left corner
    [23.77, 0, 0],       # Top-right corner
    [0, 10.97, 0],       # Bottom-left corner  
    [23.77, 10.97, 0],   # Bottom-right corner
])

def estimate_camera_parameters(court_keypoints):
    """
    Estimate camera parameters using court keypoints and known 3D court dimensions.
    
    Args:
        court_keypoints: 2D keypoints from court detection (first 4 are corners)
        court_3d_points: 3D coordinates of court corners in meters
    
    Returns:
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: distortion coefficients
        rvec: rotation vector
        tvec: translation vector
    """
    # Convert keypoints to numpy array
    keypoints_2d = np.array(court_keypoints[:4], dtype=np.float32)
    
    
    # First, estimate a rough camera matrix from keypoints
    # This is a simplified approach - in practice you'd use calibration patterns
    focal_length = max(keypoints_2d[:, 0].max() - keypoints_2d[:, 0].min(),
                      keypoints_2d[:, 1].max() - keypoints_2d[:, 1].min())
    
    # Ensure focal length is reasonable (not too small)
    focal_length = max(focal_length, 1000.0)
    
    camera_matrix = np.array([
        [focal_length, 0, keypoints_2d[:, 0].mean()],
        [0, focal_length, keypoints_2d[:, 1].mean()],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Initialize distortion coefficients
    dist_coeffs = np.zeros(5, dtype=np.float32)
    
    # Now estimate camera parameters using solvePnP with the estimated camera matrix
    success, rvec, tvec = cv2.solvePnP(
        court_3d_points, keypoints_2d, 
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        raise ValueError("Failed to estimate camera parameters")
    
    return camera_matrix, dist_coeffs, rvec, tvec

def estimate_ball_3d_position(ball_2d, court_keypoints, camera_matrix, dist_coeffs, rvec, tvec):
    """
    Estimate 3D position of ball using proper 3D ray projection.
    
    Args:
        ball_2d: 2D ball coordinates [x, y]
        court_keypoints: 2D court keypoints
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        rvec: Rotation vector
        tvec: Translation vector
    
    Returns:
        ball_3d: 3D ball coordinates [x, y, z] in meters
    """
    # Convert ball 2D to normalized coordinates
    ball_2d_np = np.array([ball_2d], dtype=np.float32)
    
    # Undistort the ball point
    ball_2d_undistorted = cv2.undistortPoints(ball_2d_np, camera_matrix, dist_coeffs)
    
    # Create ray from camera center through the ball point
    # The ray direction is the normalized undistorted point
    ray_direction = np.array([ball_2d_undistorted[0, 0, 0], ball_2d_undistorted[0, 0, 1], 1.0])
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Transform ray to world coordinates
    ray_world = R.T @ ray_direction
    
    # Camera position in world coordinates
    camera_pos_world = -R.T @ tvec.flatten()
    
    # Court plane equation: z = 0
    # Ray equation: camera_pos + t * ray_world
    # Intersection: camera_pos.z + t * ray_world.z = 0
    # Therefore: t = -camera_pos.z / ray_world.z
    
    if abs(ray_world[2]) < 1e-6:  # Ray is parallel to court plane
        # Fallback: use homography for ground projection
        court_corners_2d = np.array(court_keypoints[:4], dtype=np.float32)
        H = cv2.findHomography(court_3d_points[:, :2], court_corners_2d)[0]
        ball_2d_homogeneous = np.array([ball_2d[0], ball_2d[1], 1])
        ball_court_2d = H @ ball_2d_homogeneous
        ball_court_2d = ball_court_2d[:2] / ball_court_2d[2]
        return np.array([ball_court_2d[0], ball_court_2d[1], 0.5])
    
    # Calculate intersection parameter t
    t = -camera_pos_world[2] / ray_world[2]
    
    # Calculate intersection point on court plane
    ball_3d_ground = camera_pos_world + t * ray_world
    
    # Now we need to estimate the ball height
    # We can use the fact that the ball appears at a certain position in the image
    # and use the camera parameters to estimate its height
    
    # For a given height h, the ball would appear at a different image position
    # We can solve for h by finding where the ray intersects with the ball's actual height
    
    # Let's estimate height based on the ball's vertical position in the image
    # Higher balls appear higher in the image
    image_height = 1080  # Assuming standard video height
    ball_y_normalized = (ball_2d[1] - image_height/2) / (image_height/2)  # -1 to 1
    
    # Simple height estimation based on vertical position
    # This is a heuristic - in practice you'd use more sophisticated methods
    ball_height = max(0.1, abs(ball_y_normalized) * 2.0)  # 0.1m to 2.0m range
    
    # Calculate the 3D position at the estimated height
    # Ray equation: camera_pos + t * ray_world
    # For height h: camera_pos.z + t * ray_world.z = h
    # Therefore: t = (h - camera_pos.z) / ray_world.z
    
    t_height = (ball_height - camera_pos_world[2]) / ray_world[2]
    ball_3d = camera_pos_world + t_height * ray_world
    
    return ball_3d


def estimate_3d_position(ball_detections, court_keypoints):
    camera_matrix, dist_coeffs, rvec, tvec = estimate_camera_parameters(court_keypoints)
    ball_3d_positions = []
    for ball_detection in ball_detections:
        if any(bd is None for bd in ball_detection):
            ball_3d_positions.append([None, None, None])
            continue
        ball_2d = ball_detection
        ball_3d = estimate_ball_3d_position(ball_2d, court_keypoints, camera_matrix, dist_coeffs, rvec, tvec)
        ball_3d_positions.append(ball_3d)
    return ball_3d_positions

