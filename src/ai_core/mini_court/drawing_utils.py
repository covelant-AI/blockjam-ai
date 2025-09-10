import cv2
from utils import get_foot_position

def draw_court_video(frames, player_mini_court_coordinates, ball_mini_court_coordinates, mini_court):
    """
    Draw the mini court visualization with player and ball positions on the right side of the screen.
    
    Args:
        frames (list): List of video frames
        player_mini_court_coordinates (list): List of dictionaries containing player positions in mini court coordinates
        ball_mini_court_coordinates (list): List of dictionaries containing ball positions in mini court coordinates
        mini_court (MiniCourt): MiniCourt instance containing court dimensions and positions
        
    Returns:
        list: List of frames with mini court visualization
    """
    output_frames = []
    
    for frame_idx, frame in enumerate(frames):
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Calculate the right side position for the mini court
        frame_width = output_frame.shape[1]
        right_margin = 20  # pixels from the right edge
        mini_court.mini_court_x = frame_width - mini_court.mini_court_width - right_margin
        
        # Create mini court background (light gray rectangle)
        cv2.rectangle(output_frame,
                    (mini_court.mini_court_x, mini_court.mini_court_y),
                    (mini_court.mini_court_x + mini_court.mini_court_width, 
                     mini_court.mini_court_y + mini_court.mini_court_height),
                    (200, 200, 200),
                    -1)
        
        # Draw court outline (black rectangle)
        cv2.rectangle(output_frame,
                    (mini_court.mini_court_x + mini_court.court_x, 
                     mini_court.mini_court_y + mini_court.court_y),
                    (mini_court.mini_court_x + mini_court.court_x + mini_court.court_width, 
                     mini_court.mini_court_y + mini_court.court_y + mini_court.court_height),
                    (0, 0, 0),
                    2)
        
        # Draw net line (dashed line in the middle)
        net_y = mini_court.mini_court_y + mini_court.court_y + mini_court.court_height // 2
        dash_length = 10
        gap_length = 5
        x_start = mini_court.mini_court_x + mini_court.court_x
        x_end = mini_court.mini_court_x + mini_court.court_x + mini_court.court_width
        
        for x in range(x_start, x_end, dash_length + gap_length):
            cv2.line(output_frame,
                    (x, net_y),
                    (min(x + dash_length, x_end), net_y),
                    (0, 0, 0),
                    1)
        
        # Draw player positions if available for this frame
        if frame_idx < len(player_mini_court_coordinates):
            for player_id, coords in player_mini_court_coordinates[frame_idx].items():
                # Convert normalized coordinates to pixel coordinates
                x = int(mini_court.mini_court_x + mini_court.court_x + (coords[0] + 1) * mini_court.court_width / 2)
                y = int(mini_court.mini_court_y + mini_court.court_y + (coords[1] + 1) * mini_court.court_height / 2)
                
                # Blue for player 1, Red for player 2
                color = (255, 0, 0) if player_id == 1 else (0, 0, 255)
                
                # Draw player position
                cv2.circle(output_frame, (x, y), 5, color, -1)
                
                # Draw coordinates
                coord_text = f"P{player_id}: ({coords[0]:.2f}, {coords[1]:.2f})"
                cv2.putText(output_frame,
                          coord_text,
                          (x + 10, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.4,
                          color,
                          1)
        
        # Draw ball position if available for this frame
        if frame_idx < len(ball_mini_court_coordinates):
            for ball_id, coords in ball_mini_court_coordinates[frame_idx].items():
                # Convert normalized coordinates to pixel coordinates
                x = int(mini_court.mini_court_x + mini_court.court_x + (coords[0] + 1) * mini_court.court_width / 2)
                y = int(mini_court.mini_court_y + mini_court.court_y + (coords[1] + 1) * mini_court.court_height / 2)
                
                # Draw ball position (yellow)
                cv2.circle(output_frame, (x, y), 3, (0, 255, 255), -1)
                
                # Draw coordinates
                coord_text = f"Ball: ({coords[0]:.2f}, {coords[1]:.2f})"
                cv2.putText(output_frame,
                          coord_text,
                          (x + 10, y + 10),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.4,
                          (0, 255, 255),
                          1)
        
        output_frames.append(output_frame)
    
    return output_frames

def draw_screen_points(frames, player_boxes, court_keypoints):
    """
    Draw the players' feet positions and court keypoints on the main video frames.
    
    Args:
        frames (list): List of video frames
        player_boxes (list): List of dictionaries containing player bounding boxes for each frame
        court_keypoints (list): List of court keypoints [x1,y1,x2,y2,x3,y3,x4,y4]
        
    Returns:
        list: List of frames with player feet positions and court keypoints marked
    """
    output_frames = []
    
    for frame_idx, frame in enumerate(frames):
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Draw court keypoints
        keypoint_color = (0, 255, 0)  # Green color for keypoints
        for i in range(0, len(court_keypoints), 2):
            x, y = int(court_keypoints[i]), int(court_keypoints[i+1])
            # Draw keypoint
            cv2.circle(output_frame, (x, y), 5, keypoint_color, -1)
            # Draw keypoint number
            cv2.putText(output_frame,
                      f"K{i//2}",
                      (x + 10, y - 10),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.5,
                      keypoint_color,
                      2)
        
        # Draw court lines connecting keypoints
        # Top line
        cv2.line(output_frame,
                (int(court_keypoints[0]), int(court_keypoints[1])),
                (int(court_keypoints[2]), int(court_keypoints[3])),
                keypoint_color, 2)
        # Right line
        cv2.line(output_frame,
                (int(court_keypoints[2]), int(court_keypoints[3])),
                (int(court_keypoints[6]), int(court_keypoints[7])),
                keypoint_color, 2)
        # Bottom line
        cv2.line(output_frame,
                (int(court_keypoints[6]), int(court_keypoints[7])),
                (int(court_keypoints[4]), int(court_keypoints[5])),
                keypoint_color, 2)
        # Left line
        cv2.line(output_frame,
                (int(court_keypoints[4]), int(court_keypoints[5])),
                (int(court_keypoints[0]), int(court_keypoints[1])),
                keypoint_color, 2)
        
        # Draw player feet if available for this frame
        if frame_idx < len(player_boxes):
            for player_id, bbox in player_boxes[frame_idx].items():
                # Get foot position from bounding box
                foot_position = get_foot_position(bbox)
                
                # Draw foot position marker
                # Blue for player 1, Red for player 2
                color = (255, 0, 0) if player_id == 1 else (0, 0, 255)
                
                # Draw a circle at the foot position
                cv2.circle(output_frame, 
                         (int(foot_position[0]), int(foot_position[1])), 
                         5,  # radius
                         color, 
                         -1)  # filled circle
                
                # Draw a small crosshair for better visibility
                crosshair_size = 10
                cv2.line(output_frame,
                        (int(foot_position[0] - crosshair_size), int(foot_position[1])),
                        (int(foot_position[0] + crosshair_size), int(foot_position[1])),
                        color, 2)
                cv2.line(output_frame,
                        (int(foot_position[0]), int(foot_position[1] - crosshair_size)),
                        (int(foot_position[0]), int(foot_position[1] + crosshair_size)),
                        color, 2)
                
                # Add player ID label
                cv2.putText(output_frame,
                          f"Player {player_id}",
                          (int(foot_position[0] + 10), int(foot_position[1] - 10)),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5,
                          color,
                          2)
        
        output_frames.append(output_frame)
    
    return output_frames 