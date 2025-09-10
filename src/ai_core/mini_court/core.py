import cv2
import numpy as np
from .coordinate_utils import (
    get_distance_in_meters,
    convert_player_boxes_to_mini_court_coordinates,
    convert_screen_to_mini_court_coordinates,
    convert_ball_detections_to_mini_court_coordinates
)
from .drawing_utils import draw_court_video, draw_screen_points
from utils import constants
from .compute_homography_matrix import compute_court_homography_matrix

class MiniCourt:
    def __init__(self, output_width=1280, output_height=720):
        """
        Initialize MiniCourt with output dimensions.
        
        Args:
            output_width (int): Desired output video width
            output_height (int): Desired output video height
        """
        self.output_width = output_width
        self.output_height = output_height
        
        # Mini court dimensions (as a fraction of the output frame)
        self.mini_court_width = int(output_width * 0.25)  # 25% of frame width
        self.mini_court_height = int(output_height * 0.5)  # 50% of frame height
        self.mini_court_padding = 20
        
        # Calculate mini court position (top-right corner)
        self.mini_court_x = output_width - self.mini_court_width - self.mini_court_padding
        self.mini_court_y = self.mini_court_padding
        
        # Calculate the actual court dimensions based on real tennis court measurements
        court_aspect_ratio = constants.COURT_LENGTH / constants.DOUBLE_LINE_WIDTH
        
        # Calculate court dimensions maintaining the aspect ratio
        if court_aspect_ratio > (self.mini_court_height / self.mini_court_width):
            # Height is the limiting factor
            self.court_height = int(self.mini_court_height * 0.9)  # 90% of mini court height
            self.court_width = int(self.court_height / court_aspect_ratio)
        else:
            # Width is the limiting factor
            self.court_width = int(self.mini_court_width * 0.9)  # 90% of mini court width
            self.court_height = int(self.court_width * court_aspect_ratio)
        
        # Calculate court position within mini court area (centered)
        self.court_x = (self.mini_court_width - self.court_width) // 2
        self.court_y = (self.mini_court_height - self.court_height) // 2

        # Get the ratios of mini court distance to meters for both dimensions
        self.mini_court_length_to_meters_ratio = constants.COURT_LENGTH / 2
        self.mini_court_width_to_meters_ratio = constants.DOUBLE_LINE_WIDTH / 2

    def get_distance_in_meters(self, p1, p2):
        """Get distance between two points in meters."""
        return get_distance_in_meters(p1, p2, self)
    
    def convert_screen_to_mini_court_coordinates(self, screen_position, original_court_key_points):
        """Convert screen coordinates to mini court coordinates."""
        return convert_screen_to_mini_court_coordinates(screen_position, original_court_key_points)

    def convert_player_boxes_to_mini_court_coordinates(self, player_boxes, original_court_key_points):
        """Convert player bounding boxes to mini court coordinates."""
        return convert_player_boxes_to_mini_court_coordinates(player_boxes, original_court_key_points)
    
    def convert_ball_detections_to_mini_court_coordinates(self, ball_detections, original_court_key_points):
        """Convert ball detections to mini court coordinates."""
        return convert_ball_detections_to_mini_court_coordinates(ball_detections, original_court_key_points)

    def draw_court_video(self, frames, player_mini_court_coordinates, ball_mini_court_coordinates):
        """Draw the mini court visualization with player and ball positions."""
        return draw_court_video(frames, player_mini_court_coordinates, ball_mini_court_coordinates, self)

    def draw_screen_points(self, frames, player_boxes, court_keypoints):
        """Draw the players' feet positions and court keypoints on the main video frames."""
        return draw_screen_points(frames, player_boxes, court_keypoints)
    
    def compute_court_homography_matrix(self, court_keypoints):
        """Compute the court homography matrix in real world meters"""
        return compute_court_homography_matrix(court_keypoints, constants.DOUBLE_LINE_WIDTH, constants.COURT_LENGTH)