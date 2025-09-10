import pandas as pd
from ai_core.mini_court import MiniCourt
import numpy as np

def smooth_speed_data(speeds:list[float], alpha=0.3, fps=30, time_step=0.5, min_speed_kmh=0, max_speed_kmh=200) -> list[float] | None:
    df = pd.DataFrame(speeds, columns=['speed'])
    #check if any of the values are not None
    if not any(speed is not None and not np.isnan(speed) for speed in speeds):
        return None
    window_size = int(fps * time_step * 0.5)
    df['smooth_speed'] = (
        df['speed']
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
        .fillna(method='bfill')
        .fillna(method='ffill')
    )
    # df['smooth_speed'] = df['smooth_speed'].ewm(alpha=alpha, adjust=False).mean()
    df['smooth_speed'] = df['smooth_speed'].clip(lower=min_speed_kmh, upper=max_speed_kmh)
    target_times = [i * time_step for i in range(int(len(speeds) / (fps * time_step)) + 1)]
    sampled_indices = [int(t * fps) for t in target_times if int(t * fps) < len(speeds)]
    return df.iloc[sampled_indices]['smooth_speed'].tolist()

def _calculate_speed_data(detections, fps, time_step, distance_func, max_speed_kmh):
    """
    Generic function to calculate speed data from detections.
    
    Args:
        detections: List of detection data
        fps: Frames per second
        time_step: Time step for calculations
        mini_court: Mini court object for distance calculations
    
    Returns:
        List of speeds
    """
    max_dist_per_frame = max_speed_kmh / (fps * 3.6)
    try:
        distances = [0]
        for i in range(1, len(detections)):
            curr = detections[i]
            prev = detections[i-1]
            if prev is None or curr is None or any(bd is None for bd in prev) or any(bd is None for bd in curr):
                distances.append(None)
            else:
                dist = distance_func(prev, curr)
                distances.append(min(dist, max_dist_per_frame))

        df = pd.DataFrame(detections)
        df['dist'] = distances
        
        # Apply minimal smoothing to reduce noise while preserving variations
        # Use a smaller window size and exponential moving average for more responsive smoothing
        window_size = 3
        df['smooth_dist'] = (
            df['dist']
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
            .fillna(method='bfill')
            .fillna(method='ffill')
        )
        
        # Apply exponential moving average for additional smoothing without over-smoothing
        alpha = 0.3  # Lower alpha = more smoothing, higher alpha = less smoothing
        df['smooth_dist'] = df['smooth_dist'].ewm(alpha=alpha, adjust=False).mean()
        
        # Ensure smooth_dist is non-negative and within reasonable bounds
        df['smooth_dist'] = df['smooth_dist'].clip(lower=0, upper=max_dist_per_frame)

        df['speed'] = (df['smooth_dist'] * fps * 3.6)  # Speed in km/h

        target_times = [i * time_step for i in range(int(len(detections) / (fps * time_step)) + 1)]
        sampled_indices = [int(t * fps) for t in target_times if int(t * fps) < len(detections)]
        speeds = df.iloc[sampled_indices]['speed'].tolist()
        return speeds
    except Exception as e:
        raise Exception(f"Error in _calculate_speed_data: {e}")

def get_player_speed_data(player_mini_court_detections, fps, time_step, mini_court: MiniCourt, max_speed_kmh=16):
    def extract_points(detection, key):
        # Try the key as-is, then as string, then as integer
        return (detection.get(key) or 
                detection.get(str(key)) or 
                detection.get(int(key)))
    
    p1_points = [extract_points(detection, 1) for detection in player_mini_court_detections]
    p2_points = [extract_points(detection, 2) for detection in player_mini_court_detections]
    
    p1_speeds = _calculate_speed_data(p1_points, fps, time_step, mini_court.get_distance_in_meters, max_speed_kmh)
    p2_speeds = _calculate_speed_data(p2_points, fps, time_step, mini_court.get_distance_in_meters, max_speed_kmh)

    #print min, max and average of p1_speeds and p2_speeds
    print(f"p1_speeds: min: {min(p1_speeds)}, max: {max(p1_speeds)}, average: {sum(p1_speeds)/len(p1_speeds)}")
    print(f"p2_speeds: min: {min(p2_speeds)}, max: {max(p2_speeds)}, average: {sum(p2_speeds)/len(p2_speeds)}")

    return p1_speeds, p2_speeds


def get_ball_speed_data(ball_3d_positions, fps, time_step, max_speed_kmh=200):
    def dist_3d(prev, curr):
        x1, y1, z1 = prev
        x2, y2, z2 = curr
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
    ball_speeds = _calculate_speed_data(ball_3d_positions, fps, time_step, dist_3d, max_speed_kmh)
    
    print(f"ball_speeds: min: {min(ball_speeds)}, max: {max(ball_speeds)}, average: {sum(ball_speeds)/len(ball_speeds)}")
    
    return ball_speeds
