import pandas as pd
from ai_core.mini_court import MiniCourt

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
        
        # Apply a rolling average to the distance
        window_size = 5
        # Apply a rolling average to the distance and fill NaNs
        df['smooth_dist'] = (
            df['dist']
            .rolling(window=window_size, center=True)
            .mean()
            .interpolate(method='polynomial', order=2, limit_direction='both')
            .combine_first(df['dist']) 
            .interpolate(method='linear', limit_direction='both')
        )
        
        # Ensure smooth_dist is non-negative
        df['smooth_dist'] = df['smooth_dist'].clip(lower=0, upper=max_dist_per_frame)

        df['speed'] =  (df['smooth_dist'] * fps * 3.6) # Speed in km/h

        speeds = df[df.index % (fps * time_step) == 0]['speed'].tolist()
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
