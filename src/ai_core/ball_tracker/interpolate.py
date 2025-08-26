import pandas as pd
import numpy as np

def interpolate_ball_track(ball_track):
    """
    Interpolate ball track using velocity-based interpolation with constraints.
    This function handles missing ball detections by considering ball physics and movement patterns.
    """
    
    # Convert to pandas DataFrame
    df_ball_track = pd.DataFrame(ball_track, columns=['center_x', 'center_y'])
    
    # Normalize coordinates to [0,1] range
    max_x = df_ball_track['center_x'].max()
    max_y = df_ball_track['center_y'].max()
    df_ball_track['center_x'] = df_ball_track['center_x'] / max_x
    df_ball_track['center_y'] = df_ball_track['center_y'] / max_y
    
    # Calculate velocities (normalized)
    df_ball_track['vx'] = df_ball_track['center_x'].diff()
    df_ball_track['vy'] = df_ball_track['center_y'].diff()
    
    # Interpolate missing values with constraints
    for col in ['center_x', 'center_y']:
        # First pass: linear interpolation
        df_ball_track[col] = df_ball_track[col].interpolate(method='linear')
        
        # Second pass: forward fill for any remaining NaNs
        df_ball_track[col] = df_ball_track[col].bfill()
    
    # Apply velocity constraints (normalized)
    max_velocity = 0.1  # Maximum normalized velocity per frame
    for i in range(1, len(df_ball_track)):
        if pd.isna(df_ball_track.iloc[i-1]['vx']) or pd.isna(df_ball_track.iloc[i]['vx']):
            continue
            
        # Limit velocity changes
        vx = df_ball_track.iloc[i]['center_x'] - df_ball_track.iloc[i-1]['center_x']
        vy = df_ball_track.iloc[i]['center_y'] - df_ball_track.iloc[i-1]['center_y']
        
        # Apply velocity constraints
        if abs(vx) > max_velocity:
            df_ball_track.iloc[i, df_ball_track.columns.get_loc('center_x')] = (
                df_ball_track.iloc[i-1]['center_x'] + np.sign(vx) * max_velocity
            )
        if abs(vy) > max_velocity:
            df_ball_track.iloc[i, df_ball_track.columns.get_loc('center_y')] = (
                df_ball_track.iloc[i-1]['center_y'] + np.sign(vy) * max_velocity
            )
    
    # Apply smoothing to reduce jitter
    window_size = 1
    for col in ['center_x', 'center_y']:
        df_ball_track[col] = df_ball_track[col].rolling(window=window_size, center=True).mean()
        # Fill NaN values at the edges
        df_ball_track[col] = df_ball_track[col].bfill().ffill()
    
    # Convert back to original coordinate system
    df_ball_track['center_x'] = df_ball_track['center_x'] * max_x
    df_ball_track['center_y'] = df_ball_track['center_y'] * max_y
    
    # Convert back to original format
    ball_track = [x for x in df_ball_track[['center_x', 'center_y']].to_numpy().tolist()]
    return ball_track