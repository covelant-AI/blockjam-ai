from .model import BallTrackerNet
import torch
import cv2
from tqdm import tqdm
import numpy as np
from scipy.spatial import distance
from itertools import groupby
import pandas as pd
import gc

class TrackNetBallTracker:
    def __init__(self, model_path, device):
        self.device = device
        self.model = BallTrackerNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def convert_track_to_video_resolution(self, ball_track, video_info, ball_track_resolution=(1280, 720)):
        """ Convert ball track to video resolution
        :params
            ball_track: list of detected ball points
            video_info: video info
            ball_track_resolution: ball track resolution
        """
        converted_track = []
        for x, y in ball_track:
            if x is not None and y is not None:
                # Convert coordinates to video resolution
                converted_x = float(x * video_info['width'] / ball_track_resolution[0])
                converted_y = float(y * video_info['height'] / ball_track_resolution[1])
                converted_track.append((converted_x, converted_y))
            else:
                # Keep None values as is
                converted_track.append((x, y))
        return converted_track

    def preprocess_video_resolution(self, frames, ball_track_resolution=(1280, 720)):
        """ Preprocess video resolution
        :params
            frames: list of consecutive video frames
            video_info: video info
            ball_track_resolution: ball track resolution
        """
        frames = [cv2.resize(frame, (ball_track_resolution[0], ball_track_resolution[1])) for frame in frames]
        return frames

    def postprocess(self, feature_map, scale=2):
        feature_map *= 255
        feature_map = feature_map.reshape((360, 640))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                maxRadius=7)
        x,y = None, None
        if circles is not None:
            if len(circles) == 1:
                x = circles[0][0][0]*scale
                y = circles[0][0][1]*scale
        return x, y

    def infer_model(self,frames):
        """ Run pretrained model on a consecutive list of frames    
        :params
            frames: list of consecutive video frames
        :return    
            ball_track: list of detected ball points
            dists: list of euclidean distances between two neighbouring ball points
        """
        height = 360
        width = 640
        dists = [-1]*2
        ball_track = [(None,None)]*2
        
        for num in tqdm(range(2, len(frames))):
            img = cv2.resize(frames[num], (width, height))
            img_prev = cv2.resize(frames[num-1], (width, height))
            img_preprev = cv2.resize(frames[num-2], (width, height))
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32)/255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            # Create tensor and run inference
            input_tensor = torch.from_numpy(inp).float().to(self.device)
            with torch.no_grad():  # Disable gradient computation for inference
                out = self.model(input_tensor)
                output = out.argmax(dim=1).detach().cpu().numpy()
            
            # Clear GPU tensors immediately (but don't call empty_cache every frame)
            del input_tensor, out
            
            x_pred, y_pred = self.postprocess(output)
            ball_track.append((x_pred, y_pred))

            if ball_track[-1][0] and ball_track[-2][0]:
                dist = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                dist = -1
            dists.append(dist)
            
            # Periodic memory cleanup every 50 frames (much less frequent)
            if num % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        return ball_track, dists
    

    def split_track(self, ball_track, max_gap=4, max_dist_gap=80, min_track=5):
        """ Split ball track into several subtracks in each of which we will perform
        ball interpolation.    
        :params
            ball_track: list of detected ball points
            max_gap: maximun number of coherent None values for interpolation  
            max_dist_gap: maximum distance at which neighboring points remain in one subtrack
            min_track: minimum number of frames in each subtrack    
        :return
            result: list of subtrack indexes    
        """
        list_det = [0 if x[0] else 1 for x in ball_track]
        groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

        cursor = 0
        min_value = 0
        result = []
        for i, (k, l) in enumerate(groups):
            if (k == 1) & (i > 0) & (i < len(groups) - 1):
                dist = distance.euclidean(ball_track[cursor-1], ball_track[cursor+l])
                if (l >=max_gap) | (dist/l > max_dist_gap):
                    if cursor - min_value > min_track:
                        result.append([min_value, cursor])
                        min_value = cursor + l - 1        
            cursor += l
        if len(list_det) - min_value > min_track: 
            result.append([min_value, len(list_det)]) 
        return result
    
    def extrapolation(self, ball_track):
        subtracks = self.split_track(ball_track)
        for r in subtracks:
            ball_subtrack = ball_track[r[0]:r[1]]
            ball_subtrack = self.interpolation(ball_subtrack)
            ball_track[r[0]:r[1]] = ball_subtrack
        return ball_track
    
    def interpolation(self,coords):
        """ Run ball interpolation in one subtrack    
        :params
            coords: list of ball coordinates of one subtrack    
        :return
            track: list of interpolated ball coordinates of one subtrack
        """
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
        y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

        nons, yy = nan_helper(x)
        x[nons]= np.interp(yy(nons), yy(~nons), x[~nons])
        nans, xx = nan_helper(y)
        y[nans]= np.interp(xx(nans), xx(~nans), y[~nans])

        track = [*zip(x,y)]
        return track

    def remove_outliers(self, ball_track, dists, max_dist = 100):
        """ Remove outliers from model prediction    
        :params
            ball_track: list of detected ball points
            dists: list of euclidean distances between two neighbouring ball points
            max_dist: maximum distance between two neighbouring ball points
        :return
            ball_track: list of ball points
        """
        outliers = list(np.where(np.array(dists) > max_dist)[0])
        for i in outliers:
            if (i < len(dists) - 1) and ((dists[i+1] > max_dist) | (dists[i+1] == -1)):       
                ball_track[i] = (None, None)
            elif (i > 0) and (dists[i-1] == -1):
                ball_track[i-1] = (None, None)
        return ball_track
    
    def write_track(self, frames, ball_track, path_output_video, fps, trace=7):
        """ Write .avi file with detected ball tracks
        :params
            frames: list of original video frames
            ball_track: list of ball coordinates
            path_output_video: path to output video
            fps: frames per second
            trace: number of frames with detected trace
        """
        height, width = frames[0].shape[:2]
        out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), 
                            fps, (width, height))
        for num in range(len(frames)):
            frame = frames[num]
            for i in range(trace):
                if (num-i > 0):
                    if ball_track[num-i][0]:
                        x = int(ball_track[num-i][0])
                        y = int(ball_track[num-i][1])
                        frame = cv2.circle(frame, (x,y), radius=0, color=(0, 0, 255), thickness=10-i)
                    else:
                        break
            out.write(frame) 
        out.release()


    def interpolate_ball_track(self, ball_track):
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
    

    def detect_frames(self, frames, video_info):
        frames = self.preprocess_video_resolution(frames)
        ball_track, dists = self.infer_model(frames)
        ball_track = self.remove_outliers(ball_track, dists)
        ball_track = self.convert_track_to_video_resolution(ball_track, video_info)
        return ball_track