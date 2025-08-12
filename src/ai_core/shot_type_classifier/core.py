from .human_pose_extractor import HumanPoseExtractor
from .shot_counter import ShotCounter
import numpy as np
import torch
from .model_architecture import GRUModel

class ShotTypeClassifier:
    def __init__(self, model_path: str, pose_extractor_model_path: str, device):
        self.NB_IMAGES = 30
        try:
            self.model = GRUModel(model_path, device=device)
            print("Successfully loaded model")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        self.human_pose_extractor = HumanPoseExtractor(pose_extractor_model_path)
        self.shot_counter = ShotCounter()

    def classify(self, frames, bboxes, flip=False) -> str:
        features_pool = []
        for i in range(len(frames)):
            bbox = bboxes[i]
            if bbox is None:
                continue
            frame = frames[i]
            self.human_pose_extractor.extract_from_bbox(frame, bbox)
                        
            # Remove specific keypoints (left_eye, right_eye, left_ear, right_ear) - indices 1, 2, 3, 4
            keypoints = self.human_pose_extractor.keypoints_with_scores[0][0]  # Remove batch dimension
            print(f"Original keypoints shape: {keypoints.shape}")
            print(f"Original keypoints size: {keypoints.size}")
            
            # Keep only keypoints 0, 5-16 (nose, shoulders, elbows, wrists, hips, knees, ankles)
            filtered_keypoints = np.vstack([keypoints[0:1], keypoints[5:17]])  # 13 keypoints total
            
            # Debug: print the shape to understand the structure
            print(f"Filtered keypoints shape: {filtered_keypoints.shape}")
            print(f"Filtered keypoints size: {filtered_keypoints.size}")
            
            # Dynamically calculate the number of keypoints
            num_keypoints = filtered_keypoints.shape[0]
            features = filtered_keypoints.reshape(13, 3)


            if flip:
                features[:, 1] = 1 - features[:, 1]
            
            features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)

            features_pool.append(features)

            if len(features_pool) == self.NB_IMAGES:
                features_seq = np.array(features_pool).reshape(1, self.NB_IMAGES, 26)
                assert features_seq.shape == (1, 30, 26)
                
                # Convert numpy array to PyTorch tensor and ensure correct data type
                features_tensor = torch.tensor(features_seq, dtype=torch.float32)
                
                # Use the model's forward method (PyTorch standard)
                with torch.no_grad():
                    probs_tensor = self.model.forward(features_tensor)
                
                # Convert PyTorch tensor to numpy array and get the first (and only) batch
                probs = probs_tensor.detach().cpu().numpy()[0]
                self.shot_counter.update(probs, i)

                # Give space to pool
                features_pool = features_pool[1:]
        return self.shot_counter.results