import cv2
import numpy as np
import tensorflow as tf
import os

# Configure TensorFlow to avoid GPU memory issues
def configure_tensorflow():
    """Configure TensorFlow to avoid GPU memory issues"""
    try:
        # Set memory growth to avoid allocating all GPU memory at once
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        else:
            print("No GPU devices found, using CPU")
    except Exception as e:
        print(f"Warning: Could not configure GPU memory growth: {e}")
        print("Falling back to CPU operations")

# Configure TensorFlow on import
configure_tensorflow()

def clear_gpu_memory():
    """Clear GPU memory if available"""
    try:
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
            print("GPU memory cleared")
    except Exception as e:
        print(f"Warning: Could not clear GPU memory: {e}")

class HumanPoseExtractor:
    """
    Defines mapping between movenet key points and human readable body points
    with realistic edges to be drawn"""

    EDGES = {
        (0, 1): "m",
        (0, 2): "c",
        (1, 3): "m",
        (2, 4): "c",
        (0, 5): "m",
        (0, 6): "c",
        (5, 7): "m",
        (7, 9): "m",
        (6, 8): "c",
        (8, 10): "c",
        (5, 6): "y",
        (5, 11): "m",
        (6, 12): "c",
        (11, 12): "y",
        (11, 13): "m",
        (13, 15): "m",
        (12, 14): "c",
        (14, 16): "c",
    }

    COLORS = {"c": (255, 255, 0), "m": (255, 0, 255), "y": (0, 255, 255)}

    # Dictionary that maps from joint names to keypoint indices.
    KEYPOINT_DICT = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

    def __init__(self, model_path: str):
        # Initialize the TFLite interpreter with CPU execution to avoid GPU memory issues
        try:
            # Force CPU execution for TFLite interpreter
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=4  # Use multiple CPU threads instead of GPU
            )
            self.interpreter.allocate_tensors()
            print("TFLite interpreter initialized with CPU execution")
        except Exception as e:
            print(f"Warning: Could not initialize TFLite interpreter with CPU: {e}")
            # Fallback to default initialization
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
    
    def extract_from_bbox(self, frame, bbox):
        """
        Extract pose from a specific bounding box region
        
        Args:
            frame: Input frame
            bbox: Bounding box as (x1, y1, x2, y2) where (x1, y1) is top-left and (x2, y2) is bottom-right
        """
        if len(bbox) != 4:
            raise ValueError("bbox must be a tuple of 4 values: (x1, y1, x2, y2)")
        bbox = [int(b) for b in bbox]
        x1, y1, x2, y2 = bbox
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        
        # Ensure bbox is within frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        # Extract subframe from bbox
        subframe = frame[y:y+h, x:x+w]
        
        # Resize for model input using CPU operations to avoid GPU memory issues
        try:
            # Use OpenCV for resizing instead of TensorFlow to avoid GPU memory issues
            img = cv2.resize(subframe, (192, 192))
            img = np.expand_dims(img, axis=0)
            input_image = img.astype(np.uint8)
        except Exception as e:
            print(f"Warning: OpenCV resize failed, falling back to TensorFlow CPU: {e}")
            # Fallback to TensorFlow CPU operations
            with tf.device('/CPU:0'):
                img = tf.image.resize_with_pad(np.expand_dims(subframe, axis=0), 192, 192)
                input_image = tf.cast(img, dtype=tf.uint8).numpy()
        
        # Setup input and output
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # Make predictions
        self.interpreter.set_tensor(input_details[0]["index"], input_image)
        self.interpreter.invoke()
        self.keypoints_with_scores = self.interpreter.get_tensor(
            output_details[0]["index"]
        )
        
        # Transform keypoints to frame coordinates
        self.keypoints_pixels_frame = self._transform_bbox_keypoints_to_frame(
            self.keypoints_with_scores, x, y, w, h
        )
        
        return self.keypoints_pixels_frame
    
    def _transform_bbox_keypoints_to_frame(self, keypoints_from_tf, x, y, w, h):
        """
        Transform keypoints from bbox subframe coordinates to frame coordinates
        
        Args:
            keypoints_from_tf: Keypoints from tensorflow (normalized 0-1)
            x, y, w, h: Bbox coordinates and dimensions
        """
        # Convert normalized coordinates to bbox pixel coordinates
        keypoints_pixels_bbox = np.squeeze(
            np.multiply(keypoints_from_tf, [w, h, 1])
        )
        
        # Transform to frame coordinates
        keypoints_pixels_frame = keypoints_pixels_bbox.copy()
        keypoints_pixels_frame[:, 0] += y  # y coordinate
        keypoints_pixels_frame[:, 1] += x  # x coordinate
        
        return keypoints_pixels_frame
    
    def cleanup(self):
        """Clean up resources and clear memory"""
        try:
            if hasattr(self, 'interpreter'):
                del self.interpreter
            clear_gpu_memory()
            print("HumanPoseExtractor resources cleaned up")
        except Exception as e:
            print(f"Warning: Could not clean up resources: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()