import torch
import cv2
import numpy as np
from ultralytics import YOLO

class Segmentation:
    def __init__(self, model_path, device):
        self.device = device
        self.model = YOLO(model_path)  
        self.model.model = self.model.model.to(self.device)  

    def predict(self, frame, conf=0.2, iou=0.5):
        """
        Perform segmentation prediction on a frame
        
        Args:
            frame: Input image/frame (numpy array)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            
        Returns:
            results: YOLO results object containing segmentation masks
        """
        results = self.model.predict(frame, conf=conf, iou=iou, verbose=False)
        return results[0]  # Return first result since we're processing single frame

    def get_masks(self, results):
        """
        Extract segmentation masks from results
        
        Args:
            results: YOLO results object
            
        Returns:
            masks: List of segmentation masks
            boxes: List of bounding boxes
            classes: List of class labels
            confidences: List of confidence scores
        """
        masks = []
        boxes = []
        classes = []
        confidences = []
        
        if results.masks is not None:
            for i in range(len(results.masks)):
                mask = results.masks[i].data.cpu().numpy()  # Get mask data
                box = results.boxes[i].xyxy.cpu().numpy()[0]  # Get bounding box
                cls = int(results.boxes[i].cls.cpu().numpy()[0])  # Get class
                conf = float(results.boxes[i].conf.cpu().numpy()[0])  # Get confidence
                
                masks.append(mask)
                boxes.append(box)
                classes.append(cls)
                confidences.append(conf)
                
        return masks, boxes, classes, confidences

    def draw_segmentation(self, frame, results, alpha=0.5, colors=None):
        """
        Draw segmentation masks on the frame
        
        Args:
            frame: Input frame
            results: YOLO results object
            alpha: Transparency for overlay
            colors: List of colors for different classes
            
        Returns:
            frame: Frame with segmentation overlays
        """
        if results.masks is None:
            return frame
            
        # Generate colors if not provided
        if colors is None:
            np.random.seed(42)  # For consistent colors
            colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(100)]
        
        # Create a copy of the frame for overlay
        overlay = frame.copy()
        
        for i in range(len(results.masks)):
            mask = results.masks[i].data.cpu().numpy()[0]
            cls = int(results.boxes[i].cls.cpu().numpy()[0])
            conf = float(results.boxes[i].conf.cpu().numpy()[0])
            
            print(mask.shape)
            print(frame.shape[1], frame.shape[0])            # Resize mask to frame size
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask = (mask > 0.5).astype(np.uint8)
            
            # Create colored mask
            color_mask = np.zeros_like(frame)
            color_mask[mask == 1] = colors[cls % len(colors)]
            
            # Apply mask overlay
            frame = cv2.addWeighted(frame, 1-alpha, color_mask, alpha, 0)
            
            # Draw bounding box
            box = results.boxes[i].xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[cls % len(colors)], 2)
            
            # Add label
            label = f"{results.names[cls]}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls % len(colors)], 2)
        
        return frame

    def extract_largest_mask(self, results):
        """
        Extract the largest segmentation mask from results
        
        Args:
            results: YOLO results object
            
        Returns:
            largest_mask: Largest mask as numpy array, or None if no masks
        """
        if results.masks is None:
            return None
            
        largest_area = 0
        largest_mask = None
        
        for mask in results.masks:
            mask_data = mask.data.cpu().numpy()
            area = np.sum(mask_data)
            if area > largest_area:
                largest_area = area
                largest_mask = mask_data
                
        return largest_mask

    def mask_to_polygon(self, mask, simplify_tolerance=2.0):
        """
        Convert mask to polygon coordinates
        
        Args:
            mask: Binary mask
            simplify_tolerance: Tolerance for polygon simplification
            
        Returns:
            polygons: List of polygon coordinates
        """
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            # Simplify polygon
            epsilon = simplify_tolerance
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygons.append(approx.reshape(-1, 2).tolist())
            
        return polygons

    def draw_keypoints(self, frame, keypoints):
        """
        Draw keypoints on frame (placeholder for compatibility)
        
        Args:
            frame: Input frame
            keypoints: List of keypoint coordinates
            
        Returns:
            frame: Frame with keypoints drawn
        """
        if keypoints is None:
            return frame
            
        for i, (x, y) in enumerate(keypoints):
            if x is not None and y is not None:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (int(x)+10, int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame