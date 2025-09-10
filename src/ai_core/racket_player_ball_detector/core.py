from ultralytics import YOLO 
import cv2
import sys
import numpy as np
sys.path.append('../')

class RacketPlayerBallDetector:
    def __init__(self, model_path, device):
        self.device = device
        self.model = YOLO(model_path)  
        self.model.model = self.model.model.to(self.device)  

        print(next(self.model.model.parameters()).device)  # should print

    def detect_frame(self, frame):
        results = self.model.predict(frame, device=self.device, imgsz=640, conf=0.6)[0]
        id_name_dict = results.names
        players = []
        rackets = []
        balls = []

        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name.lower() == "player":
                players.append(result)
            elif object_cls_name.lower() == "racket":
                rackets.append(result)
            elif object_cls_name.lower() == "ball":
                balls.append(result)

        return players, rackets, balls
    
    def detect_frames(self, frames):
        results = self.model.predict(frames, device=self.device, imgsz=640, conf=0.6)

        data = []
        for result in results:
            id_name_dict = result.names
            players = []
            rackets = []
            balls = []

            for box in result.boxes:
                result = box.xyxy.tolist()[0]
                object_cls_id = box.cls.tolist()[0]
                object_cls_name = id_name_dict[object_cls_id]
                if object_cls_name.lower() == "player":
                    players.append(result)
                elif object_cls_name.lower() == "racket":
                    rackets.append(result)
                elif object_cls_name.lower() == "ball":
                    balls.append(result)

            data.append({
                'players': players,
                'rackets': rackets,
                'balls': balls
            })

        return data