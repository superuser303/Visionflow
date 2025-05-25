import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        return results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]