import cv2
import pytest
from src.detection.yolo_wrapper import YOLODetector

@pytest.fixture
def sample_image():
    return cv2.imread("data/samples/test_image.jpg")

def test_yolo_detection(sample_image):
    detector = YOLODetector()
    detections = detector.detect(sample_image)
    assert len(detections) > 0, "No detections made"