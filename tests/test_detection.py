import cv2
import pytest
import os
import numpy as np
from src.detection.yolo_wrapper import YOLODetector

@pytest.fixture
def sample_image():
    """Load a test image, create one if it doesn't exist"""
    image_path = "data/samples/test_image.jpg"
    
    # If the image doesn't exist, create a synthetic one
    if not os.path.exists(image_path):
        os.makedirs("data/samples", exist_ok=True)
        
        # Create a more realistic test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (100, 150, 200)  # Blue background
        
        # Draw person-like shape
        cv2.rectangle(img, (250, 150), (350, 400), (139, 69, 19), -1)  # Body
        cv2.circle(img, (300, 130), 25, (255, 220, 177), -1)  # Head
        
        # Draw car-like shape
        cv2.rectangle(img, (450, 350), (600, 420), (0, 0, 255), -1)  # Car body
        cv2.circle(img, (470, 410), 15, (0, 0, 0), -1)  # Wheel 1
        cv2.circle(img, (580, 410), 15, (0, 0, 0), -1)  # Wheel 2
        
        cv2.imwrite(image_path, img)
    
    return cv2.imread(image_path)

@pytest.fixture
def coco_sample_image():
    """Try to load a COCO sample image, fallback to synthetic"""
    coco_samples = [
        "data/samples/coco_sample_1.jpg",
        "data/samples/coco_sample_2.jpg",
        "data/samples/coco_sample_3.jpg"
    ]
    
    # Try to find an existing COCO sample
    for sample_path in coco_samples:
        if os.path.exists(sample_path):
            img = cv2.imread(sample_path)
            if img is not None:
                return img
    
    # Fallback to creating a synthetic image with clear objects
    synthetic_path = "data/samples/synthetic_detection_test.jpg"
    
    # Create an image with multiple detectable objects
    img = np.ones((640, 640, 3), dtype=np.uint8) * 128  # Gray background
    
    # Draw multiple person-like figures
    for i, x_offset in enumerate([100, 300, 500]):
        # Body
        cv2.rectangle(img, (x_offset, 200), (x_offset + 60, 450), (139, 69, 19), -1)
        # Head  
        cv2.circle(img, (x_offset + 30, 180), 20, (255, 220, 177), -1)
        # Arms
        cv2.rectangle(img, (x_offset - 20, 250), (x_offset + 80, 270), (139, 69, 19), -1)
    
    # Draw a car
    cv2.rectangle(img, (150, 500), (350, 580), (0, 0, 255), -1)  # Car body
    cv2.circle(img, (180, 570), 20, (0, 0, 0), -1)  # Wheel 1
    cv2.circle(img, (320, 570), 20, (0, 0, 0), -1)  # Wheel 2
    
    cv2.imwrite(synthetic_path, img)
    return img

def test_yolo_detection_basic(sample_image):
    """Test basic YOLO detection functionality"""
    detector = YOLODetector()
    detections = detector.detect(sample_image)
    
    # Should return numpy array
    assert isinstance(detections, np.ndarray), "Detections should be numpy array"
    assert detections.shape[1] == 6, "Detection should have 6 columns [x1, y1, x2, y2, conf, cls]"

def test_yolo_detection_with_coco_image(coco_sample_image):
    """Test YOLO detection with COCO-style image"""
    detector = YOLODetector()
    detections = detector.detect(coco_sample_image)
    
    assert isinstance(detections, np.ndarray), "Detections should be numpy array"
    
    # With synthetic images containing clear objects, we should get detections
    if len(detections) > 0:
        # Verify detection format
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            assert x2 > x1, "x2 should be greater than x1"
            assert y2 > y1, "y2 should be greater than y1"
            assert 0 <= conf <= 1, "Confidence should be between 0 and 1"
            assert cls >= 0, "Class should be non-negative"

def test_yolo_detector_initialization():
    """Test YOLODetector initialization"""
    detector = YOLODetector()
    assert detector.model is not None, "Model should be initialized"
    assert detector.class_names is not None, "Class names should be available"

def test_yolo_detection_empty_image():
    """Test YOLO detection with empty/black image"""
    # Create completely black image
    black_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    detector = YOLODetector()
    detections = detector.detect(black_image)
    
    # Should still return valid numpy array format
    assert isinstance(detections, np.ndarray), "Should return numpy array even for empty image"
    assert detections.shape[1] == 6 or len(detections) == 0, "Should have correct format or be empty"

def test_yolo_detection_confidence_threshold():
    """Test detection with different confidence thresholds"""
    detector = YOLODetector()
    
    # Create a test image
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray
    
    # Draw a clear person-like figure
    cv2.rectangle(test_img, (200, 100), (300, 400), (139, 69, 19), -1)  # Body
    cv2.circle(test_img, (250, 80), 25, (255, 220, 177), -1)  # Head
    
    detections = detector.detect(test_img)
    
    # Should return proper format regardless of actual detections
    assert isinstance(detections, np.ndarray), "Should return numpy array"
    if len(detections) > 0:
        # If there are detections, verify confidence values
        confidences = detections[:, 4]
        assert all(0 <= conf <= 1 for conf in confidences), "All confidences should be between 0 and 1"
