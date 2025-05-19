import cv2
import pytest
from src.pose.mediapipe_wrapper import PoseEstimator

@pytest.fixture
def sample_frame():
    return cv2.imread("data/samples/pose_sample.jpg")

def test_pose_estimation(sample_frame):
    estimator = PoseEstimator()
    landmarks = estimator.estimate(sample_frame)
    assert landmarks is not None, "Failed to detect pose"