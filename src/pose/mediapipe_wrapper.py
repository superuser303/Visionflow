import mediapipe as mp

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.7)
        
    def estimate(self, frame):
        results = self.mp_pose.process(frame)
        return results.pose_landmarks