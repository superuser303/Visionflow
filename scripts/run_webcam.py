import cv2
from src.detection import YOLODetector
from src.pose import PoseEstimator

def main():
    detector = YOLODetector()
    pose_estimator = PoseEstimator()
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Detection
        detections = detector.detect(frame)
        
        # Pose estimation
        landmarks = pose_estimator.estimate(frame)
        
        cv2.imshow('VisionFlow', frame)
        if cv2.waitKey(1) == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()