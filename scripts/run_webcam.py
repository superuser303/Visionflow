import cv2
import argparse
import numpy as np
from src.detection.yolo_wrapper import YOLODetector
from src.pose.mediapipe_wrapper import PoseEstimator
import mediapipe as mp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, path for video file)')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model path')
    args = parser.parse_args()
    
    # Initialize models
    print("Loading models...")
    detector = YOLODetector(args.model)
    pose_estimator = PoseEstimator()
    
    # Initialize video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
    
    print("Starting VisionFlow... Press 'q' to quit")
    
    # Drawing utilities for pose
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Make a copy for annotations
        annotated_frame = frame.copy()
        
        try:
            # Object Detection
            detections = detector.detect(frame)
            
            # Draw detection boxes
            if len(detections) > 0:
                for detection in detections:
                    x1, y1, x2, y2, conf, cls = detection
                    if conf > 0.5:  # Confidence threshold
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Draw label
                        class_name = detector.class_names[int(cls)]
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Pose Estimation
            landmarks = pose_estimator.estimate(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Draw pose landmarks
            if landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            
            # Add info text
            cv2.putText(annotated_frame, f"Detections: {len(detections)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "Press 'q' to quit", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        except Exception as e:
            print(f"Processing error: {e}")
            # Show original frame if processing fails
            annotated_frame = frame
        
        # Display the frame
        cv2.imshow('VisionFlow - Real-time Detection & Pose', annotated_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("VisionFlow stopped.")

if __name__ == "__main__":
    main()
