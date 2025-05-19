import cv2

class VideoHandler:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
    
    def read(self):
        return self.cap.read()
    
    def release(self):
        self.cap.release()
    
    def get_properties(self):
        return {
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }