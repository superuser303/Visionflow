import os
import onnx
from ultralytics import YOLO

def export_to_onnx():
    model = YOLO("models/custom/yolov8_custom.pt")
    model.export(format="onnx", imgsz=(640, 640))

def convert_to_tensorrt():
    os.system("trtexec --onnx=models/custom/yolov8_custom.onnx "
              "--saveEngine=models/custom/yolov8_custom.engine --fp16")

if __name__ == "__main__":
    export_to_onnx()
    convert_to_tensorrt()