# VisionFlow 🚀

Advanced Real-Time Object Detection & Action Recognition System

## Features
- Real-time multi-object tracking with YOLOv8
- 3D pose estimation using MediaPipe  
- Action recognition (SlowFast + ST-GCN)
- TensorRT optimization for edge deployment
- Docker support with GPU acceleration
- CI/CD pipeline with automated testing

## Prerequisites

- Python 3.8+ (recommended: 3.10)
- CUDA 11.8+ (for GPU acceleration)
- OpenCV 4.7+
- At least 4GB RAM
- Webcam or video files for testing

## Installation

### Option 1: Local Setup

1. Clone the repository:
```bash
git clone https://github.com/superuser303/VisionFlow
cd VisionFlow
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Set up Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # On Windows: set PYTHONPATH=%PYTHONPATH%;%cd%
```

### Option 2: Docker Setup

1. Build the container:
```bash
docker build -t visionflow:latest .
```

2. Run with GPU support:
```bash
docker run --gpus all -it --rm \
  -v /dev/video0:/dev/video0 \
  -v $(pwd)/data:/app/data \
  visionflow:latest
```

### Option 3: Dev Container (VS Code)

1. Open project in VS Code
2. Install "Dev Containers" extension
3. Press `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"

## Quick Start

### 1. Webcam Demo
```bash
python scripts/run_webcam.py
```

### 2. Process Video File
```bash
python scripts/run_webcam.py --source path/to/video.mp4
```

### 3. Train Custom Model
```bash
python scripts/train.py --config configs/model_config.yaml --data data/dataset.yaml
```

### 4. Jupyter Notebook Demo
```bash
jupyter notebook notebooks/VisionFlow_Demo.ipynb
```

## Configuration

Edit `configs/model_config.yaml` to customize:

- **Detection**: Model type, confidence thresholds, input size
- **Pose Estimation**: MediaPipe complexity settings  
- **Tracking**: DeepSORT parameters
- **Training**: Batch size, learning rate, epochs

## Dataset Setup

### For Training Custom Models:

1. Organize your dataset:
```
data/custom_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

2. Update `data/dataset.yaml`:
```yaml
path: ../data/custom_dataset
train: train/images
val: val/images
names:
  0: person
  1: car
  2: custom_class
```

### For Testing:
The system automatically downloads COCO sample images for testing, or creates synthetic test data if download fails.

## Usage Examples

### Basic Detection
```python
from src.detection.yolo_wrapper import YOLODetector
import cv2

detector = YOLODetector("yolov8n.pt")
frame = cv2.imread("test_image.jpg")
detections = detector.detect(frame)
```

### Pose Estimation
```python
from src.pose.mediapipe_wrapper import PoseEstimator
import cv2

estimator = PoseEstimator()
frame = cv2.imread("person_image.jpg")
landmarks = estimator.estimate(frame)
```

### Video Processing
```python
from src.utils.video_processor import VideoHandler

video = VideoHandler("input_video.mp4")
while True:
    ret, frame = video.read()
    if not ret: break
    # Process frame here
```

## Development

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov flake8

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality
```bash
# Linting
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Code formatting
black src/ tests/ scripts/
```

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```

## Deployment

### Edge Devices (Jetson/RPi)
```bash
python scripts/deploy_edge.py  # Converts to ONNX/TensorRT
```

### Export Models
```bash
# Export to ONNX
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"

# Export to TensorRT (requires TensorRT)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='engine')"
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **CUDA Not Found**:
   - Install CUDA 11.8+
   - Verify: `nvidia-smi`
   - Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

3. **Camera Not Working**:
   ```bash
   # Check available cameras
   python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
   ```

4. **Memory Issues**:
   - Reduce batch size in `configs/model_config.yaml`
   - Use smaller model: `yolov8n.pt` instead of `yolov8x.pt`

5. **Missing Test Data**:
   ```bash
   python -c "from src.utils.coco_utils import setup_test_dataset; setup_test_dataset('data/samples')"
   ```

## Performance Optimization

### For Real-time Processing:
- Use smaller models (`yolov8n.pt`)
- Reduce input resolution (320x320)
- Enable TensorRT optimization
- Use GPU acceleration

### For Accuracy:
- Use larger models (`yolov8x.pt`)  
- Higher input resolution (640x640)
- Lower confidence thresholds
- Custom training on domain-specific data

## API Reference

### Detection API
- `YOLODetector.detect(frame)` → Returns bounding boxes
- `YOLODetector.class_names` → Class name mapping

### Pose API  
- `PoseEstimator.estimate(frame)` → Returns pose landmarks
- MediaPipe pose connections available

### Video API
- `VideoHandler(source)` → Initialize video capture
- `VideoHandler.read()` → Get next frame
- `VideoHandler.get_properties()` → Video metadata

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Run tests: `pytest tests/`
4. Submit pull request

## License

MIT License - see [LICENSE](LICENSE) file

## Changelog

### v0.1.0
- Initial release with YOLOv8 + MediaPipe
- Docker support
- CI/CD pipeline
- Test dataset generation

---

**Need Help?** 
- Check the [Issues](https://github.com/superuser303/VisionFlow/issues) page
- Run the Jupyter notebook demo for examples
- Review test files for usage patterns
