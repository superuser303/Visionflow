from dash import Dash, html
import os
from src.detection.yolo_wrapper import YOLODetector
import logging

app = Dash(__name__)
app.server.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fallback-key')
logging.basicConfig(level=logging.INFO)

# Preload YOLO model to avoid timeout
try:
    detector = YOLODetector("yolov8n.pt")
    app.logger.info("YOLO model loaded")
except Exception as e:
    app.logger.error(f"Model load failed: {str(e)}")

# Basic Dash layout for web interface
app.layout = html.Div([
    html.H1("Visionflow: Object Detection"),
    html.P("Welcome to the Visionflow web app. Use the /predict endpoint for object detection.")
])

@app.server.route('/health')
def health():
    return {"status": "healthy"}, 200

@app.server.route('/predict', methods=['POST'])
def predict():
    try:
        # Placeholder for YOLO/MediaPipe/COCO API logic
        app.logger.info("Prediction started")
        return {"status": "success", "prediction": "example"}, 200
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return {"status": "error", "message": str(e)}, 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run_server(host='0.0.0.0', port=port)
