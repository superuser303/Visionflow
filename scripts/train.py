import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
from ultralytics.yolo.utils.torch_utils import select_device

def train_model(config_path: str, custom_data: str = None):
    """
    Train YOLOv8 model with custom configurations
    
    Args:
        config_path (str): Path to model_config.yaml
        custom_data (str): Optional custom dataset.yaml path
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    device = select_device(config['device'])
    model = YOLO(config['detection']['model']).to(device)
    
    # Training parameters
    params = {
        'data': custom_data or config['data'],
        'epochs': config['training']['epochs'],
        'imgsz': config['detection']['input_size'],
        'batch': config['training']['batch_size'],
        'optimizer': config['training']['optimizer'],
        'lr0': config['training']['learning_rate'],
        'name': config['training']['experiment_name'],
        'patience': config['training']['early_stopping'],
        'save': True,
        'exist_ok': True
    }

    # Start training
    results = model.train(**params)
    
    # Export best model
    model.export(format='onnx')  # Export to ONNX for deployment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--data', type=str, help='Custom dataset config path')
    args = parser.parse_args()
    
    train_model(args.config, args.data)