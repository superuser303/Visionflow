import os
import json
import requests
from PIL import Image, ImageDraw
import numpy as np
from pycocotools.coco import COCO
from typing import List, Dict, Tuple, Optional

class COCODatasetHandler:
    """Handle COCO dataset operations for VisionFlow"""
    
    def __init__(self, annotations_file: Optional[str] = None, images_dir: Optional[str] = None):
        """
        Initialize COCO dataset handler
        
        Args:
            annotations_file: Path to COCO annotations JSON file
            images_dir: Directory containing COCO images
        """
        self.annotations_file = annotations_file
        self.images_dir = images_dir
        self.coco = None
        
        if annotations_file and os.path.exists(annotations_file):
            self.coco = COCO(annotations_file)
    
    def download_sample_images(self, output_dir: str, num_images: int = 5) -> List[str]:
        """
        Download sample COCO images for testing
        
        Args:
            output_dir: Directory to save images
            num_images: Number of images to download
            
        Returns:
            List of downloaded image paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # COCO sample image URLs (from COCO validation set)
        sample_urls = [
            "http://images.cocodataset.org/val2017/000000039769.jpg",  # Cats on a bench
            "http://images.cocodataset.org/val2017/000000037777.jpg",  # Person on motorcycle
            "http://images.cocodataset.org/val2017/000000581929.jpg",  # Person with umbrella
            "http://images.cocodataset.org/val2017/000000579818.jpg",  # Baseball player
            "http://images.cocodataset.org/val2017/000000438862.jpg",  # Surfer
        ]
        
        downloaded_paths = []
        
        for i, url in enumerate(sample_urls[:num_images]):
            try:
                filename = f"coco_sample_{i+1}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                if not os.path.exists(filepath):
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                
                downloaded_paths.append(filepath)
                print(f"Downloaded: {filename}")
                
            except Exception as e:
                print(f"Failed to download image {i+1}: {e}")
                # Create a synthetic test image as fallback
                fallback_path = self._create_synthetic_image(output_dir, f"synthetic_{i+1}.jpg")
                downloaded_paths.append(fallback_path)
        
        return downloaded_paths
    
    def _create_synthetic_image(self, output_dir: str, filename: str) -> str:
        """Create a synthetic image with objects for testing"""
        filepath = os.path.join(output_dir, filename)
        
        # Create a 640x480 RGB image
        img = Image.new('RGB', (640, 480), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw some basic shapes to simulate objects
        # Person-like rectangle
        draw.rectangle([200, 150, 280, 400], fill='brown', outline='black', width=2)
        # Head
        draw.ellipse([220, 120, 260, 160], fill='peach', outline='black', width=2)
        
        # Car-like rectangle
        draw.rectangle([400, 300, 550, 380], fill='red', outline='black', width=2)
        # Wheels
        draw.ellipse([415, 365, 445, 395], fill='black')
        draw.ellipse([505, 365, 535, 395], fill='black')
        
        img.save(filepath)
        return filepath
    
    def get_class_names(self) -> Dict[int, str]:
        """Get COCO class names mapping"""
        if self.coco:
            cats = self.coco.loadCats(self.coco.getCatIds())
            return {cat['id']: cat['name'] for cat in cats}
        else:
            # Return standard COCO classes
            return {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow'
                # ... add more as needed
            }
    
    def create_test_annotation(self, image_path: str, output_path: str):
        """
        Create a simple COCO annotation file for testing
        
        Args:
            image_path: Path to test image
            output_path: Path to save annotation file
        """
        # Load image to get dimensions
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Create minimal COCO annotation structure
        annotation = {
            "images": [
                {
                    "id": 1,
                    "file_name": os.path.basename(image_path),
                    "width": width,
                    "height": height
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,  # person
                    "bbox": [200, 150, 80, 250],  # x, y, width, height
                    "area": 20000,
                    "iscrowd": 0
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "person"
                }
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)
    
    def validate_dataset(self, images_dir: str, annotations_file: str) -> bool:
        """
        Validate COCO dataset structure
        
        Args:
            images_dir: Directory containing images
            annotations_file: COCO annotations file
            
        Returns:
            True if dataset is valid
        """
        try:
            # Check if annotation file exists and is valid JSON
            if not os.path.exists(annotations_file):
                return False
            
            with open(annotations_file, 'r') as f:
                data = json.load(f)
            
            # Check required fields
            required_fields = ['images', 'annotations', 'categories']
            if not all(field in data for field in required_fields):
                return False
            
            # Check if image directory exists
            if not os.path.exists(images_dir):
                return False
            
            # Check if at least one image file exists
            image_files = [f for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            return len(image_files) > 0
            
        except Exception as e:
            print(f"Dataset validation error: {e}")
            return False

def setup_test_dataset(data_dir: str = "data/samples") -> Tuple[str, str]:
    """
    Set up test dataset for CI/CD pipeline
    
    Args:
        data_dir: Directory to create test data
        
    Returns:
        Tuple of (images_directory, annotations_file)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    coco_handler = COCODatasetHandler()
    
    # Try to download real COCO images, fallback to synthetic
    image_paths = coco_handler.download_sample_images(data_dir, num_images=3)
    
    # Create test annotations
    annotations_file = os.path.join(data_dir, "test_annotations.json")
    if image_paths:
        coco_handler.create_test_annotation(image_paths[0], annotations_file)
    
    return data_dir, annotations_file
