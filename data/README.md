# Data Folder

This folder contains data and configurations required to train and evaluate the VisionFlow model.

## Structure

- `coco/`: This folder contains datasets formatted using the [COCO API](https://github.com/cocodataset/cocoapi).
- `pretrained_models/`: Pre-trained weights used for transfer learning and evaluation.
- `annotations/`: Contains COCO-style annotation JSON files.
- `images/`: Subfolders for training and validation images (`train2017/`, `val2017/`).

## Instructions

1. **Install COCO API:**
   Clone and install the COCO API repository:
   ```bash
   git clone https://github.com/cocodataset/cocoapi.git
   cd cocoapi/PythonAPI
   python setup.py install
