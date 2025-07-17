# Object Follower Robot using YOLOv3 and Raspberry Pi

This project enables a Raspberry Pi robot to detect and follow a person using OpenCV's deep learning module with YOLOv3.

## Requirements

- Raspberry Pi with GPIO support
- Pi Camera or USB Camera
- DC Motors and GPIO wiring
- Python 3
- OpenCV
- YOLOv3 model files

## Setup

1. Clone this repo.
2. Place `yolov3.cfg`, `yolov3.weights`, and `coco.names.txt` in the root folder.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the program:
   ```
   python3 main.py
   ```

## Files

- `main.py` - Main Python code for object tracking
- `yolov3.cfg` - YOLOv3 configuration
- `yolov3.weights` - YOLOv3 trained weights
- `coco.names.txt` - Object labels
- `requirements.txt` - Required Python libraries
