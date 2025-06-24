# Models Directory

This directory contains the AI models used by the bird monitoring pipeline.

## Required Models

### BirdNET (Audio Detection)
- **Location**: `models/birdnet/`
- **Files needed**:
  - `model.tflite` - BirdNET TensorFlow Lite model
  - `species_list.json` - List of supported bird species
- **Download**: Follow BirdNET installation instructions

### YOLOv10 (Object Detection)
- **Location**: `models/yolo/`
- **Files needed**:
  - `yolov10n.pt` - YOLOv10 nano weights
- **Download**:
  ```bash
  wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt
  mv yolov10n.pt models/yolo/
  ```

### TransFG (Species Classification)
- **Location**: `models/transfg/`
- **Files needed**:
  - `model.pth` - TransFG model weights
  - `species_list.json` - Species classification labels
- **Download**: Follow TransFG installation instructions

## Mock Models

If model files are not available, the system will automatically use mock implementations for development and testing.

## Model Configuration

Update the model paths in `config.yaml`:

```yaml
birdnet_model_path: /app/models/birdnet/model.tflite
yolo_model_path: /app/models/yolo/yolov10n.pt
transfg_model_path: /app/models/transfg/model.pth
```