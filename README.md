# Bird Monitoring Pipeline

A comprehensive audio-visual bird detection and classification system designed for academic research. This pipeline processes video recordings from stationary cameras to detect and identify bird species using parallel audio and video analysis.

## Features

### Core Functionality
- **Parallel Processing**: Simultaneous audio and video stream analysis
- **Multi-modal Detection**: Audio-only, video-only, and combined detections
- **Hierarchical Confidence Scoring**: Advanced confidence calculation based on detection type and species classification
- **Species Identification**: Fine-grained species classification using TransFG
- **Interactive Web Interface**: Real-time visualization and analysis tools

### Technical Highlights
- **Audio Detection**: BirdNET integration for bird call detection
- **Video Detection**: YOLOv10 for bird object detection
- **Species Classification**: TransFG for fine-grained species identification
- **Event Integration**: Temporal correlation of audio and video detections
- **Export Capabilities**: JSON and CSV export with filtering options

## System Architecture

```
Input Video (MP4/AVI/MOV)
├── Audio Stream → BirdNET → Audio Detections
├── Video Stream → YOLOv10 → Video Detections
└── Integration → Event Correlation → Species Classification → Results
```

### Processing Pipeline

1. **Media Preprocessing**: Extract audio and video streams using FFmpeg
2. **Parallel Detection**: 
   - Audio: Segment-based analysis with mel-spectrograms
   - Video: Frame-based object detection
3. **Event Integration**: Temporal correlation with configurable time windows
4. **Species Classification**: Fine-grained classification for video detections
5. **Result Export**: Structured data output with confidence metrics

## Installation

### Prerequisites
- Python 3.11+
- FFmpeg
- CUDA (optional, for GPU acceleration)

### Option 1: Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd tanbo_tori

# Create virtual environment
python -m venv claude-env
source claude-env/bin/activate  # Linux/Mac
# or
claude-env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Docker Installation

```bash
# Clone the repository
git clone <repository-url>
cd tanbo_tori

# Build and run with Docker Compose
docker-compose up -d

# Access the web interface at http://localhost
```

## Configuration

### Environment Variables
```bash
export BIRD_MONITOR_ENVIRONMENT=production
export BIRD_MONITOR_LOG_LEVEL=INFO
export BIRD_MONITOR_GPU_ENABLED=true
```

### Configuration File
Edit `config.yaml` to customize processing parameters:

```yaml
# Detection thresholds
audio_confidence_threshold: 0.3
video_confidence_threshold: 0.5
species_confidence_threshold: 0.6

# Processing settings
audio_segment_length: 3.0
temporal_correlation_window: 2.0
```

## Usage

### Command Line Interface

```bash
# Process a single video
python -m src.cli process video.mp4 --output results/

# Process with custom thresholds
python -m src.cli process video.mp4 \
  --audio-threshold 0.4 \
  --video-threshold 0.6 \
  --species-threshold 0.7

# Batch processing
python -m src.cli batch input_videos/ output_results/ --pattern "*.mp4"

# Check system status
python -m src.cli status
```

### Web Interface

1. Start the web application:
```bash
python -m src.web.app
```

2. Open your browser to `http://localhost:8000`

3. Upload a video file and configure detection parameters

4. Monitor processing progress and view results

### Python API

```python
from src.core.pipeline import BirdMonitoringPipeline
from src.models.data_models import ProcessingConfig

# Initialize pipeline
pipeline = BirdMonitoringPipeline()

# Configure processing
config = ProcessingConfig(
    input_video_path="path/to/video.mp4",
    output_dir="results/",
    audio_confidence_threshold=0.3,
    video_confidence_threshold=0.5
)

# Process video
result = pipeline.process_video("video.mp4", "output/", config)

# Access results
print(f"Total events: {result.total_events}")
print(f"Species found: {result.unique_species}")
```

## Model Setup

### BirdNET (Audio Detection)
1. Download BirdNET model files
2. Place in `models/birdnet/`
3. Update model path in configuration

### YOLOv10 (Object Detection)
```bash
# Download YOLOv10 weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt
mv yolov10n.pt models/yolo/
```

### TransFG (Species Classification)
1. Download TransFG model weights
2. Place in `models/transfg/`
3. Include species list JSON file

## Output Format

### Detection Events
```json
{
  "event_id": "evt_001",
  "start_time": 10.5,
  "end_time": 13.5,
  "detection_type": "audio_video",
  "initial_confidence": 0.7,
  "final_confidence": 0.916,
  "species_classification": {
    "species_name": "Eurasian Tree Sparrow",
    "confidence": 0.92
  },
  "bounding_box": {
    "timestamp": 11.2,
    "x": 150, "y": 200,
    "width": 50, "height": 50
  }
}
```

### Processing Results
- `detection_results.json`: Complete pipeline results
- `audio_detections.json`: Audio-only detection data
- `video_detections.json`: Video-only detection data
- `species_classifications.json`: Species identification results
- `statistics.json`: Processing and quality metrics

## Web Interface Features

### Video Player
- Synchronized video playback with detection timeline
- Interactive bounding box overlays
- Jump-to-detection functionality

### Timeline Visualization
- Color-coded detection events
- Confidence-based visual indicators
- Click-to-navigate interface

### Filtering and Search
- Filter by species, confidence, detection type
- Time-range filtering
- Real-time result updates

### Export Options
- JSON: Complete structured data
- CSV: Tabular format for analysis
- Custom filtered exports

## Performance Optimization

### Hardware Requirements
- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, 8-core CPU, GPU with 6GB VRAM
- **Storage**: 100GB+ for models and temporary files

### Processing Speed
- **CPU**: ~2-3x real-time processing
- **GPU**: ~5-8x real-time processing
- **Memory**: ~1GB per hour of video

### Scaling
```yaml
# Docker Compose scaling
docker-compose up --scale celery-worker=4
```

## Development

### Project Structure
```
tanbo_tori/
├── src/
│   ├── core/           # Core pipeline components
│   ├── models/         # Data models and schemas
│   ├── utils/          # Utilities and configuration
│   ├── web/            # Web application
│   └── cli.py          # Command-line interface
├── tests/              # Unit and integration tests
├── docker/             # Docker configuration
├── models/             # Pre-trained model storage
└── data/               # Input/output data
```

### Running Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# All tests with coverage
pytest --cov=src tests/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Monitoring and Deployment

### Health Checks
```bash
curl http://localhost:8000/health
```

### Metrics (with Docker Compose)
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

### Logging
- Application logs: `logs/bird_monitor_*.log`
- Web access logs: `logs/nginx/access.log`
- Error logs: `logs/nginx/error.log`

## Troubleshooting

### Common Issues

**FFmpeg not found**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Docker: Already included in image
```

**GPU not detected**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

**Memory errors during processing**
```yaml
# Reduce batch size in config.yaml
batch_size: 16  # Default: 32
video_frame_skip: 2  # Process every 2nd frame
```

**Model loading failures**
- Verify model files exist and are readable
- Check model paths in configuration
- Ensure sufficient disk space

### Performance Issues
- Enable GPU acceleration if available
- Increase `max_workers` for CPU processing
- Use frame skipping for faster processing
- Monitor system resources during processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this pipeline in academic research, please cite:

```bibtex
@software{bird_monitoring_pipeline,
  title={Bird Monitoring Pipeline: Audio-Visual Detection and Classification},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/tanbo_tori}
}
```

## Acknowledgments

- **BirdNET**: For audio-based bird detection
- **YOLOv10**: For real-time object detection
- **TransFG**: For fine-grained visual classification
- **FastAPI**: For web application framework
- **FFmpeg**: For media processing capabilities

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact: your.email@domain.com
- Documentation: [Project Wiki](wiki-url)