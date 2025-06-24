# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive audio-visual bird monitoring pipeline that processes video recordings from stationary cameras to detect and identify bird species. The system uses parallel audio and video analysis with hierarchical confidence scoring.

## Key Architecture

### 5-Stage Processing Pipeline

The core processing follows a sequential 5-stage pipeline orchestrated by `BirdMonitoringPipeline`:

1. **Media Preprocessing** (`MediaProcessor`): FFmpeg-based audio/video separation with timestamped frame extraction
2. **Parallel Detection**: 
   - Audio: BirdNET integration with mel-spectrogram analysis (`AudioDetector`)
   - Video: YOLOv10 object detection for birds (`VideoDetector`)
3. **Event Integration** (`EventIntegrator`): Temporal correlation of audio/video detections with configurable time windows
4. **Species Classification** (`SpeciesClassifier`): TransFG fine-grained classification using cropped bird images
5. **Results Export**: Structured JSON/CSV output with comprehensive metadata

### Hierarchical Confidence System

The system implements a mathematical confidence model:
```
Confidence_Final = Confidence_Initial + (1 - Confidence_Initial) * Confidence_FGVC
```

Base confidences are:
- Audio-only: 0.3
- Video-only: 0.4
- Audio+Video: 0.7

### Data Flow Architecture

- **Pydantic Models** (`src/models/data_models.py`): Type-safe schemas for all data structures
- **Event Types**: `DetectionType` enum with AUDIO_ONLY, VIDEO_ONLY, AUDIO_VIDEO
- **Core Data**: `DetectionEvent`, `AudioDetection`, `VideoDetection`, `BoundingBox`
- **Results**: `PipelineResult` with comprehensive statistics and metadata

### Configuration System

- **YAML Configuration** (`config.yaml`): Production settings
- **Environment Variables**: `BIRD_MONITOR_*` prefixed variables
- **Pydantic Settings** (`src/utils/config.py`): Type-safe configuration management
- **Model Paths**: Configurable paths for BirdNET, YOLOv10, and TransFG models

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv claude-env
source claude-env/bin/activate  # Linux/Mac
claude-env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_data_models.py

# Run single test
pytest tests/test_data_models.py::TestDetectionEvent::test_audio_only_event
```

### Application Execution
```bash
# CLI interface - single video
python -m src.cli process video.mp4 --output results/

# CLI interface - batch processing
python -m src.cli batch input_videos/ output/ --pattern "*.mp4"

# Web interface
python -m src.web.app

# Check system status
python -m src.cli status
```

### Docker Development
```bash
# Build and run full stack
docker-compose up -d

# Scale workers
docker-compose up --scale celery-worker=4

# View logs
docker-compose logs bird-monitor
```

## Code Architecture

### Core Components Location

- **Pipeline Orchestration**: `src/core/pipeline.py` - Main `BirdMonitoringPipeline` class
- **Media Processing**: `src/core/media_processor.py` - FFmpeg integration
- **AI Models**: Individual detector classes in `src/core/`
- **Web Interface**: `src/web/app.py` - FastAPI backend with REST APIs
- **CLI**: `src/cli.py` - Typer-based command line interface

### Model Integration Pattern

Each AI model (BirdNET, YOLOv10, TransFG) follows the same pattern:
- Model loading with fallback to mock implementations
- Input preprocessing and validation
- Batch processing with progress tracking
- Structured output with confidence scores
- Export capabilities for debugging

### Configuration Pattern

Settings flow: Environment Variables → YAML Config → Pydantic Validation → Component Initialization

### Error Handling

- **Logging Context**: `LoggingContext` for structured operation logging
- **Component Isolation**: Failed models fall back to mock implementations
- **Pipeline Continuity**: Processing continues with available models
- **Comprehensive Logging**: All errors logged with context and stack traces

### Web Application Architecture

- **FastAPI Backend**: Async processing with background tasks
- **Job Queue**: In-memory job tracking (Redis recommended for production)
- **Real-time Updates**: WebSocket support for progress monitoring
- **Interactive Frontend**: HTML5 video player with timeline synchronization

## Model Dependencies

The system expects these models in `models/` directory:
- **BirdNET**: `models/birdnet/model.tflite` + `species_list.json`
- **YOLOv10**: `models/yolo/yolov10n.pt` (auto-downloads if missing)
- **TransFG**: `models/transfg/model.pth` + `species_list.json`

All models have mock implementations for development without actual model files.

## Key Data Structures

### DetectionEvent Schema
The central data structure containing:
- Event metadata (ID, timestamps, type)
- Raw detection data (audio/video)
- Confidence scores (initial/final)
- Species classification (optional)
- Bounding box information (video events)

### Processing Configuration
Comprehensive settings including:
- Detection thresholds for each modality
- Temporal correlation parameters
- Performance tuning options
- Model paths and parameters

## Testing Strategy

- **Unit Tests**: Individual component testing with mocks
- **Integration Tests**: Pipeline end-to-end testing
- **Fixtures**: Reusable test data in `tests/conftest.py`
- **Mock Models**: All AI models have mock implementations for testing

## Performance Considerations

- **GPU Acceleration**: CUDA support for all models when available
- **Batch Processing**: Configurable batch sizes for memory management
- **Frame Skipping**: Configurable frame sampling to reduce processing time
- **Parallel Processing**: Audio and video analysis run independently
- **Memory Management**: Temporary file cleanup and resource monitoring