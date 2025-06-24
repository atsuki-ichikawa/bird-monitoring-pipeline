# Data Directory

This directory contains input videos, processing results, and temporary files.

## Directory Structure

- `input/` - Place your video files here for processing
- `output/` - Processing results and detection data
- `temp/` - Temporary files created during processing
- `uploads/` - Web interface file uploads (auto-created)

## Usage

### CLI Processing
```bash
# Process videos from input directory
python -m src.cli batch data/input/ data/output/ --pattern "*.mp4"
```

### Web Interface
Upload videos through the web interface at http://localhost:8000

## Output Files

Each processed video creates:
- `detection_results.json` - Complete pipeline results
- `audio_detections.json` - Audio-only detection data
- `video_detections.json` - Video-only detection data
- `species_classifications.json` - Species identification results
- `statistics.json` - Processing metrics

## Cleanup

Temporary files are automatically cleaned up after processing. To manually clean:

```bash
# Remove temporary files
rm -rf data/temp/*

# Remove old output files
rm -rf data/output/old_results/
```