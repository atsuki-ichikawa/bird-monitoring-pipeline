"""
Pytest configuration and fixtures for the bird monitoring pipeline tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

from src.models.data_models import ProcessingConfig
from src.core.pipeline import BirdMonitoringPipeline


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample processing configuration."""
    return ProcessingConfig(
        input_video_path=str(temp_dir / "test_video.mp4"),
        output_dir=str(temp_dir / "output"),
        audio_confidence_threshold=0.3,
        video_confidence_threshold=0.5,
        species_confidence_threshold=0.6
    )


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing."""
    pipeline = Mock(spec=BirdMonitoringPipeline)
    pipeline.process_video.return_value = Mock()
    return pipeline


@pytest.fixture
def sample_audio_detection():
    """Create a sample audio detection."""
    from src.models.data_models import AudioDetection
    return AudioDetection(
        start_time=10.0,
        end_time=13.0,
        confidence=0.85,
        frequency_range=(1000.0, 8000.0)
    )


@pytest.fixture
def sample_video_detection():
    """Create a sample video detection."""
    from src.models.data_models import VideoDetection, BoundingBox
    
    bbox = BoundingBox(
        timestamp=11.5,
        x=100, y=50,
        width=150, height=100
    )
    
    return VideoDetection(
        timestamp=11.5,
        confidence=0.92,
        bounding_box=bbox
    )


@pytest.fixture
def sample_detection_event(sample_audio_detection, sample_video_detection):
    """Create a sample detection event."""
    from src.models.data_models import DetectionEvent, DetectionType, RawScores
    
    raw_scores = RawScores(
        audio=sample_audio_detection.confidence,
        video=sample_video_detection.confidence
    )
    
    return DetectionEvent(
        event_id="test_event_001",
        start_time=sample_audio_detection.start_time,
        end_time=sample_audio_detection.end_time,
        detection_type=DetectionType.AUDIO_VIDEO,
        initial_confidence=0.7,
        final_confidence=0.85,
        audio_detection=sample_audio_detection,
        video_detection=sample_video_detection,
        raw_scores=raw_scores
    )