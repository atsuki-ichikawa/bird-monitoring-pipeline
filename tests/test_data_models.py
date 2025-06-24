"""
Tests for data models and schemas.
"""

import pytest
from datetime import datetime
from src.models.data_models import (
    DetectionEvent, AudioDetection, VideoDetection, BoundingBox,
    DetectionType, RawScores, SpeciesClassification
)


class TestBoundingBox:
    def test_valid_bounding_box(self):
        bbox = BoundingBox(
            timestamp=10.5,
            x=100, y=50,
            width=200, height=150
        )
        assert bbox.timestamp == 10.5
        assert bbox.x == 100
        assert bbox.y == 50
        assert bbox.width == 200
        assert bbox.height == 150
    
    def test_negative_timestamp_validation(self):
        with pytest.raises(ValueError):
            BoundingBox(
                timestamp=-1.0,
                x=0, y=0, width=100, height=100
            )


class TestAudioDetection:
    def test_valid_audio_detection(self):
        detection = AudioDetection(
            start_time=5.0,
            end_time=8.0,
            confidence=0.85
        )
        assert detection.start_time == 5.0
        assert detection.end_time == 8.0
        assert detection.confidence == 0.85
    
    def test_end_time_validation(self):
        with pytest.raises(ValueError):
            AudioDetection(
                start_time=10.0,
                end_time=5.0,  # Invalid: end before start
                confidence=0.8
            )


class TestVideoDetection:
    def test_valid_video_detection(self):
        bbox = BoundingBox(
            timestamp=12.0,
            x=50, y=25, width=100, height=75
        )
        detection = VideoDetection(
            timestamp=12.0,
            confidence=0.92,
            bounding_box=bbox
        )
        assert detection.timestamp == 12.0
        assert detection.confidence == 0.92
        assert detection.bounding_box == bbox


class TestDetectionEvent:
    def test_audio_only_event(self):
        audio_det = AudioDetection(
            start_time=10.0,
            end_time=13.0,
            confidence=0.7
        )
        raw_scores = RawScores(audio=0.7, video=None)
        
        event = DetectionEvent(
            event_id="test_001",
            start_time=10.0,
            end_time=13.0,
            detection_type=DetectionType.AUDIO_ONLY,
            initial_confidence=0.5,
            final_confidence=0.6,
            audio_detection=audio_det,
            video_detection=None,
            raw_scores=raw_scores
        )
        
        assert event.detection_type == DetectionType.AUDIO_ONLY
        assert event.audio_detection is not None
        assert event.video_detection is None
    
    def test_event_export_dict(self):
        raw_scores = RawScores(audio=0.8, video=None)
        event = DetectionEvent(
            event_id="test_002",
            start_time=5.0,
            end_time=8.0,
            detection_type=DetectionType.AUDIO_ONLY,
            initial_confidence=0.6,
            final_confidence=0.7,
            raw_scores=raw_scores
        )
        
        export_dict = event.to_export_dict()
        assert "created_at" in export_dict
        assert export_dict["event_id"] == "test_002"
        assert export_dict["detection_type"] == "audio_only"


class TestSpeciesClassification:
    def test_valid_classification(self):
        classification = SpeciesClassification(
            species_name="House Sparrow",
            confidence=0.89,
            alternative_species=[
                {"Eurasian Tree Sparrow": 0.65},
                {"Song Sparrow": 0.42}
            ]
        )
        
        assert classification.species_name == "House Sparrow"
        assert classification.confidence == 0.89
        assert len(classification.alternative_species) == 2


if __name__ == "__main__":
    pytest.main([__file__])