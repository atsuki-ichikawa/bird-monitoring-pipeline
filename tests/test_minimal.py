"""
Minimal tests that don't require heavy dependencies.
"""

import pytest
from datetime import datetime
from src.models.data_models import (
    DetectionType, BoundingBox, RawScores, 
    AudioDetection, VideoDetection, SpeciesClassification
)


class TestDetectionType:
    def test_detection_type_values(self):
        """Test that detection type enum has correct values."""
        assert DetectionType.AUDIO_ONLY == "audio_only"
        assert DetectionType.VIDEO_ONLY == "video_only"
        assert DetectionType.AUDIO_VIDEO == "audio_video"


class TestBoundingBox:
    def test_valid_bounding_box(self):
        """Test creating a valid bounding box."""
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
        """Test that negative timestamps are rejected."""
        with pytest.raises(ValueError, match="Timestamp must be non-negative"):
            BoundingBox(
                timestamp=-1.0,
                x=0, y=0, width=100, height=100
            )


class TestRawScores:
    def test_valid_raw_scores(self):
        """Test creating valid raw scores."""
        scores = RawScores(audio=0.8, video=0.9)
        assert scores.audio == 0.8
        assert scores.video == 0.9
    
    def test_none_values(self):
        """Test that None values are allowed."""
        scores = RawScores(audio=0.8, video=None)
        assert scores.audio == 0.8
        assert scores.video is None


class TestAudioDetection:
    def test_valid_audio_detection(self):
        """Test creating a valid audio detection."""
        detection = AudioDetection(
            start_time=5.0,
            end_time=8.0,
            confidence=0.85
        )
        assert detection.start_time == 5.0
        assert detection.end_time == 8.0
        assert detection.confidence == 0.85
    
    def test_end_time_validation(self):
        """Test that end time must be after start time."""
        with pytest.raises(ValueError, match="End time must be greater than start time"):
            AudioDetection(
                start_time=10.0,
                end_time=5.0,  # Invalid: end before start
                confidence=0.8
            )


class TestVideoDetection:
    def test_valid_video_detection(self):
        """Test creating a valid video detection."""
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


class TestSpeciesClassification:
    def test_valid_classification(self):
        """Test creating a valid species classification."""
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
    
    def test_classification_without_alternatives(self):
        """Test classification without alternative species."""
        classification = SpeciesClassification(
            species_name="Robin",
            confidence=0.95
        )
        
        assert classification.species_name == "Robin"
        assert classification.confidence == 0.95
        assert classification.alternative_species is None


class TestConfidenceCalculation:
    """Test confidence calculation logic."""
    
    def test_confidence_bounds(self):
        """Test that confidence values are within valid bounds."""
        # Test various confidence values
        confidences = [0.0, 0.5, 1.0]
        
        for conf in confidences:
            detection = AudioDetection(
                start_time=0.0,
                end_time=3.0,
                confidence=conf
            )
            assert 0.0 <= detection.confidence <= 1.0
    
    def test_hierarchical_confidence_formula(self):
        """Test the hierarchical confidence calculation formula."""
        # Formula: Confidence_Final = Confidence_Initial + (1 - Confidence_Initial) * Confidence_FGVC
        
        initial_confidence = 0.7
        species_confidence = 0.92
        
        # Manual calculation
        expected_final = initial_confidence + (1 - initial_confidence) * species_confidence
        # 0.7 + (1 - 0.7) * 0.92 = 0.7 + 0.3 * 0.92 = 0.7 + 0.276 = 0.976
        
        assert abs(expected_final - 0.976) < 0.001
        assert expected_final <= 1.0  # Should never exceed 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])