"""
Data models for the bird monitoring pipeline.
Defines Pydantic models for detection events, results, and pipeline data structures.
"""

from typing import Optional, List, Dict, Union, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class DetectionType(str, Enum):
    """Types of detection events."""
    AUDIO_ONLY = "audio_only"
    VIDEO_ONLY = "video_only"
    AUDIO_VIDEO = "audio_video"


class BoundingBox(BaseModel):
    """Bounding box coordinates for video detections."""
    timestamp: float = Field(..., description="Time in seconds when detection occurred")
    x: float = Field(..., ge=0, description="X coordinate of top-left corner")
    y: float = Field(..., ge=0, description="Y coordinate of top-left corner")
    width: float = Field(..., gt=0, description="Width of bounding box")
    height: float = Field(..., gt=0, description="Height of bounding box")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v < 0:
            raise ValueError('Timestamp must be non-negative')
        return v


class RawScores(BaseModel):
    """Raw confidence scores from individual detection models."""
    audio: Optional[float] = Field(None, ge=0, le=1, description="Audio detection confidence")
    video: Optional[float] = Field(None, ge=0, le=1, description="Video detection confidence")
    
    @validator('audio', 'video')
    def validate_scores(cls, v):
        if v is not None and not (0 <= v <= 1):
            raise ValueError('Confidence scores must be between 0 and 1')
        return v


class AudioDetection(BaseModel):
    """Audio detection result."""
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., ge=0, description="End time in seconds")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    frequency_range: Optional[tuple[float, float]] = Field(None, description="Frequency range (Hz)")
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('End time must be greater than start time')
        return v


class VideoDetection(BaseModel):
    """Video detection result."""
    timestamp: float = Field(..., ge=0, description="Time in seconds")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    bounding_box: BoundingBox = Field(..., description="Bounding box coordinates")


class SpeciesClassification(BaseModel):
    """Fine-grained species classification result."""
    species_name: str = Field(..., description="Scientific or common name of species")
    confidence: float = Field(..., ge=0, le=1, description="Classification confidence")
    alternative_species: Optional[List[Dict[str, float]]] = Field(
        None, description="Alternative species predictions with confidence scores"
    )


class DetectionEvent(BaseModel):
    """Complete detection event with all associated data."""
    event_id: str = Field(..., description="Unique identifier for the event")
    start_time: float = Field(..., ge=0, description="Event start time in seconds")
    end_time: float = Field(..., ge=0, description="Event end time in seconds")
    detection_type: DetectionType = Field(..., description="Type of detection (audio/video/both)")
    
    # Confidence scores
    initial_confidence: float = Field(..., ge=0, le=1, description="Initial confidence score")
    final_confidence: float = Field(..., ge=0, le=1, description="Final confidence score")
    
    # Raw detection data
    audio_detection: Optional[AudioDetection] = Field(None, description="Audio detection data")
    video_detection: Optional[VideoDetection] = Field(None, description="Video detection data")
    raw_scores: RawScores = Field(..., description="Raw model confidence scores")
    
    # Species classification
    species_classification: Optional[SpeciesClassification] = Field(
        None, description="Fine-grained species classification"
    )
    
    # Additional metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('End time must be greater than start time')
        return v
    
    @validator('detection_type')
    def validate_detection_type(cls, v, values):
        """Ensure detection type matches available detection data."""
        # This validation will be applied after other fields are validated
        return v
    
    def to_export_dict(self) -> Dict:
        """Convert to dictionary suitable for JSON export."""
        data = self.dict()
        data['created_at'] = self.created_at.isoformat()
        return data


class PipelineResult(BaseModel):
    """Complete pipeline processing result."""
    video_path: str = Field(..., description="Path to processed video file")
    processing_started: datetime = Field(..., description="Processing start time")
    processing_completed: Optional[datetime] = Field(None, description="Processing completion time")
    
    # Detection statistics
    total_events: int = Field(0, ge=0, description="Total number of detection events")
    audio_only_events: int = Field(0, ge=0, description="Number of audio-only events")
    video_only_events: int = Field(0, ge=0, description="Number of video-only events")
    audio_video_events: int = Field(0, ge=0, description="Number of combined events")
    
    # Species identification statistics
    species_identified: int = Field(0, ge=0, description="Number of events with species identification")
    unique_species: List[str] = Field(default_factory=list, description="List of unique species found")
    
    # All detection events
    events: List[DetectionEvent] = Field(default_factory=list, description="All detection events")
    
    # Processing metadata
    pipeline_version: str = Field("1.0.0", description="Pipeline version")
    processing_parameters: Dict = Field(default_factory=dict, description="Processing parameters used")
    
    def add_event(self, event: DetectionEvent) -> None:
        """Add a detection event and update statistics."""
        self.events.append(event)
        self.total_events += 1
        
        # Update type-specific counters
        if event.detection_type == DetectionType.AUDIO_ONLY:
            self.audio_only_events += 1
        elif event.detection_type == DetectionType.VIDEO_ONLY:
            self.video_only_events += 1
        elif event.detection_type == DetectionType.AUDIO_VIDEO:
            self.audio_video_events += 1
        
        # Update species statistics
        if event.species_classification:
            self.species_identified += 1
            species_name = event.species_classification.species_name
            if species_name not in self.unique_species:
                self.unique_species.append(species_name)
    
    def get_events_by_type(self, detection_type: DetectionType) -> List[DetectionEvent]:
        """Get all events of a specific detection type."""
        return [event for event in self.events if event.detection_type == detection_type]
    
    def get_events_by_species(self, species_name: str) -> List[DetectionEvent]:
        """Get all events for a specific species."""
        return [
            event for event in self.events 
            if event.species_classification and event.species_classification.species_name == species_name
        ]
    
    def get_high_confidence_events(self, threshold: float = 0.8) -> List[DetectionEvent]:
        """Get events with final confidence above threshold."""
        return [event for event in self.events if event.final_confidence >= threshold]
    
    def to_export_dict(self) -> Dict:
        """Convert to dictionary suitable for JSON export."""
        data = self.dict()
        data['processing_started'] = self.processing_started.isoformat()
        if self.processing_completed:
            data['processing_completed'] = self.processing_completed.isoformat()
        
        # Convert events to export format
        data['events'] = [event.to_export_dict() for event in self.events]
        
        return data


class ProcessingConfig(BaseModel):
    """Configuration parameters for pipeline processing."""
    # Input settings
    input_video_path: str = Field(..., description="Path to input video file")
    output_dir: str = Field(..., description="Output directory for results")
    
    # Audio processing settings
    audio_segment_length: float = Field(3.0, gt=0, description="Audio segment length in seconds")
    audio_overlap: float = Field(0.5, ge=0, lt=1, description="Audio segment overlap ratio")
    audio_sample_rate: int = Field(22050, gt=0, description="Audio sample rate")
    
    # Video processing settings
    video_frame_skip: int = Field(1, ge=1, description="Process every N frames")
    video_resize_height: Optional[int] = Field(None, gt=0, description="Resize video height")
    
    # Detection thresholds
    audio_confidence_threshold: float = Field(0.3, ge=0, le=1, description="Audio detection threshold")
    video_confidence_threshold: float = Field(0.5, ge=0, le=1, description="Video detection threshold")
    species_confidence_threshold: float = Field(0.6, ge=0, le=1, description="Species classification threshold")
    
    # Temporal correlation settings
    temporal_correlation_window: float = Field(2.0, gt=0, description="Time window for correlating audio/video events")
    
    # Initial confidence scores
    audio_only_base_confidence: float = Field(0.3, ge=0, le=1, description="Base confidence for audio-only events")
    video_only_base_confidence: float = Field(0.4, ge=0, le=1, description="Base confidence for video-only events")
    audio_video_base_confidence: float = Field(0.7, ge=0, le=1, description="Base confidence for audio+video events")