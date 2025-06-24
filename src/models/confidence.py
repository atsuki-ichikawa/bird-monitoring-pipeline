"""
Hierarchical confidence calculation system for bird detection events.
Implements confidence scoring based on detection type and species classification.
"""

from typing import Optional
from .data_models import DetectionType, DetectionEvent, RawScores, SpeciesClassification


class ConfidenceCalculator:
    """Calculates hierarchical confidence scores for detection events."""
    
    def __init__(
        self,
        audio_only_base: float = 0.3,
        video_only_base: float = 0.4,
        audio_video_base: float = 0.7
    ):
        """
        Initialize confidence calculator with base confidence values.
        
        Args:
            audio_only_base: Base confidence for audio-only detections
            video_only_base: Base confidence for video-only detections  
            audio_video_base: Base confidence for audio+video detections
        """
        self.audio_only_base = audio_only_base
        self.video_only_base = video_only_base
        self.audio_video_base = audio_video_base
    
    def calculate_initial_confidence(
        self, 
        detection_type: DetectionType,
        raw_scores: RawScores
    ) -> float:
        """
        Calculate initial confidence based on detection type and raw scores.
        
        Args:
            detection_type: Type of detection (audio/video/both)
            raw_scores: Raw confidence scores from detection models
            
        Returns:
            Initial confidence score (0-1)
        """
        if detection_type == DetectionType.AUDIO_ONLY:
            base_confidence = self.audio_only_base
            # Adjust based on audio confidence if available
            if raw_scores.audio is not None:
                # Weighted combination of base confidence and raw audio score
                return 0.5 * base_confidence + 0.5 * raw_scores.audio
            return base_confidence
            
        elif detection_type == DetectionType.VIDEO_ONLY:
            base_confidence = self.video_only_base
            # Adjust based on video confidence if available
            if raw_scores.video is not None:
                # Weighted combination of base confidence and raw video score
                return 0.5 * base_confidence + 0.5 * raw_scores.video
            return base_confidence
            
        elif detection_type == DetectionType.AUDIO_VIDEO:
            base_confidence = self.audio_video_base
            # Combine audio and video scores if available
            if raw_scores.audio is not None and raw_scores.video is not None:
                # Weighted combination incorporating both modalities
                combined_raw = 0.6 * raw_scores.video + 0.4 * raw_scores.audio
                return 0.3 * base_confidence + 0.7 * combined_raw
            elif raw_scores.audio is not None:
                return 0.5 * base_confidence + 0.5 * raw_scores.audio
            elif raw_scores.video is not None:
                return 0.5 * base_confidence + 0.5 * raw_scores.video
            return base_confidence
        
        # Fallback
        return 0.5
    
    def calculate_final_confidence(
        self,
        initial_confidence: float,
        species_classification: Optional[SpeciesClassification] = None
    ) -> float:
        """
        Calculate final confidence incorporating species classification.
        
        Uses the formula from the design document:
        Confidence_Final = Confidence_Initial + (1 - Confidence_Initial) * Confidence_FGVC
        
        Args:
            initial_confidence: Initial confidence score
            species_classification: Fine-grained species classification result
            
        Returns:
            Final confidence score (0-1)
        """
        if species_classification is None:
            return initial_confidence
        
        # Apply hierarchical confidence formula
        species_confidence = species_classification.confidence
        final_confidence = initial_confidence + (1 - initial_confidence) * species_confidence
        
        # Ensure final confidence doesn't exceed 1.0
        return min(final_confidence, 1.0)
    
    def update_event_confidence(self, event: DetectionEvent) -> DetectionEvent:
        """
        Update confidence scores for a detection event.
        
        Args:
            event: Detection event to update
            
        Returns:
            Updated detection event with confidence scores
        """
        # Calculate initial confidence
        initial_confidence = self.calculate_initial_confidence(
            event.detection_type, 
            event.raw_scores
        )
        
        # Calculate final confidence
        final_confidence = self.calculate_final_confidence(
            initial_confidence,
            event.species_classification
        )
        
        # Update event
        event.initial_confidence = initial_confidence
        event.final_confidence = final_confidence
        
        return event
    
    def get_confidence_breakdown(self, event: DetectionEvent) -> dict:
        """
        Get detailed confidence score breakdown for analysis.
        
        Args:
            event: Detection event to analyze
            
        Returns:
            Dictionary with confidence score breakdown
        """
        breakdown = {
            'detection_type': event.detection_type.value,
            'base_confidence': self._get_base_confidence(event.detection_type),
            'raw_audio_score': event.raw_scores.audio,
            'raw_video_score': event.raw_scores.video,
            'initial_confidence': event.initial_confidence,
            'final_confidence': event.final_confidence,
            'has_species_classification': event.species_classification is not None
        }
        
        if event.species_classification:
            breakdown['species_confidence'] = event.species_classification.confidence
            breakdown['species_name'] = event.species_classification.species_name
            breakdown['confidence_boost'] = event.final_confidence - event.initial_confidence
        
        return breakdown
    
    def _get_base_confidence(self, detection_type: DetectionType) -> float:
        """Get base confidence for a detection type."""
        if detection_type == DetectionType.AUDIO_ONLY:
            return self.audio_only_base
        elif detection_type == DetectionType.VIDEO_ONLY:
            return self.video_only_base
        elif detection_type == DetectionType.AUDIO_VIDEO:
            return self.audio_video_base
        return 0.5


class ConfidenceValidator:
    """Validates confidence scores and provides quality metrics."""
    
    @staticmethod
    def validate_raw_scores(raw_scores: RawScores) -> bool:
        """Validate raw confidence scores."""
        if raw_scores.audio is not None and not (0 <= raw_scores.audio <= 1):
            return False
        if raw_scores.video is not None and not (0 <= raw_scores.video <= 1):
            return False
        return True
    
    @staticmethod
    def validate_confidence_progression(
        initial_confidence: float,
        final_confidence: float,
        has_species_classification: bool
    ) -> bool:
        """Validate that confidence scores follow expected progression."""
        # Basic range validation
        if not (0 <= initial_confidence <= 1 and 0 <= final_confidence <= 1):
            return False
        
        # If there's species classification, final should be >= initial
        if has_species_classification and final_confidence < initial_confidence:
            return False
        
        # If no species classification, final should equal initial
        if not has_species_classification and final_confidence != initial_confidence:
            return False
        
        return True
    
    @staticmethod
    def get_confidence_quality_metrics(events: list[DetectionEvent]) -> dict:
        """Calculate quality metrics for a set of detection events."""
        if not events:
            return {}
        
        total_events = len(events)
        high_confidence_events = sum(1 for e in events if e.final_confidence >= 0.8)
        medium_confidence_events = sum(1 for e in events if 0.5 <= e.final_confidence < 0.8)
        low_confidence_events = sum(1 for e in events if e.final_confidence < 0.5)
        
        events_with_species = sum(1 for e in events if e.species_classification is not None)
        
        confidence_scores = [e.final_confidence for e in events]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        return {
            'total_events': total_events,
            'high_confidence_events': high_confidence_events,
            'medium_confidence_events': medium_confidence_events,
            'low_confidence_events': low_confidence_events,
            'high_confidence_ratio': high_confidence_events / total_events,
            'events_with_species': events_with_species,
            'species_identification_ratio': events_with_species / total_events,
            'average_confidence': avg_confidence,
            'min_confidence': min(confidence_scores),
            'max_confidence': max(confidence_scores)
        }