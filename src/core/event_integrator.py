"""
Event integration system for temporal correlation of audio and video detections.
Combines detection results from audio and video streams to create unified detection events.
"""

import uuid
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
from datetime import datetime

from ..models.data_models import (
    DetectionEvent, AudioDetection, VideoDetection, DetectionType, 
    RawScores, ProcessingConfig
)
from ..models.confidence import ConfidenceCalculator
from ..utils.logging import get_logger, LoggingContext
from ..utils.config import get_settings


class TemporalCorrelator:
    """Handles temporal correlation between audio and video detections."""
    
    def __init__(self, correlation_window: float = 2.0):
        """
        Initialize temporal correlator.
        
        Args:
            correlation_window: Time window for correlating events (seconds)
        """
        self.correlation_window = correlation_window
        self.logger = get_logger("pipeline")
    
    def find_overlapping_events(
        self, 
        audio_detections: List[AudioDetection], 
        video_detections: List[VideoDetection]
    ) -> Dict[str, List]:
        """
        Find temporally overlapping audio and video detections.
        
        Args:
            audio_detections: List of audio detections
            video_detections: List of video detections
            
        Returns:
            Dictionary with correlation results
        """
        correlations = {
            'audio_only': [],
            'video_only': [],
            'overlapping': []
        }
        
        # Track which detections have been matched
        matched_audio_indices = set()
        matched_video_indices = set()
        
        # Find overlapping detections
        for i, audio_det in enumerate(audio_detections):
            overlapping_video = []
            
            for j, video_det in enumerate(video_detections):
                if self._is_temporally_related(audio_det, video_det):
                    overlapping_video.append((j, video_det))
                    matched_video_indices.add(j)
            
            if overlapping_video:
                matched_audio_indices.add(i)
                correlations['overlapping'].append({
                    'audio_detection': audio_det,
                    'audio_index': i,
                    'video_detections': overlapping_video
                })
        
        # Add unmatched audio detections
        for i, audio_det in enumerate(audio_detections):
            if i not in matched_audio_indices:
                correlations['audio_only'].append((i, audio_det))
        
        # Add unmatched video detections
        for j, video_det in enumerate(video_detections):
            if j not in matched_video_indices:
                correlations['video_only'].append((j, video_det))
        
        return correlations
    
    def _is_temporally_related(
        self, 
        audio_detection: AudioDetection, 
        video_detection: VideoDetection
    ) -> bool:
        """
        Check if audio and video detections are temporally related.
        
        Args:
            audio_detection: Audio detection
            video_detection: Video detection
            
        Returns:
            True if detections overlap within correlation window
        """
        # Check if video timestamp falls within audio detection window
        if audio_detection.start_time <= video_detection.timestamp <= audio_detection.end_time:
            return True
        
        # Check if video timestamp is within correlation window of audio detection
        audio_center = (audio_detection.start_time + audio_detection.end_time) / 2
        time_diff = abs(video_detection.timestamp - audio_center)
        
        return time_diff <= self.correlation_window
    
    def merge_overlapping_audio_detections(
        self, 
        audio_detections: List[AudioDetection],
        merge_threshold: float = 1.0
    ) -> List[AudioDetection]:
        """
        Merge overlapping or nearby audio detections.
        
        Args:
            audio_detections: List of audio detections
            merge_threshold: Maximum gap between detections to merge (seconds)
            
        Returns:
            List of merged audio detections
        """
        if not audio_detections:
            return []
        
        # Sort by start time
        sorted_detections = sorted(audio_detections, key=lambda x: x.start_time)
        merged = []
        current_detection = sorted_detections[0]
        
        for next_detection in sorted_detections[1:]:
            # Check if detections should be merged
            gap = next_detection.start_time - current_detection.end_time
            
            if gap <= merge_threshold:
                # Merge detections
                current_detection = AudioDetection(
                    start_time=current_detection.start_time,
                    end_time=max(current_detection.end_time, next_detection.end_time),
                    confidence=max(current_detection.confidence, next_detection.confidence),
                    frequency_range=current_detection.frequency_range or next_detection.frequency_range
                )
            else:
                # Add current detection and start new one
                merged.append(current_detection)
                current_detection = next_detection
        
        # Add the last detection
        merged.append(current_detection)
        
        return merged
    
    def cluster_video_detections(
        self,
        video_detections: List[VideoDetection],
        time_threshold: float = 0.5
    ) -> List[List[VideoDetection]]:
        """
        Cluster video detections that are close in time.
        
        Args:
            video_detections: List of video detections
            time_threshold: Maximum time difference for clustering (seconds)
            
        Returns:
            List of detection clusters
        """
        if not video_detections:
            return []
        
        # Sort by timestamp
        sorted_detections = sorted(video_detections, key=lambda x: x.timestamp)
        clusters = []
        current_cluster = [sorted_detections[0]]
        
        for detection in sorted_detections[1:]:
            # Check if detection belongs to current cluster
            if detection.timestamp - current_cluster[-1].timestamp <= time_threshold:
                current_cluster.append(detection)
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = [detection]
        
        # Add the last cluster
        clusters.append(current_cluster)
        
        return clusters


class EventIntegrator:
    """Integrates audio and video detections into unified detection events."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize event integrator.
        
        Args:
            config: Processing configuration
        """
        self.logger = get_logger("pipeline")
        
        if config is None:
            settings = get_settings()
            self.correlation_window = settings.temporal_correlation_window
            self.audio_base_confidence = settings.audio_only_base_confidence
            self.video_base_confidence = settings.video_only_base_confidence
            self.audio_video_base_confidence = settings.audio_video_base_confidence
        else:
            self.correlation_window = config.temporal_correlation_window
            self.audio_base_confidence = config.audio_only_base_confidence
            self.video_base_confidence = config.video_only_base_confidence
            self.audio_video_base_confidence = config.audio_video_base_confidence
        
        # Initialize components
        self.correlator = TemporalCorrelator(self.correlation_window)
        self.confidence_calculator = ConfidenceCalculator(
            audio_only_base=self.audio_base_confidence,
            video_only_base=self.video_base_confidence,
            audio_video_base=self.audio_video_base_confidence
        )
    
    def integrate_detections(
        self,
        audio_detections: List[AudioDetection],
        video_detections: List[VideoDetection]
    ) -> List[DetectionEvent]:
        """
        Integrate audio and video detections into unified events.
        
        Args:
            audio_detections: List of audio detections
            video_detections: List of video detections
            
        Returns:
            List of integrated detection events
        """
        with LoggingContext("Integrating audio and video detections", "pipeline") as ctx:
            ctx.log_metric("audio_detections", len(audio_detections))
            ctx.log_metric("video_detections", len(video_detections))
            
            # Preprocess detections
            ctx.log_progress("Preprocessing detections")
            merged_audio = self.correlator.merge_overlapping_audio_detections(audio_detections)
            video_clusters = self.correlator.cluster_video_detections(video_detections)
            
            ctx.log_metric("merged_audio_detections", len(merged_audio))
            ctx.log_metric("video_clusters", len(video_clusters))
            
            # Find temporal correlations
            ctx.log_progress("Finding temporal correlations")
            flat_video = [det for cluster in video_clusters for det in cluster]
            correlations = self.correlator.find_overlapping_events(merged_audio, flat_video)
            
            ctx.log_metric("audio_only_events", len(correlations['audio_only']))
            ctx.log_metric("video_only_events", len(correlations['video_only']))
            ctx.log_metric("overlapping_events", len(correlations['overlapping']))
            
            # Create detection events
            ctx.log_progress("Creating detection events")
            events = []
            
            # Process audio-only events
            for audio_idx, audio_det in correlations['audio_only']:
                event = self._create_audio_only_event(audio_det)
                events.append(event)
            
            # Process video-only events (group by clusters)
            video_only_by_cluster = self._group_video_only_by_cluster(
                correlations['video_only'], video_clusters
            )
            for cluster_detections in video_only_by_cluster:
                event = self._create_video_only_event(cluster_detections)
                events.append(event)
            
            # Process overlapping events
            for overlap_data in correlations['overlapping']:
                event = self._create_audio_video_event(overlap_data)
                events.append(event)
            
            ctx.log_metric("total_events_created", len(events))
            
            return events
    
    def _create_audio_only_event(self, audio_detection: AudioDetection) -> DetectionEvent:
        """Create a detection event for audio-only detection."""
        event_id = f"evt_audio_{uuid.uuid4().hex[:8]}"
        
        raw_scores = RawScores(audio=audio_detection.confidence, video=None)
        
        event = DetectionEvent(
            event_id=event_id,
            start_time=audio_detection.start_time,
            end_time=audio_detection.end_time,
            detection_type=DetectionType.AUDIO_ONLY,
            initial_confidence=0.0,  # Will be calculated
            final_confidence=0.0,    # Will be calculated
            audio_detection=audio_detection,
            video_detection=None,
            raw_scores=raw_scores,
            species_classification=None
        )
        
        # Calculate confidence scores
        event = self.confidence_calculator.update_event_confidence(event)
        
        return event
    
    def _create_video_only_event(self, video_detections: List[VideoDetection]) -> DetectionEvent:
        """Create a detection event for video-only detections."""
        if not video_detections:
            raise ValueError("Cannot create event from empty video detections")
        
        event_id = f"evt_video_{uuid.uuid4().hex[:8]}"
        
        # Use the detection with highest confidence as primary
        primary_detection = max(video_detections, key=lambda x: x.confidence)
        
        # Calculate time bounds
        timestamps = [det.timestamp for det in video_detections]
        start_time = min(timestamps)
        end_time = max(timestamps)
        
        # If all detections are at the same timestamp, add a small duration
        if start_time == end_time:
            end_time = start_time + 1.0
        
        raw_scores = RawScores(audio=None, video=primary_detection.confidence)
        
        event = DetectionEvent(
            event_id=event_id,
            start_time=start_time,
            end_time=end_time,
            detection_type=DetectionType.VIDEO_ONLY,
            initial_confidence=0.0,  # Will be calculated
            final_confidence=0.0,    # Will be calculated
            audio_detection=None,
            video_detection=primary_detection,
            raw_scores=raw_scores,
            species_classification=None
        )
        
        # Calculate confidence scores
        event = self.confidence_calculator.update_event_confidence(event)
        
        return event
    
    def _create_audio_video_event(self, overlap_data: Dict) -> DetectionEvent:
        """Create a detection event for overlapping audio and video detections."""
        audio_detection = overlap_data['audio_detection']
        video_detections = [v[1] for v in overlap_data['video_detections']]
        
        event_id = f"evt_combined_{uuid.uuid4().hex[:8]}"
        
        # Use the video detection with highest confidence
        primary_video = max(video_detections, key=lambda x: x.confidence)
        
        # Use audio detection time bounds
        start_time = audio_detection.start_time
        end_time = audio_detection.end_time
        
        raw_scores = RawScores(
            audio=audio_detection.confidence,
            video=primary_video.confidence
        )
        
        event = DetectionEvent(
            event_id=event_id,
            start_time=start_time,
            end_time=end_time,
            detection_type=DetectionType.AUDIO_VIDEO,
            initial_confidence=0.0,  # Will be calculated
            final_confidence=0.0,    # Will be calculated
            audio_detection=audio_detection,
            video_detection=primary_video,
            raw_scores=raw_scores,
            species_classification=None
        )
        
        # Calculate confidence scores
        event = self.confidence_calculator.update_event_confidence(event)
        
        return event
    
    def _group_video_only_by_cluster(
        self, 
        video_only_detections: List[Tuple[int, VideoDetection]], 
        video_clusters: List[List[VideoDetection]]
    ) -> List[List[VideoDetection]]:
        """Group video-only detections by their original clusters."""
        # Create mapping from detection to cluster
        detection_to_cluster = {}
        for cluster_idx, cluster in enumerate(video_clusters):
            for detection in cluster:
                detection_to_cluster[id(detection)] = cluster_idx
        
        # Group video-only detections by cluster
        cluster_groups = defaultdict(list)
        singleton_detections = []
        
        for det_idx, detection in video_only_detections:
            det_id = id(detection)
            if det_id in detection_to_cluster:
                cluster_idx = detection_to_cluster[det_id]
                cluster_groups[cluster_idx].append(detection)
            else:
                # Single detection not in any cluster
                singleton_detections.append([detection])
        
        # Combine cluster groups and singletons
        result = list(cluster_groups.values()) + singleton_detections
        
        return result
    
    def get_integration_statistics(self, events: List[DetectionEvent]) -> Dict[str, Any]:
        """
        Get statistics about the integration results.
        
        Args:
            events: List of detection events
            
        Returns:
            Dictionary with integration statistics
        """
        if not events:
            return {
                'total_events': 0,
                'audio_only_events': 0,
                'video_only_events': 0,
                'audio_video_events': 0,
                'average_confidence': 0.0,
                'confidence_distribution': {}
            }
        
        # Count by type
        type_counts = {
            DetectionType.AUDIO_ONLY: 0,
            DetectionType.VIDEO_ONLY: 0,
            DetectionType.AUDIO_VIDEO: 0
        }
        
        confidences = []
        
        for event in events:
            type_counts[event.detection_type] += 1
            confidences.append(event.final_confidence)
        
        # Confidence distribution
        conf_dist = {
            'high (>0.8)': sum(1 for c in confidences if c > 0.8),
            'medium (0.5-0.8)': sum(1 for c in confidences if 0.5 <= c <= 0.8),
            'low (<0.5)': sum(1 for c in confidences if c < 0.5)
        }
        
        return {
            'total_events': len(events),
            'audio_only_events': type_counts[DetectionType.AUDIO_ONLY],
            'video_only_events': type_counts[DetectionType.VIDEO_ONLY],
            'audio_video_events': type_counts[DetectionType.AUDIO_VIDEO],
            'average_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'confidence_distribution': conf_dist,
            'temporal_span': max(e.end_time for e in events) - min(e.start_time for e in events)
        }
    
    def filter_events_by_confidence(
        self, 
        events: List[DetectionEvent], 
        min_confidence: float = 0.5
    ) -> List[DetectionEvent]:
        """
        Filter events by minimum confidence threshold.
        
        Args:
            events: List of detection events
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of events
        """
        filtered = [event for event in events if event.final_confidence >= min_confidence]
        
        self.logger.info(
            f"Filtered events by confidence >= {min_confidence}: "
            f"{len(filtered)}/{len(events)} events retained"
        )
        
        return filtered
    
    def sort_events_by_time(self, events: List[DetectionEvent]) -> List[DetectionEvent]:
        """Sort events chronologically by start time."""
        return sorted(events, key=lambda x: x.start_time)