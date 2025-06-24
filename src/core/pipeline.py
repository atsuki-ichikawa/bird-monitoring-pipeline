"""
Main pipeline orchestrator for the bird monitoring system.
Coordinates all components to process videos and detect bird events.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import traceback

from ..models.data_models import PipelineResult, ProcessingConfig, DetectionEvent
from ..utils.logging import (
    get_logger, LoggingContext, log_processing_start, log_processing_complete,
    log_error, log_performance_metric
)
from ..utils.config import get_settings

from .media_processor import MediaProcessor
from .audio_detector import AudioDetector
from .video_detector import VideoDetector
from .event_integrator import EventIntegrator
from .species_classifier import SpeciesClassifier


class BirdMonitoringPipeline:
    """Main pipeline for processing video files and detecting bird events."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the bird monitoring pipeline.
        
        Args:
            config: Processing configuration (uses defaults if None)
        """
        self.logger = get_logger("pipeline")
        self.settings = get_settings()
        
        # Store configuration
        self.config = config
        
        # Initialize components
        self.media_processor = None
        self.audio_detector = None
        self.video_detector = None
        self.event_integrator = None
        self.species_classifier = None
        
        # Processing state
        self.current_result: Optional[PipelineResult] = None
        self.temp_files = []
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        try:
            with LoggingContext("Initializing pipeline components") as ctx:
                # Initialize media processor
                self.media_processor = MediaProcessor()
                ctx.log_progress("Media processor initialized")
                
                # Initialize audio detector
                self.audio_detector = AudioDetector()
                ctx.log_progress("Audio detector initialized")
                
                # Initialize video detector
                self.video_detector = VideoDetector()
                ctx.log_progress("Video detector initialized")
                
                # Initialize event integrator
                self.event_integrator = EventIntegrator(self.config)
                ctx.log_progress("Event integrator initialized")
                
                # Initialize species classifier
                self.species_classifier = SpeciesClassifier()
                ctx.log_progress("Species classifier initialized")
                
                # Log component status
                self._log_component_status()
                
        except Exception as e:
            log_error("pipeline", e, "Component initialization")
            raise RuntimeError(f"Failed to initialize pipeline components: {e}")
    
    def _log_component_status(self) -> None:
        """Log the status of all components."""
        status = {
            'audio_model_available': self.audio_detector.is_model_available(),
            'video_model_available': self.video_detector.is_model_available(),
            'species_model_available': self.species_classifier.is_model_available(),
            'media_processor_ready': self.media_processor is not None
        }
        
        self.logger.info(f"Component status: {status}")
        
        # Log warnings for missing models
        if not status['audio_model_available']:
            self.logger.warning("BirdNET model not available - using mock audio detection")
        if not status['video_model_available']:
            self.logger.warning("YOLO model not available - using mock video detection")
        if not status['species_model_available']:
            self.logger.warning("TransFG model not available - using mock species classification")
    
    def process_video(
        self, 
        video_path: Path, 
        output_dir: Optional[Path] = None,
        config: Optional[ProcessingConfig] = None
    ) -> PipelineResult:
        """
        Process a video file through the complete pipeline.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory for output files (auto-generated if None)
            config: Processing configuration (uses instance config if None)
            
        Returns:
            Complete pipeline processing result
        """
        video_path = Path(video_path)
        
        # Use provided config or instance config
        processing_config = config or self.config
        if processing_config is None:
            # Create default config
            if output_dir is None:
                output_dir = self.settings.data_dir / "output" / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            processing_config = self.settings.create_processing_config(
                str(video_path), str(output_dir)
            )
        
        # Set up output directory
        if output_dir is None:
            output_dir = Path(processing_config.output_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize result
        self.current_result = PipelineResult(
            video_path=str(video_path),
            processing_started=datetime.now(),
            processing_parameters=processing_config.dict()
        )
        
        # Log processing start
        config_summary = {
            'video_path': str(video_path),
            'output_dir': str(output_dir),
            'audio_threshold': processing_config.audio_confidence_threshold,
            'video_threshold': processing_config.video_confidence_threshold
        }
        log_processing_start(str(video_path), config_summary)
        
        try:
            with LoggingContext(f"Processing video {video_path.name}") as ctx:
                # Stage 1: Media preprocessing
                ctx.log_progress("Stage 1: Media preprocessing")
                start_time = datetime.now()
                
                audio_path, frames_dir = self._stage_1_media_preprocessing(
                    video_path, output_dir
                )
                
                log_performance_metric("Media preprocessing", 
                                     (datetime.now() - start_time).total_seconds())
                
                # Stage 2: Parallel detection
                ctx.log_progress("Stage 2: Parallel detection")
                start_time = datetime.now()
                
                audio_detections, video_detections = self._stage_2_parallel_detection(
                    audio_path, frames_dir
                )
                
                log_performance_metric("Parallel detection", 
                                     (datetime.now() - start_time).total_seconds())
                
                # Stage 3: Event integration
                ctx.log_progress("Stage 3: Event integration")
                start_time = datetime.now()
                
                integrated_events = self._stage_3_event_integration(
                    audio_detections, video_detections
                )
                
                log_performance_metric("Event integration", 
                                     (datetime.now() - start_time).total_seconds())
                
                # Stage 4: Species classification
                ctx.log_progress("Stage 4: Species classification")
                start_time = datetime.now()
                
                final_events = self._stage_4_species_classification(
                    integrated_events, frames_dir
                )
                
                log_performance_metric("Species classification", 
                                     (datetime.now() - start_time).total_seconds())
                
                # Stage 5: Results finalization
                ctx.log_progress("Stage 5: Results finalization")
                self._stage_5_results_finalization(final_events, output_dir)
                
                # Complete processing
                self.current_result.processing_completed = datetime.now()
                
                # Log completion
                results_summary = {
                    'total_events': self.current_result.total_events,
                    'audio_only': self.current_result.audio_only_events,
                    'video_only': self.current_result.video_only_events,
                    'audio_video': self.current_result.audio_video_events,
                    'species_identified': self.current_result.species_identified,
                    'processing_time': (self.current_result.processing_completed - 
                                      self.current_result.processing_started).total_seconds()
                }
                log_processing_complete(str(video_path), results_summary)
                
                return self.current_result
                
        except Exception as e:
            log_error("pipeline", e, f"Video processing: {video_path}")
            
            # Update result with error status
            if self.current_result:
                self.current_result.processing_completed = datetime.now()
            
            raise RuntimeError(f"Pipeline processing failed: {e}")
        
        finally:
            # Cleanup temporary files if configured
            self._cleanup_temporary_files()
    
    def _stage_1_media_preprocessing(
        self, 
        video_path: Path, 
        output_dir: Path
    ) -> tuple[Path, Path]:
        """Stage 1: Extract and preprocess audio and video streams."""
        with LoggingContext("Media preprocessing", "pipeline") as ctx:
            # Validate video file
            if not self.media_processor.validate_video_file(video_path):
                raise ValueError(f"Invalid video file: {video_path}")
            
            # Extract audio and video
            audio_path, frames_dir = self.media_processor.extract_audio_video(
                video_path,
                output_audio_path=output_dir / "audio.wav",
                output_video_dir=output_dir / "frames"
            )
            
            # Track temporary files
            self.temp_files.extend([audio_path, frames_dir])
            
            # Log extraction info
            video_info = self.media_processor.get_video_info()
            if video_info:
                ctx.log_metric("video_duration", video_info['duration'], "seconds")
                ctx.log_metric("video_fps", video_info['fps'])
                ctx.log_metric("has_audio", video_info['has_audio'])
            
            frame_count = len(list(frames_dir.glob("*.jpg")))
            ctx.log_metric("extracted_frames", frame_count)
            
            return audio_path, frames_dir
    
    def _stage_2_parallel_detection(
        self, 
        audio_path: Path, 
        frames_dir: Path
    ) -> tuple[List, List]:
        """Stage 2: Run parallel audio and video detection."""
        with LoggingContext("Parallel detection", "pipeline") as ctx:
            # Audio detection
            ctx.log_progress("Running audio detection")
            audio_detections = self.audio_detector.process_audio_file(audio_path)
            ctx.log_metric("audio_detections", len(audio_detections))
            
            # Video detection
            ctx.log_progress("Running video detection")
            video_detections = self.video_detector.process_frames_directory(frames_dir)
            ctx.log_metric("video_detections", len(video_detections))
            
            return audio_detections, video_detections
    
    def _stage_3_event_integration(
        self, 
        audio_detections: List, 
        video_detections: List
    ) -> List[DetectionEvent]:
        """Stage 3: Integrate audio and video detections into events."""
        with LoggingContext("Event integration", "pipeline") as ctx:
            # Integrate detections
            events = self.event_integrator.integrate_detections(
                audio_detections, video_detections
            )
            
            # Sort events chronologically
            events = self.event_integrator.sort_events_by_time(events)
            
            # Update result statistics
            for event in events:
                self.current_result.add_event(event)
            
            ctx.log_metric("integrated_events", len(events))
            
            return events
    
    def _stage_4_species_classification(
        self, 
        events: List[DetectionEvent], 
        frames_dir: Path
    ) -> List[DetectionEvent]:
        """Stage 4: Classify species for events with video components."""
        with LoggingContext("Species classification", "pipeline") as ctx:
            # Filter events that can be classified
            video_events = [e for e in events if e.video_detection is not None]
            ctx.log_metric("classifiable_events", len(video_events))
            
            if not video_events:
                ctx.log_progress("No video events to classify")
                return events
            
            # Run species classification
            classified_events = self.species_classifier.classify_detection_events(
                events, frames_dir
            )
            
            # Update result statistics
            classified_count = sum(
                1 for e in classified_events 
                if e.species_classification is not None
            )
            ctx.log_metric("classified_events", classified_count)
            
            return classified_events
    
    def _stage_5_results_finalization(
        self, 
        events: List[DetectionEvent], 
        output_dir: Path
    ) -> None:
        """Stage 5: Finalize and export results."""
        with LoggingContext("Results finalization", "pipeline") as ctx:
            # Update events in result
            self.current_result.events = events
            
            # Export main results
            results_file = output_dir / "detection_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.current_result.to_export_dict(), f, indent=2)
            
            ctx.log_progress(f"Exported results to {results_file}")
            
            # Export component-specific results
            self._export_component_results(events, output_dir)
            
            # Generate summary statistics
            stats_file = output_dir / "statistics.json"
            self._export_statistics(events, stats_file)
            
            ctx.log_metric("output_files_created", len(list(output_dir.glob("*.json"))))
    
    def _export_component_results(self, events: List[DetectionEvent], output_dir: Path) -> None:
        """Export results from individual components."""
        # Export audio detections
        audio_detections = [e.audio_detection for e in events if e.audio_detection]
        if audio_detections:
            self.audio_detector.export_detections(
                audio_detections, output_dir / "audio_detections.json"
            )
        
        # Export video detections
        video_detections = [e.video_detection for e in events if e.video_detection]
        if video_detections:
            self.video_detector.export_detections(
                video_detections, output_dir / "video_detections.json"
            )
        
        # Export species classifications
        if any(e.species_classification for e in events):
            self.species_classifier.export_classifications(
                events, output_dir / "species_classifications.json"
            )
    
    def _export_statistics(self, events: List[DetectionEvent], output_file: Path) -> None:
        """Export comprehensive statistics."""
        stats = {
            'processing_info': {
                'video_path': self.current_result.video_path,
                'processing_started': self.current_result.processing_started.isoformat(),
                'processing_completed': self.current_result.processing_completed.isoformat(),
                'processing_duration_seconds': (
                    self.current_result.processing_completed - 
                    self.current_result.processing_started
                ).total_seconds()
            },
            'detection_statistics': self.event_integrator.get_integration_statistics(events),
            'audio_statistics': self.audio_detector.get_detection_statistics(
                [e.audio_detection for e in events if e.audio_detection]
            ),
            'video_statistics': self.video_detector.get_detection_statistics(
                [e.video_detection for e in events if e.video_detection]
            ),
            'species_statistics': self.species_classifier.get_classification_statistics(events),
            'component_info': {
                'audio_model': self.audio_detector.get_model_info(),
                'video_model': self.video_detector.get_model_info(),
                'species_model': self.species_classifier.get_model_info()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _cleanup_temporary_files(self) -> None:
        """Clean up temporary files created during processing."""
        try:
            if hasattr(self.media_processor, 'cleanup_temp_files'):
                self.media_processor.cleanup_temp_files()
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temporary files: {e}")
    
    def process_batch(
        self, 
        video_paths: List[Path], 
        output_base_dir: Path,
        continue_on_error: bool = True
    ) -> List[PipelineResult]:
        """
        Process multiple video files in batch.
        
        Args:
            video_paths: List of video file paths
            output_base_dir: Base directory for all outputs
            continue_on_error: Whether to continue processing if one video fails
            
        Returns:
            List of pipeline results (may contain None for failed videos)
        """
        results = []
        
        with LoggingContext(f"Batch processing {len(video_paths)} videos") as ctx:
            for i, video_path in enumerate(video_paths):
                try:
                    ctx.log_progress(f"Processing video {i+1}/{len(video_paths)}: {video_path.name}")
                    
                    # Create output directory for this video
                    video_output_dir = output_base_dir / f"{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Process video
                    result = self.process_video(video_path, video_output_dir)
                    results.append(result)
                    
                except Exception as e:
                    log_error("pipeline", e, f"Batch processing video: {video_path}")
                    
                    if continue_on_error:
                        results.append(None)  # Placeholder for failed processing
                        ctx.log_progress(f"Skipping failed video: {video_path.name}")
                    else:
                        raise RuntimeError(f"Batch processing failed at video {video_path}: {e}")
            
            successful_results = [r for r in results if r is not None]
            ctx.log_metric("successful_videos", len(successful_results))
            ctx.log_metric("failed_videos", len(results) - len(successful_results))
        
        return results
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get status information for all pipeline components."""
        return {
            'media_processor': {
                'available': self.media_processor is not None,
                'supported_formats': self.media_processor.get_supported_formats() if self.media_processor else []
            },
            'audio_detector': self.audio_detector.get_model_info() if self.audio_detector else {},
            'video_detector': self.video_detector.get_model_info() if self.video_detector else {},
            'species_classifier': self.species_classifier.get_model_info() if self.species_classifier else {},
            'event_integrator': {
                'available': self.event_integrator is not None,
                'correlation_window': getattr(self.event_integrator, 'correlation_window', None)
            }
        }
    
    def estimate_processing_requirements(self, video_path: Path) -> Dict[str, Any]:
        """
        Estimate processing time and resource requirements.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with processing estimates
        """
        if not self.media_processor:
            return {}
        
        return self.media_processor.estimate_processing_time(video_path)