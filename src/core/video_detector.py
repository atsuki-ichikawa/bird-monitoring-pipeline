"""
Video detection pipeline using YOLOv10 for bird object detection.
Processes video frames and identifies bird objects with bounding boxes and confidence scores.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Generator
import json
from datetime import datetime
from ultralytics import YOLO

from ..models.data_models import VideoDetection, BoundingBox
from ..utils.logging import get_logger, LoggingContext
from ..utils.config import get_settings


class VideoDetector:
    """Video detection pipeline using YOLOv10 for bird object detection."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize video detector.
        
        Args:
            model_path: Path to YOLOv10 model file
        """
        self.logger = get_logger("video")
        self.settings = get_settings()
        
        # Model configuration
        self.model_path = model_path or self.settings.yolo_model_path
        self.model = None
        
        # Detection parameters
        self.confidence_threshold = self.settings.video_confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() and self.settings.gpu_enabled else "cpu"
        
        # COCO class IDs for birds (YOLOv10 is typically trained on COCO dataset)
        self.bird_class_ids = [14]  # 'bird' class in COCO dataset
        self.class_names = {14: 'bird'}
        
        # Initialize model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLOv10 model."""
        try:
            with LoggingContext("Loading YOLOv10 model", "video") as ctx:
                if self.model_path is None:
                    self.logger.info("No YOLO model path specified, using default YOLOv10n")
                    self.model = YOLO('yolov10n.pt')  # Download default model
                elif Path(self.model_path).exists():
                    self.model = YOLO(str(self.model_path))
                else:
                    self.logger.warning(f"YOLO model not found at {self.model_path}, using default")
                    self.model = YOLO('yolov10n.pt')
                
                # Move model to device
                self.model.to(self.device)
                
                ctx.log_metric("device", self.device)
                ctx.log_metric("model_loaded", True)
                
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.logger.info("Creating mock video detector")
            self._create_mock_model()
    
    def _create_mock_model(self) -> None:
        """Create a mock model for testing when YOLO is not available."""
        self.logger.info("Creating mock video detection model")
        self.model = "mock_model"
    
    def process_frames_directory(self, frames_dir: Path) -> List[VideoDetection]:
        """
        Process a directory of extracted video frames.
        
        Args:
            frames_dir: Directory containing video frames
            
        Returns:
            List of video detection results
        """
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        
        with LoggingContext(f"Processing frames from {frames_dir.name}", "video") as ctx:
            # Get all frame files
            frame_files = sorted(frames_dir.glob("frame_*.jpg"))
            ctx.log_metric("total_frames", len(frame_files))
            
            if not frame_files:
                self.logger.warning("No frame files found in directory")
                return []
            
            detections = []
            processed_count = 0
            
            for frame_file in frame_files:
                try:
                    # Extract timestamp from filename
                    timestamp = self._extract_timestamp_from_filename(frame_file)
                    
                    # Load and process frame
                    frame = cv2.imread(str(frame_file))
                    if frame is None:
                        self.logger.warning(f"Could not load frame: {frame_file}")
                        continue
                    
                    # Run detection on frame
                    frame_detections = self._detect_birds_in_frame(frame, timestamp)
                    detections.extend(frame_detections)
                    
                    processed_count += 1
                    
                    # Log progress periodically
                    if processed_count % 100 == 0:
                        ctx.log_progress(f"Processed {processed_count}/{len(frame_files)} frames")
                
                except Exception as e:
                    self.logger.warning(f"Error processing frame {frame_file}: {e}")
                    continue
            
            ctx.log_metric("processed_frames", processed_count)
            ctx.log_metric("detections_found", len(detections))
            
            return detections
    
    def process_frame_iterator(
        self, 
        frame_iterator: Generator[Tuple[float, np.ndarray], None, None]
    ) -> List[VideoDetection]:
        """
        Process frames from an iterator.
        
        Args:
            frame_iterator: Iterator yielding (timestamp, frame) tuples
            
        Returns:
            List of video detection results
        """
        with LoggingContext("Processing frame iterator", "video") as ctx:
            detections = []
            processed_count = 0
            
            for timestamp, frame in frame_iterator:
                try:
                    # Run detection on frame
                    frame_detections = self._detect_birds_in_frame(frame, timestamp)
                    detections.extend(frame_detections)
                    
                    processed_count += 1
                    
                    # Log progress periodically
                    if processed_count % 100 == 0:
                        ctx.log_progress(f"Processed {processed_count} frames")
                
                except Exception as e:
                    self.logger.warning(f"Error processing frame at {timestamp}s: {e}")
                    continue
            
            ctx.log_metric("processed_frames", processed_count)
            ctx.log_metric("detections_found", len(detections))
            
            return detections
    
    def _detect_birds_in_frame(self, frame: np.ndarray, timestamp: float) -> List[VideoDetection]:
        """
        Detect birds in a single frame.
        
        Args:
            frame: Frame image array
            timestamp: Frame timestamp in seconds
            
        Returns:
            List of detections in this frame
        """
        if self.model == "mock_model":
            return self._mock_detection(frame, timestamp)
        
        try:
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            detections = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                # Extract detection data
                xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes
                confidences = boxes.conf.cpu().numpy()  # Confidence scores
                class_ids = boxes.cls.cpu().numpy().astype(int)  # Class IDs
                
                # Filter for bird detections
                for i, class_id in enumerate(class_ids):
                    if class_id in self.bird_class_ids:
                        confidence = float(confidences[i])
                        
                        # Apply confidence threshold
                        if confidence >= self.confidence_threshold:
                            # Extract bounding box coordinates
                            x1, y1, x2, y2 = xyxy[i]
                            
                            # Create bounding box
                            bbox = BoundingBox(
                                timestamp=timestamp,
                                x=float(x1),
                                y=float(y1),
                                width=float(x2 - x1),
                                height=float(y2 - y1)
                            )
                            
                            # Create detection
                            detection = VideoDetection(
                                timestamp=timestamp,
                                confidence=confidence,
                                bounding_box=bbox
                            )
                            
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.warning(f"YOLO detection failed: {e}")
            return []
    
    def _mock_detection(self, frame: np.ndarray, timestamp: float) -> List[VideoDetection]:
        """
        Mock detection for testing purposes.
        
        Args:
            frame: Frame image array
            timestamp: Frame timestamp
            
        Returns:
            Mock detection results
        """
        # Simulate occasional bird detections
        np.random.seed(int(timestamp * 1000) % 1000)
        
        detections = []
        
        # 20% chance of detecting a bird
        if np.random.random() < 0.2:
            height, width = frame.shape[:2]
            
            # Generate random bounding box
            x = np.random.randint(0, max(1, width - 100))
            y = np.random.randint(0, max(1, height - 100))
            w = np.random.randint(50, min(200, width - x))
            h = np.random.randint(50, min(200, height - y))
            
            # Generate random confidence
            confidence = np.random.uniform(0.5, 0.95)
            
            # Create bounding box
            bbox = BoundingBox(
                timestamp=timestamp,
                x=float(x),
                y=float(y),
                width=float(w),
                height=float(h)
            )
            
            # Create detection
            detection = VideoDetection(
                timestamp=timestamp,
                confidence=confidence,
                bounding_box=bbox
            )
            
            detections.append(detection)
        
        return detections
    
    def _extract_timestamp_from_filename(self, frame_file: Path) -> float:
        """
        Extract timestamp from frame filename.
        
        Args:
            frame_file: Path to frame file
            
        Returns:
            Timestamp in seconds
        """
        try:
            # Format: frame_000001_12.345s.jpg
            parts = frame_file.stem.split('_')
            timestamp_str = parts[-1]  # "12.345s"
            return float(timestamp_str.rstrip('s'))
        except (IndexError, ValueError):
            self.logger.warning(f"Could not parse timestamp from {frame_file.name}")
            return 0.0
    
    def visualize_detections(
        self, 
        frame: np.ndarray, 
        detections: List[VideoDetection],
        save_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Visualize detections on a frame.
        
        Args:
            frame: Frame image array
            detections: List of detections for this frame
            save_path: Optional path to save visualization
            
        Returns:
            Frame with detection visualizations
        """
        vis_frame = frame.copy()
        
        for detection in detections:
            bbox = detection.bounding_box
            confidence = detection.confidence
            
            # Draw bounding box
            x1 = int(bbox.x)
            y1 = int(bbox.y)
            x2 = int(bbox.x + bbox.width)
            y2 = int(bbox.y + bbox.height)
            
            # Color based on confidence (green = high, red = low)
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            
            # Draw rectangle
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence text
            text = f"Bird {confidence:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_frame, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0], y1), color, -1)
            cv2.putText(vis_frame, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(str(save_path), vis_frame)
        
        return vis_frame
    
    def get_detection_statistics(self, detections: List[VideoDetection]) -> Dict[str, Any]:
        """
        Get statistics about video detections.
        
        Args:
            detections: List of video detections
            
        Returns:
            Dictionary with detection statistics
        """
        if not detections:
            return {
                'total_detections': 0,
                'unique_timestamps': 0,
                'average_confidence': 0.0,
                'confidence_distribution': {},
                'bounding_box_stats': {}
            }
        
        confidences = [d.confidence for d in detections]
        timestamps = list(set(d.timestamp for d in detections))
        
        # Bounding box statistics
        widths = [d.bounding_box.width for d in detections]
        heights = [d.bounding_box.height for d in detections]
        areas = [w * h for w, h in zip(widths, heights)]
        
        # Confidence distribution
        conf_dist = {
            'high (>0.8)': sum(1 for c in confidences if c > 0.8),
            'medium (0.5-0.8)': sum(1 for c in confidences if 0.5 <= c <= 0.8),
            'low (<0.5)': sum(1 for c in confidences if c < 0.5)
        }
        
        return {
            'total_detections': len(detections),
            'unique_timestamps': len(timestamps),
            'average_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'confidence_distribution': conf_dist,
            'bounding_box_stats': {
                'average_width': np.mean(widths),
                'average_height': np.mean(heights),
                'average_area': np.mean(areas),
                'min_area': np.min(areas),
                'max_area': np.max(areas)
            },
            'time_span': max(timestamps) - min(timestamps) if timestamps else 0.0
        }
    
    def export_detections(self, detections: List[VideoDetection], output_path: Path) -> None:
        """
        Export detections to JSON file.
        
        Args:
            detections: List of video detections
            output_path: Output file path
        """
        export_data = {
            'metadata': {
                'detector': 'YOLOv10',
                'processed_at': datetime.now().isoformat(),
                'settings': {
                    'confidence_threshold': self.confidence_threshold,
                    'device': self.device,
                    'model_path': str(self.model_path) if self.model_path else None
                }
            },
            'detections': [detection.dict() for detection in detections],
            'statistics': self.get_detection_statistics(detections)
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported {len(detections)} video detections to {output_path}")
    
    def create_detection_video(
        self, 
        frames_dir: Path,
        detections: List[VideoDetection],
        output_path: Path,
        fps: float = 10.0
    ) -> None:
        """
        Create a video with detection visualizations.
        
        Args:
            frames_dir: Directory containing original frames
            detections: List of detections to visualize
            output_path: Output video path
            fps: Output video frame rate
        """
        try:
            with LoggingContext(f"Creating detection video", "video") as ctx:
                # Group detections by timestamp
                detections_by_time = {}
                for detection in detections:
                    timestamp = detection.timestamp
                    if timestamp not in detections_by_time:
                        detections_by_time[timestamp] = []
                    detections_by_time[timestamp].append(detection)
                
                # Get frame files
                frame_files = sorted(frames_dir.glob("frame_*.jpg"))
                if not frame_files:
                    raise ValueError("No frame files found")
                
                # Get frame dimensions
                first_frame = cv2.imread(str(frame_files[0]))
                height, width = first_frame.shape[:2]
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                
                processed_frames = 0
                
                for frame_file in frame_files:
                    timestamp = self._extract_timestamp_from_filename(frame_file)
                    frame = cv2.imread(str(frame_file))
                    
                    if frame is None:
                        continue
                    
                    # Get detections for this timestamp
                    frame_detections = detections_by_time.get(timestamp, [])
                    
                    # Visualize detections
                    vis_frame = self.visualize_detections(frame, frame_detections)
                    
                    # Write frame
                    out.write(vis_frame)
                    processed_frames += 1
                
                out.release()
                ctx.log_metric("processed_frames", processed_frames)
                ctx.log_metric("output_path", str(output_path))
                
        except Exception as e:
            self.logger.error(f"Failed to create detection video: {e}")
            raise
    
    def is_model_available(self) -> bool:
        """Check if YOLO model is available."""
        return self.model is not None and self.model != "mock_model"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_path': str(self.model_path) if self.model_path else None,
            'model_available': self.is_model_available(),
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'bird_class_ids': self.bird_class_ids,
            'gpu_available': torch.cuda.is_available()
        }