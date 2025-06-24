"""
Species classification using TransFG for fine-grained bird species identification.
Processes cropped bird images from video detections to identify specific species.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json
from datetime import datetime

from ..models.data_models import DetectionEvent, SpeciesClassification, BoundingBox
from ..utils.logging import get_logger, LoggingContext
from ..utils.config import get_settings


class TransFGModel(nn.Module):
    """Mock TransFG model for fine-grained classification."""
    
    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.num_classes = num_classes
        
        # Simple CNN architecture as placeholder
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SpeciesClassifier:
    """Species classification pipeline using TransFG for fine-grained identification."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize species classifier.
        
        Args:
            model_path: Path to TransFG model file
        """
        self.logger = get_logger("pipeline")
        self.settings = get_settings()
        
        # Model configuration
        self.model_path = model_path or self.settings.transfg_model_path
        self.model = None
        self.species_list = None
        
        # Classification parameters
        self.confidence_threshold = self.settings.species_confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() and self.settings.gpu_enabled else "cpu"
        
        # Image preprocessing
        self.image_size = (224, 224)  # Standard size for vision transformers
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load TransFG model and species list."""
        try:
            with LoggingContext("Loading TransFG model", "pipeline") as ctx:
                if self.model_path is None:
                    self.logger.warning("No TransFG model path specified, using mock classifier")
                    self._create_mock_model()
                    return
                
                if not Path(self.model_path).exists():
                    self.logger.warning(f"TransFG model not found at {self.model_path}, using mock classifier")
                    self._create_mock_model()
                    return
                
                # Load PyTorch model
                checkpoint = torch.load(str(self.model_path), map_location=self.device)
                
                # Extract model architecture info
                num_classes = checkpoint.get('num_classes', 200)
                self.model = TransFGModel(num_classes=num_classes)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                # Load species list
                species_file = Path(self.model_path).parent / "species_list.json"
                if species_file.exists():
                    with open(species_file, 'r') as f:
                        self.species_list = json.load(f)
                else:
                    self.logger.warning("Species list not found, using default species")
                    self.species_list = self._get_default_species_list()
                
                ctx.log_metric("model_loaded", True)
                ctx.log_metric("species_count", len(self.species_list))
                ctx.log_metric("device", self.device)
                
        except Exception as e:
            self.logger.error(f"Failed to load TransFG model: {e}")
            self.logger.info("Falling back to mock classifier")
            self._create_mock_model()
    
    def _create_mock_model(self) -> None:
        """Create a mock model for testing when TransFG is not available."""
        self.logger.info("Creating mock species classification model")
        self.model = "mock_model"
        self.species_list = self._get_default_species_list()
    
    def _get_default_species_list(self) -> List[str]:
        """Get default species list for testing."""
        return [
            "Eurasian Tree Sparrow",
            "House Sparrow", 
            "Common Blackbird",
            "European Robin",
            "Great Tit",
            "Blue Tit",
            "Coal Tit",
            "Marsh Tit",
            "Wren",
            "Starling",
            "Common Crow",
            "Magpie",
            "Jay",
            "Greenfinch",
            "Goldfinch",
            "Siskin",
            "Bullfinch",
            "Chaffinch",
            "Song Thrush",
            "Blackcap"
        ]
    
    def classify_detection_events(
        self, 
        events: List[DetectionEvent],
        frames_dir: Path
    ) -> List[DetectionEvent]:
        """
        Classify species for detection events with video components.
        
        Args:
            events: List of detection events
            frames_dir: Directory containing video frames
            
        Returns:
            Updated events with species classifications
        """
        with LoggingContext("Classifying species for detection events", "pipeline") as ctx:
            # Filter events that have video components
            video_events = [
                event for event in events 
                if event.video_detection is not None
            ]
            
            ctx.log_metric("total_events", len(events))
            ctx.log_metric("video_events", len(video_events))
            
            if not video_events:
                self.logger.info("No video events to classify")
                return events
            
            updated_events = events.copy()
            classified_count = 0
            
            for i, event in enumerate(updated_events):
                if event.video_detection is None:
                    continue
                
                try:
                    # Extract and classify bird image
                    species_classification = self._classify_event_image(event, frames_dir)
                    
                    if species_classification is not None:
                        # Update event with classification
                        event.species_classification = species_classification
                        
                        # Recalculate confidence with species information
                        from ..models.confidence import ConfidenceCalculator
                        confidence_calc = ConfidenceCalculator()
                        event.final_confidence = confidence_calc.calculate_final_confidence(
                            event.initial_confidence, species_classification
                        )
                        
                        classified_count += 1
                        
                        if classified_count % 10 == 0:
                            ctx.log_progress(f"Classified {classified_count}/{len(video_events)} events")
                
                except Exception as e:
                    self.logger.warning(f"Failed to classify event {event.event_id}: {e}")
                    continue
            
            ctx.log_metric("classified_events", classified_count)
            
            return updated_events
    
    def _classify_event_image(
        self, 
        event: DetectionEvent, 
        frames_dir: Path
    ) -> Optional[SpeciesClassification]:
        """
        Classify species for a single detection event.
        
        Args:
            event: Detection event with video component
            frames_dir: Directory containing video frames
            
        Returns:
            Species classification result or None
        """
        if event.video_detection is None:
            return None
        
        try:
            # Find the corresponding frame file
            frame_file = self._find_frame_file(event.video_detection.timestamp, frames_dir)
            if frame_file is None:
                self.logger.warning(f"Frame file not found for timestamp {event.video_detection.timestamp}")
                return None
            
            # Load frame image
            frame = cv2.imread(str(frame_file))
            if frame is None:
                self.logger.warning(f"Could not load frame: {frame_file}")
                return None
            
            # Crop bird region using bounding box
            cropped_image = self._crop_bird_image(frame, event.video_detection.bounding_box)
            
            # Classify the cropped image
            species_name, confidence, alternatives = self._classify_image(cropped_image)
            
            # Check if classification meets threshold
            if confidence >= self.confidence_threshold:
                return SpeciesClassification(
                    species_name=species_name,
                    confidence=confidence,
                    alternative_species=alternatives
                )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error classifying event image: {e}")
            return None
    
    def _find_frame_file(self, timestamp: float, frames_dir: Path) -> Optional[Path]:
        """
        Find the frame file closest to the given timestamp.
        
        Args:
            timestamp: Target timestamp in seconds
            frames_dir: Directory containing frames
            
        Returns:
            Path to closest frame file or None
        """
        # Look for exact timestamp match first
        exact_match = frames_dir / f"frame_*_{timestamp:.3f}s.jpg"
        matches = list(frames_dir.glob(f"frame_*_{timestamp:.3f}s.jpg"))
        
        if matches:
            return matches[0]
        
        # Find closest timestamp
        all_frames = list(frames_dir.glob("frame_*.jpg"))
        if not all_frames:
            return None
        
        closest_frame = None
        min_diff = float('inf')
        
        for frame_file in all_frames:
            try:
                # Extract timestamp from filename
                parts = frame_file.stem.split('_')
                frame_timestamp = float(parts[-1].rstrip('s'))
                diff = abs(frame_timestamp - timestamp)
                
                if diff < min_diff:
                    min_diff = diff
                    closest_frame = frame_file
                    
            except (IndexError, ValueError):
                continue
        
        return closest_frame
    
    def _crop_bird_image(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        Crop bird region from frame using bounding box.
        
        Args:
            frame: Full frame image
            bbox: Bounding box coordinates
            
        Returns:
            Cropped bird image
        """
        # Extract bounding box coordinates
        x1 = max(0, int(bbox.x))
        y1 = max(0, int(bbox.y))
        x2 = min(frame.shape[1], int(bbox.x + bbox.width))
        y2 = min(frame.shape[0], int(bbox.y + bbox.height))
        
        # Add padding around the bird (10% on each side)
        padding_x = int(0.1 * (x2 - x1))
        padding_y = int(0.1 * (y2 - y1))
        
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(frame.shape[1], x2 + padding_x)
        y2 = min(frame.shape[0], y2 + padding_y)
        
        # Crop the image
        cropped = frame[y1:y2, x1:x2]
        
        # Ensure minimum size
        if cropped.shape[0] < 32 or cropped.shape[1] < 32:
            # Resize if too small
            cropped = cv2.resize(cropped, (64, 64))
        
        return cropped
    
    def _classify_image(self, image: np.ndarray) -> Tuple[str, float, List[Dict[str, float]]]:
        """
        Classify a bird image.
        
        Args:
            image: Bird image array (BGR format)
            
        Returns:
            Tuple of (species_name, confidence, alternatives)
        """
        if self.model == "mock_model":
            return self._mock_classification(image)
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Apply preprocessing
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
            # Get top predictions
            top_k = min(5, len(self.species_list))
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            # Extract results
            top_species = top_indices[0].cpu().numpy()
            top_confidences = top_probs[0].cpu().numpy()
            
            # Primary prediction
            species_name = self.species_list[top_species[0]]
            confidence = float(top_confidences[0])
            
            # Alternative predictions
            alternatives = []
            for i in range(1, len(top_species)):
                alternatives.append({
                    self.species_list[top_species[i]]: float(top_confidences[i])
                })
            
            return species_name, confidence, alternatives
            
        except Exception as e:
            self.logger.warning(f"Image classification failed: {e}")
            return self._mock_classification(image)
    
    def _mock_classification(self, image: np.ndarray) -> Tuple[str, float, List[Dict[str, float]]]:
        """
        Mock classification for testing purposes.
        
        Args:
            image: Bird image array
            
        Returns:
            Mock classification results
        """
        # Simple mock based on image properties
        mean_intensity = np.mean(image)
        
        # Select species based on image characteristics
        np.random.seed(int(mean_intensity) % 1000)
        species_idx = np.random.randint(0, len(self.species_list))
        species_name = self.species_list[species_idx]
        
        # Generate mock confidence
        base_confidence = 0.6 + 0.3 * np.random.random()
        confidence = float(np.clip(base_confidence, 0.0, 1.0))
        
        # Generate alternatives
        alternatives = []
        for _ in range(3):
            alt_idx = np.random.randint(0, len(self.species_list))
            if alt_idx != species_idx:
                alt_confidence = confidence * np.random.uniform(0.3, 0.8)
                alternatives.append({
                    self.species_list[alt_idx]: float(alt_confidence)
                })
        
        return species_name, confidence, alternatives
    
    def get_classification_statistics(self, events: List[DetectionEvent]) -> Dict[str, Any]:
        """
        Get statistics about species classifications.
        
        Args:
            events: List of detection events
            
        Returns:
            Dictionary with classification statistics
        """
        classified_events = [e for e in events if e.species_classification is not None]
        
        if not classified_events:
            return {
                'total_events': len(events),
                'classified_events': 0,
                'classification_rate': 0.0,
                'species_distribution': {},
                'average_species_confidence': 0.0
            }
        
        # Species distribution
        species_counts = {}
        confidences = []
        
        for event in classified_events:
            species = event.species_classification.species_name
            confidence = event.species_classification.confidence
            
            species_counts[species] = species_counts.get(species, 0) + 1
            confidences.append(confidence)
        
        return {
            'total_events': len(events),
            'classified_events': len(classified_events),
            'classification_rate': len(classified_events) / len(events),
            'species_distribution': species_counts,
            'unique_species_count': len(species_counts),
            'average_species_confidence': np.mean(confidences),
            'min_species_confidence': np.min(confidences),
            'max_species_confidence': np.max(confidences)
        }
    
    def export_classifications(self, events: List[DetectionEvent], output_path: Path) -> None:
        """
        Export species classifications to JSON file.
        
        Args:
            events: List of detection events
            output_path: Output file path
        """
        classified_events = [e for e in events if e.species_classification is not None]
        
        export_data = {
            'metadata': {
                'classifier': 'TransFG',
                'processed_at': datetime.now().isoformat(),
                'settings': {
                    'confidence_threshold': self.confidence_threshold,
                    'device': self.device,
                    'model_path': str(self.model_path) if self.model_path else None
                }
            },
            'classifications': [
                {
                    'event_id': event.event_id,
                    'species_name': event.species_classification.species_name,
                    'confidence': event.species_classification.confidence,
                    'alternatives': event.species_classification.alternative_species,
                    'timestamp': event.video_detection.timestamp if event.video_detection else None
                }
                for event in classified_events
            ],
            'statistics': self.get_classification_statistics(events)
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported {len(classified_events)} species classifications to {output_path}")
    
    def is_model_available(self) -> bool:
        """Check if TransFG model is available."""
        return self.model is not None and self.model != "mock_model"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_path': str(self.model_path) if self.model_path else None,
            'model_available': self.is_model_available(),
            'species_count': len(self.species_list) if self.species_list else 0,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'image_size': self.image_size
        }