"""
Audio detection pipeline using BirdNET for bird call detection.
Processes audio segments and identifies bird vocalizations with confidence scores.
"""

import librosa
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import tempfile
import json
from datetime import datetime

from ..models.data_models import AudioDetection, RawScores
from ..utils.logging import get_logger, LoggingContext
from ..utils.config import get_settings


class AudioDetector:
    """Audio detection pipeline using BirdNET for bird call detection."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize audio detector.
        
        Args:
            model_path: Path to BirdNET model file
        """
        self.logger = get_logger("audio")
        self.settings = get_settings()
        
        # Model configuration
        self.model_path = model_path or self.settings.birdnet_model_path
        self.model = None
        self.species_list = None
        
        # Audio processing parameters
        self.sample_rate = self.settings.audio_sample_rate
        self.segment_length = self.settings.audio_segment_length
        self.overlap = self.settings.audio_overlap
        self.confidence_threshold = self.settings.audio_confidence_threshold
        
        # Mel-spectrogram parameters (BirdNET standard)
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        self.fmin = 0
        self.fmax = 12000
        
        # Initialize model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load BirdNET model and species list."""
        try:
            with LoggingContext("Loading BirdNET model", "audio") as ctx:
                if self.model_path is None:
                    self.logger.warning("No BirdNET model path specified, using mock detector")
                    self._create_mock_model()
                    return
                
                if not Path(self.model_path).exists():
                    self.logger.warning(f"BirdNET model not found at {self.model_path}, using mock detector")
                    self._create_mock_model()
                    return
                
                # Load TensorFlow model
                self.model = tf.keras.models.load_model(str(self.model_path))
                
                # Load species list (should be in same directory as model)
                species_file = Path(self.model_path).parent / "species_list.json"
                if species_file.exists():
                    with open(species_file, 'r') as f:
                        self.species_list = json.load(f)
                else:
                    self.logger.warning("Species list not found, using default species")
                    self.species_list = self._get_default_species_list()
                
                ctx.log_metric("model_loaded", True)
                ctx.log_metric("species_count", len(self.species_list))
                
        except Exception as e:
            self.logger.error(f"Failed to load BirdNET model: {e}")
            self.logger.info("Falling back to mock detector")
            self._create_mock_model()
    
    def _create_mock_model(self) -> None:
        """Create a mock model for testing when BirdNET is not available."""
        self.logger.info("Creating mock audio detection model")
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
            "Wren",
            "Starling",
            "Crow",
            "Magpie"
        ]
    
    def process_audio_file(self, audio_path: Path) -> List[AudioDetection]:
        """
        Process an audio file and detect bird calls.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of audio detection results
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        with LoggingContext(f"Processing audio file {audio_path.name}", "audio") as ctx:
            try:
                # Load audio
                audio_data, sr = librosa.load(str(audio_path), sr=self.sample_rate)
                ctx.log_metric("audio_duration", len(audio_data) / sr, "seconds")
                ctx.log_metric("sample_rate", sr, "Hz")
                
                # Create segments
                segments = self._create_audio_segments(audio_data, sr)
                ctx.log_metric("segments_created", len(segments))
                
                # Process each segment
                detections = []
                for i, (start_time, end_time, segment_data) in enumerate(segments):
                    ctx.log_progress(f"Processing segment {i+1}/{len(segments)}")
                    
                    # Create mel-spectrogram
                    mel_spec = self._create_mel_spectrogram(segment_data)
                    
                    # Run detection
                    confidence = self._detect_bird_in_segment(mel_spec)
                    
                    # Create detection if confidence exceeds threshold
                    if confidence >= self.confidence_threshold:
                        detection = AudioDetection(
                            start_time=start_time,
                            end_time=end_time,
                            confidence=confidence,
                            frequency_range=self._estimate_frequency_range(mel_spec)
                        )
                        detections.append(detection)
                
                ctx.log_metric("detections_found", len(detections))
                return detections
                
            except Exception as e:
                ctx.log_progress(f"Error processing audio: {e}")
                raise RuntimeError(f"Audio processing failed: {e}")
    
    def _create_audio_segments(self, audio_data: np.ndarray, sample_rate: int) -> List[Tuple[float, float, np.ndarray]]:
        """
        Create overlapping audio segments for processing.
        
        Args:
            audio_data: Audio signal array
            sample_rate: Audio sample rate
            
        Returns:
            List of (start_time, end_time, segment_data) tuples
        """
        segment_samples = int(self.segment_length * sample_rate)
        hop_samples = int(segment_samples * (1 - self.overlap))
        
        segments = []
        start_sample = 0
        
        while start_sample < len(audio_data):
            end_sample = min(start_sample + segment_samples, len(audio_data))
            
            # Skip segments that are too short
            if end_sample - start_sample < segment_samples * 0.8:
                break
            
            segment_data = audio_data[start_sample:end_sample]
            
            # Pad if necessary
            if len(segment_data) < segment_samples:
                segment_data = np.pad(segment_data, (0, segment_samples - len(segment_data)))
            
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            
            segments.append((start_time, end_time, segment_data))
            start_sample += hop_samples
        
        return segments
    
    def _create_mel_spectrogram(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Create mel-spectrogram from audio segment.
        
        Args:
            audio_segment: Audio segment array
            
        Returns:
            Mel-spectrogram array
        """
        # Create mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_segment,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        return mel_spec_norm
    
    def _detect_bird_in_segment(self, mel_spectrogram: np.ndarray) -> float:
        """
        Detect bird presence in a mel-spectrogram segment.
        
        Args:
            mel_spectrogram: Mel-spectrogram array
            
        Returns:
            Confidence score (0-1)
        """
        if self.model == "mock_model":
            return self._mock_detection(mel_spectrogram)
        
        try:
            # Prepare input for BirdNET model
            input_data = self._prepare_model_input(mel_spectrogram)
            
            # Run prediction
            predictions = self.model.predict(input_data, verbose=0)
            
            # Get maximum confidence across all species
            max_confidence = float(np.max(predictions))
            
            return max_confidence
            
        except Exception as e:
            self.logger.warning(f"Model prediction failed: {e}")
            return 0.0
    
    def _prepare_model_input(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """
        Prepare mel-spectrogram for model input.
        
        Args:
            mel_spectrogram: Mel-spectrogram array
            
        Returns:
            Model input array
        """
        # Reshape for model (assuming model expects specific input shape)
        # This would need to be adjusted based on actual BirdNET model architecture
        input_shape = (1, mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1)
        model_input = mel_spectrogram.reshape(input_shape)
        
        return model_input
    
    def _mock_detection(self, mel_spectrogram: np.ndarray) -> float:
        """
        Mock detection for testing purposes.
        
        Args:
            mel_spectrogram: Mel-spectrogram array
            
        Returns:
            Mock confidence score
        """
        # Simple mock detection based on spectrogram energy
        energy = np.mean(mel_spectrogram)
        
        # Add some randomness for realistic behavior
        np.random.seed(int(datetime.now().timestamp() * 1000) % 1000)
        noise = np.random.normal(0, 0.1)
        
        # Convert energy to confidence-like score
        confidence = np.clip(energy + noise, 0, 1)
        
        # Simulate bird detection with some probability
        if np.random.random() < 0.3:  # 30% chance of "bird detection"
            confidence = max(confidence, 0.5)
        
        return float(confidence)
    
    def _estimate_frequency_range(self, mel_spectrogram: np.ndarray) -> Tuple[float, float]:
        """
        Estimate the frequency range of detected bird call.
        
        Args:
            mel_spectrogram: Mel-spectrogram array
            
        Returns:
            Tuple of (min_freq, max_freq) in Hz
        """
        # Find frequency bins with highest energy
        freq_energy = np.mean(mel_spectrogram, axis=1)
        
        # Convert mel bins to frequencies
        mel_frequencies = librosa.mel_frequencies(n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
        
        # Find active frequency range (above threshold)
        threshold = np.mean(freq_energy) + np.std(freq_energy)
        active_bins = np.where(freq_energy > threshold)[0]
        
        if len(active_bins) > 0:
            min_freq = float(mel_frequencies[active_bins[0]])
            max_freq = float(mel_frequencies[active_bins[-1]])
        else:
            # Default bird call frequency range
            min_freq = 1000.0
            max_freq = 8000.0
        
        return min_freq, max_freq
    
    def get_detection_statistics(self, detections: List[AudioDetection]) -> Dict[str, Any]:
        """
        Get statistics about audio detections.
        
        Args:
            detections: List of audio detections
            
        Returns:
            Dictionary with detection statistics
        """
        if not detections:
            return {
                'total_detections': 0,
                'total_duration': 0.0,
                'average_confidence': 0.0,
                'confidence_distribution': {}
            }
        
        total_duration = sum(d.end_time - d.start_time for d in detections)
        confidences = [d.confidence for d in detections]
        
        # Confidence distribution
        conf_dist = {
            'high (>0.8)': sum(1 for c in confidences if c > 0.8),
            'medium (0.5-0.8)': sum(1 for c in confidences if 0.5 <= c <= 0.8),
            'low (<0.5)': sum(1 for c in confidences if c < 0.5)
        }
        
        return {
            'total_detections': len(detections),
            'total_duration': total_duration,
            'average_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'confidence_distribution': conf_dist,
            'average_detection_length': total_duration / len(detections)
        }
    
    def export_detections(self, detections: List[AudioDetection], output_path: Path) -> None:
        """
        Export detections to JSON file.
        
        Args:
            detections: List of audio detections
            output_path: Output file path
        """
        export_data = {
            'metadata': {
                'detector': 'BirdNET',
                'processed_at': datetime.now().isoformat(),
                'settings': {
                    'segment_length': self.segment_length,
                    'overlap': self.overlap,
                    'confidence_threshold': self.confidence_threshold,
                    'sample_rate': self.sample_rate
                }
            },
            'detections': [detection.dict() for detection in detections],
            'statistics': self.get_detection_statistics(detections)
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported {len(detections)} audio detections to {output_path}")
    
    def is_model_available(self) -> bool:
        """Check if BirdNET model is available."""
        return self.model is not None and self.model != "mock_model"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_path': str(self.model_path) if self.model_path else None,
            'model_available': self.is_model_available(),
            'species_count': len(self.species_list) if self.species_list else 0,
            'sample_rate': self.sample_rate,
            'segment_length': self.segment_length,
            'confidence_threshold': self.confidence_threshold
        }