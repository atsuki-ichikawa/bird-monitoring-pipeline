"""
Media processing module for video/audio separation and preprocessing.
Handles video file input and extracts audio and video streams for parallel processing.
"""

import os
import cv2
import ffmpeg
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Generator, Dict, Any
import tempfile
import shutil
from datetime import timedelta

from ..utils.logging import get_logger, LoggingContext
from ..utils.config import get_settings


class MediaProcessor:
    """Handles video/audio separation and preprocessing for the bird monitoring pipeline."""
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize media processor.
        
        Args:
            temp_dir: Directory for temporary files (auto-created if None)
        """
        self.logger = get_logger("pipeline")
        self.settings = get_settings()
        
        if temp_dir is None:
            self.temp_dir = self.settings.data_dir / "temp"
        else:
            self.temp_dir = Path(temp_dir)
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Video processing state
        self._video_info: Optional[Dict[str, Any]] = None
        self._current_video_path: Optional[Path] = None
    
    def extract_audio_video(
        self, 
        video_path: Path,
        output_audio_path: Optional[Path] = None,
        output_video_dir: Optional[Path] = None
    ) -> Tuple[Path, Path]:
        """
        Extract audio and video streams from input video file.
        
        Args:
            video_path: Path to input video file
            output_audio_path: Optional output path for audio file
            output_video_dir: Optional directory for video frames
            
        Returns:
            Tuple of (audio_file_path, video_frames_dir)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        with LoggingContext(f"Extracting media from {video_path.name}") as ctx:
            # Get video information
            self._video_info = self._get_video_info(video_path)
            self._current_video_path = video_path
            
            ctx.log_metric("duration", self._video_info['duration'], "seconds")
            ctx.log_metric("fps", self._video_info['fps'], "fps")
            ctx.log_metric("resolution", f"{self._video_info['width']}x{self._video_info['height']}")
            
            # Set up output paths
            if output_audio_path is None:
                output_audio_path = self.temp_dir / f"{video_path.stem}_audio.wav"
            
            if output_video_dir is None:
                output_video_dir = self.temp_dir / f"{video_path.stem}_frames"
            
            output_video_dir = Path(output_video_dir)
            output_video_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract audio
            ctx.log_progress("Extracting audio stream")
            self._extract_audio(video_path, output_audio_path)
            
            # Extract video frames  
            ctx.log_progress("Extracting video frames")
            self._extract_video_frames(video_path, output_video_dir)
            
            return output_audio_path, output_video_dir
    
    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video file information using ffprobe."""
        try:
            probe = ffmpeg.probe(str(video_path))
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), 
                None
            )
            audio_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), 
                None
            )
            
            if video_stream is None:
                raise ValueError("No video stream found in file")
            
            # Extract key information
            info = {
                'duration': float(video_stream.get('duration', 0)),
                'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'codec': video_stream.get('codec_name', 'unknown'),
                'has_audio': audio_stream is not None
            }
            
            if audio_stream:
                info.update({
                    'audio_codec': audio_stream.get('codec_name', 'unknown'),
                    'audio_sample_rate': int(audio_stream.get('sample_rate', 0)),
                    'audio_channels': int(audio_stream.get('channels', 0))
                })
            
            return info
            
        except Exception as e:
            raise RuntimeError(f"Failed to get video information: {e}")
    
    def _extract_audio(self, video_path: Path, output_path: Path) -> None:
        """Extract audio stream to WAV file."""
        try:
            # Configure audio extraction
            stream = ffmpeg.input(str(video_path))
            audio = stream.audio
            
            # Apply audio settings
            sample_rate = self.settings.audio_sample_rate
            
            out = ffmpeg.output(
                audio,
                str(output_path),
                acodec='pcm_s16le',  # 16-bit PCM
                ac=1,  # Mono
                ar=sample_rate,  # Sample rate
                y=None  # Overwrite output file
            )
            
            ffmpeg.run(out, overwrite_output=True, quiet=True)
            
            if not output_path.exists():
                raise RuntimeError("Audio extraction failed - output file not created")
                
        except Exception as e:
            raise RuntimeError(f"Audio extraction failed: {e}")
    
    def _extract_video_frames(self, video_path: Path, output_dir: Path) -> None:
        """Extract video frames using OpenCV for better control."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError("Failed to open video file")
            
            frame_count = 0
            saved_count = 0
            frame_skip = self.settings.video_frame_skip
            resize_height = self.settings.video_resize_height
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on configuration
                if frame_count % frame_skip == 0:
                    # Resize frame if configured
                    if resize_height is not None:
                        height, width = frame.shape[:2]
                        new_width = int(width * (resize_height / height))
                        frame = cv2.resize(frame, (new_width, resize_height))
                    
                    # Calculate timestamp
                    timestamp = frame_count / self._video_info['fps']
                    
                    # Save frame
                    frame_filename = f"frame_{saved_count:06d}_{timestamp:.3f}s.jpg"
                    frame_path = output_dir / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame)
                    saved_count += 1
                
                frame_count += 1
            
            cap.release()
            
            if saved_count == 0:
                raise RuntimeError("No frames extracted from video")
            
            self.logger.info(f"Extracted {saved_count} frames from {frame_count} total frames")
            
        except Exception as e:
            raise RuntimeError(f"Video frame extraction failed: {e}")
    
    def get_frame_timestamps(self, frames_dir: Path) -> Dict[str, float]:
        """
        Get timestamps for extracted frames.
        
        Args:
            frames_dir: Directory containing extracted frames
            
        Returns:
            Dictionary mapping frame filenames to timestamps
        """
        timestamps = {}
        
        for frame_file in frames_dir.glob("frame_*.jpg"):
            # Extract timestamp from filename
            try:
                # Format: frame_000001_12.345s.jpg
                parts = frame_file.stem.split('_')
                timestamp_str = parts[-1]  # "12.345s"
                timestamp = float(timestamp_str.rstrip('s'))
                timestamps[frame_file.name] = timestamp
            except (IndexError, ValueError) as e:
                self.logger.warning(f"Could not parse timestamp from {frame_file.name}: {e}")
                continue
        
        return timestamps
    
    def create_frame_iterator(
        self, 
        frames_dir: Path,
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> Generator[Tuple[float, np.ndarray], None, None]:
        """
        Create an iterator over video frames with timestamps.
        
        Args:
            frames_dir: Directory containing extracted frames
            start_time: Start time in seconds
            end_time: End time in seconds (None for full video)
            
        Yields:
            Tuples of (timestamp, frame_array)
        """
        timestamps = self.get_frame_timestamps(frames_dir)
        
        # Sort frames by timestamp
        sorted_frames = sorted(timestamps.items(), key=lambda x: x[1])
        
        for frame_name, timestamp in sorted_frames:
            # Skip frames outside time range
            if timestamp < start_time:
                continue
            if end_time is not None and timestamp > end_time:
                break
            
            # Load frame
            frame_path = frames_dir / frame_name
            frame = cv2.imread(str(frame_path))
            
            if frame is not None:
                yield timestamp, frame
    
    def get_video_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently processed video."""
        return self._video_info
    
    def get_video_duration(self) -> float:
        """Get duration of the currently processed video in seconds."""
        if self._video_info:
            return self._video_info['duration']
        return 0.0
    
    def cleanup_temp_files(self, keep_results: bool = False) -> None:
        """
        Clean up temporary files.
        
        Args:
            keep_results: If True, keep extracted audio/video files
        """
        if not keep_results and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info("Cleaned up temporary files")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temp files: {e}")
    
    def validate_video_file(self, video_path: Path) -> bool:
        """
        Validate that a video file can be processed.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if file is valid and processable
        """
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                return False
            
            # Try to get basic video info
            info = self._get_video_info(video_path)
            
            # Check minimum requirements
            if info['duration'] <= 0:
                self.logger.error(f"Video has invalid duration: {info['duration']}")
                return False
            
            if info['fps'] <= 0:
                self.logger.error(f"Video has invalid frame rate: {info['fps']}")
                return False
            
            if info['width'] <= 0 or info['height'] <= 0:
                self.logger.error(f"Video has invalid resolution: {info['width']}x{info['height']}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Video validation failed: {e}")
            return False
    
    def get_supported_formats(self) -> list[str]:
        """Get list of supported video formats."""
        return ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    def estimate_processing_time(self, video_path: Path) -> dict:
        """
        Estimate processing time and resource requirements.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with time and resource estimates
        """
        try:
            info = self._get_video_info(video_path)
            duration = info['duration']
            total_frames = int(duration * info['fps'])
            processed_frames = total_frames // self.settings.video_frame_skip
            
            # Rough estimates based on typical processing speeds
            estimates = {
                'video_duration': duration,
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'estimated_extraction_time': duration * 0.1,  # ~10% of video duration
                'estimated_disk_space_mb': processed_frames * 0.5,  # ~0.5MB per frame
                'audio_file_size_mb': duration * 0.17  # ~10MB per minute for mono 22kHz
            }
            
            return estimates
            
        except Exception as e:
            self.logger.error(f"Failed to estimate processing time: {e}")
            return {}