"""
Configuration management for the bird monitoring pipeline.
Handles loading configuration from files, environment variables, and CLI arguments.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Union
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

from ..models.data_models import ProcessingConfig


class PipelineSettings(BaseSettings):
    """Main configuration settings for the bird monitoring pipeline."""
    
    # Environment and paths
    environment: str = Field("development", description="Environment (development, production)")
    project_root: Path = Field(default_factory=lambda: Path.cwd(), description="Project root directory")
    models_dir: Path = Field(Path("models"), description="Directory containing ML models")
    data_dir: Path = Field(Path("data"), description="Data directory")
    logs_dir: Path = Field(Path("logs"), description="Logs directory")
    
    # Model paths and configurations
    birdnet_model_path: Optional[Path] = Field(None, description="Path to BirdNET model")
    yolo_model_path: Optional[Path] = Field(None, description="Path to YOLO model")
    transfg_model_path: Optional[Path] = Field(None, description="Path to TransFG model")
    
    # Processing settings (defaults from ProcessingConfig)
    audio_segment_length: float = Field(3.0, gt=0, description="Audio segment length in seconds")
    audio_overlap: float = Field(0.5, ge=0, lt=1, description="Audio segment overlap ratio")
    audio_sample_rate: int = Field(22050, gt=0, description="Audio sample rate")
    
    video_frame_skip: int = Field(1, ge=1, description="Process every N frames")
    video_resize_height: Optional[int] = Field(None, gt=0, description="Resize video height")
    
    # Detection thresholds
    audio_confidence_threshold: float = Field(0.3, ge=0, le=1, description="Audio detection threshold")
    video_confidence_threshold: float = Field(0.5, ge=0, le=1, description="Video detection threshold")
    species_confidence_threshold: float = Field(0.6, ge=0, le=1, description="Species classification threshold")
    
    # Temporal correlation
    temporal_correlation_window: float = Field(2.0, gt=0, description="Time window for correlating events")
    
    # Confidence scoring
    audio_only_base_confidence: float = Field(0.3, ge=0, le=1, description="Base confidence for audio-only events")
    video_only_base_confidence: float = Field(0.4, ge=0, le=1, description="Base confidence for video-only events")
    audio_video_base_confidence: float = Field(0.7, ge=0, le=1, description="Base confidence for combined events")
    
    # Web application settings
    web_host: str = Field("localhost", description="Web server host")
    web_port: int = Field(8000, ge=1, le=65535, description="Web server port")
    web_reload: bool = Field(True, description="Enable auto-reload in development")
    
    # Logging configuration
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    
    # Performance settings
    max_workers: int = Field(4, ge=1, description="Maximum number of worker processes")
    batch_size: int = Field(32, ge=1, description="Batch size for model inference")
    gpu_enabled: bool = Field(True, description="Enable GPU acceleration if available")
    
    class Config:
        env_file = ".env"
        env_prefix = "BIRD_MONITOR_"
        case_sensitive = False
    
    @validator('models_dir', 'data_dir', 'logs_dir', pre=True)
    def resolve_paths(cls, v, values):
        """Resolve paths relative to project root."""
        if isinstance(v, str):
            v = Path(v)
        
        if not v.is_absolute() and 'project_root' in values:
            return values['project_root'] / v
        return v
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment value."""
        valid_envs = {'development', 'production', 'testing'}
        if v.lower() not in valid_envs:
            raise ValueError(f'Environment must be one of {valid_envs}')
        return v.lower()
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()
    
    def create_processing_config(self, input_video_path: str, output_dir: str) -> ProcessingConfig:
        """Create a ProcessingConfig from the current settings."""
        return ProcessingConfig(
            input_video_path=input_video_path,
            output_dir=output_dir,
            audio_segment_length=self.audio_segment_length,
            audio_overlap=self.audio_overlap,
            audio_sample_rate=self.audio_sample_rate,
            video_frame_skip=self.video_frame_skip,
            video_resize_height=self.video_resize_height,
            audio_confidence_threshold=self.audio_confidence_threshold,
            video_confidence_threshold=self.video_confidence_threshold,
            species_confidence_threshold=self.species_confidence_threshold,
            temporal_correlation_window=self.temporal_correlation_window,
            audio_only_base_confidence=self.audio_only_base_confidence,
            video_only_base_confidence=self.video_only_base_confidence,
            audio_video_base_confidence=self.audio_video_base_confidence
        )
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.models_dir,
            self.data_dir,
            self.data_dir / "input",
            self.data_dir / "output", 
            self.data_dir / "temp",
            self.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """Manages configuration loading from multiple sources."""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to YAML configuration file
        """
        self.config_file = Path(config_file) if config_file else None
        self._settings: Optional[PipelineSettings] = None
    
    def load_config(
        self, 
        config_file: Optional[Union[str, Path]] = None,
        env_file: Optional[Union[str, Path]] = None,
        **overrides
    ) -> PipelineSettings:
        """
        Load configuration from multiple sources.
        
        Priority order (highest to lowest):
        1. Function overrides
        2. Environment variables
        3. Configuration file
        4. Default values
        
        Args:
            config_file: Path to YAML configuration file
            env_file: Path to .env file
            **overrides: Direct configuration overrides
            
        Returns:
            Loaded configuration settings
        """
        # Load environment file if specified
        if env_file:
            load_dotenv(env_file)
        elif Path(".env").exists():
            load_dotenv(".env")
        
        # Load configuration from YAML file
        config_data = {}
        if config_file:
            config_data = self._load_yaml_config(config_file)
        elif self.config_file and self.config_file.exists():
            config_data = self._load_yaml_config(self.config_file)
        elif Path("config.yaml").exists():
            config_data = self._load_yaml_config("config.yaml")
        
        # Merge with overrides
        config_data.update(overrides)
        
        # Create settings instance
        self._settings = PipelineSettings(**config_data)
        
        # Ensure required directories exist
        self._settings.ensure_directories()
        
        return self._settings
    
    def _load_yaml_config(self, config_file: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file {config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error reading configuration file {config_path}: {e}")
    
    def save_config(self, output_file: Union[str, Path], settings: Optional[PipelineSettings] = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_file: Path to output YAML file
            settings: Settings to save (uses loaded settings if None)
        """
        if settings is None:
            if self._settings is None:
                raise ValueError("No configuration loaded to save")
            settings = self._settings
        
        config_dict = settings.dict()
        
        # Convert Path objects to strings for YAML serialization
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @property
    def settings(self) -> Optional[PipelineSettings]:
        """Get current settings."""
        return self._settings
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get path to a specific model."""
        if not self._settings:
            return None
        
        model_paths = {
            'birdnet': self._settings.birdnet_model_path,
            'yolo': self._settings.yolo_model_path,
            'transfg': self._settings.transfg_model_path
        }
        
        return model_paths.get(model_name.lower())
    
    def validate_model_paths(self) -> Dict[str, bool]:
        """Validate that required model files exist."""
        if not self._settings:
            return {}
        
        validation_results = {}
        model_configs = {
            'birdnet': self._settings.birdnet_model_path,
            'yolo': self._settings.yolo_model_path,
            'transfg': self._settings.transfg_model_path
        }
        
        for model_name, model_path in model_configs.items():
            if model_path is None:
                validation_results[model_name] = False
            else:
                validation_results[model_name] = model_path.exists()
        
        return validation_results


# Global configuration manager instance
config_manager = ConfigManager()


def get_settings() -> PipelineSettings:
    """Get current pipeline settings."""
    if config_manager.settings is None:
        return config_manager.load_config()
    return config_manager.settings


def load_config_from_file(config_file: Union[str, Path]) -> PipelineSettings:
    """Load configuration from a specific file."""
    return config_manager.load_config(config_file=config_file)