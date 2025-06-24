"""
Logging configuration for the bird monitoring pipeline.
Provides centralized logging setup with file and console handlers.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

from .config import get_settings


class PipelineLogger:
    """Centralized logger for the bird monitoring pipeline."""
    
    def __init__(self, name: str = "bird_monitor"):
        """
        Initialize pipeline logger.
        
        Args:
            name: Logger name
        """
        self.name = name
        self._logger: Optional[logging.Logger] = None
        self._handlers_configured = False
    
    def setup_logging(
        self,
        log_level: Optional[str] = None,
        log_file: Optional[Union[str, Path]] = None,
        console_output: bool = True,
        file_output: bool = True
    ) -> logging.Logger:
        """
        Set up logging configuration.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional specific log file path
            console_output: Enable console logging
            file_output: Enable file logging
            
        Returns:
            Configured logger instance
        """
        if self._logger is not None and self._handlers_configured:
            return self._logger
        
        # Get settings
        try:
            settings = get_settings()
            if log_level is None:
                log_level = settings.log_level
            log_format = settings.log_format
            logs_dir = settings.logs_dir
        except Exception:
            # Fallback if settings not available
            log_level = log_level or "INFO"
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            logs_dir = Path("logs")
        
        # Create logger
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(getattr(logging, log_level.upper()))
        
        # Avoid duplicate handlers
        if self._handlers_configured:
            return self._logger
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)
        
        # File handler
        if file_output:
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            if log_file is None:
                # Generate timestamped log file name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = logs_dir / f"bird_monitor_{timestamp}.log"
            else:
                log_file = Path(log_file)
                if not log_file.is_absolute():
                    log_file = logs_dir / log_file
            
            # Rotating file handler to prevent huge log files
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
        
        self._handlers_configured = True
        return self._logger
    
    def get_logger(self) -> logging.Logger:
        """Get the logger instance, setting it up if needed."""
        if self._logger is None:
            return self.setup_logging()
        return self._logger


# Global logger instances
_main_logger = PipelineLogger("bird_monitor")
_pipeline_logger = PipelineLogger("bird_monitor.pipeline")
_audio_logger = PipelineLogger("bird_monitor.audio")
_video_logger = PipelineLogger("bird_monitor.video")
_web_logger = PipelineLogger("bird_monitor.web")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for the specified component.
    
    Args:
        name: Component name (pipeline, audio, video, web) or None for main logger
        
    Returns:
        Logger instance
    """
    if name is None:
        return _main_logger.get_logger()
    
    logger_map = {
        "pipeline": _pipeline_logger,
        "audio": _audio_logger,
        "video": _video_logger,
        "web": _web_logger
    }
    
    if name in logger_map:
        return logger_map[name].get_logger()
    else:
        # Create custom logger for unknown names
        custom_logger = PipelineLogger(f"bird_monitor.{name}")
        return custom_logger.get_logger()


def setup_all_loggers(
    log_level: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True
) -> None:
    """
    Set up all pipeline loggers with consistent configuration.
    
    Args:
        log_level: Logging level for all loggers
        console_output: Enable console output
        file_output: Enable file output
    """
    loggers = [_main_logger, _pipeline_logger, _audio_logger, _video_logger, _web_logger]
    
    for logger in loggers:
        logger.setup_logging(
            log_level=log_level,
            console_output=console_output,
            file_output=file_output
        )


def log_processing_start(video_path: str, config_summary: dict) -> None:
    """Log the start of video processing."""
    logger = get_logger("pipeline")
    logger.info(f"Starting video processing: {video_path}")
    logger.info(f"Configuration summary: {config_summary}")


def log_processing_complete(video_path: str, results_summary: dict) -> None:
    """Log the completion of video processing."""
    logger = get_logger("pipeline")
    logger.info(f"Completed video processing: {video_path}")
    logger.info(f"Results summary: {results_summary}")


def log_detection_event(event_type: str, timestamp: float, confidence: float) -> None:
    """Log a detection event."""
    logger = get_logger("pipeline")
    logger.debug(f"Detection event - Type: {event_type}, Time: {timestamp:.2f}s, Confidence: {confidence:.3f}")


def log_model_loading(model_name: str, model_path: str) -> None:
    """Log model loading."""
    logger = get_logger("pipeline")
    logger.info(f"Loading {model_name} model from: {model_path}")


def log_model_loaded(model_name: str, load_time: float) -> None:
    """Log successful model loading."""
    logger = get_logger("pipeline")
    logger.info(f"Successfully loaded {model_name} model in {load_time:.2f} seconds")


def log_error(component: str, error: Exception, context: Optional[str] = None) -> None:
    """Log an error with context."""
    logger = get_logger(component)
    if context:
        logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    else:
        logger.error(f"Error: {str(error)}", exc_info=True)


def log_warning(component: str, message: str) -> None:
    """Log a warning message."""
    logger = get_logger(component)
    logger.warning(message)


def log_performance_metric(operation: str, duration: float, details: Optional[dict] = None) -> None:
    """Log performance metrics."""
    logger = get_logger("pipeline")
    message = f"Performance - {operation}: {duration:.3f}s"
    if details:
        message += f" | Details: {details}"
    logger.info(message)


class LoggingContext:
    """Context manager for structured logging within operations."""
    
    def __init__(self, operation: str, component: str = "pipeline", **kwargs):
        """
        Initialize logging context.
        
        Args:
            operation: Name of the operation
            component: Component name for logger
            **kwargs: Additional context information
        """
        self.operation = operation
        self.logger = get_logger(component)
        self.context = kwargs
        self.start_time = None
    
    def __enter__(self):
        """Enter the logging context."""
        self.start_time = datetime.now()
        context_str = f" with {self.context}" if self.context else ""
        self.logger.info(f"Starting {self.operation}{context_str}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the logging context."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed {self.operation} in {duration:.3f}s")
        else:
            self.logger.error(f"Failed {self.operation} after {duration:.3f}s: {exc_val}")
        
        return False  # Don't suppress exceptions
    
    def log_progress(self, message: str, **kwargs) -> None:
        """Log progress within the operation."""
        context_str = f" | {kwargs}" if kwargs else ""
        self.logger.info(f"{self.operation} - {message}{context_str}")
    
    def log_metric(self, metric_name: str, value: Union[int, float], unit: str = "") -> None:
        """Log a metric within the operation."""
        unit_str = f" {unit}" if unit else ""
        self.logger.info(f"{self.operation} - {metric_name}: {value}{unit_str}")


# Example usage functions
def create_processing_context(video_path: str, **kwargs) -> LoggingContext:
    """Create a logging context for video processing."""
    return LoggingContext(
        f"Processing {Path(video_path).name}",
        component="pipeline",
        video_path=video_path,
        **kwargs
    )


def create_model_context(model_name: str, **kwargs) -> LoggingContext:
    """Create a logging context for model operations."""
    return LoggingContext(
        f"Model operation: {model_name}",
        component="pipeline",
        model=model_name,
        **kwargs
    )