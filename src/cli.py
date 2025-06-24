"""
Command-line interface for the bird monitoring pipeline.
Provides easy access to pipeline functionality from the terminal.
"""

import typer
import json
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint

from .core.pipeline import BirdMonitoringPipeline
from .models.data_models import ProcessingConfig
from .utils.config import get_settings, load_config_from_file
from .utils.logging import setup_all_loggers

# Create CLI app
app = typer.Typer(
    name="bird-monitor",
    help="Bird Monitoring Pipeline - Audio-Visual Bird Detection System",
    add_completion=False
)

# Rich console for pretty output
console = Console()


@app.command()
def process(
    video_path: Path = typer.Argument(..., help="Path to video file to process"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
    audio_threshold: Optional[float] = typer.Option(None, "--audio-threshold", help="Audio detection threshold"),
    video_threshold: Optional[float] = typer.Option(None, "--video-threshold", help="Video detection threshold"),
    species_threshold: Optional[float] = typer.Option(None, "--species-threshold", help="Species classification threshold"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output except errors")
):
    """Process a single video file through the bird detection pipeline."""
    
    # Set up logging
    log_level = "DEBUG" if verbose else "ERROR" if quiet else "INFO"
    setup_all_loggers(log_level=log_level)
    
    # Load configuration
    if config_file:
        settings = load_config_from_file(config_file)
    else:
        settings = get_settings()
    
    # Validate video file
    if not video_path.exists():
        rprint(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(1)
    
    # Set up output directory
    if output_dir is None:
        from datetime import datetime
        output_dir = settings.data_dir / "output" / f"{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create processing config
    config = settings.create_processing_config(str(video_path), str(output_dir))
    
    # Override thresholds if provided
    if audio_threshold is not None:
        config.audio_confidence_threshold = audio_threshold
    if video_threshold is not None:
        config.video_confidence_threshold = video_threshold
    if species_threshold is not None:
        config.species_confidence_threshold = species_threshold
    
    try:
        # Display processing info
        if not quiet:
            console.print(f"\n[bold blue]Processing Video:[/bold blue] {video_path}")
            console.print(f"[bold blue]Output Directory:[/bold blue] {output_dir}")
            console.print(f"[bold blue]Configuration:[/bold blue]")
            
            config_table = Table(show_header=False)
            config_table.add_column("Setting", style="cyan")
            config_table.add_column("Value", style="white")
            
            config_table.add_row("Audio Threshold", f"{config.audio_confidence_threshold:.2f}")
            config_table.add_row("Video Threshold", f"{config.video_confidence_threshold:.2f}")
            config_table.add_row("Species Threshold", f"{config.species_confidence_threshold:.2f}")
            config_table.add_row("Correlation Window", f"{config.temporal_correlation_window:.1f}s")
            
            console.print(config_table)
            console.print()
        
        # Initialize and run pipeline
        pipeline = BirdMonitoringPipeline(config)
        
        with Progress(console=console, disable=quiet) as progress:
            task = progress.add_task("Processing video...", total=100)
            
            # Process video
            result = pipeline.process_video(video_path, output_dir, config)
            progress.update(task, completed=100)
        
        # Display results
        if not quiet:
            _display_results(result)
        
        console.print(f"\n[green]✓ Processing completed successfully![/green]")
        console.print(f"[green]Results saved to: {output_dir}[/green]")
        
    except Exception as e:
        console.print(f"\n[red]✗ Processing failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory containing video files"),
    output_dir: Path = typer.Argument(..., help="Base output directory"),
    pattern: str = typer.Option("*.mp4", "--pattern", "-p", help="File pattern to match"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error", help="Continue if one video fails"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output except errors")
):
    """Process multiple video files in batch."""
    
    # Set up logging
    log_level = "DEBUG" if verbose else "ERROR" if quiet else "INFO"
    setup_all_loggers(log_level=log_level)
    
    # Load configuration
    if config_file:
        settings = load_config_from_file(config_file)
    else:
        settings = get_settings()
    
    # Find video files
    if not input_dir.exists():
        rprint(f"[red]Error: Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)
    
    video_files = list(input_dir.glob(pattern))
    if not video_files:
        rprint(f"[red]Error: No video files found matching pattern '{pattern}' in {input_dir}[/red]")
        raise typer.Exit(1)
    
    if not quiet:
        console.print(f"\n[bold blue]Batch Processing:[/bold blue]")
        console.print(f"Found {len(video_files)} video files")
        console.print(f"Output directory: {output_dir}")
        console.print()
    
    try:
        # Initialize pipeline
        pipeline = BirdMonitoringPipeline()
        
        # Process videos
        results = pipeline.process_batch(video_files, output_dir, continue_on_error)
        
        # Display batch results
        if not quiet:
            _display_batch_results(results, video_files)
        
        successful_count = sum(1 for r in results if r is not None)
        console.print(f"\n[green]✓ Batch processing completed![/green]")
        console.print(f"[green]Successfully processed {successful_count}/{len(video_files)} videos[/green]")
        
    except Exception as e:
        console.print(f"\n[red]✗ Batch processing failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def status():
    """Check the status of pipeline components and models."""
    
    console.print("\n[bold blue]Bird Monitoring Pipeline Status[/bold blue]\n")
    
    try:
        # Initialize pipeline to check component status
        pipeline = BirdMonitoringPipeline()
        status = pipeline.get_component_status()
        
        # Create status table
        status_table = Table()
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="white")
        status_table.add_column("Details", style="dim")
        
        # Media processor
        media_status = "✓ Available" if status['media_processor']['available'] else "✗ Not Available"
        media_details = f"Formats: {', '.join(status['media_processor']['supported_formats'][:3])}..."
        status_table.add_row("Media Processor", media_status, media_details)
        
        # Audio detector
        audio_info = status['audio_detector']
        audio_status = "✓ Model Loaded" if audio_info.get('model_available') else "⚠ Mock Model"
        audio_details = f"Threshold: {audio_info.get('confidence_threshold', 'N/A')}"
        status_table.add_row("Audio Detector", audio_status, audio_details)
        
        # Video detector
        video_info = status['video_detector']
        video_status = "✓ Model Loaded" if video_info.get('model_available') else "⚠ Mock Model"
        video_details = f"Device: {video_info.get('device', 'N/A')}"
        status_table.add_row("Video Detector", video_status, video_details)
        
        # Species classifier
        species_info = status['species_classifier']
        species_status = "✓ Model Loaded" if species_info.get('model_available') else "⚠ Mock Model"
        species_details = f"Species: {species_info.get('species_count', 'N/A')}"
        status_table.add_row("Species Classifier", species_status, species_details)
        
        # Event integrator
        integrator_status = "✓ Available" if status['event_integrator']['available'] else "✗ Not Available"
        integrator_details = f"Window: {status['event_integrator'].get('correlation_window', 'N/A')}s"
        status_table.add_row("Event Integrator", integrator_status, integrator_details)
        
        console.print(status_table)
        
        # Display warnings
        warnings = []
        if not audio_info.get('model_available'):
            warnings.append("BirdNET model not found - using mock audio detection")
        if not video_info.get('model_available'):
            warnings.append("YOLO model not found - using mock video detection")
        if not species_info.get('model_available'):
            warnings.append("TransFG model not found - using mock species classification")
        
        if warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  • {warning}")
        
    except Exception as e:
        console.print(f"[red]Error checking status: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def config(
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output configuration file"),
    show: bool = typer.Option(False, "--show", help="Show current configuration")
):
    """Manage pipeline configuration."""
    
    if show:
        # Display current configuration
        settings = get_settings()
        
        console.print("\n[bold blue]Current Configuration:[/bold blue]\n")
        
        config_data = settings.dict()
        for section, values in {
            "Paths": {k: v for k, v in config_data.items() if 'dir' in k or 'path' in k},
            "Audio Settings": {k: v for k, v in config_data.items() if 'audio' in k},
            "Video Settings": {k: v for k, v in config_data.items() if 'video' in k},
            "Thresholds": {k: v for k, v in config_data.items() if 'threshold' in k or 'confidence' in k},
            "Web Settings": {k: v for k, v in config_data.items() if 'web' in k}
        }.items():
            if values:
                console.print(f"[bold cyan]{section}:[/bold cyan]")
                for key, value in values.items():
                    console.print(f"  {key}: {value}")
                console.print()
    
    if output_file:
        # Export configuration
        from .utils.config import config_manager
        config_manager.save_config(output_file)
        console.print(f"[green]Configuration saved to: {output_file}[/green]")


def _display_results(result) -> None:
    """Display processing results in a formatted table."""
    
    # Summary table
    summary_table = Table(title="Detection Results Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", style="white")
    
    summary_table.add_row("Total Events", str(result.total_events))
    summary_table.add_row("Audio Only", str(result.audio_only_events))
    summary_table.add_row("Video Only", str(result.video_only_events))
    summary_table.add_row("Audio + Video", str(result.audio_video_events))
    summary_table.add_row("Species Identified", str(result.species_identified))
    summary_table.add_row("Unique Species", str(len(result.unique_species)))
    
    console.print(summary_table)
    
    # Species table
    if result.unique_species:
        console.print("\n[bold cyan]Species Found:[/bold cyan]")
        for species in result.unique_species[:10]:  # Show top 10
            console.print(f"  • {species}")
        
        if len(result.unique_species) > 10:
            console.print(f"  ... and {len(result.unique_species) - 10} more")


def _display_batch_results(results: List, video_files: List[Path]) -> None:
    """Display batch processing results."""
    
    # Results table
    results_table = Table(title="Batch Processing Results")
    results_table.add_column("Video File", style="cyan")
    results_table.add_column("Status", style="white")
    results_table.add_column("Events", style="white")
    results_table.add_column("Species", style="white")
    
    for video_file, result in zip(video_files, results):
        if result is not None:
            status = "[green]✓ Success[/green]"
            events = str(result.total_events)
            species = str(len(result.unique_species))
        else:
            status = "[red]✗ Failed[/red]"
            events = "-"
            species = "-"
        
        results_table.add_row(video_file.name, status, events, species)
    
    console.print(results_table)


if __name__ == "__main__":
    app()