"""
Minimal command-line interface for testing without heavy dependencies.
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Create CLI app
app = typer.Typer(
    name="bird-monitor-minimal",
    help="Bird Monitoring Pipeline - Minimal Test Version",
    add_completion=False
)

# Rich console for pretty output
console = Console()


@app.command()
def status():
    """Check the status of pipeline components (minimal version)."""
    
    console.print("\n[bold blue]Bird Monitoring Pipeline Status (Minimal)[/bold blue]\n")
    
    # Create status table
    status_table = Table()
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="white")
    status_table.add_column("Details", style="dim")
    
    # Mock component status
    status_table.add_row("Media Processor", "⚠ Mock", "FFmpeg not available")
    status_table.add_row("Audio Detector", "⚠ Mock", "BirdNET not installed")
    status_table.add_row("Video Detector", "⚠ Mock", "OpenCV not installed")
    status_table.add_row("Species Classifier", "⚠ Mock", "TransFG not installed")
    status_table.add_row("Web Interface", "✓ Available", "FastAPI ready")
    
    console.print(status_table)
    
    console.print("\n[yellow]Note: This is a minimal test version.[/yellow]")
    console.print("[yellow]Install full requirements for complete functionality.[/yellow]")


@app.command()
def process(
    video_path: Path = typer.Argument(..., help="Path to video file"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Process a video file (minimal mock version)."""
    
    console.print(f"\n[bold blue]Processing Video (Mock):[/bold blue] {video_path}")
    
    if not video_path.exists():
        rprint(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(1)
    
    if output_dir is None:
        from datetime import datetime
        output_dir = Path(f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    console.print(f"[bold blue]Output Directory:[/bold blue] {output_dir}")
    
    # Mock processing
    import time
    with console.status("[bold green]Processing video...") as status:
        time.sleep(2)  # Simulate processing
        status.update("[bold green]Creating mock results...")
        time.sleep(1)
    
    # Create mock output
    output_dir.mkdir(exist_ok=True)
    
    mock_result = {
        "video_path": str(video_path),
        "total_events": 5,
        "audio_only_events": 2,
        "video_only_events": 1,
        "audio_video_events": 2,
        "species_identified": 3,
        "unique_species": ["House Sparrow", "Robin", "Blue Jay"],
        "processing_note": "This is mock data for testing"
    }
    
    import json
    result_file = output_dir / "mock_results.json"
    with open(result_file, 'w') as f:
        json.dump(mock_result, f, indent=2)
    
    # Display results
    results_table = Table(title="Mock Processing Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Count", style="white")
    
    results_table.add_row("Total Events", str(mock_result["total_events"]))
    results_table.add_row("Audio Only", str(mock_result["audio_only_events"]))
    results_table.add_row("Video Only", str(mock_result["video_only_events"]))
    results_table.add_row("Audio + Video", str(mock_result["audio_video_events"]))
    results_table.add_row("Species Identified", str(mock_result["species_identified"]))
    
    console.print(results_table)
    
    console.print(f"\n[green]✓ Mock processing completed![/green]")
    console.print(f"[green]Results saved to: {result_file}[/green]")


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory containing video files"),
    output_dir: Path = typer.Argument(..., help="Base output directory"),
    pattern: str = typer.Option("*.mp4", "--pattern", "-p", help="File pattern to match")
):
    """Process multiple video files in batch (minimal mock version)."""
    
    if not input_dir.exists():
        rprint(f"[red]Error: Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)
    
    video_files = list(input_dir.glob(pattern))
    if not video_files:
        rprint(f"[red]Error: No video files found matching pattern '{pattern}' in {input_dir}[/red]")
        raise typer.Exit(1)
    
    console.print(f"\n[bold blue]Batch Processing (Mock):[/bold blue]")
    console.print(f"Found {len(video_files)} video files")
    console.print(f"Output directory: {output_dir}")
    
    # Mock batch processing
    results_table = Table(title="Batch Processing Results (Mock)")
    results_table.add_column("Video File", style="cyan")
    results_table.add_column("Status", style="white")
    results_table.add_column("Events", style="white")
    
    for video_file in video_files:
        # Mock processing
        import random
        events = random.randint(1, 10)
        
        results_table.add_row(
            video_file.name, 
            "[green]✓ Success[/green]", 
            str(events)
        )
    
    console.print(results_table)
    console.print(f"\n[green]✓ Mock batch processing completed![/green]")


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Port to run web server on"),
    host: str = typer.Option("localhost", "--host", help="Host to bind server to")
):
    """Start the web interface server (minimal version)."""
    
    console.print(f"\n[bold blue]Starting Web Interface (Minimal)...[/bold blue]")
    console.print(f"Host: {host}")
    console.print(f"Port: {port}")
    console.print(f"URL: http://{host}:{port}")
    
    try:
        import uvicorn
        from .web.app_minimal import app as web_app
        
        console.print("\n[green]✓ Starting server...[/green]")
        uvicorn.run(web_app, host=host, port=port)
        
    except ImportError:
        rprint("[red]Error: Web dependencies not installed[/red]")
        rprint("Install with: pip install fastapi uvicorn")
        raise typer.Exit(1)
    except Exception as e:
        rprint(f"[red]Error starting server: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def config(show: bool = typer.Option(False, "--show", help="Show current configuration")):
    """Manage pipeline configuration (minimal version)."""
    
    if show:
        console.print("\n[bold blue]Current Configuration (Minimal):[/bold blue]\n")
        
        config_table = Table()
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="white")
        
        config_table.add_row("Mode", "Minimal Test")
        config_table.add_row("Audio Threshold", "0.3")
        config_table.add_row("Video Threshold", "0.5")
        config_table.add_row("Species Threshold", "0.6")
        config_table.add_row("Models Available", "False (Mock mode)")
        
        console.print(config_table)


if __name__ == "__main__":
    app()