"""
FastAPI web application for the bird monitoring pipeline.
Provides REST APIs and web interface for video processing and result visualization.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..core.pipeline import BirdMonitoringPipeline
from ..models.data_models import PipelineResult, ProcessingConfig, DetectionEvent
from ..utils.config import get_settings
from ..utils.logging import get_logger

# Initialize FastAPI app
app = FastAPI(
    title="Bird Monitoring Pipeline API",
    description="Audio-Visual Bird Detection and Classification System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get settings and logger
settings = get_settings()
logger = get_logger("web")

# Templates and static files
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# In-memory storage for processing jobs (use Redis/database for production)
processing_jobs: Dict[str, Dict] = {}
completed_results: Dict[str, PipelineResult] = {}


# Request/Response models
class ProcessingJobRequest(BaseModel):
    audio_threshold: Optional[float] = None
    video_threshold: Optional[float] = None
    species_threshold: Optional[float] = None
    correlation_window: Optional[float] = None


class ProcessingJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    created_at: datetime


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None
    result_id: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class DetectionEventResponse(BaseModel):
    event_id: str
    start_time: float
    end_time: float
    detection_type: str
    initial_confidence: float
    final_confidence: float
    species_name: Optional[str] = None
    species_confidence: Optional[float] = None
    bounding_box: Optional[Dict] = None


class ResultSummaryResponse(BaseModel):
    result_id: str
    video_path: str
    processing_started: datetime
    processing_completed: Optional[datetime]
    total_events: int
    audio_only_events: int
    video_only_events: int
    audio_video_events: int
    species_identified: int
    unique_species: List[str]


# Dependency to get pipeline instance
def get_pipeline() -> BirdMonitoringPipeline:
    """Get a pipeline instance."""
    return BirdMonitoringPipeline()


# Background task to process video
async def process_video_task(
    job_id: str,
    video_path: Path,
    output_dir: Path,
    config: ProcessingConfig
):
    """Background task for video processing."""
    try:
        # Update job status
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["progress"] = 0.0
        
        # Initialize pipeline
        pipeline = BirdMonitoringPipeline(config)
        
        # Process video
        result = pipeline.process_video(video_path, output_dir, config)
        
        # Store result
        result_id = str(uuid.uuid4())
        completed_results[result_id] = result
        
        # Update job status
        processing_jobs[job_id].update({
            "status": "completed",
            "progress": 100.0,
            "result_id": result_id,
            "completed_at": datetime.now()
        })
        
        logger.info(f"Video processing completed for job {job_id}")
        
    except Exception as e:
        # Update job status with error
        processing_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now()
        })
        
        logger.error(f"Video processing failed for job {job_id}: {e}")


# API Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": {}})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}


@app.get("/api/status")
async def get_system_status():
    """Get system status and component information."""
    try:
        pipeline = get_pipeline()
        status = pipeline.get_component_status()
        return {
            "system_status": "operational",
            "components": status,
            "active_jobs": len([j for j in processing_jobs.values() if j["status"] == "processing"]),
            "completed_jobs": len([j for j in processing_jobs.values() if j["status"] == "completed"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {e}")


@app.post("/api/upload", response_model=ProcessingJobResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request_data: ProcessingJobRequest = Depends()
):
    """Upload a video file for processing."""
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    try:
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Create upload directory
        upload_dir = settings.data_dir / "uploads" / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        video_path = upload_dir / file.filename
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create output directory
        output_dir = settings.data_dir / "output" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create processing config
        config = settings.create_processing_config(str(video_path), str(output_dir))
        
        # Override settings if provided
        if request_data.audio_threshold is not None:
            config.audio_confidence_threshold = request_data.audio_threshold
        if request_data.video_threshold is not None:
            config.video_confidence_threshold = request_data.video_threshold
        if request_data.species_threshold is not None:
            config.species_confidence_threshold = request_data.species_threshold
        if request_data.correlation_window is not None:
            config.temporal_correlation_window = request_data.correlation_window
        
        # Store job info
        processing_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "created_at": datetime.now(),
            "completed_at": None,
            "result_id": None,
            "error": None
        }
        
        # Start background processing
        background_tasks.add_task(
            process_video_task, job_id, video_path, output_dir, config
        )
        
        return ProcessingJobResponse(
            job_id=job_id,
            status="queued",
            message="Video uploaded successfully, processing started",
            created_at=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = processing_jobs[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job_data["status"],
        progress=job_data.get("progress"),
        message=job_data.get("message"),
        result_id=job_data.get("result_id"),
        error=job_data.get("error"),
        created_at=job_data["created_at"],
        completed_at=job_data.get("completed_at")
    )


@app.get("/api/jobs")
async def list_jobs():
    """List all processing jobs."""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job_data["status"],
                "created_at": job_data["created_at"],
                "completed_at": job_data.get("completed_at")
            }
            for job_id, job_data in processing_jobs.items()
        ]
    }


@app.get("/api/results/{result_id}", response_model=ResultSummaryResponse)
async def get_result_summary(result_id: str):
    """Get summary of processing results."""
    
    if result_id not in completed_results:
        raise HTTPException(status_code=404, detail="Result not found")
    
    result = completed_results[result_id]
    
    return ResultSummaryResponse(
        result_id=result_id,
        video_path=result.video_path,
        processing_started=result.processing_started,
        processing_completed=result.processing_completed,
        total_events=result.total_events,
        audio_only_events=result.audio_only_events,
        video_only_events=result.video_only_events,
        audio_video_events=result.audio_video_events,
        species_identified=result.species_identified,
        unique_species=result.unique_species
    )


@app.get("/api/results/{result_id}/events")
async def get_result_events(
    result_id: str,
    limit: Optional[int] = 100,
    offset: Optional[int] = 0,
    species: Optional[str] = None,
    min_confidence: Optional[float] = None,
    detection_type: Optional[str] = None
):
    """Get detection events from results with filtering."""
    
    if result_id not in completed_results:
        raise HTTPException(status_code=404, detail="Result not found")
    
    result = completed_results[result_id]
    events = result.events
    
    # Apply filters
    if species:
        events = [
            e for e in events 
            if e.species_classification and e.species_classification.species_name == species
        ]
    
    if min_confidence is not None:
        events = [e for e in events if e.final_confidence >= min_confidence]
    
    if detection_type:
        events = [e for e in events if e.detection_type.value == detection_type]
    
    # Apply pagination
    total_count = len(events)
    events = events[offset:offset + limit]
    
    # Convert to response format
    event_responses = []
    for event in events:
        event_data = DetectionEventResponse(
            event_id=event.event_id,
            start_time=event.start_time,
            end_time=event.end_time,
            detection_type=event.detection_type.value,
            initial_confidence=event.initial_confidence,
            final_confidence=event.final_confidence,
        )
        
        if event.species_classification:
            event_data.species_name = event.species_classification.species_name
            event_data.species_confidence = event.species_classification.confidence
        
        if event.video_detection:
            bbox = event.video_detection.bounding_box
            event_data.bounding_box = {
                "timestamp": bbox.timestamp,
                "x": bbox.x,
                "y": bbox.y,
                "width": bbox.width,
                "height": bbox.height
            }
        
        event_responses.append(event_data)
    
    return {
        "events": event_responses,
        "total_count": total_count,
        "limit": limit,
        "offset": offset
    }


@app.get("/api/results/{result_id}/species")
async def get_result_species(result_id: str):
    """Get species statistics from results."""
    
    if result_id not in completed_results:
        raise HTTPException(status_code=404, detail="Result not found")
    
    result = completed_results[result_id]
    
    # Calculate species statistics
    species_stats = {}
    for event in result.events:
        if event.species_classification:
            species = event.species_classification.species_name
            if species not in species_stats:
                species_stats[species] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "max_confidence": 0.0,
                    "min_confidence": 1.0
                }
            
            stats = species_stats[species]
            stats["count"] += 1
            confidence = event.species_classification.confidence
            stats["max_confidence"] = max(stats["max_confidence"], confidence)
            stats["min_confidence"] = min(stats["min_confidence"], confidence)
            
            # Update average confidence
            stats["avg_confidence"] = (
                (stats["avg_confidence"] * (stats["count"] - 1) + confidence) / stats["count"]
            )
    
    return {
        "species_statistics": species_stats,
        "total_species": len(species_stats)
    }


@app.get("/api/results/{result_id}/timeline")
async def get_result_timeline(result_id: str, resolution: Optional[int] = 100):
    """Get timeline data for visualization."""
    
    if result_id not in completed_results:
        raise HTTPException(status_code=404, detail="Result not found")
    
    result = completed_results[result_id]
    
    if not result.events:
        return {"timeline": [], "duration": 0}
    
    # Calculate video duration
    max_time = max(event.end_time for event in result.events)
    duration = max_time
    
    # Create timeline bins
    bin_size = duration / resolution
    timeline = []
    
    for i in range(resolution):
        bin_start = i * bin_size
        bin_end = (i + 1) * bin_size
        
        # Count events in this bin
        bin_events = [
            event for event in result.events
            if not (event.end_time < bin_start or event.start_time > bin_end)
        ]
        
        timeline.append({
            "time": bin_start,
            "event_count": len(bin_events),
            "max_confidence": max([e.final_confidence for e in bin_events], default=0),
            "detection_types": {
                "audio_only": len([e for e in bin_events if e.detection_type.value == "audio_only"]),
                "video_only": len([e for e in bin_events if e.detection_type.value == "video_only"]),
                "audio_video": len([e for e in bin_events if e.detection_type.value == "audio_video"])
            }
        })
    
    return {
        "timeline": timeline,
        "duration": duration,
        "resolution": resolution,
        "total_events": len(result.events)
    }


@app.get("/api/results/{result_id}/export")
async def export_results(result_id: str, format: str = "json"):
    """Export results in specified format."""
    
    if result_id not in completed_results:
        raise HTTPException(status_code=404, detail="Result not found")
    
    result = completed_results[result_id]
    
    if format.lower() == "json":
        # Return JSON export
        return JSONResponse(content=result.to_export_dict())
    
    elif format.lower() == "csv":
        # Create CSV export
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "event_id", "start_time", "end_time", "detection_type",
            "initial_confidence", "final_confidence", "species_name",
            "species_confidence", "bbox_x", "bbox_y", "bbox_width", "bbox_height"
        ])
        
        # Write events
        for event in result.events:
            row = [
                event.event_id,
                event.start_time,
                event.end_time,
                event.detection_type.value,
                event.initial_confidence,
                event.final_confidence,
                event.species_classification.species_name if event.species_classification else "",
                event.species_classification.confidence if event.species_classification else "",
                event.video_detection.bounding_box.x if event.video_detection else "",
                event.video_detection.bounding_box.y if event.video_detection else "",
                event.video_detection.bounding_box.width if event.video_detection else "",
                event.video_detection.bounding_box.height if event.video_detection else ""
            ]
            writer.writerow(row)
        
        output.seek(0)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=results_{result_id}.csv"}
        )
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")


# WebSocket endpoint for real-time updates (optional)
@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_updates(websocket, job_id: str):
    """WebSocket endpoint for real-time job updates."""
    await websocket.accept()
    
    try:
        while True:
            if job_id in processing_jobs:
                job_data = processing_jobs[job_id]
                await websocket.send_json(job_data)
                
                # If job is completed or failed, close connection
                if job_data["status"] in ["completed", "failed"]:
                    break
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=settings.web_host,
        port=settings.web_port,
        reload=settings.web_reload
    )