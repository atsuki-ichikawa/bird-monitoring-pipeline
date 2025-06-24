"""
Minimal FastAPI web application for testing without heavy dependencies.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Bird Monitoring Pipeline API (Minimal)",
    description="Audio-Visual Bird Detection and Classification System - Test Version",
    version="1.0.0-minimal"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and static files
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# Mock data for testing
mock_jobs = {}
mock_results = {}

# Request/Response models
class ProcessingJobRequest(BaseModel):
    audio_threshold: Optional[float] = None
    video_threshold: Optional[float] = None
    species_threshold: Optional[float] = None


class ProcessingJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    created_at: datetime


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(), "mode": "minimal"}


@app.get("/api/status")
async def get_system_status():
    """Get system status and component information."""
    return {
        "system_status": "operational",
        "mode": "minimal_test",
        "components": {
            "audio_detector": {"model_available": False, "status": "mock"},
            "video_detector": {"model_available": False, "status": "mock"},
            "species_classifier": {"model_available": False, "status": "mock"},
            "media_processor": {"available": True, "status": "ready"}
        },
        "active_jobs": len([j for j in mock_jobs.values() if j.get("status") == "processing"]),
        "completed_jobs": len([j for j in mock_jobs.values() if j.get("status") == "completed"])
    }


@app.post("/api/upload", response_model=ProcessingJobResponse)
async def upload_video(
    file: UploadFile = File(...),
    audio_threshold: Optional[float] = 0.3,
    video_threshold: Optional[float] = 0.5,
    species_threshold: Optional[float] = 0.6
):
    """Upload a video file for processing (mock implementation)."""
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Create mock job
    job_id = str(uuid.uuid4())
    
    # Store mock job info
    mock_jobs[job_id] = {
        "job_id": job_id,
        "status": "completed",  # Immediately mark as completed for testing
        "progress": 100.0,
        "filename": file.filename,
        "created_at": datetime.now(),
        "completed_at": datetime.now(),
        "result_id": str(uuid.uuid4()),
        "error": None
    }
    
    # Create mock result
    result_id = mock_jobs[job_id]["result_id"]
    mock_results[result_id] = {
        "result_id": result_id,
        "video_path": file.filename,
        "processing_started": datetime.now(),
        "processing_completed": datetime.now(),
        "total_events": 5,
        "audio_only_events": 2,
        "video_only_events": 1,
        "audio_video_events": 2,
        "species_identified": 3,
        "unique_species": ["House Sparrow", "Robin", "Blue Jay"]
    }
    
    return ProcessingJobResponse(
        job_id=job_id,
        status="completed",
        message="Video uploaded and processed successfully (mock)",
        created_at=datetime.now()
    )


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    
    if job_id not in mock_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return mock_jobs[job_id]


@app.get("/api/jobs")
async def list_jobs():
    """List all processing jobs."""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job_data["status"],
                "filename": job_data.get("filename"),
                "created_at": job_data["created_at"],
                "completed_at": job_data.get("completed_at")
            }
            for job_id, job_data in mock_jobs.items()
        ]
    }


@app.get("/api/results/{result_id}")
async def get_result_summary(result_id: str):
    """Get summary of processing results."""
    
    if result_id not in mock_results:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return mock_results[result_id]


@app.get("/api/results/{result_id}/events")
async def get_result_events(
    result_id: str,
    limit: Optional[int] = 100,
    offset: Optional[int] = 0
):
    """Get detection events from results (mock data)."""
    
    if result_id not in mock_results:
        raise HTTPException(status_code=404, detail="Result not found")
    
    # Mock events
    mock_events = [
        {
            "event_id": f"evt_{i:03d}",
            "start_time": i * 10.0,
            "end_time": i * 10.0 + 3.0,
            "detection_type": ["audio_only", "video_only", "audio_video"][i % 3],
            "initial_confidence": 0.5 + (i * 0.1) % 0.4,
            "final_confidence": 0.7 + (i * 0.05) % 0.3,
            "species_name": ["House Sparrow", "Robin", "Blue Jay"][i % 3] if i % 2 == 0 else None,
            "species_confidence": 0.8 + (i * 0.02) % 0.2 if i % 2 == 0 else None,
            "bounding_box": {
                "timestamp": i * 10.0 + 1.5,
                "x": 100 + i * 20,
                "y": 50 + i * 10,
                "width": 150,
                "height": 100
            } if i % 3 != 0 else None
        }
        for i in range(5)
    ]
    
    # Apply pagination
    total_count = len(mock_events)
    events = mock_events[offset:offset + limit]
    
    return {
        "events": events,
        "total_count": total_count,
        "limit": limit,
        "offset": offset
    }


@app.get("/api/results/{result_id}/timeline")
async def get_result_timeline(result_id: str, resolution: Optional[int] = 100):
    """Get timeline data for visualization (mock data)."""
    
    if result_id not in mock_results:
        raise HTTPException(status_code=404, detail="Result not found")
    
    # Mock timeline data
    timeline = []
    duration = 50.0  # 50 seconds mock video
    
    for i in range(resolution):
        time_point = (i / resolution) * duration
        event_count = max(0, int(3 * abs(0.5 - (i / resolution))))  # Bell curve
        
        timeline.append({
            "time": time_point,
            "event_count": event_count,
            "max_confidence": 0.8 if event_count > 0 else 0.0,
            "detection_types": {
                "audio_only": event_count // 3,
                "video_only": event_count // 3,
                "audio_video": event_count - 2 * (event_count // 3)
            }
        })
    
    return {
        "timeline": timeline,
        "duration": duration,
        "resolution": resolution,
        "total_events": 5
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )