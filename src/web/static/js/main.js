/**
 * Main JavaScript for Bird Monitoring Pipeline Web Interface
 */

class BirdMonitorApp {
    constructor() {
        this.currentResultId = null;
        this.currentEvents = [];
        this.selectedEventId = null;
        this.videoPlayer = null;
        this.timelineCanvas = null;
        this.timelineContext = null;
        
        this.init();
    }
    
    init() {
        // Initialize DOM elements
        this.videoPlayer = document.getElementById('video-player');
        this.timelineCanvas = document.getElementById('timeline-canvas');
        this.timelineContext = this.timelineCanvas?.getContext('2d');
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Load initial data
        this.loadSystemStatus();
        this.loadJobs();
        
        // Set up periodic updates
        setInterval(() => {
            this.loadSystemStatus();
            this.loadJobs();
        }, 5000);
    }
    
    setupEventListeners() {
        // Upload form
        const uploadForm = document.getElementById('upload-form');
        if (uploadForm) {
            uploadForm.addEventListener('submit', (e) => this.handleUpload(e));
        }
        
        // Confidence filter
        const confidenceFilter = document.getElementById('confidence-filter');
        if (confidenceFilter) {
            confidenceFilter.addEventListener('input', (e) => {
                document.getElementById('confidence-value').textContent = 
                    parseFloat(e.target.value).toFixed(1);
            });
        }
        
        // Apply filters button
        const applyFilters = document.getElementById('apply-filters');
        if (applyFilters) {
            applyFilters.addEventListener('click', () => this.applyFilters());
        }
        
        // Export buttons
        const exportJson = document.getElementById('export-json');
        if (exportJson) {
            exportJson.addEventListener('click', () => this.exportResults('json'));
        }
        
        const exportCsv = document.getElementById('export-csv');
        if (exportCsv) {
            exportCsv.addEventListener('click', () => this.exportResults('csv'));
        }
        
        // Timeline canvas
        if (this.timelineCanvas) {
            this.timelineCanvas.addEventListener('click', (e) => this.handleTimelineClick(e));
            this.timelineCanvas.addEventListener('mousemove', (e) => this.handleTimelineHover(e));
        }
        
        // Video player
        if (this.videoPlayer) {
            this.videoPlayer.addEventListener('timeupdate', () => this.updateTimelineMarker());
        }
    }
    
    async loadSystemStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            this.updateSystemStatus(data);
        } catch (error) {
            console.error('Failed to load system status:', error);
        }
    }
    
    updateSystemStatus(data) {
        const componentStatus = document.getElementById('component-status');
        const queueStatus = document.getElementById('queue-status');
        
        if (componentStatus) {
            const components = data.components;
            let html = '<div class="list-group list-group-flush">';
            
            // Audio detector
            const audioStatus = components.audio_detector?.model_available ? 'available' : 'warning';
            html += `
                <div class="list-group-item d-flex justify-content-between align-items-center">
                    <span><span class="status-indicator status-${audioStatus}"></span>Audio Detector</span>
                    <span class="badge bg-${audioStatus === 'available' ? 'success' : 'warning'}">
                        ${audioStatus === 'available' ? 'Ready' : 'Mock'}
                    </span>
                </div>
            `;
            
            // Video detector
            const videoStatus = components.video_detector?.model_available ? 'available' : 'warning';
            html += `
                <div class="list-group-item d-flex justify-content-between align-items-center">
                    <span><span class="status-indicator status-${videoStatus}"></span>Video Detector</span>
                    <span class="badge bg-${videoStatus === 'available' ? 'success' : 'warning'}">
                        ${videoStatus === 'available' ? 'Ready' : 'Mock'}
                    </span>
                </div>
            `;
            
            // Species classifier
            const speciesStatus = components.species_classifier?.model_available ? 'available' : 'warning';
            html += `
                <div class="list-group-item d-flex justify-content-between align-items-center">
                    <span><span class="status-indicator status-${speciesStatus}"></span>Species Classifier</span>
                    <span class="badge bg-${speciesStatus === 'available' ? 'success' : 'warning'}">
                        ${speciesStatus === 'available' ? 'Ready' : 'Mock'}
                    </span>
                </div>
            `;
            
            html += '</div>';
            componentStatus.innerHTML = html;
        }
        
        if (queueStatus) {
            queueStatus.innerHTML = `
                <div class="row text-center">
                    <div class="col-6">
                        <h4 class="text-primary">${data.active_jobs || 0}</h4>
                        <p class="mb-0">Active Jobs</p>
                    </div>
                    <div class="col-6">
                        <h4 class="text-success">${data.completed_jobs || 0}</h4>
                        <p class="mb-0">Completed</p>
                    </div>
                </div>
            `;
        }
    }
    
    async loadJobs() {
        try {
            const response = await fetch('/api/jobs');
            const data = await response.json();
            this.updateJobsList(data.jobs);
        } catch (error) {
            console.error('Failed to load jobs:', error);
        }
    }
    
    updateJobsList(jobs) {
        const jobsList = document.getElementById('jobs-list');
        if (!jobsList) return;
        
        if (jobs.length === 0) {
            jobsList.innerHTML = `
                <div class="text-center text-muted">
                    No jobs found. Upload a video to get started.
                </div>
            `;
            return;
        }
        
        let html = '';
        jobs.forEach(job => {
            const statusClass = this.getStatusClass(job.status);
            const timeAgo = this.getTimeAgo(job.created_at);
            
            html += `
                <div class="job-card card">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <h6 class="card-title mb-1">Job ${job.job_id.substring(0, 8)}</h6>
                                <small class="text-muted">${timeAgo}</small>
                            </div>
                            <div class="text-end">
                                <span class="job-status ${job.status}">${job.status}</span>
                                ${job.status === 'completed' ? 
                                    `<button class="btn btn-sm btn-outline-primary ms-2" 
                                             onclick="app.viewResults('${job.job_id}')">
                                        View Results
                                    </button>` : ''}
                            </div>
                        </div>
                        ${job.status === 'processing' ? `
                            <div class="progress mt-2">
                                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                     role="progressbar" style="width: ${job.progress || 0}%"></div>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        });
        
        jobsList.innerHTML = html;
    }
    
    getStatusClass(status) {
        const classes = {
            'queued': 'info',
            'processing': 'warning',
            'completed': 'success',
            'failed': 'danger'
        };
        return classes[status] || 'secondary';
    }
    
    getTimeAgo(timestamp) {
        const now = new Date();
        const time = new Date(timestamp);
        const diffInSeconds = Math.floor((now - time) / 1000);
        
        if (diffInSeconds < 60) return 'Just now';
        if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`;
        if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`;
        return `${Math.floor(diffInSeconds / 86400)} days ago`;
    }
    
    async handleUpload(event) {
        event.preventDefault();
        
        const formData = new FormData();
        const videoFile = document.getElementById('video-file').files[0];
        const audioThreshold = document.getElementById('audio-threshold').value;
        const videoThreshold = document.getElementById('video-threshold').value;
        const speciesThreshold = document.getElementById('species-threshold').value;
        
        if (!videoFile) {
            this.showNotification('Please select a video file', 'error');
            return;
        }
        
        formData.append('file', videoFile);
        formData.append('audio_threshold', audioThreshold);
        formData.append('video_threshold', videoThreshold);
        formData.append('species_threshold', speciesThreshold);
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                this.showNotification('Video uploaded successfully!', 'success');
                document.getElementById('upload-form').reset();
                this.loadJobs(); // Refresh jobs list
            } else {
                const error = await response.json();
                this.showNotification(error.detail || 'Upload failed', 'error');
            }
        } catch (error) {
            this.showNotification('Upload failed: ' + error.message, 'error');
        }
    }
    
    async viewResults(jobId) {
        try {
            // Get job status to find result ID
            const jobResponse = await fetch(`/api/jobs/${jobId}`);
            const jobData = await jobResponse.json();
            
            if (!jobData.result_id) {
                this.showNotification('Results not available yet', 'warning');
                return;
            }
            
            // Load results
            await this.loadResults(jobData.result_id);
            
            // Show results section
            document.getElementById('results').style.display = 'block';
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
            
        } catch (error) {
            this.showNotification('Failed to load results: ' + error.message, 'error');
        }
    }
    
    async loadResults(resultId) {
        this.currentResultId = resultId;
        
        try {
            // Load result summary
            const summaryResponse = await fetch(`/api/results/${resultId}`);
            const summary = await summaryResponse.json();
            this.updateResultsSummary(summary);
            
            // Load events
            const eventsResponse = await fetch(`/api/results/${resultId}/events?limit=1000`);
            const eventsData = await eventsResponse.json();
            this.currentEvents = eventsData.events;
            this.updateEventsTable(this.currentEvents);
            
            // Load species for filter
            const speciesResponse = await fetch(`/api/results/${resultId}/species`);
            const speciesData = await speciesResponse.json();
            this.updateSpeciesFilter(Object.keys(speciesData.species_statistics || {}));
            
            // Load timeline
            const timelineResponse = await fetch(`/api/results/${resultId}/timeline`);
            const timelineData = await timelineResponse.json();
            this.drawTimeline(timelineData);
            
        } catch (error) {
            this.showNotification('Failed to load results: ' + error.message, 'error');
        }
    }
    
    updateResultsSummary(summary) {
        document.getElementById('total-events').textContent = summary.total_events;
        document.getElementById('audio-video-events').textContent = summary.audio_video_events;
        document.getElementById('species-identified').textContent = summary.species_identified;
        document.getElementById('unique-species').textContent = summary.unique_species.length;
    }
    
    updateEventsTable(events) {
        const tbody = document.getElementById('events-tbody');
        if (!tbody) return;
        
        let html = '';
        events.forEach(event => {
            const timeStr = `${this.formatTime(event.start_time)} - ${this.formatTime(event.end_time)}`;
            const confidenceClass = this.getConfidenceClass(event.final_confidence);
            
            html += `
                <tr onclick="app.selectEvent('${event.event_id}')" data-event-id="${event.event_id}">
                    <td>${timeStr}</td>
                    <td>
                        <span class="detection-type ${event.detection_type.replace('_', '-')}">
                            ${event.detection_type.replace('_', ' ')}
                        </span>
                    </td>
                    <td>
                        <div class="confidence-bar">
                            <div class="confidence-fill ${confidenceClass}" 
                                 style="width: ${event.final_confidence * 100}%"></div>
                        </div>
                        <small>${(event.final_confidence * 100).toFixed(1)}%</small>
                    </td>
                    <td>
                        ${event.species_name ? 
                            `<span class="species-tag">${event.species_name}</span>` : 
                            '<span class="text-muted">—</span>'}
                    </td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary" 
                                onclick="event.stopPropagation(); app.jumpToTime(${event.start_time})">
                            <i class="bi bi-play"></i>
                        </button>
                    </td>
                </tr>
            `;
        });
        
        tbody.innerHTML = html;
    }
    
    updateSpeciesFilter(species) {
        const select = document.getElementById('species-filter');
        if (!select) return;
        
        // Clear existing options except "All Species"
        select.innerHTML = '<option value="">All Species</option>';
        
        species.forEach(species => {
            const option = document.createElement('option');
            option.value = species;
            option.textContent = species;
            select.appendChild(option);
        });
    }
    
    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${minutes}:${secs.toString().padStart(2, '0')}`;
    }
    
    getConfidenceClass(confidence) {
        if (confidence >= 0.8) return 'confidence-high';
        if (confidence >= 0.5) return 'confidence-medium';
        return 'confidence-low';
    }
    
    selectEvent(eventId) {
        // Remove previous selection
        document.querySelectorAll('#events-tbody tr').forEach(tr => {
            tr.classList.remove('selected');
        });
        
        // Add selection to current row
        const row = document.querySelector(`tr[data-event-id="${eventId}"]`);
        if (row) {
            row.classList.add('selected');
        }
        
        // Update detection info panel
        const event = this.currentEvents.find(e => e.event_id === eventId);
        if (event) {
            this.updateDetectionInfo(event);
            this.selectedEventId = eventId;
        }
    }
    
    updateDetectionInfo(event) {
        const infoPanel = document.getElementById('detection-info');
        if (!infoPanel) return;
        
        let html = `
            <h6>Event Details</h6>
            <p><strong>Time:</strong> ${this.formatTime(event.start_time)} - ${this.formatTime(event.end_time)}</p>
            <p><strong>Type:</strong> ${event.detection_type.replace('_', ' ')}</p>
            <p><strong>Confidence:</strong> ${(event.final_confidence * 100).toFixed(1)}%</p>
        `;
        
        if (event.species_name) {
            html += `
                <p><strong>Species:</strong> ${event.species_name}</p>
                <p><strong>Species Confidence:</strong> ${(event.species_confidence * 100).toFixed(1)}%</p>
            `;
        }
        
        if (event.bounding_box) {
            html += `
                <p><strong>Bounding Box:</strong></p>
                <small>
                    X: ${Math.round(event.bounding_box.x)}, 
                    Y: ${Math.round(event.bounding_box.y)}, 
                    W: ${Math.round(event.bounding_box.width)}, 
                    H: ${Math.round(event.bounding_box.height)}
                </small>
            `;
        }
        
        infoPanel.innerHTML = html;
    }
    
    jumpToTime(time) {
        if (this.videoPlayer) {
            this.videoPlayer.currentTime = time;
            this.videoPlayer.play();
        }
    }
    
    drawTimeline(timelineData) {
        if (!this.timelineContext || !timelineData.timeline) return;
        
        const canvas = this.timelineCanvas;
        const ctx = this.timelineContext;
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw background
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, width, height);
        
        // Draw timeline events
        const timelineEvents = timelineData.timeline;
        const maxEvents = Math.max(...timelineEvents.map(t => t.event_count));
        
        timelineEvents.forEach((timePoint, index) => {
            const x = (index / timelineEvents.length) * width;
            const barHeight = (timePoint.event_count / maxEvents) * (height - 20);
            
            // Color based on detection types
            if (timePoint.detection_types.audio_video > 0) {
                ctx.fillStyle = '#198754'; // Green for audio+video
            } else if (timePoint.detection_types.video_only > 0) {
                ctx.fillStyle = '#ffc107'; // Yellow for video only
            } else if (timePoint.detection_types.audio_only > 0) {
                ctx.fillStyle = '#0dcaf0'; // Blue for audio only
            } else {
                ctx.fillStyle = '#dee2e6'; // Gray for no events
            }
            
            ctx.fillRect(x, height - barHeight, Math.max(1, width / timelineEvents.length - 1), barHeight);
        });
        
        // Draw time axis
        ctx.strokeStyle = '#6c757d';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, height - 1);
        ctx.lineTo(width, height - 1);
        ctx.stroke();
    }
    
    handleTimelineClick(event) {
        if (!this.currentEvents.length) return;
        
        const rect = this.timelineCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const ratio = x / rect.width;
        
        // Find the maximum time from events
        const maxTime = Math.max(...this.currentEvents.map(e => e.end_time));
        const clickTime = ratio * maxTime;
        
        // Jump to that time
        this.jumpToTime(clickTime);
    }
    
    updateTimelineMarker() {
        // This would update a visual marker on the timeline showing current video position
        // Implementation depends on timeline visualization approach
    }
    
    async applyFilters() {
        if (!this.currentResultId) return;
        
        const species = document.getElementById('species-filter').value;
        const minConfidence = document.getElementById('confidence-filter').value;
        const detectionType = document.getElementById('type-filter').value;
        
        const params = new URLSearchParams();
        if (species) params.append('species', species);
        if (minConfidence && minConfidence > 0) params.append('min_confidence', minConfidence);
        if (detectionType) params.append('detection_type', detectionType);
        
        try {
            const response = await fetch(`/api/results/${this.currentResultId}/events?${params}`);
            const data = await response.json();
            this.updateEventsTable(data.events);
        } catch (error) {
            this.showNotification('Failed to apply filters: ' + error.message, 'error');
        }
    }
    
    async exportResults(format) {
        if (!this.currentResultId) {
            this.showNotification('No results to export', 'warning');
            return;
        }
        
        try {
            const response = await fetch(`/api/results/${this.currentResultId}/export?format=${format}`);
            
            if (format === 'json') {
                const data = await response.json();
                this.downloadJson(data, `bird_detection_results_${this.currentResultId}.json`);
            } else if (format === 'csv') {
                const blob = await response.blob();
                this.downloadBlob(blob, `bird_detection_results_${this.currentResultId}.csv`);
            }
            
            this.showNotification(`Results exported as ${format.toUpperCase()}`, 'success');
        } catch (error) {
            this.showNotification('Export failed: ' + error.message, 'error');
        }
    }
    
    downloadJson(data, filename) {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        this.downloadBlob(blob, filename);
    }
    
    downloadBlob(blob, filename) {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }
    
    showNotification(message, type = 'info') {
        const toast = document.getElementById('notification-toast');
        const messageEl = document.getElementById('notification-message');
        
        if (toast && messageEl) {
            messageEl.textContent = message;
            
            // Update toast styling based on type
            toast.className = `toast show bg-${type === 'error' ? 'danger' : type}`;
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                toast.classList.remove('show');
            }, 5000);
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new BirdMonitorApp();
});