from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import asyncio
import json
import uuid
from datetime import datetime
import logging
import re
import threading
import time

# Import your existing orchestrator
from host import LLMOrchestrator, ProcessingResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Job Search Orchestrator API",
    description="AI-powered job search API with LLM orchestration and progress tracking",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class JobSearchRequest(BaseModel):
    query: str = Field(..., description="Company name or job URL to search")
    max_pages: int = Field(default=1, ge=1, le=10, description="Maximum pages to crawl")
    model: str = Field(default="llama3-70b-8192", description="LLM model to use")

class JobOffer(BaseModel):
    title: str
    company: str
    location: str
    description: str
    salary: Optional[str] = "not mentioned"
    requirements: Optional[str] = "not mentioned"
    contrat_type: Optional[str] = "not mentioned"
    required_skill: Optional[str] = "not mentioned"
    post_date: Optional[str] = "not mentioned"
    url: Optional[str] = None

class ProgressStep(BaseModel):
    type: str  # MCP_INIT, LLM_ORCHESTRATOR, etc.
    message: str
    details: Optional[str] = None
    status: str  # pending, processing, completed, error
    timing: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ProgressResponse(BaseModel):
    steps: List[ProgressStep] = []
    current_step: Optional[int] = None
    progress: float = 0.0  # 0-100
    total_steps: int = 0

class JobSearchResponse(BaseModel):
    success: bool
    jobs: List[JobOffer] = []
    execution_plan: Optional[str] = None
    tools_used: List[str] = []
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    request_id: str

class JobSearchStatus(BaseModel):
    request_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: str = ""
    result: Optional[JobSearchResponse] = None

# Progress tracking storage
progress_storage: Dict[str, ProgressResponse] = {}
job_requests: Dict[str, JobSearchStatus] = {}

class ProgressTracker:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.steps = []
        self.current_step_index = None
        self.start_time = datetime.now()
        
        # Initialize progress storage
        progress_storage[request_id] = ProgressResponse(
            steps=[],
            current_step=None,
            progress=0.0,
            total_steps=0
        )
    
    def add_step(self, step_type: str, message: str, details: str = None, status: str = "processing"):
        """Add a new progress step"""
        step = ProgressStep(
            type=step_type,
            message=message,
            details=details,
            status=status,
            timing=self._get_timing()
        )
        
        self.steps.append(step)
        self.current_step_index = len(self.steps) - 1
        
        # Update storage
        progress_storage[self.request_id].steps = self.steps
        progress_storage[self.request_id].current_step = self.current_step_index
        progress_storage[self.request_id].total_steps = len(self.steps)
        progress_storage[self.request_id].progress = self._calculate_progress()
        
        logger.info(f"[PROGRESS] {self.request_id}: {step_type} - {message}")
    
    def update_current_step(self, message: str = None, details: str = None, status: str = None):
        """Update the current step"""
        if self.current_step_index is not None and self.current_step_index < len(self.steps):
            if message:
                self.steps[self.current_step_index].message = message
            if details:
                self.steps[self.current_step_index].details = details
            if status:
                self.steps[self.current_step_index].status = status
            
            # Update storage
            progress_storage[self.request_id].steps = self.steps
    
    def complete_current_step(self):
        """Mark the current step as completed"""
        if self.current_step_index is not None and self.current_step_index < len(self.steps):
            self.steps[self.current_step_index].status = "completed"
            progress_storage[self.request_id].steps = self.steps
            progress_storage[self.request_id].progress = self._calculate_progress()
    
    def error_current_step(self, error_message: str):
        """Mark the current step as error"""
        if self.current_step_index is not None and self.current_step_index < len(self.steps):
            self.steps[self.current_step_index].status = "error"
            self.steps[self.current_step_index].details = error_message
            progress_storage[self.request_id].steps = self.steps
    
    def _get_timing(self) -> str:
        """Get elapsed time since start"""
        elapsed = datetime.now() - self.start_time
        return f"{elapsed.total_seconds():.1f}s"
    
    def _calculate_progress(self) -> float:
        """Calculate progress percentage"""
        if not self.steps:
            return 0.0
        
        completed_steps = sum(1 for step in self.steps if step.status == "completed")
        return (completed_steps / len(self.steps)) * 100

# Enhanced LLM Orchestrator with progress tracking
class ProgressAwareLLMOrchestrator(LLMOrchestrator):
    def __init__(self, progress_tracker: ProgressTracker = None):
        super().__init__()
        self.progress_tracker = progress_tracker
    
    async def initialize_mcp_client(self, server_script: str) -> bool:
        """Override to add progress tracking"""
        if self.progress_tracker:
            self.progress_tracker.add_step(
                "MCP_INIT", 
                f"Connecting to MCP server: {server_script}",
                "Initializing Model Context Protocol client"
            )
        
        result = await super().initialize_mcp_client(server_script)
        
        if self.progress_tracker:
            if result:
                self.progress_tracker.complete_current_step()
                # FIX: Handle available_tools properly based on its actual type
                if hasattr(self, 'available_tools'):
                    if isinstance(self.available_tools, dict):
                        tools_info = f"Connected! Available tools: {list(self.available_tools.keys())}"
                        tools_count = len(self.available_tools)
                    elif isinstance(self.available_tools, list):
                        tools_info = f"Connected! Available tools: {self.available_tools}"
                        tools_count = len(self.available_tools)
                    else:
                        tools_info = "Connected! Tools available"
                        tools_count = 0
                else:
                    tools_info = "Connected! Tools info not available"
                    tools_count = 0
                    
                self.progress_tracker.add_step(
                    "MCP_INIT",
                    tools_info,
                    f"Successfully connected with {tools_count} tools",
                    "completed"
                )
            else:
                self.progress_tracker.error_current_step("Failed to connect to MCP server")
        
        return result
    
    async def llm_orchestrate(self, user_input: str, max_pages: int = 1):
        """Override to add progress tracking"""
        if self.progress_tracker:
            self.progress_tracker.add_step(
                "LLM_ORCHESTRATOR",
                f"Analyzing input: '{user_input}'",
                "LLM is analyzing the user input to determine the best approach"
            )
        
        # Call the original method but intercept the logging
        original_method = super().llm_orchestrate
        result = await self._track_orchestration(original_method, user_input, max_pages)
        
        return result
    
    async def _track_orchestration(self, original_method, user_input: str, max_pages: int):
        """Track the orchestration process by intercepting logs"""
        # Capture the original logging
        import sys
        from io import StringIO
        
        # Store original stdout
        old_stdout = sys.stdout
        
        try:
            # Redirect stdout to capture logs
            sys.stdout = StringIO()
            
            # Run the original method
            result = await original_method(user_input, max_pages)
            
            # Get the captured output
            captured_output = sys.stdout.getvalue()
            
            # Parse the logs and update progress
            if self.progress_tracker:
                self._parse_logs_and_update_progress(captured_output)
            
            return result
            
        finally:
            # Restore stdout
            sys.stdout = old_stdout
    
    def _parse_logs_and_update_progress(self, log_output: str):
        """Parse the log output and create progress steps"""
        lines = log_output.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse different log patterns
            if '[LLM_ORCHESTRATOR]' in line:
                if 'Input classified as:' in line:
                    classification = line.split('Input classified as: ')[1]
                    self.progress_tracker.add_step(
                        "LLM_ORCHESTRATOR",
                        f"Input classified as: {classification}",
                        status="completed"
                    )
                elif 'Workflow:' in line:
                    workflow = line.split('Workflow: ')[1]
                    self.progress_tracker.add_step(
                        "LLM_ORCHESTRATOR",
                        f"Selected workflow: {workflow}",
                        status="completed"
                    )
            
            elif '[PLAN_EXECUTION]' in line:
                if 'Executing' in line and 'with' in line and 'steps' in line:
                    self.progress_tracker.add_step(
                        "PLAN_EXECUTION",
                        "Starting workflow execution",
                        line.split('] ')[1] if '] ' in line else line
                    )
                elif 'Step' in line:
                    step_info = line.split('] ')[1] if '] ' in line else line
                    self.progress_tracker.add_step(
                        "PLAN_EXECUTION",
                        step_info,
                        "Executing workflow step"
                    )
            
            elif '[TOOL_CALL]' in line:
                if 'Calling' in line:
                    tool_name = self._extract_tool_name(line)
                    self.progress_tracker.add_step(
                        "TOOL_CALL",
                        f"Calling tool: {tool_name}",
                        line.split('] ')[1] if '] ' in line else line
                    )
                elif 'completed successfully' in line:
                    self.progress_tracker.complete_current_step()
            
            elif '[URL_SELECTION]' in line:
                if 'Testing' in line:
                    self.progress_tracker.add_step(
                        "URL_SELECTION",
                        "Testing job URLs",
                        line.split('] ')[1] if '] ' in line else line
                    )
                elif 'Selected URL:' in line:
                    self.progress_tracker.complete_current_step()
            
            elif '[JOB_EXTRACTION]' in line or '[CHUNKED_EXTRACTION]' in line:
                if 'Processing' in line:
                    self.progress_tracker.add_step(
                        "JOB_EXTRACTION",
                        "Extracting job information",
                        line.split('] ')[1] if '] ' in line else line
                    )
                elif 'Found' in line and 'jobs' in line:
                    jobs_found = self._extract_number_from_line(line)
                    self.progress_tracker.add_step(
                        "JOB_EXTRACTION",
                        f"Found {jobs_found} job opportunities",
                        status="completed"
                    )
    
    def _extract_tool_name(self, line: str) -> str:
        """Extract tool name from log line"""
        if 'Calling' in line:
            parts = line.split('Calling ')
            if len(parts) > 1:
                tool_part = parts[1].split(' with')[0]
                return tool_part
        return "unknown_tool"
    
    def _extract_number_from_line(self, line: str) -> str:
        """Extract number from log line"""
        numbers = re.findall(r'\d+', line)
        return numbers[0] if numbers else "0"

# Server script path
MCP_SERVER_SCRIPT = "mcp_server.py"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Job Search Orchestrator API with Progress Tracking",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        async with ProgressAwareLLMOrchestrator() as orchestrator:
            await orchestrator.initialize_mcp_client(MCP_SERVER_SCRIPT)
            return {
                "status": "healthy",
                "mcp_client": "connected",
                "available_tools": len(orchestrator.available_tools),
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def process_job_search_async(request: JobSearchRequest, request_id: str):
    """Async job search processing with progress tracking"""
    start_time = datetime.now()
    progress_tracker = ProgressTracker(request_id)
    
    logger.info(f"Starting async job search for request {request_id}: {request.query}")
    
    try:
        async with ProgressAwareLLMOrchestrator(progress_tracker) as orchestrator:
            orchestrator.model = request.model
            
            # Initialize MCP client
            if not await orchestrator.initialize_mcp_client(MCP_SERVER_SCRIPT):
                raise Exception("Failed to initialize MCP client")
            
            # Execute job search
            result: ProcessingResult = await orchestrator.llm_orchestrate(
                request.query, 
                request.max_pages
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Convert jobs to response format
            jobs = []
            if result.jobs:
                for job_data in result.jobs:
                    # Convert dictionary keys to match JobOffer model
                    job_data_copy = dict(job_data)
                    
                    # Fix for contrat_type vs contrat-type
                    if 'contrat-type' in job_data_copy and 'contrat_type' not in job_data_copy:
                        job_data_copy['contrat_type'] = job_data_copy['contrat-type']
                    
                    job = JobOffer(
                        title=job_data_copy.get('title', 'Unknown Position'),
                        company=job_data_copy.get('company', 'Unknown Company'),
                        location=job_data_copy.get('location', 'not mentioned'),
                        description=job_data_copy.get('description', 'not mentioned'),
                        salary=job_data_copy.get('salary', 'not mentioned'),
                        requirements=job_data_copy.get('requirements', 'not mentioned'),
                        contrat_type=job_data_copy.get('contrat_type', 'not mentioned'),
                        required_skill=job_data_copy.get('required_skill', 'not mentioned'),
                        post_date=job_data_copy.get('post_date', 'not mentioned'),
                        url=job_data_copy.get('url')
                    )
                    jobs.append(job)
            
            # Create response
            response = JobSearchResponse(
                success=result.success,
                jobs=jobs,
                execution_plan=result.execution_plan,
                tools_used=result.tools_used or [],
                error_message=result.error_message,
                processing_time=processing_time,
                request_id=request_id
            )
            
            # Update request status
            job_requests[request_id].status = "completed" if result.success else "failed"
            job_requests[request_id].result = response
            job_requests[request_id].progress = f"Completed - Found {len(jobs)} jobs" if result.success else f"Failed: {result.error_message}"
            
            logger.info(f"Async job search completed for request {request_id}: {len(jobs)} jobs found")
            
    except Exception as e:
        error_msg = f"Job search failed: {str(e)}"
        logger.error(f"Error in async request {request_id}: {error_msg}")
        
        # Update request status with error
        job_requests[request_id].status = "failed"
        job_requests[request_id].progress = error_msg
        
        response = JobSearchResponse(
            success=False,
            jobs=[],
            error_message=error_msg,
            processing_time=(datetime.now() - start_time).total_seconds(),
            request_id=request_id
        )
        job_requests[request_id].result = response
        
        # Update progress tracker
        if progress_tracker:
            progress_tracker.error_current_step(error_msg)

@app.post("/search")
async def search_jobs(request: JobSearchRequest, background_tasks: BackgroundTasks):
    """Initiate job search and return immediately with request ID"""
    request_id = str(uuid.uuid4())
    
    # Initialize job request status
    job_requests[request_id] = JobSearchStatus(
        request_id=request_id,
        status="processing",
        progress="Initializing job search..."
    )
    
    # Start background task
    background_tasks.add_task(process_job_search_async, request, request_id)
    
    return {"request_id": request_id, "status": "processing"}

@app.get("/search/{request_id}/progress", response_model=ProgressResponse)
async def get_search_progress(request_id: str):
    """Get detailed progress of a job search request"""
    if request_id not in progress_storage:
        raise HTTPException(status_code=404, detail="Request not found")
    
    return progress_storage[request_id]

@app.get("/search/{request_id}/status", response_model=JobSearchStatus)
async def get_search_status(request_id: str):
    """Get status of a job search request"""
    if request_id not in job_requests:
        raise HTTPException(status_code=404, detail="Request not found")
    
    return job_requests[request_id]

@app.get("/search/{request_id}/result", response_model=JobSearchResponse)
async def get_search_result(request_id: str):
    """Get result of a completed job search request"""
    if request_id not in job_requests:
        raise HTTPException(status_code=404, detail="Request not found")
    
    job_request = job_requests[request_id]
    
    if job_request.status == "processing":
        raise HTTPException(status_code=202, detail="Request still processing")
    
    if not job_request.result:
        raise HTTPException(status_code=404, detail="Result not available")
    
    return job_request.result

@app.get("/models")
async def get_available_models():
    """Get list of available LLM models"""
    return {
        "models": [
            {
                "id": "llama3-70b-8192",
                "name": "Llama 3 70B",
                "description": "Meta's Llama 3 70B model (default)",
                "context_length": 8192
            },
            {
                "id": "mixtral-8x7b-32768",
                "name": "Mixtral 8x7B",
                "description": "Mistral's Mixtral 8x7B model",
                "context_length": 32768
            },
            {
                "id": "llama2-70b-4096",
                "name": "Llama 2 70B",
                "description": "Meta's Llama 2 70B model",
                "context_length": 4096
            },
            {
                "id": "gemma-7b-it",
                "name": "Gemma 7B IT",
                "description": "Google's Gemma 7B Instruction Tuned",
                "context_length": 8192
            }
        ]
    }

@app.delete("/search/{request_id}")
async def delete_search_request(request_id: str):
    """Delete a job search request from memory"""
    if request_id not in job_requests:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Clean up both storages
    if request_id in job_requests:
        del job_requests[request_id]
    if request_id in progress_storage:
        del progress_storage[request_id]
    
    return {"message": "Request deleted successfully"}

@app.get("/search/history")
async def get_search_history():
    """Get history of all job search requests"""
    return {
        "requests": [
            {
                "request_id": req_id,
                "status": status.status,
                "progress": status.progress,
                "has_result": status.result is not None
            }
            for req_id, status in job_requests.items()
        ],
        "total": len(job_requests)
    }

# Background task to clean up old requests
async def cleanup_old_requests():
    """Clean up requests older than 1 hour"""
    while True:
        try:
            current_time = datetime.now()
            expired_requests = []
            
            for request_id, progress in progress_storage.items():
                # Check if any step is older than 1 hour
                if progress.steps:
                    oldest_step = min(progress.steps, key=lambda s: s.timestamp)
                    if (current_time - oldest_step.timestamp).total_seconds() > 3600:
                        expired_requests.append(request_id)
            
            # Remove expired requests
            for request_id in expired_requests:
                if request_id in progress_storage:
                    del progress_storage[request_id]
                if request_id in job_requests:
                    del job_requests[request_id]
                logger.info(f"Cleaned up expired request: {request_id}")
            
            # Sleep for 10 minutes before next cleanup
            await asyncio.sleep(600)
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    logger.info("Job Search Orchestrator API with Progress Tracking starting up...")
    # Start cleanup task
    asyncio.create_task(cleanup_old_requests())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Job Search Orchestrator API shutting down...")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Job Search Orchestrator API with Progress Tracking...")
    print("📋 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("📊 Progress Tracking: http://localhost:8000/search/{request_id}/progress")
    print("🌐 Frontend should connect to: http://localhost:8000")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )