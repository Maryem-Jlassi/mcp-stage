from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import asyncio
import json
import uuid
from datetime import datetime
import logging

# Import your existing orchestrator
from best2 import LLMOrchestrator, ProcessingResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Job Search Orchestrator API",
    description="AI-powered job search API with LLM orchestration",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
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
    current_step: str = ""
    step_number: int = 0
    total_steps: int = 0
    progress_percentage: int = 0
    detailed_steps: List[Dict[str, Any]] = []
    result: Optional[JobSearchResponse] = None

# In-memory storage for job requests (in production, use Redis or database)
job_requests: Dict[str, JobSearchStatus] = {}

# Server script path - adjust this to your MCP server
MCP_SERVER_SCRIPT = "ahsen_server.py"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Job Search Orchestrator API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test orchestrator initialization
        async with LLMOrchestrator() as orchestrator:
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

@app.post("/search", response_model=JobSearchResponse)
async def search_jobs(request: JobSearchRequest, background_tasks: BackgroundTasks):
    """Initiate job search and return immediately with request ID"""
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    # Initialize job request status with detailed steps
    detailed_steps = [
        {"name": "Initializing", "description": "Setting up job search parameters", "status": "pending"},
        {"name": "MCP Connection", "description": "Connecting to MCP server", "status": "pending"},
        {"name": "Input Analysis", "description": "Analyzing input and planning workflow", "status": "pending"},
        {"name": "URL Discovery", "description": "Finding job URLs", "status": "pending"},
        {"name": "Content Fetching", "description": "Crawling web pages", "status": "pending"},
        {"name": "Content Processing", "description": "Converting HTML to markdown", "status": "pending"},
        {"name": "Job Extraction", "description": "Extracting job listings with AI", "status": "pending"},
        {"name": "Results Processing", "description": "Cleaning and formatting results", "status": "pending"}
    ]
    
    job_requests[request_id] = JobSearchStatus(
        request_id=request_id,
        status="processing",
        progress="Initializing job search...",
        current_step="Initializing",
        step_number=1,
        total_steps=len(detailed_steps),
        progress_percentage=0,
        detailed_steps=detailed_steps
    )
    
    def update_progress(step_name: str, description: str, step_number: int, status: str = "processing"):
        """Update progress for current job request"""
        if request_id in job_requests:
            job_status = job_requests[request_id]
            job_status.current_step = step_name
            job_status.step_number = step_number
            job_status.progress_percentage = int((step_number / job_status.total_steps) * 100)
            job_status.progress = description
            
            # Update detailed steps
            for i, step in enumerate(job_status.detailed_steps):
                if i < step_number - 1:
                    step["status"] = "completed"
                elif i == step_number - 1:
                    step["status"] = status
                    step["description"] = description
                else:
                    step["status"] = "pending"
    
    logger.info(f"Starting job search for request {request_id}: {request.query}")
    
    try:
        # Step 1: Initialize orchestrator
        update_progress("Initializing", "Setting up LLM orchestrator...", 1)
        await asyncio.sleep(0.5)  # Small delay to show progress
        
        async with LLMOrchestrator() as orchestrator:
            orchestrator.model = request.model
            orchestrator.current_user = "API_User"
            orchestrator.current_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Step 2: Connect to MCP
            update_progress("MCP Connection", "Connecting to MCP server and loading tools...", 2)
            
            if not await orchestrator.initialize_mcp_client(MCP_SERVER_SCRIPT):
                raise HTTPException(status_code=500, detail="Failed to initialize MCP client")
            
            # Step 3: Analyze input
            update_progress("Input Analysis", f"Analyzing input: '{request.query}' and planning workflow...", 3)
            await asyncio.sleep(0.3)
            
            # Step 4-7: Execute job search with progress tracking
            result: ProcessingResult = await orchestrator.llm_orchestrate_with_progress(
                request.query, 
                request.max_pages,
                progress_callback=update_progress
            )
            
            # Step 8: Process results
            update_progress("Results Processing", "Formatting and cleaning job results...", 8)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Convert jobs to response format
            jobs = []
            if result.jobs:
                for job_data in result.jobs:
                    job = JobOffer(
                        title=job_data.get('title', 'Unknown Position'),
                        company=job_data.get('company', 'Unknown Company'),
                        location=job_data.get('location', 'not mentioned'),
                        description=job_data.get('description', 'not mentioned'),
                        salary=job_data.get('salary', 'not mentioned'),
                        requirements=job_data.get('requirements', 'not mentioned'),
                        contrat_type=job_data.get('contrat-type', 'not mentioned'),
                        required_skill=job_data.get('required_skill', 'not mentioned'),
                        post_date=job_data.get('post_date', 'not mentioned'),
                        url=job_data.get('url')
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
            
            # Final update
            if result.success:
                update_progress("Completed", f"Successfully found {len(jobs)} job opportunities!", 8, "completed")
                job_requests[request_id].status = "completed"
                job_requests[request_id].progress_percentage = 100
            else:
                update_progress("Failed", f"Search failed: {result.error_message}", 8, "failed")
                job_requests[request_id].status = "failed"
            
            job_requests[request_id].result = response
            
            logger.info(f"Job search completed for request {request_id}: {len(jobs)} jobs found")
            return response
            
    except Exception as e:
        error_msg = f"Job search failed: {str(e)}"
        logger.error(f"Error in request {request_id}: {error_msg}")
        
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
        
        return response

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
    
    del job_requests[request_id]
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

# Background task to clean up old requests (optional)
@app.on_event("startup")
async def startup_event():
    logger.info("Job Search Orchestrator API starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Job Search Orchestrator API shutting down...")
    # Clean up any resources if needed

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Job Search Orchestrator API...")
    print("📋 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("🌐 Frontend should connect to: http://localhost:8000")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )