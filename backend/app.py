from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import asyncio
import json
import uuid
import re
import ollama
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the orchestrator components
from hosty import LLMOrchestrator, ProcessingResult

app = FastAPI(title="JobIntel AI - Job Market Analysis")

# Configure CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active job searches and their progress
active_searches = {}
# Store WebSocket connections by client ID
active_connections = {}

class SearchRequest(BaseModel):
    query: str
    max_pages: int = 2
    model: str = "llama3.2:latest"

class ProgressUpdate(BaseModel):
    task_id: str
    step: str
    status: str
    message: str
    percentage: int = 0
    
class SearchResult(BaseModel):
    task_id: str
    status: str
    jobs: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds() * 1000
    logger.info(f"Request {request.method} {request.url.path} completed in {process_time:.2f}ms with status {response.status_code}")
    return response

@app.get("/")
async def root():
    return {"status": "online", "current_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}

@app.get("/health")
async def health_check():
    # Add any additional health checks here (database, external services, etc.)
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/search", response_model=dict)
async def search_jobs(request: SearchRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    
    logger.info(f"Starting new job search with ID {task_id}: {request.query}")
    
    # Start job search in background
    background_tasks.add_task(run_job_search, task_id, request.query, request.max_pages, request.model)
    
    return {"task_id": task_id, "status": "processing"}

@app.get("/api/search/{task_id}", response_model=SearchResult)
async def get_search_result(task_id: str):
    if task_id not in active_searches:
        raise HTTPException(status_code=404, detail="Search task not found")
        
    search_data = active_searches[task_id]
    
    if search_data["status"] == "completed":
        # Generate stats if search was successful
        stats = {}
        if search_data.get("jobs"):
            jobs = search_data["jobs"]
            stats["total_jobs"] = len(jobs)
            
            # Count job types
            job_types = {}
            for job in jobs:
                contract_type = job.get("contrat-type", "not mentioned")
                job_types[contract_type] = job_types.get(contract_type, 0) + 1
            stats["job_types"] = job_types
            
            # Count skills mentioned
            skills = {}
            for job in jobs:
                if job.get("required_skill") and job["required_skill"] != "not mentioned":
                    job_skills = [s.strip() for s in job["required_skill"].split(",")]
                    for skill in job_skills:
                        if skill:  # Ensure we don't count empty strings
                            skills[skill] = skills.get(skill, 0) + 1
            stats["top_skills"] = dict(sorted(skills.items(), key=lambda x: x[1], reverse=True)[:10])
            
            # Count locations
            locations = {}
            for job in jobs:
                location = job.get("location", "not mentioned")
                locations[location] = locations.get(location, 0) + 1
            stats["locations"] = locations
            
            search_data["stats"] = stats
    
    return SearchResult(
        task_id=task_id,
        status=search_data["status"],
        jobs=search_data.get("jobs"),
        error=search_data.get("error"),
        stats=search_data.get("stats")
    )

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    active_connections[client_id] = websocket
    logger.info(f"WebSocket connection established for client {client_id}")
    
    try:
        # Implement ping/pong mechanism to keep connection alive
        while True:
            # Set a timeout for receiving messages
            try:
                # Wait for a message with a timeout
                data = await asyncio.wait_for(websocket.receive_json(), timeout=20)
                # Process the message if needed
                if data.get('type') == 'ping':
                    await websocket.send_json({'type': 'pong', 'timestamp': datetime.utcnow().isoformat()})
            except asyncio.TimeoutError:
                # No message received within timeout, send a ping
                try:
                    await websocket.send_json({
                        'type': 'ping', 
                        'timestamp': datetime.utcnow().isoformat()
                    })
                except Exception as e:
                    # Connection is probably dead
                    logger.error(f"Failed to send ping to client {client_id}: {str(e)}")
                    break
            except Exception as e:
                logger.error(f"WebSocket error for client {client_id}: {str(e)}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        if client_id in active_connections:
            del active_connections[client_id]
            logger.info(f"Removed client {client_id} from active connections")

@app.get("/api/trends", response_model=Dict[str, Any])
async def get_market_trends():
    """Analyze trends across all completed job searches"""
    all_jobs = []
    for search_data in active_searches.values():
        if search_data.get("status") == "completed" and search_data.get("jobs"):
            all_jobs.extend(search_data["jobs"])
    
    if not all_jobs:
        return {"message": "No job data available for trend analysis"}
    
    # Analyze trends
    skills_count = {}
    titles_count = {}
    locations_count = {}
    
    for job in all_jobs:
        # Count skills
        if job.get("required_skill") and job["required_skill"] != "not mentioned":
            job_skills = [s.strip() for s in job["required_skill"].split(",")]
            for skill in job_skills:
                if skill:  # Ensure we don't count empty strings
                    skills_count[skill] = skills_count.get(skill, 0) + 1
                
        # Count job titles
        title = job.get("title", "").lower()
        if title:
            titles_count[title] = titles_count.get(title, 0) + 1
        
        # Count locations
        location = job.get("location", "not mentioned")
        locations_count[location] = locations_count.get(location, 0) + 1
    
    return {
        "top_skills": dict(sorted(skills_count.items(), key=lambda x: x[1], reverse=True)[:15]),
        "top_titles": dict(sorted(titles_count.items(), key=lambda x: x[1], reverse=True)[:10]),
        "top_locations": dict(sorted(locations_count.items(), key=lambda x: x[1], reverse=True)[:10]),
        "total_jobs_analyzed": len(all_jobs)
    }

@app.get("/api/models", response_model=List[Dict[str, str]])
async def get_available_models():
    """Return a list of available LLM models"""
    try:
        # Try to get models from Ollama
        response = ollama.list()
        models = [
            {"id": model["name"], "name": model["name"].split(":")[0].title(), "description": f"Version: {model.get('tag', 'latest')}"}
            for model in response.get("models", [])
        ]
        
        # If no models found, return default models
        if not models:
            models = [
                {"id": "llama3.2:latest", "name": "Llama 3.2", "description": "Latest version with enhanced understanding"},
                {"id": "mistral:latest", "name": "Mistral", "description": "Fast and efficient for job analyses"},
                {"id": "gemma:latest", "name": "Gemma", "description": "Compact model with strong reasoning"}
            ]
        
        return models
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        # Return default models if there's an error
        return [
            {"id": "llama3.2:latest", "name": "Llama 3.2", "description": "Latest version with enhanced understanding"},
            {"id": "mistral:latest", "name": "Mistral", "description": "Fast and efficient for job analyses"},
            {"id": "gemma:latest", "name": "Gemma", "description": "Compact model with strong reasoning"}
        ]

async def run_job_search(task_id: str, query: str, max_pages: int, model: str):
    """Run the job search process and send progress updates"""
    active_searches[task_id] = {"status": "processing"}
    
    try:
        # Custom progress logger function
        def progress_logger(step: str, status: str, message: str):
            percentage = {
                "INIT": 10,
                "URL_VALIDATION": 20,
                "COMPANY_SEARCH": 30,
                "URL_SELECTION": 40,
                "HTML_FETCHING": 60,
                "MARKDOWN_CONVERSION": 70,
                "JOB_EXTRACTION": 80,
                "COMPLETED": 100
            }.get(step, 0)
            
            update = {
                "task_id": task_id,
                "step": step,
                "status": status,
                "message": message,
                "percentage": percentage
            }
            
            # Send progress update via WebSocket if client is connected
            for connection in active_connections.values():
                asyncio.create_task(send_progress_update(connection, update))
        
        # Initialize the orchestrator
        async with LLMOrchestrator(model) as orchestrator:
            # Override the log method to use our progress_logger
            original_log = orchestrator.log
            orchestrator.log = lambda step, status, message="": (
                original_log(step, status, message),
                progress_logger(step, status, message)
            )
            
            # Initialize MCP client - hardcoded path for now
            server_script = "ahsen_server.py"  # Replace with config value in production
            if not await orchestrator.initialize_mcp_client(server_script):
                active_searches[task_id] = {
                    "status": "failed",
                    "error": "Failed to initialize job search tools"
                }
                progress_logger("ERROR", "error", "Failed to initialize job search tools")
                return
            
            # Run the job search
            result = await orchestrator.llm_orchestrate(query, max_pages)
            
            if result.success and result.jobs:
                # Process job data to ensure consistent format
                processed_jobs = []
                for job in result.jobs:
                    # Add default values for missing fields and standardize field names
                    processed_job = {
                        "title": job.get("title", "Untitled Position"),
                        "company": job.get("company", "Not specified"),
                        "location": job.get("location", "Not specified"),
                        "contrat-type": job.get("contrat-type", job.get("contract_type", "Not specified")),
                        "required_skill": job.get("required_skill", job.get("required skill", "Not specified")),
                        "description": job.get("description", "No description provided"),
                        "salary": job.get("salary", "Not specified"),
                        "experience_level": job.get("experience_level", job.get("experience level", "Not specified")),
                        "posted_date": job.get("posted_date", job.get("post_date", job.get("post date", "Not specified"))),
                        "work_time": job.get("work_time", "Not specified"),
                        "url": job.get("url", "Not specified")
                    }
                    processed_jobs.append(processed_job)
                
                active_searches[task_id] = {
                    "status": "completed",
                    "jobs": processed_jobs
                }
                progress_logger("COMPLETED", "success", f"Found {len(processed_jobs)} job listings")
            else:
                active_searches[task_id] = {
                    "status": "failed",
                    "error": result.error_message or "No job listings found"
                }
                progress_logger("COMPLETED", "error", result.error_message or "No job listings found")
                
    except Exception as e:
        logger.error(f"Job search error: {str(e)}")
        active_searches[task_id] = {
            "status": "failed",
            "error": f"Job search failed: {str(e)}"
        }
        # Send final error update
        for connection in active_connections.values():
            update = {
                "task_id": task_id,
                "step": "ERROR",
                "status": "error", 
                "message": f"Job search failed: {str(e)}",
                "percentage": 100
            }
            asyncio.create_task(send_progress_update(connection, update))

async def send_progress_update(websocket: WebSocket, update: dict):
    """Send a progress update to the client via WebSocket"""
    try:
        await websocket.send_json(update)
    except Exception as e:
        # Connection might be closed, ignore errors
        logger.debug(f"Failed to send update: {str(e)}")
        pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    reload_enabled = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    logger.info(f"Starting JobIntel AI server on {host}:{port} (reload={reload_enabled})")
    uvicorn.run("app:app", host=host, port=port, reload=reload_enabled)