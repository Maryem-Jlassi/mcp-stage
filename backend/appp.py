from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import uvicorn
import os
from contextlib import asynccontextmanager

# Import the LLMOrchestrator from your existing code
from hosty import LLMOrchestrator, ProcessingResult

# Global orchestrator instance
orchestrator_instance = None
# Dictionary to store job search results by ID
search_results = {}
# Counter for generating search IDs
search_counter = 0

class UserInput(BaseModel):
    query: str
    maxPages: int = 1
    model: str = "llama3.2:latest"

class SearchResponse(BaseModel):
    search_id: str
    status: str
    message: str

class JobResult(BaseModel):
    title: str
    company: str
    location: Optional[str] = None
    description: Optional[str] = None
    salary: Optional[str] = None
    requirements: Optional[str] = None
    contrat_type: Optional[str] = None
    required_skill: Optional[str] = None
    post_date: Optional[str] = None
    url: Optional[str] = None
    field: Optional[str] = None

class SearchResult(BaseModel):
    search_id: str
    status: str
    query: str
    jobs: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    execution_plan: Optional[str] = None
    tools_used: Optional[List[str]] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize on startup
    global orchestrator_instance
    server_script = os.environ.get("MCP_SERVER", "ahsen_server.py")
    orchestrator_instance = LLMOrchestrator()
    initialized = await orchestrator_instance.initialize_mcp_client(server_script)
    if not initialized:
        print("⚠️ Warning: Failed to initialize MCP client. API may not function correctly.")
    yield
    # Cleanup on shutdown
    if orchestrator_instance:
        await orchestrator_instance.cleanup()

app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow requests from your React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # For production, replace with specific origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

async def get_orchestrator():
    if orchestrator_instance is None:
        raise HTTPException(status_code=503, detail="LLM Orchestrator not initialized")
    return orchestrator_instance

@app.get("/")
async def root():
    return {"message": "Job Search Orchestrator API is running"}

@app.get("/api/status")
async def status():
    """Simple status endpoint that doesn't depend on the orchestrator"""
    return {
        "status": "ready",
        "api_version": "1.0"
    }

@app.get("/api/debug")
async def debug_info():
    """Endpoint to check response formatting and structure"""
    sample_response = {
        "success": True,
        "jobs": [
            {
                "title": "Test Job",
                "company": "Debug Company",
                "location": "Test Location",
                "description": "This is a test job to verify response format",
                "salary": "$100k-$150k",
                "requirements": "Testing skills",
                "contrat-type": "Full-time", 
                "required_skill": "Debugging",
                "post_date": "Today"
            }
        ],
        "error_message": None,
        "execution_plan": "Debug execution plan",
        "tools_used": ["debug_tool"]
    }
    return JSONResponse(content=sample_response)
@app.post("/api/search")
async def search_jobs(input_data: UserInput):
    try:
        orchestrator = await get_orchestrator()
        
        # Print what we're receiving from the frontend
        print(f"Received search request: query={input_data.query}, maxPages={input_data.maxPages}")
        
        # Run the search
        result = await orchestrator.llm_orchestrate(input_data.query, input_data.maxPages)
        
        # Debug log the raw result
        print(f"Raw orchestrator result: success={result.success}, jobs_count={len(result.jobs) if result.jobs else 0}")
        
        # Structure response EXACTLY as frontend expects
        response = {
            "success": True,  # IMPORTANT: Set this explicitly to True on success
            "jobs": result.jobs if result.success and result.jobs else [],
            "error_message": None,  # Set explicitly to None on success
            "execution_plan": result.execution_plan if result.execution_plan else "Job search completed",
            "tools_used": result.tools_used if result.tools_used else []
        }
        
        # Print what we're sending back
        print(f"Sending response to frontend: success=True, jobs_count={len(response['jobs'])}")
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Error in search endpoint: {str(e)}")
        return JSONResponse(
            status_code=200,  # Return 200 even for errors to ensure frontend gets the response
            content={
                "success": False,
                "error_message": f"Search failed: {str(e)}",
                "jobs": [],
                "execution_plan": None,
                "tools_used": []
            }
        )
    

if __name__ == "__main__":
    # Get server script from environment or default
    server_script = os.environ.get("MCP_SERVER", "ahsen_server.py")
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Starting Job Search Orchestrator API with server script: {server_script}")
    print(f"📡 Server will run on port {port}")
    
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)