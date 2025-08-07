import asyncio
import json
import sys
import re
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from groq import AsyncGroq
from datetime import datetime
from client import MCPClient

print("Using Groq for inference - no local GPU required!")
try:
    from config import GROQ_API_KEY
except ImportError:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

@dataclass
class ProcessingResult:
    success: bool
    jobs: List[Dict[str, Any]] = None
    error_message: str = None
    execution_plan: str = None
    tools_used: List[str] = None

def classify_input(user_input: str) -> str:
    """More robust input classification"""
    ui = user_input.strip().lower()
    # Check for URL patterns more thoroughly
    url_patterns = ['http://', 'https://', 'www.', '.com/', '.org/', '.net/', '/jobs', '/careers']
    if any(pattern in ui for pattern in url_patterns):
        return "url"
    return "company_name"

class LLMOrchestrator:
    def __init__(self):
        self.model = "llama3-70b-8192"  # Correct format without groq/ prefix
        self.mcp_client: Optional[MCPClient] = None
        self.available_tools = []
        self.groq_client = None
        self.save_content = False  # Enable content saving for debugging
        self.current_time = "2025-08-06 18:21:24"  # Updated timestamp
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the Groq client"""
        try:
            self.log("MODEL_INIT", "start", f"Initializing Groq client for model: {self.model}")
            
            # Use API key from config file or environment variable
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found in config.py or environment variables")
            
            # Initialize Groq client
            self.groq_client = AsyncGroq(api_key=GROQ_API_KEY)
            
            self.log("MODEL_INIT", "success", f"Groq client initialized for {self.model}")
            
        except Exception as e:
            self.log("MODEL_INIT", "error", f"Failed to initialize Groq client: {str(e)}")
            raise e

    async def _generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response using Groq API"""
        try:
            # Format as chat message
            messages = [{"role": "user", "content": prompt}]
            
            # Call Groq API
            response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
            )
            
            # Extract the generated text
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            
            return ""
            
        except Exception as e:
            self.log("GENERATION", "error", f"Response generation failed: {str(e)}")
            return ""
        
    def log(self, step: str, status: str, message: str = ""):
        if status == "start":
            icon = "🔄"
            color = "\033[94m"
        elif status == "success":
            icon = "✅"
            color = "\033[92m"
        elif status == "error":
            icon = "❌"
            color = "\033[91m"
        elif status == "info":
            icon = "ℹ️"
            color = "\033[96m"
        else:
            icon = "📝"
            color = "\033[0m"
        reset_color = "\033[0m"
        print(f"{color}{icon} [{step}] {message}{reset_color}")

    async def initialize_mcp_client(self, server_script: str) -> bool:
        try:
            self.log("MCP_INIT", "start", f"Connecting to MCP server: {server_script}")
            self.mcp_client = MCPClient(server_script)
            await self.mcp_client.start()
            self.available_tools = self.mcp_client.list_tools()
            self.log("MCP_INIT", "success", f"Connected with {len(self.available_tools)} tools available")
            return True
        except Exception as e:
            self.log("MCP_INIT", "error", f"Failed to connect: {str(e)}")
            return False

    async def call_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.mcp_client:
            return {"success": False, "error": "MCP client not initialized"}
        try:
            self.log("TOOL_CALL", "start", f"Calling {tool_name} with params: {params}")
            result = await self.mcp_client.call_tool(tool_name, params)
            if result.get("success"):
                self.log("TOOL_CALL", "success", f"{tool_name} completed successfully")
            else:
                self.log("TOOL_CALL", "error", f"{tool_name} failed: {result.get('error', 'Unknown error')}")
            return result
        except Exception as e:
            self.log("TOOL_CALL", "error", f"Exception calling {tool_name}: {str(e)}")
            return {"success": False, "error": str(e)}

    async def find_best_url(self, job_urls: List[str], tools_used: List[str]) -> Optional[str]:
        """Find the best job URL from a list of candidates"""
        self.log("URL_SELECTION", "start", f"Testing {len(job_urls)} candidate URLs")
        
        for i, url in enumerate(job_urls):
            self.log("URL_SELECTION", "info", f"Testing URL {i+1}/{len(job_urls)}: {url}")
            result = await self.call_mcp_tool("precheck_job_offer_url", {"url": url})
            tools_used.append("precheck_job_offer_url")
            
            if result.get("success") and result.get("result", {}).get("is_offer_page"):
                self.log("URL_SELECTION", "success", f"Found valid job URL: {url}")
                return url
            else:
                reason = result.get("result", {}).get("reason", "Unknown reason")
                self.log("URL_SELECTION", "info", f"URL rejected: {reason}")
        
        # If no URL passes validation, return the first one as fallback
        fallback_url = job_urls[0] if job_urls else None
        if fallback_url:
            self.log("URL_SELECTION", "info", f"No URLs passed validation, using fallback: {fallback_url}")
        return fallback_url

    async def llm_orchestrate(self, user_input: str, max_pages: int = 1) -> ProcessingResult:
        self.log("LLM_ORCHESTRATOR", "start", f"LLM analyzing input: '{user_input}'")
        input_type = classify_input(user_input)
        self.log("LLM_ORCHESTRATOR", "info", f"Input classified as: {input_type}")
        self.log("LLM_ORCHESTRATOR", "info", f"Pagination: Will fetch up to {max_pages} pages")
        
        # Simplified and more structured prompt with explicit JSON template
        orchestration_prompt = f"""You are a job search orchestrator. Analyze this input: "{user_input}"

INPUT ANALYSIS:
- Input type: {input_type}
- Rule: If contains URL patterns → url_workflow, else → company_workflow
- Pagination: Will fetch up to {max_pages} pages

RESPOND WITH EXACTLY THIS JSON (replace values in brackets):

{{
    "workflow_type": "{('url_workflow' if input_type == 'url' else 'company_workflow')}",
    "input_type": "{input_type}",
    "extracted_value": "{user_input}",
    "execution_plan": [
        {{"tool": "{'precheck_job_offer_url' if input_type == 'url' else 'get_official_website_and_generate_job_urls'}", "params": {{"{'url' if input_type == 'url' else 'company_name'}": "{user_input}"}}, "description": "{'validate URL' if input_type == 'url' else 'find company jobs page'}"}},
        {{"tool": "{'fetch_url_html_with_pagination' if input_type == 'url' else 'precheck_job_offer_url'}", "params": {{"{'url' if input_type == 'url' else ''}": "{'{}' if input_type == 'url' else ''}", "max_pages": {max_pages}}}, "description": "{'get HTML content from up to " + str(max_pages) + " pages' if input_type == 'url' else 'validate found URL'}"}},
        {{"tool": "{'html_to_markdown' if input_type == 'url' else 'fetch_url_html_with_pagination'}", "params": {{"max_pages": {max_pages}}}, "description": "{'convert to markdown' if input_type == 'url' else 'get HTML content from up to " + str(max_pages) + " pages'}"}}{(',' if input_type != 'url' else '')}
        {('{{"tool": "html_to_markdown", "params": {{}}, "description": "convert to markdown"}}' if input_type != 'url' else '')}
    ],
    "reasoning": "Selected {('URL' if input_type == 'url' else 'company')} workflow for {input_type} input with pagination set to {max_pages} pages"
}}

Return ONLY the JSON object above with correct values filled in.
"""
        try:
            llm_decision = await self._generate_response(orchestration_prompt, max_tokens=800)
            self.log("LLM_ORCHESTRATOR", "info", f"LLM decision received: {len(llm_decision)} characters")
            
            # Save raw decision for debugging
            if self.save_content:
                with open("llm_raw_decision.txt", "w", encoding="utf-8") as f:
                    f.write(llm_decision)
            
            # Try to parse JSON with multiple strategies
            decision_data = self._parse_llm_json(llm_decision)
            
            if decision_data is None:
                # Fallback: create decision manually based on input classification
                self.log("LLM_ORCHESTRATOR", "info", "JSON parsing failed, creating fallback decision")
                decision_data = self._create_fallback_decision(user_input, input_type, max_pages)
            
            # Validate and correct the decision
            decision_data = self._validate_decision(decision_data, input_type, user_input, max_pages)
            
            self.log("LLM_ORCHESTRATOR", "success", f"Workflow: {decision_data.get('workflow_type')}")
            self.log("LLM_ORCHESTRATOR", "info", f"Reasoning: {decision_data.get('reasoning')}")
            
            return await self.execute_plan(decision_data)
            
        except Exception as e:
            self.log("LLM_ORCHESTRATOR", "error", f"LLM orchestration failed: {e}")
            # Create fallback decision with proper error handling
            decision_data = self._create_fallback_decision(user_input, input_type, max_pages)
            if isinstance(decision_data, dict):
                return await self.execute_plan(decision_data)
            else:
                return ProcessingResult(
                    success=False,
                    error_message=f"Failed to create execution plan: {e}",
                    execution_plan="Fallback execution failed",
                    tools_used=[]
                )

    def _parse_llm_json(self, llm_decision: str) -> Optional[Dict[str, Any]]:
        """Try multiple strategies to parse LLM JSON output"""
        
        # Strategy 1: Direct parsing
        try:
            return json.loads(llm_decision)
        except Exception:
            pass
        
        # Strategy 2: Use aggressive JSON fixing
        try:
            fixed_json = self._fix_json_aggressively(llm_decision)
            return json.loads(fixed_json)
        except Exception:
            pass
        
        # Strategy 3: Extract JSON with regex
        try:
            # Look for JSON object patterns
            patterns = [
                r'\{[^{}]*"workflow_type"[^{}]*\}',  # Simple single-line JSON
                r'\{.*?"workflow_type".*?\}',        # Multi-line JSON
                r'\{.*?\}',                          # Any JSON object
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, llm_decision, re.DOTALL)
                for match in matches:
                    try:
                        fixed_match = self._fix_json_aggressively(match)
                        parsed = json.loads(fixed_match)
                        if 'workflow_type' in parsed:
                            return parsed
                    except Exception:
                        continue
        except Exception:
            pass
        
        return None

    def _create_fallback_decision(self, user_input: str, input_type: str, max_pages: int = 1) -> Dict[str, Any]:
        """Create a fallback decision when LLM parsing fails"""
        if input_type == "url":
            execution_plan = [
                {"tool": "precheck_job_offer_url", "params": {"url": user_input}, "description": "validate URL contains jobs"},
                {"tool": "fetch_url_html_with_pagination", "params": {"url": user_input, "max_pages": max_pages}, "description": f"get HTML content from up to {max_pages} pages"},
                {"tool": "html_to_markdown", "params": {}, "description": "convert HTML to markdown"}
            ]
            workflow_type = "url_workflow"
        else:
            execution_plan = [
                {"tool": "get_official_website_and_generate_job_urls", "params": {"company_name": user_input}, "description": "find company website and job URLs"},
                {"tool": "precheck_job_offer_url", "params": {}, "description": "validate best URL contains jobs"},
                {"tool": "fetch_url_html_with_pagination", "params": {"max_pages": max_pages}, "description": f"get HTML content from up to {max_pages} pages"},
                {"tool": "html_to_markdown", "params": {}, "description": "convert HTML to markdown"}
            ]
            workflow_type = "company_workflow"
        
        return {
            "workflow_type": workflow_type,
            "input_type": input_type,
            "extracted_value": user_input,
            "execution_plan": execution_plan,
            "reasoning": f"Fallback decision: detected {input_type} input with pagination set to {max_pages} pages"
        }

    def _validate_decision(self, decision_data: Dict[str, Any], input_type: str, user_input: str, max_pages: int = 1) -> Dict[str, Any]:
        """Validate and correct the LLM decision"""
        
        # Ensure correct input_type and workflow_type
        if decision_data.get("input_type") != input_type:
            self.log("LLM_ORCHESTRATOR", "info", f"Correcting input_type from {decision_data.get('input_type')} to {input_type}")
            decision_data["input_type"] = input_type
        
        expected_workflow = "url_workflow" if input_type == "url" else "company_workflow"
        if decision_data.get("workflow_type") != expected_workflow:
            self.log("LLM_ORCHESTRATOR", "info", f"Correcting workflow_type to {expected_workflow}")
            decision_data["workflow_type"] = expected_workflow
        
        # Ensure extracted_value is set
        if not decision_data.get("extracted_value"):
            decision_data["extracted_value"] = user_input
        
        # Create proper execution plan based on workflow type with pagination
        if expected_workflow == "url_workflow":
            decision_data["execution_plan"] = [
                {"tool": "precheck_job_offer_url", "params": {"url": user_input}, "description": "validate URL contains jobs"},
                {"tool": "fetch_url_html_with_pagination", "params": {"url": user_input, "max_pages": max_pages}, "description": f"get HTML content from up to {max_pages} pages"},
                {"tool": "html_to_markdown", "params": {}, "description": "convert HTML to markdown"}
            ]
        else:
            decision_data["execution_plan"] = [
                {"tool": "get_official_website_and_generate_job_urls", "params": {"company_name": user_input}, "description": "find company website and job URLs"},
                {"tool": "precheck_job_offer_url", "params": {}, "description": "validate best URL contains jobs"},
                {"tool": "fetch_url_html_with_pagination", "params": {"max_pages": max_pages}, "description": f"get HTML content from up to {max_pages} pages"},
                {"tool": "html_to_markdown", "params": {}, "description": "convert HTML to markdown"}
            ]
        
        return decision_data

    async def execute_plan(self, decision_data: Dict[str, Any]) -> ProcessingResult:
        execution_plan = decision_data.get("execution_plan", [])
        workflow_type = decision_data.get("workflow_type")
        extracted_value = decision_data.get("extracted_value")
        tools_used = []
        
        self.log("PLAN_EXECUTION", "start", f"Executing {workflow_type} with {len(execution_plan)} steps")
        
        # Initialize workflow-specific variables
        if workflow_type == "url_workflow":
            source_url = extracted_value
            self.log("PLAN_EXECUTION", "info", f"URL workflow starting with: {source_url}")
        else:
            source_url = ""
            self.log("PLAN_EXECUTION", "info", f"Company workflow starting with: {extracted_value}")
        
        try:
            markdown_content = ""
            html_content = ""
            
            # Execute each step in the plan
            for step_num, step in enumerate(execution_plan, 1):
                tool_name = step.get("tool")
                base_params = step.get("params", {})
                description = step.get("description", "")
                
                self.log("PLAN_EXECUTION", "info", f"Step {step_num}/{len(execution_plan)}: {tool_name} - {description}")
                
                # Prepare parameters for each tool based on current state
                params = self._prepare_tool_params(tool_name, base_params, extracted_value, source_url, html_content)
                
                if params is None:
                    return ProcessingResult(
                        success=False,
                        error_message=f"Step {step_num} ({tool_name}): Missing required parameters",
                        execution_plan=str(decision_data.get("reasoning")),
                        tools_used=tools_used
                    )
                
                # Execute the tool
                result = await self.call_mcp_tool(tool_name, params)
                tools_used.append(tool_name)
                
                if not result.get("success"):
                    error_msg = result.get("error", "Unknown error")
                    return ProcessingResult(
                        success=False,
                        error_message=f"Step {step_num} ({tool_name}) failed: {error_msg}",
                        execution_plan=str(decision_data.get("reasoning")),
                        tools_used=tools_used
                    )
                
                # Process results and update state for next step
                processing_result = await self._process_tool_result(tool_name, result, tools_used)
                
                if processing_result.get("error"):
                    return ProcessingResult(
                        success=False,
                        error_message=f"Step {step_num} ({tool_name}): {processing_result['error']}",
                        execution_plan=str(decision_data.get("reasoning")),
                        tools_used=tools_used
                    )
                
                # Update state variables based on tool results
                if processing_result.get("source_url"):
                    source_url = processing_result["source_url"]
                    self.log("PLAN_EXECUTION", "info", f"Updated source_url: {source_url}")
                
                if processing_result.get("html_content"):
                    html_content = processing_result["html_content"]
                    self.log("PLAN_EXECUTION", "info", f"Received HTML content: {len(html_content)} characters")
                
                if processing_result.get("markdown_content"):
                    markdown_content = processing_result["markdown_content"]
                    self.log("PLAN_EXECUTION", "info", f"Received markdown content: {len(markdown_content)} characters")
                    
                    # Save markdown content for debugging
                    if self.save_content:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        company_name = extracted_value.replace(" ", "_").replace("/", "_")
                        file_path = f"extracted_content_{company_name}_{timestamp}.md"
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(f"# Content from: {source_url}\n\n")
                            f.write(markdown_content)
                        self.log("CONTENT_SAVING", "success", f"Saved content to {file_path}")
            
            # Final step: Extract jobs from markdown
            if markdown_content:
                # Check if content is large and should use chunking immediately
                if len(markdown_content) > 5000:  # If content is large, go directly to chunked extraction
                    self.log("JOB_EXTRACTION", "info", f"Content size {len(markdown_content)} characters, using chunked extraction")
                    jobs = await self._extract_jobs_from_chunks(markdown_content, source_url)
                else:
                    jobs = await self.llm_extract_jobs(markdown_content, source_url)
                
                if not jobs:
                    # Try aggressive extraction if standard extraction fails
                    self.log("JOB_EXTRACTION", "info", "No jobs found, trying aggressive extraction")
                    jobs = await self._aggressive_job_extraction(markdown_content, source_url)
                    
                if not jobs:
                    return ProcessingResult(
                        success=False,
                        error_message="No job offers could be extracted from the content",
                        execution_plan=str(decision_data.get("reasoning")),
                        tools_used=tools_used
                    )
                return ProcessingResult(
                    success=True,
                    jobs=jobs,
                    execution_plan=str(decision_data.get("reasoning")),
                    tools_used=tools_used
                )
            else:
                return ProcessingResult(
                    success=False,
                    error_message="No markdown content was generated for job extraction",
                    execution_plan=str(decision_data.get("reasoning")),
                    tools_used=tools_used
                )
                
        except Exception as e:
            self.log("PLAN_EXECUTION", "error", f"Plan execution failed: {e}")
            return ProcessingResult(
                success=False,
                error_message=f"Plan execution failed: {e}",
                execution_plan=str(decision_data.get("reasoning")),
                tools_used=tools_used
            )

    def _prepare_tool_params(self, tool_name: str, base_params: Dict[str, Any], 
                           extracted_value: str, source_url: str, html_content: str) -> Optional[Dict[str, Any]]:
        """Prepare parameters for each tool based on current workflow state"""
        
        if tool_name == "get_official_website_and_generate_job_urls":
            return {"company_name": extracted_value}
        
        elif tool_name == "precheck_job_offer_url":
            if source_url:
                return {"url": source_url}
            elif base_params.get("url"):
                return {"url": base_params["url"]}
            else:
                self.log("PARAM_PREP", "error", f"No URL available for {tool_name}")
                return None
        
        elif tool_name == "fetch_url_html_with_pagination":
            # Ensure max_pages parameter is included
            max_pages = base_params.get("max_pages", 1)
            
            if source_url:
                return {"url": source_url, "max_pages": max_pages}
            elif base_params.get("url"):
                return {"url": base_params["url"], "max_pages": max_pages}
            else:
                self.log("PARAM_PREP", "error", f"No URL available for {tool_name}")
                return None
        
        elif tool_name == "html_to_markdown":
            if html_content:
                return {"html_content": html_content}
            else:
                self.log("PARAM_PREP", "error", f"No HTML content available for {tool_name}")
                return None
        
        else:
            # For unknown tools, return base params
            return base_params

    async def _process_tool_result(self, tool_name: str, result: Dict[str, Any], tools_used: List[str]) -> Dict[str, Any]:
        """Process tool results and extract relevant data for workflow state"""
        
        if tool_name == "get_official_website_and_generate_job_urls":
            urls_data = result.get("result", {})
            if not urls_data.get("success"):
                return {"error": f"Company search failed: {urls_data.get('reason', 'Unknown reason')}"}
            
            job_urls = urls_data.get("possible_job_urls", [])
            if not job_urls:
                return {"error": "No job URLs found for company"}
            
            self.log("RESULT_PROCESSING", "info", f"Found {len(job_urls)} candidate URLs")
            
            # Find the best URL from candidates - now properly awaited
            best_url = await self.find_best_url(job_urls, tools_used)
            
            if not best_url:
                return {"error": "No valid job URLs found after validation"}
            
            return {"source_url": best_url}
        
        elif tool_name == "precheck_job_offer_url":
            precheck_result = result.get("result", {})
            if not precheck_result.get("success") or not precheck_result.get("is_offer_page"):
                return {"error": f"URL validation failed: {precheck_result.get('reason', 'URL does not contain job offers')}"}
            return {"validated": True}
        
        elif tool_name == "fetch_url_html_with_pagination":
            pages = result.get("result", {}).get("pages", [])
            if not pages or pages[0].get("status") != "success":
                return {"error": "Failed to fetch page content"}
            
            # Combine HTML content from all pages
            combined_html = ""
            for page in pages:
                if page.get("status") == "success":
                    combined_html += page.get("html", "") + "\n<!-- Next Page -->\n"
            
            if not combined_html.strip():
                return {"error": "Fetched pages have no content"}
            
            self.log("RESULT_PROCESSING", "info", f"Successfully fetched {len(pages)} pages with {len(combined_html)} characters")
            return {"html_content": combined_html}
        
        elif tool_name == "html_to_markdown":
            markdown_result = result.get("result", {})
            if not markdown_result.get("success"):
                return {"error": f"HTML to markdown conversion failed: {markdown_result.get('error', 'Unknown error')}"}
            
            markdown_content = markdown_result.get("markdown", "")
            if not markdown_content.strip():
                return {"error": "Markdown conversion produced no content"}
            
            return {"markdown_content": markdown_content}
        
        else:
            return {"success": True}

    def _chunk_markdown(self, markdown_content: str, max_chars: int = 4000) -> List[str]:
        """Split markdown content into smaller chunks for processing"""
        if len(markdown_content) <= max_chars:
            return [markdown_content]
        
        # Split by sections (headers)
        section_pattern = r'(?=(?:^|\n)#+\s+)'
        sections = re.split(section_pattern, markdown_content)
        
        chunks = []
        current_chunk = ""
        
        for section in sections:
            if len(current_chunk) + len(section) <= max_chars:
                current_chunk += section
            else:
                # If current section is too big alone, we need to break it further
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                if len(section) > max_chars:
                    # Break section by paragraphs
                    paragraphs = section.split("\n\n")
                    current_paragraph = ""
                    
                    for paragraph in paragraphs:
                        if len(current_paragraph) + len(paragraph) + 2 <= max_chars:
                            current_paragraph += paragraph + "\n\n"
                        else:
                            if current_paragraph:
                                chunks.append(current_paragraph)
                            
                            # If paragraph is still too large, just add it as a chunk
                            if len(paragraph) > max_chars:
                                # Split into smaller chunks as needed
                                for i in range(0, len(paragraph), max_chars):
                                    chunks.append(paragraph[i:i + max_chars])
                            else:
                                current_paragraph = paragraph + "\n\n"
                    
                    if current_paragraph:
                        current_chunk = current_paragraph
                else:
                    current_chunk = section
        
        if current_chunk:
            chunks.append(current_chunk)
            
        self.log("CONTENT_CHUNKING", "info", f"Split content into {len(chunks)} chunks for processing")
        return chunks

    async def _extract_jobs_from_chunks(self, markdown_content: str, source_url: str) -> List[Dict[str, Any]]:
        """Extract jobs from markdown content by splitting it into manageable chunks"""
        self.log("CHUNKED_EXTRACTION", "start", f"Processing large content ({len(markdown_content)} chars) in chunks")
        
        # Create chunks of manageable size (3000 chars each)
        chunks = self._chunk_markdown(markdown_content, 3000)
        all_jobs = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            self.log("CHUNKED_EXTRACTION", "info", f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            
            # Simple extraction prompt for each chunk
            extraction_prompt = f"""Extract job listings from this content chunk ({i+1}/{len(chunks)}).
Return ONLY a JSON array of jobs. Source: {source_url}

Format:
[
  {{
    "title": "Job Title",
    "company": "{source_url.split('/')[2].replace('www.', '').split('.')[0].capitalize() if source_url else 'Unknown'}",
    "location": "Location if mentioned or 'not mentioned'",
    "description": "Brief description if available or 'not mentioned'"
  }}
]

CONTENT CHUNK:
{chunk[:2500]}

JSON OUTPUT:"""
            
            try:
                # Generate response for this chunk
                chunk_output = await self._generate_response(extraction_prompt, max_tokens=1024)
                
                # Try direct backtick extraction first
                chunk_jobs = []
                if "```" in chunk_output:
                    try:
                        json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', chunk_output, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            chunk_jobs = json.loads(json_str)
                    except Exception:
                        pass
                
                # If backtick extraction failed, try other methods
                if not chunk_jobs:
                    try:
                        # Find any JSON array in the text
                        json_match = re.search(r'\[\s*\{.*\}\s*\]', chunk_output, re.DOTALL)
                        if json_match:
                            json_str = self._fix_json_aggressively(json_match.group(0))
                            chunk_jobs = json.loads(json_str)
                    except Exception:
                        pass
                
                # Add jobs from this chunk to our collection
                if chunk_jobs:
                    self.log("CHUNKED_EXTRACTION", "success", f"Found {len(chunk_jobs)} jobs in chunk {i+1}")
                    
                    # Check for duplicates before adding
                    for job in chunk_jobs:
                        if job.get('title') and not any(j.get('title') == job.get('title') for j in all_jobs):
                            # Add source URL to job
                            job['url'] = source_url
                            all_jobs.append(job)
            except Exception as e:
                self.log("CHUNKED_EXTRACTION", "error", f"Error processing chunk {i+1}: {e}")
        
        # Clean all jobs
        cleaned_jobs = []
        for job in all_jobs:
            clean_job = self._simplified_clean_job(job, source_url)
            cleaned_jobs.append(clean_job)
        
        self.log("CHUNKED_EXTRACTION", "success", f"Total jobs extracted from chunks: {len(cleaned_jobs)}")
        return cleaned_jobs

    async def llm_extract_jobs(self, markdown_content: str, source_url: str = "") -> List[Dict[str, Any]]:
        self.log("JOB_EXTRACTION", "start", f"Extracting jobs from {len(markdown_content)} characters")
        
        # For large content, redirect immediately to chunked extraction
        if len(markdown_content) > 5000:
            self.log("JOB_EXTRACTION", "info", f"Content too large ({len(markdown_content)} chars), switching to chunked extraction")
            return await self._extract_jobs_from_chunks(markdown_content, source_url)
            
        # Check if it's a specific site and use a more targeted prompt
        is_odoo = "odoo" in source_url.lower() or "odoo" in markdown_content.lower()
        is_keejob = "keejob" in source_url.lower()
        
        try:
            # Enhanced extraction prompt with site-specific guidance if needed
            extraction_prompt = f"""Extract ALL job listings from this markdown content. Return ONLY a JSON array.

CONTEXT:
- Source: {source_url if source_url else "a job website"}
- Content format: Markdown
{("- Company: Odoo, which uses a custom job listing format" if is_odoo else "")}
{("- Job board: KeejobCom, a Tunisian job board" if is_keejob else "")}

JOB IDENTIFICATION IN MARKDOWN:
- Look for job titles in headings (# heading) or emphasized text (**bold** or *italic*)
- Job listings often follow patterns with sections for description, requirements, etc.
- Jobs might be separated by horizontal rules (---) or headers
- Look for key phrases like "Job Title", "Position", "Role", "Opening", "We're hiring", "Apply now"
- Job details are often in lists (- item or * item) or paragraphs
{"- For Odoo specifically, look for patterns like job position followed by location and job type" if is_odoo else ""}

OUTPUT FORMAT:
[
  {{
    "title": "Exact job title",
    "company": "Company name",
    "location": "Work location or 'not mentioned'",
    "description": "Brief description of job duties",
    "salary": "Salary information or 'not mentioned'",
    "requirements": "Key qualifications or 'not mentioned'",
    "contrat-type": "Full-time, Part-time, etc. or 'not mentioned'",
    "required_skill": "Key skills needed or 'not mentioned'",
    "post_date": "When job was posted or 'not mentioned'"
  }}
]

IMPORTANT RULES:
- Use "not mentioned" for missing fields, NOT null or N/A
- Extract ALL distinct job positions - do NOT limit the number
- If the same job appears multiple times, include it only once
- For each separate job position, create a separate JSON object
- BE GENEROUS in what you consider a job - if it's mentioned at all, include it

MARKDOWN CONTENT:
{markdown_content[:4500]}  # Limit content to avoid token limits

JSON OUTPUT:"""

            # Increased token limit for better extraction
            llm_output = await self._generate_response(extraction_prompt, max_tokens=2048)
            self.log("JOB_EXTRACTION", "info", f"LLM extraction output: {len(llm_output)} characters")
            
            # Display raw output for debugging but continue processing
            print("\n" + "="*80)
            print("RAW LLM EXTRACTION OUTPUT:")
            print("="*80)
            print(llm_output)
            print("="*80)
            
            # Try direct backtick extraction first (most reliable for code blocks)
            if '```' in llm_output:
                try:
                    # Extract content between backticks
                    json_text = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', llm_output, re.DOTALL)
                    if json_text:
                        jobs = json.loads(json_text.group(1))
                        self.log("JOB_EXTRACTION", "success", f"Direct backtick extraction found {len(jobs)} jobs")
                        # Apply simplified job data cleaning
                        simplified_jobs = []
                        for job in jobs:
                            clean_job = self._simplified_clean_job(job, source_url)
                            simplified_jobs.append(clean_job)
                        return simplified_jobs
                except Exception as e:
                    self.log("JOB_EXTRACTION", "error", f"Direct backtick extraction failed: {e}")
            
            # If direct backtick extraction fails, try the more generic extraction
            try:
                jobs = self._extract_jobs_json(llm_output)
                self.log("JOB_EXTRACTION", "info", f"JSON extraction found {len(jobs)} jobs")
                
                if not jobs:
                    self.log("JOB_EXTRACTION", "info", "Still no jobs found, trying fallback extraction")
                    jobs = self._extract_jobs_fallback(llm_output)
                    self.log("JOB_EXTRACTION", "success", f"Extracted {len(jobs)} job offers")
                
                # Apply simplified job data cleaning
                simplified_jobs = []
                for job in jobs:
                    clean_job = self._simplified_clean_job(job, source_url)
                    simplified_jobs.append(clean_job)
                
                return simplified_jobs
            except Exception as e:
                self.log("JSON_PARSING", "error", f"JSON extraction failed: {e}")
                return []
            
        except Exception as e:
            self.log("JOB_EXTRACTION", "error", f"Job extraction failed: {e}")
            return []

    async def _aggressive_job_extraction(self, markdown_content: str, source_url: str) -> List[Dict[str, Any]]:
        """Aggressively extract anything that looks like a job from the content"""
        self.log("JOB_EXTRACTION", "info", "Using aggressive job extraction method")
        
        # If content is too large, use chunking for aggressive extraction too
        if len(markdown_content) > 5000:
            self.log("JOB_EXTRACTION", "info", "Content too large for aggressive extraction, using chunks")
            return await self._aggressive_extract_from_chunks(markdown_content, source_url)
        
        # Simplified prompt that focuses on just finding job titles
        aggressive_prompt = f"""Look through this content and extract ANY possible job titles mentioned.
Be extremely generous - include anything that might be a job position.
Return ONLY a JSON array with title objects.

Example output:
[
  {{"title": "Software Engineer"}},
  {{"title": "Marketing Manager"}},
  {{"title": "Data Scientist"}}
]

Content:
{markdown_content[:4000]}

OUTPUT (JSON ARRAY ONLY):"""
    
        try:
            # Get just the job titles first
            titles_output = await self._generate_response(aggressive_prompt, max_tokens=1024)
            
            # Show raw aggressive extraction output
            print("\n" + "="*80)
            print("RAW AGGRESSIVE EXTRACTION OUTPUT:")
            print("="*80)
            print(titles_output)
            print("="*80)
            
            # Try direct backtick extraction first for aggressive output
            job_objects = []
            if '```' in titles_output:
                try:
                    json_text = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', titles_output, re.DOTALL)
                    if json_text:
                        job_objects = json.loads(json_text.group(1))
                        self.log("JOB_EXTRACTION", "success", f"Aggressive backtick extraction found {len(job_objects)} jobs")
                except Exception:
                    pass
            
            # If backtick extraction failed, try other methods
            if not job_objects:
                try:
                    # Try to find array in the output - properly call fix_json_aggressively
                    array_match = re.search(r'\[\s*\{.*?\}(?:,\s*\{.*?\})*\s*\]', titles_output, re.DOTALL)
                    if array_match:
                        json_str = array_match.group(0)
                        json_str = self._fix_json_aggressively(json_str)
                        job_objects = json.loads(json_str)
                except Exception as e:
                    self.log("JOB_EXTRACTION", "error", f"Array extraction failed: {e}")
            
            # If still no jobs found, try regex extraction
            if not job_objects:
                titles = re.findall(r'(?:position|job|title|role):\s*([^\n\.]{3,50})', markdown_content, re.IGNORECASE)
                job_objects = [{"title": title.strip()} for title in titles if len(title.strip()) > 3]
                
            # Add default company name from URL
            company = "Unknown Company"
            if source_url:
                try:
                    domain = source_url.replace('https://', '').replace('http://', '').split('/')[0]
                    company = domain.replace('www.', '').split('.')[0].capitalize()
                except:
                    pass
                    
            # Add company and other fields to all jobs
            enhanced_jobs = []
            for job in job_objects:
                job_info = {
                    "title": job.get("title", "Unknown Position"),
                    "company": company,
                    "url": source_url,
                    "location": "not mentioned",
                    "description": "not mentioned",
                    "salary": "not mentioned",
                    "requirements": "not mentioned",
                    "contrat-type": "not mentioned",
                    "required_skill": "not mentioned",
                    "post_date": "not mentioned"
                }
                enhanced_jobs.append(job_info)
                
            return enhanced_jobs
            
        except Exception as e:
            self.log("JOB_EXTRACTION", "error", f"Aggressive extraction failed: {e}")
            return []

    async def _aggressive_extract_from_chunks(self, markdown_content: str, source_url: str) -> List[Dict[str, Any]]:
        """Extract job titles aggressively from large content by processing it in chunks"""
        self.log("AGGRESSIVE_CHUNKING", "start", "Processing large content in chunks for aggressive extraction")
        
        # Split the content into manageable chunks
        chunks = self._chunk_markdown(markdown_content, 4000)
        all_jobs = set()  # Use a set to avoid duplicates
        
        # Process each chunk to find job titles
        for i, chunk in enumerate(chunks):
            self.log("AGGRESSIVE_CHUNKING", "info", f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            
            # Simple prompt to extract job titles from this chunk
            extraction_prompt = f"""Extract ALL possible job titles from this text chunk.
Return ONLY a JSON array of objects with title field.

Example:
[{{"title":"Software Engineer"}},{{"title":"Product Manager"}}]

CONTENT CHUNK:
{chunk[:3500]}

JSON ARRAY:"""
            
            try:
                # Generate response for this chunk
                chunk_output = await self._generate_response(extraction_prompt, max_tokens=512)
                
                # Parse job titles from this chunk
                titles_from_chunk = []
                
                # Try backtick extraction first
                if "```" in chunk_output:
                    try:
                        json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', chunk_output, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            titles_from_chunk = json.loads(json_str)
                    except Exception:
                        pass
                
                # If backtick extraction failed, try direct array extraction
                if not titles_from_chunk:
                    try:
                        array_match = re.search(r'\[\s*\{.*?\}(?:,\s*\{.*?\})*\s*\]', chunk_output, re.DOTALL)
                        if array_match:
                            json_str = self._fix_json_aggressively(array_match.group(0))
                            titles_from_chunk = json.loads(json_str)
                    except Exception:
                        pass
                
                # Add job titles from this chunk
                for job_info in titles_from_chunk:
                    if job_info.get('title'):
                        all_jobs.add(job_info.get('title'))
                        
            except Exception as e:
                self.log("AGGRESSIVE_CHUNKING", "error", f"Error processing chunk {i+1}: {e}")
        
        # Create job objects from the collected titles
        job_objects = []
        company = "Unknown Company"
        if source_url:
            try:
                domain = source_url.replace('https://', '').replace('http://', '').split('/')[0]
                company = domain.replace('www.', '').split('.')[0].capitalize()
            except:
                pass
                
        # Create job objects with basic information
        for title in all_jobs:
            job_objects.append({
                "title": title,
                "company": company,
                "url": source_url,
                "location": "not mentioned",
                "description": "not mentioned",
                "contrat-type": "not mentioned",
                "required_skill": "not mentioned",
                "post_date": "not mentioned"
            })
            
        self.log("AGGRESSIVE_CHUNKING", "success", f"Found {len(job_objects)} unique job titles across all chunks")
        return job_objects

    def _simplified_clean_job(self, job_data: Dict[str, Any], source_url: str) -> Dict[str, Any]:
        """Simplified job data cleaning without extensive processing"""
        clean_job = {}
        
        # Copy all fields directly
        for key, value in job_data.items():
            # Basic cleaning - just ensure values are strings
            if value is None:
                clean_job[key] = "not mentioned"
            elif isinstance(value, list):
                clean_job[key] = ", ".join(str(item) for item in value if item)
            else:
                clean_job[key] = str(value).strip()
        
        # Ensure critical fields exist
        if 'title' not in clean_job or not clean_job['title'] or clean_job['title'] == "not mentioned":
            clean_job['title'] = "Untitled Position"
        
        if 'company' not in clean_job or not clean_job['company'] or clean_job['company'] == "not mentioned":
            # Simple domain extraction from URL
            try:
                domain = source_url.replace('https://', '').replace('http://', '').split('/')[0]
                company = domain.replace('www.', '').split('.')[0].capitalize()
                clean_job['company'] = company or "Unknown Company"
            except:
                clean_job['company'] = "Unknown Company"
        
        # Add URL if missing
        if 'url' not in clean_job or not clean_job['url'] or clean_job['url'] == "not mentioned":
            clean_job['url'] = source_url
        
        return clean_job
    
    def _extract_jobs_json(self, llm_output: str) -> List[Dict[str, Any]]:
        """Extract JSON array from LLM output with improved parsing"""
        
        # Clean the output first
        if not llm_output or not isinstance(llm_output, str):
            return []
            
        llm_output = llm_output.strip()
        
        # Special handling for backtick-wrapped JSON (the most common format from LLMs)
        backtick_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', llm_output, re.DOTALL)
        if backtick_match:
            try:
                json_str = backtick_match.group(1).strip()
                fixed_json = self._fix_json_aggressively(json_str)
                parsed = json.loads(fixed_json)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed
            except Exception as e:
                self.log("JSON_PARSING", "error", f"Backtick JSON parsing failed: {str(e)}")
        
        # Direct JSON array parsing
        try:
            # Look for array pattern at the start or after common prefixes
            array_match = re.search(r'(?:JSON OUTPUT:|JSON array:|Here is the|Here are the|jobs:)?\s*(\[[\s\S]*?\])\s*$', llm_output, re.DOTALL)
            if array_match:
                json_str = array_match.group(1).strip()
                fixed_json = self._fix_json_aggressively(json_str)
                parsed = json.loads(fixed_json)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed
        except Exception as e:
            self.log("JSON_PARSING", "error", f"Direct JSON parsing failed: {str(e)}")
        
        # If above methods fail, try to find any JSON array in the text
        try:
            array_pattern = r'\[\s*\{.*?\}(?:,\s*\{.*?\})*\s*\]'
            array_match = re.search(array_pattern, llm_output, re.DOTALL)
            if array_match:
                json_str = array_match.group(0).strip()
                fixed_json = self._fix_json_aggressively(json_str)
                parsed = json.loads(fixed_json)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed
        except Exception as e:
            self.log("JSON_PARSING", "error", f"Array pattern parsing failed: {str(e)}")
        
        # Last resort: Try to extract individual job objects
        try:
            job_objects = re.findall(r'\{[^{}]*?"title"[^{}]*?\}', llm_output, re.DOTALL)
            if job_objects:
                jobs = []
                for obj_str in job_objects:
                    try:
                        fixed_obj = self._fix_json_aggressively(obj_str)
                        job = json.loads(fixed_obj)
                        if job.get('title'):
                            jobs.append(job)
                    except Exception:
                        continue
                if jobs:
                    return jobs
        except Exception as e:
            self.log("JSON_PARSING", "error", f"Individual job extraction failed: {str(e)}")
        
        return []

    def _fix_json_aggressively(self, json_str: str) -> str:
        """Fixed JSON fixing to handle all regex substitutions properly"""
        if not json_str or not isinstance(json_str, str):
            return "[]"  # Return empty array for None or non-string input
        
        try:
            # Remove markdown code blocks
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*', '', json_str)
            
            # Remove comments - properly providing empty string as replacement
            json_str = re.sub(r'//.*', '', json_str)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            
            # Fix quotes
            json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)  # Single quote keys
            json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)  # Single quote values
            
            # Fix trailing commas - FIXED: Adding empty string replacement
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # Fix missing quotes around unquoted keys - more reliable pattern
            json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
            
            # Remove non-printable characters except newlines and tabs
            json_str = ''.join(c for c in json_str if c.isprintable() or c in '\n\r\t ')
            
            return json_str.strip()
            
        except Exception as e:
            self.log("JSON_PARSING", "error", f"JSON fixing failed: {e}")
            # Return the original if processing fails
            return json_str

    def _extract_jobs_fallback(self, llm_output: str) -> List[Dict[str, Any]]:
        """Fallback method to extract job objects when JSON parsing fails"""
        # Look for individual job objects in the text
        if not llm_output or not isinstance(llm_output, str):
            return []
            
        try:
            job_objects = re.findall(r'\{[^{}]*"title"[^{}]*\}', llm_output, re.DOTALL)
            if job_objects:
                jobs = []
                for obj_str in job_objects:
                    try:
                        fixed_obj = self._fix_json_aggressively(obj_str)
                        job = json.loads(fixed_obj)
                        if job.get('title'):
                            jobs.append(job)
                    except Exception:
                        continue
                if jobs:
                    return jobs
        except Exception as e:
            self.log("JSON_PARSING", "error", f"Fallback extraction failed: {str(e)}")
        
        return []

    def display_results(self, result: ProcessingResult):
        print("\n" + "="*80)
        if result.success:
            print(f"🎉 SUCCESS: Found {len(result.jobs)} job opportunities")
            print(f"🧠 Execution Plan: {result.execution_plan}")
            print(f"🔧 Tools Used: {', '.join(result.tools_used)}")
            print(f"🤖 Model Used: {self.model} (via Groq API)")
            print(f"📅 Date: {self.current_time}")
            print(f"👤 User: {self.current_user}")
            print("="*80)
            for i, job in enumerate(result.jobs, 1):
                print(f"\n📋 Job #{i}")
                print(f"   Title: {job.get('title', 'not mentioned')}")
                print(f"   Company: {job.get('company', 'not mentioned')}")
                print(f"   Location: {job.get('location', 'not mentioned')}")
                print(f"   Field: {job.get('field', 'not mentioned')}")
                print(f"   Contrat-type: {job.get('contrat-type', 'not mentioned')}")
                print(f"   Required skill: {job.get('required_skill', 'not mentioned')}")
                print(f"   Post date: {job.get('post_date', 'not mentioned')}")
                if job.get('salary') and job.get('salary') != "not mentioned":
                    print(f"   Salary: {job.get('salary')}")
                if job.get('url') and job.get('url') != "not mentioned":
                    print(f"   URL: {job.get('url')}")
                if job.get('description') and job.get('description') != "not mentioned":
                    desc = job.get('description', '')[:200]
                    print(f"   Description: {desc}{'...' if len(job.get('description', '')) > 200 else ''}")
                print("-" * 60)
        else:
            print(f"❌ FAILED: {result.error_message}")
            print(f"🧠 Execution Plan: {result.execution_plan}")
            print(f"🔧 Tools Used: {', '.join(result.tools_used) if result.tools_used else 'None'}")
            print(f"🤖 Model Used: {self.model} (via Groq API)")
            print(f"📅 Date: {self.current_time}")
            print(f"👤 User: {self.current_user}")
            print("="*80)
            print("\n💡 Suggestions:")
            print("   • Check if the input is a valid company name or URL")
            print("   • Verify the website is accessible and contains job listings")
            print("   • Try a different company name or the company's main careers page")
            print("   • For Odoo specifically, try direct URLs like 'https://www.odoo.com/jobs' or 'https://www.odoo.com/jobs/development-6'")
            print("   • Try a different model like 'mixtral-8x7b-32768' for better extraction")

    async def cleanup(self):
        if self.mcp_client:
            self.log("CLEANUP", "start", "Disconnecting MCP client")
            try:
                await self.mcp_client.stop()
                self.log("CLEANUP", "success", "MCP client disconnected")
            except Exception as e:
                self.log("CLEANUP", "error", f"Cleanup error: {e}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

async def main():
    if len(sys.argv) < 3:
        print("Usage: python host.py mcp_server.py <company_name_or_url> [max_pages] [model]")
        print("Examples:")
        print("  python host.py mcp_server.py 'Google'")
        print("  python host.py mcp_server.py 'https://careers.google.com' 3")
        print("  python host.py mcp_server.py 'Microsoft' 2 'mixtral-8x7b-32768'")
        print("\nAvailable models:")
        print("  - llama3-70b-8192 (default)")
        print("  - mixtral-8x7b-32768")
        print("  - llama2-70b-4096")
        print("  - gemma-7b-it")
        print("\nRequires GROQ_API_KEY in config.py or environment variable")
        sys.exit(1)
    
    server_script = sys.argv[1]
    user_input = sys.argv[2]
    
    # Parse max_pages from command line if provided
    max_pages = 1
    if len(sys.argv) > 3:
        try:
            max_pages = int(sys.argv[3])
            if max_pages < 1:
                max_pages = 1
        except ValueError:
            print("Invalid max_pages value. Using default: 1")
            max_pages = 1
    
    # Check for API key from either config.py or environment variable
    if not GROQ_API_KEY:
        print("⚠️ GROQ_API_KEY not found in config.py or environment variables")
        print("Please either:")
        print("1. Add GROQ_API_KEY to your config.py file")
        print("2. Set it using: set GROQ_API_KEY=your-api-key (Windows) or export GROQ_API_KEY='your-api-key' (Linux/Mac)")
        print("Get your API key at: https://console.groq.com/")
        sys.exit(1)
    else:
        print("✅ GROQ_API_KEY found")
    
    # Set the user and date from the provided values
    current_user = "Maryem-Jlassi"
    current_time = "2025-08-06 18:21:24"

    print(f"🚀 LLM Orchestrator initialized")
    print(f"🎯 User Input: {user_input}")
    print(f"📄 Pages to crawl: {max_pages}")
    print(f"👤 User: {current_user}")
    print(f"📅 Date: {current_time}")
    print("="*80)
    
    # Allow custom model selection
    model = "llama3-70b-8192"  # Default model (without groq/ prefix)
    if len(sys.argv) > 4:
        model = sys.argv[4]
        # Remove groq/ prefix if provided
        if model.startswith("groq/"):
            model = model[5:]
        
    # Create orchestrator
    async with LLMOrchestrator() as orchestrator:
        orchestrator.model = model
        orchestrator.current_user = current_user
        orchestrator.current_time = current_time
        print(f"🤖 Model: {orchestrator.model}")
        
        if not await orchestrator.initialize_mcp_client(server_script):
            print("❌ Failed to initialize MCP client. Exiting.")
            return
        
        result = await orchestrator.llm_orchestrate(user_input, max_pages)
        orchestrator.display_results(result)

if __name__ == "__main__":
    asyncio.run(main())