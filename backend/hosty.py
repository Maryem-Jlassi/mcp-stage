import asyncio
import json
import sys
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import ollama

from client_claude import MCPClient

@dataclass
class ProcessingResult:
    success: bool
    jobs: List[Dict[str, Any]] = None
    error_message: str = None
    execution_plan: str = None
    tools_used: List[str] = None

def fix_llm_json_output(llm_decision: str) -> str:
    """More robust JSON fixing function"""
    # Remove comments
    llm_decision = re.sub(r"^\s*(//|#).*?$", "", llm_decision, flags=re.MULTILINE)
    llm_decision = re.sub(r"/\*.*?\*/", "", llm_decision, flags=re.DOTALL)
    
    # Replace single quotes with double quotes (be more careful)
    llm_decision = re.sub(r"'([^']*)':", r'"\1":', llm_decision)  # Keys
    llm_decision = re.sub(r":\s*'([^']*)'", r': "\1"', llm_decision)  # Values
    
    # Remove trailing commas
    llm_decision = re.sub(r',(\s*[}\]])', r'\1', llm_decision)
    
    # Clean non-printable characters
    llm_decision = ''.join(c for c in llm_decision if c.isprintable() or c in '\n\r\t ')
    llm_decision = llm_decision.strip()
    
    # Extract JSON object more carefully
    # Look for the outermost { } pair
    brace_count = 0
    start_idx = -1
    end_idx = -1
    
    for i, char in enumerate(llm_decision):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                end_idx = i
                break
    
    if start_idx != -1 and end_idx != -1:
        return llm_decision[start_idx:end_idx + 1]
    
    return llm_decision

def classify_input(user_input: str) -> str:
    """More robust input classification"""
    ui = user_input.strip().lower()
    # Check for URL patterns more thoroughly
    url_patterns = ['http://', 'https://', 'www.', '.com/', '.org/', '.net/', '/jobs', '/careers']
    if any(pattern in ui for pattern in url_patterns):
        return "url"
    return "company_name"

class LLMOrchestrator:
    def __init__(self, model: str = "llama3.2:latest"):
        self.model = model
        self.mcp_client: Optional[MCPClient] = None
        self.available_tools = []
        
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
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': orchestration_prompt}],
                options={'temperature': 0.1, 'top_p': 0.9}
            )
            llm_decision = response['message']['content'].strip()
            self.log("LLM_ORCHESTRATOR", "info", f"LLM decision received: {len(llm_decision)} characters")
            
            # Save raw decision for debugging
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
            # Create fallback decision
            decision_data = self._create_fallback_decision(user_input, input_type, max_pages)
            return await self.execute_plan(decision_data)

    def _parse_llm_json(self, llm_decision: str) -> Optional[Dict[str, Any]]:
        """Try multiple strategies to parse LLM JSON output"""
        
        # Strategy 1: Direct parsing
        try:
            return json.loads(llm_decision)
        except Exception:
            pass
        
        # Strategy 2: Fix and parse
        try:
            fixed_json = fix_llm_json_output(llm_decision)
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
                        fixed_match = fix_llm_json_output(match)
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
            
            # Final step: Extract jobs from markdown
            if markdown_content:
                jobs = await self.llm_extract_jobs(markdown_content, source_url)
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

    async def llm_extract_jobs(self, markdown_content: str, source_url: str = "") -> List[Dict[str, Any]]:
        self.log("JOB_EXTRACTION", "start", f"Extracting jobs from {len(markdown_content)} characters")
        
        # Process raw markdown directly without optimization
        try:
            # Updated extraction prompt with "not mentioned" instead of null
            extraction_prompt = f"""Extract ALL job listings from this markdown content. Return ONLY a JSON array.

    CONTEXT:
    - Source: {source_url if source_url else "a job website"}
    - Content format: Markdown

    JOB IDENTIFICATION IN MARKDOWN:
    - Look for job titles in headings (# heading) or emphasized text (**bold** or *italic*)
    - Job listings often follow patterns with sections for description, requirements, etc.
    - Jobs might be separated by horizontal rules (---) or headers
    - Look for key phrases like "Job Title", "Position", "Role", "Opening", "We're hiring", "Apply now"
    - Job details are often in lists (- item or * item) or paragraphs
    - A single job listing often contains multiple paragraphs of related information

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
   - If no jobs found: return empty array []

   MARKDOWN CONTENT:
   {markdown_content}

    JSON OUTPUT:"""

            response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': extraction_prompt}],
            options={'temperature': 0.1, 'top_p': 0.9}
        )
        
            llm_output = response['message']['content'].strip()
            self.log("JOB_EXTRACTION", "info", f"LLM extraction output: {len(llm_output)} characters")
        
        # Extract jobs from LLM output
            jobs = self._extract_jobs_json(llm_output)
        
            if not jobs:
                self.log("JOB_EXTRACTION", "info", "Still no jobs found, trying fallback extraction")
                jobs = self._extract_jobs_fallback(llm_output)
        
        # Apply simplified job data cleaning
            simplified_jobs = []
            for job in jobs:
                clean_job = self._simplified_clean_job(job, source_url)
                simplified_jobs.append(clean_job)
        
            self.log("JOB_EXTRACTION", "success", f"Extracted {len(simplified_jobs)} job offers")
            return simplified_jobs
        
        except Exception as e:
            self.log("JOB_EXTRACTION", "error", f"Job extraction failed: {e}")
            return []


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
        llm_output = llm_output.strip()
    
    # Try to find JSON array with more patterns
        json_patterns = [
        r'\[\s*\{.*?\}\s*\]',           # Standard array
        r'\[[\s\S]*?\]',                # Any array content
        r'```json\s*(\[[\s\S]*?\])',    # JSON in code block
        r'```\s*(\[[\s\S]*?\])',        # JSON in any code block
        r'(\[[\s\S]*?\}[\s\S]*?\])',    # Flexible array pattern
    ]
    
    # Here's the fix - using json_patterns instead of patterns
        for pattern in json_patterns:
           matches = re.findall(pattern, llm_output, re.DOTALL)
           for match in matches:
            # If match is from a group, use the group content
                content = match if isinstance(match, str) else match[0] if match else ""
                try:
                # Clean and fix the JSON
                   fixed_json = self._fix_json_aggressively(content)
                   parsed = json.loads(fixed_json)
                   if isinstance(parsed, list) and len(parsed) > 0:
                    # Validate that items look like jobs
                        if all(isinstance(item, dict) and item.get('title') for item in parsed[:3]):
                           return parsed
                except Exception:
                    continue
    
    # Try to extract individual job objects if array parsing fails
        job_patterns = [
        r'\{[^{}]*"title"[^{}]*\}',
        r'\{[\s\S]*?"title"[\s\S]*?\}',
    ]
    
        jobs = []
        for pattern in job_patterns:
            matches = re.findall(pattern, llm_output, re.DOTALL)
            for match in matches:
                try:
                   fixed_obj = self._fix_json_aggressively(match)
                   job = json.loads(fixed_obj)
                   if job.get('title'):
                      jobs.append(job)
                except Exception:
                    continue
            if jobs:
                break
    
        return jobs


    def _fix_json_aggressively(self, json_str: str) -> str:
        """More aggressive JSON fixing"""
        if not json_str.strip():
            return "[]"
        
        # Remove markdown code blocks
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*', '', json_str)
        
        # Remove comments
        json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Fix quotes
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)  # Single quote keys
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)  # Single quote values
        
        # Fix trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix missing quotes around unquoted keys
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        # Remove non-printable characters except newlines and tabs
        json_str = ''.join(c for c in json_str if c.isprintable() or c in '\n\r\t ')
        
        return json_str.strip()

    def _extract_jobs_fallback(self, llm_output: str) -> List[Dict[str, Any]]:
        """Fallback method to extract job objects when JSON parsing fails"""
        # Look for individual job objects in the text
        job_objects = re.findall(r'\{[^{}]*"title"[^{}]*\}', llm_output, re.DOTALL)
        if job_objects:
            jobs = []
            for obj_str in job_objects:
                try:
                    fixed_obj = fix_llm_json_output(obj_str)
                    job = json.loads(fixed_obj)
                    jobs.append(job)
                except Exception:
                    continue
            if jobs:
                return jobs
        
        return []

    def display_results(self, result: ProcessingResult):
        print("\n" + "="*80)
        if result.success:
            print(f"🎉 SUCCESS: Found {len(result.jobs)} job opportunities")
            print(f"🧠 Execution Plan: {result.execution_plan}")
            print(f"🔧 Tools Used: {', '.join(result.tools_used)}")
            print("="*80)
            for i, job in enumerate(result.jobs, 1):
                print(f"\n📋 Job #{i}")
                print(f"   Title: {job.get('title', 'not mentioned')}")
                print(f"   Company: {job.get('company', 'not mentioned')}")
                print(f"   Location: {job.get('location', 'not mentioned')}")
                print(f"   Field: {job.get('field', 'not mentioned')}")
                print(f"   Contrat-type: {job.get('contrat-type', 'not mentioned')}")
                print(f"   Required skill: {job.get('required skill', 'not mentioned')}")
                print(f"   Post date: {job.get('post date', 'not mentioned')}")
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
            print("="*80)
            print("\n💡 Suggestions:")
            print("   • Check if the input is a valid company name or URL")
            print("   • Verify the website is accessible and contains job listings")
            print("   • Try a different company name or the company's main careers page")

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
        print("Usage: python hosty.py ahsen_server.py <company_name_or_url>")
        print("Examples:")
        print("  python hosty.py ahsen_server.py 'Google'")
        print("  python hosty.py ahsen_server.py 'https://careers.google.com' 3")
        sys.exit(1)
    
    server_script = sys.argv[1]
    user_input = sys.argv[2]
    
    # Parse max_pages from command line if provided
    max_pages = 1
    model = "llama3.2:latest"
    
    if len(sys.argv) > 3:
        try:
            max_pages = int(sys.argv[3])
            if max_pages < 1:
                max_pages = 1
        except ValueError:
            # If the argument isn't a number, assume it's the model name
            model = sys.argv[3]
    
    if len(sys.argv) > 4:
        model = sys.argv[4]
    
    print(f"🚀 LLM Orchestrator initialized with model: {model}")
    print(f"🎯 User Input: {user_input}")
    print(f"📄 Pages to crawl: {max_pages}")
    print("="*80)
    
    async with LLMOrchestrator(model) as orchestrator:
        if not await orchestrator.initialize_mcp_client(server_script):
            print("❌ Failed to initialize MCP client. Exiting.")
            return
        
        result = await orchestrator.llm_orchestrate(user_input, max_pages)
        orchestrator.display_results(result)

if __name__ == "__main__":
    asyncio.run(main())