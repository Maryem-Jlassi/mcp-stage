import streamlit as st
import asyncio
import json
import sys
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
import time

# Import your existing orchestrator (adjust the import based on your file structure)
# Assuming your orchestrator code is in a file called 'orchestrator.py'
try:
    from best2 import LLMOrchestrator, ProcessingResult
except ImportError:
    st.error("Could not import LLMOrchestrator. Make sure your orchestrator code is available.")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="AI Job Search Orchestrator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .job-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .success-banner {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .error-banner {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .skill-tag {
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'search_count' not in st.session_state:
    st.session_state.search_count = 0

def save_search_to_history(search_input, results, execution_time):
    """Save search results to history"""
    search_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'input': search_input,
        'success': results.success if results else False,
        'job_count': len(results.jobs) if results and results.jobs else 0,
        'execution_time': execution_time,
        'results': results
    }
    st.session_state.search_history.insert(0, search_entry)
    # Keep only last 10 searches
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[:10]

async def run_job_search(user_input: str, max_pages: int, model: str, server_script: str) -> ProcessingResult:
    """Async wrapper for job search"""
    async with LLMOrchestrator(model) as orchestrator:
        if not await orchestrator.initialize_mcp_client(server_script):
            return ProcessingResult(
                success=False,
                error_message="Failed to initialize MCP client. Check if the server script exists and is accessible."
            )
        
        result = await orchestrator.llm_orchestrate(user_input, max_pages)
        return result

def display_job_card(job: Dict[str, Any], index: int):
    """Display a single job card"""
    with st.container():
        st.markdown(f"""
        <div class="job-card">
            <h3 style="color: #333; margin-bottom: 0.5rem;">
                📋 {job.get('title', 'Untitled Position')}
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**🏢 Company:** {job.get('company', 'Unknown Company')}")
            if job.get('location') and job.get('location') != 'not mentioned':
                st.write(f"**📍 Location:** {job.get('location')}")
            if job.get('description') and job.get('description') != 'not mentioned':
                desc = job.get('description', '')[:200]
                st.write(f"**📝 Description:** {desc}{'...' if len(job.get('description', '')) > 200 else ''}")
        
        with col2:
            if job.get('contrat-type') and job.get('contrat-type') != 'not mentioned':
                st.write(f"**💼 Type:** {job.get('contrat-type')}")
            if job.get('salary') and job.get('salary') != 'not mentioned':
                st.write(f"**💰 Salary:** {job.get('salary')}")
            if job.get('post_date') and job.get('post_date') != 'not mentioned':
                st.write(f"**📅 Posted:** {job.get('post_date')}")
        
        with col3:
            if job.get('required_skill') and job.get('required_skill') != 'not mentioned':
                st.write("**🎯 Skills:**")
                skills = job.get('required_skill', '').split(',')[:5]  # Show max 5 skills
                for skill in skills:
                    if skill.strip():
                        st.markdown(f'<span class="skill-tag">{skill.strip()}</span>', unsafe_allow_html=True)
            
            if job.get('url') and job.get('url') != 'not mentioned':
                st.markdown(f"[🔗 Apply Now]({job.get('url')})")
        
        st.markdown("---")

def display_results_summary(results: ProcessingResult):
    """Display search results summary"""
    if results.success:
        st.markdown(f"""
        <div class="success-banner">
            <h4>✅ Search Completed Successfully!</h4>
            <p><strong>Jobs Found:</strong> {len(results.jobs)} opportunities</p>
            <p><strong>Tools Used:</strong> {', '.join(results.tools_used) if results.tools_used else 'None'}</p>
            <p><strong>Execution Plan:</strong> {results.execution_plan}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(results.jobs)}</h3>
                <p>Total Jobs</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            companies = set(job.get('company', 'Unknown') for job in results.jobs)
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(companies)}</h3>
                <p>Companies</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            remote_jobs = sum(1 for job in results.jobs if 'remote' in job.get('location', '').lower())
            st.markdown(f"""
            <div class="metric-card">
                <h3>{remote_jobs}</h3>
                <p>Remote Jobs</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            salary_jobs = sum(1 for job in results.jobs if job.get('salary') and job.get('salary') != 'not mentioned')
            st.markdown(f"""
            <div class="metric-card">
                <h3>{salary_jobs}</h3>
                <p>With Salary</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="error-banner">
            <h4>❌ Search Failed</h4>
            <p><strong>Error:</strong> {results.error_message}</p>
            <p><strong>Execution Plan:</strong> {results.execution_plan}</p>
            <p><strong>Tools Used:</strong> {', '.join(results.tools_used) if results.tools_used else 'None'}</p>
            
            <h5>💡 Suggestions:</h5>
            <ul>
                <li>Check if the input is a valid company name or URL</li>
                <li>Verify the website is accessible and contains job listings</li>
                <li>Try a different company name or the company's main careers page</li>
                <li>Ensure the MCP server script is running and accessible</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI Job Search Orchestrator</h1>
        <p>Powered by LLM intelligence to find your perfect job opportunities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model = st.selectbox(
            "🤖 LLM Model",
            ["llama3.2:latest", "llama3.1:latest", "llama2:latest", "mistral:latest"],
            index=0,
            help="Select the LLM model to use for job extraction"
        )
        
        # Server script
        server_script = st.text_input(
            "🖥️ MCP Server Script",
            value="ahsen_server.py",
            help="Path to your MCP server script"
        )
        
        # Advanced options
        st.header("🔧 Advanced Options")
        max_pages = st.slider(
            "📄 Pages to Crawl",
            min_value=1,
            max_value=10,
            value=1,
            help="Number of pages to crawl for job listings"
        )
        
        show_raw_data = st.checkbox("📊 Show Raw JSON Data", help="Display raw job data in JSON format")
        
        # Search statistics
        st.header("📈 Statistics")
        st.metric("Total Searches", st.session_state.search_count)
        if st.session_state.search_history:
            successful_searches = sum(1 for s in st.session_state.search_history if s['success'])
            st.metric("Success Rate", f"{(successful_searches/len(st.session_state.search_history)*100):.1f}%")
    
    # Main search interface
    st.header("🔍 Job Search")
    
    # Search input
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input(
            "Company Name or Job URL",
            placeholder="e.g., 'Google', 'Microsoft', or 'https://careers.example.com/jobs'",
            help="Enter a company name to search their careers page, or a direct URL to job listings"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_button = st.button("🚀 Search Jobs", type="primary", use_container_width=True)
    
    # Quick examples
    st.write("**Quick Examples:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Google", use_container_width=True):
            user_input = "Google"
            st.rerun()
    with col2:
        if st.button("Microsoft", use_container_width=True):
            user_input = "Microsoft"
            st.rerun()
    with col3:
        if st.button("OpenAI", use_container_width=True):
            user_input = "OpenAI"
            st.rerun()
    with col4:
        if st.button("Tesla", use_container_width=True):
            user_input = "Tesla"
            st.rerun()
    
    # Execute search
    if search_button and user_input:
        if not os.path.exists(server_script):
            st.error(f"❌ Server script '{server_script}' not found. Please check the path in the sidebar.")
            return
        
        st.session_state.search_count += 1
        
        with st.spinner(f"🔄 Searching jobs for '{user_input}'... This may take a few minutes."):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            
            try:
                # Update progress
                status_text.text("🔧 Initializing MCP client...")
                progress_bar.progress(20)
                
                status_text.text("🔍 Analyzing input and planning execution...")
                progress_bar.progress(40)
                
                status_text.text("🌐 Fetching job data...")
                progress_bar.progress(60)
                
                status_text.text("🤖 Extracting jobs with LLM...")
                progress_bar.progress(80)
                
                # Run the actual search
                result = asyncio.run(run_job_search(user_input, max_pages, model, server_script))
                
                progress_bar.progress(100)
                status_text.text("✅ Search completed!")
                
                execution_time = time.time() - start_time
                
                # Save to history
                save_search_to_history(user_input, result, execution_time)
                st.session_state.current_results = result
                
                # Clear progress indicators
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                progress_bar.empty()
                status_text.empty()
                return
    
    # Display results
    if st.session_state.current_results:
        results = st.session_state.current_results
        
        # Results summary
        display_results_summary(results)
        
        if results.success and results.jobs:
            # Filter and sort options
            col1, col2, col3 = st.columns(3)
            with col1:
                company_filter = st.selectbox(
                    "Filter by Company",
                    ["All"] + list(set(job.get('company', 'Unknown') for job in results.jobs))
                )
            with col2:
                location_filter = st.selectbox(
                    "Filter by Location",
                    ["All"] + list(set(job.get('location', 'Unknown') for job in results.jobs if job.get('location') != 'not mentioned'))
                )
            with col3:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Default", "Company", "Title", "Location"]
                )
            
            # Apply filters
            filtered_jobs = results.jobs
            if company_filter != "All":
                filtered_jobs = [job for job in filtered_jobs if job.get('company') == company_filter]
            if location_filter != "All":
                filtered_jobs = [job for job in filtered_jobs if job.get('location') == location_filter]
            
            # Apply sorting
            if sort_by == "Company":
                filtered_jobs = sorted(filtered_jobs, key=lambda x: x.get('company', 'Unknown'))
            elif sort_by == "Title":
                filtered_jobs = sorted(filtered_jobs, key=lambda x: x.get('title', 'Untitled'))
            elif sort_by == "Location":
                filtered_jobs = sorted(filtered_jobs, key=lambda x: x.get('location', 'Unknown'))
            
            st.write(f"**Showing {len(filtered_jobs)} of {len(results.jobs)} jobs**")
            
            # Display jobs
            for idx, job in enumerate(filtered_jobs):
                display_job_card(job, idx)
            
            # Export options
            st.header("📤 Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export as CSV
                if st.button("📊 Export as CSV"):
                    df = pd.DataFrame(filtered_jobs)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"jobs_{user_input}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                # Export as JSON
                if st.button("📋 Export as JSON"):
                    json_str = json.dumps(filtered_jobs, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"jobs_{user_input}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col3:
                # Copy to clipboard (display JSON)
                if st.button("📋 Show JSON"):
                    st.json(filtered_jobs)
            
            # Raw data display
            if show_raw_data:
                st.header("📊 Raw Job Data")
                st.json(filtered_jobs)
    
    # Search history
    if st.session_state.search_history:
        st.header("📚 Search History")
        
        history_df = pd.DataFrame([
            {
                'Timestamp': entry['timestamp'],
                'Input': entry['input'],
                'Success': '✅' if entry['success'] else '❌',
                'Jobs Found': entry['job_count'],
                'Execution Time': f"{entry['execution_time']:.2f}s"
            }
            for entry in st.session_state.search_history
        ])
        
        st.dataframe(history_df, use_container_width=True)
        
        # Option to load previous search
        selected_history = st.selectbox(
            "Load Previous Search",
            options=range(len(st.session_state.search_history)),
            format_func=lambda x: f"{st.session_state.search_history[x]['timestamp']} - {st.session_state.search_history[x]['input']}"
        )
        
        if st.button("🔄 Load Selected Search"):
            st.session_state.current_results = st.session_state.search_history[selected_history]['results']
            st.rerun()
        
        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.search_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with ❤️ using Streamlit and LLM Orchestrator</p>
        <p>Powered by Ollama LLM and MCP Tools</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()