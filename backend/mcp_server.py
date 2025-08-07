import sys
import os
import json
from datetime import datetime
from typing import Dict, Any
from urllib.parse import urlparse
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import re
from fastmcp import FastMCP
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import logging 
import mcp.types
from urllib.robotparser import RobotFileParser


# Force UTF-8 encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8')
        except Exception:
            pass

# Configuration du navigateur pour crawl4ai
browser_config = BrowserConfig(
    browser_type="chromium",
    headless=True,
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    extra_args=[
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-blink-features=AutomationControlled",
    ]
)

# MCP server instance
mcp = FastMCP(name="JobOfferServer")


#This is commonly used in web crawlers to respect site policies and avoid legal or ethical issues.
def is_allowed_by_robots(url: str, user_agent: str = "*") -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        # If robots.txt is missing or can't be read, assume allowed
        return True


def send_progress_notification(message: str, percentage: float = None, title: str = "Job HTML Finder Progress"):
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {
            "progressToken": "some-token",
            "value": {
                "kind": "report",
                "title": title,
                "message": message,
                "percentage": percentage
            }
        }
    }
    print(json.dumps(notification), flush=True)

@mcp.tool(annotations={"title": "get_official_website_and_generate_job_urls"})
async def get_official_website_and_generate_job_urls(company_name: str) -> dict:
    """
    Recherche le site officiel d'une société via DuckDuckGo et génère des URLs potentielles d'offres à partir de ce site.
    """
    try:
        send_progress_notification(f"Recherche du site officiel de {company_name}...", 10)
        
        def get_official_website(company_name: str) -> str:
            query = f"{company_name} site officiel"
            try:
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=5):
                        url = r.get("href") or r.get("url") or ""
                        if url and "linkedin.com" not in url and "wikipedia.org" not in url:
                            return url.split("?")[0]
                return ""
            except Exception as e:
                send_progress_notification(f"Erreur DuckDuckGo: {str(e)}", 0)
                return ""

        def generate_job_urls(base_url: str, company_name: str) -> list:
            base = base_url.rstrip("/")
            patterns = ["/careers", "/jobs", "/emploi", "/recrutement", "/job-offer", "/offres-emploi", "/career", "/opportunities" ,"/careers/our-job-offers.html"]
            urls = [f"{base}{p}" for p in patterns]
            urls.append(base)
            return urls

        site = get_official_website(company_name)
        if not site:
            return {"success": False, "company_name": company_name, "reason": "Site officiel introuvable."}
        
        send_progress_notification(f"Site trouvé: {site}", 50)
        urls = generate_job_urls(site, company_name)
        send_progress_notification(f"Génération de {len(urls)} URLs potentielles", 100)
        
        return {"success": True, "company_name": company_name, "official_website": site, "possible_job_urls": urls}
    
    except Exception as e:
        return {"success": False, "company_name": company_name, "reason": f"Erreur: {str(e)}"}

@mcp.tool(annotations={"title": "precheck_job_offer_url"})
async def precheck_job_offer_url(url: str) -> dict:
    """
    Précheck d'une URL (1000 premiers caractères, motifs d'offres détectés).
    """
    OFFER_PATTERNS = [
        r'/(jobs|careers|recrutement|opportunit\w+|emploi)/?',
        r'apply(-now)?', r'\b(job|emploi)\b', r'current openings',
        r'<a[^>]+href=["\'].*(job|emploi|offre|career|apply).*["\']',
        r'<form[^>]*action=["\'][^"\']*(apply|submit)[^"\']*["\']',
        r'<meta[^>]+(job|career|recrutement)[^>]*>',
        r'espace[\s_-]?candidat[s]?',
    ]
    
    # More specific error patterns - focusing on clear error indicators
    ERROR_PATTERNS = [
        r'<title[^>]*>\s*404\s*</title>',
        r'<title[^>]*>.*?(page not found|page introuvable).*?</title>',
        r'<h1[^>]*>\s*404\s*</h1>',
        r'<div[^>]*class="[^"]*error-page[^"]*"[^>]*>',
        r'<div[^>]*class="[^"]*page-404[^"]*"[^>]*>',
        r'<div[^>]*id="[^"]*error-page[^"]*"[^>]*>',
        r'<div[^>]*id="[^"]*page-404[^"]*"[^>]*>'
    ]
    
    # Specific phrases that definitely indicate empty job listings
    EMPTY_JOB_PHRASES = [
        r'<div[^>]*>\s*aucune offre (d\'emploi)? (n\'est)? disponible\s*</div>',
        r'<p[^>]*>\s*aucune offre (d\'emploi)? (n\'est)? disponible\s*</p>',
        r'<div[^>]*>\s*no (job )?positions (are )?available\s*</div>',
        r'<p[^>]*>\s*no (job )?positions (are )?available\s*</p>'
    ]
    
    try:
        send_progress_notification(f"Vérification de {url}...", 10)
        
        if not is_allowed_by_robots(url):
            return {
                "success": False,
                "url": url,
                "is_offer_page": False,
                "reason": "Blocked by robots.txt"
            }
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    session_id="precheck"
                )
            )
            
            if not result.success or not result.html:
                return {
                    "success": False,
                    "url": url,
                    "is_offer_page": False,
                    "reason": f"Fetch failed or page does not exist"
                }
            
            if hasattr(result, "status_code") and result.status_code and result.status_code >= 400:
                return {
                    "success": False,
                    "url": url,
                    "is_offer_page": False,
                    "reason": f"HTTP Error {result.status_code}"
                }
            
            text = result.html
            
            # Use BeautifulSoup for more structured analysis
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text, 'html.parser')
            
            # Check for HTTP status in title (common for error pages)
            title = soup.title.text.lower() if soup.title else ""
            if re.search(r'\b(404|error|not found|page introuvable)\b', title):
                return {
                    "success": True, 
                    "url": url, 
                    "is_offer_page": False, 
                    "reason": f"Error indicated in page title: '{title}'"
                }
                
            # Check for specific error patterns
            for pattern in ERROR_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    return {
                        "success": True, 
                        "url": url, 
                        "is_offer_page": False, 
                        "reason": "Error page structure detected"
                    }
                    
            # Check for empty job listing messages
            for pattern in EMPTY_JOB_PHRASES:
                if re.search(pattern, text, re.IGNORECASE):
                    return {
                        "success": True, 
                        "url": url, 
                        "is_offer_page": False, 
                        "reason": "Empty job listing page"
                    }
            
            # Count job offer patterns
            score = 0
            matched = []
            for pat in OFFER_PATTERNS:
                if re.search(pat, text, re.IGNORECASE):
                    score += 1
                    matched.append(pat)
            
            # URL-based detection - If URL explicitly contains job-related terms, increase score
            url_lower = url.lower()
            if any(term in url_lower for term in ['job', 'career', 'emploi', 'offre', 'recrutement']):
                score += 2  # Give extra weight to URL structure
                matched.append("URL structure")
            
            send_progress_notification(f"Analyse terminée. Score: {score}", 100)
            
            return {
                "success": True,
                "url": url,
                "is_offer_page": score >= 2,
                "patterns_found": score,
                "matched_patterns": matched[:5],  # Limit to first 5 patterns for readability
                "snippet": text[:200] + "..." if len(text) > 200 else text
            }
    except Exception as ex:
        return {
            "success": False,
            "url": url,
            "is_offer_page": False,
            "reason": str(ex)
        }
    
@mcp.tool(annotations={"title": "fetch_url_html_with_pagination"})
async def fetch_url_html_with_pagination(url: str, max_pages: int = 1) -> Dict[str, Any]:
    """
    Fait du crawling sur une URL (optionnellement paginée), retourne le HTML brut de chaque page.
    """
    try:
        session_id = f"fetchurl_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        results = []
        
        send_progress_notification(f"Début de l'extraction HTML...", 0)
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for page in range(1, max_pages + 1):
                send_progress_notification(f"Extraction page {page}/{max_pages}...", (page-1)/max_pages * 100)
                
                if max_pages > 1:
                    crawl_url = f"{url}&page={page}" if '?' in url else f"{url}?page={page}"
                else:
                    crawl_url = url
                    
                result = await crawler.arun(
                    url=crawl_url,
                    config=CrawlerRunConfig(
                        cache_mode=CacheMode.BYPASS,
                        session_id=session_id,
                    )
                )
                
                if result.success:
                    results.append({
                        "page": page,
                        "url": crawl_url,
                        "html": result.html,
                        "status": "success"
                    })
                else:
                    results.append({
                        "page": page,
                        "url": crawl_url,
                        "error": getattr(result, "error_message", "Unknown error"),
                        "status": "error"
                    })
        
        send_progress_notification(f"Extraction terminée. {len(results)} pages traitées.", 100)
        
        return {
            "success": True,
            "pages_fetched": len(results),
            "pages": results
        }
    
    except Exception as e:
        print(f"Error in fetch_url_html_with_pagination: {str(e)}", file=sys.stderr)
        return {
            "success": False,
            "error": str(e),
            "pages_fetched": 0,
            "pages": []
        }


@mcp.tool(annotations={"title": "html_to_markdown"})
async def html_to_markdown(html_content: str) -> Dict[str, Any]:
    """
    Convert HTML content to Markdown format.
    """
    try:
        send_progress_notification("Converting HTML to Markdown...", 10)
        
        # Remove script and style tags
        html_content = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Convert headings
        for i in range(1, 7):
            html_content = re.sub(rf'<h{i}[^>]*>(.*?)</h{i}>', rf'\n{"#" * i} \1\n', html_content, flags=re.IGNORECASE)
        
        # Convert paragraphs
        html_content = re.sub(r'<p[^>]*>(.*?)</p>', r'\n\1\n', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Convert line breaks
        html_content = re.sub(r'<br[^>]*>', '\n', html_content, flags=re.IGNORECASE)
        
        # Convert lists
        html_content = re.sub(r'<li[^>]*>(.*?)</li>', r'\n- \1', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove all other HTML tags
        html_content = re.sub(r'<[^>]+>', '', html_content)
        
        # Clean up whitespace
        html_content = re.sub(r'\n\s*\n', '\n\n', html_content)
        markdown_content = html_content.strip()
        
        send_progress_notification("HTML to Markdown conversion completed.", 100)
        
        return {
            "success": True,
            "markdown": markdown_content,
            "original_length": len(html_content),
            "markdown_length": len(markdown_content)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to convert HTML to Markdown: {str(e)}"
        }

if __name__ == "__main__":
    import traceback
    print("Starting JobOfferServer with FastMCP...", file=sys.stderr)
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        print(f"Server stopped due to: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)