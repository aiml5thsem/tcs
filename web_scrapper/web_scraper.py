"""
Advanced Production Web Scraper for Authenticated Organizational Websites
Supports: Auto, Manual, Hybrid modes | Existing Chrome support | Sitemap parsing

Usage Examples:
    # Auto mode - discovers and crawls all links automatically
    python main.py --url https://myapp.com --mode auto
    
    # Manual mode - you navigate, script extracts on Enter
    python main.py --url https://myapp.com --mode manual
    
    # Hybrid mode - auto-discovers, then lets you add/remove links
    python main.py --url https://myapp.com --mode hybrid
    
    # Use existing Chrome (no playwright browser needed)
    python main.py --url https://myapp.com --mode auto --use-chrome /path/to/chrome
    
    # Full configuration
    python main.py --url https://myapp.com --mode auto --max-pages 500 --max-depth 8 --use-chrome auto

Finding Chrome paths:
    macOS: 
        - Chrome: /Applications/Google Chrome.app/Contents/MacOS/Google Chrome
        - Chromium: /Applications/Chromium.app/Contents/MacOS/Chromium
        Find: ls -la "/Applications/Google Chrome.app/Contents/MacOS/"
    
    Windows:
        - Chrome: C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe
        - Chromium: C:\\Program Files\\Chromium\\Application\\chrome.exe
        Find: where chrome
    
    Linux:
        - Chrome: /usr/bin/google-chrome
        - Chromium: /usr/bin/chromium-browser
        Find: which google-chrome
"""

import asyncio
import re
import json
import html2text
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from datetime import datetime
from typing import Set, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import deque
import platform
import subprocess

from playwright.async_api import async_playwright, Page, Browser, BrowserContext


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class PageContent:
    """Represents a scraped page with metadata."""
    url: str
    title: str
    depth: int
    content_markdown: str
    parent_url: Optional[str] = None
    found_at: str = field(default_factory=lambda: datetime.now().isoformat())
    order_index: int = 0  # For maintaining discovery order


@dataclass
class CrawlConfig:
    """Configuration for the crawling session."""
    mode: str  # 'auto', 'manual', 'hybrid'
    base_url: str
    max_pages: int = 100
    max_depth: int = 6
    use_existing_chrome: Optional[str] = None  # Path to Chrome or 'auto'
    output_file: Optional[str] = None
    follow_sitemap: bool = True
    respect_robots: bool = True
    delay_between_requests: float = 0.5  # seconds


# ============================================================================
# SITEMAP PARSER
# ============================================================================

class SitemapParser:
    """Parse and extract URLs from sitemap.xml files."""
    
    @staticmethod
    async def discover_sitemap(page: Page, base_url: str) -> Optional[str]:
        """Try to find sitemap.xml at common locations."""
        common_paths = [
            '/sitemap.xml',
            '/sitemap_index.xml', 
            '/sitemap-index.xml',
            '/sitemap1.xml',
            '/sitemaps/sitemap.xml',
        ]
        
        for path in common_paths:
            sitemap_url = urljoin(base_url, path)
            try:
                response = await page.goto(sitemap_url, wait_until="domcontentloaded", timeout=5000)
                if response and response.ok:
                    content_type = response.headers.get('content-type', '')
                    if 'xml' in content_type.lower():
                        print(f"✓ Found sitemap: {sitemap_url}")
                        return sitemap_url
            except:
                continue
        
        # Try to find in robots.txt
        try:
            robots_url = urljoin(base_url, '/robots.txt')
            await page.goto(robots_url, wait_until="domcontentloaded", timeout=5000)
            content = await page.content()
            # Look for Sitemap: directive
            matches = re.findall(r'Sitemap:\s*(.+)', content, re.IGNORECASE)
            if matches:
                sitemap_url = matches[0].strip()
                print(f"✓ Found sitemap in robots.txt: {sitemap_url}")
                return sitemap_url
        except:
            pass
        
        return None
    
    @staticmethod
    async def parse_sitemap(page: Page, sitemap_url: str, url_filter) -> Set[str]:
        """Parse sitemap XML and extract all valid URLs."""
        urls = set()
        
        try:
            await page.goto(sitemap_url, wait_until="domcontentloaded", timeout=10000)
            xml_content = await page.content()
            
            # Remove HTML wrapper if present
            xml_content = re.sub(r'^.*?<\?xml', '<?xml', xml_content, flags=re.DOTALL)
            
            root = ET.fromstring(xml_content)
            
            # Handle sitemap index (contains other sitemaps)
            ns = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            sitemap_locs = root.findall('.//s:sitemap/s:loc', ns)
            
            if sitemap_locs:
                # This is a sitemap index
                print(f"  → Found sitemap index with {len(sitemap_locs)} sitemaps")
                for sitemap_loc in sitemap_locs:
                    sub_sitemap_url = sitemap_loc.text
                    sub_urls = await SitemapParser.parse_sitemap(page, sub_sitemap_url, url_filter)
                    urls.update(sub_urls)
            else:
                # This is a regular sitemap with URLs
                url_locs = root.findall('.//s:url/s:loc', ns)
                print(f"  → Found {len(url_locs)} URLs in sitemap")
                
                for url_loc in url_locs:
                    url = url_loc.text
                    if url:
                        filtered = url_filter.filter_url(url)
                        if filtered:
                            urls.add(filtered)
        
        except Exception as e:
            print(f"  [!] Error parsing sitemap: {str(e)[:80]}")
        
        return urls


# ============================================================================
# URL FILTERING & NORMALIZATION
# ============================================================================

class URLFilter:
    """Intelligent URL filtering and normalization."""
    
    SKIP_PATTERNS = [
        r'logout', r'signout', r'sign-out', r'/auth/logout',
        r'\.pdf$', r'\.zip$', r'\.exe$', r'\.dmg$', r'\.pkg$',
        r'\.xlsx?$', r'\.csv$', r'\.json$', r'\.xml$', r'\.txt$',
        r'mailto:', r'tel:', r'javascript:', r'#$',
        r'/api/', r'/v\d+/api', r'/_next/', r'/__',
        r'download', r'export', r'print', r'/cdn-cgi/',
    ]
    
    SKIP_QUERY_PARAMS = [
        'sessionid', 'session', 'sid', 'token',
        'utm_', 'tracking', 'ref=', 'source=',
        'fbclid', 'gclid', 'msclkid',
    ]
    
    def __init__(self, base_url: str):
        parsed = urlparse(base_url)
        self.base_origin = f"{parsed.scheme}://{parsed.netloc}"
        self.base_path = parsed.path.rstrip('/')
    
    def is_same_origin(self, url: str) -> bool:
        """Check if URL is from same domain."""
        try:
            parsed = urlparse(url)
            origin = f"{parsed.scheme}://{parsed.netloc}"
            return origin == self.base_origin
        except:
            return False
    
    def should_skip(self, url: str) -> bool:
        """Check if URL should be skipped."""
        url_lower = url.lower()
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, url_lower):
                return True
        return False
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        parsed = urlparse(url)
        
        # Remove fragment
        url_no_fragment = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            ''  # fragment
        ))
        
        # Clean query parameters
        if parsed.query:
            params = parsed.query.split('&')
            clean_params = []
            
            for param in params:
                skip = False
                for skip_param in self.SKIP_QUERY_PARAMS:
                    if param.lower().startswith(skip_param):
                        skip = True
                        break
                if not skip and param:
                    clean_params.append(param)
            
            query_string = '&'.join(sorted(clean_params))  # Sort for consistency
            url_no_fragment = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                query_string,
                ''
            ))
        
        # Remove trailing slash unless it's the root
        if url_no_fragment.endswith('/') and len(urlparse(url_no_fragment).path) > 1:
            url_no_fragment = url_no_fragment.rstrip('/')
        
        return url_no_fragment
    
    def filter_url(self, url: str) -> Optional[str]:
        """Apply all filters. Return normalized URL if valid, None otherwise."""
        if not url or not url.startswith('http'):
            return None
        
        if self.should_skip(url):
            return None
        
        if not self.is_same_origin(url):
            return None
        
        normalized = self.normalize_url(url)
        return normalized


# ============================================================================
# HTML TO MARKDOWN CONVERTER
# ============================================================================

class HTMLToMarkdown:
    """Convert HTML to clean, LLM-ready Markdown."""
    
    def __init__(self):
        self.converter = html2text.HTML2Text()
        self.converter.body_width = 0
        self.converter.ignore_links = False
        self.converter.ignore_images = False
        self.converter.ignore_emphasis = False
        self.converter.ignore_tables = False
        self.converter.unicode_snob = True
        self.converter.use_automatic_links = True
        self.converter.skip_internal_links = True
    
    def convert(self, html: str, url: str = "") -> str:
        """Convert HTML to Markdown with cleanup."""
        try:
            html = self._clean_html(html)
            markdown = self.converter.handle(html)
            return self._clean_markdown(markdown)
        except Exception as e:
            print(f"  [!] Markdown conversion error: {str(e)[:60]}")
            return f"<!-- Conversion failed for {url} -->\n"
    
    def _clean_html(self, html: str) -> str:
        """Pre-process HTML."""
        # Remove scripts, styles, and common noise
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        return html
    
    def _clean_markdown(self, markdown: str) -> str:
        """Post-process Markdown."""
        # Remove excessive blank lines
        markdown = re.sub(r'\n\n\n+', '\n\n', markdown)
        
        # Clean trailing whitespace
        markdown = '\n'.join(line.rstrip() for line in markdown.split('\n'))
        
        # Remove lines with only whitespace
        lines = [line for line in markdown.split('\n') if line.strip() or line == '']
        markdown = '\n'.join(lines)
        
        return markdown.strip()


# ============================================================================
# BROWSER MANAGER
# ============================================================================

class BrowserManager:
    """Manages browser lifecycle with support for existing Chrome."""
    
    @staticmethod
    def find_chrome_path() -> Optional[str]:
        """Auto-detect Chrome installation."""
        system = platform.system()
        
        paths = {
            'Darwin': [  # macOS
                '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                '/Applications/Chromium.app/Contents/MacOS/Chromium',
            ],
            'Windows': [
                'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
                'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
                'C:\\Program Files\\Chromium\\Application\\chrome.exe',
            ],
            'Linux': [
                '/usr/bin/google-chrome',
                '/usr/bin/google-chrome-stable',
                '/usr/bin/chromium-browser',
                '/usr/bin/chromium',
                '/snap/bin/chromium',
            ]
        }
        
        for path in paths.get(system, []):
            if Path(path).exists():
                print(f"✓ Found Chrome at: {path}")
                return path
        
        return None
    
    @staticmethod
    async def launch_browser(config: CrawlConfig):
        """Launch browser based on configuration."""
        playwright = await async_playwright().start()
        
        chrome_path = None
        if config.use_existing_chrome:
            if config.use_existing_chrome == 'auto':
                chrome_path = BrowserManager.find_chrome_path()
            elif Path(config.use_existing_chrome).exists():
                chrome_path = config.use_existing_chrome
            else:
                print(f"⚠ Chrome not found at: {config.use_existing_chrome}")
        
        if chrome_path:
            # Use existing Chrome
            print(f"🌐 Launching existing Chrome from: {chrome_path}")
            browser = await playwright.chromium.launch(
                headless=False,
                executable_path=chrome_path,
                args=['--disable-blink-features=AutomationControlled']
            )
        else:
            # Use Playwright's Chromium
            print("🌐 Launching Playwright Chromium")
            browser = await playwright.chromium.launch(
                headless=False,
                args=['--disable-blink-features=AutomationControlled']
            )
        
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        return playwright, browser, context, page


# ============================================================================
# MAIN SCRAPER
# ============================================================================

class AuthenticatedScraper:
    """Main scraper with Auto/Manual/Hybrid modes."""
    
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.visited_urls: Set[str] = set()
        self.pages: Dict[str, PageContent] = {}
        self.url_filter = URLFilter(config.base_url)
        self.html_converter = HTMLToMarkdown()
        self.order_counter = 0
        
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
    
    async def start_browser(self):
        """Initialize browser."""
        print("\n" + "="*70)
        print("🔐 BROWSER INITIALIZATION")
        print("="*70 + "\n")
        
        self.playwright, self.browser, self.context, self.page = \
            await BrowserManager.launch_browser(self.config)
        
        print(f"✓ Navigating to: {self.config.base_url}\n")
        await self.page.goto(self.config.base_url, wait_until="networkidle", timeout=60000)
    
    async def authenticate(self):
        """Wait for user authentication."""
        print("\n" + "="*70)
        print("🔑 AUTHENTICATION PHASE")
        print("="*70)
        print("\n📋 INSTRUCTIONS:")
        print("-" * 70)
        print("1. Authenticate in the browser (SSO, OAuth, 2FA, etc.)")
        print("2. Navigate to any important starting pages")
        print("3. Press ENTER when ready to begin crawling")
        print("-" * 70 + "\n")
        
        input("⏳ Press ENTER to continue...")
        
        current_url = self.page.url
        title = await self.page.title()
        print(f"\n✓ Current URL: {current_url}")
        print(f"✓ Page title: {title}\n")
    
    async def extract_page_content(self, url: str, depth: int, parent_url: Optional[str] = None) -> Optional[PageContent]:
        """Extract content from a single page."""
        try:
            await self.page.goto(url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(self.config.delay_between_requests)
            
            title = await self.page.title()
            html_content = await self.page.content()
            markdown = self.html_converter.convert(html_content, url)
            
            self.order_counter += 1
            return PageContent(
                url=url,
                title=title,
                depth=depth,
                content_markdown=markdown,
                parent_url=parent_url,
                order_index=self.order_counter
            )
        
        except Exception as e:
            print(f"    [!] Failed to extract {url[:80]}: {str(e)[:50]}")
            return None
    
    async def discover_links_from_page(self, url: str) -> Set[str]:
        """Extract all valid links from current page."""
        try:
            links = await self.page.evaluate('''
                () => {
                    const links = [];
                    document.querySelectorAll('a[href]').forEach(a => {
                        const href = a.getAttribute('href');
                        if (href && !href.startsWith('javascript:')) {
                            links.push(href);
                        }
                    });
                    return links;
                }
            ''')
            
            discovered = set()
            for link in links:
                try:
                    absolute_url = urljoin(url, link)
                    filtered = self.url_filter.filter_url(absolute_url)
                    if filtered and filtered not in self.visited_urls:
                        discovered.add(filtered)
                except:
                    pass
            
            return discovered
        
        except Exception as e:
            print(f"    [!] Link discovery failed: {str(e)[:50]}")
            return set()
    
    async def crawl_auto(self, start_urls: Set[str]):
        """Auto mode: Breadth-first crawl of all discovered links."""
        print("\n" + "="*70)
        print("🤖 AUTO MODE - Automatic Crawling")
        print("="*70 + "\n")
        
        # Use breadth-first search for proper ordering
        queue = deque([(url, 0, None) for url in start_urls])
        
        while queue and len(self.visited_urls) < self.config.max_pages:
            url, depth, parent = queue.popleft()
            
            if url in self.visited_urls or depth > self.config.max_depth:
                if depth > self.config.max_depth:
                    print(f"  ⊘ Skipped {url[:70]} (max depth reached)")
                continue
            
            self.visited_urls.add(url)
            
            print(f"  📄 [{len(self.visited_urls)}/{self.config.max_pages}] "
                  f"Crawling (depth {depth}): {url[:70]}")
            
            content = await self.extract_page_content(url, depth, parent)
            if content:
                self.pages[url] = content
                
                # Discover child links
                child_links = await self.discover_links_from_page(url)
                for child_url in child_links:
                    if child_url not in self.visited_urls:
                        queue.append((child_url, depth + 1, url))
    
    async def crawl_manual(self):
        """Manual mode: User navigates, script extracts on demand."""
        print("\n" + "="*70)
        print("👤 MANUAL MODE - User-Controlled Extraction")
        print("="*70 + "\n")
        print("INSTRUCTIONS:")
        print("  • Navigate to pages you want to extract")
        print("  • Press ENTER to extract current page/tab")
        print("  • Type 'done' and press ENTER when finished\n")
        
        while True:
            user_input = input("⏳ Press ENTER to extract current page (or 'done' to finish): ").strip().lower()
            
            if user_input == 'done':
                break
            
            # Get all pages from all tabs
            pages_to_extract = self.context.pages
            
            for page_obj in pages_to_extract:
                try:
                    current_url = page_obj.url
                    normalized = self.url_filter.filter_url(current_url)
                    
                    if normalized and normalized not in self.visited_urls:
                        self.visited_urls.add(normalized)
                        
                        await page_obj.wait_for_load_state("networkidle")
                        title = await page_obj.title()
                        html_content = await page_obj.content()
                        markdown = self.html_converter.convert(html_content, normalized)
                        
                        self.order_counter += 1
                        self.pages[normalized] = PageContent(
                            url=normalized,
                            title=title,
                            depth=0,
                            content_markdown=markdown,
                            order_index=self.order_counter
                        )
                        
                        print(f"  ✓ Extracted: {title} ({normalized[:60]})")
                    elif normalized in self.visited_urls:
                        print(f"  ⊘ Already extracted: {current_url[:60]}")
                
                except Exception as e:
                    print(f"  [!] Error extracting page: {str(e)[:60]}")
            
            print(f"\n  📊 Total pages extracted: {len(self.pages)}\n")
    
    async def crawl_hybrid(self, discovered_urls: Set[str]):
        """Hybrid mode: Auto-discover then let user customize."""
        print("\n" + "="*70)
        print("🔀 HYBRID MODE - Discovered Links")
        print("="*70 + "\n")
        
        print(f"Found {len(discovered_urls)} URLs to crawl:\n")
        
        # Show discovered URLs
        for i, url in enumerate(sorted(discovered_urls)[:50], 1):
            print(f"  {i}. {url}")
        
        if len(discovered_urls) > 50:
            print(f"  ... and {len(discovered_urls) - 50} more")
        
        print("\n" + "="*70)
        print("OPTIONS:")
        print("  1. Press ENTER to crawl all discovered URLs")
        print("  2. Type URL(s) to add (one per line, empty line to finish)")
        print("  3. Navigate browser to additional pages, then press ENTER")
        print("="*70 + "\n")
        
        choice = input("Your choice (ENTER to proceed, or start typing URLs): ").strip()
        
        if choice:
            # User wants to add URLs
            additional_urls = set()
            additional_urls.add(choice)
            
            print("Enter additional URLs (empty line to finish):")
            while True:
                url_input = input("  URL: ").strip()
                if not url_input:
                    break
                filtered = self.url_filter.filter_url(url_input)
                if filtered:
                    additional_urls.add(filtered)
                    print(f"    ✓ Added: {filtered}")
            
            discovered_urls.update(additional_urls)
            print(f"\n✓ Total URLs to crawl: {len(discovered_urls)}\n")
        
        # Check if user navigated to additional pages
        for page_obj in self.context.pages:
            try:
                current_url = page_obj.url
                filtered = self.url_filter.filter_url(current_url)
                if filtered:
                    discovered_urls.add(filtered)
            except:
                pass
        
        # Now crawl all URLs
        await self.crawl_auto(discovered_urls)
    
    async def run(self) -> str:
        """Main execution flow."""
        try:
            print("\n" + "="*70)
            print("🚀 ADVANCED WEB SCRAPER STARTING")
            print("="*70)
            print(f"Mode: {self.config.mode.upper()}")
            print(f"Base URL: {self.config.base_url}")
            print(f"Max pages: {self.config.max_pages}")
            print(f"Max depth: {self.config.max_depth}")
            
            await self.start_browser()
            await self.authenticate()
            
            if self.config.mode == 'manual':
                await self.crawl_manual()
            
            else:
                # Auto and Hybrid modes: discover URLs first
                start_urls = set()
                
                # Try sitemap first
                if self.config.follow_sitemap:
                    print("\n" + "="*70)
                    print("🗺️  SITEMAP DISCOVERY")
                    print("="*70 + "\n")
                    
                    sitemap_url = await SitemapParser.discover_sitemap(self.page, self.config.base_url)
                    if sitemap_url:
                        sitemap_urls = await SitemapParser.parse_sitemap(self.page, sitemap_url, self.url_filter)
                        start_urls.update(sitemap_urls)
                        print(f"✓ Extracted {len(sitemap_urls)} URLs from sitemap\n")
                
                # Add current URL as starting point
                current_url = self.url_filter.filter_url(self.page.url)
                if current_url:
                    start_urls.add(current_url)
                
                # If no sitemap, discover from current page
                if not start_urls:
                    print("  → No sitemap found, discovering from current page...")
                    start_urls.add(self.config.base_url)
                
                if self.config.mode == 'auto':
                    await self.crawl_auto(start_urls)
                else:  # hybrid
                    await self.crawl_hybrid(start_urls)
            
            # Generate output
            print("\n" + "="*70)
            print("✅ CRAWL COMPLETE")
            print("="*70)
            print(f"\n📊 Statistics:")
            print(f"  • Pages scraped: {len(self.pages)}")
            print(f"  • URLs visited: {len(self.visited_urls)}")
            
            output = await self.generate_combined_markdown()
            return output
        
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
    
    async def generate_combined_markdown(self) -> str:
        """Generate single unified Markdown file."""
        if not self.pages:
            return "# No pages scraped\n"
        
        base_parsed = urlparse(self.config.base_url)
        base_title = base_parsed.netloc.replace('www.', '')
        
        output = []
        output.append(f"# Documentation: {base_title}")
        output.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
        output.append(f"_Mode: {self.config.mode.upper()}_\n")
        output.append(f"**Base URL:** {self.config.base_url}")
        output.append(f"**Total Pages:** {len(self.pages)}\n")
        output.append("---\n")
        
        # Sort pages by order_index (discovery order) then by depth
        sorted_pages = sorted(
            self.pages.items(),
            key=lambda x: (x[1].depth, x[1].order_index)
        )
        
        # Table of Contents
        output.append("## Table of Contents\n")
        for url, page in sorted_pages:
            indent = "  " * min(page.depth, 3)
            anchor = self._generate_anchor(page.title)
            output.append(f"{indent}• [{page.title or 'Untitled'}](#{anchor})")
        output.append("\n---\n")
        
        # Content sections
        for i, (url, page) in enumerate(sorted_pages, 1):
            heading_level = min(2 + page.depth, 5)
            heading = "#" * heading_level
            
            output.append(f"\n{heading} {page.title or 'Untitled'}")
            output.append(f"_URL: [{url}]({url})_")
            if page.parent_url:
                output.append(f"_Parent: {page.parent_url}_")
            output.append(f"_Depth: {page.depth} | Order: {page.order_index}_\n")
            
            output.append(page.content_markdown)
            output.append("\n")
            
            if i < len(sorted_pages):
                output.append("---\n")
        
        # Metadata footer
        output.append("\n---\n")
        output.append("## Metadata\n")
        output.append(f"- **Total Pages:** {len(self.pages)}")
        output.append(f"- **Mode:** {self.config.mode.upper()}")
        output.append(f"- **Max Depth:** {self.config.max_depth}")
        output.append(f"- **Generated:** {datetime.now().isoformat()}\n")
        
        return "\n".join(output)
    
    @staticmethod
    def _generate_anchor(title: str) -> str:
        """Generate Markdown anchor."""
        if not title:
            return "untitled"
        anchor = title.lower()
        anchor = re.sub(r'[^\w\s-]', '', anchor)
        anchor = re.sub(r'[-\s]+', '-', anchor)
        return anchor.strip('-')
    
    async def save_markdown(self, output: str) -> Path:
        """Save Markdown to file."""
        if not self.config.output_file:
            base_name = urlparse(self.config.base_url).netloc.replace('.', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.config.output_file = f"{base_name}_{self.config.mode}_{timestamp}.md"
        
        output_path = Path(self.config.output_file)
        output_path.write_text(output, encoding='utf-8')
        
        print(f"\n✅ Saved to: {output_path.absolute()}")
        print(f"   Size: {len(output):,} characters")
        print(f"   File: {output_path.name}\n")
        
        return output_path


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

async def main():
    """CLI entry point."""
    parser = ArgumentParser(
        description="Advanced web scraper for authenticated organizational websites",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Auto mode (fully automatic):
    python main.py --url https://docs.myorg.com --mode auto
  
  Manual mode (you navigate, script extracts):
    python main.py --url https://internal.myorg.com --mode manual
  
  Hybrid mode (auto-discover, then customize):
    python main.py --url https://wiki.myorg.com --mode hybrid
  
  Use existing Chrome (avoids playwright browser download):
    python main.py --url https://myorg.com --mode auto --use-chrome auto
    python main.py --url https://myorg.com --mode auto --use-chrome "/path/to/chrome"
        """
    )
    
    parser.add_argument("--url", required=True, 
                       help="Base URL to scrape")
    parser.add_argument("--mode", choices=['auto', 'manual', 'hybrid'], default='auto',
                       help="Crawling mode (default: auto)")
    parser.add_argument("--max-pages", type=int, default=100,
                       help="Maximum pages to scrape (default: 100)")
    parser.add_argument("--max-depth", type=int, default=6,
                       help="Maximum crawl depth (default: 6)")
    parser.add_argument("--output", 
                       help="Output filename (optional, auto-generated if not provided)")
    parser.add_argument("--use-chrome",
                       help="Path to existing Chrome/Chromium, or 'auto' to auto-detect")
    parser.add_argument("--no-sitemap", action='store_true',
                       help="Don't try to use sitemap.xml")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="Delay between requests in seconds (default: 0.5)")
    
    args = parser.parse_args()
    
    # Create config
    config = CrawlConfig(
        mode=args.mode,
        base_url=args.url,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        output_file=args.output,
        use_existing_chrome=args.use_chrome,
        follow_sitemap=not args.no_sitemap,
        delay_between_requests=args.delay
    )
    
    # Run scraper
    scraper = AuthenticatedScraper(config)
    markdown_output = await scraper.run()
    await scraper.save_markdown(markdown_output)
    
    print("✨ Done! Your documentation is ready for LLM ingestion.\n")


if __name__ == "__main__":
    asyncio.run(main())
