"""
Production-ready web scraper for authenticated organizational apps.
Extracts complete website into single Markdown file with hierarchical structure.

Usage:
    python web_scraper.py --url https://myapp.internal.com
    # Or with config:
    python web_scraper.py --url https://myapp.internal.com --max-pages 200 --max-depth 6
"""

import asyncio
import re
import json
import html2text
from pathlib import Path
from urllib.parse import urljoin, urlparse
from datetime import datetime
from typing import Set, Dict, List, Optional
from dataclasses import dataclass, asdict
from argparse import ArgumentParser

from playwright.async_api import async_playwright, Page, Browser, BrowserContext


@dataclass
class PageContent:
    """Represents a scraped page with metadata."""
    url: str
    title: str
    depth: int
    content_markdown: str
    found_at: str = None
    
    def __post_init__(self):
        if self.found_at is None:
            self.found_at = datetime.now().isoformat()


class URLFilter:
    """Intelligent URL filtering for crawling."""
    
    # Patterns to SKIP
    SKIP_PATTERNS = [
        r'logout', r'signout', r'sign-out', r'/auth/logout',
        r'\.pdf$', r'\.zip$', r'\.exe$', r'\.dmg$',
        r'\.xlsx$', r'\.csv$', r'\.json$', r'\.xml$',
        r'mailto:', r'tel:', r'javascript:', r'#',
        r'/api/', r'/v\d+/api',
        r'download', r'export', r'print',
    ]
    
    # URL components to normalize
    SKIP_QUERY_PARAMS = ['sessionid', 'utm_', 'tracking', 'ref=']
    
    def __init__(self, base_url: str):
        """Initialize filter with base URL for origin checking."""
        parsed = urlparse(base_url)
        self.base_origin = f"{parsed.scheme}://{parsed.netloc}"
    
    def is_same_origin(self, url: str) -> bool:
        """Check if URL matches base origin."""
        try:
            parsed = urlparse(url)
            origin = f"{parsed.scheme}://{parsed.netloc}"
            return origin == self.base_origin
        except:
            return False
    
    def should_skip(self, url: str) -> bool:
        """Check if URL matches skip patterns."""
        url_lower = url.lower()
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, url_lower):
                return True
        return False
    
    def normalize_url(self, url: str) -> str:
        """Remove tracking params and normalize URL."""
        parsed = urlparse(url)
        
        # Remove fragment
        url_without_fragment = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{parsed.query}"
        
        # Remove tracking query params
        if parsed.query:
            params = parsed.query.split('&')
            clean_params = []
            for param in params:
                skip = False
                for skip_param in self.SKIP_QUERY_PARAMS:
                    if param.lower().startswith(skip_param):
                        skip = True
                        break
                if not skip:
                    clean_params.append(param)
            
            query_string = '&'.join(clean_params)
            url_without_fragment = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if query_string:
                url_without_fragment += f"?{query_string}"
        
        # Remove trailing slash for consistency
        url_without_fragment = url_without_fragment.rstrip('/')
        
        return url_without_fragment
    
    def filter_url(self, url: str) -> Optional[str]:
        """
        Apply all filters. Return normalized URL if valid, None if should skip.
        """
        if not url or not url.startswith('http'):
            return None
        
        if self.should_skip(url):
            return None
        
        if not self.is_same_origin(url):
            return None
        
        normalized = self.normalize_url(url)
        return normalized


class HTMLToMarkdown:
    """Convert HTML to LLM-ready Markdown."""
    
    def __init__(self):
        self.converter = html2text.HTML2Text()
        self.converter.body_width = 0  # Don't wrap lines
        self.converter.ignore_links = False
        self.converter.ignore_images = False
        self.converter.ignore_emphasis = False
        self.converter.ignore_tables = False
        self.converter.unicode_snob = True
        self.converter.use_automatic_links = True
    
    def convert(self, html: str) -> str:
        """Convert HTML to Markdown."""
        try:
            # Clean up common problematic patterns
            html = self._clean_html(html)
            markdown = self.converter.handle(html)
            return self._clean_markdown(markdown)
        except Exception as e:
            print(f"[!] Markdown conversion error: {e}")
            return f"```\n{html}\n```\n"
    
    def _clean_html(self, html: str) -> str:
        """Pre-process HTML for better conversion."""
        # Remove script and style elements
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
        
        # Remove common junk
        html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL)
        html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL)
        
        return html
    
    def _clean_markdown(self, markdown: str) -> str:
        """Post-process Markdown for quality."""
        # Remove excessive blank lines
        markdown = re.sub(r'\n\n\n+', '\n\n', markdown)
        
        # Clean up trailing whitespace
        markdown = '\n'.join(line.rstrip() for line in markdown.split('\n'))
        
        # Remove lines with only whitespace
        lines = [line for line in markdown.split('\n') if line.strip() or line == '']
        markdown = '\n'.join(lines)
        
        return markdown.strip()


class AuthenticatedScraper:
    """Main scraper orchestrating authentication and crawling."""
    
    def __init__(self, base_url: str, max_pages: int = 100, max_depth: int = 6):
        self.base_url = base_url
        self.max_pages = max_pages
        self.max_depth = max_depth
        
        self.visited_urls: Set[str] = set()
        self.pages: Dict[str, PageContent] = {}
        self.url_filter = URLFilter(base_url)
        self.html_converter = HTMLToMarkdown()
        
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
    
    async def start_browser(self):
        """Launch browser in headed mode for manual authentication."""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=False)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        
        print("\n" + "="*70)
        print("🌐 AUTHENTICATION PHASE")
        print("="*70)
        print(f"\n✓ Browser launched (Chromium, headed mode)")
        print(f"✓ Navigating to: {self.base_url}\n")
    
    async def navigate_to_base(self):
        """Navigate to base URL and wait for authentication."""
        await self.page.goto(self.base_url, wait_until="networkidle", timeout=60000)
        
        print("\n📋 NEXT STEPS:")
        print("-" * 70)
        print("1. The browser is now on your base URL")
        print("2. Authenticate using your credentials (SSO, OAuth, 2FA, etc.)")
        print("3. Navigate to any important pages you want included")
        print("4. Once ready, return to the terminal and press ENTER")
        print("-" * 70 + "\n")
        
        # Wait for user to authenticate
        input("⏳ Press ENTER when authentication is complete...")
        
        # Get current URL and page title
        current_url = self.page.url
        title = await self.page.title()
        
        print(f"\n✓ Current URL: {current_url}")
        print(f"✓ Page title: {title}")
        print("✓ Starting crawl with authenticated session...\n")
    
    async def extract_page_content(self, url: str, depth: int) -> Optional[PageContent]:
        """Extract HTML and convert to Markdown."""
        try:
            # Navigate to page
            await self.page.goto(url, wait_until="networkidle", timeout=30000)
            await self.page.wait_for_load_state("networkidle")
            
            # Get page title and content
            title = await self.page.title()
            html_content = await self.page.content()
            
            # Convert to Markdown
            markdown = self.html_converter.convert(html_content)
            
            return PageContent(
                url=url,
                title=title,
                depth=depth,
                content_markdown=markdown
            )
        
        except Exception as e:
            print(f"  [!] Failed to extract {url}: {str(e)[:60]}")
            return None
    
    async def discover_links(self, url: str) -> Set[str]:
        """Extract all valid internal links from current page."""
        try:
            # Use Playwright to extract links
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
            
            # Convert relative to absolute and filter
            discovered = set()
            for link in links:
                try:
                    absolute_url = urljoin(url, link)
                    filtered = self.url_filter.filter_url(absolute_url)
                    if filtered:
                        discovered.add(filtered)
                except:
                    pass
            
            return discovered
        
        except Exception as e:
            print(f"  [!] Failed to discover links from {url}: {e}")
            return set()
    
    async def crawl(self, url: str, depth: int = 0):
        """Recursively crawl pages."""
        # Check limits
        if depth > self.max_depth:
            print(f"  ⊘ Skipped {url} (max depth {self.max_depth} reached)")
            return
        
        if len(self.visited_urls) >= self.max_pages:
            print(f"\n✓ Reached max pages limit ({self.max_pages})")
            return
        
        if url in self.visited_urls:
            return
        
        # Mark as visited
        self.visited_urls.add(url)
        
        # Extract content
        print(f"  📄 [{len(self.visited_urls)}/{self.max_pages}] Crawling (depth {depth}): {url[:70]}")
        
        content = await self.extract_page_content(url, depth)
        if not content:
            return
        
        self.pages[url] = content
        
        # Discover and crawl child links
        discovered_links = await self.discover_links(url)
        
        for link in discovered_links:
            if link not in self.visited_urls and len(self.visited_urls) < self.max_pages:
                await self.crawl(link, depth + 1)
    
    async def run(self) -> str:
        """Main execution flow."""
        try:
            print("\n" + "="*70)
            print("🔐 STARTING AUTHENTICATED WEB SCRAPER")
            print("="*70)
            
            # Start browser
            await self.start_browser()
            
            # Navigate and authenticate
            await self.navigate_to_base()
            
            # Start crawling from current page
            current_url = self.page.url
            print(f"\n" + "="*70)
            print("🕷️  CRAWLING PHASE")
            print("="*70 + "\n")
            print(f"Starting URL: {current_url}")
            print(f"Max pages: {self.max_pages}")
            print(f"Max depth: {self.max_depth}\n")
            
            await self.crawl(current_url, depth=0)
            
            print(f"\n" + "="*70)
            print("✓ CRAWL COMPLETE")
            print("="*70)
            print(f"\n📊 Statistics:")
            print(f"  • Pages scraped: {len(self.pages)}")
            print(f"  • URLs visited: {len(self.visited_urls)}")
            
            # Generate combined Markdown
            output = await self.generate_combined_markdown()
            
            return output
        
        except Exception as e:
            print(f"\n[ERROR] {e}")
            raise
        
        finally:
            if self.browser:
                await self.browser.close()
    
    async def generate_combined_markdown(self) -> str:
        """Generate single unified Markdown file with all pages."""
        
        if not self.pages:
            return "# No pages scraped\n"
        
        # Get base URL title
        base_parsed = urlparse(self.base_url)
        base_title = base_parsed.netloc.replace('www.', '').replace('.internal', '')
        
        # Start with metadata
        output = []
        output.append(f"# Documentation for {base_title}")
        output.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
        output.append(f"**Base URL:** {self.base_url}")
        output.append(f"**Total Pages:** {len(self.pages)}\n")
        output.append("---\n")
        
        # Table of contents
        output.append("## Table of Contents\n")
        for i, (url, page) in enumerate(self.pages.items(), 1):
            indent = "  " * min(page.depth, 3)
            output.append(f"{indent}• [{page.title or url}](#{self._generate_anchor(page.title)})")
        output.append("\n---\n")
        
        # Sort pages by depth (breadth-first) then by URL
        sorted_pages = sorted(self.pages.items(), key=lambda x: (x[1].depth, x[0]))
        
        # Generate content sections
        for i, (url, page) in enumerate(sorted_pages, 1):
            # Generate appropriate heading level based on depth
            heading_level = min(2 + page.depth, 5)  # H2 to H5
            heading = "#" * heading_level
            
            # Create section header
            output.append(f"\n{heading} {page.title or 'Untitled'}")
            output.append(f"_URL: [{url}]({url})_\n")
            
            # Add page content
            output.append(page.content_markdown)
            output.append("\n")
            
            # Add separator between pages
            if i < len(sorted_pages):
                output.append("---\n")
        
        # Final metadata footer
        output.append("\n---\n")
        output.append("## Metadata\n")
        output.append(f"- **Total Pages Scraped:** {len(self.pages)}\n")
        output.append(f"- **Max Depth:** {self.max_depth}\n")
        output.append(f"- **Generation Time:** {datetime.now().isoformat()}\n")
        
        return "\n".join(output)
    
    @staticmethod
    def _generate_anchor(title: str) -> str:
        """Generate valid Markdown anchor from title."""
        if not title:
            return "untitled"
        anchor = title.lower()
        anchor = re.sub(r'[^\w\s-]', '', anchor)
        anchor = re.sub(r'[-\s]+', '-', anchor)
        return anchor.strip('-')
    
    async def save_markdown(self, output: str, output_file: str = None) -> Path:
        """Save Markdown to file."""
        if not output_file:
            base_name = urlparse(self.base_url).netloc.replace('.', '_').replace(':', '_')
            output_file = f"{base_name}_documentation.md"
        
        output_path = Path(output_file)
        output_path.write_text(output, encoding='utf-8')
        
        print(f"\n✅ Markdown saved: {output_path.absolute()}")
        print(f"   Size: {len(output):,} characters")
        print(f"   File: {output_path.name}")
        
        return output_path


async def main():
    """CLI entry point."""
    parser = ArgumentParser(description="Authenticated web scraper for organizational apps")
    parser.add_argument("--url", required=True, help="Base URL to scrape (e.g., https://myapp.internal.com)")
    parser.add_argument("--max-pages", type=int, default=100, help="Maximum pages to scrape (default: 100)")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum crawl depth (default: 6)")
    parser.add_argument("--output", help="Output filename (optional)")
    
    args = parser.parse_args()
    
    # Create and run scraper
    scraper = AuthenticatedScraper(
        base_url=args.url,
        max_pages=args.max_pages,
        max_depth=args.max_depth
    )
    
    # Execute
    markdown_output = await scraper.run()
    
    # Save to file
    await scraper.save_markdown(markdown_output, args.output)
    
    print("\n✨ Done! Your documentation is ready for LLM ingestion.")


if __name__ == "__main__":
    asyncio.run(main())
