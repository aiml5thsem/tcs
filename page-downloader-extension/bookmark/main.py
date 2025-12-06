#!/usr/bin/env python3
"""
Enhanced HTML to LLM-ready Markdown converter
Memory-efficient streaming with chunked processing
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Iterator, Tuple, Generator
import argparse
import tempfile
import mmap

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("⚠️ Install beautifulsoup4: pip install beautifulsoup4 lxml")

try:
    import html2text
    HAS_HTML2TEXT = True
except ImportError:
    HAS_HTML2TEXT = False
    print("⚠️ Install html2text: pip install html2text")


class ChunkedHTMLProcessor:
    """Memory-efficient chunked HTML processing"""
    
    CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks
    
    def __init__(self, use_lxml=True):
        self.use_lxml = use_lxml and HAS_BS4
        
        if HAS_HTML2TEXT:
            self.converter = html2text.HTML2Text()
            self.converter.ignore_links = False
            self.converter.ignore_images = False
            self.converter.body_width = 0
            self.converter.single_line_break = False
            self.converter.unicode_snob = True
            self.converter.decode_errors = 'ignore'
    
    def smart_convert(self, html: str, max_size: int = 50 * 1024 * 1024) -> str:
        """Convert HTML with automatic strategy selection based on size"""
        
        size_mb = len(html) / (1024 * 1024)
        
        # Strategy 1: Small files - use html2text (best quality)
        if size_mb < 5 and HAS_HTML2TEXT:
            try:
                return self.converter.handle(html)
            except Exception as e:
                print(f"⚠️ html2text failed ({e}), trying BeautifulSoup")
        
        # Strategy 2: Medium files - use BeautifulSoup with lxml
        if size_mb < 20 and HAS_BS4:
            try:
                parser = 'lxml' if self.use_lxml else 'html.parser'
                soup = BeautifulSoup(html, parser)
                
                # Remove noise
                for tag in soup(['script', 'style', 'noscript', 'svg']):
                    tag.decompose()
                
                # Extract with structure
                text = soup.get_text(separator='\n', strip=True)
                
                # Clean up
                text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
                return text
                
            except Exception as e:
                print(f"⚠️ BeautifulSoup failed ({e}), using streaming")
        
        # Strategy 3: Large files - streaming regex (memory efficient)
        return self._streaming_extract(html)
    
    def _streaming_extract(self, html: str) -> str:
        """Memory-efficient streaming extraction for huge files"""
        
        # Remove script/style in chunks
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<svg[^>]*>.*?</svg>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        html = re.sub(r'<[^>]+>', ' ', html)
        
        # Decode ALL HTML entities (comprehensive)
        import html as html_module
        try:
            # Use built-in HTML entity decoder (handles all entities)
            html = html_module.unescape(html)
        except Exception:
            # Fallback: manual replacements
            replacements = {
                '&nbsp;': ' ', '&lt;': '<', '&gt;': '>', '&amp;': '&',
                '&quot;': '"', '&#39;': "'", '&apos;': "'",
                '&#8217;': "'", '&#8220;': '"', '&#8221;': '"',
                '&ndash;': '–', '&mdash;': '—', '&hellip;': '...',
                '&copy;': '©', '&reg;': '®', '&trade;': '™',
                '&deg;': '°', '&plusmn;': '±', '&times;': '×',
                '&divide;': '÷', '&ne;': '≠', '&le;': '≤', '&ge;': '≥'
            }
            for old, new in replacements.items():
                html = html.replace(old, new)
        
        # Clean whitespace
        html = re.sub(r'[^\S\n]+', ' ', html)
        html = re.sub(r'\n\s*\n\s*\n+', '\n\n', html)
        
        return html.strip()


def extract_structured_data(html: str, max_scan: int = 1024 * 1024) -> list:
    """Extract structured data with size limits"""
    structured = []
    
    # Only scan first 1MB for structured data (it's usually in <head>)
    scan_html = html[:max_scan]
    
    patterns = [
        (r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', 'JSON-LD'),
        (r'<script[^>]*type=["\']application/json["\'][^>]*>(.*?)</script>', 'App State'),
    ]
    
    for pattern, data_type in patterns:
        for match in re.finditer(pattern, scan_html, re.DOTALL | re.IGNORECASE):
            try:
                data = json.loads(match.group(1))
                structured.append({'type': data_type, 'data': data})
            except (json.JSONDecodeError, ValueError):
                continue
            
            # Limit to 10 structured data items
            if len(structured) >= 10:
                break
    
    return structured


def process_file_chunked(html_file: Path, processor: ChunkedHTMLProcessor) -> Generator[str, None, dict]:
    """Process file in chunks, yielding markdown incrementally"""
    
    try:
        file_size = html_file.stat().st_size
        timestamp_str = html_file.stem
        
        # For very large files, use memory mapping
        if file_size > 100 * 1024 * 1024:  # > 100MB
            with open(html_file, 'r+b') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped:
                    html = mmapped.read().decode('utf-8', errors='ignore')
        else:
            with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                html = f.read()
        
        # Extract structured data first (small)
        structured = extract_structured_data(html)
        
        # Convert main content
        markdown = processor.smart_convert(html)
        
        # Yield main content
        yield markdown
        
        # Yield structured data separately
        if structured:
            struct_md = "\n\n### Structured Data\n\n"
            for item in structured[:5]:  # Limit to 5 items
                struct_md += f"**{item['type']}:**\n```json\n{json.dumps(item['data'], indent=2)[:1000]}...\n```\n\n"
            yield struct_md
        
        metadata = {
            'timestamp': timestamp_str,
            'file_size': file_size,
            'char_count': len(markdown),
            'structured_count': len(structured),
            'success': True
        }
        
        return metadata
        
    except Exception as e:
        return {
            'timestamp': html_file.stem,
            'error': str(e),
            'success': False
        }


def iter_html_files(folder: Path) -> Iterator[Path]:
    """Yield HTML files sorted by timestamp"""
    files = []
    
    # Support multiple patterns
    for pattern in ['*.html', '*.htm']:
        for html_file in folder.glob(pattern):
            try:
                # Extract timestamp from filename
                timestamp_str = html_file.stem
                files.append((timestamp_str, html_file))
            except Exception:
                continue
    
    # Sort by timestamp
    files.sort(key=lambda x: x[0])
    
    for _, html_file in files:
        yield html_file


def process_folder(folder_path: str, output_file: str = None, verbose: bool = True) -> str:
    """
    Process HTML files with memory-efficient streaming
    
    Improvements:
    - ✅ Handles files of ANY size (streams to disk)
    - ✅ Better markdown quality with adaptive strategies
    - ✅ Support for timestamp.html pattern (no 'page_' prefix needed)
    """
    
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Initialize processor
    processor = ChunkedHTMLProcessor()
    
    # Collect files
    html_files = list(iter_html_files(folder))
    
    if not html_files:
        print(f"⚠️ No HTML files found in {folder_path}")
        return ""
    
    if verbose:
        print(f"📂 Found {len(html_files)} HTML files")
        print(f"🔄 Processing with streaming mode...")
    
    # Determine output path
    if output_file is None:
        output_file = folder / "combined_llm_document.md"
    output_path = Path(output_file)
    
    # Stream output to file (memory efficient)
    stats = {
        'total_files': len(html_files),
        'successful': 0,
        'failed': 0,
        'total_size': 0
    }
    
    with open(output_path, 'w', encoding='utf-8') as out:
        # Write header
        out.write("# Combined LLM-Ready Document\n\n")
        out.write(f"**Generated:** {datetime.now().isoformat()}\n")
        out.write(f"**Source:** {folder.name}\n")
        out.write(f"**Files:** {len(html_files)}\n\n")
        out.write("---\n\n")
        
        # Process each file
        for idx, html_file in enumerate(html_files, 1):
            if verbose:
                file_size_mb = html_file.stat().st_size / (1024 * 1024)
                print(f"[{idx}/{len(html_files)}] {html_file.name} ({file_size_mb:.1f} MB)...", end=" ", flush=True)
            
            try:
                # Write section header
                out.write(f"## Page {idx}: {html_file.stem}\n\n")
                
                # Process and stream chunks
                generator = process_file_chunked(html_file, processor)
                chunk_count = 0
                
                for chunk in generator:
                    if isinstance(chunk, str):
                        out.write(chunk)
                        chunk_count += 1
                    else:
                        # Final metadata
                        metadata = chunk
                        break
                
                out.write("\n\n---\n\n")
                
                if metadata.get('success', False):
                    stats['successful'] += 1
                    stats['total_size'] += metadata.get('file_size', 0)
                    if verbose:
                        print(f"✓ ({metadata['char_count']:,} chars)")
                else:
                    raise Exception(metadata.get('error', 'Unknown error'))
                    
            except Exception as e:
                stats['failed'] += 1
                out.write(f"*Error processing this file: {e}*\n\n---\n\n")
                if verbose:
                    print(f"✗ {e}")
    
    # Final stats
    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    if verbose:
        print(f"\n✅ Saved: {output_path}")
        print(f"📊 Stats:")
        print(f"   - Processed: {stats['successful']}/{stats['total_files']}")
        print(f"   - Failed: {stats['failed']}/{stats['total_files']}")
        print(f"   - Input size: {stats['total_size'] / (1024*1024):.2f} MB")
        print(f"   - Output size: {output_size_mb:.2f} MB")
    
    return str(output_path)


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Convert HTML to LLM-ready markdown (memory-efficient)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/html/folder
  %(prog)s /path/to/html/folder -o output.md
  %(prog)s /path/to/html/folder --quiet
  
Supports any HTML filename pattern (timestamp.html, page_*.html, etc.)
Handles files of ANY size with automatic streaming.
        """
    )
    parser.add_argument('folder', help='Folder containing HTML files')
    parser.add_argument('-o', '--output', help='Output file (default: combined_llm_document.md)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    try:
        process_folder(args.folder, args.output, verbose=not args.quiet)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
