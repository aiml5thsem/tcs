#!/usr/bin/env python3
"""
4_create_local_db.py - Universal Multi-Format Document Indexer
Supports: MD, TXT, PY, HTML, MDX, and more with smart deduplication
"""

import os
import sys
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import argparse
import logging
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from tqdm import tqdm

# Optional imports for advanced formats
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import html_to_markdown
    HAS_HTML2MD = True
except ImportError:
    HAS_HTML2MD = False

# Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 150
CODE_CHUNK_SIZE = 600
CODE_OVERLAP = 100

# Deduplication threshold (similarity)
DEDUP_THRESHOLD = 0.95

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ContentDeduplicator:
    """Intelligent content deduplication using text similarity"""
    
    def __init__(self, threshold=DEDUP_THRESHOLD):
        self.threshold = threshold
        self.seen_hashes = set()
        self.seen_content = []
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common UI elements
        text = re.sub(r'(Skip to|Back to top|Navigation|Menu|Footer|Sidebar)', '', text, flags=re.IGNORECASE)
        return text.strip().lower()
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate"""
        normalized = self._normalize_text(text)
        
        if not normalized or len(normalized) < 50:
            return True  # Too short, likely not useful
        
        # Quick hash check
        text_hash = hashlib.md5(normalized.encode()).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        
        # Similarity check against recent content
        for seen_text in self.seen_content[-100:]:  # Check last 100 chunks
            similarity = self._text_similarity(normalized, seen_text)
            if similarity > self.threshold:
                return True
        
        # Not a duplicate
        self.seen_hashes.add(text_hash)
        self.seen_content.append(normalized)
        
        # Limit memory usage
        if len(self.seen_content) > 1000:
            self.seen_content = self.seen_content[-500:]
        
        return False


class UniversalDocumentParser:
    """Parse multiple document formats into clean markdown"""
    
    def __init__(self):
        self.deduplicator = ContentDeduplicator()
    
    def parse_markdown(self, content: str) -> str:
        """Parse markdown (already in good format)"""
        return content
    
    def parse_text(self, content: str) -> str:
        """Parse plain text"""
        return content
    
    def parse_python(self, content: str, filepath: Path) -> str:
        """Parse Python file - extract docstrings and code"""
        result = [f"# Source: {filepath.name}\n"]
        
        # Extract module docstring
        module_doc_match = re.search(r'^"""(.*?)"""', content, re.DOTALL | re.MULTILINE)
        if module_doc_match:
            result.append(f"## Module Documentation\n{module_doc_match.group(1).strip()}\n")
        
        # Extract classes with docstrings
        class_pattern = r'class\s+(\w+).*?:\s*"""(.*?)"""'
        for match in re.finditer(class_pattern, content, re.DOTALL):
            class_name, doc = match.groups()
            result.append(f"## Class: {class_name}\n{doc.strip()}\n")
        
        # Extract functions with docstrings
        func_pattern = r'def\s+(\w+)\s*\([^)]*\):\s*"""(.*?)"""'
        for match in re.finditer(func_pattern, content, re.DOTALL):
            func_name, doc = match.groups()
            result.append(f"## Function: {func_name}\n{doc.strip()}\n")
        
        # Include actual code
        result.append("\n## Code\n```python\n" + content + "\n```\n")
        
        return "\n".join(result)
    
    def parse_html(self, content: str, filepath: Path) -> str:
        """Parse HTML to clean markdown with deduplication"""
        if not HAS_BS4:
            logger.warning(f"BeautifulSoup not installed, skipping HTML parsing for {filepath}")
            return ""
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                           'aside', 'iframe', 'noscript']):
            element.decompose()
        
        # Remove common UI patterns
        for element in soup.find_all(class_=re.compile(r'(sidebar|menu|navigation|breadcrumb|footer|header|cookie|banner)', re.I)):
            element.decompose()
        
        for element in soup.find_all(id=re.compile(r'(sidebar|menu|navigation|footer|header)', re.I)):
            element.decompose()
        
        # Try html-to-markdown if available
        if HAS_HTML2MD:
            try:
                markdown = html_to_markdown.convert(str(soup))
                return markdown
            except Exception as e:
                logger.debug(f"html_to_markdown failed: {e}, using fallback")
        
        # Fallback: extract text intelligently
        result = [f"# {filepath.stem}\n"]
        
        # Extract title
        title = soup.find('title')
        if title:
            result.append(f"## {title.get_text().strip()}\n")
        
        # Extract main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main', re.I))
        
        if main_content:
            # Extract headers and paragraphs
            for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'pre', 'code']):
                text = element.get_text().strip()
                
                if not text or self.deduplicator.is_duplicate(text):
                    continue
                
                if element.name.startswith('h'):
                    level = int(element.name[1])
                    result.append(f"\n{'#' * level} {text}\n")
                elif element.name == 'pre' or element.name == 'code':
                    result.append(f"\n```\n{text}\n```\n")
                else:
                    result.append(f"{text}\n")
        else:
            # No main content found, extract all text
            text = soup.get_text(separator='\n', strip=True)
            # Filter out duplicate lines
            lines = []
            for line in text.split('\n'):
                line = line.strip()
                if line and not self.deduplicator.is_duplicate(line):
                    lines.append(line)
            result.append('\n'.join(lines))
        
        return "\n".join(result)
    
    def parse_mdx(self, content: str) -> str:
        """Parse MDX (Markdown + JSX) - strip JSX, keep markdown"""
        # Remove JSX components but keep their text content
        content = re.sub(r'<(\w+)[^>]*>(.*?)</\1>', r'\2', content, flags=re.DOTALL)
        # Remove self-closing JSX
        content = re.sub(r'<\w+[^>]*/>', '', content)
        # Remove import statements
        content = re.sub(r'^import\s+.*?from\s+.*?$', '', content, flags=re.MULTILINE)
        # Remove export statements
        content = re.sub(r'^export\s+.*?$', '', content, flags=re.MULTILINE)
        
        return content
    
    def parse_file(self, filepath: Path) -> Optional[str]:
        """Parse any supported file format"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return None
        
        suffix = filepath.suffix.lower()
        
        if suffix in ['.md', '.markdown']:
            return self.parse_markdown(content)
        elif suffix == '.txt':
            return self.parse_text(content)
        elif suffix == '.py':
            return self.parse_python(content, filepath)
        elif suffix in ['.html', '.htm']:
            return self.parse_html(content, filepath)
        elif suffix == '.mdx':
            return self.parse_mdx(content)
        else:
            # Try as text
            return self.parse_text(content)


class SmartChunker:
    """Content-aware chunking"""
    
    def __init__(self, doc_size=DEFAULT_CHUNK_SIZE, doc_overlap=DEFAULT_OVERLAP,
                 code_size=CODE_CHUNK_SIZE, code_overlap=CODE_OVERLAP):
        self.doc_size = doc_size
        self.doc_overlap = doc_overlap
        self.code_size = code_size
        self.code_overlap = code_overlap
    
    def detect_content_type(self, text: str) -> str:
        """Detect if primarily code or docs"""
        code_indicators = len(re.findall(r'```|def |class |function |import |from ', text))
        total_lines = len(text.split('\n'))
        
        if total_lines == 0:
            return 'docs'
        
        ratio = code_indicators / total_lines
        return 'code' if ratio > 0.15 else 'docs'
    
    def chunk_by_headers(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Chunk by markdown headers"""
        sections = re.split(r'(\n#{1,6}\s+.+\n)', text)
        
        chunks = []
        current = ""
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            if len(current) + len(section) > max_size and current:
                chunks.append(current.strip())
                # Add overlap
                if overlap > 0:
                    overlap_text = current[-overlap:] if len(current) > overlap else current
                    current = overlap_text + "\n\n" + section
                else:
                    current = section
            else:
                current += "\n\n" + section if current else section
        
        if current.strip():
            chunks.append(current.strip())
        
        return chunks
    
    def chunk_code(self, text: str) -> List[str]:
        """Chunk code by logical units"""
        # Split by function/class definitions
        boundaries = [m.start() for m in re.finditer(r'\n(def |class |async def )', text)]
        boundaries.append(len(text))
        
        if len(boundaries) <= 1:
            return self.simple_chunk(text, self.code_size, self.code_overlap)
        
        chunks = []
        for i in range(len(boundaries) - 1):
            chunk = text[boundaries[i]:boundaries[i+1]].strip()
            if chunk:
                if len(chunk) > self.code_size * 2:
                    chunks.extend(self.simple_chunk(chunk, self.code_size, self.code_overlap))
                else:
                    chunks.append(chunk)
        
        return chunks
    
    def simple_chunk(self, text: str, size: int, overlap: int) -> List[str]:
        """Simple size-based chunking with sentence awareness"""
        if len(text) <= size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + size
            
            if end < len(text):
                # Try to break at sentence
                for sep in ['. ', '.\n', '\n\n', '! ', '? ']:
                    last = text[start:end].rfind(sep)
                    if last != -1:
                        end = start + last + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def chunk(self, text: str) -> List[Dict]:
        """Main chunking method"""
        content_type = self.detect_content_type(text)
        
        if content_type == 'code':
            chunks = self.chunk_code(text)
        else:
            chunks = self.chunk_by_headers(text, self.doc_size, self.doc_overlap)
        
        return [{'text': c, 'type': content_type} for c in chunks if c.strip()]


def sanitize_collection_name(name: str) -> str:
    """Create valid collection name"""
    name = Path(name).stem
    name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
    name = name.lower()
    if name and not (name[0].isalpha() or name[0] == '_'):
        name = '_' + name
    return name or '_collection'


def create_collection(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    filepath: Path,
    parser: UniversalDocumentParser,
    chunker: SmartChunker,
    force: bool = False
) -> Optional[str]:
    """Create collection from file"""
    
    collection_name = sanitize_collection_name(filepath.name)
    
    logger.info(f"Processing: {filepath.name} → {collection_name}")
    
    # Check if exists
    collections = client.get_collections().collections
    if any(c.name == collection_name for c in collections):
        if force:
            logger.info(f"  Deleting existing...")
            client.delete_collection(collection_name)
        else:
            logger.info(f"  Already exists (use --force)")
            return collection_name
    
    # Parse file
    content = parser.parse_file(filepath)
    if not content or not content.strip():
        logger.warning(f"  No content extracted")
        return None
    
    # Chunk
    chunks = chunker.chunk(content)
    if not chunks:
        logger.warning(f"  No chunks created")
        return None
    
    logger.info(f"  Created {len(chunks)} chunks")
    
    # Generate embeddings
    texts = [c['text'] for c in chunks]
    embeddings = list(embedding_model.embed(texts))
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )
    
    # Create points
    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = int(hashlib.md5(f"{filepath.name}_{idx}".encode()).hexdigest()[:16], 16) % (2**63)
        
        points.append(PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload={
                "text": chunk['text'],
                "metadata": {
                    "source_file": filepath.name,
                    "file_type": filepath.suffix,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "chunk_type": chunk['type']
                }
            }
        ))
    
    # Upload
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[i:i+batch_size])
    
    logger.info(f"  ✅ Created with {len(points)} vectors")
    return collection_name


def main():
    parser = argparse.ArgumentParser(
        description="Universal multi-format document indexer with deduplication"
    )
    parser.add_argument("paths", nargs="+", help="Files or directories to index")
    parser.add_argument("--output", default="./qdrant_storage", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force recreate")
    parser.add_argument("--formats", default="md,txt,py,html,htm,mdx", 
                       help="File formats to process (comma-separated)")
    
    args = parser.parse_args()
    
    # Collect files
    formats = set(f".{f.strip()}" for f in args.formats.split(','))
    files = []
    
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file() and path.suffix.lower() in formats:
            files.append(path)
        elif path.is_dir():
            for fmt in formats:
                files.extend(path.rglob(f"*{fmt}"))
    
    if not files:
        logger.error("No files found!")
        sys.exit(1)
    
    logger.info(f"Found {len(files)} files")
    logger.info(f"Output: {args.output}")
    
    # Initialize
    client = QdrantClient(path=args.output)
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    doc_parser = UniversalDocumentParser()
    chunker = SmartChunker()
    
    # Process
    created = []
    for file in tqdm(files, desc="Processing"):
        coll = create_collection(client, embedding_model, file, doc_parser, chunker, args.force)
        if coll:
            created.append(coll)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Complete! Created {len(created)}/{len(files)} collections")
    logger.info(f"Storage: {args.output}")


if __name__ == "__main__":
    main()
