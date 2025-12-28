#!/usr/bin/env python3
"""
4_create_local_db_upgraded.py - With Qdrant 1.16 Features
✅ Creates collections with BM25 sparse vectors
✅ Enables inline storage for faster disk search
✅ Supports all formats: MD, TXT, PY, HTML, MDX
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    HnswConfigDiff, SparseVectorParams, SparseIndexParams,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType
)
from fastembed import TextEmbedding, SparseTextEmbedding
from tqdm import tqdm

# Import parsers and chunker from original
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
ENABLE_BM25 = True  # Create collections with BM25 support
ENABLE_INLINE_STORAGE = True  # Faster disk search
ENABLE_QUANTIZATION = True  # Memory optimization

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ContentDeduplicator:
    """Smart deduplication"""
    
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.seen_hashes = set()
    
    def _normalize(self, text: str) -> str:
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(Skip to|Back to top|Navigation|Menu|Footer)', '', text, flags=re.I)
        return text.strip().lower()
    
    def is_duplicate(self, text: str) -> bool:
        normalized = self._normalize(text)
        if not normalized or len(normalized) < 50:
            return True
        
        text_hash = hashlib.md5(normalized.encode()).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(text_hash)
        return False


class UniversalDocumentParser:
    """Parse multiple formats"""
    
    def __init__(self):
        self.dedup = ContentDeduplicator()
    
    def parse_file(self, filepath: Path) -> Optional[str]:
        """Parse any supported file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return None
        
        suffix = filepath.suffix.lower()
        
        if suffix in ['.md', '.markdown']:
            return content
        elif suffix == '.txt':
            return content
        elif suffix == '.py':
            return self._parse_python(content, filepath)
        elif suffix in ['.html', '.htm']:
            return self._parse_html(content, filepath)
        elif suffix == '.mdx':
            return self._parse_mdx(content)
        else:
            return content
    
    def _parse_python(self, content: str, filepath: Path) -> str:
        import re
        result = [f"# {filepath.name}\n"]
        
        # Module docstring
        doc_match = re.search(r'^"""(.*?)"""', content, re.DOTALL | re.M)
        if doc_match:
            result.append(f"## Documentation\n{doc_match.group(1).strip()}\n")
        
        # Classes
        for match in re.finditer(r'class\s+(\w+).*?:\s*"""(.*?)"""', content, re.DOTALL):
            result.append(f"## Class: {match.group(1)}\n{match.group(2).strip()}\n")
        
        # Functions
        for match in re.finditer(r'def\s+(\w+)\s*\([^)]*\):\s*"""(.*?)"""', content, re.DOTALL):
            result.append(f"## Function: {match.group(1)}\n{match.group(2).strip()}\n")
        
        result.append(f"\n## Code\n```python\n{content}\n```\n")
        return "\n".join(result)
    
    def _parse_html(self, content: str, filepath: Path) -> str:
        if not HAS_BS4:
            return ""
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove unwanted
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        for element in soup.find_all(class_=lambda x: x and any(
            word in x.lower() for word in ['sidebar', 'menu', 'navigation', 'breadcrumb']
        )):
            element.decompose()
        
        result = [f"# {filepath.stem}\n"]
        
        main = soup.find('main') or soup.find('article') or soup
        for elem in main.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'pre', 'code']):
            text = elem.get_text().strip()
            if not text or self.dedup.is_duplicate(text):
                continue
            
            if elem.name.startswith('h'):
                level = int(elem.name[1])
                result.append(f"\n{'#' * level} {text}\n")
            elif elem.name in ['pre', 'code']:
                result.append(f"\n```\n{text}\n```\n")
            else:
                result.append(f"{text}\n")
        
        return "\n".join(result)
    
    def _parse_mdx(self, content: str) -> str:
        import re
        # Remove JSX
        content = re.sub(r'<(\w+)[^>]*>(.*?)</\1>', r'\2', content, flags=re.DOTALL)
        content = re.sub(r'<\w+[^>]*/>', '', content)
        content = re.sub(r'^import\s+.*?from\s+.*?$', '', content, flags=re.M)
        content = re.sub(r'^export\s+.*?$', '', content, flags=re.M)
        return content


class SmartChunker:
    """Content-aware chunking"""
    
    def __init__(self, doc_size=800, doc_overlap=150):
        self.doc_size = doc_size
        self.doc_overlap = doc_overlap
    
    def chunk(self, text: str) -> List[Dict]:
        import re
        
        # Detect content type
        code_ratio = len(re.findall(r'```|def |class |function ', text)) / max(len(text.split('\n')), 1)
        is_code = code_ratio > 0.15
        
        if is_code:
            chunks = self._chunk_code(text)
        else:
            chunks = self._chunk_docs(text)
        
        return [{'text': c, 'type': 'code' if is_code else 'docs'} for c in chunks if c.strip()]
    
    def _chunk_code(self, text: str) -> List[str]:
        import re
        boundaries = [m.start() for m in re.finditer(r'\n(def |class |async def )', text)]
        boundaries.append(len(text))
        
        if len(boundaries) <= 1:
            return self._simple_chunk(text, 600, 100)
        
        chunks = []
        for i in range(len(boundaries) - 1):
            chunk = text[boundaries[i]:boundaries[i+1]].strip()
            if chunk:
                if len(chunk) > 1200:
                    chunks.extend(self._simple_chunk(chunk, 600, 100))
                else:
                    chunks.append(chunk)
        return chunks
    
    def _chunk_docs(self, text: str) -> List[str]:
        import re
        sections = re.split(r'(\n#{1,6}\s+.+\n)', text)
        
        chunks = []
        current = ""
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            if len(current) + len(section) > self.doc_size and current:
                chunks.append(current.strip())
                current = section
            else:
                current += "\n\n" + section if current else section
        
        if current.strip():
            chunks.append(current.strip())
        
        return chunks
    
    def _simple_chunk(self, text: str, size: int, overlap: int) -> List[str]:
        if len(text) <= size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            if end < len(text):
                for sep in ['. ', '.\n', '\n\n']:
                    last = text[start:end].rfind(sep)
                    if last != -1:
                        end = start + last + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        
        return chunks


def sanitize_name(name: str) -> str:
    """Create valid collection name"""
    name = Path(name).stem
    name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name).lower()
    if name and not (name[0].isalpha() or name[0] == '_'):
        name = '_' + name
    return name or '_collection'


def create_collection_upgraded(
    client: QdrantClient,
    dense_model: TextEmbedding,
    sparse_model: Optional[SparseTextEmbedding],
    filepath: Path,
    parser: UniversalDocumentParser,
    chunker: SmartChunker,
    force: bool = False
) -> Optional[str]:
    """Create collection with Qdrant 1.16 features"""
    
    collection_name = sanitize_name(filepath.name)
    logger.info(f"Processing: {filepath.name} → {collection_name}")
    
    # Check if exists
    collections = client.get_collections().collections
    if any(c.name == collection_name for c in collections):
        if force:
            logger.info(f"  Deleting existing...")
            client.delete_collection(collection_name)
        else:
            logger.info(f"  Already exists")
            return collection_name
    
    # Parse
    content = parser.parse_file(filepath)
    if not content or not content.strip():
        logger.warning(f"  No content")
        return None
    
    # Chunk
    chunks = chunker.chunk(content)
    if not chunks:
        logger.warning(f"  No chunks")
        return None
    
    logger.info(f"  Created {len(chunks)} chunks")
    
    # Generate embeddings
    texts = [c['text'] for c in chunks]
    dense_embeddings = list(dense_model.embed(texts))
    
    # Generate BM25 sparse embeddings if available
    sparse_embeddings = None
    if sparse_model and ENABLE_BM25:
        try:
            sparse_embeddings = list(sparse_model.embed(texts))
            logger.info(f"  Generated BM25 embeddings")
        except Exception as e:
            logger.warning(f"  BM25 generation failed: {e}")
    
    # Create collection with Qdrant 1.16 features
    vectors_config = {"dense": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)}
    
    sparse_vectors_config = None
    if sparse_embeddings:
        sparse_vectors_config = {
            "sparse": SparseVectorParams(index=SparseIndexParams())
        }
    
    hnsw_config = None
    if ENABLE_INLINE_STORAGE:
        hnsw_config = HnswConfigDiff(inline_storage=True)
    
    quantization_config = None
    if ENABLE_QUANTIZATION:
        quantization_config = ScalarQuantization(
            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, quantile=0.99)
        )
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
        hnsw_config=hnsw_config,
        quantization_config=quantization_config
    )
    
    logger.info(f"  ✅ Collection created with:")
    logger.info(f"     ├─ Dense vectors: ✅")
    logger.info(f"     ├─ BM25 sparse: {'✅' if sparse_embeddings else '❌'}")
    logger.info(f"     ├─ Inline storage: {'✅' if ENABLE_INLINE_STORAGE else '❌'}")
    logger.info(f"     └─ Quantization: {'✅' if ENABLE_QUANTIZATION else '❌'}")
    
    # Create points
    points = []
    for idx, (chunk, dense_emb) in enumerate(zip(chunks, dense_embeddings)):
        point_id = int(hashlib.md5(f"{filepath.name}_{idx}".encode()).hexdigest()[:16], 16) % (2**63)
        
        # Prepare vectors
        vectors = {"dense": dense_emb.tolist()}
        if sparse_embeddings:
            sparse_emb = sparse_embeddings[idx]
            vectors["sparse"] = {
                "indices": sparse_emb.indices.tolist(),
                "values": sparse_emb.values.tolist()
            }
        
        points.append(PointStruct(
            id=point_id,
            vector=vectors,
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
    
    logger.info(f"  ✅ Uploaded {len(points)} vectors")
    return collection_name


def main():
    parser = argparse.ArgumentParser(description="Create Qdrant 1.16 collections")
    parser.add_argument("paths", nargs="+", help="Files or directories")
    parser.add_argument("--output", default="./qdrant_storage", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force recreate")
    parser.add_argument("--formats", default="md,txt,py,html,htm,mdx", help="File formats")
    parser.add_argument("--no-bm25", action="store_true", help="Disable BM25")
    parser.add_argument("--no-inline", action="store_true", help="Disable inline storage")
    
    args = parser.parse_args()
    
    global ENABLE_BM25, ENABLE_INLINE_STORAGE
    ENABLE_BM25 = not args.no_bm25
    ENABLE_INLINE_STORAGE = not args.no_inline
    
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
    logger.info(f"Features: BM25={'✅' if ENABLE_BM25 else '❌'}, "
               f"Inline={'✅' if ENABLE_INLINE_STORAGE else '❌'}")
    
    # Initialize
    client = QdrantClient(path=args.output)
    dense_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    
    sparse_model = None
    if ENABLE_BM25:
        try:
            sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
            logger.info("✅ BM25 model loaded")
        except Exception as e:
            logger.warning(f"BM25 model unavailable: {e}")
    
    doc_parser = UniversalDocumentParser()
    chunker = SmartChunker()
    
    # Process
    created = []
    for file in tqdm(files, desc="Processing"):
        coll = create_collection_upgraded(
            client, dense_model, sparse_model, file, doc_parser, chunker, args.force
        )
        if coll:
            created.append(coll)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Complete! Created {len(created)}/{len(files)} collections")
    logger.info(f"Storage: {args.output}")
    logger.info(f"Qdrant 1.16 features enabled:")
    logger.info(f"  ├─ Built-in BM25: {'✅' if ENABLE_BM25 else '❌'}")
    logger.info(f"  ├─ Inline storage: {'✅' if ENABLE_INLINE_STORAGE else '❌'}")
    logger.info(f"  ├─ Quantization: ✅")
    logger.info(f"  └─ ACORN: Enable in server with USE_ACORN=true")


if __name__ == "__main__":
    main()
