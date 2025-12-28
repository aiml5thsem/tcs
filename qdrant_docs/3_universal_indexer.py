#!/usr/bin/env python3
"""
Universal Database Creator - Best for Both Code and Non-Code Documents
Works with both server.py and server_improvised.py
Combines smart chunking with flexible CLI
"""

import os
import sys
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from tqdm import tqdm

# Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Adaptive chunking parameters
DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 150
CODE_CHUNK_SIZE = 600
CODE_OVERLAP = 100

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UniversalChunker:
    """
    Smart chunker that works well for ALL content types:
    - Pure documentation
    - Pure code
    - Mixed documentation + code
    """
    
    def __init__(self, doc_chunk_size=DEFAULT_CHUNK_SIZE, doc_overlap=DEFAULT_OVERLAP,
                 code_chunk_size=CODE_CHUNK_SIZE, code_overlap=CODE_OVERLAP):
        self.doc_chunk_size = doc_chunk_size
        self.doc_overlap = doc_overlap
        self.code_chunk_size = code_chunk_size
        self.code_overlap = code_overlap
        
        # Code detection patterns
        self.code_block_pattern = r'```[\w]*\n(.*?)```'
        self.function_patterns = {
            'python': r'(def\s+\w+|class\s+\w+)',
            'javascript': r'(function\s+\w+|const\s+\w+\s*=|class\s+\w+)',
            'java': r'(public|private|protected)\s+(class|interface|void|static)',
        }
    
    def detect_content_type(self, text: str) -> str:
        """Detect content type for adaptive chunking"""
        lines = text.split('\n')
        total_lines = len(lines)
        
        if total_lines == 0:
            return 'docs'
        
        # Count code blocks
        code_blocks = len(re.findall(r'```', text)) // 2
        
        # Calculate rough code ratio
        code_ratio = (code_blocks * 10) / total_lines
        
        if code_ratio > 0.3:
            return 'code'
        elif code_ratio > 0.1:
            return 'mixed'
        else:
            return 'docs'
    
    def extract_code_blocks(self, text: str) -> Tuple[List[Dict], str]:
        """Extract fenced code blocks, return blocks and remaining text"""
        code_blocks = []
        
        matches = list(re.finditer(self.code_block_pattern, text, re.DOTALL))
        
        for match in matches:
            code = match.group(1).strip()
            if code:
                # Detect language
                lang_match = re.search(r'```(\w+)', match.group(0))
                lang = lang_match.group(1) if lang_match else 'generic'
                
                code_blocks.append({
                    'code': code,
                    'language': lang,
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Remove code blocks from text
        remaining = text
        for block in reversed(code_blocks):
            remaining = remaining[:block['start']] + remaining[block['end']:]
        
        return code_blocks, remaining
    
    def chunk_by_functions(self, code: str, language: str) -> List[str]:
        """Try to chunk code by function boundaries"""
        pattern = self.function_patterns.get(language, self.function_patterns.get('python'))
        
        boundaries = [m.start() for m in re.finditer(pattern, code)]
        boundaries.append(len(code))
        
        if len(boundaries) <= 1:
            # No clear boundaries, use size-based
            return self._size_based_chunk(code, self.code_chunk_size, self.code_overlap)
        
        chunks = []
        for i in range(len(boundaries) - 1):
            chunk = code[boundaries[i]:boundaries[i + 1]].strip()
            
            if len(chunk) > self.code_chunk_size * 2:
                # Too large, split it
                chunks.extend(self._size_based_chunk(chunk, self.code_chunk_size, self.code_overlap))
            elif chunk:
                chunks.append(chunk)
        
        return chunks
    
    def chunk_by_semantic_boundaries(self, text: str) -> List[str]:
        """Chunk documentation by markdown structure and paragraphs"""
        # Split by headers (# Header style)
        parts = re.split(r'(\n#{1,6}\s+.+\n)', text)
        
        chunks = []
        current = ""
        current_size = 0
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            part_size = len(part)
            is_header = re.match(r'^#{1,6}\s+', part)
            
            if current_size + part_size > self.doc_chunk_size and current:
                # Save current chunk
                chunks.append(current.strip())
                
                # Start new chunk
                if self.doc_overlap > 0 and not is_header:
                    overlap_text = current[-self.doc_overlap:]
                    current = overlap_text + "\n\n" + part
                    current_size = len(current)
                else:
                    current = part
                    current_size = part_size
            else:
                # Add to current
                if current:
                    current += "\n\n" + part
                else:
                    current = part
                current_size += part_size
        
        if current.strip():
            chunks.append(current.strip())
        
        return chunks
    
    def _size_based_chunk(self, text: str, size: int, overlap: int) -> List[str]:
        """Fallback: simple size-based chunking with sentence awareness"""
        if len(text) <= size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + size
            
            # Try to break at sentence boundary
            if end < len(text):
                for sep in ['. ', '.\n', ';\n', '\n\n', '! ', '? ', '\n']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep != -1:
                        end = start + last_sep + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def chunk(self, text: str) -> List[Dict[str, any]]:
        """
        Main chunking method - returns chunks with metadata
        Returns: List of {text, type, language}
        """
        content_type = self.detect_content_type(text)
        logger.info(f"  Content type: {content_type}")
        
        all_chunks = []
        
        if content_type == 'docs':
            # Pure documentation
            doc_chunks = self.chunk_by_semantic_boundaries(text)
            for chunk in doc_chunks:
                all_chunks.append({
                    'text': chunk,
                    'type': 'documentation',
                    'language': None
                })
        
        elif content_type == 'code':
            # Primarily code - extract blocks and chunk separately
            code_blocks, remaining = self.extract_code_blocks(text)
            
            # Chunk code blocks
            for block in code_blocks:
                code_chunks = self.chunk_by_functions(block['code'], block['language'])
                for chunk in code_chunks:
                    all_chunks.append({
                        'text': chunk,
                        'type': 'code',
                        'language': block['language']
                    })
            
            # Chunk remaining documentation
            if remaining.strip():
                doc_chunks = self.chunk_by_semantic_boundaries(remaining)
                for chunk in doc_chunks:
                    if chunk.strip():
                        all_chunks.append({
                            'text': chunk,
                            'type': 'documentation',
                            'language': None
                        })
        
        else:  # mixed
            # Mixed content - preserve document flow
            code_blocks, remaining = self.extract_code_blocks(text)
            
            # Build position map
            positions = []
            for block in code_blocks:
                positions.append(('code', block['start'], block))
            
            # Add documentation sections
            last_pos = 0
            for block in sorted(code_blocks, key=lambda x: x['start']):
                if last_pos < block['start']:
                    doc_text = text[last_pos:block['start']].strip()
                    if doc_text:
                        positions.append(('doc', last_pos, doc_text))
                last_pos = block['end']
            
            # Add final documentation
            if last_pos < len(text):
                doc_text = text[last_pos:].strip()
                if doc_text:
                    positions.append(('doc', last_pos, doc_text))
            
            # Sort by position and chunk
            positions.sort(key=lambda x: x[1])
            
            for pos_type, _, content in positions:
                if pos_type == 'doc':
                    doc_chunks = self.chunk_by_semantic_boundaries(content)
                    for chunk in doc_chunks:
                        if chunk.strip():
                            all_chunks.append({
                                'text': chunk,
                                'type': 'documentation',
                                'language': None
                            })
                else:  # code block
                    code_chunks = self.chunk_by_functions(content['code'], content['language'])
                    for chunk in code_chunks:
                        all_chunks.append({
                            'text': chunk,
                            'type': 'code',
                            'language': content['language']
                        })
        
        logger.info(f"  Created {len(all_chunks)} chunks")
        return all_chunks


def extract_metadata(filepath: Path, text: str) -> Dict[str, any]:
    """Extract metadata from file"""
    metadata = {
        "source_file": filepath.name,
        "file_path": str(filepath),
        "file_size": filepath.stat().st_size if filepath.exists() else 0,
    }
    
    # Extract YAML frontmatter
    if text.startswith('---'):
        parts = text.split('---', 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            for line in frontmatter.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
    
    # Detect languages
    code_langs = set(re.findall(r'```(\w+)', text))
    if code_langs:
        metadata['languages'] = list(code_langs)
    
    # Extract top headers
    headers = re.findall(r'^#{1,3}\s+(.+)$', text, re.MULTILINE)
    if headers:
        metadata['topics'] = headers[:5]
    
    return metadata


def sanitize_collection_name(filename: str) -> str:
    """Create valid Qdrant collection name"""
    name = Path(filename).stem
    name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
    
    if name and not (name[0].isalpha() or name[0] == '_'):
        name = '_' + name
    
    name = name.lower()
    
    if not name:
        name = '_collection'
    
    return name


def create_collection_from_markdown(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    md_file: Path,
    force_recreate: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> Optional[str]:
    """Create collection from markdown file with smart chunking"""
    
    collection_name = sanitize_collection_name(md_file.name)
    
    logger.info(f"Processing: {md_file.name} → Collection: {collection_name}")
    
    # Check if exists
    collections = client.get_collections().collections
    if any(c.name == collection_name for c in collections):
        if force_recreate:
            logger.info(f"  Deleting existing collection")
            client.delete_collection(collection_name)
        else:
            logger.info(f"  Collection exists (use --force to recreate)")
            return collection_name
    
    # Read file
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"  Error reading: {e}")
        return None
    
    if not content.strip():
        logger.warning(f"  Empty file")
        return None
    
    # Extract metadata
    metadata = extract_metadata(md_file, content)
    
    # Smart chunking
    chunker = UniversalChunker(chunk_size, overlap, CODE_CHUNK_SIZE, CODE_OVERLAP)
    chunks = chunker.chunk(content)
    
    if not chunks:
        logger.warning(f"  No chunks created")
        return None
    
    # Generate embeddings
    logger.info(f"  Generating embeddings...")
    texts = [c['text'] for c in chunks]
    embeddings = list(embedding_model.embed(texts))
    
    # Create collection
    logger.info(f"  Creating collection...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE
        )
    )
    
    # Prepare points
    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = int(hashlib.md5(f"{md_file.name}_{idx}".encode()).hexdigest()[:16], 16) % (2**63)
        
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            "chunk_index": idx,
            "total_chunks": len(chunks),
            "chunk_type": chunk['type'],
            "language": chunk.get('language')
        })
        
        points.append(PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload={
                "text": chunk['text'],
                "metadata": chunk_metadata
            }
        ))
    
    # Upload in batches
    batch_size = 100
    logger.info(f"  Uploading {len(points)} vectors...")
    
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
    
    # Stats
    code_chunks = sum(1 for c in chunks if c['type'] == 'code')
    doc_chunks = len(chunks) - code_chunks
    
    logger.info(f"  ✅ Collection '{collection_name}' created")
    logger.info(f"     ├─ Total: {len(chunks)} chunks")
    logger.info(f"     ├─ Code: {code_chunks}")
    logger.info(f"     └─ Docs: {doc_chunks}")
    
    return collection_name


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Universal Database Creator - Works with both basic and advanced servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index all markdown in directory
  python create_local_db_universal.py /path/to/docs/ --output ./qdrant_storage
  
  # Index specific files
  python create_local_db_universal.py file1.md file2.md --output ./qdrant_storage
  
  # Force recreate with custom chunk size
  python create_local_db_universal.py /path/to/docs/ --force --chunk-size 1000
  
  # Use different embedding model
  python create_local_db_universal.py /path/to/docs/ --embedding-model "BAAI/bge-base-en-v1.5"
        """
    )
    
    parser.add_argument(
        "paths",
        nargs="+",
        help="Markdown files or directories to index"
    )
    parser.add_argument(
        "--output",
        default="./qdrant_storage",
        help="Qdrant storage directory (default: ./qdrant_storage)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate existing collections"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size for docs (default: {DEFAULT_CHUNK_SIZE})"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Overlap for docs (default: {DEFAULT_OVERLAP})"
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL,
        help=f"Embedding model (default: {EMBEDDING_MODEL})"
    )
    
    args = parser.parse_args()
    
    # Collect markdown files
    md_files = []
    for path_str in args.paths:
        path = Path(path_str)
        
        if path.is_file() and path.suffix.lower() in ['.md', '.markdown']:
            md_files.append(path)
        elif path.is_dir():
            md_files.extend(path.rglob("*.md"))
            md_files.extend(path.rglob("*.markdown"))
        else:
            logger.warning(f"Skipping: {path}")
    
    if not md_files:
        logger.error("No markdown files found!")
        sys.exit(1)
    
    logger.info(f"Found {len(md_files)} markdown files")
    logger.info(f"Output: {args.output}")
    logger.info(f"Model: {args.embedding_model}")
    logger.info(f"Chunk size: {args.chunk_size} (overlap: {args.overlap})")
    
    # Initialize
    logger.info("\nInitializing...")
    client = QdrantClient(path=args.output)
    embedding_model = TextEmbedding(model_name=args.embedding_model)
    
    # Process files
    logger.info("\nProcessing files...\n")
    created = []
    
    for md_file in tqdm(md_files, desc="Processing"):
        collection_name = create_collection_from_markdown(
            client,
            embedding_model,
            md_file,
            force_recreate=args.force,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
        
        if collection_name:
            created.append(collection_name)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Collections created: {len(created)}")
    logger.info(f"Storage: {args.output}")
    logger.info(f"\nCreated collections:")
    for coll in created:
        logger.info(f"  - {coll}")
    
    logger.info(f"\n💡 Next steps:")
    logger.info(f"  1. Set QDRANT_PATH={args.output}")
    logger.info(f"  2. Run server.py or server_improvised.py")
    logger.info(f"  3. Configure your MCP client")


if __name__ == "__main__":
    main()
