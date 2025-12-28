#!/usr/bin/env python3
"""
Advanced Single-Config Database Creator
Define collections in a dict with optimized chunking for docs and code
"""

import os
import sys
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from tqdm import tqdm

# ============================================================================
# CONFIGURATION DICTIONARY - EDIT THIS!
# ============================================================================

COLLECTIONS_CONFIG = {
    # Collection name: file path
    "endor": "/path/to/endor_docs.md",
    "interlinked": "/path/to/interlinked_apis_examples.md",
    "authentication": "/path/to/auth_guide.md",
    "deployment": "/path/to/deployment_notes.md",
    # Add more as needed...
}

# Advanced Configuration
QDRANT_PATH = "./qdrant_storage"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Chunking strategies
DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 150
CODE_CHUNK_SIZE = 600  # Smaller for code to keep complete functions
CODE_OVERLAP = 100

# ============================================================================
# SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SmartChunker:
    """
    Intelligent chunking that adapts to content type:
    - Code blocks: Keep functions/classes together
    - Documentation: Semantic boundary splitting
    - Mixed content: Hybrid approach
    """
    
    def __init__(self):
        self.code_patterns = {
            'python': r'(def\s+\w+|class\s+\w+)',
            'javascript': r'(function\s+\w+|const\s+\w+\s*=|class\s+\w+)',
            'java': r'(public|private|protected)\s+(class|interface|enum|void|static)',
            'generic': r'(```[\w]*\n.*?```)',  # Fenced code blocks
        }
    
    def detect_content_type(self, text: str) -> str:
        """Detect if content is primarily code, docs, or mixed"""
        code_block_count = len(re.findall(r'```', text)) // 2
        total_lines = len(text.split('\n'))
        
        if total_lines == 0:
            return 'docs'
        
        code_ratio = (code_block_count * 10) / total_lines  # Rough estimate
        
        if code_ratio > 0.3:
            return 'code'
        elif code_ratio > 0.1:
            return 'mixed'
        else:
            return 'docs'
    
    def extract_code_blocks(self, text: str) -> Tuple[List[str], str]:
        """Extract code blocks and return them separately with remaining text"""
        code_blocks = []
        
        # Find all fenced code blocks
        pattern = r'```[\w]*\n(.*?)```'
        matches = list(re.finditer(pattern, text, re.DOTALL))
        
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
        remaining_text = text
        for block in reversed(code_blocks):  # Reverse to maintain indices
            remaining_text = remaining_text[:block['start']] + remaining_text[block['end']:]
        
        return code_blocks, remaining_text
    
    def chunk_code(self, code: str, language: str = 'generic') -> List[str]:
        """Chunk code by logical units (functions, classes, etc.)"""
        chunks = []
        
        # Try to find logical boundaries
        pattern = self.code_patterns.get(language, self.code_patterns['generic'])
        
        boundaries = [m.start() for m in re.finditer(pattern, code)]
        boundaries.append(len(code))
        
        if len(boundaries) <= 1:
            # No clear boundaries, use size-based chunking
            return self._size_based_chunk(code, CODE_CHUNK_SIZE, CODE_OVERLAP)
        
        # Split by boundaries
        for i in range(len(boundaries) - 1):
            chunk = code[boundaries[i]:boundaries[i + 1]].strip()
            
            # If chunk is too large, split it
            if len(chunk) > CODE_CHUNK_SIZE * 2:
                chunks.extend(self._size_based_chunk(chunk, CODE_CHUNK_SIZE, CODE_OVERLAP))
            elif chunk:
                chunks.append(chunk)
        
        return chunks
    
    def chunk_documentation(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Chunk documentation by semantic boundaries (headers, paragraphs)"""
        # Split by markdown headers first
        sections = re.split(r'\n(#{1,6}\s+.+)\n', text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            section_size = len(section)
            
            # Check if this is a header
            is_header = re.match(r'^#{1,6}\s+', section)
            
            if current_size + section_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0 and not is_header:
                    overlap_text = current_chunk[-overlap:].strip()
                    current_chunk = overlap_text + "\n\n" + section
                    current_size = len(current_chunk)
                else:
                    current_chunk = section
                    current_size = section_size
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section
                current_size += section_size
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _size_based_chunk(self, text: str, size: int, overlap: int) -> List[str]:
        """Fallback: size-based chunking with sentence awareness"""
        if len(text) <= size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + size
            
            # Try to break at sentence boundary
            if end < len(text):
                for sep in ['. ', '.\n', ';\n', '\n\n', '! ', '? ']:
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
        Main chunking method that returns chunks with metadata
        Returns: List of {text, type, metadata}
        """
        content_type = self.detect_content_type(text)
        logger.info(f"  Content type detected: {content_type}")
        
        all_chunks = []
        
        if content_type == 'code':
            # Primarily code
            code_blocks, remaining = self.extract_code_blocks(text)
            
            # Chunk code blocks
            for block in code_blocks:
                code_chunks = self.chunk_code(block['code'], block['language'])
                for chunk in code_chunks:
                    all_chunks.append({
                        'text': chunk,
                        'type': 'code',
                        'language': block['language']
                    })
            
            # Chunk remaining text
            if remaining.strip():
                doc_chunks = self.chunk_documentation(remaining, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
                for chunk in doc_chunks:
                    all_chunks.append({
                        'text': chunk,
                        'type': 'documentation',
                        'language': None
                    })
        
        elif content_type == 'mixed':
            # Mixed content - preserve structure
            code_blocks, remaining = self.extract_code_blocks(text)
            
            # Interleave code and docs
            positions = []
            
            for block in code_blocks:
                positions.append(('code', block['start'], block))
            
            # Split remaining by position
            text_parts = []
            last_end = 0
            
            for block in sorted(code_blocks, key=lambda x: x['start']):
                if last_end < block['start']:
                    text_parts.append(('doc', text[last_end:block['start']]))
                last_end = block['end']
            
            if last_end < len(text):
                text_parts.append(('doc', text[last_end:]))
            
            # Chunk each part
            for part_type, content in text_parts:
                if part_type == 'doc' and content.strip():
                    doc_chunks = self.chunk_documentation(content, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
                    for chunk in doc_chunks:
                        all_chunks.append({
                            'text': chunk,
                            'type': 'documentation',
                            'language': None
                        })
            
            for block in code_blocks:
                code_chunks = self.chunk_code(block['code'], block['language'])
                for chunk in code_chunks:
                    all_chunks.append({
                        'text': chunk,
                        'type': 'code',
                        'language': block['language']
                    })
        
        else:
            # Pure documentation
            doc_chunks = self.chunk_documentation(text, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
            for chunk in doc_chunks:
                all_chunks.append({
                    'text': chunk,
                    'type': 'documentation',
                    'language': None
                })
        
        logger.info(f"  Created {len(all_chunks)} chunks ({content_type})")
        return all_chunks


def extract_metadata(filepath: Path, text: str) -> Dict[str, any]:
    """Extract rich metadata from file and content"""
    metadata = {
        "source_file": filepath.name,
        "file_path": str(filepath),
        "file_size": filepath.stat().st_size if filepath.exists() else 0,
        "content_length": len(text)
    }
    
    # Extract frontmatter
    if text.startswith('---'):
        parts = text.split('---', 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            for line in frontmatter.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
    
    # Detect programming languages mentioned
    code_langs = set(re.findall(r'```(\w+)', text))
    if code_langs:
        metadata['languages'] = list(code_langs)
    
    # Extract headers for overview
    headers = re.findall(r'^#{1,3}\s+(.+)$', text, re.MULTILINE)
    if headers:
        metadata['main_topics'] = headers[:5]  # Top 5 headers
    
    return metadata


def create_collection(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    collection_name: str,
    file_path: str,
    force: bool = False
) -> bool:
    """Create optimized collection from file"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {collection_name}")
    logger.info(f"Source: {file_path}")
    
    # Check if collection exists
    collections = client.get_collections().collections
    if any(c.name == collection_name for c in collections):
        if force:
            logger.info(f"  Deleting existing collection...")
            client.delete_collection(collection_name)
        else:
            logger.warning(f"  Collection exists (use --force to recreate)")
            return False
    
    # Read file
    path = Path(file_path)
    if not path.exists():
        logger.error(f"  File not found: {file_path}")
        return False
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"  Error reading file: {e}")
        return False
    
    if not content.strip():
        logger.warning(f"  Empty file, skipping")
        return False
    
    # Extract base metadata
    base_metadata = extract_metadata(path, content)
    logger.info(f"  File size: {base_metadata['file_size']:,} bytes")
    
    # Smart chunking
    chunker = SmartChunker()
    chunks = chunker.chunk(content)
    
    if not chunks:
        logger.warning(f"  No chunks created")
        return False
    
    # Generate embeddings
    logger.info(f"  Generating embeddings for {len(chunks)} chunks...")
    texts = [chunk['text'] for chunk in chunks]
    embeddings = list(embedding_model.embed(texts))
    
    # Create collection
    logger.info(f"  Creating collection in Qdrant...")
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
        # Unique ID
        point_id = int(hashlib.md5(f"{collection_name}_{idx}".encode()).hexdigest()[:16], 16) % (2**63)
        
        # Combine metadata
        chunk_metadata = base_metadata.copy()
        chunk_metadata.update({
            "chunk_index": idx,
            "total_chunks": len(chunks),
            "chunk_type": chunk['type'],
            "language": chunk.get('language'),
            "chunk_size": len(chunk['text'])
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
    
    for i in tqdm(range(0, len(points), batch_size), desc="  Uploading", leave=False):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
    
    # Statistics
    code_chunks = sum(1 for c in chunks if c['type'] == 'code')
    doc_chunks = sum(1 for c in chunks if c['type'] == 'documentation')
    
    logger.info(f"  ✅ Collection '{collection_name}' created")
    logger.info(f"     └─ Total chunks: {len(chunks)}")
    logger.info(f"     └─ Code chunks: {code_chunks}")
    logger.info(f"     └─ Doc chunks: {doc_chunks}")
    
    return True


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create Qdrant collections from configured file dict",
        epilog="Edit COLLECTIONS_CONFIG in the script to add your files"
    )
    parser.add_argument("--force", action="store_true", help="Force recreate existing collections")
    parser.add_argument("--collection", help="Process only this collection name")
    parser.add_argument("--output", default=QDRANT_PATH, help="Qdrant storage path")
    args = parser.parse_args()
    
    # Validate configuration
    if not COLLECTIONS_CONFIG:
        logger.error("COLLECTIONS_CONFIG is empty! Edit the script to add your files.")
        sys.exit(1)
    
    # Filter collections if specified
    if args.collection:
        if args.collection not in COLLECTIONS_CONFIG:
            logger.error(f"Collection '{args.collection}' not found in config")
            sys.exit(1)
        collections_to_process = {args.collection: COLLECTIONS_CONFIG[args.collection]}
    else:
        collections_to_process = COLLECTIONS_CONFIG
    
    logger.info("="*60)
    logger.info("Advanced Collection Creator")
    logger.info("="*60)
    logger.info(f"Output: {args.output}")
    logger.info(f"Model: {EMBEDDING_MODEL}")
    logger.info(f"Collections to process: {len(collections_to_process)}")
    
    # Initialize
    logger.info("\nInitializing...")
    client = QdrantClient(path=args.output)
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    
    # Process collections
    success_count = 0
    for name, filepath in collections_to_process.items():
        if create_collection(client, embedding_model, name, filepath, args.force):
            success_count += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Processing Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Successful: {success_count}/{len(collections_to_process)}")
    logger.info(f"Storage: {args.output}")
    logger.info(f"\n💡 Set QDRANT_PATH={args.output} in your MCP config")


if __name__ == "__main__":
    main()
