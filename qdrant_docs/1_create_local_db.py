#!/usr/bin/env python3
"""
Script to create local Qdrant database from markdown files
Each markdown file becomes its own collection
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict
import argparse

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from tqdm import tqdm

# Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks


def setup_logging():
    """Setup basic logging"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            for separator in ['. ', '.\n', '! ', '? ', '\n\n']:
                last_sep = text[start:end].rfind(separator)
                if last_sep != -1:
                    end = start + last_sep + len(separator)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def extract_metadata_from_markdown(filepath: Path) -> Dict[str, str]:
    """
    Extract metadata from markdown frontmatter (if exists)
    
    Args:
        filepath: Path to markdown file
    
    Returns:
        Dictionary of metadata
    """
    metadata = {
        "source_file": filepath.name,
        "file_path": str(filepath)
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Simple frontmatter extraction (YAML style)
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = parts[1].strip()
                    for line in frontmatter.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            metadata[key.strip()] = value.strip()
    except Exception as e:
        logger.warning(f"Could not extract metadata from {filepath}: {e}")
    
    return metadata


def sanitize_collection_name(filename: str) -> str:
    """
    Create valid Qdrant collection name from filename
    Collection names must start with letter/underscore and contain only letters, digits, underscores, hyphens
    
    Args:
        filename: Original filename
    
    Returns:
        Valid collection name
    """
    # Remove extension
    name = Path(filename).stem
    
    # Replace spaces and special chars with underscores
    name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
    
    # Ensure starts with letter or underscore
    if name and not (name[0].isalpha() or name[0] == '_'):
        name = '_' + name
    
    # Convert to lowercase for consistency
    name = name.lower()
    
    # Ensure not empty
    if not name:
        name = '_collection'
    
    return name


def create_collection_from_markdown(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    md_file: Path,
    force_recreate: bool = False
) -> str:
    """
    Create a Qdrant collection from a single markdown file
    
    Args:
        client: Qdrant client
        embedding_model: Embedding model
        md_file: Path to markdown file
        force_recreate: If True, delete existing collection before creating
    
    Returns:
        Collection name
    """
    collection_name = sanitize_collection_name(md_file.name)
    
    logger.info(f"Processing: {md_file.name} → Collection: {collection_name}")
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_exists = any(c.name == collection_name for c in collections)
    
    if collection_exists:
        if force_recreate:
            logger.info(f"  Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            logger.info(f"  Collection already exists: {collection_name} (use --force to recreate)")
            return collection_name
    
    # Read markdown file
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"  Error reading file: {e}")
        return None
    
    # Extract metadata
    metadata = extract_metadata_from_markdown(md_file)
    
    # Chunk the content
    chunks = chunk_text(content)
    logger.info(f"  Created {len(chunks)} chunks")
    
    if not chunks:
        logger.warning(f"  No content to index in {md_file}")
        return None
    
    # Generate embeddings
    logger.info(f"  Generating embeddings...")
    embeddings = list(embedding_model.embed(chunks))
    
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
        # Create unique ID based on content hash
        point_id = int(hashlib.md5(f"{md_file.name}_{idx}".encode()).hexdigest()[:16], 16) % (2**63)
        
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            "chunk_index": idx,
            "total_chunks": len(chunks)
        })
        
        points.append(PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload={
                "text": chunk,
                "metadata": chunk_metadata
            }
        ))
    
    # Upload to Qdrant in batches
    batch_size = 100
    logger.info(f"  Uploading {len(points)} vectors...")
    
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
    
    logger.info(f"  ✅ Collection '{collection_name}' created with {len(points)} vectors")
    return collection_name


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Create Qdrant collections from markdown files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index all markdown files in a directory
  python create_local_db.py /path/to/docs/ --output ./qdrant_storage
  
  # Index specific files
  python create_local_db.py file1.md file2.md --output ./qdrant_storage
  
  # Force recreate existing collections
  python create_local_db.py /path/to/docs/ --force
  
  # Use custom chunk size
  python create_local_db.py /path/to/docs/ --chunk-size 500 --overlap 100
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
        help="Path to Qdrant storage directory (default: ./qdrant_storage)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate existing collections"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Chunk size in characters (default: {CHUNK_SIZE})"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Overlap between chunks (default: {CHUNK_OVERLAP})"
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL,
        help=f"Embedding model to use (default: {EMBEDDING_MODEL})"
    )
    
    args = parser.parse_args()
    
    # Update global configuration
    global CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.overlap
    EMBEDDING_MODEL = args.embedding_model
    
    # Collect all markdown files
    md_files = []
    for path_str in args.paths:
        path = Path(path_str)
        
        if path.is_file() and path.suffix.lower() in ['.md', '.markdown']:
            md_files.append(path)
        elif path.is_dir():
            md_files.extend(path.rglob("*.md"))
            md_files.extend(path.rglob("*.markdown"))
        else:
            logger.warning(f"Skipping invalid path: {path}")
    
    if not md_files:
        logger.error("No markdown files found!")
        sys.exit(1)
    
    logger.info(f"Found {len(md_files)} markdown files to process")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    
    # Initialize clients
    logger.info("\nInitializing Qdrant and embedding model...")
    client = QdrantClient(path=args.output)
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    
    # Process each file
    logger.info("\nProcessing files...\n")
    created_collections = []
    
    for md_file in tqdm(md_files, desc="Processing files"):
        collection_name = create_collection_from_markdown(
            client,
            embedding_model,
            md_file,
            force_recreate=args.force
        )
        
        if collection_name:
            created_collections.append(collection_name)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Processing complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Collections created: {len(created_collections)}")
    logger.info(f"Storage location: {args.output}")
    logger.info(f"\nCreated collections:")
    for coll in created_collections:
        logger.info(f"  - {coll}")
    
    logger.info(f"\n💡 Next steps:")
    logger.info(f"  1. Configure your MCP client to use: {args.output}")
    logger.info(f"  2. Set QDRANT_PATH={args.output} in your environment")
    logger.info(f"  3. Run the MCP server: python server.py")


if __name__ == "__main__":
    main()
