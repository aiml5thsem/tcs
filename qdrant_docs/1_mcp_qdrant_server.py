#!/usr/bin/env python3
"""
Optimized Read-Only MCP Server for Qdrant Documentation
Single-file implementation with list_collections and search_documents tools
"""

import asyncio
import logging
import os
from typing import Any, Optional

from mcp.server import Server
from mcp.types import Tool, TextContent
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp-qdrant-docs")

# Configuration from environment variables
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_storage")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# Initialize components
app = Server("mcp-qdrant-docs")
embedding_model: Optional[TextEmbedding] = None
qdrant_client: Optional[QdrantClient] = None


def initialize_clients():
    """Initialize Qdrant client and embedding model"""
    global qdrant_client, embedding_model
    
    logger.info(f"Initializing Qdrant client at: {QDRANT_PATH}")
    qdrant_client = QdrantClient(path=QDRANT_PATH)
    
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    
    logger.info("✅ Initialization complete")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="list_collections",
            description="List all available documentation collections in the Qdrant database. "
                       "Returns collection names and their document counts. "
                       "Use this to discover what documentation is available.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="search_documents",
            description="Search for relevant documentation chunks across collections using semantic search. "
                       "Returns the most relevant text passages with metadata. "
                       "Optionally filter by specific collection name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documentation"
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Optional: Search only in this specific collection. "
                                     "If not provided, searches across all collections."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    
    if name == "list_collections":
        return await handle_list_collections()
    elif name == "search_documents":
        return await handle_search_documents(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def handle_list_collections() -> list[TextContent]:
    """List all collections and their stats"""
    try:
        collections = qdrant_client.get_collections().collections
        
        if not collections:
            return [TextContent(
                type="text",
                text="No collections found in the database. Please create collections using create_local_db.py"
            )]
        
        result = ["📚 Available Documentation Collections:\n"]
        
        for collection in collections:
            collection_info = qdrant_client.get_collection(collection.name)
            points_count = collection_info.points_count
            vectors_count = collection_info.vectors_count
            
            result.append(f"\n📖 Collection: {collection.name}")
            result.append(f"   Documents: {points_count}")
            result.append(f"   Vectors: {vectors_count}")
        
        return [TextContent(type="text", text="\n".join(result))]
        
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return [TextContent(
            type="text",
            text=f"Error listing collections: {str(e)}"
        )]


async def handle_search_documents(arguments: dict) -> list[TextContent]:
    """Search documents using semantic search"""
    try:
        query = arguments.get("query")
        collection_name = arguments.get("collection_name")
        limit = arguments.get("limit", 5)
        
        if not query:
            return [TextContent(type="text", text="Error: query parameter is required")]
        
        logger.info(f"Searching for: '{query}' in collection: {collection_name or 'all'}")
        
        # Generate query embedding
        query_vector = list(embedding_model.embed([query]))[0].tolist()
        
        # Determine which collections to search
        if collection_name:
            collections_to_search = [collection_name]
        else:
            all_collections = qdrant_client.get_collections().collections
            collections_to_search = [c.name for c in all_collections]
        
        # Search across collections
        all_results = []
        for coll_name in collections_to_search:
            try:
                search_results = qdrant_client.search(
                    collection_name=coll_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True
                )
                
                for result in search_results:
                    all_results.append({
                        "collection": coll_name,
                        "score": result.score,
                        "text": result.payload.get("text", ""),
                        "metadata": result.payload.get("metadata", {})
                    })
            except Exception as e:
                logger.warning(f"Error searching collection {coll_name}: {e}")
                continue
        
        # Sort by score and limit
        all_results.sort(key=lambda x: x["score"], reverse=True)
        all_results = all_results[:limit]
        
        if not all_results:
            return [TextContent(
                type="text",
                text=f"No results found for query: '{query}'"
            )]
        
        # Format results
        result_text = [f"🔍 Search Results for: '{query}'\n"]
        result_text.append(f"Found {len(all_results)} relevant passages:\n")
        
        for idx, result in enumerate(all_results, 1):
            result_text.append(f"\n{'='*60}")
            result_text.append(f"Result #{idx} | Collection: {result['collection']} | Relevance: {result['score']:.3f}")
            result_text.append(f"{'='*60}")
            
            # Add metadata if available
            metadata = result.get("metadata", {})
            if metadata:
                if "source_file" in metadata:
                    result_text.append(f"📄 Source: {metadata['source_file']}")
                if "chunk_index" in metadata:
                    result_text.append(f"📍 Chunk: {metadata['chunk_index']}")
            
            result_text.append(f"\n{result['text']}\n")
        
        return [TextContent(type="text", text="\n".join(result_text))]
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return [TextContent(
            type="text",
            text=f"Error searching documents: {str(e)}"
        )]


async def main():
    """Main entry point"""
    # Initialize clients
    initialize_clients()
    
    # Import and setup based on transport
    from mcp.server.stdio import stdio_server
    
    logger.info("🚀 Starting MCP Qdrant Documentation Server")
    logger.info(f"📁 Qdrant Path: {QDRANT_PATH}")
    logger.info(f"🤖 Embedding Model: {EMBEDDING_MODEL}")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
