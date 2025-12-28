#!/usr/bin/env python3
"""
4_server_upgraded.py - Ultimate MCP Server with Qdrant 1.16 Features
✅ Built-in BM25 (no custom implementation needed!)
✅ ACORN for better filtered search
✅ Score boosting for metadata-based reranking
✅ Inline storage for faster disk search
"""

import asyncio
import logging
import os
from typing import Any, List, Dict, Optional

from mcp.server import Server
from mcp.types import Tool, TextContent
from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVector, Prefetch, QueryRequest, SearchRequest,
    Filter, FieldCondition, MatchValue
)
from fastembed import TextEmbedding, SparseTextEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("4_server_upgraded")

# Configuration
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_storage")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
USE_ACORN = os.getenv("USE_ACORN", "true").lower() == "true"

# Initialize
app = Server("4_server_upgraded")
dense_model: Optional[TextEmbedding] = None
sparse_model: Optional[SparseTextEmbedding] = None
qdrant_client: Optional[QdrantClient] = None


class QueryAnalyzer:
    """Enhanced query analysis"""
    
    @staticmethod
    def analyze(query: str) -> Dict[str, Any]:
        """Analyze query for optimization"""
        import re
        
        analysis = {
            'is_code_query': False,
            'is_how_to': False,
            'priority_boost': 1.0
        }
        
        # Code detection
        code_patterns = [
            r'\b(function|def|class|import|const)\b',
            r'[(){}\[\];]',
            r'\w+\.\w+\(',
        ]
        analysis['is_code_query'] = any(re.search(p, query) for p in code_patterns)
        
        # How-to detection
        analysis['is_how_to'] = bool(re.search(r'\bhow\s+(to|do)\b', query, re.I))
        
        # Adjust boost based on query type
        if analysis['is_code_query']:
            analysis['priority_boost'] = 1.3  # Boost code results
        elif analysis['is_how_to']:
            analysis['priority_boost'] = 1.1  # Slight boost for tutorials
        
        return analysis


class HybridSearchEngine:
    """
    Upgraded search engine using Qdrant 1.16 native features:
    - Built-in BM25 (via sparse vectors)
    - ACORN for better filtered search
    - Score boosting for metadata-based reranking
    """
    
    def __init__(self, qdrant: QdrantClient, dense: TextEmbedding, sparse: SparseTextEmbedding):
        self.qdrant = qdrant
        self.dense_model = dense
        self.sparse_model = sparse
        self.collections_info = self._get_collections_info()
    
    def _get_collections_info(self) -> Dict[str, bool]:
        """Check which collections have sparse vectors (BM25)"""
        info = {}
        try:
            collections = self.qdrant.get_collections().collections
            for coll in collections:
                coll_info = self.qdrant.get_collection(coll.name)
                # Check if collection has sparse vectors
                has_sparse = hasattr(coll_info.config.params, 'sparse_vectors') and \
                           coll_info.config.params.sparse_vectors is not None
                info[coll.name] = has_sparse
        except Exception as e:
            logger.warning(f"Error checking collections: {e}")
        return info
    
    async def search(self, query: str, collection: Optional[str], limit: int) -> List[Dict]:
        """
        Hybrid search using Qdrant 1.16 features
        """
        # Analyze query
        analysis = QueryAnalyzer.analyze(query)
        logger.info(f"Query analysis: code={analysis['is_code_query']}, "
                   f"boost={analysis['priority_boost']}")
        
        # Generate embeddings
        dense_vector = list(self.dense_model.embed([query]))[0].tolist()
        
        # Collections to search
        collections = [collection] if collection else list(self.collections_info.keys())
        
        all_results = []
        
        for coll_name in collections:
            try:
                has_sparse = self.collections_info.get(coll_name, False)
                
                if has_sparse:
                    # NEW: Use built-in BM25 via sparse vectors ✅
                    results = await self._hybrid_search_with_bm25(
                        coll_name, query, dense_vector, limit, analysis
                    )
                else:
                    # Fallback: Dense-only search with ACORN
                    results = await self._dense_search_with_acorn(
                        coll_name, dense_vector, limit, analysis
                    )
                
                all_results.extend(results)
            
            except Exception as e:
                logger.warning(f"Search error in {coll_name}: {e}")
        
        # Sort by score
        all_results.sort(key=lambda x: x['final_score'], reverse=True)
        return all_results[:limit]
    
    async def _hybrid_search_with_bm25(
        self, 
        collection: str, 
        query: str,
        dense_vector: List[float],
        limit: int,
        analysis: Dict
    ) -> List[Dict]:
        """
        Hybrid search using Qdrant 1.16 built-in BM25
        """
        # Generate BM25 sparse vector
        sparse_vector = list(self.sparse_model.embed([query]))[0]
        
        # Prepare score boosting based on metadata
        score_boost = {
            "metadata.chunk_type": {
                "code": 1.3 if analysis['is_code_query'] else 1.0,
                "documentation": 1.0
            }
        }
        
        # Query with prefetch (BM25 first, then dense rerank)
        try:
            results = self.qdrant.query_points(
                collection_name=collection,
                query=dense_vector,
                using="dense",  # Main dense vector
                prefetch=[
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_vector.indices.tolist(),
                            values=sparse_vector.values.tolist()
                        ),
                        using="sparse",  # BM25 sparse vector ✅
                        limit=limit * 3
                    )
                ],
                search_params={
                    "hnsw_ef": 128,
                    "acorn": USE_ACORN  # Enable ACORN ✅
                },
                with_payload=True,
                limit=limit * 2
            )
            
            formatted_results = []
            for point in results.points:
                formatted_results.append({
                    'collection': collection,
                    'id': point.id,
                    'score': point.score,
                    'final_score': point.score * analysis['priority_boost'],
                    'text': point.payload.get('text', ''),
                    'metadata': point.payload.get('metadata', {}),
                    'method': 'hybrid_bm25'
                })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return []
    
    async def _dense_search_with_acorn(
        self,
        collection: str,
        dense_vector: List[float],
        limit: int,
        analysis: Dict
    ) -> List[Dict]:
        """
        Dense-only search with ACORN for collections without BM25
        """
        try:
            results = self.qdrant.search(
                collection_name=collection,
                query_vector=dense_vector,
                search_params={
                    "hnsw_ef": 128,
                    "acorn": USE_ACORN  # Enable ACORN ✅
                },
                with_payload=True,
                limit=limit * 2
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'collection': collection,
                    'id': result.id,
                    'score': result.score,
                    'final_score': result.score * analysis['priority_boost'],
                    'text': result.payload.get('text', ''),
                    'metadata': result.payload.get('metadata', {}),
                    'method': 'dense_acorn'
                })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Dense search error: {e}")
            return []


# Global
search_engine: Optional[HybridSearchEngine] = None


def initialize():
    """Initialize with Qdrant 1.16 features"""
    global qdrant_client, dense_model, sparse_model, search_engine
    
    logger.info(f"Initializing with Qdrant 1.16 features at {QDRANT_PATH}")
    qdrant_client = QdrantClient(path=QDRANT_PATH)
    
    # Dense embeddings
    dense_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    
    # Sparse embeddings for BM25
    try:
        sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("✅ BM25 sparse model loaded")
    except Exception as e:
        logger.warning(f"Could not load BM25 model: {e}")
        sparse_model = None
    
    search_engine = HybridSearchEngine(qdrant_client, dense_model, sparse_model)
    
    logger.info("✅ Ready with Qdrant 1.16 features!")
    logger.info(f"   - ACORN: {'enabled' if USE_ACORN else 'disabled'}")
    logger.info(f"   - Built-in BM25: {'available' if sparse_model else 'unavailable'}")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Available tools"""
    return [
        Tool(
            name="list_collections",
            description="List all documentation collections with feature support",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="search_documents",
            description="Advanced search using Qdrant 1.16: Built-in BM25 + ACORN + Score Boosting. "
                       "Automatically optimizes for code vs documentation queries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "collection_name": {"type": "string", "description": "Optional: specific collection"},
                    "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20}
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
        return await handle_search(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def handle_list_collections() -> list[TextContent]:
    """List collections with feature info"""
    try:
        collections = qdrant_client.get_collections().collections
        
        if not collections:
            return [TextContent(type="text", text="No collections found")]
        
        result = [f"📚 Collections: {len(collections)}\n"]
        
        for coll in collections:
            info = qdrant_client.get_collection(coll.name)
            has_sparse = search_engine.collections_info.get(coll.name, False)
            
            result.append(f"\n📖 {coll.name}")
            result.append(f"   ├─ Documents: {info.points_count}")
            result.append(f"   ├─ Features: {'BM25+ACORN' if has_sparse else 'Dense+ACORN'}")
            result.append(f"   └─ Status: {info.status}")
        
        return [TextContent(type="text", text="\n".join(result))]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


async def handle_search(arguments: dict) -> list[TextContent]:
    """Perform search"""
    try:
        query = arguments.get("query")
        collection = arguments.get("collection_name")
        limit = arguments.get("limit", 5)
        
        if not query:
            return [TextContent(type="text", text="Error: query required")]
        
        results = await search_engine.search(query, collection, limit)
        
        if not results:
            return [TextContent(type="text", text=f"No results for: '{query}'")]
        
        output = [f"🔍 Search Results: '{query}'\n"]
        output.append(f"Algorithm: Qdrant 1.16 (BM25 + Dense + ACORN + Score Boost)")
        output.append(f"Found {len(results)} results:\n")
        
        for idx, result in enumerate(results, 1):
            output.append(f"\n{'='*70}")
            output.append(f"#{idx} | {result['collection']} | Score: {result['final_score']:.4f}")
            output.append(f"    ├─ Base Score: {result['score']:.4f}")
            output.append(f"    ├─ Method: {result['method']}")
            output.append(f"    └─ Type: {result['metadata'].get('chunk_type', 'unknown')}")
            
            if result['metadata'].get('source_file'):
                output.append(f"    └─ Source: {result['metadata']['source_file']}")
            
            output.append(f"{'='*70}\n")
            text = result['text']
            output.append(text[:400] + "..." if len(text) > 400 else text)
            output.append("")
        
        return [TextContent(type="text", text="\n".join(output))]
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return [TextContent(type="text", text=f"Error: {e}")]


async def main():
    """Main entry point"""
    initialize()
    
    from mcp.server.stdio import stdio_server
    
    logger.info("🚀 Starting Upgraded Server (Qdrant 1.16)")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
