#!/usr/bin/env python3
"""
Advanced MCP Server with Hybrid Search: BM25 + Semantic + Re-ranking
Combines keyword matching with semantic understanding for best results
"""

import asyncio
import logging
import os
import re
from typing import Any, List, Dict, Optional, Tuple
from collections import defaultdict
import math

from mcp.server import Server
from mcp.types import Tool, TextContent
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp-qdrant-advanced")

# Configuration
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_storage")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# Hybrid search weights
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))  # Vector search weight
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))          # Keyword search weight
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"

# Initialize components
app = Server("mcp-qdrant-advanced")
embedding_model: Optional[TextEmbedding] = None
qdrant_client: Optional[QdrantClient] = None


class BM25:
    """Simplified BM25 implementation for keyword ranking"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.num_docs = 0
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def calculate_idf(self, term: str) -> float:
        """Calculate inverse document frequency"""
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0
        return math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
    
    def score(self, query: str, doc_text: str, doc_id: str) -> float:
        """Calculate BM25 score for a document"""
        query_tokens = self.tokenize(query)
        doc_tokens = self.tokenize(doc_text)
        
        # Term frequencies in document
        term_freqs = defaultdict(int)
        for token in doc_tokens:
            term_freqs[token] += 1
        
        # Document length
        doc_length = len(doc_tokens)
        if doc_length == 0:
            return 0.0
        
        # Calculate BM25 score
        score = 0.0
        for term in query_tokens:
            if term not in term_freqs:
                continue
            
            tf = term_freqs[term]
            idf = self.calculate_idf(term)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / max(self.avg_doc_length, 1)))
            
            score += idf * (numerator / denominator)
        
        return score


class HybridSearchEngine:
    """Advanced hybrid search combining multiple techniques"""
    
    def __init__(self, qdrant_client: QdrantClient, embedding_model: TextEmbedding):
        self.qdrant = qdrant_client
        self.embedding_model = embedding_model
        self.bm25 = BM25()
        self._init_bm25_stats()
    
    def _init_bm25_stats(self):
        """Initialize BM25 statistics from existing collections"""
        try:
            collections = self.qdrant.get_collections().collections
            total_length = 0
            doc_count = 0
            
            for collection in collections:
                try:
                    # Sample points to build document frequency statistics
                    points, _ = self.qdrant.scroll(
                        collection_name=collection.name,
                        limit=1000,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    for point in points:
                        text = point.payload.get("text", "")
                        tokens = self.bm25.tokenize(text)
                        
                        # Update document frequencies
                        seen_terms = set()
                        for token in tokens:
                            if token not in seen_terms:
                                self.bm25.doc_freqs[token] += 1
                                seen_terms.add(token)
                        
                        # Track document lengths
                        doc_id = f"{collection.name}_{point.id}"
                        self.bm25.doc_lengths[doc_id] = len(tokens)
                        total_length += len(tokens)
                        doc_count += 1
                
                except Exception as e:
                    logger.warning(f"Error initializing BM25 for {collection.name}: {e}")
            
            self.bm25.num_docs = doc_count
            self.bm25.avg_doc_length = total_length / doc_count if doc_count > 0 else 0
            
            logger.info(f"BM25 initialized: {doc_count} documents, avg length: {self.bm25.avg_doc_length:.1f}")
        
        except Exception as e:
            logger.warning(f"Could not initialize BM25: {e}")
    
    def _is_code_query(self, query: str) -> bool:
        """Detect if query is code-related (needs exact matching)"""
        code_indicators = [
            r'\b(function|class|def|import|const|let|var)\b',
            r'[(){}\[\];]',
            r'\b\w+\.\w+\(',  # method calls
            r'[A-Z][a-z]+[A-Z]',  # CamelCase
            r'_[a-z]+_',  # snake_case
        ]
        return any(re.search(pattern, query) for pattern in code_indicators)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 
                    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'should', 'could', 'may', 'might', 'can', 'how', 'what', 
                    'where', 'when', 'why', 'which', 'who', 'this', 'that', 'these', 'those'}
        
        tokens = self.bm25.tokenize(query)
        keywords = [t for t in tokens if t not in stopwords and len(t) > 2]
        
        # Add exact phrases (quoted)
        phrases = re.findall(r'"([^"]+)"', query)
        keywords.extend(phrases)
        
        return keywords
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Re-rank results based on multiple signals"""
        if not RERANK_ENABLED or not results:
            return results
        
        keywords = self._extract_keywords(query)
        is_code = self._is_code_query(query)
        
        for result in results:
            text = result['text'].lower()
            rerank_score = result['score']
            
            # Boost for exact keyword matches
            keyword_matches = sum(1 for kw in keywords if kw in text)
            if keyword_matches > 0:
                rerank_score *= (1 + 0.1 * keyword_matches)
            
            # Boost for exact phrase matches
            phrase_matches = sum(1 for phrase in re.findall(r'"([^"]+)"', query) if phrase.lower() in text)
            if phrase_matches > 0:
                rerank_score *= (1 + 0.2 * phrase_matches)
            
            # Boost for code snippets if code query
            if is_code and ('```' in result['text'] or 'def ' in text or 'function ' in text):
                rerank_score *= 1.3
            
            # Boost for shorter, focused results
            text_length = len(result['text'])
            if text_length < 500:
                rerank_score *= 1.1
            elif text_length > 2000:
                rerank_score *= 0.9
            
            # Boost for results with metadata
            if result.get('metadata', {}).get('chunk_index', 0) == 0:
                rerank_score *= 1.05  # First chunk often has overview
            
            result['rerank_score'] = rerank_score
        
        # Sort by reranked score
        results.sort(key=lambda x: x.get('rerank_score', x['score']), reverse=True)
        return results
    
    async def hybrid_search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Perform hybrid search: Semantic (vector) + BM25 (keyword) + Re-ranking
        """
        logger.info(f"Hybrid search: '{query}' in {collection_name or 'all collections'}")
        
        # Generate query embedding
        query_vector = list(self.embedding_model.embed([query]))[0].tolist()
        
        # Determine collections to search
        if collection_name:
            collections = [collection_name]
        else:
            all_collections = self.qdrant.get_collections().collections
            collections = [c.name for c in all_collections]
        
        # Perform semantic search on each collection
        semantic_results = []
        for coll in collections:
            try:
                # Get more results for re-ranking
                search_limit = limit * 3
                
                results = self.qdrant.search(
                    collection_name=coll,
                    query_vector=query_vector,
                    limit=search_limit,
                    with_payload=True
                )
                
                for result in results:
                    semantic_results.append({
                        "collection": coll,
                        "id": result.id,
                        "semantic_score": result.score,
                        "text": result.payload.get("text", ""),
                        "metadata": result.payload.get("metadata", {})
                    })
            
            except Exception as e:
                logger.warning(f"Error searching {coll}: {e}")
        
        # Calculate BM25 scores for semantic results
        for result in semantic_results:
            doc_id = f"{result['collection']}_{result['id']}"
            bm25_score = self.bm25.score(query, result['text'], doc_id)
            result['bm25_score'] = bm25_score
            
            # Combine scores (weighted average)
            # Normalize scores to 0-1 range
            norm_semantic = result['semantic_score']
            norm_bm25 = min(bm25_score / 10.0, 1.0)  # BM25 scores can vary widely
            
            combined_score = (SEMANTIC_WEIGHT * norm_semantic + BM25_WEIGHT * norm_bm25)
            result['score'] = combined_score
        
        # Sort by combined score
        semantic_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Re-rank top results
        top_results = semantic_results[:limit * 2]  # Get more for re-ranking
        reranked_results = self._rerank_results(query, top_results)
        
        # Return final top results
        return reranked_results[:limit]


# Global search engine
search_engine: Optional[HybridSearchEngine] = None


def initialize_clients():
    """Initialize Qdrant client, embedding model, and search engine"""
    global qdrant_client, embedding_model, search_engine
    
    logger.info(f"Initializing Qdrant client at: {QDRANT_PATH}")
    qdrant_client = QdrantClient(path=QDRANT_PATH)
    
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    
    logger.info("Initializing hybrid search engine...")
    search_engine = HybridSearchEngine(qdrant_client, embedding_model)
    
    logger.info("✅ Initialization complete")
    logger.info(f"⚙️  Semantic weight: {SEMANTIC_WEIGHT}, BM25 weight: {BM25_WEIGHT}")
    logger.info(f"⚙️  Re-ranking: {'enabled' if RERANK_ENABLED else 'disabled'}")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="list_collections",
            description="List all available documentation collections with statistics. "
                       "Shows collection names, document counts, and vector counts.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="search_documents",
            description="Advanced hybrid search using semantic understanding + keyword matching + re-ranking. "
                       "Automatically detects code queries for exact matching. "
                       "Best for finding both conceptual information and specific code examples.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query - can be natural language or code snippets"
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Optional: Search only in specific collection"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results (default: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_collection_info",
            description="Get detailed information about a specific collection including "
                       "sample documents and metadata statistics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection to inspect"
                    }
                },
                "required": ["collection_name"]
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
    elif name == "get_collection_info":
        return await handle_get_collection_info(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def handle_list_collections() -> list[TextContent]:
    """List all collections with statistics"""
    try:
        collections = qdrant_client.get_collections().collections
        
        if not collections:
            return [TextContent(
                type="text",
                text="No collections found. Create collections using single_create_local_db.py"
            )]
        
        result = ["📚 Available Documentation Collections:\n"]
        result.append(f"Total collections: {len(collections)}\n")
        
        for collection in collections:
            info = qdrant_client.get_collection(collection.name)
            result.append(f"\n📖 {collection.name}")
            result.append(f"   └─ Documents: {info.points_count}")
            result.append(f"   └─ Vectors: {info.vectors_count}")
            result.append(f"   └─ Status: {info.status}")
        
        return [TextContent(type="text", text="\n".join(result))]
    
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_search_documents(arguments: dict) -> list[TextContent]:
    """Perform hybrid search"""
    try:
        query = arguments.get("query")
        collection_name = arguments.get("collection_name")
        limit = arguments.get("limit", 5)
        
        if not query:
            return [TextContent(type="text", text="Error: query parameter required")]
        
        # Perform hybrid search
        results = await search_engine.hybrid_search(query, collection_name, limit)
        
        if not results:
            return [TextContent(type="text", text=f"No results found for: '{query}'")]
        
        # Format results
        output = [f"🔍 Hybrid Search Results for: '{query}'\n"]
        output.append(f"Algorithm: Semantic ({SEMANTIC_WEIGHT}) + BM25 ({BM25_WEIGHT}) + Re-ranking")
        output.append(f"Found {len(results)} results:\n")
        
        for idx, result in enumerate(results, 1):
            output.append(f"\n{'='*70}")
            output.append(f"#{idx} | {result['collection']} | Score: {result['score']:.4f}")
            
            # Show score breakdown
            output.append(f"    └─ Semantic: {result['semantic_score']:.3f} | "
                         f"BM25: {result['bm25_score']:.3f} | "
                         f"Rerank: {result.get('rerank_score', result['score']):.3f}")
            
            metadata = result.get('metadata', {})
            if metadata.get('source_file'):
                output.append(f"    └─ Source: {metadata['source_file']}")
            if 'chunk_index' in metadata:
                output.append(f"    └─ Chunk: {metadata['chunk_index']}/{metadata.get('total_chunks', '?')}")
            
            output.append(f"{'='*70}\n")
            output.append(result['text'])
            output.append("")
        
        return [TextContent(type="text", text="\n".join(output))]
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_get_collection_info(arguments: dict) -> list[TextContent]:
    """Get detailed collection information"""
    try:
        collection_name = arguments.get("collection_name")
        
        if not collection_name:
            return [TextContent(type="text", text="Error: collection_name required")]
        
        # Get collection info
        info = qdrant_client.get_collection(collection_name)
        
        # Sample some documents
        points, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=3,
            with_payload=True,
            with_vectors=False
        )
        
        output = [f"📖 Collection: {collection_name}\n"]
        output.append(f"Status: {info.status}")
        output.append(f"Points: {info.points_count}")
        output.append(f"Vectors: {info.vectors_count}")
        output.append(f"Vector size: {info.config.params.vectors.size}")
        output.append(f"Distance: {info.config.params.vectors.distance}")
        
        if points:
            output.append(f"\n📄 Sample Documents:")
            for i, point in enumerate(points, 1):
                text = point.payload.get("text", "")
                preview = text[:200] + "..." if len(text) > 200 else text
                output.append(f"\n{i}. Preview:")
                output.append(f"   {preview}")
                
                metadata = point.payload.get("metadata", {})
                if metadata:
                    output.append(f"   Metadata: {metadata}")
        
        return [TextContent(type="text", text="\n".join(output))]
    
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Main entry point"""
    initialize_clients()
    
    from mcp.server.stdio import stdio_server
    
    logger.info("🚀 Starting Advanced MCP Qdrant Server")
    logger.info(f"📁 Storage: {QDRANT_PATH}")
    logger.info(f"🤖 Model: {EMBEDDING_MODEL}")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
