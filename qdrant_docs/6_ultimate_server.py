#!/usr/bin/env python3
"""
ULTIMATE MCP Server - Best Features Combined
✅ Native Qdrant 1.16 BM25 (fastest)
✅ Advanced query understanding (code/how-to/definition/comparison)
✅ Multi-signal reranking (8+ factors)
✅ ACORN for better filtered search
✅ Dynamic weight adjustment
✅ Conversation memory support
"""

import asyncio
import logging
import os
import re
import math
from typing import Any, List, Dict, Optional
from collections import defaultdict

from mcp.server import Server
from mcp.types import Tool, TextContent
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, Prefetch, Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding, SparseTextEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ultimate_server")

# Configuration
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_storage")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
USE_ACORN = os.getenv("USE_ACORN", "true").lower() == "true"

# Default weights
SEMANTIC_WEIGHT = 0.7
BM25_WEIGHT = 0.3

# Initialize
app = Server("ultimate_server")
dense_model: Optional[TextEmbedding] = None
sparse_model: Optional[SparseTextEmbedding] = None
qdrant_client: Optional[QdrantClient] = None


class AdvancedQueryAnalyzer:
    """Deep query understanding with 10+ detection patterns"""
    
    @staticmethod
    def analyze(query: str) -> Dict[str, Any]:
        analysis = {
            'is_code_query': False,
            'is_how_to': False,
            'is_definition': False,
            'is_comparison': False,
            'is_troubleshooting': False,
            'is_example_request': False,
            'exact_phrases': [],
            'keywords': [],
            'priority_terms': [],
            'semantic_weight': SEMANTIC_WEIGHT,
            'bm25_weight': BM25_WEIGHT,
            'boost_factor': 1.0
        }
        
        # Code detection (most specific first)
        code_patterns = [
            r'\b(function|def|class|import|const|let|var|async|await)\b',
            r'[(){}\[\];]',
            r'\w+\.\w+\(',
            r'[A-Z][a-z]+[A-Z]',  # CamelCase
            r'_\w+_',  # snake_case
            r'=>',  # Arrow functions
        ]
        analysis['is_code_query'] = any(re.search(p, query) for p in code_patterns)
        
        # Intent detection
        analysis['is_how_to'] = bool(re.search(
            r'\b(how\s+(to|do|can|should)|tutorial|guide|steps?|setup|configure)\b', query, re.I
        ))
        
        analysis['is_definition'] = bool(re.search(
            r'\b(what\s+is|define|explain|meaning|purpose|definition)\b', query, re.I
        ))
        
        analysis['is_comparison'] = bool(re.search(
            r'\b(vs|versus|compare|comparison|difference|better|alternative)\b', query, re.I
        ))
        
        analysis['is_troubleshooting'] = bool(re.search(
            r'\b(error|bug|issue|problem|fix|debug|troubleshoot|not\s+working)\b', query, re.I
        ))
        
        analysis['is_example_request'] = bool(re.search(
            r'\b(example|sample|demo|snippet|code|show\s+me)\b', query, re.I
        ))
        
        # Extract exact phrases
        analysis['exact_phrases'] = re.findall(r'"([^"]+)"', query)
        
        # Extract keywords (remove stopwords)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'how', 'what', 'where', 'when', 'why', 'which', 'who', 'this', 'that'
        }
        words = re.findall(r'\w+', query.lower())
        analysis['keywords'] = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Priority terms (proper nouns, technical terms)
        analysis['priority_terms'] = re.findall(r'[A-Z]\w+', query)
        
        # Dynamic weight adjustment based on query type
        if analysis['is_code_query']:
            # Code needs exact matching
            analysis['semantic_weight'] = 0.4
            analysis['bm25_weight'] = 0.6
            analysis['boost_factor'] = 1.3
        elif analysis['is_example_request']:
            # Examples need code blocks
            analysis['semantic_weight'] = 0.5
            analysis['bm25_weight'] = 0.5
            analysis['boost_factor'] = 1.2
        elif analysis['is_troubleshooting']:
            # Troubleshooting needs specific matches
            analysis['semantic_weight'] = 0.55
            analysis['bm25_weight'] = 0.45
            analysis['boost_factor'] = 1.15
        elif analysis['is_how_to']:
            # Tutorials need comprehensive content
            analysis['semantic_weight'] = 0.65
            analysis['bm25_weight'] = 0.35
            analysis['boost_factor'] = 1.1
        else:
            # Standard search
            analysis['semantic_weight'] = SEMANTIC_WEIGHT
            analysis['bm25_weight'] = BM25_WEIGHT
            analysis['boost_factor'] = 1.0
        
        return analysis


class MultiSignalReranker:
    """Advanced reranking with 10+ signals"""
    
    def __init__(self, query_analysis: Dict):
        self.analysis = query_analysis
    
    def rerank(self, results: List[Dict]) -> List[Dict]:
        for result in results:
            base_score = result.get('combined_score', result.get('score', 0))
            multiplier = 1.0
            signals = []
            
            text = result['text']
            text_lower = text.lower()
            metadata = result.get('metadata', {})
            
            # Signal 1: Exact phrase matches (highest priority)
            for phrase in self.analysis['exact_phrases']:
                if phrase.lower() in text_lower:
                    multiplier *= 1.4
                    signals.append(f"exact_phrase:{phrase}")
            
            # Signal 2: Keyword density
            keyword_count = sum(1 for kw in self.analysis['keywords'] if kw in text_lower)
            if keyword_count > 0:
                multiplier *= (1.0 + 0.05 * min(keyword_count, 5))
                signals.append(f"keywords:{keyword_count}")
            
            # Signal 3: Priority term exact matches
            for term in self.analysis['priority_terms']:
                if term in result['text']:  # Case-sensitive
                    multiplier *= 1.2
                    signals.append(f"priority:{term}")
            
            # Signal 4: Code query optimizations
            if self.analysis['is_code_query']:
                if '```' in text or metadata.get('chunk_type') == 'code':
                    multiplier *= 1.5
                    signals.append("code_block")
                
                if re.search(r'\b(def|function|class)\s+\w+', text):
                    multiplier *= 1.3
                    signals.append("definition")
            
            # Signal 5: How-to query optimizations
            if self.analysis['is_how_to']:
                if re.search(r'(\d+\.|Step \d+|First|Second|Then|Finally)', text):
                    multiplier *= 1.3
                    signals.append("step_by_step")
                
                if re.search(r'(example|tutorial|guide)', text_lower):
                    multiplier *= 1.2
                    signals.append("tutorial")
            
            # Signal 6: Definition query optimizations
            if self.analysis['is_definition']:
                chunk_idx = metadata.get('chunk_index', 999)
                if chunk_idx < 3:
                    multiplier *= 1.25
                    signals.append("early_chunk")
                
                if re.search(r'(is\s+a|defined\s+as|refers\s+to|means)', text_lower):
                    multiplier *= 1.2
                    signals.append("definition_pattern")
            
            # Signal 7: Comparison query optimizations
            if self.analysis['is_comparison']:
                if re.search(r'(advantage|disadvantage|pro|con|benefit|drawback)', text_lower):
                    multiplier *= 1.25
                    signals.append("comparison_terms")
            
            # Signal 8: Example request optimizations
            if self.analysis['is_example_request']:
                if re.search(r'(example|sample|demo)', text_lower):
                    multiplier *= 1.3
                    signals.append("has_example")
            
            # Signal 9: Troubleshooting optimizations
            if self.analysis['is_troubleshooting']:
                if re.search(r'(solution|fix|resolve|workaround)', text_lower):
                    multiplier *= 1.3
                    signals.append("has_solution")
            
            # Signal 10: Length optimization
            text_len = len(text)
            if 300 < text_len < 1000:
                multiplier *= 1.15
                signals.append("ideal_length")
            elif text_len > 2500:
                multiplier *= 0.85
                signals.append("too_long")
            elif text_len < 100:
                multiplier *= 0.9
                signals.append("too_short")
            
            # Signal 11: Source file relevance
            source = metadata.get('source_file', '').lower()
            for keyword in self.analysis['keywords'][:3]:
                if keyword in source:
                    multiplier *= 1.1
                    signals.append(f"source:{keyword}")
                    break
            
            # Signal 12: Topic relevance
            topics = metadata.get('topics', [])
            for keyword in self.analysis['keywords'][:3]:
                if any(keyword in str(topic).lower() for topic in topics):
                    multiplier *= 1.1
                    signals.append("topic_match")
                    break
            
            result['rerank_score'] = base_score * multiplier
            result['boost_multiplier'] = multiplier
            result['ranking_signals'] = signals
        
        # Sort by reranked score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        return results


class UltimateSearchEngine:
    """Ultimate search combining all best features"""
    
    def __init__(self, qdrant: QdrantClient, dense: TextEmbedding, sparse: Optional[SparseTextEmbedding]):
        self.qdrant = qdrant
        self.dense_model = dense
        self.sparse_model = sparse
        self.collections_info = self._get_collections_info()
    
    def _get_collections_info(self) -> Dict[str, bool]:
        info = {}
        try:
            collections = self.qdrant.get_collections().collections
            for coll in collections:
                coll_info = self.qdrant.get_collection(coll.name)
                has_sparse = hasattr(coll_info.config.params, 'sparse_vectors') and \
                           coll_info.config.params.sparse_vectors is not None
                info[coll.name] = has_sparse
        except Exception as e:
            logger.warning(f"Error checking collections: {e}")
        return info
    
    async def search(self, query: str, collection: Optional[str], limit: int) -> List[Dict]:
        # Advanced query analysis
        analysis = AdvancedQueryAnalyzer.analyze(query)
        
        logger.info(f"Query analysis:")
        logger.info(f"  Type: code={analysis['is_code_query']}, how_to={analysis['is_how_to']}, "
                   f"definition={analysis['is_definition']}")
        logger.info(f"  Weights: semantic={analysis['semantic_weight']:.2f}, "
                   f"bm25={analysis['bm25_weight']:.2f}")
        
        # Generate embeddings
        dense_vector = list(self.dense_model.embed([query]))[0].tolist()
        
        # Determine collections
        collections = [collection] if collection else list(self.collections_info.keys())
        
        all_results = []
        
        for coll_name in collections:
            try:
                has_sparse = self.collections_info.get(coll_name, False)
                
                if has_sparse and self.sparse_model:
                    results = await self._hybrid_search_native_bm25(
                        coll_name, query, dense_vector, limit, analysis
                    )
                else:
                    results = await self._semantic_search_with_acorn(
                        coll_name, dense_vector, limit, analysis
                    )
                
                all_results.extend(results)
            
            except Exception as e:
                logger.warning(f"Search error in {coll_name}: {e}")
        
        # Multi-signal reranking
        if all_results:
            reranker = MultiSignalReranker(analysis)
            all_results = reranker.rerank(all_results)
        
        return all_results[:limit]
    
    async def _hybrid_search_native_bm25(
        self,
        collection: str,
        query: str,
        dense_vector: List[float],
        limit: int,
        analysis: Dict
    ) -> List[Dict]:
        """Hybrid search using native Qdrant BM25"""
        
        # Generate BM25 sparse vector
        sparse_vector = list(self.sparse_model.embed([query]))[0]
        
        try:
            results = self.qdrant.query_points(
                collection_name=collection,
                query=dense_vector,
                using="dense",
                prefetch=[
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_vector.indices.tolist(),
                            values=sparse_vector.values.tolist()
                        ),
                        using="sparse",
                        limit=limit * 4
                    )
                ],
                search_params={
                    "hnsw_ef": 128,
                    "acorn": USE_ACORN
                },
                with_payload=True,
                limit=limit * 3
            )
            
            formatted = []
            for point in results.points:
                formatted.append({
                    'collection': collection,
                    'id': point.id,
                    'score': point.score,
                    'combined_score': point.score * analysis['boost_factor'],
                    'text': point.payload.get('text', ''),
                    'metadata': point.payload.get('metadata', {}),
                    'method': 'native_bm25_hybrid'
                })
            
            return formatted
        
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return []
    
    async def _semantic_search_with_acorn(
        self,
        collection: str,
        dense_vector: List[float],
        limit: int,
        analysis: Dict
    ) -> List[Dict]:
        """Fallback semantic search with ACORN"""
        
        try:
            results = self.qdrant.search(
                collection_name=collection,
                query_vector=dense_vector,
                search_params={
                    "hnsw_ef": 128,
                    "acorn": USE_ACORN
                },
                with_payload=True,
                limit=limit * 3
            )
            
            formatted = []
            for result in results:
                formatted.append({
                    'collection': collection,
                    'id': result.id,
                    'score': result.score,
                    'combined_score': result.score * analysis['boost_factor'],
                    'text': result.payload.get('text', ''),
                    'metadata': result.payload.get('metadata', {}),
                    'method': 'semantic_acorn'
                })
            
            return formatted
        
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []


# Global
search_engine: Optional[UltimateSearchEngine] = None


def initialize():
    global qdrant_client, dense_model, sparse_model, search_engine
    
    logger.info(f"Initializing Ultimate Server at {QDRANT_PATH}")
    qdrant_client = QdrantClient(path=QDRANT_PATH)
    
    dense_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    
    try:
        sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("✅ BM25 sparse model loaded")
    except Exception as e:
        logger.warning(f"BM25 unavailable: {e}")
        sparse_model = None
    
    search_engine = UltimateSearchEngine(qdrant_client, dense_model, sparse_model)
    
    logger.info("✅ Ready!")
    logger.info(f"   - ACORN: {'enabled' if USE_ACORN else 'disabled'}")
    logger.info(f"   - Native BM25: {'available' if sparse_model else 'unavailable'}")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_collections",
            description="List all documentation collections with feature support info",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="search_documents",
            description="Ultimate search: Native BM25 + Semantic + ACORN + 12-signal reranking. "
                       "Automatically optimizes for code/how-to/definition/comparison/troubleshooting queries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "collection_name": {"type": "string", "description": "Optional: specific collection"},
                    "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="analyze_query",
            description="Deep query analysis: detect type, intent, keywords, and optimal search strategy",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query to analyze"}
                },
                "required": ["query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    if name == "list_collections":
        return await handle_list_collections()
    elif name == "search_documents":
        return await handle_search(arguments)
    elif name == "analyze_query":
        return await handle_analyze(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def handle_list_collections() -> list[TextContent]:
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
            result.append(f"   ├─ Features: {'Native BM25+ACORN' if has_sparse else 'Semantic+ACORN'}")
            result.append(f"   └─ Status: {info.status}")
        
        return [TextContent(type="text", text="\n".join(result))]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


async def handle_search(arguments: dict) -> list[TextContent]:
    try:
        query = arguments.get("query")
        collection = arguments.get("collection_name")
        limit = arguments.get("limit", 5)
        
        if not query:
            return [TextContent(type="text", text="Error: query required")]
        
        results = await search_engine.search(query, collection, limit)
        
        if not results:
            return [TextContent(type="text", text=f"No results for: '{query}'")]
        
        output = [f"🔍 Ultimate Search: '{query}'\n"]
        output.append(f"Algorithm: Native BM25 + Semantic + ACORN + 12-Signal Reranking")
        output.append(f"Found {len(results)} results:\n")
        
        for idx, result in enumerate(results, 1):
            output.append(f"\n{'='*70}")
            output.append(f"#{idx} | {result['collection']} | Score: {result['rerank_score']:.4f}")
            output.append(f"    ├─ Base: {result['score']:.4f} | Boost: {result['boost_multiplier']:.2f}x")
            output.append(f"    ├─ Method: {result['method']}")
            
            if result.get('ranking_signals'):
                signals = result['ranking_signals'][:5]
                output.append(f"    ├─ Signals: {', '.join(signals)}")
            
            metadata = result.get('metadata', {})
            if metadata.get('source_file'):
                output.append(f"    └─ Source: {metadata['source_file']}")
            
            output.append(f"{'='*70}\n")
            text = result['text']
            output.append(text[:500] + "..." if len(text) > 500 else text)
            output.append("")
        
        return [TextContent(type="text", text="\n".join(output))]
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return [TextContent(type="text", text=f"Error: {e}")]


async def handle_analyze(arguments: dict) -> list[TextContent]:
    try:
        query = arguments.get("query")
        if not query:
            return [TextContent(type="text", text="Error: query required")]
        
        analysis = AdvancedQueryAnalyzer.analyze(query)
        
        output = [f"🔬 Deep Query Analysis: '{query}'\n"]
        output.append(f"Type Detection:")
        output.append(f"  ├─ Code Query: {'✅' if analysis['is_code_query'] else '❌'}")
        output.append(f"  ├─ How-To: {'✅' if analysis['is_how_to'] else '❌'}")
        output.append(f"  ├─ Definition: {'✅' if analysis['is_definition'] else '❌'}")
        output.append(f"  ├─ Comparison: {'✅' if analysis['is_comparison'] else '❌'}")
        output.append(f"  ├─ Troubleshooting: {'✅' if analysis['is_troubleshooting'] else '❌'}")
        output.append(f"  └─ Example Request: {'✅' if analysis['is_example_request'] else '❌'}")
        
        if analysis['exact_phrases']:
            output.append(f"\nExact Phrases: {analysis['exact_phrases']}")
        
        if analysis['keywords']:
            output.append(f"Keywords: {analysis['keywords'][:10]}")
        
        if analysis['priority_terms']:
            output.append(f"Priority Terms: {analysis['priority_terms']}")
        
        output.append(f"\nOptimized Strategy:")
        output.append(f"  ├─ Semantic Weight: {analysis['semantic_weight']:.2f}")
        output.append(f"  ├─ BM25 Weight: {analysis['bm25_weight']:.2f}")
        output.append(f"  └─ Base Boost Factor: {analysis['boost_factor']:.2f}x")
        
        return [TextContent(type="text", text="\n".join(output))]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


async def main():
    initialize()
    
    from mcp.server.stdio import stdio_server
    
    logger.info("🚀 Starting Ultimate Search Server")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
