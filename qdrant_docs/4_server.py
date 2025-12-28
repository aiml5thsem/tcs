#!/usr/bin/env python3
"""
4_server.py - Ultimate MCP Server with Advanced Hybrid Search
BM25 + Semantic + Vector + Re-ranking + Query Understanding
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
from fastembed import TextEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("4_server")

# Configuration
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_storage")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# Hybrid weights
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))

# Initialize
app = Server("4_server")
embedding_model: Optional[TextEmbedding] = None
qdrant_client: Optional[QdrantClient] = None


class BM25Ranker:
    """Optimized BM25 implementation"""
    
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.num_docs = 0
    
    def tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization"""
        # Preserve code patterns
        text = re.sub(r'([A-Z][a-z]+)([A-Z])', r'\1 \2', text)  # CamelCase
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def build_index(self, documents: List[Dict]):
        """Build BM25 index"""
        self.num_docs = len(documents)
        total_length = 0
        
        for doc in documents:
            doc_id = f"{doc['collection']}_{doc['id']}"
            tokens = self.tokenize(doc['text'])
            
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            seen = set()
            for token in tokens:
                if token not in seen:
                    self.doc_freqs[token] += 1
                    seen.add(token)
        
        self.avg_doc_length = total_length / max(self.num_docs, 1)
    
    def score(self, query: str, doc_id: str, doc_text: str) -> float:
        """Calculate BM25 score"""
        query_tokens = self.tokenize(query)
        doc_tokens = self.tokenize(doc_text)
        
        term_freqs = defaultdict(int)
        for token in doc_tokens:
            term_freqs[token] += 1
        
        doc_length = len(doc_tokens)
        if doc_length == 0:
            return 0.0
        
        score = 0.0
        for term in query_tokens:
            if term not in term_freqs:
                continue
            
            tf = term_freqs[term]
            df = self.doc_freqs.get(term, 0)
            
            if df == 0:
                continue
            
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / max(self.avg_doc_length, 1)))
            
            score += idf * (numerator / denominator)
        
        return score


class QueryUnderstanding:
    """Analyze query intent and type"""
    
    @staticmethod
    def analyze(query: str) -> Dict[str, Any]:
        """Analyze query characteristics"""
        analysis = {
            'is_code_query': False,
            'is_how_to': False,
            'is_definition': False,
            'is_comparison': False,
            'exact_phrases': [],
            'keywords': [],
            'priority_terms': []
        }
        
        # Code query detection
        code_patterns = [
            r'\b(function|def|class|import|const|let|var)\b',
            r'[(){}\[\];]',
            r'\w+\.\w+\(',
            r'[A-Z][a-z]+[A-Z]',
        ]
        analysis['is_code_query'] = any(re.search(p, query) for p in code_patterns)
        
        # Intent detection
        analysis['is_how_to'] = bool(re.search(r'\bhow\s+(to|do|can)\b', query, re.I))
        analysis['is_definition'] = bool(re.search(r'\b(what\s+is|define|explain)\b', query, re.I))
        analysis['is_comparison'] = bool(re.search(r'\b(vs|versus|compared|difference|better)\b', query, re.I))
        
        # Extract exact phrases
        analysis['exact_phrases'] = re.findall(r'"([^"]+)"', query)
        
        # Extract keywords (remove stopwords)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'how', 'what'}
        words = re.findall(r'\w+', query.lower())
        analysis['keywords'] = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Identify priority terms (technical terms, proper nouns)
        analysis['priority_terms'] = [w for w in re.findall(r'[A-Z]\w+', query)]
        
        return analysis


class AdvancedReranker:
    """Multi-signal reranking"""
    
    def __init__(self, query_analysis: Dict):
        self.analysis = query_analysis
    
    def rerank(self, results: List[Dict]) -> List[Dict]:
        """Apply reranking signals"""
        for result in results:
            base_score = result['combined_score']
            multiplier = 1.0
            
            text_lower = result['text'].lower()
            
            # Exact phrase match boost
            for phrase in self.analysis['exact_phrases']:
                if phrase.lower() in text_lower:
                    multiplier *= 1.3
            
            # Keyword density boost
            keyword_count = sum(1 for kw in self.analysis['keywords'] if kw in text_lower)
            if keyword_count > 0:
                multiplier *= (1.0 + 0.05 * keyword_count)
            
            # Code query boosts
            if self.analysis['is_code_query']:
                # Boost code blocks
                if '```' in result['text'] or result.get('metadata', {}).get('chunk_type') == 'code':
                    multiplier *= 1.4
                
                # Boost function definitions
                if re.search(r'\b(def|function|class)\s+\w+', result['text']):
                    multiplier *= 1.2
            
            # How-to query boosts
            if self.analysis['is_how_to']:
                # Boost step-by-step content
                if re.search(r'\d+\.|Step \d+|First|Second|Finally', result['text']):
                    multiplier *= 1.25
            
            # Definition query boosts
            if self.analysis['is_definition']:
                # Boost content with definitions
                if result.get('metadata', {}).get('chunk_index', 999) < 3:
                    multiplier *= 1.2  # Early chunks often have definitions
            
            # Length optimization
            text_len = len(result['text'])
            if 300 < text_len < 1000:
                multiplier *= 1.1  # Ideal length
            elif text_len > 2500:
                multiplier *= 0.85  # Too long
            
            # Priority term exact match
            for term in self.analysis['priority_terms']:
                if term in result['text']:
                    multiplier *= 1.15
            
            result['rerank_score'] = base_score * multiplier
            result['boost_factor'] = multiplier
        
        # Sort by rerank score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        return results


class HybridSearchEngine:
    """Advanced hybrid search"""
    
    def __init__(self, qdrant: QdrantClient, embedding: TextEmbedding):
        self.qdrant = qdrant
        self.embedding = embedding
        self.bm25 = BM25Ranker()
        self._init_bm25()
    
    def _init_bm25(self):
        """Initialize BM25 index"""
        try:
            collections = self.qdrant.get_collections().collections
            all_docs = []
            
            for collection in collections:
                points, _ = self.qdrant.scroll(
                    collection_name=collection.name,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False
                )
                
                for point in points:
                    all_docs.append({
                        'collection': collection.name,
                        'id': point.id,
                        'text': point.payload.get('text', '')
                    })
            
            if all_docs:
                self.bm25.build_index(all_docs)
                logger.info(f"BM25 indexed {len(all_docs)} documents")
        
        except Exception as e:
            logger.warning(f"BM25 init error: {e}")
    
    async def search(self, query: str, collection: Optional[str], limit: int) -> List[Dict]:
        """Perform hybrid search"""
        
        # Analyze query
        query_analysis = QueryUnderstanding.analyze(query)
        logger.info(f"Query type: code={query_analysis['is_code_query']}, "
                   f"how_to={query_analysis['is_how_to']}")
        
        # Generate embedding
        query_vector = list(self.embedding.embed([query]))[0].tolist()
        
        # Determine collections
        collections = [collection] if collection else [c.name for c in self.qdrant.get_collections().collections]
        
        # Semantic search
        semantic_results = []
        search_limit = limit * 4  # Get more for reranking
        
        for coll in collections:
            try:
                results = self.qdrant.search(
                    collection_name=coll,
                    query_vector=query_vector,
                    limit=search_limit,
                    with_payload=True
                )
                
                for result in results:
                    semantic_results.append({
                        'collection': coll,
                        'id': result.id,
                        'semantic_score': result.score,
                        'text': result.payload.get('text', ''),
                        'metadata': result.payload.get('metadata', {})
                    })
            except Exception as e:
                logger.warning(f"Search error in {coll}: {e}")
        
        # Calculate BM25 scores
        for result in semantic_results:
            doc_id = f"{result['collection']}_{result['id']}"
            bm25_score = self.bm25.score(query, doc_id, result['text'])
            result['bm25_score'] = bm25_score
            
            # Combine scores
            norm_semantic = result['semantic_score']
            norm_bm25 = min(bm25_score / 10.0, 1.0)
            
            # Dynamic weighting based on query type
            if query_analysis['is_code_query']:
                # Prefer exact matching for code
                semantic_w = 0.4
                bm25_w = 0.6
            else:
                semantic_w = SEMANTIC_WEIGHT
                bm25_w = BM25_WEIGHT
            
            result['combined_score'] = (semantic_w * norm_semantic + bm25_w * norm_bm25)
        
        # Sort by combined score
        semantic_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Rerank top results
        top_results = semantic_results[:limit * 2]
        reranker = AdvancedReranker(query_analysis)
        reranked = reranker.rerank(top_results)
        
        return reranked[:limit]


# Global
search_engine: Optional[HybridSearchEngine] = None


def initialize():
    """Initialize components"""
    global qdrant_client, embedding_model, search_engine
    
    logger.info(f"Initializing at {QDRANT_PATH}")
    qdrant_client = QdrantClient(path=QDRANT_PATH)
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    search_engine = HybridSearchEngine(qdrant_client, embedding_model)
    logger.info("✅ Ready")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Available tools"""
    return [
        Tool(
            name="list_collections",
            description="List all documentation collections",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="search_documents",
            description="Advanced hybrid search: Semantic + BM25 + Reranking. "
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
        ),
        Tool(
            name="analyze_query",
            description="Analyze query to understand intent and optimize search strategy",
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
    """Handle tool calls"""
    
    if name == "list_collections":
        return await handle_list_collections()
    elif name == "search_documents":
        return await handle_search(arguments)
    elif name == "analyze_query":
        return await handle_analyze(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def handle_list_collections() -> list[TextContent]:
    """List collections"""
    try:
        collections = qdrant_client.get_collections().collections
        
        if not collections:
            return [TextContent(type="text", text="No collections found")]
        
        result = [f"📚 Available Collections: {len(collections)}\n"]
        
        for coll in collections:
            info = qdrant_client.get_collection(coll.name)
            result.append(f"\n📖 {coll.name}")
            result.append(f"   ├─ Documents: {info.points_count}")
            result.append(f"   └─ Status: {info.status}")
        
        return [TextContent(type="text", text="\n".join(result))]
    
    except Exception as e:
        logger.error(f"List error: {e}")
        return [TextContent(type="text", text=f"Error: {e}")]


async def handle_search(arguments: dict) -> list[TextContent]:
    """Hybrid search"""
    try:
        query = arguments.get("query")
        collection = arguments.get("collection_name")
        limit = arguments.get("limit", 5)
        
        if not query:
            return [TextContent(type="text", text="Error: query required")]
        
        results = await search_engine.search(query, collection, limit)
        
        if not results:
            return [TextContent(type="text", text=f"No results for: '{query}'")]
        
        output = [f"🔍 Hybrid Search: '{query}'\n"]
        output.append(f"Algorithm: Semantic ({SEMANTIC_WEIGHT}) + BM25 ({BM25_WEIGHT}) + Reranking")
        output.append(f"Found {len(results)} results:\n")
        
        for idx, result in enumerate(results, 1):
            output.append(f"\n{'='*70}")
            output.append(f"#{idx} | {result['collection']} | Score: {result['rerank_score']:.4f}")
            output.append(f"    ├─ Semantic: {result['semantic_score']:.3f}")
            output.append(f"    ├─ BM25: {result['bm25_score']:.3f}")
            output.append(f"    ├─ Combined: {result['combined_score']:.3f}")
            output.append(f"    └─ Boost: {result['boost_factor']:.2f}x")
            
            metadata = result.get('metadata', {})
            if metadata.get('source_file'):
                output.append(f"    └─ Source: {metadata['source_file']}")
            
            output.append(f"{'='*70}\n")
            output.append(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])
            output.append("")
        
        return [TextContent(type="text", text="\n".join(output))]
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return [TextContent(type="text", text=f"Error: {e}")]


async def handle_analyze(arguments: dict) -> list[TextContent]:
    """Analyze query"""
    try:
        query = arguments.get("query")
        if not query:
            return [TextContent(type="text", text="Error: query required")]
        
        analysis = QueryUnderstanding.analyze(query)
        
        output = [f"🔬 Query Analysis: '{query}'\n"]
        output.append(f"Type Detection:")
        output.append(f"  ├─ Code Query: {'✅' if analysis['is_code_query'] else '❌'}")
        output.append(f"  ├─ How-To: {'✅' if analysis['is_how_to'] else '❌'}")
        output.append(f"  ├─ Definition: {'✅' if analysis['is_definition'] else '❌'}")
        output.append(f"  └─ Comparison: {'✅' if analysis['is_comparison'] else '❌'}")
        
        if analysis['exact_phrases']:
            output.append(f"\nExact Phrases: {analysis['exact_phrases']}")
        
        if analysis['keywords']:
            output.append(f"Keywords: {analysis['keywords'][:10]}")
        
        if analysis['priority_terms']:
            output.append(f"Priority Terms: {analysis['priority_terms']}")
        
        output.append(f"\nOptimized Strategy:")
        if analysis['is_code_query']:
            output.append("  → Using code-optimized weights (BM25: 60%, Semantic: 40%)")
            output.append("  → Boosting code blocks and function definitions")
        else:
            output.append(f"  → Using standard weights (BM25: {BM25_WEIGHT}, Semantic: {SEMANTIC_WEIGHT})")
        
        return [TextContent(type="text", text="\n".join(output))]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


async def main():
    """Main entry point"""
    initialize()
    
    from mcp.server.stdio import stdio_server
    
    logger.info("🚀 Starting Ultimate Hybrid Search Server")
    logger.info(f"📁 Storage: {QDRANT_PATH}")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
