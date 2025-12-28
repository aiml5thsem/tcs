#!/usr/bin/env python3
"""
5_conversation_server.py - Intelligent Conversation History Memory Server
Solves: Context loss after compaction using semantic memory retrieval
Supports: stdio, sse, streamable-http transports with --history argument
"""

import asyncio
import argparse
import logging
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict

from mcp.server import Server
from mcp.types import Tool, TextContent
from fastembed import TextEmbedding
import sqlite3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("5_conv_server")

# Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Initialize
app = Server("conversation_memory")
embedding_model: Optional[TextEmbedding] = None


class ConversationMemory:
    """In-memory conversation storage with semantic search"""
    
    def __init__(self, db_path=":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        self.embeddings_cache = {}
    
    def _init_db(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()
        
        # Main messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                turn_number INTEGER,
                session_id TEXT,
                metadata TEXT
            )
        """)
        
        # Embeddings table for semantic search
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS message_embeddings (
                message_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY (message_id) REFERENCES messages(id)
            )
        """)
        
        # Topics/summaries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                summary TEXT NOT NULL,
                message_ids TEXT,
                created_at TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_role ON messages(role)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_turn ON messages(turn_number)")
        
        self.conn.commit()
    
    def add_message(self, role: str, content: str, turn: int, session_id: str = "default",
                   metadata: Optional[Dict] = None):
        """Add message to memory"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO messages (timestamp, role, content, turn_number, session_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            role,
            content,
            turn,
            session_id,
            json.dumps(metadata or {})
        ))
        
        message_id = cursor.lastrowid
        self.conn.commit()
        
        return message_id
    
    def add_embedding(self, message_id: int, embedding: List[float]):
        """Store message embedding"""
        cursor = self.conn.cursor()
        
        # Store as blob
        embedding_bytes = json.dumps(embedding).encode('utf-8')
        
        cursor.execute("""
            INSERT OR REPLACE INTO message_embeddings (message_id, embedding)
            VALUES (?, ?)
        """, (message_id, embedding_bytes))
        
        self.conn.commit()
        self.embeddings_cache[message_id] = embedding
    
    def get_all_messages(self, session_id: str = "default", limit: Optional[int] = None) -> List[Dict]:
        """Get all messages"""
        cursor = self.conn.cursor()
        
        query = """
            SELECT id, timestamp, role, content, turn_number, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY turn_number ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (session_id,))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                'id': row[0],
                'timestamp': row[1],
                'role': row[2],
                'content': row[3],
                'turn_number': row[4],
                'metadata': json.loads(row[5]) if row[5] else {}
            })
        
        return messages
    
    def semantic_search(self, query_embedding: List[float], limit: int = 5,
                       session_id: str = "default") -> List[Dict]:
        """Search messages by semantic similarity"""
        cursor = self.conn.cursor()
        
        # Get all messages with embeddings
        cursor.execute("""
            SELECT m.id, m.role, m.content, m.turn_number, e.embedding
            FROM messages m
            JOIN message_embeddings e ON m.id = e.message_id
            WHERE m.session_id = ?
            ORDER BY m.turn_number DESC
            LIMIT 100
        """, (session_id,))
        
        results = []
        for row in cursor.fetchall():
            msg_id, role, content, turn, emb_bytes = row
            
            # Decode embedding
            if msg_id in self.embeddings_cache:
                embedding = self.embeddings_cache[msg_id]
            else:
                embedding = json.loads(emb_bytes.decode('utf-8'))
                self.embeddings_cache[msg_id] = embedding
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, embedding)
            
            results.append({
                'id': msg_id,
                'role': role,
                'content': content,
                'turn_number': turn,
                'similarity': similarity
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def get_recent_messages(self, n: int = 10, session_id: str = "default") -> List[Dict]:
        """Get N most recent messages"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT id, timestamp, role, content, turn_number
            FROM messages
            WHERE session_id = ?
            ORDER BY turn_number DESC
            LIMIT ?
        """, (session_id, n))
        
        messages = []
        for row in reversed(cursor.fetchall()):  # Reverse to get chronological order
            messages.append({
                'id': row[0],
                'timestamp': row[1],
                'role': row[2],
                'content': row[3],
                'turn_number': row[4]
            })
        
        return messages
    
    def add_topic_summary(self, topic: str, summary: str, message_ids: List[int]):
        """Add topic summary for grouped messages"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversation_topics (topic, summary, message_ids, created_at)
            VALUES (?, ?, ?, ?)
        """, (topic, summary, json.dumps(message_ids), datetime.utcnow().isoformat()))
        
        self.conn.commit()
    
    def get_topics(self) -> List[Dict]:
        """Get all conversation topics"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT id, topic, summary, message_ids, created_at
            FROM conversation_topics
            ORDER BY created_at DESC
        """)
        
        topics = []
        for row in cursor.fetchall():
            topics.append({
                'id': row[0],
                'topic': row[1],
                'summary': row[2],
                'message_ids': json.loads(row[3]),
                'created_at': row[4]
            })
        
        return topics
    
    def get_stats(self, session_id: str = "default") -> Dict:
        """Get conversation statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN role='user' THEN 1 ELSE 0 END) as user_msgs,
                SUM(CASE WHEN role='assistant' THEN 1 ELSE 0 END) as assistant_msgs,
                MAX(turn_number) as max_turn
            FROM messages
            WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        return {
            'total_messages': row[0],
            'user_messages': row[1],
            'assistant_messages': row[2],
            'max_turn': row[3] or 0
        }


class ConversationParser:
    """Parse exported conversation files (MD/TXT)"""
    
    @staticmethod
    def parse_markdown(content: str) -> List[Dict]:
        """Parse markdown conversation export"""
        messages = []
        turn = 0
        
        # Pattern: ## User / ## Assistant / ## Claude
        sections = re.split(r'\n##\s+(User|Assistant|Claude|Human|AI)\s*\n', content, flags=re.IGNORECASE)
        
        if len(sections) < 2:
            # Try alternative format: **User:** or **Assistant:**
            sections = re.split(r'\n\*\*(User|Assistant|Claude|Human|AI):\*\*\s*\n', content, flags=re.IGNORECASE)
        
        current_role = None
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            
            # Check if this is a role marker
            if section.lower() in ['user', 'human']:
                current_role = 'user'
            elif section.lower() in ['assistant', 'claude', 'ai']:
                current_role = 'assistant'
            elif current_role:
                # This is content
                messages.append({
                    'role': current_role,
                    'content': section,
                    'turn': turn
                })
                turn += 1
                current_role = None
        
        return messages
    
    @staticmethod
    def parse_text(content: str) -> List[Dict]:
        """Parse plain text conversation"""
        messages = []
        turn = 0
        
        # Try to detect pattern
        lines = content.split('\n')
        current_role = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            
            # Check for role markers
            if re.match(r'^(User|Human|You):', line, re.IGNORECASE):
                if current_content and current_role:
                    messages.append({
                        'role': current_role,
                        'content': '\n'.join(current_content),
                        'turn': turn
                    })
                    turn += 1
                current_role = 'user'
                current_content = [re.sub(r'^(User|Human|You):\s*', '', line, flags=re.IGNORECASE)]
            
            elif re.match(r'^(Assistant|Claude|AI):', line, re.IGNORECASE):
                if current_content and current_role:
                    messages.append({
                        'role': current_role,
                        'content': '\n'.join(current_content),
                        'turn': turn
                    })
                    turn += 1
                current_role = 'assistant'
                current_content = [re.sub(r'^(Assistant|Claude|AI):\s*', '', line, flags=re.IGNORECASE)]
            
            elif current_role:
                current_content.append(line)
        
        # Add final message
        if current_content and current_role:
            messages.append({
                'role': current_role,
                'content': '\n'.join(current_content),
                'turn': turn
            })
        
        return messages


# Global memory
conversation_memory: Optional[ConversationMemory] = None


def initialize(history_path: Optional[str] = None):
    """Initialize server with optional history file"""
    global embedding_model, conversation_memory
    
    logger.info("Initializing Conversation Memory Server")
    
    # Initialize embedding model
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    
    # Initialize memory
    conversation_memory = ConversationMemory()
    
    # Load history if provided
    if history_path:
        load_history(history_path)
    
    logger.info("✅ Ready")


def load_history(history_path: str):
    """Load conversation history from file"""
    logger.info(f"Loading history from: {history_path}")
    
    path = Path(history_path)
    if not path.exists():
        logger.error(f"History file not found: {history_path}")
        return
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse based on file type
        if path.suffix.lower() in ['.md', '.markdown']:
            messages = ConversationParser.parse_markdown(content)
        else:
            messages = ConversationParser.parse_text(content)
        
        logger.info(f"Parsed {len(messages)} messages")
        
        # Store messages with embeddings
        for msg in messages:
            # Add to database
            msg_id = conversation_memory.add_message(
                role=msg['role'],
                content=msg['content'],
                turn=msg['turn']
            )
            
            # Generate and store embedding
            embedding = list(embedding_model.embed([msg['content']]))[0].tolist()
            conversation_memory.add_embedding(msg_id, embedding)
        
        logger.info(f"✅ Loaded {len(messages)} messages into memory")
    
    except Exception as e:
        logger.error(f"Error loading history: {e}")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Available tools"""
    return [
        Tool(
            name="get_conversation_context",
            description="Retrieve relevant conversation history based on current context. "
                       "Uses semantic search to find related past messages.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Current topic or question to find related history"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of relevant messages to retrieve"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_recent_history",
            description="Get the most recent N messages from conversation history",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Number of recent messages to retrieve"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="search_conversation",
            description="Search conversation history for specific keywords or topics",
            inputSchema={
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Keywords to search for"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5
                    }
                },
                "required": ["keywords"]
            }
        ),
        Tool(
            name="get_conversation_summary",
            description="Get statistics and overview of loaded conversation",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    
    if name == "get_conversation_context":
        return await handle_get_context(arguments)
    elif name == "get_recent_history":
        return await handle_get_recent(arguments)
    elif name == "search_conversation":
        return await handle_search(arguments)
    elif name == "get_conversation_summary":
        return await handle_summary()
    else:
        raise ValueError(f"Unknown tool: {name}")


async def handle_get_context(arguments: dict) -> list[TextContent]:
    """Get relevant context using semantic search"""
    try:
        query = arguments.get("query")
        limit = arguments.get("limit", 5)
        
        if not query:
            return [TextContent(type="text", text="Error: query required")]
        
        # Generate query embedding
        query_embedding = list(embedding_model.embed([query]))[0].tolist()
        
        # Semantic search
        results = conversation_memory.semantic_search(query_embedding, limit)
        
        if not results:
            return [TextContent(type="text", text="No relevant history found")]
        
        output = [f"🔍 Relevant Conversation History for: '{query}'\n"]
        output.append(f"Found {len(results)} relevant messages:\n")
        
        for idx, msg in enumerate(results, 1):
            output.append(f"\n{'─'*60}")
            output.append(f"#{idx} | Turn {msg['turn_number']} | {msg['role'].upper()} | "
                         f"Relevance: {msg['similarity']:.3f}")
            output.append(f"{'─'*60}")
            output.append(msg['content'][:400] + "..." if len(msg['content']) > 400 else msg['content'])
        
        output.append(f"\n\n💡 Use this context to continue the conversation where we left off!")
        
        return [TextContent(type="text", text="\n".join(output))]
    
    except Exception as e:
        logger.error(f"Context retrieval error: {e}")
        return [TextContent(type="text", text=f"Error: {e}")]


async def handle_get_recent(arguments: dict) -> list[TextContent]:
    """Get recent messages"""
    try:
        limit = arguments.get("limit", 10)
        
        messages = conversation_memory.get_recent_messages(limit)
        
        if not messages:
            return [TextContent(type="text", text="No messages in history")]
        
        output = [f"📜 Recent Conversation History (last {len(messages)} messages):\n"]
        
        for msg in messages:
            output.append(f"\n[Turn {msg['turn_number']}] {msg['role'].upper()}:")
            output.append(msg['content'][:300] + "..." if len(msg['content']) > 300 else msg['content'])
            output.append("")
        
        return [TextContent(type="text", text="\n".join(output))]
    
    except Exception as e:
        logger.error(f"Recent history error: {e}")
        return [TextContent(type="text", text=f"Error: {e}")]


async def handle_search(arguments: dict) -> list[TextContent]:
    """Search conversation"""
    try:
        keywords = arguments.get("keywords")
        limit = arguments.get("limit", 5)
        
        if not keywords:
            return [TextContent(type="text", text="Error: keywords required")]
        
        # Get all messages and filter
        all_messages = conversation_memory.get_all_messages()
        
        # Simple keyword search
        keywords_lower = keywords.lower()
        matches = [msg for msg in all_messages if keywords_lower in msg['content'].lower()]
        
        if not matches:
            return [TextContent(type="text", text=f"No messages found containing: '{keywords}'")]
        
        output = [f"🔎 Search Results for: '{keywords}'\n"]
        output.append(f"Found {len(matches)} matching messages:\n")
        
        for msg in matches[:limit]:
            output.append(f"\n[Turn {msg['turn_number']}] {msg['role'].upper()}:")
            # Highlight keywords
            content = msg['content']
            output.append(content[:400] + "..." if len(content) > 400 else content)
            output.append("")
        
        return [TextContent(type="text", text="\n".join(output))]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


async def handle_summary() -> list[TextContent]:
    """Get conversation summary"""
    try:
        stats = conversation_memory.get_stats()
        
        output = [f"📊 Conversation Memory Summary\n"]
        output.append(f"Total Messages: {stats['total_messages']}")
        output.append(f"User Messages: {stats['user_messages']}")
        output.append(f"Assistant Messages: {stats['assistant_messages']}")
        output.append(f"Conversation Turns: {stats['max_turn'] + 1}")
        
        # Get topics if any
        topics = conversation_memory.get_topics()
        if topics:
            output.append(f"\nIdentified Topics: {len(topics)}")
            for topic in topics[:5]:
                output.append(f"  • {topic['topic']}")
        
        output.append(f"\n💡 Tip: Use 'get_conversation_context' to retrieve relevant history!")
        
        return [TextContent(type="text", text="\n".join(output))]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Conversation Memory MCP Server")
    parser.add_argument("--history", help="Path to conversation history file (MD/TXT)")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"],
                       help="Transport type")
    
    args = parser.parse_args()
    
    # Initialize
    initialize(args.history)
    
    logger.info("🚀 Starting Conversation Memory Server")
    if args.history:
        logger.info(f"📂 Loaded history from: {args.history}")
    logger.info(f"🔌 Transport: {args.transport}")
    
    # Start server
    if args.transport == "stdio":
        from mcp.server.stdio import stdio_server
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    
    elif args.transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route
        
        sse = SseServerTransport("/messages")
        
        async def handle_sse(request):
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await app.run(streams[0], streams[1], app.create_initialization_options())
        
        starlette_app = Starlette(routes=[Route("/messages", endpoint=handle_sse)])
        
        import uvicorn
        config = uvicorn.Config(starlette_app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
