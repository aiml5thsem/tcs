# MCP Qdrant Documentation Server

A production-ready, read-only MCP (Model Context Protocol) server for serving internal documentation using Qdrant vector database. Each markdown file becomes its own searchable collection.

## 🎯 Features

- ✅ **Read-Only by Design** - Only exposes search and list operations, no write access
- ✅ **Collection per Document** - Each markdown file = one collection for easy organization
- ✅ **Local & Private** - Everything runs locally, no external API calls
- ✅ **Semantic Search** - Uses FastEmbed for high-quality embeddings (no OpenAI needed)
- ✅ **Zero Configuration** - Works out of the box with sensible defaults
- ✅ **Production Ready** - Proper error handling, logging, and metadata tracking

## 📋 Prerequisites

- Python 3.10 or higher
- pip or uv package manager

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install mcp qdrant-client fastembed tqdm
```

Or using uv:

```bash
uv pip install mcp qdrant-client fastembed tqdm
```

### 2. Prepare Your Documentation

Organize your markdown files in a directory:

```
docs/
├── api-reference.md
├── authentication.md
├── deployment-guide.md
└── troubleshooting.md
```

### 3. Create the Vector Database

```bash
python create_local_db.py /path/to/docs/ --output ./qdrant_storage
```

This will:
- Process each `.md` file
- Create a collection named after each file (e.g., `api_reference`, `authentication`)
- Chunk the content intelligently
- Generate embeddings using FastEmbed
- Store everything in local Qdrant storage

**Example output:**
```
Found 4 markdown files to process
Output directory: ./qdrant_storage
Embedding model: BAAI/bge-small-en-v1.5

Processing: api-reference.md → Collection: api_reference
  Created 15 chunks
  Generating embeddings...
  Creating collection...
  Uploading 15 vectors...
  ✅ Collection 'api_reference' created with 15 vectors

✅ Processing complete!
Collections created: 4
  - api_reference
  - authentication
  - deployment_guide
  - troubleshooting
```

### 4. Configure MCP Client

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "internal-docs": {
      "command": "python",
      "args": ["/absolute/path/to/server.py"],
      "env": {
        "QDRANT_PATH": "/absolute/path/to/qdrant_storage",
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "EMBEDDING_DIM": "384"
      }
    }
  }
}
```

### 5. Restart Claude Desktop

The server will now be available with two tools:
- **list_collections** - See all available documentation
- **search_documents** - Search across all or specific collections

## 🔧 Advanced Usage

### Custom Chunk Size

```bash
python create_local_db.py /path/to/docs/ \
  --chunk-size 500 \
  --overlap 100 \
  --output ./qdrant_storage
```

### Force Recreate Collections

```bash
python create_local_db.py /path/to/docs/ --force
```

### Index Specific Files

```bash
python create_local_db.py \
  docs/api.md \
  docs/auth.md \
  docs/deploy.md \
  --output ./qdrant_storage
```

### Use Different Embedding Model

```bash
python create_local_db.py /path/to/docs/ \
  --embedding-model "sentence-transformers/all-MiniLM-L6-v2" \
  --output ./qdrant_storage
```

**Note:** If you change the embedding model, update `EMBEDDING_MODEL` in your MCP config and `EMBEDDING_DIM` accordingly.

Popular embedding models:
- `BAAI/bge-small-en-v1.5` (384 dim) - Default, good balance
- `BAAI/bge-base-en-v1.5` (768 dim) - Better quality, slower
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim) - Fast, lightweight

## 📖 Usage Examples

### List Available Collections

In Claude:
```
Can you list the available documentation collections?
```

Response:
```
📚 Available Documentation Collections:

📖 Collection: api_reference
   Documents: 15
   Vectors: 15

📖 Collection: authentication
   Documents: 8
   Vectors: 8

📖 Collection: deployment_guide
   Documents: 12
   Vectors: 12
```

### Search All Documentation

```
How do I implement OAuth2 authentication?
```

The server will search across all collections and return relevant passages with metadata.

### Search Specific Collection

```
Search in the authentication collection for JWT token expiration
```

You can programmatically specify collection names using the `collection_name` parameter.

## 🏗️ Architecture

```
┌─────────────────────────┐
│   Markdown Files        │
│   (Your Docs)           │
└──────────┬──────────────┘
           │
           │ create_local_db.py
           ▼
┌─────────────────────────┐
│   Qdrant Storage        │
│   (Vector DB)           │
│                         │
│   Collections:          │
│   - library_1           │
│   - library_2           │
│   - api_docs            │
└──────────┬──────────────┘
           │
           │ server.py
           ▼
┌─────────────────────────┐
│   MCP Server            │
│   (Read-Only)           │
│                         │
│   Tools:                │
│   - list_collections    │
│   - search_documents    │
└──────────┬──────────────┘
           │
           │ MCP Protocol
           ▼
┌─────────────────────────┐
│   Claude Desktop        │
│   (MCP Client)          │
└─────────────────────────┘
```

## 🔒 Security & Privacy

- **Fully Local** - No data leaves your machine
- **No API Keys** - Uses local embedding models (FastEmbed)
- **Read-Only** - Server only exposes search operations
- **Private** - Perfect for internal/proprietary documentation

## 📝 Markdown File Format

### Basic Markdown

```markdown
# My Documentation

This is some content that will be indexed and searchable.

## Section 1

More content here...
```

### With Frontmatter (Optional)

```markdown
---
title: API Reference
version: 1.0
author: DevTeam
tags: api, rest, authentication
---

# API Reference

Your content here...
```

Frontmatter metadata will be stored and searchable.

## 🎨 Collection Naming

File names are automatically sanitized to create valid collection names:

| File Name | Collection Name |
|-----------|----------------|
| `API Reference.md` | `api_reference` |
| `Getting Started.md` | `getting_started` |
| `v2.0-migration.md` | `v2_0_migration` |
| `OAuth 2.0 Guide.md` | `oauth_2_0_guide` |

## 🔍 Search Quality Tips

1. **Chunk Size** - Smaller chunks (500-1000) work better for specific queries, larger chunks (1500-2000) for context
2. **Overlap** - Use 100-200 character overlap to avoid breaking concepts
3. **Query Phrasing** - Natural language questions work best: "How do I..." rather than keywords
4. **Collection Organization** - Group related docs into single markdown files for better context

## 🐛 Troubleshooting

### Collections Not Found

```bash
# Verify collections exist
python -c "from qdrant_client import QdrantClient; c = QdrantClient(path='./qdrant_storage'); print(c.get_collections())"
```

### Embedding Model Download Issues

FastEmbed will download models on first run. If you have network issues:

```bash
# Pre-download model
python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"
```

### Server Not Responding

Check logs for errors:

```bash
python server.py 2>&1 | tee server.log
```

### Permission Errors

Ensure the server has read access to the Qdrant storage directory:

```bash
chmod -R 755 ./qdrant_storage
```

## 🔄 Updating Documentation

To update your documentation:

1. Modify your markdown files
2. Re-run the indexing script with `--force`:

```bash
python create_local_db.py /path/to/docs/ --force --output ./qdrant_storage
```

3. Restart the MCP server (restart Claude Desktop)

## 📊 Performance

- **Indexing Speed** - ~100-500 documents/minute (depends on size)
- **Search Speed** - <100ms for most queries
- **Memory Usage** - ~200MB base + ~1MB per 1000 vectors
- **Storage** - ~1KB per document chunk

## 🛠️ Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_PATH` | Path to Qdrant storage | `./qdrant_storage` |
| `EMBEDDING_MODEL` | FastEmbed model name | `BAAI/bge-small-en-v1.5` |
| `EMBEDDING_DIM` | Embedding dimension | `384` |

## 📦 File Structure

```
mcp-qdrant-docs/
├── server.py              # MCP server (read-only)
├── create_local_db.py     # Database creation script
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── qdrant_storage/        # Generated database (gitignored)
    ├── collection/
    ├── meta.json
    └── storage.sqlite
```

## 🤝 Contributing

This is a production-ready implementation designed for internal use. Feel free to:

- Adjust chunk sizes for your use case
- Add custom metadata extractors
- Implement additional search filters
- Add support for other file formats (PDF, DOCX, etc.)

## 📄 License

MIT License - Use freely for internal documentation needs.

## 🙏 Acknowledgments

- Based on concepts from [qdrant/mcp-for-docs](https://github.com/qdrant/mcp-for-docs)
- Uses [Qdrant](https://qdrant.tech/) for vector storage
- Uses [FastEmbed](https://github.com/qdrant/fastembed) for embeddings
- Built for the [Model Context Protocol](https://modelcontextprotocol.io/)

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the logs in server output
3. Verify your configuration matches the examples

---

**Made with ❤️ for internal documentation teams**
