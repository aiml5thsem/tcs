"""
FastMCP Starter Template With Basic Auth

A minimal starter template for FastMCP servers:
- Tool
- Resource
- Resource Template
- Prompt
- Custom Route

Referencing FastMCP documentation: https://gofastmcp.com/getting-started
"""
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse
from auth import BasicAuthMiddleware
from starlette.middleware import Middleware
import uvicorn

# ============================================================================
# CREATE MCP SERVER
# ============================================================================

mcp = FastMCP("Starter Template Server")

# ============================================================================
# TOOL - Execute actions
# ============================================================================

@mcp.tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

@mcp.tool
def greet(name: str) -> str:
    """Greet someone by name"""
    return f"Hello, {name}!"

# ============================================================================
# RESOURCE - Read-only data
# ============================================================================

@mcp.resource("config://app-version")
def get_version():
    """Get the application version"""
    return "1.0.0"

@mcp.resource("config://settings")
def get_settings():
    """Get application settings"""
    return {
        "app_name": "Starter Template",
        "environment": "development",
        "debug": True
    }

# ============================================================================
# RESOURCE TEMPLATE - Dynamic resources with parameters
# ============================================================================

@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: int):
    """Get user profile by ID"""
    return {
        "user_id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com",
        "status": "active"
    }

# ============================================================================
# PROMPT - Reusable message templates
# ============================================================================

@mcp.prompt
def summarize_text(text: str) -> str:
    """Generate a prompt for text summarization"""
    return f"Please provide a concise summary of the following text:\n\n{text}"

@mcp.prompt
def code_review(code: str, language: str = "python") -> str:
    """Generate a prompt for code review"""
    return f"""Please review the following {language} code and provide feedback on:
1. Code quality
2. Best practices
3. Potential improvements

Code:
```{language}
{code}
```"""

# ============================================================================
# CUSTOM ROUTE - /health endpoint
# ============================================================================

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request):
    """Health check endpoint (no auth required)"""
    return JSONResponse({"status": "healthy"})

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    # Create middleware instance (pass secrets_dict if using dict-based secrets)
    auth_middleware = Middleware(BasicAuthMiddleware)  # Or {'MCP_BASIC_AUTH_USERNAME': 'your_user', ...}

    # Create ASGI app with auth middleware
    app = mcp.http_app(middleware=[auth_middleware])

    # Run with Uvicorn (matches original host/port)
    uvicorn.run(app, host="0.0.0.0", port=8000)
