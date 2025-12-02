"""
FastMCP Server with OIDC Authentication
Integrates with oidc-provider-mock for OAuth 2.0 / OIDC flows
"""

import logging
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastmcp import FastMCP, Context
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Route, Mount
from starlette.middleware import Middleware
import uvicorn
import httpx
import jwt
from jwt import PyJWKClient
from datetime import datetime

# Configuration
OIDC_SERVER_URL = "http://localhost:9400"  # Mock OIDC server
CONFIG_URL = f"{OIDC_SERVER_URL}/.well-known/openid-configuration"
CLIENT_ID = "mcp-demo-client"
CLIENT_SECRET = "demo-secret"
REDIRECT_URI = "http://localhost:8000/auth/callback"
MCP_BASE_URL = "http://localhost:8000"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# OIDC Discovery and Token Validation
# ============================================================================

class OIDCConfig:
    """Fetch and cache OIDC configuration"""
    _config = None
    _jwks_client = None
    
    @classmethod
    async def get_config(cls) -> dict:
        """Fetch OIDC configuration from server"""
        if cls._config is None:
            async with httpx.AsyncClient() as client:
                response = await client.get(CONFIG_URL, timeout=10)
                response.raise_for_status()
                cls._config = response.json()
                logger.info(f"✓ OIDC Config loaded from {CONFIG_URL}")
        return cls._config
    
    @classmethod
    async def get_jwks_client(cls):
        """Get JWT key client for token validation"""
        if cls._jwks_client is None:
            config = await cls.get_config()
            jwks_uri = config.get("jwks_uri")
            cls._jwks_client = PyJWKClient(jwks_uri)
            logger.info(f"✓ JWKS Client initialized: {jwks_uri}")
        return cls._jwks_client
    
    @classmethod
    async def get_tokens(cls, code: str) -> dict:
        """Exchange authorization code for tokens"""
        config = await cls.get_config()
        token_endpoint = config.get("token_endpoint")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_endpoint,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": CLIENT_ID,
                    "client_secret": CLIENT_SECRET,
                    "redirect_uri": REDIRECT_URI,
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()


# ============================================================================
# Bearer Token Middleware
# ============================================================================

class BearerTokenMiddleware(BaseHTTPMiddleware):
    """
    Validates Bearer tokens in Authorization header
    Extracts and validates JWT claims using OIDC JWKS
    """
    
    # Paths that don't require authentication
    PUBLIC_PATHS = {
        "/.well-known/openid-configuration",
        "/auth/callback",
        "/health",
    }
    
    async def dispatch(self, request, call_next):
        """Intercept requests and validate bearer token"""
        
        # Skip auth for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)
        
        # Extract Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning(f"❌ Missing/invalid auth header: {request.url.path}")
            return JSONResponse(
                {"error": "Missing or invalid Authorization header"},
                status_code=401
            )
        
        token = auth_header.replace("Bearer ", "").strip()
        
        try:
            # Validate token signature using JWKS
            jwks_client = await OIDCConfig.get_jwks_client()
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                options={"verify_exp": True}
            )
            
            logger.info(f"✓ Token valid for user: {payload.get('sub')}")
            
            # Store token info in request state for access in tools
            request.state.user = payload
            request.state.token = token
            
            return await call_next(request)
            
        except jwt.ExpiredSignatureError:
            logger.warning("❌ Token expired")
            return JSONResponse(
                {"error": "Token expired"},
                status_code=401
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"❌ Invalid token: {e}")
            return JSONResponse(
                {"error": "Invalid token"},
                status_code=401
            )
        except Exception as e:
            logger.error(f"❌ Token validation error: {e}")
            return JSONResponse(
                {"error": "Token validation failed"},
                status_code=500
            )


# ============================================================================
# OAuth Callback Handler
# ============================================================================

async def auth_callback(request):
    """Handle OAuth callback and redirect to token exchange"""
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    
    if not code:
        return JSONResponse({"error": "Missing authorization code"}, status_code=400)
    
    try:
        # Exchange code for tokens
        tokens = await OIDCConfig.get_tokens(code)
        access_token = tokens.get("access_token")
        
        logger.info(f"✓ Tokens obtained: access_token={access_token[:20]}...")
        
        # In a real app, store token in session/database
        # For demo, redirect with token in URL (NOT PRODUCTION)
        return JSONResponse({
            "message": "Authentication successful",
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": tokens.get("expires_in"),
            "instructions": "Use this token in Authorization: Bearer <token> header"
        })
        
    except Exception as e:
        logger.error(f"❌ Token exchange failed: {e}")
        return JSONResponse(
            {"error": "Token exchange failed", "details": str(e)},
            status_code=500
        )


# ============================================================================
# OIDC Discovery Endpoint (serves config for MCP clients)
# ============================================================================

async def oidc_config_endpoint(request):
    """Serve OIDC configuration for client discovery"""
    config = await OIDCConfig.get_config()
    
    # Override endpoints to point to our server (for OAuth Proxy pattern)
    # In production, these would point to actual OIDC provider
    return JSONResponse({
        **config,
        # These tell clients where to authenticate
        "issuer": OIDC_SERVER_URL,
        "authorization_endpoint": f"{OIDC_SERVER_URL}/oauth2/authorize",
        "token_endpoint": f"{OIDC_SERVER_URL}/oauth2/token",
        "userinfo_endpoint": f"{OIDC_SERVER_URL}/userinfo",
        "jwks_uri": f"{OIDC_SERVER_URL}/jwks",
    })


# ============================================================================
# FastMCP Server with Protected Tools
# ============================================================================

# Create FastMCP instance (no built-in auth yet, using middleware instead)
mcp = FastMCP(
    name="Protected MCP Server",
    description="Demo FastMCP with OIDC bearer token authentication"
)


@mcp.tool()
def add(a: int, b: int, ctx: Context) -> int:
    """Add two numbers (Protected - requires valid bearer token)"""
    
    # Access user info from context
    user = ctx.request_context.request.state.user if hasattr(ctx.request_context, 'request') else None
    user_id = user.get("sub") if user else "unknown"
    
    logger.info(f"📊 add() called by user: {user_id}")
    return a + b


@mcp.tool()
def subtract(a: int, b: int, ctx: Context) -> int:
    """Subtract two numbers (Protected - requires valid bearer token)"""
    
    # Access user info from context
    user = ctx.request_context.request.state.user if hasattr(ctx.request_context, 'request') else None
    user_id = user.get("sub") if user else "unknown"
    
    logger.info(f"📊 subtract() called by user: {user_id}")
    return a - b


@mcp.tool()
def get_user_info(ctx: Context) -> dict:
    """Get current user information from token"""
    
    user = ctx.request_context.request.state.user if hasattr(ctx.request_context, 'request') else None
    
    if not user:
        return {"error": "User information not available"}
    
    return {
        "sub": user.get("sub"),
        "email": user.get("email"),
        "name": user.get("name"),
        "iss": user.get("iss"),
        "exp": user.get("exp"),
        "token_issued_at": datetime.fromtimestamp(user.get("iat", 0)).isoformat(),
    }


# ============================================================================
# Starlette Application Setup
# ============================================================================

async def health_check(request):
    """Health check endpoint"""
    return JSONResponse({"status": "healthy"})


@asynccontextmanager
async def lifespan(app):
    """App startup/shutdown"""
    logger.info("🚀 Starting FastMCP with OIDC...")
    # Pre-load OIDC config and JWKS
    await OIDCConfig.get_config()
    await OIDCConfig.get_jwks_client()
    logger.info("✓ OIDC ready")
    yield
    logger.info("🛑 Shutting down...")

# Add this endpoint to existing server.py (before routes definition)

async def oauth_protected_resource(request):
    """MCP REQUIRED: /.well-known/oauth-protected-resource (RFC 9728)"""
    config = await OIDCConfig.get_config()
    
    protected_resource_metadata = {
        # REQUIRED: This server's canonical resource identifier
        "resource": "http://localhost:8000",
        
        # REQUIRED: List of authorization servers that can issue tokens
        "authorization_servers": [
            {
                "location": OIDC_SERVER_URL
            }
        ],
        
        # REQUIRED: Token authentication methods supported
        "token_endpoint_auth_methods_supported": [
            "client_secret_basic",
            "client_secret_post",
            "private_key_jwt"
        ],
        
        # REQUIRED: How tokens are presented (Bearer in Authorization header)
        "token_signed_response_alg": ["RS256"],
        
        # REQUIRED: Scopes this MCP server supports
        "scopes_supported": [
            "openid",
            "profile",
            "email",
            "mcp:tools.read",
            "mcp:tools.write"
        ],
        
        # REQUIRED: Response formats
        "authorization_details_types_supported": ["mcp"],
        
        # OPTIONAL: Additional MCP-specific metadata
        "mcp_version": "2025-11-24",
        "mcp_resources": ["/mcp", "/mcp/tools", "/mcp/resources"]
    }
    
    return JSONResponse(protected_resource_metadata)

# Update 401 responses to include resource_metadata header
class BearerTokenMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # ... existing code ...
        
        if not auth_header.startswith("Bearer "):
            logger.warning(f"❌ Missing/invalid auth header: {request.url.path}")
            return JSONResponse(
                {"error": "Missing or invalid Authorization header"},
                status_code=401,
                headers={
                    "WWW-Authenticate": f'Bearer realm="mcp", resource_metadata="http://localhost:8000/.well-known/oauth-protected-resource"'
                }
            )

# Update routes to include the NEW endpoint
routes = [
    Route("/health", health_check),
    Route("/.well-known/openid-configuration", oidc_config_endpoint),
    Route("/.well-known/oauth-protected-resource", oauth_protected_resource),  # ← NEW!
    Route("/auth/callback", auth_callback),
    Mount("/mcp", mcp.http_app()),
]

# Middleware stack
middleware = [
    Middleware(BearerTokenMiddleware),
]

# Create Starlette app
app = Starlette(
    routes=routes,
    middleware=middleware,
    lifespan=lifespan,
)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    logger.info(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║        FastMCP with OIDC Authentication Demo                ║
    ╚══════════════════════════════════════════════════════════════╝
    
    📋 Setup:
       1. Mock OIDC Server: {OIDC_SERVER_URL}
       2. MCP Server: http://localhost:8000
       3. Client ID: {CLIENT_ID}
       4. Redirect URI: {REDIRECT_URI}
    
    🔐 Authentication Flow:
       1. Get auth code: http://localhost:9400/oauth2/authorize?
          response_type=code&client_id={CLIENT_ID}&
          redirect_uri={REDIRECT_URI}&scope=openid%20profile%20email
       
       2. Exchange code: curl -X POST {OIDC_SERVER_URL}/oauth2/token
          with code from step 1
       
       3. Use token: curl -H "Authorization: Bearer TOKEN" \\
          http://localhost:8000/mcp/tools/add?a=5&b=3
    
    📚 Discovery:
       - OIDC Config: http://localhost:8000/.well-known/openid-configuration
       - Callback: http://localhost:8000/auth/callback
       - Health: http://localhost:8000/health
    
    🔧 Protected Tools:
       - add(a: int, b: int) → int
       - subtract(a: int, b: int) → int
       - get_user_info() → dict
    
    """)
    
    uvicorn.run(
        "server:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )