"""
Complete MCP-Compliant FastMCP Server with OIDC Authentication
RFC 8707 (Resource Indicators) + RFC 9728 (Protected Resource Metadata) Compliant

Implements:
- Bearer token validation with JWT + JWKS
- RFC 8707 Resource Indicator validation (aud claim)
- RFC 9728 Protected Resource Metadata
- Scope validation with insufficient_scope error
- Token introspection endpoint (RFC 7662)
- Proper WWW-Authenticate headers with scope requirements
"""

import logging
import base64
import hashlib
from typing import Optional, Any, Dict, List
from contextlib import asynccontextmanager
from functools import lru_cache

from fastmcp import FastMCP, Context
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, Mount
from starlette.middleware import Middleware
import uvicorn
import httpx
import jwt
from jwt import PyJWKClient
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

OIDC_SERVER_URL = "http://localhost:9400"  # Mock OIDC server
CONFIG_URL = f"{OIDC_SERVER_URL}/.well-known/openid-configuration"
CLIENT_ID = "mcp-demo-client"
CLIENT_SECRET = "demo-secret"
REDIRECT_URI = "http://localhost:8000/auth/callback"
MCP_BASE_URL = "http://localhost:8000"
MCP_RESOURCE_ID = "http://localhost:8000"  # RFC 8707 resource identifier

# Default required scopes for MCP operations
DEFAULT_REQUIRED_SCOPES = ["openid", "profile", "email"]

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
        """Exchange authorization code for tokens (with resource parameter)"""
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
                    "resource": MCP_RESOURCE_ID,  # RFC 8707
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
    
    @classmethod
    async def introspect_token(cls, token: str) -> dict:
        """RFC 7662: Token Introspection endpoint"""
        config = await cls.get_config()
        token_endpoint = config.get("token_endpoint")
        
        # Use introspection endpoint if available, otherwise userinfo
        introspection_endpoint = config.get("token_introspection_endpoint")
        if not introspection_endpoint:
            introspection_endpoint = token_endpoint.replace("/token", "/introspect")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                introspection_endpoint,
                data={"token": token},
                auth=(CLIENT_ID, CLIENT_SECRET),
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return {"active": False}


# ============================================================================
# RFC 8707 Resource Indicator Validation
# ============================================================================

def validate_resource_indicator(token_claims: dict, required_resource: str) -> tuple[bool, Optional[str]]:
    """
    RFC 8707: Validate that token's 'aud' claim includes the required resource
    
    Returns: (is_valid, error_message)
    """
    aud = token_claims.get("aud")
    
    if not aud:
        return False, "Token missing 'aud' (audience) claim - not resource-bound"
    
    # aud can be string or list
    if isinstance(aud, str):
        aud_list = [aud]
    else:
        aud_list = aud
    
    if required_resource in aud_list:
        logger.info(f"✓ Token audience valid: {required_resource}")
        return True, None
    
    logger.warning(f"❌ Token audience mismatch. Expected: {required_resource}, Got: {aud_list}")
    return False, f"Token not intended for this resource: {required_resource}"


# ============================================================================
# Scope Validation
# ============================================================================

def get_required_scopes_for_path(path: str) -> List[str]:
    """Determine required scopes based on request path"""
    if path.startswith("/mcp/tools"):
        return ["mcp:tools.read"]
    elif path.startswith("/mcp/resources"):
        return ["mcp:resources.read"]
    elif path.startswith("/mcp/prompts"):
        return ["mcp:prompts.read"]
    return DEFAULT_REQUIRED_SCOPES


def validate_scopes(token_claims: dict, required_scopes: List[str]) -> tuple[bool, Optional[str]]:
    """
    Validate that token has all required scopes
    
    Returns: (is_valid, missing_scopes_string)
    """
    token_scopes = token_claims.get("scope", "").split() if isinstance(token_claims.get("scope"), str) else []
    token_scopes_set = set(token_scopes)
    required_set = set(required_scopes)
    
    missing = required_set - token_scopes_set
    
    if not missing:
        logger.info(f"✓ Token has required scopes: {required_scopes}")
        return True, None
    
    missing_str = " ".join(missing)
    logger.warning(f"❌ Token missing scopes: {missing_str}")
    return False, missing_str


# ============================================================================
# Bearer Token Middleware (RFC 8707 Compliant)
# ============================================================================

class BearerTokenMiddleware(BaseHTTPMiddleware):
    """
    MCP-Compliant Bearer Token Middleware
    - Validates JWT signature using JWKS
    - Validates RFC 8707 resource indicator (aud claim)
    - Validates scope requirements
    - Returns proper WWW-Authenticate headers on error
    """
    
    # Paths that don't require authentication
    PUBLIC_PATHS = {
        "/.well-known/openid-configuration",
        "/.well-known/oauth-protected-resource",
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
            return self._unauthorized_response(
                "Missing or invalid Authorization header",
                required_scopes=get_required_scopes_for_path(request.url.path)
            )
        
        token = auth_header.replace("Bearer ", "").strip()
        
        try:
            # 1. Validate token signature using JWKS
            jwks_client = await OIDCConfig.get_jwks_client()
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                options={"verify_exp": True}
            )
            
            # 2. RFC 8707: Validate resource indicator (aud claim)
            valid_resource, resource_error = validate_resource_indicator(payload, MCP_RESOURCE_ID)
            if not valid_resource:
                logger.warning(f"❌ Resource validation failed: {resource_error}")
                return JSONResponse(
                    {"error": "invalid_token", "error_description": resource_error},
                    status_code=401,
                    headers=self._get_www_authenticate_header(error="invalid_token")
                )
            
            # 3. Validate scope requirements
            required_scopes = get_required_scopes_for_path(request.url.path)
            valid_scope, missing_scopes = validate_scopes(payload, required_scopes)
            if not valid_scope:
                logger.warning(f"❌ Insufficient scope: {missing_scopes}")
                return JSONResponse(
                    {"error": "insufficient_scope", "error_description": f"Missing scopes: {missing_scopes}"},
                    status_code=403,
                    headers=self._get_www_authenticate_header(
                        error="insufficient_scope",
                        required_scopes=required_scopes
                    )
                )
            
            logger.info(f"✓ Token valid for user: {payload.get('sub')} | Resource: {MCP_RESOURCE_ID}")
            
            # Store token info in request state
            request.state.user = payload
            request.state.token = token
            request.state.scopes = set(payload.get("scope", "").split())
            
            return await call_next(request)
            
        except jwt.ExpiredSignatureError:
            logger.warning("❌ Token expired")
            return JSONResponse(
                {"error": "token_expired", "error_description": "Token has expired"},
                status_code=401,
                headers=self._get_www_authenticate_header(error="invalid_token")
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"❌ Invalid token: {e}")
            return JSONResponse(
                {"error": "invalid_token", "error_description": str(e)},
                status_code=401,
                headers=self._get_www_authenticate_header(error="invalid_token")
            )
        except Exception as e:
            logger.error(f"❌ Token validation error: {e}")
            return JSONResponse(
                {"error": "server_error", "error_description": "Token validation failed"},
                status_code=500
            )
    
    def _get_www_authenticate_header(
        self, 
        error: Optional[str] = None,
        required_scopes: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        RFC 9728 & OAuth 2.0: Build proper WWW-Authenticate header
        """
        parts = [f'Bearer realm="mcp"']
        
        if error:
            parts.append(f'error="{error}"')
        
        if required_scopes:
            scope_str = " ".join(required_scopes)
            parts.append(f'scope="{scope_str}"')
        
        # Always include resource metadata URL
        parts.append(f'resource_metadata="{MCP_BASE_URL}/.well-known/oauth-protected-resource"')
        
        return {"WWW-Authenticate": ", ".join(parts)}
    
    def _unauthorized_response(self, message: str, required_scopes: Optional[List[str]] = None) -> JSONResponse:
        """Return 401 with proper headers"""
        return JSONResponse(
            {"error": message},
            status_code=401,
            headers=self._get_www_authenticate_header(
                error="invalid_request",
                required_scopes=required_scopes
            )
        )


# ============================================================================
# OIDC Endpoints
# ============================================================================

async def oidc_config_endpoint(request):
    """Serve OIDC configuration for client discovery"""
    config = await OIDCConfig.get_config()
    return JSONResponse({
        **config,
        "issuer": OIDC_SERVER_URL,
        "authorization_endpoint": f"{OIDC_SERVER_URL}/oauth2/authorize",
        "token_endpoint": f"{OIDC_SERVER_URL}/oauth2/token",
        "userinfo_endpoint": f"{OIDC_SERVER_URL}/userinfo",
        "jwks_uri": f"{OIDC_SERVER_URL}/jwks",
    })


async def oauth_protected_resource(request):
    """
    RFC 9728: OAuth 2.0 Protected Resource Metadata
    REQUIRED: Tells MCP clients where to authenticate and what scopes are needed
    """
    protected_resource_metadata = {
        # REQUIRED: This server's canonical resource identifier (RFC 8707)
        "resource": MCP_RESOURCE_ID,
        
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
        ],
        
        # REQUIRED: Signing algorithms for JWTs
        "token_signed_response_alg": ["RS256"],
        
        # REQUIRED: OAuth 2.1 PKCE for public clients
        "code_challenge_methods_supported": ["S256"],
        
        # REQUIRED: Scopes this MCP server supports
        "scopes_supported": [
            "openid",
            "profile",
            "email",
            "mcp:tools.read",
            "mcp:tools.write",
            "mcp:resources.read",
            "mcp:resources.write",
            "mcp:prompts.read",
        ],
        
        # REQUIRED: Response formats
        "response_types_supported": ["code"],
        
        # MCP-specific metadata
        "mcp_version": "2025-11-24",
        "mcp_capabilities": {
            "tools": True,
            "resources": True,
            "prompts": True,
        },
        "mcp_endpoints": [
            "/mcp/tools",
            "/mcp/resources",
            "/mcp/prompts",
        ]
    }
    
    return JSONResponse(protected_resource_metadata)


async def auth_callback(request):
    """Handle OAuth callback and redirect to token exchange"""
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    
    if not code:
        return JSONResponse({"error": "Missing authorization code"}, status_code=400)
    
    try:
        # Exchange code for tokens (with resource parameter)
        tokens = await OIDCConfig.get_tokens(code)
        access_token = tokens.get("access_token")
        
        logger.info(f"✓ Tokens obtained: access_token={access_token[:20]}...")
        
        return JSONResponse({
            "message": "Authentication successful",
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": tokens.get("expires_in"),
            "instructions": "Use this token in Authorization: Bearer <token> header for MCP requests"
        })
        
    except Exception as e:
        logger.error(f"❌ Token exchange failed: {e}")
        return JSONResponse(
            {"error": "Token exchange failed", "details": str(e)},
            status_code=500
        )


async def token_introspection(request):
    """
    RFC 7662: Token Introspection Endpoint
    Allows clients to check if a token is still valid and get its claims
    """
    # Parse request body
    body = await request.body()
    if body:
        import urllib.parse
        params = urllib.parse.parse_qs(body.decode())
        token = params.get("token", [None])[0]
    else:
        token = request.query_params.get("token")
    
    if not token:
        return JSONResponse({"error": "missing_token"}, status_code=400)
    
    try:
        # Validate token
        jwks_client = await OIDCConfig.get_jwks_client()
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_exp": False}  # We check manually
        )
        
        # Check expiration
        exp = payload.get("exp")
        active = True
        if exp and exp < datetime.now().timestamp():
            active = False
        
        # RFC 7662 response format
        return JSONResponse({
            "active": active,
            "scope": payload.get("scope", ""),
            "client_id": payload.get("client_id", CLIENT_ID),
            "username": payload.get("sub"),
            "token_type": "Bearer",
            "exp": exp,
            "iat": payload.get("iat"),
            "nbf": payload.get("nbf"),
            "sub": payload.get("sub"),
            "aud": payload.get("aud"),
            "iss": payload.get("iss"),
            "jti": payload.get("jti"),
        })
        
    except jwt.ExpiredSignatureError:
        logger.info(f"Token introspection: token expired")
        return JSONResponse({"active": False})
    except jwt.InvalidTokenError:
        logger.info(f"Token introspection: invalid token")
        return JSONResponse({"active": False})
    except Exception as e:
        logger.error(f"Token introspection error: {e}")
        return JSONResponse({"active": False})


# ============================================================================
# FastMCP Server with Protected Tools
# ============================================================================

mcp = FastMCP(
    name="MCP-Compliant Protected Server",
    description="FastMCP with RFC 8707 & RFC 9728 OIDC authentication"
)


@mcp.tool()
def add(a: int, b: int, ctx: Context) -> int:
    """Add two numbers (Requires: mcp:tools.read scope)"""
    user = ctx.request_context.request.state.user if hasattr(ctx.request_context, 'request') else None
    user_id = user.get("sub") if user else "unknown"
    scopes = ctx.request_context.request.state.scopes if hasattr(ctx.request_context, 'request') else set()
    
    logger.info(f"📊 add({a}, {b}) called by {user_id} with scopes: {scopes}")
    return a + b


@mcp.tool()
def subtract(a: int, b: int, ctx: Context) -> int:
    """Subtract two numbers (Requires: mcp:tools.read scope)"""
    user = ctx.request_context.request.state.user if hasattr(ctx.request_context, 'request') else None
    user_id = user.get("sub") if user else "unknown"
    scopes = ctx.request_context.request.state.scopes if hasattr(ctx.request_context, 'request') else set()
    
    logger.info(f"📊 subtract({a}, {b}) called by {user_id} with scopes: {scopes}")
    return a - b


@mcp.tool()
def get_user_info(ctx: Context) -> dict:
    """Get current authenticated user information with scope details"""
    user = ctx.request_context.request.state.user if hasattr(ctx.request_context, 'request') else None
    scopes = ctx.request_context.request.state.scopes if hasattr(ctx.request_context, 'request') else set()
    
    if not user:
        return {"error": "User information not available"}
    
    return {
        "sub": user.get("sub"),
        "email": user.get("email"),
        "name": user.get("name"),
        "aud": user.get("aud"),  # Resource indicator (RFC 8707)
        "iss": user.get("iss"),
        "exp": user.get("exp"),
        "iat": user.get("iat"),
        "scopes": list(scopes),
        "token_issued_at": datetime.fromtimestamp(user.get("iat", 0)).isoformat(),
        "token_expires_at": datetime.fromtimestamp(user.get("exp", 0)).isoformat(),
    }


# ============================================================================
# Health Check
# ============================================================================

async def health_check(request):
    """Health check endpoint"""
    return JSONResponse({"status": "healthy", "service": "MCP-OIDC Server"})


# ============================================================================
# Starlette Application Setup
# ============================================================================

@asynccontextmanager
async def lifespan(app):
    """App startup/shutdown"""
    logger.info("🚀 Starting MCP-Compliant FastMCP Server with OIDC...")
    # Pre-load OIDC config and JWKS
    await OIDCConfig.get_config()
    await OIDCConfig.get_jwks_client()
    logger.info("✓ OIDC ready | RFC 8707 & RFC 9728 compliant")
    yield
    logger.info("🛑 Shutting down...")


# Routes
routes = [
    Route("/health", health_check),
    Route("/.well-known/openid-configuration", oidc_config_endpoint),
    Route("/.well-known/oauth-protected-resource", oauth_protected_resource),
    Route("/oauth/introspect", token_introspection),
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
    ╔══════════════════════════════════════════════════════════════════════╗
    ║     MCP-Compliant FastMCP with OIDC Authentication (RFC 8707)      ║
    ║                                                                      ║
    ║  Compliance:                                                         ║
    ║  ✓ RFC 8707: Resource Indicators for OAuth 2.0 (aud validation)     ║
    ║  ✓ RFC 9728: Protected Resource Metadata (.well-known endpoint)    ║
    ║  ✓ RFC 7662: Token Introspection endpoint                          ║
    ║  ✓ RFC 6750: Bearer Token usage                                    ║
    ║  ✓ MCP Authorization Specification (June 2025)                     ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    📋 Configuration:
       OIDC Server: {OIDC_SERVER_URL}
       MCP Server: http://localhost:8000
       Resource ID (aud): {MCP_RESOURCE_ID}
       Client ID: {CLIENT_ID}
    
    🔐 Endpoints:
       /.well-known/openid-configuration       → OIDC Discovery
       /.well-known/oauth-protected-resource   → RFC 9728 (MCP client discovery)
       /oauth/introspect                       → RFC 7662 (Token validation)
       /auth/callback                          → OAuth callback
       /mcp/tools/add                          → Protected tool (Bearer token required)
       /mcp/tools/subtract                     → Protected tool (Bearer token required)
       /mcp/tools/get_user_info                → User info with scope details
    
    📊 Scope Requirements:
       /mcp/tools/*      → mcp:tools.read
       /mcp/resources/*  → mcp:resources.read
       /mcp/prompts/*    → mcp:prompts.read
    
    🚀 Quick Test:
       1. Get auth code: 
          http://localhost:9400/oauth2/authorize?response_type=code&
          client_id={CLIENT_ID}&redirect_uri=http://localhost:8000/auth/callback&
          scope=openid%20profile%20email%20mcp:tools.read&resource={MCP_RESOURCE_ID}
       
       2. Exchange for token with resource parameter
       3. Call: curl -H "Authorization: Bearer TOKEN" \\
                http://localhost:8000/mcp/tools/call/add
    
    📚 References:
       RFC 8707: https://datatracker.ietf.org/doc/rfc8707/
       RFC 9728: https://datatracker.ietf.org/doc/rfc9728/
       RFC 7662: https://datatracker.ietf.org/doc/rfc7662/
       MCP Spec: https://modelcontextprotocol.io/specification/draft/basic/authorization
    """)
    
    uvicorn.run(
        "server:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )
