# FastMCP with OIDC Authentication - Complete Guide

## Quick Overview

This guide integrates FastMCP with OIDC (OpenID Connect) authentication using a mock OIDC server.

### Architecture Flow

```
MCP Client
    ↓ (1. Request with Bearer token OR redirects to browser auth)
FastMCP Server (Starlette HTTP)
    ↓ (2. Token Verifier middleware validates token)
    ├─ .well-known/openid-configuration (OIDC Discovery)
    ├─ /auth/callback (OAuth callback endpoint)
    └─ /mcp (Protected MCP endpoints - tools, resources, etc)
    ↓ (3. If valid, access token claims injected to context)
Protected Tools/Resources
    ↓ (4. Use user info from context)
```

### Key Concepts

**Redirect URI**: For MCP, the redirect URI is typically:
- **Local Development:** `http://localhost:8000/auth/callback`
- **Production:** `https://your-domain.com/auth/callback`
- **MCP Clients (Claude, etc):** Usually `http://localhost:*` pattern

**Middleware Flow:**
1. Request comes to MCP server with `Authorization: Bearer TOKEN`
2. Middleware intercepts and validates token with OIDC server
3. Token claims (user info) extracted and added to MCP context
4. Tools access user info via context
5. Response sent to client

---

## Setup Steps

### 1. Start OIDC Mock Server (from previous guide)

```bash
# Terminal 1: Start mock OIDC server
pipx run oidc-provider-mock --port=9400

# Set up a test user
curl -X PUT http://localhost:9400/users/testuser \
-H "Content-Type: application/json" \
-d '{"email": "testuser@example.com", "name": "Test User"}'
```

### 2. Install FastMCP

```bash
pip install fastmcp starlette uvicorn pydantic python-jose cryptography
```

### 3. Create FastMCP Server with OIDC Auth (see server.py below)

### 4. Run Server

```bash
python server.py
```

Server runs at: `http://localhost:8000/mcp`

---

## Configuration for OIDC Mock Server

When using oidc-provider-mock (`http://localhost:9400`):

- **Config URL:** `http://localhost:9400/.well-known/openid-configuration`
- **Redirect URI:** `http://localhost:8000/auth/callback`
- **Client ID:** Any value (e.g., `mcp-demo-client`)
- **Client Secret:** Any value (e.g., `demo-secret`)
- **Scope:** `openid profile email`

---

## .well-known Endpoints Explained

These are OAuth/OIDC metadata endpoints that clients use to discover auth configuration:

```
/.well-known/openid-configuration
  ├─ authorization_endpoint: Where to send user for auth
  ├─ token_endpoint: Where to exchange code for token
  ├─ userinfo_endpoint: Where to get user info
  ├─ jwks_uri: Where to get signing keys
  └─ issuer: Who issued the token

/.well-known/oauth-authorization-server
  └─ Similar to above, for OAuth 2.0 generic servers

/.well-known/oauth-protected-resource
  └─ (Advanced) MCP-specific resource metadata
```

---

## Middleware for Protected Endpoints

The bearer token middleware:

1. **Intercepts** all incoming HTTP requests
2. **Extracts** `Authorization: Bearer TOKEN` header
3. **Validates** token signature using JWKS
4. **Decodes** claims (user info, expiration, etc)
5. **Injects** user info into request context
6. **Blocks** if token invalid (401 Unauthorized)
7. **Allows** request to proceed if valid

For MCP specifically, the middleware ensures:
- Every tool call includes user identity
- Token claims accessible in tool context
- 401 errors on invalid/expired tokens

---

## Token Claims in Tools

Once authenticated, tools can access user information:

```python
@mcp.tool()
def protected_tool(data: str, ctx: Context) -> str:
    # Get user info from token
    user_info = ctx.get_user_info()  # Returns token claims
    # or extract from context
    return f"Processing for user: {user_info.get('sub')}"
```

Token claims typically include:
- `sub`: Subject (user ID)
- `email`: User email
- `name`: User name
- `iss`: Issuer (who created token)
- `aud`: Audience (who token is for)
- `exp`: Expiration timestamp
- `iat`: Issued at timestamp