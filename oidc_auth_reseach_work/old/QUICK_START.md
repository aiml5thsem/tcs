# FastMCP + OIDC Quick Start

## 5-Minute Setup

### Prerequisites
```bash
# Install packages
pip install fastmcp starlette uvicorn python-jose cryptography pyjwt
pipx install oidc-provider-mock
```

### Run (3 Terminals)

**Terminal 1: OIDC Server**
```bash
pipx run oidc-provider-mock --port=9400
# Setup test user:
curl -X PUT http://localhost:9400/users/testuser \
-H "Content-Type: application/json" \
-d '{"email": "testuser@example.com", "name": "Test"}'
```

**Terminal 2: MCP Server**
```bash
python server.py
```

**Terminal 3: Test It**
```bash
# 1. Get auth code (copy code from redirect)
curl -L "http://localhost:9400/oauth2/authorize?response_type=code&client_id=mcp-demo-client&redirect_uri=http://localhost:8000/auth/callback&scope=openid%20profile%20email"

# 2. Exchange for token (use code from step 1)
TOKEN_RESPONSE=$(curl -s -X POST http://localhost:9400/oauth2/token \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "grant_type=authorization_code&code=YOUR_CODE&client_id=mcp-demo-client&client_secret=demo-secret&redirect_uri=http://localhost:8000/auth/callback")

ACCESS_TOKEN=$(echo $TOKEN_RESPONSE | jq -r '.access_token')

# 3. Call protected tool
curl -X GET "http://localhost:8000/mcp/tools/call/add" \
-H "Authorization: Bearer $ACCESS_TOKEN" \
-H "Content-Type: application/json" \
-d '{"a": 5, "b": 3}'
```

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Client                               │
│                    (e.g., Claude, Tools)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                   Authorization: Bearer TOKEN
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FastMCP Server (Port 8000)                    │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  BearerTokenMiddleware                                   │  │
│  │  - Extracts Bearer token from Authorization header      │  │
│  │  - Validates JWT signature using JWKS                  │  │
│  │  - Injects user claims into request.state.user         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                   Token valid? ─────→ Inject user info         │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Protected MCP Tools                                     │  │
│  │  - add(a, b, ctx: Context)                             │  │
│  │  - subtract(a, b, ctx: Context)                        │  │
│  │  - get_user_info(ctx: Context)                         │  │
│  │                                                          │  │
│  │  // Access user info: ctx.request_context.request...   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  /.well-known/openid-configuration  ← Proxied OIDC metadata   │
│  /auth/callback                      ← OAuth callback endpoint  │
│  /health                             ← Health check             │
└────────────┬────────────────────────────────────┬───────────────┘
             │                                    │
             │                    Validate token
             │                    with JWKS
             │                                    │
             ▼                                    ▼
┌──────────────────────┐         ┌───────────────────────────┐
│  oidc-provider-mock  │         │   OIDC Provider           │
│     (Port 9400)      │         │  (Keycloak, Auth0, etc)   │
│                      │         │                           │
│  /oauth2/authorize   │         │ /oauth2/authorize         │
│  /oauth2/token       │         │ /oauth2/token             │
│  /userinfo           │         │ /userinfo                 │
│  /jwks               │────────→│ /jwks                     │
│                      │         │                           │
│  Test users:         │         │ Manages auth & tokens     │
│  - testuser          │         │                           │
│  - alice1            │         │                           │
│  (auto signup)       │         │                           │
└──────────────────────┘         └───────────────────────────┘
```

---

## Key Concepts

### .well-known Endpoints

**Purpose**: OAuth/OIDC clients discover auth configuration automatically

- `/.well-known/openid-configuration` → Full OIDC config (issuer, endpoints, scopes, etc)
- Points clients to authorization_endpoint, token_endpoint, jwks_uri, etc.

### Redirect URI

**What it is**: Where OIDC provider redirects user AFTER authentication

**Development**: `http://localhost:8000/auth/callback`
**Production**: `https://yourdomain.com/auth/callback`

**In MCP context**: 
- For MCP clients (Claude, etc): Usually `http://localhost:*` (local loopback)
- For browser: Standard HTTP/HTTPS URL

### Bearer Token Middleware

**Flow**:
1. Client sends: `Authorization: Bearer eyJhbGc...`
2. Middleware extracts token
3. Validates JWT signature using JWKS public keys
4. Decodes claims (sub, email, name, iss, exp, iat)
5. Stores in `request.state.user`
6. Tools access via: `ctx.request_context.request.state.user`

### Protected vs Public Paths

**Public** (No token required):
- `/.well-known/openid-configuration`
- `/auth/callback`
- `/health`

**Protected** (Token required):
- `/mcp/*` (All MCP endpoints)

---

## File Structure

```
your-project/
├── server.py              # FastMCP + Starlette app with OIDC
├── FastMCP-OIDC-Docs.md   # Architecture & concepts
├── TEST_GUIDE.sh          # Complete testing walkthrough
└── QUICK_START.md         # This file
```

---

## Common Commands

### Get Config
```bash
curl http://localhost:8000/.well-known/openid-configuration | jq
```

### Get Auth Code (Browser)
```
http://localhost:9400/oauth2/authorize?response_type=code&client_id=mcp-demo-client&redirect_uri=http://localhost:8000/auth/callback&scope=openid%20profile%20email
```

### Get Token
```bash
curl -X POST http://localhost:9400/oauth2/token \
-d "grant_type=authorization_code&code=CODE&client_id=mcp-demo-client&client_secret=demo-secret&redirect_uri=http://localhost:8000/auth/callback"
```

### Call Protected Tool
```bash
curl -X GET "http://localhost:8000/mcp/tools/call/add" \
-H "Authorization: Bearer TOKEN" \
-H "Content-Type: application/json" \
-d '{"a": 5, "b": 3}'
```

### Get User Info
```bash
curl -H "Authorization: Bearer TOKEN" http://localhost:8000/mcp/tools/call/get_user_info
```

### Verify Token at OIDC Server
```bash
curl -H "Authorization: Bearer TOKEN" http://localhost:9400/userinfo
```

---

## Customization

### Add More Users
```bash
curl -X PUT http://localhost:9400/users/alice \
-H "Content-Type: application/json" \
-d '{"email": "alice@example.com", "name": "Alice"}'
```

### Change Client Credentials (in server.py)
```python
CLIENT_ID = "my-custom-client"
CLIENT_SECRET = "my-custom-secret"
```

### Add More Protected Tools
```python
@mcp.tool()
def multiply(a: int, b: int, ctx: Context) -> int:
    """Multiply two numbers"""
    user = ctx.request_context.request.state.user
    logger.info(f"multiply() called by {user.get('sub')}")
    return a * b
```

### Extract User Info in Tools
```python
# Method 1: Via context
user = ctx.request_context.request.state.user
user_id = user.get("sub")
user_email = user.get("email")

# Method 2: Dedicated function
def get_user_from_context(ctx: Context) -> dict:
    try:
        return ctx.request_context.request.state.user
    except:
        return None
```

---

## Troubleshooting

**❌ "Missing or invalid Authorization header"**
→ Add `-H "Authorization: Bearer TOKEN"` to curl

**❌ "Invalid token"**
→ Token may be expired or malformed. Get fresh token.

**❌ "Token validation failed"**
→ Check JWKS endpoint is accessible: `curl http://localhost:9400/jwks`

**❌ "OIDC Config not found"**
→ Make sure mock server running: `pipx run oidc-provider-mock --port=9400`

**❌ "Connection refused on localhost:9400"**
→ Start mock OIDC server first (Terminal 1)

---

## Next Steps

1. ✅ Run this demo
2. ✅ Understand the flow
3. ✅ Add your own tools
4. ✅ Replace mock OIDC with real provider (Keycloak, Auth0, Azure, etc)
5. ✅ Deploy to production

---

## References

- FastMCP Docs: https://gofastmcp.com/
- OIDC Provider Mock: https://oidc-provider-mock.readthedocs.io/
- OpenID Connect: https://openid.net/connect/
- JWT Debugging: https://jwt.io/