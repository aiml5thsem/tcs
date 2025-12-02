# FastMCP + OIDC Complete Quick Start (RFC 8707/9728 Compliant)

## 🚀 5-Minute Setup

### Install

```bash
pip install fastmcp starlette uvicorn python-jose cryptography pyjwt httpx
pipx install oidc-provider-mock
```

### Run (3 Terminals)

**Terminal 1: OIDC Server**
```bash
pipx run oidc-provider-mock --port=9400

# Setup test user
curl -X PUT http://localhost:9400/users/testuser \
-H "Content-Type: application/json" \
-d '{"email": "testuser@example.com", "name": "Test User"}'
```

**Terminal 2: MCP Server**
```bash
python server.py
```

**Terminal 3: Test Everything**

See testing section below.

---

## 🎯 Key Endpoints (Updated)

| Endpoint | Purpose | Public? |
|----------|---------|---------|
| `/.well-known/openid-configuration` | OIDC Discovery | ✅ |
| `/.well-known/oauth-protected-resource` | RFC 9728 Metadata | ✅ |
| `/oauth/introspect` | RFC 7662 Token Check | ❌ |
| `/auth/callback` | OAuth Callback | ✅ |
| `/mcp/tools/call/add` | Protected Tool | ❌ Bearer |
| `/health` | Health Check | ✅ |

---

## 📋 New: RFC 9728 Protected Resource Metadata

**Endpoint**: `/.well-known/oauth-protected-resource`

```bash
curl http://localhost:8000/.well-known/oauth-protected-resource | jq
```

**Response** (tells MCP clients where to authenticate):
```json
{
  "resource": "http://localhost:8000",
  "authorization_servers": [
    {"location": "http://localhost:9400"}
  ],
  "scopes_supported": [
    "openid", "profile", "email",
    "mcp:tools.read", "mcp:tools.write"
  ],
  "mcp_version": "2025-11-24",
  "mcp_endpoints": ["/mcp/tools", "/mcp/resources"]
}
```

---

## 📍 New: RFC 8707 Resource Parameter

**When requesting auth code**, include `resource` parameter:

```
http://localhost:9400/oauth2/authorize?
  response_type=code&
  client_id=mcp-demo-client&
  redirect_uri=http://localhost:8000/auth/callback&
  scope=openid%20profile%20email%20mcp:tools.read&
  resource=http://localhost:8000  ← RFC 8707 (NEW)
```

**Token will include**:
```json
{
  "access_token": "...",
  "aud": "http://localhost:8000",  ← Resource indicator bound to token
  "scope": "openid profile email mcp:tools.read",
  ...
}
```

---

## ✅ New: Token Introspection (RFC 7662)

**Check if token is valid**:

```bash
curl -X POST http://localhost:8000/oauth/introspect \
  -d "token=$TOKEN" | jq

# Response:
{
  "active": true,
  "scope": "openid profile email mcp:tools.read",
  "aud": "http://localhost:8000",
  "exp": 1764706744,
  "sub": "testuser"
}
```

---

## 🔍 New: Scope-Based Access Control

**Insufficient scope error**:

```bash
# Token without mcp:tools.read scope
curl -X GET http://localhost:8000/mcp/tools/call/add \
  -H "Authorization: Bearer $TOKEN_WITHOUT_SCOPE" \
  -d '{"a": 5, "b": 3}'

# Response: 403 Forbidden
# With WWW-Authenticate header showing required scopes:
# WWW-Authenticate: Bearer realm="mcp", 
#   error="insufficient_scope", 
#   scope="mcp:tools.read",
#   resource_metadata="http://localhost:8000/.well-known/oauth-protected-resource"
```

---

## 📊 Complete Testing Workflow

```bash
# 1. Setup
OIDC="http://localhost:9400"
MCP="http://localhost:8000"
RESOURCE="http://localhost:8000"

# 2. Get auth metadata
curl $MCP/.well-known/oauth-protected-resource | jq

# 3. Open browser for auth code (copy code from redirect)
open "$OIDC/oauth2/authorize?response_type=code&client_id=mcp-demo-client&redirect_uri=$MCP/auth/callback&scope=openid%20profile%20email%20mcp:tools.read&resource=$RESOURCE"

# 4. Exchange code for token (use code from step 3)
CODE="YOUR_CODE_HERE"
TOKEN_RESPONSE=$(curl -s -X POST $OIDC/oauth2/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=authorization_code&code=$CODE&client_id=mcp-demo-client&client_secret=demo-secret&redirect_uri=$MCP/auth/callback&resource=$RESOURCE")

ACCESS_TOKEN=$(echo $TOKEN_RESPONSE | jq -r '.access_token')
echo "Token: $ACCESS_TOKEN"

# 5. Introspect token (RFC 7662)
curl -X POST $MCP/oauth/introspect -d "token=$ACCESS_TOKEN" | jq

# 6. Call protected tool
curl -X GET "$MCP/mcp/tools/call/add" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"a": 5, "b": 3}' | jq

# 7. Get user info with scope details
curl -X GET "$MCP/mcp/tools/call/get_user_info" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}' | jq

# 8. Test insufficient scope (401)
curl -X GET "$MCP/mcp/tools/call/add" \
  -d '{"a": 5, "b": 3}' -v

# 9. Verify token at jwt.io
# Copy access_token and visit https://jwt.io
# You'll see: aud: "http://localhost:8000"
```

---

## 📐 MCP Compliance Summary

| RFC | Feature | What | Why |
|-----|---------|------|-----|
| **8707** | Resource Indicators | `aud` claim + `resource` param | Prevent token misuse |
| **9728** | Protected Resource Metadata | `/.well-known/oauth-protected-resource` | Client discovery |
| **7662** | Token Introspection | `/oauth/introspect` endpoint | Real-time validation |
| **6750** | Bearer Token | `Authorization: Bearer TOKEN` | Standard token format |
| **8414** | OIDC Discovery | `/.well-known/openid-configuration` | Config discovery |

---

## 🎨 Architecture Diagram

```
Browser/Client
    ↓
    1. Requests /mcp/tools/add (no token)
    ↓
    2. Receives 401 + WWW-Authenticate header
       ┌─────────────────────────────────────────┐
       │ WWW-Authenticate: Bearer                │
       │   realm="mcp"                           │
       │   resource_metadata="/.well-known/..." │
       │   error="invalid_request"               │
       └─────────────────────────────────────────┘
    ↓
    3. Fetches resource metadata (discovers OIDC server)
    ↓
    4. Redirects to http://localhost:9400/oauth2/authorize
       with resource=http://localhost:8000
    ↓
    5. User authenticates
    ↓
    6. OIDC server creates token with:
       - aud: "http://localhost:8000"
       - scope: "mcp:tools.read"
    ↓
    7. Client retries: GET /mcp/tools/add + Bearer TOKEN
    ↓
    8. Server validates:
       ✓ JWT signature (JWKS)
       ✓ aud claim (RFC 8707)
       ✓ scopes (RFC 6750)
    ↓
    9. Tool executes ✅
```

---

## 🔧 Customization

### Add Tool with Specific Scope

```python
@mcp.tool()
def admin_function(ctx: Context):
    """Admin-only tool - requires mcp:admin scope"""
    user = ctx.request_context.request.state.user
    scopes = ctx.request_context.request.state.scopes
    
    if "mcp:admin" not in scopes:
        raise ToolError("Insufficient permissions: mcp:admin required")
    
    # Admin logic
    return "Admin result"
```

### Configure Scopes

```python
def get_required_scopes_for_path(path: str):
    if path.startswith("/mcp/tools/admin"):
        return ["mcp:admin"]
    if path.startswith("/mcp/tools"):
        return ["mcp:tools.read"]
    return ["openid"]
```

---

## 📚 Files Included

```
✅ server.py              - Complete MCP server (RFC 8707/9728 compliant)
✅ FastMCP-OIDC-Docs.md   - Full technical documentation
✅ TEST_GUIDE.sh          - Step-by-step testing
✅ QUICK_START.md         - This file
```

---

## ✨ What Makes This Production-Ready

1. **RFC Compliant**: All critical OAuth/OIDC RFCs implemented
2. **Scope Control**: Fine-grained access control per endpoint
3. **Proper Errors**: Standard-compliant error responses
4. **Token Validation**: JWT + Audience + Scope checks
5. **Discovery**: Clients auto-discover auth requirements
6. **Introspection**: Real-time token validation
7. **Logging**: Detailed logs for debugging
8. **Extensible**: Easy to add tools and scopes

---

## 🚀 Next Steps

1. ✅ Run this demo locally
2. ✅ Understand the MCP auth flow
3. ✅ Add your own tools with scope requirements
4. ✅ Replace oidc-provider-mock with real provider (Keycloak, Auth0)
5. ✅ Deploy to production (Docker, Kubernetes)
6. ✅ Integrate with Claude Desktop or other MCP clients

---

## 📖 References

- MCP Spec: https://modelcontextprotocol.io/specification/draft/basic/authorization
- RFC 8707: https://datatracker.ietf.org/doc/rfc8707/
- RFC 9728: https://datatracker.ietf.org/doc/rfc9728/
- RFC 7662: https://datatracker.ietf.org/doc/rfc7662/
- FastMCP: https://gofastmcp.com/
- JWT: https://jwt.io/
