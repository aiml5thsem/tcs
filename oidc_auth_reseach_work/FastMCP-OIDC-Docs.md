# FastMCP with MCP-Compliant OIDC Authentication
## Complete Implementation Guide (RFC 8707 + RFC 9728)

---

## What's NEW & CRITICAL ✅

### Before (Incomplete)
- ❌ Missing RFC 8707 resource indicator validation
- ❌ No scope-based access control
- ❌ Missing token introspection endpoint
- ❌ Incomplete WWW-Authenticate headers
- ❌ Not MCP-specification compliant

### Now (Fully Compliant) ✅
- ✅ **RFC 8707**: Resource Indicator validation (aud claim)
- ✅ **RFC 9728**: Protected Resource Metadata endpoint
- ✅ **RFC 7662**: Token Introspection endpoint
- ✅ **Scope Validation**: insufficient_scope error handling
- ✅ **Proper Headers**: WWW-Authenticate with scope requirements
- ✅ **MCP Spec Compliant**: June 2025 Authorization specification

---

## Architecture: MCP OAuth 2.0 Flow (RFC 8707)

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Client (Claude, etc)                    │
└────────────────────┬────────────────────────────────────────────┘
                     │
          1. /mcp → 401 Unauthorized
                     │
          + WWW-Authenticate header points to resource metadata
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              FastMCP Server (Port 8000)                        │
│                                                                 │
│  GET /.well-known/oauth-protected-resource                    │
│  ↓ Returns resource metadata                                  │
│  {                                                              │
│    "resource": "http://localhost:8000",                        │
│    "authorization_servers": [                                  │
│      {"location": "http://localhost:9400"}                    │
│    ],                                                           │
│    "scopes_supported": [                                        │
│      "openid", "profile", "email",                            │
│      "mcp:tools.read", "mcp:tools.write"                      │
│    ]                                                            │
│  }                                                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
          2. Client discovers auth server
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│           OIDC Provider (http://localhost:9400)               │
│                                                                 │
│  GET /oauth2/authorize?                                        │
│    response_type=code&                                        │
│    client_id=...&                                             │
│    redirect_uri=http://localhost:8000/auth/callback&          │
│    scope=openid profile email mcp:tools.read&                │
│    resource=http://localhost:8000  ← RFC 8707                │
│                                                                 │
│  ↓ User authenticates                                         │
│  ↓ Issues token with aud=http://localhost:8000               │
│                                                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
          3. Exchange code for token
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Token Response                          │
│                                                                 │
│  {                                                              │
│    "access_token": "eyJhbGc...",  ← Contains aud claim        │
│    "token_type": "Bearer",                                     │
│    "expires_in": 3600,                                         │
│    "scope": "openid profile email mcp:tools.read"             │
│  }                                                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
          4. Client calls MCP tool
          with Authorization: Bearer TOKEN
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│           FastMCP Server Validation                            │
│                                                                 │
│  1. Extract token from Authorization header                   │
│  2. Validate JWT signature (JWKS)                              │
│  3. RFC 8707: Check aud claim contains "http://localhost:8000"│
│  4. Check scopes include "mcp:tools.read"                      │
│  5. Inject user info into tool context                        │
│  6. Execute tool                                               │
│                                                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
             Tool executes successfully ✅
```

---

## Key Concepts Explained

### RFC 8707: Resource Indicators

**What**: OAuth tokens bound to specific resources

**Why Important**: Prevents token misuse - token can only be used at its intended resource

**How it works**:
```
Authorization Request:
GET /oauth2/authorize?
  ...&
  resource=http://localhost:8000  ← Client tells server which resource

Token Response includes `aud` claim:
{
  "access_token": "...",
  "aud": "http://localhost:8000",  ← Token bound to this resource
  ...
}

MCP Server validates:
if "http://localhost:8000" NOT in token["aud"]:
  return 401 Invalid token for this resource
```

### RFC 9728: Protected Resource Metadata

**What**: Discovery endpoint telling clients where to authenticate

**Endpoint**: `/.well-known/oauth-protected-resource`

**Example Response**:
```json
{
  "resource": "http://localhost:8000",
  "authorization_servers": [
    {"location": "http://localhost:9400"}
  ],
  "scopes_supported": [
    "openid", "profile", "email",
    "mcp:tools.read", "mcp:tools.write",
    "mcp:resources.read"
  ],
  "token_endpoint_auth_methods_supported": [
    "client_secret_basic"
  ]
}
```

**How MCP clients use it**:
1. Get 401 response with `WWW-Authenticate` header
2. Read `resource_metadata` URL from header
3. Fetch metadata from resource server
4. Discover authorization_servers list
5. Use first server to authenticate
6. Retry with token

### RFC 7662: Token Introspection

**What**: Endpoint to validate/check tokens in real-time

**Endpoint**: `/oauth/introspect`

**Request**:
```bash
curl -X POST http://localhost:8000/oauth/introspect \
  -d "token=YOUR_TOKEN"
```

**Response**:
```json
{
  "active": true,
  "scope": "openid profile email mcp:tools.read",
  "client_id": "mcp-demo-client",
  "username": "testuser",
  "exp": 1764706744,
  "aud": "http://localhost:8000"
}
```

---

## Scope-Based Access Control

### Scope Requirements by Endpoint

```
/mcp/tools/*              → mcp:tools.read (+ mcp:tools.write for mutations)
/mcp/resources/*          → mcp:resources.read (+ mcp:resources.write for mutations)
/mcp/prompts/*            → mcp:prompts.read (+ mcp:prompts.write for mutations)
```

### Insufficient Scope Error (RFC 6750)

When token lacks required scopes:

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer realm="mcp", 
  error="insufficient_scope", 
  scope="mcp:tools.read mcp:tools.write",
  resource_metadata="http://localhost:8000/.well-known/oauth-protected-resource",
  error_description="Additional tool write permission required"
```

**MCP Client behavior**:
1. Parse `scope` from WWW-Authenticate
2. Request user authorization with those scopes
3. Get new token with required scopes
4. Retry request

---

## Bearer Token Validation Flow (Updated)

```python
# In BearerTokenMiddleware:

1. Extract token from Authorization: Bearer TOKEN
   ↓
2. Validate JWT signature using JWKS
   ↓
3. RFC 8707: Check aud claim contains resource_id
   ✗ → 401 Invalid token for this resource
   ✓ → Continue
   ↓
4. Extract scopes from token
   ↓
5. Get required scopes for endpoint
   ↓
6. Validate token has ALL required scopes
   ✗ → 403 Insufficient scope (with WWW-Authenticate)
   ✓ → Continue
   ↓
7. Store user info + scopes in request.state
   ↓
8. Execute tool
```

---

## Implementation Details

### Configuration

```python
# In server.py

MCP_RESOURCE_ID = "http://localhost:8000"  # RFC 8707 resource identifier

# Scope mapping (customize for your needs)
def get_required_scopes_for_path(path: str):
    if path.startswith("/mcp/tools"):
        return ["mcp:tools.read"]
    if path.startswith("/mcp/resources"):
        return ["mcp:resources.read"]
    return ["openid", "profile", "email"]
```

### Authorization Request with Resource Parameter

```python
# In OIDCConfig.get_tokens():

response = await client.post(
    token_endpoint,
    data={
        "grant_type": "authorization_code",
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "resource": MCP_RESOURCE_ID,  # ← RFC 8707 resource parameter
    }
)
```

### Token Validation with Audience Check

```python
# In BearerTokenMiddleware:

# Validate RFC 8707 resource indicator
valid_resource, error = validate_resource_indicator(payload, MCP_RESOURCE_ID)
if not valid_resource:
    return 401 Unauthorized  # Token not for this resource

# Validate scopes
valid_scope, missing = validate_scopes(payload, required_scopes)
if not valid_scope:
    return 403 Insufficient scope  # with WWW-Authenticate header
```

### Token Introspection Endpoint

```python
# RFC 7662 Token Introspection
# Endpoint: POST /oauth/introspect
# Validates tokens and returns their claims

@app.route("/oauth/introspect")
async def token_introspection(request):
    token = request.form["token"]
    # Validate token
    # Return RFC 7662 response format
    return {
        "active": True,
        "scope": "openid profile email mcp:tools.read",
        "aud": "http://localhost:8000",
        ...
    }
```

---

## Public vs Protected Paths

```python
PUBLIC_PATHS = {
    "/.well-known/openid-configuration",      # OIDC Discovery
    "/.well-known/oauth-protected-resource",  # RFC 9728 metadata
    "/auth/callback",                         # OAuth callback
    "/health",                                # Health check
}

PROTECTED_PATHS = {
    "/mcp/*",                    # All MCP endpoints require Bearer token
    "/oauth/introspect",         # Token introspection (authenticated)
}
```

---

## Testing the Implementation

### 1. Verify OIDC Discovery

```bash
# Server's OIDC config
curl http://localhost:8000/.well-known/openid-configuration | jq

# Server's resource metadata (RFC 9728)
curl http://localhost:8000/.well-known/oauth-protected-resource | jq
```

### 2. Get Auth Code with Resource Parameter

```bash
# Browser URL with resource parameter (RFC 8707)
http://localhost:9400/oauth2/authorize?
  response_type=code&
  client_id=mcp-demo-client&
  redirect_uri=http://localhost:8000/auth/callback&
  scope=openid%20profile%20email%20mcp:tools.read&
  resource=http://localhost:8000
```

### 3. Exchange Code for Token

```bash
curl -X POST http://localhost:9400/oauth2/token \
  -d "grant_type=authorization_code&
      code=YOUR_CODE&
      client_id=mcp-demo-client&
      client_secret=demo-secret&
      redirect_uri=http://localhost:8000/auth/callback&
      resource=http://localhost:8000"
```

### 4. Inspect Token at jwt.io

1. Copy access_token
2. Visit https://jwt.io/
3. Paste token
4. View claims including `aud` (audience/resource)

### 5. Call Protected Tool

```bash
TOKEN="YOUR_ACCESS_TOKEN"

curl -X GET http://localhost:8000/mcp/tools/call/add \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"a": 5, "b": 3}'
```

### 6. Test Introspection Endpoint

```bash
curl -X POST http://localhost:8000/oauth/introspect \
  -d "token=$TOKEN" | jq
```

### 7. Test Insufficient Scope Error

```bash
# Request tool with token lacking required scope
curl -X GET http://localhost:8000/mcp/tools/call/add \
  -H "Authorization: Bearer TOKEN_WITHOUT_SCOPE" \
  -d '{"a": 5, "b": 3}'

# Response: 403 Forbidden with WWW-Authenticate header
# Header includes required scopes and resource_metadata URL
```

---

## Production Deployment

### Replace OIDC Provider

```python
# Example: Keycloak
OIDC_SERVER_URL = "https://keycloak.company.com/realms/mcp"
CONFIG_URL = f"{OIDC_SERVER_URL}/.well-known/openid-configuration"

# Example: Auth0
OIDC_SERVER_URL = "https://yourtenant.auth0.com"
CONFIG_URL = f"{OIDC_SERVER_URL}/.well-known/openid-configuration"

# Example: Azure Entra ID
OIDC_SERVER_URL = "https://login.microsoftonline.com/YOUR_TENANT/v2.0"
CONFIG_URL = f"{OIDC_SERVER_URL}/.well-known/openid-configuration"
```

### Configure for Production

```python
MCP_RESOURCE_ID = "https://api.mycompany.com"
MCP_BASE_URL = "https://api.mycompany.com"
REDIRECT_URI = "https://api.mycompany.com/auth/callback"

# Register redirect_uri in your OIDC provider admin console
```

### Docker Deployment

```dockerfile
FROM python:3.11
RUN pip install fastmcp starlette uvicorn python-jose cryptography pyjwt httpx
COPY server.py .
ENV OIDC_SERVER_URL=https://keycloak.company.com/realms/mcp
ENV MCP_RESOURCE_ID=https://api.company.com
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Compliance Checklist ✅

- ✅ RFC 6750: Bearer Token Usage
- ✅ RFC 7662: Token Introspection
- ✅ RFC 8414: OAuth Authorization Server Metadata
- ✅ RFC 8707: Resource Indicators for OAuth 2.0
- ✅ RFC 9728: OAuth 2.0 Protected Resource Metadata
- ✅ MCP Authorization Spec (June 2025)

---

## Common Issues & Solutions

**❌ "Token not intended for this resource"**
→ Ensure auth request includes `resource=MCP_RESOURCE_ID`

**❌ "Insufficient scope: missing mcp:tools.read"**
→ Auth request must include scope `mcp:tools.read`

**❌ "Missing aud claim in token"**
→ OIDC provider not implementing RFC 8707; may need configuration

**❌ "WWW-Authenticate header missing"**
→ Check middleware is returning proper headers on 401/403

---

## References

- **RFC 6750**: Bearer Token Usage - https://datatracker.ietf.org/doc/rfc6750/
- **RFC 7662**: Token Introspection - https://datatracker.ietf.org/doc/rfc7662/
- **RFC 8414**: OAuth Authorization Server Metadata - https://datatracker.ietf.org/doc/rfc8414/
- **RFC 8707**: Resource Indicators for OAuth 2.0 - https://datatracker.ietf.org/doc/rfc8707/
- **RFC 9728**: OAuth 2.0 Protected Resource Metadata - https://datatracker.ietf.org/doc/rfc9728/
- **MCP Specification**: https://modelcontextprotocol.io/specification/draft/basic/authorization
- **FastMCP Docs**: https://gofastmcp.com/
- **JWT Debugging**: https://jwt.io/

