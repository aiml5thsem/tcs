# Complete OIDC Mock Server Guide (Windows/Mac)
## From Installation to Token Generation & Validation

---

## Table of Contents
1. [Installation](#installation)
2. [Option 1: Without Registration (Any Client ID/Secret)](#option-1-without-registration)
3. [Option 2: With Registration (Production-like)](#option-2-with-registration)
4. [User Authentication & Claims](#user-authentication--claims)
5. [Token Validation](#token-validation)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Python 3.7+
- `pipx` package manager

### Install oidc-provider-mock (One-time)
```bash
# Windows / Mac / Linux
pipx install oidc-provider-mock
```

### Verify Installation
```bash
oidc-provider-mock --help
```

---

# OPTION 1: WITHOUT Registration (Default Mode)
**Use Case:** Development/testing with any client credentials

## Step 1: Start the Server

### Windows (CMD)
```cmd
pipx run oidc-provider-mock --port=9400
```

### Mac/Linux (Terminal)
```bash
pipx run oidc-provider-mock --port=9400
```

**Expected Output:**
```
00:45:51.784 INFO   uvicorn.error Started server process [16472]
00:45:51.784 INFO   uvicorn.error Waiting for application startup.
00:45:51.784 INFO   uvicorn.error Application startup complete.
00:45:51.784 INFO   uvicorn.error Uvicorn running on http://127.0.0.1:9400
```

---

## Step 2: Verify Server (Discovery Endpoint)

### Windows (CMD)
```cmd
curl http://localhost:9400/.well-known/openid-configuration
```

### Mac/Linux (Terminal)
```bash
curl http://localhost:9400/.well-known/openid-configuration
```

**Expected Response:**
```json
{
  "authorization_endpoint": "http://localhost:9400/oauth2/authorize",
  "token_endpoint": "http://localhost:9400/oauth2/token",
  "userinfo_endpoint": "http://localhost:9400/userinfo",
  "jwks_uri": "http://localhost:9400/jwks",
  "issuer": "http://localhost:9400",
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "scopes_supported": ["openid", "profile", "email", "address", "phone"]
}
```

---

## Step 3: Set Up User Claims (Optional)

**Before authentication, define user data:**

### Windows (CMD)
```cmd
curl -X PUT http://localhost:9400/users/testuser ^
-H "Content-Type: application/json" ^
-d "{\"email\": \"testuser@example.com\", \"name\": \"Test User\", \"nickname\": \"tuser\"}"
```

### Mac/Linux (Terminal)
```bash
curl -X PUT http://localhost:9400/users/testuser \
-H "Content-Type: application/json" \
-d '{"email": "testuser@example.com", "name": "Test User", "nickname": "tuser"}'
```

---

## Step 4: Get Authorization Code

**Important:** Without registration, use any client_id and client_secret.

### Option A: Browser (Recommended)

**Open in browser:**
```
http://localhost:9400/oauth2/authorize?response_type=code&client_id=my-test-client&redirect_uri=https://oidcdebugger.com/debug&scope=openid%20profile%20email&nonce=n-0S6_WzA2Mj&state=af0ifjsldkj
```

**What happens:**
1. Browser redirects to: `https://oidcdebugger.com/debug?code=AUTHORIZATION_CODE&state=af0ifjsldkj`
2. **Copy the `code` parameter** (the long string)

**Example code:**
```
code=aSltnulrpd0W8HsJDkX6xNbhmQBJve3yzZu3iWqvbzfzTtVs
```

---

### Option B: Curl Command

#### Windows (CMD)
```cmd
curl -v -L "http://localhost:9400/oauth2/authorize?response_type=code&client_id=my-test-client&redirect_uri=https://oidcdebugger.com/debug&scope=openid%20profile%20email&nonce=n-0S6_WzA2Mj" 2>&1 | findstr "Location"
```

#### Mac/Linux (Terminal)
```bash
curl -v -L "http://localhost:9400/oauth2/authorize?response_type=code&client_id=my-test-client&redirect_uri=https://oidcdebugger.com/debug&scope=openid%20profile%20email&nonce=n-0S6_WzA2Mj" 2>&1 | grep "Location"
```

**Expected Output:**
```
Location: https://oidcdebugger.com/debug?code=aSltnulrpd0W8HsJDkX6xNbhmQBJve3yzZu3iWqvbzfzTtVs&state=...
```

---

## Step 5: Exchange Authorization Code for Token

**Use the `code` from Step 4 here:**

### Windows (CMD)
```cmd
curl -X POST http://localhost:9400/oauth2/token ^
-H "Content-Type: application/x-www-form-urlencoded" ^
-d "grant_type=authorization_code&code=aSltnulrpd0W8HsJDkX6xNbhmQBJve3yzZu3iWqvbzfzTtVs&client_id=my-test-client&client_secret=my-secret&redirect_uri=https://oidcdebugger.com/debug"
```

### Mac/Linux (Terminal)
```bash
curl -X POST http://localhost:9400/oauth2/token \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "grant_type=authorization_code&code=aSltnulrpd0W8HsJDkX6xNbhmQBJve3yzZu3iWqvbzfzTtVs&client_id=my-test-client&client_secret=my-secret&redirect_uri=https://oidcdebugger.com/debug"
```

**Expected Response:**
```json
{
  "access_token": "b14gVQzuHs9E3S4vnuyDC4l8Jx0X0nGwL213QtOfHM",
  "id_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InltOGw2Q2ZzZ0F2NEprN0x3MmFqaFlJbkp0bDZZSXA2UFRzTzVzZlZvazAifQ.eyJpc3MiOiJodHRwOi8vbG9jYWxob3N0Ojk0MDAiLCJhdWQiOlsiMmMyNmJjZjktMDZjYS00OTk0LWFmOWQtY2ZlMDExNTI3NTIxIl0sImlhdCI6MTc2NDcwMzE0NCwiZXhwIjoxNzY0NzA2NzQ0LCJhdXRoX3RpbWUiOjE3NjQ3MDMxNDQsImF0X2hhc2giOiJyWk5UM09ZQXNzU1AzZEJnT2JnZHlRIiwic3ViIjoiZmFiIn0.UZ0vqDYgiD8bid5OZKtH9nN6HKWP6VW5HsooUZVVlYULxlBaTlmY3dePJo8-XfZkV1Y7L5MHI7taYpDgofOO5vuRL0pNUQXPMnC3e4ysOJEojh3ee-1WFh7dHh5t57JNOYBkyljQ-z7NjHoJERwNQf-fTCicMbcYHFvUwtksvP_ieb-dZatdTWkh6ZCemCW1NcWOKxJRYE420fRvfWkbiOypU8u-gtzas3TseSA4uCxzV0W0KMGkG50gSdLjKI9iPP2Bg0ScY8Ig3iP8kE_GdnbVQAIlSOfvgNO-eeLJosMfx8-S46oICdaKDJVBonVq_pBognxo6kEtfL2khXt2Sg",
  "refresh_token": "EGb6QudG3XGsBQEsYOMn6MAFCDoYB7g8FD02H14v5GPPxyTQ",
  "scope": "openid profile email",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

---

## Step 6: Validate Token - Get User Info

### Windows (CMD)
```cmd
curl -H "Authorization: Bearer b14gVQzuHs9E3S4vnuyDC4l8Jx0X0nGwL213QtOfHM" http://localhost:9400/userinfo
```

### Mac/Linux (Terminal)
```bash
curl -H "Authorization: Bearer b14gVQzuHs9E3S4vnuyDC4l8Jx0X0nGwL213QtOfHM" http://localhost:9400/userinfo
```

**Expected Response:**
```json
{
  "sub": "testuser",
  "email": "testuser@example.com",
  "name": "Test User",
  "nickname": "tuser",
  "iss": "http://localhost:9400"
}
```

---

---

# OPTION 2: WITH Registration (Production-like)
**Use Case:** Production testing where clients must be registered first

## Step 1: Start Server with Registration Requirement

### Windows (CMD)
```cmd
pipx run oidc-provider-mock --require-registration=true --port=9400
```

### Mac/Linux (Terminal)
```bash
pipx run oidc-provider-mock --require-registration=true --port=9400
```

---

## Step 2: Register a Client

**Without registration, Step 5 will fail with:** `invalid_client` error

### Windows (CMD)
```cmd
curl -X POST http://localhost:9400/oauth2/clients ^
-H "Content-Type: application/json" ^
-d "{\"redirect_uris\": [\"https://oidcdebugger.com/debug\"], \"grant_types\": [\"authorization_code\", \"refresh_token\"]}" ^
-v
```

### Mac/Linux (Terminal)
```bash
curl -X POST http://localhost:9400/oauth2/clients \
-H "Content-Type: application/json" \
-d '{"redirect_uris": ["https://oidcdebugger.com/debug"], "grant_types": ["authorization_code", "refresh_token"]}' \
-v
```

**Expected Response:**
```json
{
  "client_id": "050d5966-fb55-4887-a1fe-c9cd27d5386f",
  "client_secret": "yso-fwkXObTx5SEOLPDruQ",
  "grant_types": ["authorization_code", "refresh_token"],
  "redirect_uris": ["https://oidcdebugger.com/debug"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "client_secret_basic"
}
```

**Save these values:**
- `client_id`: `050d5966-fb55-4887-a1fe-c9cd27d5386f`
- `client_secret`: `yso-fwkXObTx5SEOLPDruQ`

---

## Step 3: Set Up User Claims (Same as Option 1)

### Windows (CMD)
```cmd
curl -X PUT http://localhost:9400/users/alice1 ^
-H "Content-Type: application/json" ^
-d "{\"email\": \"alice@example.com\", \"name\": \"Alice Smith\"}"
```

### Mac/Linux (Terminal)
```bash
curl -X PUT http://localhost:9400/users/alice1 \
-H "Content-Type: application/json" \
-d '{"email": "alice@example.com", "name": "Alice Smith"}'
```

---

## Step 4: Get Authorization Code (Use Registered client_id)

### Browser Method
```
http://localhost:9400/oauth2/authorize?response_type=code&client_id=050d5966-fb55-4887-a1fe-c9cd27d5386f&redirect_uri=https://oidcdebugger.com/debug&scope=openid%20profile%20email&nonce=n-0S6_WzA2Mj
```

**Result:** Redirects to `https://oidcdebugger.com/debug?code=XXXXX`
- **Copy the code value**

---

## Step 5: Exchange Code for Token (Use Registered Credentials)

### Windows (CMD)
```cmd
curl -X POST http://localhost:9400/oauth2/token ^
-H "Content-Type: application/x-www-form-urlencoded" ^
-d "grant_type=authorization_code&code=YOUR_CODE_HERE&client_id=050d5966-fb55-4887-a1fe-c9cd27d5386f&client_secret=yso-fwkXObTx5SEOLPDruQ&redirect_uri=https://oidcdebugger.com/debug"
```

### Mac/Linux (Terminal)
```bash
curl -X POST http://localhost:9400/oauth2/token \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "grant_type=authorization_code&code=YOUR_CODE_HERE&client_id=050d5966-fb55-4887-a1fe-c9cd27d5386f&client_secret=yso-fwkXObTx5SEOLPDruQ&redirect_uri=https://oidcdebugger.com/debug"
```

**Expected Response:** (Same as Option 1)
```json
{
  "access_token": "...",
  "id_token": "...",
  "refresh_token": "...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

---

## Step 6: Validate Token - Get User Info (Same as Option 1)

### Windows (CMD)
```cmd
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" http://localhost:9400/userinfo
```

### Mac/Linux (Terminal)
```bash
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" http://localhost:9400/userinfo
```

**Expected Response:**
```json
{
  "sub": "alice1",
  "email": "alice@example.com",
  "name": "Alice Smith",
  "iss": "http://localhost:9400"
}
```

---

---

# User Authentication & Claims

## Scopes and Claims Mapping

**Standard OIDC scope → claims mapping:**

| Scope | Claims Included |
|-------|-----------------|
| `openid` | `sub`, `iss` |
| `profile` | `name`, `nickname`, `family_name`, `given_name` |
| `email` | `email`, `email_verified` |
| `address` | `address` (full address object) |
| `phone` | `phone_number`, `phone_number_verified` |

**Important:** Claims only appear in ID token and userinfo if both:
1. User has the claim set via `PUT /users/{sub}`
2. Authorization request includes appropriate scope

---

## Setting Custom Claims

### Windows (CMD)
```cmd
curl -X PUT http://localhost:9400/users/alice1 ^
-H "Content-Type: application/json" ^
-d "{\"email\": \"alice@example.com\", \"name\": \"Alice\", \"custom\": {\"department\": \"Engineering\", \"role\": \"Admin\"}}"
```

### Mac/Linux (Terminal)
```bash
curl -X PUT http://localhost:9400/users/alice1 \
-H "Content-Type: application/json" \
-d '{"email": "alice@example.com", "name": "Alice", "custom": {"department": "Engineering", "role": "Admin"}}'
```

**In authorization request, include scope:** `openid profile email`

**ID Token will contain:**
```json
{
  "sub": "alice1",
  "email": "alice@example.com",
  "name": "Alice",
  "custom": {
    "department": "Engineering",
    "role": "Admin"
  }
}
```

---

---

# Token Validation

## Method 1: Userinfo Endpoint (Easiest)

### Windows (CMD)
```cmd
curl -H "Authorization: Bearer ACCESS_TOKEN" http://localhost:9400/userinfo
```

### Mac/Linux (Terminal)
```bash
curl -H "Authorization: Bearer ACCESS_TOKEN" http://localhost:9400/userinfo
```

**If token is valid:** Returns user claims
**If token is invalid:** Returns 401 error

---

## Method 2: Decode ID Token (jwt.io)

1. Get `id_token` from token response
2. Visit https://jwt.io/
3. Paste ID token in "Encoded" section
4. Get JWKS public key:

### Windows (CMD)
```cmd
curl http://localhost:9400/jwks
```

### Mac/Linux (Terminal)
```bash
curl http://localhost:9400/jwks
```

5. Verify signature using JWKS key

---

## Method 3: Refresh Token

**Get new access token without re-authenticating:**

### Windows (CMD)
```cmd
curl -X POST http://localhost:9400/oauth2/token ^
-H "Content-Type: application/x-www-form-urlencoded" ^
-d "grant_type=refresh_token&refresh_token=YOUR_REFRESH_TOKEN&client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET"
```

### Mac/Linux (Terminal)
```bash
curl -X POST http://localhost:9400/oauth2/token \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "grant_type=refresh_token&refresh_token=YOUR_REFRESH_TOKEN&client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET"
```

---

---

# Troubleshooting

## Error: "invalid_client"
**Cause:** With `--require-registration=true`, client must be registered first
**Fix:** 
- Either start with `--require-registration=false` (default)
- Or register client first using Step 2 of Option 2

---

## Error: "invalid_request" or "authorization_code_expired"
**Cause:** Authorization code expires in ~10 minutes
**Fix:** Get fresh code from Step 4

---

## Error: "redirect_uri_mismatch"
**Cause:** Redirect URI in auth request doesn't match registered URI
**Fix:** Ensure exact match:
- In registration: `"redirect_uris": ["https://oidcdebugger.com/debug"]`
- In auth request: `redirect_uri=https://oidcdebugger.com/debug`

---

## UserInfo returns empty claims
**Cause:** User not set or scope not included
**Fix:**
1. Set user: `PUT /users/{sub}`
2. Include scope in auth request: `scope=openid%20profile%20email`

---

## 404 Error on `/oauth2/auth` endpoint
**Cause:** Wrong endpoint path
**Fix:** Use `/oauth2/authorize` not `/oauth2/auth`

---

---

# Quick Reference Commands

## Server Start
```bash
# Without registration (default)
pipx run oidc-provider-mock --port=9400

# With registration
pipx run oidc-provider-mock --require-registration=true --port=9400
```

## Discovery
```bash
curl http://localhost:9400/.well-known/openid-configuration
```

## Register Client (Option 2 only)
```bash
curl -X POST http://localhost:9400/oauth2/clients \
-H "Content-Type: application/json" \
-d '{"redirect_uris": ["https://oidcdebugger.com/debug"]}'
```

## Set User Claims
```bash
curl -X PUT http://localhost:9400/users/testuser \
-H "Content-Type: application/json" \
-d '{"email": "test@example.com", "name": "Test User"}'
```

## Authorization Browser Link
```
http://localhost:9400/oauth2/authorize?response_type=code&client_id=YOUR_CLIENT_ID&redirect_uri=https://oidcdebugger.com/debug&scope=openid%20profile%20email
```

## Get Token
```bash
curl -X POST http://localhost:9400/oauth2/token \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "grant_type=authorization_code&code=YOUR_CODE&client_id=YOUR_CLIENT_ID&client_secret=YOUR_SECRET&redirect_uri=https://oidcdebugger.com/debug"
```

## Validate Token
```bash
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" http://localhost:9400/userinfo
```

## Revoke User Tokens
```bash
curl -X POST http://localhost:9400/users/testuser/revoke-tokens
```

---

## References
- Official Docs: https://oidc-provider-mock.readthedocs.io/
- GitHub: https://github.com/geigerzaehler/oidc-provider-mock
- OpenID Connect Standard: https://openid.net/connect/