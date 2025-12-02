"""
Complete Testing Guide for FastMCP with OIDC Authentication
Step-by-step commands to test the entire flow
"""

# ============================================================================
# STEP 1: Start Services (Open 3 terminals)
# ============================================================================

# Terminal 1: Start Mock OIDC Server
# --------------------------------
pipx run oidc-provider-mock --port=9400

# Expected output:
# Uvicorn running on http://127.0.0.1:9400


# Terminal 2: Start FastMCP Server
# --------------------------------
python server.py

# Expected output (snippet):
# ╔══════════════════════════════════════════════════════════════╗
# ║        FastMCP with OIDC Authentication Demo                ║
# ╚══════════════════════════════════════════════════════════════╝
# ...
# Uvicorn running on http://localhost:8000


# Terminal 3: Run Test Commands
# --------------------------------
# (Use commands below)


# ============================================================================
# STEP 2: Set Up Test User in OIDC Server
# ============================================================================

curl -X PUT http://localhost:9400/users/testuser \
-H "Content-Type: application/json" \
-d '{"email": "testuser@example.com", "name": "Test User", "nickname": "tuser"}'

# Expected response:
# {"sub":"testuser","email":"testuser@example.com","name":"Test User",...}


# ============================================================================
# STEP 3: Verify OIDC Configuration
# ============================================================================

# Check Mock OIDC Server config
curl http://localhost:9400/.well-known/openid-configuration | jq

# Expected response includes:
# {
#   "authorization_endpoint": "http://localhost:9400/oauth2/authorize",
#   "token_endpoint": "http://localhost:9400/oauth2/token",
#   "userinfo_endpoint": "http://localhost:9400/userinfo",
#   "jwks_uri": "http://localhost:9400/jwks",
#   ...
# }


# Check MCP Server's OIDC config (proxied)
curl http://localhost:8000/.well-known/openid-configuration | jq

# Should return same as above


# ============================================================================
# STEP 4: Get Authorization Code (OAuth Flow)
# ============================================================================

# Option A: In Browser (Recommended for testing)
# Open: http://localhost:9400/oauth2/authorize?response_type=code&client_id=mcp-demo-client&redirect_uri=http://localhost:8000/auth/callback&scope=openid%20profile%20email

# Browser will redirect to:
# http://localhost:8000/auth/callback?code=YOUR_CODE_HERE&state=...

# Copy the code value


# Option B: Using curl
REDIRECT_RESPONSE=$(curl -v -L "http://localhost:9400/oauth2/authorize?response_type=code&client_id=mcp-demo-client&redirect_uri=http://localhost:8000/auth/callback&scope=openid%20profile%20email" 2>&1)

# Extract code from redirect URL (find "Location:" header)
CODE=$(echo $REDIRECT_RESPONSE | grep -oP '(?<=code=)[^&]*' | head -1)
echo "Authorization Code: $CODE"


# ============================================================================
# STEP 5: Exchange Code for Access Token
# ============================================================================

# Use code from Step 4
CODE="YOUR_CODE_FROM_STEP_4"

curl -X POST http://localhost:9400/oauth2/token \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "grant_type=authorization_code&code=$CODE&client_id=mcp-demo-client&client_secret=demo-secret&redirect_uri=http://localhost:8000/auth/callback"

# Expected response:
# {
#   "access_token": "eyJhbGc...",  ← COPY THIS
#   "token_type": "Bearer",
#   "expires_in": 3600,
#   "refresh_token": "...",
#   "id_token": "eyJ..."
# }


# ============================================================================
# STEP 6: Test Protected MCP Tools with Bearer Token
# ============================================================================

# Use access_token from Step 5
TOKEN="YOUR_ACCESS_TOKEN_FROM_STEP_5"

# Test: Add Tool
curl -X GET "http://localhost:8000/mcp/tools/add" \
-H "Authorization: Bearer $TOKEN" \
-d '{"a": 5, "b": 3}'

# Alternative with query params (for simple MCP):
curl -X GET "http://localhost:8000/mcp/tools/call/add" \
-H "Authorization: Bearer $TOKEN" \
-H "Content-Type: application/json" \
-d '{"a": 5, "b": 3}'

# Expected response:
# {"result": 8}


# Test: Subtract Tool
curl -X GET "http://localhost:8000/mcp/tools/call/subtract" \
-H "Authorization: Bearer $TOKEN" \
-H "Content-Type: application/json" \
-d '{"a": 10, "b": 3}'

# Expected response:
# {"result": 7}


# Test: Get User Info (Show authenticated user)
curl -X GET "http://localhost:8000/mcp/tools/call/get_user_info" \
-H "Authorization: Bearer $TOKEN" \
-H "Content-Type: application/json" \
-d '{}'

# Expected response:
# {
#   "sub": "testuser",
#   "email": "testuser@example.com",
#   "name": "Test User",
#   "iss": "http://localhost:9400",
#   "exp": 1764706744,
#   "token_issued_at": "2025-12-03T..."
# }


# ============================================================================
# STEP 7: Test Authentication Failures
# ============================================================================

# Test: Missing Bearer Token
curl -X GET "http://localhost:8000/mcp/tools/call/add" \
-H "Content-Type: application/json" \
-d '{"a": 5, "b": 3}'

# Expected response (401):
# {"error": "Missing or invalid Authorization header"}


# Test: Invalid Token
curl -X GET "http://localhost:8000/mcp/tools/call/add" \
-H "Authorization: Bearer invalid-token-xyz" \
-H "Content-Type: application/json" \
-d '{"a": 5, "b": 3}'

# Expected response (401):
# {"error": "Invalid token"}


# Test: Expired Token
# (Wait until token expires, or use old token from previous run)
curl -X GET "http://localhost:8000/mcp/tools/call/add" \
-H "Authorization: Bearer $OLD_TOKEN" \
-H "Content-Type: application/json" \
-d '{"a": 5, "b": 3}'

# Expected response (401):
# {"error": "Token expired"}


# ============================================================================
# STEP 8: Validate Token with OIDC Userinfo Endpoint
# ============================================================================

# Verify token is valid by checking userinfo at OIDC server
curl -H "Authorization: Bearer $TOKEN" http://localhost:9400/userinfo

# Expected response:
# {
#   "sub": "testuser",
#   "email": "testuser@example.com",
#   "name": "Test User",
#   "iss": "http://localhost:9400"
# }


# ============================================================================
# STEP 9: Decode JWT Token at jwt.io (Manual Verification)
# ============================================================================

# 1. Copy the access_token from Step 5
# 2. Visit https://jwt.io/
# 3. Paste token in "Encoded" textarea
# 4. Scroll down to see decoded claims:
#    - Header: {"alg": "RS256", "typ": "JWT", ...}
#    - Payload: {"sub": "testuser", "email": "...", "iss": "...", ...}
#    - Signature: Verified with public key from JWKS


# ============================================================================
# STEP 10: Get JWKS Public Keys (For Manual Token Verification)
# ============================================================================

curl http://localhost:9400/jwks | jq

# Expected response:
# {
#   "keys": [
#     {
#       "kty": "RSA",
#       "use": "sig",
#       "kid": "ym8l6Cfs...",
#       "n": "...",  ← Public key
#       "e": "AQAB",
#       ...
#     }
#   ]
# }


# ============================================================================
# STEP 11: Refresh Token (Get New Access Token)
# ============================================================================

REFRESH_TOKEN="YOUR_REFRESH_TOKEN_FROM_STEP_5"

curl -X POST http://localhost:9400/oauth2/token \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "grant_type=refresh_token&refresh_token=$REFRESH_TOKEN&client_id=mcp-demo-client&client_secret=demo-secret"

# Expected response: New access_token + refresh_token


# ============================================================================
# STEP 12: Revoke Tokens (Logout)
# ============================================================================

curl -X POST "http://localhost:9400/users/testuser/revoke-tokens"

# Expected: Tokens for testuser are revoked
# Subsequent requests with old token will fail


# ============================================================================
# SUMMARY: Complete Test Flow Script
# ============================================================================

#!/bin/bash

echo "🚀 FastMCP OIDC Test Flow"
echo "========================="

# 1. Set up user
echo -e "\n1️⃣  Setting up test user..."
curl -X PUT http://localhost:9400/users/testuser \
-H "Content-Type: application/json" \
-d '{"email": "testuser@example.com", "name": "Test User"}' > /dev/null
echo "✓ User created"

# 2. Get code
echo -e "\n2️⃣  Getting authorization code (open in browser)..."
echo "👉 http://localhost:9400/oauth2/authorize?response_type=code&client_id=mcp-demo-client&redirect_uri=http://localhost:8000/auth/callback&scope=openid%20profile%20email"
read -p "Enter code from redirect URL: " CODE

# 3. Exchange code
echo -e "\n3️⃣  Exchanging code for token..."
TOKEN_RESPONSE=$(curl -s -X POST http://localhost:9400/oauth2/token \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "grant_type=authorization_code&code=$CODE&client_id=mcp-demo-client&client_secret=demo-secret&redirect_uri=http://localhost:8000/auth/callback")
ACCESS_TOKEN=$(echo $TOKEN_RESPONSE | jq -r '.access_token')
echo "✓ Token received: ${ACCESS_TOKEN:0:20}..."

# 4. Test tools
echo -e "\n4️⃣  Testing protected tools..."
echo -n "Testing add(5, 3): "
curl -s -X GET "http://localhost:8000/mcp/tools/call/add" \
-H "Authorization: Bearer $ACCESS_TOKEN" \
-H "Content-Type: application/json" \
-d '{"a": 5, "b": 3}' | jq '.result'

echo -n "Testing subtract(10, 3): "
curl -s -X GET "http://localhost:8000/mcp/tools/call/subtract" \
-H "Authorization: Bearer $ACCESS_TOKEN" \
-H "Content-Type: application/json" \
-d '{"a": 10, "b": 3}' | jq '.result'

# 5. Get user info
echo -e "\n5️⃣  Getting authenticated user info..."
curl -s -X GET "http://localhost:8000/mcp/tools/call/get_user_info" \
-H "Authorization: Bearer $ACCESS_TOKEN" \
-H "Content-Type: application/json" \
-d '{}' | jq '.'

echo -e "\n✅ Test flow complete!"