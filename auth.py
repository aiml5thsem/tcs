import os
import base64
import logging
import json
from typing import Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.status import HTTP_401_UNAUTHORIZED
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

class SecretsManager:
    """
    Unified secrets manager that supports both:
    1. Dictionary-based secrets (passed directly)
    2. File-based secrets (from filesystem like Whisper)
    3. Default values when secrets are not found
    """

    def __init__(self, secrets_dict=None, secrets_folder=None):
        """
        Initialize secrets manager with either a dictionary or folder path.

        Args:
            secrets_dict: Optional dictionary containing secrets as key-value pairs
            secrets_folder: Optional path to folder containing secret files (default: from APP_SECRETS_PATH env var)
        """
        self.secrets_dict = secrets_dict or {}
        self.secrets_folder = secrets_folder or os.getenv('APP_SECRETS_PATH', '/app/secrets')
        self.use_dict = bool(secrets_dict)

        if self.use_dict:
            logger.info(f"SecretsManager initialized with dictionary containing {len(self.secrets_dict)} secrets")
        else:
            logger.info(f"SecretsManager initialized with folder: {self.secrets_folder}")

    def get_secret(self, secret_key, default=None):
        """
        Get secret value by key.
        First tries dictionary, then falls back to file-based lookup, then returns default.

        Args:
            secret_key: The secret key to retrieve
            default: Default value to return if secret is not found (default: None)

        Returns:
            Secret value as string, or default value if not found
        """
        # Try dictionary first if available
        if self.use_dict and secret_key in self.secrets_dict:
            logger.debug(f"Retrieved secret '{secret_key}' from dictionary")
            return self.secrets_dict[secret_key]

        # Fall back to file-based lookup
        try:
            secret_path = os.path.join(self.secrets_folder, secret_key)
            with open(secret_path, 'r') as f:
                secret_content = f.read().strip()
            logger.info(f"Successfully retrieved secret from: {secret_path}")
            return secret_content
        except FileNotFoundError:
            if default is not None:
                logger.warning(f"Secret '{secret_key}' not found, using default value")
                return default
            logger.warning(f"Secret '{secret_key}' not found in dictionary")
            return None
        except Exception as e:
            if default is not None:
                logger.error(f"Failed to retrieve secret '{secret_key}': {e}, using default value")
                return default
            logger.error(f"Failed to retrieve secret '{secret_key}': {e}")
            return None

def create_secrets_manager(secrets_dict=None):
    """
    Factory function to create a SecretsManager instance.

    Args:
        secrets_dict: Optional dictionary of secrets. If None, uses file-based secrets.

    Returns:
        SecretsManager instance
    """
    return SecretsManager(secrets_dict=secrets_dict)

class BasicAuthMiddleware(BaseHTTPMiddleware):
    """
    HTTP-level Basic Authentication middleware for FastMCP servers using Starlette.
    Supports dictionary/file-based secrets and defaults. Replaces MCP middleware for 2.13.0.2 compatibility.
    """

    def __init__(self, app: ASGIApp, secrets_dict=None, excluded_paths=None,
                 default_username='username', default_password='password'):
        """
        Initialize HTTP Basic Auth middleware.

        Args:
            app: The ASGI app to wrap.
            secrets_dict: Optional dict with 'MCP_BASIC_AUTH_USERNAME' and 'MCP_BASIC_AUTH_PASSWORD'.
            excluded_paths: List of paths to exclude from auth.
            default_username: Default username if secret not found.
            default_password: Default password if secret not found.
        """
        super().__init__(app)
        self.secrets_manager = create_secrets_manager(secrets_dict)
        self.excluded_paths = excluded_paths or ["/health", "/docs", "/redoc", "/openapi.json"]
        self.default_username = default_username
        self.default_password = default_password
        self.www_auth = 'Basic realm="MCP Server"'

    async def dispatch(self, request: Request, call_next):
        """Intercept HTTP requests and validate Basic/Bearer auth."""
        path = request.url.path

        if path in self.excluded_paths:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")

        if not auth_header:
            logger.error("Missing Authorization header")
            raise HTTPException(
                HTTP_401_UNAUTHORIZED,
                detail="Authentication Failed: Missing Authorization header",
                headers={"WWW-Authenticate": self.www_auth}
            )

        if not (auth_header.startswith("Basic ") or auth_header.startswith("Bearer ")):
            logger.error(f"Invalid Authorization header format: {auth_header[:20]}")
            raise HTTPException(
                HTTP_401_UNAUTHORIZED,
                detail="Authentication Failed: Must use Basic or Bearer authentication",
                headers={"WWW-Authenticate": self.www_auth}
            )

        try:
            encoded_credentials = auth_header.split(" ")[1]
            decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
            username, password = decoded_credentials.split(":", 1)

            expected_username = self.secrets_manager.get_secret(
                'MCP_BASIC_AUTH_USERNAME', default=self.default_username
            )
            expected_password = self.secrets_manager.get_secret(
                'MCP_BASIC_AUTH_PASSWORD', default=self.default_password
            )

            if not expected_username or not expected_password:
                logger.error("Basic auth credentials not configured")
                raise HTTPException(
                    HTTP_401_UNAUTHORIZED,
                    detail="Authentication not configured",
                    headers={"WWW-Authenticate": self.www_auth}
                )

            if username != expected_username or password != expected_password:
                logger.error("Invalid credentials provided")
                raise HTTPException(
                    HTTP_401_UNAUTHORIZED,
                    detail="Authentication Failed",
                    headers={"WWW-Authenticate": self.www_auth}
                )

            logger.debug("Authentication Successful")
            return await call_next(request)

        except HTTPException:
            raise
        except Exception as ex:
            logger.error(f"Authentication error: {ex}")
            raise HTTPException(
                HTTP_401_UNAUTHORIZED,
                detail="Authentication Failed",
                headers={"WWW-Authenticate": self.www_auth}
            )