"""
Unified MCP Server Proxy

A FastAPI application that hosts multiple MCP servers (custom and proxied) in a single deployment.

Features:
- Custom MCP servers with tools, resources, and prompts
- Proxy to local STDIO MCP servers (Python, Node, UVX, NPX)
- Proxy to remote HTTP/SSE MCP servers
- Proper lifespan management for all servers
- Comprehensive error handling and logging
- Health check and tools listing endpoints

Usage:
uvicorn main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
import logging
import inspect
from typing import Optional, Dict, List, Any
from fastapi import FastAPI
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class MCPServerConfig:
    """
    Configuration class for MCP servers supporting both custom tools and proxy modes.
    
    For Custom Tools Server:
    - Use tools, resources, prompts parameters
    
    For Proxy Server (STDIO):
    - Use command, args, env parameters
    - Examples: Python scripts, NPX packages, UVX packages
    
    For Proxy Server (HTTP/SSE):
    - Use proxy_url and headers parameters
    - Transport: Optional - specify 'http', 'sse', or 'streamable-http'
      If not specified, auto-detected from URL (ends with /sse = SSE, otherwise HTTP)
    """
    
    def __init__(
        self,
        name: str,
        path: str,
        # Custom server parameters
        tools: Optional[List] = None,
        resources: Optional[List] = None,
        prompts: Optional[List] = None,
        # Proxy parameters - STDIO
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        # Proxy parameters - HTTP/SSE
        proxy_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        transport: Optional[str] = None,  # Optional: 'http', 'sse', 'streamable-http' (auto-detected if None)
        # Common options
        enabled: bool = True
    ):
        self.name = name
        self.path = path
        
        # Custom server
        self.tools = tools or []
        self.resources = resources or []
        self.prompts = prompts or []
        
        # Proxy - STDIO
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.cwd = cwd
        
        # Proxy - HTTP/SSE
        self.proxy_url = proxy_url
        self.headers = headers or {}
        self.transport = transport  # Optional explicit transport
        
        # Options
        self.enabled = enabled
    
    def is_proxy(self) -> bool:
        """Check if this is a proxy configuration"""
        return bool(self.proxy_url or self.command)
    
    def is_custom(self) -> bool:
        """Check if this is a custom server configuration"""
        return bool(self.tools or self.resources or self.prompts)

# ============================================================================
# VALIDATION
# ============================================================================

def validate_server_config(config: MCPServerConfig) -> tuple[bool, Optional[str]]:
    """Validate server configuration before initialization"""
    
    # Check basic requirements
    if not config.name:
        return False, "Server name is required"
    
    if not config.path:
        return False, "Server path is required"
    
    # Check for conflicting configurations
    if config.is_custom() and config.is_proxy():
        return False, f"{config.name}: Cannot be both custom and proxy server"
    
    if not config.is_custom() and not config.is_proxy():
        return False, f"{config.name}: Must specify either tools/resources or proxy config"
    
    # Validate proxy configs
    if config.is_proxy():
        if config.command and config.proxy_url:
            return False, f"{config.name}: Cannot specify both command and proxy_url"
        
        if not config.command and not config.proxy_url:
            return False, f"{config.name}: Must specify either command or proxy_url for proxy"
        
        if config.command and not config.args:
            logger.warning(f"{config.name}: Command specified without args (may be intentional)")
    
    # Validate custom configs
    if config.is_custom():
        if not (config.tools or config.resources or config.prompts):
            return False, f"{config.name}: Custom server must have at least one tool, resource, or prompt"
    
    return True, None

# ============================================================================
# SERVER CREATION FUNCTIONS
# ============================================================================

def create_custom_server(config: MCPServerConfig) -> FastMCP:
    """Create a FastMCP server with custom tools, resources, and prompts"""
    mcp = FastMCP(name=config.name)
    
    # Register tools
    for tool_func in config.tools:
        mcp.tool()(tool_func)
        logger.debug(f" → Registered tool: {tool_func.__name__}")
    
    # Register resources
    for resource_func in config.resources:
        # Generate URI pattern from function signature
        sig = inspect.signature(resource_func)
        params_names = [
            p for p in sig.parameters.keys()
            if p not in ('self', 'cls', 'context')
        ]
        if params_names:
            params_str = '/'.join([f'{{{p}}}' for p in params_names])
            uri_pattern = f'/{resource_func.__name__}/{params_str}'
        else:
            uri_pattern = f'/{resource_func.__name__}/static'
        
        mcp.resource(uri_pattern)(resource_func)
        logger.debug(f" → Registered resource: {resource_func.__name__} ({uri_pattern})")
    
    # Register prompts
    for prompt_func in config.prompts:
        mcp.prompt()(prompt_func)
        logger.debug(f" → Registered prompt: {prompt_func.__name__}")
    
    return mcp


def create_proxy_server(config: MCPServerConfig) -> FastMCP:
    """
    Create a FastMCP proxy server using config-based approach.
    
    This is the CORRECT way to create proxies in FastMCP:
    1. Build a MCPConfig dictionary
    2. Pass it to FastMCP.as_proxy()
    3. Let FastMCP handle transport creation internally
    """
    
    try:
        # Build MCPConfig format
        if config.command:
            # STDIO proxy configuration
            logger.debug(f" → Creating STDIO proxy: {config.command} {' '.join(config.args)}")
            
            mcp_config = {
                "mcpServers": {
                    "default": {
                        "command": config.command,
                        "args": config.args,
                        "env": config.env,
                        "transport": "stdio"
                    }
                }
            }
            
            # Add working directory if specified
            if config.cwd:
                mcp_config["mcpServers"]["default"]["cwd"] = config.cwd
                logger.debug(f" → Working directory: {config.cwd}")
        
        elif config.proxy_url:
            # HTTP/SSE proxy configuration
            # Use explicit transport if provided, otherwise auto-detect from URL
            if config.transport:
                transport = config.transport
                logger.debug(f" → Creating {transport.upper()} proxy (explicit): {config.proxy_url}")
            else:
                transport = "sse" if config.proxy_url.endswith("/sse") else "http"
                logger.debug(f" → Creating {transport.upper()} proxy (auto-detected): {config.proxy_url}")
            
            mcp_config = {
                "mcpServers": {
                    "default": {
                        "url": config.proxy_url,
                        "transport": transport
                    }
                }
            }
            
            # Add headers if specified
            if config.headers:
                mcp_config["mcpServers"]["default"]["headers"] = config.headers
                logger.debug(f" → Headers configured: {list(config.headers.keys())}")
        else:
            raise ValueError("Proxy must have either command or proxy_url")
        
        # Create proxy using FastMCP.as_proxy with config
        # This is the pattern - FastMCP handles transport internally
        proxy = FastMCP.as_proxy(mcp_config, name=config.name)
        logger.debug(f" → Proxy created successfully")
        return proxy
    
    except FileNotFoundError as e:
        raise ValueError(f"Command not found: {config.command}") from e
    except ConnectionError as e:
        raise ConnectionError(f"Cannot connect to {config.proxy_url}") from e
    except Exception as e:
        raise Exception(f"Error creating proxy: {e}")

# ============================================================================
# SERVER INITIALIZATION
# ============================================================================

async def initialize_servers(app: FastAPI, server_configs: List[MCPServerConfig]):
    """Initialize all MCP servers asynchronously with comprehensive error handling"""
    
    successful = 0
    failed = 0
    skipped = 0
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Initializing {len(server_configs)} MCP server(s)...")
    logger.info(f"{'='*70}\n")
    
    for idx, config in enumerate(server_configs, 1):
        # Skip disabled servers
        if not config.enabled:
            logger.info(f"✓ [{idx}/{len(server_configs)}] Skipped (disabled): {config.name}")
            skipped += 1
            continue
        
        # Validate configuration
        is_valid, error_msg = validate_server_config(config)
        if not is_valid:
            logger.error(f"✗ [{idx}/{len(server_configs)}] Invalid config for {config.name}: {error_msg}")
            failed += 1
            continue
        
        logger.info(f"✓ [{idx}/{len(server_configs)}] Initializing: {config.name}")
        logger.info(f"   Type: {'PROXY' if config.is_proxy() else 'CUSTOM'}")
        logger.info(f"   Path: {config.path}")
        
        try:
            # Create server based on type
            if config.is_custom():
                mcp = create_custom_server(config)
                is_proxy = False
            else:
                mcp = create_proxy_server(config)
                is_proxy = True
            
            # Create HTTP app
            mcp_app = mcp.http_app(path='/mcp')
            
            # Store server info
            app.state.mcp_apps.append({
                'name': config.name,
                'path': config.path,
                'app': mcp_app,
                'mcp': mcp,
                'is_proxy': is_proxy,
                'config': config
            })
            
            # Mount to FastAPI
            app.mount(config.path, mcp_app)
            
            logger.info(f"   → Successfully initialized")
            logger.info(f"   → Endpoint: {config.path}/mcp\n")
            successful += 1
        
        except ImportError as e:
            logger.error(f"   ✗ Import error: {e}")
            logger.error(f"   → Check that all required packages are installed\n")
            failed += 1
        except ValueError as e:
            logger.error(f"   ✗ Configuration error: {e}\n")
            failed += 1
        except ConnectionError as e:
            logger.error(f"   ✗ Connection error: {e}")
            logger.error(f"   → Check that the proxy URL is accessible\n")
            failed += 1
        except Exception as e:
            logger.error(f"   ✗ Unexpected error: {e}")
            logger.exception("Full traceback:")
            failed += 1
    
    # Summary
    logger.info(f"{'='*70}")
    logger.info("Initialization Summary:")
    logger.info(f"   → Successful: {successful}")
    logger.info(f"   → Failed: {failed}")
    logger.info(f"   → Skipped: {skipped}")
    logger.info(f"   → Total: {len(server_configs)}")
    logger.info(f"{'='*70}\n")
    
    if successful == 0 and failed > 0:
        logger.error("⚠ NO servers initialized successfully")
        raise RuntimeError("Failed to initialize any MCP servers")

# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    """
    Manage all MCP server lifespans with proper error handling.
    
    Each MCP server's http_app() has its own lifespan that must be properly managed.
    This function coordinates all sub-lifespans, entering them in order and exiting
    in reverse order (LIFO) to handle dependencies correctly.
    """
    
    lifespan_contexts = []
    
    try:
        # Initialize servers first (before starting lifespans)
        await initialize_servers(app, MCP_SERVERS)
        
        # Startup phase - start all server lifespans
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting {len(app.state.mcp_apps)} MCP server(s)...")
        logger.info(f"{'='*70}\n")
        
        for idx, mcp_info in enumerate(app.state.mcp_apps, 1):
            mcp_app = mcp_info['app']
            name = mcp_info['name']
            path = mcp_info['path']
            is_proxy = mcp_info.get('is_proxy', False)
            
            try:
                # Enter the lifespan context
                lifespan_ctx = mcp_app.lifespan(mcp_app)
                await lifespan_ctx.__aenter__()
                lifespan_contexts.append((name, lifespan_ctx))
                
                server_type = "PROXY" if is_proxy else "CUSTOM"
                logger.info(f"✓ [{idx}/{len(app.state.mcp_apps)}] {server_type}: {name}")
                logger.info(f"   Path: {path}")
                logger.info(f"   Endpoint: {path}/mcp\n")
            
            except Exception as e:
                logger.error(f"✗ Failed to start {name}: {e}")
                # Continue with other servers even if one fails
                import traceback
                traceback.print_exc()
        
        logger.info(f"{'='*70}")
        logger.info(f"✓ {len(lifespan_contexts)}/{len(app.state.mcp_apps)} server(s) started successfully")
        logger.info(f"{'='*70}\n")
        
        # Application running
        yield
    
    finally:
        # Shutdown phase
        logger.info(f"\n{'='*70}")
        logger.info("Shutting down MCP servers...")
        logger.info(f"{'='*70}\n")
        
        # Exit all lifespans in REVERSE order (LIFO)
        for name, lifespan_ctx in reversed(lifespan_contexts):
            try:
                await lifespan_ctx.__aexit__(None, None, None)
                logger.info(f"✓ Shut down: {name}")
            except Exception as e:
                logger.error(f"✗ Error shutting down {name}: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✓ All servers shut down")
        logger.info(f"{'='*70}\n")

# ============================================================================
# DEFINE MCP SERVERS HERE
# ============================================================================

MCP_SERVERS = [
    # ========================================================================
    # Example 1: Custom Tools Server
    # ========================================================================
    # MCPServerConfig(
    #     name="custom-tools",
    #     path="/custom-tools",
    #     tools=[your_tool_function],
    #     resources=[your_resource_function],
    #     prompts=[your_prompt_function],
    #     enabled=True
    # ),
    
    # ========================================================================
    # Example 2: Proxy to Remote HTTP MCP Server (Public)
    # ========================================================================
    MCPServerConfig(
        name="context7",
        path="/context7",
        proxy_url="https://mcp.context7.com/mcp",
        transport="streamable-http",
        enabled=True
    ),
    
    # ========================================================================
    # Example 3: Proxy to Remote HTTP MCP Server (Authenticated)
    # ========================================================================
    # MCPServerConfig(
    #     name="authenticated-api",
    #     path="/authenticated-api",
    #     proxy_url="https://api.example.com/mcp",
    #     headers={
    #         "Authorization": "Bearer YOUR_API_KEY",
    #         "X-Custom-Header": "value"
    #     },
    #     enabled=False
    # ),
    
    # ========================================================================
    # Example 4: Proxy to Remote SSE MCP Server (Auto-detected)
    # ========================================================================
    # MCPServerConfig(
    #     name="sse-server",
    #     path="/sse-server",
    #     proxy_url="https://api.example.com/mcp/sse",  # URL ending in /sse
    #     headers={"Authorization": "Bearer token"},
    #     # Transport auto-detected as 'sse' from URL
    #     enabled=False
    # ),
    
    # ========================================================================
    # Example 4b: Proxy with Explicit Transport Override
    # ========================================================================
    # MCPServerConfig(
    #     name="streamable-http-server",
    #     path="/streamable",
    #     proxy_url="https://api.example.com/mcp",
    #     transport="streamable-http",  # Explicitly specify transport
    #     headers={"Authorization": "Bearer token"},
    #     enabled=False
    # ),
    
    # ========================================================================
    # Example 5: Proxy to Local STDIO MCP Server (Python)
    # ========================================================================
    # MCPServerConfig(
    #     name="local-python-server",
    #     path="/local-python",
    #     command="python",
    #     args=["/path/to/your/mcp_server.py"],
    #     env={
    #         "API_KEY": "your_api_key",
    #         "DEBUG": "true"
    #     },
    #     cwd="/path/to/working/directory",  # Optional
    #     enabled=False
    # ),
    
    # ========================================================================
    # Example 6: Proxy to NPM STDIO MCP Server (Node.js)
    # ========================================================================
    # MCPServerConfig(
    #     name="github",
    #     path="/github",
    #     command="npx",
    #     args=["-y", "@modelcontextprotocol/server-github"],
    #     env={"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"},
    #     enabled=False
    # ),
    
    # ========================================================================
    # Example 7: Proxy to UVX STDIO MCP Server
    # ========================================================================
    # MCPServerConfig(
    #     name="fetch-mcp",
    #     path="/fetch",
    #     command="uvx",
    #     args=["mcp-server-fetch"],
    #     enabled=False
    # ),
    
    # ========================================================================
    # Example 8: Proxy to FastMCP CLI Server
    # ========================================================================
    # MCPServerConfig(
    #     name="fastmcp-cli-server",
    #     path="/fastmcp-cli",
    #     command="fastmcp",
    #     args=["run", "/path/to/server.py"],
    #     env={"FASTMCP_LOG_LEVEL": "INFO"},
    #     enabled=False
    # ),
]

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Create FastAPI app with combined lifespan
app = FastAPI(
    lifespan=combined_lifespan,
    title="Unified MCP Server Proxy",
    description="FastAPI server hosting multiple MCP servers (custom + proxied)",
    version="2.0.0"
)

# Initialize state
app.state.mcp_apps = []

# Note: Server initialization is now handled in the combined_lifespan function
# This ensures proper async context and avoids the deprecated @app.on_event("startup")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Unified MCP Server Proxy",
        "version": "2.0.0",
        "description": "FastAPI server hosting multiple MCP servers",
        "total_servers": len(app.state.mcp_apps),
        "servers": [
            {
                "name": info['name'],
                "path": info['path'],
                "mcp_endpoint": f"{info['path']}/mcp",
                "type": "proxy" if info.get('is_proxy') else "custom"
            }
            for info in app.state.mcp_apps
        ]
    }

@app.get("/health")
async def health():
    """Health check endpoint with detailed status"""
    return {
        "status": "healthy",
        "total_servers": len(app.state.mcp_apps),
        "servers": [
            {
                "name": info['name'],
                "path": info['path'],
                "mcp_endpoint": f"{info['path']}/mcp",
                "type": "proxy" if info.get('is_proxy') else "custom"
            }
            for info in app.state.mcp_apps
        ]
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
