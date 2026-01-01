from contextlib import asynccontextmanager
import traceback
from fastmcp import FastMCP
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Optional, Dict, List

# Import your custom tools


class MCPServerConfig:
    """
    Configuration class for MCP servers supporting both custom tools and proxy modes.
    
    For Custom Tools Server:
        - Use tools, resources, prompts parameters
        
    For Proxy Server:
        - Set proxy_url to connect to existing MCP server
        - Supports multiple transport types: stdio, http (streamable-http), sse
        - Configure authentication via headers, env, or oauth
    """
    def __init__(
        self,
        name: str,
        path: str,
        # Custom server parameters
        tools: list = None,
        resources: list = None,
        prompts: list = None,
        # Proxy parameters
        proxy_url: Optional[str] = None,
        proxy_transport: Optional[str] = None,  # 'stdio', 'http', 'sse' (auto-detected if None)
        # Authentication & Headers (for HTTP/SSE proxies)
        headers: Optional[Dict[str, str]] = None,
        # Environment variables (for STDIO proxies)
        env: Optional[Dict[str, str]] = None,
        # Command arguments (for STDIO proxies)
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        # OAuth configuration (for authenticated HTTP/SSE proxies)
        oauth_client_id: Optional[str] = None,
        oauth_client_secret: Optional[str] = None,
        oauth_token_url: Optional[str] = None,
        # Additional options
        verify_ssl: bool = True,
        timeout: int = 60,
        enabled: bool = True
    ):
        self.name = name
        self.path = path
        
        # Custom server
        self.tools = tools or []
        self.resources = resources or []
        self.prompts = prompts or []
        
        # Proxy configuration
        self.proxy_url = proxy_url
        self.proxy_transport = proxy_transport
        self.headers = headers or {}
        self.env = env or {}
        self.command = command
        self.args = args or []
        self.oauth_client_id = oauth_client_id
        self.oauth_client_secret = oauth_client_secret
        self.oauth_token_url = oauth_token_url
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.enabled = enabled
    
    def is_proxy(self) -> bool:
        """Check if this is a proxy configuration"""
        return bool(self.proxy_url or self.command)


# ============================================================
# DEFINE MCP SERVERS HERE
# ============================================================

MCP_SERVERS = [
    # ============================================================
    # Example 1: Custom Tools Server
    # ============================================================
    # MCPServerConfig(
    #     name="new-service",
    #     path="/new-service",
    #     tools=[your_tool_function],
    #     resources=[your_resource_function],
    #     prompts=[your_prompt_function],
    #     enabled=True
    # ),
    
    # ============================================================
    # Example 2: Proxy to Context7 (HTTP Streamable - Public API)
    # ============================================================
    MCPServerConfig(
        name="context7",
        path="/context7",
        proxy_url="https://context7.com/mcp",
        proxy_transport="streamable-http",  # or 'http'
        # headers={"Authorization": "Bearer YOUR_API_KEY"},
        enabled=True
    ),
    
    # ============================================================
    # Example 3: Proxy to Authenticated HTTP MCP Server
    # ============================================================
    # MCPServerConfig(
    #     name="authenticated-api",
    #     path="/authenticated-api",
    #     proxy_url="https://api.example.com/mcp",
    #     proxy_transport="http",
    #     headers={
    #         "Authorization": "Bearer YOUR_API_KEY",
    #         "X-Custom-Header": "value"
    #     },
    #     verify_ssl=True,
    #     timeout=60,
    #     enabled=False
    # ),
    
    # ============================================================
    # Example 4: Proxy to SSE MCP Server with OAuth
    # ============================================================
    # MCPServerConfig(
    #     name="sse-with-oauth",
    #     path="/sse-oauth",
    #     proxy_url="https://api.example.com/sse",
    #     proxy_transport="sse",
    #     oauth_client_id="your_client_id",
    #     oauth_client_secret="your_client_secret",
    #     oauth_token_url="https://auth.example.com/oauth/token",
    #     enabled=False
    # ),
    
    # ============================================================
    # Example 5: Proxy to Local STDIO MCP Server (Python)
    # ============================================================
    # MCPServerConfig(
    #     name="local-stdio-python",
    #     path="/local-stdio-python",
    #     command="python",
    #     args=["/path/to/your/mcp_server.py"],
    #     env={
    #         "API_KEY": "your_api_key",
    #         "DEBUG": "true",
    #         "DATABASE_URL": "postgresql://localhost/mydb"
    #     },
    #     enabled=False
    # ),
    
    # ============================================================
    # Example 6: Proxy to STDIO MCP Server via FastMCP CLI
    # ============================================================
    # MCPServerConfig(
    #     name="fastmcp-cli-server",
    #     path="/fastmcp-cli",
    #     command="fastmcp",
    #     args=["run", "/path/to/server.py"],
    #     env={"FASTMCP_LOG_LEVEL": "INFO"},
    #     enabled=False
    # ),
    
    # ============================================================
    # Example 7: Proxy to NPM STDIO MCP Server (Node.js)
    # ============================================================
    # MCPServerConfig(
    #     name="github-mcp",
    #     path="/github",
    #     command="npx",
    #     args=["-y", "@modelcontextprotocol/server-github"],
    #     env={"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"},
    #     enabled=False
    # ),
    
    # ============================================================
    # Example 8: Proxy to UVX STDIO MCP Server
    # ============================================================
    # MCPServerConfig(
    #     name="fetch-mcp",
    #     path="/fetch",
    #     command="uvx",
    #     args=["mcp-server-fetch"],
    #     enabled=False
    # ),
    
    # ============================================================
    # Example 9: Mixed - Custom Server with Tools + Resources + Prompts
    # ============================================================
    # MCPServerConfig(
    #     name="custom-service",
    #     path="/custom",
    #     tools=[your_tool_function],
    #     resources=[your_resource_function],
    #     prompts=[your_prompt_function],
    #     enabled=False
    # ),
]


async def create_proxy_server(config: MCPServerConfig) -> FastMCP:
    """
    Create a FastMCP proxy server from configuration.
    
    Supports:
    - HTTP/SSE transports with optional authentication
    - STDIO transports with command, args, and environment variables
    - OAuth authentication for HTTP/SSE
    """
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport, SSETransport, StreamableHTTPTransport
    
    # Determine transport type and create appropriate client
    transport = None
    
    if config.command:
        # STDIO Transport
        print(f"  → Creating STDIO proxy: {config.command} {' '.join(config.args)}")
        transport = StdioTransport(
            command=config.command,
            args=config.args,
            env=config.env if config.env else None
        )
    
    elif config.proxy_url:
        # Prepare headers
        headers = dict(config.headers) if config.headers else {}
        
        # Handle OAuth if configured
        if config.oauth_client_id and config.oauth_client_secret and config.oauth_token_url:
            print(f"  → Configuring OAuth authentication")
            # OAuth token will be fetched by the transport
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config.oauth_token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": config.oauth_client_id,
                        "client_secret": config.oauth_client_secret
                    }
                ) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        access_token = token_data.get("access_token")
                        headers["Authorization"] = f"Bearer {access_token}"
                        print(f"  → OAuth token obtained")
                    else:
                        print(f"  → OAuth token fetch failed: {response.status}")
        
        # Auto-detect transport type if not specified
        if not config.proxy_transport:
            if "/sse" in config.proxy_url or "/events" in config.proxy_url:
                config.proxy_transport = "sse"
            else:
                config.proxy_transport = "http"
        
        # Create appropriate transport
        if config.proxy_transport.lower() in ["sse", "server-sent-events"]:
            print(f"  → Creating SSE proxy: {config.proxy_url}")
            transport = SSETransport(
                url=config.proxy_url,
                headers=headers if headers else None
            )
        else:  # http, streamable-http, streamablehttp
            print(f"  → Creating HTTP Streamable proxy: {config.proxy_url}")
            transport = StreamableHTTPTransport(
                url=config.proxy_url,
                headers=headers if headers else None
            )
    
    if not transport:
        raise ValueError(f"Could not create transport for {config.name}")
    
    # Create client with the transport
    client = Client(transport)
    
    # Create proxy server using FastMCP.as_proxy()
    proxy_mcp = FastMCP.as_proxy(
        client=client,
        name=config.name
    )
    
    print(f"  → Proxy server created successfully")
    return proxy_mcp


async def create_custom_server(config: MCPServerConfig) -> FastMCP:
    """Create a FastMCP server with custom tools, resources, and prompts"""
    mcp = FastMCP(name=config.name)
    
    # Register tools
    for tool_func in config.tools:
        mcp.tool()(tool_func)
        print(f"  → Registered tool: {tool_func.__name__}")
    
    # Register resources
    for resource_func in config.resources:
        import inspect
        sig = inspect.signature(resource_func)
        
        # Filter out special parameters
        param_names = [
            p for p in sig.parameters.keys() 
            if p not in ('self', 'ctx', 'context')
        ]
        
        # Create URI pattern
        if param_names:
            params_str = '/'.join([f'{{{p}}}' for p in param_names])
            uri_pattern = f"{resource_func.__name__}://{params_str}"
        else:
            uri_pattern = f"{resource_func.__name__}://static"
        
        mcp.resource(uri_pattern)(resource_func)
        print(f"  → Registered resource: {resource_func.__name__} ({uri_pattern})")
    
    # Register prompts
    for prompt_func in config.prompts:
        mcp.prompt()(prompt_func)
        print(f"  → Registered prompt: {prompt_func.__name__}")
    
    return mcp


@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    """Manage MCP server lifespans"""
    lifespan_contexts = []

    try:
        print(f"\nStarting up {len(app.state.mcp_apps)} MCP servers...\n")
        
        for mcp_app_info in app.state.mcp_apps:
            mcp_app = mcp_app_info['app']
            name = mcp_app_info['name']
            path = mcp_app_info['path']
            is_proxy = mcp_app_info.get('is_proxy', False)
            
            lifespan_ctx = mcp_app.lifespan(app)
            await lifespan_ctx.__aenter__()
            lifespan_contexts.append((name, lifespan_ctx))
            
            server_type = "PROXY" if is_proxy else "CUSTOM"
            print(f"✓ Started [{server_type}] {name} at {path}")

        print("\n✓ All MCP servers started successfully\n")
        yield

        print("\nShutting down MCP servers...")

        for name, lifespan_ctx in reversed(lifespan_contexts):
            try:
                await lifespan_ctx.__aexit__(None, None, None)
                print(f"✓ Shut down: {name}")
            except Exception as e:
                print(f"✗ Error shutting down {name}: {e}")

        print("✓ All MCP servers shut down\n")

    except Exception as e:
        print(f"\n✗ Error during startup: {e}")
        print(traceback.format_exc())
        raise


app = FastAPI(
    lifespan=combined_lifespan,
    title="Multi-MCP Server with Proxy Support",
    description="FastAPI server hosting multiple MCP servers (custom + proxied)",
)

app.state.mcp_apps = []


# Build MCP servers
async def initialize_servers():
    """Initialize all MCP servers (must be called in async context)"""
    for server_config in MCP_SERVERS:
        if not server_config.enabled:
            print(f"⊗ Skipped (disabled): {server_config.name}")
            continue

        print(f"\nInitializing: {server_config.name}")
        print(f"  Path: {server_config.path}")
        
        try:
            if server_config.is_proxy():
                # Create proxy server
                mcp = await create_proxy_server(server_config)
                is_proxy = True
            else:
                # Create custom server
                mcp = await create_custom_server(server_config)
                is_proxy = False
            
            # Create and mount the MCP app
            mcp_app = mcp.http_app(path='/mcp')
            app.state.mcp_apps.append({
                'name': server_config.name,
                'path': server_config.path,
                'app': mcp_app,
                'mcp': mcp,
                'is_proxy': is_proxy,
                'config': server_config
            })
            app.mount(server_config.path, mcp_app)
            
        except Exception as e:
            print(f"✗ Failed to initialize {server_config.name}: {e}")
            traceback.print_exc()


@app.on_event("startup")
async def startup_event():
    """Initialize servers on startup"""
    await initialize_servers()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "total_servers": len(app.state.mcp_apps),
        "servers": [
            {
                "name": info['name'],
                "path": info['path'],
                "endpoint": f"{info['path']}/mcp",
                "type": "proxy" if info.get('is_proxy') else "custom"
            }
            for info in app.state.mcp_apps
        ]
    }


@app.get("/tools")
async def list_tools(request: Request):
    """List all available MCP tools across all servers"""
    try:
        all_tools_info = []
        total_tools = 0

        for mcp_info in app.state.mcp_apps:
            server_name = mcp_info['name']
            server_path = mcp_info['path']
            mcp_instance = mcp_info['mcp']
            is_proxy = mcp_info.get('is_proxy', False)

            # Use FastMCP's internal _list_tools method
            tools_result = await mcp_instance._list_tools()

            # Handle different return types
            if hasattr(tools_result, 'tools'):
                tools_list = tools_result.tools
            elif isinstance(tools_result, list):
                tools_list = tools_result
            else:
                tools_list = []

            # Convert to simplified JSON format
            for tool in tools_list:
                tool_info = {
                    "server": server_name,
                    "server_path": server_path,
                    "server_type": "proxy" if is_proxy else "custom",
                    "mcp_endpoint": f"{server_path}/mcp",
                    "name": tool.name if hasattr(tool, 'name') else str(tool),
                    "description": tool.description if hasattr(tool, 'description') else "No description available",
                }

                # Add input schema if available
                if hasattr(tool, 'inputSchema'):
                    tool_info["parameters"] = tool.inputSchema

                all_tools_info.append(tool_info)
                total_tools += 1

        return JSONResponse({
            "status": "success",
            "total_servers": len(app.state.mcp_apps),
            "total_tools": total_tools,
            "tools": all_tools_info
        })

    except Exception as e:
        return JSONResponse({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "total_tools": 0,
            "tools": []
        }, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
