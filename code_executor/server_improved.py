#!/usr/bin/env python3
"""
🚀 FastMCP Code Execution Server - ENHANCED VERSION
===================================================

Improvements:
- Better tool descriptions with examples
- More informative return formats
- Additional utility tools
- Clearer parameter documentation
"""

from contextlib import asynccontextmanager, AsyncExitStack
import asyncio
import json
import sys
import os
import types
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import traceback

from fastmcp import FastMCP
from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from mcp.types import Tool

# ============================================================================
# LOGGING & CONFIG
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

EXECUTION_TIMEOUT = int(os.getenv("MCP_EXEC_TIMEOUT", "30"))
MAX_EXECUTION_TIMEOUT = int(os.getenv("MCP_MAX_EXEC_TIMEOUT", "120"))
CONFIG_FILE = os.getenv("MCP_CONFIG_FILE", str(Path.cwd() / "mcp_settings.json"))

# ============================================================================
# API KEY HELPER & CONFIG LOADING
# ============================================================================

def resolve_api_key_helper(value: str) -> str:
    """Resolve Claude's apiKeyHelper syntax."""
    pattern = r"\$\{apiKeyHelper\('([^']+)',\s*'([^']+)'\)\}"
    
    def replace_helper(match):
        service = match.group(1)
        key_name = match.group(2)
        env_key = f"{service.upper()}_{key_name}".replace("-", "_")
        if env_key in os.environ:
            return os.environ[env_key]
        if key_name in os.environ:
            return os.environ[key_name]
        return ""
    
    return re.sub(pattern, replace_helper, value)


def resolve_env_vars(env_dict: Dict[str, str]) -> Dict[str, str]:
    """Resolve apiKeyHelper and env vars."""
    resolved = {}
    for key, value in env_dict.items():
        if isinstance(value, str):
            value = resolve_api_key_helper(value)
            value = os.path.expandvars(value)
        resolved[key] = value
    return resolved


def load_mcp_config() -> Dict[str, Any]:
    """Load mcp_settings.json."""
    config_path = Path(CONFIG_FILE)
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            logger.info(f"✅ Loaded config from {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load {config_path}: {e}")
            return {"mcpServers": {}}
    else:
        logger.error(f"❌ Config not found: {config_path}")
        return {"mcpServers": {}}

# ============================================================================
# MCP CLIENT POOL
# ============================================================================

class MCPClientPool:
    """Manages connections to all MCP servers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mcp_servers = config.get("mcpServers", {})
        self.clients: Dict[str, Client] = {}
        self.stack: Optional[AsyncExitStack] = None
        self.tools: Dict[str, List[Tool]] = {}
    
    async def connect_all(self):
        """Connect to all MCP servers."""
        self.stack = AsyncExitStack()
        await self.stack.__aenter__()
        
        logger.info("\n📡 Connecting to MCP servers...\n")
        
        for server_name, config in self.mcp_servers.items():
            try:
                await self._connect_server(server_name, config)
            except Exception as e:
                logger.error(f"❌ Failed to connect {server_name}: {e}")
                traceback.print_exc()
    
    async def _connect_server(self, name: str, config: Dict[str, Any]):
        """Connect to a single MCP server."""
        transport_type = config.get("transport", "stdio")
        
        try:
            if transport_type == "stdio":
                command = config.get("command")
                args = config.get("args", [])
                env = resolve_env_vars(config.get("env", {}))
                
                resolved_command = command
                if not Path(command).is_absolute() and command not in ["python", "python3", "uvx", "npx"]:
                    resolved_command = str(Path(command).expanduser().resolve())
                
                resolved_args = [
                    str(Path(arg).expanduser().resolve()) if arg.startswith(("./", "../")) else arg
                    for arg in args
                ]
                
                logger.info(f"  🔗 {name}")
                logger.info(f"     Command: {resolved_command} {' '.join(resolved_args)}")
                
                transport = StdioTransport(
                    command=resolved_command,
                    args=resolved_args,
                    env=env if env else None
                )
                
                client = Client(transport)
                await self.stack.enter_async_context(client)
                
                tools_response = await client.list_tools()
                tools = tools_response.tools if hasattr(tools_response, 'tools') else tools_response
                
                self.clients[name] = client
                self.tools[name] = tools
                
                if tools:
                    logger.info(f"     ✅ Found {len(tools)} tools:")
                    for tool in tools[:3]:
                        logger.info(f"        • {tool.name}")
                    if len(tools) > 3:
                        logger.info(f"        • ... and {len(tools) - 3} more")
                else:
                    logger.info("     ✅ Connected (no tools found)")
            
            elif transport_type in ["http", "sse", "streamable-http"]:
                url = config.get("url")
                headers = config.get("headers", {})
                
                logger.info(f"  🔗 {name} ({transport_type})")
                logger.info(f"     URL: {url}")
                
                transport = StreamableHttpTransport(url=url, headers=headers)
                client = Client(transport)
                
                await self.stack.enter_async_context(client)
                
                tools_response = await client.list_tools()
                tools = tools_response.tools if hasattr(tools_response, 'tools') else tools_response
                
                self.clients[name] = client
                self.tools[name] = tools
                
                logger.info(f"     ✅ Connected - {len(tools)} tools")
        
        except Exception as e:
            logger.error(f"   Error: {e}")
            traceback.print_exc()
    
    async def disconnect_all(self):
        """Disconnect from all servers."""
        if self.stack:
            await self.stack.__aexit__(None, None, None)
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict) -> Any:
        """Call a tool on a server."""
        try:
            if server_name not in self.clients:
                return {"error": f"Server '{server_name}' not found"}
            
            client = self.clients[server_name]
            
            result = await client.call_tool(tool_name, arguments, raise_on_error=False)
            
            if result.is_error:
                error_msg = result.content[0].text if result.content else "Unknown error"
                return {"error": error_msg}
            else:
                if hasattr(result, 'data') and result.data is not None:
                    return result.data
                elif result.content:
                    if isinstance(result.content, list) and len(result.content) > 0:
                        if hasattr(result.content[0], 'text'):
                            return result.content[0].text
                        else:
                            return result.content[0]
                    return result.content
                else:
                    return result.structured_content if hasattr(result, 'structured_content') else str(result)
        
        except Exception as e:
            logger.error(f"Tool call error: {e}")
            traceback.print_exc()
            return {"error": f"Tool call failed: {str(e)}"}
    
    def get_discovered_servers(self, include_description: bool = False) -> Any:
        """Get list of discovered servers."""
        if include_description:
            servers: Dict[str, str] = {}
            for server_name in self.clients.keys():
                config = self.mcp_servers.get(server_name, {})
                servers[server_name] = config.get("description", "")
            return servers
        else:
            return list(self.clients.keys())
    
    def _format_tools(self, tools: List[Tool], detail_level: str) -> List[Any]:
        """Format tools based on detail level."""
        formatted = []
        
        for tool in tools:
            if detail_level == "name":
                formatted.append(tool.name)
            elif detail_level == "description":
                formatted.append({
                    "name": tool.name,
                    "description": tool.description or ""
                })
            elif detail_level == "full":
                item = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema or {}
                }
                if hasattr(tool, 'meta') and tool.meta:
                    fastmcp_meta = tool.meta.get('_fastmcp', {})
                    if fastmcp_meta.get('tags'):
                        item["tags"] = fastmcp_meta.get('tags', [])
                formatted.append(item)
        
        return formatted
    
    def get_pretty_tree(self, server_name: Optional[str] = None) -> str:
        """Get pretty-printed tree."""
        output = ""
        
        if server_name:
            if server_name not in self.tools:
                return f"❌ Server '{server_name}' not found"
            
            output += f"📁 {server_name}/\n"
            for tool in self.tools[server_name]:
                output += f"   ├── {tool.name}\n"
                output += f"   │   └── {tool.description or 'No description'}\n"
        else:
            servers_list = list(self.tools.keys())
            for i, sname in enumerate(servers_list):
                is_last = (i == len(servers_list) - 1)
                prefix = "└── " if is_last else "├── "
                output += f"{prefix}📁 {sname}/\n"
                
                tools = self.tools[sname]
                for j, tool in enumerate(tools):
                    is_last_tool = (j == len(tools) - 1)
                    sub_prefix = "    " if is_last else "│   "
                    tool_prefix = "└── " if is_last_tool else "├── "
                    
                    output += f"{sub_prefix}{tool_prefix}{tool.name}\n"
                    if tool.description:
                        desc_prefix = "    " if is_last_tool else "│   "
                        output += f"{sub_prefix}{desc_prefix}   └── {tool.description}\n"
        
        return output
    
    def get_structured_summary(self) -> Dict[str, Any]:
        """Get structured summary of all capabilities."""
        servers_info = {}
        
        for server_name, tools in self.tools.items():
            config = self.mcp_servers.get(server_name, {})
            servers_info[server_name] = {
                "description": config.get("description", ""),
                "transport": config.get("transport", "stdio"),
                "tool_count": len(tools),
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description or ""
                    }
                    for tool in tools
                ]
            }
        
        return {
            "total_servers": len(self.clients),
            "total_tools": sum(len(tools) for tools in self.tools.values()),
            "servers": servers_info
        }

# ============================================================================
# GLOBAL POOL
# ============================================================================

global_pool: Optional[MCPClientPool] = None

# ============================================================================
# CODE EXECUTION
# ============================================================================

def make_tool_func(server_name: str, tool_name: str):
    """Factory function to create tool functions."""
    async def tool_func(**kwargs):
        result = await global_pool.call_tool(server_name, tool_name, kwargs)
        return result
    
    tool_func.__name__ = tool_name.replace("-", "_").replace(".", "_")
    return tool_func


async def execute_code_with_mcp(code: str, timeout: Optional[int] = None) -> str:
    """Execute Python code with access to all MCP servers."""
    
    if not global_pool:
        return "[ERROR] MCP pool not initialized"
    
    timeout = min(timeout or EXECUTION_TIMEOUT, MAX_EXECUTION_TIMEOUT)
    code = code.replace("\\n", "\n").replace("\\t", "\t")
    
    namespace: Dict[str, Any] = {
        'asyncio': asyncio,
        'sys': sys,
        'types': types,
        '__name__': '__main__',
        'global_pool': global_pool,
    }
    
    # Build mcps module
    mcps_pkg = types.ModuleType("mcps")
    mcps_pkg.__path__ = []
    servers_module = types.ModuleType("mcps.servers")
    servers_module.__path__ = []
    
    sys.modules["mcps"] = mcps_pkg
    sys.modules["mcps.servers"] = servers_module
    mcps_pkg.servers = servers_module
    namespace["mcps"] = mcps_pkg
    
    # Create server modules
    for server_name, tools in global_pool.tools.items():
        server_module = types.ModuleType(f"mcps.servers.{server_name}")
        server_module.__path__ = []
        
        for tool in tools:
            tool_name = tool.name
            safe_tool_name = tool_name.replace("-", "_").replace(".", "_")
            tool_func = make_tool_func(server_name, tool_name)
            tool_func.__name__ = safe_tool_name
            tool_func.__doc__ = tool.description or ""
            setattr(server_module, safe_tool_name, tool_func)
        
        sys.modules[f"mcps.servers.{server_name}"] = server_module
        setattr(servers_module, server_name, server_module)
    
    def list_tools(server_name: Optional[str] = None):
        if server_name:
            if server_name not in global_pool.tools:
                return []
            return [t.name for t in global_pool.tools[server_name]]
        return list(global_pool.tools.keys())
    
    namespace["list_tools"] = list_tools
    
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        indented_code = "\n".join("    " + line for line in code.split("\n"))
        async_wrapper = f"async def __user_code__():\n{indented_code}"
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            try:
                compiled_code = compile(async_wrapper, '<code_execution>', 'exec')
            except SyntaxError as e:
                return f"[SYNTAX ERROR] Line {e.lineno}: {e.msg}"
            
            exec(compiled_code, namespace)
            
            await asyncio.wait_for(
                namespace["__user_code__"](),
                timeout=timeout
            )
        
        result = stdout_capture.getvalue()
        if not result:
            result = stderr_capture.getvalue()
        return result if result else "[No output]"
    
    except asyncio.TimeoutError:
        return f"[TIMEOUT] Execution exceeded {timeout}s"
    
    except Exception as e:
        error_output = stderr_capture.getvalue()
        if error_output:
            return f"[ERROR] {type(e).__name__}: {e}\n\n{error_output}"
        else:
            tb = traceback.format_exc()
            return f"[ERROR] {type(e).__name__}: {e}\n\n{tb}"
    
    finally:
        keys_to_remove = [k for k in sys.modules.keys() if k.startswith("mcps.")]
        for key in keys_to_remove:
            del sys.modules[key]

# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for FastMCP."""
    global global_pool
    
    logger.info("=" * 60)
    logger.info("🚀 MCP Code Execution Server")
    logger.info("=" * 60)
    logger.info(f"📂 Config: {Path(CONFIG_FILE).absolute()}\n")
    
    config = load_mcp_config()
    
    # Initialize pool in this event loop
    global_pool = MCPClientPool(config)
    await global_pool.connect_all()
    
    total_tools = sum(len(tools) for tools in global_pool.tools.values())
    logger.info(f"\n✅ Ready!")
    logger.info(f"   Servers: {len(global_pool.clients)}")
    logger.info(f"   Tools: {total_tools}\n")
    
    yield
    
    # Cleanup on shutdown
    logger.info("\n🔌 Shutting down...")
    await global_pool.disconnect_all()

# ============================================================================
# FASTMCP SERVER - IMPROVED TOOL DEFINITIONS
# ============================================================================

mcp = FastMCP(name="mcp-code-execution", lifespan=lifespan)


@mcp.tool()
async def execute_code(code: str, timeout_seconds: Optional[int] = None) -> str:
    """
    Execute Python code with full access to all connected MCP servers and their tools.
    
    The code runs in an async environment where you can import and call tools from any
    connected MCP server using the mcps.servers namespace.
    
    Args:
        code: Python code to execute (will be wrapped in async function automatically)
        timeout_seconds: Maximum execution time in seconds (default: 30s, max: 120s)
    
    Returns:
        String containing stdout/stderr output from code execution, or error message
    
    Usage Examples:
        
        # Example 1: Call a single tool
        from mcps.servers import filesystem
        result = await filesystem.read_file(path="/tmp/test.txt")
        print(result)
        
        # Example 2: Chain multiple tools
        from mcps.servers import weather, filesystem
        weather_data = await weather.get_forecast(city="London")
        await filesystem.write_file(path="/tmp/weather.txt", content=weather_data)
        print("Weather data saved!")
        
        # Example 3: Process data from multiple servers
        from mcps.servers import database, email
        users = await database.query(sql="SELECT * FROM users")
        for user in users:
            await email.send(to=user['email'], subject="Update", body="Hello!")
        print(f"Sent {len(users)} emails")
        
        # Example 4: List available tools
        tools = list_tools()  # Returns list of server names
        print(f"Available servers: {tools}")
        
        tools = list_tools("filesystem")  # Returns tools for specific server
        print(f"Filesystem tools: {tools}")
    
    Notes:
        - Code is automatically wrapped in an async function
        - All MCP tool calls must use 'await'
        - Standard library imports work normally
        - The helper function list_tools(server_name?) is available
        - Syntax errors are caught and reported before execution
    """
    return await execute_code_with_mcp(code, timeout_seconds)


@mcp.tool()
async def discovered_servers(include_description: bool = False) -> str:
    """
    Get list of all successfully connected MCP servers.
    
    Args:
        include_description: If True, returns dict with server descriptions;
                           if False, returns simple list of server names
    
    Returns:
        JSON string containing either:
        - List of server names: ["server1", "server2", ...]
        - Dict with descriptions: {"server1": "description...", "server2": "..."}
    
    Example Response (include_description=False):
        ["filesystem", "weather", "database"]
    
    Example Response (include_description=True):
        {
          "filesystem": "Access local filesystem operations",
          "weather": "Get weather data from APIs",
          "database": "Query PostgreSQL databases"
        }
    """
    if not global_pool:
        return json.dumps({"error": "MCP pool not initialized"})
    
    data = global_pool.get_discovered_servers(include_description)
    return json.dumps(data, indent=2)


@mcp.tool()
async def query_tool_docs(
    server: str, 
    tool: Optional[str] = None, 
    detail_level: str = "description"
) -> str:
    """
    Query documentation for tools on a specific MCP server.
    
    Args:
        server: Name of the MCP server to query
        tool: Optional specific tool name (if omitted, returns all tools for server)
        detail_level: Level of detail to return:
                     - "name": Just tool names
                     - "description": Names + descriptions (default)
                     - "full": Names + descriptions + complete input schemas
    
    Returns:
        JSON string with tool documentation in requested format
    
    Example Response (detail_level="description"):
        [
          {
            "name": "read_file",
            "description": "Read contents of a file from the filesystem"
          },
          {
            "name": "write_file",
            "description": "Write content to a file"
          }
        ]
    
    Example Response (detail_level="full"):
        [
          {
            "name": "read_file",
            "description": "Read contents of a file",
            "inputSchema": {
              "type": "object",
              "properties": {
                "path": {"type": "string", "description": "File path"}
              },
              "required": ["path"]
            }
          }
        ]
    """
    if not global_pool:
        return json.dumps({"error": "MCP pool not initialized"})
    
    if server not in global_pool.tools:
        return json.dumps({"error": f"Server '{server}' not found"})
    
    all_tools = global_pool.tools[server]
    
    if tool:
        filtered_tools = [t for t in all_tools if t.name == tool or t.name.replace("-", "_") == tool]
        if not filtered_tools:
            return json.dumps({"error": f"Tool '{tool}' not found in server '{server}'"})
        tools_to_format = filtered_tools
    else:
        tools_to_format = all_tools
    
    formatted = global_pool._format_tools(tools_to_format, detail_level)
    return json.dumps(formatted, indent=2)


@mcp.tool()
async def search_tool_docs(
    query: str, 
    limit: int = 5, 
    detail_level: str = "description"
) -> str:
    """
    Search for tools across all connected MCP servers by name or description.
    
    Performs case-insensitive search against tool names and descriptions,
    returning matching tools from any server.
    
    Args:
        query: Search term to find in tool names or descriptions
        limit: Maximum number of results to return (default: 5)
        detail_level: Detail level - "name", "description", or "full" (default: "description")
    
    Returns:
        JSON string with search results grouped by server
    
    Example Response:
        {
          "filesystem": [
            {
              "name": "read_file",
              "description": "Read contents of a file from the filesystem"
            }
          ],
          "database": [
            {
              "name": "read_table",
              "description": "Read data from a database table"
            }
          ]
        }
    """
    if not global_pool:
        return json.dumps({"error": "MCP pool not initialized"})
    
    query_lower = query.lower()
    results = {}
    count = 0
    
    for server_name, tools in global_pool.tools.items():
        matching = []
        for tool in tools:
            if count >= limit:
                break
            
            if (query_lower in tool.name.lower() or 
                (tool.description and query_lower in tool.description.lower())):
                matching.append(tool)
                count += 1
        
        if matching:
            results[server_name] = global_pool._format_tools(matching, detail_level)
    
    return json.dumps(results, indent=2)


@mcp.tool()
async def server_tools_tree(server: Optional[str] = None) -> str:
    """
    Display a visual tree structure of servers and their tools.
    
    Provides a filesystem-like hierarchical view that's easy to scan visually.
    Useful for understanding the complete capability landscape at a glance.
    
    Args:
        server: Optional server name to show tree for just that server.
               If omitted, shows tree for all servers.
    
    Returns:
        Formatted tree string with box-drawing characters
    
    Example Response (all servers):
        ├── 📁 filesystem/
        │   ├── read_file
        │   │   └── Read contents of a file
        │   └── write_file
        │       └── Write content to a file
        └── 📁 weather/
            └── get_forecast
                └── Get weather forecast for a city
    
    Example Response (single server):
        📁 filesystem/
           ├── read_file
           │   └── Read contents of a file
           └── write_file
               └── Write content to a file
    """
    if not global_pool:
        return "[ERROR] MCP pool not initialized"
    
    return global_pool.get_pretty_tree(server_name=server)


@mcp.tool()
async def mcp_capability_summary() -> str:
    """
    Get comprehensive summary of the MCP bridge's current capabilities and status.
    
    Provides a high-level overview including connection status, available servers,
    total tool count, and example usage patterns. This is the best starting point
    for understanding what's available.
    
    Returns:
        JSON string containing:
        - Total servers connected
        - Total tools available
        - Detailed breakdown by server (name, description, tool count, tool list)
        - Python version
        - Timeout settings
    
    Example Response:
        {
          "total_servers": 3,
          "total_tools": 15,
          "python_version": "3.11.0",
          "timeout_default": 30,
          "timeout_max": 120,
          "servers": {
            "filesystem": {
              "description": "Local filesystem operations",
              "transport": "stdio",
              "tool_count": 5,
              "tools": [
                {"name": "read_file", "description": "Read file contents"},
                {"name": "write_file", "description": "Write to file"},
                ...
              ]
            },
            ...
          }
        }
    
    Usage:
        Call this tool first to understand what MCP servers and tools are available,
        then use execute_code to interact with them.
    """
    if not global_pool:
        return json.dumps({"error": "MCP pool not initialized"})
    
    summary = global_pool.get_structured_summary()
    summary["python_version"] = sys.version.split()[0]
    summary["timeout_default"] = EXECUTION_TIMEOUT
    summary["timeout_max"] = MAX_EXECUTION_TIMEOUT
    
    return json.dumps(summary, indent=2)


@mcp.tool()
async def check_tool_exists(server: str, tool: str) -> str:
    """
    Check if a specific tool exists on a specific server.
    
    Useful for validation before attempting to execute code that depends
    on a particular tool being available.
    
    Args:
        server: Name of the MCP server
        tool: Name of the tool (can use either 'tool-name' or 'tool_name' format)
    
    Returns:
        JSON string with exists status and tool info if found
    
    Example Response:
        {
          "exists": true,
          "server": "filesystem",
          "tool": "read_file",
          "description": "Read contents of a file from the filesystem"
        }
    """
    if not global_pool:
        return json.dumps({"error": "MCP pool not initialized"})
    
    if server not in global_pool.tools:
        return json.dumps({
            "exists": False,
            "error": f"Server '{server}' not found",
            "available_servers": list(global_pool.clients.keys())
        })
    
    # Normalize tool name for comparison
    tool_normalized = tool.replace("_", "-")
    
    for t in global_pool.tools[server]:
        if t.name == tool or t.name.replace("-", "_") == tool or t.name == tool_normalized:
            return json.dumps({
                "exists": True,
                "server": server,
                "tool": t.name,
                "description": t.description or "No description available"
            }, indent=2)
    
    return json.dumps({
        "exists": False,
        "server": server,
        "tool": tool,
        "available_tools": [t.name for t in global_pool.tools[server]]
    }, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run MCP server."""
    transport = os.getenv("MCP_TRANSPORT", "http")
    logger.info(f"📡 Starting with {transport} transport\n")
    
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        port = int(os.getenv("MCP_PORT", "8000"))
        logger.info(f"   Listening on http://localhost:{port}\n")
        mcp.run(transport=transport, port=port)


if __name__ == "__main__":
    main()
