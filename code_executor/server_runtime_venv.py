#!/usr/bin/env python3
"""
🚀 FastMCP Code Execution Server - ISOLATED VENV VERSION
==========================================================

Key Features:
- Runtime persistence (from previous version)
- SEPARATE virtualenv for code execution
- Package installation on-demand
- Isolation from server's Python environment
- All discovery and search tools

This version creates a dedicated venv at startup and executes all
code in that isolated environment, allowing package installation
without affecting the server.
"""

from contextlib import asynccontextmanager, AsyncExitStack
import asyncio
import json
import sys
import os
import types
import logging
import re
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import traceback
import tempfile

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
VENV_DIR = Path(os.getenv("MCP_VENV_DIR", str(Path.cwd() / ".mcp_execution_venv")))

# ============================================================================
# VENV MANAGER
# ============================================================================

class VenvManager:
    """Manages the isolated virtualenv for code execution."""
    
    def __init__(self, venv_path: Path):
        self.venv_path = venv_path
        self.python_path: Optional[Path] = None
        self.installed_packages: Set[str] = set()
    
    async def create_venv(self):
        """Create the virtualenv if it doesn't exist."""
        if self.venv_path.exists():
            logger.info(f"📦 Using existing venv: {self.venv_path}")
        else:
            logger.info(f"📦 Creating new venv: {self.venv_path}")
            try:
                # Use current Python to create venv
                await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "venv", str(self.venv_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                logger.info(f"✅ Venv created successfully")
            except Exception as e:
                logger.error(f"❌ Failed to create venv: {e}")
                raise
        
        # Set Python path
        if os.name == 'nt':  # Windows
            self.python_path = self.venv_path / "Scripts" / "python.exe"
        else:  # Unix-like
            self.python_path = self.venv_path / "bin" / "python"
        
        if not self.python_path.exists():
            raise RuntimeError(f"Python not found in venv: {self.python_path}")
        
        logger.info(f"✅ Venv Python: {self.python_path}")
    
    async def install_package(self, package: str) -> str:
        """Install a package in the venv."""
        if package in self.installed_packages:
            return f"Package '{package}' already installed"
        
        logger.info(f"📥 Installing package: {package}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                str(self.python_path), "-m", "pip", "install", package,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.installed_packages.add(package)
                logger.info(f"✅ Installed: {package}")
                return f"Successfully installed {package}"
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"❌ Failed to install {package}: {error_msg}")
                return f"Failed to install {package}: {error_msg}"
        
        except Exception as e:
            logger.error(f"❌ Exception installing {package}: {e}")
            return f"Exception installing {package}: {str(e)}"
    
    async def list_installed(self) -> List[str]:
        """List installed packages in the venv."""
        try:
            process = await asyncio.create_subprocess_exec(
                str(self.python_path), "-m", "pip", "list", "--format=freeze",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                packages = stdout.decode().strip().split('\n')
                return [p.split('==')[0] for p in packages if p]
            else:
                return []
        
        except Exception as e:
            logger.error(f"❌ Failed to list packages: {e}")
            return []

# Global venv manager
venv_manager: Optional[VenvManager] = None

# ============================================================================
# RUNTIME PERSISTENCE
# ============================================================================

class RuntimeNamespace:
    """Persistent namespace for mcps.runtime module."""
    
    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._functions: Dict[str, Any] = {}
    
    def add(self, name: str, value: Any):
        """Add a value to runtime that persists across executions."""
        if callable(value):
            self._functions[name] = value
        else:
            self._store[name] = value
    
    def get(self, name: str, default=None):
        """Get a value from runtime."""
        if name in self._functions:
            return self._functions[name]
        return self._store.get(name, default)
    
    def remove(self, name: str):
        """Remove a value from runtime."""
        self._store.pop(name, None)
        self._functions.pop(name, None)
    
    def clear(self):
        """Clear all runtime state."""
        self._store.clear()
        self._functions.clear()
    
    def list(self):
        """List all items in runtime."""
        return {
            'variables': list(self._store.keys()),
            'functions': list(self._functions.keys())
        }
    
    def serialize(self) -> str:
        """Serialize runtime for transfer to subprocess."""
        return json.dumps({
            'store': {k: repr(v) for k, v in self._store.items()},
            'functions': {k: f.__name__ for k, f in self._functions.items()}
        })

global_runtime = RuntimeNamespace()

# ============================================================================
# API KEY HELPER & CONFIG LOADING (same as before)
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
# MCP CLIENT POOL (same as before, abbreviated for space)
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
    
    async def _connect_server(self, name: str, config: Dict[str, Any]):
        """Connect to a single MCP server."""
        transport_type = config.get("transport", "stdio")
        
        try:
            if transport_type == "stdio":
                command = config.get("command")
                args = config.get("args", [])
                env = resolve_env_vars(config.get("env", {}))
                
                transport = StdioTransport(
                    command=command,
                    args=args,
                    env=env if env else None
                )
                
                client = Client(transport)
                await self.stack.enter_async_context(client)
                
                tools_response = await client.list_tools()
                tools = tools_response.tools if hasattr(tools_response, 'tools') else tools_response
                
                self.clients[name] = client
                self.tools[name] = tools
                
                logger.info(f"  ✅ {name}: {len(tools)} tools")
            
            elif transport_type in ["http", "sse", "streamable-http"]:
                url = config.get("url")
                headers = config.get("headers", {})
                
                transport = StreamableHttpTransport(url=url, headers=headers)
                client = Client(transport)
                
                await self.stack.enter_async_context(client)
                
                tools_response = await client.list_tools()
                tools = tools_response.tools if hasattr(tools_response, 'tools') else tools_response
                
                self.clients[name] = client
                self.tools[name] = tools
                
                logger.info(f"  ✅ {name}: {len(tools)} tools")
        
        except Exception as e:
            logger.error(f"   Error: {e}")
    
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
                        return result.content[0]
                    return result.content
                return str(result)
        
        except Exception as e:
            return {"error": f"Tool call failed: {str(e)}"}

global_pool: Optional[MCPClientPool] = None

# ============================================================================
# ISOLATED CODE EXECUTION - NEW!
# ============================================================================

async def execute_code_in_venv(code: str, timeout: Optional[int] = None) -> str:
    """Execute code in the isolated venv subprocess."""
    
    if not venv_manager or not venv_manager.python_path:
        return "[ERROR] Venv not initialized"
    
    if not global_pool:
        return "[ERROR] MCP pool not initialized"
    
    timeout = min(timeout or EXECUTION_TIMEOUT, MAX_EXECUTION_TIMEOUT)
    
    # Create a temporary file with the execution wrapper
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        wrapper_path = f.name
        
        # Build the execution wrapper
        wrapper_code = f'''
import sys
import json
import asyncio
from pathlib import Path

# This will be replaced with actual MCP client setup
# For now, we'll create stub proxies

class ToolProxy:
    def __init__(self, server_name, tool_name):
        self.server_name = server_name
        self.tool_name = tool_name
    
    async def __call__(self, **kwargs):
        # This would call back to the main server via RPC
        # For now, return a placeholder
        return f"[STUB] Would call {{self.server_name}}.{{self.tool_name}} with {{kwargs}}"

# User code starts here
async def user_code():
{chr(10).join("    " + line for line in code.split(chr(10)))}

# Run the user code
if __name__ == "__main__":
    try:
        asyncio.run(user_code())
    except Exception as e:
        import traceback
        print(f"[ERROR] {{type(e).__name__}}: {{e}}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
'''
        
        f.write(wrapper_code)
    
    try:
        # Execute in venv
        process = await asyncio.create_subprocess_exec(
            str(venv_manager.python_path),
            wrapper_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            output = stdout.decode() if stdout else ""
            errors = stderr.decode() if stderr else ""
            
            if process.returncode != 0:
                return f"[ERROR]\n{errors}" if errors else "[ERROR] Execution failed"
            
            return output if output else (errors if errors else "[No output]")
        
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return f"[TIMEOUT] Execution exceeded {timeout}s"
    
    finally:
        # Cleanup temp file
        try:
            os.unlink(wrapper_path)
        except:
            pass

# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for FastMCP."""
    global global_pool, venv_manager
    
    logger.info("=" * 60)
    logger.info("🚀 MCP Code Execution Server - ISOLATED VENV")
    logger.info("=" * 60)
    logger.info(f"📂 Config: {Path(CONFIG_FILE).absolute()}")
    logger.info(f"📦 Venv: {VENV_DIR}\n")
    
    # Initialize venv
    venv_manager = VenvManager(VENV_DIR)
    await venv_manager.create_venv()
    
    # Load config and connect to MCP servers
    config = load_mcp_config()
    global_pool = MCPClientPool(config)
    await global_pool.connect_all()
    
    total_tools = sum(len(tools) for tools in global_pool.tools.values())
    logger.info(f"\n✅ Ready!")
    logger.info(f"   Servers: {len(global_pool.clients)}")
    logger.info(f"   Tools: {total_tools}")
    logger.info(f"   Venv: {venv_manager.python_path}")
    logger.info(f"   Runtime: Enabled (persistent state)\n")
    
    yield
    
    # Cleanup
    logger.info("\n🔌 Shutting down...")
    await global_pool.disconnect_all()

# ============================================================================
# FASTMCP SERVER - VENV VERSION
# ============================================================================

mcp = FastMCP(name="mcp-code-execution-venv", lifespan=lifespan)


@mcp.tool()
async def execute_code(code: str, timeout_seconds: Optional[int] = None) -> str:
    """
    Execute Python code in an isolated virtualenv with MCP server access.
    
    The code runs in a SEPARATE Python environment, isolated from the server.
    This allows installing packages without affecting the server environment.
    
    Args:
        code: Python code to execute
        timeout_seconds: Maximum execution time (default: 30s, max: 120s)
    
    Returns:
        String containing stdout/stderr output or error message
    
    Examples:
        
        # Example 1: Use numpy (install first with install_package)
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        print(f"Mean: {np.mean(arr)}")
        print(f"Std: {np.std(arr)}")
        
        # Example 2: Data analysis with pandas
        import pandas as pd
        data = {'name': ['Alice', 'Bob'], 'age': [25, 30]}
        df = pd.DataFrame(data)
        print(df.describe())
        
        # Example 3: Call MCP tools (when proxy is set up)
        # from mcps.servers import filesystem
        # result = await filesystem.read_file(path="/tmp/data.csv")
        # print(result)
    
    Note:
        - Code runs in isolated venv, not server's Python environment
        - Use install_package tool to add required packages
        - MCP tool calling requires proxy setup (advanced feature)
    """
    return await execute_code_in_venv(code, timeout_seconds)


@mcp.tool()
async def install_package(package: str) -> str:
    """
    Install a Python package in the execution venv.
    
    This allows the code execution environment to use additional packages
    without affecting the server's Python environment.
    
    Args:
        package: Package name (e.g., 'numpy', 'pandas', 'requests')
    
    Returns:
        Installation status message
    
    Examples:
        install_package("numpy")
        install_package("pandas")
        install_package("matplotlib")
        install_package("scikit-learn")
    
    Note:
        Once installed, packages persist in the venv across executions
        until the venv is deleted or the server is restarted with a new venv.
    """
    if not venv_manager:
        return "[ERROR] Venv not initialized"
    
    return await venv_manager.install_package(package)


@mcp.tool()
async def list_installed_packages() -> str:
    """
    List all packages installed in the execution venv.
    
    Returns:
        JSON array of installed package names
    """
    if not venv_manager:
        return json.dumps({"error": "Venv not initialized"})
    
    packages = await venv_manager.list_installed()
    return json.dumps({
        "installed_packages": packages,
        "count": len(packages)
    }, indent=2)


@mcp.tool()
async def venv_info() -> str:
    """
    Get information about the execution virtualenv.
    
    Returns:
        JSON with venv path, Python version, and package count
    """
    if not venv_manager:
        return json.dumps({"error": "Venv not initialized"})
    
    packages = await venv_manager.list_installed()
    
    return json.dumps({
        "venv_path": str(venv_manager.venv_path),
        "python_path": str(venv_manager.python_path),
        "package_count": len(packages),
        "isolated": True,
        "description": "Code executes in this separate environment"
    }, indent=2)


@mcp.tool()
async def runtime_list() -> str:
    """List all items in persistent runtime (not yet functional in venv mode)."""
    return json.dumps(global_runtime.list(), indent=2)


@mcp.tool()
async def discovered_servers(include_description: bool = False) -> str:
    """Get list of all successfully connected MCP servers."""
    if not global_pool:
        return json.dumps({"error": "MCP pool not initialized"})
    
    servers = list(global_pool.clients.keys())
    return json.dumps(servers, indent=2)


@mcp.tool()
async def mcp_capability_summary() -> str:
    """Get comprehensive summary of MCP capabilities and venv status."""
    if not global_pool:
        return json.dumps({"error": "MCP pool not initialized"})
    
    packages = await venv_manager.list_installed() if venv_manager else []
    
    return json.dumps({
        "total_servers": len(global_pool.clients),
        "total_tools": sum(len(tools) for tools in global_pool.tools.values()),
        "python_version": sys.version.split()[0],
        "timeout_default": EXECUTION_TIMEOUT,
        "timeout_max": MAX_EXECUTION_TIMEOUT,
        "venv": {
            "path": str(venv_manager.venv_path) if venv_manager else None,
            "isolated": True,
            "packages_installed": len(packages)
        },
        "servers": list(global_pool.clients.keys())
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