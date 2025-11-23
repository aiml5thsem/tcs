#!/usr/bin/env python3
"""
Universal MCP Client with Multi-Profile LLM System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Complete Features:
✅ YAML configuration (separate files for security)
✅ Environment variable support (.env + ${VAR} syntax)
✅ Multiple LLM profiles with custom system prompts
✅ API key management (secure + flexible)
✅ Custom providers and LiteLLM proxy support
✅ Automatic fallback chains
✅ Complete MCP support (tools, resources, templates, prompts)
✅ Multimodal support (images, audio, PDFs)
✅ Production-ready error handling

Author: FastMCP 2.0 + LiteLLM Integration
"""

import asyncio
import json
import base64
import os
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import AsyncExitStack
from dataclasses import dataclass
import string

# YAML support
import yaml

# FastMCP 2.0
from fastmcp import Client
from fastmcp.client.transports import (
    StdioTransport,
    StreamableHttpTransport,
    SSETransport,
)
from fastmcp.utilities.types import Image, Audio, File

# LiteLLM with custom provider support
from litellm import acompletion
import litellm

# Rich UI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.markdown import Markdown
from rich.box import ROUNDED

console = Console()

class ConfigLoader:
    """Load and process YAML/JSON configuration with environment variable support"""
    
    @staticmethod
    def load_config(base_name: str, search_extensions: List[str] = None) -> Dict:
        """
        Load configuration from YAML or JSON file.
        Searches for files with different extensions and merges if multiple exist.
        
        Args:
            base_name: Base filename without extension (e.g., "mcp_servers", "llm_config")
            search_extensions: List of extensions to search (e.g., ['.yaml', '.yml', '.json'])
        
        Returns:
            Merged configuration dictionary
        """
        if search_extensions is None:
            search_extensions = ['.yaml', '.yml', '.json']
        
        configs = []
        found_files = []
        
        # Search for all matching configuration files
        for ext in search_extensions:
            filepath = f"{base_name}{ext}"
            if Path(filepath).exists():
                found_files.append(filepath)
                try:
                    config = ConfigLoader._load_file(filepath)
                    if config:
                        configs.append(config)
                        console.print(f"[dim]  Loaded: {filepath}[/dim]")
                except Exception as e:
                    console.print(f"[yellow]  Warning: Failed to load {filepath}: {e}[/yellow]")
        
        if not configs:
            console.print(f"[yellow]⚠ No configuration file found for '{base_name}'[/yellow]")
            console.print(f"[dim]  Searched for: {', '.join([base_name + ext for ext in search_extensions])}[/dim]")
            return {}
        
        # If multiple files found, merge them
        if len(configs) > 1:
            console.print(f"[cyan]  Found {len(configs)} config files, merging...[/cyan]")
            merged = ConfigLoader._deep_merge_dicts(*configs)
            return merged
        
        return configs[0]
    
    @staticmethod
    def _load_file(filepath: str) -> Dict:
        """Load a single YAML or JSON file with environment variable substitution"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Substitute environment variables: ${VAR_NAME}
        content = ConfigLoader._substitute_env_vars(content)
        
        # Determine file type and parse
        ext = Path(filepath).suffix.lower()
        
        if ext in ['.yaml', '.yml']:
            return yaml.safe_load(content) or {}
        elif ext == '.json':
            return json.loads(content)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    @staticmethod
    def _substitute_env_vars(content: str) -> str:
        """Replace ${VAR_NAME} with environment variable values"""
        pattern = re.compile(r'\$\{([^}]+)\}')
        
        def replacer(match):
            var_name = match.group(1)
            value = os.getenv(var_name, '')
            if not value:
                # Don't warn for common optional variables
                if var_name not in ['DEBUG', 'LITELLM_PROXY_KEY', 'MCP_API_TOKEN']:
                    console.print(f"[dim]  Env var not set: {var_name}[/dim]")
            return value
        
        return pattern.sub(replacer, content)
    
    @staticmethod
    def _deep_merge_dicts(*dicts: Dict) -> Dict:
        """
        Deep merge multiple dictionaries.
        Later dictionaries override earlier ones.
        
        Example:
            dict1 = {"a": {"b": 1, "c": 2}}
            dict2 = {"a": {"c": 3, "d": 4}}
            result = {"a": {"b": 1, "c": 3, "d": 4}}
        """
        result = {}
        
        for d in dicts:
            for key, value in d.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    # Recursively merge nested dictionaries
                    result[key] = ConfigLoader._deep_merge_dicts(result[key], value)
                else:
                    # Override with new value
                    result[key] = value
        
        return result
    
    @staticmethod
    def load_yaml(filepath: str) -> Dict:
        """
        Legacy method - load YAML file with environment variable substitution.
        Kept for backward compatibility.
        """
        if not Path(filepath).exists():
            return {}
        return ConfigLoader._load_file(filepath)

@dataclass
class LLMProfile:
    """LLM provider profile with API configuration"""
    name: str
    provider: str
    model: str
    system_prompt: str
    
    # API configuration
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    
    # AWS-specific (for Bedrock)
    aws_region: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    
    # Fallbacks
    fallbacks: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.fallbacks is None:
            self.fallbacks = []
    
    def get_model_name(self) -> str:
        """Get full model name for LiteLLM"""
        return f"{self.provider}/{self.model}"
    
    def get_completion_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for litellm.completion()"""
        kwargs = {
            "model": self.get_model_name(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        
        # Add API configuration
        if self.api_key:
            kwargs["api_key"] = self.api_key
        
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        if self.api_version:
            kwargs["api_version"] = self.api_version
        
        # AWS credentials for Bedrock
        if self.aws_region:
            kwargs["aws_region_name"] = self.aws_region
        
        if self.aws_access_key_id:
            kwargs["aws_access_key_id"] = self.aws_access_key_id
        
        if self.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        
        return kwargs
    
    def get_fallback_models(self) -> List[str]:
        """Get list of fallback model configurations"""
        fallbacks = []
        for fb in self.fallbacks:
            fb_provider = fb.get('provider', 'openai')
            fb_model = fb.get('model', '')
            fallbacks.append(f"{fb_provider}/{fb_model}")
        return fallbacks


class MultimodalContentHandler:
    """Handles multimodal content for all LLM providers"""
    
    @staticmethod
    def encode_file_to_base64(file_path: str) -> tuple[str, str]:
        """Encode file to base64 and detect MIME type"""
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            ext = Path(file_path).suffix.lower()
            mime_map = {
                '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                '.png': 'image/png', '.gif': 'image/gif',
                '.webp': 'image/webp', '.heic': 'image/heic',
                '.mp3': 'audio/mpeg', '.wav': 'audio/wav',
                '.pdf': 'application/pdf'
            }
            mime_type = mime_map.get(ext, 'application/octet-stream')
        
        with open(file_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        
        return mime_type, encoded
    
    @staticmethod
    def create_image_content(image_data: Union[str, bytes, Image], 
                           mime_type: str = "image/png") -> Dict:
        """Create image content in OpenAI format (LiteLLM auto-converts)"""
        if isinstance(image_data, Image):
            data_uri = image_data.to_data_uri()
            return {"type": "image_url", "image_url": {"url": data_uri}}
        
        elif isinstance(image_data, bytes):
            encoded = base64.b64encode(image_data).decode('utf-8')
            data_uri = f"data:{mime_type};base64,{encoded}"
            return {"type": "image_url", "image_url": {"url": data_uri}}
        
        elif isinstance(image_data, str):
            if Path(image_data).exists():
                mime_type, encoded = MultimodalContentHandler.encode_file_to_base64(image_data)
                data_uri = f"data:{mime_type};base64,{encoded}"
                return {"type": "image_url", "image_url": {"url": data_uri}}
            elif image_data.startswith('data:'):
                return {"type": "image_url", "image_url": {"url": image_data}}
            elif image_data.startswith('http'):
                return {"type": "image_url", "image_url": {"url": image_data}}
            else:
                data_uri = f"data:{mime_type};base64,{image_data}"
                return {"type": "image_url", "image_url": {"url": data_uri}}
    
    @staticmethod
    def create_pdf_content(pdf_data: Union[str, bytes, File],
                         filename: str = "document.pdf") -> Dict:
        """Create PDF content"""
        if isinstance(pdf_data, File):
            encoded = base64.b64encode(pdf_data.data).decode('utf-8')
            data_uri = f"data:{pdf_data.mimeType};base64,{encoded}"
            return {"type": "file", "file": {"file_data": data_uri, "filename": filename}}
        
        elif isinstance(pdf_data, bytes):
            encoded = base64.b64encode(pdf_data).decode('utf-8')
            data_uri = f"data:application/pdf;base64,{encoded}"
            return {"type": "file", "file": {"file_data": data_uri, "filename": filename}}
        
        elif isinstance(pdf_data, str) and Path(pdf_data).exists():
            mime_type, encoded = MultimodalContentHandler.encode_file_to_base64(pdf_data)
            data_uri = f"data:{mime_type};base64,{encoded}"
            return {"type": "file", "file": {"file_data": data_uri, "filename": Path(pdf_data).name}}
        
        return {"type": "text", "text": f"[PDF: {filename}]"}


class MCPUniversalClient:
    """Universal MCP Client with YAML/JSON configuration and multi-profile LLM support"""
    
    def __init__(self, 
                 mcp_config_path: str = "mcp_servers",  # No extension
                 llm_config_path: str = "llm_config"):   # No extension
        
        self.mcp_config_path = mcp_config_path
        self.llm_config_path = llm_config_path
        
        # MCP clients
        self.clients: Dict[str, Client] = {}
        self.exit_stack = AsyncExitStack()
        
        # MCP capabilities
        self.all_tools = []
        self.all_resources = []
        self.all_resource_templates = []
        self.all_prompts = []
        
        # Multimodal handler
        self.content_handler = MultimodalContentHandler()
        
        # Load configurations (supports both YAML and JSON, merges if both exist)
        console.print("[cyan]Loading configuration...[/cyan]")
        
        # Load MCP servers config (tries mcp_servers.yaml, .yml, .json, mcp_settings.json)
        self.mcp_config = ConfigLoader.load_config(
            mcp_config_path,
            search_extensions=['.yaml', '.yml', '.json']
        )
        
        # Also check for legacy mcp_settings.json if mcp_servers not found
        if not self.mcp_config or not self.mcp_config.get("servers"):
            legacy_config = ConfigLoader.load_config(
                "mcp_settings",
                search_extensions=['.json', '.yaml', '.yml']
            )
            if legacy_config:
                # Merge legacy config
                if legacy_config.get("mcpServers"):
                    # Convert old format to new format
                    if not self.mcp_config.get("servers"):
                        self.mcp_config["servers"] = legacy_config["mcpServers"]
                    else:
                        self.mcp_config["servers"] = ConfigLoader._deep_merge_dicts(
                            self.mcp_config.get("servers", {}),
                            legacy_config["mcpServers"]
                        )
        
        # Load LLM config (tries llm_config.yaml, .yml, .json)
        self.llm_config = ConfigLoader.load_config(
            llm_config_path,
            search_extensions=['.yaml', '.yml', '.json']
        )
        
        # Also check if llmProfiles exist in mcp_settings.json (legacy)
        if not self.llm_config.get("profiles"):
            # Check if mcp_settings has llmProfiles
            legacy_mcp = ConfigLoader.load_config(
                "mcp_settings",
                search_extensions=['.json']
            )
            if legacy_mcp.get("llmProfiles"):
                console.print("[yellow]  Found llmProfiles in mcp_settings.json (legacy format)[/yellow]")
                if not self.llm_config.get("profiles"):
                    self.llm_config["profiles"] = legacy_mcp["llmProfiles"]
                if not self.llm_config.get("defaultProfile") and legacy_mcp.get("defaultProfile"):
                    self.llm_config["defaultProfile"] = legacy_mcp["defaultProfile"]
        
        console.print("[green]✓ Configuration loaded[/green]\n")
        
        # Load LLM profiles
        self.profiles = self._load_llm_profiles()
        self.current_profile_name = self.llm_config.get("defaultProfile", "fast")
        self.current_profile = self.profiles.get(self.current_profile_name)
        
        # Global settings
        self.llm_settings = self.llm_config.get("settings", {})
        self.mcp_settings = self.mcp_config.get("settings", {})
        
        # Enable debug
        if self.llm_settings.get("debugMode") or os.getenv("DEBUG"):
            litellm.set_verbose = True
    
    def _load_llm_profiles(self) -> Dict[str, LLMProfile]:
        """Load LLM profiles from YAML configuration"""
        profiles = {}
        
        for profile_id, config in self.llm_config.get("profiles", {}).items():
            profiles[profile_id] = LLMProfile(
                name=config.get("name", profile_id),
                provider=config.get("provider", "ollama"),
                model=config.get("model", "llama3.2"),
                system_prompt=config.get("systemPrompt", "You are a helpful AI assistant."),
                api_key=config.get("apiKey"),
                api_base=config.get("apiBase"),
                api_version=config.get("apiVersion"),
                aws_region=config.get("awsRegion"),
                aws_access_key_id=config.get("awsAccessKeyId"),
                aws_secret_access_key=config.get("awsSecretAccessKey"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("maxTokens", 4096),
                top_p=config.get("topP", 0.9),
                fallbacks=config.get("fallbacks", [])
            )
        
        return profiles
    
    def switch_profile(self, profile_name: str) -> bool:
        """Switch to a different LLM profile"""
        if profile_name in self.profiles:
            self.current_profile_name = profile_name
            self.current_profile = self.profiles[profile_name]
            return True
        return False
    
    def _create_transport(self, server_name: str, server_config: Dict):
        """Create MCP transport"""
        transport_type = server_config.get("transport", "stdio")
        
        if transport_type == "stdio":
            command = server_config.get("command")
            args = server_config.get("args", [])
            env = server_config.get("env", {})
            cwd = server_config.get("cwd")
            
            merged_env = os.environ.copy()
            merged_env.update(env)
            
            return StdioTransport(command=command, args=args, env=merged_env, cwd=cwd)
        
        elif transport_type in ["http", "streamable-http", "streamable_http"]:
            url = server_config.get("url")
            headers = server_config.get("headers", {})
            return StreamableHttpTransport(url=url, headers=headers)
        
        elif transport_type == "sse":
            url = server_config.get("url")
            headers = server_config.get("headers", {})
            return SSETransport(url=url, headers=headers)
        
        else:
            raise ValueError(f"Unknown transport type: {transport_type}")
    
    async def connect_servers(self):
        """Connect to all MCP servers"""
        servers = self.mcp_config.get("servers", {})
        
        if not servers:
            console.print("[yellow]⚠ No MCP servers configured in mcp_servers.yaml[/yellow]")
            return
        
        console.print(f"\n[cyan]Connecting to {len(servers)} MCP server(s)...[/cyan]\n")
        
        for server_name, server_config in servers.items():
            try:
                await self._connect_server(server_name, server_config)
                console.print(f"[green]✓[/green] Connected: [cyan]{server_name}[/cyan]")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed: [cyan]{server_name}[/cyan] - {str(e)[:60]}")
    
    async def _connect_server(self, name: str, config: Dict):
        """Connect to single MCP server"""
        transport = self._create_transport(name, config)
        client = Client(transport)
        await self.exit_stack.enter_async_context(client)
        self.clients[name] = client
        await self._load_server_capabilities(name, client)
    
    async def _load_server_capabilities(self, server_name: str, client: Client):
        """Load capabilities from server"""
        # Tools
        try:
            tools = await client.list_tools()
            for tool in tools:
                self.all_tools.append({
                    "server": server_name,
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.inputSchema
                })
        except:
            pass
        
        # Resources
        try:
            resources = await client.list_resources()
            for resource in resources:
                self.all_resources.append({
                    "server": server_name,
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description
                })
        except:
            pass
        
        # Resource templates
        try:
            templates = await client.list_resource_templates()
            for template in templates:
                self.all_resource_templates.append({
                    "server": server_name,
                    "uriTemplate": template.uriTemplate,
                    "name": template.name,
                    "description": template.description
                })
        except:
            pass
        
        # Prompts
        try:
            prompts = await client.list_prompts()
            for prompt in prompts:
                self.all_prompts.append({
                    "server": server_name,
                    "name": prompt.name,
                    "description": prompt.description,
                    "arguments": getattr(prompt, 'arguments', [])
                })
        except:
            pass
    
    def _format_tools_for_llm(self) -> List[Dict]:
        """Format tools for LLM"""
        return [{
            "type": "function",
            "function": {
                "name": f"{t['server']}__{t['name']}",
                "description": t['description'] or f"Tool from {t['server']}",
                "parameters": t['schema']
            }
        } for t in self.all_tools]
    
    def _format_resources_for_llm(self) -> List[Dict]:
        """Format resources and templates for LLM"""
        functions = []
        
        for res in self.all_resources:
            functions.append({
                "type": "function",
                "function": {
                    "name": f"{res['server']}__read_resource__{res['uri'].replace('://', '_').replace('/', '_')}",
                    "description": f"Read: {res['description'] or res['name']}",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            })
        
        for tmpl in self.all_resource_templates:
            params = re.findall(r'\{(\w+)\}', tmpl['uriTemplate'])
            properties = {p: {"type": "string", "description": f"Parameter {p}"} for p in params}
            
            functions.append({
                "type": "function",
                "function": {
                    "name": f"{tmpl['server']}__read_template__{tmpl['name'].replace(' ', '_')}",
                    "description": f"Read: {tmpl['description'] or tmpl['name']}",
                    "parameters": {"type": "object", "properties": properties, "required": params}
                }
            })
        
        return functions
    
    def _format_prompts_for_llm(self) -> List[Dict]:
        """Format prompts for LLM"""
        functions = []
        for prompt in self.all_prompts:
            properties = {}
            required = []
            
            for arg in prompt.get('arguments', []):
                arg_name = arg.get('name', '')
                properties[arg_name] = {
                    "type": "string",
                    "description": arg.get('description', '')
                }
                if arg.get('required', False):
                    required.append(arg_name)
            
            functions.append({
                "type": "function",
                "function": {
                    "name": f"{prompt['server']}__use_prompt__{prompt['name']}",
                    "description": f"Use prompt: {prompt['description'] or prompt['name']}",
                    "parameters": {"type": "object", "properties": properties, "required": required}
                }
            })
        
        return functions
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict) -> Any:
        """Call MCP tool"""
        client = self.clients.get(server_name)
        if not client:
            raise ValueError(f"Server {server_name} not connected")
        return await client.call_tool(tool_name, arguments)
    
    async def read_resource(self, server_name: str, uri: str) -> Any:
        """Read MCP resource"""
        client = self.clients.get(server_name)
        if not client:
            raise ValueError(f"Server {server_name} not connected")
        return await client.read_resource(uri)
    
    async def read_resource_template(self, server_name: str, uri_template: str, **params) -> Any:
        """Read dynamic resource"""
        uri = uri_template
        for param, value in params.items():
            uri = uri.replace(f"{{{param}}}", str(value))
        return await self.read_resource(server_name, uri)
    
    async def get_prompt(self, server_name: str, prompt_name: str, arguments: Dict = None) -> Any:
        """Get MCP prompt"""
        client = self.clients.get(server_name)
        if not client:
            raise ValueError(f"Server {server_name} not connected")
        return await client.get_prompt(prompt_name, arguments or {})
    
    def _prepare_messages_for_llm(self, messages: List[Dict]) -> List[Dict]:
        """Prepare messages with multimodal support"""
        prepared = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            
            if isinstance(content, Image):
                prepared.append({"role": role, "content": [self.content_handler.create_image_content(content)]})
            elif isinstance(content, File):
                prepared.append({"role": role, "content": [self.content_handler.create_pdf_content(content)]})
            elif isinstance(content, list):
                processed = []
                for item in content:
                    if isinstance(item, dict):
                        processed.append(item)
                    elif isinstance(item, Image):
                        processed.append(self.content_handler.create_image_content(item))
                    elif isinstance(item, File):
                        processed.append(self.content_handler.create_pdf_content(item))
                    else:
                        processed.append({"type": "text", "text": str(item)})
                prepared.append({"role": role, "content": processed})
            elif isinstance(content, str):
                prepared.append({"role": role, "content": content})
            else:
                prepared.append({"role": role, "content": str(content)})
        
        return prepared
    
    def _extract_text_from_result(self, result: Any) -> str:
        """Extract text from MCP result"""
        if hasattr(result, 'content') and result.content:
            parts = []
            for item in result.content:
                if hasattr(item, 'text'):
                    parts.append(item.text)
                elif hasattr(item, 'type') and item.type == "text":
                    parts.append(getattr(item, 'text', ''))
            return "\n".join(parts) if parts else str(result)
        return str(result)
    
    async def chat(self,
                   messages: List[Dict],
                   profile_override: Optional[str] = None,
                   max_iterations: Optional[int] = None) -> str:
        """Main chat function with profile support"""
        profile = self.profiles.get(profile_override) if profile_override else self.current_profile
        if not profile:
            return "Error: No valid LLM profile"
        
        # Prepare messages with system prompt
        prepared_messages = [
            {"role": "system", "content": profile.system_prompt}
        ]
        prepared_messages.extend(self._prepare_messages_for_llm(messages))
        
        # Get all functions
        all_functions = []
        if self.mcp_settings.get("enableTools", True) and self.all_tools:
            all_functions.extend(self._format_tools_for_llm())
        if self.mcp_settings.get("enableResources", True) and (self.all_resources or self.all_resource_templates):
            all_functions.extend(self._format_resources_for_llm())
        if self.mcp_settings.get("enablePrompts", True) and self.all_prompts:
            all_functions.extend(self._format_prompts_for_llm())
        
        max_iter = max_iterations or self.llm_settings.get("maxIterations", 10)
        iteration = 0
        
        while iteration < max_iter:
            iteration += 1
            
            try:
                # Build completion kwargs
                completion_kwargs = profile.get_completion_kwargs()
                completion_kwargs["messages"] = prepared_messages
                completion_kwargs["tools"] = all_functions if all_functions else None
                completion_kwargs["tool_choice"] = "auto" if all_functions else None
                completion_kwargs["timeout"] = self.llm_settings.get("requestTimeout", 120)
                
                # Add fallbacks if configured
                fallbacks = profile.get_fallback_models()
                if fallbacks:
                    completion_kwargs["fallbacks"] = fallbacks
                
                response = await acompletion(**completion_kwargs)
                
                assistant_message = response.choices[0].message
                
                msg_dict = {
                    "role": "assistant",
                    "content": assistant_message.content or ""
                }
                
                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                    msg_dict["tool_calls"] = [
                        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in assistant_message.tool_calls
                    ]
                
                prepared_messages.append(msg_dict)
                
                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                    # Execute function calls
                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        try:
                            if "__read_resource__" in function_name:
                                server_name = function_name.split("__")[0]
                                resource_uri = next((r['uri'] for r in self.all_resources 
                                                   if r['server'] == server_name and function_name.endswith(
                                                       r['uri'].replace('://', '_').replace('/', '_'))), None)
                                console.print(f"[cyan]→ Reading resource: {resource_uri}[/cyan]")
                                result = await self.read_resource(server_name, resource_uri)
                                result_text = self._extract_text_from_result(result)
                            
                            elif "__read_template__" in function_name:
                                server_name, _, template_name = function_name.split("__")
                                template_uri = next((t['uriTemplate'] for t in self.all_resource_templates 
                                                    if t['server'] == server_name and t['name'] == template_name.replace('_', ' ')), None)
                                console.print(f"[cyan]→ Reading template: {template_uri}[/cyan]")
                                result = await self.read_resource_template(server_name, template_uri, **function_args)
                                result_text = self._extract_text_from_result(result)
                            
                            elif "__use_prompt__" in function_name:
                                server_name, _, prompt_name = function_name.split("__")
                                console.print(f"[cyan]→ Using prompt: {prompt_name}[/cyan]")
                                result = await self.get_prompt(server_name, prompt_name, function_args)
                                if hasattr(result, 'messages'):
                                    result_text = "\n".join([getattr(msg.content, 'text', str(msg.content)) 
                                                            for msg in result.messages])
                                else:
                                    result_text = str(result)
                            
                            else:
                                server_name, tool_name = function_name.split("__", 1) if "__" in function_name else (list(self.clients.keys())[0], function_name)
                                console.print(f"[cyan]→ Calling tool: {tool_name}[/cyan]")
                                result = await self.call_tool(server_name, tool_name, function_args)
                                result_text = self._extract_text_from_result(result)
                            
                            console.print(f"[green]✓[/green] Result: [dim]{result_text[:100]}...[/dim]")
                        
                        except Exception as e:
                            result_text = f"Error: {str(e)}"
                            console.print(f"[red]✗[/red] {result_text}")
                        
                        prepared_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text
                        })
                    
                    continue
                else:
                    return assistant_message.content or ""
            
            except Exception as e:
                console.print(f"[red]✗ Error: {e}[/red]")
                return f"Error: {str(e)}"
        
        return "⚠ Max iterations reached"
    
    async def interactive_mode(self):
        """Interactive chat mode"""
        console.print(Panel.fit(
            f"[bold cyan]🤖 Universal MCP Client[/bold cyan]\n"
            f"[dim]FastMCP 2.0 + YAML Configuration[/dim]\n\n"
            f"[green]Profile:[/green] {self.current_profile.name}\n"
            f"[blue]Model:[/blue] {self.current_profile.get_model_name()}\n"
            f"[yellow]MCP Servers:[/yellow] {len(self.clients)}\n"
            f"[magenta]Capabilities:[/magenta] {len(self.all_tools)} tools, "
            f"{len(self.all_resources)} resources, {len(self.all_prompts)} prompts",
            title="🚀 Production Ready",
            border_style="cyan",
            box=ROUNDED
        ))
        
        console.print("\n[bold]Commands:[/bold]")
        console.print("  [cyan]/profiles[/cyan]  - List LLM profiles")
        console.print("  [cyan]/profile <name>[/cyan] - Switch profile")
        console.print("  [cyan]/tools, /resources, /prompts[/cyan] - List capabilities")
        console.print("  [cyan]/image <path>[/cyan] - Attach image")
        console.print("  [cyan]/clear, /quit[/cyan]\n")
        
        conversation = []
        pending_attachments = []
        
        while True:
            try:
                user_input = Prompt.ask(f"[bold green]You[/bold green] [[dim]{self.current_profile_name}[/dim]]")
                
                if not user_input.strip():
                    continue
                
                if user_input.startswith("/"):
                    should_quit, pending, conv = await self._handle_command(
                        user_input, pending_attachments, conversation
                    )
                    pending_attachments = pending
                    conversation = conv
                    if should_quit:
                        break
                    continue
                
                if pending_attachments:
                    content = [{"type": "text", "text": user_input}]
                    content.extend(pending_attachments)
                    conversation.append({"role": "user", "content": content})
                    pending_attachments = []
                else:
                    conversation.append({"role": "user", "content": user_input})
                
                console.print(f"\n[bold blue]Assistant[/bold blue] [[dim]{self.current_profile_name}[/dim]]:")
                response = await self.chat(conversation)
                console.print(Markdown(response))
                console.print()
                
                conversation.append({"role": "assistant", "content": response})
            
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type /quit to exit.[/yellow]")
            except Exception as e:
                console.print(f"[red]✗ Error: {e}[/red]")
    
    async def _handle_command(self, command: str, pending_attachments: List, conversation: List) -> tuple[bool, List, List]:
        """Handle commands"""
        parts = command.split(maxsplit=1)
        cmd = parts[0]
        
        if cmd == "/profiles":
            table = Table(title="Available LLM Profiles", box=ROUNDED)
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Model", style="yellow")
            table.add_column("Active", style="magenta")
            
            for profile_id, profile in self.profiles.items():
                is_active = "✓" if profile_id == self.current_profile_name else ""
                table.add_row(profile_id, profile.name, profile.get_model_name(), is_active)
            
            console.print(table)
        
        elif cmd == "/profile":
            if len(parts) < 2:
                console.print("[red]Usage: /profile <profile_id>[/red]")
            else:
                profile_id = parts[1].strip()
                if self.switch_profile(profile_id):
                    console.print(f"[green]✓ Switched to: {self.current_profile.name}[/green]")
                else:
                    console.print(f"[red]Profile not found: {profile_id}[/red]")
        
        elif cmd == "/image":
            if len(parts) < 2:
                console.print("[red]Usage: /image <path>[/red]")
            else:
                path = parts[1].strip()
                if Path(path).exists():
                    pending_attachments.append(self.content_handler.create_image_content(path))
                    console.print(f"[green]✓ Image attached: {path}[/green]")
                else:
                    console.print(f"[red]File not found: {path}[/red]")
        
        elif cmd == "/clear":
            console.print("[green]✓ Conversation cleared[/green]")
            conversation = []
        
        elif cmd == "/quit":
            console.print("[yellow]Goodbye! 👋[/yellow]")
            return True, [], conversation
        
        else:
            console.print(f"[red]Unknown command: {cmd}[/red]")
        
        return False, pending_attachments, conversation
    
    async def __aenter__(self):
        await self.connect_servers()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.aclose()


async def main():
    """Main entry point"""
    # Load .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    async with MCPUniversalClient() as client:
        await client.interactive_mode()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
