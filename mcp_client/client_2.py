# mcp_universal_client.py
"""
Universal MCP Client using FastMCP 2.0 with Multi-Provider LLM Support
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Complete MCP Feature Support:
✅ Tools - Call server tools with automatic LLM integration
✅ Resources - Read resource contents by URI
✅ Resource Templates - Dynamic resources with URI parameters
✅ Prompts - Use prompt templates with arguments
✅ All Transports - stdio, streamable-http, SSE
✅ Multimodal - Images, audio, PDFs across all LLM providers

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

# FastMCP 2.0 imports
from fastmcp import Client
from fastmcp.client.transports import (
    StdioTransport,
    StreamableHttpTransport,
    SSETransport,
)
from fastmcp.utilities.types import Image, Audio, File

# LiteLLM for universal LLM support
from litellm import acompletion
import litellm

# Rich for beautiful terminal UI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.tree import Tree

console = Console()


class MultimodalContentHandler:
    """Handles multimodal content (images, audio, PDFs) for all LLM providers."""
    
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
        """Create image content in OpenAI format (LiteLLM auto-converts)."""
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
            elif image_data.startswith('http://') or image_data.startswith('https://'):
                return {"type": "image_url", "image_url": {"url": image_data}}
            else:
                data_uri = f"data:{mime_type};base64,{image_data}"
                return {"type": "image_url", "image_url": {"url": data_uri}}
    
    @staticmethod
    def create_audio_content(audio_data: Union[str, bytes, Audio],
                           mime_type: str = "audio/mpeg") -> Dict:
        """Create audio content for transcription/analysis."""
        if isinstance(audio_data, Audio):
            data_uri = f"data:{audio_data.mimeType};base64,{base64.b64encode(audio_data.data).decode('utf-8')}"
            return {"type": "audio_url", "audio_url": {"url": data_uri}}
        
        elif isinstance(audio_data, bytes):
            encoded = base64.b64encode(audio_data).decode('utf-8')
            data_uri = f"data:{mime_type};base64,{encoded}"
            return {"type": "audio_url", "audio_url": {"url": data_uri}}
        
        elif isinstance(audio_data, str) and Path(audio_data).exists():
            mime_type, encoded = MultimodalContentHandler.encode_file_to_base64(audio_data)
            data_uri = f"data:{mime_type};base64,{encoded}"
            return {"type": "audio_url", "audio_url": {"url": data_uri}}
        
        return {"type": "text", "text": "[Audio content provided]"}
    
    @staticmethod
    def create_pdf_content(pdf_data: Union[str, bytes, File],
                         filename: str = "document.pdf") -> Dict:
        """Create PDF/document content for analysis."""
        if isinstance(pdf_data, File):
            encoded = base64.b64encode(pdf_data.data).decode('utf-8')
            data_uri = f"data:{pdf_data.mimeType};base64,{encoded}"
            return {
                "type": "file",
                "file": {"file_data": data_uri, "filename": filename}
            }
        
        elif isinstance(pdf_data, bytes):
            encoded = base64.b64encode(pdf_data).decode('utf-8')
            data_uri = f"data:application/pdf;base64,{encoded}"
            return {
                "type": "file",
                "file": {"file_data": data_uri, "filename": filename}
            }
        
        elif isinstance(pdf_data, str):
            if Path(pdf_data).exists():
                mime_type, encoded = MultimodalContentHandler.encode_file_to_base64(pdf_data)
                data_uri = f"data:{mime_type};base64,{encoded}"
                return {
                    "type": "file",
                    "file": {"file_data": data_uri, "filename": Path(pdf_data).name}
                }
            elif pdf_data.startswith('http'):
                return {
                    "type": "file",
                    "file": {"file_id": pdf_data, "filename": filename}
                }
        
        return {"type": "text", "text": f"[PDF document: {filename}]"}


class MCPUniversalClient:
    """Universal MCP Client using FastMCP 2.0 with complete feature support"""
    
    def __init__(self, config_path: str = "mcp_settings.json"):
        self.config_path = config_path
        self.clients: Dict[str, Client] = {}
        self.exit_stack = AsyncExitStack()
        
        # MCP capabilities storage
        self.all_tools = []
        self.all_resources = []
        self.all_resource_templates = []
        self.all_prompts = []
        
        # Multimodal content handler
        self.content_handler = MultimodalContentHandler()
        
        # Load configuration
        self.config = self._load_config()
        
        # LLM settings
        self.llm_provider = os.getenv("LLM_PROVIDER", "gemini")
        self.llm_model = os.getenv("LLM_MODEL", "gemini-2.5-pro")
        
        # Enable debug logging if needed
        if os.getenv("DEBUG"):
            litellm.set_verbose = True
    
    def _load_config(self) -> Dict:
        """Load MCP server configuration from JSON file"""
        config_file = Path(self.config_path)
        if not config_file.exists():
            console.print(f"[yellow]⚠ {self.config_path} not found. Using empty config.[/yellow]")
            return {"mcpServers": {}}
        
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def _create_transport(self, server_name: str, server_config: Dict):
        """Create appropriate transport based on configuration"""
        transport_type = server_config.get("transport", "stdio")
        
        if transport_type == "stdio":
            command = server_config.get("command")
            args = server_config.get("args", [])
            env = server_config.get("env", {})
            cwd = server_config.get("cwd")
            
            merged_env = os.environ.copy()
            merged_env.update(env)
            
            return StdioTransport(
                command=command,
                args=args,
                env=merged_env,
                cwd=cwd
            )
        
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
        """Connect to all configured MCP servers"""
        servers = self.config.get("mcpServers", {})
        
        if not servers:
            console.print("[yellow]⚠ No servers configured in mcp_settings.json[/yellow]")
            return
        
        console.print(f"\n[cyan]Connecting to {len(servers)} server(s)...[/cyan]\n")
        
        for server_name, server_config in servers.items():
            try:
                await self._connect_server(server_name, server_config)
                console.print(f"[green]✓[/green] Connected: [cyan]{server_name}[/cyan]")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed: [cyan]{server_name}[/cyan] - {str(e)[:60]}")
                if os.getenv("DEBUG"):
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    async def _connect_server(self, name: str, config: Dict):
        """Connect to a single MCP server using FastMCP 2.0"""
        transport = self._create_transport(name, config)
        client = Client(transport)
        await self.exit_stack.enter_async_context(client)
        self.clients[name] = client
        await self._load_server_capabilities(name, client)
    
    async def _load_server_capabilities(self, server_name: str, client: Client):
        """Load tools, resources, resource templates, and prompts"""
        try:
            init_result = client.initialize_result
            if init_result:
                console.print(f"  [dim]└─ {init_result.serverInfo.name} v{init_result.serverInfo.version}[/dim]")
        except:
            pass
        
        # Load tools
        try:
            tools = await client.list_tools()
            for tool in tools:
                self.all_tools.append({
                    "server": server_name,
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.inputSchema
                })
        except Exception as e:
            if os.getenv("DEBUG"):
                console.print(f"  [dim]└─ No tools: {e}[/dim]")
        
        # Load resources
        try:
            resources = await client.list_resources()
            for resource in resources:
                self.all_resources.append({
                    "server": server_name,
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mimeType": getattr(resource, 'mimeType', None)
                })
        except:
            pass
        
        # Load resource templates
        try:
            templates = await client.list_resource_templates()
            for template in templates:
                self.all_resource_templates.append({
                    "server": server_name,
                    "uriTemplate": template.uriTemplate,
                    "name": template.name,
                    "description": template.description,
                    "mimeType": getattr(template, 'mimeType', None)
                })
        except:
            pass
        
        # Load prompts
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
        """Format MCP tools for LLM tool calling"""
        llm_tools = []
        for tool in self.all_tools:
            llm_tools.append({
                "type": "function",
                "function": {
                    "name": f"{tool['server']}__{tool['name']}",
                    "description": tool['description'] or f"Tool from {tool['server']}",
                    "parameters": tool['schema']
                }
            })
        return llm_tools
    
    def _format_resources_for_llm(self) -> List[Dict]:
        """Format resources for LLM to understand what's available"""
        llm_resources = []
        for resource in self.all_resources:
            llm_resources.append({
                "type": "function",
                "function": {
                    "name": f"{resource['server']}__read_resource__{resource['uri'].replace('://', '_').replace('/', '_')}",
                    "description": f"Read resource: {resource['description'] or resource['name']} (URI: {resource['uri']})",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            })
        
        # Add resource templates
        for template in self.all_resource_templates:
            # Extract parameters from URI template {param}
            params = re.findall(r'\{(\w+)\}', template['uriTemplate'])
            properties = {
                param: {
                    "type": "string",
                    "description": f"Parameter for {param}"
                } for param in params
            }
            
            llm_resources.append({
                "type": "function",
                "function": {
                    "name": f"{template['server']}__read_template__{template['name'].replace(' ', '_')}",
                    "description": f"Read dynamic resource: {template['description'] or template['name']} (Template: {template['uriTemplate']})",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": params
                    }
                }
            })
        
        return llm_resources
    
    def _format_prompts_for_llm(self) -> List[Dict]:
        """Format prompts for LLM to understand what's available"""
        llm_prompts = []
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
            
            llm_prompts.append({
                "type": "function",
                "function": {
                    "name": f"{prompt['server']}__use_prompt__{prompt['name']}",
                    "description": f"Use prompt template: {prompt['description'] or prompt['name']}",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })
        
        return llm_prompts
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict) -> Any:
        """Call a tool on a specific MCP server"""
        client = self.clients.get(server_name)
        if not client:
            raise ValueError(f"Server {server_name} not connected")
        
        result = await client.call_tool(tool_name, arguments)
        return result
    
    async def read_resource(self, server_name: str, uri: str) -> Any:
        """Read a resource from a specific MCP server"""
        client = self.clients.get(server_name)
        if not client:
            raise ValueError(f"Server {server_name} not connected")
        
        result = await client.read_resource(uri)
        return result
    
    async def read_resource_template(self, server_name: str, uri_template: str, **params) -> Any:
        """Read a dynamic resource by filling in template parameters"""
        # Replace {param} with actual values
        uri = uri_template
        for param, value in params.items():
            uri = uri.replace(f"{{{param}}}", str(value))
        
        return await self.read_resource(server_name, uri)
    
    async def get_prompt(self, server_name: str, prompt_name: str, arguments: Dict = None) -> Any:
        """Get a prompt from a specific MCP server"""
        client = self.clients.get(server_name)
        if not client:
            raise ValueError(f"Server {server_name} not connected")
        
        result = await client.get_prompt(prompt_name, arguments or {})
        return result
    
    def _prepare_messages_for_llm(self, messages: List[Dict]) -> List[Dict]:
        """Prepare messages for LLM with multimodal support."""
        prepared = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            
            if isinstance(content, Image):
                image_content = self.content_handler.create_image_content(content)
                prepared.append({"role": role, "content": [image_content]})
            
            elif isinstance(content, Audio):
                audio_content = self.content_handler.create_audio_content(content)
                prepared.append({"role": role, "content": [audio_content]})
            
            elif isinstance(content, File):
                file_content = self.content_handler.create_pdf_content(content)
                prepared.append({"role": role, "content": [file_content]})
            
            elif isinstance(content, list):
                processed_content = []
                for item in content:
                    if isinstance(item, dict):
                        processed_content.append(item)
                    elif isinstance(item, Image):
                        processed_content.append(
                            self.content_handler.create_image_content(item)
                        )
                    elif isinstance(item, Audio):
                        processed_content.append(
                            self.content_handler.create_audio_content(item)
                        )
                    elif isinstance(item, File):
                        processed_content.append(
                            self.content_handler.create_pdf_content(item)
                        )
                    else:
                        processed_content.append({"type": "text", "text": str(item)})
                
                prepared.append({"role": role, "content": processed_content})
            
            elif isinstance(content, str):
                prepared.append({"role": role, "content": content})
            
            elif isinstance(content, dict):
                prepared.append({"role": role, "content": [content]})
            
            else:
                prepared.append({"role": role, "content": str(content)})
        
        return prepared
    
    def _extract_text_from_tool_result(self, result: Any) -> str:
        """Extract text content from MCP tool result"""
        if hasattr(result, 'content') and result.content:
            text_parts = []
            for content_item in result.content:
                if hasattr(content_item, 'text'):
                    text_parts.append(content_item.text)
                elif hasattr(content_item, 'type'):
                    if content_item.type == "text":
                        text_parts.append(getattr(content_item, 'text', ''))
                    elif content_item.type == "image":
                        text_parts.append(f"[Image: {getattr(content_item, 'mimeType', 'unknown')}]")
                    elif content_item.type == "audio":
                        text_parts.append(f"[Audio: {getattr(content_item, 'mimeType', 'unknown')}]")
                    elif content_item.type == "resource":
                        text_parts.append(f"[Resource: {getattr(content_item, 'uri', 'unknown')}]")
            return "\n".join(text_parts) if text_parts else str(result)
        return str(result)
    
    async def chat(self,
                   messages: List[Dict],
                   provider: Optional[str] = None,
                   model: Optional[str] = None,
                   system_prompt: Optional[str] = None,
                   system_prompt_file: Optional[str] = None,
                   use_tools: bool = True,
                   use_resources: bool = True,
                   use_prompts: bool = True,
                   max_iterations: int = 10,
                   temperature: float = 0.7,
                   max_tokens: Optional[int] = None) -> str:
        """
        Main chat function with automatic tool/resource/prompt calling.
        
        Args:
            messages: List of message dicts
            provider: LLM provider
            model: Model name
            system_prompt: Custom system prompt (overrides profile)
            system_prompt_file: Path to system prompt file
            use_tools: Enable automatic tool calling
            use_resources: Enable automatic resource reading
            use_prompts: Enable automatic prompt usage
            max_iterations: Max calling iterations
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
        """
        provider = provider or self.llm_provider
        model = model or self.llm_model
        model_name = f"{provider}/{model}"
        
        # Determine system prompt
        final_system_prompt = None
        
        # Priority: 1. Parameter, 2. File, 3. Default
        if system_prompt:
            final_system_prompt = system_prompt
        elif system_prompt_file:
            try:
                with open(system_prompt_file, 'r', encoding='utf-8') as f:
                    final_system_prompt = f.read()
            except FileNotFoundError:
                console.print(f"[yellow]⚠ System prompt file not found: {system_prompt_file}[/yellow]")
                final_system_prompt = "You are a helpful AI assistant."
            except Exception as e:
                console.print(f"[red]✗ Error reading system prompt: {e}[/red]")
                final_system_prompt = "You are a helpful AI assistant."
        else:
            final_system_prompt = """You are a helpful AI assistant with access to MCP tools and resources."""
        
        # Build prepared messages with system prompt FIRST
        prepared_messages = [
            {"role": "system", "content": final_system_prompt}
        ]
        
        # Now add user messages (this EXTENDS, not overwrites)
        prepared_messages.extend(self._prepare_messages_for_llm(messages))
        
        # Combine all available functions (tools, resources, prompts)
        all_functions = []
        if use_tools and self.all_tools:
            all_functions.extend(self._format_tools_for_llm())
        if use_resources and (self.all_resources or self.all_resource_templates):
            all_functions.extend(self._format_resources_for_llm())
        if use_prompts and self.all_prompts:
            all_functions.extend(self._format_prompts_for_llm())
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            try:
                response = await acompletion(
                    model=model_name,
                    messages=prepared_messages,
                    tools=all_functions if all_functions else None,
                    tool_choice="auto" if all_functions else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=120
                )
                
                assistant_message = response.choices[0].message
                
                msg_dict = {
                    "role": "assistant",
                    "content": assistant_message.content or ""
                }
                
                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                
                prepared_messages.append(msg_dict)
                
                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        try:
                            # Determine function type and execute
                            if "__read_resource__" in function_name:
                                # Static resource
                                parts = function_name.split("__")
                                server_name = parts[0]
                                
                                # Find the resource URI
                                resource_uri = None
                                for res in self.all_resources:
                                    if res['server'] == server_name and function_name.endswith(
                                        res['uri'].replace('://', '_').replace('/', '_')
                                    ):
                                        resource_uri = res['uri']
                                        break
                                
                                console.print(f"[cyan]→ Reading resource: [bold]{resource_uri}[/bold] from [bold]{server_name}[/bold][/cyan]")
                                result = await self.read_resource(server_name, resource_uri)
                                result_text = self._extract_text_from_tool_result(result)
                                console.print(f"[green]✓[/green] Resource content: [dim]{result_text[:150]}...[/dim]")
                            
                            elif "__read_template__" in function_name:
                                # Dynamic resource template
                                parts = function_name.split("__")
                                server_name = parts[0]
                                template_name = parts[2].replace('_', ' ')
                                
                                # Find the template
                                template_uri = None
                                for tmpl in self.all_resource_templates:
                                    if tmpl['server'] == server_name and tmpl['name'] == template_name:
                                        template_uri = tmpl['uriTemplate']
                                        break
                                
                                console.print(f"[cyan]→ Reading template: [bold]{template_uri}[/bold] with args {function_args}[/cyan]")
                                result = await self.read_resource_template(server_name, template_uri, **function_args)
                                result_text = self._extract_text_from_tool_result(result)
                                console.print(f"[green]✓[/green] Template content: [dim]{result_text[:150]}...[/dim]")
                            
                            elif "__use_prompt__" in function_name:
                                # Prompt template
                                parts = function_name.split("__")
                                server_name = parts[0]
                                prompt_name = parts[2]
                                
                                console.print(f"[cyan]→ Using prompt: [bold]{prompt_name}[/bold] from [bold]{server_name}[/bold][/cyan]")
                                result = await self.get_prompt(server_name, prompt_name, function_args)
                                
                                # Extract messages from prompt result
                                if hasattr(result, 'messages'):
                                    prompt_messages = []
                                    for msg in result.messages:
                                        if hasattr(msg, 'content'):
                                            if hasattr(msg.content, 'text'):
                                                prompt_messages.append(msg.content.text)
                                            else:
                                                prompt_messages.append(str(msg.content))
                                    result_text = "\n".join(prompt_messages)
                                else:
                                    result_text = str(result)
                                
                                console.print(f"[green]✓[/green] Prompt content: [dim]{result_text[:150]}...[/dim]")
                            
                            else:
                                # Regular tool
                                if "__" in function_name:
                                    server_name, tool_name = function_name.split("__", 1)
                                else:
                                    server_name = list(self.clients.keys())[0] if self.clients else "unknown"
                                    tool_name = function_name
                                
                                console.print(f"[cyan]→ Calling tool: [bold]{tool_name}[/bold] on [bold]{server_name}[/bold][/cyan]")
                                console.print(f"  [dim]Arguments: {json.dumps(function_args, indent=2)[:200]}...[/dim]")
                                
                                result = await self.call_tool(server_name, tool_name, function_args)
                                result_text = self._extract_text_from_tool_result(result)
                                console.print(f"[green]✓[/green] Tool result: [dim]{result_text[:150]}...[/dim]")
                        
                        except Exception as e:
                            result_text = f"Error executing function: {str(e)}"
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
                console.print(f"[red]✗ Error during chat: {e}[/red]")
                if os.getenv("DEBUG"):
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")
                return f"Error: {str(e)}"
        
        return "⚠ Max iterations reached"
    
    async def interactive_mode(self):
        """Interactive terminal chat mode with full MCP support"""
        console.print(Panel.fit(
            "[bold cyan]🤖 Universal MCP Client[/bold cyan]\n"
            "[dim]FastMCP 2.0 + LiteLLM Integration[/dim]\n\n"
            f"[green]LLM:[/green] {self.llm_provider}/{self.llm_model}\n"
            f"[yellow]Servers:[/yellow] {', '.join(self.clients.keys()) or 'None'}\n"
            f"[blue]Tools:[/blue] {len(self.all_tools)} | "
            f"[blue]Resources:[/blue] {len(self.all_resources)} | "
            f"[blue]Templates:[/blue] {len(self.all_resource_templates)} | "
            f"[blue]Prompts:[/blue] {len(self.all_prompts)}",
            title="🚀 Complete MCP Features",
            border_style="cyan"
        ))
        
        console.print("\n[bold]Commands:[/bold]")
        console.print("  [cyan]/tools[/cyan]      - List available tools")
        console.print("  [cyan]/resources[/cyan]  - List available resources")
        console.print("  [cyan]/templates[/cyan]  - List resource templates")
        console.print("  [cyan]/prompts[/cyan]    - List available prompts")
        console.print("  [cyan]/use-prompt <name> [args][/cyan] - Use a prompt template")
        console.print("  [cyan]/read-resource <uri>[/cyan] - Read a resource")
        console.print("  [cyan]/switch <provider> <model>[/cyan] - Change LLM")
        console.print("  [cyan]/image <path>[/cyan] - Attach image")
        console.print("  [cyan]/pdf <path>[/cyan]   - Attach PDF")
        console.print("  [cyan]/clear[/cyan]       - Clear conversation")
        console.print("  [cyan]/quit[/cyan]        - Exit\n")
        
        conversation = []
        pending_attachments = []
        
        while True:
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]")
                
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
                
                console.print("\n[bold blue]Assistant[/bold blue]:")
                response = await self.chat(conversation)
                console.print(Markdown(response))
                console.print()
                
                conversation.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type /quit to exit.[/yellow]")
            except Exception as e:
                console.print(f"[red]✗ Error: {e}[/red]")
                if os.getenv("DEBUG"):
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    async def _handle_command(self, command: str, pending_attachments: List, conversation: List) -> tuple[bool, List, List]:
        """Handle special commands. Returns (should_quit, updated_attachments, updated_conversation)."""
        parts = command.split(maxsplit=1)
        cmd = parts[0]
        
        if cmd == "/tools":
            if not self.all_tools:
                console.print("[yellow]No tools available[/yellow]")
                return False, pending_attachments, conversation
            
            table = Table(title="Available Tools", show_lines=True)
            table.add_column("Server", style="cyan", no_wrap=True)
            table.add_column("Tool Name", style="green")
            table.add_column("Description", style="white")
            
            for tool in self.all_tools:
                table.add_row(tool['server'], tool['name'], tool['description'] or "")
            console.print(table)
        
        elif cmd == "/resources":
            if not self.all_resources:
                console.print("[yellow]No resources available[/yellow]")
                return False, pending_attachments, conversation
            
            table = Table(title="Available Resources", show_lines=True)
            table.add_column("Server", style="cyan", no_wrap=True)
            table.add_column("URI", style="green")
            table.add_column("Name", style="blue")
            table.add_column("Description", style="white")
            
            for resource in self.all_resources:
                table.add_row(
                    resource['server'],
                    resource['uri'],
                    resource.get('name', ''),
                    resource.get('description', '')
                )
            console.print(table)
        
        elif cmd == "/templates":
            if not self.all_resource_templates:
                console.print("[yellow]No resource templates available[/yellow]")
                return False, pending_attachments, conversation
            
            table = Table(title="Available Resource Templates", show_lines=True)
            table.add_column("Server", style="cyan", no_wrap=True)
            table.add_column("URI Template", style="green")
            table.add_column("Name", style="blue")
            table.add_column("Description", style="white")
            
            for template in self.all_resource_templates:
                table.add_row(
                    template['server'],
                    template['uriTemplate'],
                    template.get('name', ''),
                    template.get('description', '')
                )
            console.print(table)
        
        elif cmd == "/prompts":
            if not self.all_prompts:
                console.print("[yellow]No prompts available[/yellow]")
                return False, pending_attachments, conversation
            
            table = Table(title="Available Prompts", show_lines=True)
            table.add_column("Server", style="cyan", no_wrap=True)
            table.add_column("Name", style="green")
            table.add_column("Arguments", style="yellow")
            table.add_column("Description", style="white")
            
            for prompt in self.all_prompts:
                args_str = ", ".join([
                    f"{arg.get('name')}{'*' if arg.get('required') else ''}"
                    for arg in prompt.get('arguments', [])
                ]) or "none"
                
                table.add_row(
                    prompt['server'],
                    prompt['name'],
                    args_str,
                    prompt.get('description', '')
                )
            console.print(table)
        
        elif cmd == "/use-prompt":
            if len(parts) < 2:
                console.print("[red]Usage: /use-prompt <name> [key=value ...][ /red]")
                return False, pending_attachments, conversation
            
            # Parse prompt name and arguments
            args_str = parts[1]
            prompt_parts = args_str.split()
            prompt_name = prompt_parts[0]
            
            # Parse key=value arguments
            prompt_args = {}
            for arg in prompt_parts[1:]:
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    prompt_args[key] = value
            
            # Find the prompt
            prompt_info = None
            for p in self.all_prompts:
                if p['name'] == prompt_name:
                    prompt_info = p
                    break
            
            if not prompt_info:
                console.print(f"[red]Prompt not found: {prompt_name}[/red]")
                return False, pending_attachments, conversation
            
            try:
                console.print(f"[cyan]Retrieving prompt: {prompt_name}...[/cyan]")
                result = await self.get_prompt(prompt_info['server'], prompt_name, prompt_args)
                
                # Extract and display prompt messages
                if hasattr(result, 'messages'):
                    console.print(f"[green]✓ Prompt retrieved with {len(result.messages)} message(s)[/green]\n")
                    for msg in result.messages:
                        role = getattr(msg, 'role', 'unknown')
                        if hasattr(msg, 'content'):
                            if hasattr(msg.content, 'text'):
                                content = msg.content.text
                            else:
                                content = str(msg.content)
                            console.print(Panel(Markdown(content), title=f"[bold]{role.upper()}[/bold]"))
                else:
                    console.print(result)
            except Exception as e:
                console.print(f"[red]✗ Error: {e}[/red]")
        
        elif cmd == "/read-resource":
            if len(parts) < 2:
                console.print("[red]Usage: /read-resource <uri>[/red]")
                return False, pending_attachments, conversation
            
            uri = parts[1].strip()
            
            # Find which server has this resource
            server_name = None
            for res in self.all_resources:
                if res['uri'] == uri:
                    server_name = res['server']
                    break
            
            if not server_name:
                console.print(f"[red]Resource not found: {uri}[/red]")
                return False, pending_attachments, conversation
            
            try:
                console.print(f"[cyan]Reading resource: {uri}...[/cyan]")
                result = await self.read_resource(server_name, uri)
                content = self._extract_text_from_tool_result(result)
                console.print(Panel(Markdown(content), title=f"[bold]Resource: {uri}[/bold]"))
            except Exception as e:
                console.print(f"[red]✗ Error: {e}[/red]")
        
        elif cmd == "/switch":
            if len(parts) < 2:
                console.print("[red]Usage: /switch <provider> <model>[/red]")
                console.print("[dim]Examples:[/dim]")
                console.print("  /switch anthropic claude-3-5-sonnet-20241022")
                console.print("  /switch groq llama-3.3-70b-versatile")
                console.print("  /switch gemini gemini-2.0-flash-exp")
                console.print("  /switch ollama llama3.2")
            else:
                provider_model = parts[1].split(maxsplit=1)
                if len(provider_model) >= 2:
                    self.llm_provider = provider_model[0]
                    self.llm_model = provider_model[1]
                    console.print(f"[green]✓ Switched to {self.llm_provider}/{self.llm_model}[/green]")
                else:
                    console.print("[red]Please provide both provider and model[/red]")
        
        elif cmd == "/image":
            if len(parts) < 2:
                console.print("[red]Usage: /image <path>[/red]")
            else:
                image_path = parts[1].strip()
                if Path(image_path).exists():
                    image_content = self.content_handler.create_image_content(image_path)
                    pending_attachments.append(image_content)
                    console.print(f"[green]✓ Image attached: {image_path}[/green]")
                else:
                    console.print(f"[red]File not found: {image_path}[/red]")
        
        elif cmd == "/pdf":
            if len(parts) < 2:
                console.print("[red]Usage: /pdf <path>[/red]")
            else:
                pdf_path = parts[1].strip()
                if Path(pdf_path).exists():
                    pdf_content = self.content_handler.create_pdf_content(pdf_path)
                    pending_attachments.append(pdf_content)
                    console.print(f"[green]✓ PDF attached: {pdf_path}[/green]")
                else:
                    console.print(f"[red]File not found: {pdf_path}[/red]")
        
        elif cmd == "/clear":
            console.print("[green]✓ Conversation cleared[/green]")
            conversation = []
        
        elif cmd == "/quit" or cmd == "/exit":
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
