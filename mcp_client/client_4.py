# mcp_universal_client.py
"""
Universal MCP Client with Multiple LLM Profiles & System Prompts
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Features:
✅ Multiple LLM provider profiles with custom system prompts
✅ Automatic fallback chains for reliability
✅ Profile switching at runtime
✅ Role-based agent behavior (coding, research, creative, etc.)
✅ Complete MCP support (tools, resources, templates, prompts)
✅ Multimodal support (images, audio, PDFs)

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

# FastMCP 2.0
from fastmcp import Client
from fastmcp.client.transports import (
    StdioTransport,
    StreamableHttpTransport,
    SSETransport,
)
from fastmcp.utilities.types import Image, Audio, File

# LiteLLM
from litellm import acompletion
import litellm

# Rich UI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.markdown import Markdown
from rich.tree import Tree
from rich.box import ROUNDED

console = Console()


@dataclass
class LLMProfile:
    """LLM provider profile configuration"""
    name: str
    provider: str
    model: str
    system_prompt: str
    fallbacks: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 4096
    
    def get_model_name(self) -> str:
        """Get full model name for LiteLLM"""
        return f"{self.provider}/{self.model}"
    
    def get_fallback_models(self) -> List[str]:
        """Get list of fallback model names"""
        return [f"{fb['provider']}/{fb['model']}" for fb in self.fallbacks]


class MultimodalContentHandler:
    """Handles multimodal content for all LLM providers."""
    
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
        """Create image content in OpenAI format"""
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
    """Universal MCP Client with multiple LLM profiles"""
    
    def __init__(self, config_path: str = "mcp_settings.json"):
        self.config_path = config_path
        self.clients: Dict[str, Client] = {}
        self.exit_stack = AsyncExitStack()
        
        # MCP capabilities
        self.all_tools = []
        self.all_resources = []
        self.all_resource_templates = []
        self.all_prompts = []
        
        # Multimodal handler
        self.content_handler = MultimodalContentHandler()
        
        # Load configuration
        self.config = self._load_config()
        
        # LLM profiles
        self.profiles = self._load_llm_profiles()
        self.current_profile_name = self.config.get("defaultProfile", "default")
        self.current_profile = self.profiles.get(self.current_profile_name)
        
        # Global settings
        self.global_settings = self.config.get("globalSettings", {})
        
        # Enable debug
        if self.global_settings.get("debugMode") or os.getenv("DEBUG"):
            litellm.set_verbose = True
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file"""
        config_file = Path(self.config_path)
        if not config_file.exists():
            console.print(f"[yellow]⚠ {self.config_path} not found. Using defaults.[/yellow]")
            return {
                "mcpServers": {},
                "llmProfiles": {
                    "default": {
                        "name": "Default",
                        "provider": "ollama",
                        "model": "llama3.2",
                        "systemPrompt": "You are a helpful AI assistant.",
                        "fallbacks": []
                    }
                },
                "defaultProfile": "default"
            }
        
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def _load_llm_profiles(self) -> Dict[str, LLMProfile]:
        """Load LLM profiles from configuration"""
        profiles = {}
        for profile_id, profile_config in self.config.get("llmProfiles", {}).items():
            profiles[profile_id] = LLMProfile(
                name=profile_config.get("name", profile_id),
                provider=profile_config.get("provider", "ollama"),
                model=profile_config.get("model", "llama3.2"),
                system_prompt=profile_config.get("systemPrompt", "You are a helpful AI assistant."),
                fallbacks=profile_config.get("fallbacks", []),
                temperature=profile_config.get("temperature", 0.7),
                max_tokens=profile_config.get("maxTokens", 4096)
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
        """Create transport for MCP server"""
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
        servers = self.config.get("mcpServers", {})
        
        if not servers:
            console.print("[yellow]⚠ No MCP servers configured[/yellow]")
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
        """Load all capabilities from server"""
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
        
        # Static resources
        for res in self.all_resources:
            functions.append({
                "type": "function",
                "function": {
                    "name": f"{res['server']}__read_resource__{res['uri'].replace('://', '_').replace('/', '_')}",
                    "description": f"Read: {res['description'] or res['name']} (URI: {res['uri']})",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            })
        
        # Dynamic templates
        for tmpl in self.all_resource_templates:
            params = re.findall(r'\{(\w+)\}', tmpl['uriTemplate'])
            properties = {p: {"type": "string", "description": f"Parameter {p}"} for p in params}
            
            functions.append({
                "type": "function",
                "function": {
                    "name": f"{tmpl['server']}__read_template__{tmpl['name'].replace(' ', '_')}",
                    "description": f"Read dynamic: {tmpl['description'] or tmpl['name']} (Template: {tmpl['uriTemplate']})",
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
            
            if isinstance(content, (Image, Audio, File)):
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
                elif hasattr(item, 'type'):
                    if item.type == "text":
                        parts.append(getattr(item, 'text', ''))
            return "\n".join(parts) if parts else str(result)
        return str(result)
    
    async def chat(self,
                   messages: List[Dict],
                   profile_override: Optional[str] = None,
                   max_iterations: Optional[int] = None) -> str:
        """
        Main chat function with profile support and fallbacks.
        
        Args:
            messages: Conversation messages
            profile_override: Temporarily use different profile
            max_iterations: Override max iterations
        """
        # Use profile override or current profile
        profile = self.profiles.get(profile_override) if profile_override else self.current_profile
        if not profile:
            return "Error: No valid LLM profile configured"
        
        # Prepare messages with system prompt
        prepared_messages = [
            {"role": "system", "content": profile.system_prompt}
        ]
        prepared_messages.extend(self._prepare_messages_for_llm(messages))
        
        # Get all functions
        all_functions = []
        if self.global_settings.get("enableToolCalling", True) and self.all_tools:
            all_functions.extend(self._format_tools_for_llm())
        if self.global_settings.get("enableResources", True) and (self.all_resources or self.all_resource_templates):
            all_functions.extend(self._format_resources_for_llm())
        if self.global_settings.get("enablePrompts", True) and self.all_prompts:
            all_functions.extend(self._format_prompts_for_llm())
        
        max_iter = max_iterations or self.global_settings.get("maxIterations", 10)
        iteration = 0
        
        while iteration < max_iter:
            iteration += 1
            
            try:
                # Try primary model with fallbacks
                response = await acompletion(
                    model=profile.get_model_name(),
                    messages=prepared_messages,
                    tools=all_functions if all_functions else None,
                    tool_choice="auto" if all_functions else None,
                    temperature=profile.temperature,
                    max_tokens=profile.max_tokens,
                    fallbacks=profile.get_fallback_models(),
                    timeout=120
                )
                
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
                            # Dispatch to appropriate handler
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
                                console.print(f"[cyan]→ Reading template: {template_uri} with {function_args}[/cyan]")
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
                                # Regular tool
                                server_name, tool_name = function_name.split("__", 1) if "__" in function_name else (list(self.clients.keys())[0], function_name)
                                console.print(f"[cyan]→ Calling tool: {tool_name} on {server_name}[/cyan]")
                                result = await self.call_tool(server_name, tool_name, function_args)
                                result_text = self._extract_text_from_result(result)
                            
                            console.print(f"[green]✓[/green] Result: [dim]{result_text[:150]}...[/dim]")
                        
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
        # Display banner
        console.print(Panel.fit(
            f"[bold cyan]🤖 Universal MCP Client[/bold cyan]\n"
            f"[dim]FastMCP 2.0 + Multi-Profile LLM System[/dim]\n\n"
            f"[green]Current Profile:[/green] {self.current_profile.name}\n"
            f"[blue]Model:[/blue] {self.current_profile.get_model_name()}\n"
            f"[yellow]Servers:[/yellow] {len(self.clients)}\n"
            f"[magenta]Capabilities:[/magenta] {len(self.all_tools)} tools, "
            f"{len(self.all_resources)} resources, {len(self.all_prompts)} prompts",
            title="🚀 Multi-Agent System",
            border_style="cyan",
            box=ROUNDED
        ))
        
        # Commands
        console.print("\n[bold]Commands:[/bold]")
        console.print("  [cyan]/profiles[/cyan]     - List available LLM profiles")
        console.print("  [cyan]/profile <name>[/cyan] - Switch to different profile")
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
            table = Table(title="Available LLM Profiles", show_header=True, box=ROUNDED)
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Model", style="yellow")
            table.add_column("Fallbacks", style="blue")
            table.add_column("Active", style="magenta")
            
            for profile_id, profile in self.profiles.items():
                is_active = "✓" if profile_id == self.current_profile_name else ""
                fallbacks_str = ", ".join([f"{fb['provider']}/{fb['model']}" for fb in profile.fallbacks]) or "none"
                table.add_row(profile_id, profile.name, profile.get_model_name(), fallbacks_str, is_active)
            
            console.print(table)
        
        elif cmd == "/profile":
            if len(parts) < 2:
                console.print("[red]Usage: /profile <profile_id>[/red]")
                console.print("[dim]Use /profiles to see available profiles[/dim]")
            else:
                profile_id = parts[1].strip()
                if self.switch_profile(profile_id):
                    console.print(f"[green]✓ Switched to profile: {self.current_profile.name}[/green]")
                    console.print(f"  Model: {self.current_profile.get_model_name()}")
                    console.print(f"  System Prompt: {self.current_profile.system_prompt[:100]}...")
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
