import os
import asyncio
import json
import google.generativeai as genai
from fastmcp import Client

# ============================================================================
# Configuration
# ============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY environment variable")

MODEL_NAME = "gemini-2.5-pro"

SYSTEM_PROMPT = """You are an advanced computer automation agent with vision capabilities.

Analyze screenshots, plan actions step-by-step, and use available MCP tools to complete tasks.

Rules:
- ALWAYS take screenshot first to see current state
- Be precise with coordinates based on visual analysis
- If app needs opening, use open_app first
- Verify after each action with new screenshot
- When complete, explain what was accomplished
"""

# ============================================================================
# FastMCP Client (STDIO) - Auto-launches mcp_tools.py
# ============================================================================

# Client will automatically launch the server as a subprocess
mcp_client = Client("mcp_tools.py")  # FastMCP infers STDIO transport from .py file

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

async def run_agent(user_task: str):
    """Run Gemini agent with MCP tools via STDIO"""
    print(f"\n🤖 Starting task with Gemini + MCP (STDIO): {user_task}\n")
    
    async with mcp_client:
        # List available tools
        tools_list = await mcp_client.list_tools()
        print(f"📦 Loaded {len(tools_list)} MCP tools\n")
        
        # Convert MCP tools to Gemini function declarations
        gemini_tools = []
        for tool in tools_list:
            # Convert MCP tool schema to Gemini format
            function_declaration = {
                "name": tool.name,
                "description": tool.description or f"Tool: {tool.name}",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}}
            }
            gemini_tools.append(function_declaration)
        
        # Create Gemini model with tools
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            tools=gemini_tools,
            system_instruction=SYSTEM_PROMPT
        )
        
        # Start chat
        chat = model.start_chat(enable_automatic_function_calling=True)
        
        # Send user task
        response = await chat.send_message_async(user_task)
        
        # Handle function calls via MCP
        while response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            function_name = function_call.name
            function_args = dict(function_call.args)
            
            print(f"🔧 Calling tool: {function_name}")
            print(f"   Args: {function_args}")
            
            # Call MCP tool
            result = await mcp_client.call_tool(function_name, function_args)
            
            # Send result back to Gemini
            response = await chat.send_message_async(
                genai.protos.Content(
                    parts=[genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=function_name,
                            response={"result": str(result)}
                        )
                    )]
                )
            )
        
        # Final response
        final_text = response.text
        
        final_result = {
            "task": user_task,
            "status": "complete",
            "model": MODEL_NAME,
            "agent_response": final_text
        }
        
        print("\n" + "="*60)
        print("FINAL RESULT:")
        print(json.dumps(final_result, indent=2))
        print("="*60)
        
        return final_result

def main():
    print("\n" + "="*60)
    print("🤖 GEMINI + MCP AUTOMATION AGENT (STDIO)")
    print(f"📡 Model: {MODEL_NAME}")
    print("="*60)
    
    user_task = input("\n💬 Enter your task: ")
    asyncio.run(run_agent(user_task))

if __name__ == "__main__":
    main()
