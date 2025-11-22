import json
import os
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from mcp.client.http import http_client
from mcp.client.stdio import stdio_client
from tools import reset_action_history, get_action_history

# Model configuration
provider = GoogleProvider(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = GoogleModel("gemini-2.5-pro", provider=provider)

SYSTEM_PROMPT = """You are an advanced computer automation agent with vision capabilities.

Analyze screenshots, plan actions step-by-step, and use available MCP tools to complete tasks.

Rules:
- ALWAYS take screenshot first to see current state
- Be precise with coordinates based on visual analysis  
- If app needs opening, use open_app first
- Verify after each action with new screenshot
- When complete, explain what was accomplished
"""

# Create agent
agent = Agent(model=MODEL, system_prompt=SYSTEM_PROMPT)

async def run_with_mcp(user_task: str):
    """Run agent with MCP tools"""
    print(f"\n🤖 Starting task with Pydantic AI + MCP: {user_task}\n")
    reset_action_history()
    
    # Connect to MCP server
    server_params = HttpServerParameters(
        url="http://localhost:8000/mcp"   # or your actual host:port path
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Get available tools from MCP
            tools_list = await session.list_tools()
            print(f"📦 Loaded {len(tools_list.tools)} MCP tools")
            
            # Run agent (Pydantic AI will use MCP tools automatically)
            result = await agent.run(user_task)
            
            final_result = {
                "task": user_task,
                "status": "complete",
                "agent_response": result.data,
                "actions_performed": get_action_history()
            }
            
            print("\n" + "="*60)
            print("FINAL RESULT:")
            print(json.dumps(final_result, indent=2))
            print("="*60)
            
            return final_result

def main():
    print("\n🤖 PYDANTIC AI + MCP AUTOMATION AGENT\n" + "="*60)
    task = input("\n💬 Enter your task: ")
    asyncio.run(run_with_mcp(task))

if __name__ == "__main__":
    main()
