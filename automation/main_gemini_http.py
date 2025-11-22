import os
import asyncio
import json
from fastmcp import Client
from google import genai

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
# FastMCP Client + Gemini Integration
# ============================================================================

# FastMCP client (HTTP mode)
mcp_client = Client("http://localhost:8000/mcp")

# Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

async def run_agent(user_task: str):
    """Run Gemini agent with MCP tools via HTTP"""
    print(f"\n🤖 Starting task with Gemini + MCP (HTTP): {user_task}\n")
    
    # Connect to MCP server
    async with mcp_client:
        # List available tools
        tools_list = await mcp_client.list_tools()
        print(f"📦 Loaded {len(tools_list)} MCP tools\n")
        
        # Build full prompt with system instruction
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser Task: {user_task}"
        
        # Call Gemini with MCP session
        # The key: Pass mcp_client.session directly as a tool!
        response = await gemini_client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.7,
                tools=[mcp_client.session],  # Pass FastMCP session directly!
            ),
        )
        
        # Extract response text
        final_text = response.text if hasattr(response, 'text') else str(response)
        
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
    print("🤖 GEMINI + MCP AUTOMATION AGENT (HTTP)")
    print(f"📡 Model: {MODEL_NAME}")
    print("="*60)
    
    user_task = input("\n💬 Enter your task: ")
    asyncio.run(run_agent(user_task))

if __name__ == "__main__":
    main()
