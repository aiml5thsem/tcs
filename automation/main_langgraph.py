import json
import os
import asyncio
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tools import reset_action_history, get_action_history

# Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

SYSTEM_PROMPT = """You are a computer automation agent with vision capabilities.
Analyze screenshots and use tools to complete tasks step-by-step."""

async def run_langgraph_agent(user_task: str):
    """Run LangGraph agent with MCP tools"""
    print(f"\n🤖 Starting task with LangGraph + MCP: {user_task}\n")
    reset_action_history()
    
    # Connect to MCP
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_tools.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Get MCP tools
            tools_list = await session.list_tools()
            print(f"📦 Loaded {len(tools_list.tools)} MCP tools")
            
            # Convert MCP tools to LangChain tools
            langchain_tools = []
            for tool_info in tools_list.tools:
                @tool(name=tool_info.name, description=tool_info.description or "")
                async def mcp_tool_wrapper(tool_name=tool_info.name, **kwargs):
                    result = await session.call_tool(tool_name, kwargs)
                    return result.content[0].text if result.content else ""
                
                langchain_tools.append(mcp_tool_wrapper)
            
            # Create agent
            agent = create_react_agent(llm, langchain_tools, state_modifier=SYSTEM_PROMPT)
            
            # Run agent
            result = await agent.ainvoke({"messages": [("human", user_task)]})
            
            final_result = {
                "task": user_task,
                "status": "complete",
                "agent_response": result["messages"][-1].content,
                "actions_performed": get_action_history()
            }
            
            print("\n" + "="*60)
            print("FINAL RESULT:")
            print(json.dumps(final_result, indent=2))
            print("="*60)

def main():
    print("\n🤖 LANGGRAPH + MCP AUTOMATION AGENT\n" + "="*60)
    task = input("\n💬 Enter your task: ")
    asyncio.run(run_langgraph_agent(task))

if __name__ == "__main__":
    main()
