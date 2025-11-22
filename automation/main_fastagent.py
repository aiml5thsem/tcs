import json
import asyncio
from fast_agent.core.fastagent import FastAgent
from tools import reset_action_history, get_action_history

# Create FastAgent app
fast = FastAgent("Computer Automation Agent")

# Define agent with MCP server
@fast.agent(
    name="computer_agent",
    instruction="""You are an advanced computer automation agent.

Analyze screenshots, plan actions, and use MCP tools to complete tasks.

Rules:
- Take screenshot first to see current state
- Be precise with coordinates
- Open apps when needed
- Verify after each action
- Explain completion""",
    servers=["computer_control"]  # Reference to MCP server in config
)

async def main():
    """Main entry point"""
    print("\n🤖 FAST-AGENT-MCP AUTOMATION AGENT\n" + "="*60)
    
    task = input("\n💬 Enter your task: ")
    reset_action_history()
    
    async with fast.run() as agent:
        # Run the agent
        result = await agent(task)
        
        final_result = {
            "task": task,
            "status": "complete",
            "agent_response": result,
            "actions_performed": get_action_history()
        }
        
        print("\n" + "="*60)
        print("FINAL RESULT:")
        print(json.dumps(final_result, indent=2))
        print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
