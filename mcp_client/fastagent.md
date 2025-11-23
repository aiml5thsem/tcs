# Advanced MCP Agent System - Complete Implementation

This document provides a comprehensive overview of the enhanced agent system with advanced orchestration capabilities, parallel execution, routing, chain workflows, and multi-agent coordination.

## Overview

The Advanced MCP Agent System extends your current implementation with:

- **Chain Workflows** - Sequential agent execution with state management
- **Parallel Workflows** - Fan-out/fan-in patterns for concurrent execution
- **Router Workflows** - Intelligent LLM-based routing to specialized agents
- **Orchestrator Pattern** - Dynamic task planning and delegation
- **Agent Registry** - Centralized agent management and discovery
- **Workflow State Management** - Context preservation across agent calls
- **Error Recovery** - Retry logic, fallbacks, and graceful degradation
- **Observability** - Built-in tracing, metrics, and logging
- **Resource Pooling** - Efficient connection and model management

## Architecture Comparison

### Your Current Implementation

**Strengths:**
- YAML/JSON configuration with environment variable support
- Multiple LLM profiles with custom system prompts
- Complete MCP protocol support (tools, resources, templates, prompts)
- Multimodal content handling (images, audio, PDFs)
- Interactive CLI with profile switching

**Limitations:**
- Single-agent execution model
- No workflow orchestration patterns
- Limited error recovery mechanisms
- No agent composition capabilities
- Sequential tool execution only

### Fast-Agent MCP Features

Fast-agent implements these advanced patterns:

1. **Chain Pattern** - Sequential agent composition
2. **Parallel Pattern** - Concurrent execution with aggregation
3. **Router Pattern** - LLM-driven agent selection
4. **Orchestrator Pattern** - Dynamic task planning
5. **Human-in-the-Loop** - Interactive intervention points
6. **Sampling** - MCP server-initiated LLM calls
7. **Agent-as-Server** - Expose agents as MCP servers

## Implementation Guide

### Core Components to Add

#### 1. Agent Registry

class AgentRegistry:
    """Central registry for agent discovery and management"""
    
    def __init__(self):
        self.agents: Dict[str, 'BaseAgent'] = {}
        self.workflows: Dict[str, 'BaseWorkflow'] = {}
        self._lock = asyncio.Lock()
    
    async def register_agent(self, name: str, agent: 'BaseAgent'):
        """Register an agent with the registry"""
        async with self._lock:
            if name in self.agents:
                raise ValueError(f"Agent {name} already registered")
            self.agents[name] = agent
            console.print(f"[green]✓[/green] Registered agent: [cyan]{name}[/cyan]")
    
    async def get_agent(self, name: str) -> 'BaseAgent':
        """Retrieve agent by name"""
        if name not in self.agents:
            raise ValueError(f"Agent {name} not found")
        return self.agents[name]
    
    def list_agents(self) -> List[str]:
        """List all registered agents"""
        return list(self.agents.keys())

#### 2. Base Agent Class

@dataclass
class AgentCapabilities:
    """Define what an agent can do"""
    tools: List[str]
    resources: List[str]
    prompts: List[str]
    supports_multimodal: bool = False
    max_tokens: int = 4096

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(
        self,
        name: str,
        instruction: str,
        profile: LLMProfile,
        servers: List[str] = None,
        capabilities: AgentCapabilities = None
    ):
        self.name = name
        self.instruction = instruction
        self.profile = profile
        self.servers = servers or []
        self.capabilities = capabilities or AgentCapabilities([], [], [])
        self.conversation_history: List[Dict] = []
        self.metadata: Dict = {}
    
    async def execute(
        self,
        message: Union[str, Dict, List[Dict]],
        context: Optional[Dict] = None
    ) -> AgentResponse:
        """Execute agent with message and optional context"""
        raise NotImplementedError
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

#### 3. Chain Workflow

class ChainWorkflow:
    """Sequential agent execution workflow"""
    
    def __init__(
        self,
        name: str,
        sequence: List[str],
        registry: AgentRegistry,
        cumulative: bool = False,
        continue_with_final: bool = False
    ):
        self.name = name
        self.sequence = sequence
        self.registry = registry
        self.cumulative = cumulative
        self.continue_with_final = continue_with_final
    
    async def execute(
        self,
        initial_message: str,
        context: Optional[Dict] = None
    ) -> ChainResult:
        """Execute chain of agents sequentially"""
        results = []
        current_message = initial_message
        accumulated_context = context or {}
        
        console.print(f"\n[bold cyan]→ Chain Workflow:[/bold cyan] {self.name}")
        console.print(f"[dim]  Sequence: {' → '.join(self.sequence)}[/dim]\n")
        
        for i, agent_name in enumerate(self.sequence, 1):
            try:
                agent = await self.registry.get_agent(agent_name)
                console.print(f"[cyan]  [{i}/{len(self.sequence)}] Executing:[/cyan] {agent_name}")
                
                # Execute agent
                response = await agent.execute(
                    current_message,
                    context=accumulated_context
                )
                
                results.append({
                    'agent': agent_name,
                    'response': response,
                    'step': i
                })
                
                # Update message for next agent
                if self.cumulative:
                    # Accumulate all previous responses
                    current_message = f"{current_message}\n\n--- Response from {agent_name} ---\n{response.text}"
                else:
                    # Only pass the last response
                    current_message = response.text
                
                # Update context with agent metadata
                accumulated_context[f'agent_{i}_output'] = response.text
                
                console.print(f"[green]  ✓[/green] Completed: {agent_name}\n")
                
            except Exception as e:
                console.print(f"[red]  ✗[/red] Failed: {agent_name} - {e}\n")
                return ChainResult(
                    success=False,
                    results=results,
                    error=str(e),
                    failed_at=agent_name
                )
        
        return ChainResult(
            success=True,
            results=results,
            final_output=results[-1]['response'].text if results else None
        )

#### 4. Parallel Workflow

class ParallelWorkflow:
    """Fan-out/fan-in parallel execution workflow"""
    
    def __init__(
        self,
        name: str,
        fan_out: List[str],
        fan_in: Optional[str] = None,
        registry: AgentRegistry,
        include_request: bool = True,
        max_concurrency: int = 10
    ):
        self.name = name
        self.fan_out = fan_out
        self.fan_in = fan_in
        self.registry = registry
        self.include_request = include_request
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
    
    async def _execute_agent(
        self,
        agent_name: str,
        message: str,
        context: Dict
    ) -> Tuple[str, AgentResponse]:
        """Execute single agent with semaphore control"""
        async with self.semaphore:
            try:
                agent = await self.registry.get_agent(agent_name)
                console.print(f"[cyan]  → Starting:[/cyan] {agent_name}")
                
                response = await agent.execute(message, context=context)
                
                console.print(f"[green]  ✓ Completed:[/green] {agent_name}")
                return agent_name, response
                
            except Exception as e:
                console.print(f"[red]  ✗ Failed:[/red] {agent_name} - {e}")
                return agent_name, AgentResponse(
                    text=f"Error: {str(e)}",
                    success=False,
                    error=str(e)
                )
    
    async def execute(
        self,
        message: str,
        context: Optional[Dict] = None
    ) -> ParallelResult:
        """Execute agents in parallel with fan-in aggregation"""
        context = context or {}
        
        console.print(f"\n[bold cyan]→ Parallel Workflow:[/bold cyan] {self.name}")
        console.print(f"[dim]  Fan-out: {len(self.fan_out)} agents | Max concurrency: {self.max_concurrency}[/dim]\n")
        
        # Fan-out: Execute all agents in parallel
        tasks = [
            self._execute_agent(agent_name, message, context)
            for agent_name in self.fan_out
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Collect successful results
        agent_outputs = {}
        failed_agents = []
        
        for agent_name, response in results:
            if response.success:
                agent_outputs[agent_name] = response.text
            else:
                failed_agents.append(agent_name)
        
        console.print(f"\n[cyan]  Fan-out complete:[/cyan] {len(agent_outputs)}/{len(self.fan_out)} successful\n")
        
        # Fan-in: Aggregate results if fan_in agent specified
        final_output = None
        if self.fan_in:
            try:
                fan_in_agent = await self.registry.get_agent(self.fan_in)
                
                # Prepare aggregation message
                aggregation_parts = []
                if self.include_request:
                    aggregation_parts.append(f"Original Request: {message}\n")
                
                aggregation_parts.append("--- Agent Responses ---")
                for agent_name, output in agent_outputs.items():
                    aggregation_parts.append(f"\n[{agent_name}]:\n{output}")
                
                aggregation_message = "\n".join(aggregation_parts)
                
                console.print(f"[cyan]  → Fan-in:[/cyan] {self.fan_in}")
                fan_in_response = await fan_in_agent.execute(
                    aggregation_message,
                    context={'parallel_results': agent_outputs}
                )
                
                final_output = fan_in_response.text
                console.print(f"[green]  ✓ Fan-in complete[/green]\n")
                
            except Exception as e:
                console.print(f"[red]  ✗ Fan-in failed:[/red] {e}\n")
                final_output = "\n\n".join(f"[{k}]: {v}" for k, v in agent_outputs.items())
        else:
            final_output = "\n\n".join(f"[{k}]: {v}" for k, v in agent_outputs.items())
        
        return ParallelResult(
            success=len(failed_agents) == 0,
            agent_outputs=agent_outputs,
            failed_agents=failed_agents,
            final_output=final_output
        )

#### 5. Router Workflow

class RouterWorkflow:
    """LLM-based intelligent routing to specialized agents"""
    
    def __init__(
        self,
        name: str,
        agents: List[str],
        registry: AgentRegistry,
        routing_instruction: Optional[str] = None,
        routing_profile: Optional[LLMProfile] = None,
        use_history: bool = False
    ):
        self.name = name
        self.agents = agents
        self.registry = registry
        self.routing_instruction = routing_instruction
        self.routing_profile = routing_profile
        self.use_history = use_history
        self.routing_history: List[Dict] = []
    
    async def _generate_routing_decision(
        self,
        message: str,
        agent_descriptions: Dict[str, str]
    ) -> str:
        """Use LLM to decide which agent to route to"""
        
        # Build routing prompt
        agent_list = []
        for agent_name, description in agent_descriptions.items():
            agent_list.append(f"- {agent_name}: {description}")
        
        routing_prompt = f"""You are a routing assistant. Analyze the user's request and select the most appropriate agent to handle it.

Available Agents:
{chr(10).join(agent_list)}

User Request: {message}

{self.routing_instruction or ''}

Respond with ONLY the agent name (e.g., "agent1"). Do not include any explanation."""
        
        # Call LLM for routing decision
        try:
            completion_kwargs = self.routing_profile.get_completion_kwargs()
            completion_kwargs["messages"] = [
                {"role": "system", "content": "You are a routing assistant that selects the best agent for a task."},
                {"role": "user", "content": routing_prompt}
            ]
            completion_kwargs["temperature"] = 0.1  # Low temperature for consistent routing
            completion_kwargs["max_tokens"] = 50
            
            response = await acompletion(**completion_kwargs)
            selected_agent = response.choices[0].message.content.strip()
            
            # Validate selection
            if selected_agent not in agent_descriptions:
                console.print(f"[yellow]⚠ Invalid routing decision: {selected_agent}. Using first agent.[/yellow]")
                selected_agent = self.agents[0]
            
            return selected_agent
            
        except Exception as e:
            console.print(f"[red]✗ Routing decision failed:[/red] {e}")
            return self.agents[0]  # Fallback to first agent
    
    async def execute(
        self,
        message: str,
        context: Optional[Dict] = None
    ) -> RouterResult:
        """Route message to appropriate agent"""
        
        console.print(f"\n[bold cyan]→ Router Workflow:[/bold cyan] {self.name}")
        console.print(f"[dim]  Available agents: {', '.join(self.agents)}[/dim]\n")
        
        # Get agent descriptions
        agent_descriptions = {}
        for agent_name in self.agents:
            agent = await self.registry.get_agent(agent_name)
            agent_descriptions[agent_name] = agent.instruction
        
        # Determine routing
        console.print(f"[cyan]  → Routing decision...[/cyan]")
        selected_agent_name = await self._generate_routing_decision(message, agent_descriptions)
        console.print(f"[green]  ✓ Routed to:[/green] {selected_agent_name}\n")
        
        # Execute selected agent
        selected_agent = await self.registry.get_agent(selected_agent_name)
        response = await selected_agent.execute(message, context=context)
        
        # Track routing history
        if self.use_history:
            self.routing_history.append({
                'message': message,
                'selected_agent': selected_agent_name,
                'timestamp': asyncio.get_event_loop().time()
            })
        
        return RouterResult(
            success=response.success,
            selected_agent=selected_agent_name,
            response=response,
            routing_confidence=0.9  # Could be enhanced with actual confidence scores
        )

#### 6. Orchestrator Pattern

class OrchestratorWorkflow:
    """Dynamic task planning and delegation"""
    
    def __init__(
        self,
        name: str,
        agents: List[str],
        registry: AgentRegistry,
        orchestrator_profile: LLMProfile,
        plan_type: str = "iterative"  # "full" or "iterative"
    ):
        self.name = name
        self.agents = agents
        self.registry = registry
        self.orchestrator_profile = orchestrator_profile
        self.plan_type = plan_type
    
    async def _generate_plan(
        self,
        task: str,
        agent_capabilities: Dict[str, str]
    ) -> List[Dict]:
        """Generate execution plan using LLM"""
        
        agent_list = []
        for agent_name, capabilities in agent_capabilities.items():
            agent_list.append(f"- {agent_name}: {capabilities}")
        
        planning_prompt = f"""You are an AI orchestrator. Break down the following complex task into subtasks and assign them to available agents.

Available Agents:
{chr(10).join(agent_list)}

Complex Task: {task}

Generate a step-by-step execution plan. For each step, specify:
1. Step number
2. Agent to use
3. Specific instruction for that agent
4. Dependencies on previous steps (if any)

Respond in JSON format:
{{
  "steps": [
    {{
      "step": 1,
      "agent": "agent_name",
      "instruction": "specific task",
      "depends_on": []
    }},
    ...
  ]
}}"""
        
        try:
            completion_kwargs = self.orchestrator_profile.get_completion_kwargs()
            completion_kwargs["messages"] = [
                {"role": "system", "content": "You are an orchestration planner. Generate structured execution plans in JSON format."},
                {"role": "user", "content": planning_prompt}
            ]
            completion_kwargs["temperature"] = 0.3
            
            response = await acompletion(**completion_kwargs)
            plan_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            import json
            import re
            json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
                return plan_data.get('steps', [])
            else:
                raise ValueError("No valid JSON plan found")
                
        except Exception as e:
            console.print(f"[red]✗ Plan generation failed:[/red] {e}")
            # Fallback: simple sequential execution
            return [{
                "step": i + 1,
                "agent": agent_name,
                "instruction": task,
                "depends_on": [i] if i > 0 else []
            } for i, agent_name in enumerate(self.agents)]
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict] = None
    ) -> OrchestratorResult:
        """Execute orchestrated workflow"""
        
        console.print(f"\n[bold cyan]→ Orchestrator Workflow:[/bold cyan] {self.name}")
        console.print(f"[dim]  Plan type: {self.plan_type} | Available agents: {len(self.agents)}[/dim]\n")
        
        # Get agent capabilities
        agent_capabilities = {}
        for agent_name in self.agents:
            agent = await self.registry.get_agent(agent_name)
            agent_capabilities[agent_name] = agent.instruction
        
        # Generate plan
        console.print(f"[cyan]  → Generating execution plan...[/cyan]")
        plan = await self._generate_plan(task, agent_capabilities)
        console.print(f"[green]  ✓ Plan generated:[/green] {len(plan)} steps\n")
        
        # Display plan
        console.print("[bold]Execution Plan:[/bold]")
        for step in plan:
            console.print(f"  {step['step']}. [{step['agent']}] {step['instruction']}")
        console.print()
        
        # Execute plan
        results = {}
        step_outputs = {}
        
        for step in plan:
            step_num = step['step']
            agent_name = step['agent']
            instruction = step['instruction']
            dependencies = step.get('depends_on', [])
            
            # Wait for dependencies
            for dep_step in dependencies:
                if dep_step not in step_outputs:
                    console.print(f"[yellow]⚠ Dependency not met: step {dep_step}[/yellow]")
                    continue
            
            # Prepare instruction with dependency outputs
            enriched_instruction = instruction
            if dependencies:
                dep_context = "\n\n".join([
                    f"--- Output from step {dep} ---\n{step_outputs[dep]}"
                    for dep in dependencies if dep in step_outputs
                ])
                enriched_instruction = f"{instruction}\n\nContext from previous steps:\n{dep_context}"
            
            # Execute step
            try:
                console.print(f"[cyan]  → Step {step_num}:[/cyan] {agent_name}")
                agent = await self.registry.get_agent(agent_name)
                response = await agent.execute(
                    enriched_instruction,
                    context=context
                )
                
                step_outputs[step_num] = response.text
                results[step_num] = {
                    'agent': agent_name,
                    'instruction': instruction,
                    'output': response.text,
                    'success': response.success
                }
                
                console.print(f"[green]  ✓ Step {step_num} complete[/green]\n")
                
            except Exception as e:
                console.print(f"[red]  ✗ Step {step_num} failed:[/red] {e}\n")
                results[step_num] = {
                    'agent': agent_name,
                    'instruction': instruction,
                    'output': None,
                    'success': False,
                    'error': str(e)
                }
        
        # Aggregate final result
        final_output = step_outputs.get(len(plan), "")
        
        return OrchestratorResult(
            success=all(r['success'] for r in results.values()),
            plan=plan,
            results=results,
            final_output=final_output
        )

#### 7. Enhanced MCPUniversalClient

class EnhancedMCPClient(MCPUniversalClient):
    """Enhanced MCP client with advanced orchestration capabilities"""
    
    def __init__(
        self,
        mcp_config_path: str = "mcp_servers",
        llm_config_path: str = "llm_config"
    ):
        super().__init__(mcp_config_path, llm_config_path)
        
        # Advanced components
        self.registry = AgentRegistry()
        self.workflows: Dict[str, Union[ChainWorkflow, ParallelWorkflow, RouterWorkflow, OrchestratorWorkflow]] = {}
        self.metrics = MetricsCollector()
        self.error_handler = ErrorRecoveryManager()
    
    async def register_agent(
        self,
        name: str,
        instruction: str,
        profile_name: Optional[str] = None,
        servers: List[str] = None
    ):
        """Register a new agent"""
        profile = self.profiles.get(profile_name or self.current_profile_name)
        
        agent = MCPAgent(
            name=name,
            instruction=instruction,
            profile=profile,
            servers=servers or [],
            mcp_client=self
        )
        
        await self.registry.register_agent(name, agent)
    
    async def create_chain(
        self,
        name: str,
        sequence: List[str],
        cumulative: bool = False
    ):
        """Create a chain workflow"""
        workflow = ChainWorkflow(
            name=name,
            sequence=sequence,
            registry=self.registry,
            cumulative=cumulative
        )
        self.workflows[name] = workflow
        console.print(f"[green]✓[/green] Created chain workflow: [cyan]{name}[/cyan]")
    
    async def create_parallel(
        self,
        name: str,
        fan_out: List[str],
        fan_in: Optional[str] = None,
        max_concurrency: int = 10
    ):
        """Create a parallel workflow"""
        workflow = ParallelWorkflow(
            name=name,
            fan_out=fan_out,
            fan_in=fan_in,
            registry=self.registry,
            max_concurrency=max_concurrency
        )
        self.workflows[name] = workflow
        console.print(f"[green]✓[/green] Created parallel workflow: [cyan]{name}[/cyan]")
    
    async def create_router(
        self,
        name: str,
        agents: List[str],
        routing_instruction: Optional[str] = None
    ):
        """Create a router workflow"""
        workflow = RouterWorkflow(
            name=name,
            agents=agents,
            registry=self.registry,
            routing_instruction=routing_instruction,
            routing_profile=self.current_profile
        )
        self.workflows[name] = workflow
        console.print(f"[green]✓[/green] Created router workflow: [cyan]{name}[/cyan]")
    
    async def create_orchestrator(
        self,
        name: str,
        agents: List[str],
        plan_type: str = "iterative"
    ):
        """Create an orchestrator workflow"""
        workflow = OrchestratorWorkflow(
            name=name,
            agents=agents,
            registry=self.registry,
            orchestrator_profile=self.current_profile,
            plan_type=plan_type
        )
        self.workflows[name] = workflow
        console.print(f"[green]✓[/green] Created orchestrator workflow: [cyan]{name}[/cyan]")
    
    async def execute_workflow(
        self,
        workflow_name: str,
        message: str,
        context: Optional[Dict] = None
    ):
        """Execute a workflow by name"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} not found")
        
        workflow = self.workflows[workflow_name]
        
        # Track metrics
        start_time = time.time()
        
        try:
            result = await workflow.execute(message, context=context)
            elapsed = time.time() - start_time
            
            self.metrics.record_workflow_execution(
                workflow_name=workflow_name,
                success=result.success,
                duration=elapsed
            )
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.metrics.record_workflow_execution(
                workflow_name=workflow_name,
                success=False,
                duration=elapsed,
                error=str(e)
            )
            raise

### Usage Examples

#### Example 1: Chain Workflow

async def example_chain():
    """Chain multiple agents for sequential processing"""
    
    async with EnhancedMCPClient() as client:
        # Register agents
        await client.register_agent(
            "url_fetcher",
            "Fetch and summarize content from a URL",
            servers=["fetch"]
        )
        
        await client.register_agent(
            "social_media",
            "Create engaging social media posts (max 280 chars)",
            servers=[]
        )
        
        await client.register_agent(
            "translator",
            "Translate text to Spanish",
            servers=[]
        )
        
        # Create chain
        await client.create_chain(
            "content_pipeline",
            sequence=["url_fetcher", "social_media", "translator"],
            cumulative=False
        )
        
        # Execute
        result = await client.execute_workflow(
            "content_pipeline",
            "https://fast-agent.ai"
        )
        
        console.print(Panel(result.final_output, title="Final Result"))

#### Example 2: Parallel Workflow

async def example_parallel():
    """Execute multiple agents in parallel with aggregation"""
    
    async with EnhancedMCPClient() as client:
        # Register specialized agents
        await client.register_agent("analyst_financial", "Analyze from financial perspective")
        await client.register_agent("analyst_technical", "Analyze from technical perspective")
        await client.register_agent("analyst_market", "Analyze from market perspective")
        await client.register_agent("synthesizer", "Synthesize multiple analyses into executive summary")
        
        # Create parallel workflow
        await client.create_parallel(
            "multi_perspective_analysis",
            fan_out=["analyst_financial", "analyst_technical", "analyst_market"],
            fan_in="synthesizer",
            max_concurrency=3
        )
        
        # Execute
        result = await client.execute_workflow(
            "multi_perspective_analysis",
            "Analyze the impact of AI on the job market"
        )
        
        console.print(Panel(result.final_output, title="Synthesized Analysis"))

#### Example 3: Router Workflow

async def example_router():
    """Route queries to specialized agents"""
    
    async with EnhancedMCPClient() as client:
        # Register specialized agents
        await client.register_agent(
            "code_expert",
            "Expert in programming, debugging, and software architecture",
            servers=["filesystem"]
        )
        
        await client.register_agent(
            "research_expert",
            "Expert in research, fact-checking, and information gathering",
            servers=["fetch", "browser"]
        )
        
        await client.register_agent(
            "creative_expert",
            "Expert in creative writing, storytelling, and content creation"
        )
        
        # Create router
        await client.create_router(
            "intelligent_router",
            agents=["code_expert", "research_expert", "creative_expert"],
            routing_instruction="Route based on query type and required expertise"
        )
        
        # Execute multiple queries
        queries = [
            "Fix this Python bug: list index out of range",
            "Research the latest AI trends in 2025",
            "Write a creative story about time travel"
        ]
        
        for query in queries:
            result = await client.execute_workflow("intelligent_router", query)
            console.print(f"\n[cyan]Query:[/cyan] {query}")
            console.print(f"[green]Routed to:[/green] {result.selected_agent}")
            console.print(Panel(result.response.text, title="Response"))

#### Example 4: Orchestrator Pattern

async def example_orchestrator():
    """Dynamic task planning and execution"""
    
    async with EnhancedMCPClient() as client:
        # Register agents with different capabilities
        await client.register_agent("researcher", "Research topics and gather information", servers=["fetch"])
        await client.register_agent("analyzer", "Analyze data and extract insights")
        await client.register_agent("writer", "Write clear, structured content")
        await client.register_agent("reviewer", "Review and improve content quality")
        
        # Create orchestrator
        await client.create_orchestrator(
            "content_creation",
            agents=["researcher", "analyzer", "writer", "reviewer"],
            plan_type="iterative"
        )
        
        # Execute complex task
        result = await client.execute_workflow(
            "content_creation",
            "Create a comprehensive blog post about quantum computing for beginners"
        )
        
        console.print(Panel(result.final_output, title="Final Content"))

#### Example 5: Nested Workflows

async def example_nested():
    """Combine multiple workflow patterns"""
    
    async with EnhancedMCPClient() as client:
        # Register agents
        await client.register_agent("data_collector", "Collect data from multiple sources", servers=["fetch"])
        await client.register_agent("validator", "Validate data quality and consistency")
        await client.register_agent("transformer", "Transform and normalize data")
        
        await client.register_agent("model_a", "ML model for prediction type A")
        await client.register_agent("model_b", "ML model for prediction type B")
        await client.register_agent("model_c", "ML model for prediction type C")
        await client.register_agent("ensemble", "Combine predictions from multiple models")
        
        await client.register_agent("reporter", "Generate detailed reports")
        
        # Create workflows
        # 1. Data preparation chain
        await client.create_chain(
            "data_prep",
            sequence=["data_collector", "validator", "transformer"]
        )
        
        # 2. Parallel model inference
        await client.create_parallel(
            "model_ensemble",
            fan_out=["model_a", "model_b", "model_c"],
            fan_in="ensemble"
        )
        
        # 3. Master orchestrator that uses other workflows
        await client.register_agent(
            "workflow_coordinator",
            "Coordinate data prep, model inference, and reporting"
        )
        
        # Execute nested workflow manually
        # Step 1: Data preparation
        data_result = await client.execute_workflow("data_prep", "Collect sales data for Q4 2024")
        
        # Step 2: Model inference
        model_result = await client.execute_workflow("model_ensemble", data_result.final_output)
        
        # Step 3: Reporting
        agent = await client.registry.get_agent("reporter")
        report = await agent.execute(
            f"Generate report based on predictions: {model_result.final_output}"
        )
        
        console.print(Panel(report.text, title="Final Report"))

## Advanced Features

### 1. Error Recovery

class ErrorRecoveryManager:
    """Manage error recovery and retry logic"""
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_multiplier: float = 2.0,
        initial_delay: float = 1.0
    ):
        self.max_retries = max_retries
        self.backoff_multiplier = backoff_multiplier
        self.initial_delay = initial_delay
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with exponential backoff retry"""
        delay = self.initial_delay
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    console.print(f"[yellow]⚠ Attempt {attempt + 1} failed, retrying in {delay}s...[/yellow]")
                    await asyncio.sleep(delay)
                    delay *= self.backoff_multiplier
        
        raise last_exception

### 2. Metrics Collection

class MetricsCollector:
    """Collect and report workflow metrics"""
    
    def __init__(self):
        self.workflow_executions: List[Dict] = []
        self.agent_calls: List[Dict] = []
    
    def record_workflow_execution(
        self,
        workflow_name: str,
        success: bool,
        duration: float,
        error: Optional[str] = None
    ):
        """Record workflow execution metrics"""
        self.workflow_executions.append({
            'workflow': workflow_name,
            'success': success,
            'duration': duration,
            'error': error,
            'timestamp': time.time()
        })
    
    def get_workflow_stats(self, workflow_name: str) -> Dict:
        """Get statistics for a specific workflow"""
        executions = [e for e in self.workflow_executions if e['workflow'] == workflow_name]
        
        if not executions:
            return {}
        
        successful = [e for e in executions if e['success']]
        
        return {
            'total_executions': len(executions),
            'successful': len(successful),
            'failed': len(executions) - len(successful),
            'success_rate': len(successful) / len(executions),
            'avg_duration': sum(e['duration'] for e in executions) / len(executions),
            'min_duration': min(e['duration'] for e in executions),
            'max_duration': max(e['duration'] for e in executions)
        }
    
    def display_stats(self):
        """Display metrics in table format"""
        table = Table(title="Workflow Statistics", box=ROUNDED)
        table.add_column("Workflow", style="cyan")
        table.add_column("Executions", style="white")
        table.add_column("Success Rate", style="green")
        table.add_column("Avg Duration", style="yellow")
        
        workflows = set(e['workflow'] for e in self.workflow_executions)
        
        for workflow in workflows:
            stats = self.get_workflow_stats(workflow)
            table.add_row(
                workflow,
                str(stats['total_executions']),
                f"{stats['success_rate']:.1%}",
                f"{stats['avg_duration']:.2f}s"
            )
        
        console.print(table)

### 3. Human-in-the-Loop

class HumanInputHandler:
    """Handle human intervention points in workflows"""
    
    async def request_approval(
        self,
        action: str,
        context: Dict,
        timeout: int = 300
    ) -> bool:
        """Request human approval for an action"""
        console.print(Panel(
            f"[bold yellow]Human Approval Required[/bold yellow]\n\n"
            f"Action: {action}\n"
            f"Context: {json.dumps(context, indent=2)}",
            title="⚠️  Waiting for Approval"
        ))
        
        response = Prompt.ask(
            "Approve this action?",
            choices=["yes", "no"],
            default="no"
        )
        
        return response.lower() == "yes"
    
    async def request_input(
        self,
        prompt: str,
        input_type: str = "text"
    ) -> Any:
        """Request input from human"""
        console.print(Panel(
            f"[bold cyan]Human Input Required[/bold cyan]\n\n{prompt}",
            title="💬 Input Needed"
        ))
        
        if input_type == "text":
            return Prompt.ask("Your input")
        elif input_type == "choice":
            # Could be enhanced with choices parameter
            return Prompt.ask("Your choice")

### 4. Workflow Visualization

class WorkflowVisualizer:
    """Visualize workflow execution"""
    
    def visualize_chain(self, chain: ChainWorkflow):
        """Display chain workflow diagram"""
        console.print(f"\n[bold]Chain Workflow:[/bold] {chain.name}\n")
        
        for i, agent in enumerate(chain.sequence):
            if i == 0:
                console.print(f"  📥 Input")
            console.print(f"   ↓")
            console.print(f"  🤖 {agent}")
        
        console.print(f"   ↓")
        console.print(f"  📤 Output\n")
    
    def visualize_parallel(self, parallel: ParallelWorkflow):
        """Display parallel workflow diagram"""
        console.print(f"\n[bold]Parallel Workflow:[/bold] {parallel.name}\n")
        
        console.print(f"  📥 Input")
        console.print(f"   ┃")
        console.print(f"   ┣━━━━━━ Fan-Out ━━━━━━┓")
        
        for i, agent in enumerate(parallel.fan_out):
            prefix = "   ┣" if i < len(parallel.fan_out) - 1 else "   ┗"
            console.print(f"{prefix}━━ 🤖 {agent}")
        
        if parallel.fan_in:
            console.print(f"   ┃")
            console.print(f"   ┗━━━━━━ Fan-In ━━━━━━┓")
            console.print(f"         🤖 {parallel.fan_in}")
        
        console.print(f"   ↓")
        console.print(f"  📤 Output\n")

## Configuration Examples

### llm_config.yaml

# Enhanced LLM configuration with multiple profiles
profiles:
  fast:
    name: "Fast (GPT-4o-mini)"
    provider: "openai"
    model: "gpt-4o-mini"
    apiKey: "${OPENAI_API_KEY}"
    systemPrompt: "You are a helpful AI assistant optimized for speed."
    temperature: 0.7
    maxTokens: 4096
    
  orchestrator:
    name: "Orchestrator (GPT-4)"
    provider: "openai"
    model: "gpt-4-turbo"
    apiKey: "${OPENAI_API_KEY}"
    systemPrompt: "You are an orchestration planner. Generate structured execution plans."
    temperature: 0.3
    maxTokens: 8192
    
  router:
    name: "Router (Claude-3-Haiku)"
    provider: "anthropic"
    model: "claude-3-haiku-20240307"
    apiKey: "${ANTHROPIC_API_KEY}"
    systemPrompt: "You are a routing assistant that selects the best agent for each task."
    temperature: 0.1
    maxTokens: 1024
    
  creative:
    name: "Creative (Claude-3-Opus)"
    provider: "anthropic"
    model: "claude-3-opus-20240229"
    apiKey: "${ANTHROPIC_API_KEY}"
    systemPrompt: "You are a creative writing expert."
    temperature: 0.9
    maxTokens: 4096
    
  local:
    name: "Local (Llama-3.2)"
    provider: "ollama"
    model: "llama3.2"
    apiBase: "http://localhost:11434"
    systemPrompt: "You are a helpful assistant."
    temperature: 0.7
    maxTokens: 4096
    fallbacks:
      - provider: "openai"
        model: "gpt-4o-mini"

defaultProfile: "fast"

settings:
  maxIterations: 15
  requestTimeout: 180
  debugMode: false
  enableStreaming: false
  maxConcurrency: 10
  retryAttempts: 3
  retryDelay: 1.0

### workflow_config.yaml

# Workflow definitions
workflows:
  content_pipeline:
    type: "chain"
    sequence:
      - url_fetcher
      - summarizer
      - social_media_writer
    cumulative: false
    
  research_analysis:
    type: "parallel"
    fanOut:
      - financial_analyst
      - technical_analyst
      - market_analyst
    fanIn: synthesizer
    maxConcurrency: 3
    includeRequest: true
    
  intelligent_dispatch:
    type: "router"
    agents:
      - code_expert
      - research_expert
      - creative_expert
    routingInstruction: "Route based on query type and expertise required"
    profile: "router"
    
  complex_project:
    type: "orchestrator"
    agents:
      - researcher
      - analyzer
      - writer
      - reviewer
      - publisher
    planType: "iterative"
    profile: "orchestrator"

## Performance Optimization

### Connection Pooling

class ConnectionPool:
    """Manage MCP server connections efficiently"""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections: Dict[str, List[Client]] = {}
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
    
    async def get_connection(self, server_name: str) -> Client:
        """Get or create connection to server"""
        if server_name not in self.semaphores:
            self.semaphores[server_name] = asyncio.Semaphore(self.max_connections)
        
        async with self.semaphores[server_name]:
            # Connection pooling logic
            pass

### Request Batching

class RequestBatcher:
    """Batch multiple requests to reduce overhead"""
    
    def __init__(self, batch_size: int = 10, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.pending_requests: List[Dict] = []
        self._lock = asyncio.Lock()
    
    async def add_request(self, request: Dict) -> asyncio.Future:
        """Add request to batch"""
        future = asyncio.Future()
        
        async with self._lock:
            self.pending_requests.append({
                'request': request,
                'future': future
            })
            
            if len(self.pending_requests) >= self.batch_size:
                await self._flush()
        
        return future
    
    async def _flush(self):
        """Process batched requests"""
        if not self.pending_requests:
            return
        
        batch = self.pending_requests[:]
        self.pending_requests.clear()
        
        # Process batch
        # ... implementation

## Testing Utilities

class MockLLM:
    """Mock LLM for testing without API calls"""
    
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0
    
    async def complete(self, messages: List[Dict]) -> str:
        """Return predefined response"""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response

class WorkflowTester:
    """Test workflows with mock components"""
    
    async def test_chain(self, chain: ChainWorkflow, test_cases: List[Dict]):
        """Test chain workflow with multiple cases"""
        results = []
        
        for case in test_cases:
            result = await chain.execute(
                case['input'],
                context=case.get('context')
            )
            
            results.append({
                'input': case['input'],
                'expected': case.get('expected'),
                'actual': result.final_output,
                'success': result.success
            })
        
        return results

## Deployment Considerations

### 1. Environment Variables

# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LITELLM_PROXY_KEY=...

# Performance
MAX_CONCURRENCY=10
REQUEST_TIMEOUT=180
RETRY_ATTEMPTS=3

# Observability
DEBUG_MODE=false
ENABLE_TRACING=true
METRICS_PORT=9090

### 2. Docker Deployment

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "agent.py"]

### 3. Kubernetes Deployment

apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-agent-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-agent
  template:
    metadata:
      labels:
        app: mcp-agent
    spec:
      containers:
      - name: agent
        image: mcp-agent:latest
        env:
        - name: MAX_CONCURRENCY
          value: "10"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"

## Next Steps

1. **Integration**: Merge these components into your existing `agent.py`
2. **Testing**: Create test suite for each workflow pattern
3. **Documentation**: Document your specific agents and workflows
4. **Monitoring**: Set up observability with OpenTelemetry
5. **Optimization**: Profile and optimize bottlenecks
6. **Production**: Deploy with proper error handling and monitoring

## References

- [fast-agent Documentation](https://fast-agent.ai)
- [MCP Protocol Specification](https://modelcontextprotocol.io)
- [LiteLLM Documentation](https://docs.litellm.ai)
- [Agent Orchestration Patterns](https://www.getdynamiq.ai/blog/agent-orchestration-patterns)
- [Microsoft Agent Framework](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/)