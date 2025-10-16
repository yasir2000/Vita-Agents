"""
Agent orchestrator for managing and coordinating multiple agents.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
import structlog
from pydantic import BaseModel
from collections import defaultdict

from vita_agents.core.agent import (
    BaseAgent,
    AgentMessage,
    MessageType,
    TaskRequest,
    TaskResponse,
    AgentStatus,
    Priority
)
from vita_agents.core.config import get_settings


logger = structlog.get_logger(__name__)


class WorkflowStep(BaseModel):
    """A single step in a workflow."""
    
    id: str
    agent_id: str
    task_type: str
    parameters: Dict[str, Any]
    dependencies: List[str] = []  # IDs of steps that must complete first
    timeout_seconds: int = 300
    retry_attempts: int = 3
    condition: Optional[str] = None  # Python expression for conditional execution


class WorkflowDefinition(BaseModel):
    """Workflow definition containing multiple steps."""
    
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    timeout_seconds: int = 600
    metadata: Dict[str, Any] = {}


class WorkflowExecution(BaseModel):
    """Workflow execution state."""
    
    id: str
    workflow_id: str
    status: str  # pending, running, completed, failed
    started_at: datetime
    completed_at: Optional[datetime] = None
    step_results: Dict[str, TaskResponse] = {}
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


class AgentOrchestrator:
    """
    Main orchestrator for managing agents and coordinating workflows.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.settings = get_settings()
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Agent management
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[str, type] = {}
        
        # Workflow management
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.workflow_executions: Dict[str, WorkflowExecution] = {}
        
        # Message routing
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.response_callbacks: Dict[str, Callable] = {}
        
        # Load balancing
        self.agent_load: Dict[str, int] = defaultdict(int)
        
        # State
        self._running = False
        self._message_router_task: Optional[asyncio.Task] = None
        
        self.logger.info("Agent orchestrator initialized")
    
    async def start(self) -> None:
        """Start the orchestrator."""
        self.logger.info("Starting agent orchestrator")
        self._running = True
        
        # Start all registered agents
        for agent in self.agents.values():
            await agent.start()
        
        # Start message router
        self._message_router_task = asyncio.create_task(self._message_router_loop())
        
        self.logger.info("Agent orchestrator started")
    
    async def stop(self) -> None:
        """Stop the orchestrator."""
        self.logger.info("Stopping agent orchestrator")
        self._running = False
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()
        
        # Stop message router
        if self._message_router_task:
            self._message_router_task.cancel()
            try:
                await self._message_router_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Agent orchestrator stopped")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_id] = agent
        self.agent_load[agent.agent_id] = 0
        
        # Override the agent's send_message method to route through orchestrator
        agent.send_message = self._route_message
        
        self.logger.info("Agent registered", agent_id=agent.agent_id, agent_name=agent.name)
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the orchestrator."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.agent_load[agent_id]
            self.logger.info("Agent unregistered", agent_id=agent_id)
    
    def register_agent_type(self, name: str, agent_class: type) -> None:
        """Register an agent type for dynamic creation."""
        self.agent_types[name] = agent_class
        self.logger.info("Agent type registered", name=name, class_name=agent_class.__name__)
    
    async def create_agent(
        self,
        agent_type: str,
        agent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseAgent:
        """Create and register a new agent instance."""
        if agent_type not in self.agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_id = agent_id or str(uuid.uuid4())
        agent_class = self.agent_types[agent_type]
        
        # Create agent instance
        agent = agent_class(agent_id=agent_id, config=config or {})
        self.register_agent(agent)
        
        # Start the agent if orchestrator is running
        if self._running:
            await agent.start()
        
        return agent
    
    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """Register a workflow definition."""
        self.workflows[workflow.id] = workflow
        self.logger.info("Workflow registered", workflow_id=workflow.id, workflow_name=workflow.name)
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
        execution_id: Optional[str] = None
    ) -> WorkflowExecution:
        """Execute a workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        execution_id = execution_id or str(uuid.uuid4())
        
        execution = WorkflowExecution(
            id=execution_id,
            workflow_id=workflow_id,
            status="pending",
            started_at=datetime.utcnow(),
            metadata={"input_data": input_data}
        )
        
        self.workflow_executions[execution_id] = execution
        
        # Start workflow execution
        asyncio.create_task(self._execute_workflow_steps(execution, workflow, input_data))
        
        return execution
    
    async def _execute_workflow_steps(
        self,
        execution: WorkflowExecution,
        workflow: WorkflowDefinition,
        input_data: Dict[str, Any]
    ) -> None:
        """Execute workflow steps."""
        try:
            execution.status = "running"
            
            # Track completed steps
            completed_steps: Set[str] = set()
            available_data = {"input": input_data}
            
            while len(completed_steps) < len(workflow.steps):
                # Find steps that can be executed (dependencies satisfied)
                ready_steps = [
                    step for step in workflow.steps
                    if step.id not in completed_steps and
                    all(dep in completed_steps for dep in step.dependencies)
                ]
                
                if not ready_steps:
                    break  # No more steps can be executed
                
                # Execute ready steps in parallel
                tasks = []
                for step in ready_steps:
                    task = asyncio.create_task(
                        self._execute_workflow_step(step, available_data, execution)
                    )
                    tasks.append((step.id, task))
                
                # Wait for all tasks to complete
                for step_id, task in tasks:
                    try:
                        result = await task
                        execution.step_results[step_id] = result
                        completed_steps.add(step_id)
                        
                        # Add result to available data
                        available_data[step_id] = result.result
                        
                    except Exception as e:
                        self.logger.error("Workflow step failed", step_id=step_id, error=str(e))
                        execution.status = "failed"
                        execution.error_message = f"Step {step_id} failed: {str(e)}"
                        execution.completed_at = datetime.utcnow()
                        return
            
            execution.status = "completed"
            execution.completed_at = datetime.utcnow()
            
        except Exception as e:
            self.logger.error("Workflow execution failed", workflow_id=workflow.id, error=str(e))
            execution.status = "failed"
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
    
    async def _execute_workflow_step(
        self,
        step: WorkflowStep,
        available_data: Dict[str, Any],
        execution: WorkflowExecution
    ) -> TaskResponse:
        """Execute a single workflow step."""
        # Check condition if specified
        if step.condition:
            # Simple condition evaluation (could be enhanced with safe_eval)
            if not eval(step.condition, {"data": available_data}):
                # Skip this step
                return TaskResponse(
                    task_id=step.id,
                    agent_id=step.agent_id,
                    status="skipped",
                    execution_time_ms=0
                )
        
        # Find agent
        agent = self.agents.get(step.agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {step.agent_id}")
        
        # Create task request
        task = TaskRequest(
            id=step.id,
            task_type=step.task_type,
            parameters=step.parameters,
            timeout_seconds=step.timeout_seconds,
            retry_attempts=step.retry_attempts,
            metadata={"workflow_execution_id": execution.id}
        )
        
        # Execute task
        execution.current_step = step.id
        result = await agent.execute_task(task)
        execution.current_step = None
        
        return result
    
    async def send_task_to_agent(
        self,
        agent_id: str,
        task: TaskRequest
    ) -> TaskResponse:
        """Send a task directly to a specific agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")
        
        return await agent.execute_task(task)
    
    async def send_task_to_best_agent(
        self,
        task_type: str,
        task: TaskRequest
    ) -> TaskResponse:
        """Send a task to the best available agent that can handle it."""
        # Find agents that can handle this task type
        capable_agents = [
            agent for agent in self.agents.values()
            if task_type in [cap.name for cap in agent.capabilities] and
            agent.status == AgentStatus.ACTIVE
        ]
        
        if not capable_agents:
            raise ValueError(f"No capable agents found for task type: {task_type}")
        
        # Select agent with lowest load
        best_agent = min(capable_agents, key=lambda a: self.agent_load[a.agent_id])
        
        # Update load tracking
        self.agent_load[best_agent.agent_id] += 1
        
        try:
            result = await best_agent.execute_task(task)
            return result
        finally:
            self.agent_load[best_agent.agent_id] -= 1
    
    async def broadcast_message(self, message: AgentMessage) -> None:
        """Broadcast a message to all agents."""
        for agent in self.agents.values():
            await agent.receive_message(message)
    
    async def _route_message(self, message: AgentMessage) -> None:
        """Route a message to its destination."""
        if message.receiver_id is None:
            # Broadcast message
            await self.broadcast_message(message)
        else:
            # Direct message
            agent = self.agents.get(message.receiver_id)
            if agent:
                await agent.receive_message(message)
            else:
                self.logger.warning("Message recipient not found", receiver_id=message.receiver_id)
    
    async def _message_router_loop(self) -> None:
        """Main message routing loop."""
        while self._running:
            try:
                # Process any queued messages
                while not self.message_queue.empty():
                    message = await self.message_queue.get()
                    await self._route_message(message)
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error("Error in message router", error=str(e))
    
    def get_agent_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of agents."""
        if agent_id:
            agent = self.agents.get(agent_id)
            return agent.get_info() if agent else {}
        
        return {
            "agents": [agent.get_info() for agent in self.agents.values()],
            "total_agents": len(self.agents),
            "active_agents": sum(1 for agent in self.agents.values() if agent.status == AgentStatus.ACTIVE),
            "total_load": sum(self.agent_load.values())
        }
    
    def get_workflow_status(self, execution_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of workflows."""
        if execution_id:
            execution = self.workflow_executions.get(execution_id)
            return execution.dict() if execution else {}
        
        return {
            "executions": [execution.dict() for execution in self.workflow_executions.values()],
            "total_executions": len(self.workflow_executions),
            "active_executions": sum(
                1 for execution in self.workflow_executions.values()
                if execution.status in ["pending", "running"]
            )
        }


# Global orchestrator instance
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


def set_orchestrator(orchestrator: AgentOrchestrator) -> None:
    """Set the global orchestrator instance."""
    global _orchestrator
    _orchestrator = orchestrator