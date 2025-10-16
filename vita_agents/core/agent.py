"""
Base agent class and communication protocols for Vita Agents.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import structlog
from pydantic import BaseModel, Field
import json


logger = structlog.get_logger(__name__)


class AgentStatus(str, Enum):
    """Agent status enumeration."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


class MessageType(str, Enum):
    """Message type enumeration."""
    TASK = "task"
    RESPONSE = "response"
    ERROR = "error"
    STATUS = "status"
    HEARTBEAT = "heartbeat"
    BROADCAST = "broadcast"


class Priority(str, Enum):
    """Task priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class AgentMessage(BaseModel):
    """Standard message format for agent communication."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    sender_id: str
    receiver_id: Optional[str] = None  # None for broadcast
    content: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    priority: Priority = Priority.NORMAL
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TaskRequest(BaseModel):
    """Task request model."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str
    parameters: Dict[str, Any]
    priority: Priority = Priority.NORMAL
    timeout_seconds: int = 300
    retry_attempts: int = 3
    callback_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TaskResponse(BaseModel):
    """Task response model."""
    
    task_id: str
    agent_id: str
    status: str  # success, error, timeout
    result: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time_ms: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=datetime.utcnow)


class AgentCapability(BaseModel):
    """Agent capability description."""
    
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    supported_formats: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)


class AgentMetrics(BaseModel):
    """Agent performance metrics."""
    
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_execution_time_ms: float = 0.0
    last_activity: Optional[datetime] = None
    uptime_seconds: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


class BaseAgent(ABC):
    """Base class for all Vita agents."""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        version: str = "1.0.0",
        capabilities: Optional[List[AgentCapability]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.version = version
        self.capabilities = capabilities or []
        self.config = config or {}
        self.status = AgentStatus.INACTIVE
        self.metrics = AgentMetrics()
        self.logger = structlog.get_logger(self.__class__.__name__).bind(agent_id=agent_id)
        
        # Message handling
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.task_handlers: Dict[str, Callable] = {}
        
        # Event callbacks
        self.on_start_callbacks: List[Callable] = []
        self.on_stop_callbacks: List[Callable] = []
        self.on_error_callbacks: List[Callable] = []
        
        # Communication
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.response_callbacks: Dict[str, Callable] = {}
        
        self._running = False
        self._start_time: Optional[datetime] = None
        
        # Register default message handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self.register_message_handler(MessageType.TASK, self._handle_task_message)
        self.register_message_handler(MessageType.STATUS, self._handle_status_message)
        self.register_message_handler(MessageType.HEARTBEAT, self._handle_heartbeat_message)
    
    def register_message_handler(self, message_type: MessageType, handler: Callable) -> None:
        """Register a message handler for a specific message type."""
        self.message_handlers[message_type] = handler
    
    def register_task_handler(self, task_type: str, handler: Callable) -> None:
        """Register a task handler for a specific task type."""
        self.task_handlers[task_type] = handler
    
    async def start(self) -> None:
        """Start the agent."""
        self.logger.info("Starting agent")
        self._running = True
        self._start_time = datetime.utcnow()
        self.status = AgentStatus.ACTIVE
        
        # Call start callbacks
        for callback in self.on_start_callbacks:
            await callback(self)
        
        # Start message processing loop
        asyncio.create_task(self._message_processing_loop())
        
        await self._on_start()
    
    async def stop(self) -> None:
        """Stop the agent."""
        self.logger.info("Stopping agent")
        self._running = False
        self.status = AgentStatus.STOPPED
        
        # Call stop callbacks
        for callback in self.on_stop_callbacks:
            await callback(self)
        
        await self._on_stop()
    
    async def send_message(self, message: AgentMessage) -> None:
        """Send a message to another agent or broadcast."""
        # This would be implemented by the orchestrator
        # For now, just log the message
        self.logger.info("Sending message", message_type=message.type, receiver=message.receiver_id)
    
    async def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from another agent."""
        await self.message_queue.put(message)
    
    async def execute_task(self, task: TaskRequest) -> TaskResponse:
        """Execute a task and return the response."""
        start_time = datetime.utcnow()
        task_response = TaskResponse(
            task_id=task.id,
            agent_id=self.agent_id,
            status="success",
            execution_time_ms=0
        )
        
        try:
            self.status = AgentStatus.BUSY
            self.logger.info("Executing task", task_type=task.task_type, task_id=task.id)
            
            # Get the appropriate task handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler registered for task type: {task.task_type}")
            
            # Execute the task
            result = await handler(task)
            task_response.result = result if isinstance(result, dict) else {"data": result}
            
            # Update metrics
            self.metrics.tasks_completed += 1
            self.metrics.last_activity = datetime.utcnow()
            
        except Exception as e:
            self.logger.error("Task execution failed", task_id=task.id, error=str(e))
            task_response.status = "error"
            task_response.error_message = str(e)
            self.metrics.tasks_failed += 1
            
            # Call error callbacks
            for callback in self.on_error_callbacks:
                await callback(self, e)
        
        finally:
            self.status = AgentStatus.ACTIVE
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            task_response.execution_time_ms = int(execution_time)
            
            # Update average execution time
            total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
            if total_tasks > 0:
                self.metrics.average_execution_time_ms = (
                    (self.metrics.average_execution_time_ms * (total_tasks - 1) + execution_time) 
                    / total_tasks
                )
        
        return task_response
    
    async def _message_processing_loop(self) -> None:
        """Main message processing loop."""
        while self._running:
            try:
                # Wait for a message with a timeout
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._process_message(message)
            except asyncio.TimeoutError:
                # No message received, continue
                continue
            except Exception as e:
                self.logger.error("Error processing message", error=str(e))
    
    async def _process_message(self, message: AgentMessage) -> None:
        """Process a received message."""
        handler = self.message_handlers.get(message.type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                self.logger.error("Error handling message", message_type=message.type, error=str(e))
        else:
            self.logger.warning("No handler for message type", message_type=message.type)
    
    async def _handle_task_message(self, message: AgentMessage) -> None:
        """Handle task messages."""
        try:
            task_data = message.content
            task = TaskRequest(**task_data)
            response = await self.execute_task(task)
            
            # Send response back
            response_message = AgentMessage(
                type=MessageType.RESPONSE,
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content=response.dict(),
                correlation_id=message.id
            )
            await self.send_message(response_message)
            
        except Exception as e:
            self.logger.error("Error handling task message", error=str(e))
    
    async def _handle_status_message(self, message: AgentMessage) -> None:
        """Handle status messages."""
        # Return current agent status
        status_response = AgentMessage(
            type=MessageType.STATUS,
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            content={
                "status": self.status.value,
                "metrics": self.metrics.dict(),
                "capabilities": [cap.dict() for cap in self.capabilities]
            },
            correlation_id=message.id
        )
        await self.send_message(status_response)
    
    async def _handle_heartbeat_message(self, message: AgentMessage) -> None:
        """Handle heartbeat messages."""
        # Update uptime
        if self._start_time:
            self.metrics.uptime_seconds = int((datetime.utcnow() - self._start_time).total_seconds())
        
        # Send heartbeat response
        heartbeat_response = AgentMessage(
            type=MessageType.HEARTBEAT,
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            content={"status": self.status.value, "uptime": self.metrics.uptime_seconds},
            correlation_id=message.id
        )
        await self.send_message(heartbeat_response)
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.value,
            "capabilities": [cap.dict() for cap in self.capabilities],
            "metrics": self.metrics.dict(),
            "config": self.config
        }
    
    @abstractmethod
    async def _on_start(self) -> None:
        """Called when the agent starts. Override in subclasses."""
        pass
    
    @abstractmethod
    async def _on_stop(self) -> None:
        """Called when the agent stops. Override in subclasses."""
        pass


class HealthcareAgent(BaseAgent):
    """Base class for healthcare-specific agents."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Healthcare-specific capabilities
        self.supported_standards: List[str] = []
        self.data_formats: List[str] = []
        self.compliance_features: List[str] = []
    
    def validate_healthcare_data(self, data: Any, standard: str) -> bool:
        """Validate healthcare data against a standard."""
        # Base implementation - override in subclasses
        return True
    
    def ensure_hipaa_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure data is HIPAA compliant."""
        # Base implementation - override in subclasses
        return data
    
    def audit_log_action(self, action: str, data_type: str, details: Dict[str, Any]) -> None:
        """Log an action for audit purposes."""
        self.logger.info(
            "Healthcare action logged",
            action=action,
            data_type=data_type,
            details=details,
            agent_id=self.agent_id,
            timestamp=datetime.utcnow().isoformat()
        )