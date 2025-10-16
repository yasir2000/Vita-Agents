"""
FastAPI application for Vita Agents REST API.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog

from vita_agents.core.orchestrator import AgentOrchestrator, get_orchestrator
from vita_agents.core.agent import TaskRequest, TaskResponse, WorkflowDefinition, WorkflowStep
from vita_agents.core.config import get_settings
from vita_agents.agents import FHIRAgent, HL7Agent, EHRAgent


logger = structlog.get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Vita Agents API",
    description="Multi-Agent AI Framework for Healthcare Interoperability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Security
security = HTTPBearer()

# Global orchestrator
orchestrator: Optional[AgentOrchestrator] = None


# Request/Response Models
class TaskRequestModel(BaseModel):
    """Task request model for API."""
    task_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    agent_id: Optional[str] = None
    priority: str = "normal"
    timeout_seconds: int = 300


class TaskResponseModel(BaseModel):
    """Task response model for API."""
    task_id: str
    agent_id: str
    status: str
    result: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time_ms: int
    timestamp: str


class AgentStatusModel(BaseModel):
    """Agent status model for API."""
    agent_id: str
    name: str
    status: str
    capabilities: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class WorkflowExecutionModel(BaseModel):
    """Workflow execution model for API."""
    workflow_id: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    execution_id: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    agents: Dict[str, str]


# Middleware
@app.middleware("http")
async def logging_middleware(request, call_next):
    """Log all requests."""
    start_time = datetime.utcnow()
    
    response = await call_next(request)
    
    process_time = (datetime.utcnow() - start_time).total_seconds()
    
    logger.info(
        "HTTP request",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    
    return response


# Dependency functions
def get_current_orchestrator() -> AgentOrchestrator:
    """Get the current orchestrator instance."""
    global orchestrator
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not initialized"
        )
    return orchestrator


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token (simplified implementation)."""
    # In production, implement proper JWT verification
    token = credentials.credentials
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    global orchestrator
    
    logger.info("Starting Vita Agents API")
    
    # Initialize orchestrator
    orchestrator = get_orchestrator()
    
    # Register agent types
    orchestrator.register_agent_type("fhir", FHIRAgent)
    orchestrator.register_agent_type("hl7", HL7Agent)
    orchestrator.register_agent_type("ehr", EHRAgent)
    
    # Create default agents
    try:
        await orchestrator.create_agent("fhir")
        await orchestrator.create_agent("hl7") 
        await orchestrator.create_agent("ehr")
        
        # Start orchestrator
        await orchestrator.start()
        
        logger.info("Orchestrator and agents started successfully")
        
    except Exception as e:
        logger.error("Failed to start orchestrator", error=str(e))
        raise
    
    # Setup CORS
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up the application."""
    global orchestrator
    
    logger.info("Shutting down Vita Agents API")
    
    if orchestrator:
        await orchestrator.stop()
    
    logger.info("Shutdown complete")


# Health check endpoints
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    orch = get_current_orchestrator()
    
    agent_status = orch.get_agent_status()
    agents = {
        agent["agent_id"]: agent["status"] 
        for agent in agent_status.get("agents", [])
    }
    
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        agents=agents
    )


@app.get("/health/ready")
async def readiness_check():
    """Readiness check endpoint."""
    orch = get_current_orchestrator()
    
    agent_status = orch.get_agent_status()
    active_agents = agent_status.get("active_agents", 0)
    
    if active_agents > 0:
        return {"status": "ready", "active_agents": active_agents}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No active agents available"
        )


# Agent management endpoints
@app.get("/api/v1/agents", response_model=List[AgentStatusModel])
async def list_agents(orch: AgentOrchestrator = Depends(get_current_orchestrator)):
    """List all agents and their status."""
    agent_status = orch.get_agent_status()
    
    agents = []
    for agent_info in agent_status.get("agents", []):
        agents.append(AgentStatusModel(
            agent_id=agent_info["agent_id"],
            name=agent_info["name"],
            status=agent_info["status"],
            capabilities=agent_info["capabilities"],
            metrics=agent_info["metrics"]
        ))
    
    return agents


@app.get("/api/v1/agents/status")
async def get_agents_status(orch: AgentOrchestrator = Depends(get_current_orchestrator)):
    """Get status of all agents as mentioned in README."""
    agent_status = orch.get_agent_status()
    
    agents = []
    for agent_info in agent_status.get("agents", []):
        agents.append({
            "name": agent_info["name"],
            "status": agent_info["status"],
            "last_activity": agent_info.get("last_activity", datetime.utcnow().isoformat() + "Z")
        })
    
    return {"agents": agents}


@app.get("/api/v1/agents/{agent_id}", response_model=AgentStatusModel)
async def get_agent(
    agent_id: str,
    orch: AgentOrchestrator = Depends(get_current_orchestrator)
):
    """Get specific agent information."""
    agent_status = orch.get_agent_status(agent_id)
    
    if not agent_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )
    
    return AgentStatusModel(
        agent_id=agent_status["agent_id"],
        name=agent_status["name"],
        status=agent_status["status"],
        capabilities=agent_status["capabilities"],
        metrics=agent_status["metrics"]
    )


@app.post("/api/v1/agents")
async def create_agent(
    agent_type: str,
    agent_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    orch: AgentOrchestrator = Depends(get_current_orchestrator),
    token: str = Depends(verify_token)
):
    """Create a new agent instance."""
    try:
        agent = await orch.create_agent(agent_type, agent_id, config)
        
        return {
            "agent_id": agent.agent_id,
            "agent_type": agent_type,
            "status": "created",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to create agent", agent_type=agent_type, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create agent: {str(e)}"
        )


# Task execution endpoints
@app.post("/api/v1/tasks", response_model=TaskResponseModel)
async def execute_task(
    task_request: TaskRequestModel,
    orch: AgentOrchestrator = Depends(get_current_orchestrator),
    token: str = Depends(verify_token)
):
    """Execute a task on an agent."""
    try:
        # Create task request
        task = TaskRequest(
            task_type=task_request.task_type,
            parameters=task_request.parameters,
            timeout_seconds=task_request.timeout_seconds
        )
        
        # Execute task
        if task_request.agent_id:
            # Send to specific agent
            result = await orch.send_task_to_agent(task_request.agent_id, task)
        else:
            # Send to best available agent
            result = await orch.send_task_to_best_agent(task_request.task_type, task)
        
        return TaskResponseModel(
            task_id=result.task_id,
            agent_id=result.agent_id,
            status=result.status,
            result=result.result,
            error_message=result.error_message,
            execution_time_ms=result.execution_time_ms,
            timestamp=result.completed_at.isoformat()
        )
        
    except Exception as e:
        logger.error("Task execution failed", task_type=task_request.task_type, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task execution failed: {str(e)}"
        )


@app.post("/api/v1/tasks/async")
async def execute_task_async(
    task_request: TaskRequestModel,
    background_tasks: BackgroundTasks,
    orch: AgentOrchestrator = Depends(get_current_orchestrator),
    token: str = Depends(verify_token)
):
    """Execute a task asynchronously."""
    task_id = f"task-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    
    # Add task to background tasks
    background_tasks.add_task(
        _execute_background_task,
        orch,
        task_request,
        task_id
    )
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Task queued for background execution",
        "timestamp": datetime.utcnow().isoformat()
    }


async def _execute_background_task(
    orch: AgentOrchestrator,
    task_request: TaskRequestModel,
    task_id: str
):
    """Execute a task in the background."""
    try:
        task = TaskRequest(
            id=task_id,
            task_type=task_request.task_type,
            parameters=task_request.parameters,
            timeout_seconds=task_request.timeout_seconds
        )
        
        if task_request.agent_id:
            result = await orch.send_task_to_agent(task_request.agent_id, task)
        else:
            result = await orch.send_task_to_best_agent(task_request.task_type, task)
        
        logger.info("Background task completed", task_id=task_id, status=result.status)
        
    except Exception as e:
        logger.error("Background task failed", task_id=task_id, error=str(e))


# Workflow endpoints
@app.post("/api/v1/workflows")
async def register_workflow(
    workflow: WorkflowDefinition,
    orch: AgentOrchestrator = Depends(get_current_orchestrator),
    token: str = Depends(verify_token)
):
    """Register a new workflow definition."""
    try:
        orch.register_workflow(workflow)
        
        return {
            "workflow_id": workflow.id,
            "status": "registered",
            "steps": len(workflow.steps),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Workflow registration failed", workflow_id=workflow.id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workflow registration failed: {str(e)}"
        )


@app.post("/api/v1/workflows/execute")
async def execute_workflow(
    execution_request: WorkflowExecutionModel,
    orch: AgentOrchestrator = Depends(get_current_orchestrator),
    token: str = Depends(verify_token)
):
    """Execute a workflow."""
    try:
        execution = await orch.execute_workflow(
            execution_request.workflow_id,
            execution_request.input_data,
            execution_request.execution_id
        )
        
        return {
            "execution_id": execution.id,
            "workflow_id": execution.workflow_id,
            "status": execution.status,
            "started_at": execution.started_at.isoformat(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Workflow execution failed", workflow_id=execution_request.workflow_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workflow execution failed: {str(e)}"
        )


@app.get("/api/v1/workflows/{execution_id}")
async def get_workflow_execution(
    execution_id: str,
    orch: AgentOrchestrator = Depends(get_current_orchestrator)
):
    """Get workflow execution status."""
    workflow_status = orch.get_workflow_status(execution_id)
    
    if not workflow_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow execution {execution_id} not found"
        )
    
    return workflow_status


# FHIR-specific endpoints
@app.post("/api/v1/fhir/validate")
async def validate_fhir(
    resource: Dict[str, Any],
    version: str = "R4",
    orch: AgentOrchestrator = Depends(get_current_orchestrator),
    token: str = Depends(verify_token)
):
    """Validate a FHIR resource."""
    task = TaskRequest(
        task_type="validate_fhir_resource",
        parameters={
            "resource": resource,
            "version": version
        }
    )
    
    try:
        result = await orch.send_task_to_best_agent("validate_fhir_resource", task)
        return result.result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"FHIR validation failed: {str(e)}"
        )


@app.post("/api/v1/fhir/quality-check")
async def fhir_quality_check(
    resource: Dict[str, Any],
    quality_rules: List[str] = [],
    orch: AgentOrchestrator = Depends(get_current_orchestrator),
    token: str = Depends(verify_token)
):
    """Perform quality check on FHIR resource."""
    task = TaskRequest(
        task_type="quality_check",
        parameters={
            "resource": resource,
            "quality_rules": quality_rules
        }
    )
    
    try:
        result = await orch.send_task_to_best_agent("quality_check", task)
        return result.result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Quality check failed: {str(e)}"
        )


# HL7-specific endpoints
@app.post("/api/v1/hl7/validate")
async def validate_hl7(
    message: str,
    version: str = "2.8",
    orch: AgentOrchestrator = Depends(get_current_orchestrator),
    token: str = Depends(verify_token)
):
    """Validate an HL7 message."""
    task = TaskRequest(
        task_type="validate_hl7_message",
        parameters={
            "message": message,
            "version": version
        }
    )
    
    try:
        result = await orch.send_task_to_best_agent("validate_hl7_message", task)
        return result.result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"HL7 validation failed: {str(e)}"
        )


@app.post("/api/v1/hl7/to-fhir")
async def hl7_to_fhir(
    hl7_message: str,
    target_resources: List[str] = [],
    orch: AgentOrchestrator = Depends(get_current_orchestrator),
    token: str = Depends(verify_token)
):
    """Convert HL7 message to FHIR resources."""
    task = TaskRequest(
        task_type="hl7_to_fhir",
        parameters={
            "hl7_message": hl7_message,
            "target_resources": target_resources
        }
    )
    
    try:
        result = await orch.send_task_to_best_agent("hl7_to_fhir", task)
        return result.result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"HL7 to FHIR conversion failed: {str(e)}"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "vita_agents.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.api.workers
    )