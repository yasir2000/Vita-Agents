"""
Enhanced FHIR Agent with Open Source FHIR Engines Support
Integration with multiple free FHIR servers and comprehensive client management
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import structlog
from pydantic import BaseModel, Field

from vita_agents.core.agent import HealthcareAgent, AgentCapability, TaskRequest, TaskResponse
from vita_agents.core.config import get_settings
from vita_agents.fhir_engines.open_source_clients import (
    FHIREngineManager, FHIRServerConfiguration, FHIREngineType, 
    FHIRVersion, AuthenticationType, FHIRSearchParameters,
    FHIROperationResult, get_server_template, list_server_templates
)

logger = structlog.get_logger(__name__)


class FHIREngineConfig(BaseModel):
    """Configuration for FHIR engines in the agent"""
    enabled_engines: List[str] = Field(default_factory=list)
    default_engine: Optional[str] = None
    auto_connect: bool = True
    connection_timeout: int = 30
    max_concurrent_operations: int = 10
    retry_failed_operations: bool = True
    max_retries: int = 3


class FHIRMultiEngineResult(BaseModel):
    """Result from operations across multiple FHIR engines"""
    operation: str
    total_engines: int
    successful_engines: int
    failed_engines: int
    results: Dict[str, FHIROperationResult] = Field(default_factory=dict)
    execution_time_ms: Optional[float] = None
    errors: List[str] = Field(default_factory=list)


class EnhancedFHIRAgent(HealthcareAgent):
    """
    Enhanced FHIR Agent with support for multiple open source FHIR engines
    
    Supported Engines:
    - HAPI FHIR Server (https://hapifhir.io/)
    - IBM FHIR Server (https://github.com/IBM/FHIR)
    - Firely .NET SDK (https://fire.ly/)
    - Spark FHIR Server (https://github.com/FirelyTeam/spark)
    - Medplum FHIR Server (https://www.medplum.com/)
    - LinuxForHealth FHIR Server
    - Aidbox FHIR Platform
    - And more...
    
    Features:
    - Multi-engine operations (parallel execution)
    - Engine-specific optimizations
    - Automatic failover and load balancing
    - Performance monitoring and metrics
    - FHIR validation across engines
    - Batch and transaction operations
    """
    
    def __init__(
        self,
        agent_id: str = "enhanced-fhir-agent",
        name: str = "Enhanced Multi-Engine FHIR Agent",
        description: str = "FHIR agent with support for multiple open source FHIR engines",
        version: str = "2.0.0",
        config: Optional[Dict[str, Any]] = None,
    ):
        # Define enhanced capabilities
        capabilities = [
            AgentCapability(
                name="connect_fhir_engine",
                description="Connect to open source FHIR engines (HAPI, IBM, Firely, etc.)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "engine_type": {"type": "string", "enum": [e.value for e in FHIREngineType]},
                        "server_config": {"type": "object"},
                        "template_name": {"type": "string"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "connected": {"type": "boolean"},
                        "server_id": {"type": "string"},
                        "engine_info": {"type": "object"}
                    }
                }
            ),
            AgentCapability(
                name="multi_engine_search",
                description="Search resources across multiple FHIR engines simultaneously",
                input_schema={
                    "type": "object",
                    "properties": {
                        "resource_type": {"type": "string"},
                        "search_parameters": {"type": "object"},
                        "engines": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["resource_type"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "results": {"type": "object"},
                        "performance_metrics": {"type": "object"}
                    }
                }
            ),
            AgentCapability(
                name="engine_performance_analysis",
                description="Analyze and compare performance across FHIR engines",
                input_schema={
                    "type": "object",
                    "properties": {
                        "operation_type": {"type": "string"},
                        "resource_type": {"type": "string"},
                        "sample_size": {"type": "integer", "minimum": 1, "maximum": 1000}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "performance_report": {"type": "object"},
                        "recommendations": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="fhir_engine_migration",
                description="Migrate data between different FHIR engines",
                input_schema={
                    "type": "object",
                    "properties": {
                        "source_engine": {"type": "string"},
                        "target_engine": {"type": "string"},
                        "resource_types": {"type": "array"},
                        "migration_strategy": {"type": "string", "enum": ["full", "incremental", "selective"]}
                    },
                    "required": ["source_engine", "target_engine"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "migration_result": {"type": "object"},
                        "migrated_resources": {"type": "integer"},
                        "errors": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="validate_across_engines",
                description="Validate FHIR resources against multiple engines for compliance",
                input_schema={
                    "type": "object",
                    "properties": {
                        "resource": {"type": "object"},
                        "resource_type": {"type": "string"},
                        "engines": {"type": "array"}
                    },
                    "required": ["resource", "resource_type"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "validation_results": {"type": "object"},
                        "consensus_valid": {"type": "boolean"},
                        "engine_differences": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="bulk_fhir_operations",
                description="Execute bulk operations across multiple FHIR engines",
                input_schema={
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["export", "import", "delete", "update"]},
                        "resource_types": {"type": "array"},
                        "engines": {"type": "array"},
                        "parameters": {"type": "object"}
                    },
                    "required": ["operation"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "bulk_result": {"type": "object"},
                        "processed_count": {"type": "integer"},
                        "engine_results": {"type": "object"}
                    }
                }
            )
        ]
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            version=version,
            capabilities=capabilities,
            config=config or {}
        )
        
        # Initialize FHIR engine manager
        self.engine_manager = FHIREngineManager()
        self.fhir_config = FHIREngineConfig(**self.config.get("fhir_engines", {}))
        
        # Performance tracking
        self.performance_metrics = {
            "operations_count": 0,
            "total_execution_time": 0.0,
            "engine_performance": {},
            "error_count": 0
        }
        
        # Register task handlers
        self.register_task_handler("connect_fhir_engine", self._connect_fhir_engine)
        self.register_task_handler("list_engines", self._list_engines)
        self.register_task_handler("multi_engine_search", self._multi_engine_search)
        self.register_task_handler("multi_engine_create", self._multi_engine_create)
        self.register_task_handler("multi_engine_read", self._multi_engine_read)
        self.register_task_handler("multi_engine_update", self._multi_engine_update)
        self.register_task_handler("multi_engine_delete", self._multi_engine_delete)
        self.register_task_handler("validate_across_engines", self._validate_across_engines)
        self.register_task_handler("engine_performance_analysis", self._engine_performance_analysis)
        self.register_task_handler("fhir_engine_migration", self._fhir_engine_migration)
        self.register_task_handler("bulk_fhir_operations", self._bulk_fhir_operations)
        self.register_task_handler("get_engine_capabilities", self._get_engine_capabilities)
        self.register_task_handler("test_engine_connection", self._test_engine_connection)
    
    async def _on_start(self) -> None:
        """Initialize FHIR engines on agent start"""
        self.logger.info("Starting Enhanced FHIR Agent with multi-engine support")
        
        # Auto-connect to configured engines
        if self.fhir_config.auto_connect and self.fhir_config.enabled_engines:
            await self._auto_connect_engines()
        
        # Load common server templates
        await self._load_default_templates()
        
        self.logger.info("Enhanced FHIR Agent started", 
                        engines_count=len(self.engine_manager.clients))
    
    async def _on_stop(self) -> None:
        """Clean up FHIR engine connections"""
        await self.engine_manager.close_all_connections()
        self.logger.info("Enhanced FHIR Agent stopped")
    
    async def _auto_connect_engines(self) -> None:
        """Auto-connect to configured FHIR engines"""
        for engine_id in self.fhir_config.enabled_engines:
            try:
                template = get_server_template(engine_id)
                if template:
                    await self.engine_manager.add_server(template)
                    self.logger.info("Auto-connected to FHIR engine", engine_id=engine_id)
            except Exception as e:
                self.logger.error("Failed to auto-connect to engine", 
                                engine_id=engine_id, error=str(e))
    
    async def _load_default_templates(self) -> None:
        """Load default server templates"""
        templates = list_server_templates()
        self.logger.info("Available FHIR server templates", count=len(templates), templates=templates)
    
    async def _connect_fhir_engine(self, task: TaskRequest) -> TaskResponse:
        """Connect to a FHIR engine"""
        try:
            params = task.parameters
            
            # Option 1: Use pre-configured template
            if "template_name" in params:
                template = get_server_template(params["template_name"])
                if not template:
                    return TaskResponse(
                        task_id=task.task_id,
                        success=False,
                        error_message=f"Template not found: {params['template_name']}"
                    )
                config = template
            
            # Option 2: Create from server config
            elif "server_config" in params:
                config_data = params["server_config"]
                config = FHIRServerConfiguration(**config_data)
            
            else:
                return TaskResponse(
                    task_id=task.task_id,
                    success=False,
                    error_message="Either 'template_name' or 'server_config' required"
                )
            
            # Connect to the engine
            connected = await self.engine_manager.add_server(config)
            
            if connected:
                # Get engine capabilities
                client = self.engine_manager.get_client(config.server_id)
                capability_result = await client.get_capability_statement()
                
                return TaskResponse(
                    task_id=task.task_id,
                    success=True,
                    data={
                        "connected": True,
                        "server_id": config.server_id,
                        "engine_type": config.engine_type.value,
                        "base_url": config.base_url,
                        "fhir_version": config.fhir_version.value,
                        "capabilities": capability_result.data if capability_result.success else None
                    }
                )
            else:
                return TaskResponse(
                    task_id=task.task_id,
                    success=False,
                    error_message=f"Failed to connect to {config.engine_type.value} server"
                )
        
        except Exception as e:
            self.logger.error("Error connecting to FHIR engine", error=str(e))
            return TaskResponse(
                task_id=task.task_id,
                success=False,
                error_message=str(e)
            )
    
    async def _list_engines(self, task: TaskRequest) -> TaskResponse:
        """List all connected FHIR engines"""
        try:
            servers = self.engine_manager.list_servers()
            templates = list_server_templates()
            
            return TaskResponse(
                task_id=task.task_id,
                success=True,
                data={
                    "connected_servers": servers,
                    "available_templates": templates,
                    "total_connections": len(servers)
                }
            )
        
        except Exception as e:
            self.logger.error("Error listing FHIR engines", error=str(e))
            return TaskResponse(
                task_id=task.task_id,
                success=False,
                error_message=str(e)
            )
    
    async def _multi_engine_search(self, task: TaskRequest) -> TaskResponse:
        """Search resources across multiple FHIR engines"""
        try:
            params = task.parameters
            resource_type = params["resource_type"]
            search_params = FHIRSearchParameters(
                resource_type=resource_type,
                parameters=params.get("search_parameters", {}),
                count=params.get("count"),
                page=params.get("page"),
                sort=params.get("sort"),
                include=params.get("include"),
                revinclude=params.get("revinclude")
            )
            
            start_time = datetime.now()
            
            # Execute search on specified engines or all engines
            engines = params.get("engines", list(self.engine_manager.clients.keys()))
            if not engines:
                return TaskResponse(
                    task_id=task.task_id,
                    success=False,
                    error_message="No FHIR engines connected"
                )
            
            # Execute searches in parallel
            tasks = []
            for engine_id in engines:
                if engine_id in self.engine_manager.clients:
                    client = self.engine_manager.clients[engine_id]
                    search_task = asyncio.create_task(client.search_resources(search_params))
                    tasks.append((engine_id, search_task))
            
            # Collect results
            results = {}
            total_resources = 0
            successful_engines = 0
            
            for engine_id, search_task in tasks:
                try:
                    result = await search_task
                    results[engine_id] = {
                        "success": result.success,
                        "status_code": result.status_code,
                        "data": result.data,
                        "execution_time_ms": result.execution_time_ms,
                        "error_message": result.error_message
                    }
                    
                    if result.success and result.data:
                        # Count resources in bundle
                        if "entry" in result.data:
                            total_resources += len(result.data["entry"])
                        successful_engines += 1
                
                except Exception as e:
                    results[engine_id] = {
                        "success": False,
                        "error_message": str(e)
                    }
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update performance metrics
            self.performance_metrics["operations_count"] += 1
            self.performance_metrics["total_execution_time"] += execution_time
            
            return TaskResponse(
                task_id=task.task_id,
                success=True,
                data={
                    "operation": "multi_engine_search",
                    "resource_type": resource_type,
                    "total_engines": len(engines),
                    "successful_engines": successful_engines,
                    "total_resources_found": total_resources,
                    "execution_time_ms": execution_time,
                    "results": results,
                    "search_parameters": search_params.dict()
                }
            )
        
        except Exception as e:
            self.logger.error("Error in multi-engine search", error=str(e))
            return TaskResponse(
                task_id=task.task_id,
                success=False,
                error_message=str(e)
            )
    
    async def _validate_across_engines(self, task: TaskRequest) -> TaskResponse:
        """Validate FHIR resources across multiple engines"""
        try:
            params = task.parameters
            resource = params["resource"]
            resource_type = params["resource_type"]
            engines = params.get("engines", list(self.engine_manager.clients.keys()))
            
            if not engines:
                return TaskResponse(
                    task_id=task.task_id,
                    success=False,
                    error_message="No FHIR engines connected"
                )
            
            start_time = datetime.now()
            
            # Validate across engines in parallel
            validation_tasks = []
            for engine_id in engines:
                if engine_id in self.engine_manager.clients:
                    client = self.engine_manager.clients[engine_id]
                    validate_task = asyncio.create_task(
                        client.validate_resource(resource_type, resource)
                    )
                    validation_tasks.append((engine_id, validate_task))
            
            # Collect validation results
            validation_results = {}
            valid_count = 0
            engine_differences = []
            
            for engine_id, validate_task in validation_tasks:
                try:
                    result = await validate_task
                    validation_results[engine_id] = {
                        "valid": result.success,
                        "status_code": result.status_code,
                        "operation_outcome": result.data,
                        "execution_time_ms": result.execution_time_ms
                    }
                    
                    if result.success:
                        valid_count += 1
                    else:
                        # Track differences between engines
                        if result.data and "issue" in result.data:
                            engine_differences.append({
                                "engine": engine_id,
                                "issues": result.data["issue"]
                            })
                
                except Exception as e:
                    validation_results[engine_id] = {
                        "valid": False,
                        "error": str(e)
                    }
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            consensus_valid = valid_count == len(engines)
            
            return TaskResponse(
                task_id=task.task_id,
                success=True,
                data={
                    "operation": "validate_across_engines",
                    "resource_type": resource_type,
                    "total_engines": len(engines),
                    "valid_engines": valid_count,
                    "consensus_valid": consensus_valid,
                    "validation_results": validation_results,
                    "engine_differences": engine_differences,
                    "execution_time_ms": execution_time
                }
            )
        
        except Exception as e:
            self.logger.error("Error in cross-engine validation", error=str(e))
            return TaskResponse(
                task_id=task.task_id,
                success=False,
                error_message=str(e)
            )
    
    async def _engine_performance_analysis(self, task: TaskRequest) -> TaskResponse:
        """Analyze performance across FHIR engines"""
        try:
            params = task.parameters
            operation_type = params.get("operation_type", "read")
            resource_type = params.get("resource_type", "Patient")
            sample_size = params.get("sample_size", 10)
            
            engines = list(self.engine_manager.clients.keys())
            if not engines:
                return TaskResponse(
                    task_id=task.task_id,
                    success=False,
                    error_message="No FHIR engines connected"
                )
            
            performance_results = {}
            
            # Test each engine
            for engine_id in engines:
                client = self.engine_manager.clients[engine_id]
                engine_metrics = {
                    "response_times": [],
                    "success_count": 0,
                    "error_count": 0,
                    "avg_response_time": 0,
                    "min_response_time": float('inf'),
                    "max_response_time": 0
                }
                
                # Run sample operations
                for i in range(sample_size):
                    start_time = datetime.now()
                    
                    try:
                        # Perform operation based on type
                        if operation_type == "capability":
                            result = await client.get_capability_statement()
                        elif operation_type == "search":
                            search_params = FHIRSearchParameters(
                                resource_type=resource_type,
                                parameters={"_count": "1"}
                            )
                            result = await client.search_resources(search_params)
                        else:
                            # Default to capability statement
                            result = await client.get_capability_statement()
                        
                        execution_time = (datetime.now() - start_time).total_seconds() * 1000
                        engine_metrics["response_times"].append(execution_time)
                        
                        if result.success:
                            engine_metrics["success_count"] += 1
                        else:
                            engine_metrics["error_count"] += 1
                        
                        # Update min/max times
                        if execution_time < engine_metrics["min_response_time"]:
                            engine_metrics["min_response_time"] = execution_time
                        if execution_time > engine_metrics["max_response_time"]:
                            engine_metrics["max_response_time"] = execution_time
                    
                    except Exception as e:
                        engine_metrics["error_count"] += 1
                        self.logger.warning("Performance test error", 
                                          engine=engine_id, error=str(e))
                
                # Calculate averages
                if engine_metrics["response_times"]:
                    engine_metrics["avg_response_time"] = sum(engine_metrics["response_times"]) / len(engine_metrics["response_times"])
                
                performance_results[engine_id] = engine_metrics
            
            # Generate recommendations
            recommendations = []
            fastest_engine = min(performance_results.keys(), 
                               key=lambda x: performance_results[x]["avg_response_time"] 
                               if performance_results[x]["response_times"] else float('inf'))
            
            recommendations.append(f"Fastest engine for {operation_type} operations: {fastest_engine}")
            
            # Check for reliability
            for engine_id, metrics in performance_results.items():
                success_rate = metrics["success_count"] / sample_size if sample_size > 0 else 0
                if success_rate < 0.9:
                    recommendations.append(f"Engine {engine_id} has low success rate: {success_rate:.2%}")
            
            return TaskResponse(
                task_id=task.task_id,
                success=True,
                data={
                    "operation": "engine_performance_analysis",
                    "operation_type": operation_type,
                    "resource_type": resource_type,
                    "sample_size": sample_size,
                    "performance_results": performance_results,
                    "recommendations": recommendations,
                    "fastest_engine": fastest_engine
                }
            )
        
        except Exception as e:
            self.logger.error("Error in performance analysis", error=str(e))
            return TaskResponse(
                task_id=task.task_id,
                success=False,
                error_message=str(e)
            )
    
    async def _fhir_engine_migration(self, task: TaskRequest) -> TaskResponse:
        """Migrate data between FHIR engines"""
        try:
            params = task.parameters
            source_engine = params["source_engine"]
            target_engine = params["target_engine"]
            resource_types = params.get("resource_types", ["Patient"])
            migration_strategy = params.get("migration_strategy", "selective")
            
            # Validate engines exist
            source_client = self.engine_manager.get_client(source_engine)
            target_client = self.engine_manager.get_client(target_engine)
            
            if not source_client:
                return TaskResponse(
                    task_id=task.task_id,
                    success=False,
                    error_message=f"Source engine not found: {source_engine}"
                )
            
            if not target_client:
                return TaskResponse(
                    task_id=task.task_id,
                    success=False,
                    error_message=f"Target engine not found: {target_engine}"
                )
            
            migration_results = {
                "total_resources": 0,
                "migrated_resources": 0,
                "failed_resources": 0,
                "errors": [],
                "resource_type_results": {}
            }
            
            # Migrate each resource type
            for resource_type in resource_types:
                self.logger.info("Migrating resource type", 
                               resource_type=resource_type, 
                               source=source_engine, 
                               target=target_engine)
                
                # Search for resources in source
                search_params = FHIRSearchParameters(
                    resource_type=resource_type,
                    parameters={"_count": "100"}  # Batch size
                )
                
                search_result = await source_client.search_resources(search_params)
                
                if not search_result.success:
                    error_msg = f"Failed to search {resource_type} in source engine"
                    migration_results["errors"].append(error_msg)
                    continue
                
                # Process search results
                resources = []
                if search_result.data and "entry" in search_result.data:
                    resources = [entry["resource"] for entry in search_result.data["entry"]]
                
                migration_results["total_resources"] += len(resources)
                
                type_results = {
                    "total": len(resources),
                    "migrated": 0,
                    "failed": 0,
                    "errors": []
                }
                
                # Migrate each resource
                for resource in resources:
                    try:
                        # Remove server-specific elements
                        clean_resource = resource.copy()
                        clean_resource.pop("id", None)  # Let target assign new ID
                        clean_resource.pop("meta", None)  # Remove metadata
                        
                        # Create in target engine
                        create_result = await target_client.create_resource(
                            resource_type, clean_resource
                        )
                        
                        if create_result.success:
                            type_results["migrated"] += 1
                            migration_results["migrated_resources"] += 1
                        else:
                            type_results["failed"] += 1
                            migration_results["failed_resources"] += 1
                            type_results["errors"].append(create_result.error_message)
                    
                    except Exception as e:
                        type_results["failed"] += 1
                        migration_results["failed_resources"] += 1
                        type_results["errors"].append(str(e))
                
                migration_results["resource_type_results"][resource_type] = type_results
            
            success = migration_results["failed_resources"] == 0
            
            return TaskResponse(
                task_id=task.task_id,
                success=success,
                data={
                    "operation": "fhir_engine_migration",
                    "source_engine": source_engine,
                    "target_engine": target_engine,
                    "migration_strategy": migration_strategy,
                    "migration_results": migration_results
                }
            )
        
        except Exception as e:
            self.logger.error("Error in FHIR engine migration", error=str(e))
            return TaskResponse(
                task_id=task.task_id,
                success=False,
                error_message=str(e)
            )
    
    async def _bulk_fhir_operations(self, task: TaskRequest) -> TaskResponse:
        """Execute bulk operations across FHIR engines"""
        # Implementation for bulk operations
        # This would include export, import, bulk delete, etc.
        return TaskResponse(
            task_id=task.task_id,
            success=True,
            data={"message": "Bulk operations implementation coming soon"}
        )
    
    async def _get_engine_capabilities(self, task: TaskRequest) -> TaskResponse:
        """Get capabilities for all connected FHIR engines"""
        try:
            capabilities = {}
            
            for engine_id, client in self.engine_manager.clients.items():
                capability_result = await client.get_capability_statement()
                capabilities[engine_id] = {
                    "success": capability_result.success,
                    "capabilities": capability_result.data if capability_result.success else None,
                    "error": capability_result.error_message
                }
            
            return TaskResponse(
                task_id=task.task_id,
                success=True,
                data={
                    "engine_capabilities": capabilities,
                    "total_engines": len(capabilities)
                }
            )
        
        except Exception as e:
            self.logger.error("Error getting engine capabilities", error=str(e))
            return TaskResponse(
                task_id=task.task_id,
                success=False,
                error_message=str(e)
            )
    
    async def _test_engine_connection(self, task: TaskRequest) -> TaskResponse:
        """Test connection to specified FHIR engines"""
        try:
            params = task.parameters
            engines = params.get("engines", list(self.engine_manager.clients.keys()))
            
            connection_results = {}
            
            for engine_id in engines:
                if engine_id in self.engine_manager.clients:
                    result = await self.engine_manager.test_connection(engine_id)
                    connection_results[engine_id] = {
                        "connected": result.success,
                        "status_code": result.status_code,
                        "response_time_ms": result.execution_time_ms,
                        "error": result.error_message
                    }
                else:
                    connection_results[engine_id] = {
                        "connected": False,
                        "error": "Engine not found"
                    }
            
            return TaskResponse(
                task_id=task.task_id,
                success=True,
                data={
                    "connection_tests": connection_results,
                    "total_tested": len(engines)
                }
            )
        
        except Exception as e:
            self.logger.error("Error testing engine connections", error=str(e))
            return TaskResponse(
                task_id=task.task_id,
                success=False,
                error_message=str(e)
            )
    
    # Additional helper methods for multi-engine operations
    async def _multi_engine_create(self, task: TaskRequest) -> TaskResponse:
        """Create resource across multiple engines"""
        params = task.parameters
        resource_type = params["resource_type"]
        resource = params["resource"]
        engines = params.get("engines", list(self.engine_manager.clients.keys()))
        
        results = await self.engine_manager.execute_on_all_servers(
            "create", resource_type=resource_type, resource=resource
        )
        
        successful = sum(1 for r in results.values() if r.success)
        
        return TaskResponse(
            task_id=task.task_id,
            success=successful > 0,
            data={
                "operation": "multi_engine_create",
                "total_engines": len(results),
                "successful_engines": successful,
                "results": {k: {"success": v.success, "data": v.data, "error": v.error_message} 
                          for k, v in results.items()}
            }
        )
    
    async def _multi_engine_read(self, task: TaskRequest) -> TaskResponse:
        """Read resource from multiple engines"""
        params = task.parameters
        resource_type = params["resource_type"]
        resource_id = params["resource_id"]
        engines = params.get("engines", list(self.engine_manager.clients.keys()))
        
        results = {}
        for engine_id in engines:
            if engine_id in self.engine_manager.clients:
                result = await self.engine_manager.execute_on_server(
                    engine_id, "read", resource_type=resource_type, resource_id=resource_id
                )
                results[engine_id] = result
        
        successful = sum(1 for r in results.values() if r.success)
        
        return TaskResponse(
            task_id=task.task_id,
            success=successful > 0,
            data={
                "operation": "multi_engine_read",
                "resource_type": resource_type,
                "resource_id": resource_id,
                "total_engines": len(results),
                "successful_engines": successful,
                "results": {k: {"success": v.success, "data": v.data, "error": v.error_message} 
                          for k, v in results.items()}
            }
        )
    
    async def _multi_engine_update(self, task: TaskRequest) -> TaskResponse:
        """Update resource across multiple engines"""
        params = task.parameters
        resource_type = params["resource_type"]
        resource_id = params["resource_id"]
        resource = params["resource"]
        engines = params.get("engines", list(self.engine_manager.clients.keys()))
        
        results = {}
        for engine_id in engines:
            if engine_id in self.engine_manager.clients:
                result = await self.engine_manager.execute_on_server(
                    engine_id, "update", 
                    resource_type=resource_type, 
                    resource_id=resource_id, 
                    resource=resource
                )
                results[engine_id] = result
        
        successful = sum(1 for r in results.values() if r.success)
        
        return TaskResponse(
            task_id=task.task_id,
            success=successful > 0,
            data={
                "operation": "multi_engine_update",
                "total_engines": len(results),
                "successful_engines": successful,
                "results": {k: {"success": v.success, "data": v.data, "error": v.error_message} 
                          for k, v in results.items()}
            }
        )
    
    async def _multi_engine_delete(self, task: TaskRequest) -> TaskResponse:
        """Delete resource from multiple engines"""
        params = task.parameters
        resource_type = params["resource_type"]
        resource_id = params["resource_id"]
        engines = params.get("engines", list(self.engine_manager.clients.keys()))
        
        results = {}
        for engine_id in engines:
            if engine_id in self.engine_manager.clients:
                result = await self.engine_manager.execute_on_server(
                    engine_id, "delete", 
                    resource_type=resource_type, 
                    resource_id=resource_id
                )
                results[engine_id] = result
        
        successful = sum(1 for r in results.values() if r.success)
        
        return TaskResponse(
            task_id=task.task_id,
            success=successful > 0,
            data={
                "operation": "multi_engine_delete",
                "total_engines": len(results),
                "successful_engines": successful,
                "results": {k: {"success": v.success, "data": v.data, "error": v.error_message} 
                          for k, v in results.items()}
            }
        )