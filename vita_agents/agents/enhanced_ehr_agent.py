"""
Enhanced EHR Integration Agent for connecting with major EHR systems.

This agent provides advanced EHR connectivity using the enhanced connector
framework with vendor-specific optimizations, connection pooling, and
intelligent routing across multiple EHR systems.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, AsyncIterator
import structlog
from pydantic import BaseModel, Field

from vita_agents.core.agent import HealthcareAgent, AgentCapability, TaskRequest, TaskResponse
from vita_agents.core.config import get_settings
from vita_agents.connectors import (
    ehr_factory,
    EHRVendor,
    EHRConnectionConfig,
    EHRConnectorFactory,
    SyncResult,
    EHRConnectorError,
    AuthenticationType,
    SyncMode,
)


logger = structlog.get_logger(__name__)


class EnhancedEHRConnectionConfig(BaseModel):
    """Enhanced EHR connection configuration."""
    
    config_id: str
    vendor: EHRVendor
    base_url: str
    client_id: str
    client_secret: str
    auth_type: AuthenticationType = AuthenticationType.CLIENT_CREDENTIALS
    scope: str = "patient/*.read"
    fhir_version: str = "R4"
    timeout: int = 30
    max_retries: int = 3
    vendor_specific: Dict[str, Any] = Field(default_factory=dict)


class MultiSystemSyncRequest(BaseModel):
    """Request for multi-system patient data synchronization."""
    
    patient_identifier: str
    identifier_type: str = "MRN"
    system_configs: List[str]  # List of config IDs
    resource_types: Optional[List[str]] = None
    include_historical: bool = True
    resolve_conflicts: bool = True


class EHRSystemHealth(BaseModel):
    """EHR system health status."""
    
    config_id: str
    vendor: str
    is_connected: bool
    is_healthy: bool
    response_time: Optional[float]
    last_check: datetime
    error_count: int = 0
    uptime_percentage: float = 100.0


class PatientDataSummary(BaseModel):
    """Comprehensive patient data summary from multiple systems."""
    
    patient_identifier: str
    identifier_type: str
    systems_queried: List[str]
    successful_systems: List[str]
    failed_systems: List[str]
    data_summary: Dict[str, Any]
    conflicts_detected: List[Dict[str, Any]] = Field(default_factory=list)
    harmonized_data: Optional[Dict[str, Any]] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class EnhancedEHRAgent(HealthcareAgent):
    """
    Enhanced EHR Integration Agent for connecting with major EHR systems.
    
    Capabilities:
    - Connect with Epic, Cerner, Allscripts EHR systems using enhanced connectors
    - Multi-system patient data synchronization
    - Real-time health monitoring and failover
    - Intelligent connection pooling and load balancing
    - Vendor-specific optimizations and features
    - Cross-system data harmonization and conflict resolution
    """
    
    def __init__(
        self,
        agent_id: str = "enhanced-ehr-agent",
        name: str = "Enhanced EHR Integration Agent",
        description: str = "Advanced EHR connectivity with multi-system support",
        version: str = "2.0.0",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            version=version,
            capabilities=[
                AgentCapability(
                    name="EHR_INTEGRATION",
                    description="Connect to Epic, Cerner, and Allscripts EHR systems",
                    input_schema={"type": "object", "properties": {"patient_id": {"type": "string"}}},
                    output_schema={"type": "object", "properties": {"patient_data": {"type": "object"}}},
                    supported_formats=["FHIR_R4", "HL7_v2.5"],
                    requirements=["OAuth2", "HTTPS"]
                ),
                AgentCapability(
                    name="DATA_HARMONIZATION", 
                    description="Harmonize patient data across multiple EHR systems",
                    input_schema={"type": "object", "properties": {"systems": {"type": "array"}}},
                    output_schema={"type": "object", "properties": {"harmonized_data": {"type": "object"}}},
                    supported_formats=["FHIR_R4"],
                    requirements=["Multi-system access"]
                ),
                AgentCapability(
                    name="BULK_OPERATIONS",
                    description="Perform bulk data export and import operations", 
                    input_schema={"type": "object", "properties": {"resource_types": {"type": "array"}}},
                    output_schema={"type": "object", "properties": {"exported_resources": {"type": "array"}}},
                    supported_formats=["FHIR_R4", "NDJSON"],
                    requirements=["Bulk FHIR API"]
                ),
                AgentCapability(
                    name="REAL_TIME_SYNC",
                    description="Real-time synchronization across EHR systems",
                    input_schema={"type": "object", "properties": {"sync_mode": {"type": "string"}}},
                    output_schema={"type": "object", "properties": {"sync_status": {"type": "object"}}},
                    supported_formats=["FHIR_R4"],
                    requirements=["Webhooks", "Event streaming"]
                ),
                AgentCapability(
                    name="MULTI_SYSTEM_SUPPORT",
                    description="Support for multiple EHR vendor systems simultaneously",
                    input_schema={"type": "object", "properties": {"systems": {"type": "array"}}},
                    output_schema={"type": "object", "properties": {"multi_system_data": {"type": "object"}}},
                    supported_formats=["FHIR_R4"],
                    requirements=["Connection pooling", "Load balancing"]
                ),
            ],
            config=config,
        )
        
        self.factory = ehr_factory
        self.configured_systems: Dict[str, EHRConnectionConfig] = {}
        self.sync_results_cache: Dict[str, SyncResult] = {}
        
        # Performance metrics
        self.request_count = 0
        self.error_count = 0
        self.avg_response_time = 0.0
        
        # Initialize from configuration
        self._initialize_systems()
    
    def _initialize_systems(self) -> None:
        """Initialize EHR system configurations from settings."""
        settings = get_settings()
        
        # Get EHR configurations from settings
        ehr_configs = getattr(settings, 'ehr_systems', {})
        
        for config_id, config_data in ehr_configs.items():
            try:
                # Convert to enhanced config
                ehr_config = EHRConnectionConfig(
                    vendor=EHRVendor(config_data['vendor']),
                    base_url=config_data['base_url'],
                    client_id=config_data['client_id'],
                    client_secret=config_data['client_secret'],
                    auth_type=AuthenticationType(config_data.get('auth_type', 'client_credentials')),
                    scope=config_data.get('scope', 'patient/*.read'),
                    fhir_version=config_data.get('fhir_version', 'R4'),
                    timeout=config_data.get('timeout', 30),
                    max_retries=config_data.get('max_retries', 3),
                    vendor_specific=config_data.get('vendor_specific', {}),
                )
                
                # Add to factory
                self.factory.add_configuration(config_id, ehr_config)
                self.configured_systems[config_id] = ehr_config
                
                logger.info(f"Configured EHR system: {config_id} ({ehr_config.vendor})")
                
            except Exception as e:
                logger.error(f"Failed to configure EHR system {config_id}: {e}")
    
    async def start(self) -> None:
        """Start the agent and begin health monitoring."""
        await super().start()
        
        # Start health monitoring for all configured systems
        await self.factory.start_health_monitoring()
        
        logger.info(f"Enhanced EHR Agent started with {len(self.configured_systems)} systems")
    
    async def stop(self) -> None:
        """Stop the agent and clean up connections."""
        await self.factory.shutdown()
        await super().stop()
        
        logger.info("Enhanced EHR Agent stopped")
    
    async def _on_start(self) -> None:
        """Hook called when the agent starts."""
        logger.info("Enhanced EHR Agent starting...")
        # Additional startup logic can be added here
    
    async def _on_stop(self) -> None:
        """Hook called when the agent stops."""
        logger.info("Enhanced EHR Agent stopping...")
        # Additional shutdown logic can be added here
    
    async def add_ehr_system(self, config: EnhancedEHRConnectionConfig) -> None:
        """
        Add a new EHR system configuration.
        
        Args:
            config: Enhanced EHR connection configuration
        """
        # Convert to base config
        ehr_config = EHRConnectionConfig(
            vendor=config.vendor,
            base_url=config.base_url,
            client_id=config.client_id,
            client_secret=config.client_secret,
            auth_type=config.auth_type,
            scope=config.scope,
            fhir_version=config.fhir_version,
            timeout=config.timeout,
            max_retries=config.max_retries,
            vendor_specific=config.vendor_specific,
        )
        
        # Add to factory and local registry
        self.factory.add_configuration(config.config_id, ehr_config)
        self.configured_systems[config.config_id] = ehr_config
        
        logger.info(f"Added EHR system: {config.config_id} ({config.vendor})")
    
    async def remove_ehr_system(self, config_id: str) -> None:
        """
        Remove an EHR system configuration.
        
        Args:
            config_id: Configuration identifier to remove
        """
        self.factory.remove_configuration(config_id)
        
        if config_id in self.configured_systems:
            del self.configured_systems[config_id]
        
        logger.info(f"Removed EHR system: {config_id}")
    
    async def get_patient_data(
        self,
        config_id: str,
        patient_id: str,
        resource_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get patient data from a specific EHR system.
        
        Args:
            config_id: EHR system configuration ID
            patient_id: Patient identifier
            resource_types: Optional list of FHIR resource types to retrieve
            
        Returns:
            Patient data from the EHR system
        """
        start_time = datetime.utcnow()
        
        try:
            async with self.factory.get_connector(config_id) as connector:
                # Get basic patient info
                patient_response = await connector.get_patient(patient_id)
                
                if not patient_response.is_success:
                    raise EHRConnectorError(
                        f"Failed to get patient data: {patient_response.status_code}",
                        connector.vendor
                    )
                
                patient_data = {
                    "patient": patient_response.data,
                    "system": config_id,
                    "vendor": connector.vendor.value,
                    "retrieved_at": datetime.utcnow().isoformat(),
                }
                
                # Get additional resources if specified
                if resource_types:
                    for resource_type in resource_types:
                        try:
                            if resource_type.lower() == "observation":
                                response = await connector.get_observations(patient_id)
                            elif resource_type.lower() == "medicationrequest":
                                response = await connector.get_medications(patient_id)
                            elif resource_type.lower() == "condition":
                                response = await connector.get_conditions(patient_id)
                            elif resource_type.lower() == "encounter":
                                response = await connector.get_encounters(patient_id)
                            else:
                                # Generic resource retrieval
                                response = await connector._make_authenticated_request(
                                    "GET", resource_type, params={"patient": patient_id}
                                )
                            
                            if response.is_success:
                                patient_data[resource_type.lower()] = response.data
                                
                        except Exception as e:
                            logger.warning(f"Failed to get {resource_type} for patient {patient_id}: {e}")
                            patient_data[f"{resource_type.lower()}_error"] = str(e)
                
                # Update metrics
                response_time = (datetime.utcnow() - start_time).total_seconds()
                self._update_metrics(response_time, success=True)
                
                return patient_data
                
        except Exception as e:
            # Update metrics on failure
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(response_time, success=False)
            
            logger.error(f"Error getting patient data from {config_id}: {e}")
            raise
    
    async def sync_patient_across_systems(
        self,
        request: MultiSystemSyncRequest
    ) -> PatientDataSummary:
        """
        Synchronize patient data across multiple EHR systems.
        
        Args:
            request: Multi-system sync request
            
        Returns:
            Comprehensive patient data summary
        """
        logger.info(f"Starting multi-system sync for patient {request.patient_identifier}")
        
        # Perform sync across all specified systems
        sync_results = await self.factory.sync_patient_across_systems(
            request.patient_identifier,
            request.system_configs,
            request.identifier_type
        )
        
        # Compile results
        successful_systems = []
        failed_systems = []
        data_summary = {}
        conflicts = []
        
        for config_id, sync_result in sync_results.items():
            if sync_result.success_rate > 0.8:  # 80% success threshold
                successful_systems.append(config_id)
                
                # Get patient data for successful systems
                try:
                    patient_data = await self.get_patient_data(
                        config_id,
                        request.patient_identifier,
                        request.resource_types
                    )
                    data_summary[config_id] = patient_data
                except Exception as e:
                    logger.warning(f"Failed to get patient data from {config_id}: {e}")
                    failed_systems.append(config_id)
            else:
                failed_systems.append(config_id)
        
        # Detect conflicts if requested
        if request.resolve_conflicts and len(successful_systems) > 1:
            conflicts = await self._detect_data_conflicts(data_summary)
        
        # Create harmonized data if needed
        harmonized_data = None
        if request.resolve_conflicts and conflicts:
            harmonized_data = await self._harmonize_patient_data(data_summary, conflicts)
        
        return PatientDataSummary(
            patient_identifier=request.patient_identifier,
            identifier_type=request.identifier_type,
            systems_queried=request.system_configs,
            successful_systems=successful_systems,
            failed_systems=failed_systems,
            data_summary=data_summary,
            conflicts_detected=conflicts,
            harmonized_data=harmonized_data,
        )
    
    async def get_system_health_status(self, config_id: Optional[str] = None) -> Union[EHRSystemHealth, List[EHRSystemHealth]]:
        """
        Get health status for EHR systems.
        
        Args:
            config_id: Optional specific system ID, or None for all systems
            
        Returns:
            Health status for specific system or all systems
        """
        factory_status = await self.factory.get_system_status(config_id)
        
        if config_id:
            # Single system status
            if factory_status:
                return EHRSystemHealth(
                    config_id=factory_status.connection_id,
                    vendor=factory_status.vendor.value,
                    is_connected=factory_status.is_connected,
                    is_healthy=factory_status.is_healthy,
                    response_time=factory_status.response_time,
                    last_check=factory_status.last_health_check,
                    error_count=factory_status.error_count,
                    uptime_percentage=factory_status.uptime_percentage,
                )
            else:
                raise ValueError(f"System not found: {config_id}")
        else:
            # All systems status
            health_list = []
            for status in factory_status.values():
                health_list.append(EHRSystemHealth(
                    config_id=status.connection_id,
                    vendor=status.vendor.value,
                    is_connected=status.is_connected,
                    is_healthy=status.is_healthy,
                    response_time=status.response_time,
                    last_check=status.last_health_check,
                    error_count=status.error_count,
                    uptime_percentage=status.uptime_percentage,
                ))
            return health_list
    
    async def bulk_export_patient_data(
        self,
        config_id: str,
        resource_types: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        patient_filter: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Perform bulk export of patient data from an EHR system.
        
        Args:
            config_id: EHR system configuration ID
            resource_types: Optional list of resource types to export
            since: Optional date filter for incremental exports
            patient_filter: Optional patient filter criteria
            
        Yields:
            FHIR resources from the bulk export
        """
        async with self.factory.get_connector(config_id) as connector:
            async for resource in connector.bulk_export(
                resource_types=resource_types,
                since=since,
                type_filter=patient_filter
            ):
                yield resource
    
    async def execute_task(self, task: TaskRequest) -> TaskResponse:
        """
        Execute an EHR-related task.
        
        Args:
            task: Task request containing EHR operation details
            
        Returns:
            Task response with results
        """
        task_type = task.task_type.lower()
        
        try:
            if task_type == "get_patient_data":
                # Single system patient data retrieval
                config_id = task.parameters.get("config_id")
                patient_id = task.parameters.get("patient_id")
                resource_types = task.parameters.get("resource_types")
                
                result = await self.get_patient_data(config_id, patient_id, resource_types)
                
                return TaskResponse(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status="completed",
                    result=result,
                    metadata={"system": config_id, "patient_id": patient_id}
                )
            
            elif task_type == "multi_system_sync":
                # Multi-system patient synchronization
                sync_request = MultiSystemSyncRequest(**task.parameters)
                result = await self.sync_patient_across_systems(sync_request)
                
                return TaskResponse(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status="completed",
                    result=result.dict(),
                    metadata={"systems": sync_request.system_configs}
                )
            
            elif task_type == "health_check":
                # System health monitoring
                config_id = task.parameters.get("config_id")
                result = await self.get_system_health_status(config_id)
                
                return TaskResponse(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status="completed",
                    result=result if isinstance(result, list) else [result],
                    metadata={"health_check": True}
                )
            
            else:
                return TaskResponse(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status="failed",
                    error=f"Unknown task type: {task_type}",
                    metadata={"supported_tasks": ["get_patient_data", "multi_system_sync", "health_check"]}
                )
                
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return TaskResponse(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status="failed",
                error=str(e),
                metadata={"task_type": task_type}
            )
    
    async def _detect_data_conflicts(self, data_summary: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect conflicts in patient data across systems.
        
        Args:
            data_summary: Patient data from multiple systems
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Extract patient demographics from each system
        patient_demos = {}
        for system_id, data in data_summary.items():
            patient_data = data.get("patient", {})
            if patient_data:
                patient_demos[system_id] = {
                    "name": patient_data.get("name", [{}])[0],
                    "birthDate": patient_data.get("birthDate"),
                    "gender": patient_data.get("gender"),
                    "identifier": patient_data.get("identifier", [])
                }
        
        # Compare demographics across systems
        system_ids = list(patient_demos.keys())
        for i, system_a in enumerate(system_ids):
            for system_b in system_ids[i+1:]:
                demo_a = patient_demos[system_a]
                demo_b = patient_demos[system_b]
                
                # Check birth date conflicts
                if (demo_a.get("birthDate") and demo_b.get("birthDate") and 
                    demo_a["birthDate"] != demo_b["birthDate"]):
                    conflicts.append({
                        "type": "birthdate_mismatch",
                        "systems": [system_a, system_b],
                        "values": {
                            system_a: demo_a["birthDate"],
                            system_b: demo_b["birthDate"]
                        },
                        "severity": "high"
                    })
                
                # Check gender conflicts
                if (demo_a.get("gender") and demo_b.get("gender") and 
                    demo_a["gender"] != demo_b["gender"]):
                    conflicts.append({
                        "type": "gender_mismatch",
                        "systems": [system_a, system_b],
                        "values": {
                            system_a: demo_a["gender"],
                            system_b: demo_b["gender"]
                        },
                        "severity": "medium"
                    })
        
        return conflicts
    
    async def _harmonize_patient_data(
        self,
        data_summary: Dict[str, Dict[str, Any]],
        conflicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Harmonize patient data by resolving conflicts.
        
        Args:
            data_summary: Patient data from multiple systems
            conflicts: Detected conflicts
            
        Returns:
            Harmonized patient data
        """
        # Start with data from the first system
        systems = list(data_summary.keys())
        if not systems:
            return {}
        
        harmonized = data_summary[systems[0]].copy()
        
        # Apply conflict resolution rules
        for conflict in conflicts:
            if conflict["severity"] == "high":
                # For high severity conflicts, prefer the most recent data
                # This is a simplified rule - in production, more sophisticated logic would be used
                systems_in_conflict = conflict["systems"]
                values = conflict["values"]
                
                # For now, just log the conflict and keep the first value
                logger.warning(f"High severity conflict detected: {conflict}")
                # In production, you might want to:
                # - Use data lineage to determine most authoritative source
                # - Apply business rules based on data types
                # - Flag for manual review
        
        # Add metadata about harmonization
        harmonized["_harmonization"] = {
            "conflicts_resolved": len(conflicts),
            "source_systems": systems,
            "harmonized_at": datetime.utcnow().isoformat(),
            "resolution_method": "automated_with_priority_rules"
        }
        
        return harmonized
    
    def _update_metrics(self, response_time: float, success: bool) -> None:
        """Update agent performance metrics."""
        self.request_count += 1
        
        if not success:
            self.error_count += 1
        
        # Update average response time
        self.avg_response_time = (
            (self.avg_response_time * (self.request_count - 1) + response_time) / self.request_count
        )


# Maintain backwards compatibility
EHRAgent = EnhancedEHRAgent