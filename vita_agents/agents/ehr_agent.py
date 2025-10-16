"""
EHR Integration Agent for connecting with major EHR systems.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import structlog
from pydantic import BaseModel, Field
import httpx
import base64
from urllib.parse import urlencode

from vita_agents.core.agent import HealthcareAgent, AgentCapability, TaskRequest
from vita_agents.core.config import get_settings


logger = structlog.get_logger(__name__)


class EHRConnectionConfig(BaseModel):
    """EHR connection configuration."""
    
    vendor: str  # epic, cerner, allscripts
    base_url: str
    client_id: str
    client_secret: str
    auth_url: str
    token_url: str
    api_version: str = "R4"
    timeout: int = 30


class BulkDataRequest(BaseModel):
    """Bulk data export request."""
    
    resource_types: List[str]
    since: Optional[datetime] = None
    patient_filter: Optional[str] = None
    group_id: Optional[str] = None
    export_format: str = "ndjson"


class EHRAgent(HealthcareAgent):
    """
    EHR Integration Agent for connecting with major EHR systems.
    
    Capabilities:
    - Connect with Epic, Cerner, Allscripts EHR systems
    - Handle OAuth 2.0 authentication and token management
    - Perform bulk data operations and FHIR exports
    - Real-time data synchronization
    - Patient data retrieval and updates
    - Clinical data search and filtering
    """
    
    def __init__(
        self,
        agent_id: str = "ehr-agent",
        name: str = "EHR Integration Agent",
        description: str = "Connects with major EHR systems for data integration",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None,
    ):
        # Define EHR-specific capabilities
        capabilities = [
            AgentCapability(
                name="connect_ehr",
                description="Establish connection with EHR system",
                input_schema={
                    "type": "object",
                    "properties": {
                        "vendor": {"type": "string", "enum": ["epic", "cerner", "allscripts"]},
                        "connection_config": {"type": "object"},
                        "test_connection": {"type": "boolean", "default": True}
                    },
                    "required": ["vendor", "connection_config"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "connection_id": {"type": "string"},
                        "status": {"type": "string"},
                        "capabilities": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="bulk_export",
                description="Perform bulk data export from EHR",
                input_schema={
                    "type": "object",
                    "properties": {
                        "connection_id": {"type": "string"},
                        "export_request": {"type": "object"},
                        "callback_url": {"type": "string"}
                    },
                    "required": ["connection_id", "export_request"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "export_id": {"type": "string"},
                        "status": {"type": "string"},
                        "estimated_completion": {"type": "string"}
                    }
                }
            ),
            AgentCapability(
                name="search_patients",
                description="Search for patients in EHR system",
                input_schema={
                    "type": "object",
                    "properties": {
                        "connection_id": {"type": "string"},
                        "search_criteria": {"type": "object"},
                        "limit": {"type": "integer", "default": 50}
                    },
                    "required": ["connection_id", "search_criteria"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "patients": {"type": "array"},
                        "total": {"type": "integer"},
                        "has_more": {"type": "boolean"}
                    }
                }
            ),
            AgentCapability(
                name="get_patient_data",
                description="Retrieve comprehensive patient data",
                input_schema={
                    "type": "object",
                    "properties": {
                        "connection_id": {"type": "string"},
                        "patient_id": {"type": "string"},
                        "include_resources": {"type": "array"},
                        "date_range": {"type": "object"}
                    },
                    "required": ["connection_id", "patient_id"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "patient": {"type": "object"},
                        "resources": {"type": "object"},
                        "summary": {"type": "object"}
                    }
                }
            ),
        ]
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            version=version,
            capabilities=capabilities,
            config=config or {}
        )
        
        # EHR-specific configuration
        self.supported_vendors = ["epic", "cerner", "allscripts"]
        self.connections: Dict[str, EHRConnectionConfig] = {}
        self.access_tokens: Dict[str, Dict[str, Any]] = {}
        
        # Healthcare standards
        self.supported_standards = ["FHIR", "HL7", "EHR"]
        self.data_formats = ["json", "xml", "ndjson"]
        self.compliance_features = ["HIPAA", "OAuth2", "audit-logging", "rate-limiting"]
        
        # HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Register task handlers
        self.register_task_handler("connect_ehr", self._connect_ehr)
        self.register_task_handler("bulk_export", self._bulk_export)
        self.register_task_handler("search_patients", self._search_patients)
        self.register_task_handler("get_patient_data", self._get_patient_data)
        self.register_task_handler("sync_data", self._sync_data)
        self.register_task_handler("check_export_status", self._check_export_status)
    
    async def _on_start(self) -> None:
        """Initialize EHR connections."""
        self.logger.info("Starting EHR agent")
        
        # Load EHR configurations from settings
        settings = get_settings()
        
        # Epic configuration
        if settings.ehr.epic_client_id and settings.ehr.epic_client_secret:
            epic_config = EHRConnectionConfig(
                vendor="epic",
                base_url="https://fhir.epic.com/interconnect-fhir-oauth",
                client_id=settings.ehr.epic_client_id,
                client_secret=settings.ehr.epic_client_secret,
                auth_url="https://fhir.epic.com/interconnect-fhir-oauth/oauth2/authorize",
                token_url="https://fhir.epic.com/interconnect-fhir-oauth/oauth2/token"
            )
            self.connections["epic"] = epic_config
        
        # Cerner configuration
        if settings.ehr.cerner_client_id and settings.ehr.cerner_client_secret:
            cerner_config = EHRConnectionConfig(
                vendor="cerner",
                base_url="https://fhir-open.cerner.com",
                client_id=settings.ehr.cerner_client_id,
                client_secret=settings.ehr.cerner_client_secret,
                auth_url="https://authorization.cerner.com/tenants/{tenant_id}/protocols/oauth2/profiles/smart-v1/personas/provider/authorize",
                token_url="https://authorization.cerner.com/tenants/{tenant_id}/protocols/oauth2/profiles/smart-v1/token"
            )
            self.connections["cerner"] = cerner_config
        
        self.logger.info("EHR agent initialized", connections=list(self.connections.keys()))
    
    async def _on_stop(self) -> None:
        """Clean up EHR connections."""
        self.logger.info("Stopping EHR agent")
        await self.http_client.aclose()
    
    async def _connect_ehr(self, task: TaskRequest) -> Dict[str, Any]:
        """Establish connection with EHR system."""
        try:
            vendor = task.parameters.get("vendor")
            connection_config = task.parameters.get("connection_config", {})
            test_connection = task.parameters.get("test_connection", True)
            
            if vendor not in self.supported_vendors:
                raise ValueError(f"Unsupported EHR vendor: {vendor}")
            
            self.audit_log_action(
                action="connect_ehr",
                data_type="EHR Connection",
                details={
                    "vendor": vendor,
                    "test_connection": test_connection,
                    "task_id": task.id
                }
            )
            
            # Create or update connection configuration
            if vendor in self.connections:
                config = self.connections[vendor]
            else:
                config = EHRConnectionConfig(**connection_config)
                self.connections[vendor] = config
            
            connection_id = f"{vendor}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            # Obtain access token
            token_info = await self._obtain_access_token(config)
            self.access_tokens[connection_id] = token_info
            
            # Test connection if requested
            capabilities = []
            if test_connection:
                test_result = await self._test_connection(config, token_info["access_token"])
                capabilities = test_result.get("capabilities", [])
            
            return {
                "connection_id": connection_id,
                "status": "connected",
                "vendor": vendor,
                "capabilities": capabilities,
                "token_expires_at": token_info["expires_at"],
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("EHR connection failed", error=str(e), task_id=task.id)
            raise
    
    async def _bulk_export(self, task: TaskRequest) -> Dict[str, Any]:
        """Perform bulk data export from EHR."""
        try:
            connection_id = task.parameters.get("connection_id")
            export_request = BulkDataRequest(**task.parameters.get("export_request", {}))
            callback_url = task.parameters.get("callback_url")
            
            if connection_id not in self.access_tokens:
                raise ValueError(f"Connection not found: {connection_id}")
            
            self.audit_log_action(
                action="bulk_export",
                data_type="EHR Bulk Export",
                details={
                    "connection_id": connection_id,
                    "resource_types": export_request.resource_types,
                    "task_id": task.id
                }
            )
            
            vendor = connection_id.split("-")[0]
            config = self.connections[vendor]
            token_info = self.access_tokens[connection_id]
            
            # Prepare export request
            export_url = f"{config.base_url}/Patient/$export"
            
            # Build query parameters
            params = {
                "_type": ",".join(export_request.resource_types),
                "_outputFormat": export_request.export_format
            }
            
            if export_request.since:
                params["_since"] = export_request.since.isoformat()
            
            # Make bulk export request
            headers = {
                "Authorization": f"Bearer {token_info['access_token']}",
                "Accept": "application/fhir+json",
                "Prefer": "respond-async"
            }
            
            response = await self.http_client.get(export_url, params=params, headers=headers)
            
            if response.status_code == 202:  # Accepted - async processing
                content_location = response.headers.get("Content-Location")
                export_id = content_location.split("/")[-1] if content_location else str(datetime.utcnow().timestamp())
                
                return {
                    "export_id": export_id,
                    "status": "in_progress",
                    "status_url": content_location,
                    "estimated_completion": (datetime.utcnow() + timedelta(minutes=30)).isoformat(),
                    "callback_url": callback_url,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                raise Exception(f"Bulk export failed: {response.status_code} - {response.text}")
            
        except Exception as e:
            self.logger.error("Bulk export failed", error=str(e), task_id=task.id)
            raise
    
    async def _search_patients(self, task: TaskRequest) -> Dict[str, Any]:
        """Search for patients in EHR system."""
        try:
            connection_id = task.parameters.get("connection_id")
            search_criteria = task.parameters.get("search_criteria", {})
            limit = task.parameters.get("limit", 50)
            
            if connection_id not in self.access_tokens:
                raise ValueError(f"Connection not found: {connection_id}")
            
            self.audit_log_action(
                action="search_patients",
                data_type="EHR Patient Search",
                details={
                    "connection_id": connection_id,
                    "search_criteria": search_criteria,
                    "task_id": task.id
                }
            )
            
            vendor = connection_id.split("-")[0]
            config = self.connections[vendor]
            token_info = self.access_tokens[connection_id]
            
            # Build search URL
            search_url = f"{config.base_url}/Patient"
            
            # Prepare search parameters
            params = search_criteria.copy()
            params["_count"] = limit
            params["_format"] = "json"
            
            headers = {
                "Authorization": f"Bearer {token_info['access_token']}",
                "Accept": "application/fhir+json"
            }
            
            response = await self.http_client.get(search_url, params=params, headers=headers)
            
            if response.status_code == 200:
                bundle = response.json()
                patients = []
                
                for entry in bundle.get("entry", []):
                    patient = entry.get("resource", {})
                    # Ensure HIPAA compliance
                    if self.config.get("enforce_hipaa", True):
                        patient = self.ensure_hipaa_compliance(patient)
                    patients.append(patient)
                
                return {
                    "patients": patients,
                    "total": bundle.get("total", len(patients)),
                    "has_more": len(patients) == limit,
                    "search_criteria": search_criteria,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                raise Exception(f"Patient search failed: {response.status_code} - {response.text}")
            
        except Exception as e:
            self.logger.error("Patient search failed", error=str(e), task_id=task.id)
            raise
    
    async def _get_patient_data(self, task: TaskRequest) -> Dict[str, Any]:
        """Retrieve comprehensive patient data."""
        try:
            connection_id = task.parameters.get("connection_id")
            patient_id = task.parameters.get("patient_id")
            include_resources = task.parameters.get("include_resources", [])
            date_range = task.parameters.get("date_range", {})
            
            if connection_id not in self.access_tokens:
                raise ValueError(f"Connection not found: {connection_id}")
            
            self.audit_log_action(
                action="get_patient_data",
                data_type="EHR Patient Data",
                details={
                    "connection_id": connection_id,
                    "patient_id": patient_id,
                    "include_resources": include_resources,
                    "task_id": task.id
                }
            )
            
            vendor = connection_id.split("-")[0]
            config = self.connections[vendor]
            token_info = self.access_tokens[connection_id]
            
            headers = {
                "Authorization": f"Bearer {token_info['access_token']}",
                "Accept": "application/fhir+json"
            }
            
            # Get patient resource
            patient_url = f"{config.base_url}/Patient/{patient_id}"
            patient_response = await self.http_client.get(patient_url, headers=headers)
            
            if patient_response.status_code != 200:
                raise Exception(f"Patient not found: {patient_response.status_code}")
            
            patient = patient_response.json()
            
            # Ensure HIPAA compliance
            if self.config.get("enforce_hipaa", True):
                patient = self.ensure_hipaa_compliance(patient)
            
            # Get related resources
            resources = {}
            
            if not include_resources:
                include_resources = ["Observation", "Condition", "MedicationRequest", "Encounter"]
            
            for resource_type in include_resources:
                try:
                    resource_url = f"{config.base_url}/{resource_type}"
                    params = {
                        "patient": patient_id,
                        "_count": 100,
                        "_format": "json"
                    }
                    
                    # Add date range if specified
                    if date_range and resource_type in ["Observation", "Encounter"]:
                        if "start" in date_range:
                            params["date"] = f"ge{date_range['start']}"
                        if "end" in date_range:
                            date_param = params.get("date", "")
                            params["date"] = f"{date_param}&le{date_range['end']}" if date_param else f"le{date_range['end']}"
                    
                    resource_response = await self.http_client.get(resource_url, params=params, headers=headers)
                    
                    if resource_response.status_code == 200:
                        bundle = resource_response.json()
                        resource_list = []
                        
                        for entry in bundle.get("entry", []):
                            resource = entry.get("resource", {})
                            # Ensure HIPAA compliance
                            if self.config.get("enforce_hipaa", True):
                                resource = self.ensure_hipaa_compliance(resource)
                            resource_list.append(resource)
                        
                        resources[resource_type] = resource_list
                    
                except Exception as e:
                    self.logger.warning(f"Failed to retrieve {resource_type}", error=str(e))
                    resources[resource_type] = []
            
            # Generate summary
            summary = self._generate_patient_summary(patient, resources)
            
            return {
                "patient": patient,
                "resources": resources,
                "summary": summary,
                "retrieved_at": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Patient data retrieval failed", error=str(e), task_id=task.id)
            raise
    
    async def _sync_data(self, task: TaskRequest) -> Dict[str, Any]:
        """Synchronize data between EHR and local storage."""
        try:
            connection_id = task.parameters.get("connection_id")
            sync_config = task.parameters.get("sync_config", {})
            
            self.audit_log_action(
                action="sync_data",
                data_type="EHR Data Sync",
                details={
                    "connection_id": connection_id,
                    "sync_config": sync_config,
                    "task_id": task.id
                }
            )
            
            # Simplified sync implementation
            sync_results = {
                "patients_synced": 0,
                "resources_synced": 0,
                "errors": [],
                "started_at": datetime.utcnow().isoformat()
            }
            
            # This would implement actual synchronization logic
            # For now, return a placeholder result
            
            sync_results["completed_at"] = datetime.utcnow().isoformat()
            sync_results["status"] = "completed"
            
            return {
                "sync_results": sync_results,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Data sync failed", error=str(e), task_id=task.id)
            raise
    
    async def _check_export_status(self, task: TaskRequest) -> Dict[str, Any]:
        """Check the status of a bulk export operation."""
        try:
            connection_id = task.parameters.get("connection_id")
            export_id = task.parameters.get("export_id")
            status_url = task.parameters.get("status_url")
            
            if connection_id not in self.access_tokens:
                raise ValueError(f"Connection not found: {connection_id}")
            
            token_info = self.access_tokens[connection_id]
            
            headers = {
                "Authorization": f"Bearer {token_info['access_token']}",
                "Accept": "application/json"
            }
            
            response = await self.http_client.get(status_url, headers=headers)
            
            if response.status_code == 200:
                # Export completed
                export_manifest = response.json()
                return {
                    "status": "completed",
                    "export_id": export_id,
                    "files": export_manifest.get("output", []),
                    "completed_at": datetime.utcnow().isoformat(),
                    "agent_id": self.agent_id
                }
            elif response.status_code == 202:
                # Still in progress
                return {
                    "status": "in_progress",
                    "export_id": export_id,
                    "progress": response.headers.get("X-Progress", "unknown"),
                    "agent_id": self.agent_id
                }
            else:
                raise Exception(f"Export status check failed: {response.status_code}")
            
        except Exception as e:
            self.logger.error("Export status check failed", error=str(e), task_id=task.id)
            raise
    
    async def _obtain_access_token(self, config: EHRConnectionConfig) -> Dict[str, Any]:
        """Obtain OAuth 2.0 access token."""
        # Client credentials flow for backend services
        token_data = {
            "grant_type": "client_credentials",
            "scope": "system/*.read"
        }
        
        # Create basic auth header
        credentials = base64.b64encode(f"{config.client_id}:{config.client_secret}".encode()).decode()
        headers = {
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        response = await self.http_client.post(
            config.token_url,
            data=urlencode(token_data),
            headers=headers
        )
        
        if response.status_code == 200:
            token_response = response.json()
            expires_in = token_response.get("expires_in", 3600)
            
            return {
                "access_token": token_response["access_token"],
                "token_type": token_response.get("token_type", "Bearer"),
                "expires_at": (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat(),
                "scope": token_response.get("scope", "")
            }
        else:
            raise Exception(f"Token request failed: {response.status_code} - {response.text}")
    
    async def _test_connection(self, config: EHRConnectionConfig, access_token: str) -> Dict[str, Any]:
        """Test EHR connection and retrieve capabilities."""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/fhir+json"
        }
        
        # Test with capability statement
        capability_url = f"{config.base_url}/metadata"
        response = await self.http_client.get(capability_url, headers=headers)
        
        if response.status_code == 200:
            capability_statement = response.json()
            
            # Extract supported resources
            supported_resources = []
            if "rest" in capability_statement:
                for rest in capability_statement["rest"]:
                    for resource in rest.get("resource", []):
                        supported_resources.append(resource.get("type"))
            
            return {
                "status": "success",
                "capabilities": supported_resources,
                "fhir_version": capability_statement.get("fhirVersion", "unknown"),
                "implementation": capability_statement.get("implementation", {})
            }
        else:
            raise Exception(f"Connection test failed: {response.status_code}")
    
    def _generate_patient_summary(self, patient: Dict[str, Any], resources: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate a summary of patient data."""
        summary = {
            "patient_id": patient.get("id"),
            "demographics": {
                "name": self._extract_patient_name(patient),
                "gender": patient.get("gender"),
                "birth_date": patient.get("birthDate"),
                "age": self._calculate_age(patient.get("birthDate"))
            },
            "resource_counts": {
                resource_type: len(resource_list)
                for resource_type, resource_list in resources.items()
            },
            "recent_encounters": len([
                enc for enc in resources.get("Encounter", [])
                if self._is_recent(enc.get("period", {}).get("start"))
            ]),
            "active_medications": len([
                med for med in resources.get("MedicationRequest", [])
                if med.get("status") == "active"
            ]),
            "critical_results": len([
                obs for obs in resources.get("Observation", [])
                if self._is_critical_result(obs)
            ])
        }
        
        return summary
    
    def _extract_patient_name(self, patient: Dict[str, Any]) -> str:
        """Extract patient name from FHIR Patient resource."""
        names = patient.get("name", [])
        if names:
            name = names[0]
            given = " ".join(name.get("given", []))
            family = name.get("family", "")
            return f"{given} {family}".strip()
        return "Unknown"
    
    def _calculate_age(self, birth_date: str) -> Optional[int]:
        """Calculate age from birth date."""
        if not birth_date:
            return None
        
        try:
            birth = datetime.fromisoformat(birth_date.replace("Z", "+00:00"))
            today = datetime.utcnow()
            age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
            return age
        except:
            return None
    
    def _is_recent(self, date_str: str, days: int = 30) -> bool:
        """Check if date is within recent days."""
        if not date_str:
            return False
        
        try:
            date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            cutoff = datetime.utcnow() - timedelta(days=days)
            return date >= cutoff
        except:
            return False
    
    def _is_critical_result(self, observation: Dict[str, Any]) -> bool:
        """Check if observation result is critical."""
        interpretation = observation.get("interpretation", [])
        for interp in interpretation:
            for coding in interp.get("coding", []):
                if coding.get("code") in ["H", "L", "A", "AA"]:  # High, Low, Abnormal, Critical
                    return True
        return False