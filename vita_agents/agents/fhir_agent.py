"""
FHIR Parser Agent for handling FHIR resources, validation, and data quality checks.
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import structlog
from pydantic import BaseModel, Field
from fhirclient import client
from fhirclient.models import fhirreference, patient, observation, medication, encounter
import requests

from vita_agents.core.agent import HealthcareAgent, AgentCapability, TaskRequest, TaskResponse
from vita_agents.core.config import get_settings


logger = structlog.get_logger(__name__)


class FHIRValidationResult(BaseModel):
    """FHIR validation result."""
    
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    resource_type: Optional[str] = None
    version: Optional[str] = None


class FHIRQualityMetrics(BaseModel):
    """Data quality metrics for FHIR resources."""
    
    completeness_score: float  # 0-1
    consistency_score: float   # 0-1
    validity_score: float      # 0-1
    missing_required_fields: List[str] = Field(default_factory=list)
    inconsistent_data: List[str] = Field(default_factory=list)
    invalid_values: List[str] = Field(default_factory=list)


class FHIRAgent(HealthcareAgent):
    """
    FHIR Parser Agent for validating, parsing, and processing FHIR resources.
    
    Capabilities:
    - Parse and validate FHIR resources (Patient, Observation, Medication, etc.)
    - Support multiple FHIR versions (DSTU2, STU3, R4, R5)
    - Perform data quality checks
    - Convert between FHIR versions
    - Extract clinical insights from FHIR data
    """
    
    def __init__(
        self,
        agent_id: str = "fhir-agent",
        name: str = "FHIR Parser Agent",
        description: str = "Validates and processes FHIR resources with data quality checks",
        version: str = "1.0.0",
        fhir_version: str = "R4",
        fhir_server_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        # Define FHIR-specific capabilities
        capabilities = [
            AgentCapability(
                name="validate_fhir_resource",
                description="Validate FHIR resources against specification",
                input_schema={
                    "type": "object",
                    "properties": {
                        "resource": {"type": "object"},
                        "resource_type": {"type": "string"},
                        "version": {"type": "string", "enum": ["DSTU2", "STU3", "R4", "R5"]}
                    },
                    "required": ["resource"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "is_valid": {"type": "boolean"},
                        "errors": {"type": "array", "items": {"type": "string"}},
                        "warnings": {"type": "array", "items": {"type": "string"}}
                    }
                },
                supported_formats=["json", "xml"],
                requirements=["fhirclient", "requests"]
            ),
            AgentCapability(
                name="parse_fhir_bundle",
                description="Parse and extract resources from FHIR bundles",
                input_schema={
                    "type": "object",
                    "properties": {
                        "bundle": {"type": "object"},
                        "extract_types": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["bundle"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "resources": {"type": "object"},
                        "count": {"type": "integer"}
                    }
                }
            ),
            AgentCapability(
                name="quality_check",
                description="Perform data quality assessment on FHIR resources",
                input_schema={
                    "type": "object",
                    "properties": {
                        "resource": {"type": "object"},
                        "quality_rules": {"type": "array"}
                    },
                    "required": ["resource"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "quality_metrics": {"type": "object"},
                        "recommendations": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="extract_clinical_data",
                description="Extract clinical insights from FHIR resources",
                input_schema={
                    "type": "object",
                    "properties": {
                        "resources": {"type": "array"},
                        "extraction_type": {"type": "string"}
                    },
                    "required": ["resources"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "clinical_data": {"type": "object"},
                        "insights": {"type": "array"}
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
        
        # FHIR-specific configuration
        self.fhir_version = fhir_version
        self.fhir_server_url = fhir_server_url or get_settings().healthcare.fhir_server_url
        self.supported_versions = ["DSTU2", "STU3", "R4", "R5"]
        
        # Healthcare standards
        self.supported_standards = ["FHIR", "HL7"]
        self.data_formats = ["json", "xml"]
        self.compliance_features = ["HIPAA", "audit-logging", "data-validation"]
        
        # FHIR client
        self.fhir_client = None
        
        # Resource type mappings
        self.resource_types = {
            "Patient": patient.Patient,
            "Observation": observation.Observation,
            "Medication": medication.Medication,
            "Encounter": encounter.Encounter,
            # Add more as needed
        }
        
        # Register task handlers
        self.register_task_handler("validate_fhir_resource", self._validate_fhir_resource)
        self.register_task_handler("parse_fhir_bundle", self._parse_fhir_bundle)
        self.register_task_handler("quality_check", self._perform_quality_check)
        self.register_task_handler("extract_clinical_data", self._extract_clinical_data)
        self.register_task_handler("convert_fhir_version", self._convert_fhir_version)
        self.register_task_handler("search_fhir_resources", self._search_fhir_resources)
    
    async def _on_start(self) -> None:
        """Initialize FHIR client and connections."""
        self.logger.info("Starting FHIR agent", fhir_version=self.fhir_version)
        
        try:
            # Initialize FHIR client
            settings = {
                'app_id': 'vita-agents',
                'api_base': self.fhir_server_url
            }
            self.fhir_client = client.FHIRClient(settings=settings)
            
            self.logger.info("FHIR client initialized", server_url=self.fhir_server_url)
            
        except Exception as e:
            self.logger.error("Failed to initialize FHIR client", error=str(e))
    
    async def _on_stop(self) -> None:
        """Clean up FHIR connections."""
        self.logger.info("Stopping FHIR agent")
        self.fhir_client = None
    
    async def _validate_fhir_resource(self, task: TaskRequest) -> Dict[str, Any]:
        """Validate a FHIR resource against the specification."""
        try:
            resource_data = task.parameters.get("resource")
            resource_type = task.parameters.get("resource_type")
            version = task.parameters.get("version", self.fhir_version)
            
            if not resource_data:
                raise ValueError("Resource data is required")
            
            # Log the validation action
            self.audit_log_action(
                action="validate_fhir_resource",
                data_type="FHIR",
                details={
                    "resource_type": resource_type,
                    "version": version,
                    "task_id": task.id
                }
            )
            
            # Basic structure validation
            validation_result = FHIRValidationResult(is_valid=True)
            
            # Check if resourceType is present
            if "resourceType" not in resource_data:
                validation_result.is_valid = False
                validation_result.errors.append("Missing required field: resourceType")
            else:
                validation_result.resource_type = resource_data["resourceType"]
            
            # Check if id is present (warning if missing)
            if "id" not in resource_data:
                validation_result.warnings.append("Resource ID is missing - recommended for tracking")
            
            # Validate against FHIR specification using fhirclient
            if validation_result.resource_type in self.resource_types:
                try:
                    resource_class = self.resource_types[validation_result.resource_type]
                    fhir_resource = resource_class(resource_data)
                    
                    # Additional validation can be added here
                    if not fhir_resource.id and validation_result.resource_type == "Patient":
                        validation_result.warnings.append("Patient resource should have an identifier")
                    
                except Exception as e:
                    validation_result.is_valid = False
                    validation_result.errors.append(f"FHIR validation error: {str(e)}")
            
            # Version-specific validation
            validation_result.version = version
            if version not in self.supported_versions:
                validation_result.warnings.append(f"FHIR version {version} may not be fully supported")
            
            # Perform HIPAA compliance check
            if self.config.get("enforce_hipaa", True):
                resource_data = self.ensure_hipaa_compliance(resource_data)
            
            return {
                "validation_result": validation_result.dict(),
                "processed_resource": resource_data,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("FHIR validation failed", error=str(e), task_id=task.id)
            raise
    
    async def _parse_fhir_bundle(self, task: TaskRequest) -> Dict[str, Any]:
        """Parse and extract resources from a FHIR bundle."""
        try:
            bundle_data = task.parameters.get("bundle")
            extract_types = task.parameters.get("extract_types", [])
            
            if not bundle_data:
                raise ValueError("Bundle data is required")
            
            self.audit_log_action(
                action="parse_fhir_bundle",
                data_type="FHIR Bundle",
                details={
                    "extract_types": extract_types,
                    "task_id": task.id
                }
            )
            
            resources = {}
            resource_count = 0
            
            # Check if it's a valid bundle
            if bundle_data.get("resourceType") != "Bundle":
                raise ValueError("Resource is not a FHIR Bundle")
            
            # Extract entries
            entries = bundle_data.get("entry", [])
            
            for entry in entries:
                resource = entry.get("resource", {})
                resource_type = resource.get("resourceType")
                
                if resource_type:
                    # Filter by requested types if specified
                    if not extract_types or resource_type in extract_types:
                        if resource_type not in resources:
                            resources[resource_type] = []
                        
                        # Ensure HIPAA compliance
                        if self.config.get("enforce_hipaa", True):
                            resource = self.ensure_hipaa_compliance(resource)
                        
                        resources[resource_type].append(resource)
                        resource_count += 1
            
            return {
                "resources": resources,
                "count": resource_count,
                "bundle_type": bundle_data.get("type"),
                "total_entries": len(entries),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Bundle parsing failed", error=str(e), task_id=task.id)
            raise
    
    async def _perform_quality_check(self, task: TaskRequest) -> Dict[str, Any]:
        """Perform comprehensive data quality assessment."""
        try:
            resource_data = task.parameters.get("resource")
            quality_rules = task.parameters.get("quality_rules", [])
            
            if not resource_data:
                raise ValueError("Resource data is required")
            
            self.audit_log_action(
                action="quality_check",
                data_type="FHIR",
                details={
                    "resource_type": resource_data.get("resourceType"),
                    "rules_count": len(quality_rules),
                    "task_id": task.id
                }
            )
            
            # Initialize quality metrics
            metrics = FHIRQualityMetrics(
                completeness_score=0.0,
                consistency_score=0.0,
                validity_score=0.0
            )
            
            resource_type = resource_data.get("resourceType")
            
            # Completeness check
            required_fields = self._get_required_fields(resource_type)
            present_fields = 0
            
            for field in required_fields:
                if self._has_nested_field(resource_data, field):
                    present_fields += 1
                else:
                    metrics.missing_required_fields.append(field)
            
            metrics.completeness_score = present_fields / len(required_fields) if required_fields else 1.0
            
            # Validity check
            valid_count = 0
            total_checks = 0
            
            # Check data types and formats
            if resource_type == "Patient":
                total_checks += 3
                
                # Check birthDate format
                birth_date = resource_data.get("birthDate")
                if birth_date and self._is_valid_date(birth_date):
                    valid_count += 1
                elif birth_date:
                    metrics.invalid_values.append("birthDate: Invalid date format")
                
                # Check gender value
                gender = resource_data.get("gender")
                if gender and gender in ["male", "female", "other", "unknown"]:
                    valid_count += 1
                elif gender:
                    metrics.invalid_values.append("gender: Invalid value")
                
                # Check identifier system
                identifiers = resource_data.get("identifier", [])
                if identifiers and all(id.get("system") for id in identifiers):
                    valid_count += 1
                elif identifiers:
                    metrics.invalid_values.append("identifier: Missing system")
            
            metrics.validity_score = valid_count / total_checks if total_checks > 0 else 1.0
            
            # Consistency check (simplified)
            metrics.consistency_score = 0.9  # Placeholder for more complex consistency rules
            
            # Generate recommendations
            recommendations = []
            
            if metrics.completeness_score < 0.8:
                recommendations.append("Consider adding missing required fields to improve completeness")
            
            if metrics.validity_score < 0.9:
                recommendations.append("Review and correct invalid field values")
            
            if metrics.missing_required_fields:
                recommendations.append(f"Add missing required fields: {', '.join(metrics.missing_required_fields)}")
            
            return {
                "quality_metrics": metrics.dict(),
                "recommendations": recommendations,
                "overall_score": (metrics.completeness_score + metrics.validity_score + metrics.consistency_score) / 3,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Quality check failed", error=str(e), task_id=task.id)
            raise
    
    async def _extract_clinical_data(self, task: TaskRequest) -> Dict[str, Any]:
        """Extract clinical insights from FHIR resources."""
        try:
            resources = task.parameters.get("resources", [])
            extraction_type = task.parameters.get("extraction_type", "general")
            
            if not resources:
                raise ValueError("Resources list is required")
            
            self.audit_log_action(
                action="extract_clinical_data",
                data_type="FHIR",
                details={
                    "resource_count": len(resources),
                    "extraction_type": extraction_type,
                    "task_id": task.id
                }
            )
            
            clinical_data = {
                "patient_summary": {},
                "observations": [],
                "medications": [],
                "encounters": [],
                "conditions": []
            }
            
            insights = []
            
            # Process each resource
            for resource in resources:
                resource_type = resource.get("resourceType")
                
                if resource_type == "Patient":
                    clinical_data["patient_summary"] = self._extract_patient_summary(resource)
                
                elif resource_type == "Observation":
                    obs_data = self._extract_observation_data(resource)
                    clinical_data["observations"].append(obs_data)
                    
                    # Generate insights for abnormal values
                    if obs_data.get("abnormal_flag"):
                        insights.append(f"Abnormal {obs_data.get('code', 'observation')} detected")
                
                elif resource_type == "Medication":
                    med_data = self._extract_medication_data(resource)
                    clinical_data["medications"].append(med_data)
                
                elif resource_type == "Encounter":
                    enc_data = self._extract_encounter_data(resource)
                    clinical_data["encounters"].append(enc_data)
            
            # Generate high-level insights
            if len(clinical_data["observations"]) > 10:
                insights.append("High number of observations - consider care coordination")
            
            if len(clinical_data["medications"]) > 5:
                insights.append("Multiple medications - review for potential interactions")
            
            return {
                "clinical_data": clinical_data,
                "insights": insights,
                "summary_stats": {
                    "total_resources": len(resources),
                    "observations_count": len(clinical_data["observations"]),
                    "medications_count": len(clinical_data["medications"]),
                    "encounters_count": len(clinical_data["encounters"])
                },
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Clinical data extraction failed", error=str(e), task_id=task.id)
            raise
    
    async def _convert_fhir_version(self, task: TaskRequest) -> Dict[str, Any]:
        """Convert FHIR resource between versions."""
        try:
            resource_data = task.parameters.get("resource")
            target_version = task.parameters.get("target_version")
            source_version = task.parameters.get("source_version", "R4")
            
            if not resource_data or not target_version:
                raise ValueError("Resource data and target version are required")
            
            self.audit_log_action(
                action="convert_fhir_version",
                data_type="FHIR",
                details={
                    "source_version": source_version,
                    "target_version": target_version,
                    "resource_type": resource_data.get("resourceType"),
                    "task_id": task.id
                }
            )
            
            # Simplified version conversion (in practice, this would be more complex)
            converted_resource = resource_data.copy()
            conversion_notes = []
            
            if source_version == "STU3" and target_version == "R4":
                # Example conversion logic
                if "meta" in converted_resource:
                    if "versionId" in converted_resource["meta"]:
                        conversion_notes.append("versionId handling updated for R4")
            
            # Add meta information about conversion
            if "meta" not in converted_resource:
                converted_resource["meta"] = {}
            
            converted_resource["meta"]["profile"] = [f"http://hl7.org/fhir/{target_version}/StructureDefinition/{resource_data.get('resourceType', 'Resource')}"]
            
            return {
                "converted_resource": converted_resource,
                "source_version": source_version,
                "target_version": target_version,
                "conversion_notes": conversion_notes,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("FHIR version conversion failed", error=str(e), task_id=task.id)
            raise
    
    async def _search_fhir_resources(self, task: TaskRequest) -> Dict[str, Any]:
        """Search for FHIR resources on the server."""
        try:
            resource_type = task.parameters.get("resource_type")
            search_params = task.parameters.get("search_params", {})
            
            if not resource_type:
                raise ValueError("Resource type is required")
            
            if not self.fhir_client:
                raise ValueError("FHIR client not initialized")
            
            self.audit_log_action(
                action="search_fhir_resources",
                data_type="FHIR",
                details={
                    "resource_type": resource_type,
                    "search_params": search_params,
                    "task_id": task.id
                }
            )
            
            # Perform search using fhirclient
            search_result = self.fhir_client.server.request_json(
                f"{resource_type}",
                "GET",
                params=search_params
            )
            
            resources = []
            if search_result and search_result.get("entry"):
                for entry in search_result["entry"]:
                    resource = entry.get("resource")
                    if resource:
                        # Ensure HIPAA compliance
                        if self.config.get("enforce_hipaa", True):
                            resource = self.ensure_hipaa_compliance(resource)
                        resources.append(resource)
            
            return {
                "resources": resources,
                "total": search_result.get("total", 0),
                "resource_type": resource_type,
                "search_params": search_params,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("FHIR search failed", error=str(e), task_id=task.id)
            raise
    
    def _get_required_fields(self, resource_type: str) -> List[str]:
        """Get required fields for a FHIR resource type."""
        required_fields_map = {
            "Patient": ["resourceType"],
            "Observation": ["resourceType", "status", "code"],
            "Medication": ["resourceType", "code"],
            "Encounter": ["resourceType", "status", "class"],
        }
        return required_fields_map.get(resource_type, ["resourceType"])
    
    def _has_nested_field(self, data: Dict[str, Any], field_path: str) -> bool:
        """Check if a nested field exists in the data."""
        fields = field_path.split(".")
        current = data
        
        for field in fields:
            if isinstance(current, dict) and field in current:
                current = current[field]
            else:
                return False
        
        return current is not None
    
    def _is_valid_date(self, date_str: str) -> bool:
        """Validate date string format."""
        try:
            # Try common FHIR date formats
            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"]:
                try:
                    datetime.strptime(date_str, fmt)
                    return True
                except ValueError:
                    continue
            return False
        except:
            return False
    
    def _extract_patient_summary(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        """Extract patient summary data."""
        return {
            "id": patient.get("id"),
            "name": self._extract_human_name(patient.get("name", [])),
            "gender": patient.get("gender"),
            "birth_date": patient.get("birthDate"),
            "identifiers": patient.get("identifier", [])
        }
    
    def _extract_human_name(self, names: List[Dict[str, Any]]) -> str:
        """Extract human name from FHIR name array."""
        if not names:
            return "Unknown"
        
        name = names[0]  # Use first name
        given = name.get("given", [])
        family = name.get("family", "")
        
        return f"{' '.join(given)} {family}".strip()
    
    def _extract_observation_data(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract observation data."""
        value_quantity = observation.get("valueQuantity", {})
        
        return {
            "id": observation.get("id"),
            "code": observation.get("code", {}).get("coding", [{}])[0].get("display", "Unknown"),
            "value": value_quantity.get("value"),
            "unit": value_quantity.get("unit"),
            "status": observation.get("status"),
            "abnormal_flag": observation.get("interpretation", {}).get("coding", [{}])[0].get("code") == "A"
        }
    
    def _extract_medication_data(self, medication: Dict[str, Any]) -> Dict[str, Any]:
        """Extract medication data."""
        return {
            "id": medication.get("id"),
            "code": medication.get("code", {}).get("coding", [{}])[0].get("display", "Unknown"),
            "form": medication.get("form", {}).get("coding", [{}])[0].get("display"),
            "status": medication.get("status")
        }
    
    def _extract_encounter_data(self, encounter: Dict[str, Any]) -> Dict[str, Any]:
        """Extract encounter data."""
        return {
            "id": encounter.get("id"),
            "status": encounter.get("status"),
            "class": encounter.get("class", {}).get("display"),
            "period": encounter.get("period", {}),
            "type": encounter.get("type", [{}])[0].get("coding", [{}])[0].get("display")
        }