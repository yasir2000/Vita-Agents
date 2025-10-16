"""
HL7 Translation Agent for converting between HL7 v2.x messages and FHIR resources.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import structlog
from pydantic import BaseModel, Field
import hl7apy
from hl7apy.core import Message, Segment, Field
from hl7apy.parser import parse_message
from hl7apy.validation import VALIDATION_LEVEL

from vita_agents.core.agent import HealthcareAgent, AgentCapability, TaskRequest
from vita_agents.core.config import get_settings


logger = structlog.get_logger(__name__)


class HL7ValidationResult(BaseModel):
    """HL7 validation result."""
    
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    message_type: Optional[str] = None
    version: Optional[str] = None


class TerminologyMapping(BaseModel):
    """Terminology mapping between HL7 and FHIR."""
    
    hl7_code: str
    hl7_system: str
    fhir_code: str
    fhir_system: str
    display: str


class ConversionResult(BaseModel):
    """Result of HL7 to FHIR conversion."""
    
    success: bool
    fhir_resources: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []
    conversion_notes: List[str] = []


class HL7Agent(HealthcareAgent):
    """
    HL7 Translation Agent for converting between HL7 v2.x messages and FHIR resources.
    
    Capabilities:
    - Parse and validate HL7 v2.x messages
    - Convert HL7 messages to FHIR resources
    - Convert FHIR resources to HL7 messages
    - Handle CDA (Clinical Document Architecture) processing
    - Manage terminology mapping (SNOMED CT, ICD-10, LOINC)
    - Support multiple HL7 versions (2.3, 2.4, 2.5, 2.6, 2.8)
    """
    
    def __init__(
        self,
        agent_id: str = "hl7-agent",
        name: str = "HL7 Translation Agent",
        description: str = "Converts between HL7 v2.x messages and FHIR resources",
        version: str = "1.0.0",
        hl7_version: str = "2.8",
        config: Optional[Dict[str, Any]] = None,
    ):
        # Define HL7-specific capabilities
        capabilities = [
            AgentCapability(
                name="validate_hl7_message",
                description="Validate HL7 v2.x messages against specification",
                input_schema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "version": {"type": "string", "enum": ["2.3", "2.4", "2.5", "2.6", "2.8"]},
                        "strict_validation": {"type": "boolean", "default": True}
                    },
                    "required": ["message"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "is_valid": {"type": "boolean"},
                        "errors": {"type": "array", "items": {"type": "string"}},
                        "warnings": {"type": "array", "items": {"type": "string"}},
                        "message_type": {"type": "string"}
                    }
                },
                supported_formats=["hl7v2", "pipe-delimited"],
                requirements=["hl7apy"]
            ),
            AgentCapability(
                name="hl7_to_fhir",
                description="Convert HL7 v2.x messages to FHIR resources",
                input_schema={
                    "type": "object",
                    "properties": {
                        "hl7_message": {"type": "string"},
                        "target_resources": {"type": "array", "items": {"type": "string"}},
                        "terminology_mappings": {"type": "object"}
                    },
                    "required": ["hl7_message"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "fhir_resources": {"type": "array"},
                        "conversion_notes": {"type": "array"},
                        "success": {"type": "boolean"}
                    }
                }
            ),
            AgentCapability(
                name="fhir_to_hl7",
                description="Convert FHIR resources to HL7 v2.x messages",
                input_schema={
                    "type": "object",
                    "properties": {
                        "fhir_resources": {"type": "array"},
                        "message_type": {"type": "string"},
                        "hl7_version": {"type": "string"}
                    },
                    "required": ["fhir_resources", "message_type"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "hl7_message": {"type": "string"},
                        "conversion_notes": {"type": "array"},
                        "success": {"type": "boolean"}
                    }
                }
            ),
            AgentCapability(
                name="parse_cda_document",
                description="Parse CDA (Clinical Document Architecture) documents",
                input_schema={
                    "type": "object",
                    "properties": {
                        "cda_document": {"type": "string"},
                        "extract_sections": {"type": "array"}
                    },
                    "required": ["cda_document"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "structured_data": {"type": "object"},
                        "fhir_resources": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="map_terminology",
                description="Map terminology codes between systems",
                input_schema={
                    "type": "object",
                    "properties": {
                        "source_code": {"type": "string"},
                        "source_system": {"type": "string"},
                        "target_system": {"type": "string"}
                    },
                    "required": ["source_code", "source_system", "target_system"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "mapped_code": {"type": "string"},
                        "display": {"type": "string"},
                        "confidence": {"type": "number"}
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
        
        # HL7-specific configuration
        self.hl7_version = hl7_version
        self.supported_versions = ["2.3", "2.4", "2.5", "2.6", "2.8"]
        
        # Healthcare standards
        self.supported_standards = ["HL7", "FHIR", "CDA"]
        self.data_formats = ["hl7v2", "xml", "json"]
        self.compliance_features = ["HIPAA", "audit-logging", "terminology-mapping"]
        
        # Terminology mappings
        self.terminology_maps = self._load_terminology_mappings()
        
        # HL7 message type mappings to FHIR resources
        self.message_type_mappings = {
            "ADT": ["Patient", "Encounter"],
            "ORM": ["MedicationRequest", "ServiceRequest"],
            "ORU": ["Observation", "DiagnosticReport"],
            "SIU": ["Appointment"],
            "MDM": ["DocumentReference"],
            "VXU": ["Immunization"],
            "DFT": ["ChargeItem", "Account"]
        }
        
        # Register task handlers
        self.register_task_handler("validate_hl7_message", self._validate_hl7_message)
        self.register_task_handler("hl7_to_fhir", self._hl7_to_fhir)
        self.register_task_handler("fhir_to_hl7", self._fhir_to_hl7)
        self.register_task_handler("parse_cda_document", self._parse_cda_document)
        self.register_task_handler("map_terminology", self._map_terminology)
        self.register_task_handler("extract_segments", self._extract_segments)
    
    async def _on_start(self) -> None:
        """Initialize HL7 processing capabilities."""
        self.logger.info("Starting HL7 agent", hl7_version=self.hl7_version)
        
        # Set HL7apy validation level
        hl7apy.set_default_validation_level(VALIDATION_LEVEL.STRICT)
        
        self.logger.info("HL7 agent initialized with hl7apy")
    
    async def _on_stop(self) -> None:
        """Clean up HL7 processing."""
        self.logger.info("Stopping HL7 agent")
    
    async def _validate_hl7_message(self, task: TaskRequest) -> Dict[str, Any]:
        """Validate an HL7 v2.x message."""
        try:
            message_text = task.parameters.get("message")
            version = task.parameters.get("version", self.hl7_version)
            strict_validation = task.parameters.get("strict_validation", True)
            
            if not message_text:
                raise ValueError("HL7 message text is required")
            
            self.audit_log_action(
                action="validate_hl7_message",
                data_type="HL7v2",
                details={
                    "version": version,
                    "strict_validation": strict_validation,
                    "task_id": task.id
                }
            )
            
            validation_result = HL7ValidationResult(is_valid=True)
            
            try:
                # Parse the message using hl7apy
                parsed_message = parse_message(message_text, version=version)
                
                # Extract message type
                msh_segment = parsed_message.msh
                message_type = msh_segment.msh_9.msh_9_1.value
                validation_result.message_type = message_type
                validation_result.version = version
                
                # Basic validation checks
                if not msh_segment.msh_3.value:  # Sending Application
                    validation_result.warnings.append("MSH.3 (Sending Application) is empty")
                
                if not msh_segment.msh_4.value:  # Sending Facility
                    validation_result.warnings.append("MSH.4 (Sending Facility) is empty")
                
                # Check required segments based on message type
                required_segments = self._get_required_segments(message_type)
                for segment_name in required_segments:
                    if not hasattr(parsed_message, segment_name.lower()):
                        validation_result.errors.append(f"Required segment {segment_name} is missing")
                        validation_result.is_valid = False
                
                self.logger.info(
                    "HL7 message validated",
                    message_type=message_type,
                    version=version,
                    is_valid=validation_result.is_valid
                )
                
            except Exception as e:
                validation_result.is_valid = False
                validation_result.errors.append(f"Parsing error: {str(e)}")
            
            return {
                "validation_result": validation_result.dict(),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("HL7 validation failed", error=str(e), task_id=task.id)
            raise
    
    async def _hl7_to_fhir(self, task: TaskRequest) -> Dict[str, Any]:
        """Convert HL7 v2.x message to FHIR resources."""
        try:
            hl7_message = task.parameters.get("hl7_message")
            target_resources = task.parameters.get("target_resources", [])
            terminology_mappings = task.parameters.get("terminology_mappings", {})
            
            if not hl7_message:
                raise ValueError("HL7 message is required")
            
            self.audit_log_action(
                action="hl7_to_fhir",
                data_type="HL7v2 to FHIR",
                details={
                    "target_resources": target_resources,
                    "task_id": task.id
                }
            )
            
            conversion_result = ConversionResult(success=True)
            
            try:
                # Parse HL7 message
                parsed_message = parse_message(hl7_message)
                msh_segment = parsed_message.msh
                message_type = msh_segment.msh_9.msh_9_1.value
                
                self.logger.info("Converting HL7 to FHIR", message_type=message_type)
                
                # Convert based on message type
                if message_type.startswith("ADT"):
                    fhir_resources = await self._convert_adt_to_fhir(parsed_message)
                elif message_type.startswith("ORU"):
                    fhir_resources = await self._convert_oru_to_fhir(parsed_message)
                elif message_type.startswith("ORM"):
                    fhir_resources = await self._convert_orm_to_fhir(parsed_message)
                else:
                    conversion_result.warnings.append(f"Message type {message_type} conversion not fully implemented")
                    fhir_resources = await self._convert_generic_to_fhir(parsed_message)
                
                # Filter by target resources if specified
                if target_resources:
                    fhir_resources = [
                        resource for resource in fhir_resources
                        if resource.get("resourceType") in target_resources
                    ]
                
                # Apply terminology mappings
                for resource in fhir_resources:
                    self._apply_terminology_mappings(resource, terminology_mappings)
                
                conversion_result.fhir_resources = fhir_resources
                conversion_result.conversion_notes.append(f"Converted {message_type} message to {len(fhir_resources)} FHIR resources")
                
            except Exception as e:
                conversion_result.success = False
                conversion_result.errors.append(f"Conversion error: {str(e)}")
            
            return {
                "conversion_result": conversion_result.dict(),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("HL7 to FHIR conversion failed", error=str(e), task_id=task.id)
            raise
    
    async def _fhir_to_hl7(self, task: TaskRequest) -> Dict[str, Any]:
        """Convert FHIR resources to HL7 v2.x message."""
        try:
            fhir_resources = task.parameters.get("fhir_resources", [])
            message_type = task.parameters.get("message_type")
            hl7_version = task.parameters.get("hl7_version", self.hl7_version)
            
            if not fhir_resources or not message_type:
                raise ValueError("FHIR resources and message type are required")
            
            self.audit_log_action(
                action="fhir_to_hl7",
                data_type="FHIR to HL7v2",
                details={
                    "message_type": message_type,
                    "resource_count": len(fhir_resources),
                    "task_id": task.id
                }
            )
            
            # Create HL7 message structure
            hl7_message = await self._create_hl7_message(fhir_resources, message_type, hl7_version)
            
            return {
                "hl7_message": hl7_message,
                "message_type": message_type,
                "resource_count": len(fhir_resources),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("FHIR to HL7 conversion failed", error=str(e), task_id=task.id)
            raise
    
    async def _parse_cda_document(self, task: TaskRequest) -> Dict[str, Any]:
        """Parse CDA (Clinical Document Architecture) document."""
        try:
            cda_document = task.parameters.get("cda_document")
            extract_sections = task.parameters.get("extract_sections", [])
            
            if not cda_document:
                raise ValueError("CDA document is required")
            
            self.audit_log_action(
                action="parse_cda_document",
                data_type="CDA",
                details={
                    "extract_sections": extract_sections,
                    "task_id": task.id
                }
            )
            
            # Parse CDA document (simplified implementation)
            structured_data = {
                "header": {},
                "sections": {},
                "clinical_data": {}
            }
            
            fhir_resources = []
            
            # Extract header information
            # This is a simplified implementation - real CDA parsing would be more complex
            structured_data["header"] = {
                "document_id": "extracted_from_cda",
                "creation_time": datetime.utcnow().isoformat(),
                "patient_info": {}
            }
            
            # Convert to FHIR DocumentReference
            document_reference = {
                "resourceType": "DocumentReference",
                "id": "cda-document",
                "status": "current",
                "type": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": "34133-9",
                        "display": "Summarization of Episode Note"
                    }]
                },
                "content": [{
                    "attachment": {
                        "contentType": "application/xml",
                        "data": cda_document  # Base64 encoded in real implementation
                    }
                }]
            }
            
            fhir_resources.append(document_reference)
            
            return {
                "structured_data": structured_data,
                "fhir_resources": fhir_resources,
                "sections_extracted": len(extract_sections),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("CDA parsing failed", error=str(e), task_id=task.id)
            raise
    
    async def _map_terminology(self, task: TaskRequest) -> Dict[str, Any]:
        """Map terminology codes between systems."""
        try:
            source_code = task.parameters.get("source_code")
            source_system = task.parameters.get("source_system")
            target_system = task.parameters.get("target_system")
            
            if not all([source_code, source_system, target_system]):
                raise ValueError("Source code, source system, and target system are required")
            
            self.audit_log_action(
                action="map_terminology",
                data_type="Terminology",
                details={
                    "source_system": source_system,
                    "target_system": target_system,
                    "source_code": source_code,
                    "task_id": task.id
                }
            )
            
            # Look up mapping in terminology maps
            mapping_key = f"{source_system}:{source_code}"
            mapping = self.terminology_maps.get(mapping_key)
            
            if mapping and mapping.fhir_system == target_system:
                return {
                    "mapped_code": mapping.fhir_code,
                    "display": mapping.display,
                    "confidence": 1.0,
                    "mapping_source": "predefined",
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # If no direct mapping found, return approximate mapping
            return {
                "mapped_code": source_code,  # Fallback to original code
                "display": f"Unmapped code from {source_system}",
                "confidence": 0.1,
                "mapping_source": "fallback",
                "warning": f"No mapping found from {source_system} to {target_system}",
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Terminology mapping failed", error=str(e), task_id=task.id)
            raise
    
    async def _extract_segments(self, task: TaskRequest) -> Dict[str, Any]:
        """Extract specific segments from HL7 message."""
        try:
            hl7_message = task.parameters.get("hl7_message")
            segment_types = task.parameters.get("segment_types", [])
            
            if not hl7_message:
                raise ValueError("HL7 message is required")
            
            self.audit_log_action(
                action="extract_segments",
                data_type="HL7v2",
                details={
                    "segment_types": segment_types,
                    "task_id": task.id
                }
            )
            
            parsed_message = parse_message(hl7_message)
            extracted_segments = {}
            
            # Extract all segments if none specified
            if not segment_types:
                segment_types = [segment.name for segment in parsed_message.children]
            
            for segment_type in segment_types:
                segments = []
                for segment in parsed_message.children:
                    if segment.name.upper() == segment_type.upper():
                        segment_data = self._segment_to_dict(segment)
                        segments.append(segment_data)
                
                if segments:
                    extracted_segments[segment_type] = segments
            
            return {
                "extracted_segments": extracted_segments,
                "total_segments": len(extracted_segments),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Segment extraction failed", error=str(e), task_id=task.id)
            raise
    
    def _load_terminology_mappings(self) -> Dict[str, TerminologyMapping]:
        """Load predefined terminology mappings."""
        # This would typically load from a database or configuration file
        mappings = {
            # Example mappings
            "HL70001:M": TerminologyMapping(
                hl7_code="M",
                hl7_system="HL70001",
                fhir_code="male",
                fhir_system="http://hl7.org/fhir/administrative-gender",
                display="Male"
            ),
            "HL70001:F": TerminologyMapping(
                hl7_code="F",
                hl7_system="HL70001",
                fhir_code="female",
                fhir_system="http://hl7.org/fhir/administrative-gender",
                display="Female"
            ),
        }
        return mappings
    
    def _get_required_segments(self, message_type: str) -> List[str]:
        """Get required segments for a message type."""
        required_segments_map = {
            "ADT^A01": ["MSH", "EVN", "PID"],
            "ADT^A08": ["MSH", "EVN", "PID"],
            "ORU^R01": ["MSH", "PID", "OBR", "OBX"],
            "ORM^O01": ["MSH", "PID", "ORC"],
        }
        return required_segments_map.get(message_type, ["MSH"])
    
    async def _convert_adt_to_fhir(self, parsed_message: Message) -> List[Dict[str, Any]]:
        """Convert ADT message to FHIR resources."""
        resources = []
        
        # Extract patient information from PID segment
        if hasattr(parsed_message, 'pid'):
            patient = self._create_patient_from_pid(parsed_message.pid)
            resources.append(patient)
        
        # Extract encounter information from PV1 segment
        if hasattr(parsed_message, 'pv1'):
            encounter = self._create_encounter_from_pv1(parsed_message.pv1)
            resources.append(encounter)
        
        return resources
    
    async def _convert_oru_to_fhir(self, parsed_message: Message) -> List[Dict[str, Any]]:
        """Convert ORU message to FHIR resources."""
        resources = []
        
        # Extract patient information
        if hasattr(parsed_message, 'pid'):
            patient = self._create_patient_from_pid(parsed_message.pid)
            resources.append(patient)
        
        # Extract observations from OBX segments
        for segment in parsed_message.children:
            if segment.name == 'OBX':
                observation = self._create_observation_from_obx(segment)
                resources.append(observation)
        
        return resources
    
    async def _convert_orm_to_fhir(self, parsed_message: Message) -> List[Dict[str, Any]]:
        """Convert ORM message to FHIR resources."""
        resources = []
        
        # Extract patient information
        if hasattr(parsed_message, 'pid'):
            patient = self._create_patient_from_pid(parsed_message.pid)
            resources.append(patient)
        
        # Extract service request from ORC/OBR segments
        if hasattr(parsed_message, 'orc'):
            service_request = self._create_service_request_from_orc(parsed_message.orc)
            resources.append(service_request)
        
        return resources
    
    async def _convert_generic_to_fhir(self, parsed_message: Message) -> List[Dict[str, Any]]:
        """Generic conversion for unsupported message types."""
        resources = []
        
        # Always try to extract patient if PID segment exists
        if hasattr(parsed_message, 'pid'):
            patient = self._create_patient_from_pid(parsed_message.pid)
            resources.append(patient)
        
        return resources
    
    def _create_patient_from_pid(self, pid_segment: Segment) -> Dict[str, Any]:
        """Create FHIR Patient resource from PID segment."""
        patient = {
            "resourceType": "Patient",
            "id": pid_segment.pid_3.pid_3_1.value if pid_segment.pid_3.pid_3_1.value else "unknown",
            "identifier": [{
                "value": pid_segment.pid_3.pid_3_1.value,
                "system": pid_segment.pid_3.pid_3_4.value if pid_segment.pid_3.pid_3_4.value else "unknown"
            }],
            "name": [{
                "family": pid_segment.pid_5.pid_5_1.value if pid_segment.pid_5.pid_5_1.value else "Unknown",
                "given": [pid_segment.pid_5.pid_5_2.value] if pid_segment.pid_5.pid_5_2.value else []
            }],
            "gender": self._map_gender(pid_segment.pid_8.value if pid_segment.pid_8.value else "unknown"),
            "birthDate": self._format_hl7_date(pid_segment.pid_7.value) if pid_segment.pid_7.value else None
        }
        
        # Remove None values
        return {k: v for k, v in patient.items() if v is not None}
    
    def _create_encounter_from_pv1(self, pv1_segment: Segment) -> Dict[str, Any]:
        """Create FHIR Encounter resource from PV1 segment."""
        encounter = {
            "resourceType": "Encounter",
            "id": f"encounter-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "status": "finished",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": self._map_patient_class(pv1_segment.pv1_2.value if pv1_segment.pv1_2.value else "AMB"),
                "display": pv1_segment.pv1_2.value if pv1_segment.pv1_2.value else "Ambulatory"
            }
        }
        
        return encounter
    
    def _create_observation_from_obx(self, obx_segment: Segment) -> Dict[str, Any]:
        """Create FHIR Observation resource from OBX segment."""
        observation = {
            "resourceType": "Observation",
            "id": f"obs-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "status": "final",
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": obx_segment.obx_3.obx_3_1.value if obx_segment.obx_3.obx_3_1.value else "unknown",
                    "display": obx_segment.obx_3.obx_3_2.value if obx_segment.obx_3.obx_3_2.value else "Unknown"
                }]
            },
            "valueString": obx_segment.obx_5.value if obx_segment.obx_5.value else "No value"
        }
        
        return observation
    
    def _create_service_request_from_orc(self, orc_segment: Segment) -> Dict[str, Any]:
        """Create FHIR ServiceRequest resource from ORC segment."""
        service_request = {
            "resourceType": "ServiceRequest",
            "id": f"req-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "status": self._map_order_status(orc_segment.orc_1.value if orc_segment.orc_1.value else "unknown"),
            "intent": "order",
            "code": {
                "text": "Laboratory order"
            }
        }
        
        return service_request
    
    async def _create_hl7_message(self, fhir_resources: List[Dict[str, Any]], message_type: str, version: str) -> str:
        """Create HL7 message from FHIR resources."""
        # This is a simplified implementation
        # Real implementation would be more comprehensive
        
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        
        # Create MSH segment
        msh = f"MSH|^~\\&|VITA-AGENTS|VITA-AGENTS|RECEIVING-APP|RECEIVING-FACILITY|{timestamp}||{message_type}|{timestamp}|P|{version}"
        
        segments = [msh]
        
        # Add segments based on FHIR resources
        for resource in fhir_resources:
            resource_type = resource.get("resourceType")
            
            if resource_type == "Patient":
                pid = self._create_pid_from_patient(resource)
                segments.append(pid)
            
            elif resource_type == "Observation":
                obx = self._create_obx_from_observation(resource)
                segments.append(obx)
        
        return "\r".join(segments)
    
    def _create_pid_from_patient(self, patient: Dict[str, Any]) -> str:
        """Create PID segment from FHIR Patient resource."""
        patient_id = patient.get("id", "")
        name = patient.get("name", [{}])[0]
        family = name.get("family", "")
        given = name.get("given", [""])[0]
        gender = patient.get("gender", "")
        birth_date = patient.get("birthDate", "")
        
        pid = f"PID|1||{patient_id}|||{family}^{given}||{birth_date}|{gender.upper()}"
        return pid
    
    def _create_obx_from_observation(self, observation: Dict[str, Any]) -> str:
        """Create OBX segment from FHIR Observation resource."""
        code = observation.get("code", {}).get("coding", [{}])[0]
        code_value = code.get("code", "")
        code_display = code.get("display", "")
        value = observation.get("valueString", "")
        
        obx = f"OBX|1|ST|{code_value}^{code_display}||{value}"
        return obx
    
    def _apply_terminology_mappings(self, resource: Dict[str, Any], mappings: Dict[str, Any]) -> None:
        """Apply terminology mappings to a FHIR resource."""
        # This would recursively apply mappings to coded fields
        # Simplified implementation
        pass
    
    def _segment_to_dict(self, segment: Segment) -> Dict[str, Any]:
        """Convert HL7 segment to dictionary."""
        segment_dict = {
            "name": segment.name,
            "fields": {}
        }
        
        for field in segment.children:
            field_name = field.name if hasattr(field, 'name') else f"field_{len(segment_dict['fields'])}"
            segment_dict["fields"][field_name] = field.value if hasattr(field, 'value') else str(field)
        
        return segment_dict
    
    def _map_gender(self, hl7_gender: str) -> str:
        """Map HL7 gender to FHIR gender."""
        gender_map = {
            "M": "male",
            "F": "female",
            "O": "other",
            "U": "unknown"
        }
        return gender_map.get(hl7_gender.upper(), "unknown")
    
    def _map_patient_class(self, hl7_class: str) -> str:
        """Map HL7 patient class to FHIR encounter class."""
        class_map = {
            "I": "IMP",      # Inpatient
            "O": "AMB",      # Outpatient
            "E": "EMER",     # Emergency
            "AMB": "AMB"     # Ambulatory
        }
        return class_map.get(hl7_class.upper(), "AMB")
    
    def _map_order_status(self, hl7_status: str) -> str:
        """Map HL7 order status to FHIR request status."""
        status_map = {
            "NW": "active",
            "IP": "active", 
            "CM": "completed",
            "CA": "cancelled",
            "DC": "revoked"
        }
        return status_map.get(hl7_status.upper(), "unknown")
    
    def _format_hl7_date(self, hl7_date: str) -> Optional[str]:
        """Format HL7 date to FHIR date format."""
        if not hl7_date:
            return None
        
        try:
            # HL7 date format: YYYYMMDD or YYYYMMDDHHMMSS
            if len(hl7_date) >= 8:
                year = hl7_date[:4]
                month = hl7_date[4:6]
                day = hl7_date[6:8]
                return f"{year}-{month}-{day}"
        except:
            pass
        
        return None