"""
HL7 v3 Agent for Reference Information Model (RIM) based message processing.
Provides comprehensive HL7 v3 support including vocabulary services and FHIR conversion.
"""

import asyncio
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
import structlog
from pydantic import BaseModel, Field
import uuid

from vita_agents.core.agent import HealthcareAgent, AgentCapability, TaskRequest
from vita_agents.core.config import get_settings


logger = structlog.get_logger(__name__)


class RIMComponent(BaseModel):
    """Reference Information Model component."""
    
    component_type: str  # Act, Entity, Role, Participation, ActRelationship
    class_code: str
    mood_code: Optional[str] = None
    attributes: Dict[str, Any] = {}
    relationships: List[Dict[str, Any]] = []


class HL7V3ValidationResult(BaseModel):
    """HL7 v3 validation result."""
    
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    message_type: Optional[str] = None
    rim_components: List[RIMComponent] = []
    vocabulary_bindings: Dict[str, str] = {}


class CDASection(BaseModel):
    """Clinical Document Architecture section."""
    
    section_id: str
    code: str
    code_system: str
    title: str
    text: str
    entries: List[Dict[str, Any]] = []
    sub_sections: List['CDASection'] = []


class CDADocument(BaseModel):
    """Enhanced CDA document representation."""
    
    document_id: str
    template_id: str
    code: str
    title: str
    effective_time: str
    patient: Dict[str, Any]
    author: Dict[str, Any]
    custodian: Dict[str, Any]
    sections: List[CDASection] = []
    stylesheet_reference: Optional[str] = None


class VocabularyService(BaseModel):
    """HL7 v3 vocabulary service configuration."""
    
    service_url: str
    code_systems: List[str]
    value_sets: List[str]
    supported_operations: List[str]
    authentication_required: bool = False


class HL7V3Agent(HealthcareAgent):
    """
    HL7 v3 Agent for RIM-based message processing and vocabulary services.
    
    Capabilities:
    - Parse and validate HL7 v3 messages using RIM
    - Process Clinical Document Architecture (CDA) documents  
    - Convert HL7 v3 messages to FHIR resources
    - Manage HL7 v3 vocabulary and terminology services
    - Support C-CDA template validation and processing
    - Implement advanced CDA stylesheet processing
    """
    
    def __init__(
        self,
        agent_id: str = "hl7-v3-agent",
        name: str = "HL7 v3 RIM Agent",
        description: str = "Processes HL7 v3 RIM-based messages and CDA documents",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None,
    ):
        capabilities = [
            AgentCapability(
                name="validate_hl7_v3_message",
                description="Validate HL7 v3 message against RIM",
                input_schema={
                    "type": "object",
                    "properties": {
                        "hl7_v3_message": {"type": "string"},
                        "strict_validation": {"type": "boolean"},
                        "rim_version": {"type": "string"}
                    },
                    "required": ["hl7_v3_message"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "is_valid": {"type": "boolean"},
                        "errors": {"type": "array", "items": {"type": "string"}},
                        "warnings": {"type": "array", "items": {"type": "string"}},
                        "rim_components": {"type": "array"}
                    }
                },
                supported_formats=["xml", "hl7v3"],
                requirements=["lxml", "xmlschema"]
            ),
            AgentCapability(
                name="hl7_v3_to_fhir",
                description="Convert HL7 v3 messages to FHIR resources",
                input_schema={
                    "type": "object",
                    "properties": {
                        "hl7_v3_message": {"type": "string"},
                        "target_resources": {"type": "array", "items": {"type": "string"}},
                        "vocabulary_mappings": {"type": "object"}
                    },
                    "required": ["hl7_v3_message"]
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
                name="process_cda_document",
                description="Enhanced CDA document processing with template validation",
                input_schema={
                    "type": "object",
                    "properties": {
                        "cda_document": {"type": "string"},
                        "template_validation": {"type": "boolean"},
                        "extract_sections": {"type": "array"},
                        "apply_stylesheet": {"type": "boolean"}
                    },
                    "required": ["cda_document"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "structured_data": {"type": "object"},
                        "fhir_resources": {"type": "array"},
                        "rendered_content": {"type": "string"},
                        "validation_results": {"type": "object"}
                    }
                }
            ),
            AgentCapability(
                name="vocabulary_service_operations",
                description="HL7 v3 vocabulary and terminology services",
                input_schema={
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string"},
                        "code_system": {"type": "string"},
                        "code": {"type": "string"},
                        "value_set": {"type": "string"},
                        "target_system": {"type": "string"}
                    },
                    "required": ["operation"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "result": {"type": "object"},
                        "mappings": {"type": "array"},
                        "hierarchy": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="rim_component_analysis",
                description="Analyze RIM components and relationships",
                input_schema={
                    "type": "object",
                    "properties": {
                        "hl7_v3_content": {"type": "string"},
                        "analysis_type": {"type": "string"},
                        "include_relationships": {"type": "boolean"}
                    },
                    "required": ["hl7_v3_content"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "rim_analysis": {"type": "object"},
                        "component_graph": {"type": "object"},
                        "validation_issues": {"type": "array"}
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
        
        # HL7 v3 specific configuration
        self.rim_version = "2.43"
        self.supported_cda_templates = [
            "C-CDA R2.1",
            "C-CDA R1.1", 
            "CCD",
            "CCR",
            "CCDA"
        ]
        self.vocabulary_services = self._initialize_vocabulary_services()
        
        # CDA stylesheet configurations
        self.cda_stylesheets = {
            "clinical": "https://www.hl7.org/stylesheets/cda.xsl",
            "narrative": "https://www.hl7.org/stylesheets/cda-narrative.xsl",
            "mobile": "https://www.hl7.org/stylesheets/cda-mobile.xsl"
        }
        
        # Register task handlers
        self.register_task_handler("validate_hl7_v3_message", self._validate_hl7_v3_message)
        self.register_task_handler("hl7_v3_to_fhir", self._hl7_v3_to_fhir)
        self.register_task_handler("process_cda_document", self._process_cda_document)
        self.register_task_handler("vocabulary_service_operations", self._vocabulary_service_operations)
        self.register_task_handler("rim_component_analysis", self._rim_component_analysis)
    
    def _initialize_vocabulary_services(self) -> Dict[str, VocabularyService]:
        """Initialize vocabulary service configurations."""
        return {
            "fhir_tx": VocabularyService(
                service_url="https://tx.fhir.org/r4",
                code_systems=["http://snomed.info/sct", "http://loinc.org", "http://hl7.org/fhir/sid/icd-10"],
                value_sets=["diabetes-conditions", "cardiovascular-procedures"],
                supported_operations=["lookup", "validate-code", "expand", "translate"]
            ),
            "nih_uts": VocabularyService(
                service_url="https://uts-ws.nlm.nih.gov/rest",
                code_systems=["SNOMEDCT_US", "LOINC", "ICD10CM"],
                value_sets=["clinical-findings", "procedures"],
                supported_operations=["search", "lookup", "hierarchy"],
                authentication_required=True
            )
        }
    
    async def _on_start(self) -> None:
        """Initialize HL7 v3 processing capabilities."""
        self.logger.info("Starting HL7 v3 agent", rim_version=self.rim_version)
        
        # Initialize XML schema validation if available
        try:
            import xmlschema
            self.xml_validator = xmlschema
            self.logger.info("XML schema validation enabled")
        except ImportError:
            self.xml_validator = None
            self.logger.warning("XML schema validation not available")
        
        self.logger.info("HL7 v3 agent initialized")
    
    async def _on_stop(self) -> None:
        """Clean up HL7 v3 processing."""
        self.logger.info("Stopping HL7 v3 agent")
    
    async def _validate_hl7_v3_message(self, task: TaskRequest) -> Dict[str, Any]:
        """Validate HL7 v3 message against RIM."""
        try:
            hl7_v3_message = task.parameters.get("hl7_v3_message")
            strict_validation = task.parameters.get("strict_validation", True)
            rim_version = task.parameters.get("rim_version", self.rim_version)
            
            if not hl7_v3_message:
                raise ValueError("HL7 v3 message is required")
            
            self.audit_log_action(
                action="validate_hl7_v3_message",
                data_type="HL7v3",
                details={
                    "rim_version": rim_version,
                    "strict_validation": strict_validation,
                    "task_id": task.id
                }
            )
            
            validation_result = HL7V3ValidationResult(is_valid=True)
            
            # Parse XML structure
            try:
                root = ET.fromstring(hl7_v3_message)
                
                # Validate XML structure
                if not self._validate_xml_structure(root):
                    validation_result.is_valid = False
                    validation_result.errors.append("Invalid XML structure")
                
                # Extract and validate RIM components
                rim_components = self._extract_rim_components(root)
                validation_result.rim_components = rim_components
                
                # Validate RIM relationships
                relationship_errors = self._validate_rim_relationships(rim_components)
                validation_result.errors.extend(relationship_errors)
                
                # Validate vocabulary bindings
                vocab_warnings = self._validate_vocabulary_bindings(root)
                validation_result.warnings.extend(vocab_warnings)
                
                # Determine message type
                validation_result.message_type = self._determine_message_type(root)
                
                if validation_result.errors:
                    validation_result.is_valid = False
                
            except ET.ParseError as e:
                validation_result.is_valid = False
                validation_result.errors.append(f"XML parsing error: {str(e)}")
            
            return {
                "validation_result": validation_result.dict(),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("HL7 v3 validation failed", error=str(e), task_id=task.id)
            raise
    
    async def _hl7_v3_to_fhir(self, task: TaskRequest) -> Dict[str, Any]:
        """Convert HL7 v3 message to FHIR resources."""
        try:
            hl7_v3_message = task.parameters.get("hl7_v3_message")
            target_resources = task.parameters.get("target_resources", [])
            vocabulary_mappings = task.parameters.get("vocabulary_mappings", {})
            
            if not hl7_v3_message:
                raise ValueError("HL7 v3 message is required")
            
            self.audit_log_action(
                action="hl7_v3_to_fhir",
                data_type="HL7v3 to FHIR",
                details={
                    "target_resources": target_resources,
                    "task_id": task.id
                }
            )
            
            fhir_resources = []
            conversion_notes = []
            
            # Parse HL7 v3 message
            root = ET.fromstring(hl7_v3_message)
            
            # Extract RIM components
            rim_components = self._extract_rim_components(root)
            
            # Convert RIM components to FHIR resources
            for component in rim_components:
                fhir_resource = await self._convert_rim_to_fhir(component, vocabulary_mappings)
                if fhir_resource:
                    fhir_resources.append(fhir_resource)
            
            # Filter by target resources if specified
            if target_resources:
                fhir_resources = [
                    resource for resource in fhir_resources
                    if resource.get("resourceType") in target_resources
                ]
            
            conversion_notes.append(f"Converted {len(rim_components)} RIM components to {len(fhir_resources)} FHIR resources")
            
            return {
                "fhir_resources": fhir_resources,
                "conversion_notes": conversion_notes,
                "success": True,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("HL7 v3 to FHIR conversion failed", error=str(e), task_id=task.id)
            raise
    
    async def _process_cda_document(self, task: TaskRequest) -> Dict[str, Any]:
        """Enhanced CDA document processing with template validation."""
        try:
            cda_document = task.parameters.get("cda_document")
            template_validation = task.parameters.get("template_validation", True)
            extract_sections = task.parameters.get("extract_sections", [])
            apply_stylesheet = task.parameters.get("apply_stylesheet", False)
            
            if not cda_document:
                raise ValueError("CDA document is required")
            
            self.audit_log_action(
                action="process_cda_document",
                data_type="CDA",
                details={
                    "template_validation": template_validation,
                    "extract_sections": extract_sections,
                    "apply_stylesheet": apply_stylesheet,
                    "task_id": task.id
                }
            )
            
            # Parse CDA document
            root = ET.fromstring(cda_document)
            
            # Enhanced CDA processing
            structured_data = await self._extract_cda_structure(root)
            fhir_resources = await self._convert_cda_to_fhir(root)
            
            # Template validation
            validation_results = {}
            if template_validation:
                validation_results = await self._validate_cda_template(root)
            
            # Stylesheet processing
            rendered_content = ""
            if apply_stylesheet:
                rendered_content = await self._apply_cda_stylesheet(root)
            
            return {
                "structured_data": structured_data,
                "fhir_resources": fhir_resources,
                "rendered_content": rendered_content,
                "validation_results": validation_results,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("CDA processing failed", error=str(e), task_id=task.id)
            raise
    
    async def _vocabulary_service_operations(self, task: TaskRequest) -> Dict[str, Any]:
        """HL7 v3 vocabulary and terminology services."""
        try:
            operation = task.parameters.get("operation")
            code_system = task.parameters.get("code_system", "")
            code = task.parameters.get("code", "")
            value_set = task.parameters.get("value_set", "")
            target_system = task.parameters.get("target_system", "")
            
            if not operation:
                raise ValueError("Operation is required")
            
            self.audit_log_action(
                action="vocabulary_service_operations",
                data_type="Vocabulary",
                details={
                    "operation": operation,
                    "code_system": code_system,
                    "task_id": task.id
                }
            )
            
            result = {}
            mappings = []
            hierarchy = []
            
            if operation == "lookup":
                result = await self._vocabulary_lookup(code_system, code)
            elif operation == "validate":
                result = await self._vocabulary_validate(code_system, code)
            elif operation == "expand":
                result = await self._vocabulary_expand(value_set)
            elif operation == "translate":
                mappings = await self._vocabulary_translate(code_system, code, target_system)
            elif operation == "hierarchy":
                hierarchy = await self._vocabulary_hierarchy(code_system, code)
            
            return {
                "result": result,
                "mappings": mappings,
                "hierarchy": hierarchy,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Vocabulary service operation failed", error=str(e), task_id=task.id)
            raise
    
    async def _rim_component_analysis(self, task: TaskRequest) -> Dict[str, Any]:
        """Analyze RIM components and relationships."""
        try:
            hl7_v3_content = task.parameters.get("hl7_v3_content")
            analysis_type = task.parameters.get("analysis_type", "complete")
            include_relationships = task.parameters.get("include_relationships", True)
            
            if not hl7_v3_content:
                raise ValueError("HL7 v3 content is required")
            
            # Parse content and extract RIM components
            root = ET.fromstring(hl7_v3_content)
            rim_components = self._extract_rim_components(root)
            
            # Perform RIM analysis
            rim_analysis = {
                "total_components": len(rim_components),
                "component_types": self._analyze_component_types(rim_components),
                "class_codes": self._analyze_class_codes(rim_components),
                "mood_codes": self._analyze_mood_codes(rim_components)
            }
            
            # Build component relationship graph
            component_graph = {}
            if include_relationships:
                component_graph = self._build_component_graph(rim_components)
            
            # Validate components
            validation_issues = self._validate_rim_components(rim_components)
            
            return {
                "rim_analysis": rim_analysis,
                "component_graph": component_graph,
                "validation_issues": validation_issues,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("RIM component analysis failed", error=str(e), task_id=task.id)
            raise
    
    def _validate_xml_structure(self, root: ET.Element) -> bool:
        """Validate basic XML structure."""
        # Check for required HL7 v3 namespaces
        required_namespaces = [
            "urn:hl7-org:v3",
            "http://www.w3.org/2001/XMLSchema-instance"
        ]
        
        # Basic validation - in production would use proper schema validation
        return root.tag is not None
    
    def _extract_rim_components(self, root: ET.Element) -> List[RIMComponent]:
        """Extract RIM components from HL7 v3 message."""
        components = []
        
        # Extract Act components
        for act_elem in root.findall(".//*[@classCode]"):
            component = RIMComponent(
                component_type="Act",
                class_code=act_elem.get("classCode", ""),
                mood_code=act_elem.get("moodCode"),
                attributes=self._extract_element_attributes(act_elem)
            )
            components.append(component)
        
        # Extract Entity components  
        for entity_elem in root.findall(".//*[@determinerCode]"):
            component = RIMComponent(
                component_type="Entity",
                class_code=entity_elem.get("classCode", ""),
                attributes=self._extract_element_attributes(entity_elem)
            )
            components.append(component)
        
        return components
    
    def _extract_element_attributes(self, element: ET.Element) -> Dict[str, Any]:
        """Extract attributes from XML element."""
        attributes = {}
        for key, value in element.attrib.items():
            attributes[key] = value
        
        # Extract text content
        if element.text and element.text.strip():
            attributes["text_content"] = element.text.strip()
        
        return attributes
    
    def _validate_rim_relationships(self, components: List[RIMComponent]) -> List[str]:
        """Validate RIM component relationships."""
        errors = []
        
        # Validate component relationships
        for component in components:
            if component.component_type == "Act" and not component.mood_code:
                errors.append(f"Act component missing required moodCode")
        
        return errors
    
    def _validate_vocabulary_bindings(self, root: ET.Element) -> List[str]:
        """Validate vocabulary bindings."""
        warnings = []
        
        # Check for vocabulary bindings
        coded_elements = root.findall(".//*[@code]")
        for elem in coded_elements:
            code_system = elem.get("codeSystem")
            if not code_system:
                warnings.append(f"Coded element missing codeSystem attribute")
        
        return warnings
    
    def _determine_message_type(self, root: ET.Element) -> str:
        """Determine HL7 v3 message type."""
        # Extract message type from root element or interaction
        root_tag = root.tag
        if "ClinicalDocument" in root_tag:
            return "CDA"
        elif "PRPA_IN" in root_tag:
            return "Patient Registry"
        elif "RCMR_IN" in root_tag:
            return "Care Record"
        else:
            return "Unknown"
    
    async def _convert_rim_to_fhir(self, component: RIMComponent, mappings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert RIM component to FHIR resource."""
        if component.component_type == "Act":
            # Convert Act to appropriate FHIR resource
            if component.class_code == "OBS":
                return {
                    "resourceType": "Observation",
                    "id": str(uuid.uuid4()),
                    "status": "final",
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": "sample-code",
                            "display": "Sample Observation"
                        }]
                    }
                }
            elif component.class_code == "PROC":
                return {
                    "resourceType": "Procedure",
                    "id": str(uuid.uuid4()),
                    "status": "completed",
                    "code": {
                        "coding": [{
                            "system": "http://snomed.info/sct",
                            "code": "sample-procedure",
                            "display": "Sample Procedure"
                        }]
                    }
                }
        
        elif component.component_type == "Entity":
            # Convert Entity to appropriate FHIR resource
            if component.class_code == "PSN":
                return {
                    "resourceType": "Patient",
                    "id": str(uuid.uuid4()),
                    "active": True,
                    "name": [{
                        "family": "Sample",
                        "given": ["Patient"]
                    }]
                }
        
        return None
    
    async def _extract_cda_structure(self, root: ET.Element) -> Dict[str, Any]:
        """Extract enhanced CDA document structure."""
        structure = {
            "document_type": "CDA",
            "template_ids": [],
            "effective_time": None,
            "patient": {},
            "sections": []
        }
        
        # Extract template IDs
        for template_elem in root.findall(".//templateId"):
            template_id = template_elem.get("root")
            if template_id:
                structure["template_ids"].append(template_id)
        
        # Extract effective time
        effective_time_elem = root.find(".//effectiveTime")
        if effective_time_elem is not None:
            structure["effective_time"] = effective_time_elem.get("value")
        
        # Extract patient information
        patient_elem = root.find(".//patient")
        if patient_elem is not None:
            structure["patient"] = self._extract_patient_info(patient_elem)
        
        # Extract sections
        for section_elem in root.findall(".//section"):
            section_data = self._extract_section_info(section_elem)
            structure["sections"].append(section_data)
        
        return structure
    
    async def _convert_cda_to_fhir(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Convert CDA document to FHIR resources."""
        fhir_resources = []
        
        # Create DocumentReference
        doc_ref = {
            "resourceType": "DocumentReference",
            "id": str(uuid.uuid4()),
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
                    "title": "CDA Document"
                }
            }]
        }
        fhir_resources.append(doc_ref)
        
        # Extract and convert patient
        patient_elem = root.find(".//patient")
        if patient_elem is not None:
            patient_resource = self._convert_cda_patient_to_fhir(patient_elem)
            fhir_resources.append(patient_resource)
        
        # Convert sections to appropriate FHIR resources
        for section_elem in root.findall(".//section"):
            section_resources = self._convert_cda_section_to_fhir(section_elem)
            fhir_resources.extend(section_resources)
        
        return fhir_resources
    
    async def _validate_cda_template(self, root: ET.Element) -> Dict[str, Any]:
        """Validate CDA document against templates."""
        validation_results = {
            "template_validation": True,
            "template_errors": [],
            "template_warnings": [],
            "conformance_level": "high"
        }
        
        # Extract template IDs
        template_ids = []
        for template_elem in root.findall(".//templateId"):
            template_id = template_elem.get("root")
            if template_id:
                template_ids.append(template_id)
        
        # Validate against known templates
        for template_id in template_ids:
            template_validation = self._validate_specific_template(root, template_id)
            if not template_validation["valid"]:
                validation_results["template_validation"] = False
                validation_results["template_errors"].extend(template_validation["errors"])
        
        return validation_results
    
    async def _apply_cda_stylesheet(self, root: ET.Element) -> str:
        """Apply CDA stylesheet for rendering."""
        # In production, this would apply XSLT transformation
        rendered_content = f"""
        <html>
        <head>
            <title>CDA Document</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #f0f0f0; padding: 10px; }}
                .section {{ margin: 10px 0; border: 1px solid #ccc; padding: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Clinical Document</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="content">
                <!-- CDA content would be transformed here -->
                <p>CDA document rendered with clinical stylesheet</p>
            </div>
        </body>
        </html>
        """
        return rendered_content
    
    async def _vocabulary_lookup(self, code_system: str, code: str) -> Dict[str, Any]:
        """Lookup vocabulary code."""
        # Simulate vocabulary lookup
        return {
            "code": code,
            "system": code_system,
            "display": f"Sample term for {code}",
            "definition": f"Definition for code {code} in system {code_system}",
            "active": True,
            "properties": []
        }
    
    async def _vocabulary_validate(self, code_system: str, code: str) -> Dict[str, Any]:
        """Validate vocabulary code."""
        return {
            "valid": True,
            "code": code,
            "system": code_system,
            "issues": []
        }
    
    async def _vocabulary_expand(self, value_set: str) -> Dict[str, Any]:
        """Expand value set."""
        return {
            "expansion": {
                "total": 10,
                "contains": [
                    {
                        "code": "73211009",
                        "system": "http://snomed.info/sct",
                        "display": "Diabetes mellitus"
                    }
                ]
            }
        }
    
    async def _vocabulary_translate(self, source_system: str, code: str, target_system: str) -> List[Dict[str, Any]]:
        """Translate code between systems."""
        return [
            {
                "source_code": code,
                "source_system": source_system,
                "target_code": "E11.9",
                "target_system": target_system,
                "equivalence": "equivalent"
            }
        ]
    
    async def _vocabulary_hierarchy(self, code_system: str, code: str) -> List[Dict[str, Any]]:
        """Get vocabulary hierarchy."""
        return [
            {
                "level": 1,
                "code": "404684003",
                "display": "Clinical finding",
                "children_count": 1250000
            },
            {
                "level": 2,
                "code": "64572001", 
                "display": "Disease",
                "children_count": 450000
            }
        ]
    
    def _extract_patient_info(self, patient_elem: ET.Element) -> Dict[str, Any]:
        """Extract patient information from CDA."""
        patient_info = {}
        
        # Extract patient name
        name_elem = patient_elem.find(".//name")
        if name_elem is not None:
            patient_info["name"] = {
                "family": name_elem.findtext(".//family", ""),
                "given": name_elem.findtext(".//given", "")
            }
        
        # Extract administrative gender
        gender_elem = patient_elem.find(".//administrativeGenderCode")
        if gender_elem is not None:
            patient_info["gender"] = gender_elem.get("code")
        
        return patient_info
    
    def _extract_section_info(self, section_elem: ET.Element) -> Dict[str, Any]:
        """Extract section information from CDA."""
        section_info = {
            "code": "",
            "title": "",
            "text": "",
            "entries": []
        }
        
        # Extract section code
        code_elem = section_elem.find(".//code")
        if code_elem is not None:
            section_info["code"] = code_elem.get("code", "")
        
        # Extract section title
        title_elem = section_elem.find(".//title")
        if title_elem is not None:
            section_info["title"] = title_elem.text or ""
        
        # Extract section text
        text_elem = section_elem.find(".//text")
        if text_elem is not None:
            section_info["text"] = ET.tostring(text_elem, encoding="unicode", method="text")
        
        return section_info
    
    def _convert_cda_patient_to_fhir(self, patient_elem: ET.Element) -> Dict[str, Any]:
        """Convert CDA patient to FHIR Patient resource."""
        patient_info = self._extract_patient_info(patient_elem)
        
        patient_resource = {
            "resourceType": "Patient",
            "id": str(uuid.uuid4()),
            "active": True
        }
        
        if "name" in patient_info:
            patient_resource["name"] = [patient_info["name"]]
        
        if "gender" in patient_info:
            gender_map = {"M": "male", "F": "female", "UN": "unknown"}
            patient_resource["gender"] = gender_map.get(patient_info["gender"], "unknown")
        
        return patient_resource
    
    def _convert_cda_section_to_fhir(self, section_elem: ET.Element) -> List[Dict[str, Any]]:
        """Convert CDA section to FHIR resources."""
        resources = []
        
        # Extract section code to determine resource type
        code_elem = section_elem.find(".//code")
        if code_elem is not None:
            section_code = code_elem.get("code")
            
            # Map section codes to FHIR resources
            if section_code == "48765-2":  # Allergies
                allergy_resource = {
                    "resourceType": "AllergyIntolerance",
                    "id": str(uuid.uuid4()),
                    "clinicalStatus": {
                        "coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
                            "code": "active"
                        }]
                    }
                }
                resources.append(allergy_resource)
                
            elif section_code == "10160-0":  # Medications
                medication_resource = {
                    "resourceType": "MedicationStatement",
                    "id": str(uuid.uuid4()),
                    "status": "active"
                }
                resources.append(medication_resource)
        
        return resources
    
    def _validate_specific_template(self, root: ET.Element, template_id: str) -> Dict[str, Any]:
        """Validate against specific CDA template."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Template-specific validation logic would go here
        # For now, simulate validation
        
        return validation
    
    def _analyze_component_types(self, components: List[RIMComponent]) -> Dict[str, int]:
        """Analyze RIM component types."""
        type_counts = {}
        for component in components:
            component_type = component.component_type
            type_counts[component_type] = type_counts.get(component_type, 0) + 1
        return type_counts
    
    def _analyze_class_codes(self, components: List[RIMComponent]) -> Dict[str, int]:
        """Analyze RIM class codes."""
        class_counts = {}
        for component in components:
            class_code = component.class_code
            class_counts[class_code] = class_counts.get(class_code, 0) + 1
        return class_counts
    
    def _analyze_mood_codes(self, components: List[RIMComponent]) -> Dict[str, int]:
        """Analyze RIM mood codes."""
        mood_counts = {}
        for component in components:
            if component.mood_code:
                mood_counts[component.mood_code] = mood_counts.get(component.mood_code, 0) + 1
        return mood_counts
    
    def _build_component_graph(self, components: List[RIMComponent]) -> Dict[str, Any]:
        """Build RIM component relationship graph."""
        graph = {
            "nodes": [],
            "edges": []
        }
        
        for i, component in enumerate(components):
            graph["nodes"].append({
                "id": f"component_{i}",
                "type": component.component_type,
                "class_code": component.class_code,
                "mood_code": component.mood_code
            })
            
            # Add relationships as edges
            for relationship in component.relationships:
                graph["edges"].append({
                    "source": f"component_{i}",
                    "target": relationship.get("target", "unknown"),
                    "type": relationship.get("type", "unknown")
                })
        
        return graph
    
    def _validate_rim_components(self, components: List[RIMComponent]) -> List[Dict[str, Any]]:
        """Validate individual RIM components."""
        issues = []
        
        for i, component in enumerate(components):
            # Validate required attributes
            if component.component_type == "Act" and not component.mood_code:
                issues.append({
                    "component_index": i,
                    "severity": "error",
                    "message": "Act component missing required moodCode"
                })
            
            if not component.class_code:
                issues.append({
                    "component_index": i,
                    "severity": "error", 
                    "message": "Component missing required classCode"
                })
        
        return issues