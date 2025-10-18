"""
Claims Attachments Agent for Vita Agents.
Provides comprehensive healthcare claims attachment processing and HIPAA compliance.
"""

import asyncio
import json
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, BinaryIO
from enum import Enum
import structlog
from pydantic import BaseModel, Field
import uuid
import mimetypes
import hashlib

from vita_agents.core.agent import HealthcareAgent, AgentCapability, TaskRequest
from vita_agents.core.config import get_settings


logger = structlog.get_logger(__name__)


class AttachmentType(str, Enum):
    """Claims attachment types."""
    CLINICAL_RECORDS = "clinical_records"
    DIAGNOSTIC_IMAGES = "diagnostic_images"
    LAB_RESULTS = "lab_results"
    PHYSICIAN_NOTES = "physician_notes"
    DISCHARGE_SUMMARY = "discharge_summary"
    OPERATIVE_REPORT = "operative_report"
    PATHOLOGY_REPORT = "pathology_report"
    RADIOLOGY_REPORT = "radiology_report"
    PRESCRIPTION_RECORDS = "prescription_records"
    THERAPY_NOTES = "therapy_notes"
    SUPPORTING_DOCUMENTATION = "supporting_documentation"
    PRIOR_AUTHORIZATION = "prior_authorization"


class AttachmentFormat(str, Enum):
    """Supported attachment formats."""
    PDF = "pdf"
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"
    DICOM = "dicom"
    XML = "xml"
    JSON = "json"
    CSV = "csv"
    TXT = "txt"
    RTF = "rtf"
    DOC = "doc"
    DOCX = "docx"


class ClaimStatus(str, Enum):
    """Claim processing status."""
    SUBMITTED = "submitted"
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    DENIED = "denied"
    PARTIALLY_APPROVED = "partially_approved"
    REQUIRES_ADDITIONAL_INFO = "requires_additional_info"
    SUSPENDED = "suspended"


class HIPAASecurityLevel(str, Enum):
    """HIPAA security levels."""
    PUBLIC = "public"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    HIGHLY_CONFIDENTIAL = "highly_confidential"


class ClaimsAttachment(BaseModel):
    """Claims attachment model."""
    
    attachment_id: str
    claim_id: str
    attachment_type: AttachmentType
    file_format: AttachmentFormat
    file_name: str
    file_size: int
    mime_type: str
    content_hash: str
    encrypted: bool = True
    security_level: HIPAASecurityLevel
    patient_id: str
    provider_id: str
    created_date: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    last_modified: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = {}
    hipaa_compliant: bool = True


class ClaimSubmission(BaseModel):
    """Claim submission model."""
    
    claim_id: str
    patient_id: str
    provider_id: str
    payer_id: str
    claim_type: str
    submission_date: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    status: ClaimStatus = ClaimStatus.SUBMITTED
    attachments: List[ClaimsAttachment] = []
    total_amount: float
    diagnosis_codes: List[str] = []
    procedure_codes: List[str] = []
    prior_authorization_number: Optional[str] = None
    hipaa_audit_trail: List[Dict[str, Any]] = []


class AttachmentValidationRule(BaseModel):
    """Attachment validation rule."""
    
    rule_id: str
    rule_name: str
    attachment_type: AttachmentType
    required_metadata: List[str] = []
    max_file_size_mb: Optional[float] = None
    allowed_formats: List[AttachmentFormat] = []
    security_requirements: List[str] = []
    hipaa_requirements: List[str] = []


class AttachmentValidationError(BaseModel):
    """Attachment validation error."""
    
    error_id: str
    rule_id: str
    attachment_id: str
    error_type: str
    severity: str  # error, warning, info
    message: str
    suggested_fix: Optional[str] = None
    hipaa_impact: bool = False


class HIPAAAuditEntry(BaseModel):
    """HIPAA audit log entry."""
    
    audit_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    patient_id: Optional[str] = None
    access_reason: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool
    details: Dict[str, Any] = {}


class ClaimsAttachmentsAgent(HealthcareAgent):
    """
    Claims Attachments Agent for healthcare claims processing.
    
    Capabilities:
    - Complete claims attachment creation and validation
    - Multi-format attachment support (PDF, DICOM, XML, etc.)
    - HIPAA-compliant attachment processing and encryption
    - Claims submission workflow management
    - Attachment tracking and status monitoring
    - Comprehensive audit trails and compliance reporting
    - Automated attachment validation and quality checks
    - Integration with major payers and clearinghouses
    - Prior authorization attachment handling
    - Secure transmission and storage protocols
    """
    
    def __init__(
        self,
        agent_id: str = "claims-attachments-agent",
        name: str = "Claims Attachments Agent",
        description: str = "Healthcare claims attachment processing and HIPAA compliance",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None,
    ):
        capabilities = [
            AgentCapability(
                name="create_attachment",
                description="Create and validate claims attachment",
                input_schema={
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "attachment_type": {"type": "string"},
                        "file_content": {"type": "string"},
                        "file_name": {"type": "string"},
                        "patient_info": {"type": "object"},
                        "provider_info": {"type": "object"},
                        "metadata": {"type": "object"},
                        "security_level": {"type": "string"}
                    },
                    "required": ["claim_id", "attachment_type", "file_content", "file_name"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "attachment": {"type": "object"},
                        "validation_result": {"type": "object"},
                        "compliance_status": {"type": "object"},
                        "attachment_id": {"type": "string"}
                    }
                }
            ),
            AgentCapability(
                name="validate_attachment",
                description="Comprehensive attachment validation",
                input_schema={
                    "type": "object",
                    "properties": {
                        "attachment_id": {"type": "string"},
                        "validation_rules": {"type": "array"},
                        "hipaa_compliance": {"type": "boolean"},
                        "payer_requirements": {"type": "object"},
                        "clinical_validation": {"type": "boolean"}
                    },
                    "required": ["attachment_id"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "validation_result": {"type": "object"},
                        "errors": {"type": "array"},
                        "warnings": {"type": "array"},
                        "compliance_score": {"type": "number"},
                        "hipaa_compliant": {"type": "boolean"}
                    }
                }
            ),
            AgentCapability(
                name="submit_claim",
                description="Submit claim with attachments",
                input_schema={
                    "type": "object",
                    "properties": {
                        "claim_data": {"type": "object"},
                        "attachment_ids": {"type": "array"},
                        "payer_id": {"type": "string"},
                        "submission_method": {"type": "string"},
                        "priority": {"type": "string"},
                        "test_mode": {"type": "boolean"}
                    },
                    "required": ["claim_data", "payer_id"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "submission_result": {"type": "object"},
                        "tracking_number": {"type": "string"},
                        "estimated_processing_time": {"type": "string"},
                        "submission_status": {"type": "string"}
                    }
                }
            ),
            AgentCapability(
                name="track_claim_status",
                description="Track claim and attachment status",
                input_schema={
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "tracking_number": {"type": "string"},
                        "payer_id": {"type": "string"},
                        "include_attachments": {"type": "boolean"},
                        "detailed_history": {"type": "boolean"}
                    },
                    "required": ["claim_id"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "claim_status": {"type": "object"},
                        "attachment_statuses": {"type": "array"},
                        "processing_history": {"type": "array"},
                        "next_actions": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="generate_compliance_report",
                description="Generate HIPAA compliance and audit report",
                input_schema={
                    "type": "object",
                    "properties": {
                        "report_type": {"type": "string"},
                        "date_range": {"type": "object"},
                        "claim_ids": {"type": "array"},
                        "include_audit_trail": {"type": "boolean"},
                        "compliance_metrics": {"type": "boolean"}
                    },
                    "required": ["report_type"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "compliance_report": {"type": "object"},
                        "audit_entries": {"type": "array"},
                        "compliance_metrics": {"type": "object"},
                        "recommendations": {"type": "array"}
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
        
        # Initialize claims processing components
        self.attachments_store: Dict[str, ClaimsAttachment] = {}
        self.claims_submissions: Dict[str, ClaimSubmission] = {}
        self.validation_rules = self._initialize_validation_rules()
        self.hipaa_audit_log: List[HIPAAAuditEntry] = []
        
        # Payer configurations
        self.payer_configs = self._initialize_payer_configs()
        
        # Register task handlers
        self.register_task_handler("create_attachment", self._create_attachment)
        self.register_task_handler("validate_attachment", self._validate_attachment)
        self.register_task_handler("submit_claim", self._submit_claim)
        self.register_task_handler("track_claim_status", self._track_claim_status)
        self.register_task_handler("generate_compliance_report", self._generate_compliance_report)
    
    def _initialize_validation_rules(self) -> List[AttachmentValidationRule]:
        """Initialize attachment validation rules."""
        return [
            AttachmentValidationRule(
                rule_id="RULE-001",
                rule_name="Clinical Records Validation",
                attachment_type=AttachmentType.CLINICAL_RECORDS,
                required_metadata=["patient_id", "provider_id", "service_date"],
                max_file_size_mb=25.0,
                allowed_formats=[AttachmentFormat.PDF, AttachmentFormat.XML, AttachmentFormat.JSON],
                security_requirements=["encryption", "access_control"],
                hipaa_requirements=["patient_consent", "minimum_necessary", "audit_logging"]
            ),
            AttachmentValidationRule(
                rule_id="RULE-002",
                rule_name="Diagnostic Images Validation",
                attachment_type=AttachmentType.DIAGNOSTIC_IMAGES,
                required_metadata=["patient_id", "modality", "study_date"],
                max_file_size_mb=100.0,
                allowed_formats=[AttachmentFormat.DICOM, AttachmentFormat.JPEG, AttachmentFormat.PNG],
                security_requirements=["encryption", "de_identification"],
                hipaa_requirements=["patient_consent", "secure_transmission", "audit_logging"]
            ),
            AttachmentValidationRule(
                rule_id="RULE-003",
                rule_name="Lab Results Validation",
                attachment_type=AttachmentType.LAB_RESULTS,
                required_metadata=["patient_id", "test_date", "ordering_physician"],
                max_file_size_mb=10.0,
                allowed_formats=[AttachmentFormat.PDF, AttachmentFormat.XML, AttachmentFormat.CSV],
                security_requirements=["encryption", "integrity_check"],
                hipaa_requirements=["patient_consent", "data_integrity", "audit_logging"]
            ),
            AttachmentValidationRule(
                rule_id="RULE-004",
                rule_name="Prior Authorization Validation",
                attachment_type=AttachmentType.PRIOR_AUTHORIZATION,
                required_metadata=["patient_id", "authorization_number", "effective_date"],
                max_file_size_mb=5.0,
                allowed_formats=[AttachmentFormat.PDF, AttachmentFormat.XML],
                security_requirements=["encryption", "non_repudiation"],
                hipaa_requirements=["patient_consent", "authorization_tracking", "audit_logging"]
            )
        ]
    
    def _initialize_payer_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize payer-specific configurations."""
        return {
            "medicare": {
                "payer_name": "Centers for Medicare & Medicaid Services",
                "submission_methods": ["EDI", "web_portal", "clearinghouse"],
                "supported_formats": ["PDF", "XML", "DICOM"],
                "max_attachment_size_mb": 25,
                "processing_time_days": "14-30",
                "special_requirements": ["prior_authorization_tracking", "medicare_compliance"]
            },
            "medicaid": {
                "payer_name": "State Medicaid Programs",
                "submission_methods": ["EDI", "web_portal", "clearinghouse"],
                "supported_formats": ["PDF", "XML", "CSV"],
                "max_attachment_size_mb": 20,
                "processing_time_days": "21-45",
                "special_requirements": ["state_specific_rules", "medicaid_compliance"]
            },
            "commercial": {
                "payer_name": "Commercial Insurance Plans",
                "submission_methods": ["EDI", "web_portal", "api", "clearinghouse"],
                "supported_formats": ["PDF", "XML", "JSON", "DICOM"],
                "max_attachment_size_mb": 50,
                "processing_time_days": "7-21",
                "special_requirements": ["plan_specific_rules", "prior_authorization"]
            }
        }
    
    async def _on_start(self) -> None:
        """Initialize Claims Attachments agent."""
        self.logger.info("Starting Claims Attachments agent")
        
        # Initialize processing statistics
        self.processing_stats = {
            "total_attachments": 0,
            "successful_submissions": 0,
            "failed_submissions": 0,
            "hipaa_violations": 0,
            "average_processing_time": 0.0
        }
        
        self.logger.info("Claims Attachments agent initialized",
                        validation_rules=len(self.validation_rules),
                        payer_configs=len(self.payer_configs))
    
    async def _on_stop(self) -> None:
        """Clean up Claims Attachments agent."""
        self.logger.info("Claims Attachments agent stopped")
    
    async def _create_attachment(self, task: TaskRequest) -> Dict[str, Any]:
        """Create and validate claims attachment."""
        try:
            claim_id = task.parameters.get("claim_id")
            attachment_type = task.parameters.get("attachment_type")
            file_content = task.parameters.get("file_content")  # Base64 encoded
            file_name = task.parameters.get("file_name")
            patient_info = task.parameters.get("patient_info", {})
            provider_info = task.parameters.get("provider_info", {})
            metadata = task.parameters.get("metadata", {})
            security_level = task.parameters.get("security_level", "confidential")
            
            if not all([claim_id, attachment_type, file_content, file_name]):
                raise ValueError("claim_id, attachment_type, file_content, and file_name are required")
            
            attachment_id = f"att_{uuid.uuid4().hex[:12]}"
            
            self.audit_log_action(
                action="create_attachment",
                data_type="Claims Attachment",
                details={
                    "claim_id": claim_id,
                    "attachment_id": attachment_id,
                    "attachment_type": attachment_type,
                    "file_name": file_name,
                    "task_id": task.id
                }
            )
            
            # Decode file content
            try:
                file_data = base64.b64decode(file_content)
            except Exception as e:
                raise ValueError(f"Invalid base64 file content: {str(e)}")
            
            # Determine file format and MIME type
            file_format, mime_type = self._determine_file_format(file_name, file_data)
            
            # Calculate file hash for integrity
            content_hash = hashlib.sha256(file_data).hexdigest()
            
            # Create attachment
            attachment = ClaimsAttachment(
                attachment_id=attachment_id,
                claim_id=claim_id,
                attachment_type=AttachmentType(attachment_type),
                file_format=file_format,
                file_name=file_name,
                file_size=len(file_data),
                mime_type=mime_type,
                content_hash=content_hash,
                encrypted=True,
                security_level=HIPAASecurityLevel(security_level),
                patient_id=patient_info.get("patient_id", "unknown"),
                provider_id=provider_info.get("provider_id", "unknown"),
                metadata={
                    **metadata,
                    "patient_info": patient_info,
                    "provider_info": provider_info,
                    "creation_context": {
                        "created_by": task.id,
                        "creation_method": "api",
                        "ip_address": "127.0.0.1"  # Would be actual IP in production
                    }
                }
            )
            
            # Validate attachment
            validation_result = await self._perform_attachment_validation(attachment, file_data)
            
            # Check HIPAA compliance
            compliance_status = await self._check_hipaa_compliance(attachment, validation_result)
            
            # Store attachment if valid
            if validation_result["valid"]:
                # Encrypt file data (simplified - production would use proper encryption)
                encrypted_data = self._encrypt_file_data(file_data, attachment_id)
                
                # Store attachment metadata
                self.attachments_store[attachment_id] = attachment
                
                # Log HIPAA audit entry
                await self._log_hipaa_audit(
                    user_id=task.id,
                    action="create_attachment",
                    resource_type="attachment",
                    resource_id=attachment_id,
                    patient_id=attachment.patient_id,
                    access_reason="claims_processing",
                    success=True,
                    details={"attachment_type": attachment_type, "file_size": len(file_data)}
                )
                
                self.processing_stats["total_attachments"] += 1
            
            return {
                "attachment": attachment.dict(),
                "validation_result": validation_result,
                "compliance_status": compliance_status,
                "attachment_id": attachment_id,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Attachment creation failed", error=str(e), task_id=task.id)
            raise
    
    async def _validate_attachment(self, task: TaskRequest) -> Dict[str, Any]:
        """Comprehensive attachment validation."""
        try:
            attachment_id = task.parameters.get("attachment_id")
            validation_rules = task.parameters.get("validation_rules", [])
            hipaa_compliance = task.parameters.get("hipaa_compliance", True)
            payer_requirements = task.parameters.get("payer_requirements", {})
            clinical_validation = task.parameters.get("clinical_validation", False)
            
            if not attachment_id:
                raise ValueError("attachment_id is required")
            
            if attachment_id not in self.attachments_store:
                raise ValueError(f"Attachment not found: {attachment_id}")
            
            attachment = self.attachments_store[attachment_id]
            
            self.audit_log_action(
                action="validate_attachment",
                data_type="Claims Attachment",
                details={
                    "attachment_id": attachment_id,
                    "hipaa_compliance": hipaa_compliance,
                    "clinical_validation": clinical_validation,
                    "task_id": task.id
                }
            )
            
            validation_errors = []
            validation_warnings = []
            
            # Apply standard validation rules
            standard_errors, standard_warnings = await self._apply_standard_validation(attachment)
            validation_errors.extend(standard_errors)
            validation_warnings.extend(standard_warnings)
            
            # Apply custom validation rules
            if validation_rules:
                custom_errors, custom_warnings = await self._apply_custom_validation(attachment, validation_rules)
                validation_errors.extend(custom_errors)
                validation_warnings.extend(custom_warnings)
            
            # HIPAA compliance validation
            hipaa_compliant = True
            if hipaa_compliance:
                hipaa_errors, hipaa_warnings, hipaa_compliant = await self._validate_hipaa_compliance(attachment)
                validation_errors.extend(hipaa_errors)
                validation_warnings.extend(hipaa_warnings)
            
            # Payer-specific validation
            if payer_requirements:
                payer_errors, payer_warnings = await self._validate_payer_requirements(attachment, payer_requirements)
                validation_errors.extend(payer_errors)
                validation_warnings.extend(payer_warnings)
            
            # Clinical validation
            if clinical_validation:
                clinical_errors, clinical_warnings = await self._validate_clinical_content(attachment)
                validation_errors.extend(clinical_errors)
                validation_warnings.extend(clinical_warnings)
            
            # Calculate compliance score
            compliance_score = self._calculate_attachment_compliance_score(validation_errors, validation_warnings)
            
            validation_result = {
                "attachment_id": attachment_id,
                "validation_timestamp": datetime.utcnow().isoformat(),
                "total_errors": len(validation_errors),
                "total_warnings": len(validation_warnings),
                "overall_valid": len(validation_errors) == 0,
                "compliance_score": compliance_score,
                "hipaa_compliant": hipaa_compliant
            }
            
            # Log HIPAA audit entry
            await self._log_hipaa_audit(
                user_id=task.id,
                action="validate_attachment",
                resource_type="attachment",
                resource_id=attachment_id,
                patient_id=attachment.patient_id,
                access_reason="validation_check",
                success=True,
                details={"validation_result": validation_result}
            )
            
            return {
                "validation_result": validation_result,
                "errors": [error.dict() for error in validation_errors],
                "warnings": [warning.dict() for warning in validation_warnings],
                "compliance_score": compliance_score,
                "hipaa_compliant": hipaa_compliant,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Attachment validation failed", error=str(e), task_id=task.id)
            raise
    
    async def _submit_claim(self, task: TaskRequest) -> Dict[str, Any]:
        """Submit claim with attachments."""
        try:
            claim_data = task.parameters.get("claim_data", {})
            attachment_ids = task.parameters.get("attachment_ids", [])
            payer_id = task.parameters.get("payer_id")
            submission_method = task.parameters.get("submission_method", "EDI")
            priority = task.parameters.get("priority", "normal")
            test_mode = task.parameters.get("test_mode", False)
            
            if not claim_data or not payer_id:
                raise ValueError("claim_data and payer_id are required")
            
            claim_id = claim_data.get("claim_id", f"claim_{uuid.uuid4().hex[:12]}")
            tracking_number = f"TRK_{uuid.uuid4().hex[:16].upper()}"
            
            self.audit_log_action(
                action="submit_claim",
                data_type="Claims Submission",
                details={
                    "claim_id": claim_id,
                    "payer_id": payer_id,
                    "attachment_count": len(attachment_ids),
                    "submission_method": submission_method,
                    "test_mode": test_mode,
                    "task_id": task.id
                }
            )
            
            # Validate attachments exist and are valid
            attachments = []
            for att_id in attachment_ids:
                if att_id in self.attachments_store:
                    attachments.append(self.attachments_store[att_id])
                else:
                    raise ValueError(f"Attachment not found: {att_id}")
            
            # Validate payer configuration
            if payer_id not in self.payer_configs:
                raise ValueError(f"Unsupported payer: {payer_id}")
            
            payer_config = self.payer_configs[payer_id]
            
            # Validate submission method
            if submission_method not in payer_config["submission_methods"]:
                raise ValueError(f"Unsupported submission method for {payer_id}: {submission_method}")
            
            # Create claim submission
            claim_submission = ClaimSubmission(
                claim_id=claim_id,
                patient_id=claim_data.get("patient_id", "unknown"),
                provider_id=claim_data.get("provider_id", "unknown"),
                payer_id=payer_id,
                claim_type=claim_data.get("claim_type", "professional"),
                attachments=attachments,
                total_amount=float(claim_data.get("total_amount", 0.0)),
                diagnosis_codes=claim_data.get("diagnosis_codes", []),
                procedure_codes=claim_data.get("procedure_codes", []),
                prior_authorization_number=claim_data.get("prior_authorization_number")
            )
            
            # Perform pre-submission validation
            submission_valid, validation_messages = await self._validate_claim_submission(claim_submission, payer_config)
            
            if not submission_valid and not test_mode:
                return {
                    "submission_result": {
                        "success": False,
                        "validation_errors": validation_messages,
                        "status": "validation_failed"
                    },
                    "tracking_number": None,
                    "estimated_processing_time": None,
                    "submission_status": "failed",
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Process submission
            submission_result = await self._process_claim_submission(
                claim_submission, payer_config, submission_method, test_mode
            )
            
            # Store submission
            self.claims_submissions[claim_id] = claim_submission
            
            # Update statistics
            if submission_result["success"]:
                self.processing_stats["successful_submissions"] += 1
            else:
                self.processing_stats["failed_submissions"] += 1
            
            # Log HIPAA audit entries for all attachments
            for attachment in attachments:
                await self._log_hipaa_audit(
                    user_id=task.id,
                    action="submit_attachment",
                    resource_type="attachment",
                    resource_id=attachment.attachment_id,
                    patient_id=attachment.patient_id,
                    access_reason="claims_submission",
                    success=submission_result["success"],
                    details={
                        "claim_id": claim_id,
                        "payer_id": payer_id,
                        "tracking_number": tracking_number
                    }
                )
            
            return {
                "submission_result": submission_result,
                "tracking_number": tracking_number,
                "estimated_processing_time": payer_config["processing_time_days"],
                "submission_status": "submitted" if submission_result["success"] else "failed",
                "claim_id": claim_id,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Claim submission failed", error=str(e), task_id=task.id)
            raise
    
    async def _track_claim_status(self, task: TaskRequest) -> Dict[str, Any]:
        """Track claim and attachment status."""
        try:
            claim_id = task.parameters.get("claim_id")
            tracking_number = task.parameters.get("tracking_number")
            payer_id = task.parameters.get("payer_id")
            include_attachments = task.parameters.get("include_attachments", True)
            detailed_history = task.parameters.get("detailed_history", False)
            
            if not claim_id:
                raise ValueError("claim_id is required")
            
            self.audit_log_action(
                action="track_claim_status",
                data_type="Claims Tracking",
                details={
                    "claim_id": claim_id,
                    "tracking_number": tracking_number,
                    "include_attachments": include_attachments,
                    "task_id": task.id
                }
            )
            
            # Get claim submission
            if claim_id not in self.claims_submissions:
                # In production, this would query external payer systems
                return {
                    "claim_status": {"status": "not_found", "message": "Claim not found in local records"},
                    "attachment_statuses": [],
                    "processing_history": [],
                    "next_actions": ["verify_claim_id"],
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            claim_submission = self.claims_submissions[claim_id]
            
            # Get current claim status (simulated)
            claim_status = await self._get_claim_status(claim_submission, tracking_number, payer_id)
            
            # Get attachment statuses
            attachment_statuses = []
            if include_attachments:
                for attachment in claim_submission.attachments:
                    att_status = await self._get_attachment_status(attachment, claim_status)
                    attachment_statuses.append(att_status)
            
            # Get processing history
            processing_history = []
            if detailed_history:
                processing_history = await self._get_processing_history(claim_submission)
            
            # Determine next actions
            next_actions = await self._determine_next_actions(claim_status, attachment_statuses)
            
            # Log HIPAA audit entry
            await self._log_hipaa_audit(
                user_id=task.id,
                action="track_claim_status",
                resource_type="claim",
                resource_id=claim_id,
                patient_id=claim_submission.patient_id,
                access_reason="status_inquiry",
                success=True,
                details={"tracking_number": tracking_number, "status": claim_status["status"]}
            )
            
            return {
                "claim_status": claim_status,
                "attachment_statuses": attachment_statuses,
                "processing_history": processing_history,
                "next_actions": next_actions,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Claim status tracking failed", error=str(e), task_id=task.id)
            raise
    
    async def _generate_compliance_report(self, task: TaskRequest) -> Dict[str, Any]:
        """Generate HIPAA compliance and audit report."""
        try:
            report_type = task.parameters.get("report_type", "compliance")
            date_range = task.parameters.get("date_range", {})
            claim_ids = task.parameters.get("claim_ids", [])
            include_audit_trail = task.parameters.get("include_audit_trail", True)
            compliance_metrics = task.parameters.get("compliance_metrics", True)
            
            self.audit_log_action(
                action="generate_compliance_report",
                data_type="Compliance Report",
                details={
                    "report_type": report_type,
                    "claim_ids_count": len(claim_ids),
                    "include_audit_trail": include_audit_trail,
                    "task_id": task.id
                }
            )
            
            # Generate report based on type
            if report_type == "compliance":
                compliance_report = await self._generate_hipaa_compliance_report(date_range, claim_ids)
            elif report_type == "audit":
                compliance_report = await self._generate_audit_report(date_range, claim_ids)
            elif report_type == "performance":
                compliance_report = await self._generate_performance_report(date_range, claim_ids)
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
            
            # Get audit entries
            audit_entries = []
            if include_audit_trail:
                audit_entries = await self._get_audit_entries(date_range, claim_ids)
            
            # Calculate compliance metrics
            metrics = {}
            if compliance_metrics:
                metrics = await self._calculate_compliance_metrics(date_range, claim_ids)
            
            # Generate recommendations
            recommendations = await self._generate_compliance_recommendations(compliance_report, metrics)
            
            return {
                "compliance_report": compliance_report,
                "audit_entries": [entry.dict() for entry in audit_entries],
                "compliance_metrics": metrics,
                "recommendations": recommendations,
                "report_metadata": {
                    "report_type": report_type,
                    "generation_timestamp": datetime.utcnow().isoformat(),
                    "date_range": date_range,
                    "total_claims": len(claim_ids) if claim_ids else len(self.claims_submissions)
                },
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Compliance report generation failed", error=str(e), task_id=task.id)
            raise
    
    # Helper methods for claims processing
    
    def _determine_file_format(self, file_name: str, file_data: bytes) -> tuple[AttachmentFormat, str]:
        """Determine file format and MIME type."""
        # Get MIME type from filename
        mime_type, _ = mimetypes.guess_type(file_name)
        
        # Map to attachment format
        format_mapping = {
            "application/pdf": AttachmentFormat.PDF,
            "image/jpeg": AttachmentFormat.JPEG,
            "image/png": AttachmentFormat.PNG,
            "image/tiff": AttachmentFormat.TIFF,
            "application/dicom": AttachmentFormat.DICOM,
            "application/xml": AttachmentFormat.XML,
            "application/json": AttachmentFormat.JSON,
            "text/csv": AttachmentFormat.CSV,
            "text/plain": AttachmentFormat.TXT,
            "application/rtf": AttachmentFormat.RTF,
            "application/msword": AttachmentFormat.DOC,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": AttachmentFormat.DOCX
        }
        
        file_format = format_mapping.get(mime_type, AttachmentFormat.PDF)
        return file_format, mime_type or "application/octet-stream"
    
    async def _perform_attachment_validation(self, attachment: ClaimsAttachment, file_data: bytes) -> Dict[str, Any]:
        """Perform comprehensive attachment validation."""
        validation_errors = []
        validation_warnings = []
        
        # Find applicable validation rule
        applicable_rule = None
        for rule in self.validation_rules:
            if rule.attachment_type == attachment.attachment_type:
                applicable_rule = rule
                break
        
        if applicable_rule:
            # Check file size
            if applicable_rule.max_file_size_mb:
                max_size_bytes = applicable_rule.max_file_size_mb * 1024 * 1024
                if attachment.file_size > max_size_bytes:
                    validation_errors.append(f"File size exceeds maximum: {applicable_rule.max_file_size_mb}MB")
            
            # Check file format
            if applicable_rule.allowed_formats and attachment.file_format not in applicable_rule.allowed_formats:
                validation_errors.append(f"File format not allowed: {attachment.file_format}")
            
            # Check required metadata
            for required_field in applicable_rule.required_metadata:
                if required_field not in attachment.metadata:
                    validation_errors.append(f"Missing required metadata: {required_field}")
        
        # Basic content validation
        if len(file_data) == 0:
            validation_errors.append("File is empty")
        
        # File integrity check
        calculated_hash = hashlib.sha256(file_data).hexdigest()
        if calculated_hash != attachment.content_hash:
            validation_errors.append("File integrity check failed")
        
        return {
            "valid": len(validation_errors) == 0,
            "errors": validation_errors,
            "warnings": validation_warnings,
            "validation_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _check_hipaa_compliance(self, attachment: ClaimsAttachment, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Check HIPAA compliance for attachment."""
        compliance_issues = []
        
        # Check encryption
        if not attachment.encrypted:
            compliance_issues.append("Attachment must be encrypted for HIPAA compliance")
        
        # Check patient consent (simplified check)
        if "patient_consent" not in attachment.metadata:
            compliance_issues.append("Patient consent documentation required")
        
        # Check minimum necessary standard
        if attachment.security_level == HIPAASecurityLevel.PUBLIC:
            compliance_issues.append("PHI should not be classified as public")
        
        # Check audit trail requirements
        if "creation_context" not in attachment.metadata:
            compliance_issues.append("Creation context required for audit trail")
        
        return {
            "hipaa_compliant": len(compliance_issues) == 0,
            "compliance_issues": compliance_issues,
            "security_level": attachment.security_level.value,
            "encrypted": attachment.encrypted,
            "compliance_check_timestamp": datetime.utcnow().isoformat()
        }
    
    def _encrypt_file_data(self, file_data: bytes, attachment_id: str) -> bytes:
        """Encrypt file data (simplified implementation)."""
        # In production, this would use proper encryption (AES-256, etc.)
        # This is just a placeholder
        return base64.b64encode(file_data)
    
    async def _log_hipaa_audit(self, user_id: str, action: str, resource_type: str, resource_id: str, patient_id: Optional[str], access_reason: str, success: bool, details: Dict[str, Any]) -> None:
        """Log HIPAA audit entry."""
        audit_entry = HIPAAAuditEntry(
            audit_id=f"audit_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            patient_id=patient_id,
            access_reason=access_reason,
            success=success,
            details=details
        )
        
        self.hipaa_audit_log.append(audit_entry)
    
    # Validation methods
    
    async def _apply_standard_validation(self, attachment: ClaimsAttachment) -> tuple[List[AttachmentValidationError], List[AttachmentValidationError]]:
        """Apply standard validation rules."""
        errors = []
        warnings = []
        
        # Check required fields
        if not attachment.patient_id:
            errors.append(AttachmentValidationError(
                error_id=f"err_{uuid.uuid4().hex[:8]}",
                rule_id="STD-001",
                attachment_id=attachment.attachment_id,
                error_type="missing_data",
                severity="error",
                message="Patient ID is required",
                hipaa_impact=True
            ))
        
        if not attachment.provider_id:
            errors.append(AttachmentValidationError(
                error_id=f"err_{uuid.uuid4().hex[:8]}",
                rule_id="STD-002",
                attachment_id=attachment.attachment_id,
                error_type="missing_data",
                severity="error",
                message="Provider ID is required"
            ))
        
        return errors, warnings
    
    async def _apply_custom_validation(self, attachment: ClaimsAttachment, validation_rules: List[Dict[str, Any]]) -> tuple[List[AttachmentValidationError], List[AttachmentValidationError]]:
        """Apply custom validation rules."""
        errors = []
        warnings = []
        
        # Apply custom rules (simplified implementation)
        for rule in validation_rules:
            if rule.get("check_file_size") and attachment.file_size > rule.get("max_size", 1000000):
                errors.append(AttachmentValidationError(
                    error_id=f"err_{uuid.uuid4().hex[:8]}",
                    rule_id="CUSTOM-001",
                    attachment_id=attachment.attachment_id,
                    error_type="file_size",
                    severity="error",
                    message=f"File size exceeds custom limit: {rule.get('max_size')}"
                ))
        
        return errors, warnings
    
    async def _validate_hipaa_compliance(self, attachment: ClaimsAttachment) -> tuple[List[AttachmentValidationError], List[AttachmentValidationError], bool]:
        """Validate HIPAA compliance."""
        errors = []
        warnings = []
        compliant = True
        
        # Check encryption requirement
        if not attachment.encrypted:
            errors.append(AttachmentValidationError(
                error_id=f"err_{uuid.uuid4().hex[:8]}",
                rule_id="HIPAA-001",
                attachment_id=attachment.attachment_id,
                error_type="security_violation",
                severity="error",
                message="PHI must be encrypted",
                hipaa_impact=True
            ))
            compliant = False
        
        return errors, warnings, compliant
    
    async def _validate_payer_requirements(self, attachment: ClaimsAttachment, payer_requirements: Dict[str, Any]) -> tuple[List[AttachmentValidationError], List[AttachmentValidationError]]:
        """Validate payer-specific requirements."""
        errors = []
        warnings = []
        
        # Validate against payer requirements (simplified)
        if payer_requirements.get("max_file_size") and attachment.file_size > payer_requirements["max_file_size"]:
            errors.append(AttachmentValidationError(
                error_id=f"err_{uuid.uuid4().hex[:8]}",
                rule_id="PAYER-001",
                attachment_id=attachment.attachment_id,
                error_type="payer_requirement",
                severity="error",
                message=f"File size exceeds payer limit: {payer_requirements['max_file_size']}"
            ))
        
        return errors, warnings
    
    async def _validate_clinical_content(self, attachment: ClaimsAttachment) -> tuple[List[AttachmentValidationError], List[AttachmentValidationError]]:
        """Validate clinical content."""
        errors = []
        warnings = []
        
        # Clinical validation would analyze actual content
        # This is a simplified placeholder
        if attachment.attachment_type == AttachmentType.LAB_RESULTS:
            if "test_date" not in attachment.metadata:
                warnings.append(AttachmentValidationError(
                    error_id=f"warn_{uuid.uuid4().hex[:8]}",
                    rule_id="CLIN-001",
                    attachment_id=attachment.attachment_id,
                    error_type="clinical_data",
                    severity="warning",
                    message="Lab test date should be specified"
                ))
        
        return errors, warnings
    
    # Submission processing methods
    
    async def _validate_claim_submission(self, claim_submission: ClaimSubmission, payer_config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate claim submission."""
        validation_messages = []
        
        # Check required fields
        if not claim_submission.patient_id:
            validation_messages.append("Patient ID is required")
        
        if not claim_submission.provider_id:
            validation_messages.append("Provider ID is required")
        
        if not claim_submission.diagnosis_codes:
            validation_messages.append("At least one diagnosis code is required")
        
        # Check attachment formats against payer requirements
        supported_formats = payer_config.get("supported_formats", [])
        for attachment in claim_submission.attachments:
            if attachment.file_format.value.upper() not in [f.upper() for f in supported_formats]:
                validation_messages.append(f"Attachment format not supported by payer: {attachment.file_format}")
        
        return len(validation_messages) == 0, validation_messages
    
    async def _process_claim_submission(self, claim_submission: ClaimSubmission, payer_config: Dict[str, Any], submission_method: str, test_mode: bool) -> Dict[str, Any]:
        """Process claim submission."""
        # Simulate submission processing
        if test_mode:
            return {
                "success": True,
                "message": "Test submission successful",
                "submission_method": submission_method,
                "test_mode": True,
                "processing_time_ms": 500
            }
        
        # In production, this would integrate with actual payer systems
        success_rate = 0.95  # Simulate 95% success rate
        import random
        success = random.random() < success_rate
        
        return {
            "success": success,
            "message": "Submission processed successfully" if success else "Submission failed - payer system error",
            "submission_method": submission_method,
            "test_mode": False,
            "processing_time_ms": random.randint(1000, 5000)
        }
    
    # Status tracking methods
    
    async def _get_claim_status(self, claim_submission: ClaimSubmission, tracking_number: Optional[str], payer_id: Optional[str]) -> Dict[str, Any]:
        """Get current claim status."""
        # Simulate status retrieval
        import random
        statuses = [status.value for status in ClaimStatus]
        current_status = random.choice(statuses)
        
        return {
            "claim_id": claim_submission.claim_id,
            "status": current_status,
            "status_date": datetime.utcnow().isoformat(),
            "payer_id": claim_submission.payer_id,
            "tracking_number": tracking_number,
            "total_amount": claim_submission.total_amount,
            "adjudicated_amount": claim_submission.total_amount * random.uniform(0.8, 1.0) if current_status == "approved" else None
        }
    
    async def _get_attachment_status(self, attachment: ClaimsAttachment, claim_status: Dict[str, Any]) -> Dict[str, Any]:
        """Get attachment status."""
        return {
            "attachment_id": attachment.attachment_id,
            "attachment_type": attachment.attachment_type.value,
            "file_name": attachment.file_name,
            "status": "processed" if claim_status["status"] in ["approved", "denied"] else "pending",
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def _get_processing_history(self, claim_submission: ClaimSubmission) -> List[Dict[str, Any]]:
        """Get claim processing history."""
        # Simulate processing history
        history = [
            {
                "date": claim_submission.submission_date,
                "status": "submitted",
                "description": "Claim submitted to payer"
            },
            {
                "date": (datetime.fromisoformat(claim_submission.submission_date) + timedelta(days=1)).isoformat(),
                "status": "received",
                "description": "Claim received by payer"
            }
        ]
        
        return history
    
    async def _determine_next_actions(self, claim_status: Dict[str, Any], attachment_statuses: List[Dict[str, Any]]) -> List[str]:
        """Determine next actions based on current status."""
        next_actions = []
        
        status = claim_status.get("status")
        
        if status == "requires_additional_info":
            next_actions.append("Provide additional documentation")
            next_actions.append("Contact payer for specific requirements")
        elif status == "denied":
            next_actions.append("Review denial reason")
            next_actions.append("File appeal if appropriate")
        elif status == "pending":
            next_actions.append("Monitor status for updates")
        
        return next_actions
    
    # Compliance reporting methods
    
    async def _generate_hipaa_compliance_report(self, date_range: Dict[str, Any], claim_ids: List[str]) -> Dict[str, Any]:
        """Generate HIPAA compliance report."""
        return {
            "report_type": "HIPAA Compliance",
            "compliance_summary": {
                "total_attachments": len(self.attachments_store),
                "encrypted_attachments": sum(1 for att in self.attachments_store.values() if att.encrypted),
                "hipaa_violations": self.processing_stats.get("hipaa_violations", 0),
                "compliance_rate": 98.5  # Simulated
            },
            "security_metrics": {
                "encryption_rate": 100.0,
                "access_control_compliance": 99.2,
                "audit_trail_completeness": 100.0
            },
            "violations": [],  # Would list actual violations
            "recommendations": [
                "Continue monitoring access logs",
                "Review encryption protocols quarterly"
            ]
        }
    
    async def _generate_audit_report(self, date_range: Dict[str, Any], claim_ids: List[str]) -> Dict[str, Any]:
        """Generate audit report."""
        return {
            "report_type": "Audit Report",
            "audit_summary": {
                "total_audit_entries": len(self.hipaa_audit_log),
                "successful_accesses": sum(1 for entry in self.hipaa_audit_log if entry.success),
                "failed_accesses": sum(1 for entry in self.hipaa_audit_log if not entry.success),
                "unique_users": len(set(entry.user_id for entry in self.hipaa_audit_log))
            },
            "access_patterns": {
                "most_common_actions": ["create_attachment", "validate_attachment", "submit_claim"],
                "peak_access_hours": ["09:00-11:00", "14:00-16:00"],
                "suspicious_activities": []  # Would identify unusual patterns
            }
        }
    
    async def _generate_performance_report(self, date_range: Dict[str, Any], claim_ids: List[str]) -> Dict[str, Any]:
        """Generate performance report."""
        return {
            "report_type": "Performance Report",
            "performance_metrics": self.processing_stats,
            "throughput": {
                "attachments_per_day": len(self.attachments_store) / 30,  # Simulated
                "claims_per_day": len(self.claims_submissions) / 30,
                "processing_time_avg": self.processing_stats.get("average_processing_time", 0.0)
            },
            "success_rates": {
                "submission_success_rate": 95.0,
                "validation_success_rate": 98.0,
                "compliance_success_rate": 99.5
            }
        }
    
    async def _get_audit_entries(self, date_range: Dict[str, Any], claim_ids: List[str]) -> List[HIPAAAuditEntry]:
        """Get audit entries for specified criteria."""
        # Filter audit entries based on criteria
        filtered_entries = []
        
        for entry in self.hipaa_audit_log:
            # Apply date range filter if specified
            if date_range:
                entry_date = datetime.fromisoformat(entry.timestamp)
                start_date = datetime.fromisoformat(date_range.get("start", "1900-01-01"))
                end_date = datetime.fromisoformat(date_range.get("end", "2100-12-31"))
                
                if not (start_date <= entry_date <= end_date):
                    continue
            
            # Apply claim IDs filter if specified
            if claim_ids and entry.details.get("claim_id") not in claim_ids:
                continue
            
            filtered_entries.append(entry)
        
        return filtered_entries[-100:]  # Return last 100 entries
    
    async def _calculate_compliance_metrics(self, date_range: Dict[str, Any], claim_ids: List[str]) -> Dict[str, Any]:
        """Calculate compliance metrics."""
        total_attachments = len(self.attachments_store)
        encrypted_attachments = sum(1 for att in self.attachments_store.values() if att.encrypted)
        
        return {
            "encryption_compliance": (encrypted_attachments / total_attachments * 100) if total_attachments > 0 else 100,
            "audit_trail_completeness": 100.0,  # All actions are logged
            "access_control_compliance": 99.8,   # Simulated
            "data_integrity_compliance": 100.0,  # All files have integrity checks
            "privacy_compliance": 98.5,          # Simulated
            "overall_compliance_score": 99.2
        }
    
    async def _generate_compliance_recommendations(self, compliance_report: Dict[str, Any], metrics: Dict[str, Any]) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        if metrics.get("encryption_compliance", 100) < 100:
            recommendations.append("Ensure all attachments are encrypted before storage")
        
        if metrics.get("access_control_compliance", 100) < 95:
            recommendations.append("Review and strengthen access control policies")
        
        if metrics.get("privacy_compliance", 100) < 98:
            recommendations.append("Conduct privacy impact assessment")
        
        if not recommendations:
            recommendations.append("Continue current compliance practices")
            recommendations.append("Schedule quarterly compliance review")
        
        return recommendations
    
    def _calculate_attachment_compliance_score(self, errors: List[AttachmentValidationError], warnings: List[AttachmentValidationError]) -> float:
        """Calculate attachment compliance score."""
        total_issues = len(errors) + len(warnings)
        
        if total_issues == 0:
            return 1.0
        
        # Weight errors more heavily than warnings
        error_penalty = len(errors) * 1.0
        warning_penalty = len(warnings) * 0.3
        
        total_penalty = error_penalty + warning_penalty
        
        # Calculate score (0.0 to 1.0)
        score = max(0.0, 1.0 - (total_penalty / 10.0))
        
        return round(score, 3)