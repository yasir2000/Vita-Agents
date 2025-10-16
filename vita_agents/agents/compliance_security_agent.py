"""
Compliance & Security Agent for HIPAA compliance, patient consent, and security monitoring.
"""

import asyncio
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set
from enum import Enum
from dataclasses import dataclass, asdict
import structlog
from pydantic import BaseModel, Field

from vita_agents.core.agent import BaseAgent, TaskRequest, TaskResponse, MessageType
from vita_agents.core.security import (
    EncryptionManager, 
    AuditLogger, 
    ComplianceValidator,
    AuditEvent, 
    AuditAction, 
    ComplianceLevel
)


logger = structlog.get_logger(__name__)


class ConsentType(str, Enum):
    """Types of patient consent."""
    TREATMENT = "treatment"
    RESEARCH = "research"
    DATA_SHARING = "data_sharing"
    MARKETING = "marketing"
    THIRD_PARTY_DISCLOSURE = "third_party_disclosure"


class ConsentStatus(str, Enum):
    """Status of patient consent."""
    GRANTED = "granted"
    DENIED = "denied"
    EXPIRED = "expired"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"


class SecurityIncidentType(str, Enum):
    """Types of security incidents."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    MALICIOUS_ACTIVITY = "malicious_activity"
    POLICY_VIOLATION = "policy_violation"
    TECHNICAL_VULNERABILITY = "technical_vulnerability"


class SecurityIncidentSeverity(str, Enum):
    """Severity levels for security incidents."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PatientConsent:
    """Patient consent record."""
    consent_id: str
    patient_id: str
    consent_type: ConsentType
    status: ConsentStatus
    granted_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    withdrawn_date: Optional[datetime] = None
    purpose: str = ""
    scope: List[str] = None
    grantor_id: str = ""  # Who granted the consent
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.scope is None:
            self.scope = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SecurityIncident:
    """Security incident record."""
    incident_id: str
    incident_type: SecurityIncidentType
    severity: SecurityIncidentSeverity
    detected_at: datetime
    description: str
    affected_resources: List[str] = None
    affected_patients: List[str] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    source_system: Optional[str] = None
    mitigation_actions: List[str] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.affected_resources is None:
            self.affected_resources = []
        if self.affected_patients is None:
            self.affected_patients = []
        if self.mitigation_actions is None:
            self.mitigation_actions = []
        if self.metadata is None:
            self.metadata = {}


class ComplianceReport(BaseModel):
    """Compliance assessment report."""
    report_id: str
    generated_at: datetime
    assessment_period: Dict[str, datetime]
    compliance_score: float
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    audit_summary: Dict[str, Any]
    consent_summary: Dict[str, Any]
    security_summary: Dict[str, Any]


class PrivacyImpactAssessment(BaseModel):
    """Privacy impact assessment for data processing activities."""
    assessment_id: str
    activity_description: str
    data_types: List[str]
    processing_purpose: str
    legal_basis: str
    privacy_risks: List[Dict[str, Any]]
    mitigation_measures: List[str]
    impact_score: float
    approved: bool
    assessor_id: str
    assessment_date: datetime


class ComplianceSecurityAgent(BaseAgent):
    """
    Agent responsible for HIPAA compliance, patient consent management,
    and security monitoring.
    """
    
    def __init__(
        self,
        agent_id: str,
        settings,
        database,
        encryption_manager: Optional[EncryptionManager] = None,
        audit_logger: Optional[AuditLogger] = None,
        compliance_validator: Optional[ComplianceValidator] = None
    ):
        super().__init__(agent_id, "compliance_security")
        self.settings = settings
        self.database = database
        self.encryption_manager = encryption_manager or EncryptionManager(settings)
        self.audit_logger = audit_logger or AuditLogger(settings, database)
        self.compliance_validator = compliance_validator or ComplianceValidator(settings)
        
        # Task handlers
        self.task_handlers = {
            "validate_phi_access": self._validate_phi_access,
            "check_consent": self._check_consent,
            "grant_consent": self._grant_consent,
            "withdraw_consent": self._withdraw_consent,
            "detect_security_incident": self._detect_security_incident,
            "investigate_incident": self._investigate_incident,
            "generate_compliance_report": self._generate_compliance_report,
            "assess_privacy_impact": self._assess_privacy_impact,
            "audit_data_access": self._audit_data_access,
            "encrypt_sensitive_data": self._encrypt_sensitive_data,
            "validate_data_retention": self._validate_data_retention,
            "check_minimum_necessary": self._check_minimum_necessary,
        }
        
        logger.info("Compliance & Security Agent initialized", agent_id=agent_id)
    
    async def process_task(self, task: TaskRequest) -> TaskResponse:
        """Process compliance and security tasks."""
        try:
            if task.task_type not in self.task_handlers:
                return TaskResponse(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status="failed",
                    error=f"Unknown task type: {task.task_type}",
                    result={}
                )
            
            handler = self.task_handlers[task.task_type]
            result = await handler(task.parameters)
            
            return TaskResponse(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status="completed",
                result=result
            )
            
        except Exception as e:
            logger.error("Task processing failed", error=str(e), task_id=task.task_id)
            return TaskResponse(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status="failed",
                error=str(e),
                result={}
            )
    
    async def _validate_phi_access(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate PHI access request against HIPAA requirements."""
        user_id = parameters.get("user_id")
        patient_id = parameters.get("patient_id")
        access_reason = parameters.get("access_reason")
        requested_fields = parameters.get("requested_fields", [])
        user_role = parameters.get("user_role")
        user_permissions = parameters.get("user_permissions", [])
        
        # Validate minimum necessary access
        allowed_fields = self.compliance_validator.validate_minimum_necessary(
            requested_fields, user_role
        )
        
        # Check user permissions
        has_phi_access = self.compliance_validator.validate_phi_access(
            user_permissions, "Patient", AuditAction.READ
        )
        
        # Check if consent exists
        consent_valid = await self._check_patient_consent(
            patient_id, user_id, "treatment"
        )
        
        # Log access attempt
        await self.audit_logger.log_access(AuditEvent(
            action=AuditAction.READ,
            resource_type="Patient",
            resource_id=patient_id,
            user_id=user_id,
            patient_id=patient_id,
            agent_id=self.agent_id,
            access_reason=access_reason,
            compliance_level=ComplianceLevel.RESTRICTED,
            timestamp=datetime.utcnow(),
            details={
                "requested_fields": requested_fields,
                "allowed_fields": allowed_fields,
                "user_role": user_role
            },
            success=has_phi_access and consent_valid
        ))
        
        return {
            "access_granted": has_phi_access and consent_valid,
            "allowed_fields": allowed_fields,
            "consent_valid": consent_valid,
            "compliance_notes": self._generate_compliance_notes(
                has_phi_access, consent_valid, user_role
            )
        }
    
    async def _check_consent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check patient consent status."""
        patient_id = parameters.get("patient_id")
        consent_type = parameters.get("consent_type", ConsentType.TREATMENT)
        purpose = parameters.get("purpose", "")
        
        consent_valid = await self._check_patient_consent(
            patient_id, None, consent_type, purpose
        )
        
        # Get consent details
        consent_records = await self._get_consent_records(patient_id, consent_type)
        
        return {
            "consent_valid": consent_valid,
            "consent_records": [asdict(record) for record in consent_records],
            "compliance_status": "compliant" if consent_valid else "non_compliant"
        }
    
    async def _grant_consent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Grant patient consent."""
        patient_id = parameters.get("patient_id")
        consent_type = ConsentType(parameters.get("consent_type"))
        purpose = parameters.get("purpose", "")
        scope = parameters.get("scope", [])
        grantor_id = parameters.get("grantor_id")
        expiry_date = parameters.get("expiry_date")
        
        if expiry_date and isinstance(expiry_date, str):
            expiry_date = datetime.fromisoformat(expiry_date)
        
        consent = PatientConsent(
            consent_id=str(uuid.uuid4()),
            patient_id=patient_id,
            consent_type=consent_type,
            status=ConsentStatus.GRANTED,
            granted_date=datetime.utcnow(),
            expiry_date=expiry_date,
            purpose=purpose,
            scope=scope,
            grantor_id=grantor_id
        )
        
        # Store consent record
        await self._store_consent_record(consent)
        
        # Log consent grant
        await self.audit_logger.log_consent_action(
            patient_id, "consent_granted", consent_type.value, grantor_id
        )
        
        return {
            "consent_id": consent.consent_id,
            "status": "granted",
            "granted_date": consent.granted_date.isoformat(),
            "expiry_date": consent.expiry_date.isoformat() if consent.expiry_date else None
        }
    
    async def _withdraw_consent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Withdraw patient consent."""
        patient_id = parameters.get("patient_id")
        consent_type = ConsentType(parameters.get("consent_type"))
        grantor_id = parameters.get("grantor_id")
        
        # Update consent status
        withdrawn_count = await self._update_consent_status(
            patient_id, consent_type, ConsentStatus.WITHDRAWN
        )
        
        # Log consent withdrawal
        await self.audit_logger.log_consent_action(
            patient_id, "consent_withdrawn", consent_type.value, grantor_id
        )
        
        return {
            "status": "withdrawn",
            "withdrawn_date": datetime.utcnow().isoformat(),
            "records_updated": withdrawn_count
        }
    
    async def _detect_security_incident(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and classify security incidents."""
        event_data = parameters.get("event_data", {})
        user_id = event_data.get("user_id")
        ip_address = event_data.get("ip_address")
        resource_accessed = event_data.get("resource_accessed")
        access_pattern = event_data.get("access_pattern", {})
        
        # Analyze for suspicious patterns
        incidents = []
        
        # Check for unusual access patterns
        if await self._is_unusual_access_pattern(user_id, access_pattern):
            incident = SecurityIncident(
                incident_id=str(uuid.uuid4()),
                incident_type=SecurityIncidentType.UNAUTHORIZED_ACCESS,
                severity=SecurityIncidentSeverity.MEDIUM,
                detected_at=datetime.utcnow(),
                description="Unusual access pattern detected",
                user_id=user_id,
                ip_address=ip_address,
                affected_resources=[resource_accessed] if resource_accessed else [],
                metadata=access_pattern
            )
            incidents.append(incident)
        
        # Check for policy violations
        if await self._check_policy_violations(event_data):
            incident = SecurityIncident(
                incident_id=str(uuid.uuid4()),
                incident_type=SecurityIncidentType.POLICY_VIOLATION,
                severity=SecurityIncidentSeverity.HIGH,
                detected_at=datetime.utcnow(),
                description="HIPAA policy violation detected",
                user_id=user_id,
                metadata=event_data
            )
            incidents.append(incident)
        
        # Store incidents
        for incident in incidents:
            await self._store_security_incident(incident)
        
        return {
            "incidents_detected": len(incidents),
            "incidents": [asdict(incident) for incident in incidents],
            "requires_investigation": len(incidents) > 0
        }
    
    async def _investigate_incident(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Investigate security incident."""
        incident_id = parameters.get("incident_id")
        investigator_id = parameters.get("investigator_id")
        
        # Get incident details
        incident = await self._get_security_incident(incident_id)
        if not incident:
            return {"error": "Incident not found"}
        
        # Perform investigation steps
        investigation_results = {
            "incident_id": incident_id,
            "investigator_id": investigator_id,
            "investigation_date": datetime.utcnow().isoformat(),
            "findings": [],
            "recommendations": [],
            "risk_assessment": "medium"
        }
        
        # Analyze related events
        related_events = await self._find_related_security_events(incident)
        investigation_results["related_events"] = len(related_events)
        
        # Generate findings
        if incident.incident_type == SecurityIncidentType.UNAUTHORIZED_ACCESS:
            investigation_results["findings"].append(
                "Unauthorized access attempt detected from unusual IP range"
            )
            investigation_results["recommendations"].append(
                "Review and update access controls for affected user"
            )
        
        # Update incident with investigation results
        await self._update_incident_investigation(incident_id, investigation_results)
        
        return investigation_results
    
    async def _generate_compliance_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate HIPAA compliance report."""
        start_date = datetime.fromisoformat(parameters.get("start_date"))
        end_date = datetime.fromisoformat(parameters.get("end_date"))
        
        # Gather compliance metrics
        audit_summary = await self._get_audit_summary(start_date, end_date)
        consent_summary = await self._get_consent_summary(start_date, end_date)
        security_summary = await self._get_security_summary(start_date, end_date)
        violations = await self._get_compliance_violations(start_date, end_date)
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(
            audit_summary, consent_summary, security_summary, violations
        )
        
        report = ComplianceReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.utcnow(),
            assessment_period={"start": start_date, "end": end_date},
            compliance_score=compliance_score,
            violations=violations,
            recommendations=self._generate_compliance_recommendations(violations),
            audit_summary=audit_summary,
            consent_summary=consent_summary,
            security_summary=security_summary
        )
        
        return report.dict()
    
    async def _assess_privacy_impact(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Assess privacy impact of data processing activity."""
        activity_description = parameters.get("activity_description")
        data_types = parameters.get("data_types", [])
        processing_purpose = parameters.get("processing_purpose")
        assessor_id = parameters.get("assessor_id")
        
        # Identify privacy risks
        privacy_risks = self._identify_privacy_risks(data_types, processing_purpose)
        
        # Calculate impact score
        impact_score = self._calculate_privacy_impact_score(privacy_risks)
        
        # Generate mitigation measures
        mitigation_measures = self._generate_mitigation_measures(privacy_risks)
        
        assessment = PrivacyImpactAssessment(
            assessment_id=str(uuid.uuid4()),
            activity_description=activity_description,
            data_types=data_types,
            processing_purpose=processing_purpose,
            legal_basis="healthcare_treatment",  # Default for healthcare
            privacy_risks=privacy_risks,
            mitigation_measures=mitigation_measures,
            impact_score=impact_score,
            approved=impact_score < 0.7,  # Auto-approve low-risk activities
            assessor_id=assessor_id,
            assessment_date=datetime.utcnow()
        )
        
        return assessment.dict()
    
    async def _audit_data_access(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Audit data access for compliance."""
        user_id = parameters.get("user_id")
        resource_type = parameters.get("resource_type")
        resource_id = parameters.get("resource_id")
        action = AuditAction(parameters.get("action", "read"))
        
        # Create audit event
        audit_event = AuditEvent(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            patient_id=parameters.get("patient_id"),
            agent_id=self.agent_id,
            access_reason=parameters.get("access_reason", ""),
            compliance_level=ComplianceLevel(parameters.get("compliance_level", "restricted")),
            timestamp=datetime.utcnow(),
            details=parameters.get("details", {}),
            success=parameters.get("success", True)
        )
        
        # Log to audit trail
        await self.audit_logger.log_access(audit_event)
        
        return {
            "audited": True,
            "audit_id": str(uuid.uuid4()),
            "timestamp": audit_event.timestamp.isoformat()
        }
    
    async def _encrypt_sensitive_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive healthcare data."""
        data = parameters.get("data")
        compliance_level = ComplianceLevel(parameters.get("compliance_level", "restricted"))
        
        encrypted_data = self.encryption_manager.encrypt_sensitive_data(
            data, compliance_level
        )
        
        return {
            "encrypted": True,
            "encrypted_data": encrypted_data,
            "compliance_level": compliance_level.value
        }
    
    async def _validate_data_retention(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data retention policy compliance."""
        data_date = datetime.fromisoformat(parameters.get("data_date"))
        data_type = parameters.get("data_type", "medical_record")
        
        compliant = self.compliance_validator.check_data_retention_policy(data_date)
        
        if not compliant:
            days_overdue = (datetime.utcnow() - data_date).days - (7 * 365)  # 7 year retention
            return {
                "compliant": False,
                "action_required": "data_purge",
                "days_overdue": days_overdue
            }
        
        return {
            "compliant": True,
            "action_required": None
        }
    
    async def _check_minimum_necessary(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check minimum necessary access principle."""
        requested_fields = parameters.get("requested_fields", [])
        user_role = parameters.get("user_role")
        access_purpose = parameters.get("access_purpose")
        
        allowed_fields = self.compliance_validator.validate_minimum_necessary(
            requested_fields, user_role
        )
        
        excessive_fields = set(requested_fields) - set(allowed_fields)
        
        return {
            "compliant": len(excessive_fields) == 0,
            "allowed_fields": allowed_fields,
            "denied_fields": list(excessive_fields),
            "compliance_notes": f"Minimum necessary principle applied for {user_role}"
        }
    
    # Helper methods
    async def _check_patient_consent(
        self, 
        patient_id: str, 
        user_id: Optional[str], 
        consent_type: Union[str, ConsentType],
        purpose: str = ""
    ) -> bool:
        """Check if patient consent exists and is valid."""
        # This would query the database for consent records
        # For now, return a mock implementation
        return True  # Assume consent exists
    
    async def _get_consent_records(
        self, 
        patient_id: str, 
        consent_type: ConsentType
    ) -> List[PatientConsent]:
        """Get consent records for a patient."""
        # Mock implementation - would query database
        return []
    
    async def _store_consent_record(self, consent: PatientConsent) -> None:
        """Store consent record in database."""
        # Mock implementation - would store in database
        pass
    
    async def _update_consent_status(
        self, 
        patient_id: str, 
        consent_type: ConsentType, 
        status: ConsentStatus
    ) -> int:
        """Update consent status and return number of records updated."""
        # Mock implementation - would update database
        return 1
    
    async def _is_unusual_access_pattern(
        self, 
        user_id: str, 
        access_pattern: Dict[str, Any]
    ) -> bool:
        """Check if access pattern is unusual."""
        # Mock implementation - would analyze historical patterns
        return False
    
    async def _check_policy_violations(self, event_data: Dict[str, Any]) -> bool:
        """Check for HIPAA policy violations."""
        # Mock implementation - would check against policy rules
        return False
    
    async def _store_security_incident(self, incident: SecurityIncident) -> None:
        """Store security incident in database."""
        # Mock implementation - would store in database
        pass
    
    async def _get_security_incident(self, incident_id: str) -> Optional[SecurityIncident]:
        """Get security incident by ID."""
        # Mock implementation - would query database
        return None
    
    async def _find_related_security_events(
        self, 
        incident: SecurityIncident
    ) -> List[Dict[str, Any]]:
        """Find related security events."""
        # Mock implementation - would search for related events
        return []
    
    async def _update_incident_investigation(
        self, 
        incident_id: str, 
        investigation_results: Dict[str, Any]
    ) -> None:
        """Update incident with investigation results."""
        # Mock implementation - would update database
        pass
    
    async def _get_audit_summary(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get audit summary for date range."""
        return {
            "total_accesses": 1000,
            "successful_accesses": 995,
            "failed_accesses": 5,
            "unique_users": 50
        }
    
    async def _get_consent_summary(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get consent summary for date range."""
        return {
            "total_consents": 500,
            "active_consents": 480,
            "expired_consents": 15,
            "withdrawn_consents": 5
        }
    
    async def _get_security_summary(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get security summary for date range."""
        return {
            "total_incidents": 3,
            "resolved_incidents": 2,
            "open_incidents": 1,
            "critical_incidents": 0
        }
    
    async def _get_compliance_violations(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get compliance violations for date range."""
        return []
    
    def _calculate_compliance_score(
        self, 
        audit_summary: Dict[str, Any],
        consent_summary: Dict[str, Any],
        security_summary: Dict[str, Any],
        violations: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall compliance score."""
        base_score = 1.0
        
        # Deduct for failed accesses
        if audit_summary["total_accesses"] > 0:
            failure_rate = audit_summary["failed_accesses"] / audit_summary["total_accesses"]
            base_score -= failure_rate * 0.2
        
        # Deduct for open security incidents
        if security_summary["total_incidents"] > 0:
            open_rate = security_summary["open_incidents"] / security_summary["total_incidents"]
            base_score -= open_rate * 0.3
        
        # Deduct for violations
        base_score -= len(violations) * 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _generate_compliance_recommendations(
        self, 
        violations: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        if len(violations) > 0:
            recommendations.append("Review and address compliance violations immediately")
            recommendations.append("Conduct additional staff training on HIPAA requirements")
        
        recommendations.extend([
            "Regularly review access logs for unusual patterns",
            "Ensure patient consent forms are up to date",
            "Monitor security incidents and response times"
        ])
        
        return recommendations
    
    def _identify_privacy_risks(
        self, 
        data_types: List[str], 
        processing_purpose: str
    ) -> List[Dict[str, Any]]:
        """Identify privacy risks for data processing."""
        risks = []
        
        high_risk_data = ["ssn", "genetic_data", "mental_health"]
        for data_type in data_types:
            if data_type in high_risk_data:
                risks.append({
                    "type": "high_sensitivity_data",
                    "description": f"Processing of {data_type} carries high privacy risk",
                    "severity": "high",
                    "data_element": data_type
                })
        
        return risks
    
    def _calculate_privacy_impact_score(self, privacy_risks: List[Dict[str, Any]]) -> float:
        """Calculate privacy impact score."""
        if not privacy_risks:
            return 0.1
        
        total_score = 0.0
        for risk in privacy_risks:
            if risk["severity"] == "high":
                total_score += 0.3
            elif risk["severity"] == "medium":
                total_score += 0.2
            else:
                total_score += 0.1
        
        return min(1.0, total_score)
    
    def _generate_mitigation_measures(
        self, 
        privacy_risks: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate privacy risk mitigation measures."""
        measures = [
            "Implement data minimization principles",
            "Use encryption for data at rest and in transit",
            "Conduct regular privacy training",
            "Implement access controls and audit logging"
        ]
        
        for risk in privacy_risks:
            if risk["severity"] == "high":
                measures.append(f"Additional controls for {risk['data_element']}")
        
        return measures
    
    def _generate_compliance_notes(
        self, 
        has_phi_access: bool, 
        consent_valid: bool, 
        user_role: str
    ) -> List[str]:
        """Generate compliance notes for access decision."""
        notes = []
        
        if not has_phi_access:
            notes.append("User lacks required PHI access permissions")
        
        if not consent_valid:
            notes.append("Valid patient consent not found")
        
        notes.append(f"Minimum necessary principle applied for {user_role}")
        
        return notes