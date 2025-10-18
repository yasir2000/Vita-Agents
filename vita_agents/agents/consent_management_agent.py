"""
Consent Management Agent for Vita Agents.
Provides comprehensive patient consent management and privacy controls.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import structlog
from pydantic import BaseModel, Field
import uuid

from vita_agents.core.agent import HealthcareAgent, AgentCapability, TaskRequest
from vita_agents.core.config import get_settings


logger = structlog.get_logger(__name__)


class ConsentStatus(str, Enum):
    """Consent status enumeration."""
    PROPOSED = "proposed"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ENTERED_IN_ERROR = "entered-in-error"
    DRAFT = "draft"
    REJECTED = "rejected"


class ConsentType(str, Enum):
    """Consent type enumeration."""
    RESEARCH = "research"
    TREATMENT = "treatment"
    PAYMENT = "payment"
    OPERATIONS = "operations"
    MARKETING = "marketing"
    DISCLOSURE = "disclosure"
    ADVANCE_DIRECTIVE = "advance-directive"


class DataCategory(str, Enum):
    """Data category enumeration."""
    CLINICAL = "clinical"
    ADMINISTRATIVE = "administrative"
    FINANCIAL = "financial"
    RESEARCH = "research"
    QUALITY = "quality"
    SAFETY = "safety"
    GENOMIC = "genomic"
    BEHAVIORAL = "behavioral"


class ConsentScope(BaseModel):
    """Consent scope definition."""
    
    data_categories: List[DataCategory]
    resource_types: List[str]
    time_period: Optional[Dict[str, str]] = None
    purposes: List[str]
    recipients: List[str]
    restrictions: Dict[str, Any] = {}


class ConsentAction(str, Enum):
    """Consent action enumeration."""
    ACCESS = "access"
    COLLECT = "collect"
    USE = "use"
    DISCLOSE = "disclose"
    CORRECT = "correct"
    DELETE = "delete"
    OPT_OUT = "opt-out"
    OPT_IN = "opt-in"


class ConsentPolicy(BaseModel):
    """Consent policy definition."""
    
    policy_id: str
    name: str
    description: str
    rule_type: str  # permit, deny, obligation
    actions: List[ConsentAction]
    conditions: Dict[str, Any] = {}
    exceptions: List[Dict[str, Any]] = []


class ConsentDirective(BaseModel):
    """Patient consent directive."""
    
    id: str
    patient_id: str
    status: ConsentStatus
    consent_type: ConsentType
    scope: ConsentScope
    policies: List[ConsentPolicy]
    created_date: datetime
    effective_date: datetime
    expiry_date: Optional[datetime] = None
    source_reference: Optional[str] = None
    custodian: Optional[str] = None
    version: str = "1.0"
    signature: Optional[Dict[str, Any]] = None


class PrivacyPreference(BaseModel):
    """Patient privacy preferences."""
    
    patient_id: str
    communication_preferences: Dict[str, Any] = {}
    marketing_opt_out: bool = False
    research_participation: bool = True
    data_sharing_level: str = "standard"  # minimal, standard, full
    emergency_access: bool = True
    family_access_allowed: List[str] = []
    provider_restrictions: List[str] = []


class ConsentViolation(BaseModel):
    """Consent violation record."""
    
    id: str
    patient_id: str
    violation_type: str
    description: str
    severity: str  # low, medium, high, critical
    detected_date: datetime
    affected_data: List[str]
    remedial_actions: List[str] = []
    status: str = "open"  # open, investigating, resolved


class DataAccessLog(BaseModel):
    """Data access log entry."""
    
    id: str
    patient_id: str
    user_id: str
    resource_type: str
    action: str
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    consent_reference: Optional[str] = None
    purpose: Optional[str] = None
    outcome: str = "success"


class ConsentManagementAgent(HealthcareAgent):
    """
    Consent Management Agent.
    
    Capabilities:
    - Create and manage patient consent directives
    - Validate data access against consent policies
    - Privacy preference management
    - Consent violation detection and reporting
    - Audit trail for consent changes
    - Automated consent enforcement
    - Right to be forgotten implementation
    - Cross-border data transfer compliance
    """
    
    def __init__(
        self,
        agent_id: str = "consent-management-agent",
        name: str = "Consent Management Agent",
        description: str = "Patient consent and privacy management",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None,
    ):
        capabilities = [
            AgentCapability(
                name="create_consent_directive",
                description="Create new patient consent directive",
                input_schema={
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string"},
                        "consent_type": {"type": "string"},
                        "scope": {"type": "object"},
                        "policies": {"type": "array"},
                        "effective_date": {"type": "string"},
                        "expiry_date": {"type": "string"}
                    },
                    "required": ["patient_id", "consent_type", "scope"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "consent_id": {"type": "string"},
                        "status": {"type": "string"},
                        "effective_date": {"type": "string"}
                    }
                }
            ),
            AgentCapability(
                name="validate_data_access",
                description="Validate data access request against consent policies",
                input_schema={
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string"},
                        "requester_id": {"type": "string"},
                        "resource_type": {"type": "string"},
                        "action": {"type": "string"},
                        "purpose": {"type": "string"},
                        "context": {"type": "object"}
                    },
                    "required": ["patient_id", "requester_id", "resource_type", "action"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "access_granted": {"type": "boolean"},
                        "applicable_consents": {"type": "array"},
                        "restrictions": {"type": "object"},
                        "audit_trail": {"type": "object"}
                    }
                }
            ),
            AgentCapability(
                name="manage_privacy_preferences",
                description="Manage patient privacy preferences",
                input_schema={
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string"},
                        "preferences": {"type": "object"},
                        "action": {"type": "string"}
                    },
                    "required": ["patient_id", "action"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "preferences_id": {"type": "string"},
                        "updated_preferences": {"type": "object"},
                        "effective_date": {"type": "string"}
                    }
                }
            ),
            AgentCapability(
                name="detect_consent_violations",
                description="Detect and report consent violations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string"},
                        "access_logs": {"type": "array"},
                        "time_period": {"type": "object"}
                    },
                    "required": ["patient_id"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "violations": {"type": "array"},
                        "violation_count": {"type": "integer"},
                        "recommendations": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="exercise_data_rights",
                description="Exercise patient data rights (access, portability, erasure)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string"},
                        "right_type": {"type": "string"},
                        "scope": {"type": "object"},
                        "identity_verification": {"type": "object"}
                    },
                    "required": ["patient_id", "right_type"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "request_id": {"type": "string"},
                        "status": {"type": "string"},
                        "estimated_completion": {"type": "string"},
                        "data_export": {"type": "object"}
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
        
        # Storage for consent directives and related data
        self.consent_directives: Dict[str, ConsentDirective] = {}
        self.privacy_preferences: Dict[str, PrivacyPreference] = {}
        self.consent_violations: List[ConsentViolation] = []
        self.access_logs: List[DataAccessLog] = []
        
        # Default consent policies
        self.default_policies = self._initialize_default_policies()
        
        # Register task handlers
        self.register_task_handler("create_consent_directive", self._create_consent_directive)
        self.register_task_handler("validate_data_access", self._validate_data_access)
        self.register_task_handler("manage_privacy_preferences", self._manage_privacy_preferences)
        self.register_task_handler("detect_consent_violations", self._detect_consent_violations)
        self.register_task_handler("exercise_data_rights", self._exercise_data_rights)
    
    def _initialize_default_policies(self) -> Dict[str, ConsentPolicy]:
        """Initialize default consent policies."""
        return {
            "treatment_access": ConsentPolicy(
                policy_id="treatment_access",
                name="Treatment Access Policy",
                description="Allow healthcare providers to access patient data for treatment purposes",
                rule_type="permit",
                actions=[ConsentAction.ACCESS, ConsentAction.USE],
                conditions={
                    "purpose": ["treatment"],
                    "recipient_type": ["healthcare-provider"],
                    "data_categories": ["clinical"]
                }
            ),
            "emergency_access": ConsentPolicy(
                policy_id="emergency_access",
                name="Emergency Access Policy",
                description="Allow emergency access to critical patient data",
                rule_type="permit",
                actions=[ConsentAction.ACCESS, ConsentAction.USE],
                conditions={
                    "emergency": True,
                    "data_categories": ["clinical"],
                    "resource_types": ["Patient", "Condition", "AllergyIntolerance", "MedicationStatement"]
                }
            ),
            "research_restriction": ConsentPolicy(
                policy_id="research_restriction",
                name="Research Restriction Policy",
                description="Restrict use of patient data for research without explicit consent",
                rule_type="deny",
                actions=[ConsentAction.USE, ConsentAction.DISCLOSE],
                conditions={
                    "purpose": ["research"],
                    "explicit_consent_required": True
                }
            ),
            "marketing_opt_out": ConsentPolicy(
                policy_id="marketing_opt_out",
                name="Marketing Opt-out Policy",
                description="Deny use of patient data for marketing purposes",
                rule_type="deny",
                actions=[ConsentAction.USE, ConsentAction.DISCLOSE],
                conditions={
                    "purpose": ["marketing", "commercial"]
                }
            )
        }
    
    async def _on_start(self) -> None:
        """Initialize Consent Management agent."""
        self.logger.info("Starting Consent Management agent",
                        default_policies=len(self.default_policies))
        
        # Initialize compliance framework
        self.compliance_frameworks = {
            "HIPAA": {
                "minimum_necessary": True,
                "business_associate_agreements": True,
                "individual_rights": ["access", "amendment", "accounting", "restriction"]
            },
            "GDPR": {
                "lawful_basis_required": True,
                "data_subject_rights": ["access", "rectification", "erasure", "portability", "restriction"],
                "consent_withdrawal": True,
                "data_protection_impact_assessment": True
            },
            "21CFR11": {
                "electronic_signatures": True,
                "audit_trails": True,
                "system_validation": True
            }
        }
        
        self.logger.info("Consent Management agent initialized")
    
    async def _on_stop(self) -> None:
        """Clean up Consent Management agent."""
        self.logger.info("Consent Management agent stopped")
    
    async def _create_consent_directive(self, task: TaskRequest) -> Dict[str, Any]:
        """Create new patient consent directive."""
        try:
            patient_id = task.parameters.get("patient_id")
            consent_type = task.parameters.get("consent_type")
            scope_data = task.parameters.get("scope")
            policies_data = task.parameters.get("policies", [])
            effective_date = task.parameters.get("effective_date")
            expiry_date = task.parameters.get("expiry_date")
            
            if not patient_id or not consent_type or not scope_data:
                raise ValueError("patient_id, consent_type, and scope are required")
            
            # Generate consent ID
            consent_id = f"consent_{uuid.uuid4().hex}"
            
            # Parse effective date
            if effective_date:
                effective_dt = datetime.fromisoformat(effective_date.replace("Z", "+00:00"))
            else:
                effective_dt = datetime.utcnow()
            
            # Parse expiry date
            expiry_dt = None
            if expiry_date:
                expiry_dt = datetime.fromisoformat(expiry_date.replace("Z", "+00:00"))
            
            # Create consent scope
            scope = ConsentScope(
                data_categories=[DataCategory(cat) for cat in scope_data.get("data_categories", [])],
                resource_types=scope_data.get("resource_types", []),
                time_period=scope_data.get("time_period"),
                purposes=scope_data.get("purposes", []),
                recipients=scope_data.get("recipients", []),
                restrictions=scope_data.get("restrictions", {})
            )
            
            # Create consent policies
            policies = []
            for policy_data in policies_data:
                policy = ConsentPolicy(
                    policy_id=policy_data.get("policy_id", f"policy_{uuid.uuid4().hex}"),
                    name=policy_data.get("name", "Custom Policy"),
                    description=policy_data.get("description", ""),
                    rule_type=policy_data.get("rule_type", "permit"),
                    actions=[ConsentAction(action) for action in policy_data.get("actions", [])],
                    conditions=policy_data.get("conditions", {}),
                    exceptions=policy_data.get("exceptions", [])
                )
                policies.append(policy)
            
            # Add default policies if none specified
            if not policies:
                policies = [
                    self.default_policies["treatment_access"],
                    self.default_policies["emergency_access"]
                ]
            
            # Create consent directive
            consent_directive = ConsentDirective(
                id=consent_id,
                patient_id=patient_id,
                status=ConsentStatus.ACTIVE,
                consent_type=ConsentType(consent_type),
                scope=scope,
                policies=policies,
                created_date=datetime.utcnow(),
                effective_date=effective_dt,
                expiry_date=expiry_dt,
                version="1.0"
            )
            
            # Store consent directive
            self.consent_directives[consent_id] = consent_directive
            
            # Log consent creation
            await self._log_consent_event(
                "consent_created",
                patient_id=patient_id,
                consent_id=consent_id,
                details={
                    "consent_type": consent_type,
                    "scope": scope_data,
                    "policies_count": len(policies)
                }
            )
            
            self.audit_log_action(
                action="create_consent_directive",
                data_type="Consent",
                details={
                    "patient_id": patient_id,
                    "consent_id": consent_id,
                    "consent_type": consent_type,
                    "task_id": task.id
                }
            )
            
            return {
                "consent_id": consent_id,
                "status": consent_directive.status.value,
                "effective_date": consent_directive.effective_date.isoformat(),
                "expiry_date": consent_directive.expiry_date.isoformat() if consent_directive.expiry_date else None,
                "policies_count": len(policies),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Consent directive creation failed", error=str(e), task_id=task.id)
            raise
    
    async def _validate_data_access(self, task: TaskRequest) -> Dict[str, Any]:
        """Validate data access request against consent policies."""
        try:
            patient_id = task.parameters.get("patient_id")
            requester_id = task.parameters.get("requester_id")
            resource_type = task.parameters.get("resource_type")
            action = task.parameters.get("action")
            purpose = task.parameters.get("purpose", "treatment")
            context = task.parameters.get("context", {})
            
            if not patient_id or not requester_id or not resource_type or not action:
                raise ValueError("patient_id, requester_id, resource_type, and action are required")
            
            # Find applicable consent directives
            applicable_consents = []
            for consent in self.consent_directives.values():
                if (consent.patient_id == patient_id and 
                    consent.status == ConsentStatus.ACTIVE and
                    self._is_consent_effective(consent)):
                    applicable_consents.append(consent)
            
            # Evaluate consent policies
            access_decision = self._evaluate_consent_policies(
                applicable_consents,
                requester_id,
                resource_type,
                action,
                purpose,
                context
            )
            
            # Create access log entry
            access_log = DataAccessLog(
                id=f"access_{uuid.uuid4().hex}",
                patient_id=patient_id,
                user_id=requester_id,
                resource_type=resource_type,
                action=action,
                timestamp=datetime.utcnow(),
                purpose=purpose,
                outcome="granted" if access_decision["access_granted"] else "denied",
                consent_reference=",".join([c.id for c in applicable_consents])
            )
            
            self.access_logs.append(access_log)
            
            # Log access validation
            await self._log_consent_event(
                "access_validation",
                patient_id=patient_id,
                details={
                    "requester_id": requester_id,
                    "resource_type": resource_type,
                    "action": action,
                    "purpose": purpose,
                    "access_granted": access_decision["access_granted"],
                    "applicable_consents": len(applicable_consents)
                }
            )
            
            self.audit_log_action(
                action="validate_data_access",
                data_type="Access Control",
                details={
                    "patient_id": patient_id,
                    "requester_id": requester_id,
                    "resource_type": resource_type,
                    "access_granted": access_decision["access_granted"],
                    "task_id": task.id
                }
            )
            
            return {
                **access_decision,
                "applicable_consents": [c.id for c in applicable_consents],
                "audit_trail": {
                    "access_log_id": access_log.id,
                    "timestamp": access_log.timestamp.isoformat()
                },
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Data access validation failed", error=str(e), task_id=task.id)
            raise
    
    async def _manage_privacy_preferences(self, task: TaskRequest) -> Dict[str, Any]:
        """Manage patient privacy preferences."""
        try:
            patient_id = task.parameters.get("patient_id")
            preferences_data = task.parameters.get("preferences", {})
            action = task.parameters.get("action")  # create, update, delete, query
            
            if not patient_id or not action:
                raise ValueError("patient_id and action are required")
            
            if action == "create" or action == "update":
                # Create or update privacy preferences
                if patient_id in self.privacy_preferences:
                    preferences = self.privacy_preferences[patient_id]
                else:
                    preferences = PrivacyPreference(patient_id=patient_id)
                
                # Update preferences
                for key, value in preferences_data.items():
                    if hasattr(preferences, key):
                        setattr(preferences, key, value)
                
                self.privacy_preferences[patient_id] = preferences
                preferences_id = f"prefs_{patient_id}"
                
            elif action == "delete":
                if patient_id in self.privacy_preferences:
                    del self.privacy_preferences[patient_id]
                preferences_id = None
                preferences = None
                
            elif action == "query":
                preferences = self.privacy_preferences.get(patient_id)
                preferences_id = f"prefs_{patient_id}" if preferences else None
                
            else:
                raise ValueError(f"Unknown action: {action}")
            
            # Log privacy preference change
            await self._log_consent_event(
                f"privacy_preferences_{action}",
                patient_id=patient_id,
                details={
                    "preferences_updated": list(preferences_data.keys()) if preferences_data else []
                }
            )
            
            self.audit_log_action(
                action="manage_privacy_preferences",
                data_type="Privacy Preferences",
                details={
                    "patient_id": patient_id,
                    "action": action,
                    "task_id": task.id
                }
            )
            
            return {
                "preferences_id": preferences_id,
                "updated_preferences": preferences.dict() if preferences else None,
                "effective_date": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Privacy preferences management failed", error=str(e), task_id=task.id)
            raise
    
    async def _detect_consent_violations(self, task: TaskRequest) -> Dict[str, Any]:
        """Detect and report consent violations."""
        try:
            patient_id = task.parameters.get("patient_id")
            access_logs_data = task.parameters.get("access_logs", [])
            time_period = task.parameters.get("time_period", {})
            
            if not patient_id:
                raise ValueError("patient_id is required")
            
            # Get access logs for patient
            if access_logs_data:
                # Use provided access logs
                access_logs = [DataAccessLog(**log_data) for log_data in access_logs_data]
            else:
                # Use stored access logs
                access_logs = [log for log in self.access_logs if log.patient_id == patient_id]
            
            # Filter by time period
            if time_period:
                start_date = datetime.fromisoformat(time_period.get("start", "1900-01-01"))
                end_date = datetime.fromisoformat(time_period.get("end", "2100-01-01"))
                access_logs = [log for log in access_logs 
                             if start_date <= log.timestamp <= end_date]
            
            # Detect violations
            violations = []
            
            # Check for unauthorized access
            unauthorized_access = [log for log in access_logs if log.outcome == "denied"]
            for log in unauthorized_access:
                violation = ConsentViolation(
                    id=f"violation_{uuid.uuid4().hex}",
                    patient_id=patient_id,
                    violation_type="unauthorized_access",
                    description=f"Unauthorized access attempt by {log.user_id} to {log.resource_type}",
                    severity="medium",
                    detected_date=log.timestamp,
                    affected_data=[log.resource_type]
                )
                violations.append(violation)
            
            # Check for purpose violations
            for log in access_logs:
                if log.outcome == "granted" and log.consent_reference:
                    consent_ids = log.consent_reference.split(",")
                    consent_violations = self._check_purpose_compliance(log, consent_ids)
                    violations.extend(consent_violations)
            
            # Check for data retention violations
            retention_violations = self._check_data_retention_compliance(patient_id, access_logs)
            violations.extend(retention_violations)
            
            # Store violations
            self.consent_violations.extend(violations)
            
            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(violations)
            
            # Log violation detection
            await self._log_consent_event(
                "violation_detection",
                patient_id=patient_id,
                details={
                    "violations_detected": len(violations),
                    "access_logs_reviewed": len(access_logs),
                    "time_period": time_period
                }
            )
            
            self.audit_log_action(
                action="detect_consent_violations",
                data_type="Compliance",
                details={
                    "patient_id": patient_id,
                    "violations_count": len(violations),
                    "task_id": task.id
                }
            )
            
            return {
                "violations": [v.dict() for v in violations],
                "violation_count": len(violations),
                "recommendations": recommendations,
                "access_logs_reviewed": len(access_logs),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Consent violation detection failed", error=str(e), task_id=task.id)
            raise
    
    async def _exercise_data_rights(self, task: TaskRequest) -> Dict[str, Any]:
        """Exercise patient data rights (access, portability, erasure)."""
        try:
            patient_id = task.parameters.get("patient_id")
            right_type = task.parameters.get("right_type")
            scope = task.parameters.get("scope", {})
            identity_verification = task.parameters.get("identity_verification", {})
            
            if not patient_id or not right_type:
                raise ValueError("patient_id and right_type are required")
            
            # Generate request ID
            request_id = f"right_{uuid.uuid4().hex}"
            
            # Verify identity (simplified)
            if not self._verify_identity(patient_id, identity_verification):
                raise ValueError("Identity verification failed")
            
            result = {}
            
            if right_type == "access":
                # Right to access personal data
                result = await self._handle_data_access_right(patient_id, scope)
                
            elif right_type == "portability":
                # Right to data portability
                result = await self._handle_data_portability_right(patient_id, scope)
                
            elif right_type == "erasure":
                # Right to be forgotten
                result = await self._handle_data_erasure_right(patient_id, scope)
                
            elif right_type == "rectification":
                # Right to rectification
                result = await self._handle_data_rectification_right(patient_id, scope)
                
            elif right_type == "restriction":
                # Right to restriction of processing
                result = await self._handle_data_restriction_right(patient_id, scope)
                
            else:
                raise ValueError(f"Unknown right type: {right_type}")
            
            # Log data rights exercise
            await self._log_consent_event(
                f"data_right_{right_type}",
                patient_id=patient_id,
                details={
                    "request_id": request_id,
                    "scope": scope,
                    "identity_verified": True
                }
            )
            
            self.audit_log_action(
                action="exercise_data_rights",
                data_type="Data Rights",
                details={
                    "patient_id": patient_id,
                    "right_type": right_type,
                    "request_id": request_id,
                    "task_id": task.id
                }
            )
            
            return {
                "request_id": request_id,
                "status": "completed",
                "estimated_completion": datetime.utcnow().isoformat(),
                **result,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Data rights exercise failed", error=str(e), task_id=task.id)
            raise
    
    def _is_consent_effective(self, consent: ConsentDirective) -> bool:
        """Check if consent is currently effective."""
        now = datetime.utcnow()
        
        if consent.effective_date > now:
            return False
        
        if consent.expiry_date and consent.expiry_date < now:
            return False
        
        return True
    
    def _evaluate_consent_policies(
        self,
        consents: List[ConsentDirective],
        requester_id: str,
        resource_type: str,
        action: str,
        purpose: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate consent policies for access decision."""
        
        access_granted = False
        applicable_policies = []
        restrictions = {}
        
        # Evaluate each consent directive
        for consent in consents:
            for policy in consent.policies:
                if self._policy_applies(policy, resource_type, action, purpose, context):
                    applicable_policies.append(policy)
                    
                    if policy.rule_type == "permit":
                        access_granted = True
                    elif policy.rule_type == "deny":
                        access_granted = False
                        break
                    elif policy.rule_type == "obligation":
                        restrictions.update(policy.conditions)
        
        # Apply default deny if no permit policy found
        if not any(p.rule_type == "permit" for p in applicable_policies):
            access_granted = False
        
        return {
            "access_granted": access_granted,
            "applicable_policies": [p.policy_id for p in applicable_policies],
            "restrictions": restrictions
        }
    
    def _policy_applies(
        self,
        policy: ConsentPolicy,
        resource_type: str,
        action: str,
        purpose: str,
        context: Dict[str, Any]
    ) -> bool:
        """Check if a policy applies to the access request."""
        
        # Check action
        if ConsentAction(action) not in policy.actions and ConsentAction.ACCESS not in policy.actions:
            return False
        
        # Check purpose
        if "purpose" in policy.conditions:
            if purpose not in policy.conditions["purpose"]:
                return False
        
        # Check resource type
        if "resource_types" in policy.conditions:
            if resource_type not in policy.conditions["resource_types"]:
                return False
        
        # Check emergency context
        if policy.conditions.get("emergency") and not context.get("emergency"):
            return False
        
        return True
    
    def _check_purpose_compliance(self, access_log: DataAccessLog, consent_ids: List[str]) -> List[ConsentViolation]:
        """Check if access complies with consent purpose."""
        violations = []
        
        for consent_id in consent_ids:
            if consent_id in self.consent_directives:
                consent = self.consent_directives[consent_id]
                
                # Check if purpose is allowed
                allowed_purposes = consent.scope.purposes
                if access_log.purpose and access_log.purpose not in allowed_purposes:
                    violation = ConsentViolation(
                        id=f"violation_{uuid.uuid4().hex}",
                        patient_id=access_log.patient_id,
                        violation_type="purpose_violation",
                        description=f"Data accessed for unauthorized purpose: {access_log.purpose}",
                        severity="high",
                        detected_date=access_log.timestamp,
                        affected_data=[access_log.resource_type]
                    )
                    violations.append(violation)
        
        return violations
    
    def _check_data_retention_compliance(self, patient_id: str, access_logs: List[DataAccessLog]) -> List[ConsentViolation]:
        """Check data retention compliance."""
        violations = []
        
        # Find consent directives with retention periods
        patient_consents = [c for c in self.consent_directives.values() if c.patient_id == patient_id]
        
        for consent in patient_consents:
            if consent.scope.time_period and "retention_period" in consent.scope.time_period:
                retention_days = int(consent.scope.time_period["retention_period"])
                cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                
                # Check for access to expired data
                expired_access = [log for log in access_logs if log.timestamp < cutoff_date]
                
                for log in expired_access:
                    violation = ConsentViolation(
                        id=f"violation_{uuid.uuid4().hex}",
                        patient_id=patient_id,
                        violation_type="retention_violation",
                        description=f"Access to data beyond retention period: {log.resource_type}",
                        severity="medium",
                        detected_date=log.timestamp,
                        affected_data=[log.resource_type]
                    )
                    violations.append(violation)
        
        return violations
    
    def _generate_compliance_recommendations(self, violations: List[ConsentViolation]) -> List[str]:
        """Generate compliance recommendations based on violations."""
        recommendations = []
        
        violation_types = set(v.violation_type for v in violations)
        
        if "unauthorized_access" in violation_types:
            recommendations.append("Review access controls and user permissions")
            recommendations.append("Implement additional authentication measures")
        
        if "purpose_violation" in violation_types:
            recommendations.append("Provide additional training on data use policies")
            recommendations.append("Implement purpose-based access controls")
        
        if "retention_violation" in violation_types:
            recommendations.append("Implement automated data retention policies")
            recommendations.append("Regular data purging based on retention schedules")
        
        # High severity violations
        high_severity = [v for v in violations if v.severity == "high"]
        if high_severity:
            recommendations.append("Immediate review of high-severity violations required")
            recommendations.append("Consider notifying affected patients")
        
        return recommendations
    
    def _verify_identity(self, patient_id: str, verification: Dict[str, Any]) -> bool:
        """Verify patient identity for data rights exercise."""
        # Simplified identity verification
        # In production, implement proper identity verification
        
        required_fields = ["date_of_birth", "last_four_ssn"]
        
        for field in required_fields:
            if field not in verification:
                return False
        
        # Additional verification logic would go here
        return True
    
    async def _handle_data_access_right(self, patient_id: str, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Handle right to access personal data."""
        
        # Collect patient data (simplified)
        patient_data = {
            "personal_information": {"patient_id": patient_id},
            "consent_directives": [c.dict() for c in self.consent_directives.values() if c.patient_id == patient_id],
            "privacy_preferences": self.privacy_preferences.get(patient_id, {}).dict() if patient_id in self.privacy_preferences else {},
            "access_logs": [log.dict() for log in self.access_logs if log.patient_id == patient_id]
        }
        
        # Apply scope filters
        if scope.get("data_types"):
            filtered_data = {}
            for data_type in scope["data_types"]:
                if data_type in patient_data:
                    filtered_data[data_type] = patient_data[data_type]
            patient_data = filtered_data
        
        return {
            "data_export": patient_data,
            "export_format": "json",
            "records_count": sum(len(v) if isinstance(v, list) else 1 for v in patient_data.values())
        }
    
    async def _handle_data_portability_right(self, patient_id: str, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Handle right to data portability."""
        
        # Get data in portable format
        access_result = await self._handle_data_access_right(patient_id, scope)
        
        # Convert to portable formats
        portable_formats = ["json", "xml", "csv"]
        
        return {
            "data_export": access_result["data_export"],
            "available_formats": portable_formats,
            "default_format": "json",
            "portable": True
        }
    
    async def _handle_data_erasure_right(self, patient_id: str, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Handle right to be forgotten (data erasure)."""
        
        erased_items = []
        
        # Remove consent directives
        if not scope.get("preserve_consents"):
            patient_consents = [cid for cid, c in self.consent_directives.items() if c.patient_id == patient_id]
            for consent_id in patient_consents:
                del self.consent_directives[consent_id]
                erased_items.append(f"consent_{consent_id}")
        
        # Remove privacy preferences
        if patient_id in self.privacy_preferences:
            del self.privacy_preferences[patient_id]
            erased_items.append("privacy_preferences")
        
        # Anonymize access logs (can't delete for audit purposes)
        for log in self.access_logs:
            if log.patient_id == patient_id:
                log.patient_id = "anonymized"
                erased_items.append(f"access_log_{log.id}")
        
        return {
            "erasure_completed": True,
            "items_erased": erased_items,
            "erasure_count": len(erased_items)
        }
    
    async def _handle_data_rectification_right(self, patient_id: str, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Handle right to rectification."""
        
        corrections = scope.get("corrections", {})
        corrected_items = []
        
        # Update consent directives if specified
        if "consent_updates" in corrections:
            for consent_id, updates in corrections["consent_updates"].items():
                if consent_id in self.consent_directives and self.consent_directives[consent_id].patient_id == patient_id:
                    consent = self.consent_directives[consent_id]
                    for field, value in updates.items():
                        if hasattr(consent, field):
                            setattr(consent, field, value)
                            corrected_items.append(f"consent_{consent_id}.{field}")
        
        # Update privacy preferences if specified
        if "privacy_updates" in corrections and patient_id in self.privacy_preferences:
            preferences = self.privacy_preferences[patient_id]
            for field, value in corrections["privacy_updates"].items():
                if hasattr(preferences, field):
                    setattr(preferences, field, value)
                    corrected_items.append(f"privacy_preferences.{field}")
        
        return {
            "rectification_completed": True,
            "items_corrected": corrected_items,
            "correction_count": len(corrected_items)
        }
    
    async def _handle_data_restriction_right(self, patient_id: str, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Handle right to restriction of processing."""
        
        restrictions = scope.get("restrictions", {})
        restricted_items = []
        
        # Add restriction policies to consents
        patient_consents = [c for c in self.consent_directives.values() if c.patient_id == patient_id]
        
        for consent in patient_consents:
            restriction_policy = ConsentPolicy(
                policy_id=f"restriction_{uuid.uuid4().hex}",
                name="Data Processing Restriction",
                description="Patient-requested restriction on data processing",
                rule_type="deny",
                actions=[ConsentAction.USE, ConsentAction.DISCLOSE],
                conditions=restrictions
            )
            
            consent.policies.append(restriction_policy)
            restricted_items.append(f"consent_{consent.id}")
        
        return {
            "restriction_applied": True,
            "restricted_items": restricted_items,
            "restriction_count": len(restricted_items)
        }
    
    async def _log_consent_event(self, event_type: str, patient_id: str = None, **kwargs) -> None:
        """Log consent-related events."""
        self.logger.info(
            f"Consent event: {event_type}",
            patient_id=patient_id,
            **kwargs
        )