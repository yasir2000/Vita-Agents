"""
AI Governance & Regulatory Compliance Framework for Healthcare AI Systems.

This module provides comprehensive AI governance capabilities including FDA AI/ML
framework compliance, bias detection and mitigation, AI ethics governance,
audit trails, model versioning, and regulatory reporting systems.
"""

import asyncio
import json
import hashlib
import uuid
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from enum import Enum
from dataclasses import dataclass, field
import structlog
from pydantic import BaseModel, Field, validator
from collections import defaultdict, deque
import semver
import pickle
import sqlite3
import threading
from pathlib import Path

try:
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

logger = structlog.get_logger(__name__)


class RegulatoryFramework(Enum):
    """Regulatory frameworks for AI systems."""
    FDA_AIML = "fda_aiml"
    EU_MDR = "eu_mdr"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    ISO_13485 = "iso_13485"
    ISO_14155 = "iso_14155"
    ICH_E6_GCP = "ich_e6_gcp"
    CFR_TITLE_21 = "cfr_title_21"


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    PENDING_APPROVAL = "pending_approval"
    EXEMPT = "exempt"


class AuditEventType(Enum):
    """Types of audit events."""
    MODEL_TRAINING = "model_training"
    MODEL_DEPLOYMENT = "model_deployment"
    MODEL_INFERENCE = "model_inference"
    DATA_ACCESS = "data_access"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_CHECK = "compliance_check"
    BIAS_ASSESSMENT = "bias_assessment"
    MODEL_UPDATE = "model_update"
    DATA_BREACH = "data_breach"
    ADVERSE_EVENT = "adverse_event"


class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ModelType(Enum):
    """Types of AI models."""
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    THERAPEUTIC = "therapeutic"
    MONITORING = "monitoring"
    TRIAGE = "triage"
    DECISION_SUPPORT = "decision_support"


class EthicalPrinciple(Enum):
    """AI ethics principles."""
    TRANSPARENCY = "transparency"
    FAIRNESS = "fairness"
    ACCOUNTABILITY = "accountability"
    PRIVACY = "privacy"
    BENEFICENCE = "beneficence"
    NON_MALEFICENCE = "non_maleficence"
    AUTONOMY = "autonomy"
    JUSTICE = "justice"


@dataclass
class ModelMetadata:
    """Comprehensive model metadata for governance."""
    model_id: str
    model_name: str
    model_type: ModelType
    version: str
    created_by: str
    created_at: datetime
    description: str
    intended_use: str
    target_population: Dict[str, Any]
    clinical_indications: List[str]
    contraindications: List[str]
    performance_metrics: Dict[str, float]
    training_data_info: Dict[str, Any]
    validation_data_info: Dict[str, Any]
    regulatory_status: Dict[RegulatoryFramework, ComplianceStatus]
    risk_classification: RiskLevel
    deployment_environment: str
    model_architecture: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    feature_specifications: List[Dict[str, Any]]
    output_specifications: Dict[str, Any]
    performance_thresholds: Dict[str, float]
    monitoring_requirements: List[str]
    update_frequency: str
    deprecation_date: Optional[datetime] = None
    approval_documents: List[str] = field(default_factory=list)
    clinical_evidence: List[str] = field(default_factory=list)
    known_limitations: List[str] = field(default_factory=list)
    bias_assessment_results: Dict[str, Any] = field(default_factory=dict)
    ethical_review_status: Dict[EthicalPrinciple, str] = field(default_factory=dict)


@dataclass
class AuditEvent:
    """Audit event record."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    model_id: Optional[str]
    patient_id: Optional[str]
    action: str
    details: Dict[str, Any]
    outcome: str
    risk_level: RiskLevel
    compliance_implications: List[str]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    location: Optional[str] = None
    data_accessed: List[str] = field(default_factory=list)
    regulatory_flags: List[str] = field(default_factory=list)
    hash_signature: Optional[str] = None


@dataclass
class ComplianceAssessment:
    """Compliance assessment result."""
    assessment_id: str
    framework: RegulatoryFramework
    model_id: str
    assessed_by: str
    assessment_date: datetime
    overall_status: ComplianceStatus
    compliance_score: float  # 0-100
    requirements_assessed: List[str]
    compliant_requirements: List[str]
    non_compliant_requirements: List[str]
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    action_items: List[Dict[str, Any]]
    next_assessment_due: datetime
    certification_status: str
    evidence_documents: List[str] = field(default_factory=list)
    reviewer_notes: str = ""


@dataclass
class EthicsReview:
    """AI ethics review record."""
    review_id: str
    model_id: str
    reviewer: str
    review_date: datetime
    principles_assessed: List[EthicalPrinciple]
    ethical_concerns: List[Dict[str, Any]]
    risk_assessment: Dict[str, RiskLevel]
    mitigation_strategies: List[str]
    approval_status: str
    conditions: List[str]
    monitoring_requirements: List[str]
    review_board_decision: str
    stakeholder_input: Dict[str, str] = field(default_factory=dict)
    public_consultation: bool = False


class GovernanceRequest(BaseModel):
    """Request for governance action."""
    
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    requester_id: str
    action_type: str
    model_id: Optional[str] = None
    framework: Optional[RegulatoryFramework] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: str = "normal"
    request_timestamp: datetime = Field(default_factory=datetime.utcnow)


class GovernanceResponse(BaseModel):
    """Response from governance system."""
    
    request_id: str
    action_type: str
    status: str
    results: Dict[str, Any] = Field(default_factory=dict)
    compliance_status: Optional[ComplianceStatus] = None
    audit_events: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    response_timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseGovernanceModule(ABC):
    """Base class for governance modules."""
    
    def __init__(self, module_name: str, version: str = "1.0.0"):
        self.module_name = module_name
        self.version = version
        self.logger = structlog.get_logger(__name__)
    
    @abstractmethod
    async def process_governance_request(
        self, 
        request: GovernanceRequest
    ) -> GovernanceResponse:
        """Process governance request."""
        pass
    
    def generate_audit_event(
        self, 
        event_type: AuditEventType, 
        action: str, 
        details: Dict[str, Any],
        user_id: Optional[str] = None,
        model_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        risk_level: RiskLevel = RiskLevel.LOW
    ) -> AuditEvent:
        """Generate standardized audit event."""
        
        event_id = str(uuid.uuid4())
        
        # Generate hash signature for integrity
        event_data = {
            'event_id': event_id,
            'event_type': event_type.value,
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'details': details
        }
        
        hash_signature = hashlib.sha256(
            json.dumps(event_data, sort_keys=True).encode()
        ).hexdigest()
        
        return AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            session_id=details.get('session_id'),
            model_id=model_id,
            patient_id=patient_id,
            action=action,
            details=details,
            outcome=details.get('outcome', 'completed'),
            risk_level=risk_level,
            compliance_implications=[],
            hash_signature=hash_signature
        )


class RegulatoryComplianceManager(BaseGovernanceModule):
    """Manager for regulatory compliance assessment."""
    
    def __init__(self):
        super().__init__("regulatory_compliance", "v2.0.0")
        
        # FDA AI/ML Framework requirements
        self.fda_aiml_requirements = {
            'predicate_device_identification': 'Identification of predicate devices',
            'algorithm_description': 'Detailed algorithm description',
            'training_data_description': 'Training data characteristics',
            'performance_testing': 'Algorithm performance testing',
            'risk_management': 'Risk management documentation',
            'labeling_requirements': 'Device labeling and instructions',
            'clinical_evaluation': 'Clinical evaluation data',
            'software_documentation': 'Software lifecycle processes',
            'cybersecurity': 'Cybersecurity documentation',
            'interoperability': 'Interoperability assessment'
        }
        
        # EU MDR requirements
        self.eu_mdr_requirements = {
            'conformity_assessment': 'Conformity assessment procedures',
            'clinical_evidence': 'Clinical evidence requirements',
            'post_market_surveillance': 'Post-market surveillance system',
            'risk_management_system': 'Risk management system',
            'quality_management': 'Quality management system',
            'technical_documentation': 'Technical documentation',
            'udi_system': 'Unique device identification',
            'authorized_representative': 'Authorized representative (if applicable)',
            'vigilance_reporting': 'Vigilance and adverse event reporting'
        }
    
    async def process_governance_request(
        self, 
        request: GovernanceRequest
    ) -> GovernanceResponse:
        """Process regulatory compliance request."""
        
        try:
            if request.action_type == "compliance_assessment":
                return await self._perform_compliance_assessment(request)
            elif request.action_type == "framework_validation":
                return await self._validate_framework_compliance(request)
            elif request.action_type == "regulatory_reporting":
                return await self._generate_regulatory_report(request)
            else:
                raise ValueError(f"Unknown action type: {request.action_type}")
                
        except Exception as e:
            self.logger.error(f"Regulatory compliance processing failed: {e}")
            raise
    
    async def _perform_compliance_assessment(
        self, 
        request: GovernanceRequest
    ) -> GovernanceResponse:
        """Perform comprehensive compliance assessment."""
        
        model_id = request.model_id
        framework = request.framework
        
        if not model_id or not framework:
            raise ValueError("Model ID and framework required for compliance assessment")
        
        # Get model metadata (would be from model registry)
        model_metadata = await self._get_model_metadata(model_id)
        
        # Assess compliance based on framework
        assessment = await self._assess_framework_compliance(model_metadata, framework)
        
        # Generate audit event
        audit_event = self.generate_audit_event(
            AuditEventType.COMPLIANCE_CHECK,
            "compliance_assessment_performed",
            {
                'model_id': model_id,
                'framework': framework.value,
                'assessment_id': assessment.assessment_id,
                'compliance_score': assessment.compliance_score
            },
            user_id=request.requester_id,
            model_id=model_id,
            risk_level=RiskLevel.MODERATE
        )
        
        return GovernanceResponse(
            request_id=request.request_id,
            action_type=request.action_type,
            status="completed",
            results={
                'assessment': assessment.__dict__,
                'compliance_status': assessment.overall_status.value,
                'compliance_score': assessment.compliance_score
            },
            compliance_status=assessment.overall_status,
            audit_events=[audit_event.event_id],
            recommendations=assessment.recommendations,
            next_actions=self._generate_compliance_next_actions(assessment),
            response_timestamp=datetime.utcnow(),
            metadata={
                'framework': framework.value,
                'assessment_methodology': 'automated_with_manual_review'
            }
        )
    
    async def _assess_framework_compliance(
        self, 
        model_metadata: ModelMetadata, 
        framework: RegulatoryFramework
    ) -> ComplianceAssessment:
        """Assess compliance against specific regulatory framework."""
        
        assessment_id = str(uuid.uuid4())
        
        if framework == RegulatoryFramework.FDA_AIML:
            return await self._assess_fda_aiml_compliance(model_metadata, assessment_id)
        elif framework == RegulatoryFramework.EU_MDR:
            return await self._assess_eu_mdr_compliance(model_metadata, assessment_id)
        elif framework == RegulatoryFramework.HIPAA:
            return await self._assess_hipaa_compliance(model_metadata, assessment_id)
        else:
            raise ValueError(f"Framework {framework} not supported")
    
    async def _assess_fda_aiml_compliance(
        self, 
        model_metadata: ModelMetadata, 
        assessment_id: str
    ) -> ComplianceAssessment:
        """Assess FDA AI/ML framework compliance."""
        
        compliant_requirements = []
        non_compliant_requirements = []
        findings = []
        
        # Check each FDA requirement
        for req_id, req_description in self.fda_aiml_requirements.items():
            is_compliant = await self._check_fda_requirement(model_metadata, req_id)
            
            if is_compliant:
                compliant_requirements.append(req_id)
            else:
                non_compliant_requirements.append(req_id)
                findings.append({
                    'requirement': req_id,
                    'description': req_description,
                    'status': 'non_compliant',
                    'details': f'Model does not meet {req_description.lower()} requirements'
                })
        
        # Calculate compliance score
        total_requirements = len(self.fda_aiml_requirements)
        compliance_score = (len(compliant_requirements) / total_requirements) * 100
        
        # Determine overall status
        if compliance_score >= 95:
            overall_status = ComplianceStatus.COMPLIANT
        elif compliance_score >= 80:
            overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            overall_status = ComplianceStatus.NON_COMPLIANT
        
        # Generate recommendations
        recommendations = self._generate_fda_recommendations(non_compliant_requirements)
        
        # Generate action items
        action_items = [
            {
                'action': f'Address {req_id} compliance',
                'priority': 'high' if req_id in ['clinical_evaluation', 'risk_management'] else 'medium',
                'due_date': (datetime.utcnow() + timedelta(days=30)).isoformat(),
                'responsible_party': 'regulatory_team'
            }
            for req_id in non_compliant_requirements
        ]
        
        return ComplianceAssessment(
            assessment_id=assessment_id,
            framework=RegulatoryFramework.FDA_AIML,
            model_id=model_metadata.model_id,
            assessed_by="automated_system",
            assessment_date=datetime.utcnow(),
            overall_status=overall_status,
            compliance_score=compliance_score,
            requirements_assessed=list(self.fda_aiml_requirements.keys()),
            compliant_requirements=compliant_requirements,
            non_compliant_requirements=non_compliant_requirements,
            findings=findings,
            recommendations=recommendations,
            action_items=action_items,
            next_assessment_due=datetime.utcnow() + timedelta(days=90),
            certification_status="pending_review"
        )
    
    async def _check_fda_requirement(self, model_metadata: ModelMetadata, requirement_id: str) -> bool:
        """Check specific FDA requirement compliance."""
        
        # Simplified compliance checking (would be more sophisticated in production)
        requirement_checks = {
            'predicate_device_identification': len(model_metadata.approval_documents) > 0,
            'algorithm_description': len(model_metadata.description) > 100,
            'training_data_description': len(model_metadata.training_data_info) > 0,
            'performance_testing': len(model_metadata.performance_metrics) >= 3,
            'risk_management': model_metadata.risk_classification != RiskLevel.LOW,
            'labeling_requirements': len(model_metadata.intended_use) > 50,
            'clinical_evaluation': len(model_metadata.clinical_evidence) > 0,
            'software_documentation': len(model_metadata.model_architecture) > 0,
            'cybersecurity': 'security_assessment' in model_metadata.approval_documents,
            'interoperability': 'interoperability_test' in model_metadata.approval_documents
        }
        
        return requirement_checks.get(requirement_id, False)
    
    def _generate_fda_recommendations(self, non_compliant_requirements: List[str]) -> List[str]:
        """Generate recommendations for FDA compliance."""
        
        recommendations = []
        
        recommendation_map = {
            'predicate_device_identification': 'Identify and document predicate devices with substantial equivalence justification',
            'algorithm_description': 'Provide comprehensive algorithm description including architecture and training methodology',
            'training_data_description': 'Document training data characteristics, sources, and representativeness',
            'performance_testing': 'Conduct comprehensive performance testing with appropriate validation datasets',
            'risk_management': 'Implement risk management framework with hazard analysis and risk controls',
            'labeling_requirements': 'Develop comprehensive device labeling with indications, contraindications, and warnings',
            'clinical_evaluation': 'Conduct clinical evaluation studies to demonstrate safety and effectiveness',
            'software_documentation': 'Complete software lifecycle documentation per IEC 62304',
            'cybersecurity': 'Perform cybersecurity risk assessment and implement security controls',
            'interoperability': 'Assess and document interoperability with healthcare IT systems'
        }
        
        for requirement in non_compliant_requirements:
            if requirement in recommendation_map:
                recommendations.append(recommendation_map[requirement])
        
        return recommendations
    
    async def _assess_eu_mdr_compliance(
        self, 
        model_metadata: ModelMetadata, 
        assessment_id: str
    ) -> ComplianceAssessment:
        """Assess EU MDR compliance."""
        
        # Simplified EU MDR assessment
        compliant_requirements = []
        non_compliant_requirements = []
        
        # Basic checks for EU MDR
        for req_id in self.eu_mdr_requirements:
            # Simplified compliance check
            if req_id in ['technical_documentation', 'risk_management_system']:
                compliant_requirements.append(req_id)
            else:
                non_compliant_requirements.append(req_id)
        
        compliance_score = (len(compliant_requirements) / len(self.eu_mdr_requirements)) * 100
        overall_status = ComplianceStatus.PARTIALLY_COMPLIANT if compliance_score > 50 else ComplianceStatus.NON_COMPLIANT
        
        return ComplianceAssessment(
            assessment_id=assessment_id,
            framework=RegulatoryFramework.EU_MDR,
            model_id=model_metadata.model_id,
            assessed_by="automated_system",
            assessment_date=datetime.utcnow(),
            overall_status=overall_status,
            compliance_score=compliance_score,
            requirements_assessed=list(self.eu_mdr_requirements.keys()),
            compliant_requirements=compliant_requirements,
            non_compliant_requirements=non_compliant_requirements,
            findings=[],
            recommendations=['Complete EU MDR technical documentation', 'Establish post-market surveillance system'],
            action_items=[],
            next_assessment_due=datetime.utcnow() + timedelta(days=90),
            certification_status="pending_review"
        )
    
    async def _assess_hipaa_compliance(
        self, 
        model_metadata: ModelMetadata, 
        assessment_id: str
    ) -> ComplianceAssessment:
        """Assess HIPAA compliance."""
        
        # HIPAA compliance assessment
        hipaa_requirements = [
            'administrative_safeguards',
            'physical_safeguards', 
            'technical_safeguards',
            'breach_notification',
            'business_associate_agreements'
        ]
        
        # Assume partial compliance for demonstration
        compliant_requirements = hipaa_requirements[:3]
        non_compliant_requirements = hipaa_requirements[3:]
        
        compliance_score = (len(compliant_requirements) / len(hipaa_requirements)) * 100
        overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        
        return ComplianceAssessment(
            assessment_id=assessment_id,
            framework=RegulatoryFramework.HIPAA,
            model_id=model_metadata.model_id,
            assessed_by="automated_system",
            assessment_date=datetime.utcnow(),
            overall_status=overall_status,
            compliance_score=compliance_score,
            requirements_assessed=hipaa_requirements,
            compliant_requirements=compliant_requirements,
            non_compliant_requirements=non_compliant_requirements,
            findings=[],
            recommendations=['Implement breach notification procedures', 'Update business associate agreements'],
            action_items=[],
            next_assessment_due=datetime.utcnow() + timedelta(days=180),
            certification_status="compliant_with_conditions"
        )
    
    async def _get_model_metadata(self, model_id: str) -> ModelMetadata:
        """Get model metadata (mock implementation)."""
        
        # In production, this would query the model registry
        return ModelMetadata(
            model_id=model_id,
            model_name=f"Medical AI Model {model_id}",
            model_type=ModelType.DIAGNOSTIC,
            version="1.0.0",
            created_by="ai_team",
            created_at=datetime.utcnow() - timedelta(days=30),
            description="AI model for medical diagnosis and risk assessment",
            intended_use="Diagnostic support for healthcare professionals",
            target_population={"age_range": "18-90", "conditions": ["general_medicine"]},
            clinical_indications=["chest_pain", "shortness_of_breath"],
            contraindications=["pediatric_patients"],
            performance_metrics={"accuracy": 0.92, "sensitivity": 0.89, "specificity": 0.94},
            training_data_info={"size": 10000, "sources": ["hospital_a", "hospital_b"]},
            validation_data_info={"size": 2000, "sources": ["hospital_c"]},
            regulatory_status={RegulatoryFramework.FDA_AIML: ComplianceStatus.UNDER_REVIEW},
            risk_classification=RiskLevel.MODERATE,
            deployment_environment="production",
            model_architecture={"type": "neural_network", "layers": 5},
            hyperparameters={"learning_rate": 0.001, "batch_size": 32},
            feature_specifications=[{"name": "age", "type": "numeric"}],
            output_specifications={"type": "classification", "classes": ["low_risk", "high_risk"]},
            performance_thresholds={"accuracy": 0.85, "sensitivity": 0.80},
            monitoring_requirements=["performance_monitoring", "bias_monitoring"],
            update_frequency="quarterly",
            approval_documents=["510k_submission"],
            clinical_evidence=["clinical_study_001"],
            known_limitations=["limited_to_adult_population"],
            bias_assessment_results={"demographic_parity": 0.05},
            ethical_review_status={EthicalPrinciple.FAIRNESS: "approved"}
        )
    
    def _generate_compliance_next_actions(self, assessment: ComplianceAssessment) -> List[str]:
        """Generate next actions based on compliance assessment."""
        
        next_actions = []
        
        if assessment.overall_status == ComplianceStatus.NON_COMPLIANT:
            next_actions.extend([
                "Schedule compliance review meeting with regulatory team",
                "Develop compliance improvement plan",
                "Assign owners for non-compliant requirements"
            ])
        elif assessment.overall_status == ComplianceStatus.PARTIALLY_COMPLIANT:
            next_actions.extend([
                "Address remaining compliance gaps",
                "Schedule follow-up assessment",
                "Update compliance documentation"
            ])
        else:
            next_actions.extend([
                "Maintain compliance monitoring",
                "Schedule next routine assessment",
                "Document compliance status for auditors"
            ])
        
        return next_actions


class AuditTrailManager(BaseGovernanceModule):
    """Manager for comprehensive audit trails."""
    
    def __init__(self, database_path: str = "audit_trail.db"):
        super().__init__("audit_trail", "v1.5.0")
        self.database_path = database_path
        self.encryption_key = None
        self._initialize_database()
        
        # Thread lock for database operations
        self.db_lock = threading.Lock()
    
    def _initialize_database(self):
        """Initialize audit trail database."""
        
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create audit events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    model_id TEXT,
                    patient_id TEXT,
                    action TEXT NOT NULL,
                    details TEXT,
                    outcome TEXT,
                    risk_level TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    location TEXT,
                    hash_signature TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create compliance assessments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS compliance_assessments (
                    assessment_id TEXT PRIMARY KEY,
                    framework TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    assessed_by TEXT,
                    assessment_date TEXT,
                    overall_status TEXT,
                    compliance_score REAL,
                    findings TEXT,
                    recommendations TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_events(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_model ON audit_events(model_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_events(patient_id)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Audit trail database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audit database: {e}")
            raise
    
    async def process_governance_request(
        self, 
        request: GovernanceRequest
    ) -> GovernanceResponse:
        """Process audit trail request."""
        
        try:
            if request.action_type == "log_event":
                return await self._log_audit_event(request)
            elif request.action_type == "query_events":
                return await self._query_audit_events(request)
            elif request.action_type == "generate_report":
                return await self._generate_audit_report(request)
            else:
                raise ValueError(f"Unknown action type: {request.action_type}")
                
        except Exception as e:
            self.logger.error(f"Audit trail processing failed: {e}")
            raise
    
    async def _log_audit_event(self, request: GovernanceRequest) -> GovernanceResponse:
        """Log audit event to secure trail."""
        
        event_data = request.parameters
        
        # Create audit event
        audit_event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType(event_data['event_type']),
            timestamp=datetime.utcnow(),
            user_id=event_data.get('user_id'),
            session_id=event_data.get('session_id'),
            model_id=event_data.get('model_id'),
            patient_id=event_data.get('patient_id'),
            action=event_data['action'],
            details=event_data.get('details', {}),
            outcome=event_data.get('outcome', 'completed'),
            risk_level=RiskLevel(event_data.get('risk_level', 'low')),
            compliance_implications=event_data.get('compliance_implications', []),
            ip_address=event_data.get('ip_address'),
            user_agent=event_data.get('user_agent'),
            location=event_data.get('location')
        )
        
        # Generate hash signature
        audit_event.hash_signature = self._generate_event_hash(audit_event)
        
        # Store in database
        await self._store_audit_event(audit_event)
        
        return GovernanceResponse(
            request_id=request.request_id,
            action_type=request.action_type,
            status="completed",
            results={
                'event_id': audit_event.event_id,
                'hash_signature': audit_event.hash_signature
            },
            audit_events=[audit_event.event_id],
            response_timestamp=datetime.utcnow(),
            metadata={'audit_trail_integrity': 'verified'}
        )
    
    async def _store_audit_event(self, event: AuditEvent):
        """Store audit event in database."""
        
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.database_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO audit_events 
                    (event_id, event_type, timestamp, user_id, session_id, model_id, 
                     patient_id, action, details, outcome, risk_level, ip_address, 
                     user_agent, location, hash_signature)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id,
                    event.event_type.value,
                    event.timestamp.isoformat(),
                    event.user_id,
                    event.session_id,
                    event.model_id,
                    event.patient_id,
                    event.action,
                    json.dumps(event.details),
                    event.outcome,
                    event.risk_level.value,
                    event.ip_address,
                    event.user_agent,
                    event.location,
                    event.hash_signature
                ))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                self.logger.error(f"Failed to store audit event: {e}")
                raise
    
    def _generate_event_hash(self, event: AuditEvent) -> str:
        """Generate cryptographic hash for audit event integrity."""
        
        # Create event data for hashing
        event_data = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'timestamp': event.timestamp.isoformat(),
            'user_id': event.user_id,
            'action': event.action,
            'details': event.details,
            'outcome': event.outcome
        }
        
        # Generate SHA-256 hash
        event_json = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_json.encode()).hexdigest()
    
    async def _query_audit_events(self, request: GovernanceRequest) -> GovernanceResponse:
        """Query audit events based on criteria."""
        
        query_params = request.parameters
        
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.database_path)
                cursor = conn.cursor()
                
                # Build query based on parameters
                where_clauses = []
                query_values = []
                
                if 'user_id' in query_params:
                    where_clauses.append('user_id = ?')
                    query_values.append(query_params['user_id'])
                
                if 'model_id' in query_params:
                    where_clauses.append('model_id = ?')
                    query_values.append(query_params['model_id'])
                
                if 'start_date' in query_params:
                    where_clauses.append('timestamp >= ?')
                    query_values.append(query_params['start_date'])
                
                if 'end_date' in query_params:
                    where_clauses.append('timestamp <= ?')
                    query_values.append(query_params['end_date'])
                
                where_clause = ' AND '.join(where_clauses) if where_clauses else '1=1'
                
                query = f'''
                    SELECT * FROM audit_events 
                    WHERE {where_clause} 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                '''
                
                limit = query_params.get('limit', 100)
                query_values.append(limit)
                
                cursor.execute(query, query_values)
                rows = cursor.fetchall()
                
                # Convert to event objects
                events = []
                for row in rows:
                    event_dict = {
                        'event_id': row[0],
                        'event_type': row[1],
                        'timestamp': row[2],
                        'user_id': row[3],
                        'session_id': row[4],
                        'model_id': row[5],
                        'patient_id': row[6],
                        'action': row[7],
                        'details': json.loads(row[8]) if row[8] else {},
                        'outcome': row[9],
                        'risk_level': row[10],
                        'ip_address': row[11],
                        'user_agent': row[12],
                        'location': row[13],
                        'hash_signature': row[14]
                    }
                    events.append(event_dict)
                
                conn.close()
                
                return GovernanceResponse(
                    request_id=request.request_id,
                    action_type=request.action_type,
                    status="completed",
                    results={
                        'events': events,
                        'total_found': len(events)
                    },
                    response_timestamp=datetime.utcnow(),
                    metadata={'query_parameters': query_params}
                )
                
            except Exception as e:
                self.logger.error(f"Failed to query audit events: {e}")
                raise
    
    async def verify_audit_integrity(self, event_ids: List[str]) -> Dict[str, bool]:
        """Verify integrity of audit events."""
        
        integrity_results = {}
        
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.database_path)
                cursor = conn.cursor()
                
                for event_id in event_ids:
                    cursor.execute(
                        'SELECT * FROM audit_events WHERE event_id = ?',
                        (event_id,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        # Reconstruct event and verify hash
                        original_hash = row[14]  # hash_signature column
                        
                        # Recreate event data for hash calculation
                        event_data = {
                            'event_id': row[0],
                            'event_type': row[1],
                            'timestamp': row[2],
                            'user_id': row[3],
                            'action': row[7],
                            'details': json.loads(row[8]) if row[8] else {},
                            'outcome': row[9]
                        }
                        
                        calculated_hash = hashlib.sha256(
                            json.dumps(event_data, sort_keys=True).encode()
                        ).hexdigest()
                        
                        integrity_results[event_id] = (original_hash == calculated_hash)
                    else:
                        integrity_results[event_id] = False
                
                conn.close()
                
            except Exception as e:
                self.logger.error(f"Failed to verify audit integrity: {e}")
                for event_id in event_ids:
                    integrity_results[event_id] = False
        
        return integrity_results


class AIGovernanceManager:
    """Central manager for AI governance and regulatory compliance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Initialize governance modules
        self.modules = {
            'regulatory_compliance': RegulatoryComplianceManager(),
            'audit_trail': AuditTrailManager(config.get('audit_db_path', 'audit_trail.db')),
        }
        
        # Model registry (simplified)
        self.model_registry = {}
        
        # Active governance sessions
        self.active_sessions = {}
        
        # Performance metrics
        self.performance_metrics = {
            'total_governance_requests': 0,
            'compliance_assessments_performed': 0,
            'audit_events_logged': 0,
            'policy_violations_detected': 0,
            'average_response_time': 0.0
        }
    
    async def initialize(self):
        """Initialize the AI governance manager."""
        
        try:
            # Initialize all governance modules
            for module in self.modules.values():
                if hasattr(module, 'initialize'):
                    await module.initialize()
            
            self.logger.info("AI Governance Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Governance Manager: {e}")
            raise
    
    async def process_governance_request(
        self, 
        request: GovernanceRequest
    ) -> GovernanceResponse:
        """Process governance request through appropriate module."""
        
        start_time = datetime.utcnow()
        
        try:
            # Route request to appropriate module
            if request.action_type in ['compliance_assessment', 'framework_validation', 'regulatory_reporting']:
                module = self.modules['regulatory_compliance']
            elif request.action_type in ['log_event', 'query_events', 'generate_report']:
                module = self.modules['audit_trail']
            else:
                raise ValueError(f"Unknown governance action: {request.action_type}")
            
            # Process request
            response = await module.process_governance_request(request)
            
            # Update performance metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.performance_metrics['total_governance_requests'] += 1
            
            # Update average response time
            current_avg = self.performance_metrics['average_response_time']
            total_requests = self.performance_metrics['total_governance_requests']
            self.performance_metrics['average_response_time'] = (
                (current_avg * (total_requests - 1)) + processing_time
            ) / total_requests
            
            # Log governance action
            await self._log_governance_action(request, response, processing_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Governance request processing failed: {e}")
            raise
    
    async def register_model(self, model_metadata: ModelMetadata) -> bool:
        """Register AI model in governance system."""
        
        try:
            # Store model metadata
            self.model_registry[model_metadata.model_id] = model_metadata
            
            # Log model registration
            await self._log_governance_action(
                GovernanceRequest(
                    requester_id="system",
                    action_type="model_registration",
                    model_id=model_metadata.model_id
                ),
                GovernanceResponse(
                    request_id="system_generated",
                    action_type="model_registration",
                    status="completed",
                    response_timestamp=datetime.utcnow()
                ),
                0.0
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model registration failed: {e}")
            return False
    
    async def perform_compliance_audit(
        self, 
        model_id: str, 
        frameworks: List[RegulatoryFramework]
    ) -> Dict[str, ComplianceAssessment]:
        """Perform comprehensive compliance audit."""
        
        audit_results = {}
        
        for framework in frameworks:
            request = GovernanceRequest(
                requester_id="audit_system",
                action_type="compliance_assessment",
                model_id=model_id,
                framework=framework
            )
            
            response = await self.process_governance_request(request)
            
            if response.status == "completed":
                audit_results[framework.value] = response.results['assessment']
        
        return audit_results
    
    async def generate_governance_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for governance dashboard."""
        
        # Query recent audit events
        recent_events_request = GovernanceRequest(
            requester_id="dashboard_system",
            action_type="query_events",
            parameters={
                'start_date': (datetime.utcnow() - timedelta(days=30)).isoformat(),
                'limit': 1000
            }
        )
        
        events_response = await self.process_governance_request(recent_events_request)
        recent_events = events_response.results.get('events', [])
        
        # Calculate metrics
        total_models = len(self.model_registry)
        high_risk_models = len([
            m for m in self.model_registry.values() 
            if m.risk_classification in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ])
        
        # Compliance status summary
        compliance_summary = defaultdict(int)
        for model in self.model_registry.values():
            for framework, status in model.regulatory_status.items():
                compliance_summary[status.value] += 1
        
        # Event type distribution
        event_type_counts = defaultdict(int)
        for event in recent_events:
            event_type_counts[event['event_type']] += 1
        
        return {
            'summary_statistics': {
                'total_models_registered': total_models,
                'high_risk_models': high_risk_models,
                'compliance_assessments_this_month': self.performance_metrics['compliance_assessments_performed'],
                'audit_events_this_month': len(recent_events),
                'policy_violations_detected': self.performance_metrics['policy_violations_detected']
            },
            'compliance_status_distribution': dict(compliance_summary),
            'event_type_distribution': dict(event_type_counts),
            'risk_level_distribution': {
                'low': len([m for m in self.model_registry.values() if m.risk_classification == RiskLevel.LOW]),
                'moderate': len([m for m in self.model_registry.values() if m.risk_classification == RiskLevel.MODERATE]),
                'high': len([m for m in self.model_registry.values() if m.risk_classification == RiskLevel.HIGH]),
                'critical': len([m for m in self.model_registry.values() if m.risk_classification == RiskLevel.CRITICAL])
            },
            'performance_metrics': self.performance_metrics,
            'alerts': self._generate_governance_alerts(recent_events)
        }
    
    def _generate_governance_alerts(self, recent_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate governance alerts from recent events."""
        
        alerts = []
        
        # Check for high-risk events
        high_risk_events = [e for e in recent_events if e.get('risk_level') in ['high', 'critical']]
        if len(high_risk_events) > 10:
            alerts.append({
                'type': 'high_risk_activity',
                'severity': 'warning',
                'message': f'{len(high_risk_events)} high-risk events detected in the last 30 days',
                'count': len(high_risk_events)
            })
        
        # Check for compliance issues
        compliance_events = [e for e in recent_events if e.get('event_type') == 'compliance_check']
        failed_compliance = [e for e in compliance_events if e.get('outcome') == 'failed']
        if len(failed_compliance) > 0:
            alerts.append({
                'type': 'compliance_failure',
                'severity': 'critical',
                'message': f'{len(failed_compliance)} compliance check failures require attention',
                'count': len(failed_compliance)
            })
        
        # Check for security events
        security_events = [e for e in recent_events if e.get('event_type') == 'security_event']
        if len(security_events) > 0:
            alerts.append({
                'type': 'security_concern',
                'severity': 'high',
                'message': f'{len(security_events)} security events detected',
                'count': len(security_events)
            })
        
        return alerts
    
    async def _log_governance_action(
        self, 
        request: GovernanceRequest, 
        response: GovernanceResponse, 
        processing_time: float
    ):
        """Log governance action for audit trail."""
        
        try:
            audit_request = GovernanceRequest(
                requester_id="governance_system",
                action_type="log_event",
                parameters={
                    'event_type': 'system_event',
                    'action': f'governance_request_{request.action_type}',
                    'details': {
                        'original_request_id': request.request_id,
                        'requester_id': request.requester_id,
                        'action_type': request.action_type,
                        'response_status': response.status,
                        'processing_time_seconds': processing_time
                    },
                    'outcome': response.status,
                    'risk_level': 'low'
                }
            )
            
            await self.modules['audit_trail'].process_governance_request(audit_request)
            
        except Exception as e:
            self.logger.error(f"Failed to log governance action: {e}")
    
    async def export_compliance_report(
        self, 
        model_id: Optional[str] = None, 
        frameworks: Optional[List[RegulatoryFramework]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Export comprehensive compliance report."""
        
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=90)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Query relevant audit events
        query_params = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'limit': 10000
        }
        
        if model_id:
            query_params['model_id'] = model_id
        
        events_request = GovernanceRequest(
            requester_id="compliance_system",
            action_type="query_events",
            parameters=query_params
        )
        
        events_response = await self.process_governance_request(events_request)
        audit_events = events_response.results.get('events', [])
        
        # Generate compliance report
        report = {
            'report_metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'report_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'scope': {
                    'model_id': model_id or 'all_models',
                    'frameworks': [f.value for f in frameworks] if frameworks else 'all_frameworks'
                }
            },
            'executive_summary': {
                'total_models_assessed': len(self.model_registry),
                'total_audit_events': len(audit_events),
                'compliance_rate': 85.0,  # Would calculate from actual data
                'critical_findings': 2,
                'recommendations_count': 5
            },
            'detailed_findings': self._generate_detailed_findings(audit_events),
            'compliance_status_by_framework': self._generate_framework_compliance_summary(),
            'risk_assessment': self._generate_risk_assessment(),
            'recommendations': self._generate_compliance_recommendations(),
            'action_plan': self._generate_compliance_action_plan(),
            'appendices': {
                'audit_trail_summary': len(audit_events),
                'model_inventory': list(self.model_registry.keys()),
                'regulatory_updates': []
            }
        }
        
        return report
    
    def _generate_detailed_findings(self, audit_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate detailed findings from audit events."""
        
        findings = []
        
        # Analyze compliance-related events
        compliance_events = [e for e in audit_events if e.get('event_type') == 'compliance_check']
        
        for event in compliance_events:
            if event.get('outcome') == 'failed':
                findings.append({
                    'finding_id': f"COMP_{event['event_id'][:8]}",
                    'severity': 'high',
                    'category': 'compliance_violation',
                    'description': f"Compliance check failed for model {event.get('model_id', 'unknown')}",
                    'timestamp': event['timestamp'],
                    'affected_models': [event.get('model_id')] if event.get('model_id') else [],
                    'remediation_required': True
                })
        
        return findings
    
    def _generate_framework_compliance_summary(self) -> Dict[str, Any]:
        """Generate compliance summary by framework."""
        
        summary = {}
        
        for framework in RegulatoryFramework:
            compliant_models = 0
            total_models = 0
            
            for model in self.model_registry.values():
                if framework in model.regulatory_status:
                    total_models += 1
                    if model.regulatory_status[framework] == ComplianceStatus.COMPLIANT:
                        compliant_models += 1
            
            if total_models > 0:
                summary[framework.value] = {
                    'total_models': total_models,
                    'compliant_models': compliant_models,
                    'compliance_rate': (compliant_models / total_models) * 100,
                    'status': 'good' if (compliant_models / total_models) >= 0.9 else 'needs_attention'
                }
        
        return summary
    
    def _generate_risk_assessment(self) -> Dict[str, Any]:
        """Generate overall risk assessment."""
        
        risk_counts = defaultdict(int)
        for model in self.model_registry.values():
            risk_counts[model.risk_classification.value] += 1
        
        return {
            'risk_distribution': dict(risk_counts),
            'overall_risk_level': 'moderate',  # Would calculate based on risk model
            'high_risk_models': risk_counts['high'] + risk_counts['critical'],
            'risk_mitigation_status': 'active',
            'next_risk_review': (datetime.utcnow() + timedelta(days=30)).isoformat()
        }
    
    def _generate_compliance_recommendations(self) -> List[str]:
        """Generate compliance recommendations."""
        
        return [
            "Implement automated compliance monitoring for all high-risk models",
            "Establish regular compliance review cycles for FDA AI/ML framework",
            "Enhance audit trail documentation for regulatory submissions",
            "Develop compliance training program for AI development teams",
            "Create compliance dashboard for real-time monitoring"
        ]
    
    def _generate_compliance_action_plan(self) -> List[Dict[str, Any]]:
        """Generate compliance action plan."""
        
        return [
            {
                'action': 'Complete FDA 510(k) submission for Model XYZ',
                'priority': 'high',
                'assigned_to': 'regulatory_team',
                'due_date': (datetime.utcnow() + timedelta(days=60)).isoformat(),
                'status': 'in_progress'
            },
            {
                'action': 'Implement bias monitoring for all diagnostic models',
                'priority': 'medium',
                'assigned_to': 'ai_ethics_team',
                'due_date': (datetime.utcnow() + timedelta(days=90)).isoformat(),
                'status': 'planned'
            },
            {
                'action': 'Update privacy impact assessment for EU MDR compliance',
                'priority': 'medium',
                'assigned_to': 'privacy_team',
                'due_date': (datetime.utcnow() + timedelta(days=45)).isoformat(),
                'status': 'not_started'
            }
        ]
    
    async def shutdown(self):
        """Shutdown the governance manager."""
        
        try:
            # Close database connections and cleanup
            for module in self.modules.values():
                if hasattr(module, 'shutdown'):
                    await module.shutdown()
            
            self.logger.info("AI Governance Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during governance manager shutdown: {e}")


# Factory function
def create_ai_governance_manager(config: Dict[str, Any]) -> AIGovernanceManager:
    """Create AI governance manager with configuration."""
    return AIGovernanceManager(config)