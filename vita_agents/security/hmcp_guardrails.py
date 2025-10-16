#!/usr/bin/env python3
"""
Comprehensive Healthcare Guardrails System for HMCP

Implements enterprise-grade healthcare guardrails including:
- Prompt injection protection
- PHI detection and masking
- Clinical safety validations
- Medication interaction checks
- Regulatory compliance (HIPAA, FDA, etc.)
- Content filtering and safety monitoring
"""

import re
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

try:
    # Healthcare and medical libraries
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    # Advanced text processing
    import transformers
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    # FHIR validation
    from fhir.resources import Patient, Medication, AllergyIntolerance
    FHIR_AVAILABLE = True
except ImportError:
    FHIR_AVAILABLE = False

from vita_agents.protocols.hmcp import (
    ClinicalUrgency, HealthcareRole, PatientContext, ClinicalContext,
    HMCPMessage, HMCPMessageType
)

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of guardrail violations"""
    PHI_EXPOSURE = "phi_exposure"
    PROMPT_INJECTION = "prompt_injection"
    CLINICAL_SAFETY = "clinical_safety"
    MEDICATION_SAFETY = "medication_safety"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    CONTENT_INAPPROPRIATE = "content_inappropriate"
    DATA_INTEGRITY = "data_integrity"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    RATE_LIMITING = "rate_limiting"


class SeverityLevel(Enum):
    """Severity levels for violations"""
    CRITICAL = "critical"  # Immediate system shutdown required
    HIGH = "high"         # Block request, alert administrators
    MEDIUM = "medium"     # Log warning, allow with restrictions
    LOW = "low"          # Log for monitoring, allow request
    INFO = "info"        # Informational logging only


@dataclass
class GuardrailViolation:
    """Represents a guardrail violation"""
    violation_id: str
    violation_type: ViolationType
    severity: SeverityLevel
    message: str
    details: Dict[str, Any]
    detected_content: Optional[str] = None
    remediation_suggestions: List[str] = field(default_factory=list)
    compliance_frameworks: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailResult:
    """Result of guardrail evaluation"""
    passed: bool
    violations: List[GuardrailViolation] = field(default_factory=list)
    processed_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    guardrails_applied: List[str] = field(default_factory=list)


class PHIDetector:
    """Detects and masks Protected Health Information (PHI)"""
    
    def __init__(self):
        # PHI patterns (simplified for demonstration)
        self.phi_patterns = {
            "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "mrn": r'\b(MRN|mrn)[:\s]*[A-Za-z0-9]{6,15}\b',
            "dob": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            "address": r'\b\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|place|pl)\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "insurance_id": r'\b[A-Za-z]{2,4}\d{8,12}\b'
        }
        
        # Common PHI indicators
        self.phi_keywords = {
            "names": ["patient", "mr.", "mrs.", "ms.", "dr.", "doctor"],
            "medical": ["diagnosis", "condition", "treatment", "medication", "allergy"],
            "identifiers": ["id", "number", "account", "member", "policy"]
        }
        
        # Load NLP model if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    
    def detect_phi(self, text: str) -> List[Dict[str, Any]]:
        """Detect PHI in text"""
        phi_findings = []
        
        # Pattern-based detection
        for phi_type, pattern in self.phi_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                phi_findings.append({
                    "type": phi_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9,
                    "method": "regex"
                })
        
        # NLP-based detection
        if self.nlp:
            phi_findings.extend(self._detect_phi_nlp(text))
        
        return phi_findings
    
    def _detect_phi_nlp(self, text: str) -> List[Dict[str, Any]]:
        """NLP-based PHI detection using spaCy"""
        phi_findings = []
        
        doc = self.nlp(text)
        
        # Named entity recognition
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "CARDINAL"]:
                # Additional context analysis to determine if it's PHI
                phi_likelihood = self._assess_phi_likelihood(ent.text, ent.label_, text)
                
                if phi_likelihood > 0.5:
                    phi_findings.append({
                        "type": "named_entity",
                        "value": ent.text,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": phi_likelihood,
                        "method": "nlp",
                        "entity_label": ent.label_
                    })
        
        return phi_findings
    
    def _assess_phi_likelihood(self, entity: str, label: str, context: str) -> float:
        """Assess likelihood that an entity is PHI"""
        likelihood = 0.0
        
        # Base likelihood by entity type
        if label == "PERSON":
            likelihood = 0.8
        elif label == "DATE":
            likelihood = 0.6
        elif label == "ORG":
            likelihood = 0.4
        elif label == "CARDINAL":
            likelihood = 0.3
        
        # Context-based adjustments
        context_lower = context.lower()
        
        # Medical context increases likelihood
        medical_keywords = ["patient", "diagnosis", "treatment", "medication", "doctor", "hospital"]
        if any(keyword in context_lower for keyword in medical_keywords):
            likelihood += 0.2
        
        # Identifier context
        id_keywords = ["id", "number", "ssn", "mrn", "account"]
        if any(keyword in context_lower for keyword in id_keywords):
            likelihood += 0.3
        
        return min(likelihood, 1.0)
    
    def mask_phi(self, text: str, phi_findings: List[Dict[str, Any]]) -> str:
        """Mask detected PHI in text"""
        masked_text = text
        
        # Sort findings by start position (reverse order for proper replacement)
        sorted_findings = sorted(phi_findings, key=lambda x: x["start"], reverse=True)
        
        for finding in sorted_findings:
            start = finding["start"]
            end = finding["end"]
            phi_type = finding["type"]
            
            # Create appropriate mask
            if phi_type == "ssn":
                mask = "***-**-****"
            elif phi_type == "phone":
                mask = "***-***-****"
            elif phi_type == "email":
                mask = "[EMAIL_REDACTED]"
            elif phi_type == "mrn":
                mask = "[MRN_REDACTED]"
            elif phi_type == "dob":
                mask = "[DOB_REDACTED]"
            elif phi_type == "address":
                mask = "[ADDRESS_REDACTED]"
            elif phi_type == "named_entity":
                mask = f"[{finding.get('entity_label', 'ENTITY')}_REDACTED]"
            else:
                mask = "[REDACTED]"
            
            # Replace in text
            masked_text = masked_text[:start] + mask + masked_text[end:]
        
        return masked_text


class PromptInjectionDetector:
    """Detects prompt injection attacks"""
    
    def __init__(self):
        # Prompt injection patterns
        self.injection_patterns = [
            # Direct instruction overrides
            r'ignore\s+(?:previous|all|above)\s+(?:instructions|prompts?|commands?)',
            r'forget\s+(?:everything|all|previous)',
            r'disregard\s+(?:previous|all|above)',
            
            # Role manipulation
            r'you\s+are\s+(?:no\s+longer|not)\s+(?:a|an)\s+\w+',
            r'act\s+as\s+(?:a|an)\s+(?!healthcare|medical|clinical)\w+',
            r'pretend\s+(?:you\s+are|to\s+be)',
            
            # System message injection
            r'<\|?(?:system|assistant|user)\|?>',
            r'\[?(?:SYSTEM|ASSISTANT|USER)\]?:',
            
            # Encoding/decoding attempts
            r'base64|hex|unicode|decode|encode',
            
            # Jailbreaking attempts
            r'jailbreak|bypass|override|hack',
            r'developer\s+mode|debug\s+mode',
            
            # Healthcare-specific injection attempts
            r'provide\s+(?:unauthorized|illegal)\s+medical\s+advice',
            r'diagnose\s+without\s+(?:proper|medical)\s+(?:training|license)',
            r'prescribe\s+(?:medication|drugs)\s+without\s+(?:license|authorization)'
        ]
        
        # Suspicious character patterns
        self.suspicious_chars = [
            r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]',  # Control characters
            r'[^\x00-\x7F]{10,}',  # Long non-ASCII sequences
            r'[<>{}[\]]{5,}',  # Excessive markup characters
        ]
    
    def detect_injection(self, text: str) -> List[Dict[str, Any]]:
        """Detect prompt injection attempts"""
        injections = []
        
        # Pattern-based detection
        for pattern in self.injection_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                injections.append({
                    "type": "prompt_injection",
                    "pattern": pattern,
                    "match": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.8,
                    "severity": "high"
                })
        
        # Suspicious character detection
        for pattern in self.suspicious_chars:
            matches = re.finditer(pattern, text)
            for match in matches:
                injections.append({
                    "type": "suspicious_encoding",
                    "pattern": pattern,
                    "match": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.6,
                    "severity": "medium"
                })
        
        # Statistical anomaly detection
        injections.extend(self._detect_statistical_anomalies(text))
        
        return injections
    
    def _detect_statistical_anomalies(self, text: str) -> List[Dict[str, Any]]:
        """Detect statistical anomalies that might indicate injection"""
        anomalies = []
        
        # Excessive repetition
        words = text.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_count = max(word_counts.values())
            if max_count > len(words) * 0.3:  # More than 30% repetition
                anomalies.append({
                    "type": "excessive_repetition",
                    "details": f"Word repeated {max_count} times out of {len(words)}",
                    "confidence": 0.7,
                    "severity": "medium"
                })
        
        # Unusual character distribution
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Check for excessive special characters
        special_chars = sum(1 for char in text if not char.isalnum() and not char.isspace())
        if len(text) > 0 and special_chars / len(text) > 0.5:
            anomalies.append({
                "type": "excessive_special_chars",
                "details": f"{special_chars}/{len(text)} characters are special",
                "confidence": 0.6,
                "severity": "medium"
            })
        
        return anomalies


class ClinicalSafetyValidator:
    """Validates clinical safety and appropriateness"""
    
    def __init__(self):
        # Dangerous medical advice patterns
        self.dangerous_patterns = [
            r'stop\s+taking\s+(?:all\s+)?(?:your\s+)?medications?',
            r'ignore\s+(?:your\s+)?doctor[\'s]*\s+(?:advice|recommendations?)',
            r'self-medicate|self-treat',
            r'diagnose\s+yourself',
            r'emergency\s+(?:room|department)\s+is\s+(?:not\s+)?(?:necessary|needed)',
            r'surgery\s+is\s+(?:not\s+)?(?:necessary|needed|required)',
            r'wait\s+and\s+see\s+if\s+(?:symptoms?\s+)?(?:get\s+)?worse'
        ]
        
        # High-risk medical terms requiring careful handling
        self.high_risk_terms = [
            "suicide", "self-harm", "overdose", "poisoning", "emergency",
            "chest pain", "heart attack", "stroke", "bleeding", "unconscious",
            "difficulty breathing", "severe pain", "allergic reaction"
        ]
        
        # Medication safety patterns
        self.medication_warnings = [
            r'take\s+(?:more|extra|additional)\s+(?:doses?|pills?)',
            r'mix\s+with\s+alcohol',
            r'stop\s+(?:suddenly|abruptly)',
            r'share\s+(?:your\s+)?medication',
            r'expired\s+medication'
        ]
    
    def validate_clinical_safety(self, 
                               text: str, 
                               clinical_context: Optional[ClinicalContext] = None) -> List[Dict[str, Any]]:
        """Validate clinical safety of content"""
        safety_issues = []
        
        # Check for dangerous medical advice
        for pattern in self.dangerous_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                safety_issues.append({
                    "type": "dangerous_medical_advice",
                    "match": match.group(),
                    "severity": "critical",
                    "risk_level": "high",
                    "recommendation": "Requires healthcare professional review"
                })
        
        # Check for high-risk terms
        text_lower = text.lower()
        for term in self.high_risk_terms:
            if term in text_lower:
                safety_issues.append({
                    "type": "high_risk_medical_term",
                    "term": term,
                    "severity": "high",
                    "risk_level": "elevated",
                    "recommendation": f"Exercise caution with {term}-related content"
                })
        
        # Check medication safety
        for pattern in self.medication_warnings:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                safety_issues.append({
                    "type": "medication_safety_concern",
                    "match": match.group(),
                    "severity": "high",
                    "risk_level": "medication",
                    "recommendation": "Review medication safety guidelines"
                })
        
        # Context-specific validation
        if clinical_context:
            safety_issues.extend(self._validate_clinical_context(text, clinical_context))
        
        return safety_issues
    
    def _validate_clinical_context(self, 
                                 text: str, 
                                 clinical_context: ClinicalContext) -> List[Dict[str, Any]]:
        """Validate against specific clinical context"""
        context_issues = []
        
        # Check urgency appropriateness
        urgency_keywords = {
            ClinicalUrgency.CRITICAL: ["immediate", "urgent", "emergency", "stat"],
            ClinicalUrgency.HIGH: ["soon", "quickly", "promptly"],
            ClinicalUrgency.MEDIUM: ["routine", "standard"],
            ClinicalUrgency.LOW: ["when convenient", "no rush"]
        }
        
        text_lower = text.lower()
        expected_keywords = urgency_keywords.get(clinical_context.urgency, [])
        
        # Check for urgency mismatch
        if clinical_context.urgency == ClinicalUrgency.CRITICAL:
            non_urgent_keywords = ["wait", "delay", "postpone", "no hurry"]
            if any(keyword in text_lower for keyword in non_urgent_keywords):
                context_issues.append({
                    "type": "urgency_mismatch",
                    "severity": "high",
                    "details": f"Non-urgent language used for {clinical_context.urgency.value} case",
                    "recommendation": "Review urgency appropriateness"
                })
        
        # Check medication conflicts
        if clinical_context.allergies:
            for allergy in clinical_context.allergies:
                if allergy.lower() in text_lower:
                    context_issues.append({
                        "type": "allergy_conflict",
                        "severity": "critical",
                        "allergy": allergy,
                        "details": f"Content mentions known allergen: {allergy}",
                        "recommendation": "Remove allergen reference or add warning"
                    })
        
        return context_issues


class MedicationInteractionChecker:
    """Checks for medication interactions and contraindications"""
    
    def __init__(self):
        # Simplified drug interaction database
        self.interactions = {
            "warfarin": {
                "severe": ["aspirin", "ibuprofen", "naproxen"],
                "moderate": ["omeprazole", "amiodarone"],
                "monitor": ["atorvastatin", "metoprolol"]
            },
            "metformin": {
                "severe": ["iodinated_contrast"],
                "moderate": ["alcohol"],
                "monitor": ["furosemide", "prednisone"]
            },
            "digoxin": {
                "severe": ["quinidine", "verapamil"],
                "moderate": ["amiodarone", "clarithromycin"],
                "monitor": ["furosemide", "spironolactone"]
            }
        }
        
        # Drug contraindications
        self.contraindications = {
            "metformin": ["kidney_disease", "liver_disease", "heart_failure"],
            "nsaids": ["kidney_disease", "heart_failure", "peptic_ulcer"],
            "ace_inhibitors": ["pregnancy", "angioedema_history"]
        }
    
    def check_interactions(self, medications: List[str]) -> List[Dict[str, Any]]:
        """Check for drug-drug interactions"""
        interactions = []
        
        # Normalize medication names
        normalized_meds = [med.lower().strip() for med in medications]
        
        for i, med1 in enumerate(normalized_meds):
            for j, med2 in enumerate(normalized_meds[i+1:], i+1):
                interaction = self._check_drug_pair(med1, med2)
                if interaction:
                    interactions.append({
                        "medication1": medications[i],
                        "medication2": medications[j],
                        "severity": interaction["severity"],
                        "description": interaction["description"],
                        "recommendation": interaction["recommendation"]
                    })
        
        return interactions
    
    def _check_drug_pair(self, drug1: str, drug2: str) -> Optional[Dict[str, Any]]:
        """Check interaction between two drugs"""
        
        # Check both directions
        for primary, secondary in [(drug1, drug2), (drug2, drug1)]:
            if primary in self.interactions:
                drug_interactions = self.interactions[primary]
                
                if secondary in drug_interactions.get("severe", []):
                    return {
                        "severity": "severe",
                        "description": f"Severe interaction between {primary} and {secondary}",
                        "recommendation": "Consider alternative medications"
                    }
                elif secondary in drug_interactions.get("moderate", []):
                    return {
                        "severity": "moderate",
                        "description": f"Moderate interaction between {primary} and {secondary}",
                        "recommendation": "Monitor patient closely"
                    }
                elif secondary in drug_interactions.get("monitor", []):
                    return {
                        "severity": "monitor",
                        "description": f"Monitor interaction between {primary} and {secondary}",
                        "recommendation": "Regular monitoring recommended"
                    }
        
        return None
    
    def check_contraindications(self, 
                              medications: List[str], 
                              conditions: List[str]) -> List[Dict[str, Any]]:
        """Check for contraindications"""
        contraindications = []
        
        normalized_meds = [med.lower().strip() for med in medications]
        normalized_conditions = [cond.lower().strip() for cond in conditions]
        
        for i, med in enumerate(normalized_meds):
            if med in self.contraindications:
                med_contraindications = self.contraindications[med]
                
                for condition in normalized_conditions:
                    if condition in med_contraindications:
                        contraindications.append({
                            "medication": medications[i],
                            "condition": condition,
                            "severity": "contraindicated",
                            "description": f"{medications[i]} is contraindicated with {condition}",
                            "recommendation": "Consider alternative medication"
                        })
        
        return contraindications


class HMCPGuardrailSystem:
    """
    Comprehensive Healthcare Guardrails System for HMCP
    
    Provides enterprise-grade security and safety validation for healthcare AI.
    """
    
    def __init__(self, 
                 enable_phi_detection: bool = True,
                 enable_prompt_injection: bool = True,
                 enable_clinical_safety: bool = True,
                 enable_medication_checking: bool = True,
                 strict_mode: bool = True):
        
        self.enable_phi_detection = enable_phi_detection
        self.enable_prompt_injection = enable_prompt_injection
        self.enable_clinical_safety = enable_clinical_safety
        self.enable_medication_checking = enable_medication_checking
        self.strict_mode = strict_mode
        
        # Initialize components
        self.phi_detector = PHIDetector() if enable_phi_detection else None
        self.injection_detector = PromptInjectionDetector() if enable_prompt_injection else None
        self.safety_validator = ClinicalSafetyValidator() if enable_clinical_safety else None
        self.medication_checker = MedicationInteractionChecker() if enable_medication_checking else None
        
        # Violation tracking
        self.violations_log: List[GuardrailViolation] = []
        self.blocked_requests: int = 0
        self.warnings_issued: int = 0
        
        logger.info("HMCP Guardrail System initialized")
    
    async def validate_request(self, 
                             content: str,
                             patient_context: Optional[PatientContext] = None,
                             clinical_context: Optional[ClinicalContext] = None,
                             user_role: Optional[HealthcareRole] = None) -> GuardrailResult:
        """Validate request against all guardrails"""
        
        start_time = datetime.now()
        violations = []
        processed_content = content
        guardrails_applied = []
        
        # 1. PHI Detection and Masking
        if self.enable_phi_detection and self.phi_detector:
            phi_result = await self._check_phi(content)
            if phi_result.violations:
                violations.extend(phi_result.violations)
                if phi_result.processed_content:
                    processed_content = phi_result.processed_content
            guardrails_applied.append("phi_detection")
        
        # 2. Prompt Injection Detection
        if self.enable_prompt_injection and self.injection_detector:
            injection_result = await self._check_prompt_injection(content)
            violations.extend(injection_result.violations)
            guardrails_applied.append("prompt_injection")
        
        # 3. Clinical Safety Validation
        if self.enable_clinical_safety and self.safety_validator:
            safety_result = await self._check_clinical_safety(content, clinical_context)
            violations.extend(safety_result.violations)
            guardrails_applied.append("clinical_safety")
        
        # 4. Medication Safety
        if self.enable_medication_checking and self.medication_checker and clinical_context:
            med_result = await self._check_medication_safety(clinical_context)
            violations.extend(med_result.violations)
            guardrails_applied.append("medication_safety")
        
        # 5. Authorization Check
        auth_result = await self._check_authorization(user_role, patient_context, clinical_context)
        violations.extend(auth_result.violations)
        guardrails_applied.append("authorization")
        
        # 6. Content Appropriateness
        content_result = await self._check_content_appropriateness(content)
        violations.extend(content_result.violations)
        guardrails_applied.append("content_filter")
        
        # Determine overall result
        critical_violations = [v for v in violations if v.severity == SeverityLevel.CRITICAL]
        high_violations = [v for v in violations if v.severity == SeverityLevel.HIGH]
        
        # In strict mode, block on any high or critical violation
        passed = True
        if self.strict_mode:
            if critical_violations or high_violations:
                passed = False
                self.blocked_requests += 1
        else:
            if critical_violations:
                passed = False
                self.blocked_requests += 1
        
        # Log violations
        for violation in violations:
            self.violations_log.append(violation)
            if violation.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                logger.warning(f"Guardrail violation: {violation.violation_type.value} - {violation.message}")
        
        if violations and passed:
            self.warnings_issued += 1
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return GuardrailResult(
            passed=passed,
            violations=violations,
            processed_content=processed_content if passed else None,
            metadata={
                "strict_mode": self.strict_mode,
                "total_violations": len(violations),
                "critical_violations": len(critical_violations),
                "high_violations": len(high_violations),
                "user_role": user_role.value if user_role else None
            },
            processing_time=processing_time,
            guardrails_applied=guardrails_applied
        )
    
    async def _check_phi(self, content: str) -> GuardrailResult:
        """Check for PHI exposure"""
        violations = []
        processed_content = content
        
        phi_findings = self.phi_detector.detect_phi(content)
        
        if phi_findings:
            # High-confidence PHI findings are violations
            high_confidence_phi = [f for f in phi_findings if f["confidence"] > 0.7]
            
            if high_confidence_phi:
                violation = GuardrailViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type=ViolationType.PHI_EXPOSURE,
                    severity=SeverityLevel.HIGH,
                    message=f"Detected {len(high_confidence_phi)} potential PHI exposures",
                    details={"phi_findings": high_confidence_phi},
                    compliance_frameworks=["HIPAA", "HITECH"],
                    remediation_suggestions=[
                        "Remove or mask identified PHI",
                        "Use de-identified data",
                        "Obtain proper authorization"
                    ]
                )
                violations.append(violation)
                
                # Mask PHI in processed content
                processed_content = self.phi_detector.mask_phi(content, phi_findings)
        
        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations,
            processed_content=processed_content
        )
    
    async def _check_prompt_injection(self, content: str) -> GuardrailResult:
        """Check for prompt injection attempts"""
        violations = []
        
        injections = self.injection_detector.detect_injection(content)
        
        if injections:
            high_risk_injections = [i for i in injections if i.get("severity") == "high"]
            
            if high_risk_injections:
                violation = GuardrailViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type=ViolationType.PROMPT_INJECTION,
                    severity=SeverityLevel.HIGH,
                    message=f"Detected {len(high_risk_injections)} potential prompt injection attempts",
                    details={"injections": high_risk_injections},
                    compliance_frameworks=["Security"],
                    remediation_suggestions=[
                        "Review and sanitize input",
                        "Use input validation",
                        "Implement content filtering"
                    ]
                )
                violations.append(violation)
        
        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations
        )
    
    async def _check_clinical_safety(self, 
                                   content: str, 
                                   clinical_context: Optional[ClinicalContext]) -> GuardrailResult:
        """Check clinical safety"""
        violations = []
        
        safety_issues = self.safety_validator.validate_clinical_safety(content, clinical_context)
        
        for issue in safety_issues:
            severity_map = {
                "critical": SeverityLevel.CRITICAL,
                "high": SeverityLevel.HIGH,
                "medium": SeverityLevel.MEDIUM,
                "low": SeverityLevel.LOW
            }
            
            severity = severity_map.get(issue.get("severity", "medium"), SeverityLevel.MEDIUM)
            
            violation = GuardrailViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.CLINICAL_SAFETY,
                severity=severity,
                message=f"Clinical safety concern: {issue['type']}",
                details=issue,
                compliance_frameworks=["Clinical Guidelines", "FDA"],
                remediation_suggestions=[
                    issue.get("recommendation", "Review clinical appropriateness")
                ]
            )
            violations.append(violation)
        
        return GuardrailResult(
            passed=len([v for v in violations if v.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]]) == 0,
            violations=violations
        )
    
    async def _check_medication_safety(self, clinical_context: ClinicalContext) -> GuardrailResult:
        """Check medication safety"""
        violations = []
        
        if clinical_context.medications:
            # Check drug interactions
            interactions = self.medication_checker.check_interactions(clinical_context.medications)
            
            for interaction in interactions:
                severity_map = {
                    "severe": SeverityLevel.CRITICAL,
                    "moderate": SeverityLevel.HIGH,
                    "monitor": SeverityLevel.MEDIUM
                }
                
                severity = severity_map.get(interaction["severity"], SeverityLevel.MEDIUM)
                
                violation = GuardrailViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type=ViolationType.MEDICATION_SAFETY,
                    severity=severity,
                    message=f"Medication interaction: {interaction['description']}",
                    details=interaction,
                    compliance_frameworks=["FDA", "Clinical Pharmacology"],
                    remediation_suggestions=[interaction["recommendation"]]
                )
                violations.append(violation)
            
            # Check contraindications if we have patient conditions
            if hasattr(clinical_context, 'conditions') and clinical_context.conditions:
                contraindications = self.medication_checker.check_contraindications(
                    clinical_context.medications, 
                    clinical_context.conditions
                )
                
                for contraindication in contraindications:
                    violation = GuardrailViolation(
                        violation_id=str(uuid.uuid4()),
                        violation_type=ViolationType.MEDICATION_SAFETY,
                        severity=SeverityLevel.CRITICAL,
                        message=f"Medication contraindication: {contraindication['description']}",
                        details=contraindication,
                        compliance_frameworks=["FDA", "Clinical Guidelines"],
                        remediation_suggestions=[contraindication["recommendation"]]
                    )
                    violations.append(violation)
        
        return GuardrailResult(
            passed=len([v for v in violations if v.severity == SeverityLevel.CRITICAL]) == 0,
            violations=violations
        )
    
    async def _check_authorization(self, 
                                 user_role: Optional[HealthcareRole],
                                 patient_context: Optional[PatientContext],
                                 clinical_context: Optional[ClinicalContext]) -> GuardrailResult:
        """Check user authorization"""
        violations = []
        
        # Check if user role is provided
        if not user_role:
            violation = GuardrailViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.AUTHENTICATION_FAILURE,
                severity=SeverityLevel.HIGH,
                message="User role not specified",
                details={"issue": "missing_user_role"},
                compliance_frameworks=["RBAC", "HIPAA"],
                remediation_suggestions=["Provide valid user role credentials"]
            )
            violations.append(violation)
        
        # Check role-based access
        if user_role and clinical_context:
            # Students and trainees have limited access
            if user_role in [HealthcareRole.STUDENT, HealthcareRole.TRAINEE]:
                if clinical_context.urgency == ClinicalUrgency.CRITICAL:
                    violation = GuardrailViolation(
                        violation_id=str(uuid.uuid4()),
                        violation_type=ViolationType.AUTHORIZATION_FAILURE,
                        severity=SeverityLevel.HIGH,
                        message="Insufficient privileges for critical care access",
                        details={"user_role": user_role.value, "urgency": clinical_context.urgency.value},
                        compliance_frameworks=["RBAC", "Medical Training"],
                        remediation_suggestions=["Require supervisor oversight", "Escalate to licensed provider"]
                    )
                    violations.append(violation)
        
        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations
        )
    
    async def _check_content_appropriateness(self, content: str) -> GuardrailResult:
        """Check content appropriateness"""
        violations = []
        
        # Check for inappropriate content
        inappropriate_patterns = [
            r'\b(?:hate|discrimin|bias|racist|sexist)\b',
            r'\b(?:violence|violent|assault|abuse)\b',
            r'\b(?:illegal|illicit|unlawful)\b'
        ]
        
        content_lower = content.lower()
        
        for pattern in inappropriate_patterns:
            if re.search(pattern, content_lower):
                violation = GuardrailViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type=ViolationType.CONTENT_INAPPROPRIATE,
                    severity=SeverityLevel.MEDIUM,
                    message="Potentially inappropriate content detected",
                    details={"pattern": pattern},
                    compliance_frameworks=["Content Policy"],
                    remediation_suggestions=["Review content appropriateness", "Use professional language"]
                )
                violations.append(violation)
        
        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations
        )
    
    def get_guardrail_statistics(self) -> Dict[str, Any]:
        """Get guardrail system statistics"""
        total_violations = len(self.violations_log)
        
        # Count by type
        violation_counts = {}
        for violation in self.violations_log:
            violation_type = violation.violation_type.value
            violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for violation in self.violations_log:
            severity = violation.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_violations": total_violations,
            "blocked_requests": self.blocked_requests,
            "warnings_issued": self.warnings_issued,
            "violation_types": violation_counts,
            "severity_distribution": severity_counts,
            "enabled_guardrails": {
                "phi_detection": self.enable_phi_detection,
                "prompt_injection": self.enable_prompt_injection,
                "clinical_safety": self.enable_clinical_safety,
                "medication_checking": self.enable_medication_checking
            },
            "strict_mode": self.strict_mode
        }
    
    def get_recent_violations(self, limit: int = 10) -> List[GuardrailViolation]:
        """Get recent violations"""
        return self.violations_log[-limit:] if self.violations_log else []


# Example usage
async def guardrails_example():
    """Example of using HMCP Guardrails System"""
    
    # Initialize guardrail system
    guardrails = HMCPGuardrailSystem(strict_mode=True)
    
    # Example patient context
    patient_context = PatientContext(
        patient_id="PT12345",
        mrn="MRN-12345",
        demographics={"age": 65, "ssn": "123-45-6789"}  # Contains PHI
    )
    
    # Example clinical context
    clinical_context = ClinicalContext(
        chief_complaint="Chest pain",
        medications=["warfarin", "aspirin"],  # Interaction risk
        allergies=["penicillin"],
        urgency=ClinicalUrgency.HIGH
    )
    
    # Test content with various issues
    test_content = """
    Patient John Doe (SSN: 123-45-6789) should stop taking all medications immediately.
    Ignore previous medical advice and self-medicate with aspirin.
    """
    
    # Validate request
    result = await guardrails.validate_request(
        content=test_content,
        patient_context=patient_context,
        clinical_context=clinical_context,
        user_role=HealthcareRole.PHYSICIAN
    )
    
    print(f"Request passed: {result.passed}")
    print(f"Violations found: {len(result.violations)}")
    
    for violation in result.violations:
        print(f"- {violation.violation_type.value}: {violation.message} ({violation.severity.value})")
    
    if result.processed_content:
        print(f"Processed content: {result.processed_content}")
    
    # Get statistics
    stats = guardrails.get_guardrail_statistics()
    print(f"Guardrail statistics: {stats}")


if __name__ == "__main__":
    # Run example
    import asyncio
    asyncio.run(guardrails_example())