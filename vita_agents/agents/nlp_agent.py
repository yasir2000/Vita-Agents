"""
Natural Language Processing Agent for clinical text analysis and processing.
"""

import asyncio
import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass
import structlog
from pydantic import BaseModel

from vita_agents.core.agent import BaseAgent, TaskRequest, TaskResponse


logger = structlog.get_logger(__name__)


class EntityType(str, Enum):
    """Types of medical entities that can be extracted."""
    MEDICATION = "medication"
    CONDITION = "condition"
    PROCEDURE = "procedure"
    ANATOMY = "anatomy"
    SYMPTOM = "symptom"
    DOSAGE = "dosage"
    FREQUENCY = "frequency"
    DURATION = "duration"
    PERSON = "person"
    DATE = "date"
    MEASUREMENT = "measurement"


class SentimentType(str, Enum):
    """Sentiment analysis results."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CONCERNING = "concerning"


@dataclass
class MedicalEntity:
    """Extracted medical entity."""
    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float
    normalized_form: Optional[str] = None
    codes: Dict[str, str] = None  # Medical codes (ICD-10, SNOMED, etc.)
    
    def __post_init__(self):
        if self.codes is None:
            self.codes = {}


@dataclass
class ClinicalConcept:
    """Clinical concept with relationships."""
    concept: str
    concept_type: EntityType
    modifiers: List[str] = None
    negated: bool = False
    uncertain: bool = False
    family_history: bool = False
    severity: Optional[str] = None
    
    def __post_init__(self):
        if self.modifiers is None:
            self.modifiers = []


class ClinicalNoteAnalysis(BaseModel):
    """Analysis results for clinical notes."""
    note_id: str
    analysis_timestamp: datetime
    entities: List[Dict[str, Any]]
    concepts: List[Dict[str, Any]]
    sentiment: Dict[str, Any]
    summary: str
    key_findings: List[str]
    recommendations: List[str]
    quality_score: float
    confidence_score: float


class NaturalLanguageProcessingAgent(BaseAgent):
    """
    Agent for processing clinical text and extracting medical information.
    """
    
    def __init__(self, agent_id: str, settings, database):
        super().__init__(agent_id, "nlp")
        self.settings = settings
        self.database = database
        
        # Initialize medical terminology patterns
        self._init_medical_patterns()
        
        # Task handlers
        self.task_handlers = {
            "extract_entities": self._extract_entities,
            "analyze_clinical_note": self._analyze_clinical_note,
            "extract_medications": self._extract_medications,
            "identify_conditions": self._identify_conditions,
            "analyze_sentiment": self._analyze_sentiment,
            "generate_summary": self._generate_summary,
            "detect_clinical_concepts": self._detect_clinical_concepts,
            "validate_note_quality": self._validate_note_quality,
            "anonymize_text": self._anonymize_text,
            "standardize_terminology": self._standardize_terminology,
        }
        
        logger.info("NLP Agent initialized", agent_id=agent_id)
    
    def _init_medical_patterns(self):
        """Initialize medical terminology patterns."""
        
        # Medication patterns
        self.medication_patterns = [
            r'\b(?:mg|mcg|g|ml|units?)\b',  # Dosage units
            r'\b\d+\s*(?:mg|mcg|g|ml|units?)\b',  # Dose amounts
            r'\b(?:twice|once|three times?)\s+(?:daily|a day|per day)\b',  # Frequency
            r'\bq\d+h\b',  # Every X hours
            r'\b(?:prn|as needed|when necessary)\b',  # As needed
        ]
        
        # Common medications (simplified list)
        self.medication_names = {
            'aspirin', 'ibuprofen', 'acetaminophen', 'paracetamol',
            'metformin', 'insulin', 'lisinopril', 'amlodipine',
            'atorvastatin', 'simvastatin', 'omeprazole', 'levothyroxine'
        }
        
        # Condition patterns
        self.condition_patterns = [
            r'\b(?:diagnosis|dx|impression):\s*([^\n\r.]+)',
            r'\b(?:history of|h/o|past medical history)\s*([^\n\r.]+)',
            r'\b(?:presents? with|presenting with|c/o|complain(?:s|ing) of)\s*([^\n\r.]+)',
        ]
        
        # Vital signs patterns
        self.vital_patterns = {
            'blood_pressure': r'\b\d+/\d+\s*(?:mmhg|mm hg)?\b',
            'heart_rate': r'\b(?:hr|heart rate|pulse):\s*\d+\s*(?:bpm)?\b',
            'temperature': r'\b(?:temp|temperature):\s*\d+\.?\d*\s*(?:f|fahrenheit|c|celsius)?\b',
            'respiratory_rate': r'\b(?:rr|resp|respiratory rate):\s*\d+\b',
            'oxygen_saturation': r'\b(?:o2 sat|spo2|oxygen saturation):\s*\d+%?\b'
        }
        
        # Negation patterns
        self.negation_patterns = [
            r'\b(?:no|not|without|absent|denies?|negative)\b',
            r'\b(?:rules? out|r/o)\b',
            r'\b(?:unlikely|doubtful)\b'
        ]
        
        # Uncertainty patterns
        self.uncertainty_patterns = [
            r'\b(?:possible|possibly|probable|probably|likely|suggests?|appears?)\b',
            r'\b(?:concerning for|suspicious for|consistent with)\b',
            r'\b(?:may|might|could|would)\b'
        ]
    
    async def process_task(self, task: TaskRequest) -> TaskResponse:
        """Process NLP tasks."""
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
            logger.error("NLP task processing failed", error=str(e), task_id=task.task_id)
            return TaskResponse(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status="failed",
                error=str(e),
                result={}
            )
    
    async def _extract_entities(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract medical entities from text."""
        text = parameters.get("text", "")
        entity_types = parameters.get("entity_types", list(EntityType))
        
        entities = []
        
        # Extract medications
        if EntityType.MEDICATION in entity_types:
            medication_entities = self._extract_medication_entities(text)
            entities.extend(medication_entities)
        
        # Extract conditions
        if EntityType.CONDITION in entity_types:
            condition_entities = self._extract_condition_entities(text)
            entities.extend(condition_entities)
        
        # Extract measurements
        if EntityType.MEASUREMENT in entity_types:
            measurement_entities = self._extract_measurement_entities(text)
            entities.extend(measurement_entities)
        
        # Extract dates
        if EntityType.DATE in entity_types:
            date_entities = self._extract_date_entities(text)
            entities.extend(date_entities)
        
        return {
            "entities": [entity.__dict__ for entity in entities],
            "entity_count": len(entities),
            "entity_types_found": list(set(entity.entity_type for entity in entities))
        }
    
    async def _analyze_clinical_note(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive analysis of clinical notes."""
        text = parameters.get("text", "")
        note_id = parameters.get("note_id", "unknown")
        
        # Extract entities
        entities_result = await self._extract_entities({"text": text})
        entities = entities_result["entities"]
        
        # Detect clinical concepts
        concepts = self._detect_clinical_concepts_internal(text)
        
        # Analyze sentiment
        sentiment = self._analyze_clinical_sentiment(text)
        
        # Generate summary
        summary = self._generate_clinical_summary(text, entities, concepts)
        
        # Extract key findings
        key_findings = self._extract_key_findings(text, entities, concepts)
        
        # Generate recommendations
        recommendations = self._generate_clinical_recommendations(entities, concepts)
        
        # Assess quality
        quality_score = self._assess_note_quality(text)
        
        # Calculate confidence
        confidence_score = self._calculate_confidence(entities, concepts)
        
        analysis = ClinicalNoteAnalysis(
            note_id=note_id,
            analysis_timestamp=datetime.utcnow(),
            entities=entities,
            concepts=[concept.__dict__ for concept in concepts],
            sentiment=sentiment,
            summary=summary,
            key_findings=key_findings,
            recommendations=recommendations,
            quality_score=quality_score,
            confidence_score=confidence_score
        )
        
        return analysis.dict()
    
    async def _extract_medications(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract medication information from text."""
        text = parameters.get("text", "")
        
        medications = []
        medication_entities = self._extract_medication_entities(text)
        
        for entity in medication_entities:
            # Try to extract dosage, frequency, and duration
            medication_context = self._get_medication_context(text, entity)
            
            medication_info = {
                "name": entity.text,
                "normalized_name": entity.normalized_form,
                "position": {"start": entity.start_pos, "end": entity.end_pos},
                "confidence": entity.confidence,
                "dosage": medication_context.get("dosage"),
                "frequency": medication_context.get("frequency"),
                "duration": medication_context.get("duration"),
                "route": medication_context.get("route"),
                "indication": medication_context.get("indication")
            }
            medications.append(medication_info)
        
        return {
            "medications": medications,
            "medication_count": len(medications)
        }
    
    async def _identify_conditions(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Identify medical conditions in text."""
        text = parameters.get("text", "")
        
        conditions = []
        condition_entities = self._extract_condition_entities(text)
        
        for entity in conditions:
            # Analyze modifiers (negation, uncertainty, etc.)
            context = self._analyze_condition_context(text, entity)
            
            condition_info = {
                "condition": entity.text,
                "normalized_condition": entity.normalized_form,
                "position": {"start": entity.start_pos, "end": entity.end_pos},
                "confidence": entity.confidence,
                "negated": context.get("negated", False),
                "uncertain": context.get("uncertain", False),
                "family_history": context.get("family_history", False),
                "severity": context.get("severity"),
                "codes": entity.codes
            }
            conditions.append(condition_info)
        
        return {
            "conditions": conditions,
            "condition_count": len(conditions)
        }
    
    async def _analyze_sentiment(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of clinical text."""
        text = parameters.get("text", "")
        
        sentiment_result = self._analyze_clinical_sentiment(text)
        
        return sentiment_result
    
    async def _generate_summary(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of clinical text."""
        text = parameters.get("text", "")
        max_length = parameters.get("max_length", 200)
        
        # Extract key entities first
        entities_result = await self._extract_entities({"text": text})
        entities = entities_result["entities"]
        
        # Detect concepts
        concepts = self._detect_clinical_concepts_internal(text)
        
        # Generate summary
        summary = self._generate_clinical_summary(text, entities, concepts)
        
        # Truncate if necessary
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / len(text) if text else 0
        }
    
    async def _detect_clinical_concepts(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect clinical concepts and relationships."""
        text = parameters.get("text", "")
        
        concepts = self._detect_clinical_concepts_internal(text)
        
        return {
            "concepts": [concept.__dict__ for concept in concepts],
            "concept_count": len(concepts),
            "concept_types": list(set(concept.concept_type for concept in concepts))
        }
    
    async def _validate_note_quality(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate quality of clinical note."""
        text = parameters.get("text", "")
        
        quality_score = self._assess_note_quality(text)
        quality_issues = self._identify_quality_issues(text)
        
        return {
            "quality_score": quality_score,
            "quality_grade": self._get_quality_grade(quality_score),
            "quality_issues": quality_issues,
            "recommendations": self._get_quality_recommendations(quality_issues)
        }
    
    async def _anonymize_text(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize clinical text by removing PHI."""
        text = parameters.get("text", "")
        
        # Identify PHI entities
        phi_entities = self._identify_phi_entities(text)
        
        # Replace PHI with placeholders
        anonymized_text = self._replace_phi_entities(text, phi_entities)
        
        return {
            "anonymized_text": anonymized_text,
            "phi_entities_removed": len(phi_entities),
            "phi_types": list(set(entity["type"] for entity in phi_entities))
        }
    
    async def _standardize_terminology(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize medical terminology in text."""
        text = parameters.get("text", "")
        target_standard = parameters.get("target_standard", "SNOMED")
        
        # Extract entities
        entities_result = await self._extract_entities({"text": text})
        entities = entities_result["entities"]
        
        # Map to standard terminology
        standardized_terms = []
        for entity in entities:
            standard_term = self._map_to_standard_terminology(
                entity["text"], entity["entity_type"], target_standard
            )
            if standard_term:
                standardized_terms.append({
                    "original_term": entity["text"],
                    "standard_term": standard_term["term"],
                    "standard_code": standard_term["code"],
                    "terminology": target_standard,
                    "confidence": standard_term["confidence"]
                })
        
        return {
            "standardized_terms": standardized_terms,
            "target_standard": target_standard,
            "mapping_count": len(standardized_terms)
        }
    
    # Helper methods for entity extraction
    def _extract_medication_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medication entities from text."""
        entities = []
        
        # Look for medication names
        for med_name in self.medication_names:
            pattern = r'\b' + re.escape(med_name) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = MedicalEntity(
                    text=match.group(),
                    entity_type=EntityType.MEDICATION,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8,
                    normalized_form=med_name.lower()
                )
                entities.append(entity)
        
        return entities
    
    def _extract_condition_entities(self, text: str) -> List[MedicalEntity]:
        """Extract condition entities from text."""
        entities = []
        
        # Use condition patterns to find conditions
        for pattern in self.condition_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                condition_text = match.group(1).strip()
                entity = MedicalEntity(
                    text=condition_text,
                    entity_type=EntityType.CONDITION,
                    start_pos=match.start(1),
                    end_pos=match.end(1),
                    confidence=0.7
                )
                entities.append(entity)
        
        return entities
    
    def _extract_measurement_entities(self, text: str) -> List[MedicalEntity]:
        """Extract measurement entities from text."""
        entities = []
        
        # Extract vital signs
        for vital_type, pattern in self.vital_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = MedicalEntity(
                    text=match.group(),
                    entity_type=EntityType.MEASUREMENT,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9,
                    normalized_form=vital_type
                )
                entities.append(entity)
        
        return entities
    
    def _extract_date_entities(self, text: str) -> List[MedicalEntity]:
        """Extract date entities from text."""
        entities = []
        
        # Simple date patterns
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2,4}\b',
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = MedicalEntity(
                    text=match.group(),
                    entity_type=EntityType.DATE,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8
                )
                entities.append(entity)
        
        return entities
    
    def _detect_clinical_concepts_internal(self, text: str) -> List[ClinicalConcept]:
        """Detect clinical concepts with modifiers."""
        concepts = []
        
        # This is a simplified implementation
        # In a real system, this would use advanced NLP models
        
        # Look for chief complaints
        cc_pattern = r'(?:chief complaint|cc|presenting complaint):\s*([^\n\r.]+)'
        for match in re.finditer(cc_pattern, text, re.IGNORECASE):
            concept_text = match.group(1).strip()
            concept = ClinicalConcept(
                concept=concept_text,
                concept_type=EntityType.SYMPTOM
            )
            concepts.append(concept)
        
        return concepts
    
    def _analyze_clinical_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of clinical text."""
        # Simplified sentiment analysis
        concerning_words = [
            'severe', 'critical', 'emergency', 'urgent', 'deteriorating',
            'worsening', 'pain', 'distress', 'complications'
        ]
        
        positive_words = [
            'improving', 'stable', 'resolved', 'normal', 'good',
            'excellent', 'satisfactory', 'healing'
        ]
        
        concerning_count = sum(1 for word in concerning_words if word in text.lower())
        positive_count = sum(1 for word in positive_words if word in text.lower())
        
        if concerning_count > positive_count:
            sentiment = SentimentType.CONCERNING
        elif positive_count > concerning_count:
            sentiment = SentimentType.POSITIVE
        else:
            sentiment = SentimentType.NEUTRAL
        
        return {
            "sentiment": sentiment.value,
            "confidence": 0.7,
            "concerning_indicators": concerning_count,
            "positive_indicators": positive_count
        }
    
    def _generate_clinical_summary(
        self, 
        text: str, 
        entities: List[Dict[str, Any]], 
        concepts: List[ClinicalConcept]
    ) -> str:
        """Generate clinical summary from text and extracted information."""
        # Simplified summary generation
        summary_parts = []
        
        # Extract patient age/gender if available
        age_pattern = r'(\d+)\s*(?:year|yo|y/o)'
        age_match = re.search(age_pattern, text, re.IGNORECASE)
        if age_match:
            summary_parts.append(f"{age_match.group(1)}-year-old patient")
        
        # Add chief complaint
        cc_pattern = r'(?:chief complaint|cc):\s*([^\n\r.]+)'
        cc_match = re.search(cc_pattern, text, re.IGNORECASE)
        if cc_match:
            summary_parts.append(f"presenting with {cc_match.group(1).strip()}")
        
        # Add key medications if any
        medications = [e for e in entities if e.get("entity_type") == EntityType.MEDICATION.value]
        if medications:
            med_names = [med["text"] for med in medications[:3]]  # Top 3
            summary_parts.append(f"Current medications include {', '.join(med_names)}")
        
        if summary_parts:
            return ". ".join(summary_parts) + "."
        else:
            return "Clinical note analysis completed with limited extractable information."
    
    def _extract_key_findings(
        self, 
        text: str, 
        entities: List[Dict[str, Any]], 
        concepts: List[ClinicalConcept]
    ) -> List[str]:
        """Extract key clinical findings."""
        findings = []
        
        # Look for assessment and plan
        assessment_pattern = r'(?:assessment|impression|plan):\s*([^\n\r]+)'
        for match in re.finditer(assessment_pattern, text, re.IGNORECASE):
            findings.append(match.group(1).strip())
        
        # Add significant medications
        medications = [e for e in entities if e.get("entity_type") == EntityType.MEDICATION.value]
        if len(medications) > 3:
            findings.append(f"Patient on multiple medications ({len(medications)} identified)")
        
        return findings[:5]  # Limit to 5 key findings
    
    def _generate_clinical_recommendations(
        self, 
        entities: List[Dict[str, Any]], 
        concepts: List[ClinicalConcept]
    ) -> List[str]:
        """Generate clinical recommendations based on analysis."""
        recommendations = []
        
        # Check for medication count
        medications = [e for e in entities if e.get("entity_type") == EntityType.MEDICATION.value]
        if len(medications) > 5:
            recommendations.append("Review medication list for potential interactions")
        
        # Check for missing information
        measurements = [e for e in entities if e.get("entity_type") == EntityType.MEASUREMENT.value]
        if not measurements:
            recommendations.append("Consider documenting vital signs")
        
        # Generic recommendations
        recommendations.append("Ensure proper follow-up scheduling")
        recommendations.append("Verify patient understanding of treatment plan")
        
        return recommendations
    
    def _assess_note_quality(self, text: str) -> float:
        """Assess quality of clinical note."""
        quality_score = 1.0
        
        # Check length
        if len(text) < 100:
            quality_score -= 0.3
        
        # Check for key sections
        key_sections = ['chief complaint', 'history', 'assessment', 'plan']
        sections_found = sum(1 for section in key_sections if section in text.lower())
        if sections_found < 2:
            quality_score -= 0.2
        
        # Check for measurements
        if not any(pattern in text.lower() for pattern in ['bp', 'temperature', 'hr', 'pulse']):
            quality_score -= 0.1
        
        return max(0.0, quality_score)
    
    def _calculate_confidence(
        self, 
        entities: List[Dict[str, Any]], 
        concepts: List[ClinicalConcept]
    ) -> float:
        """Calculate confidence score for analysis."""
        if not entities:
            return 0.1
        
        # Average entity confidence
        entity_confidences = [e.get("confidence", 0.5) for e in entities]
        avg_confidence = sum(entity_confidences) / len(entity_confidences)
        
        return avg_confidence
    
    def _get_medication_context(self, text: str, entity: MedicalEntity) -> Dict[str, Any]:
        """Get medication context (dosage, frequency, etc.)."""
        context = {}
        
        # Look for dosage near medication
        start = max(0, entity.start_pos - 50)
        end = min(len(text), entity.end_pos + 50)
        context_text = text[start:end]
        
        # Simple dosage pattern
        dosage_pattern = r'\b\d+\s*(?:mg|mcg|g|ml|units?)\b'
        dosage_match = re.search(dosage_pattern, context_text, re.IGNORECASE)
        if dosage_match:
            context["dosage"] = dosage_match.group()
        
        return context
    
    def _analyze_condition_context(self, text: str, entity: MedicalEntity) -> Dict[str, Any]:
        """Analyze condition context for modifiers."""
        context = {}
        
        # Look for negation
        start = max(0, entity.start_pos - 30)
        context_text = text[start:entity.end_pos]
        
        for pattern in self.negation_patterns:
            if re.search(pattern, context_text, re.IGNORECASE):
                context["negated"] = True
                break
        
        # Look for uncertainty
        for pattern in self.uncertainty_patterns:
            if re.search(pattern, context_text, re.IGNORECASE):
                context["uncertain"] = True
                break
        
        return context
    
    def _identify_quality_issues(self, text: str) -> List[str]:
        """Identify quality issues in clinical text."""
        issues = []
        
        if len(text) < 100:
            issues.append("Note is very short and may lack detail")
        
        if not re.search(r'(?:chief complaint|cc)', text, re.IGNORECASE):
            issues.append("Chief complaint not clearly documented")
        
        if not re.search(r'(?:assessment|plan)', text, re.IGNORECASE):
            issues.append("Assessment or plan section missing")
        
        return issues
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Fair"
        else:
            return "Poor"
    
    def _get_quality_recommendations(self, issues: List[str]) -> List[str]:
        """Get recommendations to improve note quality."""
        recommendations = []
        
        for issue in issues:
            if "short" in issue:
                recommendations.append("Include more detailed clinical information")
            elif "chief complaint" in issue:
                recommendations.append("Clearly document patient's chief complaint")
            elif "assessment" in issue:
                recommendations.append("Include assessment and plan sections")
        
        return recommendations
    
    def _identify_phi_entities(self, text: str) -> List[Dict[str, Any]]:
        """Identify PHI entities for anonymization."""
        phi_entities = []
        
        # Phone numbers
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        for match in re.finditer(phone_pattern, text):
            phi_entities.append({
                "text": match.group(),
                "type": "phone",
                "start": match.start(),
                "end": match.end()
            })
        
        # SSN pattern
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        for match in re.finditer(ssn_pattern, text):
            phi_entities.append({
                "text": match.group(),
                "type": "ssn",
                "start": match.start(),
                "end": match.end()
            })
        
        return phi_entities
    
    def _replace_phi_entities(self, text: str, phi_entities: List[Dict[str, Any]]) -> str:
        """Replace PHI entities with placeholders."""
        anonymized_text = text
        
        # Sort by position (reverse order to maintain positions)
        phi_entities.sort(key=lambda x: x["start"], reverse=True)
        
        for entity in phi_entities:
            placeholder = f"[{entity['type'].upper()}]"
            anonymized_text = (
                anonymized_text[:entity["start"]] + 
                placeholder + 
                anonymized_text[entity["end"]:]
            )
        
        return anonymized_text
    
    def _map_to_standard_terminology(
        self, 
        term: str, 
        entity_type: str, 
        target_standard: str
    ) -> Optional[Dict[str, Any]]:
        """Map term to standard terminology."""
        # Simplified mapping - in real implementation, this would use
        # terminology services like UMLS, SNOMED CT, etc.
        
        standard_mappings = {
            "aspirin": {
                "term": "Aspirin",
                "code": "387458008",
                "confidence": 0.9
            },
            "diabetes": {
                "term": "Diabetes mellitus",
                "code": "73211009", 
                "confidence": 0.9
            }
        }
        
        return standard_mappings.get(term.lower())