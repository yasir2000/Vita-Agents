"""
🔬 Diagnostic Agent - AI-Powered Medical Diagnosis and Analysis
============================================================

This agent provides AI-powered diagnostic support, differential diagnosis,
and clinical decision assistance for healthcare providers.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import structlog
from pydantic import BaseModel, Field
import json

from ..core.agent import BaseAgent, AgentMessage, MessageType, Priority, AgentStatus


logger = structlog.get_logger(__name__)


class DiagnosisConfidence(str, Enum):
    """Confidence levels for diagnostic suggestions"""
    VERY_HIGH = "very_high"     # 90-100%
    HIGH = "high"               # 70-89%
    MODERATE = "moderate"       # 50-69%
    LOW = "low"                 # 30-49%
    VERY_LOW = "very_low"       # 0-29%


class DiagnosisType(str, Enum):
    """Types of diagnostic assessments"""
    PRIMARY = "primary"         # Primary diagnosis
    DIFFERENTIAL = "differential"  # Differential diagnosis
    RULE_OUT = "rule_out"       # Rule-out diagnosis
    COMORBID = "comorbid"       # Comorbid condition


class ClinicalEvidence(BaseModel):
    """Clinical evidence supporting a diagnosis"""
    type: str = Field(description="Type of evidence (symptom, sign, test, etc.)")
    value: str = Field(description="Evidence description")
    weight: float = Field(description="Evidence weight (0-1)")
    supports: bool = Field(description="Whether evidence supports or contradicts")


class DiagnosticSuggestion(BaseModel):
    """Diagnostic suggestion with confidence and reasoning"""
    diagnosis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    condition_name: str
    icd_10_code: Optional[str] = None
    confidence: DiagnosisConfidence
    confidence_score: float = Field(description="Numerical confidence (0-1)")
    diagnosis_type: DiagnosisType = DiagnosisType.PRIMARY
    
    # Supporting information
    clinical_evidence: List[ClinicalEvidence] = Field(default_factory=list)
    reasoning: str = Field(description="Diagnostic reasoning")
    recommended_tests: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    
    # Additional context
    prevalence: Optional[str] = None
    urgency: str = Field(default="routine")  # immediate, urgent, routine
    specialty_referral: Optional[str] = None


class DiagnosticCase(BaseModel):
    """Complete diagnostic case data"""
    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Patient information
    age: Optional[int] = None
    gender: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    
    # Clinical presentation
    chief_complaint: str
    history_of_present_illness: Optional[str] = None
    symptoms: List[str] = Field(default_factory=list)
    symptom_duration: Optional[str] = None
    symptom_onset: Optional[str] = None
    
    # Physical examination
    vital_signs: Optional[Dict[str, Any]] = None
    physical_findings: List[str] = Field(default_factory=list)
    
    # Medical history
    past_medical_history: List[str] = Field(default_factory=list)
    family_history: List[str] = Field(default_factory=list)
    social_history: Optional[Dict[str, str]] = None
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    
    # Diagnostic results
    lab_results: Optional[Dict[str, Any]] = None
    imaging_results: Optional[Dict[str, Any]] = None
    other_tests: Optional[Dict[str, Any]] = None
    
    # AI Analysis results
    diagnostic_suggestions: List[DiagnosticSuggestion] = Field(default_factory=list)
    differential_diagnoses: List[DiagnosticSuggestion] = Field(default_factory=list)
    analysis_summary: Optional[str] = None


class DiagnosticAgent(BaseAgent):
    """
    AI-Powered Diagnostic Agent for healthcare systems.
    
    Responsibilities:
    - Analyze symptoms and clinical data
    - Generate differential diagnoses
    - Provide diagnostic confidence scores
    - Recommend additional tests and investigations
    - Support clinical decision making
    - Identify potential red flags and emergencies
    """
    
    def __init__(self, agent_id: str = None):
        super().__init__(
            agent_id=agent_id or "diagnostic_agent",
            name="AI Diagnostic Agent",
            description="Provides AI-powered diagnostic support and analysis"
        )
        
        # Disease knowledge base (simplified for demonstration)
        self.disease_patterns = {
            "myocardial_infarction": {
                "symptoms": ["chest pain", "shortness of breath", "nausea", "sweating", "arm pain"],
                "risk_factors": ["diabetes", "hypertension", "smoking", "family history"],
                "age_groups": ["45+"],
                "urgency": "immediate",
                "icd_10": "I21.9",
                "tests": ["ECG", "troponin", "CK-MB"],
                "specialty": "cardiology"
            },
            "pneumonia": {
                "symptoms": ["cough", "fever", "shortness of breath", "chest pain", "fatigue"],
                "risk_factors": ["age > 65", "smoking", "immunocompromised"],
                "urgency": "urgent",
                "icd_10": "J18.9",
                "tests": ["chest X-ray", "CBC", "blood cultures"],
                "specialty": "pulmonology"
            },
            "stroke": {
                "symptoms": ["weakness", "speech difficulty", "facial drooping", "confusion"],
                "risk_factors": ["hypertension", "atrial fibrillation", "diabetes"],
                "urgency": "immediate",
                "icd_10": "I64",
                "tests": ["CT head", "MRI", "carotid ultrasound"],
                "specialty": "neurology"
            },
            "appendicitis": {
                "symptoms": ["abdominal pain", "nausea", "vomiting", "fever"],
                "physical_signs": ["McBurney's point tenderness", "rebound tenderness"],
                "urgency": "urgent",
                "icd_10": "K35.9",
                "tests": ["CT abdomen", "CBC", "urinalysis"],
                "specialty": "surgery"
            },
            "diabetes_mellitus": {
                "symptoms": ["polyuria", "polydipsia", "weight loss", "fatigue"],
                "risk_factors": ["obesity", "family history", "sedentary lifestyle"],
                "urgency": "routine",
                "icd_10": "E11.9",
                "tests": ["fasting glucose", "HbA1c", "oral glucose tolerance test"],
                "specialty": "endocrinology"
            }
        }
        
        # Symptom-to-system mapping
        self.symptom_systems = {
            "chest pain": ["cardiovascular", "respiratory", "gastrointestinal"],
            "shortness of breath": ["cardiovascular", "respiratory"],
            "abdominal pain": ["gastrointestinal", "genitourinary"],
            "headache": ["neurological", "ophthalmological"],
            "fever": ["infectious", "inflammatory", "neoplastic"],
            "fatigue": ["endocrine", "cardiovascular", "infectious"],
            "weakness": ["neurological", "musculoskeletal", "endocrine"]
        }

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process incoming diagnostic requests"""
        try:
            if message.type == MessageType.TASK:
                task_type = message.data.get("task_type")
                
                if task_type == "diagnostic_analysis":
                    return await self._perform_diagnostic_analysis(message)
                elif task_type == "differential_diagnosis":
                    return await self._generate_differential_diagnosis(message)
                elif task_type == "symptom_analysis":
                    return await self._analyze_symptoms(message)
                elif task_type == "test_recommendations":
                    return await self._recommend_tests(message)
                elif task_type == "risk_assessment":
                    return await self._assess_risk(message)
                else:
                    raise ValueError(f"Unknown task type: {task_type}")
            
            return await super().process_message(message)
            
        except Exception as e:
            logger.error("Error in diagnostic agent", error=str(e))
            return AgentMessage(
                type=MessageType.ERROR,
                sender=self.agent_id,
                receiver=message.sender,
                data={"error": str(e)}
            )

    async def _perform_diagnostic_analysis(self, message: AgentMessage) -> AgentMessage:
        """Perform comprehensive diagnostic analysis"""
        case_data = message.data.get("case_data", {})
        diagnostic_case = DiagnosticCase(**case_data)
        
        # Analyze symptoms and generate hypotheses
        await self._analyze_clinical_presentation(diagnostic_case)
        await self._generate_diagnostic_hypotheses(diagnostic_case)
        await self._calculate_diagnostic_confidence(diagnostic_case)
        await self._recommend_investigations(diagnostic_case)
        await self._identify_red_flags(diagnostic_case)
        await self._generate_analysis_summary(diagnostic_case)
        
        logger.info(
            "Diagnostic analysis completed",
            case_id=diagnostic_case.case_id,
            suggestions_count=len(diagnostic_case.diagnostic_suggestions)
        )
        
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            data={
                "case_id": diagnostic_case.case_id,
                "diagnostic_analysis": diagnostic_case.dict(),
                "status": "completed"
            }
        )

    async def _analyze_clinical_presentation(self, case: DiagnosticCase) -> None:
        """Analyze the clinical presentation and symptoms"""
        # Categorize symptoms by body system
        symptom_systems = {}
        for symptom in case.symptoms:
            systems = self.symptom_systems.get(symptom.lower(), ["unknown"])
            for system in systems:
                if system not in symptom_systems:
                    symptom_systems[system] = []
                symptom_systems[system].append(symptom)
        
        # Store analysis for later use
        case.analysis_summary = f"Symptoms primarily affect: {', '.join(symptom_systems.keys())}"

    async def _generate_diagnostic_hypotheses(self, case: DiagnosticCase) -> None:
        """Generate diagnostic hypotheses based on pattern matching"""
        case_symptoms = [s.lower() for s in case.symptoms]
        complaint_lower = case.chief_complaint.lower()
        
        for disease, pattern in self.disease_patterns.items():
            score = 0
            matching_symptoms = []
            
            # Check symptom matches
            for symptom in pattern.get("symptoms", []):
                if any(symptom in cs for cs in case_symptoms) or symptom in complaint_lower:
                    score += 1
                    matching_symptoms.append(symptom)
            
            # Check risk factors
            risk_factor_matches = []
            for risk_factor in pattern.get("risk_factors", []):
                if any(risk_factor in pmh.lower() for pmh in case.past_medical_history):
                    score += 0.5
                    risk_factor_matches.append(risk_factor)
            
            # Age considerations
            age_appropriate = True
            if case.age and "age_groups" in pattern:
                age_groups = pattern["age_groups"]
                for age_group in age_groups:
                    if "+" in age_group:
                        min_age = int(age_group.replace("+", ""))
                        if case.age < min_age:
                            age_appropriate = False
            
            # Generate suggestion if sufficient matches
            if score >= 1:  # At least one symptom match
                confidence_score = min(score / len(pattern.get("symptoms", [1])), 1.0)
                
                # Adjust for age appropriateness
                if not age_appropriate:
                    confidence_score *= 0.7
                
                # Determine confidence level
                if confidence_score >= 0.8:
                    confidence = DiagnosisConfidence.VERY_HIGH
                elif confidence_score >= 0.6:
                    confidence = DiagnosisConfidence.HIGH
                elif confidence_score >= 0.4:
                    confidence = DiagnosisConfidence.MODERATE
                elif confidence_score >= 0.2:
                    confidence = DiagnosisConfidence.LOW
                else:
                    confidence = DiagnosisConfidence.VERY_LOW
                
                # Create evidence list
                evidence = []
                for symptom in matching_symptoms:
                    evidence.append(ClinicalEvidence(
                        type="symptom",
                        value=symptom,
                        weight=0.8,
                        supports=True
                    ))
                
                for risk_factor in risk_factor_matches:
                    evidence.append(ClinicalEvidence(
                        type="risk_factor",
                        value=risk_factor,
                        weight=0.6,
                        supports=True
                    ))
                
                suggestion = DiagnosticSuggestion(
                    condition_name=disease.replace("_", " ").title(),
                    icd_10_code=pattern.get("icd_10"),
                    confidence=confidence,
                    confidence_score=confidence_score,
                    clinical_evidence=evidence,
                    reasoning=f"Based on {len(matching_symptoms)} matching symptoms and {len(risk_factor_matches)} risk factors",
                    recommended_tests=pattern.get("tests", []),
                    urgency=pattern.get("urgency", "routine"),
                    specialty_referral=pattern.get("specialty")
                )
                
                case.diagnostic_suggestions.append(suggestion)
        
        # Sort by confidence score
        case.diagnostic_suggestions.sort(key=lambda x: x.confidence_score, reverse=True)

    async def _calculate_diagnostic_confidence(self, case: DiagnosticCase) -> None:
        """Calculate and adjust diagnostic confidence based on additional factors"""
        for suggestion in case.diagnostic_suggestions:
            # Adjust confidence based on age appropriateness
            if case.age:
                age_factor = self._get_age_factor(suggestion.condition_name, case.age)
                suggestion.confidence_score *= age_factor
            
            # Adjust for gender specificity
            if case.gender:
                gender_factor = self._get_gender_factor(suggestion.condition_name, case.gender)
                suggestion.confidence_score *= gender_factor
            
            # Recalculate confidence level
            if suggestion.confidence_score >= 0.9:
                suggestion.confidence = DiagnosisConfidence.VERY_HIGH
            elif suggestion.confidence_score >= 0.7:
                suggestion.confidence = DiagnosisConfidence.HIGH
            elif suggestion.confidence_score >= 0.5:
                suggestion.confidence = DiagnosisConfidence.MODERATE
            elif suggestion.confidence_score >= 0.3:
                suggestion.confidence = DiagnosisConfidence.LOW
            else:
                suggestion.confidence = DiagnosisConfidence.VERY_LOW

    def _get_age_factor(self, condition: str, age: int) -> float:
        """Get age appropriateness factor for condition"""
        age_factors = {
            "Myocardial Infarction": 1.2 if age > 45 else 0.6,
            "Stroke": 1.3 if age > 55 else 0.7,
            "Appendicitis": 1.1 if 10 <= age <= 40 else 0.8,
            "Diabetes Mellitus": 1.2 if age > 35 else 0.8
        }
        return age_factors.get(condition, 1.0)

    def _get_gender_factor(self, condition: str, gender: str) -> float:
        """Get gender appropriateness factor for condition"""
        # Generally conditions are equally likely, but some have gender predispositions
        gender_factors = {
            "Myocardial Infarction": 1.1 if gender.lower() == "male" else 0.9
        }
        return gender_factors.get(condition, 1.0)

    async def _recommend_investigations(self, case: DiagnosticCase) -> None:
        """Recommend diagnostic tests and investigations"""
        all_tests = set()
        
        for suggestion in case.diagnostic_suggestions:
            all_tests.update(suggestion.recommended_tests)
        
        # Prioritize tests based on urgency and diagnostic value
        urgent_tests = []
        routine_tests = []
        
        for suggestion in case.diagnostic_suggestions:
            if suggestion.urgency == "immediate":
                urgent_tests.extend(suggestion.recommended_tests)
            elif suggestion.urgency == "urgent":
                urgent_tests.extend(suggestion.recommended_tests)
            else:
                routine_tests.extend(suggestion.recommended_tests)
        
        # Update suggestions with prioritized test recommendations
        for suggestion in case.diagnostic_suggestions:
            if suggestion.urgency in ["immediate", "urgent"]:
                suggestion.next_steps.extend([
                    "Immediate evaluation required",
                    "Consider emergency department if not already there"
                ])

    async def _identify_red_flags(self, case: DiagnosticCase) -> None:
        """Identify red flag symptoms or combinations requiring immediate attention"""
        red_flags = []
        
        # Check for emergency combinations
        emergency_combinations = [
            (["chest pain", "shortness of breath"], "Possible cardiac emergency"),
            (["severe headache", "fever", "neck stiffness"], "Possible meningitis"),
            (["abdominal pain", "vomiting", "fever"], "Possible surgical emergency"),
            (["weakness", "speech difficulty"], "Possible stroke")
        ]
        
        case_symptoms_lower = [s.lower() for s in case.symptoms]
        
        for symptom_combo, warning in emergency_combinations:
            if all(any(sym in cs for cs in case_symptoms_lower) for sym in symptom_combo):
                red_flags.append(warning)
        
        # Add red flags to high-priority suggestions
        for suggestion in case.diagnostic_suggestions:
            if suggestion.urgency == "immediate":
                suggestion.red_flags.extend(red_flags)

    async def _generate_analysis_summary(self, case: DiagnosticCase) -> None:
        """Generate a comprehensive analysis summary"""
        if not case.diagnostic_suggestions:
            case.analysis_summary = "No clear diagnostic patterns identified. Consider broader differential diagnosis."
            return
        
        top_suggestion = case.diagnostic_suggestions[0]
        
        summary_parts = [
            f"Primary consideration: {top_suggestion.condition_name} (confidence: {top_suggestion.confidence.value})",
            f"Based on: {top_suggestion.reasoning}",
        ]
        
        if top_suggestion.urgency == "immediate":
            summary_parts.append("⚠️ IMMEDIATE EVALUATION REQUIRED")
        elif top_suggestion.urgency == "urgent":
            summary_parts.append("⚠️ Urgent evaluation recommended")
        
        if len(case.diagnostic_suggestions) > 1:
            other_conditions = [s.condition_name for s in case.diagnostic_suggestions[1:3]]
            summary_parts.append(f"Differential includes: {', '.join(other_conditions)}")
        
        case.analysis_summary = ". ".join(summary_parts)

    async def _generate_differential_diagnosis(self, message: AgentMessage) -> AgentMessage:
        """Generate differential diagnosis list"""
        # This would typically be called after initial analysis
        return await self._perform_diagnostic_analysis(message)

    async def _analyze_symptoms(self, message: AgentMessage) -> AgentMessage:
        """Analyze specific symptoms"""
        symptoms = message.data.get("symptoms", [])
        
        analysis = {}
        for symptom in symptoms:
            systems = self.symptom_systems.get(symptom.lower(), ["unknown"])
            analysis[symptom] = {
                "affected_systems": systems,
                "urgency_indicators": self._assess_symptom_urgency(symptom)
            }
        
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            data={"symptom_analysis": analysis}
        )

    def _assess_symptom_urgency(self, symptom: str) -> List[str]:
        """Assess urgency indicators for individual symptoms"""
        urgent_symptoms = {
            "chest pain": ["Consider cardiac emergency", "Immediate evaluation if severe"],
            "shortness of breath": ["Assess respiratory distress", "Consider cardiac/pulmonary emergency"],
            "severe headache": ["Consider neurological emergency", "Assess for meningitis signs"],
            "abdominal pain": ["Consider surgical emergency", "Assess severity and location"]
        }
        
        return urgent_symptoms.get(symptom.lower(), ["Standard evaluation appropriate"])

    async def _recommend_tests(self, message: AgentMessage) -> AgentMessage:
        """Recommend diagnostic tests based on clinical presentation"""
        # Implementation would analyze the case and recommend appropriate tests
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            data={"test_recommendations": "Feature under development"}
        )

    async def _assess_risk(self, message: AgentMessage) -> AgentMessage:
        """Assess patient risk factors for various conditions"""
        # Implementation would analyze risk factors
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            data={"risk_assessment": "Feature under development"}
        )

    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        return [
            "diagnostic_analysis",
            "differential_diagnosis",
            "symptom_analysis",
            "pattern_recognition",
            "confidence_scoring",
            "test_recommendations",
            "red_flag_identification",
            "urgency_assessment",
            "clinical_reasoning"
        ]


# Example usage and testing
async def test_diagnostic_agent():
    """Test the diagnostic agent with sample cases"""
    agent = DiagnosticAgent()
    await agent.start()
    
    # Test case: Chest pain evaluation
    chest_pain_case = {
        "patient_id": "P002",
        "age": 58,
        "gender": "male",
        "chief_complaint": "severe chest pain radiating to left arm",
        "symptoms": ["chest pain", "shortness of breath", "nausea", "sweating"],
        "past_medical_history": ["hypertension", "diabetes", "smoking history"],
        "current_medications": ["metformin", "lisinopril"],
        "vital_signs": {
            "blood_pressure": "165/95",
            "heart_rate": 105,
            "respiratory_rate": 20,
            "oxygen_saturation": 96
        }
    }
    
    message = AgentMessage(
        type=MessageType.TASK,
        sender="test",
        receiver=agent.agent_id,
        data={
            "task_type": "diagnostic_analysis",
            "case_data": chest_pain_case
        }
    )
    
    response = await agent.process_message(message)
    analysis = response.data["diagnostic_analysis"]
    
    print("Diagnostic Analysis Results:")
    print(f"Analysis Summary: {analysis['analysis_summary']}")
    print("\nTop Diagnostic Suggestions:")
    
    for i, suggestion in enumerate(analysis["diagnostic_suggestions"][:3], 1):
        print(f"{i}. {suggestion['condition_name']}")
        print(f"   Confidence: {suggestion['confidence']} ({suggestion['confidence_score']:.2f})")
        print(f"   Reasoning: {suggestion['reasoning']}")
        print(f"   Urgency: {suggestion['urgency']}")
        if suggestion['recommended_tests']:
            print(f"   Recommended Tests: {', '.join(suggestion['recommended_tests'])}")
        print()
    
    await agent.stop()


if __name__ == "__main__":
    asyncio.run(test_diagnostic_agent())