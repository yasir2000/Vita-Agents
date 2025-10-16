"""
🏥 Triage Agent - Emergency Case Prioritization and Symptom Assessment
=====================================================================

This agent handles emergency triage, symptom assessment, and case prioritization
based on medical protocols and AI-powered analysis.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import structlog
from pydantic import BaseModel, Field

from ..core.agent import BaseAgent, AgentMessage, MessageType, Priority, AgentStatus


logger = structlog.get_logger(__name__)


class TriageLevel(str, Enum):
    """Emergency triage levels based on ESI (Emergency Severity Index)"""
    LEVEL_1 = "immediate"      # Life-threatening - immediate intervention
    LEVEL_2 = "emergent"       # High risk - within 10 minutes
    LEVEL_3 = "urgent"         # Stable but urgent - within 30 minutes
    LEVEL_4 = "less_urgent"    # Non-urgent - within 60 minutes
    LEVEL_5 = "non_urgent"     # Non-urgent - within 120 minutes


class VitalSigns(BaseModel):
    """Patient vital signs for triage assessment"""
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    blood_pressure_systolic: Optional[int] = Field(None, description="Systolic BP")
    blood_pressure_diastolic: Optional[int] = Field(None, description="Diastolic BP")
    heart_rate: Optional[int] = Field(None, description="Heart rate per minute")
    respiratory_rate: Optional[int] = Field(None, description="Breaths per minute")
    oxygen_saturation: Optional[float] = Field(None, description="SpO2 percentage")
    pain_score: Optional[int] = Field(None, description="Pain scale 0-10")
    consciousness_level: Optional[str] = Field(None, description="AVPU scale")


class TriageCase(BaseModel):
    """Triage case data structure"""
    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    arrival_time: datetime = Field(default_factory=datetime.now)
    chief_complaint: str
    symptoms: List[str] = Field(default_factory=list)
    vital_signs: Optional[VitalSigns] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    
    # Triage results
    triage_level: Optional[TriageLevel] = None
    priority_score: Optional[float] = None
    recommended_specialty: Optional[str] = None
    estimated_wait_time: Optional[int] = None  # minutes
    red_flags: List[str] = Field(default_factory=list)
    assessment_notes: Optional[str] = None


class TriageAgent(BaseAgent):
    """
    Emergency Triage Agent for healthcare systems.
    
    Responsibilities:
    - Assess patient urgency based on symptoms and vital signs
    - Assign triage levels according to medical protocols
    - Route patients to appropriate medical specialties
    - Identify red flag symptoms requiring immediate attention
    - Estimate wait times based on current capacity
    """
    
    def __init__(self, agent_id: str = None):
        super().__init__(
            agent_id=agent_id or "triage_agent",
            name="Emergency Triage Agent",
            description="Handles emergency triage and patient prioritization"
        )
        
        # Critical symptom patterns that require immediate attention
        self.red_flag_symptoms = {
            "cardiac": [
                "chest pain with radiation",
                "severe chest pain",
                "cardiac arrest",
                "severe arrhythmia",
                "st elevation"
            ],
            "respiratory": [
                "severe respiratory distress",
                "airway obstruction",
                "respiratory arrest",
                "oxygen saturation < 90%",
                "severe asthma attack"
            ],
            "neurological": [
                "stroke symptoms",
                "severe head injury",
                "altered mental status",
                "seizure",
                "loss of consciousness"
            ],
            "trauma": [
                "severe trauma",
                "multiple injuries",
                "penetrating injury",
                "severe bleeding",
                "suspected spinal injury"
            ],
            "sepsis": [
                "signs of sepsis",
                "fever with altered mental status",
                "severe infection",
                "septic shock"
            ]
        }
        
        # Specialty routing based on chief complaints
        self.specialty_routing = {
            "chest pain": "cardiology",
            "heart attack": "cardiology",
            "stroke": "neurology",
            "severe headache": "neurology",
            "trauma": "emergency_medicine",
            "fracture": "orthopedics",
            "respiratory distress": "pulmonology",
            "abdominal pain": "gastroenterology",
            "pregnancy complications": "obstetrics",
            "pediatric emergency": "pediatrics",
            "psychiatric emergency": "psychiatry",
            "overdose": "toxicology",
            "burns": "plastic_surgery",
            "eye injury": "ophthalmology"
        }

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process incoming triage requests"""
        try:
            if message.type == MessageType.TASK:
                task_type = message.data.get("task_type")
                
                if task_type == "triage_assessment":
                    return await self._perform_triage(message)
                elif task_type == "emergency_triage":
                    return await self._emergency_triage(message)
                elif task_type == "re_triage":
                    return await self._re_triage(message)
                elif task_type == "capacity_check":
                    return await self._check_capacity(message)
                else:
                    raise ValueError(f"Unknown task type: {task_type}")
            
            return await super().process_message(message)
            
        except Exception as e:
            logger.error("Error in triage agent", error=str(e))
            return AgentMessage(
                type=MessageType.ERROR,
                sender=self.agent_id,
                receiver=message.sender,
                data={"error": str(e)}
            )

    async def _perform_triage(self, message: AgentMessage) -> AgentMessage:
        """Perform comprehensive triage assessment"""
        case_data = message.data.get("case_data", {})
        triage_case = TriageCase(**case_data)
        
        # Perform triage assessment
        await self._assess_vital_signs(triage_case)
        await self._identify_red_flags(triage_case)
        await self._calculate_priority_score(triage_case)
        await self._assign_triage_level(triage_case)
        await self._route_to_specialty(triage_case)
        await self._estimate_wait_time(triage_case)
        
        logger.info(
            "Triage assessment completed",
            case_id=triage_case.case_id,
            triage_level=triage_case.triage_level,
            priority_score=triage_case.priority_score
        )
        
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            data={
                "case_id": triage_case.case_id,
                "triage_result": triage_case.dict(),
                "status": "completed"
            }
        )

    async def _assess_vital_signs(self, case: TriageCase) -> None:
        """Assess vital signs for abnormalities"""
        if not case.vital_signs:
            return
            
        vs = case.vital_signs
        
        # Critical vital sign thresholds
        if vs.temperature and (vs.temperature > 39.0 or vs.temperature < 35.0):
            case.red_flags.append("Critical temperature")
            
        if vs.blood_pressure_systolic:
            if vs.blood_pressure_systolic > 180 or vs.blood_pressure_systolic < 90:
                case.red_flags.append("Critical blood pressure")
                
        if vs.heart_rate:
            if vs.heart_rate > 120 or vs.heart_rate < 50:
                case.red_flags.append("Critical heart rate")
                
        if vs.respiratory_rate:
            if vs.respiratory_rate > 30 or vs.respiratory_rate < 8:
                case.red_flags.append("Critical respiratory rate")
                
        if vs.oxygen_saturation and vs.oxygen_saturation < 90:
            case.red_flags.append("Critical oxygen saturation")
            
        if vs.consciousness_level and vs.consciousness_level != "Alert":
            case.red_flags.append("Altered consciousness")

    async def _identify_red_flags(self, case: TriageCase) -> None:
        """Identify red flag symptoms requiring immediate attention"""
        complaint_lower = case.chief_complaint.lower()
        symptoms_lower = [s.lower() for s in case.symptoms]
        all_text = complaint_lower + " " + " ".join(symptoms_lower)
        
        for category, symptoms in self.red_flag_symptoms.items():
            for symptom in symptoms:
                if symptom in all_text:
                    case.red_flags.append(f"Red flag: {symptom} ({category})")

    async def _calculate_priority_score(self, case: TriageCase) -> None:
        """Calculate numerical priority score (0-100, higher = more urgent)"""
        score = 0
        
        # Base score from red flags
        score += len(case.red_flags) * 25
        
        # Age-based adjustments
        if case.age:
            if case.age < 2 or case.age > 75:
                score += 10
            elif case.age < 18 or case.age > 65:
                score += 5
                
        # Vital signs scoring
        if case.vital_signs:
            vs = case.vital_signs
            
            # Pain score
            if vs.pain_score and vs.pain_score >= 8:
                score += 15
            elif vs.pain_score and vs.pain_score >= 6:
                score += 10
                
            # Temperature
            if vs.temperature:
                if vs.temperature > 38.5 or vs.temperature < 36.0:
                    score += 10
                    
            # Heart rate variability
            if vs.heart_rate:
                if vs.heart_rate > 100 or vs.heart_rate < 60:
                    score += 8
                    
        # Chief complaint severity
        high_priority_complaints = [
            "chest pain", "difficulty breathing", "severe pain",
            "head injury", "stroke", "heart attack", "severe bleeding"
        ]
        
        for complaint in high_priority_complaints:
            if complaint in case.chief_complaint.lower():
                score += 20
                break
                
        # Medical history considerations
        high_risk_history = [
            "heart disease", "diabetes", "cancer", "stroke history",
            "blood clotting disorder", "immunocompromised"
        ]
        
        for condition in case.medical_history:
            for risk_condition in high_risk_history:
                if risk_condition in condition.lower():
                    score += 5
                    break
        
        case.priority_score = min(score, 100)  # Cap at 100

    async def _assign_triage_level(self, case: TriageCase) -> None:
        """Assign ESI triage level based on assessment"""
        # Level 1: Immediate (life-threatening)
        if case.red_flags or (case.priority_score and case.priority_score >= 80):
            case.triage_level = TriageLevel.LEVEL_1
            
        # Level 2: Emergent (high risk)
        elif case.priority_score and case.priority_score >= 60:
            case.triage_level = TriageLevel.LEVEL_2
            
        # Level 3: Urgent (stable but urgent)
        elif case.priority_score and case.priority_score >= 40:
            case.triage_level = TriageLevel.LEVEL_3
            
        # Level 4: Less urgent
        elif case.priority_score and case.priority_score >= 20:
            case.triage_level = TriageLevel.LEVEL_4
            
        # Level 5: Non-urgent
        else:
            case.triage_level = TriageLevel.LEVEL_5

    async def _route_to_specialty(self, case: TriageCase) -> None:
        """Route case to appropriate medical specialty"""
        complaint_lower = case.chief_complaint.lower()
        
        # Direct routing based on chief complaint
        for pattern, specialty in self.specialty_routing.items():
            if pattern in complaint_lower:
                case.recommended_specialty = specialty
                return
                
        # Default routing based on triage level
        if case.triage_level in [TriageLevel.LEVEL_1, TriageLevel.LEVEL_2]:
            case.recommended_specialty = "emergency_medicine"
        else:
            case.recommended_specialty = "general_medicine"

    async def _estimate_wait_time(self, case: TriageCase) -> None:
        """Estimate wait time based on triage level and current capacity"""
        # Base wait times by triage level (in minutes)
        base_times = {
            TriageLevel.LEVEL_1: 0,    # Immediate
            TriageLevel.LEVEL_2: 10,   # Within 10 minutes
            TriageLevel.LEVEL_3: 30,   # Within 30 minutes
            TriageLevel.LEVEL_4: 60,   # Within 60 minutes
            TriageLevel.LEVEL_5: 120   # Within 120 minutes
        }
        
        base_time = base_times.get(case.triage_level, 60)
        
        # TODO: Integrate with actual capacity management system
        # For now, use a simple calculation
        capacity_multiplier = 1.2  # Assuming 20% above normal capacity
        
        case.estimated_wait_time = int(base_time * capacity_multiplier)

    async def _emergency_triage(self, message: AgentMessage) -> AgentMessage:
        """Handle emergency triage (fast-track processing)"""
        case_data = message.data.get("case_data", {})
        triage_case = TriageCase(**case_data)
        
        # Fast emergency assessment
        await self._identify_red_flags(triage_case)
        
        # If red flags present, immediate triage
        if triage_case.red_flags:
            triage_case.triage_level = TriageLevel.LEVEL_1
            triage_case.priority_score = 100
            triage_case.recommended_specialty = "emergency_medicine"
            triage_case.estimated_wait_time = 0
            
            logger.critical(
                "Emergency triage - immediate attention required",
                case_id=triage_case.case_id,
                red_flags=triage_case.red_flags
            )
        else:
            # Continue with normal triage
            await self._perform_triage(message)
            
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            data={
                "case_id": triage_case.case_id,
                "emergency_triage": True,
                "triage_result": triage_case.dict()
            }
        )

    async def _re_triage(self, message: AgentMessage) -> AgentMessage:
        """Re-assess patient if condition changes"""
        case_id = message.data.get("case_id")
        updated_data = message.data.get("updated_data", {})
        
        # TODO: Retrieve existing case and update with new data
        # For now, perform fresh triage
        return await self._perform_triage(message)

    async def _check_capacity(self, message: AgentMessage) -> AgentMessage:
        """Check current emergency department capacity"""
        # TODO: Integrate with real capacity management system
        
        mock_capacity = {
            "total_beds": 50,
            "occupied_beds": 35,
            "available_beds": 15,
            "current_wait_times": {
                "level_1": 0,
                "level_2": 8,
                "level_3": 25,
                "level_4": 45,
                "level_5": 90
            },
            "staffing_level": "normal",
            "special_alerts": []
        }
        
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            data={"capacity_status": mock_capacity}
        )

    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        return [
            "emergency_triage",
            "symptom_assessment", 
            "vital_signs_analysis",
            "red_flag_identification",
            "specialty_routing",
            "wait_time_estimation",
            "capacity_management",
            "priority_scoring",
            "re_triage"
        ]


# Example usage and testing
async def test_triage_agent():
    """Test the triage agent with sample cases"""
    agent = TriageAgent()
    await agent.start()
    
    # Test case 1: High priority cardiac case
    cardiac_case = {
        "patient_id": "P001",
        "chief_complaint": "severe chest pain with radiation to left arm",
        "symptoms": ["chest pain", "shortness of breath", "nausea"],
        "age": 55,
        "gender": "male",
        "vital_signs": {
            "blood_pressure_systolic": 160,
            "blood_pressure_diastolic": 100,
            "heart_rate": 110,
            "respiratory_rate": 22,
            "oxygen_saturation": 95,
            "pain_score": 9
        },
        "medical_history": ["hypertension", "diabetes"]
    }
    
    message = AgentMessage(
        type=MessageType.TASK,
        sender="test",
        receiver=agent.agent_id,
        data={
            "task_type": "triage_assessment",
            "case_data": cardiac_case
        }
    )
    
    response = await agent.process_message(message)
    print("Cardiac Case Triage Result:")
    print(f"Triage Level: {response.data['triage_result']['triage_level']}")
    print(f"Priority Score: {response.data['triage_result']['priority_score']}")
    print(f"Red Flags: {response.data['triage_result']['red_flags']}")
    print(f"Specialty: {response.data['triage_result']['recommended_specialty']}")
    print(f"Wait Time: {response.data['triage_result']['estimated_wait_time']} minutes")
    
    await agent.stop()


if __name__ == "__main__":
    asyncio.run(test_triage_agent())