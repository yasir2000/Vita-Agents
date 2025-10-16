"""
🔗 Medical Router Agent - Intelligent Healthcare Routing and Care Coordination
============================================================================

This agent provides intelligent routing of patients, cases, and data between
healthcare services, specialists, and care teams.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import structlog
from pydantic import BaseModel, Field
import json

from ..core.agent import BaseAgent, AgentMessage, MessageType, Priority, AgentStatus


logger = structlog.get_logger(__name__)


class RoutingPriority(str, Enum):
    """Priority levels for routing decisions"""
    EMERGENCY = "emergency"          # Immediate routing required
    URGENT = "urgent"               # Within 24 hours
    SEMI_URGENT = "semi_urgent"     # Within 1 week
    ROUTINE = "routine"             # Within 1 month
    ELECTIVE = "elective"           # Scheduled convenience


class SpecialtyType(str, Enum):
    """Medical specialties for routing"""
    EMERGENCY_MEDICINE = "emergency_medicine"
    CARDIOLOGY = "cardiology"
    NEUROLOGY = "neurology"
    PULMONOLOGY = "pulmonology"
    GASTROENTEROLOGY = "gastroenterology"
    ENDOCRINOLOGY = "endocrinology"
    NEPHROLOGY = "nephrology"
    ONCOLOGY = "oncology"
    ORTHOPEDICS = "orthopedics"
    SURGERY = "surgery"
    PSYCHIATRY = "psychiatry"
    DERMATOLOGY = "dermatology"
    OPHTHALMOLOGY = "ophthalmology"
    ENT = "ent"
    RADIOLOGY = "radiology"
    PATHOLOGY = "pathology"
    PRIMARY_CARE = "primary_care"
    PEDIATRICS = "pediatrics"
    GERIATRICS = "geriatrics"
    OBSTETRICS_GYNECOLOGY = "obstetrics_gynecology"


class CareLevel(str, Enum):
    """Levels of care for routing"""
    EMERGENCY_DEPARTMENT = "emergency_department"
    INTENSIVE_CARE = "intensive_care"
    INPATIENT = "inpatient"
    OUTPATIENT = "outpatient"
    URGENT_CARE = "urgent_care"
    PRIMARY_CARE = "primary_care"
    TELEHEALTH = "telehealth"
    HOME_CARE = "home_care"


class RoutingReason(str, Enum):
    """Reasons for routing decisions"""
    EMERGENCY_CONDITION = "emergency_condition"
    SPECIALTY_CONSULTATION = "specialty_consultation"
    DIAGNOSTIC_WORKUP = "diagnostic_workup"
    TREATMENT_PLAN = "treatment_plan"
    FOLLOW_UP = "follow_up"
    SECOND_OPINION = "second_opinion"
    CARE_COORDINATION = "care_coordination"
    RESOURCE_AVAILABILITY = "resource_availability"


class ProviderProfile(BaseModel):
    """Healthcare provider profile for routing"""
    provider_id: str
    name: str
    specialty: SpecialtyType
    sub_specialties: List[str] = Field(default_factory=list)
    care_levels: List[CareLevel] = Field(default_factory=list)
    
    # Availability and capacity
    availability_status: str = "available"  # available, busy, unavailable
    current_capacity: int = 0
    max_capacity: int = 100
    next_available: Optional[datetime] = None
    
    # Quality metrics
    experience_years: Optional[int] = None
    patient_satisfaction: Optional[float] = None
    success_rate: Optional[float] = None
    response_time_avg: Optional[int] = None  # minutes
    
    # Contact and location
    location: Optional[str] = None
    contact_info: Optional[Dict[str, str]] = None
    
    # Preferences and constraints
    preferred_conditions: List[str] = Field(default_factory=list)
    language_capabilities: List[str] = Field(default_factory=list)
    insurance_accepted: List[str] = Field(default_factory=list)


class RoutingCriteria(BaseModel):
    """Criteria for routing decisions"""
    condition_category: Optional[str] = None
    urgency_level: RoutingPriority
    required_specialty: Optional[SpecialtyType] = None
    preferred_care_level: Optional[CareLevel] = None
    
    # Patient preferences
    location_preference: Optional[str] = None
    provider_preference: Optional[str] = None
    language_requirement: Optional[str] = None
    insurance_type: Optional[str] = None
    
    # Clinical requirements
    required_equipment: List[str] = Field(default_factory=list)
    required_certifications: List[str] = Field(default_factory=list)
    contraindications: List[str] = Field(default_factory=list)
    
    # Timing constraints
    preferred_time: Optional[datetime] = None
    max_wait_time: Optional[int] = None  # hours
    follow_up_required: bool = False


class RoutingDecision(BaseModel):
    """Routing decision with justification"""
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Routing details
    destination_provider: str
    destination_specialty: SpecialtyType
    destination_care_level: CareLevel
    priority: RoutingPriority
    estimated_wait_time: Optional[int] = None  # minutes
    
    # Decision justification
    routing_reason: RoutingReason
    confidence_score: float = Field(description="Routing confidence (0-1)")
    justification: str
    alternative_options: List[str] = Field(default_factory=list)
    
    # Follow-up and coordination
    requires_coordination: bool = False
    coordination_agents: List[str] = Field(default_factory=list)
    follow_up_instructions: List[str] = Field(default_factory=list)
    
    # Quality assurance
    expected_outcome: Optional[str] = None
    quality_indicators: Dict[str, Any] = Field(default_factory=dict)


class RoutingCase(BaseModel):
    """Complete routing case with patient and clinical data"""
    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Patient demographics
    age: Optional[int] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    insurance: Optional[str] = None
    language: Optional[str] = None
    
    # Clinical presentation
    chief_complaint: str
    symptoms: List[str] = Field(default_factory=list)
    severity_score: Optional[float] = None
    triage_level: Optional[str] = None
    vital_signs: Optional[Dict[str, Any]] = None
    
    # Current context
    current_location: Optional[str] = None
    referring_provider: Optional[str] = None
    previous_providers: List[str] = Field(default_factory=list)
    
    # Clinical history
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    
    # Routing requirements
    routing_criteria: RoutingCriteria
    routing_decision: Optional[RoutingDecision] = None
    routing_history: List[RoutingDecision] = Field(default_factory=list)


class MedicalRouterAgent(BaseAgent):
    """
    Intelligent Medical Router Agent for healthcare systems.
    
    Responsibilities:
    - Route patients to appropriate specialists and care levels
    - Coordinate care between multiple providers
    - Optimize resource utilization and wait times
    - Consider patient preferences and constraints
    - Track and analyze routing decisions
    - Provide real-time routing recommendations
    - Support emergency and urgent routing
    """
    
    def __init__(self, agent_id: str = None):
        super().__init__(
            agent_id=agent_id or "medical_router_agent",
            name="Medical Router Agent",
            description="Intelligent routing and care coordination for healthcare systems"
        )
        
        # Provider network (simplified for demonstration)
        self.provider_network: Dict[str, ProviderProfile] = {}
        self.specialty_mapping = self._initialize_specialty_mapping()
        self.condition_routing_rules = self._initialize_routing_rules()
        
        # Initialize sample provider network
        self._setup_sample_providers()

    def _initialize_specialty_mapping(self) -> Dict[str, List[SpecialtyType]]:
        """Initialize condition to specialty mapping"""
        return {
            "chest_pain": [SpecialtyType.CARDIOLOGY, SpecialtyType.EMERGENCY_MEDICINE],
            "heart_attack": [SpecialtyType.CARDIOLOGY, SpecialtyType.EMERGENCY_MEDICINE],
            "stroke": [SpecialtyType.NEUROLOGY, SpecialtyType.EMERGENCY_MEDICINE],
            "seizure": [SpecialtyType.NEUROLOGY, SpecialtyType.EMERGENCY_MEDICINE],
            "shortness_of_breath": [SpecialtyType.PULMONOLOGY, SpecialtyType.CARDIOLOGY],
            "pneumonia": [SpecialtyType.PULMONOLOGY, SpecialtyType.PRIMARY_CARE],
            "abdominal_pain": [SpecialtyType.GASTROENTEROLOGY, SpecialtyType.SURGERY],
            "appendicitis": [SpecialtyType.SURGERY, SpecialtyType.EMERGENCY_MEDICINE],
            "diabetes": [SpecialtyType.ENDOCRINOLOGY, SpecialtyType.PRIMARY_CARE],
            "cancer": [SpecialtyType.ONCOLOGY],
            "broken_bone": [SpecialtyType.ORTHOPEDICS, SpecialtyType.EMERGENCY_MEDICINE],
            "mental_health": [SpecialtyType.PSYCHIATRY, SpecialtyType.PRIMARY_CARE],
            "skin_condition": [SpecialtyType.DERMATOLOGY, SpecialtyType.PRIMARY_CARE],
            "eye_problem": [SpecialtyType.OPHTHALMOLOGY, SpecialtyType.PRIMARY_CARE],
            "pregnancy": [SpecialtyType.OBSTETRICS_GYNECOLOGY, SpecialtyType.PRIMARY_CARE]
        }

    def _initialize_routing_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize condition-based routing rules"""
        return {
            "emergency_conditions": {
                "conditions": ["heart_attack", "stroke", "severe_trauma", "respiratory_failure"],
                "priority": RoutingPriority.EMERGENCY,
                "care_level": CareLevel.EMERGENCY_DEPARTMENT,
                "max_wait_time": 0
            },
            "urgent_conditions": {
                "conditions": ["chest_pain", "severe_abdominal_pain", "difficulty_breathing"],
                "priority": RoutingPriority.URGENT,
                "care_level": CareLevel.URGENT_CARE,
                "max_wait_time": 60  # 1 hour
            },
            "routine_conditions": {
                "conditions": ["diabetes_management", "hypertension", "routine_checkup"],
                "priority": RoutingPriority.ROUTINE,
                "care_level": CareLevel.PRIMARY_CARE,
                "max_wait_time": 10080  # 1 week
            }
        }

    def _setup_sample_providers(self) -> None:
        """Setup sample provider network for demonstration"""
        providers = [
            ProviderProfile(
                provider_id="emergency_01",
                name="City Emergency Department",
                specialty=SpecialtyType.EMERGENCY_MEDICINE,
                care_levels=[CareLevel.EMERGENCY_DEPARTMENT],
                current_capacity=15,
                max_capacity=50,
                response_time_avg=5,
                location="Downtown Medical Center"
            ),
            ProviderProfile(
                provider_id="cardio_01",
                name="Dr. Sarah Johnson - Cardiologist",
                specialty=SpecialtyType.CARDIOLOGY,
                sub_specialties=["interventional_cardiology", "heart_failure"],
                care_levels=[CareLevel.OUTPATIENT, CareLevel.INPATIENT],
                current_capacity=8,
                max_capacity=20,
                experience_years=15,
                patient_satisfaction=4.8,
                response_time_avg=30
            ),
            ProviderProfile(
                provider_id="neuro_01",
                name="Dr. Michael Chen - Neurologist",
                specialty=SpecialtyType.NEUROLOGY,
                sub_specialties=["stroke", "epilepsy"],
                care_levels=[CareLevel.OUTPATIENT, CareLevel.EMERGENCY_DEPARTMENT],
                current_capacity=5,
                max_capacity=15,
                experience_years=12,
                patient_satisfaction=4.6,
                response_time_avg=45
            ),
            ProviderProfile(
                provider_id="primary_01",
                name="Dr. Emily Rodriguez - Family Medicine",
                specialty=SpecialtyType.PRIMARY_CARE,
                care_levels=[CareLevel.PRIMARY_CARE, CareLevel.TELEHEALTH],
                current_capacity=12,
                max_capacity=25,
                experience_years=8,
                patient_satisfaction=4.9,
                response_time_avg=15,
                language_capabilities=["English", "Spanish"]
            ),
            ProviderProfile(
                provider_id="surgery_01",
                name="Dr. Robert Kim - General Surgeon",
                specialty=SpecialtyType.SURGERY,
                sub_specialties=["abdominal_surgery", "emergency_surgery"],
                care_levels=[CareLevel.INPATIENT, CareLevel.EMERGENCY_DEPARTMENT],
                current_capacity=3,
                max_capacity=10,
                experience_years=20,
                patient_satisfaction=4.7,
                response_time_avg=60
            )
        ]
        
        for provider in providers:
            self.provider_network[provider.provider_id] = provider

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process incoming routing requests"""
        try:
            if message.type == MessageType.TASK:
                task_type = message.data.get("task_type")
                
                if task_type == "route_patient":
                    return await self._route_patient(message)
                elif task_type == "find_providers":
                    return await self._find_providers(message)
                elif task_type == "check_availability":
                    return await self._check_availability(message)
                elif task_type == "coordinate_care":
                    return await self._coordinate_care(message)
                elif task_type == "optimize_routing":
                    return await self._optimize_routing(message)
                elif task_type == "emergency_routing":
                    return await self._emergency_routing(message)
                else:
                    raise ValueError(f"Unknown task type: {task_type}")
            
            return await super().process_message(message)
            
        except Exception as e:
            logger.error("Error in medical router agent", error=str(e))
            return AgentMessage(
                type=MessageType.ERROR,
                sender=self.agent_id,
                receiver=message.sender,
                data={"error": str(e)}
            )

    async def _route_patient(self, message: AgentMessage) -> AgentMessage:
        """Route patient to appropriate provider"""
        case_data = message.data.get("case_data", {})
        routing_case = RoutingCase(**case_data)
        
        # Analyze routing requirements
        await self._analyze_routing_requirements(routing_case)
        
        # Find suitable providers
        suitable_providers = await self._find_suitable_providers(routing_case)
        
        # Make routing decision
        routing_decision = await self._make_routing_decision(routing_case, suitable_providers)
        routing_case.routing_decision = routing_decision
        routing_case.routing_history.append(routing_decision)
        
        # Update provider capacity
        await self._update_provider_capacity(routing_decision.destination_provider)
        
        logger.info(
            "Patient routed successfully",
            case_id=routing_case.case_id,
            destination=routing_decision.destination_provider,
            priority=routing_decision.priority.value
        )
        
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            data={
                "case_id": routing_case.case_id,
                "routing_decision": routing_decision.dict(),
                "routing_case": routing_case.dict(),
                "status": "routed"
            }
        )

    async def _analyze_routing_requirements(self, case: RoutingCase) -> None:
        """Analyze case to determine routing requirements"""
        # Determine priority based on symptoms and triage
        priority = await self._determine_priority(case)
        
        # Determine required specialty
        required_specialty = await self._determine_specialty(case)
        
        # Determine care level
        care_level = await self._determine_care_level(case)
        
        # Update routing criteria
        case.routing_criteria.urgency_level = priority
        case.routing_criteria.required_specialty = required_specialty
        case.routing_criteria.preferred_care_level = care_level

    async def _determine_priority(self, case: RoutingCase) -> RoutingPriority:
        """Determine routing priority based on clinical presentation"""
        # Check for emergency conditions
        emergency_keywords = ["heart attack", "stroke", "severe", "critical", "emergency"]
        complaint_lower = case.chief_complaint.lower()
        
        if any(keyword in complaint_lower for keyword in emergency_keywords):
            return RoutingPriority.EMERGENCY
        
        # Check triage level if available
        if case.triage_level:
            triage_priority_map = {
                "1": RoutingPriority.EMERGENCY,
                "2": RoutingPriority.URGENT,
                "3": RoutingPriority.SEMI_URGENT,
                "4": RoutingPriority.ROUTINE,
                "5": RoutingPriority.ELECTIVE
            }
            return triage_priority_map.get(case.triage_level, RoutingPriority.ROUTINE)
        
        # Check severity score
        if case.severity_score:
            if case.severity_score >= 8:
                return RoutingPriority.EMERGENCY
            elif case.severity_score >= 6:
                return RoutingPriority.URGENT
            elif case.severity_score >= 4:
                return RoutingPriority.SEMI_URGENT
            else:
                return RoutingPriority.ROUTINE
        
        # Default based on symptoms
        urgent_symptoms = ["chest pain", "shortness of breath", "severe pain"]
        if any(symptom in case.symptoms for symptom in urgent_symptoms):
            return RoutingPriority.URGENT
        
        return RoutingPriority.ROUTINE

    async def _determine_specialty(self, case: RoutingCase) -> Optional[SpecialtyType]:
        """Determine required medical specialty"""
        complaint_lower = case.chief_complaint.lower()
        
        # Check specialty mapping
        for condition, specialties in self.specialty_mapping.items():
            if condition.replace("_", " ") in complaint_lower:
                return specialties[0]  # Return primary specialty
        
        # Check symptoms
        for symptom in case.symptoms:
            symptom_lower = symptom.lower()
            for condition, specialties in self.specialty_mapping.items():
                if condition.replace("_", " ") in symptom_lower:
                    return specialties[0]
        
        # Default to primary care for routine issues
        return SpecialtyType.PRIMARY_CARE

    async def _determine_care_level(self, case: RoutingCase) -> CareLevel:
        """Determine appropriate care level"""
        priority = case.routing_criteria.urgency_level
        
        if priority == RoutingPriority.EMERGENCY:
            return CareLevel.EMERGENCY_DEPARTMENT
        elif priority == RoutingPriority.URGENT:
            return CareLevel.URGENT_CARE
        elif case.routing_criteria.location_preference == "telehealth":
            return CareLevel.TELEHEALTH
        else:
            return CareLevel.OUTPATIENT

    async def _find_suitable_providers(self, case: RoutingCase) -> List[ProviderProfile]:
        """Find providers suitable for the routing case"""
        suitable_providers = []
        
        required_specialty = case.routing_criteria.required_specialty
        preferred_care_level = case.routing_criteria.preferred_care_level
        
        for provider in self.provider_network.values():
            # Check specialty match
            if required_specialty and provider.specialty != required_specialty:
                continue
            
            # Check care level
            if preferred_care_level and preferred_care_level not in provider.care_levels:
                continue
            
            # Check availability
            if provider.availability_status == "unavailable":
                continue
            
            # Check capacity
            if provider.current_capacity >= provider.max_capacity:
                continue
            
            # Check language requirements
            if (case.language and 
                provider.language_capabilities and 
                case.language not in provider.language_capabilities):
                continue
            
            # Check insurance
            if (case.insurance and 
                provider.insurance_accepted and 
                case.insurance not in provider.insurance_accepted):
                continue
            
            suitable_providers.append(provider)
        
        # Sort by suitability score
        suitable_providers.sort(key=lambda p: self._calculate_suitability_score(p, case), reverse=True)
        
        return suitable_providers

    def _calculate_suitability_score(self, provider: ProviderProfile, case: RoutingCase) -> float:
        """Calculate suitability score for provider-case match"""
        score = 0.0
        
        # Specialty match
        if provider.specialty == case.routing_criteria.required_specialty:
            score += 3.0
        
        # Care level match
        if case.routing_criteria.preferred_care_level in provider.care_levels:
            score += 2.0
        
        # Capacity utilization (prefer less busy providers)
        capacity_ratio = provider.current_capacity / provider.max_capacity
        score += (1.0 - capacity_ratio) * 2.0
        
        # Quality metrics
        if provider.patient_satisfaction:
            score += provider.patient_satisfaction * 0.5
        
        if provider.experience_years:
            score += min(provider.experience_years / 20.0, 1.0) * 1.0
        
        # Response time (prefer faster response)
        if provider.response_time_avg:
            response_score = max(0, 1.0 - (provider.response_time_avg / 120.0))  # Normalize to 2 hours
            score += response_score * 1.0
        
        # Location preference
        if (case.routing_criteria.location_preference and 
            provider.location and 
            case.routing_criteria.location_preference.lower() in provider.location.lower()):
            score += 1.0
        
        return score

    async def _make_routing_decision(self, case: RoutingCase, providers: List[ProviderProfile]) -> RoutingDecision:
        """Make final routing decision"""
        if not providers:
            # No suitable providers found - route to emergency if urgent
            if case.routing_criteria.urgency_level in [RoutingPriority.EMERGENCY, RoutingPriority.URGENT]:
                emergency_provider = self._find_emergency_provider()
                if emergency_provider:
                    return RoutingDecision(
                        destination_provider=emergency_provider.provider_id,
                        destination_specialty=emergency_provider.specialty,
                        destination_care_level=CareLevel.EMERGENCY_DEPARTMENT,
                        priority=RoutingPriority.EMERGENCY,
                        routing_reason=RoutingReason.EMERGENCY_CONDITION,
                        confidence_score=0.7,
                        justification="No suitable specialists available - routing to emergency department",
                        estimated_wait_time=10
                    )
            
            raise ValueError("No suitable providers available")
        
        # Select best provider
        best_provider = providers[0]
        
        # Calculate confidence
        confidence = min(0.9, 0.6 + (len(providers) * 0.1))
        
        # Estimate wait time
        wait_time = self._estimate_wait_time(best_provider, case.routing_criteria.urgency_level)
        
        # Determine routing reason
        routing_reason = self._determine_routing_reason(case, best_provider)
        
        # Create alternative options
        alternatives = [p.name for p in providers[1:3]]
        
        return RoutingDecision(
            destination_provider=best_provider.provider_id,
            destination_specialty=best_provider.specialty,
            destination_care_level=case.routing_criteria.preferred_care_level or CareLevel.OUTPATIENT,
            priority=case.routing_criteria.urgency_level,
            routing_reason=routing_reason,
            confidence_score=confidence,
            justification=f"Best match: {best_provider.name} - specialty expertise and availability",
            alternative_options=alternatives,
            estimated_wait_time=wait_time,
            requires_coordination=len(case.medical_history) > 3  # Complex cases need coordination
        )

    def _find_emergency_provider(self) -> Optional[ProviderProfile]:
        """Find emergency department provider"""
        for provider in self.provider_network.values():
            if (provider.specialty == SpecialtyType.EMERGENCY_MEDICINE and 
                CareLevel.EMERGENCY_DEPARTMENT in provider.care_levels):
                return provider
        return None

    def _estimate_wait_time(self, provider: ProviderProfile, priority: RoutingPriority) -> int:
        """Estimate wait time based on provider capacity and priority"""
        base_wait = provider.response_time_avg or 30
        
        # Adjust for capacity
        capacity_factor = provider.current_capacity / provider.max_capacity
        capacity_wait = base_wait * (1 + capacity_factor)
        
        # Adjust for priority
        priority_multipliers = {
            RoutingPriority.EMERGENCY: 0.1,
            RoutingPriority.URGENT: 0.5,
            RoutingPriority.SEMI_URGENT: 1.0,
            RoutingPriority.ROUTINE: 2.0,
            RoutingPriority.ELECTIVE: 5.0
        }
        
        final_wait = capacity_wait * priority_multipliers.get(priority, 1.0)
        return int(min(final_wait, 480))  # Cap at 8 hours

    def _determine_routing_reason(self, case: RoutingCase, provider: ProviderProfile) -> RoutingReason:
        """Determine the reason for routing decision"""
        if case.routing_criteria.urgency_level == RoutingPriority.EMERGENCY:
            return RoutingReason.EMERGENCY_CONDITION
        elif provider.specialty != SpecialtyType.PRIMARY_CARE:
            return RoutingReason.SPECIALTY_CONSULTATION
        elif case.routing_criteria.follow_up_required:
            return RoutingReason.FOLLOW_UP
        else:
            return RoutingReason.TREATMENT_PLAN

    async def _update_provider_capacity(self, provider_id: str) -> None:
        """Update provider capacity after routing"""
        if provider_id in self.provider_network:
            provider = self.provider_network[provider_id]
            provider.current_capacity += 1
            logger.info("Provider capacity updated", provider_id=provider_id, new_capacity=provider.current_capacity)

    async def _find_providers(self, message: AgentMessage) -> AgentMessage:
        """Find providers based on criteria"""
        criteria = message.data.get("criteria", {})
        
        filtered_providers = []
        for provider in self.provider_network.values():
            if self._matches_criteria(provider, criteria):
                filtered_providers.append(provider.dict())
        
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            data={"providers": filtered_providers}
        )

    def _matches_criteria(self, provider: ProviderProfile, criteria: Dict[str, Any]) -> bool:
        """Check if provider matches search criteria"""
        if "specialty" in criteria and provider.specialty != criteria["specialty"]:
            return False
        
        if "location" in criteria and provider.location:
            if criteria["location"].lower() not in provider.location.lower():
                return False
        
        if "available_only" in criteria and criteria["available_only"]:
            if provider.availability_status != "available":
                return False
        
        return True

    async def _check_availability(self, message: AgentMessage) -> AgentMessage:
        """Check provider availability"""
        provider_id = message.data.get("provider_id")
        
        if provider_id not in self.provider_network:
            return AgentMessage(
                type=MessageType.ERROR,
                sender=self.agent_id,
                receiver=message.sender,
                data={"error": "Provider not found"}
            )
        
        provider = self.provider_network[provider_id]
        availability = {
            "provider_id": provider_id,
            "status": provider.availability_status,
            "current_capacity": provider.current_capacity,
            "max_capacity": provider.max_capacity,
            "next_available": provider.next_available.isoformat() if provider.next_available else None
        }
        
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            data={"availability": availability}
        )

    async def _coordinate_care(self, message: AgentMessage) -> AgentMessage:
        """Coordinate care between multiple providers"""
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            data={"coordination": "Care coordination feature under development"}
        )

    async def _optimize_routing(self, message: AgentMessage) -> AgentMessage:
        """Optimize routing decisions across multiple cases"""
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            data={"optimization": "Routing optimization feature under development"}
        )

    async def _emergency_routing(self, message: AgentMessage) -> AgentMessage:
        """Handle emergency routing with highest priority"""
        case_data = message.data.get("case_data", {})
        
        # Override priority to emergency
        case_data["routing_criteria"] = case_data.get("routing_criteria", {})
        case_data["routing_criteria"]["urgency_level"] = RoutingPriority.EMERGENCY
        case_data["routing_criteria"]["preferred_care_level"] = CareLevel.EMERGENCY_DEPARTMENT
        
        # Use standard routing with emergency override
        return await self._route_patient(AgentMessage(
            type=MessageType.TASK,
            sender=message.sender,
            receiver=self.agent_id,
            data={"task_type": "route_patient", "case_data": case_data}
        ))

    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        return [
            "patient_routing",
            "provider_matching",
            "availability_checking",
            "priority_assessment",
            "care_coordination",
            "emergency_routing",
            "capacity_management",
            "quality_optimization"
        ]


# Example usage and testing
async def test_medical_router():
    """Test the medical router agent with sample cases"""
    agent = MedicalRouterAgent()
    await agent.start()
    
    # Test case: Emergency chest pain
    emergency_case = {
        "patient_id": "P003",
        "age": 55,
        "gender": "male",
        "chief_complaint": "severe chest pain with shortness of breath",
        "symptoms": ["chest pain", "shortness of breath", "sweating"],
        "severity_score": 8.5,
        "triage_level": "2",
        "location": "Downtown",
        "insurance": "Medicare",
        "routing_criteria": {
            "urgency_level": "urgent",
            "location_preference": "Downtown Medical Center"
        }
    }
    
    message = AgentMessage(
        type=MessageType.TASK,
        sender="test",
        receiver=agent.agent_id,
        data={
            "task_type": "route_patient",
            "case_data": emergency_case
        }
    )
    
    response = await agent.process_message(message)
    routing_result = response.data
    
    print("Medical Routing Results:")
    print(f"Case ID: {routing_result['case_id']}")
    
    decision = routing_result["routing_decision"]
    print(f"\nRouting Decision:")
    print(f"Destination: {decision['destination_provider']}")
    print(f"Specialty: {decision['destination_specialty']}")
    print(f"Care Level: {decision['destination_care_level']}")
    print(f"Priority: {decision['priority']}")
    print(f"Estimated Wait: {decision['estimated_wait_time']} minutes")
    print(f"Confidence: {decision['confidence_score']:.2f}")
    print(f"Justification: {decision['justification']}")
    
    if decision['alternative_options']:
        print(f"Alternatives: {', '.join(decision['alternative_options'])}")
    
    await agent.stop()


if __name__ == "__main__":
    asyncio.run(test_medical_router())