#!/usr/bin/env python3
"""
Healthcare Agent Framework Integration

Implements OpenAI agents framework integration for HMCP with:
- Specialist agent routing (cardiologist, oncologist, etc.)
- Context passing and workflow orchestration
- Healthcare-specific handoff patterns
- Clinical specialty routing
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

try:
    # OpenAI agents framework
    from openai import OpenAI
    from openai.types.beta import Assistant, Thread
    from openai.types.beta.threads import Run, Message
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Fallback types
    class Assistant: pass
    class Thread: pass
    class Run: pass
    class Message: pass

try:
    # MCP types for integration
    from mcp.types import SamplingMessage, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    class SamplingMessage: pass
    class TextContent: pass

from vita_agents.protocols.hmcp import (
    ClinicalUrgency, HealthcareRole, PatientContext, ClinicalContext,
    HMCPMessage, HMCPMessageType
)
from vita_agents.protocols.hmcp_client import HMCPClient

logger = logging.getLogger(__name__)


class MedicalSpecialty(Enum):
    """Medical specialties for agent routing"""
    GENERAL_PRACTICE = "general_practice"
    CARDIOLOGY = "cardiology"
    ONCOLOGY = "oncology"
    NEUROLOGY = "neurology"
    EMERGENCY_MEDICINE = "emergency_medicine"
    RADIOLOGY = "radiology"
    PATHOLOGY = "pathology"
    PHARMACY = "pharmacy"
    NURSING = "nursing"
    PSYCHIATRY = "psychiatry"
    PEDIATRICS = "pediatrics"
    GERIATRICS = "geriatrics"
    INFECTIOUS_DISEASE = "infectious_disease"
    ENDOCRINOLOGY = "endocrinology"
    DERMATOLOGY = "dermatology"


@dataclass
class SpecialistAgent:
    """Configuration for a specialist healthcare agent"""
    specialty: MedicalSpecialty
    assistant_id: str
    name: str
    description: str
    instructions: str
    model: str = "gpt-4-turbo-preview"
    tools: List[Dict[str, Any]] = field(default_factory=list)
    file_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Healthcare-specific properties
    license_requirements: List[str] = field(default_factory=list)
    consultation_patterns: List[str] = field(default_factory=list)
    emergency_protocols: bool = False
    patient_age_range: Optional[str] = None


@dataclass
class HandoffContext:
    """Context for agent handoffs"""
    source_agent: str
    target_agent: str
    patient_context: PatientContext
    clinical_context: ClinicalContext
    handoff_reason: str
    handoff_type: str  # "consultation", "transfer", "emergency", "follow_up"
    priority: ClinicalUrgency
    conversation_history: List[Dict[str, Any]]
    relevant_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class HealthcareAgentFramework:
    """
    Healthcare Agent Framework Integration
    
    Provides specialist agent routing, context passing, and workflow orchestration
    for healthcare scenarios using OpenAI agents framework with HMCP integration.
    """
    
    def __init__(self, 
                 openai_api_key: str,
                 hmcp_client: Optional[HMCPClient] = None,
                 organization_id: Optional[str] = None):
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        self.client = OpenAI(
            api_key=openai_api_key,
            organization=organization_id
        )
        self.hmcp_client = hmcp_client
        
        # Agent registry
        self.specialist_agents: Dict[MedicalSpecialty, SpecialistAgent] = {}
        self.active_assistants: Dict[str, Assistant] = {}
        self.active_threads: Dict[str, Thread] = {}
        
        # Handoff tracking
        self.handoff_history: List[HandoffContext] = []
        self.active_handoffs: Dict[str, HandoffContext] = {}
        
        # Routing rules
        self.routing_rules = self._initialize_routing_rules()
        
        logger.info("Healthcare Agent Framework initialized")
    
    def _initialize_routing_rules(self) -> Dict[str, List[MedicalSpecialty]]:
        """Initialize clinical routing rules based on symptoms/conditions"""
        return {
            # Cardiovascular
            "chest_pain": [MedicalSpecialty.CARDIOLOGY, MedicalSpecialty.EMERGENCY_MEDICINE],
            "heart_palpitations": [MedicalSpecialty.CARDIOLOGY],
            "shortness_of_breath": [MedicalSpecialty.CARDIOLOGY, MedicalSpecialty.EMERGENCY_MEDICINE],
            "hypertension": [MedicalSpecialty.CARDIOLOGY, MedicalSpecialty.GENERAL_PRACTICE],
            
            # Oncology
            "tumor": [MedicalSpecialty.ONCOLOGY, MedicalSpecialty.RADIOLOGY],
            "cancer": [MedicalSpecialty.ONCOLOGY],
            "chemotherapy": [MedicalSpecialty.ONCOLOGY, MedicalSpecialty.PHARMACY],
            "radiation": [MedicalSpecialty.ONCOLOGY, MedicalSpecialty.RADIOLOGY],
            
            # Neurology
            "headache": [MedicalSpecialty.NEUROLOGY, MedicalSpecialty.GENERAL_PRACTICE],
            "seizure": [MedicalSpecialty.NEUROLOGY, MedicalSpecialty.EMERGENCY_MEDICINE],
            "stroke": [MedicalSpecialty.NEUROLOGY, MedicalSpecialty.EMERGENCY_MEDICINE],
            "memory_loss": [MedicalSpecialty.NEUROLOGY, MedicalSpecialty.GERIATRICS],
            
            # Emergency
            "trauma": [MedicalSpecialty.EMERGENCY_MEDICINE],
            "poisoning": [MedicalSpecialty.EMERGENCY_MEDICINE, MedicalSpecialty.PHARMACY],
            "overdose": [MedicalSpecialty.EMERGENCY_MEDICINE, MedicalSpecialty.PSYCHIATRY],
            
            # Pediatrics
            "pediatric": [MedicalSpecialty.PEDIATRICS],
            "child": [MedicalSpecialty.PEDIATRICS],
            "infant": [MedicalSpecialty.PEDIATRICS],
            
            # Mental Health
            "depression": [MedicalSpecialty.PSYCHIATRY, MedicalSpecialty.GENERAL_PRACTICE],
            "anxiety": [MedicalSpecialty.PSYCHIATRY, MedicalSpecialty.GENERAL_PRACTICE],
            "psychosis": [MedicalSpecialty.PSYCHIATRY, MedicalSpecialty.EMERGENCY_MEDICINE],
            
            # General
            "medication": [MedicalSpecialty.PHARMACY, MedicalSpecialty.GENERAL_PRACTICE],
            "vaccination": [MedicalSpecialty.GENERAL_PRACTICE, MedicalSpecialty.PEDIATRICS]
        }
    
    async def register_specialist_agent(self, 
                                      specialty: MedicalSpecialty,
                                      name: str,
                                      instructions: str,
                                      tools: Optional[List[Dict[str, Any]]] = None) -> SpecialistAgent:
        """Register a new specialist agent"""
        
        # Create OpenAI assistant
        assistant = await asyncio.to_thread(
            self.client.beta.assistants.create,
            name=name,
            instructions=instructions,
            model="gpt-4-turbo-preview",
            tools=tools or []
        )
        
        # Create specialist agent configuration
        specialist_agent = SpecialistAgent(
            specialty=specialty,
            assistant_id=assistant.id,
            name=name,
            description=f"Specialist agent for {specialty.value}",
            instructions=instructions,
            tools=tools or []
        )
        
        # Store in registry
        self.specialist_agents[specialty] = specialist_agent
        self.active_assistants[assistant.id] = assistant
        
        logger.info(f"Registered specialist agent: {name} ({specialty.value})")
        return specialist_agent
    
    def route_to_specialist(self, 
                          clinical_text: str,
                          patient_context: PatientContext,
                          current_specialty: Optional[MedicalSpecialty] = None) -> List[MedicalSpecialty]:
        """Route patient case to appropriate specialist(s)"""
        
        clinical_text_lower = clinical_text.lower()
        recommended_specialties = set()
        
        # Apply routing rules
        for keyword, specialties in self.routing_rules.items():
            if keyword in clinical_text_lower:
                recommended_specialties.update(specialties)
        
        # Age-based routing
        if patient_context.demographics.get("age"):
            age = patient_context.demographics["age"]
            if age < 18:
                recommended_specialties.add(MedicalSpecialty.PEDIATRICS)
            elif age > 65:
                recommended_specialties.add(MedicalSpecialty.GERIATRICS)
        
        # Emergency detection
        emergency_keywords = ["emergency", "urgent", "critical", "life-threatening", "code"]
        if any(keyword in clinical_text_lower for keyword in emergency_keywords):
            recommended_specialties.add(MedicalSpecialty.EMERGENCY_MEDICINE)
        
        # Exclude current specialty to avoid circular routing
        if current_specialty:
            recommended_specialties.discard(current_specialty)
        
        # Default to general practice if no specific routing
        if not recommended_specialties:
            recommended_specialties.add(MedicalSpecialty.GENERAL_PRACTICE)
        
        return list(recommended_specialties)
    
    async def initiate_handoff(self, 
                             source_specialty: MedicalSpecialty,
                             target_specialty: MedicalSpecialty,
                             patient_context: PatientContext,
                             clinical_context: ClinicalContext,
                             handoff_reason: str,
                             handoff_type: str = "consultation") -> str:
        """Initiate handoff between specialist agents"""
        
        if target_specialty not in self.specialist_agents:
            raise ValueError(f"Target specialty {target_specialty.value} not registered")
        
        # Create handoff context
        handoff_context = HandoffContext(
            source_agent=source_specialty.value if source_specialty else "system",
            target_agent=target_specialty.value,
            patient_context=patient_context,
            clinical_context=clinical_context,
            handoff_reason=handoff_reason,
            handoff_type=handoff_type,
            priority=clinical_context.urgency,
            conversation_history=[],
            relevant_data={}
        )
        
        # Generate handoff ID
        handoff_id = f"handoff_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{target_specialty.value}"
        
        # Store handoff
        self.active_handoffs[handoff_id] = handoff_context
        self.handoff_history.append(handoff_context)
        
        # Create thread for the handoff
        thread = await asyncio.to_thread(self.client.beta.threads.create)
        self.active_threads[handoff_id] = thread
        
        # Prepare handoff message
        handoff_message = self._prepare_handoff_message(handoff_context)
        
        # Send initial message to target specialist
        await asyncio.to_thread(
            self.client.beta.threads.messages.create,
            thread_id=thread.id,
            role="user",
            content=handoff_message
        )
        
        logger.info(f"Initiated handoff from {source_specialty} to {target_specialty}: {handoff_id}")
        return handoff_id
    
    def _prepare_handoff_message(self, handoff_context: HandoffContext) -> str:
        """Prepare handoff message with clinical context"""
        
        patient_summary = f"""
PATIENT HANDOFF - {handoff_context.handoff_type.upper()}

Patient ID: {handoff_context.patient_context.patient_id}
MRN: {handoff_context.patient_context.mrn}

HANDOFF DETAILS:
- From: {handoff_context.source_agent}
- Reason: {handoff_context.handoff_reason}
- Priority: {handoff_context.priority.value}
- Type: {handoff_context.handoff_type}

CLINICAL CONTEXT:
- Chief Complaint: {handoff_context.clinical_context.chief_complaint}
- Clinical Notes: {handoff_context.clinical_context.clinical_notes}
- Current Medications: {', '.join(handoff_context.clinical_context.medications)}
- Allergies: {', '.join(handoff_context.clinical_context.allergies)}
- Relevant History: {handoff_context.clinical_context.relevant_history}

PATIENT DEMOGRAPHICS:
{json.dumps(handoff_context.patient_context.demographics, indent=2)}

Please provide your specialist consultation based on this clinical information.
"""
        return patient_summary
    
    async def get_specialist_response(self, 
                                    handoff_id: str,
                                    additional_questions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get response from specialist agent"""
        
        if handoff_id not in self.active_handoffs:
            raise ValueError(f"Handoff {handoff_id} not found")
        
        handoff_context = self.active_handoffs[handoff_id]
        target_specialty = MedicalSpecialty(handoff_context.target_agent)
        
        if target_specialty not in self.specialist_agents:
            raise ValueError(f"Specialist {target_specialty.value} not registered")
        
        specialist_agent = self.specialist_agents[target_specialty]
        thread = self.active_threads[handoff_id]
        
        # Add additional questions if provided
        if additional_questions:
            for question in additional_questions:
                await asyncio.to_thread(
                    self.client.beta.threads.messages.create,
                    thread_id=thread.id,
                    role="user",
                    content=question
                )
        
        # Create run with specialist assistant
        run = await asyncio.to_thread(
            self.client.beta.threads.runs.create,
            thread_id=thread.id,
            assistant_id=specialist_agent.assistant_id
        )
        
        # Wait for completion
        while True:
            run_status = await asyncio.to_thread(
                self.client.beta.threads.runs.retrieve,
                thread_id=thread.id,
                run_id=run.id
            )
            
            if run_status.status == "completed":
                break
            elif run_status.status in ["failed", "expired", "cancelled"]:
                raise Exception(f"Run failed with status: {run_status.status}")
            
            await asyncio.sleep(1)
        
        # Get messages
        messages = await asyncio.to_thread(
            self.client.beta.threads.messages.list,
            thread_id=thread.id,
            order="desc",
            limit=10
        )
        
        # Extract assistant response
        assistant_messages = [
            msg for msg in messages.data 
            if msg.role == "assistant"
        ]
        
        if not assistant_messages:
            raise Exception("No response from specialist")
        
        latest_response = assistant_messages[0]
        response_text = latest_response.content[0].text.value if latest_response.content else ""
        
        return {
            "handoff_id": handoff_id,
            "specialist": target_specialty.value,
            "response": response_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run.id,
            "thread_id": thread.id
        }
    
    async def complete_handoff(self, handoff_id: str, outcome: str = "completed") -> Dict[str, Any]:
        """Complete a handoff and clean up resources"""
        
        if handoff_id not in self.active_handoffs:
            raise ValueError(f"Handoff {handoff_id} not found")
        
        handoff_context = self.active_handoffs[handoff_id]
        
        # Update handoff status
        completion_data = {
            "handoff_id": handoff_id,
            "outcome": outcome,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration": (datetime.now(timezone.utc) - handoff_context.timestamp).total_seconds()
        }
        
        # Clean up active resources
        del self.active_handoffs[handoff_id]
        if handoff_id in self.active_threads:
            del self.active_threads[handoff_id]
        
        logger.info(f"Completed handoff {handoff_id} with outcome: {outcome}")
        return completion_data
    
    async def get_specialist_recommendations(self, 
                                           clinical_text: str,
                                           patient_context: PatientContext) -> Dict[str, Any]:
        """Get specialist recommendations for a clinical case"""
        
        # Route to specialists
        recommended_specialties = self.route_to_specialist(clinical_text, patient_context)
        
        recommendations = []
        for specialty in recommended_specialties[:3]:  # Limit to top 3
            if specialty in self.specialist_agents:
                specialist = self.specialist_agents[specialty]
                recommendations.append({
                    "specialty": specialty.value,
                    "name": specialist.name,
                    "description": specialist.description,
                    "confidence": self._calculate_routing_confidence(clinical_text, specialty),
                    "emergency_capable": specialist.emergency_protocols
                })
        
        return {
            "recommendations": recommendations,
            "routing_keywords": self._extract_routing_keywords(clinical_text),
            "patient_factors": {
                "age_group": self._get_age_group(patient_context.demographics.get("age")),
                "priority": self._assess_clinical_priority(clinical_text)
            }
        }
    
    def _calculate_routing_confidence(self, clinical_text: str, specialty: MedicalSpecialty) -> float:
        """Calculate confidence score for routing to specialty"""
        clinical_text_lower = clinical_text.lower()
        specialty_keywords = [
            keyword for keyword, specialties in self.routing_rules.items()
            if specialty in specialties
        ]
        
        matches = sum(1 for keyword in specialty_keywords if keyword in clinical_text_lower)
        total_keywords = len(specialty_keywords)
        
        return min(matches / max(total_keywords, 1), 1.0) if total_keywords > 0 else 0.0
    
    def _extract_routing_keywords(self, clinical_text: str) -> List[str]:
        """Extract routing keywords from clinical text"""
        clinical_text_lower = clinical_text.lower()
        found_keywords = [
            keyword for keyword in self.routing_rules.keys()
            if keyword in clinical_text_lower
        ]
        return found_keywords
    
    def _get_age_group(self, age: Optional[int]) -> str:
        """Categorize patient by age group"""
        if not age:
            return "unknown"
        if age < 18:
            return "pediatric"
        elif age > 65:
            return "geriatric"
        else:
            return "adult"
    
    def _assess_clinical_priority(self, clinical_text: str) -> str:
        """Assess clinical priority from text"""
        clinical_text_lower = clinical_text.lower()
        
        emergency_keywords = ["emergency", "urgent", "critical", "life-threatening", "code", "stat"]
        high_priority_keywords = ["severe", "acute", "sudden", "rapid"]
        
        if any(keyword in clinical_text_lower for keyword in emergency_keywords):
            return "emergency"
        elif any(keyword in clinical_text_lower for keyword in high_priority_keywords):
            return "high"
        else:
            return "routine"
    
    async def setup_default_specialists(self):
        """Set up default specialist agents"""
        
        default_specialists = [
            {
                "specialty": MedicalSpecialty.GENERAL_PRACTICE,
                "name": "Dr. Primary Care",
                "instructions": """You are a primary care physician providing comprehensive healthcare.
                Focus on preventive care, health maintenance, and coordination of specialist referrals.
                Consider the whole patient and provide evidence-based recommendations."""
            },
            {
                "specialty": MedicalSpecialty.CARDIOLOGY,
                "name": "Dr. Heart Specialist",
                "instructions": """You are a cardiologist specializing in heart and cardiovascular conditions.
                Provide expert guidance on cardiac symptoms, diagnostic workup, and treatment recommendations.
                Consider both acute and chronic cardiovascular conditions."""
            },
            {
                "specialty": MedicalSpecialty.EMERGENCY_MEDICINE,
                "name": "Dr. Emergency Physician",
                "instructions": """You are an emergency medicine physician focused on acute care.
                Prioritize life-threatening conditions, provide rapid assessment and stabilization guidance.
                Use ABCDE approach and consider immediate interventions."""
            },
            {
                "specialty": MedicalSpecialty.PHARMACY,
                "name": "Dr. Clinical Pharmacist",
                "instructions": """You are a clinical pharmacist specializing in medication management.
                Focus on drug interactions, dosing, adverse effects, and therapeutic optimization.
                Provide medication reconciliation and safety recommendations."""
            }
        ]
        
        for spec_config in default_specialists:
            try:
                await self.register_specialist_agent(
                    specialty=spec_config["specialty"],
                    name=spec_config["name"],
                    instructions=spec_config["instructions"]
                )
                logger.info(f"Set up default specialist: {spec_config['name']}")
            except Exception as e:
                logger.error(f"Failed to set up {spec_config['name']}: {e}")
    
    def get_framework_status(self) -> Dict[str, Any]:
        """Get framework status and statistics"""
        return {
            "registered_specialists": len(self.specialist_agents),
            "active_handoffs": len(self.active_handoffs),
            "total_handoffs": len(self.handoff_history),
            "active_threads": len(self.active_threads),
            "specialties": [specialty.value for specialty in self.specialist_agents.keys()],
            "routing_rules_count": len(self.routing_rules)
        }


# Example usage
async def healthcare_agent_framework_example():
    """Example of using the Healthcare Agent Framework"""
    
    # Initialize framework
    framework = HealthcareAgentFramework(
        openai_api_key="your-openai-api-key"  # Replace with actual API key
    )
    
    # Set up default specialists
    await framework.setup_default_specialists()
    
    # Example patient context
    patient_context = PatientContext(
        patient_id="PT12345",
        mrn="MRN-12345",
        demographics={"age": 65, "gender": "male"}
    )
    
    # Example clinical context
    clinical_context = ClinicalContext(
        chief_complaint="Chest pain and shortness of breath",
        clinical_notes="Patient reports substernal chest pain with exertion",
        medications=["metoprolol", "atorvastatin"],
        allergies=["penicillin"],
        urgency=ClinicalUrgency.HIGH,
        relevant_history="History of hypertension and hyperlipidemia"
    )
    
    # Get specialist recommendations
    recommendations = await framework.get_specialist_recommendations(
        "chest pain shortness of breath elderly male",
        patient_context
    )
    print(f"Specialist recommendations: {recommendations}")
    
    # Initiate handoff to cardiology
    handoff_id = await framework.initiate_handoff(
        source_specialty=MedicalSpecialty.GENERAL_PRACTICE,
        target_specialty=MedicalSpecialty.CARDIOLOGY,
        patient_context=patient_context,
        clinical_context=clinical_context,
        handoff_reason="Chest pain evaluation and cardiac workup",
        handoff_type="consultation"
    )
    
    print(f"Handoff initiated: {handoff_id}")
    
    # Get specialist response
    response = await framework.get_specialist_response(handoff_id)
    print(f"Cardiology consultation: {response}")
    
    # Complete handoff
    completion = await framework.complete_handoff(handoff_id, "completed")
    print(f"Handoff completed: {completion}")


if __name__ == "__main__":
    # Run example
    asyncio.run(healthcare_agent_framework_example())