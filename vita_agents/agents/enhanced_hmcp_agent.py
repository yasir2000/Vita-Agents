#!/usr/bin/env python3
"""
Enhanced Healthcare Model Context Protocol (HMCP) Implementation

This enhanced implementation follows the MCP pattern but specifically designed 
for healthcare workflows, supporting the multi-step conversational patterns
shown in healthcare agent interactions.

Key Features:
- Context preservation across multi-step conversations
- Iterative information gathering workflows
- Healthcare-specific state management
- Clinical decision support integration
- Care coordination and scheduling workflows
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

from vita_agents.core.agent import HealthcareAgent
from vita_agents.core.security import HIPAACompliantAgent
from vita_agents.protocols.hmcp import (
    HMCPMessage, HMCPMessageType, ClinicalUrgency, HealthcareRole,
    PatientContext, ClinicalContext, SecurityContext
)

logger = logging.getLogger(__name__)


class WorkflowState(str, Enum):
    """Healthcare workflow states"""
    INITIATED = "initiated"
    GATHERING_INFO = "gathering_info" 
    ANALYZING = "analyzing"
    CONSULTING = "consulting"
    RECOMMENDING = "recommending"
    COORDINATING = "coordinating"
    SCHEDULING = "scheduling"
    COMPLETED = "completed"
    PAUSED = "paused"
    ERROR = "error"


class ConversationContext:
    """Context for multi-step healthcare conversations"""
    
    def __init__(self, conversation_id: str, patient_id: str, initiator: str):
        self.conversation_id = conversation_id
        self.patient_id = patient_id
        self.initiator = initiator
        self.state = WorkflowState.INITIATED
        self.created_at = datetime.now(timezone.utc)
        self.last_activity = self.created_at
        
        # Healthcare-specific context
        self.symptoms: List[str] = []
        self.patient_identifiers: Dict[str, Any] = {}
        self.clinical_data: Dict[str, Any] = {}
        self.medications: List[Dict[str, Any]] = []
        self.allergies: List[str] = []
        self.vital_signs: Dict[str, Any] = {}
        
        # Workflow tracking
        self.information_needed: List[str] = []
        self.information_gathered: List[str] = []
        self.recommendations: List[Dict[str, Any]] = []
        self.care_plan: Dict[str, Any] = {}
        self.scheduled_appointments: List[Dict[str, Any]] = []
        
        # Agent participation
        self.participating_agents: Set[str] = set()
        self.agent_responses: Dict[str, List[Dict[str, Any]]] = {}
        
        # Message history
        self.message_history: List[HMCPMessage] = []
    
    def add_message(self, message: HMCPMessage):
        """Add message to conversation history"""
        self.message_history.append(message)
        self.last_activity = datetime.now(timezone.utc)
        
        if message.sender_id:
            self.participating_agents.add(message.sender_id)
            
        # Extract healthcare information from message content
        self._extract_healthcare_info(message)
    
    def _extract_healthcare_info(self, message: HMCPMessage):
        """Extract and update healthcare information from message"""
        content = message.content
        
        # Extract symptoms
        if "symptoms" in content:
            new_symptoms = content["symptoms"]
            if isinstance(new_symptoms, list):
                self.symptoms.extend([s for s in new_symptoms if s not in self.symptoms])
            elif isinstance(new_symptoms, str):
                if new_symptoms not in self.symptoms:
                    self.symptoms.append(new_symptoms)
        
        # Extract patient identifiers
        if "patient_identifiers" in content:
            self.patient_identifiers.update(content["patient_identifiers"])
        
        # Extract clinical data
        if "clinical_data" in content:
            self.clinical_data.update(content["clinical_data"])
        
        # Extract vital signs
        if "vital_signs" in content:
            self.vital_signs.update(content["vital_signs"])
        
        # Extract medications
        if "medications" in content:
            new_meds = content["medications"]
            if isinstance(new_meds, list):
                self.medications.extend(new_meds)
        
        # Extract allergies
        if "allergies" in content:
            new_allergies = content["allergies"]
            if isinstance(new_allergies, list):
                self.allergies.extend([a for a in new_allergies if a not in self.allergies])
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get complete context summary for agents"""
        return {
            "conversation_id": self.conversation_id,
            "patient_id": self.patient_id,
            "state": self.state,
            "symptoms": self.symptoms,
            "patient_identifiers": self.patient_identifiers,
            "clinical_data": self.clinical_data,
            "vital_signs": self.vital_signs,
            "medications": self.medications,
            "allergies": self.allergies,
            "information_needed": self.information_needed,
            "information_gathered": self.information_gathered,
            "recommendations": self.recommendations,
            "care_plan": self.care_plan,
            "participating_agents": list(self.participating_agents),
            "duration_minutes": (datetime.now(timezone.utc) - self.created_at).total_seconds() / 60
        }


class EnhancedHMCPAgent(HIPAACompliantAgent, HealthcareAgent):
    """
    Enhanced Healthcare Model Context Protocol Agent
    
    Implements MCP-like patterns for healthcare with:
    - Multi-step conversational workflows
    - Context preservation across interactions
    - Healthcare-specific state management
    - Iterative information gathering
    - Clinical decision support integration
    """
    
    def __init__(self, agent_id: str, settings: Dict[str, Any]):
        super().__init__(agent_id, settings)
        
        # Enhanced HMCP capabilities
        self.agent_type = settings.get("agent_type", "general")  # diagnostic, patient_data, medical_knowledge, scheduling
        self.specialized_functions = self._init_specialized_functions()
        
        # Conversation management
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.conversation_timeouts: Dict[str, datetime] = {}
        
        # Healthcare-specific handlers
        self.workflow_handlers = {
            "symptom_analysis": self._handle_symptom_analysis_workflow,
            "patient_identification": self._handle_patient_identification_workflow,
            "clinical_consultation": self._handle_clinical_consultation_workflow,
            "care_coordination": self._handle_care_coordination_workflow,
            "appointment_scheduling": self._handle_appointment_scheduling_workflow
        }
        
        # Information gathering templates
        self.info_gathering_templates = {
            "patient_identifiers": [
                "date_of_birth", "medical_record_number", "social_security_number",
                "phone_number", "address", "emergency_contact"
            ],
            "clinical_history": [
                "chief_complaint", "history_of_present_illness", "past_medical_history",
                "family_history", "social_history", "review_of_systems"
            ],
            "medications": [
                "current_medications", "allergies", "adverse_reactions",
                "dosages", "adherence", "recent_changes"
            ]
        }
    
    def _init_specialized_functions(self) -> Dict[str, Callable]:
        """Initialize agent-specific functions based on agent type"""
        base_functions = {
            "process_symptoms": self._process_symptoms,
            "validate_patient_data": self._validate_patient_data,
            "provide_clinical_guidance": self._provide_clinical_guidance,
            "coordinate_care": self._coordinate_care
        }
        
        if self.agent_type == "diagnostic":
            base_functions.update({
                "analyze_symptoms": self._analyze_symptoms,
                "generate_differential_diagnosis": self._generate_differential_diagnosis,
                "request_additional_tests": self._request_additional_tests,
                "provide_diagnostic_reasoning": self._provide_diagnostic_reasoning
            })
        elif self.agent_type == "patient_data":
            base_functions.update({
                "retrieve_patient_records": self._retrieve_patient_records,
                "validate_patient_identity": self._validate_patient_identity,
                "request_missing_identifiers": self._request_missing_identifiers,
                "consolidate_patient_data": self._consolidate_patient_data
            })
        elif self.agent_type == "medical_knowledge":
            base_functions.update({
                "search_medical_literature": self._search_medical_literature,
                "provide_treatment_guidelines": self._provide_treatment_guidelines,
                "check_drug_interactions": self._check_drug_interactions,
                "suggest_clinical_protocols": self._suggest_clinical_protocols
            })
        elif self.agent_type == "scheduling":
            base_functions.update({
                "check_availability": self._check_availability,
                "schedule_appointment": self._schedule_appointment,
                "coordinate_resources": self._coordinate_resources,
                "send_appointment_reminders": self._send_appointment_reminders
            })
        
        return base_functions
    
    async def process_hmcp_message(self, message: HMCPMessage) -> HMCPMessage:
        """Enhanced message processing with conversation context"""
        try:
            # Get or create conversation context
            conversation_id = message.content.get("conversation_id")
            if not conversation_id and message.patient_context:
                conversation_id = f"conv_{message.patient_context.patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if conversation_id and conversation_id not in self.active_conversations:
                self.active_conversations[conversation_id] = ConversationContext(
                    conversation_id=conversation_id,
                    patient_id=message.patient_context.patient_id if message.patient_context else "unknown",
                    initiator=message.sender_id
                )
            
            context = self.active_conversations.get(conversation_id)
            if context:
                context.add_message(message)
            
            # Process message based on type and workflow
            if message.type == HMCPMessageType.REQUEST:
                return await self._handle_workflow_request(message, context)
            elif message.type == HMCPMessageType.COORDINATION:
                return await self._handle_workflow_coordination(message, context)
            elif message.type == HMCPMessageType.NOTIFICATION:
                return await self._handle_workflow_notification(message, context)
            else:
                return await self._handle_general_message(message, context)
                
        except Exception as e:
            logger.error(f"Error processing HMCP message: {e}")
            return self._create_error_response(message, str(e))
    
    async def _handle_workflow_request(self, message: HMCPMessage, context: Optional[ConversationContext]) -> HMCPMessage:
        """Handle workflow-based requests with context"""
        action = message.content.get("action", "unknown")
        
        # Route to appropriate workflow handler
        if action.startswith("symptom"):
            return await self._handle_symptom_analysis_workflow(message, context)
        elif action.startswith("patient"):
            return await self._handle_patient_identification_workflow(message, context)
        elif action.startswith("clinical"):
            return await self._handle_clinical_consultation_workflow(message, context)
        elif action.startswith("schedule"):
            return await self._handle_appointment_scheduling_workflow(message, context)
        else:
            return await self._handle_general_workflow(message, context)
    
    async def _handle_symptom_analysis_workflow(self, message: HMCPMessage, context: Optional[ConversationContext]) -> HMCPMessage:
        """Handle symptom analysis workflow like in the diagram"""
        content = message.content
        action = content.get("action", "")
        
        if action == "provide_initial_symptoms":
            # Step 1: Physician provides initial symptoms
            symptoms = content.get("symptoms", [])
            if context:
                context.symptoms.extend(symptoms)
                context.state = WorkflowState.GATHERING_INFO
            
            # Request basic patient records
            response_content = {
                "status": "symptoms_received",
                "next_action": "request_patient_records",
                "message": "Initial symptoms recorded. Requesting basic patient records.",
                "symptoms_recorded": symptoms,
                "conversation_id": context.conversation_id if context else None
            }
            
        elif action == "request_patient_records":
            # Step 2: Request basic patient records
            response_content = {
                "status": "requesting_patient_records",
                "required_identifiers": self.info_gathering_templates["patient_identifiers"],
                "message": "Need additional patient identifiers to retrieve records.",
                "conversation_id": context.conversation_id if context else None
            }
            
        elif action == "provide_patient_identifiers":
            # Step 3: Receive patient identifiers
            identifiers = content.get("identifiers", {})
            if context:
                context.patient_identifiers.update(identifiers)
                context.information_gathered.append("patient_identifiers")
            
            response_content = {
                "status": "identifiers_received",
                "next_action": "retrieve_patient_records",
                "message": "Patient identifiers received. Retrieving medical records.",
                "conversation_id": context.conversation_id if context else None
            }
            
        elif action == "analyze_symptoms_with_records":
            # Step 4: Analyze symptoms with patient records
            if context:
                context.state = WorkflowState.ANALYZING
            
            # Perform analysis (this would integrate with actual diagnostic systems)
            analysis_result = await self._perform_symptom_analysis(
                symptoms=context.symptoms if context else content.get("symptoms", []),
                patient_data=context.clinical_data if context else content.get("patient_data", {}),
                context=context
            )
            
            response_content = {
                "status": "analysis_completed",
                "analysis_result": analysis_result,
                "next_action": "request_medical_knowledge",
                "conversation_id": context.conversation_id if context else None
            }
            
        else:
            response_content = {
                "status": "unknown_action",
                "message": f"Unknown symptom analysis action: {action}",
                "conversation_id": context.conversation_id if context else None
            }
        
        return self._create_workflow_response(message, response_content, context)
    
    async def _handle_patient_identification_workflow(self, message: HMCPMessage, context: Optional[ConversationContext]) -> HMCPMessage:
        """Handle patient identification and data retrieval workflow"""
        content = message.content
        action = content.get("action", "")
        
        if action == "request_basic_patient_records":
            # Check if we have sufficient identifiers
            required_ids = ["date_of_birth", "medical_record_number"]
            missing_ids = []
            
            if context:
                provided_ids = context.patient_identifiers.keys()
                missing_ids = [id for id in required_ids if id not in provided_ids]
            
            if missing_ids:
                response_content = {
                    "status": "missing_identifiers",
                    "missing_identifiers": missing_ids,
                    "message": f"Need additional identifiers: {', '.join(missing_ids)}",
                    "conversation_id": context.conversation_id if context else None
                }
            else:
                # Retrieve records
                patient_records = await self._retrieve_patient_records_internal(context)
                response_content = {
                    "status": "records_retrieved",
                    "patient_records": patient_records,
                    "next_action": "forward_to_diagnosis",
                    "conversation_id": context.conversation_id if context else None
                }
        
        elif action == "provide_additional_identifiers":
            identifiers = content.get("identifiers", {})
            if context:
                context.patient_identifiers.update(identifiers)
            
            response_content = {
                "status": "identifiers_updated",
                "message": "Additional identifiers received.",
                "next_action": "retry_record_retrieval",
                "conversation_id": context.conversation_id if context else None
            }
        
        else:
            response_content = {
                "status": "unknown_action", 
                "message": f"Unknown patient identification action: {action}",
                "conversation_id": context.conversation_id if context else None
            }
        
        return self._create_workflow_response(message, response_content, context)
    
    async def _handle_clinical_consultation_workflow(self, message: HMCPMessage, context: Optional[ConversationContext]) -> HMCPMessage:
        """Handle clinical consultation and knowledge retrieval"""
        content = message.content
        action = content.get("action", "")
        
        if action == "request_relevant_articles":
            # Search medical knowledge based on symptoms/records
            symptoms = content.get("symptoms", context.symptoms if context else [])
            articles = await self._search_medical_literature_internal(symptoms, context)
            
            response_content = {
                "status": "articles_found",
                "relevant_articles": articles,
                "next_action": "request_more_details",
                "conversation_id": context.conversation_id if context else None
            }
            
        elif action == "request_more_specific_symptoms":
            # Request additional clinical details
            if context:
                context.state = WorkflowState.GATHERING_INFO
                context.information_needed.append("specific_symptoms")
            
            response_content = {
                "status": "requesting_details",
                "requested_information": [
                    "symptom_onset", "symptom_duration", "severity_scale",
                    "aggravating_factors", "relieving_factors", "associated_symptoms"
                ],
                "message": "Need more specific symptom details for accurate diagnosis.",
                "conversation_id": context.conversation_id if context else None
            }
            
        elif action == "provide_clinical_recommendations":
            # Generate clinical recommendations
            if context:
                context.state = WorkflowState.RECOMMENDING
            
            recommendations = await self._generate_clinical_recommendations(content, context)
            
            response_content = {
                "status": "recommendations_ready",
                "clinical_recommendations": recommendations,
                "next_action": "confirm_care_plan",
                "conversation_id": context.conversation_id if context else None
            }
        
        else:
            response_content = {
                "status": "unknown_action",
                "message": f"Unknown clinical consultation action: {action}",
                "conversation_id": context.conversation_id if context else None
            }
        
        return self._create_workflow_response(message, response_content, context)
    
    async def _handle_appointment_scheduling_workflow(self, message: HMCPMessage, context: Optional[ConversationContext]) -> HMCPMessage:
        """Handle appointment scheduling workflow"""
        content = message.content
        action = content.get("action", "")
        
        if action == "request_to_schedule_appointment":
            # Check patient availability
            if context:
                context.state = WorkflowState.SCHEDULING
            
            response_content = {
                "status": "checking_availability",
                "message": "Checking patient and provider availability.",
                "next_action": "report_availability",
                "conversation_id": context.conversation_id if context else None
            }
            
        elif action == "report_patient_availability":
            # Receive patient availability
            availability = content.get("availability", {})
            if context:
                context.care_plan["patient_availability"] = availability
            
            response_content = {
                "status": "availability_received",
                "next_action": "provide_appointment_options",
                "message": "Patient availability recorded. Providing appointment options.",
                "conversation_id": context.conversation_id if context else None
            }
            
        elif action == "provide_preferences":
            # Finalize appointment
            preferences = content.get("preferences", {})
            appointment = await self._schedule_appointment_internal(preferences, context)
            
            if context:
                context.scheduled_appointments.append(appointment)
                context.state = WorkflowState.COMPLETED
            
            response_content = {
                "status": "appointment_scheduled",
                "appointment_details": appointment,
                "confirmation": "Appointment confirmed and notifications sent.",
                "conversation_id": context.conversation_id if context else None
            }
        
        else:
            response_content = {
                "status": "unknown_action",
                "message": f"Unknown scheduling action: {action}",
                "conversation_id": context.conversation_id if context else None
            }
        
        return self._create_workflow_response(message, response_content, context)
    
    # Helper methods for actual implementation
    
    async def _perform_symptom_analysis(self, symptoms: List[str], patient_data: Dict[str, Any], context: Optional[ConversationContext]) -> Dict[str, Any]:
        """Perform actual symptom analysis"""
        # This would integrate with real diagnostic systems
        return {
            "primary_symptoms": symptoms[:3],
            "symptom_severity": "moderate",
            "preliminary_assessment": "requires_further_evaluation",
            "recommended_tests": ["blood_work", "imaging"],
            "urgency_level": "routine"
        }
    
    async def _retrieve_patient_records_internal(self, context: Optional[ConversationContext]) -> Dict[str, Any]:
        """Retrieve patient medical records"""
        # This would integrate with actual EHR systems
        return {
            "patient_id": context.patient_id if context else "unknown",
            "basic_demographics": context.patient_identifiers if context else {},
            "medical_history": ["hypertension", "diabetes_type_2"],
            "current_medications": ["metformin", "lisinopril"],
            "allergies": ["penicillin"],
            "last_visit": "2024-10-01"
        }
    
    async def _search_medical_literature_internal(self, symptoms: List[str], context: Optional[ConversationContext]) -> List[Dict[str, Any]]:
        """Search medical literature and guidelines"""
        # This would integrate with medical knowledge databases
        return [
            {
                "title": "Clinical Guidelines for Chest Pain Evaluation",
                "relevance_score": 0.95,
                "source": "American College of Cardiology",
                "summary": "Evidence-based approach to chest pain assessment"
            },
            {
                "title": "Differential Diagnosis of Acute Chest Pain",
                "relevance_score": 0.87,
                "source": "New England Journal of Medicine",
                "summary": "Systematic approach to chest pain differential diagnosis"
            }
        ]
    
    async def _generate_clinical_recommendations(self, content: Dict[str, Any], context: Optional[ConversationContext]) -> List[Dict[str, Any]]:
        """Generate clinical recommendations"""
        return [
            {
                "recommendation": "Order 12-lead ECG",
                "priority": "high",
                "rationale": "Rule out acute coronary syndrome"
            },
            {
                "recommendation": "Obtain cardiac enzymes",
                "priority": "high", 
                "rationale": "Assess for myocardial injury"
            },
            {
                "recommendation": "Schedule cardiology consultation",
                "priority": "medium",
                "rationale": "Specialist evaluation recommended"
            }
        ]
    
    async def _schedule_appointment_internal(self, preferences: Dict[str, Any], context: Optional[ConversationContext]) -> Dict[str, Any]:
        """Schedule appointment with preferences"""
        return {
            "appointment_id": f"APT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "appointment_date": "2024-10-25",
            "appointment_time": "10:00 AM",
            "provider": "Dr. Smith, Cardiology",
            "location": "Main Hospital, Suite 200",
            "appointment_type": "consultation",
            "duration_minutes": 30
        }
    
    def _create_workflow_response(self, original_message: HMCPMessage, content: Dict[str, Any], context: Optional[ConversationContext]) -> HMCPMessage:
        """Create workflow response message"""
        return HMCPMessage(
            type=HMCPMessageType.RESPONSE,
            sender_id=self.agent_id,
            receiver_id=original_message.sender_id,
            correlation_id=original_message.id,
            content=content,
            patient_context=original_message.patient_context,
            clinical_context=original_message.clinical_context,
            security_context=SecurityContext(
                user_id=self.agent_id,
                role=self.healthcare_role,
                phi_flag=True
            )
        )
    
    def _create_error_response(self, original_message: HMCPMessage, error: str) -> HMCPMessage:
        """Create error response message"""
        return HMCPMessage(
            type=HMCPMessageType.RESPONSE,
            sender_id=self.agent_id,
            receiver_id=original_message.sender_id,
            correlation_id=original_message.id,
            content={"error": error, "status": "failed"},
            urgency=ClinicalUrgency.URGENT
        )
    
    # Additional specialized methods would be implemented here for each agent type
    async def _analyze_symptoms(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnostic agent symptom analysis"""
        pass
    
    async def _validate_patient_identity(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Patient data agent identity validation"""
        pass
    
    async def _provide_treatment_guidelines(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Medical knowledge agent treatment guidelines"""
        pass
    
    async def _check_availability(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Scheduling agent availability check"""
        pass
    
    def get_conversation_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of active conversation"""
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id].get_context_summary()
        return None
    
    def cleanup_expired_conversations(self, timeout_hours: int = 24):
        """Clean up expired conversations"""
        now = datetime.now(timezone.utc)
        expired = []
        
        for conv_id, context in self.active_conversations.items():
            if (now - context.last_activity).total_seconds() > (timeout_hours * 3600):
                expired.append(conv_id)
        
        for conv_id in expired:
            del self.active_conversations[conv_id]
            logger.info(f"Cleaned up expired conversation: {conv_id}")