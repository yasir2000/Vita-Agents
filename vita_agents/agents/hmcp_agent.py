#!/usr/bin/env python3
"""
HMCP Agent Implementation for Vita Agents

Healthcare Model Context Protocol agent that enables seamless
communication between healthcare AI agents with clinical context awareness,
security, and workflow orchestration capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Callable, Union

from vita_agents.core.agent import HealthcareAgent
from vita_agents.core.security import HIPAACompliantAgent
from vita_agents.protocols.hmcp import (
    HMCPMessage, HMCPMessageType, HMCPClient, HMCPServer, HMCPProtocol,
    ClinicalUrgency, HealthcareRole, PatientContext, ClinicalContext, 
    SecurityContext, hmcp_protocol
)

logger = logging.getLogger(__name__)


class HMCPAgent(HIPAACompliantAgent, HealthcareAgent):
    """
    Healthcare Model Context Protocol Agent
    
    Enables healthcare agents to communicate using HMCP with:
    - Clinical context awareness
    - Healthcare-specific message routing
    - Emergency response protocols
    - Multi-agent workflow coordination
    - Security and audit compliance
    """
    
    def __init__(self, agent_id: str, settings: Dict[str, Any]):
        super().__init__(agent_id, settings)
        
        self.hmcp_client: Optional[HMCPClient] = None
        self.hmcp_server: Optional[HMCPServer] = None
        self.supported_protocols = ["HMCP", "FHIR_MESSAGING", "HL7_COMMUNICATION"]
        self.capabilities = settings.get("capabilities", [])
        self.healthcare_role = HealthcareRole(settings.get("role", "ai_agent"))
        self.emergency_capable = settings.get("emergency_capable", False)
        
        # Message handling
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.active_conversations: Dict[str, List[HMCPMessage]] = {}
        self.workflow_handlers: Dict[str, Callable] = {}
        
        # Performance metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "emergency_responses": 0,
            "workflow_completions": 0,
            "average_response_time": 0.0,
            "error_count": 0
        }
        
        # Initialize HMCP components
        self._initialize_hmcp()
    
    def _initialize_hmcp(self):
        """Initialize HMCP client and server components"""
        try:
            # Create HMCP client and server
            self.hmcp_client = hmcp_protocol.create_client(self.agent_id)
            self.hmcp_server = hmcp_protocol.create_server(
                self.agent_id, 
                port=8080 + hash(self.agent_id) % 1000
            )
            
            # Register with router
            hmcp_protocol.router.register_agent(
                self.agent_id,
                self.capabilities,
                self.healthcare_role,
                self.emergency_capable
            )
            
            # Set up message handlers
            self._setup_message_handlers()
            
            logger.info(f"HMCP Agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize HMCP for agent {self.agent_id}: {e}")
            raise
    
    def _setup_message_handlers(self):
        """Set up HMCP message handlers"""
        if not self.hmcp_server:
            return
            
        self.hmcp_server.register_handler(
            HMCPMessageType.REQUEST, self._handle_request
        )
        self.hmcp_server.register_handler(
            HMCPMessageType.NOTIFICATION, self._handle_notification
        )
        self.hmcp_server.register_handler(
            HMCPMessageType.EMERGENCY, self._handle_emergency
        )
        self.hmcp_server.register_handler(
            HMCPMessageType.COORDINATION, self._handle_coordination
        )
        self.hmcp_server.register_handler(
            HMCPMessageType.EVENT, self._handle_event
        )
    
    async def _handle_request(self, message: HMCPMessage) -> HMCPMessage:
        """Handle incoming HMCP request messages"""
        start_time = datetime.now()
        
        try:
            self.metrics["messages_received"] += 1
            
            # Log clinical context
            if message.patient_context:
                logger.info(f"Processing request for patient {message.patient_context.patient_id}")
            
            # Process based on content
            content = message.content
            action = content.get("action", "unknown")
            
            response_content = {}
            
            if action == "validate_fhir":
                response_content = await self._validate_fhir_resource(content.get("resource"))
            elif action == "process_hl7":
                response_content = await self._process_hl7_message(content.get("message"))
            elif action == "clinical_decision_support":
                response_content = await self._provide_clinical_decision_support(content)
            elif action == "medication_check":
                response_content = await self._check_medications(content)
            else:
                response_content = await self._process_healthcare_data(content)
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_response_time_metric(response_time)
            
            return HMCPMessage(
                type=HMCPMessageType.RESPONSE,
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                correlation_id=message.id,
                content=response_content,
                patient_context=message.patient_context,
                clinical_context=message.clinical_context,
                security_context=SecurityContext(
                    user_id=self.agent_id,
                    role=self.healthcare_role,
                    phi_flag=response_content.get("contains_phi", False)
                )
            )
            
        except Exception as e:
            self.metrics["error_count"] += 1
            logger.error(f"Error handling request {message.id}: {e}")
            
            return HMCPMessage(
                type=HMCPMessageType.RESPONSE,
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                correlation_id=message.id,
                content={"error": str(e), "status": "failed"},
                urgency=ClinicalUrgency.URGENT
            )
    
    async def _handle_notification(self, message: HMCPMessage) -> HMCPMessage:
        """Handle incoming HMCP notification messages"""
        try:
            self.metrics["messages_received"] += 1
            
            notification_type = message.content.get("type", "general")
            
            if notification_type == "critical_lab_value":
                await self._handle_critical_lab_notification(message)
            elif notification_type == "medication_alert":
                await self._handle_medication_alert(message)
            elif notification_type == "vital_signs_alert":
                await self._handle_vital_signs_alert(message)
            elif notification_type == "care_plan_update":
                await self._handle_care_plan_update(message)
            
            logger.info(f"Processed {notification_type} notification for agent {self.agent_id}")
            
            return HMCPMessage(
                type=HMCPMessageType.RESPONSE,
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content={"status": "notification_processed", "type": notification_type}
            )
            
        except Exception as e:
            logger.error(f"Error handling notification {message.id}: {e}")
            raise
    
    async def _handle_emergency(self, message: HMCPMessage) -> HMCPMessage:
        """Handle emergency HMCP messages"""
        try:
            self.metrics["emergency_responses"] += 1
            
            emergency_type = message.content.get("emergency_type", "unknown")
            patient_id = message.patient_context.patient_id if message.patient_context else "unknown"
            location = message.content.get("location", "unknown")
            
            logger.critical(f"EMERGENCY: {emergency_type} for patient {patient_id} at {location}")
            
            # Process emergency based on type
            if emergency_type == "cardiac_arrest":
                response = await self._handle_cardiac_emergency(message)
            elif emergency_type == "respiratory_failure":
                response = await self._handle_respiratory_emergency(message)
            elif emergency_type == "stroke_alert":
                response = await self._handle_stroke_emergency(message)
            elif emergency_type == "sepsis_alert":
                response = await self._handle_sepsis_emergency(message)
            else:
                response = await self._handle_general_emergency(message)
            
            return HMCPMessage(
                type=HMCPMessageType.RESPONSE,
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                urgency=ClinicalUrgency.EMERGENCY,
                content=response,
                patient_context=message.patient_context
            )
            
        except Exception as e:
            logger.error(f"Error handling emergency {message.id}: {e}")
            raise
    
    async def _handle_coordination(self, message: HMCPMessage) -> HMCPMessage:
        """Handle care coordination messages"""
        try:
            coordination_type = message.content.get("coordination_type", "general")
            
            if coordination_type == "multidisciplinary_care":
                response = await self._coordinate_multidisciplinary_care(message)
            elif coordination_type == "discharge_planning":
                response = await self._coordinate_discharge_planning(message)
            elif coordination_type == "transfer_of_care":
                response = await self._coordinate_transfer_of_care(message)
            else:
                response = {"status": "coordination_acknowledged", "type": coordination_type}
            
            return HMCPMessage(
                type=HMCPMessageType.RESPONSE,
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content=response,
                workflow_id=message.workflow_id,
                patient_context=message.patient_context
            )
            
        except Exception as e:
            logger.error(f"Error handling coordination {message.id}: {e}")
            raise
    
    async def _handle_event(self, message: HMCPMessage) -> HMCPMessage:
        """Handle healthcare event messages"""
        try:
            event_type = message.content.get("event_type", "unknown")
            
            if event_type == "patient_admission":
                await self._handle_patient_admission_event(message)
            elif event_type == "patient_discharge":
                await self._handle_patient_discharge_event(message)
            elif event_type == "procedure_completed":
                await self._handle_procedure_completion_event(message)
            elif event_type == "lab_results_available":
                await self._handle_lab_results_event(message)
            
            return HMCPMessage(
                type=HMCPMessageType.RESPONSE,
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content={"status": "event_processed", "event_type": event_type}
            )
            
        except Exception as e:
            logger.error(f"Error handling event {message.id}: {e}")
            raise
    
    # Healthcare-specific processing methods
    
    async def _validate_fhir_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Validate FHIR resource using HMCP context"""
        # Implementation would validate FHIR resource
        return {
            "validation_status": "valid",
            "resource_type": resource.get("resourceType", "unknown"),
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "contains_phi": True
        }
    
    async def _process_hl7_message(self, hl7_message: str) -> Dict[str, Any]:
        """Process HL7 message with clinical context"""
        # Implementation would process HL7 message
        return {
            "processing_status": "completed",
            "message_type": "ADT^A01",  # Example
            "patient_extracted": True,
            "contains_phi": True
        }
    
    async def _provide_clinical_decision_support(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Provide clinical decision support recommendations"""
        patient_data = content.get("patient_data", {})
        condition = content.get("condition", "")
        
        # Mock clinical decision support
        recommendations = []
        if "hypertension" in condition.lower():
            recommendations.extend([
                "Monitor blood pressure daily",
                "Consider ACE inhibitor therapy",
                "Lifestyle modifications: diet and exercise",
                "Follow up in 2-4 weeks"
            ])
        
        return {
            "recommendations": recommendations,
            "confidence_score": 0.85,
            "evidence_level": "high",
            "contains_phi": True
        }
    
    async def _check_medications(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Check medications for interactions and contraindications"""
        medications = content.get("medications", [])
        patient_allergies = content.get("allergies", [])
        
        # Mock medication checking
        alerts = []
        if "warfarin" in str(medications).lower() and "aspirin" in str(medications).lower():
            alerts.append({
                "severity": "high",
                "type": "drug_interaction",
                "message": "Warfarin + Aspirin: Increased bleeding risk"
            })
        
        return {
            "alerts": alerts,
            "total_medications": len(medications),
            "interaction_check_completed": True,
            "contains_phi": True
        }
    
    # Emergency handling methods
    
    async def _handle_cardiac_emergency(self, message: HMCPMessage) -> Dict[str, Any]:
        """Handle cardiac emergency response"""
        return {
            "response_type": "cardiac_arrest_protocol",
            "actions_initiated": [
                "cardiology_team_notified",
                "crash_cart_requested",
                "cath_lab_prepared",
                "family_notification_initiated"
            ],
            "estimated_response_time": "3_minutes",
            "priority": "STAT"
        }
    
    async def _handle_respiratory_emergency(self, message: HMCPMessage) -> Dict[str, Any]:
        """Handle respiratory emergency response"""
        return {
            "response_type": "respiratory_failure_protocol",
            "actions_initiated": [
                "respiratory_therapy_notified",
                "icu_bed_reserved",
                "ventilator_prepared",
                "pulmonology_consult_requested"
            ],
            "estimated_response_time": "5_minutes",
            "priority": "STAT"
        }
    
    async def _handle_stroke_emergency(self, message: HMCPMessage) -> Dict[str, Any]:
        """Handle stroke emergency response"""
        return {
            "response_type": "stroke_alert_protocol",
            "actions_initiated": [
                "neurology_team_notified",
                "ct_scanner_reserved",
                "stroke_team_assembled",
                "tpa_prepared"
            ],
            "estimated_response_time": "15_minutes",
            "priority": "STAT"
        }
    
    async def _handle_sepsis_emergency(self, message: HMCPMessage) -> Dict[str, Any]:
        """Handle sepsis emergency response"""
        return {
            "response_type": "sepsis_protocol",
            "actions_initiated": [
                "sepsis_bundle_initiated",
                "blood_cultures_ordered",
                "broad_spectrum_antibiotics_prepared",
                "icu_notified"
            ],
            "estimated_response_time": "60_minutes",
            "priority": "URGENT"
        }
    
    async def _handle_general_emergency(self, message: HMCPMessage) -> Dict[str, Any]:
        """Handle general emergency response"""
        return {
            "response_type": "general_emergency_protocol",
            "actions_initiated": [
                "emergency_team_notified",
                "charge_nurse_alerted",
                "physician_contacted"
            ],
            "estimated_response_time": "10_minutes",
            "priority": "URGENT"
        }
    
    # Coordination methods
    
    async def _coordinate_multidisciplinary_care(self, message: HMCPMessage) -> Dict[str, Any]:
        """Coordinate multidisciplinary care team"""
        care_plan = message.content.get("care_plan", {})
        team_members = message.content.get("team_members", [])
        
        return {
            "coordination_status": "initiated",
            "team_members_notified": len(team_members),
            "care_plan_distributed": True,
            "next_meeting_scheduled": True,
            "coordination_id": message.workflow_id
        }
    
    async def _coordinate_discharge_planning(self, message: HMCPMessage) -> Dict[str, Any]:
        """Coordinate patient discharge planning"""
        return {
            "discharge_planning_status": "in_progress",
            "assessments_completed": [
                "medical_clearance",
                "social_work_assessment",
                "pharmacy_reconciliation"
            ],
            "discharge_criteria_met": True,
            "estimated_discharge_date": "2024-12-16"
        }
    
    async def _coordinate_transfer_of_care(self, message: HMCPMessage) -> Dict[str, Any]:
        """Coordinate transfer of care between units/facilities"""
        return {
            "transfer_status": "approved",
            "receiving_unit_notified": True,
            "handoff_report_prepared": True,
            "transport_arranged": True,
            "estimated_transfer_time": "30_minutes"
        }
    
    # Event handling methods
    
    async def _handle_patient_admission_event(self, message: HMCPMessage):
        """Handle patient admission event"""
        logger.info(f"Processing patient admission for {message.patient_context.patient_id}")
        # Implementation would trigger admission workflow
    
    async def _handle_patient_discharge_event(self, message: HMCPMessage):
        """Handle patient discharge event"""
        logger.info(f"Processing patient discharge for {message.patient_context.patient_id}")
        # Implementation would trigger discharge workflow
    
    async def _handle_procedure_completion_event(self, message: HMCPMessage):
        """Handle procedure completion event"""
        procedure_type = message.content.get("procedure_type", "unknown")
        logger.info(f"Processing procedure completion: {procedure_type}")
        # Implementation would update care plan and notify relevant teams
    
    async def _handle_lab_results_event(self, message: HMCPMessage):
        """Handle lab results availability event"""
        logger.info("Processing lab results availability")
        # Implementation would check for critical values and notify providers
    
    # Notification handlers
    
    async def _handle_critical_lab_notification(self, message: HMCPMessage):
        """Handle critical lab value notification"""
        lab_value = message.content.get("lab_value", {})
        logger.warning(f"Critical lab value received: {lab_value}")
        # Implementation would escalate to appropriate provider
    
    async def _handle_medication_alert(self, message: HMCPMessage):
        """Handle medication alert notification"""
        alert_type = message.content.get("alert_type", "unknown")
        logger.warning(f"Medication alert: {alert_type}")
        # Implementation would notify pharmacist and provider
    
    async def _handle_vital_signs_alert(self, message: HMCPMessage):
        """Handle vital signs alert notification"""
        vital_signs = message.content.get("vital_signs", {})
        logger.warning(f"Vital signs alert: {vital_signs}")
        # Implementation would notify nursing staff
    
    async def _handle_care_plan_update(self, message: HMCPMessage):
        """Handle care plan update notification"""
        logger.info("Care plan update received")
        # Implementation would update local care plan and notify team
    
    # Utility methods
    
    def _update_response_time_metric(self, response_time: float):
        """Update average response time metric"""
        current_avg = self.metrics["average_response_time"]
        total_responses = self.metrics["messages_received"]
        
        if total_responses == 1:
            self.metrics["average_response_time"] = response_time
        else:
            self.metrics["average_response_time"] = (
                (current_avg * (total_responses - 1) + response_time) / total_responses
            )
    
    # Public API methods
    
    async def send_clinical_message(self, 
                                  receiver_id: str,
                                  message_type: HMCPMessageType,
                                  content: Dict[str, Any],
                                  patient_id: Optional[str] = None,
                                  urgency: ClinicalUrgency = ClinicalUrgency.ROUTINE) -> bool:
        """Send a clinical message to another healthcare agent"""
        
        message = HMCPMessage(
            type=message_type,
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content=content,
            urgency=urgency,
            patient_context=PatientContext(patient_id=patient_id) if patient_id else None,
            security_context=SecurityContext(
                user_id=self.agent_id,
                role=self.healthcare_role,
                phi_flag=bool(patient_id)
            )
        )
        
        if self.hmcp_client:
            # Mock endpoint URL - in real implementation, this would be resolved
            endpoint_url = f"http://localhost:808{hash(receiver_id) % 10}"
            success = await self.hmcp_client.send_message(message, endpoint_url)
            
            if success:
                self.metrics["messages_sent"] += 1
            
            return success
        
        return False
    
    async def initiate_emergency_response(self,
                                        patient_id: str,
                                        emergency_type: str,
                                        location: str,
                                        details: Dict[str, Any]) -> str:
        """Initiate emergency response protocol"""
        
        emergency_id = await hmcp_protocol.emergency_response(
            self.agent_id,
            patient_id,
            emergency_type,
            location,
            details
        )
        
        self.metrics["emergency_responses"] += 1
        
        return emergency_id
    
    async def coordinate_care_workflow(self,
                                     patient_id: str,
                                     workflow_type: str,
                                     participants: List[str],
                                     care_plan: Dict[str, Any]) -> str:
        """Coordinate a care workflow with multiple agents"""
        
        workflow_id = await hmcp_protocol.coordinate_care_team(
            self.agent_id,
            patient_id,
            care_plan,
            participants
        )
        
        self.metrics["workflow_completions"] += 1
        
        return workflow_id
    
    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            **self.metrics,
            "agent_id": self.agent_id,
            "role": self.healthcare_role,
            "capabilities": self.capabilities,
            "emergency_capable": self.emergency_capable,
            "active_conversations": len(self.active_conversations),
            "uptime": (datetime.now(timezone.utc) - self.created_at).total_seconds()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        return {
            "status": "healthy",
            "hmcp_client_connected": self.hmcp_client is not None,
            "hmcp_server_running": self.hmcp_server is not None,
            "message_queue_size": self.message_queue.qsize(),
            "error_rate": self.metrics["error_count"] / max(1, self.metrics["messages_received"]),
            "average_response_time": self.metrics["average_response_time"],
            "last_activity": datetime.now(timezone.utc).isoformat()
        }