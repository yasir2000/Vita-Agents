#!/usr/bin/env python3
"""
Healthcare Multi-agent Communication Protocol (HMCP) Implementation for Vita Agents

Based on Innovaccer's HMCP specification, this module provides comprehensive support
for multi-agent healthcare workflows with bidirectional communication, security,
and healthcare-specific context management.

Key Features:
- Bidirectional agent-to-agent communication
- Healthcare-specific message types and routing
- Clinical workflow orchestration
- Security and compliance (HIPAA-ready)
- Real-time emergency response protocols
- Structured clinical data exchange

References:
- https://innovaccer.com/blogs/building-a-multi-agent-workflow-in-healthcare-systems-using-hmcp
- https://github.com/innovaccer/Healthcare-MCP
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, TYPE_CHECKING

from pydantic import BaseModel, Field, validator
import aiohttp
from cryptography.fernet import Fernet

if TYPE_CHECKING:
    from vita_agents.core.agent import HealthcareAgent

logger = logging.getLogger(__name__)


class HMCPMessageType(str, Enum):
    """HMCP message types for healthcare communication"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    EVENT = "event"
    COORDINATION = "coordination"
    EMERGENCY = "emergency"


class ClinicalUrgency(str, Enum):
    """Clinical urgency levels for healthcare messages"""
    ROUTINE = "routine"
    URGENT = "urgent"
    STAT = "stat"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class HealthcareRole(str, Enum):
    """Healthcare professional roles for agent identification"""
    PHYSICIAN = "physician"
    NURSE = "nurse"
    PHARMACIST = "pharmacist"
    TECHNICIAN = "technician"
    THERAPIST = "therapist"
    ADMINISTRATOR = "administrator"
    SYSTEM = "system"
    AI_AGENT = "ai_agent"


@dataclass
class PatientContext:
    """Patient context information for HMCP messages"""
    patient_id: str
    mrn: Optional[str] = None  # Medical Record Number
    visit_id: Optional[str] = None
    encounter_id: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    condition: Optional[str] = None
    location: Optional[str] = None  # Room, unit, department
    acuity_level: Optional[str] = None


@dataclass
class ClinicalContext:
    """Clinical context for healthcare-specific processing"""
    specialty: Optional[str] = None
    department: Optional[str] = None
    workflow_type: Optional[str] = None  # admission, discharge, consult, etc.
    care_plan_id: Optional[str] = None
    provider_id: Optional[str] = None
    facility_id: Optional[str] = None
    insurance_info: Optional[Dict[str, Any]] = None


@dataclass
class SecurityContext:
    """Security context for HMCP messages"""
    user_id: str
    role: HealthcareRole
    permissions: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    authentication_token: Optional[str] = None
    encryption_required: bool = True
    audit_required: bool = True
    phi_flag: bool = False  # Contains Protected Health Information


class HMCPMessage(BaseModel):
    """Healthcare Multi-agent Communication Protocol Message"""
    
    # Core message fields
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: HMCPMessageType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sender_id: str
    receiver_id: Optional[str] = None  # None for broadcast messages
    correlation_id: Optional[str] = None  # For request-response correlation
    
    # Healthcare-specific fields
    urgency: ClinicalUrgency = ClinicalUrgency.ROUTINE
    patient_context: Optional[PatientContext] = None
    clinical_context: Optional[ClinicalContext] = None
    security_context: Optional[SecurityContext] = None
    
    # Message content
    subject: Optional[str] = None
    content: Dict[str, Any] = Field(default_factory=dict)
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Workflow and routing
    workflow_id: Optional[str] = None
    step_id: Optional[str] = None
    route_history: List[str] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.now(timezone.utc)
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def can_retry(self) -> bool:
        """Check if message can be retried"""
        return self.retry_count < self.max_retries
    
    def encrypt_content(self, encryption_key: bytes) -> None:
        """Encrypt message content for PHI protection"""
        if not self.security_context or not self.security_context.encryption_required:
            return
            
        fernet = Fernet(encryption_key)
        content_json = json.dumps(self.content)
        encrypted_content = fernet.encrypt(content_json.encode())
        self.content = {"encrypted": encrypted_content.decode()}
        self.metadata["encrypted"] = True
    
    def decrypt_content(self, encryption_key: bytes) -> None:
        """Decrypt message content"""
        if not self.metadata.get("encrypted"):
            return
            
        fernet = Fernet(encryption_key)
        encrypted_content = self.content["encrypted"].encode()
        decrypted_content = fernet.decrypt(encrypted_content)
        self.content = json.loads(decrypted_content.decode())
        self.metadata["encrypted"] = False


class HMCPMessageRouter:
    """Router for HMCP messages with healthcare-specific routing logic"""
    
    def __init__(self):
        self.routes: Dict[str, List[str]] = {}  # agent_id -> [capabilities]
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.emergency_agents: List[str] = []
        
    def register_agent(self, agent_id: str, capabilities: List[str], 
                      role: HealthcareRole, emergency_capable: bool = False):
        """Register an agent with the router"""
        self.routes[agent_id] = capabilities
        self.agent_registry[agent_id] = {
            "capabilities": capabilities,
            "role": role,
            "emergency_capable": emergency_capable,
            "last_seen": datetime.now(timezone.utc)
        }
        
        if emergency_capable:
            self.emergency_agents.append(agent_id)
    
    def find_capable_agents(self, capability: str) -> List[str]:
        """Find agents capable of handling a specific capability"""
        capable_agents = []
        for agent_id, capabilities in self.routes.items():
            if capability in capabilities:
                capable_agents.append(agent_id)
        return capable_agents
    
    def route_message(self, message: HMCPMessage) -> List[str]:
        """Route message to appropriate agents based on healthcare context"""
        if message.receiver_id:
            return [message.receiver_id]
        
        target_agents = []
        
        # Emergency routing
        if message.urgency in [ClinicalUrgency.CRITICAL, ClinicalUrgency.EMERGENCY]:
            target_agents.extend(self.emergency_agents)
            logger.warning(f"Emergency message {message.id} routed to {len(self.emergency_agents)} emergency agents")
        
        # Clinical specialty routing
        if message.clinical_context and message.clinical_context.specialty:
            specialty_agents = self.find_capable_agents(f"specialty:{message.clinical_context.specialty}")
            target_agents.extend(specialty_agents)
        
        # Workflow-based routing
        if message.clinical_context and message.clinical_context.workflow_type:
            workflow_agents = self.find_capable_agents(f"workflow:{message.clinical_context.workflow_type}")
            target_agents.extend(workflow_agents)
        
        # Role-based routing
        if message.security_context:
            role_agents = self.find_capable_agents(f"role:{message.security_context.role}")
            target_agents.extend(role_agents)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_agents = []
        for agent in target_agents:
            if agent not in seen:
                seen.add(agent)
                unique_agents.append(agent)
        
        return unique_agents or self.get_default_agents()
    
    def get_default_agents(self) -> List[str]:
        """Get default agents when no specific routing applies"""
        return list(self.routes.keys())[:3]  # Limit to first 3 agents


class HMCPClient:
    """HMCP client for agent-to-agent communication"""
    
    def __init__(self, agent_id: str, encryption_key: Optional[bytes] = None):
        self.agent_id = agent_id
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.session: Optional[aiohttp.ClientSession] = None
        self.connections: Dict[str, str] = {}  # agent_id -> endpoint_url
        self.message_handlers: Dict[HMCPMessageType, Callable] = {}
        
    async def connect(self, server_url: str, auth_token: Optional[str] = None):
        """Connect to HMCP server"""
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
            
        self.session = aiohttp.ClientSession(headers=headers)
        
        # Test connection
        try:
            async with self.session.get(f"{server_url}/health") as response:
                if response.status == 200:
                    logger.info(f"Connected to HMCP server at {server_url}")
                    return True
        except Exception as e:
            logger.error(f"Failed to connect to HMCP server: {e}")
            return False
    
    async def send_message(self, message: HMCPMessage, endpoint_url: str) -> bool:
        """Send HMCP message to another agent"""
        if not self.session:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        try:
            # Encrypt content if required
            if message.security_context and message.security_context.encryption_required:
                message.encrypt_content(self.encryption_key)
            
            # Add to route history
            message.route_history.append(self.agent_id)
            
            # Send message
            async with self.session.post(
                f"{endpoint_url}/hmcp/message",
                json=message.dict()
            ) as response:
                if response.status == 200:
                    logger.info(f"Message {message.id} sent successfully")
                    return True
                else:
                    logger.error(f"Failed to send message {message.id}: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending message {message.id}: {e}")
            return False
    
    async def receive_message(self, message_data: Dict[str, Any]) -> HMCPMessage:
        """Receive and process HMCP message"""
        message = HMCPMessage(**message_data)
        
        # Decrypt content if encrypted
        if message.metadata.get("encrypted"):
            message.decrypt_content(self.encryption_key)
        
        # Handle message based on type
        if message.type in self.message_handlers:
            await self.message_handlers[message.type](message)
        
        return message
    
    def register_handler(self, message_type: HMCPMessageType, 
                        handler: Callable[[HMCPMessage], None]):
        """Register a message handler for a specific message type"""
        self.message_handlers[message_type] = handler
    
    async def close(self):
        """Close the HMCP client connection"""
        if self.session:
            await self.session.close()


class HMCPServer:
    """HMCP server for receiving and processing healthcare messages"""
    
    def __init__(self, agent_id: str, port: int = 8080):
        self.agent_id = agent_id
        self.port = port
        self.router = HMCPMessageRouter()
        self.message_handlers: Dict[HMCPMessageType, Callable] = {}
        self.audit_log: List[Dict[str, Any]] = []
        
    def register_handler(self, message_type: HMCPMessageType, 
                        handler: Callable[[HMCPMessage], HMCPMessage]):
        """Register a message handler"""
        self.message_handlers[message_type] = handler
    
    async def process_message(self, message: HMCPMessage) -> HMCPMessage:
        """Process incoming HMCP message"""
        # Audit logging
        self.audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_id": message.id,
            "sender_id": message.sender_id,
            "receiver_id": self.agent_id,
            "type": message.type,
            "urgency": message.urgency,
            "patient_id": message.patient_context.patient_id if message.patient_context else None,
            "phi_flag": message.security_context.phi_flag if message.security_context else False
        })
        
        # Process based on message type
        if message.type in self.message_handlers:
            response = await self.message_handlers[message.type](message)
            response.correlation_id = message.id
            return response
        
        # Default response
        return HMCPMessage(
            type=HMCPMessageType.RESPONSE,
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            correlation_id=message.id,
            content={"status": "received", "message": "Message processed"}
        )


class HMCPProtocol:
    """Main HMCP protocol implementation for Vita Agents"""
    
    def __init__(self):
        self.clients: Dict[str, HMCPClient] = {}
        self.servers: Dict[str, HMCPServer] = {}
        self.router = HMCPMessageRouter()
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        
    def create_client(self, agent_id: str) -> HMCPClient:
        """Create an HMCP client for an agent"""
        client = HMCPClient(agent_id)
        self.clients[agent_id] = client
        return client
    
    def create_server(self, agent_id: str, port: int = 8080) -> HMCPServer:
        """Create an HMCP server for an agent"""
        server = HMCPServer(agent_id, port)
        self.servers[agent_id] = server
        return server
    
    async def send_clinical_notification(self, 
                                       sender_id: str,
                                       patient_id: str,
                                       notification_type: str,
                                       content: Dict[str, Any],
                                       urgency: ClinicalUrgency = ClinicalUrgency.ROUTINE,
                                       target_specialty: Optional[str] = None) -> bool:
        """Send a clinical notification to relevant healthcare agents"""
        
        message = HMCPMessage(
            type=HMCPMessageType.NOTIFICATION,
            sender_id=sender_id,
            urgency=urgency,
            subject=f"Clinical Notification: {notification_type}",
            content=content,
            patient_context=PatientContext(patient_id=patient_id),
            clinical_context=ClinicalContext(specialty=target_specialty),
            security_context=SecurityContext(
                user_id=sender_id,
                role=HealthcareRole.AI_AGENT,
                phi_flag=True,
                encryption_required=True
            )
        )
        
        # Route to appropriate agents
        target_agents = self.router.route_message(message)
        
        success_count = 0
        for agent_id in target_agents:
            if agent_id in self.clients:
                client = self.clients[agent_id]
                # This would need the actual endpoint URL in a real implementation
                endpoint_url = f"http://localhost:808{hash(agent_id) % 10}"
                if await client.send_message(message, endpoint_url):
                    success_count += 1
        
        return success_count > 0
    
    async def coordinate_care_team(self,
                                 coordinator_id: str,
                                 patient_id: str,
                                 care_plan: Dict[str, Any],
                                 team_members: List[str]) -> str:
        """Coordinate care team communication for a patient"""
        
        workflow_id = str(uuid.uuid4())
        
        coordination_message = HMCPMessage(
            type=HMCPMessageType.COORDINATION,
            sender_id=coordinator_id,
            urgency=ClinicalUrgency.URGENT,
            subject="Care Team Coordination",
            content={
                "action": "coordinate_care",
                "care_plan": care_plan,
                "team_members": team_members,
                "coordination_type": "multidisciplinary_care"
            },
            workflow_id=workflow_id,
            patient_context=PatientContext(patient_id=patient_id),
            clinical_context=ClinicalContext(workflow_type="care_coordination"),
            security_context=SecurityContext(
                user_id=coordinator_id,
                role=HealthcareRole.AI_AGENT,
                phi_flag=True
            )
        )
        
        # Store workflow information
        self.active_workflows[workflow_id] = {
            "type": "care_coordination",
            "patient_id": patient_id,
            "coordinator": coordinator_id,
            "team_members": team_members,
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Send to all team members
        for member_id in team_members:
            if member_id in self.clients:
                client = self.clients[member_id]
                endpoint_url = f"http://localhost:808{hash(member_id) % 10}"
                await client.send_message(coordination_message, endpoint_url)
        
        return workflow_id
    
    async def emergency_response(self,
                               initiator_id: str,
                               patient_id: str,
                               emergency_type: str,
                               location: str,
                               details: Dict[str, Any]) -> str:
        """Initiate emergency response protocol"""
        
        emergency_id = str(uuid.uuid4())
        
        emergency_message = HMCPMessage(
            type=HMCPMessageType.EMERGENCY,
            sender_id=initiator_id,
            urgency=ClinicalUrgency.EMERGENCY,
            subject=f"EMERGENCY: {emergency_type}",
            content={
                "emergency_type": emergency_type,
                "location": location,
                "details": details,
                "response_required": True,
                "emergency_id": emergency_id
            },
            patient_context=PatientContext(
                patient_id=patient_id,
                location=location,
                acuity_level="critical"
            ),
            clinical_context=ClinicalContext(workflow_type="emergency_response"),
            security_context=SecurityContext(
                user_id=initiator_id,
                role=HealthcareRole.AI_AGENT,
                phi_flag=True,
                encryption_required=True
            )
        )
        
        # Route to emergency-capable agents
        emergency_agents = self.router.emergency_agents
        
        for agent_id in emergency_agents:
            if agent_id in self.clients:
                client = self.clients[agent_id]
                endpoint_url = f"http://localhost:808{hash(agent_id) % 10}"
                await client.send_message(emergency_message, endpoint_url)
        
        logger.critical(f"Emergency response {emergency_id} initiated for patient {patient_id}")
        
        return emergency_id
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active workflow"""
        return self.active_workflows.get(workflow_id)
    
    def get_audit_trail(self, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get audit trail for HMCP messages"""
        audit_trail = []
        
        for server in self.servers.values():
            for entry in server.audit_log:
                if patient_id is None or entry.get("patient_id") == patient_id:
                    audit_trail.append(entry)
        
        return sorted(audit_trail, key=lambda x: x["timestamp"])


# Global HMCP protocol instance
hmcp_protocol = HMCPProtocol()