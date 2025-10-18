"""
Clinical Context Object Workgroup (CCOW) Integration Agent for Vita Agents.
Provides comprehensive CCOW context management and visual integration capabilities.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
import structlog
from pydantic import BaseModel, Field
import uuid
import threading
import weakref

from vita_agents.core.agent import HealthcareAgent, AgentCapability, TaskRequest
from vita_agents.core.config import get_settings


logger = structlog.get_logger(__name__)


class CCOWContextType(str, Enum):
    """CCOW context types."""
    PATIENT = "patient"
    USER = "user"
    ENCOUNTER = "encounter" 
    OBSERVATION = "observation"
    ORDER = "order"
    CLINICAL_IMAGE = "clinical_image"
    DOCUMENT = "document"
    APPLICATION = "application"
    CUSTOM = "custom"


class CCOWParticipantType(str, Enum):
    """CCOW participant types."""
    CONTEXT_MANAGER = "context_manager"
    CONTEXT_PARTICIPANT = "context_participant"
    SECURE_CONTEXT_PARTICIPANT = "secure_context_participant"
    CONTEXT_AGENT = "context_agent"
    ANNOTATION_MANAGER = "annotation_manager"


class CCOWSecurityLevel(str, Enum):
    """CCOW security levels."""
    NONE = "none"
    AUTHENTICATION = "authentication"
    ACCESS_CONTROL = "access_control"
    DIGITAL_SIGNATURE = "digital_signature"
    ENCRYPTION = "encryption"
    NON_REPUDIATION = "non_repudiation"


class CCOWContextItem(BaseModel):
    """CCOW context item."""
    
    name: str
    value: Any
    context_type: CCOWContextType
    security_level: CCOWSecurityLevel = CCOWSecurityLevel.NONE
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    version: int = 1
    metadata: Dict[str, Any] = {}


class CCOWParticipant(BaseModel):
    """CCOW participant information."""
    
    participant_id: str
    participant_type: CCOWParticipantType
    application_name: str
    vendor_name: str
    version: str
    security_level: CCOWSecurityLevel
    capabilities: List[str] = []
    context_filters: List[str] = []
    notification_url: Optional[str] = None
    is_active: bool = True
    joined_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class CCOWContextSession(BaseModel):
    """CCOW context session."""
    
    session_id: str
    context_manager_id: str
    participants: List[CCOWParticipant] = []
    context_items: Dict[str, CCOWContextItem] = {}
    transaction_id: Optional[str] = None
    is_locked: bool = False
    locked_by: Optional[str] = None
    lock_timestamp: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class CCOWTransaction(BaseModel):
    """CCOW context change transaction."""
    
    transaction_id: str
    session_id: str
    initiator_id: str
    proposed_changes: List[Dict[str, Any]] = []
    participant_responses: Dict[str, str] = {}  # participant_id -> response
    status: str = "pending"  # pending, committed, cancelled
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None


class CCOWSecurityToken(BaseModel):
    """CCOW security token."""
    
    token_id: str
    participant_id: str
    session_id: str
    token_type: str = "bearer"
    permissions: List[str] = []
    issued_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: str
    issuer: str = "ccow-agent"


class CCOWAgent(HealthcareAgent):
    """
    Clinical Context Object Workgroup (CCOW) Integration Agent.
    
    Capabilities:
    - Complete CCOW context management (patient, user, encounter contexts)
    - Visual integration coordination between healthcare applications
    - Secure context sharing with authentication and access control
    - Context change transaction management with two-phase commit
    - Multi-application session synchronization
    - CCOW-compliant participant registration and management
    - Real-time context notifications and updates
    - Security token management and validation
    - Context persistence and audit trails
    - Advanced context filtering and routing
    """
    
    def __init__(
        self,
        agent_id: str = "ccow-agent",
        name: str = "CCOW Agent",
        description: str = "Clinical Context Object Workgroup integration and visual coordination",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None,
    ):
        capabilities = [
            AgentCapability(
                name="connect_to_context",
                description="Connect application to CCOW context session",
                input_schema={
                    "type": "object",
                    "properties": {
                        "application_info": {"type": "object"},
                        "participant_type": {"type": "string"},
                        "security_requirements": {"type": "object"},
                        "context_filters": {"type": "array"},
                        "notification_endpoint": {"type": "string"}
                    },
                    "required": ["application_info", "participant_type"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "participant_id": {"type": "string"},
                        "security_token": {"type": "object"},
                        "current_context": {"type": "object"},
                        "connection_status": {"type": "string"}
                    }
                }
            ),
            AgentCapability(
                name="participate_in_context",
                description="Participate in context changes and synchronization",
                input_schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "participant_id": {"type": "string"},
                        "context_changes": {"type": "array"},
                        "transaction_mode": {"type": "string"},
                        "notification_preferences": {"type": "object"}
                    },
                    "required": ["session_id", "participant_id"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "transaction_id": {"type": "string"},
                        "context_accepted": {"type": "boolean"},
                        "updated_context": {"type": "object"},
                        "notifications": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="secure_context_access",
                description="Manage secure context access and permissions",
                input_schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "participant_id": {"type": "string"},
                        "security_token": {"type": "string"},
                        "requested_permissions": {"type": "array"},
                        "context_scope": {"type": "object"}
                    },
                    "required": ["session_id", "participant_id", "security_token"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "access_granted": {"type": "boolean"},
                        "granted_permissions": {"type": "array"},
                        "security_context": {"type": "object"},
                        "audit_trail": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="synchronize_applications",
                description="Synchronize multiple healthcare applications",
                input_schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "synchronization_scope": {"type": "string"},
                        "target_applications": {"type": "array"},
                        "sync_policies": {"type": "object"},
                        "conflict_resolution": {"type": "string"}
                    },
                    "required": ["session_id", "synchronization_scope"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "synchronization_result": {"type": "object"},
                        "synchronized_contexts": {"type": "array"},
                        "conflicts_resolved": {"type": "array"},
                        "sync_status": {"type": "string"}
                    }
                }
            ),
            AgentCapability(
                name="manage_visual_integration",
                description="Manage visual integration between applications",
                input_schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "integration_type": {"type": "string"},
                        "visual_policies": {"type": "object"},
                        "application_layouts": {"type": "array"},
                        "user_preferences": {"type": "object"}
                    },
                    "required": ["session_id", "integration_type"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "integration_config": {"type": "object"},
                        "layout_adjustments": {"type": "array"},
                        "visual_cues": {"type": "object"},
                        "integration_status": {"type": "string"}
                    }
                }
            )
        ]
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            version=version,
            capabilities=capabilities,
            config=config or {}
        )
        
        # Initialize CCOW components
        self.context_sessions: Dict[str, CCOWContextSession] = {}
        self.active_transactions: Dict[str, CCOWTransaction] = {}
        self.security_tokens: Dict[str, CCOWSecurityToken] = {}
        self.participant_registry: Dict[str, CCOWParticipant] = {}
        
        # Context change notification callbacks
        self.context_listeners: Dict[str, List[Callable]] = {}
        self.transaction_listeners: Dict[str, List[Callable]] = {}
        
        # Threading for real-time notifications
        self._notification_lock = threading.Lock()
        self._notification_thread = None
        self._stop_notifications = False
        
        # Register task handlers
        self.register_task_handler("connect_to_context", self._connect_to_context)
        self.register_task_handler("participate_in_context", self._participate_in_context)
        self.register_task_handler("secure_context_access", self._secure_context_access)
        self.register_task_handler("synchronize_applications", self._synchronize_applications)
        self.register_task_handler("manage_visual_integration", self._manage_visual_integration)
    
    async def _on_start(self) -> None:
        """Initialize CCOW agent."""
        self.logger.info("Starting CCOW agent")
        
        # Start notification processing thread
        self._start_notification_processing()
        
        # Initialize default context session
        await self._create_default_context_session()
        
        self.logger.info("CCOW agent initialized",
                        sessions_count=len(self.context_sessions),
                        participants_count=len(self.participant_registry))
    
    async def _on_stop(self) -> None:
        """Clean up CCOW agent."""
        self.logger.info("Stopping CCOW agent")
        
        # Stop notification processing
        self._stop_notification_processing()
        
        # Clean up active sessions
        await self._cleanup_sessions()
        
        self.logger.info("CCOW agent stopped")
    
    async def _connect_to_context(self, task: TaskRequest) -> Dict[str, Any]:
        """Connect application to CCOW context session."""
        try:
            application_info = task.parameters.get("application_info", {})
            participant_type = task.parameters.get("participant_type", "context_participant")
            security_requirements = task.parameters.get("security_requirements", {})
            context_filters = task.parameters.get("context_filters", [])
            notification_endpoint = task.parameters.get("notification_endpoint")
            
            if not application_info.get("name"):
                raise ValueError("Application name is required")
            
            self.audit_log_action(
                action="connect_to_context",
                data_type="CCOW Connection",
                details={
                    "application": application_info.get("name"),
                    "participant_type": participant_type,
                    "task_id": task.id
                }
            )
            
            # Generate participant ID
            participant_id = f"participant_{uuid.uuid4().hex[:8]}"
            
            # Create participant
            participant = CCOWParticipant(
                participant_id=participant_id,
                participant_type=CCOWParticipantType(participant_type),
                application_name=application_info.get("name", "Unknown"),
                vendor_name=application_info.get("vendor", "Unknown"),
                version=application_info.get("version", "1.0"),
                security_level=CCOWSecurityLevel(security_requirements.get("level", "none")),
                capabilities=application_info.get("capabilities", []),
                context_filters=context_filters,
                notification_url=notification_endpoint
            )
            
            # Find or create context session
            session_id = await self._find_or_create_session(participant)
            session = self.context_sessions[session_id]
            
            # Add participant to session
            session.participants.append(participant)
            session.last_updated = datetime.utcnow().isoformat()
            
            # Register participant
            self.participant_registry[participant_id] = participant
            
            # Generate security token
            security_token = await self._generate_security_token(participant_id, session_id, security_requirements)
            
            # Get current context
            current_context = await self._get_filtered_context(session_id, context_filters)
            
            # Notify other participants of new connection
            await self._notify_participant_joined(session_id, participant)
            
            return {
                "session_id": session_id,
                "participant_id": participant_id,
                "security_token": security_token.dict() if security_token else None,
                "current_context": current_context,
                "connection_status": "connected",
                "session_info": {
                    "participant_count": len(session.participants),
                    "context_items": len(session.context_items),
                    "is_locked": session.is_locked
                },
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("CCOW connection failed", error=str(e), task_id=task.id)
            raise
    
    async def _participate_in_context(self, task: TaskRequest) -> Dict[str, Any]:
        """Participate in context changes and synchronization."""
        try:
            session_id = task.parameters.get("session_id")
            participant_id = task.parameters.get("participant_id")
            context_changes = task.parameters.get("context_changes", [])
            transaction_mode = task.parameters.get("transaction_mode", "immediate")
            notification_preferences = task.parameters.get("notification_preferences", {})
            
            if not session_id or not participant_id:
                raise ValueError("session_id and participant_id are required")
            
            # Validate session and participant
            if session_id not in self.context_sessions:
                raise ValueError(f"Session not found: {session_id}")
            
            if participant_id not in self.participant_registry:
                raise ValueError(f"Participant not registered: {participant_id}")
            
            session = self.context_sessions[session_id]
            participant = self.participant_registry[participant_id]
            
            self.audit_log_action(
                action="participate_in_context",
                data_type="CCOW Participation",
                details={
                    "session_id": session_id,
                    "participant_id": participant_id,
                    "changes_count": len(context_changes),
                    "transaction_mode": transaction_mode,
                    "task_id": task.id
                }
            )
            
            # Process context changes
            transaction_id = None
            context_accepted = True
            notifications = []
            
            if context_changes:
                if transaction_mode == "transactional":
                    # Start transaction for context changes
                    transaction_id = await self._start_context_transaction(
                        session_id, participant_id, context_changes
                    )
                    
                    # Wait for transaction completion
                    transaction_result = await self._process_transaction(transaction_id)
                    context_accepted = transaction_result["committed"]
                    
                    if context_accepted:
                        await self._apply_context_changes(session_id, context_changes)
                        notifications.append({
                            "type": "context_changed",
                            "transaction_id": transaction_id,
                            "status": "committed"
                        })
                    else:
                        notifications.append({
                            "type": "context_change_rejected",
                            "transaction_id": transaction_id,
                            "status": "cancelled"
                        })
                else:
                    # Immediate context changes
                    context_accepted = await self._validate_context_changes(session_id, context_changes)
                    
                    if context_accepted:
                        await self._apply_context_changes(session_id, context_changes)
                        await self._notify_context_changed(session_id, participant_id, context_changes)
                        notifications.append({
                            "type": "context_updated",
                            "status": "applied"
                        })
            
            # Get updated context
            updated_context = await self._get_filtered_context(
                session_id, participant.context_filters
            )
            
            # Register for notifications if requested
            if notification_preferences.get("enable_notifications", True):
                await self._register_notification_listener(session_id, participant_id, notification_preferences)
            
            return {
                "transaction_id": transaction_id,
                "context_accepted": context_accepted,
                "updated_context": updated_context,
                "notifications": notifications,
                "participation_status": "active",
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("CCOW participation failed", error=str(e), task_id=task.id)
            raise
    
    async def _secure_context_access(self, task: TaskRequest) -> Dict[str, Any]:
        """Manage secure context access and permissions."""
        try:
            session_id = task.parameters.get("session_id")
            participant_id = task.parameters.get("participant_id")
            security_token = task.parameters.get("security_token")
            requested_permissions = task.parameters.get("requested_permissions", [])
            context_scope = task.parameters.get("context_scope", {})
            
            if not all([session_id, participant_id, security_token]):
                raise ValueError("session_id, participant_id, and security_token are required")
            
            self.audit_log_action(
                action="secure_context_access",
                data_type="CCOW Security",
                details={
                    "session_id": session_id,
                    "participant_id": participant_id,
                    "permissions_requested": len(requested_permissions),
                    "task_id": task.id
                }
            )
            
            # Validate security token
            token_valid, token_obj = await self._validate_security_token(security_token, participant_id, session_id)
            
            if not token_valid:
                return {
                    "access_granted": False,
                    "granted_permissions": [],
                    "security_context": {},
                    "audit_trail": [{
                        "action": "access_denied",
                        "reason": "invalid_token",
                        "timestamp": datetime.utcnow().isoformat()
                    }],
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Check participant permissions
            participant = self.participant_registry.get(participant_id)
            if not participant:
                raise ValueError(f"Participant not found: {participant_id}")
            
            # Evaluate permission requests
            granted_permissions = await self._evaluate_permissions(
                participant, requested_permissions, context_scope
            )
            
            access_granted = len(granted_permissions) > 0
            
            # Create security context
            security_context = {
                "participant_id": participant_id,
                "security_level": participant.security_level.value,
                "granted_permissions": granted_permissions,
                "context_scope": context_scope,
                "token_expires_at": token_obj.expires_at if token_obj else None,
                "session_security": await self._get_session_security_info(session_id)
            }
            
            # Audit trail
            audit_trail = [{
                "action": "access_granted" if access_granted else "access_denied",
                "participant_id": participant_id,
                "permissions_requested": requested_permissions,
                "permissions_granted": granted_permissions,
                "timestamp": datetime.utcnow().isoformat()
            }]
            
            # Update token permissions if access granted
            if access_granted and token_obj:
                token_obj.permissions.extend([p for p in granted_permissions if p not in token_obj.permissions])
                self.security_tokens[token_obj.token_id] = token_obj
            
            return {
                "access_granted": access_granted,
                "granted_permissions": granted_permissions,
                "security_context": security_context,
                "audit_trail": audit_trail,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Secure context access failed", error=str(e), task_id=task.id)
            raise
    
    async def _synchronize_applications(self, task: TaskRequest) -> Dict[str, Any]:
        """Synchronize multiple healthcare applications."""
        try:
            session_id = task.parameters.get("session_id")
            synchronization_scope = task.parameters.get("synchronization_scope", "full")
            target_applications = task.parameters.get("target_applications", [])
            sync_policies = task.parameters.get("sync_policies", {})
            conflict_resolution = task.parameters.get("conflict_resolution", "latest_wins")
            
            if not session_id:
                raise ValueError("session_id is required")
            
            if session_id not in self.context_sessions:
                raise ValueError(f"Session not found: {session_id}")
            
            session = self.context_sessions[session_id]
            
            self.audit_log_action(
                action="synchronize_applications",
                data_type="CCOW Synchronization",
                details={
                    "session_id": session_id,
                    "scope": synchronization_scope,
                    "target_count": len(target_applications),
                    "task_id": task.id
                }
            )
            
            # Determine participants to synchronize
            sync_participants = []
            if target_applications:
                sync_participants = [
                    p for p in session.participants 
                    if p.application_name in target_applications
                ]
            else:
                sync_participants = session.participants.copy()
            
            if not sync_participants:
                return {
                    "synchronization_result": {"status": "no_participants"},
                    "synchronized_contexts": [],
                    "conflicts_resolved": [],
                    "sync_status": "completed",
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Lock session for synchronization
            await self._lock_session(session_id, "sync_operation")
            
            try:
                # Collect context from all participants
                participant_contexts = {}
                for participant in sync_participants:
                    context = await self._get_participant_context(session_id, participant.participant_id)
                    participant_contexts[participant.participant_id] = context
                
                # Detect conflicts
                conflicts = await self._detect_context_conflicts(participant_contexts, sync_policies)
                
                # Resolve conflicts
                resolved_conflicts = []
                if conflicts:
                    resolved_conflicts = await self._resolve_context_conflicts(
                        conflicts, conflict_resolution, sync_policies
                    )
                
                # Generate synchronized context
                synchronized_context = await self._merge_participant_contexts(
                    participant_contexts, resolved_conflicts, sync_policies
                )
                
                # Apply synchronized context to session
                if synchronization_scope == "full":
                    session.context_items = {
                        name: CCOWContextItem(**item) if isinstance(item, dict) else item
                        for name, item in synchronized_context.items()
                    }
                else:
                    # Partial synchronization based on scope
                    await self._apply_partial_sync(session_id, synchronized_context, synchronization_scope)
                
                session.last_updated = datetime.utcnow().isoformat()
                
                # Notify participants of synchronization
                await self._notify_synchronization_complete(session_id, sync_participants, synchronized_context)
                
                synchronization_result = {
                    "status": "completed",
                    "participants_synchronized": len(sync_participants),
                    "conflicts_found": len(conflicts),
                    "conflicts_resolved": len(resolved_conflicts),
                    "context_items_synchronized": len(synchronized_context)
                }
                
                return {
                    "synchronization_result": synchronization_result,
                    "synchronized_contexts": list(synchronized_context.keys()),
                    "conflicts_resolved": resolved_conflicts,
                    "sync_status": "completed",
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            finally:
                # Unlock session
                await self._unlock_session(session_id)
            
        except Exception as e:
            self.logger.error("Application synchronization failed", error=str(e), task_id=task.id)
            raise
    
    async def _manage_visual_integration(self, task: TaskRequest) -> Dict[str, Any]:
        """Manage visual integration between applications."""
        try:
            session_id = task.parameters.get("session_id")
            integration_type = task.parameters.get("integration_type", "standard")
            visual_policies = task.parameters.get("visual_policies", {})
            application_layouts = task.parameters.get("application_layouts", [])
            user_preferences = task.parameters.get("user_preferences", {})
            
            if not session_id:
                raise ValueError("session_id is required")
            
            if session_id not in self.context_sessions:
                raise ValueError(f"Session not found: {session_id}")
            
            session = self.context_sessions[session_id]
            
            self.audit_log_action(
                action="manage_visual_integration",
                data_type="CCOW Visual Integration",
                details={
                    "session_id": session_id,
                    "integration_type": integration_type,
                    "layouts_count": len(application_layouts),
                    "task_id": task.id
                }
            )
            
            # Generate integration configuration
            integration_config = await self._generate_integration_config(
                session, integration_type, visual_policies, user_preferences
            )
            
            # Calculate layout adjustments
            layout_adjustments = []
            if application_layouts:
                layout_adjustments = await self._calculate_layout_adjustments(
                    application_layouts, integration_config, visual_policies
                )
            
            # Define visual cues for context sharing
            visual_cues = await self._generate_visual_cues(
                session, integration_config, user_preferences
            )
            
            # Apply integration settings to session
            session.metadata = session.metadata or {}
            session.metadata["visual_integration"] = {
                "type": integration_type,
                "config": integration_config,
                "visual_cues": visual_cues,
                "layout_adjustments": layout_adjustments,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Notify participants of visual integration changes
            await self._notify_visual_integration_update(session_id, integration_config, visual_cues)
            
            integration_status = "active" if len(session.participants) > 1 else "ready"
            
            return {
                "integration_config": integration_config,
                "layout_adjustments": layout_adjustments,
                "visual_cues": visual_cues,
                "integration_status": integration_status,
                "participants_integrated": len(session.participants),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Visual integration management failed", error=str(e), task_id=task.id)
            raise
    
    # Helper methods for CCOW operations
    
    async def _find_or_create_session(self, participant: CCOWParticipant) -> str:
        """Find existing session or create new one."""
        # For now, create a default session or find existing one
        # In production, this would involve more sophisticated session management
        
        for session_id, session in self.context_sessions.items():
            if len(session.participants) < 10:  # Max participants per session
                return session_id
        
        # Create new session
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        context_manager_id = f"cm_{uuid.uuid4().hex[:8]}"
        
        session = CCOWContextSession(
            session_id=session_id,
            context_manager_id=context_manager_id
        )
        
        self.context_sessions[session_id] = session
        return session_id
    
    async def _create_default_context_session(self) -> None:
        """Create default context session."""
        session_id = "default_session"
        context_manager_id = "default_cm"
        
        session = CCOWContextSession(
            session_id=session_id,
            context_manager_id=context_manager_id
        )
        
        self.context_sessions[session_id] = session
    
    async def _generate_security_token(self, participant_id: str, session_id: str, security_requirements: Dict[str, Any]) -> Optional[CCOWSecurityToken]:
        """Generate security token for participant."""
        if security_requirements.get("level", "none") == "none":
            return None
        
        token_id = f"token_{uuid.uuid4().hex}"
        expires_at = datetime.utcnow().replace(hour=23, minute=59, second=59).isoformat()
        
        token = CCOWSecurityToken(
            token_id=token_id,
            participant_id=participant_id,
            session_id=session_id,
            permissions=security_requirements.get("permissions", []),
            expires_at=expires_at
        )
        
        self.security_tokens[token_id] = token
        return token
    
    async def _get_filtered_context(self, session_id: str, context_filters: List[str]) -> Dict[str, Any]:
        """Get context filtered by participant filters."""
        session = self.context_sessions[session_id]
        filtered_context = {}
        
        for name, item in session.context_items.items():
            if not context_filters or item.context_type.value in context_filters:
                filtered_context[name] = {
                    "value": item.value,
                    "type": item.context_type.value,
                    "timestamp": item.timestamp,
                    "version": item.version
                }
        
        return filtered_context
    
    async def _notify_participant_joined(self, session_id: str, participant: CCOWParticipant) -> None:
        """Notify other participants that a new participant joined."""
        session = self.context_sessions[session_id]
        
        notification = {
            "type": "participant_joined",
            "session_id": session_id,
            "participant": {
                "id": participant.participant_id,
                "application": participant.application_name,
                "vendor": participant.vendor_name
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._send_session_notification(session_id, notification, exclude_participant=participant.participant_id)
    
    async def _start_context_transaction(self, session_id: str, initiator_id: str, context_changes: List[Dict[str, Any]]) -> str:
        """Start a context change transaction."""
        transaction_id = f"txn_{uuid.uuid4().hex[:8]}"
        
        transaction = CCOWTransaction(
            transaction_id=transaction_id,
            session_id=session_id,
            initiator_id=initiator_id,
            proposed_changes=context_changes
        )
        
        self.active_transactions[transaction_id] = transaction
        
        # Notify all participants about pending transaction
        await self._notify_transaction_pending(transaction)
        
        return transaction_id
    
    async def _process_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """Process context change transaction with two-phase commit."""
        if transaction_id not in self.active_transactions:
            raise ValueError(f"Transaction not found: {transaction_id}")
        
        transaction = self.active_transactions[transaction_id]
        session = self.context_sessions[transaction.session_id]
        
        # Phase 1: Prepare - ask all participants if they can accept changes
        prepare_responses = {}
        for participant in session.participants:
            if participant.participant_id != transaction.initiator_id:
                response = await self._request_transaction_prepare(participant, transaction)
                prepare_responses[participant.participant_id] = response
        
        # Check if all participants agreed
        all_agreed = all(response == "prepared" for response in prepare_responses.values())
        
        if all_agreed:
            # Phase 2: Commit
            transaction.status = "committed"
            transaction.participant_responses = prepare_responses
            transaction.completed_at = datetime.utcnow().isoformat()
            
            await self._notify_transaction_committed(transaction)
            
            return {"committed": True, "responses": prepare_responses}
        else:
            # Abort transaction
            transaction.status = "cancelled"
            transaction.participant_responses = prepare_responses
            transaction.completed_at = datetime.utcnow().isoformat()
            
            await self._notify_transaction_cancelled(transaction)
            
            return {"committed": False, "responses": prepare_responses}
    
    async def _apply_context_changes(self, session_id: str, context_changes: List[Dict[str, Any]]) -> None:
        """Apply context changes to session."""
        session = self.context_sessions[session_id]
        
        for change in context_changes:
            context_name = change.get("name")
            context_value = change.get("value")
            context_type = change.get("type", "custom")
            
            if context_name:
                context_item = CCOWContextItem(
                    name=context_name,
                    value=context_value,
                    context_type=CCOWContextType(context_type)
                )
                
                session.context_items[context_name] = context_item
        
        session.last_updated = datetime.utcnow().isoformat()
    
    async def _validate_context_changes(self, session_id: str, context_changes: List[Dict[str, Any]]) -> bool:
        """Validate proposed context changes."""
        # Basic validation - in production this would be more sophisticated
        for change in context_changes:
            if not change.get("name"):
                return False
        return True
    
    async def _notify_context_changed(self, session_id: str, initiator_id: str, context_changes: List[Dict[str, Any]]) -> None:
        """Notify participants of context changes."""
        notification = {
            "type": "context_changed",
            "session_id": session_id,
            "initiator_id": initiator_id,
            "changes": context_changes,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._send_session_notification(session_id, notification, exclude_participant=initiator_id)
    
    async def _register_notification_listener(self, session_id: str, participant_id: str, preferences: Dict[str, Any]) -> None:
        """Register participant for notifications."""
        # Store notification preferences for participant
        # In production, this would set up actual notification channels
        pass
    
    async def _validate_security_token(self, token_string: str, participant_id: str, session_id: str) -> tuple[bool, Optional[CCOWSecurityToken]]:
        """Validate security token."""
        for token in self.security_tokens.values():
            if (token.token_id == token_string and 
                token.participant_id == participant_id and 
                token.session_id == session_id):
                
                # Check expiration
                expires_at = datetime.fromisoformat(token.expires_at)
                if expires_at > datetime.utcnow():
                    return True, token
                else:
                    # Token expired
                    del self.security_tokens[token.token_id]
                    return False, None
        
        return False, None
    
    async def _evaluate_permissions(self, participant: CCOWParticipant, requested_permissions: List[str], context_scope: Dict[str, Any]) -> List[str]:
        """Evaluate permission requests for participant."""
        granted_permissions = []
        
        # Basic permission evaluation based on participant type and security level
        if participant.participant_type == CCOWParticipantType.CONTEXT_MANAGER:
            # Context managers get all requested permissions
            granted_permissions = requested_permissions.copy()
        elif participant.participant_type == CCOWParticipantType.SECURE_CONTEXT_PARTICIPANT:
            # Secure participants get most permissions
            allowed_permissions = ["read_context", "write_context", "notify"]
            granted_permissions = [p for p in requested_permissions if p in allowed_permissions]
        else:
            # Regular participants get limited permissions
            allowed_permissions = ["read_context", "notify"]
            granted_permissions = [p for p in requested_permissions if p in allowed_permissions]
        
        return granted_permissions
    
    async def _get_session_security_info(self, session_id: str) -> Dict[str, Any]:
        """Get security information for session."""
        session = self.context_sessions[session_id]
        
        return {
            "session_locked": session.is_locked,
            "participant_count": len(session.participants),
            "security_levels": [p.security_level.value for p in session.participants],
            "has_secure_participants": any(p.security_level != CCOWSecurityLevel.NONE for p in session.participants)
        }
    
    async def _get_participant_context(self, session_id: str, participant_id: str) -> Dict[str, Any]:
        """Get context specific to participant."""
        participant = self.participant_registry.get(participant_id)
        if not participant:
            return {}
        
        return await self._get_filtered_context(session_id, participant.context_filters)
    
    async def _detect_context_conflicts(self, participant_contexts: Dict[str, Dict[str, Any]], sync_policies: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect conflicts between participant contexts."""
        conflicts = []
        
        # Compare context items across participants
        all_context_names = set()
        for context in participant_contexts.values():
            all_context_names.update(context.keys())
        
        for context_name in all_context_names:
            values = {}
            for participant_id, context in participant_contexts.items():
                if context_name in context:
                    values[participant_id] = context[context_name]["value"]
            
            # Check for conflicts (different values for same context)
            if len(set(str(v) for v in values.values())) > 1:
                conflicts.append({
                    "context_name": context_name,
                    "conflicting_values": values,
                    "conflict_type": "value_mismatch"
                })
        
        return conflicts
    
    async def _resolve_context_conflicts(self, conflicts: List[Dict[str, Any]], resolution_strategy: str, sync_policies: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Resolve context conflicts using specified strategy."""
        resolved_conflicts = []
        
        for conflict in conflicts:
            if resolution_strategy == "latest_wins":
                # Use the value from the most recent update (simplified)
                resolved_value = list(conflict["conflicting_values"].values())[0]
            elif resolution_strategy == "priority_based":
                # Use value from highest priority participant (simplified)
                resolved_value = list(conflict["conflicting_values"].values())[0]
            else:
                # Default to first value
                resolved_value = list(conflict["conflicting_values"].values())[0]
            
            resolved_conflicts.append({
                "context_name": conflict["context_name"],
                "resolution_strategy": resolution_strategy,
                "resolved_value": resolved_value,
                "original_conflict": conflict
            })
        
        return resolved_conflicts
    
    async def _merge_participant_contexts(self, participant_contexts: Dict[str, Dict[str, Any]], resolved_conflicts: List[Dict[str, Any]], sync_policies: Dict[str, Any]) -> Dict[str, Any]:
        """Merge participant contexts into synchronized context."""
        merged_context = {}
        
        # Apply resolved conflicts first
        for resolution in resolved_conflicts:
            context_name = resolution["context_name"]
            merged_context[context_name] = {
                "value": resolution["resolved_value"],
                "type": "custom",
                "timestamp": datetime.utcnow().isoformat(),
                "version": 1,
                "synchronized": True
            }
        
        # Add non-conflicting items
        all_context_names = set()
        for context in participant_contexts.values():
            all_context_names.update(context.keys())
        
        conflict_names = {r["context_name"] for r in resolved_conflicts}
        
        for context_name in all_context_names:
            if context_name not in conflict_names:
                # Take first occurrence of non-conflicting item
                for context in participant_contexts.values():
                    if context_name in context:
                        merged_context[context_name] = context[context_name]
                        break
        
        return merged_context
    
    async def _apply_partial_sync(self, session_id: str, synchronized_context: Dict[str, Any], scope: str) -> None:
        """Apply partial synchronization based on scope."""
        session = self.context_sessions[session_id]
        
        if scope == "patient_only":
            # Only sync patient-related context
            for name, item in synchronized_context.items():
                if item.get("type") == "patient":
                    session.context_items[name] = CCOWContextItem(**item)
        elif scope == "user_only":
            # Only sync user-related context
            for name, item in synchronized_context.items():
                if item.get("type") == "user":
                    session.context_items[name] = CCOWContextItem(**item)
        # Add more scope types as needed
    
    async def _notify_synchronization_complete(self, session_id: str, participants: List[CCOWParticipant], synchronized_context: Dict[str, Any]) -> None:
        """Notify participants that synchronization is complete."""
        notification = {
            "type": "synchronization_complete",
            "session_id": session_id,
            "synchronized_items": list(synchronized_context.keys()),
            "participant_count": len(participants),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._send_session_notification(session_id, notification)
    
    async def _lock_session(self, session_id: str, lock_reason: str) -> None:
        """Lock session for exclusive operations."""
        session = self.context_sessions[session_id]
        session.is_locked = True
        session.locked_by = lock_reason
        session.lock_timestamp = datetime.utcnow().isoformat()
    
    async def _unlock_session(self, session_id: str) -> None:
        """Unlock session."""
        session = self.context_sessions[session_id]
        session.is_locked = False
        session.locked_by = None
        session.lock_timestamp = None
    
    # Visual integration methods
    
    async def _generate_integration_config(self, session: CCOWContextSession, integration_type: str, visual_policies: Dict[str, Any], user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visual integration configuration."""
        return {
            "integration_type": integration_type,
            "context_highlighting": visual_policies.get("highlight_shared_context", True),
            "synchronized_scrolling": visual_policies.get("synchronized_scrolling", False),
            "unified_toolbar": visual_policies.get("unified_toolbar", True),
            "color_coding": user_preferences.get("enable_color_coding", True),
            "notification_style": user_preferences.get("notification_style", "subtle"),
            "layout_mode": user_preferences.get("layout_mode", "side_by_side")
        }
    
    async def _calculate_layout_adjustments(self, application_layouts: List[Dict[str, Any]], integration_config: Dict[str, Any], visual_policies: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate layout adjustments for visual integration."""
        adjustments = []
        
        for layout in application_layouts:
            adjustment = {
                "application_id": layout.get("application_id"),
                "original_size": layout.get("size", {}),
                "recommended_size": {},
                "position_adjustment": {},
                "visual_elements": []
            }
            
            # Calculate recommended size based on integration config
            if integration_config.get("layout_mode") == "side_by_side":
                adjustment["recommended_size"] = {
                    "width": layout.get("size", {}).get("width", 800) // 2,
                    "height": layout.get("size", {}).get("height", 600)
                }
            
            adjustments.append(adjustment)
        
        return adjustments
    
    async def _generate_visual_cues(self, session: CCOWContextSession, integration_config: Dict[str, Any], user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visual cues for context sharing."""
        return {
            "context_indicators": {
                "patient_highlight_color": "#4CAF50",
                "user_highlight_color": "#2196F3",
                "encounter_highlight_color": "#FF9800"
            },
            "notification_styles": {
                "context_change": {
                    "background_color": "#E3F2FD",
                    "border_color": "#2196F3",
                    "duration": 3000
                },
                "sync_complete": {
                    "background_color": "#E8F5E8",
                    "border_color": "#4CAF50",
                    "duration": 2000
                }
            },
            "toolbar_elements": {
                "sync_status_indicator": True,
                "participant_count_display": True,
                "context_lock_indicator": True
            }
        }
    
    async def _notify_visual_integration_update(self, session_id: str, integration_config: Dict[str, Any], visual_cues: Dict[str, Any]) -> None:
        """Notify participants of visual integration updates."""
        notification = {
            "type": "visual_integration_update",
            "session_id": session_id,
            "integration_config": integration_config,
            "visual_cues": visual_cues,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._send_session_notification(session_id, notification)
    
    # Notification and messaging methods
    
    async def _send_session_notification(self, session_id: str, notification: Dict[str, Any], exclude_participant: Optional[str] = None) -> None:
        """Send notification to all participants in session."""
        session = self.context_sessions.get(session_id)
        if not session:
            return
        
        for participant in session.participants:
            if exclude_participant and participant.participant_id == exclude_participant:
                continue
            
            await self._send_participant_notification(participant, notification)
    
    async def _send_participant_notification(self, participant: CCOWParticipant, notification: Dict[str, Any]) -> None:
        """Send notification to specific participant."""
        # In production, this would use actual notification channels (HTTP, WebSocket, etc.)
        self.logger.debug(f"Notification to {participant.participant_id}: {notification['type']}")
    
    async def _notify_transaction_pending(self, transaction: CCOWTransaction) -> None:
        """Notify participants about pending transaction."""
        notification = {
            "type": "transaction_pending",
            "transaction_id": transaction.transaction_id,
            "proposed_changes": transaction.proposed_changes,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._send_session_notification(transaction.session_id, notification, exclude_participant=transaction.initiator_id)
    
    async def _notify_transaction_committed(self, transaction: CCOWTransaction) -> None:
        """Notify participants that transaction was committed."""
        notification = {
            "type": "transaction_committed",
            "transaction_id": transaction.transaction_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._send_session_notification(transaction.session_id, notification)
    
    async def _notify_transaction_cancelled(self, transaction: CCOWTransaction) -> None:
        """Notify participants that transaction was cancelled."""
        notification = {
            "type": "transaction_cancelled",
            "transaction_id": transaction.transaction_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._send_session_notification(transaction.session_id, notification)
    
    async def _request_transaction_prepare(self, participant: CCOWParticipant, transaction: CCOWTransaction) -> str:
        """Request participant to prepare for transaction."""
        # In production, this would make actual request to participant
        # For now, simulate agreement
        return "prepared"
    
    # Threading and cleanup methods
    
    def _start_notification_processing(self) -> None:
        """Start background notification processing."""
        self._stop_notifications = False
        self._notification_thread = threading.Thread(target=self._notification_worker)
        self._notification_thread.daemon = True
        self._notification_thread.start()
    
    def _stop_notification_processing(self) -> None:
        """Stop background notification processing."""
        self._stop_notifications = True
        if self._notification_thread:
            self._notification_thread.join(timeout=5.0)
    
    def _notification_worker(self) -> None:
        """Background worker for processing notifications."""
        while not self._stop_notifications:
            try:
                # Process pending notifications
                # In production, this would handle queued notifications
                threading.Event().wait(1.0)  # Sleep for 1 second
            except Exception as e:
                self.logger.error(f"Notification worker error: {str(e)}")
    
    async def _cleanup_sessions(self) -> None:
        """Clean up active sessions and resources."""
        for session_id in list(self.context_sessions.keys()):
            session = self.context_sessions[session_id]
            
            # Notify participants of session cleanup
            cleanup_notification = {
                "type": "session_cleanup",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._send_session_notification(session_id, cleanup_notification)
        
        # Clear all sessions
        self.context_sessions.clear()
        self.active_transactions.clear()
        self.security_tokens.clear()
        self.participant_registry.clear()