"""
SMART on FHIR Security Agent for Vita Agents.
Provides comprehensive OAuth2, SMART on FHIR, and advanced security capabilities.
"""

import asyncio
import json
import base64
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Union
import structlog
from pydantic import BaseModel, Field
import jwt
import httpx
from urllib.parse import urlencode, parse_qs, urlparse

from vita_agents.core.agent import HealthcareAgent, AgentCapability, TaskRequest
from vita_agents.core.config import get_settings


logger = structlog.get_logger(__name__)


class SMARTScope(BaseModel):
    """SMART on FHIR scope definition."""
    
    resource_type: str
    permission: str  # read, write, *
    clinical_scope: Optional[str] = None  # patient, user, system


class OAuth2Token(BaseModel):
    """OAuth2 token response."""
    
    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    patient: Optional[str] = None
    encounter: Optional[str] = None


class AuthorizationCode(BaseModel):
    """Authorization code for OAuth2 flow."""
    
    code: str
    state: str
    client_id: str
    redirect_uri: str
    scope: str
    code_challenge: Optional[str] = None
    code_challenge_method: Optional[str] = None


class SMARTLaunchContext(BaseModel):
    """SMART launch context."""
    
    iss: str  # FHIR server URL
    launch: Optional[str] = None  # Launch token
    patient: Optional[str] = None
    encounter: Optional[str] = None
    location: Optional[str] = None
    resource: Optional[str] = None
    intent: Optional[str] = None


class SecurityEvent(BaseModel):
    """Security audit event."""
    
    event_type: str
    timestamp: datetime
    user_id: Optional[str] = None
    patient_id: Optional[str] = None
    resource_type: Optional[str] = None
    action: str
    outcome: str  # success, failure, warning
    details: Dict[str, Any] = {}


class ConsentDirective(BaseModel):
    """Patient consent directive."""
    
    patient_id: str
    consent_type: str  # research, treatment, sharing
    status: str  # active, inactive, proposed, rejected
    category: List[str]
    purpose: List[str]
    data_types: List[str]
    recipients: List[str]
    period: Optional[Dict[str, str]] = None
    restrictions: Dict[str, Any] = {}


class SMARTSecurityAgent(HealthcareAgent):
    """
    SMART on FHIR Security Agent.
    
    Capabilities:
    - OAuth2 authorization flows (authorization code, client credentials)
    - SMART on FHIR app launch (standalone, EHR-integrated)
    - Token validation and refresh
    - Scope-based access control
    - Patient consent management
    - Security audit logging
    - Advanced authentication (PKCE, dynamic client registration)
    - Multi-tenant security
    """
    
    def __init__(
        self,
        agent_id: str = "smart-security-agent",
        name: str = "SMART Security Agent",
        description: str = "OAuth2 and SMART on FHIR security management",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None,
    ):
        capabilities = [
            AgentCapability(
                name="authorize_smart_app",
                description="Authorize SMART on FHIR application using OAuth2",
                input_schema={
                    "type": "object",
                    "properties": {
                        "client_id": {"type": "string"},
                        "redirect_uri": {"type": "string"},
                        "scope": {"type": "string"},
                        "launch_context": {"type": "object"},
                        "use_pkce": {"type": "boolean"}
                    },
                    "required": ["client_id", "redirect_uri", "scope"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "authorization_url": {"type": "string"},
                        "state": {"type": "string"},
                        "code_challenge": {"type": "string"}
                    }
                }
            ),
            AgentCapability(
                name="exchange_authorization_code",
                description="Exchange authorization code for access token",
                input_schema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "client_id": {"type": "string"},
                        "client_secret": {"type": "string"},
                        "redirect_uri": {"type": "string"},
                        "code_verifier": {"type": "string"}
                    },
                    "required": ["code", "client_id", "redirect_uri"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "access_token": {"type": "string"},
                        "token_type": {"type": "string"},
                        "expires_in": {"type": "integer"},
                        "scope": {"type": "string"},
                        "patient": {"type": "string"}
                    }
                }
            ),
            AgentCapability(
                name="validate_token",
                description="Validate and introspect OAuth2 token",
                input_schema={
                    "type": "object",
                    "properties": {
                        "access_token": {"type": "string"},
                        "resource_request": {"type": "object"}
                    },
                    "required": ["access_token"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "valid": {"type": "boolean"},
                        "scopes": {"type": "array"},
                        "patient_id": {"type": "string"},
                        "expires_at": {"type": "string"}
                    }
                }
            ),
            AgentCapability(
                name="manage_patient_consent",
                description="Manage patient consent directives",
                input_schema={
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string"},
                        "consent_directive": {"type": "object"},
                        "action": {"type": "string"}
                    },
                    "required": ["patient_id", "action"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "consent_status": {"type": "string"},
                        "consent_id": {"type": "string"},
                        "effective_permissions": {"type": "object"}
                    }
                }
            ),
            AgentCapability(
                name="audit_security_event",
                description="Log and audit security events",
                input_schema={
                    "type": "object",
                    "properties": {
                        "event_type": {"type": "string"},
                        "user_id": {"type": "string"},
                        "resource_details": {"type": "object"},
                        "outcome": {"type": "string"}
                    },
                    "required": ["event_type", "outcome"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "audit_id": {"type": "string"},
                        "timestamp": {"type": "string"},
                        "compliance_status": {"type": "string"}
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
        
        # Security configuration
        self.fhir_servers = self._initialize_fhir_servers()
        self.registered_clients = self._initialize_registered_clients()
        self.supported_scopes = self._initialize_supported_scopes()
        self.consent_directives = {}
        self.security_events = []
        
        # Register task handlers
        self.register_task_handler("authorize_smart_app", self._authorize_smart_app)
        self.register_task_handler("exchange_authorization_code", self._exchange_authorization_code)
        self.register_task_handler("validate_token", self._validate_token)
        self.register_task_handler("manage_patient_consent", self._manage_patient_consent)
        self.register_task_handler("audit_security_event", self._audit_security_event)
    
    def _initialize_fhir_servers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize FHIR server configurations."""
        return {
            "ehr_sandbox": {
                "base_url": "https://launch.smarthealthit.org/v/r4/fhir",
                "authorization_endpoint": "https://launch.smarthealthit.org/v/r4/auth/authorize",
                "token_endpoint": "https://launch.smarthealthit.org/v/r4/auth/token",
                "introspection_endpoint": "https://launch.smarthealthit.org/v/r4/auth/introspect",
                "capabilities": ["launch-ehr", "launch-standalone", "client-public", "client-confidential-symmetric"],
                "supported_grant_types": ["authorization_code", "client_credentials"],
                "supported_scopes": ["openid", "fhirUser", "launch", "launch/patient", "patient/*.*", "user/*.*"]
            },
            "epic_sandbox": {
                "base_url": "https://fhir.epic.com/interconnect-fhir-oauth",
                "authorization_endpoint": "https://fhir.epic.com/interconnect-fhir-oauth/oauth2/authorize",
                "token_endpoint": "https://fhir.epic.com/interconnect-fhir-oauth/oauth2/token",
                "registration_endpoint": "https://fhir.epic.com/interconnect-fhir-oauth/oauth2/register",
                "capabilities": ["launch-ehr", "client-confidential-symmetric"],
                "supported_grant_types": ["authorization_code"],
                "supported_scopes": ["patient/*.read", "user/*.read", "launch"]
            },
            "cerner_sandbox": {
                "base_url": "https://fhir-ehr-code.cerner.com/r4/ec2458f2-1e24-41c8-b71b-0e701af7583d",
                "authorization_endpoint": "https://authorization.cerner.com/tenants/ec2458f2-1e24-41c8-b71b-0e701af7583d/protocols/oauth2/profiles/smart-v1/personas/provider/authorize",
                "token_endpoint": "https://authorization.cerner.com/tenants/ec2458f2-1e24-41c8-b71b-0e701af7583d/protocols/oauth2/profiles/smart-v1/token",
                "capabilities": ["launch-ehr", "launch-standalone", "client-public"],
                "supported_grant_types": ["authorization_code"],
                "supported_scopes": ["patient/*.read", "user/*.read", "launch", "launch/patient"]
            }
        }
    
    def _initialize_registered_clients(self) -> Dict[str, Dict[str, Any]]:
        """Initialize registered OAuth2 clients."""
        return {
            "vita_agents_ehr": {
                "client_id": "vita_agents_ehr_app",
                "client_secret": "secure_client_secret_123",
                "client_type": "confidential",
                "redirect_uris": ["https://app.vita-agents.com/oauth/callback"],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "scopes": ["patient/*.read", "patient/*.write", "user/*.read", "launch", "launch/patient"]
            },
            "vita_agents_standalone": {
                "client_id": "vita_agents_standalone",
                "client_secret": None,  # Public client
                "client_type": "public",
                "redirect_uris": ["https://app.vita-agents.com/standalone/callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "scopes": ["patient/*.read", "launch/patient", "offline_access"]
            },
            "vita_agents_research": {
                "client_id": "vita_agents_research",
                "client_secret": "research_secret_456",
                "client_type": "confidential",
                "redirect_uris": ["https://research.vita-agents.com/oauth/callback"],
                "grant_types": ["client_credentials"],
                "response_types": [],
                "scopes": ["system/*.read", "system/Patient.read", "system/Observation.read"]
            }
        }
    
    def _initialize_supported_scopes(self) -> Dict[str, SMARTScope]:
        """Initialize supported SMART scopes."""
        scopes = {}
        
        # Patient scopes
        for resource in ["Patient", "Observation", "Condition", "MedicationRequest", "DiagnosticReport", "Encounter"]:
            for permission in ["read", "write", "*"]:
                scope_name = f"patient/{resource}.{permission}"
                scopes[scope_name] = SMARTScope(
                    resource_type=resource,
                    permission=permission,
                    clinical_scope="patient"
                )
        
        # User scopes
        for resource in ["Patient", "Observation", "Condition", "MedicationRequest", "DiagnosticReport"]:
            for permission in ["read", "write", "*"]:
                scope_name = f"user/{resource}.{permission}"
                scopes[scope_name] = SMARTScope(
                    resource_type=resource,
                    permission=permission,
                    clinical_scope="user"
                )
        
        # System scopes
        for resource in ["Patient", "Observation", "Condition"]:
            for permission in ["read", "write", "*"]:
                scope_name = f"system/{resource}.{permission}"
                scopes[scope_name] = SMARTScope(
                    resource_type=resource,
                    permission=permission,
                    clinical_scope="system"
                )
        
        # Special scopes
        scopes["openid"] = SMARTScope(resource_type="openid", permission="read")
        scopes["profile"] = SMARTScope(resource_type="profile", permission="read")
        scopes["fhirUser"] = SMARTScope(resource_type="fhirUser", permission="read")
        scopes["launch"] = SMARTScope(resource_type="launch", permission="read")
        scopes["launch/patient"] = SMARTScope(resource_type="launch", permission="read", clinical_scope="patient")
        scopes["launch/encounter"] = SMARTScope(resource_type="launch", permission="read", clinical_scope="encounter")
        scopes["offline_access"] = SMARTScope(resource_type="offline_access", permission="read")
        
        return scopes
    
    async def _on_start(self) -> None:
        """Initialize SMART Security agent."""
        self.logger.info("Starting SMART Security agent",
                        fhir_servers=len(self.fhir_servers),
                        registered_clients=len(self.registered_clients))
        
        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Initialize token storage (in production, use secure storage)
        self.active_tokens = {}
        self.authorization_codes = {}
        
        self.logger.info("SMART Security agent initialized")
    
    async def _on_stop(self) -> None:
        """Clean up SMART Security agent."""
        if hasattr(self, 'http_client'):
            await self.http_client.aclose()
        self.logger.info("SMART Security agent stopped")
    
    async def _authorize_smart_app(self, task: TaskRequest) -> Dict[str, Any]:
        """Authorize SMART on FHIR application using OAuth2."""
        try:
            client_id = task.parameters.get("client_id")
            redirect_uri = task.parameters.get("redirect_uri")
            scope = task.parameters.get("scope")
            launch_context = task.parameters.get("launch_context", {})
            use_pkce = task.parameters.get("use_pkce", True)
            
            if not client_id or not redirect_uri or not scope:
                raise ValueError("client_id, redirect_uri, and scope are required")
            
            # Validate client registration
            if client_id not in self.registered_clients:
                raise ValueError(f"Unknown client: {client_id}")
            
            client = self.registered_clients[client_id]
            
            # Validate redirect URI
            if redirect_uri not in client["redirect_uris"]:
                raise ValueError("Invalid redirect URI")
            
            # Generate authorization parameters
            state = self._generate_state()
            
            # PKCE support
            code_challenge = None
            code_verifier = None
            if use_pkce and client["client_type"] == "public":
                code_verifier = self._generate_code_verifier()
                code_challenge = self._generate_code_challenge(code_verifier)
            
            # Build authorization URL
            auth_params = {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scope": scope,
                "state": state
            }
            
            # Add launch context
            if launch_context.get("iss"):
                auth_params["iss"] = launch_context["iss"]
            if launch_context.get("launch"):
                auth_params["launch"] = launch_context["launch"]
            
            # Add PKCE parameters
            if code_challenge:
                auth_params["code_challenge"] = code_challenge
                auth_params["code_challenge_method"] = "S256"
            
            # Store authorization request
            auth_code_data = AuthorizationCode(
                code="",  # Will be filled when code is issued
                state=state,
                client_id=client_id,
                redirect_uri=redirect_uri,
                scope=scope,
                code_challenge=code_challenge
            )
            
            self.authorization_codes[state] = {
                "auth_data": auth_code_data,
                "code_verifier": code_verifier,
                "launch_context": launch_context,
                "timestamp": datetime.utcnow()
            }
            
            # Determine authorization endpoint
            authorization_endpoint = self._get_authorization_endpoint(launch_context.get("iss"))
            authorization_url = f"{authorization_endpoint}?{urlencode(auth_params)}"
            
            # Log security event
            await self._log_security_event(
                "authorization_request",
                client_id=client_id,
                outcome="success",
                details={
                    "scope": scope,
                    "use_pkce": use_pkce,
                    "launch_context": launch_context
                }
            )
            
            result = {
                "authorization_url": authorization_url,
                "state": state,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if code_challenge:
                result["code_challenge"] = code_challenge
                result["code_verifier"] = code_verifier
            
            return result
            
        except Exception as e:
            await self._log_security_event(
                "authorization_request",
                outcome="failure",
                details={"error": str(e)}
            )
            self.logger.error("Authorization request failed", error=str(e), task_id=task.id)
            raise
    
    async def _exchange_authorization_code(self, task: TaskRequest) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        try:
            code = task.parameters.get("code")
            client_id = task.parameters.get("client_id")
            client_secret = task.parameters.get("client_secret")
            redirect_uri = task.parameters.get("redirect_uri")
            code_verifier = task.parameters.get("code_verifier")
            
            if not code or not client_id or not redirect_uri:
                raise ValueError("code, client_id, and redirect_uri are required")
            
            # Validate client
            if client_id not in self.registered_clients:
                raise ValueError(f"Unknown client: {client_id}")
            
            client = self.registered_clients[client_id]
            
            # Validate client secret for confidential clients
            if client["client_type"] == "confidential" and client_secret != client["client_secret"]:
                raise ValueError("Invalid client credentials")
            
            # Build token request
            token_params = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id
            }
            
            if client_secret:
                token_params["client_secret"] = client_secret
            
            if code_verifier:
                token_params["code_verifier"] = code_verifier
            
            # Get token endpoint
            token_endpoint = self._get_token_endpoint()
            
            # Exchange code for token
            response = await self.http_client.post(
                token_endpoint,
                data=token_params,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if response.status_code != 200:
                raise ValueError(f"Token exchange failed: {response.text}")
            
            token_data = response.json()
            
            # Create OAuth2 token
            oauth_token = OAuth2Token(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in"),
                refresh_token=token_data.get("refresh_token"),
                scope=token_data.get("scope"),
                patient=token_data.get("patient"),
                encounter=token_data.get("encounter")
            )
            
            # Store token
            token_id = self._generate_token_id()
            self.active_tokens[token_id] = {
                "token": oauth_token,
                "client_id": client_id,
                "issued_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(seconds=oauth_token.expires_in or 3600)
            }
            
            # Log security event
            await self._log_security_event(
                "token_issued",
                client_id=client_id,
                outcome="success",
                details={
                    "scope": oauth_token.scope,
                    "patient": oauth_token.patient,
                    "expires_in": oauth_token.expires_in
                }
            )
            
            return {
                "access_token": oauth_token.access_token,
                "token_type": oauth_token.token_type,
                "expires_in": oauth_token.expires_in,
                "scope": oauth_token.scope,
                "patient": oauth_token.patient,
                "encounter": oauth_token.encounter,
                "refresh_token": oauth_token.refresh_token,
                "token_id": token_id,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            await self._log_security_event(
                "token_exchange",
                outcome="failure",
                details={"error": str(e)}
            )
            self.logger.error("Token exchange failed", error=str(e), task_id=task.id)
            raise
    
    async def _validate_token(self, task: TaskRequest) -> Dict[str, Any]:
        """Validate and introspect OAuth2 token."""
        try:
            access_token = task.parameters.get("access_token")
            resource_request = task.parameters.get("resource_request", {})
            
            if not access_token:
                raise ValueError("access_token is required")
            
            # Find token in active tokens
            token_info = None
            for token_id, token_data in self.active_tokens.items():
                if token_data["token"].access_token == access_token:
                    token_info = token_data
                    break
            
            if not token_info:
                return {
                    "valid": False,
                    "error": "token_not_found",
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Check token expiration
            if datetime.utcnow() > token_info["expires_at"]:
                return {
                    "valid": False,
                    "error": "token_expired",
                    "expired_at": token_info["expires_at"].isoformat(),
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            token = token_info["token"]
            
            # Validate scopes for resource request
            access_granted = True
            required_scope = None
            
            if resource_request:
                resource_type = resource_request.get("resourceType")
                method = resource_request.get("method", "GET")
                patient_id = resource_request.get("patient")
                
                if resource_type and method:
                    permission = "read" if method == "GET" else "write"
                    
                    # Check patient context
                    if token.patient and patient_id and token.patient != patient_id:
                        access_granted = False
                        required_scope = f"Access denied: token patient ({token.patient}) != requested patient ({patient_id})"
                    
                    # Check scope permissions
                    if access_granted:
                        required_scope = f"patient/{resource_type}.{permission}"
                        token_scopes = token.scope.split() if token.scope else []
                        
                        # Check for specific scope or wildcard
                        scope_patterns = [
                            required_scope,
                            f"patient/{resource_type}.*",
                            f"patient/*.*",
                            f"user/{resource_type}.{permission}",
                            f"user/{resource_type}.*",
                            f"user/*.*"
                        ]
                        
                        access_granted = any(pattern in token_scopes for pattern in scope_patterns)
            
            # Log validation event
            await self._log_security_event(
                "token_validation",
                outcome="success" if access_granted else "failure",
                details={
                    "token_valid": True,
                    "access_granted": access_granted,
                    "required_scope": required_scope,
                    "patient_context": token.patient
                }
            )
            
            return {
                "valid": True,
                "access_granted": access_granted,
                "scopes": token.scope.split() if token.scope else [],
                "patient_id": token.patient,
                "encounter_id": token.encounter,
                "expires_at": token_info["expires_at"].isoformat(),
                "client_id": token_info["client_id"],
                "required_scope": required_scope,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            await self._log_security_event(
                "token_validation",
                outcome="failure",
                details={"error": str(e)}
            )
            self.logger.error("Token validation failed", error=str(e), task_id=task.id)
            raise
    
    async def _manage_patient_consent(self, task: TaskRequest) -> Dict[str, Any]:
        """Manage patient consent directives."""
        try:
            patient_id = task.parameters.get("patient_id")
            consent_directive = task.parameters.get("consent_directive", {})
            action = task.parameters.get("action")  # create, update, revoke, query
            
            if not patient_id or not action:
                raise ValueError("patient_id and action are required")
            
            consent_id = consent_directive.get("id") or self._generate_consent_id()
            
            if action == "create":
                consent = ConsentDirective(
                    patient_id=patient_id,
                    consent_type=consent_directive.get("consent_type", "treatment"),
                    status="active",
                    category=consent_directive.get("category", ["treatment"]),
                    purpose=consent_directive.get("purpose", ["TREAT"]),
                    data_types=consent_directive.get("data_types", ["*"]),
                    recipients=consent_directive.get("recipients", ["healthcare-provider"]),
                    period=consent_directive.get("period"),
                    restrictions=consent_directive.get("restrictions", {})
                )
                
                if patient_id not in self.consent_directives:
                    self.consent_directives[patient_id] = {}
                
                self.consent_directives[patient_id][consent_id] = consent
                
            elif action == "update":
                if patient_id not in self.consent_directives or consent_id not in self.consent_directives[patient_id]:
                    raise ValueError("Consent directive not found")
                
                consent = self.consent_directives[patient_id][consent_id]
                
                # Update consent fields
                for field, value in consent_directive.items():
                    if hasattr(consent, field):
                        setattr(consent, field, value)
                
            elif action == "revoke":
                if patient_id not in self.consent_directives or consent_id not in self.consent_directives[patient_id]:
                    raise ValueError("Consent directive not found")
                
                consent = self.consent_directives[patient_id][consent_id]
                consent.status = "inactive"
                
            elif action == "query":
                consents = self.consent_directives.get(patient_id, {})
                return {
                    "consent_directives": [consent.dict() for consent in consents.values()],
                    "patient_id": patient_id,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            else:
                raise ValueError(f"Unknown action: {action}")
            
            # Calculate effective permissions
            effective_permissions = self._calculate_effective_permissions(patient_id)
            
            # Log consent event
            await self._log_security_event(
                f"consent_{action}",
                patient_id=patient_id,
                outcome="success",
                details={
                    "consent_id": consent_id,
                    "consent_type": consent_directive.get("consent_type"),
                    "effective_permissions": effective_permissions
                }
            )
            
            consent = self.consent_directives[patient_id][consent_id]
            
            return {
                "consent_status": consent.status,
                "consent_id": consent_id,
                "effective_permissions": effective_permissions,
                "patient_id": patient_id,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            await self._log_security_event(
                f"consent_{action}",
                patient_id=patient_id,
                outcome="failure",
                details={"error": str(e)}
            )
            self.logger.error("Consent management failed", error=str(e), task_id=task.id)
            raise
    
    async def _audit_security_event(self, task: TaskRequest) -> Dict[str, Any]:
        """Log and audit security events."""
        try:
            event_type = task.parameters.get("event_type")
            user_id = task.parameters.get("user_id")
            resource_details = task.parameters.get("resource_details", {})
            outcome = task.parameters.get("outcome")
            
            if not event_type or not outcome:
                raise ValueError("event_type and outcome are required")
            
            # Create security event
            security_event = SecurityEvent(
                event_type=event_type,
                timestamp=datetime.utcnow(),
                user_id=user_id,
                patient_id=resource_details.get("patient_id"),
                resource_type=resource_details.get("resource_type"),
                action=resource_details.get("action", "unknown"),
                outcome=outcome,
                details=resource_details
            )
            
            # Store security event
            audit_id = self._generate_audit_id()
            self.security_events.append({
                "audit_id": audit_id,
                "event": security_event,
                "compliance_status": self._check_compliance_status(security_event)
            })
            
            # Check for security violations
            violations = self._check_security_violations(security_event)
            
            return {
                "audit_id": audit_id,
                "timestamp": security_event.timestamp.isoformat(),
                "compliance_status": "compliant" if not violations else "violation",
                "violations": violations,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            self.logger.error("Security audit failed", error=str(e), task_id=task.id)
            raise
    
    def _generate_state(self) -> str:
        """Generate OAuth2 state parameter."""
        return secrets.token_urlsafe(32)
    
    def _generate_code_verifier(self) -> str:
        """Generate PKCE code verifier."""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
    
    def _generate_code_challenge(self, verifier: str) -> str:
        """Generate PKCE code challenge."""
        digest = hashlib.sha256(verifier.encode('utf-8')).digest()
        return base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')
    
    def _generate_token_id(self) -> str:
        """Generate unique token ID."""
        return f"token_{secrets.token_hex(16)}"
    
    def _generate_consent_id(self) -> str:
        """Generate unique consent ID."""
        return f"consent_{secrets.token_hex(16)}"
    
    def _generate_audit_id(self) -> str:
        """Generate unique audit ID."""
        return f"audit_{secrets.token_hex(16)}"
    
    def _get_authorization_endpoint(self, iss: Optional[str] = None) -> str:
        """Get authorization endpoint for FHIR server."""
        if iss:
            for server_config in self.fhir_servers.values():
                if server_config["base_url"] == iss:
                    return server_config["authorization_endpoint"]
        
        # Default to SMART sandbox
        return self.fhir_servers["ehr_sandbox"]["authorization_endpoint"]
    
    def _get_token_endpoint(self, iss: Optional[str] = None) -> str:
        """Get token endpoint for FHIR server."""
        if iss:
            for server_config in self.fhir_servers.values():
                if server_config["base_url"] == iss:
                    return server_config["token_endpoint"]
        
        # Default to SMART sandbox
        return self.fhir_servers["ehr_sandbox"]["token_endpoint"]
    
    def _calculate_effective_permissions(self, patient_id: str) -> Dict[str, Any]:
        """Calculate effective permissions based on consent directives."""
        if patient_id not in self.consent_directives:
            return {"default": "full_access"}
        
        permissions = {
            "data_access": [],
            "sharing_allowed": [],
            "restrictions": []
        }
        
        for consent in self.consent_directives[patient_id].values():
            if consent.status == "active":
                permissions["data_access"].extend(consent.data_types)
                permissions["sharing_allowed"].extend(consent.recipients)
                
                if consent.restrictions:
                    permissions["restrictions"].append(consent.restrictions)
        
        return permissions
    
    def _check_compliance_status(self, event: SecurityEvent) -> str:
        """Check compliance status of security event."""
        # Simplified compliance checking
        if event.outcome == "failure":
            return "requires_review"
        elif event.event_type in ["unauthorized_access", "data_breach"]:
            return "violation"
        else:
            return "compliant"
    
    def _check_security_violations(self, event: SecurityEvent) -> List[str]:
        """Check for security violations."""
        violations = []
        
        if event.outcome == "failure" and event.event_type == "authorization_request":
            violations.append("Failed authorization attempt")
        
        if event.event_type == "token_validation" and event.details.get("access_granted") is False:
            violations.append("Unauthorized resource access attempt")
        
        if event.patient_id and event.event_type == "data_access":
            # Check consent violations
            patient_consents = self.consent_directives.get(event.patient_id, {})
            if not patient_consents:
                violations.append("Data access without documented consent")
        
        return violations
    
    async def _log_security_event(self, event_type: str, outcome: str, **kwargs) -> None:
        """Log security event."""
        event = SecurityEvent(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            outcome=outcome,
            details=kwargs
        )
        
        audit_id = self._generate_audit_id()
        self.security_events.append({
            "audit_id": audit_id,
            "event": event,
            "compliance_status": self._check_compliance_status(event)
        })
        
        self.logger.info("Security event logged",
                        event_type=event_type,
                        outcome=outcome,
                        audit_id=audit_id)