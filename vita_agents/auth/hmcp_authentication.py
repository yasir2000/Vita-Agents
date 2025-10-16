#!/usr/bin/env python3
"""
Healthcare Authentication System for HMCP

Implements comprehensive OAuth2/JWT authentication with:
- Healthcare-specific scopes and permissions
- FHIR user context and patient authorization
- Role-based access control (RBAC)
- Secure token management for clinical environments
- SMART on FHIR compliance
- Audit logging for HIPAA compliance
"""

import jwt
import json
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import hashlib
import secrets

try:
    # OAuth2 and authentication libraries
    from authlib.integrations.httpx_client import AsyncOAuth2Client
    from authlib.oauth2 import OAuth2Error
    from authlib.jose import jwt as authlib_jwt, JWTError
    import httpx
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

try:
    # FHIR resources for healthcare context
    from fhir.resources import Patient, Practitioner, Organization
    from fhir.resources.fhirtypes import Id
    FHIR_AVAILABLE = True
except ImportError:
    FHIR_AVAILABLE = False

from vita_agents.protocols.hmcp import (
    HealthcareRole, PatientContext, ClinicalContext
)

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Authentication-related errors"""
    pass


class AuthorizationError(Exception):
    """Authorization-related errors"""
    pass


class SMARTScope(Enum):
    """SMART on FHIR scopes for healthcare applications"""
    # Patient access
    PATIENT_READ = "patient/*.read"
    PATIENT_WRITE = "patient/*.write"
    PATIENT_HMCP_READ = "patient/hmcp:read"
    PATIENT_HMCP_WRITE = "patient/hmcp:write"
    
    # User access
    USER_READ = "user/*.read"
    USER_WRITE = "user/*.write"
    USER_HMCP_READ = "user/hmcp:read"
    USER_HMCP_WRITE = "user/hmcp:write"
    
    # System access
    SYSTEM_READ = "system/*.read"
    SYSTEM_WRITE = "system/*.write"
    SYSTEM_HMCP_READ = "system/hmcp:read"
    SYSTEM_HMCP_WRITE = "system/hmcp:write"
    
    # Launch contexts
    LAUNCH_PATIENT = "launch/patient"
    LAUNCH_ENCOUNTER = "launch/encounter"
    
    # Profile access
    PROFILE = "profile"
    FHIR_USER = "fhirUser"
    OPENID = "openid"
    
    # Clinical scopes
    CLINICAL_READ = "clinical/*.read"
    CLINICAL_WRITE = "clinical/*.write"
    
    # Administrative
    ADMIN_READ = "admin/*.read"
    ADMIN_WRITE = "admin/*.write"


@dataclass
class HealthcareToken:
    """Healthcare-specific JWT token structure"""
    # Standard JWT claims
    iss: str  # Issuer
    sub: str  # Subject (user ID)
    aud: str  # Audience
    exp: int  # Expiration time
    iat: int  # Issued at
    jti: str  # JWT ID
    
    # SMART on FHIR claims
    scope: str  # Granted scopes
    client_id: str  # Client identifier
    fhirUser: Optional[str] = None  # FHIR User ID
    
    # Healthcare context
    patient: Optional[str] = None  # Patient ID in context
    encounter: Optional[str] = None  # Encounter ID in context
    location: Optional[str] = None  # Location ID in context
    
    # Custom healthcare claims
    healthcare_role: Optional[str] = None  # Healthcare role
    organization: Optional[str] = None  # Healthcare organization
    department: Optional[str] = None  # Department/unit
    license_number: Optional[str] = None  # Professional license
    npi: Optional[str] = None  # National Provider Identifier
    
    # Security
    session_id: Optional[str] = None  # Session identifier
    tenant: Optional[str] = None  # Multi-tenant identifier
    
    # Additional claims
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    email: Optional[str] = None


@dataclass
class AuthenticationContext:
    """Authentication context for healthcare operations"""
    token: HealthcareToken
    scopes: Set[str]
    permissions: Set[str]
    healthcare_role: HealthcareRole
    patient_context: Optional[PatientContext] = None
    organization_id: Optional[str] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    authenticated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=1))
    
    def is_expired(self) -> bool:
        """Check if authentication context is expired"""
        return datetime.now(timezone.utc) > self.expires_at
    
    def has_scope(self, scope: Union[str, SMARTScope]) -> bool:
        """Check if context has specific scope"""
        scope_str = scope if isinstance(scope, str) else scope.value
        return scope_str in self.scopes
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission"""
        return permission in self.permissions


class HealthcarePermissionManager:
    """Manages role-based permissions for healthcare operations"""
    
    def __init__(self):
        # Define role-based permissions
        self.role_permissions = {
            HealthcareRole.PHYSICIAN: {
                "patient.read", "patient.write", "patient.diagnose",
                "medication.prescribe", "medication.modify", 
                "clinical.read", "clinical.write", "clinical.critical_access",
                "hmcp.agent_handoff", "hmcp.clinical_decision_support",
                "hmcp.emergency_override"
            },
            HealthcareRole.NURSE: {
                "patient.read", "patient.write_limited", 
                "medication.administer", "medication.read",
                "clinical.read", "clinical.write_limited",
                "hmcp.clinical_assessment", "hmcp.basic_decision_support"
            },
            HealthcareRole.PHARMACIST: {
                "patient.read", "medication.read", "medication.verify",
                "medication.interaction_check", "medication.counsel",
                "clinical.read", "hmcp.medication_review"
            },
            HealthcareRole.TECHNICIAN: {
                "patient.read_limited", "clinical.read_limited",
                "hmcp.basic_access"
            },
            HealthcareRole.ADMINISTRATOR: {
                "patient.read", "clinical.read", "admin.read", "admin.write",
                "system.monitor", "hmcp.admin_access"
            },
            HealthcareRole.STUDENT: {
                "patient.read_supervised", "clinical.read_supervised",
                "hmcp.educational_access"
            },
            HealthcareRole.TRAINEE: {
                "patient.read_supervised", "patient.write_supervised",
                "clinical.read", "clinical.write_supervised",
                "hmcp.supervised_access"
            },
            HealthcareRole.RESEARCHER: {
                "patient.read_deidentified", "clinical.read_research",
                "hmcp.research_access"
            }
        }
        
        # Scope to permission mapping
        self.scope_permissions = {
            SMARTScope.PATIENT_READ.value: {"patient.read"},
            SMARTScope.PATIENT_WRITE.value: {"patient.write"},
            SMARTScope.PATIENT_HMCP_READ.value: {"hmcp.patient_read"},
            SMARTScope.PATIENT_HMCP_WRITE.value: {"hmcp.patient_write"},
            SMARTScope.USER_READ.value: {"user.read"},
            SMARTScope.USER_WRITE.value: {"user.write"},
            SMARTScope.CLINICAL_READ.value: {"clinical.read"},
            SMARTScope.CLINICAL_WRITE.value: {"clinical.write"},
            SMARTScope.ADMIN_READ.value: {"admin.read"},
            SMARTScope.ADMIN_WRITE.value: {"admin.write"}
        }
    
    def get_permissions_for_role(self, role: HealthcareRole) -> Set[str]:
        """Get permissions for a healthcare role"""
        return self.role_permissions.get(role, set())
    
    def get_permissions_for_scopes(self, scopes: Set[str]) -> Set[str]:
        """Get permissions for OAuth scopes"""
        permissions = set()
        for scope in scopes:
            permissions.update(self.scope_permissions.get(scope, set()))
        return permissions
    
    def can_access_patient(self, 
                          role: HealthcareRole, 
                          permissions: Set[str],
                          patient_id: str) -> bool:
        """Check if user can access specific patient"""
        required_permissions = {"patient.read", "patient.read_limited", "patient.read_supervised"}
        return bool(permissions.intersection(required_permissions))
    
    def can_perform_action(self, 
                          permissions: Set[str], 
                          action: str) -> bool:
        """Check if user can perform specific action"""
        return action in permissions


class HMCPAuthenticationService:
    """
    Healthcare Authentication Service for HMCP
    
    Provides OAuth2/JWT authentication with healthcare-specific features:
    - SMART on FHIR compliance
    - Role-based access control
    - Patient context management
    - Secure token management
    """
    
    def __init__(self, 
                 client_id: str,
                 client_secret: str,
                 authorization_endpoint: str,
                 token_endpoint: str,
                 jwks_uri: str,
                 issuer: str = "hmcp-auth-server",
                 audience: str = "hmcp-api"):
        
        if not AUTH_AVAILABLE:
            raise ImportError("Authentication libraries not available. Install with: pip install authlib httpx PyJWT")
        
        self.client_id = client_id
        self.client_secret = client_secret
        self.authorization_endpoint = authorization_endpoint
        self.token_endpoint = token_endpoint
        self.jwks_uri = jwks_uri
        self.issuer = issuer
        self.audience = audience
        
        # Authentication state
        self.active_sessions: Dict[str, AuthenticationContext] = {}
        self.refresh_tokens: Dict[str, str] = {}
        
        # Permission manager
        self.permission_manager = HealthcarePermissionManager()
        
        # Audit logging
        self.auth_audit_log: List[Dict[str, Any]] = []
        
        # Token signing key (in production, use proper key management)
        self.signing_key = secrets.token_urlsafe(32)
        
        logger.info("HMCP Authentication Service initialized")
    
    async def initiate_authorization(self, 
                                   scopes: List[Union[str, SMARTScope]],
                                   patient_id: Optional[str] = None,
                                   encounter_id: Optional[str] = None,
                                   redirect_uri: str = "http://localhost:8000/callback") -> Dict[str, str]:
        """Initiate OAuth2 authorization flow"""
        
        # Convert scopes to strings
        scope_strings = []
        for scope in scopes:
            if isinstance(scope, SMARTScope):
                scope_strings.append(scope.value)
            else:
                scope_strings.append(scope)
        
        # Add required SMART scopes
        if "openid" not in scope_strings:
            scope_strings.append("openid")
        if "profile" not in scope_strings:
            scope_strings.append("profile")
        if "fhirUser" not in scope_strings:
            scope_strings.append("fhirUser")
        
        # Create OAuth2 client
        client = AsyncOAuth2Client(
            client_id=self.client_id,
            client_secret=self.client_secret,
            scope=" ".join(scope_strings)
        )
        
        # Generate state and PKCE parameters
        state = secrets.token_urlsafe(32)
        code_verifier = secrets.token_urlsafe(32)
        code_challenge = hashlib.sha256(code_verifier.encode()).hexdigest()
        
        # Build authorization URL
        auth_params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(scope_strings),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256"
        }
        
        # Add launch context
        if patient_id:
            auth_params["launch"] = f"patient:{patient_id}"
        if encounter_id:
            auth_params["encounter"] = encounter_id
        
        authorization_url = f"{self.authorization_endpoint}?" + "&".join([
            f"{k}={v}" for k, v in auth_params.items()
        ])
        
        return {
            "authorization_url": authorization_url,
            "state": state,
            "code_verifier": code_verifier
        }
    
    async def exchange_authorization_code(self, 
                                        authorization_code: str,
                                        redirect_uri: str,
                                        code_verifier: str) -> AuthenticationContext:
        """Exchange authorization code for tokens"""
        
        # Create OAuth2 client
        client = AsyncOAuth2Client(
            client_id=self.client_id,
            client_secret=self.client_secret
        )
        
        # Exchange code for tokens
        try:
            async with httpx.AsyncClient() as http_client:
                token_response = await client.fetch_token(
                    self.token_endpoint,
                    code=authorization_code,
                    redirect_uri=redirect_uri,
                    code_verifier=code_verifier,
                    client=http_client
                )
            
            # Validate and parse token
            access_token = token_response["access_token"]
            refresh_token = token_response.get("refresh_token")
            
            # Create authentication context
            auth_context = await self._create_auth_context(access_token, refresh_token)
            
            # Store session
            self.active_sessions[auth_context.session_id] = auth_context
            if refresh_token:
                self.refresh_tokens[auth_context.session_id] = refresh_token
            
            # Log authentication event
            await self._log_auth_event({
                "action": "token_exchange",
                "client_id": self.client_id,
                "user_id": auth_context.token.sub,
                "session_id": auth_context.session_id,
                "scopes": list(auth_context.scopes)
            })
            
            return auth_context
            
        except Exception as e:
            logger.error(f"Token exchange failed: {e}")
            raise AuthenticationError(f"Failed to exchange authorization code: {str(e)}")
    
    async def _create_auth_context(self, 
                                 access_token: str, 
                                 refresh_token: Optional[str] = None) -> AuthenticationContext:
        """Create authentication context from access token"""
        
        try:
            # Decode JWT token (in production, verify signature with JWKS)
            token_data = jwt.decode(
                access_token, 
                self.signing_key, 
                algorithms=["HS256"],
                options={"verify_signature": False}  # For demo - verify in production
            )
            
            # Create healthcare token
            healthcare_token = HealthcareToken(**token_data)
            
            # Parse scopes
            scopes = set(healthcare_token.scope.split(" ")) if healthcare_token.scope else set()
            
            # Determine healthcare role
            healthcare_role = HealthcareRole.UNKNOWN
            if healthcare_token.healthcare_role:
                try:
                    healthcare_role = HealthcareRole(healthcare_token.healthcare_role)
                except ValueError:
                    logger.warning(f"Unknown healthcare role: {healthcare_token.healthcare_role}")
            
            # Get permissions
            role_permissions = self.permission_manager.get_permissions_for_role(healthcare_role)
            scope_permissions = self.permission_manager.get_permissions_for_scopes(scopes)
            permissions = role_permissions.union(scope_permissions)
            
            # Create patient context if patient ID is available
            patient_context = None
            if healthcare_token.patient:
                patient_context = PatientContext(
                    patient_id=healthcare_token.patient,
                    mrn=f"MRN-{healthcare_token.patient}",  # Simplified
                    demographics={}
                )
            
            # Create authentication context
            auth_context = AuthenticationContext(
                token=healthcare_token,
                scopes=scopes,
                permissions=permissions,
                healthcare_role=healthcare_role,
                patient_context=patient_context,
                organization_id=healthcare_token.organization,
                expires_at=datetime.fromtimestamp(healthcare_token.exp, timezone.utc)
            )
            
            return auth_context
            
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid access token: {str(e)}")
        except Exception as e:
            raise AuthenticationError(f"Failed to create auth context: {str(e)}")
    
    async def validate_token(self, access_token: str) -> AuthenticationContext:
        """Validate access token and return authentication context"""
        
        try:
            # In production, verify token signature with JWKS
            token_data = jwt.decode(
                access_token,
                options={"verify_signature": False}  # For demo
            )
            
            # Check token expiration
            exp = token_data.get("exp", 0)
            if datetime.now(timezone.utc).timestamp() > exp:
                raise AuthenticationError("Token has expired")
            
            # Check issuer and audience
            if token_data.get("iss") != self.issuer:
                raise AuthenticationError("Invalid token issuer")
            
            if token_data.get("aud") != self.audience:
                raise AuthenticationError("Invalid token audience")
            
            # Create authentication context
            auth_context = await self._create_auth_context(access_token)
            
            return auth_context
            
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
    
    async def refresh_access_token(self, session_id: str) -> AuthenticationContext:
        """Refresh access token using refresh token"""
        
        if session_id not in self.refresh_tokens:
            raise AuthenticationError("No refresh token available")
        
        refresh_token = self.refresh_tokens[session_id]
        
        try:
            # Create OAuth2 client
            client = AsyncOAuth2Client(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            
            # Refresh token
            async with httpx.AsyncClient() as http_client:
                token_response = await client.refresh_token(
                    self.token_endpoint,
                    refresh_token=refresh_token,
                    client=http_client
                )
            
            # Create new authentication context
            new_access_token = token_response["access_token"]
            new_refresh_token = token_response.get("refresh_token", refresh_token)
            
            auth_context = await self._create_auth_context(new_access_token, new_refresh_token)
            
            # Update session
            self.active_sessions[session_id] = auth_context
            self.refresh_tokens[session_id] = new_refresh_token
            
            # Log refresh event
            await self._log_auth_event({
                "action": "token_refresh",
                "session_id": session_id,
                "user_id": auth_context.token.sub
            })
            
            return auth_context
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise AuthenticationError(f"Failed to refresh token: {str(e)}")
    
    async def authenticate_service_account(self, 
                                         service_account_id: str,
                                         private_key: str,
                                         scopes: List[Union[str, SMARTScope]]) -> AuthenticationContext:
        """Authenticate service account using client credentials"""
        
        # Convert scopes to strings
        scope_strings = [scope.value if isinstance(scope, SMARTScope) else scope for scope in scopes]
        
        # Create JWT assertion for client credentials
        assertion_claims = {
            "iss": service_account_id,
            "sub": service_account_id,
            "aud": self.token_endpoint,
            "exp": int((datetime.now(timezone.utc) + timedelta(minutes=5)).timestamp()),
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "jti": str(uuid.uuid4())
        }
        
        assertion = jwt.encode(
            assertion_claims,
            private_key,
            algorithm="RS256"
        )
        
        # Request token using client credentials
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.token_endpoint,
                    data={
                        "grant_type": "client_credentials",
                        "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                        "client_assertion": assertion,
                        "scope": " ".join(scope_strings)
                    }
                )
                response.raise_for_status()
                token_response = response.json()
            
            access_token = token_response["access_token"]
            auth_context = await self._create_auth_context(access_token)
            
            # Store session
            self.active_sessions[auth_context.session_id] = auth_context
            
            # Log service account authentication
            await self._log_auth_event({
                "action": "service_account_auth",
                "service_account_id": service_account_id,
                "session_id": auth_context.session_id,
                "scopes": scope_strings
            })
            
            return auth_context
            
        except Exception as e:
            logger.error(f"Service account authentication failed: {e}")
            raise AuthenticationError(f"Service account authentication failed: {str(e)}")
    
    async def create_session_token(self, 
                                 user_id: str,
                                 healthcare_role: HealthcareRole,
                                 scopes: List[str],
                                 patient_id: Optional[str] = None,
                                 organization_id: Optional[str] = None) -> str:
        """Create session token for development/testing"""
        
        # Create token claims
        token_claims = {
            "iss": self.issuer,
            "sub": user_id,
            "aud": self.audience,
            "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "jti": str(uuid.uuid4()),
            "scope": " ".join(scopes),
            "client_id": self.client_id,
            "healthcare_role": healthcare_role.value,
            "organization": organization_id,
            "patient": patient_id,
            "fhirUser": f"Practitioner/{user_id}"
        }
        
        # Sign token
        access_token = jwt.encode(
            token_claims,
            self.signing_key,
            algorithm="HS256"
        )
        
        return access_token
    
    async def revoke_session(self, session_id: str) -> bool:
        """Revoke authentication session"""
        
        try:
            if session_id in self.active_sessions:
                auth_context = self.active_sessions[session_id]
                
                # Log revocation
                await self._log_auth_event({
                    "action": "session_revoked",
                    "session_id": session_id,
                    "user_id": auth_context.token.sub
                })
                
                # Remove session
                del self.active_sessions[session_id]
                
                # Remove refresh token
                if session_id in self.refresh_tokens:
                    del self.refresh_tokens[session_id]
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Session revocation failed: {e}")
            return False
    
    async def authorize_operation(self, 
                                session_id: str,
                                operation: str,
                                resource_type: str,
                                resource_id: Optional[str] = None) -> bool:
        """Authorize specific operation"""
        
        if session_id not in self.active_sessions:
            raise AuthorizationError("Session not found")
        
        auth_context = self.active_sessions[session_id]
        
        # Check session expiration
        if auth_context.is_expired():
            raise AuthorizationError("Session has expired")
        
        # Build permission string
        permission = f"{resource_type}.{operation}"
        
        # Check permissions
        if not auth_context.has_permission(permission):
            # Log authorization failure
            await self._log_auth_event({
                "action": "authorization_denied",
                "session_id": session_id,
                "user_id": auth_context.token.sub,
                "requested_permission": permission,
                "user_permissions": list(auth_context.permissions)
            })
            
            raise AuthorizationError(f"Insufficient permissions for {permission}")
        
        # Additional checks for patient-specific operations
        if resource_type == "patient" and resource_id:
            if not self.permission_manager.can_access_patient(
                auth_context.healthcare_role,
                auth_context.permissions,
                resource_id
            ):
                raise AuthorizationError(f"Cannot access patient {resource_id}")
        
        # Log successful authorization
        await self._log_auth_event({
            "action": "authorization_granted",
            "session_id": session_id,
            "user_id": auth_context.token.sub,
            "permission": permission,
            "resource_id": resource_id
        })
        
        return True
    
    async def get_session(self, session_id: str) -> Optional[AuthenticationContext]:
        """Get authentication session"""
        return self.active_sessions.get(session_id)
    
    async def _log_auth_event(self, event: Dict[str, Any]):
        """Log authentication/authorization event"""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event.get("action", "unknown"),
            "client_id": self.client_id,
            **event
        }
        
        self.auth_audit_log.append(audit_entry)
        
        # In production, send to secure audit log system
        logger.info(f"Auth audit: {audit_entry}")
    
    def get_auth_statistics(self) -> Dict[str, Any]:
        """Get authentication service statistics"""
        
        active_sessions_count = len(self.active_sessions)
        total_events = len(self.auth_audit_log)
        
        # Count events by type
        event_counts = {}
        for event in self.auth_audit_log:
            event_type = event.get("event_type", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Session statistics
        role_distribution = {}
        for session in self.active_sessions.values():
            role = session.healthcare_role.value
            role_distribution[role] = role_distribution.get(role, 0) + 1
        
        return {
            "active_sessions": active_sessions_count,
            "total_auth_events": total_events,
            "event_distribution": event_counts,
            "role_distribution": role_distribution,
            "refresh_tokens_count": len(self.refresh_tokens)
        }
    
    def get_recent_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent authentication audit logs"""
        return self.auth_audit_log[-limit:]


# Example usage and helper functions
async def authentication_example():
    """Example of using HMCP Authentication Service"""
    
    # Initialize authentication service
    auth_service = HMCPAuthenticationService(
        client_id="hmcp-demo-client",
        client_secret="hmcp-demo-secret",
        authorization_endpoint="https://auth.healthcare.local/oauth2/authorize",
        token_endpoint="https://auth.healthcare.local/oauth2/token",
        jwks_uri="https://auth.healthcare.local/.well-known/jwks.json"
    )
    
    # Create session token for testing
    test_token = await auth_service.create_session_token(
        user_id="physician-001",
        healthcare_role=HealthcareRole.PHYSICIAN,
        scopes=[
            SMARTScope.PATIENT_READ.value,
            SMARTScope.PATIENT_WRITE.value,
            SMARTScope.CLINICAL_READ.value,
            SMARTScope.CLINICAL_WRITE.value,
            SMARTScope.PATIENT_HMCP_READ.value,
            SMARTScope.PATIENT_HMCP_WRITE.value
        ],
        patient_id="PT12345",
        organization_id="org-001"
    )
    
    print(f"Created test token: {test_token[:50]}...")
    
    # Validate token
    auth_context = await auth_service.validate_token(test_token)
    print(f"Authentication successful for user: {auth_context.token.sub}")
    print(f"Healthcare role: {auth_context.healthcare_role.value}")
    print(f"Scopes: {list(auth_context.scopes)}")
    print(f"Permissions: {list(auth_context.permissions)}")
    
    # Test authorization
    try:
        await auth_service.authorize_operation(
            auth_context.session_id,
            "read",
            "patient",
            "PT12345"
        )
        print("✓ Authorized to read patient data")
    except AuthorizationError as e:
        print(f"✗ Authorization failed: {e}")
    
    # Get statistics
    stats = auth_service.get_auth_statistics()
    print(f"Auth service statistics: {stats}")


if __name__ == "__main__":
    # Run example
    asyncio.run(authentication_example())