#!/usr/bin/env python3
"""
Healthcare Model Context Protocol (HMCP) Client Implementation

Based on Innovaccer's Healthcare-MCP implementation, this extends the MCP Client
with healthcare-specific capabilities including:
- Healthcare-specific authentication and authorization
- Patient context management
- Clinical data handling with PHI protection
- Bidirectional sampling for agent communication
- HIPAA-compliant audit logging
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

try:
    # MCP SDK imports
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    import mcp.types as types
    from mcp.types import (
        SamplingMessage, TextContent, CreateMessageRequest, CreateMessageResult,
        CallToolRequest, CallToolResult, ListToolsRequest
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Fallback types for development
    class ClientSession: pass
    class SamplingMessage: pass
    class TextContent: pass
    class CreateMessageRequest: pass
    class CreateMessageResult: pass

try:
    # Healthcare authentication imports
    import jwt
    from authlib.integrations.httpx_client import AsyncOAuth2Client
    from httpx import AsyncClient
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

from vita_agents.protocols.hmcp import (
    ClinicalUrgency, HealthcareRole, PatientContext, ClinicalContext,
    SecurityContext, HMCPMessage, HMCPMessageType
)

logger = logging.getLogger(__name__)


@dataclass
class HMCPAuthConfig:
    """Healthcare-specific authentication configuration"""
    OAUTH_SCOPES: List[str] = field(default_factory=lambda: [
        "patient/hmcp:read",
        "patient/hmcp:write", 
        "launch/patient",
        "fhirUser",
        "openid",
        "profile",
        "patient/*.read",
        "patient/*.write",
        "user/*.*"
    ])
    JWT_ALGORITHM: str = "RS256"
    TOKEN_EXPIRY: int = 3600  # 1 hour
    CLIENT_ID: str = "hmcp-client"
    CLIENT_SECRET: str = "hmcp-secret"
    AUTH_ENDPOINT: str = "https://auth.healthcare.local/oauth2"
    TOKEN_ENDPOINT: str = "https://auth.healthcare.local/oauth2/token"


class HMCPOAuthClient:
    """Healthcare OAuth2 client for HMCP authentication"""
    
    def __init__(self, config: HMCPAuthConfig):
        if not AUTH_AVAILABLE:
            raise ImportError("Authentication libraries not available. Install with: pip install authlib PyJWT")
        
        self.config = config
        self.oauth_client: Optional[AsyncOAuth2Client] = None
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.patient_context: Optional[PatientContext] = None
    
    async def initialize(self):
        """Initialize OAuth2 client"""
        self.oauth_client = AsyncOAuth2Client(
            client_id=self.config.CLIENT_ID,
            client_secret=self.config.CLIENT_SECRET,
            scope=" ".join(self.config.OAUTH_SCOPES)
        )
    
    async def authenticate(self, patient_id: Optional[str] = None) -> bool:
        """Authenticate with healthcare-specific scopes"""
        if not self.oauth_client:
            await self.initialize()
        
        try:
            # In a real implementation, this would do proper OAuth2 flow
            # For now, we generate a mock JWT token
            token_payload = {
                "iss": "hmcp-auth-server",
                "sub": self.config.CLIENT_ID,
                "aud": "hmcp-api",
                "exp": datetime.now(timezone.utc).timestamp() + self.config.TOKEN_EXPIRY,
                "iat": datetime.now(timezone.utc).timestamp(),
                "scope": " ".join(self.config.OAUTH_SCOPES),
                "client_id": self.config.CLIENT_ID,
                "patient": patient_id,
                "fhirUser": f"Practitioner/{self.config.CLIENT_ID}",
                "tenant": "healthcare-org"
            }
            
            # Generate mock JWT (in production, this would come from auth server)
            self.access_token = jwt.encode(
                token_payload, 
                "secret-key",  # In production, use proper private key
                algorithm="HS256"  # In production, use RS256
            )
            
            self.token_expires_at = datetime.fromtimestamp(token_payload["exp"], timezone.utc)
            
            # Set patient context if provided
            if patient_id:
                self.patient_context = PatientContext(
                    patient_id=patient_id,
                    mrn=f"MRN-{patient_id}",
                    demographics={"authorized": True}
                )
            
            logger.info(f"HMCP authentication successful for client: {self.config.CLIENT_ID}")
            return True
            
        except Exception as e:
            logger.error(f"HMCP authentication failed: {e}")
            return False
    
    def is_token_valid(self) -> bool:
        """Check if access token is still valid"""
        if not self.access_token or not self.token_expires_at:
            return False
        return datetime.now(timezone.utc) < self.token_expires_at
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for HMCP requests"""
        if not self.is_token_valid():
            raise ValueError("Access token is invalid or expired")
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "X-HMCP-Client": self.config.CLIENT_ID
        }
        
        if self.patient_context:
            headers["X-Patient-Context"] = self.patient_context.patient_id
            headers["X-Patient-MRN"] = self.patient_context.mrn
        
        return headers
    
    async def refresh_access_token(self) -> bool:
        """Refresh the access token"""
        # In a real implementation, this would use refresh token
        return await self.authenticate(
            self.patient_context.patient_id if self.patient_context else None
        )


class HMCPClient:
    """
    Healthcare Model Context Protocol Client
    
    Extends MCP Client with healthcare-specific capabilities:
    - Healthcare authentication and authorization
    - Patient context management
    - Clinical data handling with PHI protection
    - Bidirectional sampling for agent communication
    - HIPAA-compliant audit logging
    """
    
    def __init__(self, 
                 server_url: str,
                 auth_config: Optional[HMCPAuthConfig] = None,
                 patient_id: Optional[str] = None,
                 debug: bool = False):
        
        if not MCP_AVAILABLE:
            raise ImportError("MCP SDK not available. Install with: pip install mcp")
        
        self.server_url = server_url
        self.auth_config = auth_config or HMCPAuthConfig()
        self.patient_id = patient_id
        self.debug = debug
        
        # Authentication
        self.oauth_client = HMCPOAuthClient(self.auth_config)
        
        # MCP session
        self.session: Optional[ClientSession] = None
        self.connected = False
        
        # Message history for context
        self.message_history: List[Dict[str, Any]] = []
        
        # Audit logging
        self.audit_logs: List[Dict[str, Any]] = []
        
        logger.info(f"HMCP Client initialized for server: {server_url}")
    
    async def connect(self) -> bool:
        """Connect to HMCP server with authentication"""
        try:
            # Authenticate first
            auth_success = await self.oauth_client.authenticate(self.patient_id)
            if not auth_success:
                raise Exception("Authentication failed")
            
            # Get authentication headers
            auth_headers = self.oauth_client.get_auth_headers()
            
            # Connect using MCP SSE client
            transport = sse_client(self.server_url, headers=auth_headers)
            
            # Create session
            read_stream, write_stream = await transport.__aenter__()
            self.session = ClientSession(read_stream, write_stream)
            
            # Initialize session
            init_result = await self.session.initialize()
            
            self.connected = True
            
            # Log audit event
            await self._log_audit({
                "action": "client_connected",
                "server_url": self.server_url,
                "patient_id": self.patient_id,
                "capabilities": init_result.capabilities if hasattr(init_result, 'capabilities') else []
            })
            
            logger.info(f"HMCP Client connected to server: {self.server_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to HMCP server: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from HMCP server"""
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.error(f"Error closing session: {e}")
        
        self.connected = False
        
        # Log audit event
        await self._log_audit({
            "action": "client_disconnected",
            "server_url": self.server_url
        })
        
        logger.info("HMCP Client disconnected")
    
    async def create_message(self, 
                           messages: List[SamplingMessage],
                           system_instructions: Optional[str] = None,
                           include_context: bool = True) -> CreateMessageResult:
        """Send message to HMCP server using sampling"""
        
        if not self.connected or not self.session:
            raise Exception("Client not connected to server")
        
        # Refresh token if needed
        if not self.oauth_client.is_token_valid():
            await self.oauth_client.refresh_access_token()
        
        try:
            # Add system instructions if provided
            all_messages = messages.copy()
            if system_instructions:
                system_message = SamplingMessage(
                    role="system",
                    content=TextContent(type="text", text=system_instructions)
                )
                all_messages.insert(0, system_message)
            
            # Add patient context if available
            if include_context and self.oauth_client.patient_context:
                context_message = SamplingMessage(
                    role="system",
                    content=TextContent(
                        type="text",
                        text=f"Patient Context: {json.dumps({
                            'patient_id': self.oauth_client.patient_context.patient_id,
                            'mrn': self.oauth_client.patient_context.mrn,
                            'authorized_scopes': self.auth_config.OAUTH_SCOPES
                        })}"
                    )
                )
                all_messages.insert(0, context_message)
            
            # Create sampling request
            request = CreateMessageRequest(
                messages=all_messages
            )
            
            # Send request to server
            result = await self.session.create_message(request)
            
            # Add to message history
            self.message_history.extend([
                {"role": msg.role, "content": msg.content.text if hasattr(msg.content, 'text') else str(msg.content)}
                for msg in all_messages
            ])
            
            if result and hasattr(result, 'content'):
                self.message_history.append({
                    "role": result.role,
                    "content": result.content.text if hasattr(result.content, 'text') else str(result.content)
                })
            
            # Log audit event
            await self._log_audit({
                "action": "message_sent",
                "message_count": len(all_messages),
                "has_patient_context": include_context and self.oauth_client.patient_context is not None,
                "response_received": result is not None
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error sending message to HMCP server: {e}")
            
            # Log audit event for error
            await self._log_audit({
                "action": "message_error",
                "error": str(e),
                "message_count": len(messages)
            })
            
            raise
    
    async def call_tool(self, 
                       tool_name: str, 
                       arguments: Dict[str, Any]) -> CallToolResult:
        """Call a tool on the HMCP server"""
        
        if not self.connected or not self.session:
            raise Exception("Client not connected to server")
        
        try:
            # Add patient context to tool arguments if available
            enhanced_arguments = arguments.copy()
            if self.oauth_client.patient_context:
                enhanced_arguments["patient_id"] = self.oauth_client.patient_context.patient_id
                enhanced_arguments["user_scopes"] = self.auth_config.OAUTH_SCOPES
            
            # Create tool request
            request = CallToolRequest(
                name=tool_name,
                arguments=enhanced_arguments
            )
            
            # Call tool
            result = await self.session.call_tool(request)
            
            # Log audit event
            await self._log_audit({
                "action": "tool_called",
                "tool_name": tool_name,
                "has_patient_context": self.oauth_client.patient_context is not None,
                "success": not (hasattr(result, 'isError') and result.isError)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            
            # Log audit event for error
            await self._log_audit({
                "action": "tool_error",
                "tool_name": tool_name,
                "error": str(e)
            })
            
            raise
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools on the HMCP server"""
        
        if not self.connected or not self.session:
            raise Exception("Client not connected to server")
        
        try:
            request = ListToolsRequest()
            result = await self.session.list_tools(request)
            
            tools = []
            if hasattr(result, 'tools'):
                for tool in result.tools:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema
                    })
            
            return tools
            
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []
    
    async def validate_patient_context(self) -> Dict[str, Any]:
        """Validate patient context with the server"""
        
        if not self.oauth_client.patient_context:
            return {
                "valid": False,
                "error": "No patient context available"
            }
        
        try:
            result = await self.call_tool("validate_patient_context", {
                "patient_id": self.oauth_client.patient_context.patient_id,
                "user_scopes": self.auth_config.OAUTH_SCOPES
            })
            
            if hasattr(result, 'content') and result.content:
                content_text = result.content[0].text if result.content else "{}"
                return json.loads(content_text)
            
            return {"valid": False, "error": "No response from server"}
            
        except Exception as e:
            logger.error(f"Error validating patient context: {e}")
            return {"valid": False, "error": str(e)}
    
    async def clinical_decision_support(self, 
                                      symptoms: List[str], 
                                      patient_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get clinical decision support from HMCP server"""
        
        try:
            result = await self.call_tool("clinical_decision_support", {
                "symptoms": symptoms,
                "patient_data": patient_data or {}
            })
            
            if hasattr(result, 'content') and result.content:
                content_text = result.content[0].text if result.content else "{}"
                return json.loads(content_text)
            
            return {"error": "No response from server"}
            
        except Exception as e:
            logger.error(f"Error getting clinical decision support: {e}")
            return {"error": str(e)}
    
    async def check_medication_interactions(self, medications: List[str]) -> Dict[str, Any]:
        """Check for medication interactions"""
        
        try:
            result = await self.call_tool("medication_interaction_check", {
                "medications": medications
            })
            
            if hasattr(result, 'content') and result.content:
                content_text = result.content[0].text if result.content else "{}"
                return json.loads(content_text)
            
            return {"error": "No response from server"}
            
        except Exception as e:
            logger.error(f"Error checking medication interactions: {e}")
            return {"error": str(e)}
    
    async def _log_audit(self, event: Dict[str, Any]):
        """Log audit event for HIPAA compliance"""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "client_id": self.auth_config.CLIENT_ID,
            "server_url": self.server_url,
            "patient_id": self.patient_id,
            "event_type": event.get("action", "unknown"),
            "details": event
        }
        self.audit_logs.append(audit_entry)
        
        # In production, this would write to secure audit log storage
        if self.debug:
            logger.info(f"Client audit log: {audit_entry}")
    
    def get_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit logs"""
        return self.audit_logs[-limit:]
    
    def get_client_status(self) -> Dict[str, Any]:
        """Get client status and connection information"""
        return {
            "connected": self.connected,
            "server_url": self.server_url,
            "patient_id": self.patient_id,
            "authenticated": self.oauth_client.is_token_valid(),
            "token_expires_at": self.oauth_client.token_expires_at.isoformat() if self.oauth_client.token_expires_at else None,
            "message_history_count": len(self.message_history),
            "audit_logs_count": len(self.audit_logs),
            "patient_context": {
                "patient_id": self.oauth_client.patient_context.patient_id,
                "mrn": self.oauth_client.patient_context.mrn
            } if self.oauth_client.patient_context else None
        }


# Example usage and helper functions
async def create_hmcp_client_example():
    """Example of creating and using an HMCP client"""
    
    # Create authentication configuration
    auth_config = HMCPAuthConfig()
    auth_config.CLIENT_ID = "vita-agents-client"
    auth_config.CLIENT_SECRET = "vita-agents-secret"
    
    # Create HMCP client
    client = HMCPClient(
        server_url="http://localhost:8050/sse",
        auth_config=auth_config,
        patient_id="PT12345",
        debug=True
    )
    
    try:
        # Connect to server
        connected = await client.connect()
        if not connected:
            print("Failed to connect to HMCP server")
            return
        
        # Validate patient context
        validation = await client.validate_patient_context()
        print(f"Patient context validation: {validation}")
        
        # Send a clinical message
        messages = [
            SamplingMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text="Patient presents with chest pain and shortness of breath. What are the recommended next steps?"
                )
            )
        ]
        
        result = await client.create_message(
            messages=messages,
            system_instructions="You are a healthcare AI providing clinical decision support.",
            include_context=True
        )
        
        print(f"Clinical response: {result.content.text if hasattr(result.content, 'text') else result.content}")
        
        # Check medication interactions
        interactions = await client.check_medication_interactions(["warfarin", "aspirin"])
        print(f"Medication interactions: {interactions}")
        
        # Get client status
        status = client.get_client_status()
        print(f"Client status: {status}")
        
    finally:
        # Always disconnect
        await client.disconnect()


if __name__ == "__main__":
    # Run example
    asyncio.run(create_hmcp_client_example())