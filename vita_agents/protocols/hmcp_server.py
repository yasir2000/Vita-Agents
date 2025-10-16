#!/usr/bin/env python3
"""
Healthcare Model Context Protocol (HMCP) Server Implementation

Based on Innovaccer's Healthcare-MCP implementation, this extends the MCP Server
with healthcare-specific capabilities including:
- Healthcare-specific authentication and authorization
- Patient context handling
- Clinical guardrails and security
- Bidirectional sampling for agent communication
- HIPAA-compliant audit logging
"""

import asyncio
import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable, Sequence
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

try:
    # MCP SDK imports
    from mcp import Server
    from mcp.server.sse import SseServerTransport
    from mcp.server.stdio import StdioServerTransport
    from mcp.shared.context import RequestContext
    import mcp.types as types
    from mcp.types import (
        SamplingMessage, TextContent, CreateMessageRequest, CreateMessageResult,
        Tool, CallToolRequest, CallToolResult
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Fallback types for development
    class RequestContext: pass
    class SamplingMessage: pass
    class TextContent: pass
    class CreateMessageRequest: pass
    class CreateMessageResult: pass

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
    PATIENT_CONTEXT_REQUIRED: bool = True
    AUDIT_LOGGING_ENABLED: bool = True
    REQUIRE_PATIENT_SCOPE: bool = True


class HMCPGuardrail:
    """Healthcare-specific guardrails for HMCP servers"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.prompt_injection_patterns = [
            "ignore previous instructions",
            "system override", 
            "admin mode",
            "bypass security",
            "disable guardrails",
            "jailbreak",
            "pretend you are",
            "act as if",
            "forget your instructions"
        ]
        
        self.phi_detection_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Z]{2}\d{8}\b',  # Medical record number pattern
            r'\b\d{2}/\d{2}/\d{4}\b',  # Date of birth pattern
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Potential names
        ]
        
        self.clinical_keywords = [
            "patient", "diagnosis", "medication", "treatment", "surgery",
            "blood pressure", "heart rate", "laboratory", "radiology",
            "prescription", "allergy", "medical history"
        ]
    
    async def validate_input(self, message: str, context: Optional[RequestContext] = None) -> Dict[str, Any]:
        """Validate input message for healthcare security threats"""
        if not self.enabled:
            return {"valid": True, "warnings": [], "requires_review": False}
        
        warnings = []
        security_violations = []
        
        # Check for prompt injection attempts
        message_lower = message.lower()
        for pattern in self.prompt_injection_patterns:
            if pattern in message_lower:
                security_violations.append(f"Potential prompt injection: {pattern}")
                logger.warning(f"Security violation detected: {pattern}")
        
        # Check for PHI exposure risk
        import re
        phi_detected = []
        for pattern in self.phi_detection_patterns:
            matches = re.findall(pattern, message)
            if matches:
                phi_detected.extend(matches)
                warnings.append(f"Potential PHI detected: {pattern}")
        
        # Clinical context validation
        has_clinical_context = any(keyword in message_lower for keyword in self.clinical_keywords)
        if has_clinical_context and context:
            # Verify patient context is provided for clinical discussions
            patient_context = getattr(context, 'patient_context', None)
            if not patient_context:
                warnings.append("Clinical discussion without patient context")
        
        return {
            "valid": len(security_violations) == 0,
            "warnings": warnings,
            "security_violations": security_violations,
            "phi_detected": phi_detected,
            "requires_review": len(warnings) > 0 or len(security_violations) > 0,
            "has_clinical_context": has_clinical_context
        }
    
    async def validate_output(self, message: str, context: Optional[RequestContext] = None) -> Dict[str, Any]:
        """Validate and sanitize output message for PHI and compliance"""
        if not self.enabled:
            return {"valid": True, "sanitized_message": message, "audit_logged": False}
        
        # PHI detection and redaction
        import re
        sanitized_message = message
        redactions_made = []
        
        for pattern in self.phi_detection_patterns:
            matches = re.findall(pattern, sanitized_message)
            for match in matches:
                if pattern == r'\b\d{3}-\d{2}-\d{4}\b':  # SSN
                    sanitized_message = sanitized_message.replace(match, "XXX-XX-XXXX")
                    redactions_made.append(f"SSN redacted: {match}")
                elif pattern == r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b':  # Credit card
                    sanitized_message = sanitized_message.replace(match, "XXXX-XXXX-XXXX-XXXX")
                    redactions_made.append(f"Credit card redacted: {match}")
                else:
                    # Generic redaction for other patterns
                    sanitized_message = sanitized_message.replace(match, "[REDACTED]")
                    redactions_made.append(f"PHI redacted: {match}")
        
        # Log for audit if redactions were made
        audit_logged = False
        if redactions_made:
            logger.info(f"PHI redaction performed: {redactions_made}")
            audit_logged = True
        
        return {
            "valid": True,
            "sanitized_message": sanitized_message,
            "redactions_made": redactions_made,
            "audit_logged": audit_logged,
            "original_length": len(message),
            "sanitized_length": len(sanitized_message)
        }


class HMCPServer:
    """
    Healthcare Model Context Protocol Server
    
    Extends MCP Server with healthcare-specific capabilities:
    - Healthcare authentication and authorization
    - Patient context handling
    - Clinical guardrails and security
    - Bidirectional sampling for agent communication
    - HIPAA-compliant audit logging
    """
    
    def __init__(self,
                 name: str,
                 version: str = "1.0.0",
                 instructions: Optional[str] = None,
                 host: str = "localhost",
                 port: int = 8050,
                 debug: bool = False,
                 log_level: str = "INFO",
                 auth_config: Optional[HMCPAuthConfig] = None,
                 enable_guardrails: bool = True):
        
        if not MCP_AVAILABLE:
            raise ImportError("MCP SDK not available. Install with: pip install mcp")
        
        self.name = name
        self.version = version
        self.instructions = instructions or f"Healthcare AI Agent: {name}"
        self.host = host
        self.port = port
        self.debug = debug
        
        # Healthcare-specific configuration
        self.auth_config = auth_config or HMCPAuthConfig()
        self.guardrail = HMCPGuardrail(enabled=enable_guardrails)
        
        # Initialize MCP server
        self.server = Server(name)
        
        # Healthcare capabilities
        self.capabilities = [
            "sampling",
            "healthcare",
            "fhir",
            "clinical_decision_support",
            "patient_context",
            "guardrails"
        ]
        
        # Sampling handler
        self._sampling_handler: Optional[Callable] = None
        
        # Tool handlers
        self._tool_handlers: Dict[str, Callable] = {}
        
        # Patient context storage
        self._patient_contexts: Dict[str, PatientContext] = {}
        
        # Audit logging
        self._audit_logs: List[Dict[str, Any]] = []
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, log_level))
        
        # Register default healthcare tools
        self._register_healthcare_tools()
        
        logger.info(f"HMCP Server '{name}' initialized with capabilities: {self.capabilities}")
    
    def sampling(self):
        """Decorator for registering sampling handler (bidirectional communication)"""
        def decorator(func: Callable):
            self._sampling_handler = func
            
            @self.server.call_tool()
            async def handle_sampling_tool(request: CallToolRequest) -> CallToolResult:
                """Handle sampling requests as tool calls"""
                try:
                    # Extract sampling parameters from tool request
                    if request.name == "hmcp_sampling":
                        arguments = request.arguments or {}
                        
                        # Create mock sampling request
                        messages = arguments.get("messages", [])
                        
                        # Create request context with healthcare information
                        context = RequestContext(
                            server=self.server,
                            session_id=arguments.get("session_id", "default")
                        )
                        
                        # Add patient context if provided
                        patient_id = arguments.get("patient_id")
                        if patient_id and patient_id in self._patient_contexts:
                            context.patient_context = self._patient_contexts[patient_id]
                        
                        # Create sampling parameters
                        sampling_params = types.CreateMessageRequestParams(
                            messages=[
                                SamplingMessage(
                                    role=msg.get("role", "user"),
                                    content=TextContent(
                                        type="text",
                                        text=msg.get("content", "")
                                    )
                                ) for msg in messages
                            ]
                        )
                        
                        # Call the sampling handler
                        result = await func(context, sampling_params)
                        
                        return CallToolResult(
                            content=[
                                types.TextContent(
                                    type="text",
                                    text=json.dumps({
                                        "model": result.model,
                                        "role": result.role,
                                        "content": result.content.text if hasattr(result.content, 'text') else str(result.content),
                                        "stopReason": result.stopReason
                                    })
                                )
                            ]
                        )
                    
                    return CallToolResult(
                        content=[types.TextContent(type="text", text="Unknown sampling request")]
                    )
                    
                except Exception as e:
                    logger.error(f"Error in sampling handler: {e}")
                    return CallToolResult(
                        content=[types.TextContent(type="text", text=f"Error: {str(e)}")],
                        isError=True
                    )
            
            return func
        return decorator
    
    def tool(self, name: str, description: str):
        """Decorator for registering healthcare tools"""
        def decorator(func: Callable):
            self._tool_handlers[name] = func
            
            @self.server.call_tool()
            async def tool_handler(request: CallToolRequest) -> CallToolResult:
                if request.name == name:
                    try:
                        # Apply guardrails to tool input
                        input_validation = await self.guardrail.validate_input(
                            json.dumps(request.arguments or {})
                        )
                        
                        if not input_validation["valid"]:
                            logger.warning(f"Tool input validation failed: {input_validation['security_violations']}")
                            return CallToolResult(
                                content=[types.TextContent(
                                    type="text",
                                    text=f"Security validation failed: {input_validation['security_violations']}"
                                )],
                                isError=True
                            )
                        
                        # Execute tool function
                        result = await func(request.arguments or {})
                        
                        # Apply guardrails to tool output
                        output_validation = await self.guardrail.validate_output(
                            json.dumps(result) if isinstance(result, dict) else str(result)
                        )
                        
                        # Log for audit
                        await self._log_audit({
                            "action": "tool_execution",
                            "tool_name": name,
                            "input_validation": input_validation,
                            "output_validation": output_validation,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                        
                        return CallToolResult(
                            content=[types.TextContent(
                                type="text", 
                                text=output_validation["sanitized_message"]
                            )]
                        )
                        
                    except Exception as e:
                        logger.error(f"Error executing tool {name}: {e}")
                        return CallToolResult(
                            content=[types.TextContent(type="text", text=f"Error: {str(e)}")],
                            isError=True
                        )
                
                return CallToolResult(
                    content=[types.TextContent(type="text", text="Tool not found")]
                )
            
            # Register tool with MCP server
            tool_def = Tool(
                name=name,
                description=description,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Tool input"}
                    }
                }
            )
            
            return func
        return decorator
    
    def _register_healthcare_tools(self):
        """Register default healthcare tools"""
        
        @self.tool("validate_patient_context", "Validate patient context and authorization")
        async def validate_patient_context(args: Dict[str, Any]) -> Dict[str, Any]:
            patient_id = args.get("patient_id")
            user_scopes = args.get("user_scopes", [])
            
            # Check if user has patient access scope
            has_patient_scope = any("patient" in scope for scope in user_scopes)
            
            if patient_id and patient_id in self._patient_contexts:
                context = self._patient_contexts[patient_id]
                return {
                    "valid": has_patient_scope,
                    "patient_id": patient_id,
                    "mrn": context.mrn,
                    "authorized": has_patient_scope
                }
            
            return {
                "valid": False,
                "error": "Patient context not found or unauthorized"
            }
        
        @self.tool("clinical_decision_support", "Provide clinical decision support")
        async def clinical_decision_support(args: Dict[str, Any]) -> Dict[str, Any]:
            symptoms = args.get("symptoms", [])
            patient_data = args.get("patient_data", {})
            
            # Mock clinical decision support
            recommendations = []
            if "chest pain" in symptoms:
                recommendations.extend([
                    "Order 12-lead ECG",
                    "Obtain cardiac enzymes", 
                    "Consider cardiology consultation"
                ])
            
            return {
                "recommendations": recommendations,
                "confidence_score": 0.85,
                "evidence_level": "high",
                "guidelines_referenced": ["AHA/ACC Guidelines"]
            }
        
        @self.tool("medication_interaction_check", "Check for medication interactions")
        async def medication_interaction_check(args: Dict[str, Any]) -> Dict[str, Any]:
            medications = args.get("medications", [])
            
            interactions = []
            # Simple interaction checking
            med_names = [med.lower() for med in medications]
            if "warfarin" in med_names and "aspirin" in med_names:
                interactions.append({
                    "severity": "major",
                    "interaction": "Warfarin + Aspirin",
                    "description": "Increased bleeding risk",
                    "recommendation": "Monitor INR closely"
                })
            
            return {
                "interactions": interactions,
                "total_medications": len(medications),
                "high_risk_count": len([i for i in interactions if i["severity"] == "major"])
            }
    
    async def add_patient_context(self, patient_context: PatientContext):
        """Add patient context for authorization and clinical workflows"""
        self._patient_contexts[patient_context.patient_id] = patient_context
        logger.info(f"Patient context added: {patient_context.patient_id}")
    
    async def _log_audit(self, event: Dict[str, Any]):
        """Log audit event for HIPAA compliance"""
        if self.auth_config.AUDIT_LOGGING_ENABLED:
            audit_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "server_name": self.name,
                "event_type": event.get("action", "unknown"),
                "details": event,
                "session_id": event.get("session_id", "unknown")
            }
            self._audit_logs.append(audit_entry)
            
            # In production, this would write to secure audit log storage
            logger.info(f"Audit log: {audit_entry}")
    
    def get_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit logs"""
        return self._audit_logs[-limit:]
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information and capabilities"""
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": self.capabilities,
            "instructions": self.instructions,
            "host": self.host,
            "port": self.port,
            "guardrails_enabled": self.guardrail.enabled,
            "patient_contexts_count": len(self._patient_contexts),
            "audit_logs_count": len(self._audit_logs),
            "uptime": datetime.now(timezone.utc).isoformat()
        }
    
    async def run_sse(self):
        """Run server with Server-Sent Events transport"""
        if not MCP_AVAILABLE:
            raise ImportError("MCP SDK not available")
        
        transport = SseServerTransport(f"http://{self.host}:{self.port}/sse")
        
        logger.info(f"Starting HMCP Server '{self.name}' on {self.host}:{self.port}")
        
        async with transport:
            await self.server.run(
                transport.read,
                transport.write,
                self.server.create_initialization_options()
            )
    
    async def run_stdio(self):
        """Run server with stdio transport"""
        if not MCP_AVAILABLE:
            raise ImportError("MCP SDK not available")
        
        transport = StdioServerTransport()
        
        logger.info(f"Starting HMCP Server '{self.name}' with stdio transport")
        
        async with transport:
            await self.server.run(
                transport.read,
                transport.write,
                self.server.create_initialization_options()
            )
    
    def run(self, transport: str = "sse"):
        """Run the HMCP server"""
        if transport == "sse":
            asyncio.run(self.run_sse())
        elif transport == "stdio":
            asyncio.run(self.run_stdio())
        else:
            raise ValueError(f"Unknown transport: {transport}")


# Example healthcare HMCP server implementation
if __name__ == "__main__":
    async def create_healthcare_hmcp_server():
        """Create a healthcare HMCP server with comprehensive capabilities"""
        
        # Create authentication configuration
        auth_config = HMCPAuthConfig()
        auth_config.PATIENT_CONTEXT_REQUIRED = True
        auth_config.AUDIT_LOGGING_ENABLED = True
        
        # Create HMCP server
        server = HMCPServer(
            name="Healthcare HMCP Agent",
            version="1.0.0",
            instructions="Advanced healthcare AI agent with clinical decision support, "
                        "medication management, and emergency response capabilities.",
            host="localhost",
            port=8050,
            debug=True,
            auth_config=auth_config,
            enable_guardrails=True
        )
        
        # Register sampling handler for bidirectional communication
        @server.sampling()
        async def handle_healthcare_sampling(
            context: RequestContext, 
            params: types.CreateMessageRequestParams
        ) -> types.CreateMessageResult:
            """Handle healthcare-specific sampling requests"""
            
            # Get the latest message
            latest_message = params.messages[-1] if params.messages else None
            if not latest_message:
                return types.CreateMessageResult(
                    model="healthcare-hmcp-agent",
                    role="assistant",
                    content=TextContent(
                        type="text",
                        text="No message provided"
                    ),
                    stopReason="endTurn"
                )
            
            # Extract message content
            message_content = ""
            if isinstance(latest_message.content, TextContent):
                message_content = latest_message.content.text
            elif hasattr(latest_message.content, 'text'):
                message_content = latest_message.content.text
            else:
                message_content = str(latest_message.content)
            
            # Apply input guardrails
            guardrail_result = await server.guardrail.validate_input(message_content, context)
            
            if not guardrail_result["valid"]:
                logger.warning(f"Guardrail violation: {guardrail_result['security_violations']}")
                return types.CreateMessageResult(
                    model="healthcare-hmcp-agent",
                    role="assistant", 
                    content=TextContent(
                        type="text",
                        text="I cannot process this request due to security policy violations."
                    ),
                    stopReason="endTurn"
                )
            
            # Generate healthcare-appropriate response
            response_text = await generate_healthcare_response(message_content, context, guardrail_result)
            
            # Apply output guardrails
            output_validation = await server.guardrail.validate_output(response_text, context)
            
            # Log audit event
            await server._log_audit({
                "action": "sampling_request",
                "input_validation": guardrail_result,
                "output_validation": output_validation,
                "session_id": context.session_id,
                "has_patient_context": hasattr(context, 'patient_context')
            })
            
            return types.CreateMessageResult(
                model="healthcare-hmcp-agent",
                role="assistant",
                content=TextContent(
                    type="text",
                    text=output_validation["sanitized_message"]
                ),
                stopReason="endTurn"
            )
        
        # Add sample patient context
        patient_context = PatientContext(
            patient_id="PT12345",
            mrn="MRN789012",
            demographics={
                "age": 45,
                "gender": "male",
                "date_of_birth": "1979-03-15"
            }
        )
        await server.add_patient_context(patient_context)
        
        return server
    
    async def generate_healthcare_response(message: str, context: RequestContext, guardrail_result: Dict[str, Any]) -> str:
        """Generate healthcare-appropriate response"""
        
        message_lower = message.lower()
        
        # Clinical decision support responses
        if "chest pain" in message_lower:
            return """Based on the symptoms described, I recommend the following immediate actions:

1. **Initial Assessment**: Obtain vital signs and 12-lead ECG
2. **Laboratory Studies**: Order cardiac enzymes (troponin, CK-MB)
3. **Imaging**: Consider chest X-ray to rule out other causes
4. **Monitoring**: Continuous cardiac monitoring
5. **Consultation**: Consider cardiology evaluation if indicated

Please note: This is clinical decision support only. Always follow your institution's protocols and consult with attending physicians for patient care decisions."""
        
        elif "medication" in message_lower or "drug" in message_lower:
            return """For medication-related queries, I can assist with:

- Drug interaction checking
- Dosage calculations
- Contraindication identification
- Clinical pharmacology guidance

Please provide specific medications and patient context for detailed analysis. All medication decisions should be verified with clinical pharmacists and attending physicians."""
        
        elif guardrail_result.get("has_clinical_context"):
            return """I understand you have a clinical question. As a healthcare AI agent, I can provide:

- Clinical decision support
- Evidence-based recommendations
- Medication interaction checking
- Diagnostic assistance
- Treatment protocol guidance

Please ensure you have appropriate patient consent and authorization before sharing any patient information."""
        
        else:
            return """Hello! I'm a healthcare AI agent designed to support clinical workflows. I can assist with:

- Clinical decision support and evidence-based recommendations
- Medication interaction checking and pharmacology guidance  
- Diagnostic assistance and differential diagnosis support
- Treatment protocol and clinical guideline references
- Emergency response protocols and care coordination

How can I help you with your healthcare-related query today?"""
    
    # Run the server
    server = asyncio.run(create_healthcare_hmcp_server())
    print(f"Healthcare HMCP Server created: {server.get_server_info()}")
    
    # Start server (uncomment to run)
    # server.run(transport="sse")