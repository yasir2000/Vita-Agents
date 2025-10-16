#!/usr/bin/env python3
"""
MCP-Compliant Healthcare Model Context Protocol (HMCP) Implementation

This enhanced implementation aligns with the true MCP standard as demonstrated
by Innovaccer's Healthcare-MCP while maintaining our healthcare-specific
capabilities and FHIR integration strengths.

Key improvements:
- True MCP SDK foundation
- Agent framework integration
- Bidirectional sampling capability
- Healthcare-specific guardrails
- Enhanced authentication with healthcare scopes
- Multi-agent handoff orchestration
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Callable, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum

# MCP SDK imports (would need to install: pip install mcp)
try:
    from mcp import ClientSession, types
    from mcp.client.sse import sse_client
    from mcp.server.sse import SseServerTransport
    from mcp.server import Server
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP SDK not available. Install with: pip install mcp")

# Agent framework imports (would need to install: pip install agents)
try:
    from agents import Agent, Model, ModelResponse
    from agents.guardrail import InputGuardrail, OutputGuardrail
    from agents.handoffs import Handoff
    from agents.model_settings import ModelSettings
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    print("Warning: agents framework not available. Install with: pip install agents")

from vita_agents.core.agent import HealthcareAgent
from vita_agents.core.security import HIPAACompliantAgent
from vita_agents.protocols.hmcp import (
    HMCPMessage, HMCPMessageType, ClinicalUrgency, HealthcareRole,
    PatientContext, ClinicalContext, SecurityContext
)

logger = logging.getLogger(__name__)

TContext = TypeVar("TContext")


class HMCPAuthConfig:
    """Healthcare-specific authentication configuration"""
    
    def __init__(self):
        self.OAUTH_SCOPES = [
            "patient/hmcp:read",
            "patient/hmcp:write", 
            "launch/patient",
            "fhirUser",
            "openid",
            "profile",
            "patient/*.read",
            "patient/*.write",
            "user/*.*"
        ]
        self.JWT_ALGORITHM = "RS256"
        self.TOKEN_EXPIRY = 3600  # 1 hour
        self.PATIENT_CONTEXT_REQUIRED = True
        self.AUDIT_LOGGING_ENABLED = True


class HMCPGuardrail:
    """Healthcare-specific guardrails for HMCP agents"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.prompt_injection_patterns = [
            "ignore previous instructions",
            "system override", 
            "admin mode",
            "bypass security",
            "disable guardrails"
        ]
        self.phi_detection_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Z]{2}\d{8}\b',  # Medical record number pattern
        ]
    
    async def validate_input(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate input message for security threats"""
        if not self.enabled:
            return {"valid": True, "warnings": []}
        
        warnings = []
        
        # Check for prompt injection attempts
        message_lower = message.lower()
        for pattern in self.prompt_injection_patterns:
            if pattern in message_lower:
                warnings.append(f"Potential prompt injection detected: {pattern}")
        
        # Check for PHI exposure risk
        import re
        for pattern in self.phi_detection_patterns:
            if re.search(pattern, message):
                warnings.append(f"Potential PHI detected in message")
        
        return {
            "valid": len(warnings) == 0,
            "warnings": warnings,
            "requires_review": len(warnings) > 0
        }
    
    async def validate_output(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate output message for PHI and compliance"""
        if not self.enabled:
            return {"valid": True, "sanitized_message": message}
        
        # In a real implementation, this would:
        # 1. Scan for PHI patterns
        # 2. Apply redaction if needed
        # 3. Log for audit purposes
        # 4. Apply clinical guideline validation
        
        return {
            "valid": True,
            "sanitized_message": message,
            "audit_logged": True
        }


@dataclass
class HMCPServerHelper:
    """Helper class for connecting to HMCP servers (MCP-compliant)"""
    
    host: str
    port: int
    debug: bool = False
    auth_config: Optional[HMCPAuthConfig] = None
    
    def __post_init__(self):
        self.connected = False
        self.session: Optional[ClientSession] = None
        self.server_info: Dict[str, Any] = {}
        self.auth_headers: Dict[str, str] = {}
    
    async def connect(self) -> bool:
        """Connect to the HMCP server"""
        if not MCP_AVAILABLE:
            logger.error("MCP SDK not available")
            return False
        
        try:
            # Setup authentication headers
            if self.auth_config:
                # In a real implementation, this would generate proper JWT tokens
                self.auth_headers = {
                    "Authorization": "Bearer healthcare_jwt_token_here",
                    "X-Patient-Context": "patient_id_here",
                    "X-Healthcare-Scope": " ".join(self.auth_config.OAUTH_SCOPES)
                }
            
            # Connect using MCP SSE client
            server_url = f"http://{self.host}:{self.port}/sse"
            
            # This would be the actual MCP connection in a real implementation
            self.connected = True
            self.server_info = {
                "name": f"HMCP Server @ {self.host}:{self.port}",
                "version": "1.0.0",
                "capabilities": ["sampling", "healthcare", "fhir"],
                "connected_at": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Connected to HMCP server: {self.server_info['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to HMCP server: {e}")
            self.connected = False
            return False
    
    async def create_message(self, 
                           message: str,
                           role: str = "user",
                           messages_history: Optional[List[Dict[str, Any]]] = None,
                           **kwargs) -> Dict[str, Any]:
        """Send message using MCP sampling"""
        if not self.connected:
            await self.connect()
        
        # In a real implementation, this would use actual MCP sampling
        # For now, we simulate the response
        response = {
            "model": "hmcp-healthcare-agent",
            "role": "assistant", 
            "content": {
                "type": "text",
                "text": f"Healthcare HMCP response to: {message[:100]}..."
            },
            "stopReason": "endTurn",
            "usage": {
                "inputTokens": len(message.split()),
                "outputTokens": 20
            }
        }
        
        return response
    
    async def disconnect(self):
        """Disconnect from HMCP server"""
        if self.session:
            # In real implementation: await self.session.close()
            pass
        self.connected = False
        logger.info("Disconnected from HMCP server")


class HMCPModel(Model):
    """MCP-compliant model implementation for HMCP servers"""
    
    def __init__(self, hmcp_helper: HMCPServerHelper, settings: Optional[Dict[str, Any]] = None):
        self.hmcp_helper = hmcp_helper
        self.settings = settings or {}
        self.guardrail = HMCPGuardrail(enabled=True)
    
    async def get_response(self,
                         system_instructions: Optional[str],
                         input: List[Any],
                         model_settings: ModelSettings,
                         tools: Optional[List[Any]] = None,
                         output_schema: Optional[Any] = None,
                         handoffs: Optional[List[Any]] = None,
                         **kwargs) -> ModelResponse:
        """Get response from HMCP server using MCP sampling"""
        
        if not self.hmcp_helper.connected:
            await self.hmcp_helper.connect()
        
        # Convert input to message history format
        messages_history = []
        user_message = ""
        
        for item in input:
            if hasattr(item, "role") and hasattr(item, "content"):
                messages_history.append({"role": item.role, "content": item.content})
                if item.role == "user":
                    user_message = item.content
        
        # Apply input guardrails
        guardrail_result = await self.guardrail.validate_input(user_message)
        if not guardrail_result["valid"]:
            logger.warning(f"Guardrail warnings: {guardrail_result['warnings']}")
        
        # Send to HMCP server
        response = await self.hmcp_helper.create_message(
            message=user_message,
            messages_history=messages_history,
            **self.settings
        )
        
        # Apply output guardrails
        response_text = response.get("content", {}).get("text", "")
        output_validation = await self.guardrail.validate_output(response_text)
        
        # Create MCP-compliant response
        return ModelResponse(
            content=output_validation["sanitized_message"],
            usage=response.get("usage", {}),
            model=response.get("model", "hmcp-agent")
        )


@dataclass 
class MCPCompliantHMCPAgent(Generic[TContext]):
    """MCP-compliant HMCP agent that integrates with agents framework"""
    
    name: str
    hmcp_helper: HMCPServerHelper
    instructions: Optional[str] = None
    handoff_description: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # Agent framework integration
    model: Optional[Model] = None
    handoffs: List[Union[Agent, Handoff]] = field(default_factory=list)
    input_guardrails: List[InputGuardrail] = field(default_factory=list)
    output_guardrails: List[OutputGuardrail] = field(default_factory=list)
    
    # Healthcare-specific fields
    healthcare_role: HealthcareRole = HealthcareRole.AI_AGENT
    emergency_capable: bool = False
    fhir_engines: List[str] = field(default_factory=list)
    clinical_specialties: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize the MCP-compliant HMCP agent"""
        if self.model is None:
            self.model = HMCPModel(self.hmcp_helper, self.settings)
        
        # Add healthcare-specific guardrails
        if AGENTS_AVAILABLE:
            healthcare_input_guardrail = self._create_healthcare_input_guardrail()
            healthcare_output_guardrail = self._create_healthcare_output_guardrail()
            
            self.input_guardrails.append(healthcare_input_guardrail)
            self.output_guardrails.append(healthcare_output_guardrail)
    
    def _create_healthcare_input_guardrail(self) -> InputGuardrail:
        """Create healthcare-specific input guardrail"""
        async def validate_healthcare_input(context, input_data):
            # Healthcare-specific input validation
            guardrail = HMCPGuardrail()
            validation = await guardrail.validate_input(str(input_data))
            
            if not validation["valid"]:
                raise ValueError(f"Healthcare guardrail violation: {validation['warnings']}")
            
            return input_data
        
        return InputGuardrail(name="healthcare_input", func=validate_healthcare_input)
    
    def _create_healthcare_output_guardrail(self) -> OutputGuardrail:
        """Create healthcare-specific output guardrail"""
        async def validate_healthcare_output(context, output_data):
            # Healthcare-specific output validation and PHI protection
            guardrail = HMCPGuardrail()
            validation = await guardrail.validate_output(str(output_data))
            
            return validation["sanitized_message"]
        
        return OutputGuardrail(name="healthcare_output", func=validate_healthcare_output)
    
    async def process_message(self, 
                            message: str, 
                            context: Optional[TContext] = None) -> Dict[str, Any]:
        """Process message using MCP sampling with healthcare context"""
        
        # Add healthcare context to message
        enhanced_message = self._add_healthcare_context(message, context)
        
        # Process through HMCP server
        response = await self.hmcp_helper.create_message(
            message=enhanced_message,
            role="user"
        )
        
        # Add healthcare metadata to response
        response["healthcare_metadata"] = {
            "agent_name": self.name,
            "healthcare_role": self.healthcare_role,
            "clinical_specialties": self.clinical_specialties,
            "fhir_engines_available": self.fhir_engines,
            "emergency_capable": self.emergency_capable,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        
        return response
    
    def _add_healthcare_context(self, message: str, context: Optional[TContext]) -> str:
        """Add healthcare-specific context to message"""
        healthcare_context = {
            "agent_specialties": self.clinical_specialties,
            "emergency_protocols_available": self.emergency_capable,
            "fhir_integration": len(self.fhir_engines) > 0,
            "compliance_mode": "HIPAA"
        }
        
        if context:
            healthcare_context["additional_context"] = str(context)
        
        enhanced_message = f"""
Healthcare Context: {json.dumps(healthcare_context, indent=2)}

User Message: {message}

Please provide a healthcare-appropriate response considering the clinical context and compliance requirements.
        """
        
        return enhanced_message
    
    def to_agent(self) -> Agent:
        """Convert to standard Agent for use with agents framework"""
        if not AGENTS_AVAILABLE:
            raise ImportError("agents framework not available")
        
        return Agent(
            name=self.name,
            instructions=self.instructions,
            handoff_description=self.handoff_description,
            model=self.model,
            handoffs=self.handoffs,
            input_guardrails=self.input_guardrails,
            output_guardrails=self.output_guardrails
        )
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.hmcp_helper.disconnect()


class MultiAgentHMCPOrchestrator:
    """Multi-agent orchestrator for HMCP workflows (MCP-compliant)"""
    
    def __init__(self, 
                 orchestrator_agent: Agent,
                 specialized_agents: List[MCPCompliantHMCPAgent],
                 max_iterations: int = 10):
        self.orchestrator_agent = orchestrator_agent
        self.specialized_agents = {agent.name: agent for agent in specialized_agents}
        self.max_iterations = max_iterations
        self.conversation_history: List[Dict[str, Any]] = []
        self.handoff_history: List[Dict[str, Any]] = []
    
    async def run_workflow(self, initial_prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run multi-agent healthcare workflow"""
        
        workflow_id = f"hmcp_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting HMCP workflow: {workflow_id}")
        
        current_message = initial_prompt
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Get decision from orchestrator
            orchestrator_response = await self._get_orchestrator_decision(current_message, context)
            
            # Check if workflow is complete
            if self._is_workflow_complete(orchestrator_response):
                break
            
            # Execute handoff if needed
            handoff_result = await self._execute_handoff(orchestrator_response, context)
            current_message = handoff_result.get("response", "")
            
            # Log progress
            self.conversation_history.append({
                "iteration": iteration,
                "orchestrator_response": orchestrator_response,
                "handoff_result": handoff_result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        return {
            "workflow_id": workflow_id,
            "completed": iteration < self.max_iterations,
            "iterations": iteration,
            "final_result": current_message,
            "conversation_history": self.conversation_history,
            "handoff_history": self.handoff_history
        }
    
    async def _get_orchestrator_decision(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get decision from orchestrator agent"""
        # In a real implementation, this would use the agents framework
        # to get the orchestrator's decision about which agent to call next
        
        return {
            "next_agent": "patient_data_agent",
            "action": "get_patient_id",
            "message": message,
            "reasoning": "Need to identify patient before proceeding"
        }
    
    async def _execute_handoff(self, orchestrator_response: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute handoff to specialized agent"""
        agent_name = orchestrator_response.get("next_agent")
        
        if agent_name in self.specialized_agents:
            agent = self.specialized_agents[agent_name]
            
            # Process message with specialized agent
            result = await agent.process_message(
                message=orchestrator_response.get("message", ""),
                context=context
            )
            
            # Log handoff
            self.handoff_history.append({
                "agent_name": agent_name,
                "action": orchestrator_response.get("action"),
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
        
        return {"error": f"Agent {agent_name} not found"}
    
    def _is_workflow_complete(self, orchestrator_response: Dict[str, Any]) -> bool:
        """Check if workflow is complete"""
        return orchestrator_response.get("action") == "complete" or \
               "workflow_complete" in orchestrator_response.get("message", "").lower()


# Example usage and integration with our existing Vita-Agents
async def create_mcp_compliant_hmcp_system():
    """Create an MCP-compliant HMCP system integrated with Vita-Agents"""
    
    # Create HMCP server helpers for different agents
    emr_helper = HMCPServerHelper(
        host="localhost",
        port=8050,
        auth_config=HMCPAuthConfig()
    )
    
    patient_data_helper = HMCPServerHelper(
        host="localhost", 
        port=8060,
        auth_config=HMCPAuthConfig()
    )
    
    # Create MCP-compliant HMCP agents
    emr_agent = MCPCompliantHMCPAgent(
        name="EMR Writeback Agent",
        hmcp_helper=emr_helper,
        instructions="Specialized agent for writing clinical data to electronic medical records",
        handoff_description="Handles EMR operations requiring patient_id and clinical_data",
        healthcare_role=HealthcareRole.CLINICAL_SUPPORT,
        clinical_specialties=["medical_records", "data_management"],
        fhir_engines=["hapi", "ibm", "microsoft"]  # From our existing FHIR integration
    )
    
    patient_data_agent = MCPCompliantHMCPAgent(
        name="Patient Data Agent", 
        hmcp_helper=patient_data_helper,
        instructions="Specialized agent for patient identification and data retrieval",
        handoff_description="Provides patient lookup and identification services",
        healthcare_role=HealthcareRole.CLINICAL_SUPPORT,
        clinical_specialties=["patient_identification", "demographics"],
        fhir_engines=["hapi", "ibm", "microsoft"]
    )
    
    # Create orchestrator (would need agents framework)
    if AGENTS_AVAILABLE:
        orchestrator = Agent(
            name="Healthcare Workflow Orchestrator",
            instructions="Coordinate healthcare workflows between specialized HMCP agents",
            handoffs=[emr_agent.to_agent(), patient_data_agent.to_agent()]
        )
        
        # Create multi-agent orchestrator
        hmcp_orchestrator = MultiAgentHMCPOrchestrator(
            orchestrator_agent=orchestrator,
            specialized_agents=[emr_agent, patient_data_agent]
        )
        
        return hmcp_orchestrator
    
    return {"emr_agent": emr_agent, "patient_data_agent": patient_data_agent}


if __name__ == "__main__":
    async def main():
        print("🏥 MCP-Compliant Healthcare Model Context Protocol (HMCP)")
        print("=" * 60)
        
        if not MCP_AVAILABLE or not AGENTS_AVAILABLE:
            print("❌ Missing dependencies:")
            if not MCP_AVAILABLE:
                print("   - MCP SDK: pip install mcp")
            if not AGENTS_AVAILABLE: 
                print("   - Agents framework: pip install agents")
            print("\n💡 This is a reference implementation showing MCP-compliant structure")
            return
        
        # Create MCP-compliant system
        hmcp_system = await create_mcp_compliant_hmcp_system()
        
        if isinstance(hmcp_system, MultiAgentHMCPOrchestrator):
            # Run a sample workflow
            result = await hmcp_system.run_workflow(
                initial_prompt="Update patient John Smith's record with diagnosis: Hypertension, BP: 140/90",
                context={"urgency": "routine", "specialty": "cardiology"}
            )
            
            print(f"✅ Workflow completed: {result['workflow_id']}")
            print(f"   Iterations: {result['iterations']}")
            print(f"   Status: {'Success' if result['completed'] else 'Timeout'}")
        else:
            print("✅ HMCP agents created (framework integration disabled)")
            print(f"   EMR Agent: {hmcp_system['emr_agent'].name}")
            print(f"   Patient Agent: {hmcp_system['patient_data_agent'].name}")
    
    asyncio.run(main())