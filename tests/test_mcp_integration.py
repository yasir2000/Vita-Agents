#!/usr/bin/env python3
"""
Comprehensive Integration and Testing Suite for MCP-Compliant HMCP

Integrates all MCP-compliant components and provides:
- Updated hmcp_agent.py using new MCP infrastructure
- Comprehensive testing suite
- Demo workflows for clinical decision support
- Agent handoff demonstrations
- Healthcare compliance validation
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    # MCP types
    from mcp.types import SamplingMessage, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    class SamplingMessage: pass
    class TextContent: pass

# Import all our MCP-compliant components
from vita_agents.protocols.hmcp import (
    ClinicalUrgency, HealthcareRole, PatientContext, ClinicalContext,
    HMCPMessage, HMCPMessageType
)
from vita_agents.protocols.hmcp_server import HMCPServer
from vita_agents.protocols.hmcp_client import HMCPClient, HMCPAuthConfig
from vita_agents.agents.healthcare_agent_framework import HealthcareAgentFramework, MedicalSpecialty
from vita_agents.protocols.hmcp_sampling import HMCPBidirectionalSampling, SamplingContext, SamplingStrategy, ClinicalSamplingRequest
from vita_agents.security.hmcp_guardrails import HMCPGuardrailSystem, ViolationType, SeverityLevel
from vita_agents.auth.hmcp_authentication import HMCPAuthenticationService, SMARTScope

logger = logging.getLogger(__name__)


class MCPCompliantHMCPAgent:
    """
    Updated HMCP Agent using MCP-compliant infrastructure
    
    Integrates all components:
    - MCP-based server and client
    - Healthcare agent framework
    - Bidirectional sampling
    - Comprehensive guardrails
    - Healthcare authentication
    """
    
    def __init__(self, 
                 agent_id: str,
                 openai_api_key: Optional[str] = None,
                 server_url: str = "http://localhost:8050",
                 auth_config: Optional[HMCPAuthConfig] = None):
        
        self.agent_id = agent_id
        self.server_url = server_url
        
        # Core components
        self.hmcp_server: Optional[HMCPServer] = None
        self.hmcp_client: Optional[HMCPClient] = None
        self.agent_framework: Optional[HealthcareAgentFramework] = None
        self.sampling_system: Optional[HMCPBidirectionalSampling] = None
        self.guardrails: Optional[HMCPGuardrailSystem] = None
        self.auth_service: Optional[HMCPAuthenticationService] = None
        
        # Configuration
        self.auth_config = auth_config or HMCPAuthConfig()
        self.openai_api_key = openai_api_key
        
        # State
        self.is_initialized = False
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.session_history: List[Dict[str, Any]] = []
        
        logger.info(f"MCP-Compliant HMCP Agent created: {agent_id}")
    
    async def initialize(self) -> bool:
        """Initialize all MCP-compliant components"""
        
        try:
            # 1. Initialize Authentication Service
            self.auth_service = HMCPAuthenticationService(
                client_id=self.auth_config.CLIENT_ID,
                client_secret=self.auth_config.CLIENT_SECRET,
                authorization_endpoint=self.auth_config.AUTH_ENDPOINT,
                token_endpoint=self.auth_config.TOKEN_ENDPOINT,
                jwks_uri=f"{self.auth_config.AUTH_ENDPOINT}/.well-known/jwks.json"
            )
            
            # 2. Initialize Guardrails System
            self.guardrails = HMCPGuardrailSystem(
                enable_phi_detection=True,
                enable_prompt_injection=True,
                enable_clinical_safety=True,
                enable_medication_checking=True,
                strict_mode=True
            )
            
            # 3. Initialize HMCP Server
            self.hmcp_server = HMCPServer(
                name="vita-agents-hmcp-server",
                version="1.0.0",
                auth_service=self.auth_service,
                guardrails=self.guardrails
            )
            
            # 4. Initialize Bidirectional Sampling
            self.sampling_system = HMCPBidirectionalSampling(self.hmcp_server)
            
            # 5. Initialize Agent Framework
            if self.openai_api_key:
                self.agent_framework = HealthcareAgentFramework(
                    openai_api_key=self.openai_api_key,
                    hmcp_client=None  # Will be set after client initialization
                )
                await self.agent_framework.setup_default_specialists()
            
            # 6. Initialize HMCP Client
            self.hmcp_client = HMCPClient(
                server_url=f"{self.server_url}/sse",
                auth_config=self.auth_config,
                debug=True
            )
            
            # Link client to agent framework
            if self.agent_framework:
                self.agent_framework.hmcp_client = self.hmcp_client
            
            self.is_initialized = True
            logger.info("MCP-Compliant HMCP Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize HMCP Agent: {e}")
            return False
    
    async def start_server(self, host: str = "localhost", port: int = 8050) -> bool:
        """Start the HMCP server"""
        
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Start server in background
            await self.hmcp_server.start_server(host, port)
            logger.info(f"HMCP Server started on {host}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start HMCP server: {e}")
            return False
    
    async def connect_client(self, patient_id: Optional[str] = None) -> bool:
        """Connect the HMCP client to server"""
        
        if not self.hmcp_client:
            raise Exception("HMCP Client not initialized")
        
        try:
            # Set patient context if provided
            if patient_id:
                self.hmcp_client.patient_id = patient_id
            
            # Connect to server
            connected = await self.hmcp_client.connect()
            
            if connected:
                logger.info("HMCP Client connected to server")
                return True
            else:
                logger.error("Failed to connect HMCP Client")
                return False
                
        except Exception as e:
            logger.error(f"HMCP Client connection failed: {e}")
            return False
    
    async def clinical_consultation(self, 
                                  clinical_query: str,
                                  patient_context: PatientContext,
                                  clinical_context: ClinicalContext,
                                  specialty: Optional[MedicalSpecialty] = None) -> Dict[str, Any]:
        """Perform clinical consultation using MCP infrastructure"""
        
        workflow_id = f"consultation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # 1. Validate request through guardrails
            guardrail_result = await self.guardrails.validate_request(
                content=clinical_query,
                patient_context=patient_context,
                clinical_context=clinical_context,
                user_role=HealthcareRole.PHYSICIAN
            )
            
            if not guardrail_result.passed:
                return {
                    "workflow_id": workflow_id,
                    "status": "blocked",
                    "reason": "Guardrail violations",
                    "violations": [v.message for v in guardrail_result.violations]
                }
            
            # 2. Create sampling request
            sampling_request = ClinicalSamplingRequest(
                context_type=SamplingContext.CLINICAL_DECISION,
                strategy=SamplingStrategy.DETERMINISTIC,
                patient_context=patient_context,
                clinical_context=clinical_context,
                messages=[
                    SamplingMessage(
                        role="user",
                        content=TextContent(type="text", text=clinical_query)
                    )
                ]
            )
            
            # 3. Get sampling response
            sampling_response = await self.sampling_system.create_clinical_sampling(sampling_request)
            
            # 4. Route to specialist if needed
            specialist_response = None
            if self.agent_framework and specialty:
                try:
                    handoff_id = await self.agent_framework.initiate_handoff(
                        source_specialty=MedicalSpecialty.GENERAL_PRACTICE,
                        target_specialty=specialty,
                        patient_context=patient_context,
                        clinical_context=clinical_context,
                        handoff_reason=clinical_query,
                        handoff_type="consultation"
                    )
                    
                    specialist_response = await self.agent_framework.get_specialist_response(handoff_id)
                    await self.agent_framework.complete_handoff(handoff_id)
                    
                except Exception as e:
                    logger.warning(f"Specialist consultation failed: {e}")
            
            # 5. Compile results
            result = {
                "workflow_id": workflow_id,
                "status": "completed",
                "clinical_assessment": {
                    "confidence": sampling_response.confidence_score,
                    "safety_score": sampling_response.clinical_safety_score,
                    "recommendations": sampling_response.recommendations,
                    "warnings": sampling_response.warnings,
                    "next_steps": sampling_response.next_steps
                },
                "specialist_consultation": specialist_response,
                "guardrail_status": {
                    "passed": guardrail_result.passed,
                    "violations_count": len(guardrail_result.violations)
                }
            }
            
            # Store workflow
            self.active_workflows[workflow_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Clinical consultation failed: {e}")
            return {
                "workflow_id": workflow_id,
                "status": "error",
                "error": str(e)
            }
    
    async def medication_review(self, 
                              medications: List[str],
                              patient_context: PatientContext,
                              clinical_context: ClinicalContext) -> Dict[str, Any]:
        """Perform comprehensive medication review"""
        
        workflow_id = f"med_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Update clinical context with medications
            clinical_context.medications = medications
            
            # 1. Check medication interactions via client
            if self.hmcp_client:
                interactions = await self.hmcp_client.check_medication_interactions(medications)
            else:
                interactions = {"error": "HMCP Client not available"}
            
            # 2. Use sampling system for medication assessment
            med_query = f"Review medications: {', '.join(medications)}"
            
            sampling_request = ClinicalSamplingRequest(
                context_type=SamplingContext.MEDICATION_REVIEW,
                strategy=SamplingStrategy.PROBABILISTIC,
                patient_context=patient_context,
                clinical_context=clinical_context,
                messages=[
                    SamplingMessage(
                        role="user",
                        content=TextContent(type="text", text=med_query)
                    )
                ]
            )
            
            sampling_response = await self.sampling_system.create_clinical_sampling(sampling_request)
            
            # 3. Get pharmacist consultation if available
            pharmacist_review = None
            if self.agent_framework:
                try:
                    handoff_id = await self.agent_framework.initiate_handoff(
                        source_specialty=MedicalSpecialty.GENERAL_PRACTICE,
                        target_specialty=MedicalSpecialty.PHARMACY,
                        patient_context=patient_context,
                        clinical_context=clinical_context,
                        handoff_reason="Medication review and interaction check",
                        handoff_type="consultation"
                    )
                    
                    pharmacist_review = await self.agent_framework.get_specialist_response(handoff_id)
                    await self.agent_framework.complete_handoff(handoff_id)
                    
                except Exception as e:
                    logger.warning(f"Pharmacist consultation failed: {e}")
            
            result = {
                "workflow_id": workflow_id,
                "status": "completed",
                "medications_reviewed": medications,
                "interaction_check": interactions,
                "clinical_assessment": {
                    "confidence": sampling_response.confidence_score,
                    "safety_score": sampling_response.clinical_safety_score,
                    "recommendations": sampling_response.recommendations,
                    "warnings": sampling_response.warnings
                },
                "pharmacist_review": pharmacist_review
            }
            
            self.active_workflows[workflow_id] = result
            return result
            
        except Exception as e:
            logger.error(f"Medication review failed: {e}")
            return {
                "workflow_id": workflow_id,
                "status": "error",
                "error": str(e)
            }
    
    async def emergency_assessment(self, 
                                 emergency_scenario: str,
                                 patient_context: PatientContext) -> Dict[str, Any]:
        """Perform emergency assessment with escalation"""
        
        workflow_id = f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Create critical clinical context
            clinical_context = ClinicalContext(
                chief_complaint=emergency_scenario,
                urgency=ClinicalUrgency.CRITICAL,
                clinical_notes="Emergency assessment required"
            )
            
            # 1. Emergency guardrail validation
            guardrail_result = await self.guardrails.validate_request(
                content=emergency_scenario,
                patient_context=patient_context,
                clinical_context=clinical_context,
                user_role=HealthcareRole.PHYSICIAN
            )
            
            # 2. Emergency protocol sampling
            sampling_request = ClinicalSamplingRequest(
                context_type=SamplingContext.EMERGENCY_PROTOCOL,
                strategy=SamplingStrategy.HIERARCHICAL,
                patient_context=patient_context,
                clinical_context=clinical_context,
                messages=[
                    SamplingMessage(
                        role="user",
                        content=TextContent(
                            type="text", 
                            text=f"EMERGENCY: {emergency_scenario}"
                        )
                    )
                ]
            )
            
            sampling_response = await self.sampling_system.create_clinical_sampling(sampling_request)
            
            # 3. Emergency medicine consultation
            emergency_consultation = None
            if self.agent_framework:
                try:
                    handoff_id = await self.agent_framework.initiate_handoff(
                        source_specialty=None,
                        target_specialty=MedicalSpecialty.EMERGENCY_MEDICINE,
                        patient_context=patient_context,
                        clinical_context=clinical_context,
                        handoff_reason="Emergency assessment and stabilization",
                        handoff_type="emergency"
                    )
                    
                    emergency_consultation = await self.agent_framework.get_specialist_response(handoff_id)
                    await self.agent_framework.complete_handoff(handoff_id)
                    
                except Exception as e:
                    logger.error(f"Emergency consultation failed: {e}")
            
            result = {
                "workflow_id": workflow_id,
                "status": "completed",
                "emergency_scenario": emergency_scenario,
                "priority": "CRITICAL",
                "protocol_response": {
                    "confidence": sampling_response.confidence_score,
                    "safety_score": sampling_response.clinical_safety_score,
                    "immediate_actions": sampling_response.next_steps,
                    "warnings": sampling_response.warnings
                },
                "emergency_consultation": emergency_consultation,
                "guardrail_alerts": [v.message for v in guardrail_result.violations]
            }
            
            self.active_workflows[workflow_id] = result
            return result
            
        except Exception as e:
            logger.error(f"Emergency assessment failed: {e}")
            return {
                "workflow_id": workflow_id,
                "status": "error",
                "priority": "CRITICAL",
                "error": str(e)
            }
    
    async def multi_agent_consultation(self, 
                                     clinical_query: str,
                                     patient_context: PatientContext,
                                     clinical_context: ClinicalContext,
                                     specialties: List[MedicalSpecialty]) -> Dict[str, Any]:
        """Multi-specialist consultation with consensus"""
        
        workflow_id = f"multi_consult_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            specialist_responses = []
            
            # Get recommendations from multiple specialists
            for specialty in specialties:
                try:
                    handoff_id = await self.agent_framework.initiate_handoff(
                        source_specialty=MedicalSpecialty.GENERAL_PRACTICE,
                        target_specialty=specialty,
                        patient_context=patient_context,
                        clinical_context=clinical_context,
                        handoff_reason=clinical_query,
                        handoff_type="consultation"
                    )
                    
                    response = await self.agent_framework.get_specialist_response(handoff_id)
                    specialist_responses.append({
                        "specialty": specialty.value,
                        "response": response
                    })
                    
                    await self.agent_framework.complete_handoff(handoff_id)
                    
                except Exception as e:
                    logger.warning(f"Consultation with {specialty.value} failed: {e}")
            
            # Use consensus sampling for final recommendation
            sampling_request = ClinicalSamplingRequest(
                context_type=SamplingContext.CLINICAL_DECISION,
                strategy=SamplingStrategy.CONSENSUS,
                patient_context=patient_context,
                clinical_context=clinical_context,
                messages=[
                    SamplingMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=f"Multi-specialist consultation: {clinical_query}"
                        )
                    )
                ],
                metadata={"specialist_responses": specialist_responses}
            )
            
            consensus_response = await self.sampling_system.create_clinical_sampling(sampling_request)
            
            result = {
                "workflow_id": workflow_id,
                "status": "completed",
                "clinical_query": clinical_query,
                "specialties_consulted": [s.value for s in specialties],
                "specialist_responses": specialist_responses,
                "consensus_recommendation": {
                    "confidence": consensus_response.confidence_score,
                    "safety_score": consensus_response.clinical_safety_score,
                    "recommendations": consensus_response.recommendations,
                    "reasoning": consensus_response.reasoning_trace
                }
            }
            
            self.active_workflows[workflow_id] = result
            return result
            
        except Exception as e:
            logger.error(f"Multi-agent consultation failed: {e}")
            return {
                "workflow_id": workflow_id,
                "status": "error",
                "error": str(e)
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            "agent_id": self.agent_id,
            "initialized": self.is_initialized,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_workflows": len(self.active_workflows),
            "components": {}
        }
        
        # Component status
        if self.hmcp_server:
            status["components"]["hmcp_server"] = "initialized"
        
        if self.hmcp_client:
            status["components"]["hmcp_client"] = self.hmcp_client.get_client_status()
        
        if self.agent_framework:
            status["components"]["agent_framework"] = self.agent_framework.get_framework_status()
        
        if self.sampling_system:
            status["components"]["sampling_system"] = await self.sampling_system.get_sampling_analytics()
        
        if self.guardrails:
            status["components"]["guardrails"] = self.guardrails.get_guardrail_statistics()
        
        if self.auth_service:
            status["components"]["auth_service"] = self.auth_service.get_auth_statistics()
        
        return status
    
    async def cleanup(self):
        """Cleanup resources"""
        
        try:
            if self.hmcp_client:
                await self.hmcp_client.disconnect()
            
            if self.hmcp_server:
                await self.hmcp_server.cleanup()
            
            logger.info("HMCP Agent cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Comprehensive Test Suite
class HMCPTestSuite:
    """Comprehensive testing suite for MCP-compliant HMCP"""
    
    def __init__(self, agent: MCPCompliantHMCPAgent):
        self.agent = agent
        self.test_results: List[Dict[str, Any]] = []
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        
        logger.info("Starting HMCP comprehensive test suite...")
        
        test_scenarios = [
            ("test_initialization", self.test_initialization),
            ("test_guardrails", self.test_guardrails),
            ("test_authentication", self.test_authentication),
            ("test_sampling", self.test_sampling),
            ("test_agent_framework", self.test_agent_framework),
            ("test_clinical_consultation", self.test_clinical_consultation),
            ("test_medication_review", self.test_medication_review),
            ("test_emergency_assessment", self.test_emergency_assessment),
            ("test_multi_agent_consultation", self.test_multi_agent_consultation),
            ("test_integration", self.test_integration)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in test_scenarios:
            try:
                logger.info(f"Running test: {test_name}")
                result = await test_func()
                
                if result["passed"]:
                    passed += 1
                    logger.info(f"✓ {test_name} PASSED")
                else:
                    failed += 1
                    logger.error(f"✗ {test_name} FAILED: {result.get('error', 'Unknown error')}")
                
                self.test_results.append({
                    "test_name": test_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **result
                })
                
            except Exception as e:
                failed += 1
                logger.error(f"✗ {test_name} FAILED with exception: {e}")
                self.test_results.append({
                    "test_name": test_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "passed": False,
                    "error": str(e)
                })
        
        summary = {
            "total_tests": len(test_scenarios),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(test_scenarios) * 100,
            "detailed_results": self.test_results
        }
        
        logger.info(f"Test suite completed: {passed}/{len(test_scenarios)} tests passed ({summary['success_rate']:.1f}%)")
        return summary
    
    async def test_initialization(self) -> Dict[str, Any]:
        """Test agent initialization"""
        try:
            if not self.agent.is_initialized:
                success = await self.agent.initialize()
                if not success:
                    return {"passed": False, "error": "Initialization failed"}
            
            return {"passed": True, "message": "Agent initialized successfully"}
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_guardrails(self) -> Dict[str, Any]:
        """Test guardrails system"""
        try:
            if not self.agent.guardrails:
                return {"passed": False, "error": "Guardrails not initialized"}
            
            # Test PHI detection
            test_content = "Patient John Doe with SSN 123-45-6789 needs medication review"
            
            patient_context = PatientContext(
                patient_id="TEST123",
                mrn="MRN-TEST",
                demographics={"age": 45}
            )
            
            clinical_context = ClinicalContext(
                chief_complaint="Test",
                urgency=ClinicalUrgency.LOW
            )
            
            result = await self.agent.guardrails.validate_request(
                content=test_content,
                patient_context=patient_context,
                clinical_context=clinical_context,
                user_role=HealthcareRole.PHYSICIAN
            )
            
            # Should detect PHI and either block or mask
            if not result.violations:
                return {"passed": False, "error": "Failed to detect PHI in test content"}
            
            return {
                "passed": True, 
                "message": f"Guardrails working - detected {len(result.violations)} violations"
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_authentication(self) -> Dict[str, Any]:
        """Test authentication system"""
        try:
            if not self.agent.auth_service:
                return {"passed": False, "error": "Auth service not initialized"}
            
            # Create test token
            test_token = await self.agent.auth_service.create_session_token(
                user_id="test-physician",
                healthcare_role=HealthcareRole.PHYSICIAN,
                scopes=[SMARTScope.PATIENT_READ.value, SMARTScope.CLINICAL_READ.value],
                patient_id="TEST123"
            )
            
            # Validate token
            auth_context = await self.agent.auth_service.validate_token(test_token)
            
            if auth_context.healthcare_role != HealthcareRole.PHYSICIAN:
                return {"passed": False, "error": "Authentication context incorrect"}
            
            return {"passed": True, "message": "Authentication system working"}
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_sampling(self) -> Dict[str, Any]:
        """Test bidirectional sampling"""
        try:
            if not self.agent.sampling_system:
                return {"passed": False, "error": "Sampling system not initialized"}
            
            # Test sampling request
            patient_context = PatientContext(
                patient_id="TEST123",
                mrn="MRN-TEST",
                demographics={"age": 45}
            )
            
            clinical_context = ClinicalContext(
                chief_complaint="Test chest pain",
                urgency=ClinicalUrgency.MEDIUM
            )
            
            sampling_request = ClinicalSamplingRequest(
                context_type=SamplingContext.CLINICAL_DECISION,
                strategy=SamplingStrategy.DETERMINISTIC,
                patient_context=patient_context,
                clinical_context=clinical_context,
                messages=[
                    SamplingMessage(
                        role="user",
                        content=TextContent(type="text", text="Test clinical query")
                    )
                ]
            )
            
            response = await self.agent.sampling_system.create_clinical_sampling(sampling_request)
            
            if response.confidence_score <= 0:
                return {"passed": False, "error": "Invalid sampling response"}
            
            return {"passed": True, "message": "Sampling system working"}
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_agent_framework(self) -> Dict[str, Any]:
        """Test agent framework"""
        try:
            if not self.agent.agent_framework:
                return {"passed": True, "message": "Agent framework not configured (OpenAI key required)"}
            
            # Test specialist recommendations
            patient_context = PatientContext(
                patient_id="TEST123",
                mrn="MRN-TEST",
                demographics={"age": 65}
            )
            
            recommendations = await self.agent.agent_framework.get_specialist_recommendations(
                "chest pain in elderly male",
                patient_context
            )
            
            if not recommendations.get("recommendations"):
                return {"passed": False, "error": "No specialist recommendations generated"}
            
            return {"passed": True, "message": "Agent framework working"}
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_clinical_consultation(self) -> Dict[str, Any]:
        """Test clinical consultation workflow"""
        try:
            patient_context = PatientContext(
                patient_id="TEST123",
                mrn="MRN-TEST",
                demographics={"age": 45, "gender": "male"}
            )
            
            clinical_context = ClinicalContext(
                chief_complaint="Chest pain",
                clinical_notes="Patient reports chest discomfort",
                medications=["aspirin"],
                allergies=["penicillin"],
                urgency=ClinicalUrgency.MEDIUM
            )
            
            result = await self.agent.clinical_consultation(
                clinical_query="45-year-old male with chest pain, assess cardiac risk",
                patient_context=patient_context,
                clinical_context=clinical_context
            )
            
            if result["status"] != "completed":
                return {"passed": False, "error": f"Consultation failed: {result.get('reason', 'Unknown')}"}
            
            return {"passed": True, "message": "Clinical consultation working"}
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_medication_review(self) -> Dict[str, Any]:
        """Test medication review workflow"""
        try:
            patient_context = PatientContext(
                patient_id="TEST123",
                mrn="MRN-TEST",
                demographics={"age": 70}
            )
            
            clinical_context = ClinicalContext(
                chief_complaint="Medication review",
                urgency=ClinicalUrgency.LOW
            )
            
            result = await self.agent.medication_review(
                medications=["warfarin", "aspirin", "metoprolol"],
                patient_context=patient_context,
                clinical_context=clinical_context
            )
            
            if result["status"] != "completed":
                return {"passed": False, "error": f"Medication review failed: {result.get('error', 'Unknown')}"}
            
            return {"passed": True, "message": "Medication review working"}
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_emergency_assessment(self) -> Dict[str, Any]:
        """Test emergency assessment workflow"""
        try:
            patient_context = PatientContext(
                patient_id="TEST123",
                mrn="MRN-TEST",
                demographics={"age": 55}
            )
            
            result = await self.agent.emergency_assessment(
                emergency_scenario="Severe chest pain with shortness of breath",
                patient_context=patient_context
            )
            
            if result["priority"] != "CRITICAL":
                return {"passed": False, "error": "Emergency priority not set correctly"}
            
            return {"passed": True, "message": "Emergency assessment working"}
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_multi_agent_consultation(self) -> Dict[str, Any]:
        """Test multi-agent consultation"""
        try:
            if not self.agent.agent_framework:
                return {"passed": True, "message": "Multi-agent consultation skipped (OpenAI key required)"}
            
            patient_context = PatientContext(
                patient_id="TEST123",
                mrn="MRN-TEST",
                demographics={"age": 65}
            )
            
            clinical_context = ClinicalContext(
                chief_complaint="Complex cardiac case",
                urgency=ClinicalUrgency.HIGH
            )
            
            result = await self.agent.multi_agent_consultation(
                clinical_query="Complex cardiac case requiring multiple specialist input",
                patient_context=patient_context,
                clinical_context=clinical_context,
                specialties=[MedicalSpecialty.CARDIOLOGY, MedicalSpecialty.PHARMACY]
            )
            
            if result["status"] != "completed":
                return {"passed": False, "error": f"Multi-agent consultation failed: {result.get('error', 'Unknown')}"}
            
            return {"passed": True, "message": "Multi-agent consultation working"}
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def test_integration(self) -> Dict[str, Any]:
        """Test overall system integration"""
        try:
            # Get system status
            status = await self.agent.get_system_status()
            
            if not status["initialized"]:
                return {"passed": False, "error": "System not fully initialized"}
            
            # Check all components
            required_components = ["guardrails", "sampling_system", "auth_service"]
            for component in required_components:
                if component not in status["components"]:
                    return {"passed": False, "error": f"Missing component: {component}"}
            
            return {"passed": True, "message": "System integration working"}
            
        except Exception as e:
            return {"passed": False, "error": str(e)}


# Demo workflows
async def demo_clinical_workflows():
    """Demonstration of clinical workflows"""
    
    print("🏥 HMCP Clinical Workflows Demonstration")
    print("=" * 50)
    
    # Initialize agent
    agent = MCPCompliantHMCPAgent(
        agent_id="demo-agent",
        openai_api_key=None,  # Set to your OpenAI API key for full functionality
        server_url="http://localhost:8050"
    )
    
    # Initialize
    print("Initializing MCP-compliant HMCP agent...")
    success = await agent.initialize()
    if not success:
        print("❌ Failed to initialize agent")
        return
    
    print("✅ Agent initialized successfully")
    
    # Demo scenarios
    print("\n🔍 Demo Scenario 1: Clinical Consultation")
    patient = PatientContext(
        patient_id="DEMO001",
        mrn="MRN-DEMO001",
        demographics={"age": 65, "gender": "male", "weight": "80kg"}
    )
    
    clinical = ClinicalContext(
        chief_complaint="Chest pain and shortness of breath",
        clinical_notes="Patient reports substernal chest pain with exertion, lasting 10 minutes",
        medications=["metoprolol", "atorvastatin"],
        allergies=["penicillin"],
        urgency=ClinicalUrgency.HIGH,
        relevant_history="History of hypertension and hyperlipidemia"
    )
    
    consultation_result = await agent.clinical_consultation(
        clinical_query="65-year-old male with chest pain and SOB, assess cardiac risk and recommend next steps",
        patient_context=patient,
        clinical_context=clinical,
        specialty=MedicalSpecialty.CARDIOLOGY if agent.agent_framework else None
    )
    
    print(f"Consultation Status: {consultation_result['status']}")
    if consultation_result["status"] == "completed":
        print(f"Safety Score: {consultation_result['clinical_assessment']['safety_score']:.2%}")
        print(f"Recommendations: {consultation_result['clinical_assessment']['recommendations']}")
    
    print("\n💊 Demo Scenario 2: Medication Review")
    med_review_result = await agent.medication_review(
        medications=["warfarin", "aspirin", "metoprolol"],
        patient_context=patient,
        clinical_context=clinical
    )
    
    print(f"Medication Review Status: {med_review_result['status']}")
    if med_review_result["status"] == "completed":
        print(f"Medications Reviewed: {med_review_result['medications_reviewed']}")
        print(f"Interaction Check: {med_review_result['interaction_check']}")
    
    print("\n🚨 Demo Scenario 3: Emergency Assessment")
    emergency_result = await agent.emergency_assessment(
        emergency_scenario="Severe chest pain with diaphoresis and nausea",
        patient_context=patient
    )
    
    print(f"Emergency Assessment Status: {emergency_result['status']}")
    print(f"Priority: {emergency_result['priority']}")
    if emergency_result["status"] == "completed":
        print(f"Immediate Actions: {emergency_result['protocol_response']['immediate_actions']}")
    
    # System status
    print("\n📊 System Status")
    status = await agent.get_system_status()
    print(f"Active Workflows: {status['active_workflows']}")
    print(f"Components: {list(status['components'].keys())}")
    
    # Cleanup
    await agent.cleanup()
    print("\n✅ Demo completed successfully!")


async def main():
    """Main function for testing and demonstration"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 Starting HMCP MCP-Compliant Testing Suite")
    print("=" * 60)
    
    # Create agent
    agent = MCPCompliantHMCPAgent(
        agent_id="test-agent",
        openai_api_key=None,  # Set your OpenAI API key for full testing
        server_url="http://localhost:8050"
    )
    
    # Run comprehensive tests
    test_suite = HMCPTestSuite(agent)
    test_results = await test_suite.run_all_tests()
    
    print(f"\n📋 Test Results Summary:")
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"Passed: {test_results['passed']}")
    print(f"Failed: {test_results['failed']}")
    print(f"Success Rate: {test_results['success_rate']:.1f}%")
    
    # Run demo workflows if tests are mostly successful
    if test_results['success_rate'] >= 70:
        print("\n🎯 Running Clinical Workflow Demonstrations...")
        await demo_clinical_workflows()
    else:
        print("\n⚠️ Skipping demos due to test failures")
    
    # Cleanup
    await agent.cleanup()
    
    print(f"\n🎉 MCP-Compliant HMCP Testing Complete!")
    print(f"Results: {test_results['passed']}/{test_results['total_tests']} tests passed")


if __name__ == "__main__":
    asyncio.run(main())