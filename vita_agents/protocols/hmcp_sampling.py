#!/usr/bin/env python3
"""
Enhanced Bidirectional Sampling for Healthcare Model Context Protocol (HMCP)

Implements comprehensive bidirectional sampling with:
- Clinical workflow sampling contexts
- Agent-to-agent communication
- Healthcare decision trees
- Real-time clinical decision support
- Multi-modal healthcare data sampling
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
import uuid

try:
    # MCP types for sampling
    from mcp.types import (
        SamplingMessage, TextContent, ImageContent, 
        CreateMessageRequest, CreateMessageResult
    )
    from mcp import Server
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Fallback types
    class SamplingMessage: pass
    class TextContent: pass
    class ImageContent: pass
    class CreateMessageRequest: pass
    class CreateMessageResult: pass
    class Server: pass

from vita_agents.protocols.hmcp import (
    ClinicalUrgency, HealthcareRole, PatientContext, ClinicalContext,
    HMCPMessage, HMCPMessageType
)

logger = logging.getLogger(__name__)


class SamplingContext(Enum):
    """Types of sampling contexts for healthcare workflows"""
    CLINICAL_DECISION = "clinical_decision"
    AGENT_HANDOFF = "agent_handoff"
    EMERGENCY_PROTOCOL = "emergency_protocol"
    MEDICATION_REVIEW = "medication_review"
    DIAGNOSTIC_WORKFLOW = "diagnostic_workflow"
    PATIENT_EDUCATION = "patient_education"
    QUALITY_IMPROVEMENT = "quality_improvement"
    RESEARCH_PROTOCOL = "research_protocol"
    CARE_COORDINATION = "care_coordination"
    SYMPTOM_ASSESSMENT = "symptom_assessment"


class SamplingStrategy(Enum):
    """Sampling strategies for different clinical scenarios"""
    DETERMINISTIC = "deterministic"  # Fixed clinical protocols
    PROBABILISTIC = "probabilistic"  # Evidence-based probability sampling
    ADAPTIVE = "adaptive"  # Learning from outcomes
    CONSENSUS = "consensus"  # Multi-agent agreement
    HIERARCHICAL = "hierarchical"  # Escalation-based sampling
    TEMPORAL = "temporal"  # Time-based sampling


@dataclass
class ClinicalSamplingRequest:
    """Request for clinical sampling with healthcare context"""
    context_type: SamplingContext
    strategy: SamplingStrategy
    patient_context: PatientContext
    clinical_context: ClinicalContext
    messages: List[SamplingMessage]
    sampling_parameters: Dict[str, Any] = field(default_factory=dict)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    include_thinking: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ClinicalSamplingResponse:
    """Response from clinical sampling"""
    request_id: str
    context_type: SamplingContext
    strategy: SamplingStrategy
    response_message: SamplingMessage
    confidence_score: float
    clinical_safety_score: float
    reasoning_trace: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DecisionNode:
    """Decision tree node for clinical workflows"""
    node_id: str
    question: str
    criteria: Dict[str, Any]
    children: List['DecisionNode'] = field(default_factory=list)
    action: Optional[str] = None
    confidence: float = 1.0
    clinical_evidence: List[str] = field(default_factory=list)
    safety_checks: List[str] = field(default_factory=list)


class ClinicalDecisionTree:
    """Clinical decision tree for structured sampling"""
    
    def __init__(self, name: str, root_node: DecisionNode):
        self.name = name
        self.root_node = root_node
        self.current_path: List[str] = []
        self.decision_history: List[Dict[str, Any]] = []
    
    async def traverse(self, 
                      patient_data: Dict[str, Any],
                      clinical_context: ClinicalContext) -> Dict[str, Any]:
        """Traverse decision tree based on patient data"""
        
        current_node = self.root_node
        path = []
        decisions = []
        
        while current_node:
            path.append(current_node.node_id)
            
            # Evaluate criteria
            decision_result = await self._evaluate_criteria(
                current_node.criteria,
                patient_data,
                clinical_context
            )
            
            decisions.append({
                "node_id": current_node.node_id,
                "question": current_node.question,
                "evaluation": decision_result,
                "confidence": current_node.confidence
            })
            
            # Find next node
            next_node = None
            if decision_result["meets_criteria"] and current_node.children:
                # Select best matching child
                for child in current_node.children:
                    child_eval = await self._evaluate_criteria(
                        child.criteria,
                        patient_data,
                        clinical_context
                    )
                    if child_eval["meets_criteria"]:
                        next_node = child
                        break
            
            current_node = next_node
        
        return {
            "path": path,
            "decisions": decisions,
            "final_action": current_node.action if current_node else None,
            "confidence": min([d["confidence"] for d in decisions]) if decisions else 0.0
        }
    
    async def _evaluate_criteria(self, 
                                criteria: Dict[str, Any],
                                patient_data: Dict[str, Any],
                                clinical_context: ClinicalContext) -> Dict[str, Any]:
        """Evaluate decision criteria"""
        
        meets_criteria = True
        evaluation_details = []
        
        for key, expected_value in criteria.items():
            actual_value = patient_data.get(key)
            
            if isinstance(expected_value, dict):
                # Range or comparison criteria
                if "min" in expected_value and actual_value is not None:
                    meets = actual_value >= expected_value["min"]
                    meets_criteria = meets_criteria and meets
                    evaluation_details.append({
                        "criteria": f"{key} >= {expected_value['min']}",
                        "actual": actual_value,
                        "met": meets
                    })
                
                if "max" in expected_value and actual_value is not None:
                    meets = actual_value <= expected_value["max"]
                    meets_criteria = meets_criteria and meets
                    evaluation_details.append({
                        "criteria": f"{key} <= {expected_value['max']}",
                        "actual": actual_value,
                        "met": meets
                    })
            else:
                # Direct comparison
                meets = actual_value == expected_value
                meets_criteria = meets_criteria and meets
                evaluation_details.append({
                    "criteria": f"{key} == {expected_value}",
                    "actual": actual_value,
                    "met": meets
                })
        
        return {
            "meets_criteria": meets_criteria,
            "details": evaluation_details
        }


class HMCPBidirectionalSampling:
    """
    Enhanced Bidirectional Sampling for HMCP
    
    Provides comprehensive sampling capabilities for clinical workflows,
    agent communication, and healthcare decision support.
    """
    
    def __init__(self, server: Optional[Server] = None):
        self.server = server
        
        # Sampling state
        self.active_samplings: Dict[str, ClinicalSamplingRequest] = {}
        self.sampling_history: List[ClinicalSamplingResponse] = []
        
        # Decision trees
        self.decision_trees: Dict[str, ClinicalDecisionTree] = {}
        
        # Sampling strategies
        self.strategy_handlers = {
            SamplingStrategy.DETERMINISTIC: self._deterministic_sampling,
            SamplingStrategy.PROBABILISTIC: self._probabilistic_sampling,
            SamplingStrategy.ADAPTIVE: self._adaptive_sampling,
            SamplingStrategy.CONSENSUS: self._consensus_sampling,
            SamplingStrategy.HIERARCHICAL: self._hierarchical_sampling,
            SamplingStrategy.TEMPORAL: self._temporal_sampling
        }
        
        # Initialize default decision trees
        self._initialize_decision_trees()
        
        logger.info("HMCP Bidirectional Sampling initialized")
    
    def _initialize_decision_trees(self):
        """Initialize default clinical decision trees"""
        
        # Chest pain decision tree
        chest_pain_tree = self._create_chest_pain_decision_tree()
        self.decision_trees["chest_pain"] = chest_pain_tree
        
        # Medication interaction tree
        med_interaction_tree = self._create_medication_interaction_tree()
        self.decision_trees["medication_interaction"] = med_interaction_tree
        
        # Emergency triage tree
        emergency_tree = self._create_emergency_triage_tree()
        self.decision_trees["emergency_triage"] = emergency_tree
    
    def _create_chest_pain_decision_tree(self) -> ClinicalDecisionTree:
        """Create chest pain assessment decision tree"""
        
        # High-risk criteria node
        high_risk_node = DecisionNode(
            node_id="chest_pain_high_risk",
            question="Does patient meet high-risk criteria?",
            criteria={
                "troponin_elevated": True,
                "ecg_changes": True,
                "hemodynamic_instability": True
            },
            action="immediate_cardiology_consult",
            confidence=0.95,
            clinical_evidence=["ACS guidelines", "Troponin elevation significant"],
            safety_checks=["Verify troponin levels", "Check ECG interpretation"]
        )
        
        # Moderate risk criteria node
        moderate_risk_node = DecisionNode(
            node_id="chest_pain_moderate_risk",
            question="Does patient have cardiac risk factors?",
            criteria={
                "age": {"min": 50},
                "diabetes": True,
                "hypertension": True,
                "smoking": True
            },
            action="stress_testing_consideration",
            confidence=0.80,
            clinical_evidence=["Framingham risk factors"],
            safety_checks=["Assess exercise tolerance"]
        )
        
        # Root node
        root_node = DecisionNode(
            node_id="chest_pain_assessment",
            question="Patient presents with chest pain",
            criteria={},
            children=[high_risk_node, moderate_risk_node],
            confidence=1.0
        )
        
        return ClinicalDecisionTree("chest_pain_assessment", root_node)
    
    def _create_medication_interaction_tree(self) -> ClinicalDecisionTree:
        """Create medication interaction decision tree"""
        
        # Severe interaction node
        severe_interaction_node = DecisionNode(
            node_id="severe_interaction",
            question="Severe drug interaction detected?",
            criteria={
                "interaction_severity": "severe",
                "contraindicated": True
            },
            action="discontinue_medication",
            confidence=0.98,
            clinical_evidence=["Drug interaction database"],
            safety_checks=["Verify interaction severity", "Check alternatives"]
        )
        
        # Moderate interaction node
        moderate_interaction_node = DecisionNode(
            node_id="moderate_interaction",
            question="Moderate drug interaction with monitoring needed?",
            criteria={
                "interaction_severity": "moderate",
                "monitoring_required": True
            },
            action="increase_monitoring",
            confidence=0.85,
            clinical_evidence=["Clinical pharmacology"],
            safety_checks=["Establish monitoring plan"]
        )
        
        # Root node
        root_node = DecisionNode(
            node_id="medication_interaction_check",
            question="Checking for drug interactions",
            criteria={},
            children=[severe_interaction_node, moderate_interaction_node],
            confidence=1.0
        )
        
        return ClinicalDecisionTree("medication_interaction", root_node)
    
    def _create_emergency_triage_tree(self) -> ClinicalDecisionTree:
        """Create emergency triage decision tree"""
        
        # Critical/Immediate
        critical_node = DecisionNode(
            node_id="triage_critical",
            question="Life-threatening emergency?",
            criteria={
                "respiratory_distress": True,
                "cardiac_arrest": True,
                "severe_trauma": True,
                "altered_mental_status": True
            },
            action="immediate_intervention",
            confidence=0.99,
            clinical_evidence=["Emergency triage protocols"],
            safety_checks=["Verify vital signs", "ABC assessment"]
        )
        
        # Urgent
        urgent_node = DecisionNode(
            node_id="triage_urgent",
            question="Urgent care needed?",
            criteria={
                "pain_scale": {"min": 7},
                "fever": {"min": 101},
                "blood_pressure_high": True
            },
            action="urgent_evaluation",
            confidence=0.85,
            clinical_evidence=["Triage guidelines"],
            safety_checks=["Reassess within 30 minutes"]
        )
        
        # Root node
        root_node = DecisionNode(
            node_id="emergency_triage",
            question="Emergency department triage assessment",
            criteria={},
            children=[critical_node, urgent_node],
            confidence=1.0
        )
        
        return ClinicalDecisionTree("emergency_triage", root_node)
    
    async def create_clinical_sampling(self, 
                                     request: ClinicalSamplingRequest) -> ClinicalSamplingResponse:
        """Create sampling request for clinical context"""
        
        start_time = datetime.now()
        
        # Store active sampling
        self.active_samplings[request.request_id] = request
        
        try:
            # Select strategy handler
            strategy_handler = self.strategy_handlers.get(request.strategy)
            if not strategy_handler:
                raise ValueError(f"Unknown sampling strategy: {request.strategy}")
            
            # Execute sampling
            response = await strategy_handler(request)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            response.processing_time = processing_time
            
            # Store in history
            self.sampling_history.append(response)
            
            # Clean up active sampling
            del self.active_samplings[request.request_id]
            
            logger.info(f"Completed clinical sampling: {request.request_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error in clinical sampling {request.request_id}: {e}")
            
            # Create error response
            error_response = ClinicalSamplingResponse(
                request_id=request.request_id,
                context_type=request.context_type,
                strategy=request.strategy,
                response_message=SamplingMessage(
                    role="assistant",
                    content=TextContent(type="text", text=f"Error in sampling: {str(e)}")
                ),
                confidence_score=0.0,
                clinical_safety_score=0.0,
                warnings=[f"Sampling error: {str(e)}"],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            return error_response
    
    async def _deterministic_sampling(self, 
                                    request: ClinicalSamplingRequest) -> ClinicalSamplingResponse:
        """Deterministic sampling using clinical protocols"""
        
        # Use decision trees for deterministic protocols
        tree_name = request.sampling_parameters.get("decision_tree", "chest_pain")
        
        if tree_name in self.decision_trees:
            decision_tree = self.decision_trees[tree_name]
            
            # Extract patient data from context
            patient_data = {
                "age": request.patient_context.demographics.get("age"),
                "medications": request.clinical_context.medications,
                "allergies": request.clinical_context.allergies,
                **request.sampling_parameters.get("patient_data", {})
            }
            
            # Traverse decision tree
            tree_result = await decision_tree.traverse(patient_data, request.clinical_context)
            
            response_text = f"""
DETERMINISTIC CLINICAL PROTOCOL RESULT:

Decision Tree: {tree_name}
Decision Path: {' -> '.join(tree_result['path'])}
Final Action: {tree_result['final_action']}
Confidence: {tree_result['confidence']:.2f}

CLINICAL REASONING:
{chr(10).join([f"- {d['question']}: {d['evaluation']['meets_criteria']}" for d in tree_result['decisions']])}

RECOMMENDED ACTION:
{tree_result['final_action'] or 'No specific action determined'}
"""
            
            return ClinicalSamplingResponse(
                request_id=request.request_id,
                context_type=request.context_type,
                strategy=request.strategy,
                response_message=SamplingMessage(
                    role="assistant",
                    content=TextContent(type="text", text=response_text)
                ),
                confidence_score=tree_result['confidence'],
                clinical_safety_score=0.95,  # High safety for protocol-based decisions
                reasoning_trace=[d['question'] for d in tree_result['decisions']],
                recommendations=[tree_result['final_action']] if tree_result['final_action'] else [],
                metadata={"decision_tree_result": tree_result}
            )
        
        # Fallback deterministic response
        response_text = "Following standard clinical protocol for deterministic decision making."
        
        return ClinicalSamplingResponse(
            request_id=request.request_id,
            context_type=request.context_type,
            strategy=request.strategy,
            response_message=SamplingMessage(
                role="assistant",
                content=TextContent(type="text", text=response_text)
            ),
            confidence_score=0.8,
            clinical_safety_score=0.9
        )
    
    async def _probabilistic_sampling(self, 
                                    request: ClinicalSamplingRequest) -> ClinicalSamplingResponse:
        """Probabilistic sampling based on clinical evidence"""
        
        # Calculate evidence-based probabilities
        symptoms = request.clinical_context.chief_complaint.lower()
        age = request.patient_context.demographics.get("age", 50)
        
        # Example: Chest pain probability assessment
        if "chest pain" in symptoms:
            # Simplified cardiac risk calculation
            cardiac_risk = 0.1  # Base risk
            if age > 50:
                cardiac_risk += 0.2
            if "diabetes" in str(request.clinical_context.relevant_history).lower():
                cardiac_risk += 0.15
            if any("statin" in med.lower() for med in request.clinical_context.medications):
                cardiac_risk += 0.1
            
            response_text = f"""
PROBABILISTIC CLINICAL ASSESSMENT:

Chief Complaint: {request.clinical_context.chief_complaint}
Estimated Cardiac Risk: {cardiac_risk:.2%}

RISK FACTORS CONSIDERED:
- Age: {age} years
- Medical History: {request.clinical_context.relevant_history}
- Current Medications: {', '.join(request.clinical_context.medications)}

PROBABILISTIC RECOMMENDATIONS:
- If risk > 30%: Consider cardiology consultation
- If risk > 20%: Obtain ECG and troponins
- If risk < 10%: Consider discharge with follow-up

Current Risk Level: {cardiac_risk:.2%}
"""
            
            return ClinicalSamplingResponse(
                request_id=request.request_id,
                context_type=request.context_type,
                strategy=request.strategy,
                response_message=SamplingMessage(
                    role="assistant",
                    content=TextContent(type="text", text=response_text)
                ),
                confidence_score=0.85,
                clinical_safety_score=0.80,
                recommendations=[
                    f"Cardiac risk assessment: {cardiac_risk:.2%}",
                    "Consider evidence-based risk stratification"
                ],
                metadata={"calculated_risk": cardiac_risk}
            )
        
        # Default probabilistic response
        response_text = "Applying evidence-based probabilistic assessment to clinical scenario."
        
        return ClinicalSamplingResponse(
            request_id=request.request_id,
            context_type=request.context_type,
            strategy=request.strategy,
            response_message=SamplingMessage(
                role="assistant",
                content=TextContent(type="text", text=response_text)
            ),
            confidence_score=0.75,
            clinical_safety_score=0.80
        )
    
    async def _adaptive_sampling(self, 
                               request: ClinicalSamplingRequest) -> ClinicalSamplingResponse:
        """Adaptive sampling that learns from outcomes"""
        
        # Analyze historical outcomes for similar cases
        similar_cases = [
            resp for resp in self.sampling_history[-50:]  # Last 50 cases
            if resp.context_type == request.context_type
        ]
        
        if similar_cases:
            avg_confidence = sum(case.confidence_score for case in similar_cases) / len(similar_cases)
            avg_safety = sum(case.clinical_safety_score for case in similar_cases) / len(similar_cases)
            
            response_text = f"""
ADAPTIVE CLINICAL LEARNING:

Similar Cases Analyzed: {len(similar_cases)}
Historical Average Confidence: {avg_confidence:.2%}
Historical Average Safety Score: {avg_safety:.2%}

ADAPTIVE RECOMMENDATIONS:
Based on analysis of similar {request.context_type.value} cases, the system recommends:
- Maintaining confidence threshold above {avg_confidence:.2%}
- Ensuring safety measures meet {avg_safety:.2%} standard
- Incorporating lessons learned from previous outcomes

CONTEXT-SPECIFIC INSIGHTS:
Learning from {len(similar_cases)} similar cases to optimize decision-making for current scenario.
"""
            
            return ClinicalSamplingResponse(
                request_id=request.request_id,
                context_type=request.context_type,
                strategy=request.strategy,
                response_message=SamplingMessage(
                    role="assistant",
                    content=TextContent(type="text", text=response_text)
                ),
                confidence_score=min(avg_confidence + 0.1, 1.0),
                clinical_safety_score=min(avg_safety + 0.05, 1.0),
                recommendations=[
                    f"Learned from {len(similar_cases)} similar cases",
                    f"Target confidence: {avg_confidence:.2%}",
                    f"Target safety: {avg_safety:.2%}"
                ],
                metadata={
                    "historical_cases": len(similar_cases),
                    "avg_confidence": avg_confidence,
                    "avg_safety": avg_safety
                }
            )
        
        # No historical data available
        response_text = "Adaptive learning mode: Insufficient historical data for this context type."
        
        return ClinicalSamplingResponse(
            request_id=request.request_id,
            context_type=request.context_type,
            strategy=request.strategy,
            response_message=SamplingMessage(
                role="assistant",
                content=TextContent(type="text", text=response_text)
            ),
            confidence_score=0.7,
            clinical_safety_score=0.85
        )
    
    async def _consensus_sampling(self, 
                                request: ClinicalSamplingRequest) -> ClinicalSamplingResponse:
        """Consensus sampling using multi-agent agreement"""
        
        # Simulate multiple agent perspectives
        agent_opinions = [
            {"agent": "primary_care", "recommendation": "Conservative management", "confidence": 0.8},
            {"agent": "specialist", "recommendation": "Specialist consultation", "confidence": 0.9},
            {"agent": "emergency", "recommendation": "Rule out emergent causes", "confidence": 0.85}
        ]
        
        # Calculate consensus
        avg_confidence = sum(opinion["confidence"] for opinion in agent_opinions) / len(agent_opinions)
        
        response_text = f"""
MULTI-AGENT CONSENSUS ASSESSMENT:

AGENT PERSPECTIVES:
{chr(10).join([f"- {op['agent'].title()}: {op['recommendation']} (confidence: {op['confidence']:.2%})" for op in agent_opinions])}

CONSENSUS ANALYSIS:
Average Confidence: {avg_confidence:.2%}
Agreement Level: {"High" if all(abs(op["confidence"] - avg_confidence) < 0.1 for op in agent_opinions) else "Moderate"}

CONSENSUS RECOMMENDATION:
Based on multi-agent analysis, the consensus approach combines conservative management
with appropriate specialist input and emergency rule-out protocols.
"""
        
        return ClinicalSamplingResponse(
            request_id=request.request_id,
            context_type=request.context_type,
            strategy=request.strategy,
            response_message=SamplingMessage(
                role="assistant",
                content=TextContent(type="text", text=response_text)
            ),
            confidence_score=avg_confidence,
            clinical_safety_score=0.90,  # High safety from consensus
            recommendations=[op["recommendation"] for op in agent_opinions],
            metadata={"agent_opinions": agent_opinions}
        )
    
    async def _hierarchical_sampling(self, 
                                   request: ClinicalSamplingRequest) -> ClinicalSamplingResponse:
        """Hierarchical sampling with escalation protocols"""
        
        urgency_level = request.clinical_context.urgency
        
        # Determine escalation level
        if urgency_level == ClinicalUrgency.CRITICAL:
            level = "LEVEL 1 - CRITICAL"
            action = "Immediate physician notification and intervention"
            confidence = 0.95
        elif urgency_level == ClinicalUrgency.HIGH:
            level = "LEVEL 2 - HIGH PRIORITY"
            action = "Expedited evaluation within 30 minutes"
            confidence = 0.90
        elif urgency_level == ClinicalUrgency.MEDIUM:
            level = "LEVEL 3 - ROUTINE URGENT"
            action = "Standard evaluation within 2 hours"
            confidence = 0.85
        else:
            level = "LEVEL 4 - ROUTINE"
            action = "Routine scheduling and evaluation"
            confidence = 0.80
        
        response_text = f"""
HIERARCHICAL ESCALATION PROTOCOL:

Clinical Urgency: {urgency_level.value}
Escalation Level: {level}

PROTOCOL ACTION:
{action}

ESCALATION CRITERIA MET:
- Chief Complaint: {request.clinical_context.chief_complaint}
- Clinical Urgency: {urgency_level.value}
- Patient Context: {request.patient_context.patient_id}

NEXT STEPS:
1. Execute {action.lower()}
2. Document escalation reasoning
3. Monitor for changes in clinical status
4. Follow institutional escalation protocols
"""
        
        return ClinicalSamplingResponse(
            request_id=request.request_id,
            context_type=request.context_type,
            strategy=request.strategy,
            response_message=SamplingMessage(
                role="assistant",
                content=TextContent(type="text", text=response_text)
            ),
            confidence_score=confidence,
            clinical_safety_score=0.95,  # High safety from structured escalation
            recommendations=[action],
            next_steps=[
                "Execute escalation protocol",
                "Document decision reasoning",
                "Monitor patient status"
            ],
            metadata={"escalation_level": level, "urgency": urgency_level.value}
        )
    
    async def _temporal_sampling(self, 
                               request: ClinicalSamplingRequest) -> ClinicalSamplingResponse:
        """Temporal sampling considering time-based factors"""
        
        current_time = datetime.now()
        time_factors = []
        
        # Time of day considerations
        hour = current_time.hour
        if 0 <= hour < 6:
            time_factors.append("Night shift - limited resources")
        elif 6 <= hour < 18:
            time_factors.append("Day shift - full resources available")
        else:
            time_factors.append("Evening shift - reduced specialist availability")
        
        # Day of week considerations
        weekday = current_time.weekday()
        if weekday >= 5:  # Weekend
            time_factors.append("Weekend - limited specialist availability")
        else:
            time_factors.append("Weekday - standard specialist availability")
        
        # Seasonal considerations (simplified)
        month = current_time.month
        if month in [12, 1, 2]:
            time_factors.append("Winter - increased respiratory illness risk")
        elif month in [6, 7, 8]:
            time_factors.append("Summer - increased heat-related conditions")
        
        response_text = f"""
TEMPORAL CLINICAL ASSESSMENT:

Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
Day of Week: {current_time.strftime('%A')}

TEMPORAL FACTORS:
{chr(10).join([f"- {factor}" for factor in time_factors])}

TIME-BASED RECOMMENDATIONS:
Based on current temporal context, consider:
1. Resource availability adjustments
2. Escalation pathway modifications
3. Timing-sensitive interventions
4. Shift-specific protocols

TEMPORAL PRIORITY ADJUSTMENTS:
Current time factors may influence clinical decision-making and resource allocation.
"""
        
        return ClinicalSamplingResponse(
            request_id=request.request_id,
            context_type=request.context_type,
            strategy=request.strategy,
            response_message=SamplingMessage(
                role="assistant",
                content=TextContent(type="text", text=response_text)
            ),
            confidence_score=0.85,
            clinical_safety_score=0.85,
            recommendations=[
                "Consider temporal factors in decision making",
                "Adjust for current resource availability",
                "Account for time-sensitive interventions"
            ],
            metadata={
                "temporal_factors": time_factors,
                "current_time": current_time.isoformat(),
                "hour": hour,
                "weekday": weekday
            }
        )
    
    async def get_sampling_analytics(self) -> Dict[str, Any]:
        """Get analytics on sampling performance"""
        
        if not self.sampling_history:
            return {"message": "No sampling history available"}
        
        total_samples = len(self.sampling_history)
        
        # Confidence statistics
        confidences = [resp.confidence_score for resp in self.sampling_history]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Safety statistics
        safety_scores = [resp.clinical_safety_score for resp in self.sampling_history]
        avg_safety = sum(safety_scores) / len(safety_scores)
        
        # Context type distribution
        context_counts = {}
        for resp in self.sampling_history:
            context_type = resp.context_type.value
            context_counts[context_type] = context_counts.get(context_type, 0) + 1
        
        # Strategy distribution
        strategy_counts = {}
        for resp in self.sampling_history:
            strategy = resp.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Processing time statistics
        processing_times = [resp.processing_time for resp in self.sampling_history]
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        return {
            "total_samples": total_samples,
            "average_confidence": avg_confidence,
            "average_safety_score": avg_safety,
            "average_processing_time": avg_processing_time,
            "context_distribution": context_counts,
            "strategy_distribution": strategy_counts,
            "active_samplings": len(self.active_samplings),
            "decision_trees_available": len(self.decision_trees)
        }
    
    def get_sampling_status(self) -> Dict[str, Any]:
        """Get current sampling system status"""
        return {
            "system_ready": True,
            "active_samplings": len(self.active_samplings),
            "total_completed": len(self.sampling_history),
            "decision_trees": list(self.decision_trees.keys()),
            "available_strategies": [strategy.value for strategy in SamplingStrategy],
            "available_contexts": [context.value for context in SamplingContext]
        }


# Example usage and helper functions
async def create_sampling_example():
    """Example of using HMCP Bidirectional Sampling"""
    
    # Initialize sampling system
    sampling_system = HMCPBidirectionalSampling()
    
    # Example patient context
    patient_context = PatientContext(
        patient_id="PT12345",
        mrn="MRN-12345",
        demographics={"age": 65, "gender": "male"}
    )
    
    # Example clinical context
    clinical_context = ClinicalContext(
        chief_complaint="Chest pain and shortness of breath",
        clinical_notes="Patient reports substernal chest pain with exertion",
        medications=["metoprolol", "atorvastatin"],
        allergies=["penicillin"],
        urgency=ClinicalUrgency.HIGH,
        relevant_history="History of hypertension and hyperlipidemia"
    )
    
    # Create sampling request
    sampling_request = ClinicalSamplingRequest(
        context_type=SamplingContext.CLINICAL_DECISION,
        strategy=SamplingStrategy.DETERMINISTIC,
        patient_context=patient_context,
        clinical_context=clinical_context,
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text="65-year-old male with chest pain and SOB. Risk factors include HTN and hyperlipidemia."
                )
            )
        ],
        sampling_parameters={
            "decision_tree": "chest_pain",
            "patient_data": {
                "troponin_elevated": True,
                "ecg_changes": False,
                "hemodynamic_instability": False
            }
        }
    )
    
    # Execute sampling
    response = await sampling_system.create_clinical_sampling(sampling_request)
    
    print(f"Sampling Response:")
    print(f"Context: {response.context_type.value}")
    print(f"Strategy: {response.strategy.value}")
    print(f"Confidence: {response.confidence_score:.2%}")
    print(f"Safety Score: {response.clinical_safety_score:.2%}")
    print(f"Response: {response.response_message.content.text if hasattr(response.response_message.content, 'text') else response.response_message.content}")
    
    # Get analytics
    analytics = await sampling_system.get_sampling_analytics()
    print(f"\nSampling Analytics: {analytics}")


if __name__ == "__main__":
    # Run example
    asyncio.run(create_sampling_example())