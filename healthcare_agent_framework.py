"""
Healthcare Agent Framework for Vita Agents
Inspired by Feriq's multi-agent architecture, specialized for healthcare workflows

This framework provides:
- Healthcare-specific agent roles (Diagnostician, Pharmacist, Radiologist, etc.)
- Medical reasoning and clinical decision support
- Patient-centered care coordination
- Evidence-based collaborative workflows
- Compliance and safety monitoring
"""

import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import logging

from llm_integration import llm_manager

logger = logging.getLogger(__name__)

# Healthcare-specific enums
class HealthcareRole(Enum):
    """Healthcare professional roles for agents"""
    DIAGNOSTICIAN = "diagnostician"
    PHARMACIST = "pharmacist"
    RADIOLOGIST = "radiologist"
    PATHOLOGIST = "pathologist"
    CARDIOLOGIST = "cardiologist"
    NEUROLOGIST = "neurologist"
    GENERAL_PRACTITIONER = "general_practitioner"
    NURSE_PRACTITIONER = "nurse_practitioner"
    CARE_COORDINATOR = "care_coordinator"
    QUALITY_MONITOR = "quality_monitor"

class PatientSeverity(Enum):
    """Patient case severity levels"""
    ROUTINE = "routine"
    URGENT = "urgent"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ClinicalTaskType(Enum):
    """Types of clinical tasks"""
    DIAGNOSIS = "diagnosis"
    TREATMENT_PLAN = "treatment_plan"
    MEDICATION_REVIEW = "medication_review"
    IMAGE_INTERPRETATION = "image_interpretation"
    LAB_ANALYSIS = "lab_analysis"
    CARE_COORDINATION = "care_coordination"
    PATIENT_EDUCATION = "patient_education"
    QUALITY_ASSESSMENT = "quality_assessment"

class ReasoningType(Enum):
    """Medical reasoning types"""
    DIFFERENTIAL_DIAGNOSIS = "differential_diagnosis"
    EVIDENCE_BASED = "evidence_based"
    CLINICAL_GUIDELINE = "clinical_guideline"
    RISK_ASSESSMENT = "risk_assessment"
    BENEFIT_HARM_ANALYSIS = "benefit_harm_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"

@dataclass
class PatientContext:
    """Patient information and clinical context"""
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    medical_history: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    vital_signs: Dict[str, Any] = field(default_factory=dict)
    lab_results: Dict[str, Any] = field(default_factory=dict)
    imaging: Dict[str, Any] = field(default_factory=dict)
    severity: PatientSeverity = PatientSeverity.ROUTINE
    additional_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClinicalTask:
    """Clinical task definition"""
    id: str
    task_type: ClinicalTaskType
    patient_context: PatientContext
    description: str
    assigned_agent: Optional[str] = None
    collaborating_agents: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    priority: int = 1  # 1-5, 5 being highest
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    evidence: List[str] = field(default_factory=list)

@dataclass
class HealthcareCapability:
    """Healthcare-specific capability definition"""
    name: str
    specialty: str
    proficiency_level: float  # 0.0 to 1.0
    evidence_sources: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

class HealthcareAgent(BaseModel):
    """Healthcare professional agent"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    role: HealthcareRole
    specialties: List[str] = Field(default_factory=list)
    capabilities: Dict[str, HealthcareCapability] = Field(default_factory=dict)
    current_tasks: List[str] = Field(default_factory=list)
    completed_tasks: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    collaboration_history: List[str] = Field(default_factory=list)
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
    
    def can_handle_task(self, task: ClinicalTask) -> tuple[bool, float]:
        """Determine if agent can handle a task and with what confidence"""
        role_compatibility = self._check_role_compatibility(task)
        capability_match = self._check_capability_match(task)
        workload_capacity = self._check_workload_capacity()
        
        can_handle = role_compatibility and capability_match and workload_capacity
        confidence = self._calculate_confidence(task) if can_handle else 0.0
        
        return can_handle, confidence
    
    def _check_role_compatibility(self, task: ClinicalTask) -> bool:
        """Check if agent's role is compatible with the task"""
        role_task_mapping = {
            HealthcareRole.DIAGNOSTICIAN: [ClinicalTaskType.DIAGNOSIS, ClinicalTaskType.QUALITY_ASSESSMENT],
            HealthcareRole.PHARMACIST: [ClinicalTaskType.MEDICATION_REVIEW, ClinicalTaskType.TREATMENT_PLAN],
            HealthcareRole.RADIOLOGIST: [ClinicalTaskType.IMAGE_INTERPRETATION],
            HealthcareRole.PATHOLOGIST: [ClinicalTaskType.LAB_ANALYSIS],
            HealthcareRole.CARE_COORDINATOR: [ClinicalTaskType.CARE_COORDINATION, ClinicalTaskType.PATIENT_EDUCATION],
            HealthcareRole.GENERAL_PRACTITIONER: list(ClinicalTaskType),  # GPs can handle all task types
        }
        
        compatible_tasks = role_task_mapping.get(self.role, [])
        return task.task_type in compatible_tasks
    
    def _check_capability_match(self, task: ClinicalTask) -> bool:
        """Check if agent has required capabilities for the task"""
        required_capabilities = self._get_required_capabilities(task)
        
        for capability in required_capabilities:
            if capability not in self.capabilities:
                return False
            if self.capabilities[capability].proficiency_level < 0.6:  # Minimum proficiency threshold
                return False
        
        return True
    
    def _check_workload_capacity(self) -> bool:
        """Check if agent has capacity for additional tasks"""
        max_concurrent_tasks = 3  # Configurable limit
        return len(self.current_tasks) < max_concurrent_tasks
    
    def _get_required_capabilities(self, task: ClinicalTask) -> List[str]:
        """Get required capabilities for a task"""
        task_capability_mapping = {
            ClinicalTaskType.DIAGNOSIS: ["clinical_reasoning", "differential_diagnosis", "pattern_recognition"],
            ClinicalTaskType.MEDICATION_REVIEW: ["pharmacology", "drug_interactions", "dosing"],
            ClinicalTaskType.IMAGE_INTERPRETATION: ["radiology", "imaging_analysis", "anatomical_knowledge"],
            ClinicalTaskType.LAB_ANALYSIS: ["laboratory_medicine", "pathology", "biomarkers"],
            ClinicalTaskType.CARE_COORDINATION: ["care_planning", "communication", "workflow_management"],
        }
        
        return task_capability_mapping.get(task.task_type, ["general_medicine"])
    
    def _calculate_confidence(self, task: ClinicalTask) -> float:
        """Calculate confidence score for handling a task"""
        required_capabilities = self._get_required_capabilities(task)
        if not required_capabilities:
            return 0.5
        
        capability_scores = []
        for capability in required_capabilities:
            if capability in self.capabilities:
                capability_scores.append(self.capabilities[capability].proficiency_level)
            else:
                capability_scores.append(0.0)
        
        base_confidence = sum(capability_scores) / len(capability_scores)
        
        # Adjust based on experience and performance
        experience_bonus = min(len(self.completed_tasks) * 0.01, 0.2)  # Max 20% bonus
        performance_score = self.performance_metrics.get("success_rate", 0.5)
        
        final_confidence = min(base_confidence + experience_bonus * performance_score, 1.0)
        return final_confidence

class ClinicalWorkflow:
    """Manages clinical workflows and agent coordination"""
    
    def __init__(self):
        self.agents: Dict[str, HealthcareAgent] = {}
        self.active_tasks: Dict[str, ClinicalTask] = {}
        self.completed_tasks: Dict[str, ClinicalTask] = {}
        self.workflows: Dict[str, Dict[str, Any]] = {}
        
    def register_agent(self, agent: HealthcareAgent):
        """Register a healthcare agent"""
        self.agents[agent.id] = agent
        logger.info(f"Registered {agent.role.value} agent: {agent.name}")
    
    def create_clinical_task(self, task_type: ClinicalTaskType, patient_context: PatientContext, 
                           description: str, priority: int = 1) -> ClinicalTask:
        """Create a new clinical task"""
        task = ClinicalTask(
            id=str(uuid.uuid4()),
            task_type=task_type,
            patient_context=patient_context,
            description=description,
            priority=priority
        )
        
        # Set due date based on severity and priority
        if patient_context.severity == PatientSeverity.EMERGENCY:
            task.due_date = datetime.now() + timedelta(minutes=15)
        elif patient_context.severity == PatientSeverity.CRITICAL:
            task.due_date = datetime.now() + timedelta(hours=1)
        elif patient_context.severity == PatientSeverity.URGENT:
            task.due_date = datetime.now() + timedelta(hours=4)
        else:
            task.due_date = datetime.now() + timedelta(days=1)
        
        self.active_tasks[task.id] = task
        return task
    
    def assign_task(self, task: ClinicalTask) -> Optional[HealthcareAgent]:
        """Assign a task to the most suitable agent"""
        suitable_agents = []
        
        for agent in self.agents.values():
            if not agent.active:
                continue
                
            can_handle, confidence = agent.can_handle_task(task)
            if can_handle:
                suitable_agents.append((agent, confidence))
        
        if not suitable_agents:
            logger.warning(f"No suitable agent found for task {task.id}")
            return None
        
        # Sort by confidence and select the best agent
        suitable_agents.sort(key=lambda x: x[1], reverse=True)
        best_agent = suitable_agents[0][0]
        
        # Assign task
        task.assigned_agent = best_agent.id
        best_agent.current_tasks.append(task.id)
        task.status = "assigned"
        
        logger.info(f"Assigned task {task.id} to {best_agent.name} ({best_agent.role.value})")
        return best_agent
    
    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """Execute a clinical task with the assigned agent"""
        if task_id not in self.active_tasks:
            return {"error": "Task not found"}
        
        task = self.active_tasks[task_id]
        if not task.assigned_agent:
            return {"error": "Task not assigned to any agent"}
        
        agent = self.agents[task.assigned_agent]
        task.status = "in_progress"
        
        try:
            # Get appropriate clinical prompt for the task
            prompt = self._generate_clinical_prompt(task, agent)
            
            # Use LLM to process the clinical task
            result = await llm_manager.generate_response(
                prompt=prompt,
                context=self._build_clinical_context(task),
                temperature=0.3,  # Lower temperature for clinical accuracy
                max_tokens=1000
            )
            
            if "error" in result:
                task.status = "failed"
                return {"error": result["error"]}
            
            # Process and structure the result
            task_result = self._process_clinical_result(result, task, agent)
            task.result = task_result
            task.status = "completed"
            task.confidence_score = task_result.get("confidence", 0.0)
            
            # Update agent metrics
            self._update_agent_performance(agent, task, True)
            
            # Move task to completed
            agent.current_tasks.remove(task_id)
            agent.completed_tasks.append(task_id)
            self.completed_tasks[task_id] = self.active_tasks.pop(task_id)
            
            return task_result
            
        except Exception as e:
            task.status = "failed"
            self._update_agent_performance(agent, task, False)
            logger.error(f"Task execution failed: {e}")
            return {"error": str(e)}
    
    def _generate_clinical_prompt(self, task: ClinicalTask, agent: HealthcareAgent) -> str:
        """Generate appropriate clinical prompt based on task type and agent role"""
        base_context = f"""
You are a {agent.role.value.replace('_', ' ').title()} with expertise in {', '.join(agent.specialties)}.
You are working on the following clinical case:

Patient: {task.patient_context.age}-year-old {task.patient_context.gender}
Chief Complaint: {task.patient_context.chief_complaint}
"""
        
        if task.task_type == ClinicalTaskType.DIAGNOSIS:
            return f"""{base_context}
Medical History: {', '.join(task.patient_context.medical_history) if task.patient_context.medical_history else 'None significant'}
Current Medications: {', '.join(task.patient_context.current_medications) if task.patient_context.current_medications else 'None'}
Vital Signs: {task.patient_context.vital_signs}

Task: {task.description}

Please provide:
1. Differential diagnosis with probability estimates
2. Recommended diagnostic tests
3. Immediate management considerations
4. Red flags or concerning features
5. Confidence level (0-100%)

Format your response as a structured clinical assessment."""
        
        elif task.task_type == ClinicalTaskType.MEDICATION_REVIEW:
            return f"""{base_context}
Current Medications: {', '.join(task.patient_context.current_medications)}
Allergies: {', '.join(task.patient_context.allergies) if task.patient_context.allergies else 'None known'}
Medical History: {', '.join(task.patient_context.medical_history)}

Task: {task.description}

Please analyze:
1. Drug-drug interactions
2. Drug-allergy contraindications
3. Dosing appropriateness
4. Monitoring requirements
5. Alternative recommendations if needed
6. Confidence level (0-100%)

Provide evidence-based pharmaceutical recommendations."""
        
        else:
            return f"""{base_context}
Task: {task.description}

Please provide a comprehensive clinical assessment appropriate for your role as a {agent.role.value.replace('_', ' ')}.
Include your confidence level (0-100%) and evidence-based recommendations."""
    
    def _build_clinical_context(self, task: ClinicalTask) -> str:
        """Build clinical context for LLM processing"""
        context_parts = []
        
        if task.patient_context.lab_results:
            context_parts.append(f"Lab Results: {task.patient_context.lab_results}")
        
        if task.patient_context.imaging:
            context_parts.append(f"Imaging: {task.patient_context.imaging}")
        
        if task.patient_context.additional_context:
            context_parts.append(f"Additional Context: {task.patient_context.additional_context}")
        
        return "\n".join(context_parts)
    
    def _process_clinical_result(self, llm_result: Dict[str, Any], task: ClinicalTask, 
                               agent: HealthcareAgent) -> Dict[str, Any]:
        """Process and structure the clinical result"""
        response_text = llm_result.get("response", "")
        
        # Extract confidence if mentioned in response
        confidence = 0.7  # Default confidence
        if "confidence" in response_text.lower():
            # Try to extract confidence percentage
            import re
            confidence_match = re.search(r'confidence[:\s]*(\d+)%?', response_text.lower())
            if confidence_match:
                confidence = float(confidence_match.group(1)) / 100
        
        return {
            "response": response_text,
            "agent_id": agent.id,
            "agent_role": agent.role.value,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "task_type": task.task_type.value,
            "model_used": llm_manager.active_model,
            "tokens_used": llm_result.get("tokens_used", 0)
        }
    
    def _update_agent_performance(self, agent: HealthcareAgent, task: ClinicalTask, success: bool):
        """Update agent performance metrics"""
        if "total_tasks" not in agent.performance_metrics:
            agent.performance_metrics["total_tasks"] = 0
        if "successful_tasks" not in agent.performance_metrics:
            agent.performance_metrics["successful_tasks"] = 0
        
        agent.performance_metrics["total_tasks"] += 1
        if success:
            agent.performance_metrics["successful_tasks"] += 1
        
        # Calculate success rate
        total = agent.performance_metrics["total_tasks"]
        successful = agent.performance_metrics["successful_tasks"]
        agent.performance_metrics["success_rate"] = successful / total if total > 0 else 0.0
        
        # Update task-specific metrics
        task_type_key = f"{task.task_type.value}_tasks"
        if task_type_key not in agent.performance_metrics:
            agent.performance_metrics[task_type_key] = 0
        agent.performance_metrics[task_type_key] += 1
    
    def get_agent_recommendations(self, task_type: ClinicalTaskType) -> List[HealthcareAgent]:
        """Get recommended agents for a specific task type"""
        recommendations = []
        
        for agent in self.agents.values():
            if not agent.active:
                continue
            
            # Create a dummy task to test compatibility
            dummy_patient = PatientContext(
                patient_id="test",
                age=30,
                gender="unknown",
                chief_complaint="test"
            )
            dummy_task = ClinicalTask(
                id="test",
                task_type=task_type,
                patient_context=dummy_patient,
                description="test"
            )
            
            can_handle, confidence = agent.can_handle_task(dummy_task)
            if can_handle:
                recommendations.append((agent, confidence))
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [agent for agent, _ in recommendations]
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.active]),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "agents_by_role": self._get_agents_by_role(),
            "tasks_by_type": self._get_tasks_by_type(),
            "average_confidence": self._calculate_average_confidence()
        }
    
    def _get_agents_by_role(self) -> Dict[str, int]:
        """Get count of agents by role"""
        role_counts = {}
        for agent in self.agents.values():
            role = agent.role.value
            role_counts[role] = role_counts.get(role, 0) + 1
        return role_counts
    
    def _get_tasks_by_type(self) -> Dict[str, int]:
        """Get count of tasks by type"""
        type_counts = {}
        all_tasks = {**self.active_tasks, **self.completed_tasks}
        for task in all_tasks.values():
            task_type = task.task_type.value
            type_counts[task_type] = type_counts.get(task_type, 0) + 1
        return type_counts
    
    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence across completed tasks"""
        completed_with_confidence = [
            task for task in self.completed_tasks.values() 
            if task.confidence_score > 0
        ]
        
        if not completed_with_confidence:
            return 0.0
        
        total_confidence = sum(task.confidence_score for task in completed_with_confidence)
        return total_confidence / len(completed_with_confidence)

# Global healthcare workflow instance
healthcare_workflow = ClinicalWorkflow()

def create_default_healthcare_agents():
    """Create default healthcare agents with specialized capabilities"""
    
    # Diagnostician Agent
    diagnostician = HealthcareAgent(
        name="Dr. DiagnosisBot",
        role=HealthcareRole.DIAGNOSTICIAN,
        specialties=["Internal Medicine", "Differential Diagnosis", "Clinical Reasoning"],
        capabilities={
            "clinical_reasoning": HealthcareCapability("clinical_reasoning", "Internal Medicine", 0.9),
            "differential_diagnosis": HealthcareCapability("differential_diagnosis", "Diagnostics", 0.95),
            "pattern_recognition": HealthcareCapability("pattern_recognition", "Clinical Assessment", 0.85),
            "risk_assessment": HealthcareCapability("risk_assessment", "Patient Safety", 0.8)
        }
    )
    
    # Pharmacist Agent
    pharmacist = HealthcareAgent(
        name="PharmBot",
        role=HealthcareRole.PHARMACIST,
        specialties=["Clinical Pharmacy", "Drug Interactions", "Pharmacology"],
        capabilities={
            "pharmacology": HealthcareCapability("pharmacology", "Pharmacy", 0.95),
            "drug_interactions": HealthcareCapability("drug_interactions", "Pharmacy", 0.9),
            "dosing": HealthcareCapability("dosing", "Clinical Pharmacy", 0.88),
            "adverse_effects": HealthcareCapability("adverse_effects", "Pharmacovigilance", 0.85)
        }
    )
    
    # Care Coordinator Agent
    coordinator = HealthcareAgent(
        name="CareCoordBot",
        role=HealthcareRole.CARE_COORDINATOR,
        specialties=["Care Management", "Patient Navigation", "Workflow optimization"],
        capabilities={
            "care_planning": HealthcareCapability("care_planning", "Care Management", 0.9),
            "communication": HealthcareCapability("communication", "Patient Relations", 0.85),
            "workflow_management": HealthcareCapability("workflow_management", "Operations", 0.8),
            "patient_education": HealthcareCapability("patient_education", "Health Education", 0.75)
        }
    )
    
    # Register agents
    healthcare_workflow.register_agent(diagnostician)
    healthcare_workflow.register_agent(pharmacist)
    healthcare_workflow.register_agent(coordinator)
    
    return [diagnostician, pharmacist, coordinator]

# Initialize default agents
default_agents = create_default_healthcare_agents()