"""
Autonomous Clinical Workflows for AI-Driven Healthcare Operations.

This module provides intelligent clinical pathway automation, resource optimization,
automated scheduling, clinical decision automation, and real-time care coordination.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
import structlog
from pydantic import BaseModel, Field
import uuid
from collections import defaultdict, deque

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = structlog.get_logger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ResourceType(Enum):
    """Healthcare resource types."""
    PHYSICIAN = "physician"
    NURSE = "nurse"
    SPECIALIST = "specialist"
    OPERATING_ROOM = "operating_room"
    ICU_BED = "icu_bed"
    REGULAR_BED = "regular_bed"
    EQUIPMENT = "equipment"
    LABORATORY = "laboratory"
    IMAGING = "imaging"
    PHARMACY = "pharmacy"


class Priority(Enum):
    """Task and workflow priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    EMERGENCY = 5


class ClinicalPathwayType(Enum):
    """Types of clinical pathways."""
    EMERGENCY_DEPARTMENT = "emergency_department"
    SURGICAL_PATHWAY = "surgical_pathway"
    MEDICATION_MANAGEMENT = "medication_management"
    DISCHARGE_PLANNING = "discharge_planning"
    CHRONIC_CARE_MANAGEMENT = "chronic_care_management"
    DIAGNOSTIC_WORKUP = "diagnostic_workup"
    INFECTION_CONTROL = "infection_control"
    QUALITY_IMPROVEMENT = "quality_improvement"


@dataclass
class ClinicalTask:
    """Represents a clinical task within a workflow."""
    task_id: str
    name: str
    description: str
    task_type: str
    priority: Priority
    estimated_duration: timedelta
    required_resources: List[ResourceType]
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    assigned_to: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_start: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Resource:
    """Represents a healthcare resource."""
    resource_id: str
    resource_type: ResourceType
    name: str
    capacity: int
    current_utilization: int = 0
    location: Optional[str] = None
    specialties: List[str] = field(default_factory=list)
    availability_schedule: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def utilization_rate(self) -> float:
        """Calculate current utilization rate."""
        return self.current_utilization / self.capacity if self.capacity > 0 else 0.0
    
    @property
    def available_capacity(self) -> int:
        """Get available capacity."""
        return max(0, self.capacity - self.current_utilization)


@dataclass
class ClinicalWorkflow:
    """Represents a clinical workflow/pathway."""
    workflow_id: str
    name: str
    pathway_type: ClinicalPathwayType
    patient_id: str
    tasks: List[ClinicalTask]
    status: WorkflowStatus = WorkflowStatus.PENDING
    priority: Priority = Priority.NORMAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowRequest(BaseModel):
    """Request to create or execute a workflow."""
    
    patient_id: str
    pathway_type: ClinicalPathwayType
    priority: Priority = Priority.NORMAL
    clinical_context: Dict[str, Any] = Field(default_factory=dict)
    patient_data: Dict[str, Any] = Field(default_factory=dict)
    resource_preferences: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    request_timestamp: datetime = Field(default_factory=datetime.utcnow)


class WorkflowResponse(BaseModel):
    """Response from workflow execution."""
    
    workflow_id: str
    status: WorkflowStatus
    estimated_completion: Optional[datetime] = None
    scheduled_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    resource_assignments: Dict[str, str] = Field(default_factory=dict)
    next_steps: List[str] = Field(default_factory=list)
    alerts: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseWorkflowEngine(ABC):
    """Base class for workflow execution engines."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = structlog.get_logger(__name__)
    
    @abstractmethod
    async def execute_workflow(
        self, 
        workflow: ClinicalWorkflow,
        resources: List[Resource]
    ) -> WorkflowResponse:
        """Execute a clinical workflow."""
        pass
    
    @abstractmethod
    async def optimize_schedule(
        self, 
        workflows: List[ClinicalWorkflow],
        resources: List[Resource]
    ) -> Dict[str, Any]:
        """Optimize workflow scheduling."""
        pass


class EmergencyDepartmentWorkflowEngine(BaseWorkflowEngine):
    """Workflow engine for emergency department operations."""
    
    def __init__(self):
        super().__init__("emergency_department")
        
        # ED-specific workflow templates
        self.ed_pathways = {
            'chest_pain': [
                ('triage_assessment', 5, [ResourceType.NURSE]),
                ('ecg_acquisition', 10, [ResourceType.NURSE, ResourceType.EQUIPMENT]),
                ('blood_draw', 5, [ResourceType.NURSE]),
                ('physician_evaluation', 15, [ResourceType.PHYSICIAN]),
                ('cardiac_enzymes', 30, [ResourceType.LABORATORY]),
                ('disposition_decision', 10, [ResourceType.PHYSICIAN])
            ],
            'trauma': [
                ('primary_survey', 5, [ResourceType.PHYSICIAN, ResourceType.NURSE]),
                ('imaging_trauma_pan_scan', 20, [ResourceType.IMAGING]),
                ('blood_type_crossmatch', 15, [ResourceType.LABORATORY]),
                ('secondary_survey', 15, [ResourceType.PHYSICIAN]),
                ('specialist_consultation', 30, [ResourceType.SPECIALIST]),
                ('disposition_decision', 10, [ResourceType.PHYSICIAN])
            ],
            'sepsis_alert': [
                ('sepsis_screening', 5, [ResourceType.NURSE]),
                ('blood_cultures', 10, [ResourceType.NURSE]),
                ('lactate_measurement', 5, [ResourceType.LABORATORY]),
                ('antibiotic_administration', 15, [ResourceType.NURSE, ResourceType.PHARMACY]),
                ('fluid_resuscitation', 30, [ResourceType.NURSE]),
                ('icu_consultation', 20, [ResourceType.SPECIALIST])
            ]
        }
    
    async def execute_workflow(
        self, 
        workflow: ClinicalWorkflow,
        resources: List[Resource]
    ) -> WorkflowResponse:
        """Execute emergency department workflow."""
        
        try:
            # Determine pathway based on clinical context
            pathway_key = self._determine_ed_pathway(workflow)
            
            # Generate tasks if not already created
            if not workflow.tasks:
                workflow.tasks = self._generate_ed_tasks(pathway_key, workflow.workflow_id)
            
            # Prioritize based on acuity
            workflow.priority = self._calculate_ed_priority(workflow)
            
            # Schedule and assign resources
            schedule_result = await self._schedule_ed_workflow(workflow, resources)
            
            # Execute tasks in order
            execution_result = await self._execute_ed_tasks(workflow, resources)
            
            # Generate response
            return WorkflowResponse(
                workflow_id=workflow.workflow_id,
                status=workflow.status,
                estimated_completion=workflow.estimated_completion,
                scheduled_tasks=schedule_result['scheduled_tasks'],
                resource_assignments=schedule_result['resource_assignments'],
                next_steps=execution_result['next_steps'],
                alerts=execution_result['alerts'],
                performance_metrics=self._calculate_ed_metrics(workflow),
                metadata={
                    'pathway_used': pathway_key,
                    'acuity_level': workflow.priority.value,
                    'resource_utilization': schedule_result.get('utilization_metrics', {})
                }
            )
            
        except Exception as e:
            self.logger.error(f"ED workflow execution failed: {e}")
            raise
    
    def _determine_ed_pathway(self, workflow: ClinicalWorkflow) -> str:
        """Determine appropriate ED pathway based on clinical context."""
        
        clinical_context = workflow.metadata.get('clinical_context', {})
        
        # Chief complaint analysis
        chief_complaint = clinical_context.get('chief_complaint', '').lower()
        
        if any(keyword in chief_complaint for keyword in ['chest pain', 'chest pressure', 'heart']):
            return 'chest_pain'
        elif any(keyword in chief_complaint for keyword in ['trauma', 'accident', 'fall', 'injury']):
            return 'trauma'
        elif any(keyword in chief_complaint for keyword in ['fever', 'infection', 'sepsis', 'shock']):
            return 'sepsis_alert'
        else:
            return 'general_assessment'  # Default pathway
    
    def _generate_ed_tasks(self, pathway_key: str, workflow_id: str) -> List[ClinicalTask]:
        """Generate ED tasks based on pathway."""
        
        pathway = self.ed_pathways.get(pathway_key, [])
        tasks = []
        
        for i, (task_name, duration_mins, required_resources) in enumerate(pathway):
            task = ClinicalTask(
                task_id=f"{workflow_id}_task_{i+1}",
                name=task_name.replace('_', ' ').title(),
                description=f"ED {task_name} for pathway {pathway_key}",
                task_type=task_name,
                priority=Priority.HIGH if 'trauma' in pathway_key or 'sepsis' in pathway_key else Priority.NORMAL,
                estimated_duration=timedelta(minutes=duration_mins),
                required_resources=required_resources,
                dependencies=[f"{workflow_id}_task_{i}"] if i > 0 else []
            )
            tasks.append(task)
        
        return tasks
    
    def _calculate_ed_priority(self, workflow: ClinicalWorkflow) -> Priority:
        """Calculate ED priority based on clinical indicators."""
        
        clinical_context = workflow.metadata.get('clinical_context', {})
        
        # ESI (Emergency Severity Index) simulation
        vital_signs = clinical_context.get('vital_signs', {})
        
        # High priority indicators
        if (vital_signs.get('systolic_bp', 120) < 90 or 
            vital_signs.get('heart_rate', 80) > 120 or
            vital_signs.get('respiratory_rate', 16) > 24 or
            vital_signs.get('oxygen_saturation', 98) < 92):
            return Priority.EMERGENCY
        
        # Check for time-sensitive conditions
        chief_complaint = clinical_context.get('chief_complaint', '').lower()
        if any(keyword in chief_complaint for keyword in ['chest pain', 'stroke', 'trauma', 'sepsis']):
            return Priority.URGENT
        
        return Priority.HIGH  # Default ED priority
    
    async def _schedule_ed_workflow(
        self, 
        workflow: ClinicalWorkflow, 
        resources: List[Resource]
    ) -> Dict[str, Any]:
        """Schedule ED workflow tasks."""
        
        scheduled_tasks = []
        resource_assignments = {}
        current_time = datetime.utcnow()
        
        for task in workflow.tasks:
            # Find available resources
            assigned_resources = self._assign_resources(task, resources, current_time)
            
            if assigned_resources:
                task.scheduled_start = current_time
                task.assigned_to = ', '.join(assigned_resources.keys())
                resource_assignments[task.task_id] = assigned_resources
                
                scheduled_tasks.append({
                    'task_id': task.task_id,
                    'name': task.name,
                    'scheduled_start': task.scheduled_start,
                    'estimated_duration': task.estimated_duration.total_seconds() / 60,
                    'assigned_resources': list(assigned_resources.keys())
                })
                
                # Update current time for next task
                current_time = task.scheduled_start + task.estimated_duration
        
        workflow.estimated_completion = current_time
        
        return {
            'scheduled_tasks': scheduled_tasks,
            'resource_assignments': resource_assignments,
            'utilization_metrics': self._calculate_resource_utilization(resources)
        }
    
    def _assign_resources(
        self, 
        task: ClinicalTask, 
        resources: List[Resource], 
        start_time: datetime
    ) -> Dict[str, str]:
        """Assign resources to a task."""
        
        assignments = {}
        
        for resource_type in task.required_resources:
            # Find available resource of required type
            available_resources = [
                r for r in resources 
                if r.resource_type == resource_type and r.available_capacity > 0
            ]
            
            if available_resources:
                # Choose least utilized resource
                best_resource = min(available_resources, key=lambda r: r.utilization_rate)
                assignments[best_resource.resource_id] = best_resource.name
                
                # Update resource utilization
                best_resource.current_utilization += 1
        
        return assignments
    
    async def _execute_ed_tasks(
        self, 
        workflow: ClinicalWorkflow, 
        resources: List[Resource]
    ) -> Dict[str, Any]:
        """Execute ED tasks and monitor progress."""
        
        next_steps = []
        alerts = []
        
        # Check for delays or issues
        current_time = datetime.utcnow()
        
        for task in workflow.tasks:
            if task.scheduled_start and task.scheduled_start < current_time:
                if task.status == WorkflowStatus.PENDING:
                    alerts.append(f"Task '{task.name}' is overdue and should have started")
                    next_steps.append(f"Prioritize completion of {task.name}")
        
        # Generate next steps based on workflow progress
        pending_tasks = [t for t in workflow.tasks if t.status == WorkflowStatus.PENDING]
        if pending_tasks:
            next_task = pending_tasks[0]
            next_steps.append(f"Next: {next_task.name} (estimated {next_task.estimated_duration.total_seconds()/60} minutes)")
        
        # Check for critical pathways
        if workflow.priority == Priority.EMERGENCY:
            alerts.append("EMERGENCY workflow - expedite all tasks")
        
        return {
            'next_steps': next_steps,
            'alerts': alerts
        }
    
    def _calculate_ed_metrics(self, workflow: ClinicalWorkflow) -> Dict[str, Any]:
        """Calculate ED-specific performance metrics."""
        
        metrics = {}
        
        # Door-to-decision time
        if workflow.started_at and workflow.estimated_completion:
            total_time = workflow.estimated_completion - workflow.started_at
            metrics['estimated_door_to_decision_minutes'] = total_time.total_seconds() / 60
        
        # Task efficiency
        completed_tasks = [t for t in workflow.tasks if t.status == WorkflowStatus.COMPLETED]
        metrics['task_completion_rate'] = len(completed_tasks) / len(workflow.tasks) if workflow.tasks else 0
        
        # Resource requirements
        metrics['total_resources_required'] = sum(len(t.required_resources) for t in workflow.tasks)
        
        return metrics
    
    async def optimize_schedule(
        self, 
        workflows: List[ClinicalWorkflow], 
        resources: List[Resource]
    ) -> Dict[str, Any]:
        """Optimize ED scheduling across multiple workflows."""
        
        # Sort workflows by priority and arrival time
        sorted_workflows = sorted(
            workflows, 
            key=lambda w: (w.priority.value, w.created_at), 
            reverse=True
        )
        
        # Calculate optimal resource allocation
        optimization_result = {
            'optimized_schedule': [],
            'resource_utilization': {},
            'estimated_wait_times': {},
            'recommendations': []
        }
        
        current_time = datetime.utcnow()
        
        for workflow in sorted_workflows:
            # Estimate wait time based on current load
            wait_time = self._estimate_wait_time(workflow, resources, current_time)
            optimization_result['estimated_wait_times'][workflow.workflow_id] = wait_time
            
            # Add to optimized schedule
            optimization_result['optimized_schedule'].append({
                'workflow_id': workflow.workflow_id,
                'patient_id': workflow.patient_id,
                'priority': workflow.priority.value,
                'estimated_start': current_time + wait_time,
                'estimated_duration': sum(
                    (t.estimated_duration.total_seconds() / 60 for t in workflow.tasks), 0
                )
            })
            
            # Update current time
            workflow_duration = sum(t.estimated_duration for t in workflow.tasks, timedelta())
            current_time += wait_time + workflow_duration
        
        # Calculate resource utilization
        for resource in resources:
            optimization_result['resource_utilization'][resource.resource_id] = {
                'utilization_rate': resource.utilization_rate,
                'capacity': resource.capacity,
                'current_load': resource.current_utilization
            }
        
        # Generate recommendations
        optimization_result['recommendations'] = self._generate_ed_recommendations(
            workflows, resources, optimization_result
        )
        
        return optimization_result
    
    def _estimate_wait_time(
        self, 
        workflow: ClinicalWorkflow, 
        resources: List[Resource], 
        current_time: datetime
    ) -> timedelta:
        """Estimate wait time for workflow based on current resource load."""
        
        # Simplified wait time calculation
        required_resource_types = set()
        for task in workflow.tasks:
            required_resource_types.update(task.required_resources)
        
        max_wait = timedelta()
        for resource_type in required_resource_types:
            type_resources = [r for r in resources if r.resource_type == resource_type]
            if type_resources:
                avg_utilization = sum(r.utilization_rate for r in type_resources) / len(type_resources)
                # Higher utilization = longer wait
                wait_minutes = int(avg_utilization * 30)  # Max 30 minute wait
                max_wait = max(max_wait, timedelta(minutes=wait_minutes))
        
        return max_wait
    
    def _generate_ed_recommendations(
        self, 
        workflows: List[ClinicalWorkflow], 
        resources: List[Resource], 
        optimization_result: Dict[str, Any]
    ) -> List[str]:
        """Generate ED optimization recommendations."""
        
        recommendations = []
        
        # Resource utilization recommendations
        high_util_resources = [
            r for r in resources if r.utilization_rate > 0.8
        ]
        if high_util_resources:
            resource_names = [r.name for r in high_util_resources]
            recommendations.append(f"High utilization detected: {', '.join(resource_names)} - consider additional staffing")
        
        # Wait time recommendations
        long_wait_workflows = [
            wf_id for wf_id, wait_time in optimization_result['estimated_wait_times'].items()
            if wait_time.total_seconds() > 1800  # > 30 minutes
        ]
        if long_wait_workflows:
            recommendations.append(f"Extended wait times for {len(long_wait_workflows)} workflows - consider process optimization")
        
        # Priority recommendations
        emergency_workflows = [w for w in workflows if w.priority == Priority.EMERGENCY]
        if emergency_workflows:
            recommendations.append(f"CRITICAL: {len(emergency_workflows)} emergency workflows require immediate attention")
        
        return recommendations
    
    def _calculate_resource_utilization(self, resources: List[Resource]) -> Dict[str, float]:
        """Calculate resource utilization metrics."""
        
        utilization = {}
        for resource in resources:
            utilization[resource.resource_id] = resource.utilization_rate
        
        return utilization


class SurgicalWorkflowEngine(BaseWorkflowEngine):
    """Workflow engine for surgical operations."""
    
    def __init__(self):
        super().__init__("surgical")
        
        # Surgical pathway templates
        self.surgical_pathways = {
            'general_surgery': [
                ('pre_op_assessment', 30, [ResourceType.NURSE, ResourceType.PHYSICIAN]),
                ('anesthesia_consultation', 20, [ResourceType.SPECIALIST]),
                ('or_setup', 15, [ResourceType.NURSE, ResourceType.EQUIPMENT]),
                ('surgical_procedure', 120, [ResourceType.PHYSICIAN, ResourceType.NURSE, ResourceType.OPERATING_ROOM]),
                ('post_op_recovery', 60, [ResourceType.NURSE]),
                ('discharge_planning', 15, [ResourceType.NURSE])
            ],
            'cardiac_surgery': [
                ('pre_op_optimization', 60, [ResourceType.SPECIALIST, ResourceType.NURSE]),
                ('anesthesia_induction', 30, [ResourceType.SPECIALIST]),
                ('surgical_approach', 180, [ResourceType.SPECIALIST, ResourceType.NURSE, ResourceType.OPERATING_ROOM]),
                ('post_op_icu', 240, [ResourceType.ICU_BED, ResourceType.NURSE, ResourceType.SPECIALIST]),
                ('step_down_care', 1440, [ResourceType.REGULAR_BED, ResourceType.NURSE])  # 24 hours
            ],
            'emergency_surgery': [
                ('trauma_assessment', 10, [ResourceType.PHYSICIAN, ResourceType.NURSE]),
                ('emergency_or_prep', 10, [ResourceType.NURSE, ResourceType.OPERATING_ROOM]),
                ('emergency_procedure', 90, [ResourceType.PHYSICIAN, ResourceType.NURSE, ResourceType.OPERATING_ROOM]),
                ('critical_care', 120, [ResourceType.ICU_BED, ResourceType.SPECIALIST, ResourceType.NURSE])
            ]
        }
    
    async def execute_workflow(
        self, 
        workflow: ClinicalWorkflow, 
        resources: List[Resource]
    ) -> WorkflowResponse:
        """Execute surgical workflow."""
        
        try:
            # Determine surgical pathway
            pathway_key = self._determine_surgical_pathway(workflow)
            
            # Generate surgical tasks
            if not workflow.tasks:
                workflow.tasks = self._generate_surgical_tasks(pathway_key, workflow.workflow_id)
            
            # OR scheduling optimization
            or_schedule = await self._optimize_or_schedule(workflow, resources)
            
            # Resource coordination
            resource_coordination = await self._coordinate_surgical_resources(workflow, resources)
            
            # Safety checks
            safety_checks = self._perform_surgical_safety_checks(workflow, resources)
            
            return WorkflowResponse(
                workflow_id=workflow.workflow_id,
                status=workflow.status,
                estimated_completion=workflow.estimated_completion,
                scheduled_tasks=or_schedule['scheduled_tasks'],
                resource_assignments=resource_coordination['assignments'],
                next_steps=safety_checks['next_steps'],
                alerts=safety_checks['alerts'],
                performance_metrics=self._calculate_surgical_metrics(workflow),
                metadata={
                    'surgical_pathway': pathway_key,
                    'or_block_time': or_schedule.get('block_time'),
                    'safety_score': safety_checks.get('safety_score', 0)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Surgical workflow execution failed: {e}")
            raise
    
    def _determine_surgical_pathway(self, workflow: ClinicalWorkflow) -> str:
        """Determine surgical pathway based on procedure type."""
        
        clinical_context = workflow.metadata.get('clinical_context', {})
        procedure_type = clinical_context.get('procedure_type', '').lower()
        
        if any(keyword in procedure_type for keyword in ['cardiac', 'heart', 'cabg', 'valve']):
            return 'cardiac_surgery'
        elif any(keyword in procedure_type for keyword in ['emergency', 'trauma', 'urgent']):
            return 'emergency_surgery'
        else:
            return 'general_surgery'
    
    def _generate_surgical_tasks(self, pathway_key: str, workflow_id: str) -> List[ClinicalTask]:
        """Generate surgical tasks based on pathway."""
        
        pathway = self.surgical_pathways.get(pathway_key, [])
        tasks = []
        
        for i, (task_name, duration_mins, required_resources) in enumerate(pathway):
            # Determine priority based on task type
            priority = Priority.URGENT if 'emergency' in pathway_key else Priority.HIGH
            
            task = ClinicalTask(
                task_id=f"{workflow_id}_surgical_{i+1}",
                name=task_name.replace('_', ' ').title(),
                description=f"Surgical {task_name} for {pathway_key}",
                task_type=task_name,
                priority=priority,
                estimated_duration=timedelta(minutes=duration_mins),
                required_resources=required_resources,
                dependencies=[f"{workflow_id}_surgical_{i}"] if i > 0 else []
            )
            tasks.append(task)
        
        return tasks
    
    async def _optimize_or_schedule(
        self, 
        workflow: ClinicalWorkflow, 
        resources: List[Resource]
    ) -> Dict[str, Any]:
        """Optimize operating room scheduling."""
        
        # Find available OR resources
        or_resources = [r for r in resources if r.resource_type == ResourceType.OPERATING_ROOM]
        
        # Calculate total OR time needed
        or_tasks = [t for t in workflow.tasks if ResourceType.OPERATING_ROOM in t.required_resources]
        total_or_time = sum((t.estimated_duration for t in or_tasks), timedelta())
        
        # Find optimal OR slot
        current_time = datetime.utcnow()
        best_or = None
        earliest_start = None
        
        for or_resource in or_resources:
            if or_resource.available_capacity > 0:
                # Calculate next available slot
                available_start = current_time + timedelta(minutes=30)  # Setup time
                if earliest_start is None or available_start < earliest_start:
                    earliest_start = available_start
                    best_or = or_resource
        
        scheduled_tasks = []
        if best_or and earliest_start:
            current_start = earliest_start
            
            for task in or_tasks:
                scheduled_tasks.append({
                    'task_id': task.task_id,
                    'name': task.name,
                    'scheduled_start': current_start,
                    'estimated_duration': task.estimated_duration.total_seconds() / 60,
                    'assigned_or': best_or.resource_id
                })
                
                task.scheduled_start = current_start
                current_start += task.estimated_duration
            
            workflow.estimated_completion = current_start
        
        return {
            'scheduled_tasks': scheduled_tasks,
            'block_time': total_or_time.total_seconds() / 60,
            'assigned_or': best_or.resource_id if best_or else None
        }
    
    async def _coordinate_surgical_resources(
        self, 
        workflow: ClinicalWorkflow, 
        resources: List[Resource]
    ) -> Dict[str, Any]:
        """Coordinate all surgical resources."""
        
        assignments = {}
        
        # Assign surgeon
        surgeons = [r for r in resources if r.resource_type == ResourceType.PHYSICIAN and 'surgery' in r.specialties]
        if surgeons:
            best_surgeon = min(surgeons, key=lambda r: r.utilization_rate)
            assignments['primary_surgeon'] = best_surgeon.resource_id
        
        # Assign anesthesiologist
        anesthesiologists = [r for r in resources if 'anesthesia' in r.specialties]
        if anesthesiologists:
            best_anesthesiologist = min(anesthesiologists, key=lambda r: r.utilization_rate)
            assignments['anesthesiologist'] = best_anesthesiologist.resource_id
        
        # Assign OR nurses
        or_nurses = [r for r in resources if r.resource_type == ResourceType.NURSE and 'or' in r.specialties]
        if len(or_nurses) >= 2:  # Need circulating and scrub nurse
            or_nurses_sorted = sorted(or_nurses, key=lambda r: r.utilization_rate)
            assignments['circulating_nurse'] = or_nurses_sorted[0].resource_id
            assignments['scrub_nurse'] = or_nurses_sorted[1].resource_id
        
        return {'assignments': assignments}
    
    def _perform_surgical_safety_checks(
        self, 
        workflow: ClinicalWorkflow, 
        resources: List[Resource]
    ) -> Dict[str, Any]:
        """Perform surgical safety checks."""
        
        alerts = []
        next_steps = []
        safety_score = 100  # Start with perfect score
        
        # Check resource availability
        required_resources = set()
        for task in workflow.tasks:
            required_resources.update(task.required_resources)
        
        for resource_type in required_resources:
            available = any(r.resource_type == resource_type and r.available_capacity > 0 for r in resources)
            if not available:
                alerts.append(f"CRITICAL: No available {resource_type.value}")
                safety_score -= 20
        
        # Check time constraints
        emergency_indicators = workflow.metadata.get('clinical_context', {}).get('emergency_indicators', [])
        if emergency_indicators:
            alerts.append("EMERGENCY SURGERY: Expedite all preparations")
            next_steps.append("Activate emergency surgical protocol")
        
        # Pre-operative checklist
        next_steps.extend([
            "Verify patient identity and surgical site",
            "Confirm consent and surgical plan",
            "Review allergies and medications",
            "Ensure blood products available if needed"
        ])
        
        # Equipment checks
        required_equipment = workflow.metadata.get('required_equipment', [])
        if required_equipment:
            next_steps.append(f"Verify availability of: {', '.join(required_equipment)}")
        
        return {
            'alerts': alerts,
            'next_steps': next_steps,
            'safety_score': safety_score
        }
    
    def _calculate_surgical_metrics(self, workflow: ClinicalWorkflow) -> Dict[str, Any]:
        """Calculate surgical performance metrics."""
        
        metrics = {}
        
        # OR utilization time
        or_tasks = [t for t in workflow.tasks if ResourceType.OPERATING_ROOM in t.required_resources]
        total_or_time = sum((t.estimated_duration.total_seconds() / 60 for t in or_tasks), 0)
        metrics['total_or_time_minutes'] = total_or_time
        
        # Turnover time estimation
        metrics['estimated_turnover_time'] = 30  # Standard OR turnover
        
        # Resource coordination score
        metrics['resource_coordination_score'] = 85  # Placeholder
        
        return metrics
    
    async def optimize_schedule(
        self, 
        workflows: List[ClinicalWorkflow], 
        resources: List[Resource]
    ) -> Dict[str, Any]:
        """Optimize surgical scheduling across multiple cases."""
        
        # Sort by priority and case complexity
        sorted_workflows = sorted(
            workflows,
            key=lambda w: (w.priority.value, self._estimate_case_complexity(w)),
            reverse=True
        )
        
        optimization_result = {
            'optimized_or_schedule': [],
            'resource_utilization': {},
            'case_sequencing': [],
            'recommendations': []
        }
        
        # OR block scheduling
        or_resources = [r for r in resources if r.resource_type == ResourceType.OPERATING_ROOM]
        current_time = datetime.utcnow()
        
        for i, workflow in enumerate(sorted_workflows):
            or_time = sum(
                t.estimated_duration.total_seconds() / 60 
                for t in workflow.tasks 
                if ResourceType.OPERATING_ROOM in t.required_resources
            )
            
            # Assign to least utilized OR
            if or_resources:
                best_or = min(or_resources, key=lambda r: r.utilization_rate)
                
                optimization_result['optimized_or_schedule'].append({
                    'workflow_id': workflow.workflow_id,
                    'case_order': i + 1,
                    'estimated_start': current_time,
                    'estimated_duration': or_time,
                    'assigned_or': best_or.resource_id,
                    'case_priority': workflow.priority.value
                })
                
                # Update OR utilization
                best_or.current_utilization += or_time / 60  # Convert to hours
                current_time += timedelta(minutes=or_time + 30)  # Add turnover time
        
        return optimization_result
    
    def _estimate_case_complexity(self, workflow: ClinicalWorkflow) -> int:
        """Estimate surgical case complexity."""
        
        # Simplified complexity scoring
        complexity_score = 0
        
        # Base on estimated duration
        total_time = sum(t.estimated_duration.total_seconds() / 60 for t in workflow.tasks)
        if total_time > 240:  # > 4 hours
            complexity_score += 3
        elif total_time > 120:  # > 2 hours
            complexity_score += 2
        else:
            complexity_score += 1
        
        # Factor in procedure type
        clinical_context = workflow.metadata.get('clinical_context', {})
        procedure_type = clinical_context.get('procedure_type', '').lower()
        
        if any(keyword in procedure_type for keyword in ['cardiac', 'neurosurgery', 'transplant']):
            complexity_score += 2
        
        return complexity_score


class AutonomousWorkflowManager:
    """Manager for autonomous clinical workflow operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Initialize workflow engines
        self.engines = {
            ClinicalPathwayType.EMERGENCY_DEPARTMENT: EmergencyDepartmentWorkflowEngine(),
            ClinicalPathwayType.SURGICAL_PATHWAY: SurgicalWorkflowEngine(),
        }
        
        # Active workflows
        self.active_workflows: Dict[str, ClinicalWorkflow] = {}
        
        # Available resources
        self.resources: List[Resource] = []
        
        # Performance monitoring
        self.performance_metrics = defaultdict(list)
        
        # Workflow queue
        self.workflow_queue = deque()
    
    async def create_workflow(self, request: WorkflowRequest) -> ClinicalWorkflow:
        """Create a new clinical workflow."""
        
        workflow_id = str(uuid.uuid4())
        
        workflow = ClinicalWorkflow(
            workflow_id=workflow_id,
            name=f"{request.pathway_type.value}_{request.patient_id}",
            pathway_type=request.pathway_type,
            patient_id=request.patient_id,
            tasks=[],  # Will be generated by engine
            priority=request.priority,
            metadata={
                'clinical_context': request.clinical_context,
                'patient_data': request.patient_data,
                'resource_preferences': request.resource_preferences,
                'constraints': request.constraints
            }
        )
        
        self.active_workflows[workflow_id] = workflow
        self.workflow_queue.append(workflow)
        
        return workflow
    
    async def execute_workflow(self, workflow_id: str) -> WorkflowResponse:
        """Execute a specific workflow."""
        
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Get appropriate engine
        engine = self.engines.get(workflow.pathway_type)
        if not engine:
            raise ValueError(f"No engine available for pathway type: {workflow.pathway_type}")
        
        # Execute workflow
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.utcnow()
        
        try:
            response = await engine.execute_workflow(workflow, self.resources)
            
            # Update workflow status
            if response.status == WorkflowStatus.COMPLETED:
                workflow.status = WorkflowStatus.COMPLETED
                workflow.completed_at = datetime.utcnow()
            
            # Record performance metrics
            self._record_performance_metrics(workflow, response)
            
            return response
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            self.logger.error(f"Workflow execution failed for {workflow_id}: {e}")
            raise
    
    async def optimize_all_workflows(self) -> Dict[str, Any]:
        """Optimize scheduling across all active workflows."""
        
        optimization_results = {}
        
        # Group workflows by pathway type
        pathway_groups = defaultdict(list)
        for workflow in self.active_workflows.values():
            if workflow.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
                pathway_groups[workflow.pathway_type].append(workflow)
        
        # Optimize each pathway type
        for pathway_type, workflows in pathway_groups.items():
            engine = self.engines.get(pathway_type)
            if engine:
                optimization = await engine.optimize_schedule(workflows, self.resources)
                optimization_results[pathway_type.value] = optimization
        
        # Global optimization recommendations
        global_recommendations = self._generate_global_recommendations(optimization_results)
        
        return {
            'pathway_optimizations': optimization_results,
            'global_recommendations': global_recommendations,
            'overall_metrics': self._calculate_overall_metrics()
        }
    
    def add_resource(self, resource: Resource):
        """Add a healthcare resource."""
        self.resources.append(resource)
    
    def update_resource_capacity(self, resource_id: str, new_capacity: int):
        """Update resource capacity."""
        for resource in self.resources:
            if resource.resource_id == resource_id:
                resource.capacity = new_capacity
                break
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get detailed workflow status."""
        
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return {'error': 'Workflow not found'}
        
        return {
            'workflow_id': workflow.workflow_id,
            'status': workflow.status.value,
            'pathway_type': workflow.pathway_type.value,
            'priority': workflow.priority.value,
            'created_at': workflow.created_at,
            'started_at': workflow.started_at,
            'completed_at': workflow.completed_at,
            'estimated_completion': workflow.estimated_completion,
            'task_count': len(workflow.tasks),
            'completed_tasks': len([t for t in workflow.tasks if t.status == WorkflowStatus.COMPLETED]),
            'progress_percentage': self._calculate_workflow_progress(workflow)
        }
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization metrics."""
        
        utilization = {}
        
        for resource in self.resources:
            utilization[resource.resource_id] = {
                'name': resource.name,
                'type': resource.resource_type.value,
                'capacity': resource.capacity,
                'current_utilization': resource.current_utilization,
                'utilization_rate': resource.utilization_rate,
                'available_capacity': resource.available_capacity
            }
        
        return utilization
    
    def _record_performance_metrics(self, workflow: ClinicalWorkflow, response: WorkflowResponse):
        """Record performance metrics for workflow execution."""
        
        metrics = {
            'workflow_id': workflow.workflow_id,
            'pathway_type': workflow.pathway_type.value,
            'execution_time': (datetime.utcnow() - workflow.started_at).total_seconds() if workflow.started_at else 0,
            'task_count': len(workflow.tasks),
            'resource_efficiency': response.performance_metrics.get('resource_efficiency', 0),
            'completion_status': workflow.status.value
        }
        
        self.performance_metrics[workflow.pathway_type].append(metrics)
    
    def _calculate_workflow_progress(self, workflow: ClinicalWorkflow) -> float:
        """Calculate workflow completion percentage."""
        
        if not workflow.tasks:
            return 0.0
        
        completed_tasks = len([t for t in workflow.tasks if t.status == WorkflowStatus.COMPLETED])
        return (completed_tasks / len(workflow.tasks)) * 100
    
    def _generate_global_recommendations(self, optimization_results: Dict[str, Any]) -> List[str]:
        """Generate global workflow optimization recommendations."""
        
        recommendations = []
        
        # Analyze resource bottlenecks
        high_utilization_resources = [
            r for r in self.resources if r.utilization_rate > 0.85
        ]
        
        if high_utilization_resources:
            resource_types = set(r.resource_type.value for r in high_utilization_resources)
            recommendations.append(f"Resource bottleneck detected in: {', '.join(resource_types)}")
        
        # Workflow queue analysis
        if len(self.workflow_queue) > 10:
            recommendations.append("High workflow volume - consider additional staffing or process optimization")
        
        # Priority workflow analysis
        emergency_workflows = [
            w for w in self.active_workflows.values() 
            if w.priority == Priority.EMERGENCY and w.status != WorkflowStatus.COMPLETED
        ]
        
        if emergency_workflows:
            recommendations.append(f"URGENT: {len(emergency_workflows)} emergency workflows require immediate attention")
        
        return recommendations
    
    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall system performance metrics."""
        
        total_workflows = len(self.active_workflows)
        completed_workflows = len([
            w for w in self.active_workflows.values() 
            if w.status == WorkflowStatus.COMPLETED
        ])
        
        avg_resource_utilization = (
            sum(r.utilization_rate for r in self.resources) / len(self.resources) 
            if self.resources else 0
        )
        
        return {
            'total_workflows': total_workflows,
            'completed_workflows': completed_workflows,
            'completion_rate': completed_workflows / total_workflows if total_workflows > 0 else 0,
            'average_resource_utilization': avg_resource_utilization,
            'queue_length': len(self.workflow_queue),
            'active_emergency_workflows': len([
                w for w in self.active_workflows.values() 
                if w.priority == Priority.EMERGENCY and w.status != WorkflowStatus.COMPLETED
            ])
        }


# Factory function
def create_autonomous_workflow_manager(config: Dict[str, Any]) -> AutonomousWorkflowManager:
    """Create autonomous workflow manager with configuration."""
    return AutonomousWorkflowManager(config)