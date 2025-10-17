"""
Healthcare Team Framework for Vita Agents
Advanced team-based collaboration system for healthcare AI agents

This module provides:
- Healthcare-specific team formations (ICU Team, Surgery Team, Emergency Team, etc.)
- Team-based task coordination and workflow management
- Specialty-based team assembly and resource allocation
- Performance tracking and team analytics
- Emergency response team protocols
"""

import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import logging

from healthcare_agent_framework import (
    HealthcareAgent, HealthcareRole, ClinicalTask, ClinicalTaskType, 
    PatientContext, PatientSeverity, healthcare_workflow
)

logger = logging.getLogger(__name__)

class TeamType(Enum):
    """Healthcare team types"""
    ICU_TEAM = "icu_team"
    EMERGENCY_TEAM = "emergency_team"
    SURGICAL_TEAM = "surgical_team"
    CARDIAC_TEAM = "cardiac_team"
    STROKE_TEAM = "stroke_team"
    TRAUMA_TEAM = "trauma_team"
    MULTIDISCIPLINARY_TEAM = "multidisciplinary_team"
    PRIMARY_CARE_TEAM = "primary_care_team"
    SPECIALTY_CONSULT_TEAM = "specialty_consult_team"
    DISCHARGE_PLANNING_TEAM = "discharge_planning_team"

class TeamStatus(Enum):
    """Team operational status"""
    ASSEMBLING = "assembling"
    ACTIVE = "active"
    ON_CALL = "on_call"
    BUSY = "busy"
    DISBANDING = "disbanding"
    INACTIVE = "inactive"

class CoordinationPattern(Enum):
    """Team coordination patterns"""
    HIERARCHICAL = "hierarchical"  # Lead physician directs team
    COLLABORATIVE = "collaborative"  # Equal participation
    SPECIALIST_LED = "specialist_led"  # Specialist leads for specific cases
    EMERGENCY_RESPONSE = "emergency_response"  # Rapid response protocol
    CONSENSUS_BASED = "consensus_based"  # Group decision making

@dataclass
class TeamRole:
    """Role definition within a healthcare team"""
    name: str
    required_agent_role: HealthcareRole
    responsibilities: List[str]
    authority_level: int  # 1-10, higher = more authority
    is_lead: bool = False
    is_required: bool = True
    minimum_experience: int = 0  # Minimum completed tasks

@dataclass
class TeamProtocol:
    """Standard operating protocol for a team"""
    name: str
    triggers: List[str]  # Conditions that activate this protocol
    steps: List[Dict[str, Any]]
    priority_level: int
    max_response_time: timedelta
    required_roles: List[HealthcareRole]

@dataclass
class TeamMetrics:
    """Team performance metrics"""
    cases_handled: int = 0
    average_response_time: float = 0.0
    success_rate: float = 0.0
    patient_satisfaction: float = 0.0
    collaboration_score: float = 0.0
    efficiency_rating: float = 0.0
    cases_by_severity: Dict[str, int] = field(default_factory=dict)

class HealthcareTeam(BaseModel):
    """Healthcare team model with specialized capabilities"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    team_type: TeamType
    description: str
    
    # Team composition
    members: Dict[str, HealthcareAgent] = Field(default_factory=dict)
    team_roles: Dict[str, TeamRole] = Field(default_factory=dict)
    lead_agent_id: Optional[str] = None
    
    # Team configuration
    coordination_pattern: CoordinationPattern = CoordinationPattern.COLLABORATIVE
    protocols: List[TeamProtocol] = Field(default_factory=list)
    specialties: List[str] = Field(default_factory=list)
    
    # Team state
    status: TeamStatus = TeamStatus.INACTIVE
    current_cases: List[str] = Field(default_factory=list)
    completed_cases: List[str] = Field(default_factory=list)
    
    # Performance tracking
    metrics: TeamMetrics = Field(default_factory=TeamMetrics)
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: Optional[datetime] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_member(self, agent: HealthcareAgent, role_name: str) -> bool:
        """Add an agent to the team with a specific role"""
        if role_name not in self.team_roles:
            logger.error(f"Role {role_name} not defined for team {self.name}")
            return False
        
        role = self.team_roles[role_name]
        
        # Check if agent meets role requirements
        if agent.role != role.required_agent_role:
            logger.error(f"Agent {agent.name} role {agent.role.value} doesn't match required role {role.required_agent_role.value}")
            return False
        
        if len(agent.completed_tasks) < role.minimum_experience:
            logger.warning(f"Agent {agent.name} has insufficient experience for role {role_name}")
            if role.is_required:
                return False
        
        self.members[agent.id] = agent
        
        # Set lead if this is a lead role
        if role.is_lead:
            self.lead_agent_id = agent.id
        
        logger.info(f"Added {agent.name} to team {self.name} as {role_name}")
        return True
    
    def remove_member(self, agent_id: str) -> bool:
        """Remove an agent from the team"""
        if agent_id not in self.members:
            return False
        
        agent = self.members[agent_id]
        del self.members[agent_id]
        
        # Update lead if necessary
        if self.lead_agent_id == agent_id:
            self.lead_agent_id = None
            # Try to assign new lead
            self._assign_new_lead()
        
        logger.info(f"Removed {agent.name} from team {self.name}")
        return True
    
    def _assign_new_lead(self):
        """Assign a new team lead based on authority levels"""
        lead_candidates = []
        
        for agent_id, agent in self.members.items():
            for role in self.team_roles.values():
                if (role.required_agent_role == agent.role and 
                    role.authority_level > 5):  # High authority roles
                    lead_candidates.append((agent_id, role.authority_level))
        
        if lead_candidates:
            # Sort by authority level and assign highest
            lead_candidates.sort(key=lambda x: x[1], reverse=True)
            self.lead_agent_id = lead_candidates[0][0]
    
    def can_handle_case(self, patient_context: PatientContext, task_type: ClinicalTaskType) -> tuple[bool, float]:
        """Determine if team can handle a case and calculate confidence"""
        
        # Check if team has required roles
        required_roles = self._get_required_roles_for_case(patient_context, task_type)
        available_roles = set(agent.role for agent in self.members.values())
        
        if not required_roles.issubset(available_roles):
            return False, 0.0
        
        # Check team capacity
        if len(self.current_cases) >= self._get_max_capacity():
            return False, 0.0
        
        # Calculate confidence based on team expertise and experience
        confidence = self._calculate_team_confidence(patient_context, task_type)
        
        return True, confidence
    
    def _get_required_roles_for_case(self, patient_context: PatientContext, task_type: ClinicalTaskType) -> Set[HealthcareRole]:
        """Get required roles based on case complexity"""
        base_roles = {HealthcareRole.GENERAL_PRACTITIONER}
        
        # Add roles based on severity
        if patient_context.severity in [PatientSeverity.CRITICAL, PatientSeverity.EMERGENCY]:
            base_roles.update({
                HealthcareRole.DIAGNOSTICIAN,
                HealthcareRole.CARE_COORDINATOR
            })
        
        # Add roles based on task type
        if task_type == ClinicalTaskType.MEDICATION_REVIEW:
            base_roles.add(HealthcareRole.PHARMACIST)
        elif task_type == ClinicalTaskType.IMAGE_INTERPRETATION:
            base_roles.add(HealthcareRole.RADIOLOGIST)
        elif task_type == ClinicalTaskType.LAB_ANALYSIS:
            base_roles.add(HealthcareRole.PATHOLOGIST)
        
        # Add specialty roles based on medical history
        history_keywords = [condition.lower() for condition in patient_context.medical_history]
        if any(keyword in history_keywords for keyword in ['heart', 'cardiac', 'cardio']):
            base_roles.add(HealthcareRole.CARDIOLOGIST)
        if any(keyword in history_keywords for keyword in ['neuro', 'brain', 'stroke']):
            base_roles.add(HealthcareRole.NEUROLOGIST)
        
        return base_roles
    
    def _get_max_capacity(self) -> int:
        """Get maximum concurrent cases based on team size and type"""
        base_capacity = len(self.members)
        
        # Adjust based on team type
        if self.team_type in [TeamType.EMERGENCY_TEAM, TeamType.TRAUMA_TEAM]:
            return base_capacity * 2  # Emergency teams can handle more
        elif self.team_type == TeamType.ICU_TEAM:
            return max(1, base_capacity // 2)  # ICU cases need more attention
        else:
            return base_capacity
    
    def _calculate_team_confidence(self, patient_context: PatientContext, task_type: ClinicalTaskType) -> float:
        """Calculate team's confidence for handling a case"""
        if not self.members:
            return 0.0
        
        # Get individual agent confidences
        agent_confidences = []
        for agent in self.members.values():
            # Create a dummy task to test confidence
            dummy_task = ClinicalTask(
                id="dummy",
                task_type=task_type,
                patient_context=patient_context,
                description="dummy"
            )
            can_handle, confidence = agent.can_handle_task(dummy_task)
            if can_handle:
                agent_confidences.append(confidence)
        
        if not agent_confidences:
            return 0.0
        
        # Calculate team confidence (weighted average with team synergy bonus)
        base_confidence = sum(agent_confidences) / len(agent_confidences)
        
        # Team synergy bonus (more members = better collaboration)
        synergy_bonus = min(0.2, len(self.members) * 0.05)
        
        # Experience bonus based on completed cases
        experience_bonus = min(0.15, self.metrics.cases_handled * 0.01)
        
        final_confidence = min(1.0, base_confidence + synergy_bonus + experience_bonus)
        return final_confidence
    
    def activate_protocol(self, protocol_name: str, patient_context: PatientContext) -> Dict[str, Any]:
        """Activate a specific team protocol"""
        protocol = next((p for p in self.protocols if p.name == protocol_name), None)
        if not protocol:
            return {"error": f"Protocol {protocol_name} not found"}
        
        # Check if team has required roles for protocol
        required_roles = set(protocol.required_roles)
        available_roles = set(agent.role for agent in self.members.values())
        
        if not required_roles.issubset(available_roles):
            missing_roles = required_roles - available_roles
            return {
                "error": f"Missing required roles for protocol: {[r.value for r in missing_roles]}"
            }
        
        # Execute protocol steps
        execution_plan = {
            "protocol": protocol_name,
            "patient_id": patient_context.patient_id,
            "severity": patient_context.severity.value,
            "steps": [],
            "estimated_completion": datetime.now() + protocol.max_response_time,
            "team_members": [agent.name for agent in self.members.values()]
        }
        
        for i, step in enumerate(protocol.steps):
            execution_plan["steps"].append({
                "step_number": i + 1,
                "action": step.get("action", ""),
                "responsible_role": step.get("responsible_role", ""),
                "estimated_duration": step.get("duration", "Unknown"),
                "status": "pending"
            })
        
        self.status = TeamStatus.ACTIVE
        self.last_active = datetime.now()
        
        logger.info(f"Activated protocol {protocol_name} for team {self.name}")
        return execution_plan
    
    def get_team_status(self) -> Dict[str, Any]:
        """Get comprehensive team status"""
        return {
            "team_id": self.id,
            "name": self.name,
            "type": self.team_type.value,
            "status": self.status.value,
            "members": len(self.members),
            "active_cases": len(self.current_cases),
            "completed_cases": len(self.completed_cases),
            "lead_agent": self.members[self.lead_agent_id].name if self.lead_agent_id else "None",
            "specialties": self.specialties,
            "coordination_pattern": self.coordination_pattern.value,
            "protocols_available": len(self.protocols),
            "metrics": {
                "success_rate": self.metrics.success_rate,
                "average_response_time": self.metrics.average_response_time,
                "cases_handled": self.metrics.cases_handled,
                "efficiency_rating": self.metrics.efficiency_rating
            }
        }

class TeamManager:
    """Manages healthcare teams and team-based workflows"""
    
    def __init__(self):
        self.teams: Dict[str, HealthcareTeam] = {}
        self.team_templates: Dict[TeamType, Dict[str, Any]] = {}
        self._initialize_team_templates()
    
    def _initialize_team_templates(self):
        """Initialize standard healthcare team templates"""
        
        # Emergency Team Template
        self.team_templates[TeamType.EMERGENCY_TEAM] = {
            "name": "Emergency Response Team",
            "description": "Rapid response team for emergency situations",
            "coordination_pattern": CoordinationPattern.EMERGENCY_RESPONSE,
            "team_roles": {
                "emergency_physician": TeamRole(
                    name="Emergency Physician",
                    required_agent_role=HealthcareRole.DIAGNOSTICIAN,
                    responsibilities=["Initial assessment", "Treatment decisions", "Team coordination"],
                    authority_level=9,
                    is_lead=True,
                    is_required=True
                ),
                "emergency_pharmacist": TeamRole(
                    name="Emergency Pharmacist",
                    required_agent_role=HealthcareRole.PHARMACIST,
                    responsibilities=["Emergency medications", "Drug interactions", "Dosing"],
                    authority_level=6,
                    is_required=True
                ),
                "care_coordinator": TeamRole(
                    name="Care Coordinator",
                    required_agent_role=HealthcareRole.CARE_COORDINATOR,
                    responsibilities=["Resource allocation", "Communication", "Follow-up"],
                    authority_level=5,
                    is_required=True
                )
            },
            "protocols": [
                TeamProtocol(
                    name="Cardiac Arrest Response",
                    triggers=["cardiac arrest", "code blue", "ventricular fibrillation"],
                    steps=[
                        {"action": "Initiate CPR", "responsible_role": "emergency_physician", "duration": "immediate"},
                        {"action": "Prepare emergency medications", "responsible_role": "emergency_pharmacist", "duration": "2 minutes"},
                        {"action": "Coordinate care team", "responsible_role": "care_coordinator", "duration": "ongoing"}
                    ],
                    priority_level=10,
                    max_response_time=timedelta(minutes=2),
                    required_roles=[HealthcareRole.DIAGNOSTICIAN, HealthcareRole.PHARMACIST, HealthcareRole.CARE_COORDINATOR]
                ),
                TeamProtocol(
                    name="Stroke Alert",
                    triggers=["stroke", "CVA", "neurological deficit"],
                    steps=[
                        {"action": "Neurological assessment", "responsible_role": "emergency_physician", "duration": "5 minutes"},
                        {"action": "Prepare thrombolytics", "responsible_role": "emergency_pharmacist", "duration": "10 minutes"},
                        {"action": "Coordinate imaging", "responsible_role": "care_coordinator", "duration": "15 minutes"}
                    ],
                    priority_level=9,
                    max_response_time=timedelta(minutes=15),
                    required_roles=[HealthcareRole.DIAGNOSTICIAN, HealthcareRole.PHARMACIST, HealthcareRole.CARE_COORDINATOR]
                )
            ]
        }
        
        # ICU Team Template
        self.team_templates[TeamType.ICU_TEAM] = {
            "name": "Intensive Care Unit Team",
            "description": "Specialized team for critical care management",
            "coordination_pattern": CoordinationPattern.HIERARCHICAL,
            "team_roles": {
                "intensivist": TeamRole(
                    name="Intensivist",
                    required_agent_role=HealthcareRole.DIAGNOSTICIAN,
                    responsibilities=["Critical care decisions", "Team leadership", "Family communication"],
                    authority_level=10,
                    is_lead=True,
                    is_required=True,
                    minimum_experience=20
                ),
                "critical_care_pharmacist": TeamRole(
                    name="Critical Care Pharmacist",
                    required_agent_role=HealthcareRole.PHARMACIST,
                    responsibilities=["Complex medication management", "Drug monitoring", "Adverse effect prevention"],
                    authority_level=7,
                    is_required=True,
                    minimum_experience=15
                ),
                "care_coordinator": TeamRole(
                    name="ICU Care Coordinator",
                    required_agent_role=HealthcareRole.CARE_COORDINATOR,
                    responsibilities=["ICU logistics", "Discharge planning", "Family updates"],
                    authority_level=6,
                    is_required=True
                )
            },
            "protocols": [
                TeamProtocol(
                    name="Sepsis Management",
                    triggers=["sepsis", "systemic infection", "SIRS"],
                    steps=[
                        {"action": "Blood cultures and antibiotics", "responsible_role": "intensivist", "duration": "1 hour"},
                        {"action": "Antimicrobial optimization", "responsible_role": "critical_care_pharmacist", "duration": "ongoing"},
                        {"action": "Family notification", "responsible_role": "care_coordinator", "duration": "2 hours"}
                    ],
                    priority_level=9,
                    max_response_time=timedelta(hours=1),
                    required_roles=[HealthcareRole.DIAGNOSTICIAN, HealthcareRole.PHARMACIST, HealthcareRole.CARE_COORDINATOR]
                )
            ]
        }
        
        # Primary Care Team Template
        self.team_templates[TeamType.PRIMARY_CARE_TEAM] = {
            "name": "Primary Care Team",
            "description": "Comprehensive primary healthcare team",
            "coordination_pattern": CoordinationPattern.COLLABORATIVE,
            "team_roles": {
                "primary_physician": TeamRole(
                    name="Primary Care Physician",
                    required_agent_role=HealthcareRole.GENERAL_PRACTITIONER,
                    responsibilities=["Primary diagnosis", "Treatment planning", "Preventive care"],
                    authority_level=8,
                    is_lead=True,
                    is_required=True
                ),
                "pharmacist": TeamRole(
                    name="Clinical Pharmacist",
                    required_agent_role=HealthcareRole.PHARMACIST,
                    responsibilities=["Medication management", "Patient education", "Chronic disease management"],
                    authority_level=6,
                    is_required=False
                ),
                "care_manager": TeamRole(
                    name="Care Manager",
                    required_agent_role=HealthcareRole.CARE_COORDINATOR,
                    responsibilities=["Care coordination", "Resource navigation", "Patient advocacy"],
                    authority_level=5,
                    is_required=True
                )
            },
            "protocols": [
                TeamProtocol(
                    name="Chronic Disease Management",
                    triggers=["diabetes", "hypertension", "chronic condition"],
                    steps=[
                        {"action": "Comprehensive assessment", "responsible_role": "primary_physician", "duration": "30 minutes"},
                        {"action": "Medication review", "responsible_role": "pharmacist", "duration": "15 minutes"},
                        {"action": "Care plan development", "responsible_role": "care_manager", "duration": "20 minutes"}
                    ],
                    priority_level=5,
                    max_response_time=timedelta(days=1),
                    required_roles=[HealthcareRole.GENERAL_PRACTITIONER, HealthcareRole.CARE_COORDINATOR]
                )
            ]
        }
    
    def create_team(self, team_type: TeamType, name: str = None) -> HealthcareTeam:
        """Create a team from template"""
        if team_type not in self.team_templates:
            raise ValueError(f"No template available for team type: {team_type}")
        
        template = self.team_templates[team_type]
        
        team = HealthcareTeam(
            name=name or template["name"],
            team_type=team_type,
            description=template["description"],
            coordination_pattern=template["coordination_pattern"],
            protocols=template["protocols"]
        )
        
        # Add team roles
        for role_name, role_def in template["team_roles"].items():
            team.team_roles[role_name] = role_def
        
        self.teams[team.id] = team
        logger.info(f"Created team: {team.name} ({team_type.value})")
        return team
    
    def auto_assemble_team(self, team_type: TeamType, available_agents: List[HealthcareAgent]) -> Optional[HealthcareTeam]:
        """Automatically assemble a team from available agents"""
        team = self.create_team(team_type)
        
        # Try to fill required roles first
        for role_name, role_def in team.team_roles.items():
            if role_def.is_required:
                suitable_agents = [
                    agent for agent in available_agents 
                    if (agent.role == role_def.required_agent_role and 
                        len(agent.completed_tasks) >= role_def.minimum_experience and
                        agent.active and
                        agent.id not in team.members)
                ]
                
                if not suitable_agents:
                    logger.warning(f"No suitable agent found for required role: {role_name}")
                    if role_def.is_required:
                        # Can't form team without required roles
                        del self.teams[team.id]
                        return None
                else:
                    # Select best agent based on performance
                    best_agent = max(suitable_agents, 
                                   key=lambda a: a.performance_metrics.get("success_rate", 0))
                    team.add_member(best_agent, role_name)
        
        # Fill optional roles if agents available
        for role_name, role_def in team.team_roles.items():
            if not role_def.is_required and role_name not in [r for r in team.team_roles.keys() if team.team_roles[r].is_required]:
                suitable_agents = [
                    agent for agent in available_agents 
                    if (agent.role == role_def.required_agent_role and 
                        agent.active and
                        agent.id not in team.members)
                ]
                
                if suitable_agents:
                    best_agent = max(suitable_agents, 
                                   key=lambda a: a.performance_metrics.get("success_rate", 0))
                    team.add_member(best_agent, role_name)
        
        team.status = TeamStatus.ACTIVE
        logger.info(f"Auto-assembled team {team.name} with {len(team.members)} members")
        return team
    
    def assign_case_to_team(self, case_id: str, patient_context: PatientContext, 
                           task_type: ClinicalTaskType) -> Optional[HealthcareTeam]:
        """Assign a case to the most suitable team"""
        suitable_teams = []
        
        for team in self.teams.values():
            if team.status in [TeamStatus.ACTIVE, TeamStatus.ON_CALL]:
                can_handle, confidence = team.can_handle_case(patient_context, task_type)
                if can_handle:
                    suitable_teams.append((team, confidence))
        
        if not suitable_teams:
            # Try to auto-assemble a team
            available_agents = list(healthcare_workflow.agents.values())
            
            # Determine best team type for case
            if patient_context.severity in [PatientSeverity.EMERGENCY, PatientSeverity.CRITICAL]:
                team_type = TeamType.EMERGENCY_TEAM
            else:
                team_type = TeamType.PRIMARY_CARE_TEAM
            
            new_team = self.auto_assemble_team(team_type, available_agents)
            if new_team:
                suitable_teams = [(new_team, 0.7)]  # Default confidence for new team
        
        if not suitable_teams:
            logger.warning(f"No suitable team found for case {case_id}")
            return None
        
        # Select team with highest confidence
        suitable_teams.sort(key=lambda x: x[1], reverse=True)
        best_team = suitable_teams[0][0]
        
        best_team.current_cases.append(case_id)
        logger.info(f"Assigned case {case_id} to team {best_team.name}")
        return best_team
    
    def get_team_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all teams"""
        total_teams = len(self.teams)
        active_teams = len([t for t in self.teams.values() if t.status == TeamStatus.ACTIVE])
        total_cases = sum(t.metrics.cases_handled for t in self.teams.values())
        
        avg_success_rate = 0.0
        if self.teams:
            success_rates = [t.metrics.success_rate for t in self.teams.values() if t.metrics.cases_handled > 0]
            avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0
        
        teams_by_type = {}
        for team in self.teams.values():
            team_type = team.team_type.value
            teams_by_type[team_type] = teams_by_type.get(team_type, 0) + 1
        
        return {
            "total_teams": total_teams,
            "active_teams": active_teams,
            "total_cases_handled": total_cases,
            "average_success_rate": avg_success_rate,
            "teams_by_type": teams_by_type,
            "team_utilization": active_teams / total_teams if total_teams > 0 else 0.0
        }

# Global team manager instance
team_manager = TeamManager()

def create_default_healthcare_teams():
    """Create default healthcare teams"""
    
    # Get available agents
    available_agents = list(healthcare_workflow.agents.values())
    
    if len(available_agents) < 3:
        logger.warning("Not enough agents to create meaningful teams")
        return []
    
    created_teams = []
    
    # Create Emergency Team
    emergency_team = team_manager.auto_assemble_team(TeamType.EMERGENCY_TEAM, available_agents)
    if emergency_team:
        created_teams.append(emergency_team)
    
    # Create Primary Care Team
    primary_team = team_manager.auto_assemble_team(TeamType.PRIMARY_CARE_TEAM, available_agents)
    if primary_team:
        created_teams.append(primary_team)
    
    logger.info(f"Created {len(created_teams)} default healthcare teams")
    return created_teams