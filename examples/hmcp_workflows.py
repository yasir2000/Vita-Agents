#!/usr/bin/env python3
"""
HMCP Healthcare Workflow Examples

Demonstrates common healthcare multi-agent workflow patterns using the
Healthcare Model Context Protocol (HMCP) in Vita Agents.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any

from vita_agents.agents.hmcp_agent import HMCPAgent
from vita_agents.protocols.hmcp import (
    HMCPMessageType, ClinicalUrgency, HealthcareRole,
    PatientContext, ClinicalContext, hmcp_protocol
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HMCPWorkflowExamples:
    """Healthcare workflow examples using HMCP"""
    
    def __init__(self):
        self.agents: Dict[str, HMCPAgent] = {}
    
    async def setup_healthcare_team(self):
        """Set up a multi-disciplinary healthcare team"""
        print("🏥 Setting up healthcare team...")
        
        # Create diagnostic agent
        diagnostic_config = {
            'role': 'ai_agent',
            'capabilities': [
                'differential_diagnosis',
                'symptom_analysis',
                'clinical_reasoning',
                'medical_imaging_analysis'
            ],
            'emergency_capable': True
        }
        diagnostic_agent = HMCPAgent('diagnostic_copilot', diagnostic_config)
        self.agents['diagnostic_copilot'] = diagnostic_agent
        
        # Create medical knowledge agent
        knowledge_config = {
            'role': 'ai_agent',
            'capabilities': [
                'drug_interactions',
                'treatment_guidelines',
                'evidence_based_medicine',
                'clinical_decision_support'
            ],
            'emergency_capable': False
        }
        knowledge_agent = HMCPAgent('medical_knowledge', knowledge_config)
        self.agents['medical_knowledge'] = knowledge_agent
        
        # Create patient data agent
        patient_data_config = {
            'role': 'ai_agent',
            'capabilities': [
                'ehr_integration',
                'lab_results_processing',
                'vital_signs_monitoring',
                'medication_reconciliation'
            ],
            'emergency_capable': True
        }
        patient_data_agent = HMCPAgent('patient_data', patient_data_config)
        self.agents['patient_data'] = patient_data_agent
        
        # Create scheduling agent
        scheduling_config = {
            'role': 'ai_agent',
            'capabilities': [
                'appointment_scheduling',
                'resource_management',
                'staff_coordination',
                'workflow_optimization'
            ],
            'emergency_capable': False
        }
        scheduling_agent = HMCPAgent('scheduling_agent', scheduling_config)
        self.agents['scheduling_agent'] = scheduling_agent
        
        print("✅ Healthcare team assembled!")
        print(f"   - {len(self.agents)} agents created")
        print(f"   - Emergency capable agents: {sum(1 for a in self.agents.values() if a.emergency_capable)}")
    
    async def example_1_chest_pain_diagnosis(self):
        """Example 1: Chest pain diagnosis workflow"""
        print("\n📋 Example 1: Chest Pain Diagnosis Workflow")
        print("-" * 50)
        
        patient_id = "PATIENT_001"
        
        # Step 1: Patient data agent processes vital signs
        patient_data = {
            "action": "process_vital_signs",
            "vital_signs": {
                "blood_pressure": "160/95",
                "heart_rate": 110,
                "respiratory_rate": 22,
                "temperature": 98.6,
                "oxygen_saturation": 96
            },
            "symptoms": ["chest_pain", "shortness_of_breath", "diaphoresis"]
        }
        
        await self.agents['patient_data'].send_clinical_message(
            receiver_id='diagnostic_copilot',
            message_type=HMCPMessageType.REQUEST,
            content=patient_data,
            patient_id=patient_id,
            urgency=ClinicalUrgency.URGENT
        )
        
        print("📊 Patient data sent to diagnostic copilot")
        
        # Step 2: Diagnostic copilot requests medical knowledge
        diagnostic_request = {
            "action": "differential_diagnosis",
            "chief_complaint": "chest_pain",
            "presenting_symptoms": ["chest_pain", "shortness_of_breath", "diaphoresis"],
            "vital_signs": patient_data["vital_signs"],
            "age": 55,
            "sex": "male"
        }
        
        await self.agents['diagnostic_copilot'].send_clinical_message(
            receiver_id='medical_knowledge',
            message_type=HMCPMessageType.REQUEST,
            content=diagnostic_request,
            patient_id=patient_id,
            urgency=ClinicalUrgency.URGENT
        )
        
        print("🧠 Diagnostic request sent to medical knowledge agent")
        
        # Step 3: Medical knowledge agent provides clinical decision support
        clinical_support = {
            "action": "clinical_decision_support",
            "differential_diagnosis": [
                "acute_coronary_syndrome",
                "pulmonary_embolism",
                "aortic_dissection",
                "pneumothorax"
            ],
            "recommended_tests": [
                "ECG",
                "troponin_levels",
                "chest_xray",
                "d_dimer"
            ],
            "urgency_assessment": "immediate_evaluation_required"
        }
        
        await self.agents['medical_knowledge'].send_clinical_message(
            receiver_id='diagnostic_copilot',
            message_type=HMCPMessageType.RESPONSE,
            content=clinical_support,
            patient_id=patient_id,
            urgency=ClinicalUrgency.URGENT
        )
        
        print("💡 Clinical decision support provided")
        
        # Step 4: Schedule emergency procedures
        emergency_scheduling = {
            "action": "emergency_scheduling",
            "procedures": ["ECG", "lab_draw", "chest_xray"],
            "priority": "STAT",
            "location": "emergency_department"
        }
        
        await self.agents['diagnostic_copilot'].send_clinical_message(
            receiver_id='scheduling_agent',
            message_type=HMCPMessageType.COORDINATION,
            content=emergency_scheduling,
            patient_id=patient_id,
            urgency=ClinicalUrgency.EMERGENCY
        )
        
        print("⚡ Emergency procedures scheduled")
        print("✅ Chest pain diagnosis workflow completed")
    
    async def example_2_medication_interaction_check(self):
        """Example 2: Medication interaction checking workflow"""
        print("\n💊 Example 2: Medication Interaction Check")
        print("-" * 50)
        
        patient_id = "PATIENT_002"
        
        # Step 1: Patient data agent receives new prescription
        new_prescription = {
            "action": "medication_check",
            "new_medication": {
                "name": "warfarin",
                "dose": "5mg",
                "frequency": "daily"
            },
            "current_medications": [
                {"name": "aspirin", "dose": "81mg", "frequency": "daily"},
                {"name": "metformin", "dose": "500mg", "frequency": "twice_daily"}
            ],
            "allergies": ["penicillin", "sulfa"]
        }
        
        await self.agents['patient_data'].send_clinical_message(
            receiver_id='medical_knowledge',
            message_type=HMCPMessageType.REQUEST,
            content=new_prescription,
            patient_id=patient_id,
            urgency=ClinicalUrgency.ROUTINE
        )
        
        print("📋 New prescription sent for interaction check")
        
        # Step 2: Medical knowledge agent detects interaction
        interaction_alert = {
            "action": "medication_alert",
            "alert_type": "drug_interaction",
            "severity": "high",
            "interaction": {
                "drugs": ["warfarin", "aspirin"],
                "effect": "increased_bleeding_risk",
                "recommendation": "monitor_inr_closely"
            },
            "alternative_suggestions": [
                {"name": "clopidogrel", "note": "alternative_antiplatelet"}
            ]
        }
        
        await self.agents['medical_knowledge'].send_clinical_message(
            receiver_id='patient_data',
            message_type=HMCPMessageType.NOTIFICATION,
            content=interaction_alert,
            patient_id=patient_id,
            urgency=ClinicalUrgency.URGENT
        )
        
        print("⚠️  Drug interaction detected and alert sent")
        print("✅ Medication interaction check completed")
    
    async def example_3_emergency_cardiac_arrest(self):
        """Example 3: Emergency cardiac arrest response"""
        print("\n🚨 Example 3: Emergency Cardiac Arrest Response")
        print("-" * 50)
        
        patient_id = "PATIENT_003"
        
        # Step 1: Patient data agent detects cardiac arrest
        emergency_id = await self.agents['patient_data'].initiate_emergency_response(
            patient_id=patient_id,
            emergency_type="cardiac_arrest",
            location="room_305_icu",
            details={
                "vital_signs": {
                    "heart_rate": 0,
                    "blood_pressure": "undetectable",
                    "respiratory_rate": 0,
                    "oxygen_saturation": 85
                },
                "witnessed": True,
                "time_detected": datetime.now(timezone.utc).isoformat()
            }
        )
        
        print(f"🚨 Cardiac arrest emergency initiated (ID: {emergency_id})")
        
        # Step 2: Coordinate emergency response team
        workflow_id = await self.agents['patient_data'].coordinate_care_workflow(
            patient_id=patient_id,
            workflow_type="cardiac_arrest_response",
            participants=['diagnostic_copilot', 'medical_knowledge', 'scheduling_agent'],
            care_plan={
                "protocol": "ACLS",
                "interventions": [
                    "CPR_initiation",
                    "defibrillation_preparation",
                    "epinephrine_administration",
                    "airway_management"
                ],
                "team_roles": {
                    "code_leader": "attending_physician",
                    "chest_compressions": "nurse_1",
                    "airway_management": "respiratory_therapist",
                    "medications": "pharmacist"
                }
            }
        )
        
        print(f"🤝 Emergency care team coordinated (Workflow ID: {workflow_id})")
        
        # Step 3: Diagnostic copilot provides real-time guidance
        acls_guidance = {
            "action": "emergency_protocol",
            "protocol": "ACLS",
            "current_phase": "initial_assessment",
            "next_steps": [
                "verify_unresponsiveness",
                "check_pulse",
                "initiate_CPR",
                "attach_defibrillator"
            ],
            "medication_timing": {
                "epinephrine": "every_3_5_minutes",
                "amiodarone": "after_3rd_shock"
            }
        }
        
        await self.agents['diagnostic_copilot'].send_clinical_message(
            receiver_id='patient_data',
            message_type=HMCPMessageType.EMERGENCY,
            content=acls_guidance,
            patient_id=patient_id,
            urgency=ClinicalUrgency.EMERGENCY
        )
        
        print("📋 ACLS guidance provided to emergency team")
        print("✅ Emergency cardiac arrest response coordinated")
    
    async def example_4_discharge_planning(self):
        """Example 4: Discharge planning coordination"""
        print("\n🏠 Example 4: Discharge Planning Coordination")
        print("-" * 50)
        
        patient_id = "PATIENT_004"
        
        # Step 1: Initiate discharge planning workflow
        workflow_id = await self.agents['scheduling_agent'].coordinate_care_workflow(
            patient_id=patient_id,
            workflow_type="discharge_planning",
            participants=['patient_data', 'medical_knowledge', 'diagnostic_copilot'],
            care_plan={
                "discharge_date": "2024-12-17",
                "destination": "home_with_services",
                "required_assessments": [
                    "medical_clearance",
                    "social_work_assessment",
                    "pharmacy_reconciliation",
                    "home_safety_evaluation"
                ],
                "follow_up_appointments": [
                    {"specialty": "cardiology", "timeframe": "1_week"},
                    {"specialty": "primary_care", "timeframe": "2_weeks"}
                ]
            }
        )
        
        print(f"📋 Discharge planning initiated (Workflow ID: {workflow_id})")
        
        # Step 2: Patient data agent provides medication reconciliation
        medication_reconciliation = {
            "action": "medication_reconciliation",
            "home_medications": [
                {"name": "lisinopril", "dose": "10mg", "frequency": "daily"},
                {"name": "metoprolol", "dose": "25mg", "frequency": "twice_daily"},
                {"name": "atorvastatin", "dose": "40mg", "frequency": "daily"}
            ],
            "discontinued_medications": [
                {"name": "furosemide", "reason": "acute_therapy_only"}
            ],
            "new_medications": [
                {"name": "clopidogrel", "dose": "75mg", "frequency": "daily", "duration": "12_months"}
            ]
        }
        
        await self.agents['patient_data'].send_clinical_message(
            receiver_id='medical_knowledge',
            message_type=HMCPMessageType.COORDINATION,
            content=medication_reconciliation,
            patient_id=patient_id,
            urgency=ClinicalUrgency.ROUTINE
        )
        
        print("💊 Medication reconciliation completed")
        
        # Step 3: Medical knowledge agent provides discharge education
        discharge_education = {
            "action": "discharge_education",
            "education_topics": [
                "heart_healthy_diet",
                "medication_compliance",
                "activity_restrictions",
                "warning_signs"
            ],
            "warning_signs": [
                "chest_pain",
                "shortness_of_breath",
                "weight_gain_over_2_pounds",
                "dizziness_or_fainting"
            ],
            "emergency_contacts": {
                "cardiology_office": "555-HEART",
                "emergency": "911"
            }
        }
        
        await self.agents['medical_knowledge'].send_clinical_message(
            receiver_id='scheduling_agent',
            message_type=HMCPMessageType.COORDINATION,
            content=discharge_education,
            patient_id=patient_id,
            urgency=ClinicalUrgency.ROUTINE
        )
        
        print("📚 Discharge education materials prepared")
        
        # Step 4: Schedule follow-up appointments
        follow_up_scheduling = {
            "action": "schedule_follow_up",
            "appointments": [
                {
                    "specialty": "cardiology",
                    "provider": "Dr. Smith",
                    "date": "2024-12-24",
                    "time": "10:00 AM",
                    "type": "post_hospitalization_follow_up"
                },
                {
                    "specialty": "primary_care",
                    "provider": "Dr. Johnson",
                    "date": "2024-12-31",
                    "time": "2:00 PM",
                    "type": "routine_follow_up"
                }
            ]
        }
        
        await self.agents['scheduling_agent'].send_clinical_message(
            receiver_id='patient_data',
            message_type=HMCPMessageType.COORDINATION,
            content=follow_up_scheduling,
            patient_id=patient_id,
            urgency=ClinicalUrgency.ROUTINE
        )
        
        print("📅 Follow-up appointments scheduled")
        print("✅ Discharge planning coordination completed")
    
    async def example_5_lab_critical_value_notification(self):
        """Example 5: Critical lab value notification workflow"""
        print("\n🧪 Example 5: Critical Lab Value Notification")
        print("-" * 50)
        
        patient_id = "PATIENT_005"
        
        # Step 1: Patient data agent receives critical lab value
        critical_lab = {
            "action": "critical_lab_notification",
            "lab_test": "troponin_i",
            "value": 15.2,
            "normal_range": "0.0-0.04",
            "units": "ng/mL",
            "collection_time": "2024-12-15T14:30:00Z",
            "critical_threshold": 0.04,
            "fold_increase": 380
        }
        
        await self.agents['patient_data'].send_clinical_message(
            receiver_id='diagnostic_copilot',
            message_type=HMCPMessageType.NOTIFICATION,
            content=critical_lab,
            patient_id=patient_id,
            urgency=ClinicalUrgency.EMERGENCY
        )
        
        print("⚠️  Critical troponin value sent to diagnostic copilot")
        
        # Step 2: Diagnostic copilot assesses clinical significance
        clinical_assessment = {
            "action": "critical_value_assessment",
            "lab_value": "troponin_i_15.2",
            "clinical_significance": "acute_myocardial_infarction",
            "recommended_actions": [
                "immediate_cardiology_consultation",
                "serial_ecgs",
                "continuous_cardiac_monitoring",
                "antiplatelet_therapy_if_not_contraindicated"
            ],
            "time_sensitivity": "immediate"
        }
        
        await self.agents['diagnostic_copilot'].send_clinical_message(
            receiver_id='medical_knowledge',
            message_type=HMCPMessageType.REQUEST,
            content=clinical_assessment,
            patient_id=patient_id,
            urgency=ClinicalUrgency.EMERGENCY
        )
        
        print("🧠 Clinical assessment sent to medical knowledge agent")
        
        # Step 3: Medical knowledge agent provides treatment protocol
        treatment_protocol = {
            "action": "acute_mi_protocol",
            "protocol_type": "STEMI_protocol",
            "immediate_interventions": [
                "dual_antiplatelet_therapy",
                "anticoagulation",
                "beta_blocker_if_appropriate",
                "statin_therapy"
            ],
            "reperfusion_strategy": "primary_pci_preferred",
            "door_to_balloon_target": "90_minutes",
            "contraindications_check": [
                "active_bleeding",
                "recent_surgery",
                "severe_hypertension"
            ]
        }
        
        await self.agents['medical_knowledge'].send_clinical_message(
            receiver_id='scheduling_agent',
            message_type=HMCPMessageType.EMERGENCY,
            content=treatment_protocol,
            patient_id=patient_id,
            urgency=ClinicalUrgency.EMERGENCY
        )
        
        print("🏥 Acute MI protocol activated")
        
        # Step 4: Schedule urgent cardiac catheterization
        urgent_procedure = {
            "action": "urgent_procedure_scheduling",
            "procedure": "cardiac_catheterization",
            "urgency": "STAT",
            "target_time": "within_90_minutes",
            "team_notification": [
                "interventional_cardiology",
                "cath_lab_team",
                "cardiac_anesthesia"
            ],
            "preparation_orders": [
                "consent_for_pci",
                "type_and_screen",
                "nothing_by_mouth"
            ]
        }
        
        await self.agents['scheduling_agent'].send_clinical_message(
            receiver_id='patient_data',
            message_type=HMCPMessageType.EMERGENCY,
            content=urgent_procedure,
            patient_id=patient_id,
            urgency=ClinicalUrgency.EMERGENCY
        )
        
        print("⚡ Urgent cardiac catheterization scheduled")
        print("✅ Critical lab value notification workflow completed")
    
    async def display_metrics_summary(self):
        """Display metrics summary for all agents"""
        print("\n📊 HMCP Workflow Metrics Summary")
        print("=" * 60)
        
        total_messages_sent = 0
        total_messages_received = 0
        total_emergency_responses = 0
        total_workflows = 0
        
        for agent_id, agent in self.agents.items():
            metrics = agent.get_agent_metrics()
            health = agent.get_health_status()
            
            print(f"\n🤖 Agent: {agent_id}")
            print(f"   Role: {metrics['role']}")
            print(f"   Messages sent: {metrics['messages_sent']}")
            print(f"   Messages received: {metrics['messages_received']}")
            print(f"   Emergency responses: {metrics['emergency_responses']}")
            print(f"   Workflow completions: {metrics['workflow_completions']}")
            print(f"   Avg response time: {metrics['average_response_time']:.3f}s")
            print(f"   Health status: {health['status']}")
            
            total_messages_sent += metrics['messages_sent']
            total_messages_received += metrics['messages_received']
            total_emergency_responses += metrics['emergency_responses']
            total_workflows += metrics['workflow_completions']
        
        print(f"\n🏥 Overall Healthcare Team Performance:")
        print(f"   Total messages sent: {total_messages_sent}")
        print(f"   Total messages received: {total_messages_received}")
        print(f"   Total emergency responses: {total_emergency_responses}")
        print(f"   Total workflows completed: {total_workflows}")
        print(f"   Active agents: {len(self.agents)}")


async def main():
    """Run HMCP workflow examples"""
    print("🏥 HMCP Healthcare Workflow Examples")
    print("=" * 50)
    
    examples = HMCPWorkflowExamples()
    
    # Set up healthcare team
    await examples.setup_healthcare_team()
    
    # Wait a moment for initialization
    await asyncio.sleep(1)
    
    # Run workflow examples
    await examples.example_1_chest_pain_diagnosis()
    await asyncio.sleep(0.5)
    
    await examples.example_2_medication_interaction_check()
    await asyncio.sleep(0.5)
    
    await examples.example_3_emergency_cardiac_arrest()
    await asyncio.sleep(0.5)
    
    await examples.example_4_discharge_planning()
    await asyncio.sleep(0.5)
    
    await examples.example_5_lab_critical_value_notification()
    await asyncio.sleep(0.5)
    
    # Display final metrics
    await examples.display_metrics_summary()
    
    print("\n✅ All HMCP workflow examples completed successfully!")
    print("🏥 Healthcare Model Context Protocol demonstration finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples interrupted by user")
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print(f"❌ Error: {e}")