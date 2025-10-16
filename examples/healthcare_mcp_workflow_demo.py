#!/usr/bin/env python3
"""
Healthcare MCP Workflow Example

This demonstrates the exact workflow pattern shown in the sequence diagram:
Physician → Diagnosis Copilot → Patient Data Agent → Medical Knowledge Agent → Scheduling Agent

Following the Healthcare Model Context Protocol (HMCP) pattern.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

from vita_agents.agents.enhanced_hmcp_agent import EnhancedHMCPAgent, WorkflowState
from vita_agents.protocols.hmcp import (
    HMCPMessage, HMCPMessageType, ClinicalUrgency, HealthcareRole,
    PatientContext, ClinicalContext, SecurityContext
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareMCPWorkflowDemo:
    """
    Demonstrates the Healthcare Model Context Protocol workflow
    exactly as shown in the sequence diagram
    """
    
    def __init__(self):
        # Initialize agents for each role in the workflow
        self.agents = {
            "diagnosis_copilot": EnhancedHMCPAgent(
                agent_id="diagnosis_copilot_001",
                settings={
                    "agent_type": "diagnostic",
                    "healthcare_role": HealthcareRole.CLINICAL_SUPPORT,
                    "name": "Diagnosis Copilot"
                }
            ),
            "patient_data_agent": EnhancedHMCPAgent(
                agent_id="patient_data_agent_001", 
                settings={
                    "agent_type": "patient_data",
                    "healthcare_role": HealthcareRole.CLINICAL_SUPPORT,
                    "name": "Patient Data Agent"
                }
            ),
            "medical_knowledge_agent": EnhancedHMCPAgent(
                agent_id="medical_knowledge_agent_001",
                settings={
                    "agent_type": "medical_knowledge",
                    "healthcare_role": HealthcareRole.CLINICAL_SUPPORT,
                    "name": "Medical Knowledge Agent"
                }
            ),
            "scheduling_agent": EnhancedHMCPAgent(
                agent_id="scheduling_agent_001",
                settings={
                    "agent_type": "scheduling", 
                    "healthcare_role": HealthcareRole.ADMINISTRATION,
                    "name": "Scheduling Agent"
                }
            )
        }
        
        # Physician context (external user)
        self.physician_id = "physician_001"
        self.conversation_id = None
    
    async def demonstrate_workflow(self):
        """Run the complete workflow demonstration"""
        print("🏥 Healthcare MCP Workflow Demonstration")
        print("=" * 50)
        print("Following the sequence diagram pattern:")
        print("Physician → Diagnosis Copilot → Patient Data Agent → Medical Knowledge Agent → Scheduling Agent")
        print("")
        
        # Step 1: Physician provides initial symptoms to Diagnosis Copilot
        await self._step1_physician_provides_symptoms()
        
        # Step 2: Diagnosis Copilot requests patient records from Patient Data Agent
        await self._step2_request_patient_records()
        
        # Step 3: Patient Data Agent requests additional identifiers
        await self._step3_request_patient_identifiers()
        
        # Step 4: Diagnosis Copilot analyzes symptoms with patient records
        await self._step4_analyze_symptoms_with_records()
        
        # Step 5: Medical Knowledge Agent provides relevant articles
        await self._step5_request_medical_knowledge()
        
        # Step 6: Medical Knowledge Agent requests more details
        await self._step6_request_more_details()
        
        # Step 7: Medical Knowledge Agent provides recommendations
        await self._step7_provide_recommendations()
        
        # Step 8: Scheduling Agent handles appointment scheduling
        await self._step8_schedule_appointment()
        
        print("\n✅ Healthcare MCP Workflow completed successfully!")
        print(f"🔗 Conversation ID: {self.conversation_id}")
        
        # Show final conversation summary
        await self._show_conversation_summary()
    
    async def _step1_physician_provides_symptoms(self):
        """Step 1: Physician provides initial symptoms"""
        print("\n📋 Step 1: Physician provides initial symptoms to Diagnosis Copilot")
        
        # Create patient context
        patient_context = PatientContext(
            patient_id="patient_123456",
            mrn="MRN789012",
            demographics={
                "age": 45,
                "gender": "male",
                "date_of_birth": "1979-03-15"
            }
        )
        
        # Create clinical context
        clinical_context = ClinicalContext(
            chief_complaint="chest pain",
            urgency=ClinicalUrgency.URGENT,
            specialties_required=["cardiology"]
        )
        
        # Create initial message
        message = HMCPMessage(
            type=HMCPMessageType.REQUEST,
            sender_id=self.physician_id,
            receiver_id="diagnosis_copilot_001",
            content={
                "action": "provide_initial_symptoms",
                "symptoms": [
                    "chest pain",
                    "shortness of breath", 
                    "left arm numbness",
                    "nausea"
                ],
                "onset": "2 hours ago",
                "severity": "8/10",
                "patient_age": 45,
                "patient_gender": "male"
            },
            patient_context=patient_context,
            clinical_context=clinical_context,
            urgency=ClinicalUrgency.URGENT
        )
        
        # Process message
        response = await self.agents["diagnosis_copilot"].process_hmcp_message(message)
        self.conversation_id = response.content.get("conversation_id")
        
        print(f"   Physician: Provided symptoms - {message.content['symptoms']}")
        print(f"   Diagnosis Copilot: {response.content.get('message', 'Symptoms received')}")
        print(f"   Next Action: {response.content.get('next_action', 'continue')}")
    
    async def _step2_request_patient_records(self):
        """Step 2: Diagnosis Copilot requests patient records"""
        print("\n📄 Step 2: Diagnosis Copilot requests patient records from Patient Data Agent")
        
        message = HMCPMessage(
            type=HMCPMessageType.REQUEST,
            sender_id="diagnosis_copilot_001", 
            receiver_id="patient_data_agent_001",
            content={
                "action": "request_basic_patient_records",
                "patient_id": "patient_123456",
                "conversation_id": self.conversation_id,
                "urgency": "high",
                "required_data": ["medical_history", "current_medications", "allergies"]
            },
            urgency=ClinicalUrgency.URGENT
        )
        
        response = await self.agents["patient_data_agent"].process_hmcp_message(message)
        
        print(f"   Diagnosis Copilot: Requesting basic patient records")
        print(f"   Patient Data Agent: {response.content.get('message', 'Processing request')}")
        
        if response.content.get("status") == "missing_identifiers":
            print(f"   Missing Identifiers: {response.content.get('missing_identifiers', [])}")
    
    async def _step3_request_patient_identifiers(self):
        """Step 3: Request additional patient identifiers"""
        print("\n🆔 Step 3: Request additional patient identifiers")
        
        # Provide additional identifiers
        message = HMCPMessage(
            type=HMCPMessageType.REQUEST,
            sender_id="diagnosis_copilot_001",
            receiver_id="patient_data_agent_001", 
            content={
                "action": "provide_additional_identifiers",
                "conversation_id": self.conversation_id,
                "identifiers": {
                    "date_of_birth": "1979-03-15",
                    "medical_record_number": "MRN789012",
                    "social_security_number": "XXX-XX-1234",
                    "phone_number": "(555) 123-4567"
                }
            }
        )
        
        response = await self.agents["patient_data_agent"].process_hmcp_message(message)
        
        print(f"   Diagnosis Copilot: Provided additional patient identifiers")
        print(f"   Patient Data Agent: {response.content.get('message', 'Identifiers received')}")
        
        # Now retrieve records with complete identifiers
        message2 = HMCPMessage(
            type=HMCPMessageType.REQUEST,
            sender_id="diagnosis_copilot_001",
            receiver_id="patient_data_agent_001",
            content={
                "action": "request_basic_patient_records",
                "conversation_id": self.conversation_id,
                "patient_id": "patient_123456"
            }
        )
        
        response2 = await self.agents["patient_data_agent"].process_hmcp_message(message2)
        print(f"   Patient Data Agent: Records retrieved successfully")
        if "patient_records" in response2.content:
            records = response2.content["patient_records"]
            print(f"   Medical History: {records.get('medical_history', [])}")
            print(f"   Current Medications: {records.get('current_medications', [])}")
            print(f"   Allergies: {records.get('allergies', [])}")
    
    async def _step4_analyze_symptoms_with_records(self):
        """Step 4: Analyze symptoms with patient records"""
        print("\n🔍 Step 4: Diagnosis Copilot analyzes symptoms with patient records")
        
        message = HMCPMessage(
            type=HMCPMessageType.REQUEST,
            sender_id="diagnosis_copilot_001",
            receiver_id="diagnosis_copilot_001",  # Self-processing
            content={
                "action": "analyze_symptoms_with_records",
                "conversation_id": self.conversation_id,
                "symptoms": ["chest pain", "shortness of breath", "left arm numbness", "nausea"],
                "patient_data": {
                    "medical_history": ["hypertension", "diabetes_type_2"],
                    "current_medications": ["metformin", "lisinopril"],
                    "allergies": ["penicillin"],
                    "age": 45,
                    "gender": "male"
                }
            }
        )
        
        response = await self.agents["diagnosis_copilot"].process_hmcp_message(message)
        
        print(f"   Diagnosis Copilot: Analyzing symptoms with patient records")
        if "analysis_result" in response.content:
            analysis = response.content["analysis_result"]
            print(f"   Primary Symptoms: {analysis.get('primary_symptoms', [])}")
            print(f"   Severity: {analysis.get('symptom_severity', 'unknown')}")
            print(f"   Preliminary Assessment: {analysis.get('preliminary_assessment', 'unknown')}")
            print(f"   Recommended Tests: {analysis.get('recommended_tests', [])}")
    
    async def _step5_request_medical_knowledge(self):
        """Step 5: Request relevant medical knowledge"""
        print("\n📚 Step 5: Medical Knowledge Agent provides relevant articles")
        
        message = HMCPMessage(
            type=HMCPMessageType.REQUEST,
            sender_id="diagnosis_copilot_001",
            receiver_id="medical_knowledge_agent_001",
            content={
                "action": "request_relevant_articles",
                "conversation_id": self.conversation_id,
                "symptoms": ["chest pain", "shortness of breath", "left arm numbness"],
                "patient_demographics": {"age": 45, "gender": "male"},
                "medical_history": ["hypertension", "diabetes_type_2"]
            }
        )
        
        response = await self.agents["medical_knowledge_agent"].process_hmcp_message(message)
        
        print(f"   Diagnosis Copilot: Requesting relevant medical articles")
        print(f"   Medical Knowledge Agent: {response.content.get('message', 'Articles found')}")
        
        if "relevant_articles" in response.content:
            articles = response.content["relevant_articles"]
            for i, article in enumerate(articles, 1):
                print(f"   Article {i}: {article.get('title', 'Unknown')}")
                print(f"     Relevance: {article.get('relevance_score', 0):.2f}")
                print(f"     Source: {article.get('source', 'Unknown')}")
    
    async def _step6_request_more_details(self):
        """Step 6: Request more specific symptom details"""
        print("\n🔍 Step 6: Medical Knowledge Agent requests more specific symptoms")
        
        message = HMCPMessage(
            type=HMCPMessageType.REQUEST,
            sender_id="medical_knowledge_agent_001",
            receiver_id="diagnosis_copilot_001",
            content={
                "action": "request_more_specific_symptoms",
                "conversation_id": self.conversation_id,
                "reason": "Need additional details for accurate differential diagnosis"
            }
        )
        
        response = await self.agents["diagnosis_copilot"].process_hmcp_message(message)
        
        print(f"   Medical Knowledge Agent: Requesting more specific symptom details")
        print(f"   Requested Information: {response.content.get('requested_information', [])}")
        
        # Provide additional details
        details_message = HMCPMessage(
            type=HMCPMessageType.RESPONSE,
            sender_id=self.physician_id,
            receiver_id="medical_knowledge_agent_001",
            content={
                "action": "provide_symptom_details",
                "conversation_id": self.conversation_id,
                "symptom_details": {
                    "symptom_onset": "sudden, 2 hours ago",
                    "symptom_duration": "continuous",
                    "severity_scale": "8/10",
                    "aggravating_factors": ["physical exertion", "deep breathing"],
                    "relieving_factors": ["rest", "sitting position"],
                    "associated_symptoms": ["sweating", "anxiety"]
                }
            }
        )
        
        print(f"   Physician: Provided additional symptom details")
        print(f"   Onset: sudden, 2 hours ago")
        print(f"   Severity: 8/10")
        print(f"   Aggravating factors: physical exertion, deep breathing")
    
    async def _step7_provide_recommendations(self):
        """Step 7: Provide clinical recommendations"""
        print("\n💡 Step 7: Medical Knowledge Agent provides clinical recommendations")
        
        message = HMCPMessage(
            type=HMCPMessageType.REQUEST,
            sender_id="diagnosis_copilot_001",
            receiver_id="medical_knowledge_agent_001",
            content={
                "action": "provide_clinical_recommendations",
                "conversation_id": self.conversation_id,
                "clinical_scenario": {
                    "symptoms": ["chest pain", "shortness of breath", "left arm numbness"],
                    "severity": "8/10",
                    "patient_profile": {
                        "age": 45,
                        "gender": "male", 
                        "medical_history": ["hypertension", "diabetes_type_2"]
                    },
                    "urgency": "high"
                }
            }
        )
        
        response = await self.agents["medical_knowledge_agent"].process_hmcp_message(message)
        
        print(f"   Medical Knowledge Agent: Providing clinical recommendations")
        
        if "clinical_recommendations" in response.content:
            recommendations = response.content["clinical_recommendations"]
            for i, rec in enumerate(recommendations, 1):
                print(f"   Recommendation {i}: {rec.get('recommendation', 'Unknown')}")
                print(f"     Priority: {rec.get('priority', 'unknown')}")
                print(f"     Rationale: {rec.get('rationale', 'unknown')}")
    
    async def _step8_schedule_appointment(self):
        """Step 8: Schedule follow-up appointment"""
        print("\n📅 Step 8: Scheduling Agent handles appointment scheduling")
        
        # Request to schedule appointment
        message = HMCPMessage(
            type=HMCPMessageType.REQUEST,
            sender_id="diagnosis_copilot_001",
            receiver_id="scheduling_agent_001",
            content={
                "action": "request_to_schedule_appointment",
                "conversation_id": self.conversation_id,
                "appointment_type": "cardiology_consultation",
                "urgency": "high",
                "patient_id": "patient_123456",
                "specialty_required": "cardiology"
            }
        )
        
        response1 = await self.agents["scheduling_agent"].process_hmcp_message(message)
        print(f"   Diagnosis Copilot: Requesting cardiology consultation appointment")
        print(f"   Scheduling Agent: {response1.content.get('message', 'Checking availability')}")
        
        # Report patient availability
        message2 = HMCPMessage(
            type=HMCPMessageType.REQUEST,
            sender_id="scheduling_agent_001",
            receiver_id="scheduling_agent_001",
            content={
                "action": "report_patient_availability",
                "conversation_id": self.conversation_id,
                "availability": {
                    "preferred_days": ["Monday", "Tuesday", "Wednesday"],
                    "preferred_times": ["morning", "afternoon"],
                    "urgent_preference": True
                }
            }
        )
        
        response2 = await self.agents["scheduling_agent"].process_hmcp_message(message2)
        print(f"   Patient: Availability provided (urgent preference)")
        
        # Finalize appointment
        message3 = HMCPMessage(
            type=HMCPMessageType.REQUEST,
            sender_id="scheduling_agent_001",
            receiver_id="scheduling_agent_001",
            content={
                "action": "provide_preferences",
                "conversation_id": self.conversation_id,
                "preferences": {
                    "provider": "Dr. Smith, Cardiology",
                    "appointment_date": "2024-10-25",
                    "appointment_time": "10:00 AM",
                    "location": "Main Hospital"
                }
            }
        )
        
        response3 = await self.agents["scheduling_agent"].process_hmcp_message(message3)
        print(f"   Scheduling Agent: {response3.content.get('confirmation', 'Appointment scheduled')}")
        
        if "appointment_details" in response3.content:
            details = response3.content["appointment_details"]
            print(f"   Appointment ID: {details.get('appointment_id', 'unknown')}")
            print(f"   Date/Time: {details.get('appointment_date', 'unknown')} at {details.get('appointment_time', 'unknown')}")
            print(f"   Provider: {details.get('provider', 'unknown')}")
            print(f"   Location: {details.get('location', 'unknown')}")
    
    async def _show_conversation_summary(self):
        """Show final conversation summary"""
        print("\n📊 Final Conversation Summary")
        print("=" * 40)
        
        # Get summary from diagnosis copilot (main orchestrator)
        summary = self.agents["diagnosis_copilot"].get_conversation_summary(self.conversation_id)
        
        if summary:
            print(f"Conversation ID: {summary.get('conversation_id', 'unknown')}")
            print(f"Patient ID: {summary.get('patient_id', 'unknown')}")
            print(f"Final State: {summary.get('state', 'unknown')}")
            print(f"Duration: {summary.get('duration_minutes', 0):.1f} minutes")
            print(f"Symptoms Recorded: {', '.join(summary.get('symptoms', []))}")
            print(f"Information Gathered: {', '.join(summary.get('information_gathered', []))}")
            print(f"Participating Agents: {', '.join(summary.get('participating_agents', []))}")
            print(f"Recommendations Count: {len(summary.get('recommendations', []))}")
            print(f"Scheduled Appointments: {len(summary.get('scheduled_appointments', []))}")
        
        print("\n🎯 Workflow demonstrates successful Healthcare MCP implementation!")


async def main():
    """Run the Healthcare MCP workflow demonstration"""
    demo = HealthcareMCPWorkflowDemo()
    await demo.demonstrate_workflow()


if __name__ == "__main__":
    asyncio.run(main())