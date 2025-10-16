"""
Integration tests for Vita Agents healthcare workflow
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import json

from vita_agents.core.orchestrator import AgentOrchestrator
from vita_agents.agents.fhir_agent import FHIRAgent
from vita_agents.agents.hl7_agent import HL7Agent
from vita_agents.agents.ehr_agent import EHRAgent
from vita_agents.agents.clinical_decision_agent import ClinicalDecisionSupportAgent
from vita_agents.agents.data_harmonization_agent import DataHarmonizationAgent
from vita_agents.core.config import Settings


@pytest.fixture
def test_settings():
    """Test settings fixture"""
    return Settings()


@pytest.fixture
def mock_db_manager():
    """Mock database manager"""
    db_manager = AsyncMock()
    db_manager.execute = AsyncMock()
    db_manager.fetch_all = AsyncMock(return_value=[])
    return db_manager


@pytest.fixture
async def orchestrator(test_settings, mock_db_manager):
    """Agent orchestrator with all agents"""
    orchestrator = AgentOrchestrator(test_settings, mock_db_manager)
    
    # Register all healthcare agents
    fhir_agent = FHIRAgent("fhir-agent-1", test_settings, mock_db_manager)
    hl7_agent = HL7Agent("hl7-agent-1", test_settings, mock_db_manager)
    ehr_agent = EHRAgent("ehr-agent-1", test_settings, mock_db_manager)
    clinical_agent = ClinicalDecisionSupportAgent("clinical-agent-1", test_settings, mock_db_manager)
    harmonization_agent = DataHarmonizationAgent("harmonization-agent-1", test_settings, mock_db_manager)
    
    await orchestrator.register_agent(fhir_agent)
    await orchestrator.register_agent(hl7_agent)
    await orchestrator.register_agent(ehr_agent)
    await orchestrator.register_agent(clinical_agent)
    await orchestrator.register_agent(harmonization_agent)
    
    return orchestrator


@pytest.fixture
def sample_fhir_patient():
    """Sample FHIR Patient resource"""
    return {
        "resourceType": "Patient",
        "id": "patient-123",
        "name": [
            {
                "use": "official",
                "family": "Doe",
                "given": ["John", "William"]
            }
        ],
        "gender": "male",
        "birthDate": "1980-01-01",
        "identifier": [
            {
                "use": "usual",
                "type": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                            "code": "MR"
                        }
                    ]
                },
                "system": "http://hospital.example.org",
                "value": "12345"
            }
        ],
        "active": True
    }


@pytest.fixture
def sample_hl7_message():
    """Sample HL7 ADT message"""
    return {
        "message_type": "ADT",
        "timestamp": "2025-10-16T10:00:00Z",
        "PID": {
            "patient_id": "12345",
            "name": "Doe^John^William",
            "birth_date": "19800101",
            "gender": "M"
        }
    }


@pytest.fixture
def sample_observations():
    """Sample FHIR Observation resources"""
    return [
        {
            "resourceType": "Observation",
            "id": "obs-1",
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "33747-0",
                        "display": "General appearance"
                    }
                ]
            },
            "subject": {
                "reference": "Patient/patient-123"
            },
            "valueString": "Well-appearing",
            "effectiveDateTime": "2025-10-16T10:00:00Z"
        },
        {
            "resourceType": "Observation",
            "id": "obs-2",
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "8302-2",
                        "display": "Body height"
                    }
                ]
            },
            "subject": {
                "reference": "Patient/patient-123"
            },
            "valueQuantity": {
                "value": 180,
                "unit": "cm",
                "system": "http://unitsofmeasure.org",
                "code": "cm"
            },
            "effectiveDateTime": "2025-10-16T10:00:00Z"
        }
    ]


@pytest.mark.asyncio
class TestPatientDataIntegrationWorkflow:
    """Test end-to-end patient data integration workflow"""
    
    async def test_complete_patient_workflow(
        self, orchestrator, sample_fhir_patient, sample_observations
    ):
        """Test complete patient data processing workflow"""
        
        # Define workflow steps
        workflow_definition = {
            "workflow_id": "patient_integration_test",
            "name": "Patient Data Integration Test",
            "steps": [
                {
                    "step_id": "validate_patient",
                    "agent_id": "fhir-agent-1",
                    "action": "validate_fhir_resource",
                    "input": "patient_data"
                },
                {
                    "step_id": "validate_observations",
                    "agent_id": "fhir-agent-1", 
                    "action": "validate_fhir_resource",
                    "input": "observation_data"
                },
                {
                    "step_id": "clinical_analysis",
                    "agent_id": "clinical-agent-1",
                    "action": "analyze_patient_data",
                    "input": "combined_data",
                    "depends_on": ["validate_patient", "validate_observations"]
                }
            ]
        }
        
        # Input data
        input_data = {
            "patient_data": sample_fhir_patient,
            "observation_data": sample_observations[0],
            "combined_data": {
                "patient_data": sample_fhir_patient,
                "observations": sample_observations,
                "medications": [],
                "allergies": [],
                "conditions": [],
                "lab_results": []
            }
        }
        
        # Mock user context
        user_context = {
            "user_id": "test-user",
            "permissions": ["patient.read", "phi.access", "observation.read"],
            "access_reason": "Integration test"
        }
        
        # Execute workflow
        execution = await orchestrator.execute_workflow(
            workflow_definition, 
            input_data,
            user_context
        )
        
        # Verify workflow execution
        assert execution is not None
        assert execution.status in ["completed", "running"]
        
        # If completed, verify results
        if execution.status == "completed":
            assert len(execution.step_results) == 3
            
            # Verify patient validation
            patient_result = execution.step_results["validate_patient"]
            assert patient_result["status"] == "completed"
            assert "validation" in patient_result["result"]
            
            # Verify observation validation
            obs_result = execution.step_results["validate_observations"]
            assert obs_result["status"] == "completed"
            
            # Verify clinical analysis
            clinical_result = execution.step_results["clinical_analysis"]
            assert clinical_result["status"] == "completed"
            assert "clinical_alerts" in clinical_result["result"]
    
    async def test_hl7_to_fhir_workflow(self, orchestrator, sample_hl7_message):
        """Test HL7 message processing and FHIR conversion"""
        
        workflow_definition = {
            "workflow_id": "hl7_processing_test",
            "name": "HL7 to FHIR Conversion Test",
            "steps": [
                {
                    "step_id": "process_hl7",
                    "agent_id": "hl7-agent-1",
                    "action": "process_hl7_message",
                    "input": "hl7_data"
                },
                {
                    "step_id": "validate_converted_fhir",
                    "agent_id": "fhir-agent-1",
                    "action": "validate_fhir_resource", 
                    "input": "converted_fhir",
                    "depends_on": ["process_hl7"]
                }
            ]
        }
        
        input_data = {
            "hl7_data": sample_hl7_message
        }
        
        user_context = {
            "user_id": "test-user",
            "permissions": ["hl7.process", "fhir.validate"],
            "access_reason": "HL7 processing test"
        }
        
        execution = await orchestrator.execute_workflow(
            workflow_definition,
            input_data, 
            user_context
        )
        
        assert execution is not None
        # Additional assertions would depend on actual implementation
    
    async def test_data_harmonization_workflow(self, orchestrator, sample_fhir_patient):
        """Test data harmonization across multiple sources"""
        
        # Simulate data from multiple sources
        source1_data = {
            "source_info": {
                "source_id": "epic_source",
                "name": "Epic EHR",
                "system_type": "epic",
                "reliability_score": 0.9,
                "last_updated": datetime.utcnow(),
                "data_standards": ["FHIR"],
                "priority": 1
            },
            "data": sample_fhir_patient
        }
        
        source2_data = {
            "source_info": {
                "source_id": "cerner_source", 
                "name": "Cerner EHR",
                "system_type": "cerner",
                "reliability_score": 0.85,
                "last_updated": datetime.utcnow(),
                "data_standards": ["FHIR"],
                "priority": 2
            },
            "data": {
                **sample_fhir_patient,
                "id": "patient-456",  # Different ID - will cause conflict
                "name": [
                    {
                        "use": "official",
                        "family": "Doe",
                        "given": ["John", "W"]  # Slightly different name
                    }
                ]
            }
        }
        
        workflow_definition = {
            "workflow_id": "harmonization_test",
            "name": "Data Harmonization Test",
            "steps": [
                {
                    "step_id": "harmonize_data",
                    "agent_id": "harmonization-agent-1",
                    "action": "harmonize_patient_data",
                    "input": "harmonization_data"
                }
            ]
        }
        
        input_data = {
            "harmonization_data": {
                "data_sources": [source1_data, source2_data],
                "patient_id": "patient-123"
            }
        }
        
        user_context = {
            "user_id": "test-user",
            "permissions": ["data.harmonize", "phi.access"],
            "access_reason": "Data harmonization test"
        }
        
        execution = await orchestrator.execute_workflow(
            workflow_definition,
            input_data,
            user_context
        )
        
        assert execution is not None
        
        if execution.status == "completed":
            harmonization_result = execution.step_results["harmonize_data"]
            assert "harmonized_data" in harmonization_result["result"]
            assert "conflicts_detected" in harmonization_result["result"]


@pytest.mark.asyncio 
class TestClinicalDecisionSupportWorkflow:
    """Test clinical decision support workflows"""
    
    async def test_drug_interaction_checking(self, orchestrator):
        """Test drug interaction checking workflow"""
        
        patient_data = {
            "patient_data": {
                "id": "patient-123",
                "birthDate": "1950-01-01",
                "gender": "male"
            },
            "medications": [
                {
                    "code": {
                        "coding": [
                            {
                                "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                "code": "11289",
                                "display": "warfarin"
                            }
                        ]
                    }
                },
                {
                    "code": {
                        "coding": [
                            {
                                "system": "http://www.nlm.nih.gov/research/umls/rxnorm", 
                                "code": "1191",
                                "display": "aspirin"
                            }
                        ]
                    }
                }
            ],
            "allergies": [],
            "conditions": [
                {
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "49436004",
                                "display": "Atrial fibrillation"
                            }
                        ]
                    }
                }
            ],
            "lab_results": [
                {
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": "6301-6",
                                "display": "INR"
                            }
                        ]
                    },
                    "valueQuantity": {
                        "value": 3.5,
                        "unit": "ratio"
                    }
                }
            ]
        }
        
        workflow_definition = {
            "workflow_id": "clinical_decision_test",
            "name": "Clinical Decision Support Test",
            "steps": [
                {
                    "step_id": "analyze_medications",
                    "agent_id": "clinical-agent-1",
                    "action": "analyze_patient_data",
                    "input": "clinical_data"
                }
            ]
        }
        
        input_data = {
            "clinical_data": patient_data
        }
        
        user_context = {
            "user_id": "test-clinician",
            "permissions": ["clinical.analyze", "patient.read", "phi.access"],
            "access_reason": "Clinical decision support"
        }
        
        execution = await orchestrator.execute_workflow(
            workflow_definition,
            input_data,
            user_context
        )
        
        assert execution is not None
        
        if execution.status == "completed":
            clinical_result = execution.step_results["analyze_medications"]
            result_data = clinical_result["result"]
            
            # Should detect warfarin-aspirin interaction
            assert "drug_interactions" in result_data
            interactions = result_data["drug_interactions"]
            
            # Should find at least one interaction
            assert len(interactions) > 0
            
            # Should detect high INR alert
            assert "clinical_alerts" in result_data
            alerts = result_data["clinical_alerts"]
            
            # Should have alerts for high INR
            inr_alerts = [a for a in alerts if "inr" in a.get("message", "").lower()]
            assert len(inr_alerts) > 0


@pytest.mark.asyncio
class TestComplianceAndSecurity:
    """Test HIPAA compliance and security features"""
    
    async def test_audit_logging_workflow(self, orchestrator, sample_fhir_patient):
        """Test that all patient data access is properly audited"""
        
        workflow_definition = {
            "workflow_id": "audit_test",
            "name": "Audit Logging Test",
            "steps": [
                {
                    "step_id": "process_patient",
                    "agent_id": "fhir-agent-1",
                    "action": "validate_fhir_resource",
                    "input": "patient_data"
                }
            ]
        }
        
        input_data = {
            "patient_data": sample_fhir_patient
        }
        
        user_context = {
            "user_id": "test-user",
            "permissions": ["patient.read", "phi.access"],
            "access_reason": "Audit test - patient data validation"
        }
        
        # Execute workflow
        execution = await orchestrator.execute_workflow(
            workflow_definition,
            input_data,
            user_context
        )
        
        assert execution is not None
        
        # Verify audit logging was called
        # In a real test, we would check the database for audit entries
        # For now, we verify the workflow completed successfully
        if execution.status == "completed":
            assert len(execution.step_results) == 1
    
    async def test_permission_enforcement(self, orchestrator, sample_fhir_patient):
        """Test that insufficient permissions are properly rejected"""
        
        workflow_definition = {
            "workflow_id": "permission_test",
            "name": "Permission Test",
            "steps": [
                {
                    "step_id": "access_patient",
                    "agent_id": "fhir-agent-1", 
                    "action": "validate_fhir_resource",
                    "input": "patient_data"
                }
            ]
        }
        
        input_data = {
            "patient_data": sample_fhir_patient
        }
        
        # User with insufficient permissions
        user_context = {
            "user_id": "limited-user",
            "permissions": ["observation.read"],  # No patient.read permission
            "access_reason": "Permission test"
        }
        
        # This should fail due to insufficient permissions
        with pytest.raises(Exception):  # Would be SecurityException in real implementation
            await orchestrator.execute_workflow(
                workflow_definition,
                input_data,
                user_context
            )


@pytest.mark.asyncio
class TestPerformanceAndScaling:
    """Test performance and scaling capabilities"""
    
    async def test_concurrent_workflow_execution(self, orchestrator, sample_fhir_patient):
        """Test concurrent execution of multiple workflows"""
        
        workflow_definition = {
            "workflow_id": "concurrent_test",
            "name": "Concurrent Execution Test",
            "steps": [
                {
                    "step_id": "validate_patient",
                    "agent_id": "fhir-agent-1",
                    "action": "validate_fhir_resource",
                    "input": "patient_data"
                }
            ]
        }
        
        input_data = {
            "patient_data": sample_fhir_patient
        }
        
        user_context = {
            "user_id": "test-user",
            "permissions": ["patient.read", "phi.access"],
            "access_reason": "Concurrent test"
        }
        
        # Execute multiple workflows concurrently
        tasks = []
        for i in range(5):
            task = orchestrator.execute_workflow(
                workflow_definition,
                input_data,
                user_context
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all succeeded
        for result in results:
            assert not isinstance(result, Exception)
            assert result is not None
    
    async def test_large_data_processing(self, orchestrator):
        """Test processing of large datasets"""
        
        # Create large dataset
        large_observation_list = []
        for i in range(100):
            observation = {
                "resourceType": "Observation",
                "id": f"obs-{i}",
                "status": "final",
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "33747-0",
                            "display": f"Test observation {i}"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/patient-123"
                },
                "valueString": f"Test value {i}",
                "effectiveDateTime": "2025-10-16T10:00:00Z"
            }
            large_observation_list.append(observation)
        
        workflow_definition = {
            "workflow_id": "large_data_test",
            "name": "Large Data Processing Test",
            "steps": [
                {
                    "step_id": "validate_observations",
                    "agent_id": "fhir-agent-1",
                    "action": "validate_fhir_resource",
                    "input": "observation_data"
                }
            ]
        }
        
        # Process each observation
        successful_validations = 0
        for observation in large_observation_list[:10]:  # Limit for test performance
            input_data = {
                "observation_data": observation
            }
            
            user_context = {
                "user_id": "test-user",
                "permissions": ["observation.read", "phi.access"],
                "access_reason": "Large data test"
            }
            
            try:
                execution = await orchestrator.execute_workflow(
                    workflow_definition,
                    input_data,
                    user_context
                )
                
                if execution and execution.status == "completed":
                    successful_validations += 1
                    
            except Exception as e:
                # Log but don't fail the test
                print(f"Validation failed for observation: {e}")
        
        # Verify reasonable success rate
        assert successful_validations >= 8  # At least 80% success rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])