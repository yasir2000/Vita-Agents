"""
Comprehensive tests for the Enhanced Clinical Decision Support Agent.

Tests all major capabilities including drug interactions, allergy screening,
clinical guidelines, lab interpretation, and risk assessment.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any

from vita_agents.agents.enhanced_clinical_decision_agent import (
    EnhancedClinicalDecisionSupportAgent,
    ClinicalAnalysisRequest,
    DrugInteractionRequest,
    AllergyScreeningRequest,
    LabInterpretationRequest
)
from vita_agents.core.agent import TaskRequest


class TestEnhancedClinicalDecisionSupportAgent:
    """Test suite for Enhanced Clinical Decision Support Agent."""
    
    @pytest.fixture
    async def agent(self):
        """Create agent instance for testing."""
        agent = EnhancedClinicalDecisionSupportAgent()
        await agent._on_start()
        yield agent
        await agent._on_stop()
    
    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data for testing."""
        return {
            "patient_id": "test-patient-001",
            "age": 65,
            "sex": "male",
            "weight": 80.0,
            "height": 175.0,
            "conditions": ["diabetes", "hypertension", "hyperlipidemia"],
            "medical_history": ["myocardial_infarction"],
            "vital_signs": {
                "blood_pressure": "140/90",
                "heart_rate": 72,
                "temperature": 98.6
            }
        }
    
    @pytest.fixture
    def sample_medications(self):
        """Sample medications for testing."""
        return ["warfarin", "aspirin", "metformin", "lisinopril", "simvastatin"]
    
    @pytest.fixture
    def sample_allergies(self):
        """Sample allergies for testing."""
        return ["penicillin", "sulfa"]
    
    @pytest.fixture
    def sample_lab_results(self):
        """Sample lab results for testing."""
        return [
            {
                "test_name": "glucose",
                "value": 150.0,
                "unit": "mg/dL",
                "reference_range": "70-100",
                "collected_date": datetime.utcnow().isoformat(),
                "result_date": datetime.utcnow().isoformat()
            },
            {
                "test_name": "hemoglobin_a1c",
                "value": 8.5,
                "unit": "%",
                "reference_range": "<7.0",
                "collected_date": datetime.utcnow().isoformat(),
                "result_date": datetime.utcnow().isoformat()
            },
            {
                "test_name": "ldl_cholesterol",
                "value": 160.0,
                "unit": "mg/dL",
                "reference_range": "<100",
                "collected_date": datetime.utcnow().isoformat(),
                "result_date": datetime.utcnow().isoformat()
            }
        ]
    
    @pytest.mark.asyncio
    async def test_comprehensive_clinical_analysis(
        self,
        agent,
        sample_patient_data,
        sample_medications,
        sample_allergies,
        sample_lab_results
    ):
        """Test comprehensive clinical analysis."""
        request = ClinicalAnalysisRequest(
            patient_id=sample_patient_data["patient_id"],
            patient_data=sample_patient_data,
            medications=sample_medications,
            allergies=sample_allergies,
            lab_results=sample_lab_results,
            clinical_contexts=["diabetes", "cardiovascular"],
            include_lab_interpretation=True,
            include_risk_assessment=True,
            include_guidelines=True
        )
        
        result = await agent.comprehensive_clinical_analysis(request)
        
        # Verify structure
        assert "drug_interactions" in result
        assert "allergy_alerts" in result
        assert "clinical_guidelines" in result
        assert "lab_interpretations" in result
        assert "lab_alerts" in result
        assert "risk_assessment" in result
        assert "analysis_metadata" in result
        assert "summary" in result
        
        # Verify metadata
        metadata = result["analysis_metadata"]
        assert metadata["agent_version"] == agent.version
        assert "analysis_timestamp" in metadata
        assert "processing_time_ms" in metadata
        assert "analysis_id" in metadata
        
        # Verify drug interactions are detected
        drug_interactions = result["drug_interactions"]
        assert len(drug_interactions) > 0
        
        # Should detect warfarin-aspirin interaction
        warfarin_aspirin_interaction = any(
            "warfarin" in interaction["triggered_by"].lower() and
            "aspirin" in interaction["description"].lower()
            for interaction in drug_interactions
        )
        assert warfarin_aspirin_interaction, "Should detect warfarin-aspirin interaction"
        
        # Verify lab interpretations
        lab_interpretations = result["lab_interpretations"]
        assert len(lab_interpretations) == 3  # glucose, A1c, LDL
        
        # Verify abnormal values detected
        glucose_interp = next(
            (interp for interp in lab_interpretations if interp["test_name"] == "glucose"),
            None
        )
        assert glucose_interp is not None
        assert glucose_interp["status"] == "high"
        
        a1c_interp = next(
            (interp for interp in lab_interpretations if interp["test_name"] == "hemoglobin_a1c"),
            None
        )
        assert a1c_interp is not None
        assert a1c_interp["status"] == "high"
        
        # Verify clinical guidelines
        guidelines = result["clinical_guidelines"]
        assert len(guidelines) > 0
        
        # Should have diabetes and hypertension guidelines
        diabetes_guideline = any(
            "diabetes" in guideline["title"].lower() or
            "glucose" in guideline["title"].lower() or
            "a1c" in guideline["title"].lower()
            for guideline in guidelines
        )
        assert diabetes_guideline, "Should have diabetes-related guidelines"
        
        # Verify summary
        summary = result["summary"]
        assert summary["total_alerts"] > 0
        assert summary["drug_interaction_count"] > 0
        assert summary["recommendations_count"] > 0
    
    @pytest.mark.asyncio
    async def test_drug_interaction_checking(self, agent, sample_medications):
        """Test drug interaction checking."""
        request = DrugInteractionRequest(
            patient_id="test-patient-001",
            medications=sample_medications
        )
        
        result = await agent.check_drug_interactions(request)
        
        # Verify structure
        assert "patient_id" in result
        assert "medications" in result
        assert "interaction_count" in result
        assert "interactions" in result
        assert "recommendations" in result
        
        # Should detect interactions
        assert result["interaction_count"] > 0
        
        # Should detect warfarin-aspirin interaction
        interactions = result["interactions"]
        warfarin_aspirin = any(
            "warfarin" in interaction["triggered_by"].lower() and
            "aspirin" in interaction["description"].lower()
            for interaction in interactions
        )
        assert warfarin_aspirin, "Should detect warfarin-aspirin interaction"
        
        # Verify interaction details
        first_interaction = interactions[0]
        assert "alert_id" in first_interaction
        assert "severity" in first_interaction
        assert "description" in first_interaction
        assert "recommendation" in first_interaction
        assert first_interaction["severity"] in ["low", "moderate", "high", "critical"]
    
    @pytest.mark.asyncio
    async def test_new_medication_interaction(self, agent):
        """Test checking interactions with a new medication."""
        request = DrugInteractionRequest(
            patient_id="test-patient-001",
            medications=["warfarin", "metformin"],
            new_medication="aspirin"
        )
        
        result = await agent.check_drug_interactions(request)
        
        # Verify structure
        assert "new_medication" in result
        assert result["new_medication"] == "aspirin"
        
        # Should detect warfarin-aspirin interaction
        assert result["interaction_count"] > 0
        
        interactions = result["interactions"]
        aspirin_interaction = any(
            "aspirin" in interaction["triggered_by"].lower()
            for interaction in interactions
        )
        assert aspirin_interaction, "Should detect interaction involving aspirin"
    
    @pytest.mark.asyncio
    async def test_allergy_screening(self, agent, sample_allergies):
        """Test allergy screening."""
        request = AllergyScreeningRequest(
            patient_id="test-patient-001",
            patient_allergies=sample_allergies,
            proposed_medications=["amoxicillin", "sulfamethoxazole", "ciprofloxacin"]
        )
        
        result = await agent.screen_allergies(request)
        
        # Verify structure
        assert "patient_id" in result
        assert "patient_allergies" in result
        assert "proposed_medications" in result
        assert "allergy_alert_count" in result
        assert "allergy_alerts" in result
        assert "safe_medications" in result
        assert "recommendations" in result
        
        # Should detect allergy alerts
        assert result["allergy_alert_count"] > 0
        
        # Should detect penicillin-amoxicillin cross-reactivity
        alerts = result["allergy_alerts"]
        penicillin_alert = any(
            "penicillin" in alert["description"].lower() and
            "amoxicillin" in alert["triggered_by"].lower()
            for alert in alerts
        )
        assert penicillin_alert, "Should detect penicillin-amoxicillin cross-reactivity"
        
        # Should identify safe medications
        safe_meds = result["safe_medications"]
        assert "ciprofloxacin" in safe_meds, "Ciprofloxacin should be safe"
        
        # Should have recommendations
        recommendations = result["recommendations"]
        assert len(recommendations) > 0
        assert any("allerg" in rec.lower() for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_lab_interpretation(self, agent, sample_lab_results):
        """Test laboratory value interpretation."""
        request = LabInterpretationRequest(
            patient_id="test-patient-001",
            lab_results=sample_lab_results,
            patient_conditions=["diabetes", "hyperlipidemia"],
            patient_age=65,
            patient_sex="male"
        )
        
        result = await agent.interpret_lab_results(request)
        
        # Verify structure
        assert "patient_id" in result
        assert "interpretation_count" in result
        assert "interpretations" in result
        assert "lab_alert_count" in result
        assert "alerts" in result
        assert "critical_values" in result
        assert "abnormal_values" in result
        
        # Should have interpretations for all lab values
        assert result["interpretation_count"] == 3
        
        interpretations = result["interpretations"]
        
        # Verify glucose interpretation
        glucose_interp = next(
            (interp for interp in interpretations if interp["test_name"] == "glucose"),
            None
        )
        assert glucose_interp is not None
        assert glucose_interp["status"] == "high"
        assert glucose_interp["current_value"] == 150.0
        assert len(glucose_interp["recommendations"]) > 0
        
        # Verify A1c interpretation  
        a1c_interp = next(
            (interp for interp in interpretations if interp["test_name"] == "hemoglobin_a1c"),
            None
        )
        assert a1c_interp is not None
        assert a1c_interp["status"] == "high"
        assert a1c_interp["current_value"] == 8.5
        
        # Should have abnormal values
        abnormal_values = result["abnormal_values"]
        assert len(abnormal_values) == 3  # All values are abnormal
    
    @pytest.mark.asyncio
    async def test_patient_alert_summary(self, agent):
        """Test patient alert summary."""
        patient_id = "test-patient-001"
        
        # First generate some alerts through analysis
        request = ClinicalAnalysisRequest(
            patient_id=patient_id,
            patient_data={"age": 65, "conditions": ["diabetes"]},
            medications=["warfarin", "aspirin"],
            allergies=["penicillin"],
            clinical_contexts=["diabetes"]
        )
        await agent.comprehensive_clinical_analysis(request)
        
        # Get alert summary
        result = await agent.get_patient_alert_summary(patient_id)
        
        # Verify structure
        assert "patient_id" in result
        assert "total_alerts" in result
        assert "alert_groups" in result
        assert "severity_distribution" in result
        assert "summary_timestamp" in result
        
        assert result["patient_id"] == patient_id
        assert result["total_alerts"] > 0
        
        # Verify severity distribution
        severity_dist = result["severity_distribution"]
        assert isinstance(severity_dist, dict)
        
        # Verify alert groups
        alert_groups = result["alert_groups"]
        assert isinstance(alert_groups, dict)
    
    @pytest.mark.asyncio
    async def test_task_execution_comprehensive(self, agent, sample_patient_data, sample_medications, sample_allergies):
        """Test task execution with comprehensive analysis."""
        task = TaskRequest(
            task_id="test-task-001",
            task_type="comprehensive_analysis",
            parameters={
                "patient_id": sample_patient_data["patient_id"],
                "patient_data": sample_patient_data,
                "medications": sample_medications,
                "allergies": sample_allergies,
                "clinical_contexts": ["diabetes", "cardiovascular"]
            }
        )
        
        response = await agent.execute_task(task)
        
        # Verify response structure
        assert response.task_id == task.task_id
        assert response.agent_id == agent.agent_id
        assert response.status == "completed"
        assert response.result is not None
        assert response.metadata["analysis_type"] == "comprehensive"
        assert response.metadata["patient_id"] == sample_patient_data["patient_id"]
        
        # Verify result content
        result = response.result
        assert "drug_interactions" in result
        assert "allergy_alerts" in result
        assert "clinical_guidelines" in result
        assert "summary" in result
    
    @pytest.mark.asyncio
    async def test_task_execution_drug_interactions(self, agent, sample_medications):
        """Test task execution for drug interactions."""
        task = TaskRequest(
            task_id="test-task-002",
            task_type="drug_interactions",
            parameters={
                "patient_id": "test-patient-001",
                "medications": sample_medications
            }
        )
        
        response = await agent.execute_task(task)
        
        assert response.status == "completed"
        assert response.metadata["analysis_type"] == "drug_interactions"
        
        result = response.result
        assert "interaction_count" in result
        assert "interactions" in result
        assert "recommendations" in result
    
    @pytest.mark.asyncio
    async def test_task_execution_allergy_screening(self, agent, sample_allergies):
        """Test task execution for allergy screening."""
        task = TaskRequest(
            task_id="test-task-003",
            task_type="allergy_screening",
            parameters={
                "patient_id": "test-patient-001",
                "patient_allergies": sample_allergies,
                "proposed_medications": ["amoxicillin", "ciprofloxacin"]
            }
        )
        
        response = await agent.execute_task(task)
        
        assert response.status == "completed"
        assert response.metadata["analysis_type"] == "allergy_screening"
        
        result = response.result
        assert "allergy_alert_count" in result
        assert "allergy_alerts" in result
        assert "safe_medications" in result
    
    @pytest.mark.asyncio
    async def test_task_execution_lab_interpretation(self, agent, sample_lab_results):
        """Test task execution for lab interpretation."""
        task = TaskRequest(
            task_id="test-task-004",
            task_type="lab_interpretation",
            parameters={
                "patient_id": "test-patient-001",
                "lab_results": sample_lab_results,
                "patient_conditions": ["diabetes"],
                "patient_age": 65,
                "patient_sex": "male"
            }
        )
        
        response = await agent.execute_task(task)
        
        assert response.status == "completed"
        assert response.metadata["analysis_type"] == "lab_interpretation"
        
        result = response.result
        assert "interpretation_count" in result
        assert "interpretations" in result
        assert "lab_alert_count" in result
    
    @pytest.mark.asyncio
    async def test_task_execution_patient_alerts(self, agent):
        """Test task execution for patient alerts."""
        task = TaskRequest(
            task_id="test-task-005",
            task_type="patient_alerts",
            parameters={
                "patient_id": "test-patient-001",
                "severity_filter": "high"
            }
        )
        
        response = await agent.execute_task(task)
        
        assert response.status == "completed"
        assert response.metadata["analysis_type"] == "patient_alerts"
        
        result = response.result
        assert "patient_id" in result
        assert "total_alerts" in result
        assert "alert_groups" in result
    
    @pytest.mark.asyncio
    async def test_task_execution_unknown_type(self, agent):
        """Test task execution with unknown task type."""
        task = TaskRequest(
            task_id="test-task-006",
            task_type="unknown_task",
            parameters={}
        )
        
        response = await agent.execute_task(task)
        
        assert response.status == "failed"
        assert "Unknown task type" in response.error
        assert "supported_tasks" in response.metadata
    
    @pytest.mark.asyncio
    async def test_caching_mechanism(self, agent, sample_patient_data, sample_medications):
        """Test that caching works properly."""
        request = ClinicalAnalysisRequest(
            patient_id=sample_patient_data["patient_id"],
            patient_data=sample_patient_data,
            medications=sample_medications,
            allergies=[],
            clinical_contexts=["diabetes"]
        )
        
        # First call
        start_time = datetime.utcnow()
        result1 = await agent.comprehensive_clinical_analysis(request)
        first_call_time = datetime.utcnow() - start_time
        
        # Second call (should be cached)
        start_time = datetime.utcnow()
        result2 = await agent.comprehensive_clinical_analysis(request)
        second_call_time = datetime.utcnow() - start_time
        
        # Results should be identical
        assert result1["analysis_metadata"]["analysis_id"] == result2["analysis_metadata"]["analysis_id"]
        
        # Second call should be faster (cached)
        assert second_call_time < first_call_time
        
        # Cache should contain the entry
        assert len(agent.analysis_cache) > 0
    
    @pytest.mark.asyncio
    async def test_agent_metrics(self, agent, sample_patient_data, sample_medications):
        """Test agent metrics collection."""
        # Perform some analyses to generate metrics
        request = ClinicalAnalysisRequest(
            patient_id=sample_patient_data["patient_id"],
            patient_data=sample_patient_data,
            medications=sample_medications,
            allergies=["penicillin"],
            clinical_contexts=["diabetes"]
        )
        
        await agent.comprehensive_clinical_analysis(request)
        
        metrics = await agent.get_agent_metrics()
        
        # Verify metrics structure
        assert "agent_metrics" in metrics
        assert "system_statistics" in metrics
        assert "capabilities" in metrics
        assert "version" in metrics
        assert "last_updated" in metrics
        
        agent_metrics = metrics["agent_metrics"]
        assert "analyses_performed" in agent_metrics
        assert "alerts_generated" in agent_metrics
        assert "recommendations_made" in agent_metrics
        assert "cache_size" in agent_metrics
        
        # Verify metrics are updated
        assert agent_metrics["analyses_performed"] > 0
        assert agent_metrics["alerts_generated"] > 0
        assert agent_metrics["cache_size"] > 0
        
        # Verify capabilities
        capabilities = metrics["capabilities"]
        expected_capabilities = [
            "DRUG_INTERACTION_CHECKING",
            "ALLERGY_SCREENING", 
            "CLINICAL_GUIDELINES",
            "LAB_INTERPRETATION",
            "RISK_ASSESSMENT"
        ]
        for cap in expected_capabilities:
            assert cap in capabilities
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling in task execution."""
        # Test with invalid request data
        task = TaskRequest(
            task_id="test-task-error",
            task_type="comprehensive_analysis",
            parameters={
                "patient_id": "test-patient-001",
                # Missing required parameters
            }
        )
        
        response = await agent.execute_task(task)
        
        assert response.status == "failed"
        assert response.error is not None
        assert response.task_id == task.task_id
    
    def test_agent_capabilities(self, agent):
        """Test agent capabilities are properly defined."""
        capabilities = agent.capabilities
        
        assert len(capabilities) == 5
        
        capability_names = [cap.name for cap in capabilities]
        expected_names = [
            "DRUG_INTERACTION_CHECKING",
            "ALLERGY_SCREENING",
            "CLINICAL_GUIDELINES", 
            "LAB_INTERPRETATION",
            "RISK_ASSESSMENT"
        ]
        
        for name in expected_names:
            assert name in capability_names
        
        # Verify each capability has required fields
        for cap in capabilities:
            assert cap.name
            assert cap.description
            assert cap.input_schema
            assert cap.output_schema
            assert cap.supported_formats
            assert cap.requirements


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])