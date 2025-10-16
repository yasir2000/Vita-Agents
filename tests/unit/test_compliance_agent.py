"""
Unit tests for Compliance & Security Agent.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from vita_agents.agents.compliance_security_agent import (
    ComplianceSecurityAgent,
    ConsentType,
    ConsentStatus,
    SecurityIncidentType,
    SecurityIncidentSeverity,
    PatientConsent
)
from vita_agents.core.agent import TaskRequest
from vita_agents.core.config import Settings


class TestComplianceSecurityAgent:
    """Test Compliance & Security Agent functionality."""
    
    @pytest.fixture
    def settings(self):
        """Mock settings fixture."""
        return Settings()
    
    @pytest.fixture
    def mock_database(self):
        """Mock database fixture."""
        return AsyncMock()
    
    @pytest.fixture
    def agent(self, settings, mock_database):
        """Create agent fixture."""
        return ComplianceSecurityAgent("test-compliance", settings, mock_database)
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.agent_id == "test-compliance"
        assert agent.agent_type == "compliance_security"
        assert len(agent.task_handlers) > 0
    
    @pytest.mark.asyncio
    async def test_validate_phi_access_success(self, agent):
        """Test successful PHI access validation."""
        task = TaskRequest(
            task_id="test-task",
            task_type="validate_phi_access",
            parameters={
                "user_id": "user123",
                "patient_id": "patient456",
                "access_reason": "treatment",
                "requested_fields": ["name", "dob"],
                "user_role": "doctor",
                "user_permissions": ["patient.read", "phi.access"]
            }
        )
        
        with patch.object(agent, '_check_patient_consent', return_value=True):
            response = await agent.process_task(task)
        
        assert response.status == "completed"
        assert response.result["access_granted"] == True
        assert "allowed_fields" in response.result
    
    @pytest.mark.asyncio
    async def test_validate_phi_access_denied(self, agent):
        """Test PHI access denial."""
        task = TaskRequest(
            task_id="test-task",
            task_type="validate_phi_access",
            parameters={
                "user_id": "user123",
                "patient_id": "patient456",
                "access_reason": "research",
                "requested_fields": ["name", "ssn"],
                "user_role": "researcher",
                "user_permissions": ["observation.read"]
            }
        )
        
        with patch.object(agent, '_check_patient_consent', return_value=False):
            response = await agent.process_task(task)
        
        assert response.status == "completed"
        assert response.result["access_granted"] == False
    
    @pytest.mark.asyncio
    async def test_grant_consent(self, agent):
        """Test granting patient consent."""
        task = TaskRequest(
            task_id="test-task",
            task_type="grant_consent",
            parameters={
                "patient_id": "patient123",
                "consent_type": "treatment",
                "purpose": "routine care",
                "grantor_id": "patient123",
                "scope": ["medication", "vitals"]
            }
        )
        
        with patch.object(agent, '_store_consent_record'):
            response = await agent.process_task(task)
        
        assert response.status == "completed"
        assert response.result["status"] == "granted"
        assert "consent_id" in response.result
    
    @pytest.mark.asyncio
    async def test_withdraw_consent(self, agent):
        """Test withdrawing patient consent."""
        task = TaskRequest(
            task_id="test-task",
            task_type="withdraw_consent",
            parameters={
                "patient_id": "patient123",
                "consent_type": "research",
                "grantor_id": "patient123"
            }
        )
        
        with patch.object(agent, '_update_consent_status', return_value=1):
            response = await agent.process_task(task)
        
        assert response.status == "completed"
        assert response.result["status"] == "withdrawn"
        assert response.result["records_updated"] == 1
    
    @pytest.mark.asyncio
    async def test_detect_security_incident(self, agent):
        """Test security incident detection."""
        task = TaskRequest(
            task_id="test-task",
            task_type="detect_security_incident",
            parameters={
                "event_data": {
                    "user_id": "user123",
                    "ip_address": "192.168.1.100",
                    "resource_accessed": "Patient/123",
                    "access_pattern": {"unusual_time": True}
                }
            }
        )
        
        with patch.object(agent, '_is_unusual_access_pattern', return_value=True), \
             patch.object(agent, '_check_policy_violations', return_value=False), \
             patch.object(agent, '_store_security_incident'):
            
            response = await agent.process_task(task)
        
        assert response.status == "completed"
        assert response.result["incidents_detected"] >= 1
        assert response.result["requires_investigation"] == True
    
    @pytest.mark.asyncio
    async def test_generate_compliance_report(self, agent):
        """Test compliance report generation."""
        task = TaskRequest(
            task_id="test-task",
            task_type="generate_compliance_report",
            parameters={
                "start_date": "2025-01-01T00:00:00",
                "end_date": "2025-01-31T23:59:59"
            }
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "completed"
        assert "report_id" in response.result
        assert "compliance_score" in response.result
        assert "violations" in response.result
        assert "recommendations" in response.result
    
    @pytest.mark.asyncio
    async def test_encrypt_sensitive_data(self, agent):
        """Test data encryption."""
        task = TaskRequest(
            task_id="test-task",
            task_type="encrypt_sensitive_data",
            parameters={
                "data": "Patient: John Doe, SSN: 123-45-6789",
                "compliance_level": "restricted"
            }
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "completed"
        assert response.result["encrypted"] == True
        assert "encrypted_data" in response.result
        assert response.result["encrypted_data"] != "Patient: John Doe, SSN: 123-45-6789"
    
    @pytest.mark.asyncio
    async def test_validate_data_retention_compliant(self, agent):
        """Test data retention validation for compliant data."""
        task = TaskRequest(
            task_id="test-task",
            task_type="validate_data_retention",
            parameters={
                "data_date": "2024-01-01T00:00:00",
                "data_type": "medical_record"
            }
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "completed"
        assert response.result["compliant"] == True
        assert response.result["action_required"] is None
    
    @pytest.mark.asyncio
    async def test_validate_data_retention_non_compliant(self, agent):
        """Test data retention validation for non-compliant data."""
        old_date = datetime.utcnow() - timedelta(days=8 * 365)  # 8 years old
        
        task = TaskRequest(
            task_id="test-task",
            task_type="validate_data_retention",
            parameters={
                "data_date": old_date.isoformat(),
                "data_type": "medical_record"
            }
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "completed"
        assert response.result["compliant"] == False
        assert response.result["action_required"] == "data_purge"
    
    @pytest.mark.asyncio
    async def test_check_minimum_necessary_compliant(self, agent):
        """Test minimum necessary access check - compliant."""
        task = TaskRequest(
            task_id="test-task",
            task_type="check_minimum_necessary",
            parameters={
                "requested_fields": ["vitals", "medications"],
                "user_role": "nurse",
                "access_purpose": "patient_care"
            }
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "completed"
        assert "compliant" in response.result
        assert "allowed_fields" in response.result
    
    @pytest.mark.asyncio
    async def test_unknown_task_type(self, agent):
        """Test handling of unknown task type."""
        task = TaskRequest(
            task_id="test-task",
            task_type="unknown_task",
            parameters={}
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "failed"
        assert "Unknown task type" in response.error
    
    @pytest.mark.asyncio
    async def test_task_processing_exception(self, agent):
        """Test handling of task processing exceptions."""
        task = TaskRequest(
            task_id="test-task",
            task_type="validate_phi_access",
            parameters={
                "user_id": None,  # This should cause an error
                "patient_id": "patient456"
            }
        )
        
        response = await agent.process_task(task)
        
        assert response.status == "failed"
        assert response.error is not None


class TestPatientConsent:
    """Test PatientConsent data model."""
    
    def test_consent_creation(self):
        """Test creating a consent record."""
        consent = PatientConsent(
            consent_id="consent123",
            patient_id="patient456",
            consent_type=ConsentType.TREATMENT,
            status=ConsentStatus.GRANTED,
            granted_date=datetime.utcnow(),
            purpose="routine care"
        )
        
        assert consent.consent_id == "consent123"
        assert consent.patient_id == "patient456"
        assert consent.consent_type == ConsentType.TREATMENT
        assert consent.status == ConsentStatus.GRANTED
        assert consent.scope == []  # Default value
        assert consent.metadata == {}  # Default value
    
    def test_consent_with_scope(self):
        """Test creating consent with specific scope."""
        consent = PatientConsent(
            consent_id="consent123",
            patient_id="patient456",
            consent_type=ConsentType.RESEARCH,
            status=ConsentStatus.GRANTED,
            scope=["demographics", "conditions"],
            metadata={"study_id": "STUDY001"}
        )
        
        assert consent.scope == ["demographics", "conditions"]
        assert consent.metadata["study_id"] == "STUDY001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])