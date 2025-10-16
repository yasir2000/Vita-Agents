"""
HIPAA Compliance Tests for Vita Agents Security Framework
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import json

from vita_agents.core.security import (
    EncryptionManager,
    AuthenticationManager,
    AuditLogger,
    ComplianceValidator,
    HIPAACompliantAgent,
    SecurityException,
    ComplianceLevel,
    AuditAction,
    AuditEvent
)
from vita_agents.core.config import Settings


@pytest.fixture
def test_settings():
    """Test settings fixture"""
    return Settings(
        security={
            "jwt_secret": "test-jwt-secret-key",
            "encryption_key": "test-encryption-key-change-in-prod",
            "encryption_salt": "test-encryption-salt-change-in-prod",
            "jwt_expiration_minutes": 30,
            "data_retention_years": 7
        }
    )


@pytest.fixture
def encryption_manager(test_settings):
    """Encryption manager fixture"""
    return EncryptionManager(test_settings)


@pytest.fixture
def auth_manager(test_settings):
    """Authentication manager fixture"""
    return AuthenticationManager(test_settings)


@pytest.fixture
def compliance_validator(test_settings):
    """Compliance validator fixture"""
    return ComplianceValidator(test_settings)


@pytest.fixture
def mock_db_manager():
    """Mock database manager"""
    db_manager = AsyncMock()
    db_manager.execute = AsyncMock()
    db_manager.fetch_all = AsyncMock(return_value=[])
    return db_manager


@pytest.fixture
def audit_logger(test_settings, mock_db_manager):
    """Audit logger fixture"""
    return AuditLogger(test_settings, mock_db_manager)


class TestEncryptionManager:
    """Test encryption functionality"""
    
    def test_encrypt_decrypt_sensitive_data(self, encryption_manager):
        """Test encryption and decryption of sensitive data"""
        test_data = "Patient ID: 12345"
        
        # Test encryption
        encrypted = encryption_manager.encrypt_sensitive_data(
            test_data, 
            ComplianceLevel.RESTRICTED
        )
        
        assert encrypted != test_data
        assert len(encrypted) > 0
        
        # Test decryption
        decrypted = encryption_manager.decrypt_sensitive_data(
            encrypted, 
            ComplianceLevel.RESTRICTED
        )
        
        assert decrypted == test_data
    
    def test_encrypt_patient_id(self, encryption_manager):
        """Test patient ID encryption for audit logging"""
        patient_id = "PATIENT-123456"
        
        encrypted_id = encryption_manager.encrypt_patient_id(patient_id)
        
        assert encrypted_id != patient_id
        assert len(encrypted_id) > 0
    
    def test_hash_data_with_salt(self, encryption_manager):
        """Test data hashing with salt"""
        test_data = "sensitive-password"
        
        hashed = encryption_manager.hash_data(test_data)
        
        # Should contain salt and hash separated by colon
        assert ":" in hashed
        salt, hash_value = hashed.split(":", 1)
        assert len(salt) > 0
        assert len(hash_value) > 0
        
        # Verify hash
        assert encryption_manager.verify_hash(test_data, hashed)
        assert not encryption_manager.verify_hash("wrong-data", hashed)
    
    def test_encryption_failure_handling(self, encryption_manager):
        """Test encryption error handling"""
        with pytest.raises(SecurityException):
            # This should fail due to invalid compliance level handling
            encryption_manager.encrypt_sensitive_data("", None)


class TestAuthenticationManager:
    """Test authentication and authorization"""
    
    def test_create_and_verify_token(self, auth_manager):
        """Test JWT token creation and verification"""
        user_id = "user123"
        permissions = ["patient.read", "phi.access"]
        
        # Create token
        token = auth_manager.create_access_token(user_id, permissions)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify token
        payload = auth_manager.verify_token(token)
        
        assert payload["sub"] == user_id
        assert payload["permissions"] == permissions
        assert payload["type"] == "access"
        assert payload["compliance"] == "hipaa"
    
    def test_token_expiration(self, auth_manager):
        """Test token expiration handling"""
        user_id = "user123"
        permissions = ["patient.read"]
        
        # Create token with very short expiration
        short_expiration = timedelta(seconds=-1)  # Already expired
        token = auth_manager.create_access_token(
            user_id, 
            permissions, 
            expires_delta=short_expiration
        )
        
        # Should raise exception for expired token
        with pytest.raises(SecurityException, match="Token has expired"):
            auth_manager.verify_token(token)
    
    def test_password_hashing(self, auth_manager):
        """Test password hashing and verification"""
        password = "secure-password-123"
        
        # Hash password
        hashed = auth_manager.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
        
        # Verify password
        assert auth_manager.verify_password(password, hashed)
        assert not auth_manager.verify_password("wrong-password", hashed)
    
    def test_permission_checking(self, auth_manager):
        """Test permission validation"""
        # Test with specific permission
        user_permissions = ["patient.read", "observation.read"]
        
        assert auth_manager.check_permission(user_permissions, "patient.read")
        assert not auth_manager.check_permission(user_permissions, "patient.write")
        
        # Test with admin permission
        admin_permissions = ["admin"]
        assert auth_manager.check_permission(admin_permissions, "any.permission")


class TestComplianceValidator:
    """Test HIPAA compliance validation"""
    
    def test_validate_phi_access_permissions(self, compliance_validator):
        """Test PHI access permission validation"""
        # Test valid access
        user_permissions = ["patient.read", "phi.access"]
        assert compliance_validator.validate_phi_access(
            user_permissions, 
            "Patient", 
            AuditAction.READ
        )
        
        # Test invalid access
        limited_permissions = ["observation.read"]
        assert not compliance_validator.validate_phi_access(
            limited_permissions, 
            "Patient", 
            AuditAction.READ
        )
    
    def test_data_retention_policy(self, compliance_validator):
        """Test data retention policy compliance"""
        # Recent data should pass
        recent_date = datetime.utcnow() - timedelta(days=30)
        assert compliance_validator.check_data_retention_policy(recent_date)
        
        # Old data should fail
        old_date = datetime.utcnow() - timedelta(days=8 * 365)  # 8 years old
        assert not compliance_validator.check_data_retention_policy(old_date)
    
    def test_minimum_necessary_principle(self, compliance_validator):
        """Test minimum necessary data access principle"""
        requested_fields = ["name", "mrn", "dob", "ssn", "vitals", "diagnoses"]
        
        # Nurse should get limited access
        nurse_fields = compliance_validator.validate_minimum_necessary(
            requested_fields, 
            "nurse"
        )
        assert "name" in nurse_fields
        assert "vitals" in nurse_fields
        assert "ssn" not in nurse_fields  # Should not have access to SSN
        
        # Doctor should get more access
        doctor_fields = compliance_validator.validate_minimum_necessary(
            requested_fields, 
            "doctor"
        )
        assert "diagnoses" in doctor_fields
        assert len(doctor_fields) >= len(nurse_fields)
        
        # Researcher should get de-identified data only
        researcher_fields = compliance_validator.validate_minimum_necessary(
            requested_fields, 
            "researcher"
        )
        assert "name" not in researcher_fields
        assert "mrn" not in researcher_fields


@pytest.mark.asyncio
class TestAuditLogger:
    """Test audit logging functionality"""
    
    async def test_log_audit_event(self, audit_logger, mock_db_manager):
        """Test audit event logging"""
        audit_event = AuditEvent(
            action=AuditAction.READ,
            resource_type="Patient",
            resource_id="patient-123",
            user_id="user-456",
            patient_id="patient-123",
            agent_id="fhir-agent-1",
            ip_address="192.168.1.1",
            access_reason="Clinical care",
            compliance_level=ComplianceLevel.RESTRICTED,
            timestamp=datetime.utcnow(),
            details={"data_size": 1024},
            success=True
        )
        
        # Should not raise exception
        await audit_logger.log_audit_event(audit_event)
        
        # Verify database call was made
        mock_db_manager.execute.assert_called_once()
    
    async def test_audit_logging_failure(self, audit_logger):
        """Test audit logging failure handling"""
        # Create invalid audit event
        invalid_event = AuditEvent(
            action=AuditAction.READ,
            resource_type="Patient",
            resource_id=None,
            user_id=None,
            patient_id=None,
            agent_id=None,
            ip_address=None,
            access_reason="",
            compliance_level=ComplianceLevel.RESTRICTED,
            timestamp=datetime.utcnow(),
            details={},
            success=True
        )
        
        # Mock database failure
        audit_logger.db_manager = AsyncMock()
        audit_logger.db_manager.execute.side_effect = Exception("Database error")
        
        with pytest.raises(SecurityException, match="Audit logging failed"):
            await audit_logger.log_audit_event(invalid_event)
    
    async def test_get_audit_trail(self, audit_logger, mock_db_manager):
        """Test audit trail retrieval"""
        # Mock database response
        mock_audit_records = [
            {
                "id": "record-1",
                "action": "read",
                "resource_type": "Patient",
                "created_at": datetime.utcnow()
            }
        ]
        mock_db_manager.fetch_all.return_value = mock_audit_records
        
        # Get audit trail
        trail = await audit_logger.get_audit_trail(
            patient_id="patient-123",
            start_date=datetime.utcnow() - timedelta(days=30),
            limit=50
        )
        
        assert len(trail) == 1
        assert trail[0]["action"] == "read"
        mock_db_manager.fetch_all.assert_called_once()


@pytest.mark.asyncio
class TestHIPAACompliantAgent:
    """Test HIPAA compliant agent base class"""
    
    class TestAgent(HIPAACompliantAgent):
        """Test implementation of HIPAA compliant agent"""
        
        async def _process_healthcare_data(self, data):
            return {"processed": True, "data": data}
    
    @pytest.fixture
    def test_agent(self, test_settings, mock_db_manager):
        """Test agent fixture"""
        return self.TestAgent("test-agent", test_settings, mock_db_manager)
    
    async def test_secure_process_data_success(self, test_agent):
        """Test successful secure data processing"""
        test_data = {"resourceType": "Patient", "id": "patient-123"}
        user_permissions = ["patient.read", "phi.access"]
        
        result = await test_agent.secure_process_data(
            data=test_data,
            user_id="user-456",
            user_permissions=user_permissions,
            access_reason="Clinical care",
            action=AuditAction.READ,
            resource_type="Patient",
            patient_id="patient-123"
        )
        
        assert result["processed"] is True
        assert result["data"] == test_data
    
    async def test_secure_process_data_insufficient_permissions(self, test_agent):
        """Test data processing with insufficient permissions"""
        test_data = {"resourceType": "Patient", "id": "patient-123"}
        user_permissions = ["observation.read"]  # Insufficient for Patient access
        
        with pytest.raises(SecurityException, match="Insufficient permissions"):
            await test_agent.secure_process_data(
                data=test_data,
                user_id="user-456",
                user_permissions=user_permissions,
                access_reason="Clinical care",
                action=AuditAction.READ,
                resource_type="Patient",
                patient_id="patient-123"
            )


class TestSecurityIntegration:
    """Integration tests for security components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_secure_workflow(self, test_settings, mock_db_manager):
        """Test complete secure workflow from authentication to audit"""
        
        # Initialize components
        auth_manager = AuthenticationManager(test_settings)
        encryption_manager = EncryptionManager(test_settings)
        audit_logger = AuditLogger(test_settings, mock_db_manager)
        
        # 1. User authentication
        user_id = "healthcare-user"
        permissions = ["patient.read", "phi.access"]
        token = auth_manager.create_access_token(user_id, permissions)
        
        # 2. Token verification
        payload = auth_manager.verify_token(token)
        assert payload["sub"] == user_id
        
        # 3. Data encryption
        sensitive_data = "Patient John Doe, DOB: 1990-01-01"
        encrypted_data = encryption_manager.encrypt_sensitive_data(
            sensitive_data,
            ComplianceLevel.RESTRICTED
        )
        
        # 4. Audit logging
        audit_event = AuditEvent(
            action=AuditAction.READ,
            resource_type="Patient",
            resource_id="patient-123",
            user_id=user_id,
            patient_id="patient-123",
            agent_id="integration-test",
            ip_address="127.0.0.1",
            access_reason="Integration test",
            compliance_level=ComplianceLevel.RESTRICTED,
            timestamp=datetime.utcnow(),
            details={"test": True},
            success=True
        )
        
        await audit_logger.log_audit_event(audit_event)
        
        # 5. Data decryption
        decrypted_data = encryption_manager.decrypt_sensitive_data(
            encrypted_data,
            ComplianceLevel.RESTRICTED
        )
        
        assert decrypted_data == sensitive_data
        mock_db_manager.execute.assert_called_once()


# Performance and security benchmarks
class TestSecurityPerformance:
    """Performance tests for security operations"""
    
    def test_encryption_performance(self, encryption_manager):
        """Test encryption performance with various data sizes"""
        test_data_sizes = [100, 1000, 10000]  # bytes
        
        for size in test_data_sizes:
            test_data = "x" * size
            
            # Measure encryption time
            import time
            start_time = time.time()
            
            encrypted = encryption_manager.encrypt_sensitive_data(
                test_data,
                ComplianceLevel.RESTRICTED
            )
            
            encryption_time = time.time() - start_time
            
            # Should complete within reasonable time (adjust as needed)
            assert encryption_time < 1.0  # 1 second
            assert len(encrypted) > 0
    
    def test_token_generation_performance(self, auth_manager):
        """Test JWT token generation performance"""
        import time
        
        start_time = time.time()
        
        # Generate multiple tokens
        for i in range(100):
            token = auth_manager.create_access_token(
                f"user-{i}",
                ["patient.read", "phi.access"]
            )
            assert len(token) > 0
        
        total_time = time.time() - start_time
        
        # Should complete 100 token generations within reasonable time
        assert total_time < 5.0  # 5 seconds for 100 tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])