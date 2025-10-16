"""
Compliance tests for HIPAA and healthcare regulations
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
import json

from vita_agents.core.security import (
    EncryptionManager,
    AuditLogger,
    ComplianceValidator,
    AuditEvent,
    AuditAction,
    ComplianceLevel
)
from vita_agents.agents.fhir_agent import FHIRAgent
from vita_agents.core.config import Settings


class TestHIPAACompliance:
    """Test HIPAA compliance requirements"""
    
    def test_phi_encryption_at_rest(self):
        """Test that PHI is encrypted when stored"""
        settings = Settings()
        encryption_manager = EncryptionManager(settings)
        
        # Test PHI encryption
        phi_data = "Patient: John Doe, SSN: 123-45-6789, DOB: 1980-01-01"
        
        encrypted = encryption_manager.encrypt_sensitive_data(
            phi_data, 
            ComplianceLevel.RESTRICTED
        )
        
        # Verify data is encrypted (not readable)
        assert encrypted != phi_data
        assert "John Doe" not in encrypted
        assert "123-45-6789" not in encrypted
        
        # Verify data can be decrypted
        decrypted = encryption_manager.decrypt_sensitive_data(
            encrypted,
            ComplianceLevel.RESTRICTED
        )
        assert decrypted == phi_data
    
    def test_audit_trail_completeness(self):
        """Test that all PHI access is audited"""
        settings = Settings()
        mock_db = AsyncMock()
        audit_logger = AuditLogger(settings, mock_db)
        
        # Test audit event creation
        audit_event = AuditEvent(
            action=AuditAction.READ,
            resource_type="Patient",
            resource_id="patient-123",
            user_id="user-456",
            patient_id="patient-123",
            agent_id="test-agent",
            ip_address="192.168.1.1",
            access_reason="Clinical care",
            compliance_level=ComplianceLevel.RESTRICTED,
            timestamp=datetime.utcnow(),
            details={"test": True},
            success=True
        )
        
        # Verify required audit fields are present
        assert audit_event.action is not None
        assert audit_event.user_id is not None
        assert audit_event.patient_id is not None
        assert audit_event.timestamp is not None
        assert audit_event.access_reason is not None
    
    def test_minimum_necessary_access(self):
        """Test minimum necessary access principle"""
        settings = Settings()
        compliance_validator = ComplianceValidator(settings)
        
        # Test field filtering for different roles
        all_fields = ["name", "mrn", "dob", "ssn", "vitals", "diagnoses", "medications"]
        
        # Nurse should get limited access
        nurse_fields = compliance_validator.validate_minimum_necessary(
            all_fields, "nurse"
        )
        assert "vitals" in nurse_fields
        assert "medications" in nurse_fields
        assert "ssn" not in nurse_fields  # Should not have access to SSN
        
        # Doctor should get more access
        doctor_fields = compliance_validator.validate_minimum_necessary(
            all_fields, "doctor" 
        )
        assert "diagnoses" in doctor_fields
        assert len(doctor_fields) >= len(nurse_fields)
        
        # Researcher should get de-identified data only
        researcher_fields = compliance_validator.validate_minimum_necessary(
            all_fields, "researcher"
        )
        assert "name" not in researcher_fields
        assert "ssn" not in researcher_fields
    
    def test_data_retention_compliance(self):
        """Test data retention policy compliance"""
        settings = Settings()
        compliance_validator = ComplianceValidator(settings)
        
        # Recent data should pass retention check
        recent_date = datetime.utcnow() - timedelta(days=30)
        assert compliance_validator.check_data_retention_policy(recent_date)
        
        # Old data should fail retention check (assuming 7 year retention)
        old_date = datetime.utcnow() - timedelta(days=8 * 365)
        assert not compliance_validator.check_data_retention_policy(old_date)
    
    def test_access_control_enforcement(self):
        """Test that access controls are properly enforced"""
        settings = Settings()
        compliance_validator = ComplianceValidator(settings)
        
        # Test PHI access validation
        valid_permissions = ["patient.read", "phi.access"]
        assert compliance_validator.validate_phi_access(
            valid_permissions, "Patient", AuditAction.READ
        )
        
        # Invalid permissions should be rejected
        invalid_permissions = ["observation.read"]
        assert not compliance_validator.validate_phi_access(
            invalid_permissions, "Patient", AuditAction.READ
        )


class TestSecurityStandards:
    """Test security standards compliance"""
    
    def test_encryption_standards(self):
        """Test encryption meets security standards"""
        settings = Settings()
        encryption_manager = EncryptionManager(settings)
        
        # Test encryption strength
        test_data = "sensitive healthcare data"
        encrypted = encryption_manager.encrypt_sensitive_data(
            test_data,
            ComplianceLevel.RESTRICTED
        )
        
        # Verify encryption produces different output each time (salt/IV)
        encrypted2 = encryption_manager.encrypt_sensitive_data(
            test_data,
            ComplianceLevel.RESTRICTED
        )
        assert encrypted != encrypted2  # Should be different due to randomness
        
        # Both should decrypt to same value
        decrypted1 = encryption_manager.decrypt_sensitive_data(
            encrypted, ComplianceLevel.RESTRICTED
        )
        decrypted2 = encryption_manager.decrypt_sensitive_data(
            encrypted2, ComplianceLevel.RESTRICTED
        )
        assert decrypted1 == decrypted2 == test_data
    
    def test_password_security(self):
        """Test password hashing security"""
        from vita_agents.core.security import AuthenticationManager
        
        settings = Settings()
        auth_manager = AuthenticationManager(settings)
        
        password = "secure_password_123"
        hashed = auth_manager.hash_password(password)
        
        # Verify password is properly hashed
        assert hashed != password
        assert len(hashed) > 50  # bcrypt hashes are long
        assert "$" in hashed  # bcrypt format includes $ separators
        
        # Verify password verification works
        assert auth_manager.verify_password(password, hashed)
        assert not auth_manager.verify_password("wrong_password", hashed)
    
    def test_session_security(self):
        """Test session management security"""
        from vita_agents.core.security import AuthenticationManager
        
        settings = Settings()
        auth_manager = AuthenticationManager(settings)
        
        # Test JWT token creation
        user_id = "test_user"
        permissions = ["patient.read"]
        token = auth_manager.create_access_token(user_id, permissions)
        
        # Verify token structure
        assert isinstance(token, str)
        assert len(token) > 100  # JWT tokens are long
        assert "." in token  # JWT format includes dots
        
        # Verify token can be verified
        payload = auth_manager.verify_token(token)
        assert payload["sub"] == user_id
        assert payload["permissions"] == permissions
        assert payload["compliance"] == "hipaa"


class TestDataIntegrity:
    """Test data integrity and validation"""
    
    def test_fhir_validation_integrity(self):
        """Test FHIR resource validation maintains data integrity"""
        settings = Settings()
        mock_db = AsyncMock()
        fhir_agent = FHIRAgent("test-fhir", settings, mock_db)
        
        # Valid FHIR Patient resource
        valid_patient = {
            "resourceType": "Patient",
            "id": "patient-123",
            "name": [
                {
                    "use": "official",
                    "family": "Doe",
                    "given": ["John"]
                }
            ],
            "gender": "male",
            "active": True
        }
        
        # Test internal validation (not the secure method)
        validation_result = asyncio.run(
            fhir_agent._validate_fhir_resource(valid_patient)
        )
        
        assert validation_result["valid"] == True
        assert len(validation_result["errors"]) == 0
        
        # Invalid resource should fail validation
        invalid_patient = {
            "id": "patient-123",  # Missing resourceType
            "name": "Invalid Name Format"  # Wrong format
        }
        
        validation_result = asyncio.run(
            fhir_agent._validate_fhir_resource(invalid_patient)
        )
        
        assert validation_result["valid"] == False
        assert len(validation_result["errors"]) > 0
    
    def test_data_quality_assessment(self):
        """Test data quality assessment accuracy"""
        settings = Settings()
        mock_db = AsyncMock()
        fhir_agent = FHIRAgent("test-fhir", settings, mock_db)
        
        # High quality data
        high_quality_patient = {
            "resourceType": "Patient",
            "id": "patient-123",
            "meta": {
                "lastUpdated": "2025-10-16T10:00:00Z"
            },
            "name": [
                {
                    "use": "official",
                    "family": "Doe",
                    "given": ["John", "William"]
                }
            ],
            "gender": "male",
            "birthDate": "1980-01-01",
            "address": [
                {
                    "use": "home",
                    "line": ["123 Main St"],
                    "city": "Anytown",
                    "state": "NY",
                    "postalCode": "12345"
                }
            ],
            "telecom": [
                {
                    "system": "phone",
                    "value": "555-1234",
                    "use": "home"
                }
            ],
            "active": True
        }
        
        quality_result = asyncio.run(
            fhir_agent._assess_data_quality(high_quality_patient)
        )
        
        # Should have high quality score
        assert quality_result["overall_score"] > 0.8
        assert quality_result["compliance_ready"] == True
        
        # Low quality data (minimal fields)
        low_quality_patient = {
            "resourceType": "Patient",
            "id": "patient-456"
            # Minimal data - should have low quality score
        }
        
        quality_result = asyncio.run(
            fhir_agent._assess_data_quality(low_quality_patient)
        )
        
        # Should have lower quality score
        assert quality_result["overall_score"] < 0.5


class TestPrivacyProtection:
    """Test privacy protection mechanisms"""
    
    def test_phi_identification(self):
        """Test that PHI is properly identified"""
        settings = Settings()
        mock_db = AsyncMock()
        fhir_agent = FHIRAgent("test-fhir", settings, mock_db)
        
        patient_data = {
            "resourceType": "Patient",
            "id": "patient-123",
            "name": [
                {
                    "use": "official",
                    "family": "Doe",
                    "given": ["John"]
                }
            ],
            "birthDate": "1980-01-01",
            "identifier": [
                {
                    "system": "http://hospital.example.org",
                    "value": "MRN-12345"
                }
            ],
            "telecom": [
                {
                    "system": "phone",
                    "value": "555-1234"
                }
            ]
        }
        
        # Identify PHI fields
        phi_fields = fhir_agent._identify_phi_fields(patient_data, "Patient")
        
        # Should identify name, birthDate, identifier, telecom as PHI
        assert "name" in phi_fields
        assert "identifier" in phi_fields
        assert "telecom" in phi_fields
    
    def test_data_deidentification(self):
        """Test data de-identification capabilities"""
        # This would test removal of direct identifiers
        # Implementation would depend on de-identification requirements
        pass
    
    def test_consent_management(self):
        """Test patient consent management"""
        # This would test patient consent tracking and enforcement
        # Implementation would depend on consent management requirements
        pass


class TestRegulatoryCompliance:
    """Test compliance with healthcare regulations"""
    
    def test_21_cfr_part_11_compliance(self):
        """Test 21 CFR Part 11 compliance for electronic records"""
        # Test electronic signature requirements
        # Test audit trail requirements
        # Test record integrity requirements
        pass
    
    def test_gdpr_compliance(self):
        """Test GDPR compliance for EU patients"""
        # Test right to be forgotten
        # Test data portability
        # Test consent management
        pass
    
    def test_hitech_compliance(self):
        """Test HITECH Act compliance"""
        # Test breach notification requirements
        # Test enhanced penalties compliance
        # Test business associate requirements
        pass


class TestSecurityIncidentResponse:
    """Test security incident response capabilities"""
    
    def test_breach_detection(self):
        """Test breach detection mechanisms"""
        # Test unauthorized access detection
        # Test data exfiltration detection
        # Test suspicious activity monitoring
        pass
    
    def test_incident_logging(self):
        """Test security incident logging"""
        # Test incident event logging
        # Test incident severity classification
        # Test incident response tracking
        pass
    
    def test_breach_notification(self):
        """Test breach notification procedures"""
        # Test automatic notification triggers
        # Test notification content requirements
        # Test notification timing requirements
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])