"""
Enhanced FHIR Agent with HIPAA compliance and security features
"""

from typing import Dict, Any, List, Optional, Union
import json
import asyncio
from datetime import datetime

from fhirclient import client
from fhirclient.models import patient, observation, diagnosticreport, medication
import structlog

from ..core.agent import HealthcareAgent
from ..core.security import (
    HIPAACompliantAgent, 
    AuditAction, 
    ComplianceLevel, 
    EncryptionManager,
    SecurityException
)
from ..core.config import Settings


class SecureFHIRAgent(HIPAACompliantAgent, HealthcareAgent):
    """HIPAA-compliant FHIR agent with encryption and audit logging"""
    
    def __init__(self, agent_id: str, settings: Settings, db_manager=None):
        super().__init__(agent_id, settings, db_manager)
        
        self.capabilities = [
            "fhir_validation",
            "fhir_parsing", 
            "data_quality_assessment",
            "clinical_data_extraction",
            "phi_encryption",
            "audit_logging",
            "compliance_validation"
        ]
        
        # Initialize FHIR client with security settings
        self.fhir_settings = {
            'app_id': 'vita-agents-fhir',
            'api_base': settings.healthcare.fhir_server_url
        }
        
        self.logger = structlog.get_logger(__name__)
    
    async def _process_healthcare_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process FHIR data with security and compliance"""
        try:
            # Determine FHIR resource type
            resource_type = data.get("resourceType", "Unknown")
            
            # Validate FHIR resource
            validation_result = await self._validate_fhir_resource(data)
            
            # Assess data quality
            quality_result = await self._assess_data_quality(data)
            
            # Extract clinical information
            clinical_data = await self._extract_clinical_data(data, resource_type)
            
            # Encrypt sensitive fields
            encrypted_data = await self._encrypt_sensitive_fields(data, resource_type)
            
            return {
                "validation": validation_result,
                "quality": quality_result,
                "clinical_data": clinical_data,
                "encrypted_data": encrypted_data,
                "processed_at": datetime.utcnow().isoformat(),
                "compliance_level": ComplianceLevel.RESTRICTED.value
            }
            
        except Exception as e:
            self.logger.error("FHIR processing failed", error=str(e), resource_type=data.get("resourceType"))
            raise SecurityException(f"FHIR processing failed: {str(e)}")
    
    async def validate_fhir_resource(
        self,
        fhir_data: Union[str, Dict[str, Any]],
        user_id: str,
        user_permissions: List[str],
        access_reason: str
    ) -> Dict[str, Any]:
        """Securely validate FHIR resource"""
        
        # Parse FHIR data if string
        if isinstance(fhir_data, str):
            try:
                data = json.loads(fhir_data)
            except json.JSONDecodeError as e:
                raise SecurityException(f"Invalid JSON in FHIR data: {str(e)}")
        else:
            data = fhir_data
        
        resource_type = data.get("resourceType", "Unknown")
        patient_id = self._extract_patient_id(data)
        
        return await self.secure_process_data(
            data=data,
            user_id=user_id,
            user_permissions=user_permissions,
            access_reason=access_reason,
            action=AuditAction.VALIDATE,
            resource_type=resource_type,
            patient_id=patient_id
        )
    
    async def _validate_fhir_resource(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Internal FHIR validation logic"""
        try:
            resource_type = data.get("resourceType")
            
            if not resource_type:
                return {
                    "valid": False,
                    "errors": ["Missing resourceType field"],
                    "version": None
                }
            
            # FHIR R4/R5 validation
            required_fields = self._get_required_fields(resource_type)
            missing_fields = [field for field in required_fields if field not in data]
            
            # Validate data types and formats
            format_errors = self._validate_field_formats(data, resource_type)
            
            # Check FHIR version compatibility
            meta = data.get("meta", {})
            version_info = self._validate_fhir_version(meta)
            
            is_valid = len(missing_fields) == 0 and len(format_errors) == 0
            
            return {
                "valid": is_valid,
                "errors": missing_fields + format_errors,
                "version": version_info,
                "resource_type": resource_type,
                "validation_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation exception: {str(e)}"],
                "version": None
            }
    
    async def _assess_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess FHIR data quality with privacy considerations"""
        try:
            resource_type = data.get("resourceType")
            
            quality_metrics = {
                "completeness": self._calculate_completeness(data, resource_type),
                "consistency": self._check_consistency(data),
                "validity": self._check_validity(data, resource_type),
                "timeliness": self._check_timeliness(data),
                "accuracy": self._estimate_accuracy(data)
            }
            
            # Calculate overall quality score
            overall_score = sum(quality_metrics.values()) / len(quality_metrics)
            
            # Add privacy-preserving quality indicators
            phi_fields = self._identify_phi_fields(data, resource_type)
            
            return {
                "overall_score": round(overall_score, 2),
                "metrics": quality_metrics,
                "phi_field_count": len(phi_fields),
                "compliance_ready": overall_score >= 0.8,
                "assessment_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Data quality assessment failed", error=str(e))
            return {
                "overall_score": 0.0,
                "metrics": {},
                "error": str(e)
            }
    
    async def _extract_clinical_data(self, data: Dict[str, Any], resource_type: str) -> Dict[str, Any]:
        """Extract clinical information with de-identification options"""
        try:
            clinical_info = {}
            
            if resource_type == "Patient":
                clinical_info = {
                    "gender": data.get("gender"),
                    "birth_date": data.get("birthDate"),
                    "active": data.get("active"),
                    "deceased": data.get("deceased", False),
                    "identifier_count": len(data.get("identifier", [])),
                    "contact_count": len(data.get("telecom", []))
                }
                
            elif resource_type == "Observation":
                clinical_info = {
                    "status": data.get("status"),
                    "category": data.get("category", [{}])[0].get("coding", [{}])[0].get("code"),
                    "code": data.get("code", {}).get("coding", [{}])[0].get("code"),
                    "value_type": self._get_value_type(data),
                    "effective_date": data.get("effectiveDateTime"),
                    "has_reference_range": "referenceRange" in data
                }
                
            elif resource_type == "DiagnosticReport":
                clinical_info = {
                    "status": data.get("status"),
                    "category": data.get("category", [{}])[0].get("coding", [{}])[0].get("code"),
                    "code": data.get("code", {}).get("coding", [{}])[0].get("code"),
                    "effective_date": data.get("effectiveDateTime"),
                    "result_count": len(data.get("result", [])),
                    "has_conclusion": "conclusion" in data
                }
            
            # Add common fields
            clinical_info.update({
                "resource_type": resource_type,
                "has_patient_reference": "subject" in data,
                "last_updated": data.get("meta", {}).get("lastUpdated"),
                "extraction_timestamp": datetime.utcnow().isoformat()
            })
            
            return clinical_info
            
        except Exception as e:
            self.logger.error("Clinical data extraction failed", error=str(e))
            return {"error": str(e)}
    
    async def _encrypt_sensitive_fields(self, data: Dict[str, Any], resource_type: str) -> Dict[str, Any]:
        """Encrypt PHI fields in FHIR resource"""
        try:
            encrypted_data = data.copy()
            phi_fields = self._identify_phi_fields(data, resource_type)
            
            for field_path in phi_fields:
                field_value = self._get_nested_field(data, field_path)
                if field_value:
                    encrypted_value = self.encryption_manager.encrypt_sensitive_data(
                        str(field_value), 
                        ComplianceLevel.RESTRICTED
                    )
                    self._set_nested_field(encrypted_data, field_path, encrypted_value)
            
            # Add encryption metadata
            encrypted_data["_vita_encryption"] = {
                "encrypted_fields": phi_fields,
                "encryption_timestamp": datetime.utcnow().isoformat(),
                "compliance_level": ComplianceLevel.RESTRICTED.value
            }
            
            return encrypted_data
            
        except Exception as e:
            self.logger.error("Field encryption failed", error=str(e))
            raise SecurityException("Failed to encrypt sensitive fields")
    
    def _identify_phi_fields(self, data: Dict[str, Any], resource_type: str) -> List[str]:
        """Identify PHI fields that need encryption"""
        phi_patterns = {
            "Patient": [
                "name", "telecom", "address", "identifier", 
                "birthDate", "photo", "contact"
            ],
            "Observation": [
                "subject.reference", "performer.reference"
            ],
            "DiagnosticReport": [
                "subject.reference", "performer.reference"
            ],
            "Medication": [
                "subject.reference"
            ]
        }
        
        resource_phi = phi_patterns.get(resource_type, [])
        
        # Find actual existing fields
        existing_phi_fields = []
        for field_path in resource_phi:
            if self._field_exists(data, field_path):
                existing_phi_fields.append(field_path)
        
        return existing_phi_fields
    
    def _extract_patient_id(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract patient ID from FHIR resource"""
        if data.get("resourceType") == "Patient":
            return data.get("id")
        
        # Check for patient reference
        subject = data.get("subject", {})
        if subject.get("reference"):
            ref = subject["reference"]
            if ref.startswith("Patient/"):
                return ref.replace("Patient/", "")
        
        return None
    
    def _get_required_fields(self, resource_type: str) -> List[str]:
        """Get required fields for FHIR resource type"""
        required_fields_map = {
            "Patient": ["resourceType"],
            "Observation": ["resourceType", "status", "code"],
            "DiagnosticReport": ["resourceType", "status", "code", "subject"],
            "Medication": ["resourceType", "code"],
            "MedicationRequest": ["resourceType", "status", "medicationCodeableConcept", "subject"]
        }
        
        return required_fields_map.get(resource_type, ["resourceType"])
    
    def _validate_field_formats(self, data: Dict[str, Any], resource_type: str) -> List[str]:
        """Validate FHIR field formats"""
        errors = []
        
        # Check common format requirements
        if "birthDate" in data:
            birth_date = data["birthDate"]
            if not self._is_valid_date(birth_date):
                errors.append(f"Invalid birthDate format: {birth_date}")
        
        if "status" in data:
            status = data["status"]
            valid_statuses = self._get_valid_statuses(resource_type)
            if status not in valid_statuses:
                errors.append(f"Invalid status: {status}")
        
        return errors
    
    def _validate_fhir_version(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Validate FHIR version compatibility"""
        version_url = meta.get("profile", [""])[0] if meta.get("profile") else ""
        
        if "R4" in version_url:
            return {"version": "R4", "supported": True}
        elif "R5" in version_url:
            return {"version": "R5", "supported": True}
        else:
            return {"version": "Unknown", "supported": False}
    
    def _calculate_completeness(self, data: Dict[str, Any], resource_type: str) -> float:
        """Calculate data completeness score"""
        required_fields = self._get_required_fields(resource_type)
        optional_important_fields = self._get_important_optional_fields(resource_type)
        
        all_important_fields = required_fields + optional_important_fields
        present_fields = [field for field in all_important_fields if field in data and data[field]]
        
        if not all_important_fields:
            return 1.0
        
        return len(present_fields) / len(all_important_fields)
    
    def _check_consistency(self, data: Dict[str, Any]) -> float:
        """Check internal data consistency"""
        # Basic consistency checks
        consistency_score = 1.0
        
        # Check if patient references are consistent
        if "subject" in data and "patient" in data:
            if data["subject"].get("reference") != data["patient"].get("reference"):
                consistency_score -= 0.2
        
        return max(consistency_score, 0.0)
    
    def _check_validity(self, data: Dict[str, Any], resource_type: str) -> float:
        """Check data validity"""
        validity_score = 1.0
        
        # Check required fields
        required_fields = self._get_required_fields(resource_type)
        missing_required = [field for field in required_fields if field not in data]
        validity_score -= len(missing_required) * 0.3
        
        return max(validity_score, 0.0)
    
    def _check_timeliness(self, data: Dict[str, Any]) -> float:
        """Check data timeliness"""
        meta = data.get("meta", {})
        last_updated = meta.get("lastUpdated")
        
        if not last_updated:
            return 0.5  # No timestamp information
        
        try:
            updated_date = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            age_days = (datetime.utcnow().replace(tzinfo=updated_date.tzinfo) - updated_date).days
            
            # Recent data gets higher score
            if age_days <= 1:
                return 1.0
            elif age_days <= 7:
                return 0.9
            elif age_days <= 30:
                return 0.7
            elif age_days <= 365:
                return 0.5
            else:
                return 0.2
                
        except (ValueError, TypeError):
            return 0.3
    
    def _estimate_accuracy(self, data: Dict[str, Any]) -> float:
        """Estimate data accuracy based on heuristics"""
        accuracy_score = 1.0
        
        # Check for common data quality issues
        text_fields = self._get_text_fields(data)
        for field, value in text_fields.items():
            if isinstance(value, str):
                # Check for placeholder text
                if value.lower() in ["test", "unknown", "n/a", "null", "none"]:
                    accuracy_score -= 0.1
                
                # Check for suspicious patterns
                if value.count("x") > len(value) * 0.5:  # Too many x's
                    accuracy_score -= 0.1
        
        return max(accuracy_score, 0.0)
    
    def _get_value_type(self, observation_data: Dict[str, Any]) -> str:
        """Get the type of observation value"""
        value_keys = [key for key in observation_data.keys() if key.startswith("value")]
        return value_keys[0] if value_keys else "unknown"
    
    def _get_important_optional_fields(self, resource_type: str) -> List[str]:
        """Get important optional fields for completeness calculation"""
        optional_fields_map = {
            "Patient": ["birthDate", "gender", "address", "telecom"],
            "Observation": ["valueQuantity", "effectiveDateTime", "performer"],
            "DiagnosticReport": ["effectiveDateTime", "performer", "result"],
            "Medication": ["form", "ingredient"],
        }
        
        return optional_fields_map.get(resource_type, [])
    
    def _get_valid_statuses(self, resource_type: str) -> List[str]:
        """Get valid status values for resource type"""
        status_map = {
            "Observation": ["registered", "preliminary", "final", "amended", "corrected", "cancelled", "entered-in-error", "unknown"],
            "DiagnosticReport": ["registered", "partial", "preliminary", "final", "amended", "corrected", "appended", "cancelled", "entered-in-error", "unknown"],
            "MedicationRequest": ["active", "on-hold", "cancelled", "completed", "entered-in-error", "stopped", "draft", "unknown"]
        }
        
        return status_map.get(resource_type, [])
    
    def _is_valid_date(self, date_str: str) -> bool:
        """Validate date format"""
        try:
            datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return True
        except (ValueError, TypeError):
            return False
    
    def _get_text_fields(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
        """Recursively get all text fields from data"""
        text_fields = {}
        
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, str):
                text_fields[full_key] = value
            elif isinstance(value, dict):
                text_fields.update(self._get_text_fields(value, full_key))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                for i, item in enumerate(value):
                    text_fields.update(self._get_text_fields(item, f"{full_key}[{i}]"))
        
        return text_fields
    
    def _field_exists(self, data: Dict[str, Any], field_path: str) -> bool:
        """Check if nested field exists"""
        try:
            self._get_nested_field(data, field_path)
            return True
        except (KeyError, TypeError, IndexError):
            return False
    
    def _get_nested_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested field value using dot notation"""
        parts = field_path.split(".")
        current = data
        
        for part in parts:
            if isinstance(current, dict):
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                current = current[int(part)]
            else:
                raise KeyError(f"Field path {field_path} not found")
        
        return current
    
    def _set_nested_field(self, data: Dict[str, Any], field_path: str, value: Any) -> None:
        """Set nested field value using dot notation"""
        parts = field_path.split(".")
        current = data
        
        for part in parts[:-1]:
            if isinstance(current, dict):
                if part not in current:
                    current[part] = {}
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                current = current[int(part)]
        
        # Set the final value
        final_key = parts[-1]
        if isinstance(current, dict):
            current[final_key] = value
        elif isinstance(current, list) and final_key.isdigit():
            current[int(final_key)] = value


# Export the secure FHIR agent
__all__ = ["SecureFHIRAgent"]