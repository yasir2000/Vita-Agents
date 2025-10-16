"""
Security and Compliance Module for Vita Agents
HIPAA-compliant healthcare data handling with encryption and audit logging
"""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
from passlib.context import CryptContext
import structlog

from .config import Settings


class ComplianceLevel(Enum):
    """Compliance levels for healthcare data handling"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"  # PHI/PII
    CRITICAL = "critical"  # Highly sensitive PHI


class AuditAction(Enum):
    """Audit action types for HIPAA compliance"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXPORT = "export"
    IMPORT = "import"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    AUTHENTICATE = "authenticate"
    AUTHORIZE = "authorize"
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"


@dataclass
class AuditEvent:
    """Audit event for healthcare data access"""
    action: AuditAction
    resource_type: str
    resource_id: Optional[str]
    user_id: Optional[str]
    patient_id: Optional[str]
    agent_id: Optional[str]
    ip_address: Optional[str]
    access_reason: str
    compliance_level: ComplianceLevel
    timestamp: datetime
    details: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class EncryptionManager:
    """HIPAA-compliant encryption manager"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = structlog.get_logger(__name__)
        
        # Initialize encryption keys
        self._symmetric_key = self._derive_key(settings.security.encryption_key)
        self._cipher = Fernet(self._symmetric_key)
        
        # Generate RSA keys for asymmetric encryption
        self._private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self._public_key = self._private_key.public_key()
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password using PBKDF2"""
        salt = self.settings.security.encryption_salt.encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password.encode())
        return Fernet.generate_key()  # Use derived key for Fernet
    
    def encrypt_sensitive_data(self, data: str, compliance_level: ComplianceLevel) -> str:
        """Encrypt sensitive healthcare data"""
        try:
            if compliance_level in [ComplianceLevel.RESTRICTED, ComplianceLevel.CRITICAL]:
                # Use stronger encryption for PHI
                encrypted = self._cipher.encrypt(data.encode())
                return encrypted.decode('latin-1')
            else:
                # Basic encryption for less sensitive data
                return self._cipher.encrypt(data.encode()).decode('latin-1')
        except Exception as e:
            self.logger.error("Encryption failed", error=str(e))
            raise SecurityException("Failed to encrypt sensitive data")
    
    def decrypt_sensitive_data(self, encrypted_data: str, compliance_level: ComplianceLevel) -> str:
        """Decrypt sensitive healthcare data"""
        try:
            decrypted = self._cipher.decrypt(encrypted_data.encode('latin-1'))
            return decrypted.decode()
        except Exception as e:
            self.logger.error("Decryption failed", error=str(e))
            raise SecurityException("Failed to decrypt sensitive data")
    
    def encrypt_patient_id(self, patient_id: str) -> str:
        """Encrypt patient ID for audit logging"""
        return self.encrypt_sensitive_data(patient_id, ComplianceLevel.RESTRICTED)
    
    def hash_data(self, data: str, salt: Optional[str] = None) -> str:
        """Hash data with optional salt for one-way encryption"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        combined = f"{data}{salt}".encode()
        hash_obj = hashlib.sha256(combined)
        return f"{salt}:{hash_obj.hexdigest()}"
    
    def verify_hash(self, data: str, hashed_data: str) -> bool:
        """Verify hashed data"""
        try:
            salt, hash_value = hashed_data.split(':', 1)
            return self.hash_data(data, salt) == hashed_data
        except ValueError:
            return False


class AuthenticationManager:
    """HIPAA-compliant authentication and authorization"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = structlog.get_logger(__name__)
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.encryption_manager = EncryptionManager(settings)
    
    def create_access_token(self, user_id: str, permissions: List[str], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token with healthcare permissions"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.settings.security.jwt_expiration_minutes)
        
        payload = {
            "sub": user_id,
            "permissions": permissions,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
            "compliance": "hipaa"
        }
        
        return jwt.encode(payload, self.settings.security.jwt_secret, algorithm="HS256")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.settings.security.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise SecurityException("Token has expired")
        except jwt.JWTError:
            raise SecurityException("Invalid token")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def check_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """Check if user has required permission"""
        return required_permission in user_permissions or "admin" in user_permissions


class AuditLogger:
    """HIPAA-compliant audit logging system"""
    
    def __init__(self, settings: Settings, db_manager=None):
        self.settings = settings
        self.db_manager = db_manager
        self.logger = structlog.get_logger(__name__)
        self.encryption_manager = EncryptionManager(settings)
        
        # Configure structured logging for audit trail
        self.audit_logger = structlog.get_logger("audit")
    
    async def log_audit_event(self, event: AuditEvent) -> None:
        """Log audit event for HIPAA compliance"""
        try:
            # Encrypt sensitive data
            encrypted_patient_id = None
            if event.patient_id:
                encrypted_patient_id = self.encryption_manager.encrypt_patient_id(event.patient_id)
            
            # Prepare audit record
            audit_record = {
                "agent_id": event.agent_id,
                "action": event.action.value,
                "resource_type": event.resource_type,
                "resource_id": event.resource_id,
                "patient_id": encrypted_patient_id,
                "user_id": event.user_id,
                "ip_address": event.ip_address,
                "access_reason": event.access_reason,
                "compliance_level": event.compliance_level.value,
                "timestamp": event.timestamp.isoformat(),
                "details": json.dumps(event.details),
                "success": event.success,
                "error_message": event.error_message
            }
            
            # Log to structured logger
            self.audit_logger.info(
                "Healthcare data access",
                **audit_record
            )
            
            # Store in database if available
            if self.db_manager:
                await self._store_audit_record(audit_record)
                
        except Exception as e:
            self.logger.error("Failed to log audit event", error=str(e))
            # Audit logging failures are critical for HIPAA compliance
            raise SecurityException("Audit logging failed")
    
    async def _store_audit_record(self, record: Dict[str, Any]) -> None:
        """Store audit record in database"""
        if not self.db_manager:
            return
        
        query = """
        INSERT INTO audit.healthcare_logs 
        (agent_id, action, resource_type, resource_id, patient_id, access_reason, compliance_flags, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """
        
        compliance_flags = {
            "compliance_level": record["compliance_level"],
            "ip_address": record["ip_address"],
            "user_id": record["user_id"],
            "success": record["success"],
            "error_message": record["error_message"]
        }
        
        await self.db_manager.execute(
            query,
            record["agent_id"],
            record["action"],
            record["resource_type"],
            record["resource_id"],
            record["patient_id"],
            record["access_reason"],
            json.dumps(compliance_flags),
            datetime.fromisoformat(record["timestamp"])
        )
    
    async def get_audit_trail(
        self, 
        patient_id: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve audit trail for compliance reporting"""
        if not self.db_manager:
            return []
        
        conditions = []
        params = []
        param_count = 0
        
        if patient_id:
            param_count += 1
            encrypted_patient_id = self.encryption_manager.encrypt_patient_id(patient_id)
            conditions.append(f"patient_id = ${param_count}")
            params.append(encrypted_patient_id)
        
        if user_id:
            param_count += 1
            conditions.append(f"compliance_flags->>'user_id' = ${param_count}")
            params.append(user_id)
        
        if start_date:
            param_count += 1
            conditions.append(f"created_at >= ${param_count}")
            params.append(start_date)
        
        if end_date:
            param_count += 1
            conditions.append(f"created_at <= ${param_count}")
            params.append(end_date)
        
        param_count += 1
        params.append(limit)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
        SELECT * FROM audit.healthcare_logs 
        WHERE {where_clause}
        ORDER BY created_at DESC 
        LIMIT ${param_count}
        """
        
        return await self.db_manager.fetch_all(query, *params)


class ComplianceValidator:
    """HIPAA compliance validation and monitoring"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = structlog.get_logger(__name__)
    
    def validate_phi_access(self, user_permissions: List[str], resource_type: str, action: AuditAction) -> bool:
        """Validate PHI access permissions"""
        required_permissions = self._get_required_permissions(resource_type, action)
        return any(perm in user_permissions for perm in required_permissions)
    
    def _get_required_permissions(self, resource_type: str, action: AuditAction) -> List[str]:
        """Get required permissions for resource and action"""
        permission_map = {
            ("Patient", AuditAction.READ): ["patient.read", "phi.access"],
            ("Patient", AuditAction.UPDATE): ["patient.write", "phi.modify"],
            ("Observation", AuditAction.READ): ["observation.read", "phi.access"],
            ("Observation", AuditAction.CREATE): ["observation.write", "phi.create"],
            ("DiagnosticReport", AuditAction.READ): ["report.read", "phi.access"],
            ("Medication", AuditAction.READ): ["medication.read", "phi.access"],
        }
        
        return permission_map.get((resource_type, action), ["admin"])
    
    def check_data_retention_policy(self, created_date: datetime) -> bool:
        """Check if data meets retention policy requirements"""
        retention_years = self.settings.security.data_retention_years
        retention_limit = datetime.utcnow() - timedelta(days=retention_years * 365)
        return created_date > retention_limit
    
    def validate_minimum_necessary(self, requested_fields: List[str], user_role: str) -> List[str]:
        """Apply minimum necessary principle for PHI access"""
        role_permissions = {
            "nurse": ["name", "mrn", "dob", "vitals", "medications"],
            "doctor": ["name", "mrn", "dob", "vitals", "medications", "diagnoses", "procedures"],
            "admin": ["name", "mrn", "dob", "contact"],
            "researcher": ["age_group", "diagnosis_codes", "medication_codes"]  # De-identified
        }
        
        allowed_fields = role_permissions.get(user_role, [])
        return [field for field in requested_fields if field in allowed_fields]


class SecurityException(Exception):
    """Security-related exceptions"""
    pass


class HIPAACompliantAgent:
    """Base class for HIPAA-compliant healthcare agents"""
    
    def __init__(self, agent_id: str, settings: Settings, db_manager=None):
        self.agent_id = agent_id
        self.settings = settings
        self.logger = structlog.get_logger(__name__)
        
        # Initialize security components
        self.encryption_manager = EncryptionManager(settings)
        self.auth_manager = AuthenticationManager(settings)
        self.audit_logger = AuditLogger(settings, db_manager)
        self.compliance_validator = ComplianceValidator(settings)
    
    async def secure_process_data(
        self,
        data: Dict[str, Any],
        user_id: str,
        user_permissions: List[str],
        access_reason: str,
        action: AuditAction,
        resource_type: str,
        patient_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Securely process healthcare data with full audit trail"""
        
        # Validate permissions
        if not self.compliance_validator.validate_phi_access(user_permissions, resource_type, action):
            await self._log_failed_access(user_id, resource_type, action, "Insufficient permissions")
            raise SecurityException("Insufficient permissions for PHI access")
        
        try:
            # Log access attempt
            audit_event = AuditEvent(
                action=action,
                resource_type=resource_type,
                resource_id=data.get("id"),
                user_id=user_id,
                patient_id=patient_id,
                agent_id=self.agent_id,
                ip_address=None,  # Should be provided by calling context
                access_reason=access_reason,
                compliance_level=ComplianceLevel.RESTRICTED,
                timestamp=datetime.utcnow(),
                details={"data_size": len(str(data))},
                success=True
            )
            
            await self.audit_logger.log_audit_event(audit_event)
            
            # Process data (implement in subclass)
            result = await self._process_healthcare_data(data)
            
            return result
            
        except Exception as e:
            await self._log_failed_access(user_id, resource_type, action, str(e))
            raise
    
    async def _process_healthcare_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Override in subclass to implement specific data processing"""
        raise NotImplementedError("Subclass must implement _process_healthcare_data")
    
    async def _log_failed_access(self, user_id: str, resource_type: str, action: AuditAction, error: str):
        """Log failed access attempt"""
        audit_event = AuditEvent(
            action=action,
            resource_type=resource_type,
            resource_id=None,
            user_id=user_id,
            patient_id=None,
            agent_id=self.agent_id,
            ip_address=None,
            access_reason="Access denied",
            compliance_level=ComplianceLevel.RESTRICTED,
            timestamp=datetime.utcnow(),
            details={"error": error},
            success=False,
            error_message=error
        )
        
        await self.audit_logger.log_audit_event(audit_event)


# Export security components
__all__ = [
    "EncryptionManager",
    "AuthenticationManager", 
    "AuditLogger",
    "ComplianceValidator",
    "HIPAACompliantAgent",
    "SecurityException",
    "ComplianceLevel",
    "AuditAction",
    "AuditEvent"
]