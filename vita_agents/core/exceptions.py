"""
Core exceptions for the Vita Agents system.

This module defines the base exception classes and specific error types
used throughout the healthcare AI agent framework.
"""

from typing import Optional, Dict, Any


class VitaAgentsError(Exception):
    """
    Base exception class for all Vita Agents errors.
    
    This is the parent class for all custom exceptions in the Vita Agents
    system, providing a common interface for error handling.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None
        }


class ConfigurationError(VitaAgentsError):
    """
    Raised when there are configuration-related errors.
    
    This includes missing configuration files, invalid settings,
    or malformed configuration data.
    """
    pass


class AuthenticationError(VitaAgentsError):
    """
    Raised when authentication fails.
    
    This includes invalid credentials, expired tokens, or
    authentication service unavailability.
    """
    pass


class AuthorizationError(VitaAgentsError):
    """
    Raised when authorization fails.
    
    This occurs when a user or service lacks permission to
    perform a requested operation.
    """
    pass


class ValidationError(VitaAgentsError):
    """
    Raised when data validation fails.
    
    This includes invalid input data, schema validation failures,
    or constraint violations.
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value
        if field:
            self.details["field"] = field
        if value is not None:
            self.details["value"] = value


class ConnectionError(VitaAgentsError):
    """
    Raised when connection to external services fails.
    
    This includes network connectivity issues, service unavailability,
    or connection timeout errors.
    """
    
    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        endpoint: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.service = service
        self.endpoint = endpoint
        if service:
            self.details["service"] = service
        if endpoint:
            self.details["endpoint"] = endpoint


class RateLimitError(VitaAgentsError):
    """
    Raised when rate limits are exceeded.
    
    This occurs when the system or external service rate limits
    are reached and requests need to be throttled.
    """
    
    def __init__(
        self,
        message: str,
        limit: Optional[int] = None,
        reset_time: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.limit = limit
        self.reset_time = reset_time
        if limit:
            self.details["limit"] = limit
        if reset_time:
            self.details["reset_time"] = reset_time


class TimeoutError(VitaAgentsError):
    """
    Raised when operations timeout.
    
    This occurs when operations take longer than the specified
    timeout period to complete.
    """
    
    def __init__(
        self,
        message: str,
        timeout: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.timeout = timeout
        self.operation = operation
        if timeout:
            self.details["timeout"] = timeout
        if operation:
            self.details["operation"] = operation


class DataProcessingError(VitaAgentsError):
    """
    Raised when data processing operations fail.
    
    This includes data transformation errors, parsing failures,
    or data corruption issues.
    """
    pass


class ExternalServiceError(VitaAgentsError):
    """
    Raised when external service operations fail.
    
    This includes API errors, service-specific failures,
    or external system unavailability.
    """
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.service_name = service_name
        self.status_code = status_code
        self.response_data = response_data
        
        if service_name:
            self.details["service_name"] = service_name
        if status_code:
            self.details["status_code"] = status_code
        if response_data:
            self.details["response_data"] = response_data


class AgentError(VitaAgentsError):
    """
    Raised when agent-specific operations fail.
    
    This includes agent initialization errors, task execution failures,
    or agent communication issues.
    """
    
    def __init__(
        self,
        message: str,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.agent_id = agent_id
        self.task_id = task_id
        
        if agent_id:
            self.details["agent_id"] = agent_id
        if task_id:
            self.details["task_id"] = task_id


class SecurityError(VitaAgentsError):
    """
    Raised when security-related operations fail.
    
    This includes encryption/decryption errors, security policy violations,
    or security service failures.
    """
    pass


class ComplianceError(VitaAgentsError):
    """
    Raised when compliance requirements are not met.
    
    This includes HIPAA violations, audit failures,
    or regulatory compliance issues.
    """
    
    def __init__(
        self,
        message: str,
        regulation: Optional[str] = None,
        requirement: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.regulation = regulation
        self.requirement = requirement
        
        if regulation:
            self.details["regulation"] = regulation
        if requirement:
            self.details["requirement"] = requirement


class FHIRError(VitaAgentsError):
    """
    Raised when FHIR-related operations fail.
    
    This includes FHIR validation errors, resource processing failures,
    or FHIR server communication issues.
    """
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        fhir_version: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.fhir_version = fhir_version
        
        if resource_type:
            self.details["resource_type"] = resource_type
        if resource_id:
            self.details["resource_id"] = resource_id
        if fhir_version:
            self.details["fhir_version"] = fhir_version


class HL7Error(VitaAgentsError):
    """
    Raised when HL7-related operations fail.
    
    This includes HL7 message parsing errors, validation failures,
    or HL7 communication issues.
    """
    
    def __init__(
        self,
        message: str,
        message_type: Optional[str] = None,
        segment: Optional[str] = None,
        hl7_version: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.message_type = message_type
        self.segment = segment
        self.hl7_version = hl7_version
        
        if message_type:
            self.details["message_type"] = message_type
        if segment:
            self.details["segment"] = segment
        if hl7_version:
            self.details["hl7_version"] = hl7_version


# Exception type mapping for easy categorization
EXCEPTION_CATEGORIES = {
    "configuration": [ConfigurationError],
    "security": [AuthenticationError, AuthorizationError, SecurityError],
    "validation": [ValidationError],
    "connectivity": [ConnectionError, TimeoutError, RateLimitError],
    "processing": [DataProcessingError],
    "external": [ExternalServiceError],
    "agent": [AgentError],
    "compliance": [ComplianceError],
    "standards": [FHIRError, HL7Error],
}


def get_exception_category(exception: Exception) -> Optional[str]:
    """
    Get the category of an exception.
    
    Args:
        exception: The exception to categorize
        
    Returns:
        Category name or None if not found
    """
    exception_type = type(exception)
    
    for category, exception_types in EXCEPTION_CATEGORIES.items():
        if exception_type in exception_types:
            return category
    
    return None


def is_retryable_error(exception: Exception) -> bool:
    """
    Determine if an error is retryable.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the error is retryable, False otherwise
    """
    retryable_types = [
        ConnectionError,
        TimeoutError,
        RateLimitError,
        ExternalServiceError
    ]
    
    return any(isinstance(exception, exc_type) for exc_type in retryable_types)


def format_exception_for_logging(exception: Exception) -> Dict[str, Any]:
    """
    Format an exception for structured logging.
    
    Args:
        exception: The exception to format
        
    Returns:
        Dictionary with formatted exception data
    """
    if isinstance(exception, VitaAgentsError):
        return exception.to_dict()
    
    return {
        "error_type": type(exception).__name__,
        "message": str(exception),
        "error_code": None,
        "details": {},
        "cause": None
    }