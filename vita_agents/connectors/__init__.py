"""
EHR Connectors package initialization for Vita Agents.

This package provides enhanced connectivity to major EHR vendors with
vendor-specific optimizations, connection pooling, and health monitoring.
"""

from .base import (
    BaseEHRConnector,
    EHRConnectionConfig,
    EHRResponse,
    EHRVendor,
    AuthenticationType,
    SyncMode,
    SyncResult,
    EHRConnectorError,
    EHRAuthenticationError,
    EHRRateLimitError,
    EHRConnectionError,
)

from .epic import EpicConnector
from .cerner import CernerConnector
from .allscripts import AllscriptsConnector

from .factory import (
    EHRConnectorFactory,
    EHRConnectionPool,
    EHRSystemStatus,
    ehr_factory,
)

__all__ = [
    # Base classes and types
    "BaseEHRConnector",
    "EHRConnectionConfig",
    "EHRResponse",
    "EHRVendor",
    "AuthenticationType",
    "SyncMode",
    "SyncResult",
    
    # Exceptions
    "EHRConnectorError",
    "EHRAuthenticationError",
    "EHRRateLimitError",
    "EHRConnectionError",
    
    # Vendor connectors
    "EpicConnector",
    "CernerConnector",
    "AllscriptsConnector",
    
    # Factory and management
    "EHRConnectorFactory",
    "EHRConnectionPool",
    "EHRSystemStatus",
    "ehr_factory",
]

# Version information
__version__ = "2.0.0"