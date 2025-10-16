"""
🏥 Vita Agents: Multi-Agent AI for Healthcare Interoperability

Enterprise-grade framework for building AI agents that work with healthcare data
standards including FHIR, HL7, and EHR systems. Now with multi-engine FHIR support
for 11+ open source FHIR servers.

Key Features:
- Multi-engine FHIR operations across 11+ servers
- Advanced healthcare AI agent orchestration  
- Production-ready Docker integration
- HIPAA-compliant security and audit logging
- Professional CLI and web portal interfaces
- Real-time performance monitoring and analytics
"""

# Import version information
from vita_agents.__version__ import (
    __version__,
    __version_info__,
    __title__,
    __description__,
    __author__,
    __author_email__,
    __license__,
    __copyright__,
    __url__,
    __build__,
    __status__,
    __api_version__,
    __python_requires__,
    __fhir_versions__,
    __hl7_versions__,
    __features__,
    __release_notes__,
    get_version_info,
    print_version_info,
)

from vita_agents.core.orchestrator import AgentOrchestrator
from vita_agents.agents.fhir_agent import FHIRAgent
from vita_agents.agents.hl7_agent import HL7Agent
from vita_agents.agents.ehr_agent import EHRAgent
from vita_agents.agents.clinical_decision_agent import ClinicalDecisionSupportAgent
from vita_agents.agents.data_harmonization_agent import DataHarmonizationAgent

# Enhanced EHR Connectors
from vita_agents.connectors import (
    EHRConnectorFactory,
    ehr_factory,
    EpicConnector,
    CernerConnector,
    AllscriptsConnector,
    EHRVendor,
    EHRConnectionConfig,
)

# Enhanced FHIR Agent with multi-engine support
try:
    from vita_agents.agents.enhanced_fhir_agent import EnhancedFHIRAgent
    from vita_agents.fhir_engines.open_source_clients import (
        FHIREngineManager,
        FHIRServerConfiguration,
        FHIREngineType,
        get_server_template,
        list_server_templates,
    )
    _ENHANCED_FHIR_AVAILABLE = True
except ImportError:
    _ENHANCED_FHIR_AVAILABLE = False

__all__ = [
    # Version and metadata
    "__version__",
    "__version_info__", 
    "__title__",
    "__description__",
    "__author__",
    "__license__",
    "get_version_info",
    "print_version_info",
    
    # Core agents
    "AgentOrchestrator",
    "FHIRAgent",
    "HL7Agent", 
    "EHRAgent", 
    "ClinicalDecisionSupportAgent",
    "DataHarmonizationAgent",
    
    # Enhanced EHR Connectors
    "EHRConnectorFactory",
    "ehr_factory",
    "EpicConnector", 
    "CernerConnector",
    "AllscriptsConnector",
    "EHRVendor",
    "EHRConnectionConfig",
]

# Add enhanced FHIR exports if available
if _ENHANCED_FHIR_AVAILABLE:
    __all__.extend([
        "EnhancedFHIRAgent",
        "FHIREngineManager",
        "FHIRServerConfiguration", 
        "FHIREngineType",
        "get_server_template",
        "list_server_templates",
    ])

# Package metadata
PACKAGE_NAME = "vita-agents"
PACKAGE_DESCRIPTION = "Multi-Agent AI Framework for Healthcare Interoperability"
PACKAGE_URL = "https://github.com/yasir2000/vita-agents"
DOCUMENTATION_URL = "https://vita-agents.readthedocs.io"
BUG_TRACKER_URL = "https://github.com/yasir2000/vita-agents/issues"