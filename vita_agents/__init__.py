"""
Vita Agents: Multi-Agent AI Framework for Healthcare Interoperability

This package provides a comprehensive framework for building AI agents
that can work with healthcare data standards including FHIR, HL7, and EHR systems.
"""

__version__ = "0.1.0"
__author__ = "Yasir"
__email__ = "yasir@vita-agents.org"
__license__ = "Apache 2.0"

from vita_agents.core.orchestrator import AgentOrchestrator
from vita_agents.agents.fhir_agent import FHIRAgent
from vita_agents.agents.hl7_agent import HL7Agent
from vita_agents.agents.ehr_agent import EHRAgent
from vita_agents.agents.clinical_decision_agent import ClinicalDecisionSupportAgent
from vita_agents.agents.data_harmonization_agent import DataHarmonizationAgent

__all__ = [
    "AgentOrchestrator",
    "FHIRAgent",
    "HL7Agent", 
    "EHRAgent",
    "ClinicalDecisionSupportAgent",
    "DataHarmonizationAgent",
]

# Version info
VERSION_INFO = tuple(map(int, __version__.split(".")))

# Package metadata
PACKAGE_NAME = "vita-agents"
PACKAGE_DESCRIPTION = "Multi-Agent AI Framework for Healthcare Interoperability"
PACKAGE_URL = "https://github.com/yasir2000/vita-agents"
DOCUMENTATION_URL = "https://vita-agents.readthedocs.io"
BUG_TRACKER_URL = "https://github.com/yasir2000/vita-agents/issues"