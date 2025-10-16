"""
EHR Integration Agent for healthcare data processing and management.

This agent provides Electronic Health Record (EHR) integration capabilities
with support for multiple healthcare systems, FHIR standards, and secure data exchange.
"""

# Import the enhanced agent as the primary implementation
from .enhanced_ehr_agent import EnhancedEHRAgent as EHRAgent

# For backwards compatibility, export the enhanced agent
__all__ = ["EHRAgent"]