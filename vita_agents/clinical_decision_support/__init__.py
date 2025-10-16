"""
Clinical Decision Support Package for Vita Agents.

This package provides comprehensive clinical decision support capabilities
including drug interaction checking, allergy screening, evidence-based
recommendations, and clinical risk assessment.

Key Components:
- Drug interaction checking with major/minor/contraindicated classifications
- Allergy screening with cross-reactivity detection
- Evidence-based clinical guideline recommendations
- Clinical risk assessment and scoring
- Lab value interpretation and alerts
- Clinical workflow integration
"""

from .core import (
    # Core classes
    AdvancedClinicalDecisionSupport,
    DrugInteractionChecker,
    AllergyScreener,
    ClinicalGuidelineEngine,
    
    # Data models
    ClinicalAlert,
    DrugInteraction,
    AllergyAlert,
    ClinicalRecommendation,
    RiskAssessment,
    
    # Enums
    AlertSeverity,
    InteractionType,
    AllergyReactionType,
    RiskCategory,
    EvidenceLevel,
)

# Create default instance for easy access
clinical_decision_support = AdvancedClinicalDecisionSupport()

__all__ = [
    # Main system
    "AdvancedClinicalDecisionSupport",
    "clinical_decision_support",
    
    # Component classes
    "DrugInteractionChecker", 
    "AllergyScreener",
    "ClinicalGuidelineEngine",
    
    # Data models
    "ClinicalAlert",
    "DrugInteraction",
    "AllergyAlert", 
    "ClinicalRecommendation",
    "RiskAssessment",
    
    # Enums
    "AlertSeverity",
    "InteractionType",
    "AllergyReactionType",
    "RiskCategory",
    "EvidenceLevel",
]