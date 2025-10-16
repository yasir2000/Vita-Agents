"""
Advanced Clinical Decision Support System for Vita Agents.

This module provides comprehensive clinical decision support capabilities including:
- Drug interaction checking and alerts
- Allergy screening and contraindication detection
- Evidence-based care recommendations
- Clinical risk assessment and scoring
- Lab value interpretation and alerts
- Clinical guideline adherence checking
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import structlog
from pydantic import BaseModel, Field

from vita_agents.core.exceptions import VitaAgentsError


logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels for clinical decision support."""
    
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class InteractionType(Enum):
    """Types of drug interactions."""
    
    MAJOR = "major"
    MODERATE = "moderate"
    MINOR = "minor"
    CONTRAINDICATED = "contraindicated"


class AllergyReactionType(Enum):
    """Types of allergic reactions."""
    
    ANAPHYLAXIS = "anaphylaxis"
    SEVERE_CUTANEOUS = "severe_cutaneous"
    RESPIRATORY = "respiratory"
    GASTROINTESTINAL = "gastrointestinal"
    MILD_CUTANEOUS = "mild_cutaneous"
    OTHER = "other"


class RiskCategory(Enum):
    """Clinical risk categories."""
    
    LOW_RISK = "low_risk"
    MODERATE_RISK = "moderate_risk"
    HIGH_RISK = "high_risk"
    VERY_HIGH_RISK = "very_high_risk"


class EvidenceLevel(Enum):
    """Evidence levels for clinical recommendations."""
    
    LEVEL_A = "level_a"  # High-quality evidence
    LEVEL_B = "level_b"  # Moderate-quality evidence
    LEVEL_C = "level_c"  # Low-quality evidence
    EXPERT_OPINION = "expert_opinion"


@dataclass
class ClinicalAlert:
    """Clinical decision support alert."""
    
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    alert_type: str
    patient_id: str
    triggered_by: str
    recommendation: str
    evidence_level: Optional[EvidenceLevel] = None
    source: Optional[str] = None
    references: List[str] = None
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.references is None:
            self.references = []


@dataclass
class DrugInteraction:
    """Drug-drug interaction information."""
    
    drug_a: str
    drug_b: str
    interaction_type: InteractionType
    severity: AlertSeverity
    mechanism: str
    clinical_effect: str
    recommendation: str
    evidence_level: EvidenceLevel
    frequency: Optional[str] = None
    onset: Optional[str] = None
    documentation: Optional[str] = None
    references: List[str] = None
    
    def __post_init__(self):
        if self.references is None:
            self.references = []


@dataclass
class AllergyAlert:
    """Allergy-based clinical alert."""
    
    allergen: str
    patient_allergen: str
    reaction_type: AllergyReactionType
    severity: AlertSeverity
    cross_reactivity: bool
    recommendation: str
    alternative_medications: List[str] = None
    
    def __post_init__(self):
        if self.alternative_medications is None:
            self.alternative_medications = []


@dataclass
class ClinicalRecommendation:
    """Evidence-based clinical recommendation."""
    
    recommendation_id: str
    title: str
    description: str
    clinical_context: str
    patient_criteria: Dict[str, Any]
    recommendation_text: str
    evidence_level: EvidenceLevel
    guideline_source: str
    implementation_notes: str
    contraindications: List[str] = None
    monitoring_requirements: List[str] = None
    references: List[str] = None
    
    def __post_init__(self):
        if self.contraindications is None:
            self.contraindications = []
        if self.monitoring_requirements is None:
            self.monitoring_requirements = []
        if self.references is None:
            self.references = []


@dataclass
class RiskAssessment:
    """Clinical risk assessment result."""
    
    assessment_id: str
    risk_type: str
    risk_category: RiskCategory
    risk_score: float
    risk_factors: List[str]
    protective_factors: List[str]
    recommendations: List[str]
    time_horizon: str
    model_used: str
    confidence_level: float
    
    def __post_init__(self):
        if self.risk_factors is None:
            self.risk_factors = []
        if self.protective_factors is None:
            self.protective_factors = []
        if self.recommendations is None:
            self.recommendations = []


class DrugInteractionChecker:
    """Drug interaction checking and alert system."""
    
    def __init__(self):
        """Initialize the drug interaction checker."""
        self.interaction_database = self._load_interaction_database()
        self.drug_mappings = self._load_drug_mappings()
        
    def _load_interaction_database(self) -> Dict[str, List[DrugInteraction]]:
        """Load drug interaction database (normally from external source)."""
        # In production, this would load from a comprehensive drug interaction database
        # like First DataBank, Lexicomp, or Micromedex
        
        interactions = {
            "warfarin": [
                DrugInteraction(
                    drug_a="warfarin",
                    drug_b="aspirin", 
                    interaction_type=InteractionType.MAJOR,
                    severity=AlertSeverity.HIGH,
                    mechanism="Additive anticoagulant effects",
                    clinical_effect="Increased bleeding risk",
                    recommendation="Monitor INR closely, consider dose reduction",
                    evidence_level=EvidenceLevel.LEVEL_A,
                    frequency="Common",
                    onset="Immediate",
                    documentation="Well-documented",
                    references=["PMID:12345678", "PMID:87654321"]
                ),
                DrugInteraction(
                    drug_a="warfarin",
                    drug_b="amiodarone",
                    interaction_type=InteractionType.MAJOR,
                    severity=AlertSeverity.CRITICAL,
                    mechanism="CYP2C9 inhibition",
                    clinical_effect="Significantly increased warfarin effect",
                    recommendation="Reduce warfarin dose by 25-50%, monitor INR closely",
                    evidence_level=EvidenceLevel.LEVEL_A,
                    frequency="Common",
                    onset="Delayed (days to weeks)",
                    documentation="Well-documented",
                    references=["PMID:11111111", "PMID:22222222"]
                )
            ],
            "metformin": [
                DrugInteraction(
                    drug_a="metformin",
                    drug_b="iodinated_contrast",
                    interaction_type=InteractionType.CONTRAINDICATED,
                    severity=AlertSeverity.CRITICAL,
                    mechanism="Increased risk of lactic acidosis",
                    clinical_effect="Potentially fatal lactic acidosis",
                    recommendation="Discontinue metformin before contrast procedure",
                    evidence_level=EvidenceLevel.LEVEL_A,
                    frequency="Rare but serious",
                    onset="Hours to days",
                    documentation="Well-documented",
                    references=["PMID:33333333"]
                )
            ],
            "simvastatin": [
                DrugInteraction(
                    drug_a="simvastatin",
                    drug_b="gemfibrozil",
                    interaction_type=InteractionType.CONTRAINDICATED,
                    severity=AlertSeverity.CRITICAL,
                    mechanism="CYP3A4 inhibition and glucuronidation interference",
                    clinical_effect="Severe myopathy and rhabdomyolysis risk",
                    recommendation="Contraindicated - use alternative statin",
                    evidence_level=EvidenceLevel.LEVEL_A,
                    frequency="High risk",
                    onset="Days to weeks",
                    documentation="Well-documented",
                    references=["PMID:44444444"]
                )
            ]
        }
        
        return interactions
    
    def _load_drug_mappings(self) -> Dict[str, List[str]]:
        """Load drug name mappings (brand names, generics, etc.)."""
        return {
            "warfarin": ["coumadin", "jantoven", "marevan"],
            "aspirin": ["acetylsalicylic acid", "asa", "ecotrin", "bufferin"],
            "amiodarone": ["cordarone", "pacerone"],
            "metformin": ["glucophage", "fortamet", "glumetza"],
            "iodinated_contrast": ["iohexol", "iopamidol", "ioversol"],
            "simvastatin": ["zocor"],
            "gemfibrozil": ["lopid"]
        }
    
    def _normalize_drug_name(self, drug_name: str) -> str:
        """Normalize drug name to standard form."""
        drug_name = drug_name.lower().strip()
        
        # Check if it's already a generic name
        if drug_name in self.drug_mappings:
            return drug_name
        
        # Search through mappings for brand names
        for generic, brands in self.drug_mappings.items():
            if drug_name in brands:
                return generic
        
        return drug_name
    
    async def check_interactions(
        self,
        medications: List[str],
        patient_id: str
    ) -> List[ClinicalAlert]:
        """
        Check for drug interactions in a medication list.
        
        Args:
            medications: List of medication names
            patient_id: Patient identifier
            
        Returns:
            List of clinical alerts for detected interactions
        """
        alerts = []
        normalized_meds = [self._normalize_drug_name(med) for med in medications]
        
        # Check all pairs of medications
        for i, med_a in enumerate(normalized_meds):
            for j, med_b in enumerate(normalized_meds[i+1:], i+1):
                # Check for interactions between med_a and med_b
                interactions = self._find_interactions(med_a, med_b)
                
                for interaction in interactions:
                    alert = ClinicalAlert(
                        alert_id=f"drug_interaction_{patient_id}_{i}_{j}_{datetime.utcnow().timestamp()}",
                        title=f"Drug Interaction: {interaction.drug_a.title()} - {interaction.drug_b.title()}",
                        description=f"{interaction.clinical_effect}. {interaction.mechanism}",
                        severity=interaction.severity,
                        alert_type="drug_interaction",
                        patient_id=patient_id,
                        triggered_by=f"{medications[i]}, {medications[j]}",
                        recommendation=interaction.recommendation,
                        evidence_level=interaction.evidence_level,
                        source="drug_interaction_database",
                        references=interaction.references,
                        created_at=datetime.utcnow()
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _find_interactions(self, drug_a: str, drug_b: str) -> List[DrugInteraction]:
        """Find interactions between two drugs."""
        interactions = []
        
        # Check drug_a -> drug_b
        if drug_a in self.interaction_database:
            for interaction in self.interaction_database[drug_a]:
                if interaction.drug_b == drug_b:
                    interactions.append(interaction)
        
        # Check drug_b -> drug_a (bidirectional)
        if drug_b in self.interaction_database:
            for interaction in self.interaction_database[drug_b]:
                if interaction.drug_b == drug_a:
                    # Create reverse interaction
                    reverse_interaction = DrugInteraction(
                        drug_a=drug_b,
                        drug_b=drug_a,
                        interaction_type=interaction.interaction_type,
                        severity=interaction.severity,
                        mechanism=interaction.mechanism,
                        clinical_effect=interaction.clinical_effect,
                        recommendation=interaction.recommendation,
                        evidence_level=interaction.evidence_level,
                        frequency=interaction.frequency,
                        onset=interaction.onset,
                        documentation=interaction.documentation,
                        references=interaction.references
                    )
                    interactions.append(reverse_interaction)
        
        return interactions


class AllergyScreener:
    """Allergy screening and contraindication detection."""
    
    def __init__(self):
        """Initialize the allergy screener."""
        self.allergy_database = self._load_allergy_database()
        self.cross_reactivity_mappings = self._load_cross_reactivity_mappings()
        
    def _load_allergy_database(self) -> Dict[str, Dict[str, Any]]:
        """Load allergy and contraindication database."""
        return {
            "penicillin": {
                "related_drugs": ["amoxicillin", "ampicillin", "penicillin_v", "cloxacillin"],
                "cross_reactive_classes": ["beta_lactams"],
                "typical_reactions": [AllergyReactionType.ANAPHYLAXIS, AllergyReactionType.SEVERE_CUTANEOUS],
                "alternatives": ["cephalexin", "azithromycin", "clindamycin"]
            },
            "sulfa": {
                "related_drugs": ["sulfamethoxazole", "sulfisoxazole", "sulfadiazine"],
                "cross_reactive_classes": ["sulfonamides"],
                "typical_reactions": [AllergyReactionType.SEVERE_CUTANEOUS, AllergyReactionType.ANAPHYLAXIS],
                "alternatives": ["amoxicillin", "azithromycin", "fluoroquinolones"]
            },
            "aspirin": {
                "related_drugs": ["nsaids", "ibuprofen", "naproxen", "diclofenac"],
                "cross_reactive_classes": ["salicylates", "nsaids"],
                "typical_reactions": [AllergyReactionType.RESPIRATORY, AllergyReactionType.ANAPHYLAXIS],
                "alternatives": ["acetaminophen", "celecoxib", "tramadol"]
            },
            "shellfish": {
                "related_substances": ["iodine", "iodinated_contrast"],
                "cross_reactive_classes": ["iodine_containing"],
                "typical_reactions": [AllergyReactionType.ANAPHYLAXIS],
                "alternatives": ["non_ionic_contrast", "gadolinium_contrast"]
            }
        }
    
    def _load_cross_reactivity_mappings(self) -> Dict[str, List[str]]:
        """Load cross-reactivity mappings between allergens."""
        return {
            "penicillin": ["amoxicillin", "ampicillin", "cephalexin"],
            "sulfa": ["sulfamethoxazole", "furosemide", "hydrochlorothiazide"],
            "shellfish": ["iodinated_contrast", "povidone_iodine"],
            "latex": ["banana", "avocado", "kiwi", "chestnuts"]
        }
    
    async def screen_allergies(
        self,
        patient_allergies: List[str],
        proposed_medications: List[str],
        patient_id: str
    ) -> List[ClinicalAlert]:
        """
        Screen for allergy contraindications.
        
        Args:
            patient_allergies: List of known patient allergies
            proposed_medications: List of proposed medications
            patient_id: Patient identifier
            
        Returns:
            List of allergy-related clinical alerts
        """
        alerts = []
        
        for allergy in patient_allergies:
            allergy_lower = allergy.lower().strip()
            
            for medication in proposed_medications:
                medication_lower = medication.lower().strip()
                
                # Direct allergy match
                if self._is_direct_match(allergy_lower, medication_lower):
                    alert = self._create_allergy_alert(
                        allergy_lower, medication_lower, patient_id, 
                        is_cross_reactive=False
                    )
                    alerts.append(alert)
                
                # Cross-reactivity check
                elif self._is_cross_reactive(allergy_lower, medication_lower):
                    alert = self._create_allergy_alert(
                        allergy_lower, medication_lower, patient_id,
                        is_cross_reactive=True
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _is_direct_match(self, allergy: str, medication: str) -> bool:
        """Check if medication directly matches patient allergy."""
        # Exact match
        if allergy == medication:
            return True
        
        # Check if medication is in related drugs for this allergy
        if allergy in self.allergy_database:
            allergy_info = self.allergy_database[allergy]
            if medication in allergy_info.get("related_drugs", []):
                return True
        
        return False
    
    def _is_cross_reactive(self, allergy: str, medication: str) -> bool:
        """Check if medication has cross-reactivity with patient allergy."""
        if allergy in self.cross_reactivity_mappings:
            cross_reactive_substances = self.cross_reactivity_mappings[allergy]
            return medication in cross_reactive_substances
        
        return False
    
    def _create_allergy_alert(
        self,
        allergy: str,
        medication: str,
        patient_id: str,
        is_cross_reactive: bool
    ) -> ClinicalAlert:
        """Create an allergy-related clinical alert."""
        if is_cross_reactive:
            title = f"Cross-Reactive Allergy Alert: {medication.title()}"
            description = f"Patient has {allergy} allergy; {medication} may cause cross-reactive allergic reaction"
            severity = AlertSeverity.HIGH
        else:
            title = f"Direct Allergy Alert: {medication.title()}"
            description = f"Patient has documented {allergy} allergy; {medication} is contraindicated"
            severity = AlertSeverity.CRITICAL
        
        # Get alternatives if available
        alternatives = []
        if allergy in self.allergy_database:
            alternatives = self.allergy_database[allergy].get("alternatives", [])
        
        recommendation = f"Avoid {medication}. "
        if alternatives:
            recommendation += f"Consider alternatives: {', '.join(alternatives)}"
        else:
            recommendation += "Consult pharmacist for alternative medications."
        
        return ClinicalAlert(
            alert_id=f"allergy_{patient_id}_{allergy}_{medication}_{datetime.utcnow().timestamp()}",
            title=title,
            description=description,
            severity=severity,
            alert_type="allergy_contraindication",
            patient_id=patient_id,
            triggered_by=medication,
            recommendation=recommendation,
            evidence_level=EvidenceLevel.LEVEL_A,
            source="allergy_database",
            created_at=datetime.utcnow()
        )


class ClinicalGuidelineEngine:
    """Evidence-based clinical guideline recommendations."""
    
    def __init__(self):
        """Initialize the clinical guideline engine."""
        self.guidelines = self._load_clinical_guidelines()
        self.risk_calculators = self._load_risk_calculators()
    
    def _load_clinical_guidelines(self) -> Dict[str, List[ClinicalRecommendation]]:
        """Load clinical guidelines database."""
        guidelines = {
            "diabetes_type2": [
                ClinicalRecommendation(
                    recommendation_id="diabetes_metformin_first_line",
                    title="Metformin as First-Line Therapy",
                    description="Metformin should be first-line pharmacologic therapy for type 2 diabetes",
                    clinical_context="Type 2 diabetes management",
                    patient_criteria={
                        "diagnosis": "type_2_diabetes",
                        "hba1c": ">7.0%",
                        "contraindications": "none"
                    },
                    recommendation_text="Initiate metformin 500mg twice daily, titrate as tolerated",
                    evidence_level=EvidenceLevel.LEVEL_A,
                    guideline_source="ADA Standards of Medical Care 2024",
                    implementation_notes="Start with low dose to minimize GI side effects",
                    contraindications=["severe_kidney_disease", "metabolic_acidosis"],
                    monitoring_requirements=["kidney_function", "vitamin_b12", "lactic_acid"],
                    references=["ADA_2024_Standards", "PMID:12345678"]
                ),
                ClinicalRecommendation(
                    recommendation_id="diabetes_bp_target",
                    title="Blood Pressure Target for Diabetes",
                    description="Target blood pressure <130/80 mmHg for most diabetic patients",
                    clinical_context="Cardiovascular risk reduction in diabetes",
                    patient_criteria={
                        "diagnosis": "type_2_diabetes",
                        "cardiovascular_risk": "any"
                    },
                    recommendation_text="Target BP <130/80 mmHg using ACE-I or ARB as first-line",
                    evidence_level=EvidenceLevel.LEVEL_A,
                    guideline_source="ADA/ESC Consensus 2024",
                    implementation_notes="Use ACE inhibitor or ARB unless contraindicated",
                    monitoring_requirements=["blood_pressure", "kidney_function", "potassium"],
                    references=["ADA_ESC_2024", "PMID:87654321"]
                )
            ],
            "hypertension": [
                ClinicalRecommendation(
                    recommendation_id="htn_lifestyle_first",
                    title="Lifestyle Modifications for Hypertension",
                    description="Lifestyle modifications should be first-line for stage 1 hypertension",
                    clinical_context="Stage 1 hypertension without cardiovascular disease",
                    patient_criteria={
                        "systolic_bp": "130-139",
                        "diastolic_bp": "80-89",
                        "cardiovascular_disease": "false"
                    },
                    recommendation_text="DASH diet, sodium restriction, exercise, weight loss",
                    evidence_level=EvidenceLevel.LEVEL_A,
                    guideline_source="AHA/ACC 2017 Guidelines",
                    implementation_notes="3-month trial before adding medications",
                    monitoring_requirements=["blood_pressure", "weight", "adherence"],
                    references=["AHA_ACC_2017", "PMID:11111111"]
                )
            ],
            "cardiovascular_prevention": [
                ClinicalRecommendation(
                    recommendation_id="statin_primary_prevention",
                    title="Statin for Primary Prevention",
                    description="Consider statin therapy for primary prevention based on risk assessment",
                    clinical_context="Primary cardiovascular disease prevention",
                    patient_criteria={
                        "age": "40-75",
                        "ldl_cholesterol": ">70",
                        "cardiovascular_risk_score": ">7.5%"
                    },
                    recommendation_text="Initiate moderate-intensity statin therapy",
                    evidence_level=EvidenceLevel.LEVEL_A,
                    guideline_source="ACC/AHA Cholesterol Guidelines 2019",
                    implementation_notes="Shared decision making with patient",
                    monitoring_requirements=["lipid_panel", "liver_function", "muscle_symptoms"],
                    references=["ACC_AHA_2019", "PMID:22222222"]
                )
            ]
        }
        
        return guidelines
    
    def _load_risk_calculators(self) -> Dict[str, Any]:
        """Load clinical risk calculation algorithms."""
        return {
            "cardiovascular_risk": {
                "name": "Pooled Cohort Equations",
                "parameters": ["age", "sex", "race", "total_cholesterol", "hdl_cholesterol", 
                             "systolic_bp", "bp_treatment", "diabetes", "smoking"],
                "output": "10_year_ascvd_risk_percentage"
            },
            "bleeding_risk": {
                "name": "HAS-BLED Score",
                "parameters": ["hypertension", "abnormal_kidney_liver", "stroke", "bleeding", 
                             "labile_inr", "elderly", "drugs_alcohol"],
                "output": "bleeding_risk_score"
            },
            "ckd_progression": {
                "name": "CKD Progression Risk",
                "parameters": ["egfr", "albuminuria", "age", "diabetes", "hypertension"],
                "output": "ckd_progression_risk"
            }
        }
    
    async def get_recommendations(
        self,
        patient_data: Dict[str, Any],
        clinical_context: str,
        patient_id: str
    ) -> List[ClinicalAlert]:
        """
        Generate evidence-based clinical recommendations.
        
        Args:
            patient_data: Patient clinical data
            clinical_context: Clinical context (e.g., "diabetes_type2")
            patient_id: Patient identifier
            
        Returns:
            List of clinical recommendation alerts
        """
        alerts = []
        
        if clinical_context in self.guidelines:
            guidelines = self.guidelines[clinical_context]
            
            for guideline in guidelines:
                if self._patient_meets_criteria(patient_data, guideline.patient_criteria):
                    alert = ClinicalAlert(
                        alert_id=f"guideline_{patient_id}_{guideline.recommendation_id}_{datetime.utcnow().timestamp()}",
                        title=f"Clinical Guideline: {guideline.title}",
                        description=guideline.description,
                        severity=AlertSeverity.MODERATE,
                        alert_type="clinical_guideline",
                        patient_id=patient_id,
                        triggered_by=clinical_context,
                        recommendation=guideline.recommendation_text,
                        evidence_level=guideline.evidence_level,
                        source=guideline.guideline_source,
                        references=guideline.references,
                        created_at=datetime.utcnow()
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _patient_meets_criteria(
        self,
        patient_data: Dict[str, Any],
        criteria: Dict[str, Any]
    ) -> bool:
        """Check if patient meets guideline criteria."""
        for criterion, required_value in criteria.items():
            patient_value = patient_data.get(criterion)
            
            if patient_value is None:
                continue  # Skip missing data
            
            # Handle different types of criteria
            if isinstance(required_value, str):
                if str(patient_value).lower() != required_value.lower():
                    return False
            elif isinstance(required_value, (int, float)):
                if patient_value != required_value:
                    return False
            elif isinstance(required_value, dict):
                # Handle range criteria like {"min": 130, "max": 139}
                if "min" in required_value and patient_value < required_value["min"]:
                    return False
                if "max" in required_value and patient_value > required_value["max"]:
                    return False
        
        return True
    
    async def calculate_risk_scores(
        self,
        patient_data: Dict[str, Any],
        risk_types: List[str],
        patient_id: str
    ) -> List[RiskAssessment]:
        """
        Calculate clinical risk scores for the patient.
        
        Args:
            patient_data: Patient clinical data
            risk_types: List of risk types to calculate
            patient_id: Patient identifier
            
        Returns:
            List of risk assessments
        """
        assessments = []
        
        for risk_type in risk_types:
            if risk_type in self.risk_calculators:
                assessment = await self._calculate_single_risk(
                    patient_data, risk_type, patient_id
                )
                if assessment:
                    assessments.append(assessment)
        
        return assessments
    
    async def _calculate_single_risk(
        self,
        patient_data: Dict[str, Any],
        risk_type: str,
        patient_id: str
    ) -> Optional[RiskAssessment]:
        """Calculate a single risk score."""
        calculator = self.risk_calculators[risk_type]
        
        # Simple implementation - in production would use validated algorithms
        if risk_type == "cardiovascular_risk":
            return await self._calculate_cardiovascular_risk(patient_data, patient_id)
        elif risk_type == "bleeding_risk":
            return await self._calculate_bleeding_risk(patient_data, patient_id)
        
        return None
    
    async def _calculate_cardiovascular_risk(
        self,
        patient_data: Dict[str, Any],
        patient_id: str
    ) -> Optional[RiskAssessment]:
        """Calculate 10-year cardiovascular risk using simplified Pooled Cohort Equations."""
        required_params = ["age", "sex", "total_cholesterol", "hdl_cholesterol", 
                          "systolic_bp", "diabetes", "smoking"]
        
        # Check if we have required data
        missing_params = [p for p in required_params if p not in patient_data]
        if missing_params:
            return None
        
        # Simplified risk calculation (production would use validated algorithms)
        risk_score = 0.0
        risk_factors = []
        
        # Age factor
        age = patient_data.get("age", 0)
        if age > 65:
            risk_score += 20
            risk_factors.append("Advanced age (>65)")
        elif age > 55:
            risk_score += 10
            risk_factors.append("Age >55")
        
        # Diabetes
        if patient_data.get("diabetes", False):
            risk_score += 15
            risk_factors.append("Diabetes mellitus")
        
        # Smoking
        if patient_data.get("smoking", False):
            risk_score += 10
            risk_factors.append("Current smoking")
        
        # Hypertension
        systolic_bp = patient_data.get("systolic_bp", 120)
        if systolic_bp > 140:
            risk_score += 15
            risk_factors.append("Hypertension")
        elif systolic_bp > 130:
            risk_score += 5
            risk_factors.append("Elevated blood pressure")
        
        # Cholesterol
        ldl = patient_data.get("ldl_cholesterol", 100)
        if ldl > 160:
            risk_score += 10
            risk_factors.append("High LDL cholesterol")
        elif ldl > 130:
            risk_score += 5
            risk_factors.append("Elevated LDL cholesterol")
        
        # Determine risk category
        if risk_score >= 30:
            risk_category = RiskCategory.VERY_HIGH_RISK
        elif risk_score >= 20:
            risk_category = RiskCategory.HIGH_RISK
        elif risk_score >= 10:
            risk_category = RiskCategory.MODERATE_RISK
        else:
            risk_category = RiskCategory.LOW_RISK
        
        recommendations = []
        if risk_category in [RiskCategory.HIGH_RISK, RiskCategory.VERY_HIGH_RISK]:
            recommendations.extend([
                "Consider statin therapy",
                "Aggressive blood pressure control",
                "Smoking cessation if applicable",
                "Diabetes management if applicable"
            ])
        
        return RiskAssessment(
            assessment_id=f"cv_risk_{patient_id}_{datetime.utcnow().timestamp()}",
            risk_type="cardiovascular",
            risk_category=risk_category,
            risk_score=min(risk_score, 100),  # Cap at 100%
            risk_factors=risk_factors,
            protective_factors=[],
            recommendations=recommendations,
            time_horizon="10_years",
            model_used="Simplified Pooled Cohort Equations",
            confidence_level=0.8
        )
    
    async def _calculate_bleeding_risk(
        self,
        patient_data: Dict[str, Any],
        patient_id: str
    ) -> Optional[RiskAssessment]:
        """Calculate bleeding risk using simplified HAS-BLED score."""
        risk_score = 0
        risk_factors = []
        
        # HAS-BLED components
        if patient_data.get("hypertension", False):
            risk_score += 1
            risk_factors.append("Hypertension")
        
        if patient_data.get("abnormal_kidney_liver", False):
            risk_score += 1
            risk_factors.append("Abnormal kidney/liver function")
        
        if patient_data.get("stroke_history", False):
            risk_score += 1
            risk_factors.append("Previous stroke")
        
        if patient_data.get("bleeding_history", False):
            risk_score += 1
            risk_factors.append("Previous bleeding")
        
        if patient_data.get("age", 0) > 65:
            risk_score += 1
            risk_factors.append("Age >65")
        
        if patient_data.get("antiplatelet_use", False):
            risk_score += 1
            risk_factors.append("Antiplatelet use")
        
        # Determine risk category
        if risk_score >= 3:
            risk_category = RiskCategory.HIGH_RISK
        elif risk_score == 2:
            risk_category = RiskCategory.MODERATE_RISK
        else:
            risk_category = RiskCategory.LOW_RISK
        
        recommendations = []
        if risk_category == RiskCategory.HIGH_RISK:
            recommendations.extend([
                "Careful monitoring if on anticoagulation",
                "Consider bleeding risk vs thrombotic benefit",
                "Regular follow-up recommended"
            ])
        
        return RiskAssessment(
            assessment_id=f"bleeding_risk_{patient_id}_{datetime.utcnow().timestamp()}",
            risk_type="bleeding",
            risk_category=risk_category,
            risk_score=float(risk_score),
            risk_factors=risk_factors,
            protective_factors=[],
            recommendations=recommendations,
            time_horizon="1_year",
            model_used="HAS-BLED Score",
            confidence_level=0.9
        )


class AdvancedClinicalDecisionSupport:
    """
    Comprehensive clinical decision support system.
    
    Integrates drug interaction checking, allergy screening, guideline recommendations,
    and risk assessment to provide comprehensive clinical decision support.
    """
    
    def __init__(self):
        """Initialize the advanced clinical decision support system."""
        self.drug_checker = DrugInteractionChecker()
        self.allergy_screener = AllergyScreener()
        self.guideline_engine = ClinicalGuidelineEngine()
        
        # Alert management
        self.active_alerts: Dict[str, List[ClinicalAlert]] = {}
        self.alert_history: List[ClinicalAlert] = []
        
    async def comprehensive_analysis(
        self,
        patient_data: Dict[str, Any],
        medications: List[str],
        allergies: List[str],
        clinical_contexts: List[str],
        patient_id: str
    ) -> Dict[str, Any]:
        """
        Perform comprehensive clinical decision support analysis.
        
        Args:
            patient_data: Complete patient clinical data
            medications: List of current/proposed medications
            allergies: List of known patient allergies
            clinical_contexts: List of relevant clinical contexts
            patient_id: Patient identifier
            
        Returns:
            Comprehensive analysis results
        """
        results = {
            "patient_id": patient_id,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "drug_interactions": [],
            "allergy_alerts": [],
            "clinical_recommendations": [],
            "risk_assessments": [],
            "summary": {
                "total_alerts": 0,
                "critical_alerts": 0,
                "high_priority_alerts": 0,
                "recommendations_count": 0
            }
        }
        
        try:
            # 1. Drug interaction analysis
            if medications:
                drug_alerts = await self.drug_checker.check_interactions(
                    medications, patient_id
                )
                results["drug_interactions"] = [
                    self._alert_to_dict(alert) for alert in drug_alerts
                ]
            
            # 2. Allergy screening
            if allergies and medications:
                allergy_alerts = await self.allergy_screener.screen_allergies(
                    allergies, medications, patient_id
                )
                results["allergy_alerts"] = [
                    self._alert_to_dict(alert) for alert in allergy_alerts
                ]
            
            # 3. Clinical guideline recommendations
            guideline_alerts = []
            for context in clinical_contexts:
                context_alerts = await self.guideline_engine.get_recommendations(
                    patient_data, context, patient_id
                )
                guideline_alerts.extend(context_alerts)
            
            results["clinical_recommendations"] = [
                self._alert_to_dict(alert) for alert in guideline_alerts
            ]
            
            # 4. Risk assessments
            risk_types = ["cardiovascular_risk", "bleeding_risk"]
            risk_assessments = await self.guideline_engine.calculate_risk_scores(
                patient_data, risk_types, patient_id
            )
            results["risk_assessments"] = [
                self._risk_assessment_to_dict(assessment) for assessment in risk_assessments
            ]
            
            # 5. Generate summary
            all_alerts = (drug_alerts + allergy_alerts + guideline_alerts)
            results["summary"] = {
                "total_alerts": len(all_alerts),
                "critical_alerts": len([a for a in all_alerts if a.severity == AlertSeverity.CRITICAL]),
                "high_priority_alerts": len([a for a in all_alerts if a.severity == AlertSeverity.HIGH]),
                "recommendations_count": len(guideline_alerts),
                "risk_assessments_count": len(risk_assessments)
            }
            
            # Store alerts for tracking
            self.active_alerts[patient_id] = all_alerts
            self.alert_history.extend(all_alerts)
            
            logger.info(
                f"Clinical decision support analysis completed for patient {patient_id}",
                total_alerts=results["summary"]["total_alerts"],
                critical_alerts=results["summary"]["critical_alerts"]
            )
            
        except Exception as e:
            logger.error(f"Error in clinical decision support analysis: {e}")
            raise VitaAgentsError(f"Clinical analysis failed: {e}")
        
        return results
    
    def _alert_to_dict(self, alert: ClinicalAlert) -> Dict[str, Any]:
        """Convert ClinicalAlert to dictionary format."""
        return {
            "alert_id": alert.alert_id,
            "title": alert.title,
            "description": alert.description,
            "severity": alert.severity.value,
            "alert_type": alert.alert_type,
            "patient_id": alert.patient_id,
            "triggered_by": alert.triggered_by,
            "recommendation": alert.recommendation,
            "evidence_level": alert.evidence_level.value if alert.evidence_level else None,
            "source": alert.source,
            "references": alert.references,
            "created_at": alert.created_at.isoformat(),
            "expires_at": alert.expires_at.isoformat() if alert.expires_at else None
        }
    
    def _risk_assessment_to_dict(self, assessment: RiskAssessment) -> Dict[str, Any]:
        """Convert RiskAssessment to dictionary format."""
        return {
            "assessment_id": assessment.assessment_id,
            "risk_type": assessment.risk_type,
            "risk_category": assessment.risk_category.value,
            "risk_score": assessment.risk_score,
            "risk_factors": assessment.risk_factors,
            "protective_factors": assessment.protective_factors,
            "recommendations": assessment.recommendations,
            "time_horizon": assessment.time_horizon,
            "model_used": assessment.model_used,
            "confidence_level": assessment.confidence_level
        }
    
    async def get_patient_alerts(
        self,
        patient_id: str,
        severity_filter: Optional[AlertSeverity] = None,
        alert_type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get active alerts for a patient.
        
        Args:
            patient_id: Patient identifier
            severity_filter: Optional severity filter
            alert_type_filter: Optional alert type filter
            
        Returns:
            List of filtered alerts
        """
        if patient_id not in self.active_alerts:
            return []
        
        alerts = self.active_alerts[patient_id]
        
        # Apply filters
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]
        
        if alert_type_filter:
            alerts = [a for a in alerts if a.alert_type == alert_type_filter]
        
        return [self._alert_to_dict(alert) for alert in alerts]
    
    async def dismiss_alert(self, alert_id: str, patient_id: str) -> bool:
        """
        Dismiss an active alert.
        
        Args:
            alert_id: Alert identifier
            patient_id: Patient identifier
            
        Returns:
            True if alert was dismissed, False if not found
        """
        if patient_id not in self.active_alerts:
            return False
        
        alerts = self.active_alerts[patient_id]
        original_count = len(alerts)
        
        # Remove alert with matching ID
        self.active_alerts[patient_id] = [
            alert for alert in alerts if alert.alert_id != alert_id
        ]
        
        return len(self.active_alerts[patient_id]) < original_count
    
    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get clinical decision support system statistics."""
        total_alerts = len(self.alert_history)
        
        if total_alerts == 0:
            return {
                "total_alerts_generated": 0,
                "alert_distribution": {},
                "most_common_alert_types": [],
                "patient_coverage": 0
            }
        
        # Alert distribution by severity
        severity_counts = {}
        for alert in self.alert_history:
            severity_counts[alert.severity.value] = severity_counts.get(alert.severity.value, 0) + 1
        
        # Alert type distribution
        type_counts = {}
        for alert in self.alert_history:
            type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1
        
        return {
            "total_alerts_generated": total_alerts,
            "alert_distribution": severity_counts,
            "alert_type_distribution": type_counts,
            "most_common_alert_types": sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "patient_coverage": len(self.active_alerts),
            "average_alerts_per_patient": total_alerts / max(len(self.active_alerts), 1)
        }