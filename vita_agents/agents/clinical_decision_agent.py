"""
Clinical Decision Support Agent for Vita Agents
Provides clinical insights, drug interaction checking, and care recommendations
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import structlog

from ..core.agent import HealthcareAgent
from ..core.security import HIPAACompliantAgent, AuditAction, ComplianceLevel
from ..core.config import Settings


class AlertSeverity(Enum):
    """Clinical alert severity levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class InteractionType(Enum):
    """Drug interaction types"""
    DRUG_DRUG = "drug-drug"
    DRUG_ALLERGY = "drug-allergy"
    DRUG_CONDITION = "drug-condition"
    DRUG_LAB = "drug-lab"


@dataclass
class ClinicalAlert:
    """Clinical decision support alert"""
    alert_id: str
    severity: AlertSeverity
    message: str
    recommendation: str
    evidence: str
    source: str
    created_at: datetime
    patient_id: Optional[str] = None
    medication_involved: Optional[str] = None
    condition_involved: Optional[str] = None


@dataclass
class DrugInteraction:
    """Drug interaction information"""
    interaction_id: str
    drug1: str
    drug2: str
    interaction_type: InteractionType
    severity: AlertSeverity
    description: str
    mechanism: str
    management: str
    evidence_level: str


@dataclass
class ClinicalRecommendation:
    """Clinical care recommendation"""
    recommendation_id: str
    category: str
    title: str
    description: str
    rationale: str
    strength: str  # strong, weak, conditional
    quality_of_evidence: str  # high, moderate, low, very_low
    patient_population: str
    contraindications: List[str]
    implementation_guidance: str


class ClinicalDecisionSupportAgent(HIPAACompliantAgent, HealthcareAgent):
    """Agent for clinical decision support and care recommendations"""
    
    def __init__(self, agent_id: str, settings: Settings, db_manager=None):
        super().__init__(agent_id, settings, db_manager)
        
        self.capabilities = [
            "drug_interaction_checking",
            "allergy_screening",
            "clinical_guidelines",
            "care_recommendations",
            "risk_assessment",
            "contraindication_detection",
            "dosage_adjustment",
            "lab_value_interpretation"
        ]
        
        self.logger = structlog.get_logger(__name__)
        
        # Initialize clinical knowledge bases
        self._load_drug_interaction_database()
        self._load_clinical_guidelines()
        self._load_allergy_database()
    
    async def _process_healthcare_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process patient data for clinical decision support"""
        try:
            patient_data = data.get("patient_data", {})
            medications = data.get("medications", [])
            allergies = data.get("allergies", [])
            conditions = data.get("conditions", [])
            lab_results = data.get("lab_results", [])
            
            # Perform clinical analysis
            alerts = await self._generate_clinical_alerts(
                patient_data, medications, allergies, conditions, lab_results
            )
            
            # Check drug interactions
            interactions = await self._check_drug_interactions(medications, allergies)
            
            # Generate care recommendations
            recommendations = await self._generate_care_recommendations(
                patient_data, medications, conditions, lab_results
            )
            
            # Risk assessment
            risk_scores = await self._calculate_risk_scores(
                patient_data, conditions, medications, lab_results
            )
            
            return {
                "clinical_alerts": [alert.__dict__ for alert in alerts],
                "drug_interactions": [interaction.__dict__ for interaction in interactions],
                "care_recommendations": [rec.__dict__ for rec in recommendations],
                "risk_assessment": risk_scores,
                "processed_at": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            self.logger.error("Clinical decision support processing failed", error=str(e))
            raise
    
    async def analyze_patient_data(
        self,
        patient_data: Dict[str, Any],
        user_id: str,
        user_permissions: List[str],
        access_reason: str
    ) -> Dict[str, Any]:
        """Analyze patient data for clinical insights"""
        
        patient_id = patient_data.get("id") or patient_data.get("patient_id")
        
        return await self.secure_process_data(
            data=patient_data,
            user_id=user_id,
            user_permissions=user_permissions,
            access_reason=access_reason,
            action=AuditAction.READ,
            resource_type="ClinicalAnalysis",
            patient_id=patient_id
        )
    
    async def _generate_clinical_alerts(
        self,
        patient_data: Dict[str, Any],
        medications: List[Dict[str, Any]],
        allergies: List[Dict[str, Any]],
        conditions: List[Dict[str, Any]],
        lab_results: List[Dict[str, Any]]
    ) -> List[ClinicalAlert]:
        """Generate clinical alerts based on patient data"""
        
        alerts = []
        
        # Check for medication allergies
        allergy_alerts = await self._check_medication_allergies(medications, allergies)
        alerts.extend(allergy_alerts)
        
        # Check for contraindications
        contraindication_alerts = await self._check_contraindications(medications, conditions)
        alerts.extend(contraindication_alerts)
        
        # Check lab values
        lab_alerts = await self._check_lab_values(lab_results, medications)
        alerts.extend(lab_alerts)
        
        # Check dosing alerts
        dosing_alerts = await self._check_dosing(medications, patient_data)
        alerts.extend(dosing_alerts)
        
        return alerts
    
    async def _check_drug_interactions(
        self,
        medications: List[Dict[str, Any]],
        allergies: List[Dict[str, Any]]
    ) -> List[DrugInteraction]:
        """Check for drug-drug and drug-allergy interactions"""
        
        interactions = []
        
        # Drug-drug interactions
        for i, med1 in enumerate(medications):
            for med2 in medications[i+1:]:
                interaction = await self._check_drug_drug_interaction(med1, med2)
                if interaction:
                    interactions.append(interaction)
        
        # Drug-allergy interactions
        for medication in medications:
            for allergy in allergies:
                interaction = await self._check_drug_allergy_interaction(medication, allergy)
                if interaction:
                    interactions.append(interaction)
        
        return interactions
    
    async def _generate_care_recommendations(
        self,
        patient_data: Dict[str, Any],
        medications: List[Dict[str, Any]],
        conditions: List[Dict[str, Any]],
        lab_results: List[Dict[str, Any]]
    ) -> List[ClinicalRecommendation]:
        """Generate evidence-based care recommendations"""
        
        recommendations = []
        
        # Get patient demographics
        age = self._calculate_age(patient_data.get("birthDate"))
        gender = patient_data.get("gender")
        
        # Condition-based recommendations
        for condition in conditions:
            condition_recs = await self._get_condition_recommendations(
                condition, age, gender, medications, lab_results
            )
            recommendations.extend(condition_recs)
        
        # Preventive care recommendations
        preventive_recs = await self._get_preventive_care_recommendations(
            patient_data, conditions
        )
        recommendations.extend(preventive_recs)
        
        # Medication optimization
        med_recs = await self._get_medication_recommendations(
            medications, conditions, lab_results, age
        )
        recommendations.extend(med_recs)
        
        return recommendations
    
    async def _calculate_risk_scores(
        self,
        patient_data: Dict[str, Any],
        conditions: List[Dict[str, Any]],
        medications: List[Dict[str, Any]],
        lab_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate various clinical risk scores"""
        
        age = self._calculate_age(patient_data.get("birthDate"))
        gender = patient_data.get("gender")
        
        risk_scores = {}
        
        # Cardiovascular risk
        risk_scores["cardiovascular"] = await self._calculate_cardiovascular_risk(
            age, gender, conditions, lab_results
        )
        
        # Fall risk
        risk_scores["fall_risk"] = await self._calculate_fall_risk(
            age, medications, conditions
        )
        
        # Bleeding risk
        risk_scores["bleeding"] = await self._calculate_bleeding_risk(
            age, medications, conditions, lab_results
        )
        
        # Medication adherence risk
        risk_scores["medication_adherence"] = await self._calculate_adherence_risk(
            medications, conditions, patient_data
        )
        
        return risk_scores
    
    async def _check_medication_allergies(
        self,
        medications: List[Dict[str, Any]],
        allergies: List[Dict[str, Any]]
    ) -> List[ClinicalAlert]:
        """Check medications against known allergies"""
        
        alerts = []
        
        for medication in medications:
            med_name = self._extract_medication_name(medication)
            med_class = self._get_medication_class(med_name)
            
            for allergy in allergies:
                allergy_substance = self._extract_allergy_substance(allergy)
                
                # Direct match
                if med_name.lower() == allergy_substance.lower():
                    alert = ClinicalAlert(
                        alert_id=f"allergy_{len(alerts)}",
                        severity=AlertSeverity.CRITICAL,
                        message=f"Patient is allergic to {med_name}",
                        recommendation=f"Do not administer {med_name}. Consider alternative.",
                        evidence=f"Known allergy: {allergy_substance}",
                        source="allergy_database",
                        created_at=datetime.utcnow(),
                        medication_involved=med_name
                    )
                    alerts.append(alert)
                
                # Cross-reactivity check
                elif await self._check_cross_reactivity(med_class, allergy_substance):
                    alert = ClinicalAlert(
                        alert_id=f"cross_allergy_{len(alerts)}",
                        severity=AlertSeverity.HIGH,
                        message=f"Potential cross-reactivity between {med_name} and {allergy_substance}",
                        recommendation=f"Use caution with {med_name}. Monitor for allergic reactions.",
                        evidence=f"Cross-reactivity potential with {allergy_substance}",
                        source="allergy_database",
                        created_at=datetime.utcnow(),
                        medication_involved=med_name
                    )
                    alerts.append(alert)
        
        return alerts
    
    async def _check_contraindications(
        self,
        medications: List[Dict[str, Any]],
        conditions: List[Dict[str, Any]]
    ) -> List[ClinicalAlert]:
        """Check for medication contraindications based on conditions"""
        
        alerts = []
        
        for medication in medications:
            med_name = self._extract_medication_name(medication)
            
            for condition in conditions:
                condition_code = self._extract_condition_code(condition)
                
                if await self._is_contraindicated(med_name, condition_code):
                    alert = ClinicalAlert(
                        alert_id=f"contraindication_{len(alerts)}",
                        severity=AlertSeverity.HIGH,
                        message=f"{med_name} is contraindicated in {condition_code}",
                        recommendation=f"Consider alternative to {med_name}",
                        evidence=f"Contraindication: {condition_code}",
                        source="contraindication_database",
                        created_at=datetime.utcnow(),
                        medication_involved=med_name,
                        condition_involved=condition_code
                    )
                    alerts.append(alert)
        
        return alerts
    
    async def _check_lab_values(
        self,
        lab_results: List[Dict[str, Any]],
        medications: List[Dict[str, Any]]
    ) -> List[ClinicalAlert]:
        """Check lab values for medication-related concerns"""
        
        alerts = []
        
        for lab in lab_results:
            lab_code = self._extract_lab_code(lab)
            lab_value = self._extract_lab_value(lab)
            
            # Check for abnormal values requiring medication adjustment
            for medication in medications:
                med_name = self._extract_medication_name(medication)
                
                if await self._requires_lab_monitoring(med_name, lab_code):
                    if await self._is_lab_value_concerning(lab_code, lab_value, med_name):
                        severity = await self._get_lab_alert_severity(lab_code, lab_value)
                        
                        alert = ClinicalAlert(
                            alert_id=f"lab_alert_{len(alerts)}",
                            severity=severity,
                            message=f"Abnormal {lab_code}: {lab_value} while on {med_name}",
                            recommendation=await self._get_lab_recommendation(lab_code, lab_value, med_name),
                            evidence=f"Lab monitoring for {med_name}",
                            source="lab_monitoring_rules",
                            created_at=datetime.utcnow(),
                            medication_involved=med_name
                        )
                        alerts.append(alert)
        
        return alerts
    
    def _load_drug_interaction_database(self):
        """Load drug interaction database"""
        # In a real implementation, this would load from a comprehensive drug database
        self.drug_interactions = {
            # Example interactions
            ("warfarin", "aspirin"): DrugInteraction(
                interaction_id="warfarin_aspirin",
                drug1="warfarin",
                drug2="aspirin",
                interaction_type=InteractionType.DRUG_DRUG,
                severity=AlertSeverity.HIGH,
                description="Increased bleeding risk",
                mechanism="Additive anticoagulant effects",
                management="Monitor INR closely, consider dose reduction",
                evidence_level="high"
            ),
            ("metformin", "contrast"): DrugInteraction(
                interaction_id="metformin_contrast",
                drug1="metformin",
                drug2="iodinated contrast",
                interaction_type=InteractionType.DRUG_CONDITION,
                severity=AlertSeverity.MODERATE,
                description="Risk of lactic acidosis",
                mechanism="Impaired metformin clearance",
                management="Hold metformin 48 hours before and after contrast",
                evidence_level="moderate"
            )
        }
    
    def _load_clinical_guidelines(self):
        """Load clinical guidelines database"""
        # In a real implementation, this would load from clinical guideline databases
        self.clinical_guidelines = {
            "diabetes_type2": [
                ClinicalRecommendation(
                    recommendation_id="dm2_metformin",
                    category="pharmacotherapy",
                    title="First-line therapy with metformin",
                    description="Metformin is recommended as first-line therapy for type 2 diabetes",
                    rationale="Proven efficacy, cardiovascular benefits, low hypoglycemia risk",
                    strength="strong",
                    quality_of_evidence="high",
                    patient_population="Adults with type 2 diabetes without contraindications",
                    contraindications=["eGFR < 30", "severe heart failure", "lactic acidosis history"],
                    implementation_guidance="Start 500mg BID, titrate based on tolerance and glycemic control"
                )
            ],
            "hypertension": [
                ClinicalRecommendation(
                    recommendation_id="htn_ace_inhibitor",
                    category="pharmacotherapy",
                    title="ACE inhibitor or ARB for hypertension",
                    description="ACE inhibitors or ARBs recommended for initial therapy",
                    rationale="Cardiovascular outcome benefits, renal protection",
                    strength="strong",
                    quality_of_evidence="high",
                    patient_population="Adults with hypertension",
                    contraindications=["pregnancy", "bilateral renal artery stenosis", "hyperkalemia"],
                    implementation_guidance="Start low dose, titrate to target BP < 130/80"
                )
            ]
        }
    
    def _load_allergy_database(self):
        """Load allergy and cross-reactivity database"""
        self.allergy_cross_reactivity = {
            "penicillin": ["amoxicillin", "ampicillin", "piperacillin"],
            "sulfonamides": ["sulfamethoxazole", "sulfasalazine"],
            "cephalosporins": ["cephalexin", "ceftriaxone", "cefazolin"]
        }
    
    # Helper methods
    def _calculate_age(self, birth_date: str) -> int:
        """Calculate age from birth date"""
        if not birth_date:
            return 0
        
        try:
            birth = datetime.fromisoformat(birth_date.replace("Z", "+00:00"))
            today = datetime.utcnow().replace(tzinfo=birth.tzinfo)
            return (today - birth).days // 365
        except (ValueError, TypeError):
            return 0
    
    def _extract_medication_name(self, medication: Dict[str, Any]) -> str:
        """Extract medication name from FHIR resource"""
        if isinstance(medication, dict):
            # Try different possible structures
            if "code" in medication:
                coding = medication["code"].get("coding", [])
                if coding:
                    return coding[0].get("display", "Unknown")
            elif "medicationCodeableConcept" in medication:
                coding = medication["medicationCodeableConcept"].get("coding", [])
                if coding:
                    return coding[0].get("display", "Unknown")
            elif "name" in medication:
                return medication["name"]
        
        return "Unknown medication"
    
    def _extract_allergy_substance(self, allergy: Dict[str, Any]) -> str:
        """Extract allergen substance from allergy data"""
        if isinstance(allergy, dict):
            if "code" in allergy:
                coding = allergy["code"].get("coding", [])
                if coding:
                    return coding[0].get("display", "Unknown")
            elif "substance" in allergy:
                return allergy["substance"]
        
        return "Unknown allergen"
    
    def _extract_condition_code(self, condition: Dict[str, Any]) -> str:
        """Extract condition code from FHIR resource"""
        if isinstance(condition, dict):
            if "code" in condition:
                coding = condition["code"].get("coding", [])
                if coding:
                    return coding[0].get("code", "Unknown")
        
        return "Unknown condition"
    
    def _extract_lab_code(self, lab: Dict[str, Any]) -> str:
        """Extract lab test code"""
        if isinstance(lab, dict):
            if "code" in lab:
                coding = lab["code"].get("coding", [])
                if coding:
                    return coding[0].get("code", "Unknown")
        
        return "Unknown lab"
    
    def _extract_lab_value(self, lab: Dict[str, Any]) -> float:
        """Extract lab value"""
        if isinstance(lab, dict):
            if "valueQuantity" in lab:
                return lab["valueQuantity"].get("value", 0.0)
            elif "value" in lab:
                return float(lab["value"])
        
        return 0.0
    
    def _get_medication_class(self, medication_name: str) -> str:
        """Get medication class"""
        # Simplified medication class mapping
        medication_classes = {
            "warfarin": "anticoagulant",
            "aspirin": "antiplatelet",
            "metformin": "biguanide",
            "lisinopril": "ace_inhibitor",
            "amlodipine": "calcium_channel_blocker"
        }
        
        return medication_classes.get(medication_name.lower(), "unknown")
    
    async def _check_cross_reactivity(self, med_class: str, allergy_substance: str) -> bool:
        """Check for cross-reactivity"""
        cross_reactive = self.allergy_cross_reactivity.get(allergy_substance.lower(), [])
        return med_class.lower() in [item.lower() for item in cross_reactive]
    
    async def _is_contraindicated(self, medication: str, condition: str) -> bool:
        """Check if medication is contraindicated for condition"""
        # Simplified contraindication rules
        contraindications = {
            "metformin": ["renal_failure", "heart_failure"],
            "nsaids": ["renal_failure", "heart_failure", "peptic_ulcer"],
            "ace_inhibitor": ["pregnancy", "hyperkalemia"]
        }
        
        med_contraindications = contraindications.get(medication.lower(), [])
        return condition.lower() in med_contraindications
    
    async def _requires_lab_monitoring(self, medication: str, lab_code: str) -> bool:
        """Check if medication requires specific lab monitoring"""
        monitoring_rules = {
            "warfarin": ["inr", "pt"],
            "metformin": ["creatinine", "egfr"],
            "statin": ["alt", "ast", "ck"]
        }
        
        required_labs = monitoring_rules.get(medication.lower(), [])
        return lab_code.lower() in required_labs
    
    async def _is_lab_value_concerning(self, lab_code: str, lab_value: float, medication: str) -> bool:
        """Check if lab value is concerning for given medication"""
        # Simplified lab value rules
        if lab_code.lower() == "inr" and medication.lower() == "warfarin":
            return lab_value > 3.0 or lab_value < 1.5
        elif lab_code.lower() == "creatinine" and medication.lower() == "metformin":
            return lab_value > 1.5
        
        return False
    
    async def _get_lab_alert_severity(self, lab_code: str, lab_value: float) -> AlertSeverity:
        """Get alert severity based on lab value"""
        # Simplified severity rules
        if lab_code.lower() == "inr":
            if lab_value > 4.0:
                return AlertSeverity.CRITICAL
            elif lab_value > 3.5:
                return AlertSeverity.HIGH
            else:
                return AlertSeverity.MODERATE
        
        return AlertSeverity.MODERATE
    
    async def _get_lab_recommendation(self, lab_code: str, lab_value: float, medication: str) -> str:
        """Get recommendation based on lab value"""
        if lab_code.lower() == "inr" and medication.lower() == "warfarin":
            if lab_value > 3.0:
                return "Consider reducing warfarin dose. Recheck INR in 3-7 days."
            elif lab_value < 2.0:
                return "Consider increasing warfarin dose. Recheck INR in 3-7 days."
        
        return "Review medication dosing based on lab results."
    
    async def _check_drug_drug_interaction(self, med1: Dict[str, Any], med2: Dict[str, Any]) -> Optional[DrugInteraction]:
        """Check for drug-drug interaction"""
        med1_name = self._extract_medication_name(med1).lower()
        med2_name = self._extract_medication_name(med2).lower()
        
        # Check both directions
        interaction_key = (med1_name, med2_name)
        reverse_key = (med2_name, med1_name)
        
        return self.drug_interactions.get(interaction_key) or self.drug_interactions.get(reverse_key)
    
    async def _check_drug_allergy_interaction(self, medication: Dict[str, Any], allergy: Dict[str, Any]) -> Optional[DrugInteraction]:
        """Check for drug-allergy interaction"""
        med_name = self._extract_medication_name(medication)
        allergy_substance = self._extract_allergy_substance(allergy)
        
        if med_name.lower() == allergy_substance.lower():
            return DrugInteraction(
                interaction_id=f"allergy_{med_name}",
                drug1=med_name,
                drug2=allergy_substance,
                interaction_type=InteractionType.DRUG_ALLERGY,
                severity=AlertSeverity.CRITICAL,
                description=f"Known allergy to {allergy_substance}",
                mechanism="Allergic reaction",
                management="Avoid medication, use alternative",
                evidence_level="definitive"
            )
        
        return None
    
    async def _get_condition_recommendations(
        self, condition: Dict[str, Any], age: int, gender: str, 
        medications: List[Dict[str, Any]], lab_results: List[Dict[str, Any]]
    ) -> List[ClinicalRecommendation]:
        """Get recommendations for specific condition"""
        condition_code = self._extract_condition_code(condition)
        
        if condition_code.lower() in ["diabetes", "diabetes_type2"]:
            return self.clinical_guidelines.get("diabetes_type2", [])
        elif condition_code.lower() in ["hypertension", "high_blood_pressure"]:
            return self.clinical_guidelines.get("hypertension", [])
        
        return []
    
    async def _get_preventive_care_recommendations(
        self, patient_data: Dict[str, Any], conditions: List[Dict[str, Any]]
    ) -> List[ClinicalRecommendation]:
        """Get preventive care recommendations"""
        recommendations = []
        age = self._calculate_age(patient_data.get("birthDate"))
        gender = patient_data.get("gender", "").lower()
        
        # Age-based screening recommendations
        if age >= 50:
            recommendations.append(ClinicalRecommendation(
                recommendation_id="colonoscopy_screening",
                category="preventive_care",
                title="Colorectal cancer screening",
                description="Colonoscopy screening recommended every 10 years starting at age 50",
                rationale="Early detection of colorectal cancer",
                strength="strong",
                quality_of_evidence="high",
                patient_population="Adults 50-75 years",
                contraindications=["life expectancy < 10 years"],
                implementation_guidance="Discuss risks and benefits with patient"
            ))
        
        if gender == "female" and age >= 21:
            recommendations.append(ClinicalRecommendation(
                recommendation_id="cervical_cancer_screening",
                category="preventive_care",
                title="Cervical cancer screening",
                description="Pap smear every 3 years or HPV testing every 5 years",
                rationale="Early detection of cervical cancer",
                strength="strong",
                quality_of_evidence="high",
                patient_population="Women 21-65 years",
                contraindications=["hysterectomy with cervix removal"],
                implementation_guidance="Discuss screening options with patient"
            ))
        
        return recommendations
    
    async def _get_medication_recommendations(
        self, medications: List[Dict[str, Any]], conditions: List[Dict[str, Any]], 
        lab_results: List[Dict[str, Any]], age: int
    ) -> List[ClinicalRecommendation]:
        """Get medication optimization recommendations"""
        recommendations = []
        
        # Check for missing evidence-based medications
        condition_codes = [self._extract_condition_code(c).lower() for c in conditions]
        current_meds = [self._extract_medication_name(m).lower() for m in medications]
        
        # Diabetes without metformin
        if "diabetes" in str(condition_codes) and "metformin" not in str(current_meds):
            recommendations.append(ClinicalRecommendation(
                recommendation_id="add_metformin",
                category="medication_optimization",
                title="Consider adding metformin",
                description="Metformin is first-line therapy for type 2 diabetes",
                rationale="Proven cardiovascular benefits and glycemic efficacy",
                strength="strong",
                quality_of_evidence="high",
                patient_population="Adults with type 2 diabetes",
                contraindications=["eGFR < 30", "severe heart failure"],
                implementation_guidance="Start 500mg BID with meals"
            ))
        
        return recommendations
    
    async def _calculate_cardiovascular_risk(
        self, age: int, gender: str, conditions: List[Dict[str, Any]], 
        lab_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate cardiovascular risk score"""
        # Simplified ASCVD risk calculation
        risk_score = 0
        
        # Age factor
        if age > 65:
            risk_score += 2
        elif age > 55:
            risk_score += 1
        
        # Gender factor
        if gender.lower() == "male":
            risk_score += 1
        
        # Conditions
        condition_codes = [self._extract_condition_code(c).lower() for c in conditions]
        if "diabetes" in str(condition_codes):
            risk_score += 2
        if "hypertension" in str(condition_codes):
            risk_score += 1
        
        # Convert to percentage (simplified)
        risk_percentage = min(risk_score * 5, 50)
        
        return {
            "score": risk_score,
            "percentage": risk_percentage,
            "category": "high" if risk_percentage > 20 else "moderate" if risk_percentage > 7.5 else "low"
        }
    
    async def _calculate_fall_risk(
        self, age: int, medications: List[Dict[str, Any]], conditions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate fall risk score"""
        risk_score = 0
        
        # Age factor
        if age > 80:
            risk_score += 3
        elif age > 65:
            risk_score += 2
        
        # High-risk medications
        high_risk_meds = ["sedative", "antipsychotic", "antidepressant", "anticonvulsant"]
        current_meds = [self._get_medication_class(self._extract_medication_name(m)) for m in medications]
        
        for med_class in high_risk_meds:
            if med_class in current_meds:
                risk_score += 1
        
        return {
            "score": risk_score,
            "category": "high" if risk_score > 4 else "moderate" if risk_score > 2 else "low"
        }
    
    async def _calculate_bleeding_risk(
        self, age: int, medications: List[Dict[str, Any]], conditions: List[Dict[str, Any]], 
        lab_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate bleeding risk score"""
        risk_score = 0
        
        # Age factor
        if age > 75:
            risk_score += 2
        elif age > 65:
            risk_score += 1
        
        # Anticoagulant medications
        current_meds = [self._extract_medication_name(m).lower() for m in medications]
        anticoagulants = ["warfarin", "rivaroxaban", "apixaban", "dabigatran"]
        
        for anticoag in anticoagulants:
            if anticoag in str(current_meds):
                risk_score += 2
                break
        
        return {
            "score": risk_score,
            "category": "high" if risk_score > 3 else "moderate" if risk_score > 1 else "low"
        }
    
    async def _calculate_adherence_risk(
        self, medications: List[Dict[str, Any]], conditions: List[Dict[str, Any]], 
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate medication adherence risk"""
        risk_score = 0
        
        # Number of medications
        med_count = len(medications)
        if med_count > 10:
            risk_score += 2
        elif med_count > 5:
            risk_score += 1
        
        # Complex conditions
        condition_count = len(conditions)
        if condition_count > 5:
            risk_score += 1
        
        return {
            "score": risk_score,
            "category": "high" if risk_score > 2 else "moderate" if risk_score > 0 else "low"
        }


# Export the clinical decision support agent
__all__ = ["ClinicalDecisionSupportAgent", "ClinicalAlert", "DrugInteraction", "ClinicalRecommendation"]