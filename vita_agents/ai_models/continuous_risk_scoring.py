"""
Continuous Risk Scoring System for Real-time Clinical Monitoring.

This module provides advanced risk scoring algorithms for continuous patient monitoring
including sepsis prediction, cardiac event prediction, fall risk assessment, and
medication adherence prediction.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass
import structlog
from pydantic import BaseModel, Field

try:
    import scikit_learn as sklearn
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

logger = structlog.get_logger(__name__)


class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RiskType(Enum):
    """Types of clinical risk assessments."""
    SEPSIS = "sepsis"
    CARDIAC_EVENT = "cardiac_event"
    FALL_RISK = "fall_risk"
    MEDICATION_ADHERENCE = "medication_adherence"
    BLEEDING = "bleeding"
    VENOUS_THROMBOEMBOLISM = "vte"
    DELIRIUM = "delirium"
    PRESSURE_ULCER = "pressure_ulcer"


@dataclass
class VitalSigns:
    """Patient vital signs data."""
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    heart_rate: Optional[float] = None
    respiratory_rate: Optional[float] = None
    temperature: Optional[float] = None  # Celsius
    oxygen_saturation: Optional[float] = None
    glasgow_coma_scale: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


@dataclass
class LabResults:
    """Laboratory results data."""
    white_blood_cell_count: Optional[float] = None  # cells/µL
    lactate: Optional[float] = None  # mmol/L
    creatinine: Optional[float] = None  # mg/dL
    bilirubin_total: Optional[float] = None  # mg/dL
    platelet_count: Optional[float] = None  # cells/µL
    inr: Optional[float] = None
    glucose: Optional[float] = None  # mg/dL
    hemoglobin: Optional[float] = None  # g/dL
    sodium: Optional[float] = None  # mEq/L
    potassium: Optional[float] = None  # mEq/L
    troponin: Optional[float] = None  # ng/mL
    bnp: Optional[float] = None  # pg/mL
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RiskAssessmentRequest(BaseModel):
    """Request for risk assessment."""
    
    patient_id: str
    risk_type: RiskType
    vital_signs: Optional[VitalSigns] = None
    lab_results: Optional[LabResults] = None
    patient_demographics: Dict[str, Any] = Field(default_factory=dict)
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    clinical_notes: Optional[str] = None
    wearable_data: Optional[Dict[str, Any]] = None
    assessment_timestamp: datetime = Field(default_factory=datetime.utcnow)


class RiskAssessmentResponse(BaseModel):
    """Response from risk assessment."""
    
    patient_id: str
    risk_type: RiskType
    risk_score: float  # 0.0 to 1.0
    risk_level: RiskLevel
    confidence: float  # 0.0 to 1.0
    contributing_factors: List[str]
    recommendations: List[str]
    alert_threshold_crossed: bool
    next_assessment_recommended: Optional[datetime] = None
    model_version: str
    assessment_timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseRiskScorer(ABC):
    """Base class for risk scoring algorithms."""
    
    def __init__(self, model_version: str = "1.0.0"):
        self.model_version = model_version
        self.logger = structlog.get_logger(__name__)
        self.alert_thresholds = {
            RiskLevel.LOW: 0.3,
            RiskLevel.MODERATE: 0.6,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 0.9
        }
    
    @abstractmethod
    async def calculate_risk_score(
        self, 
        request: RiskAssessmentRequest
    ) -> RiskAssessmentResponse:
        """Calculate risk score for patient."""
        pass
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on score."""
        if risk_score >= self.alert_thresholds[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif risk_score >= self.alert_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif risk_score >= self.alert_thresholds[RiskLevel.MODERATE]:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _check_alert_threshold(self, risk_score: float, current_level: RiskLevel) -> bool:
        """Check if alert threshold is crossed."""
        return risk_score >= self.alert_thresholds.get(current_level, 0.5)


class SepsisRiskScorer(BaseRiskScorer):
    """Sepsis prediction using qSOFA and enhanced clinical indicators."""
    
    def __init__(self):
        super().__init__("sepsis_v1.2.0")
        
        # qSOFA criteria weights
        self.qsofa_weights = {
            'systolic_bp_low': 1.0,  # SBP ≤ 100 mmHg
            'altered_mental_status': 1.0,  # GCS < 15
            'respiratory_rate_high': 1.0  # RR ≥ 22/min
        }
        
        # Enhanced sepsis indicators
        self.enhanced_weights = {
            'lactate_elevated': 2.0,  # > 2 mmol/L
            'white_blood_cell_abnormal': 1.5,  # < 4000 or > 12000
            'temperature_abnormal': 1.0,  # < 36°C or > 38.3°C
            'heart_rate_high': 0.5,  # > 90 bpm
            'infection_suspected': 2.0  # Clinical notes mention infection
        }
    
    async def calculate_risk_score(
        self, 
        request: RiskAssessmentRequest
    ) -> RiskAssessmentResponse:
        """Calculate sepsis risk score."""
        
        try:
            risk_factors = []
            qsofa_score = 0
            enhanced_score = 0
            
            # Calculate qSOFA score
            if request.vital_signs:
                vitals = request.vital_signs
                
                # Systolic BP ≤ 100 mmHg
                if vitals.systolic_bp and vitals.systolic_bp <= 100:
                    qsofa_score += self.qsofa_weights['systolic_bp_low']
                    risk_factors.append("Hypotension (SBP ≤ 100 mmHg)")
                
                # Altered mental status (GCS < 15)
                if vitals.glasgow_coma_scale and vitals.glasgow_coma_scale < 15:
                    qsofa_score += self.qsofa_weights['altered_mental_status']
                    risk_factors.append(f"Altered mental status (GCS {vitals.glasgow_coma_scale})")
                
                # Respiratory rate ≥ 22/min
                if vitals.respiratory_rate and vitals.respiratory_rate >= 22:
                    qsofa_score += self.qsofa_weights['respiratory_rate_high']
                    risk_factors.append(f"Tachypnea (RR {vitals.respiratory_rate}/min)")
                
                # Enhanced indicators - Temperature
                if vitals.temperature:
                    if vitals.temperature < 36.0 or vitals.temperature > 38.3:
                        enhanced_score += self.enhanced_weights['temperature_abnormal']
                        risk_factors.append(f"Abnormal temperature ({vitals.temperature}°C)")
                
                # Enhanced indicators - Heart rate
                if vitals.heart_rate and vitals.heart_rate > 90:
                    enhanced_score += self.enhanced_weights['heart_rate_high']
                    risk_factors.append(f"Tachycardia (HR {vitals.heart_rate} bpm)")
            
            # Enhanced laboratory indicators
            if request.lab_results:
                labs = request.lab_results
                
                # Elevated lactate
                if labs.lactate and labs.lactate > 2.0:
                    enhanced_score += self.enhanced_weights['lactate_elevated']
                    risk_factors.append(f"Elevated lactate ({labs.lactate} mmol/L)")
                
                # Abnormal WBC count
                if labs.white_blood_cell_count:
                    wbc = labs.white_blood_cell_count
                    if wbc < 4000 or wbc > 12000:
                        enhanced_score += self.enhanced_weights['white_blood_cell_abnormal']
                        risk_factors.append(f"Abnormal WBC count ({wbc} cells/µL)")
            
            # Check for infection suspicion in clinical notes
            if request.clinical_notes:
                infection_keywords = ['infection', 'sepsis', 'bacteremia', 'pneumonia', 'uti', 'cellulitis']
                if any(keyword in request.clinical_notes.lower() for keyword in infection_keywords):
                    enhanced_score += self.enhanced_weights['infection_suspected']
                    risk_factors.append("Clinical suspicion of infection")
            
            # Calculate combined risk score (normalized 0-1)
            max_possible_score = sum(self.qsofa_weights.values()) + sum(self.enhanced_weights.values())
            raw_score = qsofa_score + enhanced_score
            risk_score = min(raw_score / max_possible_score, 1.0)
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_score)
            
            # Generate recommendations
            recommendations = self._generate_sepsis_recommendations(
                qsofa_score, enhanced_score, risk_factors
            )
            
            # Calculate confidence based on data completeness
            confidence = self._calculate_confidence(request)
            
            return RiskAssessmentResponse(
                patient_id=request.patient_id,
                risk_type=RiskType.SEPSIS,
                risk_score=risk_score,
                risk_level=risk_level,
                confidence=confidence,
                contributing_factors=risk_factors,
                recommendations=recommendations,
                alert_threshold_crossed=self._check_alert_threshold(risk_score, risk_level),
                next_assessment_recommended=datetime.utcnow() + timedelta(hours=2),
                model_version=self.model_version,
                assessment_timestamp=request.assessment_timestamp,
                metadata={
                    "qsofa_score": qsofa_score,
                    "enhanced_score": enhanced_score,
                    "max_possible_score": max_possible_score,
                    "data_sources_used": self._get_data_sources_used(request)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Sepsis risk calculation failed: {e}")
            raise
    
    def _generate_sepsis_recommendations(
        self, 
        qsofa_score: float, 
        enhanced_score: float, 
        risk_factors: List[str]
    ) -> List[str]:
        """Generate sepsis-specific recommendations."""
        
        recommendations = []
        
        if qsofa_score >= 2:
            recommendations.extend([
                "URGENT: qSOFA ≥ 2 - Consider sepsis protocol activation",
                "Obtain blood cultures before antibiotics",
                "Consider broad-spectrum antibiotics",
                "Fluid resuscitation if hypotensive",
                "Continuous monitoring required"
            ])
        elif qsofa_score >= 1:
            recommendations.extend([
                "Monitor closely for sepsis development",
                "Consider infection workup",
                "Frequent vital signs monitoring"
            ])
        
        if enhanced_score > 3:
            recommendations.extend([
                "Enhanced sepsis monitoring indicated",
                "Consider lactate monitoring",
                "Evaluate for source control"
            ])
        
        if not risk_factors:
            recommendations.append("Continue routine monitoring")
        
        return recommendations
    
    def _calculate_confidence(self, request: RiskAssessmentRequest) -> float:
        """Calculate confidence based on data completeness."""
        
        data_points = 0
        available_points = 0
        
        # Vital signs
        if request.vital_signs:
            vitals = request.vital_signs
            if vitals.systolic_bp is not None:
                available_points += 1
            if vitals.respiratory_rate is not None:
                available_points += 1
            if vitals.glasgow_coma_scale is not None:
                available_points += 1
            if vitals.temperature is not None:
                available_points += 1
            if vitals.heart_rate is not None:
                available_points += 1
            data_points += 5
        
        # Lab results
        if request.lab_results:
            labs = request.lab_results
            if labs.lactate is not None:
                available_points += 1
            if labs.white_blood_cell_count is not None:
                available_points += 1
            data_points += 2
        
        # Clinical notes
        if request.clinical_notes:
            available_points += 1
        data_points += 1
        
        return available_points / data_points if data_points > 0 else 0.5
    
    def _get_data_sources_used(self, request: RiskAssessmentRequest) -> List[str]:
        """Get list of data sources used in assessment."""
        sources = []
        if request.vital_signs:
            sources.append("vital_signs")
        if request.lab_results:
            sources.append("laboratory_results")
        if request.clinical_notes:
            sources.append("clinical_notes")
        return sources


class CardiacEventRiskScorer(BaseRiskScorer):
    """Cardiac event prediction using clinical indicators and biomarkers."""
    
    def __init__(self):
        super().__init__("cardiac_v1.1.0")
        
        # Risk factor weights
        self.risk_weights = {
            'chest_pain': 3.0,
            'elevated_troponin': 4.0,
            'elevated_bnp': 2.5,
            'st_changes': 4.5,
            'previous_mi': 2.0,
            'diabetes': 1.5,
            'hypertension': 1.0,
            'age_risk': 2.0,  # Age > 65
            'tachycardia': 1.5,
            'dyspnea': 2.0
        }
    
    async def calculate_risk_score(
        self, 
        request: RiskAssessmentRequest
    ) -> RiskAssessmentResponse:
        """Calculate cardiac event risk score."""
        
        try:
            risk_factors = []
            risk_score = 0
            
            # Check age risk
            age = request.patient_demographics.get('age', 0)
            if age > 65:
                risk_score += self.risk_weights['age_risk']
                risk_factors.append(f"Advanced age ({age} years)")
            
            # Check medical history
            history = [h.lower() for h in request.medical_history]
            if any('myocardial infarction' in h or 'mi' in h or 'heart attack' in h for h in history):
                risk_score += self.risk_weights['previous_mi']
                risk_factors.append("Previous myocardial infarction")
            
            if any('diabetes' in h for h in history):
                risk_score += self.risk_weights['diabetes']
                risk_factors.append("Diabetes mellitus")
            
            if any('hypertension' in h or 'high blood pressure' in h for h in history):
                risk_score += self.risk_weights['hypertension']
                risk_factors.append("Hypertension")
            
            # Check vital signs
            if request.vital_signs:
                vitals = request.vital_signs
                
                # Tachycardia
                if vitals.heart_rate and vitals.heart_rate > 100:
                    risk_score += self.risk_weights['tachycardia']
                    risk_factors.append(f"Tachycardia (HR {vitals.heart_rate} bpm)")
            
            # Check laboratory biomarkers
            if request.lab_results:
                labs = request.lab_results
                
                # Elevated troponin
                if labs.troponin and labs.troponin > 0.04:  # ng/mL
                    risk_score += self.risk_weights['elevated_troponin']
                    risk_factors.append(f"Elevated troponin ({labs.troponin} ng/mL)")
                
                # Elevated BNP
                if labs.bnp and labs.bnp > 100:  # pg/mL
                    risk_score += self.risk_weights['elevated_bnp']
                    risk_factors.append(f"Elevated BNP ({labs.bnp} pg/mL)")
            
            # Check clinical notes for symptoms
            if request.clinical_notes:
                notes = request.clinical_notes.lower()
                
                if 'chest pain' in notes or 'chest discomfort' in notes:
                    risk_score += self.risk_weights['chest_pain']
                    risk_factors.append("Chest pain reported")
                
                if 'dyspnea' in notes or 'shortness of breath' in notes or 'sob' in notes:
                    risk_score += self.risk_weights['dyspnea']
                    risk_factors.append("Dyspnea reported")
                
                if 'st elevation' in notes or 'st depression' in notes or 'st changes' in notes:
                    risk_score += self.risk_weights['st_changes']
                    risk_factors.append("ST changes on ECG")
            
            # Normalize risk score (0-1)
            max_possible_score = sum(self.risk_weights.values())
            normalized_score = min(risk_score / max_possible_score, 1.0)
            
            # Determine risk level
            risk_level = self._determine_risk_level(normalized_score)
            
            # Generate recommendations
            recommendations = self._generate_cardiac_recommendations(
                normalized_score, risk_factors
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(request)
            
            return RiskAssessmentResponse(
                patient_id=request.patient_id,
                risk_type=RiskType.CARDIAC_EVENT,
                risk_score=normalized_score,
                risk_level=risk_level,
                confidence=confidence,
                contributing_factors=risk_factors,
                recommendations=recommendations,
                alert_threshold_crossed=self._check_alert_threshold(normalized_score, risk_level),
                next_assessment_recommended=datetime.utcnow() + timedelta(hours=4),
                model_version=self.model_version,
                assessment_timestamp=request.assessment_timestamp,
                metadata={
                    "raw_risk_score": risk_score,
                    "max_possible_score": max_possible_score,
                    "risk_factors_count": len(risk_factors)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Cardiac risk calculation failed: {e}")
            raise
    
    def _generate_cardiac_recommendations(
        self, 
        risk_score: float, 
        risk_factors: List[str]
    ) -> List[str]:
        """Generate cardiac-specific recommendations."""
        
        recommendations = []
        
        if risk_score >= 0.8:
            recommendations.extend([
                "URGENT: High cardiac event risk - Consider immediate cardiology consultation",
                "Obtain 12-lead ECG",
                "Serial cardiac biomarkers",
                "Continuous cardiac monitoring",
                "Consider aspirin and anticoagulation if not contraindicated"
            ])
        elif risk_score >= 0.6:
            recommendations.extend([
                "Moderate cardiac risk - Enhanced monitoring recommended",
                "Cardiac enzyme monitoring",
                "ECG monitoring",
                "Consider cardiology consultation"
            ])
        elif risk_score >= 0.3:
            recommendations.extend([
                "Low-moderate cardiac risk",
                "Routine cardiac monitoring",
                "Symptom monitoring"
            ])
        else:
            recommendations.append("Low cardiac risk - Continue routine care")
        
        return recommendations
    
    def _calculate_confidence(self, request: RiskAssessmentRequest) -> float:
        """Calculate confidence for cardiac assessment."""
        
        # Similar to sepsis confidence calculation but cardiac-specific
        data_points = 0
        available_points = 0
        
        # Demographics
        if 'age' in request.patient_demographics:
            available_points += 1
        data_points += 1
        
        # Medical history
        if request.medical_history:
            available_points += 1
        data_points += 1
        
        # Vital signs
        if request.vital_signs and request.vital_signs.heart_rate:
            available_points += 1
        data_points += 1
        
        # Lab results (troponin, BNP)
        if request.lab_results:
            if request.lab_results.troponin is not None:
                available_points += 1
            if request.lab_results.bnp is not None:
                available_points += 1
            data_points += 2
        else:
            data_points += 2
        
        # Clinical notes
        if request.clinical_notes:
            available_points += 1
        data_points += 1
        
        return available_points / data_points if data_points > 0 else 0.5


class FallRiskScorer(BaseRiskScorer):
    """Fall risk assessment using Morse Fall Scale and mobility data."""
    
    def __init__(self):
        super().__init__("fall_risk_v1.0.0")
        
        # Morse Fall Scale weights
        self.morse_weights = {
            'history_of_falling': 25,
            'secondary_diagnosis': 15,
            'ambulatory_aid': {'none': 0, 'bed_rest': 0, 'nurse_assist': 15, 'crutches': 15, 'furniture': 30},
            'iv_therapy': 20,
            'gait': {'normal': 0, 'weak': 10, 'impaired': 20},
            'mental_status': {'oriented': 0, 'forgets_limitations': 15}
        }
    
    async def calculate_risk_score(
        self, 
        request: RiskAssessmentRequest
    ) -> RiskAssessmentResponse:
        """Calculate fall risk score using Morse Fall Scale."""
        
        try:
            morse_score = 0
            risk_factors = []
            
            # History of falling
            if any('fall' in h.lower() for h in request.medical_history):
                morse_score += self.morse_weights['history_of_falling']
                risk_factors.append("History of falling")
            
            # Secondary diagnosis (more than one medical diagnosis)
            if len(request.medical_history) > 1:
                morse_score += self.morse_weights['secondary_diagnosis']
                risk_factors.append("Multiple medical diagnoses")
            
            # Check for IV therapy in medications/treatments
            if any('iv' in med.lower() or 'intravenous' in med.lower() for med in request.current_medications):
                morse_score += self.morse_weights['iv_therapy']
                risk_factors.append("IV therapy")
            
            # Check wearable data for mobility indicators
            if request.wearable_data:
                mobility_data = request.wearable_data
                
                # Gait analysis from wearable data
                gait_score = mobility_data.get('gait_stability_score', 1.0)  # 0-1 scale
                if gait_score < 0.5:
                    morse_score += self.morse_weights['gait']['impaired']
                    risk_factors.append("Impaired gait stability")
                elif gait_score < 0.8:
                    morse_score += self.morse_weights['gait']['weak']
                    risk_factors.append("Weak gait")
                
                # Activity level
                daily_steps = mobility_data.get('daily_steps', 0)
                if daily_steps < 1000:
                    risk_factors.append("Low mobility (sedentary)")
                
                # Balance metrics
                balance_score = mobility_data.get('balance_score', 1.0)
                if balance_score < 0.6:
                    risk_factors.append("Poor balance metrics")
            
            # Check clinical notes for mental status
            if request.clinical_notes:
                notes = request.clinical_notes.lower()
                if any(term in notes for term in ['confused', 'disoriented', 'forgetful', 'dementia']):
                    morse_score += self.morse_weights['mental_status']['forgets_limitations']
                    risk_factors.append("Cognitive impairment noted")
            
            # Age-related risk
            age = request.patient_demographics.get('age', 0)
            if age >= 65:
                morse_score += 10  # Additional age-related risk
                risk_factors.append(f"Advanced age ({age} years)")
            
            # Medication-related fall risk
            high_risk_meds = ['sedative', 'hypnotic', 'antipsychotic', 'diuretic', 'antihypertensive']
            for med in request.current_medications:
                if any(risk_med in med.lower() for risk_med in high_risk_meds):
                    morse_score += 5
                    risk_factors.append(f"High-risk medication: {med}")
            
            # Normalize score (Morse scale typically 0-125, normalize to 0-1)
            max_morse_score = 125
            normalized_score = min(morse_score / max_morse_score, 1.0)
            
            # Determine risk level (adjusted for fall risk thresholds)
            if morse_score >= 45:
                risk_level = RiskLevel.HIGH
            elif morse_score >= 25:
                risk_level = RiskLevel.MODERATE
            else:
                risk_level = RiskLevel.LOW
            
            # Generate recommendations
            recommendations = self._generate_fall_recommendations(morse_score, risk_factors)
            
            # Calculate confidence
            confidence = self._calculate_fall_confidence(request)
            
            return RiskAssessmentResponse(
                patient_id=request.patient_id,
                risk_type=RiskType.FALL_RISK,
                risk_score=normalized_score,
                risk_level=risk_level,
                confidence=confidence,
                contributing_factors=risk_factors,
                recommendations=recommendations,
                alert_threshold_crossed=morse_score >= 25,
                next_assessment_recommended=datetime.utcnow() + timedelta(hours=12),
                model_version=self.model_version,
                assessment_timestamp=request.assessment_timestamp,
                metadata={
                    "morse_fall_score": morse_score,
                    "risk_category": "high" if morse_score >= 45 else "moderate" if morse_score >= 25 else "low",
                    "wearable_data_available": bool(request.wearable_data)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Fall risk calculation failed: {e}")
            raise
    
    def _generate_fall_recommendations(
        self, 
        morse_score: int, 
        risk_factors: List[str]
    ) -> List[str]:
        """Generate fall prevention recommendations."""
        
        recommendations = []
        
        if morse_score >= 45:
            recommendations.extend([
                "HIGH FALL RISK: Implement comprehensive fall prevention protocol",
                "Bed/chair alarm system",
                "Frequent rounding (hourly)",
                "Physical therapy consultation",
                "Remove environmental hazards",
                "Consider bed rails if appropriate",
                "Family education on fall prevention"
            ])
        elif morse_score >= 25:
            recommendations.extend([
                "MODERATE FALL RISK: Standard fall prevention measures",
                "Regular assistance with ambulation",
                "Clear pathways",
                "Adequate lighting",
                "Non-slip footwear",
                "Call bell within reach"
            ])
        else:
            recommendations.extend([
                "LOW FALL RISK: Basic safety measures",
                "Patient education on fall prevention",
                "Regular assessment of mobility"
            ])
        
        # Add specific recommendations based on risk factors
        if any('medication' in factor for factor in risk_factors):
            recommendations.append("Medication review for fall risk reduction")
        
        if any('gait' in factor or 'balance' in factor for factor in risk_factors):
            recommendations.append("Physical therapy evaluation for gait and balance training")
        
        return recommendations
    
    def _calculate_fall_confidence(self, request: RiskAssessmentRequest) -> float:
        """Calculate confidence for fall risk assessment."""
        
        data_points = 0
        available_points = 0
        
        # Medical history
        if request.medical_history:
            available_points += 1
        data_points += 1
        
        # Current medications
        if request.current_medications:
            available_points += 1
        data_points += 1
        
        # Demographics (age)
        if 'age' in request.patient_demographics:
            available_points += 1
        data_points += 1
        
        # Wearable data (key for fall risk)
        if request.wearable_data:
            available_points += 2  # High weight for mobility data
        data_points += 2
        
        # Clinical notes
        if request.clinical_notes:
            available_points += 1
        data_points += 1
        
        return available_points / data_points if data_points > 0 else 0.5


class MedicationAdherenceRiskScorer(BaseRiskScorer):
    """Medication adherence prediction using patient factors and behavior."""
    
    def __init__(self):
        super().__init__("med_adherence_v1.0.0")
        
        # Risk factor weights for non-adherence
        self.risk_weights = {
            'age_young': 2.0,  # < 30 years
            'age_elderly': 1.5,  # > 75 years
            'complex_regimen': 3.0,  # > 4 medications
            'frequent_dosing': 2.0,  # > 2x daily
            'side_effects': 2.5,
            'cognitive_impairment': 3.0,
            'depression': 2.0,
            'financial_barriers': 2.5,
            'poor_health_literacy': 2.0,
            'missed_appointments': 2.0
        }
    
    async def calculate_risk_score(
        self, 
        request: RiskAssessmentRequest
    ) -> RiskAssessmentResponse:
        """Calculate medication adherence risk score."""
        
        try:
            risk_score = 0
            risk_factors = []
            
            # Age-related risk
            age = request.patient_demographics.get('age', 0)
            if age < 30:
                risk_score += self.risk_weights['age_young']
                risk_factors.append("Young age (< 30 years)")
            elif age > 75:
                risk_score += self.risk_weights['age_elderly']
                risk_factors.append("Advanced age (> 75 years)")
            
            # Medication complexity
            med_count = len(request.current_medications)
            if med_count > 4:
                risk_score += self.risk_weights['complex_regimen']
                risk_factors.append(f"Complex medication regimen ({med_count} medications)")
            
            # Check for frequent dosing requirements
            frequent_dosing_meds = [
                med for med in request.current_medications 
                if any(term in med.lower() for term in ['tid', 'qid', '3 times', '4 times', 'every 6', 'every 4'])
            ]
            if frequent_dosing_meds:
                risk_score += self.risk_weights['frequent_dosing']
                risk_factors.append("Frequent dosing requirements")
            
            # Medical history factors
            history = [h.lower() for h in request.medical_history]
            
            if any('depression' in h or 'anxiety' in h for h in history):
                risk_score += self.risk_weights['depression']
                risk_factors.append("History of depression/anxiety")
            
            if any('dementia' in h or 'cognitive' in h or 'alzheimer' in h for h in history):
                risk_score += self.risk_weights['cognitive_impairment']
                risk_factors.append("Cognitive impairment")
            
            # Check clinical notes for adherence indicators
            if request.clinical_notes:
                notes = request.clinical_notes.lower()
                
                if any(term in notes for term in ['side effect', 'adverse effect', 'intolerant']):
                    risk_score += self.risk_weights['side_effects']
                    risk_factors.append("Reported side effects")
                
                if any(term in notes for term in ['missed appointment', 'no show', 'cancelled']):
                    risk_score += self.risk_weights['missed_appointments']
                    risk_factors.append("History of missed appointments")
                
                if any(term in notes for term in ['financial', 'insurance', 'cost', 'afford']):
                    risk_score += self.risk_weights['financial_barriers']
                    risk_factors.append("Financial barriers noted")
                
                if any(term in notes for term in ['education', 'understanding', 'literacy', 'confused about']):
                    risk_score += self.risk_weights['poor_health_literacy']
                    risk_factors.append("Health literacy concerns")
            
            # Social determinants
            if 'education_level' in request.patient_demographics:
                education = request.patient_demographics['education_level'].lower()
                if 'elementary' in education or 'primary' in education:
                    risk_score += self.risk_weights['poor_health_literacy']
                    risk_factors.append("Limited education")
            
            if 'insurance_status' in request.patient_demographics:
                insurance = request.patient_demographics['insurance_status'].lower()
                if 'uninsured' in insurance or 'medicaid' in insurance:
                    risk_score += self.risk_weights['financial_barriers']
                    risk_factors.append("Insurance/financial concerns")
            
            # Normalize score
            max_possible_score = sum(self.risk_weights.values())
            normalized_score = min(risk_score / max_possible_score, 1.0)
            
            # Determine risk level
            risk_level = self._determine_risk_level(normalized_score)
            
            # Generate recommendations
            recommendations = self._generate_adherence_recommendations(
                normalized_score, risk_factors
            )
            
            # Calculate confidence
            confidence = self._calculate_adherence_confidence(request)
            
            return RiskAssessmentResponse(
                patient_id=request.patient_id,
                risk_type=RiskType.MEDICATION_ADHERENCE,
                risk_score=normalized_score,
                risk_level=risk_level,
                confidence=confidence,
                contributing_factors=risk_factors,
                recommendations=recommendations,
                alert_threshold_crossed=self._check_alert_threshold(normalized_score, risk_level),
                next_assessment_recommended=datetime.utcnow() + timedelta(days=30),
                model_version=self.model_version,
                assessment_timestamp=request.assessment_timestamp,
                metadata={
                    "medication_count": len(request.current_medications),
                    "frequent_dosing_meds": len(frequent_dosing_meds),
                    "risk_factors_count": len(risk_factors)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Medication adherence risk calculation failed: {e}")
            raise
    
    def _generate_adherence_recommendations(
        self, 
        risk_score: float, 
        risk_factors: List[str]
    ) -> List[str]:
        """Generate medication adherence recommendations."""
        
        recommendations = []
        
        if risk_score >= 0.7:
            recommendations.extend([
                "HIGH NON-ADHERENCE RISK: Implement comprehensive adherence support",
                "Pharmacist consultation for medication management",
                "Pill organizers or adherence packaging",
                "Medication therapy management (MTM)",
                "Frequent follow-up appointments",
                "Consider simplified regimen if possible"
            ])
        elif risk_score >= 0.4:
            recommendations.extend([
                "MODERATE NON-ADHERENCE RISK: Enhanced adherence support",
                "Patient education on medication importance",
                "Adherence monitoring",
                "Address identified barriers"
            ])
        else:
            recommendations.extend([
                "LOW NON-ADHERENCE RISK: Standard adherence support",
                "Regular medication review",
                "Patient education materials"
            ])
        
        # Specific recommendations based on risk factors
        if any('complex' in factor or 'frequent' in factor for factor in risk_factors):
            recommendations.append("Consider medication regimen simplification")
        
        if any('side effect' in factor for factor in risk_factors):
            recommendations.append("Address side effects with prescriber")
        
        if any('financial' in factor or 'insurance' in factor for factor in risk_factors):
            recommendations.append("Social work consultation for financial assistance")
        
        if any('cognitive' in factor or 'literacy' in factor for factor in risk_factors):
            recommendations.append("Caregiver involvement in medication management")
        
        return recommendations
    
    def _calculate_adherence_confidence(self, request: RiskAssessmentRequest) -> float:
        """Calculate confidence for adherence assessment."""
        
        data_points = 0
        available_points = 0
        
        # Demographics
        demo_fields = ['age', 'education_level', 'insurance_status']
        for field in demo_fields:
            if field in request.patient_demographics:
                available_points += 1
            data_points += 1
        
        # Medical history
        if request.medical_history:
            available_points += 1
        data_points += 1
        
        # Current medications
        if request.current_medications:
            available_points += 1
        data_points += 1
        
        # Clinical notes
        if request.clinical_notes:
            available_points += 1
        data_points += 1
        
        return available_points / data_points if data_points > 0 else 0.5


class ContinuousRiskScoringManager:
    """Manager for continuous risk scoring across multiple risk types."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Initialize risk scorers
        self.scorers = {
            RiskType.SEPSIS: SepsisRiskScorer(),
            RiskType.CARDIAC_EVENT: CardiacEventRiskScorer(),
            RiskType.FALL_RISK: FallRiskScorer(),
            RiskType.MEDICATION_ADHERENCE: MedicationAdherenceRiskScorer()
        }
        
        # Risk monitoring intervals
        self.monitoring_intervals = {
            RiskType.SEPSIS: timedelta(hours=2),
            RiskType.CARDIAC_EVENT: timedelta(hours=4),
            RiskType.FALL_RISK: timedelta(hours=12),
            RiskType.MEDICATION_ADHERENCE: timedelta(days=30)
        }
        
        # Alert thresholds
        self.alert_thresholds = config.get('alert_thresholds', {
            RiskType.SEPSIS: 0.6,
            RiskType.CARDIAC_EVENT: 0.7,
            RiskType.FALL_RISK: 0.5,
            RiskType.MEDICATION_ADHERENCE: 0.6
        })
    
    async def assess_risk(
        self, 
        request: RiskAssessmentRequest
    ) -> RiskAssessmentResponse:
        """Assess risk for a specific risk type."""
        
        scorer = self.scorers.get(request.risk_type)
        if not scorer:
            raise ValueError(f"No scorer available for risk type: {request.risk_type}")
        
        return await scorer.calculate_risk_score(request)
    
    async def assess_multiple_risks(
        self, 
        patient_id: str,
        risk_types: List[RiskType],
        **data_inputs
    ) -> Dict[RiskType, RiskAssessmentResponse]:
        """Assess multiple risk types for a patient."""
        
        results = {}
        
        for risk_type in risk_types:
            request = RiskAssessmentRequest(
                patient_id=patient_id,
                risk_type=risk_type,
                **data_inputs
            )
            
            try:
                result = await self.assess_risk(request)
                results[risk_type] = result
            except Exception as e:
                self.logger.error(f"Risk assessment failed for {risk_type}: {e}")
                # Continue with other assessments
        
        return results
    
    async def continuous_monitoring(
        self, 
        patient_id: str,
        risk_types: List[RiskType],
        monitoring_duration: timedelta = timedelta(hours=24)
    ) -> List[RiskAssessmentResponse]:
        """Continuously monitor patient risks."""
        
        monitoring_results = []
        start_time = datetime.utcnow()
        
        while datetime.utcnow() - start_time < monitoring_duration:
            # Simulate getting fresh patient data
            # In production, this would pull from real-time data sources
            
            for risk_type in risk_types:
                interval = self.monitoring_intervals.get(risk_type, timedelta(hours=6))
                
                # Check if it's time for this risk assessment
                if len(monitoring_results) == 0 or \
                   datetime.utcnow() - monitoring_results[-1].assessment_timestamp >= interval:
                    
                    # Create assessment request with simulated current data
                    request = RiskAssessmentRequest(
                        patient_id=patient_id,
                        risk_type=risk_type,
                        # In production, would include real-time data
                        assessment_timestamp=datetime.utcnow()
                    )
                    
                    try:
                        result = await self.assess_risk(request)
                        monitoring_results.append(result)
                        
                        # Check for alerts
                        if result.alert_threshold_crossed:
                            await self._trigger_alert(result)
                        
                    except Exception as e:
                        self.logger.error(f"Continuous monitoring failed for {risk_type}: {e}")
            
            # Wait before next monitoring cycle
            await asyncio.sleep(300)  # 5 minutes
        
        return monitoring_results
    
    async def _trigger_alert(self, assessment: RiskAssessmentResponse):
        """Trigger alert for high-risk assessment."""
        
        alert_message = f"""
RISK ALERT - {assessment.risk_type.value.upper()}

Patient: {assessment.patient_id}
Risk Level: {assessment.risk_level.value.upper()}
Risk Score: {assessment.risk_score:.2f}
Confidence: {assessment.confidence:.2f}

Contributing Factors:
{chr(10).join(f"• {factor}" for factor in assessment.contributing_factors)}

Recommendations:
{chr(10).join(f"• {rec}" for rec in assessment.recommendations)}

Assessment Time: {assessment.assessment_timestamp}
        """
        
        self.logger.warning(f"RISK ALERT triggered", alert=alert_message)
        
        # In production, would send to alerting system
        # (PACS, SMS, email, EHR alerts, etc.)
    
    def get_risk_trends(
        self, 
        assessments: List[RiskAssessmentResponse]
    ) -> Dict[str, Any]:
        """Analyze risk trends over time."""
        
        if not assessments:
            return {}
        
        # Group by risk type
        risk_groups = {}
        for assessment in assessments:
            risk_type = assessment.risk_type
            if risk_type not in risk_groups:
                risk_groups[risk_type] = []
            risk_groups[risk_type].append(assessment)
        
        trends = {}
        for risk_type, group in risk_groups.items():
            # Sort by timestamp
            sorted_assessments = sorted(group, key=lambda x: x.assessment_timestamp)
            scores = [a.risk_score for a in sorted_assessments]
            
            trends[risk_type.value] = {
                'total_assessments': len(sorted_assessments),
                'latest_score': scores[-1] if scores else 0,
                'highest_score': max(scores) if scores else 0,
                'average_score': sum(scores) / len(scores) if scores else 0,
                'trend_direction': self._calculate_trend_direction(scores),
                'alerts_triggered': sum(1 for a in sorted_assessments if a.alert_threshold_crossed)
            }
        
        return trends
    
    def _calculate_trend_direction(self, scores: List[float]) -> str:
        """Calculate trend direction from score history."""
        
        if len(scores) < 2:
            return "insufficient_data"
        
        # Simple trend calculation
        recent_scores = scores[-3:] if len(scores) >= 3 else scores
        if len(recent_scores) < 2:
            return "stable"
        
        if recent_scores[-1] > recent_scores[0] * 1.1:
            return "increasing"
        elif recent_scores[-1] < recent_scores[0] * 0.9:
            return "decreasing"
        else:
            return "stable"


# Factory function for easy instantiation
def create_continuous_risk_scoring_manager(config: Dict[str, Any]) -> ContinuousRiskScoringManager:
    """Create a continuous risk scoring manager with given configuration."""
    return ContinuousRiskScoringManager(config)