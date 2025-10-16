"""
Laboratory Value Interpretation for Clinical Decision Support.

This module provides automated interpretation of laboratory values with
clinical alerts, trend analysis, and evidence-based recommendations.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import structlog

from .core import ClinicalAlert, AlertSeverity, EvidenceLevel


logger = structlog.get_logger(__name__)


class LabValueStatus(Enum):
    """Laboratory value status categories."""
    
    CRITICAL_LOW = "critical_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL_HIGH = "critical_high"


class TrendDirection(Enum):
    """Laboratory value trend directions."""
    
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"
    FLUCTUATING = "fluctuating"


@dataclass
class LabReference:
    """Laboratory reference range information."""
    
    test_name: str
    unit: str
    normal_low: float
    normal_high: float
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None
    age_dependent: bool = False
    sex_dependent: bool = False
    condition_specific: Dict[str, Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.condition_specific is None:
            self.condition_specific = {}


@dataclass
class LabResult:
    """Laboratory result with metadata."""
    
    test_name: str
    value: float
    unit: str
    reference_range: str
    collected_date: datetime
    result_date: datetime
    patient_id: str
    status: LabValueStatus = LabValueStatus.NORMAL
    critical: bool = False
    delta_check_flag: bool = False
    previous_value: Optional[float] = None
    
    def __post_init__(self):
        if self.collected_date is None:
            self.collected_date = datetime.utcnow()
        if self.result_date is None:
            self.result_date = datetime.utcnow()


@dataclass
class LabInterpretation:
    """Laboratory value interpretation result."""
    
    test_name: str
    current_value: float
    status: LabValueStatus
    clinical_significance: str
    recommendations: List[str]
    monitoring_frequency: Optional[str] = None
    follow_up_tests: List[str] = None
    trend_analysis: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.follow_up_tests is None:
            self.follow_up_tests = []


class LabValueInterpreter:
    """Laboratory value interpretation engine."""
    
    def __init__(self):
        """Initialize the lab value interpreter."""
        self.reference_ranges = self._load_reference_ranges()
        self.interpretation_rules = self._load_interpretation_rules()
        self.trend_thresholds = self._load_trend_thresholds()
    
    def _load_reference_ranges(self) -> Dict[str, LabReference]:
        """Load laboratory reference ranges."""
        return {
            # Basic Metabolic Panel
            "glucose": LabReference(
                test_name="glucose",
                unit="mg/dL",
                normal_low=70,
                normal_high=99,
                critical_low=40,
                critical_high=400,
                condition_specific={
                    "diabetes": (80, 130),
                    "pregnancy": (60, 95)
                }
            ),
            "bun": LabReference(
                test_name="blood_urea_nitrogen",
                unit="mg/dL", 
                normal_low=7,
                normal_high=20,
                critical_high=100
            ),
            "creatinine": LabReference(
                test_name="creatinine",
                unit="mg/dL",
                normal_low=0.6,
                normal_high=1.2,
                critical_high=5.0,
                sex_dependent=True
            ),
            "sodium": LabReference(
                test_name="sodium",
                unit="mEq/L",
                normal_low=136,
                normal_high=145,
                critical_low=120,
                critical_high=160
            ),
            "potassium": LabReference(
                test_name="potassium",
                unit="mEq/L",
                normal_low=3.5,
                normal_high=5.0,
                critical_low=2.5,
                critical_high=6.5
            ),
            "chloride": LabReference(
                test_name="chloride",
                unit="mEq/L",
                normal_low=98,
                normal_high=107,
                critical_low=80,
                critical_high=120
            ),
            
            # Complete Blood Count
            "hemoglobin": LabReference(
                test_name="hemoglobin",
                unit="g/dL",
                normal_low=12.0,
                normal_high=15.5,
                critical_low=7.0,
                critical_high=20.0,
                sex_dependent=True
            ),
            "hematocrit": LabReference(
                test_name="hematocrit",
                unit="%",
                normal_low=36,
                normal_high=46,
                critical_low=20,
                critical_high=60,
                sex_dependent=True
            ),
            "wbc": LabReference(
                test_name="white_blood_cells",
                unit="K/uL",
                normal_low=4.5,
                normal_high=11.0,
                critical_low=1.0,
                critical_high=50.0
            ),
            "platelet": LabReference(
                test_name="platelets",
                unit="K/uL",
                normal_low=150,
                normal_high=450,
                critical_low=20,
                critical_high=1000
            ),
            
            # Liver Function
            "alt": LabReference(
                test_name="alanine_aminotransferase",
                unit="U/L",
                normal_low=7,
                normal_high=40,
                critical_high=300
            ),
            "ast": LabReference(
                test_name="aspartate_aminotransferase",
                unit="U/L",
                normal_low=10,
                normal_high=40,
                critical_high=300
            ),
            "bilirubin_total": LabReference(
                test_name="total_bilirubin",
                unit="mg/dL",
                normal_low=0.2,
                normal_high=1.0,
                critical_high=20.0
            ),
            
            # Cardiac Markers
            "troponin_i": LabReference(
                test_name="troponin_i",
                unit="ng/mL",
                normal_low=0.0,
                normal_high=0.04,
                critical_high=50.0
            ),
            "bnp": LabReference(
                test_name="brain_natriuretic_peptide",
                unit="pg/mL",
                normal_low=0,
                normal_high=100,
                age_dependent=True
            ),
            
            # Lipid Panel
            "total_cholesterol": LabReference(
                test_name="total_cholesterol",
                unit="mg/dL",
                normal_low=100,
                normal_high=200,
                critical_high=500
            ),
            "ldl_cholesterol": LabReference(
                test_name="ldl_cholesterol",
                unit="mg/dL",
                normal_low=0,
                normal_high=100,
                condition_specific={
                    "diabetes": (0, 70),
                    "cardiovascular_disease": (0, 70)
                }
            ),
            "hdl_cholesterol": LabReference(
                test_name="hdl_cholesterol",
                unit="mg/dL",
                normal_low=40,
                normal_high=200,
                sex_dependent=True
            ),
            "triglycerides": LabReference(
                test_name="triglycerides",
                unit="mg/dL",
                normal_low=0,
                normal_high=150,
                critical_high=1000
            ),
            
            # Thyroid Function
            "tsh": LabReference(
                test_name="thyroid_stimulating_hormone",
                unit="mIU/L",
                normal_low=0.4,
                normal_high=4.0,
                critical_low=0.01,
                critical_high=100
            ),
            "free_t4": LabReference(
                test_name="free_thyroxine",
                unit="ng/dL",
                normal_low=0.8,
                normal_high=1.8,
                critical_low=0.2,
                critical_high=5.0
            ),
            
            # Diabetes Management
            "hba1c": LabReference(
                test_name="hemoglobin_a1c",
                unit="%",
                normal_low=4.0,
                normal_high=5.6,
                condition_specific={
                    "diabetes": (7.0, 7.0),  # Target <7%
                    "prediabetes": (5.7, 6.4)
                }
            )
        }
    
    def _load_interpretation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load clinical interpretation rules for lab values."""
        return {
            "glucose": {
                "critical_low": {
                    "significance": "Severe hypoglycemia - immediate intervention required",
                    "recommendations": [
                        "Administer IV dextrose or glucagon immediately",
                        "Monitor continuously until stable",
                        "Investigate cause of hypoglycemia"
                    ],
                    "monitoring": "continuous"
                },
                "critical_high": {
                    "significance": "Severe hyperglycemia - diabetic emergency",
                    "recommendations": [
                        "Check for diabetic ketoacidosis or hyperosmolar state",
                        "Insulin therapy may be required",
                        "Monitor electrolytes and hydration status"
                    ],
                    "monitoring": "every_1_2_hours"
                },
                "high": {
                    "significance": "Hyperglycemia - diabetes risk or poor control",
                    "recommendations": [
                        "Evaluate for diabetes if not known diabetic",
                        "Review diabetes medications if known diabetic",
                        "Consider lifestyle modifications"
                    ],
                    "monitoring": "daily_to_weekly",
                    "follow_up_tests": ["hba1c", "fructosamine"]
                }
            },
            "creatinine": {
                "high": {
                    "significance": "Elevated creatinine suggests kidney dysfunction",
                    "recommendations": [
                        "Calculate estimated GFR",
                        "Review medications for nephrotoxic agents",
                        "Consider nephrology consultation if severely elevated"
                    ],
                    "monitoring": "weekly_to_monthly",
                    "follow_up_tests": ["bun", "urinalysis", "protein_creatinine_ratio"]
                },
                "critical_high": {
                    "significance": "Severe kidney dysfunction - urgent intervention needed",
                    "recommendations": [
                        "Immediate nephrology consultation",
                        "Consider dialysis evaluation",
                        "Hold nephrotoxic medications"
                    ],
                    "monitoring": "daily"
                }
            },
            "potassium": {
                "critical_low": {
                    "significance": "Severe hypokalemia - cardiac arrhythmia risk",
                    "recommendations": [
                        "IV potassium replacement therapy",
                        "Continuous cardiac monitoring", 
                        "Check magnesium level"
                    ],
                    "monitoring": "every_4_6_hours"
                },
                "critical_high": {
                    "significance": "Severe hyperkalemia - life-threatening arrhythmia risk",
                    "recommendations": [
                        "Immediate EKG to assess for hyperkalemic changes",
                        "Consider calcium gluconate, insulin/glucose, or dialysis",
                        "Hold ACE inhibitors and potassium supplements"
                    ],
                    "monitoring": "continuous"
                }
            },
            "troponin_i": {
                "high": {
                    "significance": "Elevated troponin indicates myocardial injury",
                    "recommendations": [
                        "Evaluate for acute coronary syndrome",
                        "Serial troponin measurements",
                        "Consider cardiology consultation"
                    ],
                    "monitoring": "every_6_8_hours",
                    "follow_up_tests": ["ecg", "echocardiogram", "ck_mb"]
                }
            },
            "hemoglobin": {
                "critical_low": {
                    "significance": "Severe anemia - transfusion may be needed",
                    "recommendations": [
                        "Consider blood transfusion",
                        "Investigate cause of anemia",
                        "Monitor for signs of cardiac compromise"
                    ],
                    "monitoring": "every_6_12_hours"
                },
                "low": {
                    "significance": "Anemia - investigate underlying cause",
                    "recommendations": [
                        "Iron studies, B12, folate levels",
                        "Reticulocyte count",
                        "Consider GI workup if appropriate"
                    ],
                    "follow_up_tests": ["iron_studies", "vitamin_b12", "folate", "reticulocyte_count"]
                }
            }
        }
    
    def _load_trend_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load thresholds for trend analysis."""
        return {
            "glucose": {"significant_change": 20, "critical_change": 50},
            "creatinine": {"significant_change": 0.3, "critical_change": 0.5}, 
            "potassium": {"significant_change": 0.5, "critical_change": 1.0},
            "hemoglobin": {"significant_change": 1.0, "critical_change": 2.0},
            "platelet": {"significant_change": 50, "critical_change": 100}
        }
    
    async def interpret_lab_value(
        self,
        result: LabResult,
        patient_conditions: List[str] = None,
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None
    ) -> LabInterpretation:
        """
        Interpret a single laboratory value.
        
        Args:
            result: Laboratory result to interpret
            patient_conditions: List of patient conditions
            patient_age: Patient age
            patient_sex: Patient sex
            
        Returns:
            Laboratory interpretation
        """
        if patient_conditions is None:
            patient_conditions = []
        
        test_name = result.test_name.lower()
        
        if test_name not in self.reference_ranges:
            # Unknown test - return basic interpretation
            return LabInterpretation(
                test_name=result.test_name,
                current_value=result.value,
                status=LabValueStatus.NORMAL,
                clinical_significance="Reference ranges not available for this test",
                recommendations=["Review with clinical laboratory or physician"],
                monitoring_frequency="as_clinically_indicated"
            )
        
        reference = self.reference_ranges[test_name]
        
        # Determine appropriate reference range
        normal_low, normal_high = self._get_reference_range(
            reference, patient_conditions, patient_age, patient_sex
        )
        
        # Determine status
        status = self._determine_status(result.value, reference, normal_low, normal_high)
        
        # Get clinical interpretation
        interpretation = self._get_clinical_interpretation(test_name, status, result.value)
        
        # Perform trend analysis if previous value available
        trend_analysis = None
        if result.previous_value is not None:
            trend_analysis = self._analyze_trend(
                test_name, result.previous_value, result.value
            )
        
        return LabInterpretation(
            test_name=result.test_name,
            current_value=result.value,
            status=status,
            clinical_significance=interpretation["significance"],
            recommendations=interpretation["recommendations"],
            monitoring_frequency=interpretation.get("monitoring"),
            follow_up_tests=interpretation.get("follow_up_tests", []),
            trend_analysis=trend_analysis
        )
    
    def _get_reference_range(
        self,
        reference: LabReference,
        conditions: List[str],
        age: Optional[int],
        sex: Optional[str]
    ) -> Tuple[float, float]:
        """Get appropriate reference range based on patient characteristics."""
        normal_low = reference.normal_low
        normal_high = reference.normal_high
        
        # Check for condition-specific ranges
        for condition in conditions:
            if condition.lower() in reference.condition_specific:
                cond_low, cond_high = reference.condition_specific[condition.lower()]
                normal_low = cond_low
                normal_high = cond_high
                break
        
        # Apply sex-specific adjustments
        if reference.sex_dependent and sex:
            if reference.test_name == "creatinine":
                if sex.lower() == "female":
                    normal_low = 0.5
                    normal_high = 1.0
                elif sex.lower() == "male":
                    normal_low = 0.7
                    normal_high = 1.3
            elif reference.test_name == "hemoglobin":
                if sex.lower() == "female":
                    normal_low = 12.0
                    normal_high = 15.5
                elif sex.lower() == "male":
                    normal_low = 13.5
                    normal_high = 17.5
            elif reference.test_name == "hdl_cholesterol":
                if sex.lower() == "female":
                    normal_low = 50
                elif sex.lower() == "male":
                    normal_low = 40
        
        # Apply age-specific adjustments
        if reference.age_dependent and age:
            if reference.test_name == "bnp" and age > 75:
                normal_high = 300  # Higher threshold for elderly
        
        return normal_low, normal_high
    
    def _determine_status(
        self,
        value: float,
        reference: LabReference,
        normal_low: float,
        normal_high: float
    ) -> LabValueStatus:
        """Determine the status of a lab value."""
        # Check critical ranges first
        if reference.critical_low is not None and value <= reference.critical_low:
            return LabValueStatus.CRITICAL_LOW
        
        if reference.critical_high is not None and value >= reference.critical_high:
            return LabValueStatus.CRITICAL_HIGH
        
        # Check normal ranges
        if value < normal_low:
            return LabValueStatus.LOW
        elif value > normal_high:
            return LabValueStatus.HIGH
        else:
            return LabValueStatus.NORMAL
    
    def _get_clinical_interpretation(
        self,
        test_name: str,
        status: LabValueStatus,
        value: float
    ) -> Dict[str, Any]:
        """Get clinical interpretation for a lab value status."""
        if test_name in self.interpretation_rules:
            rules = self.interpretation_rules[test_name]
            status_key = status.value
            
            if status_key in rules:
                return rules[status_key]
        
        # Default interpretations
        default_interpretations = {
            LabValueStatus.CRITICAL_LOW: {
                "significance": "Critically low value requiring immediate attention",
                "recommendations": ["Immediate clinical evaluation and intervention required"],
                "monitoring": "continuous"
            },
            LabValueStatus.CRITICAL_HIGH: {
                "significance": "Critically high value requiring immediate attention",
                "recommendations": ["Immediate clinical evaluation and intervention required"],
                "monitoring": "continuous"
            },
            LabValueStatus.LOW: {
                "significance": "Below normal range",
                "recommendations": ["Clinical correlation recommended", "Consider repeat testing"],
                "monitoring": "as_clinically_indicated"
            },
            LabValueStatus.HIGH: {
                "significance": "Above normal range",
                "recommendations": ["Clinical correlation recommended", "Consider repeat testing"],
                "monitoring": "as_clinically_indicated"
            },
            LabValueStatus.NORMAL: {
                "significance": "Within normal limits",
                "recommendations": ["No immediate action required"],
                "monitoring": "routine"
            }
        }
        
        return default_interpretations.get(status, default_interpretations[LabValueStatus.NORMAL])
    
    def _analyze_trend(
        self,
        test_name: str,
        previous_value: float,
        current_value: float
    ) -> Dict[str, Any]:
        """Analyze trend between previous and current lab values."""
        change = current_value - previous_value
        percent_change = (change / previous_value) * 100 if previous_value != 0 else 0
        
        # Get trend thresholds
        thresholds = self.trend_thresholds.get(test_name, {
            "significant_change": abs(current_value * 0.1),  # 10% default
            "critical_change": abs(current_value * 0.2)      # 20% default
        })
        
        # Determine trend significance
        if abs(change) >= thresholds["critical_change"]:
            significance = "critical"
        elif abs(change) >= thresholds["significant_change"]:
            significance = "significant"
        else:
            significance = "minimal"
        
        # Determine direction
        if change > 0:
            direction = "increasing"
        elif change < 0:
            direction = "decreasing"
        else:
            direction = "stable"
        
        return {
            "previous_value": previous_value,
            "current_value": current_value,
            "absolute_change": change,
            "percent_change": percent_change,
            "direction": direction,
            "significance": significance,
            "trend_alert": significance in ["significant", "critical"]
        }
    
    async def generate_lab_alerts(
        self,
        interpretations: List[LabInterpretation],
        patient_id: str
    ) -> List[ClinicalAlert]:
        """
        Generate clinical alerts based on lab interpretations.
        
        Args:
            interpretations: List of lab interpretations
            patient_id: Patient identifier
            
        Returns:
            List of clinical alerts
        """
        alerts = []
        
        for interp in interpretations:
            # Create alerts for abnormal values
            if interp.status in [LabValueStatus.CRITICAL_LOW, LabValueStatus.CRITICAL_HIGH]:
                severity = AlertSeverity.CRITICAL
            elif interp.status in [LabValueStatus.LOW, LabValueStatus.HIGH]:
                severity = AlertSeverity.HIGH
            else:
                continue  # No alert for normal values
            
            # Create primary lab alert
            alert = ClinicalAlert(
                alert_id=f"lab_alert_{patient_id}_{interp.test_name}_{datetime.utcnow().timestamp()}",
                title=f"Abnormal Lab Value: {interp.test_name.title()}",
                description=f"{interp.test_name}: {interp.current_value}. {interp.clinical_significance}",
                severity=severity,
                alert_type="laboratory_value",
                patient_id=patient_id,
                triggered_by=f"{interp.test_name}={interp.current_value}",
                recommendation="; ".join(interp.recommendations),
                evidence_level=EvidenceLevel.LEVEL_A,
                source="laboratory_interpreter",
                created_at=datetime.utcnow()
            )
            alerts.append(alert)
            
            # Create trend alert if significant
            if (interp.trend_analysis and 
                interp.trend_analysis.get("trend_alert", False)):
                
                trend_alert = ClinicalAlert(
                    alert_id=f"trend_alert_{patient_id}_{interp.test_name}_{datetime.utcnow().timestamp()}",
                    title=f"Significant Lab Trend: {interp.test_name.title()}",
                    description=f"{interp.test_name} {interp.trend_analysis['direction']} significantly: {interp.trend_analysis['percent_change']:.1f}% change",
                    severity=AlertSeverity.MODERATE if interp.trend_analysis['significance'] == 'significant' else AlertSeverity.HIGH,
                    alert_type="laboratory_trend",
                    patient_id=patient_id,
                    triggered_by=f"{interp.test_name}_trend",
                    recommendation="Monitor closely and consider clinical correlation",
                    evidence_level=EvidenceLevel.LEVEL_B,
                    source="laboratory_interpreter",
                    created_at=datetime.utcnow()
                )
                alerts.append(trend_alert)
        
        return alerts