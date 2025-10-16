"""
Laboratory Medicine AI for Intelligent Lab Result Analysis.

This module provides comprehensive laboratory data analysis including intelligent
result interpretation, automated flagging systems, trend analysis, reference
range optimization, and clinical decision support integration.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
import structlog
from pydantic import BaseModel, Field
import statistics
from collections import defaultdict, deque

try:
    import scipy.stats as stats
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = structlog.get_logger(__name__)


class LabTestType(Enum):
    """Types of laboratory tests."""
    CHEMISTRY = "chemistry"
    HEMATOLOGY = "hematology"
    MICROBIOLOGY = "microbiology"
    IMMUNOLOGY = "immunology"
    MOLECULAR = "molecular"
    TOXICOLOGY = "toxicology"
    COAGULATION = "coagulation"
    ENDOCRINOLOGY = "endocrinology"
    CARDIOLOGY = "cardiology"
    GENETICS = "genetics"
    PATHOLOGY = "pathology"


class LabResultStatus(Enum):
    """Laboratory result status."""
    NORMAL = "normal"
    ABNORMAL_LOW = "abnormal_low"
    ABNORMAL_HIGH = "abnormal_high"
    CRITICAL_LOW = "critical_low"
    CRITICAL_HIGH = "critical_high"
    PANIC_VALUE = "panic_value"
    INVALID = "invalid"
    PENDING = "pending"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    PANIC = "panic"


class TrendDirection(Enum):
    """Trend direction for lab values."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    FLUCTUATING = "fluctuating"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class LabTest:
    """Represents a laboratory test definition."""
    test_code: str
    test_name: str
    test_type: LabTestType
    unit: str
    reference_ranges: Dict[str, Tuple[float, float]]  # demographic_key -> (low, high)
    critical_ranges: Dict[str, Tuple[float, float]]
    panic_ranges: Dict[str, Tuple[float, float]]
    methodology: Optional[str] = None
    turnaround_time: Optional[timedelta] = None
    interfering_factors: List[str] = field(default_factory=list)
    clinical_significance: Optional[str] = None


@dataclass
class LabResult:
    """Represents a laboratory test result."""
    result_id: str
    patient_id: str
    test_code: str
    value: Optional[float]
    text_value: Optional[str] = None
    unit: str
    status: LabResultStatus
    reference_range: Optional[Tuple[float, float]] = None
    collected_datetime: datetime
    resulted_datetime: datetime
    ordering_provider: Optional[str] = None
    performing_lab: Optional[str] = None
    specimen_type: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    comments: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LabAlert:
    """Represents a laboratory alert."""
    alert_id: str
    patient_id: str
    test_code: str
    severity: AlertSeverity
    message: str
    triggered_datetime: datetime
    value: Optional[float] = None
    reference_range: Optional[Tuple[float, float]] = None
    recommendations: List[str] = field(default_factory=list)
    auto_resolved: bool = False
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class LabAnalysisRequest(BaseModel):
    """Request for laboratory analysis."""
    
    patient_id: str
    results: List[LabResult]
    patient_demographics: Dict[str, Any] = Field(default_factory=dict)
    clinical_context: Dict[str, Any] = Field(default_factory=dict)
    historical_results: List[LabResult] = Field(default_factory=list)
    analysis_parameters: Dict[str, Any] = Field(default_factory=dict)
    request_timestamp: datetime = Field(default_factory=datetime.utcnow)


class LabAnalysisResponse(BaseModel):
    """Response from laboratory analysis."""
    
    patient_id: str
    analyzed_results: List[LabResult]
    alerts: List[LabAlert] = Field(default_factory=list)
    interpretations: Dict[str, str] = Field(default_factory=dict)
    trends: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    critical_values: List[str] = Field(default_factory=list)
    quality_indicators: Dict[str, Any] = Field(default_factory=dict)
    analysis_timestamp: datetime
    analyzer_version: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseLabAnalyzer(ABC):
    """Base class for laboratory analyzers."""
    
    def __init__(self, analyzer_name: str, version: str = "1.0.0"):
        self.analyzer_name = analyzer_name
        self.version = version
        self.logger = structlog.get_logger(__name__)
    
    @abstractmethod
    async def analyze_results(
        self, 
        request: LabAnalysisRequest
    ) -> LabAnalysisResponse:
        """Analyze laboratory results."""
        pass
    
    def _calculate_reference_range(
        self, 
        test: LabTest, 
        demographics: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate appropriate reference range based on demographics."""
        
        # Default reference range
        default_range = test.reference_ranges.get('default', (0.0, 100.0))
        
        # Age-based ranges
        age = demographics.get('age', 0)
        gender = demographics.get('gender', 'unknown').lower()
        
        # Try age-gender specific range first
        age_gender_key = f"{gender}_{self._get_age_group(age)}"
        if age_gender_key in test.reference_ranges:
            return test.reference_ranges[age_gender_key]
        
        # Try gender-specific range
        if gender in test.reference_ranges:
            return test.reference_ranges[gender]
        
        # Try age-specific range
        age_group = self._get_age_group(age)
        if age_group in test.reference_ranges:
            return test.reference_ranges[age_group]
        
        return default_range
    
    def _get_age_group(self, age: int) -> str:
        """Determine age group for reference range selection."""
        if age < 1:
            return 'neonate'
        elif age < 12:
            return 'pediatric'
        elif age < 18:
            return 'adolescent'
        elif age < 65:
            return 'adult'
        else:
            return 'elderly'
    
    def _assess_result_status(
        self, 
        value: float, 
        reference_range: Tuple[float, float],
        critical_range: Optional[Tuple[float, float]] = None,
        panic_range: Optional[Tuple[float, float]] = None
    ) -> LabResultStatus:
        """Assess the status of a lab result."""
        
        low_ref, high_ref = reference_range
        
        # Check panic values first
        if panic_range:
            panic_low, panic_high = panic_range
            if value <= panic_low or value >= panic_high:
                return LabResultStatus.PANIC_VALUE
        
        # Check critical values
        if critical_range:
            crit_low, crit_high = critical_range
            if value <= crit_low:
                return LabResultStatus.CRITICAL_LOW
            elif value >= crit_high:
                return LabResultStatus.CRITICAL_HIGH
        
        # Check normal ranges
        if value < low_ref:
            return LabResultStatus.ABNORMAL_LOW
        elif value > high_ref:
            return LabResultStatus.ABNORMAL_HIGH
        else:
            return LabResultStatus.NORMAL


class ChemistryAnalyzer(BaseLabAnalyzer):
    """Analyzer for chemistry laboratory results."""
    
    def __init__(self):
        super().__init__("chemistry_analyzer", "v2.1.0")
        
        # Define common chemistry tests
        self.chemistry_tests = {
            'glucose': LabTest(
                test_code='GLU',
                test_name='Glucose',
                test_type=LabTestType.CHEMISTRY,
                unit='mg/dL',
                reference_ranges={
                    'default': (70, 100),
                    'diabetic': (80, 130)
                },
                critical_ranges={'default': (40, 400)},
                panic_ranges={'default': (20, 600)},
                clinical_significance='Diabetes screening and monitoring'
            ),
            'creatinine': LabTest(
                test_code='CREAT',
                test_name='Creatinine',
                test_type=LabTestType.CHEMISTRY,
                unit='mg/dL',
                reference_ranges={
                    'male_adult': (0.7, 1.3),
                    'female_adult': (0.6, 1.1),
                    'pediatric': (0.3, 0.7)
                },
                critical_ranges={'default': (0.2, 10.0)},
                panic_ranges={'default': (0.1, 15.0)},
                clinical_significance='Kidney function assessment'
            ),
            'sodium': LabTest(
                test_code='NA',
                test_name='Sodium',
                test_type=LabTestType.CHEMISTRY,
                unit='mEq/L',
                reference_ranges={'default': (136, 145)},
                critical_ranges={'default': (120, 160)},
                panic_ranges={'default': (115, 170)},
                clinical_significance='Electrolyte balance and hydration status'
            ),
            'potassium': LabTest(
                test_code='K',
                test_name='Potassium',
                test_type=LabTestType.CHEMISTRY,
                unit='mEq/L',
                reference_ranges={'default': (3.5, 5.0)},
                critical_ranges={'default': (2.5, 6.0)},
                panic_ranges={'default': (2.0, 6.5)},
                clinical_significance='Cardiac and neuromuscular function'
            ),
            'troponin_i': LabTest(
                test_code='TROPI',
                test_name='Troponin I',
                test_type=LabTestType.CARDIOLOGY,
                unit='ng/mL',
                reference_ranges={'default': (0.0, 0.04)},
                critical_ranges={'default': (0.04, 50.0)},
                panic_ranges={'default': (10.0, 100.0)},
                clinical_significance='Myocardial infarction detection'
            )
        }
    
    async def analyze_results(
        self, 
        request: LabAnalysisRequest
    ) -> LabAnalysisResponse:
        """Analyze chemistry laboratory results."""
        
        alerts = []
        interpretations = {}
        trends = {}
        recommendations = []
        critical_values = []
        
        try:
            # Analyze each result
            for result in request.results:
                # Get test definition
                test = self.chemistry_tests.get(result.test_code.lower())
                if not test:
                    continue
                
                if result.value is not None:
                    # Calculate appropriate reference range
                    ref_range = self._calculate_reference_range(test, request.patient_demographics)
                    result.reference_range = ref_range
                    
                    # Assess result status
                    crit_range = test.critical_ranges.get('default')
                    panic_range = test.panic_ranges.get('default')
                    result.status = self._assess_result_status(result.value, ref_range, crit_range, panic_range)
                    
                    # Generate alerts for abnormal values
                    alert = self._generate_chemistry_alert(result, test, ref_range)
                    if alert:
                        alerts.append(alert)
                    
                    # Track critical values
                    if result.status in [LabResultStatus.CRITICAL_LOW, LabResultStatus.CRITICAL_HIGH, LabResultStatus.PANIC_VALUE]:
                        critical_values.append(f"{test.test_name}: {result.value} {result.unit}")
                    
                    # Generate interpretation
                    interpretation = self._interpret_chemistry_result(result, test, request.clinical_context)
                    interpretations[result.test_code] = interpretation
            
            # Analyze trends if historical data available
            if request.historical_results:
                trends = await self._analyze_chemistry_trends(request.results, request.historical_results)
            
            # Generate panel-specific interpretations
            panel_interpretations = self._analyze_chemistry_panels(request.results, request.clinical_context)
            interpretations.update(panel_interpretations)
            
            # Generate recommendations
            recommendations = self._generate_chemistry_recommendations(request.results, alerts, trends)
            
            # Calculate quality indicators
            quality_indicators = self._calculate_quality_indicators(request.results)
            
            return LabAnalysisResponse(
                patient_id=request.patient_id,
                analyzed_results=request.results,
                alerts=alerts,
                interpretations=interpretations,
                trends=trends,
                recommendations=recommendations,
                critical_values=critical_values,
                quality_indicators=quality_indicators,
                analysis_timestamp=datetime.utcnow(),
                analyzer_version=self.version,
                metadata={
                    'tests_analyzed': len(request.results),
                    'alerts_generated': len(alerts),
                    'critical_values_detected': len(critical_values)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Chemistry analysis failed: {e}")
            raise
    
    def _generate_chemistry_alert(
        self, 
        result: LabResult, 
        test: LabTest, 
        ref_range: Tuple[float, float]
    ) -> Optional[LabAlert]:
        """Generate alert for chemistry result if needed."""
        
        if result.status == LabResultStatus.NORMAL:
            return None
        
        # Determine severity
        if result.status == LabResultStatus.PANIC_VALUE:
            severity = AlertSeverity.PANIC
            message = f"PANIC VALUE: {test.test_name} = {result.value} {result.unit}"
        elif result.status in [LabResultStatus.CRITICAL_LOW, LabResultStatus.CRITICAL_HIGH]:
            severity = AlertSeverity.CRITICAL
            message = f"CRITICAL: {test.test_name} = {result.value} {result.unit}"
        else:
            severity = AlertSeverity.WARNING
            message = f"ABNORMAL: {test.test_name} = {result.value} {result.unit}"
        
        # Generate recommendations
        recommendations = self._generate_result_recommendations(result, test)
        
        return LabAlert(
            alert_id=f"alert_{result.result_id}",
            patient_id=result.patient_id,
            test_code=result.test_code,
            severity=severity,
            message=message,
            triggered_datetime=datetime.utcnow(),
            value=result.value,
            reference_range=ref_range,
            recommendations=recommendations
        )
    
    def _generate_result_recommendations(self, result: LabResult, test: LabTest) -> List[str]:
        """Generate recommendations based on specific test result."""
        
        recommendations = []
        test_code = result.test_code.lower()
        
        if test_code == 'glu':  # Glucose
            if result.status == LabResultStatus.ABNORMAL_HIGH:
                recommendations.extend([
                    "Consider diabetes screening",
                    "Repeat fasting glucose or HbA1c",
                    "Evaluate for diabetic complications"
                ])
            elif result.status == LabResultStatus.ABNORMAL_LOW:
                recommendations.extend([
                    "Evaluate for hypoglycemia causes",
                    "Consider immediate glucose administration if symptomatic",
                    "Monitor closely"
                ])
        
        elif test_code == 'creat':  # Creatinine
            if result.status == LabResultStatus.ABNORMAL_HIGH:
                recommendations.extend([
                    "Assess kidney function with eGFR",
                    "Review medications for nephrotoxicity",
                    "Consider nephrology consultation",
                    "Evaluate for acute kidney injury"
                ])
        
        elif test_code == 'k':  # Potassium
            if result.status == LabResultStatus.CRITICAL_HIGH:
                recommendations.extend([
                    "URGENT: Check ECG for cardiac effects",
                    "Consider immediate treatment to lower potassium",
                    "Evaluate medications and supplements"
                ])
            elif result.status == LabResultStatus.CRITICAL_LOW:
                recommendations.extend([
                    "Assess for cardiac arrhythmias",
                    "Consider potassium replacement",
                    "Evaluate for ongoing losses"
                ])
        
        elif test_code == 'tropi':  # Troponin
            if result.status in [LabResultStatus.ABNORMAL_HIGH, LabResultStatus.CRITICAL_HIGH]:
                recommendations.extend([
                    "URGENT: Evaluate for myocardial infarction",
                    "Obtain ECG and cardiology consultation",
                    "Consider cardiac catheterization",
                    "Initiate appropriate cardiac medications"
                ])
        
        return recommendations
    
    def _interpret_chemistry_result(
        self, 
        result: LabResult, 
        test: LabTest, 
        clinical_context: Dict[str, Any]
    ) -> str:
        """Generate clinical interpretation for chemistry result."""
        
        if result.status == LabResultStatus.NORMAL:
            return f"{test.test_name} is within normal limits"
        
        interpretation_parts = []
        
        # Basic interpretation
        if result.status == LabResultStatus.ABNORMAL_HIGH:
            interpretation_parts.append(f"Elevated {test.test_name.lower()}")
        elif result.status == LabResultStatus.ABNORMAL_LOW:
            interpretation_parts.append(f"Low {test.test_name.lower()}")
        elif result.status in [LabResultStatus.CRITICAL_HIGH, LabResultStatus.CRITICAL_LOW]:
            interpretation_parts.append(f"Critically {'high' if 'HIGH' in result.status.value else 'low'} {test.test_name.lower()}")
        
        # Add clinical significance
        if test.clinical_significance:
            interpretation_parts.append(f"Clinical significance: {test.clinical_significance}")
        
        # Add context-specific interpretation
        symptoms = clinical_context.get('symptoms', [])
        medications = clinical_context.get('medications', [])
        
        if result.test_code.lower() == 'glu' and 'diabetes' in symptoms:
            interpretation_parts.append("Consistent with diabetes mellitus")
        
        return ". ".join(interpretation_parts)
    
    async def _analyze_chemistry_trends(
        self, 
        current_results: List[LabResult], 
        historical_results: List[LabResult]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze trends in chemistry results."""
        
        trends = {}
        
        # Group results by test code
        test_groups = defaultdict(list)
        for result in historical_results + current_results:
            if result.value is not None:
                test_groups[result.test_code].append(result)
        
        for test_code, results in test_groups.items():
            if len(results) < 2:
                continue
            
            # Sort by date
            sorted_results = sorted(results, key=lambda r: r.resulted_datetime)
            
            # Extract values and dates
            values = [r.value for r in sorted_results]
            dates = [r.resulted_datetime for r in sorted_results]
            
            # Calculate trend
            trend_analysis = self._calculate_trend(values, dates)
            
            trends[test_code] = {
                'direction': trend_analysis['direction'],
                'slope': trend_analysis['slope'],
                'correlation': trend_analysis['correlation'],
                'recent_change_percent': trend_analysis['recent_change_percent'],
                'data_points': len(values),
                'time_span_days': (dates[-1] - dates[0]).days,
                'interpretation': self._interpret_trend(test_code, trend_analysis)
            }
        
        return trends
    
    def _calculate_trend(self, values: List[float], dates: List[datetime]) -> Dict[str, Any]:
        """Calculate statistical trend for lab values."""
        
        if len(values) < 2:
            return {
                'direction': TrendDirection.INSUFFICIENT_DATA,
                'slope': 0.0,
                'correlation': 0.0,
                'recent_change_percent': 0.0
            }
        
        # Convert dates to numeric values (days from first date)
        base_date = dates[0]
        numeric_dates = [(d - base_date).days for d in dates]
        
        # Calculate linear regression if scipy available
        if SCIPY_AVAILABLE and len(values) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(numeric_dates, values)
            correlation = r_value
        else:
            # Simple slope calculation
            slope = (values[-1] - values[0]) / (numeric_dates[-1] - numeric_dates[0]) if numeric_dates[-1] != numeric_dates[0] else 0
            correlation = 0.0
        
        # Determine trend direction
        if abs(slope) < 0.01:  # Threshold for "stable"
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING
        
        # Calculate recent change percentage
        if len(values) >= 2:
            recent_change_percent = ((values[-1] - values[-2]) / values[-2]) * 100 if values[-2] != 0 else 0
        else:
            recent_change_percent = 0.0
        
        return {
            'direction': direction,
            'slope': slope,
            'correlation': correlation,
            'recent_change_percent': recent_change_percent
        }
    
    def _interpret_trend(self, test_code: str, trend_analysis: Dict[str, Any]) -> str:
        """Interpret trend analysis for specific test."""
        
        direction = trend_analysis['direction']
        change_percent = trend_analysis['recent_change_percent']
        
        base_interpretation = f"{test_code} trend: {direction.value}"
        
        if abs(change_percent) > 20:
            base_interpretation += f" (recent change: {change_percent:.1f}%)"
        
        # Test-specific interpretations
        if test_code.lower() == 'creat' and direction == TrendDirection.INCREASING:
            base_interpretation += " - Monitor for kidney function decline"
        elif test_code.lower() == 'glu' and direction == TrendDirection.INCREASING:
            base_interpretation += " - Consider diabetes management optimization"
        
        return base_interpretation
    
    def _analyze_chemistry_panels(
        self, 
        results: List[LabResult], 
        clinical_context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Analyze chemistry panels for combined interpretations."""
        
        interpretations = {}
        
        # Group results by common panels
        test_codes = {r.test_code.lower() for r in results}
        
        # Basic Metabolic Panel (BMP)
        bmp_tests = {'na', 'k', 'cl', 'co2', 'bun', 'creat', 'glu'}
        if bmp_tests.intersection(test_codes):
            bmp_interpretation = self._interpret_bmp(results)
            if bmp_interpretation:
                interpretations['basic_metabolic_panel'] = bmp_interpretation
        
        # Liver Function Tests
        lft_tests = {'alt', 'ast', 'alp', 'bili_total', 'bili_direct'}
        if lft_tests.intersection(test_codes):
            lft_interpretation = self._interpret_lft(results)
            if lft_interpretation:
                interpretations['liver_function_tests'] = lft_interpretation
        
        # Cardiac Panel
        cardiac_tests = {'tropi', 'tropii', 'ck', 'ckmb', 'bnp', 'proBNP'}
        if cardiac_tests.intersection(test_codes):
            cardiac_interpretation = self._interpret_cardiac_panel(results)
            if cardiac_interpretation:
                interpretations['cardiac_panel'] = cardiac_interpretation
        
        return interpretations
    
    def _interpret_bmp(self, results: List[LabResult]) -> Optional[str]:
        """Interpret Basic Metabolic Panel."""
        
        abnormal_results = [r for r in results if r.status != LabResultStatus.NORMAL]
        
        if not abnormal_results:
            return "Basic metabolic panel within normal limits"
        
        interpretation_parts = []
        
        # Check for electrolyte imbalances
        electrolyte_abnormalities = [r for r in abnormal_results if r.test_code.lower() in ['na', 'k', 'cl']]
        if electrolyte_abnormalities:
            interpretation_parts.append("Electrolyte imbalance detected")
        
        # Check for kidney function issues
        kidney_abnormalities = [r for r in abnormal_results if r.test_code.lower() in ['creat', 'bun']]
        if kidney_abnormalities:
            interpretation_parts.append("Kidney function abnormality")
        
        # Check for glucose abnormalities
        glucose_abnormalities = [r for r in abnormal_results if r.test_code.lower() == 'glu']
        if glucose_abnormalities:
            interpretation_parts.append("Glucose metabolism abnormality")
        
        return ". ".join(interpretation_parts) if interpretation_parts else None
    
    def _interpret_lft(self, results: List[LabResult]) -> Optional[str]:
        """Interpret Liver Function Tests."""
        
        lft_results = [r for r in results if r.test_code.lower() in ['alt', 'ast', 'alp', 'bili_total']]
        abnormal_lft = [r for r in lft_results if r.status != LabResultStatus.NORMAL]
        
        if not abnormal_lft:
            return "Liver function tests within normal limits"
        
        # Pattern recognition
        elevated_enzymes = [r for r in abnormal_lft if r.test_code.lower() in ['alt', 'ast'] and 'HIGH' in r.status.value]
        elevated_bilirubin = [r for r in abnormal_lft if r.test_code.lower() == 'bili_total' and 'HIGH' in r.status.value]
        
        if elevated_enzymes and elevated_bilirubin:
            return "Hepatocellular injury pattern with elevated bilirubin"
        elif elevated_enzymes:
            return "Hepatocellular injury pattern"
        elif elevated_bilirubin:
            return "Elevated bilirubin - consider cholestatic process"
        
        return "Liver function abnormalities detected"
    
    def _interpret_cardiac_panel(self, results: List[LabResult]) -> Optional[str]:
        """Interpret cardiac biomarkers."""
        
        cardiac_results = [r for r in results if r.test_code.lower() in ['tropi', 'tropii', 'ck', 'ckmb', 'bnp']]
        abnormal_cardiac = [r for r in cardiac_results if r.status != LabResultStatus.NORMAL]
        
        if not abnormal_cardiac:
            return "Cardiac biomarkers within normal limits"
        
        # Check for troponin elevation
        troponin_elevated = [r for r in abnormal_cardiac if 'trop' in r.test_code.lower() and 'HIGH' in r.status.value]
        if troponin_elevated:
            return "URGENT: Elevated cardiac troponin suggests myocardial injury"
        
        # Check for heart failure markers
        bnp_elevated = [r for r in abnormal_cardiac if 'bnp' in r.test_code.lower() and 'HIGH' in r.status.value]
        if bnp_elevated:
            return "Elevated BNP suggests heart failure or cardiac stress"
        
        return "Cardiac biomarker abnormalities detected"
    
    def _generate_chemistry_recommendations(
        self, 
        results: List[LabResult], 
        alerts: List[LabAlert], 
        trends: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate clinical recommendations based on chemistry results."""
        
        recommendations = []
        
        # Critical value recommendations
        panic_alerts = [a for a in alerts if a.severity == AlertSeverity.PANIC]
        if panic_alerts:
            recommendations.append("URGENT: Immediate clinical intervention required for panic values")
        
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            recommendations.append("CRITICAL: Prompt clinical evaluation and treatment indicated")
        
        # Trend-based recommendations
        for test_code, trend_data in trends.items():
            if trend_data['direction'] == TrendDirection.INCREASING and abs(trend_data['recent_change_percent']) > 25:
                recommendations.append(f"Monitor {test_code} trend - significant increase detected")
        
        # General recommendations
        abnormal_results = [r for r in results if r.status != LabResultStatus.NORMAL]
        if len(abnormal_results) > len(results) * 0.5:  # More than 50% abnormal
            recommendations.append("Consider comprehensive metabolic evaluation")
        
        # Add result-specific recommendations from alerts
        for alert in alerts:
            recommendations.extend(alert.recommendations)
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _calculate_quality_indicators(self, results: List[LabResult]) -> Dict[str, Any]:
        """Calculate quality indicators for lab results."""
        
        if not results:
            return {}
        
        # Turnaround time analysis
        turnaround_times = []
        for result in results:
            if result.collected_datetime and result.resulted_datetime:
                tat = (result.resulted_datetime - result.collected_datetime).total_seconds() / 3600  # hours
                turnaround_times.append(tat)
        
        # Result completeness
        complete_results = len([r for r in results if r.value is not None])
        completeness_rate = complete_results / len(results) if results else 0
        
        # Critical value detection rate
        critical_results = len([r for r in results if r.status in [
            LabResultStatus.CRITICAL_LOW, LabResultStatus.CRITICAL_HIGH, LabResultStatus.PANIC_VALUE
        ]])
        
        return {
            'total_results': len(results),
            'completeness_rate': completeness_rate,
            'critical_value_rate': critical_results / len(results) if results else 0,
            'average_turnaround_hours': statistics.mean(turnaround_times) if turnaround_times else 0,
            'turnaround_time_range': {
                'min': min(turnaround_times) if turnaround_times else 0,
                'max': max(turnaround_times) if turnaround_times else 0
            }
        }


class HematologyAnalyzer(BaseLabAnalyzer):
    """Analyzer for hematology laboratory results."""
    
    def __init__(self):
        super().__init__("hematology_analyzer", "v1.8.0")
        
        # Define hematology tests
        self.hematology_tests = {
            'wbc': LabTest(
                test_code='WBC',
                test_name='White Blood Cell Count',
                test_type=LabTestType.HEMATOLOGY,
                unit='cells/µL',
                reference_ranges={
                    'adult': (4000, 11000),
                    'pediatric': (5000, 15000)
                },
                critical_ranges={'default': (1000, 50000)},
                panic_ranges={'default': (500, 100000)},
                clinical_significance='Infection and immune system assessment'
            ),
            'hemoglobin': LabTest(
                test_code='HGB',
                test_name='Hemoglobin',
                test_type=LabTestType.HEMATOLOGY,
                unit='g/dL',
                reference_ranges={
                    'male_adult': (13.5, 17.5),
                    'female_adult': (12.0, 16.0),
                    'pediatric': (11.0, 14.0)
                },
                critical_ranges={'default': (7.0, 20.0)},
                panic_ranges={'default': (5.0, 25.0)},
                clinical_significance='Oxygen-carrying capacity and anemia assessment'
            ),
            'platelet': LabTest(
                test_code='PLT',
                test_name='Platelet Count',
                test_type=LabTestType.HEMATOLOGY,
                unit='cells/µL',
                reference_ranges={'default': (150000, 450000)},
                critical_ranges={'default': (50000, 1000000)},
                panic_ranges={'default': (20000, 2000000)},
                clinical_significance='Bleeding and clotting assessment'
            )
        }
    
    async def analyze_results(
        self, 
        request: LabAnalysisRequest
    ) -> LabAnalysisResponse:
        """Analyze hematology laboratory results."""
        
        alerts = []
        interpretations = {}
        recommendations = []
        critical_values = []
        
        try:
            # Analyze each hematology result
            for result in request.results:
                test = self.hematology_tests.get(result.test_code.lower())
                if not test:
                    continue
                
                if result.value is not None:
                    # Calculate reference range
                    ref_range = self._calculate_reference_range(test, request.patient_demographics)
                    result.reference_range = ref_range
                    
                    # Assess status
                    crit_range = test.critical_ranges.get('default')
                    panic_range = test.panic_ranges.get('default')
                    result.status = self._assess_result_status(result.value, ref_range, crit_range, panic_range)
                    
                    # Generate alerts
                    alert = self._generate_hematology_alert(result, test, ref_range)
                    if alert:
                        alerts.append(alert)
                    
                    # Track critical values
                    if result.status in [LabResultStatus.CRITICAL_LOW, LabResultStatus.CRITICAL_HIGH, LabResultStatus.PANIC_VALUE]:
                        critical_values.append(f"{test.test_name}: {result.value} {result.unit}")
                    
                    # Generate interpretation
                    interpretation = self._interpret_hematology_result(result, test)
                    interpretations[result.test_code] = interpretation
            
            # Analyze CBC patterns
            cbc_interpretation = self._analyze_cbc_pattern(request.results)
            if cbc_interpretation:
                interpretations['complete_blood_count'] = cbc_interpretation
            
            # Generate recommendations
            recommendations = self._generate_hematology_recommendations(request.results, alerts)
            
            return LabAnalysisResponse(
                patient_id=request.patient_id,
                analyzed_results=request.results,
                alerts=alerts,
                interpretations=interpretations,
                recommendations=recommendations,
                critical_values=critical_values,
                quality_indicators=self._calculate_quality_indicators(request.results),
                analysis_timestamp=datetime.utcnow(),
                analyzer_version=self.version,
                metadata={
                    'hematology_tests_analyzed': len([r for r in request.results if r.test_code.lower() in self.hematology_tests])
                }
            )
            
        except Exception as e:
            self.logger.error(f"Hematology analysis failed: {e}")
            raise
    
    def _generate_hematology_alert(
        self, 
        result: LabResult, 
        test: LabTest, 
        ref_range: Tuple[float, float]
    ) -> Optional[LabAlert]:
        """Generate hematology-specific alerts."""
        
        if result.status == LabResultStatus.NORMAL:
            return None
        
        severity = AlertSeverity.WARNING
        recommendations = []
        
        test_code = result.test_code.lower()
        
        if test_code == 'wbc':
            if result.status == LabResultStatus.ABNORMAL_HIGH:
                severity = AlertSeverity.CRITICAL
                recommendations.extend([
                    "Evaluate for infection or hematologic malignancy",
                    "Consider blood cultures if febrile",
                    "Review white cell differential"
                ])
            elif result.status == LabResultStatus.CRITICAL_LOW:
                severity = AlertSeverity.CRITICAL
                recommendations.extend([
                    "URGENT: Severe leukopenia - infection risk",
                    "Consider neutropenia precautions",
                    "Evaluate for bone marrow failure"
                ])
        
        elif test_code == 'hemoglobin' or test_code == 'hgb':
            if result.status == LabResultStatus.CRITICAL_LOW:
                severity = AlertSeverity.CRITICAL
                recommendations.extend([
                    "URGENT: Severe anemia",
                    "Consider blood transfusion",
                    "Evaluate for bleeding source"
                ])
        
        elif test_code == 'platelet' or test_code == 'plt':
            if result.status == LabResultStatus.CRITICAL_LOW:
                severity = AlertSeverity.CRITICAL
                recommendations.extend([
                    "URGENT: Severe thrombocytopenia",
                    "Bleeding precautions",
                    "Consider platelet transfusion",
                    "Evaluate for cause of thrombocytopenia"
                ])
        
        message = f"{severity.value.upper()}: {test.test_name} = {result.value} {result.unit}"
        
        return LabAlert(
            alert_id=f"hematology_alert_{result.result_id}",
            patient_id=result.patient_id,
            test_code=result.test_code,
            severity=severity,
            message=message,
            triggered_datetime=datetime.utcnow(),
            value=result.value,
            reference_range=ref_range,
            recommendations=recommendations
        )
    
    def _interpret_hematology_result(self, result: LabResult, test: LabTest) -> str:
        """Interpret individual hematology result."""
        
        if result.status == LabResultStatus.NORMAL:
            return f"{test.test_name} within normal limits"
        
        test_code = result.test_code.lower()
        interpretation = ""
        
        if test_code == 'wbc':
            if result.status == LabResultStatus.ABNORMAL_HIGH:
                interpretation = "Leukocytosis - suggest infection, inflammation, or hematologic disorder"
            elif result.status == LabResultStatus.ABNORMAL_LOW:
                interpretation = "Leukopenia - suggest viral infection, medication effect, or bone marrow disorder"
        
        elif test_code in ['hemoglobin', 'hgb']:
            if result.status in [LabResultStatus.ABNORMAL_LOW, LabResultStatus.CRITICAL_LOW]:
                interpretation = "Anemia - evaluate for iron deficiency, chronic disease, or bleeding"
            elif result.status == LabResultStatus.ABNORMAL_HIGH:
                interpretation = "Polycythemia - evaluate for dehydration, lung disease, or polycythemia vera"
        
        elif test_code in ['platelet', 'plt']:
            if result.status in [LabResultStatus.ABNORMAL_LOW, LabResultStatus.CRITICAL_LOW]:
                interpretation = "Thrombocytopenia - evaluate for medication effect, immune destruction, or bone marrow disorder"
            elif result.status == LabResultStatus.ABNORMAL_HIGH:
                interpretation = "Thrombocytosis - evaluate for inflammation, malignancy, or myeloproliferative disorder"
        
        return interpretation
    
    def _analyze_cbc_pattern(self, results: List[LabResult]) -> Optional[str]:
        """Analyze complete blood count pattern."""
        
        # Extract CBC components
        cbc_results = {}
        for result in results:
            test_code = result.test_code.lower()
            if test_code in ['wbc', 'hemoglobin', 'hgb', 'platelet', 'plt'] and result.value is not None:
                cbc_results[test_code] = result
        
        if len(cbc_results) < 2:
            return None
        
        # Pattern analysis
        abnormal_results = {k: v for k, v in cbc_results.items() if v.status != LabResultStatus.NORMAL}
        
        if not abnormal_results:
            return "Complete blood count within normal limits"
        
        # Check for pancytopenia (all cell lines low)
        low_results = {k: v for k, v in abnormal_results.items() if 'LOW' in v.status.value}
        if len(low_results) >= 2:
            return "Pancytopenia pattern - suggest bone marrow failure, chemotherapy effect, or systemic disease"
        
        # Check for specific patterns
        wbc_high = any(k in ['wbc'] and 'HIGH' in v.status.value for k, v in abnormal_results.items())
        anemia = any(k in ['hemoglobin', 'hgb'] and 'LOW' in v.status.value for k, v in abnormal_results.items())
        
        if wbc_high and anemia:
            return "Leukocytosis with anemia - suggest hematologic malignancy or chronic inflammation"
        
        return f"CBC abnormalities: {len(abnormal_results)} parameters affected"
    
    def _generate_hematology_recommendations(
        self, 
        results: List[LabResult], 
        alerts: List[LabAlert]
    ) -> List[str]:
        """Generate hematology-specific recommendations."""
        
        recommendations = []
        
        # Critical alerts
        critical_alerts = [a for a in alerts if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.PANIC]]
        if critical_alerts:
            recommendations.append("URGENT: Immediate hematology consultation recommended")
        
        # Specific recommendations based on results
        hematology_results = [r for r in results if r.test_code.lower() in self.hematology_tests]
        abnormal_hematology = [r for r in hematology_results if r.status != LabResultStatus.NORMAL]
        
        if len(abnormal_hematology) > 1:
            recommendations.append("Consider comprehensive hematology evaluation")
            recommendations.append("Obtain blood smear review")
        
        # Add alert-specific recommendations
        for alert in alerts:
            recommendations.extend(alert.recommendations)
        
        return list(set(recommendations))  # Remove duplicates


class LaboratoryMedicineManager:
    """Manager for laboratory medicine AI analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Initialize analyzers
        self.analyzers = {
            LabTestType.CHEMISTRY: ChemistryAnalyzer(),
            LabTestType.HEMATOLOGY: HematologyAnalyzer(),
        }
        
        # Analysis history
        self.analysis_history = []
        
        # Performance metrics
        self.performance_metrics = {
            'total_analyses': 0,
            'critical_alerts_generated': 0,
            'average_processing_time': 0.0,
            'accuracy_metrics': {}
        }
    
    async def analyze_laboratory_results(
        self, 
        request: LabAnalysisRequest
    ) -> Dict[LabTestType, LabAnalysisResponse]:
        """Analyze laboratory results using appropriate analyzers."""
        
        # Group results by test type
        grouped_results = defaultdict(list)
        for result in request.results:
            # Determine test type based on test code
            test_type = self._determine_test_type(result.test_code)
            grouped_results[test_type].append(result)
        
        responses = {}
        
        # Analyze each group
        for test_type, results in grouped_results.items():
            analyzer = self.analyzers.get(test_type)
            if analyzer:
                type_request = LabAnalysisRequest(
                    patient_id=request.patient_id,
                    results=results,
                    patient_demographics=request.patient_demographics,
                    clinical_context=request.clinical_context,
                    historical_results=[r for r in request.historical_results if self._determine_test_type(r.test_code) == test_type],
                    analysis_parameters=request.analysis_parameters
                )
                
                try:
                    response = await analyzer.analyze_results(type_request)
                    responses[test_type] = response
                    
                    # Update performance metrics
                    self._update_performance_metrics(response)
                    
                except Exception as e:
                    self.logger.error(f"Analysis failed for {test_type}: {e}")
        
        return responses
    
    def _determine_test_type(self, test_code: str) -> LabTestType:
        """Determine test type based on test code."""
        
        test_code = test_code.lower()
        
        # Chemistry tests
        chemistry_codes = {'glu', 'creat', 'na', 'k', 'cl', 'co2', 'bun', 'alt', 'ast', 'alp', 'bili', 'tropi'}
        if test_code in chemistry_codes:
            return LabTestType.CHEMISTRY
        
        # Hematology tests
        hematology_codes = {'wbc', 'hemoglobin', 'hgb', 'hct', 'plt', 'platelet', 'rbc'}
        if test_code in hematology_codes:
            return LabTestType.HEMATOLOGY
        
        # Coagulation tests
        coagulation_codes = {'pt', 'inr', 'ptt', 'aptt'}
        if test_code in coagulation_codes:
            return LabTestType.COAGULATION
        
        # Default to chemistry
        return LabTestType.CHEMISTRY
    
    def generate_comprehensive_report(
        self, 
        analyses: Dict[LabTestType, LabAnalysisResponse]
    ) -> Dict[str, Any]:
        """Generate comprehensive laboratory medicine report."""
        
        report = {
            'patient_id': None,
            'report_timestamp': datetime.utcnow(),
            'summary': {},
            'critical_alerts': [],
            'all_recommendations': [],
            'trends_summary': {},
            'quality_assessment': {},
            'next_steps': []
        }
        
        # Consolidate findings
        for test_type, analysis in analyses.items():
            if not report['patient_id']:
                report['patient_id'] = analysis.patient_id
            
            # Summary by test type
            report['summary'][test_type.value] = {
                'tests_analyzed': len(analysis.analyzed_results),
                'alerts_generated': len(analysis.alerts),
                'critical_values': len(analysis.critical_values),
                'key_findings': list(analysis.interpretations.values())[:3]  # Top 3 findings
            }
            
            # Collect critical alerts
            critical_alerts = [a for a in analysis.alerts if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.PANIC]]
            report['critical_alerts'].extend(critical_alerts)
            
            # Collect recommendations
            report['all_recommendations'].extend(analysis.recommendations)
            
            # Trends summary
            if analysis.trends:
                report['trends_summary'][test_type.value] = analysis.trends
        
        # Overall assessment
        total_critical_alerts = len(report['critical_alerts'])
        if total_critical_alerts > 0:
            report['overall_severity'] = 'Critical'
            report['next_steps'].append(f"URGENT: {total_critical_alerts} critical alerts require immediate attention")
        else:
            warning_alerts = sum(len([a for a in analysis.alerts if a.severity == AlertSeverity.WARNING]) for analysis in analyses.values())
            if warning_alerts > 0:
                report['overall_severity'] = 'Warning'
                report['next_steps'].append(f"Monitor {warning_alerts} abnormal findings")
            else:
                report['overall_severity'] = 'Normal'
                report['next_steps'].append("Continue routine monitoring")
        
        # Remove duplicate recommendations
        report['all_recommendations'] = list(set(report['all_recommendations']))
        
        return report
    
    def get_trend_analysis(
        self, 
        patient_id: str, 
        test_codes: List[str], 
        time_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Get detailed trend analysis for specific tests."""
        
        # This would integrate with historical data systems
        # For now, return simulated trend analysis
        
        trends = {}
        start_time, end_time = time_range
        
        for test_code in test_codes:
            trends[test_code] = {
                'data_points': 10,  # Simulated
                'time_range': {
                    'start': start_time,
                    'end': end_time
                },
                'trend_direction': TrendDirection.STABLE.value,
                'statistical_significance': 0.05,
                'clinical_significance': 'Monitor for changes',
                'recommendations': ['Continue current monitoring frequency']
            }
        
        return {
            'patient_id': patient_id,
            'analysis_timestamp': datetime.utcnow(),
            'trends': trends,
            'overall_assessment': 'Laboratory values show stable trends with no significant changes requiring immediate intervention'
        }
    
    def _update_performance_metrics(self, response: LabAnalysisResponse):
        """Update performance metrics."""
        
        self.performance_metrics['total_analyses'] += 1
        
        critical_alerts = len([a for a in response.alerts if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.PANIC]])
        self.performance_metrics['critical_alerts_generated'] += critical_alerts
        
        # Record analysis in history
        self.analysis_history.append({
            'patient_id': response.patient_id,
            'timestamp': response.analysis_timestamp,
            'tests_analyzed': len(response.analyzed_results),
            'alerts_generated': len(response.alerts),
            'critical_alerts': critical_alerts
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for laboratory analysis."""
        
        return {
            'total_analyses_performed': self.performance_metrics['total_analyses'],
            'total_critical_alerts': self.performance_metrics['critical_alerts_generated'],
            'average_processing_time': self.performance_metrics['average_processing_time'],
            'analysis_history_count': len(self.analysis_history),
            'recent_activity': self.analysis_history[-10:] if self.analysis_history else []
        }


# Factory function
def create_laboratory_medicine_manager(config: Dict[str, Any]) -> LaboratoryMedicineManager:
    """Create laboratory medicine manager with configuration."""
    return LaboratoryMedicineManager(config)