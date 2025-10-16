"""
Enhanced Clinical Decision Support Agent for comprehensive healthcare analytics.

This agent provides advanced clinical decision support capabilities including:
- Drug interaction checking with severity classification
- Allergy screening and contraindication detection  
- Evidence-based clinical guideline recommendations
- Laboratory value interpretation and alerts
- Clinical risk assessment and scoring
- Real-time clinical monitoring and alerts
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import structlog
from pydantic import BaseModel, Field

from vita_agents.core.agent import HealthcareAgent, AgentCapability, TaskRequest, TaskResponse
from vita_agents.core.config import get_settings
from vita_agents.clinical_decision_support import (
    AdvancedClinicalDecisionSupport,
    clinical_decision_support,
    AlertSeverity,
    EvidenceLevel,
    ClinicalAlert
)
from vita_agents.clinical_decision_support.lab_interpreter import (
    LabValueInterpreter,
    LabResult,
    LabInterpretation
)


logger = structlog.get_logger(__name__)


class ClinicalAnalysisRequest(BaseModel):
    """Request for comprehensive clinical analysis."""
    
    patient_id: str
    patient_data: Dict[str, Any]
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    lab_results: List[Dict[str, Any]] = Field(default_factory=list)
    clinical_contexts: List[str] = Field(default_factory=list)
    include_lab_interpretation: bool = True
    include_risk_assessment: bool = True
    include_guidelines: bool = True


class DrugInteractionRequest(BaseModel):
    """Request for drug interaction analysis."""
    
    patient_id: str
    medications: List[str]
    new_medication: Optional[str] = None


class AllergyScreeningRequest(BaseModel):
    """Request for allergy screening."""
    
    patient_id: str
    patient_allergies: List[str]
    proposed_medications: List[str]


class LabInterpretationRequest(BaseModel):
    """Request for laboratory value interpretation."""
    
    patient_id: str
    lab_results: List[Dict[str, Any]]
    patient_conditions: List[str] = Field(default_factory=list)
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None


class EnhancedClinicalDecisionSupportAgent(HealthcareAgent):
    """
    Enhanced Clinical Decision Support Agent.
    
    Provides comprehensive clinical decision support including drug interactions,
    allergy screening, evidence-based recommendations, lab interpretation,
    and clinical risk assessment.
    """
    
    def __init__(
        self,
        agent_id: str = "enhanced-clinical-decision-support-agent",
        name: str = "Enhanced Clinical Decision Support Agent",
        description: str = "Advanced clinical decision support with comprehensive analytics",
        version: str = "2.0.0",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            version=version,
            capabilities=[
                AgentCapability(
                    name="DRUG_INTERACTION_CHECKING",
                    description="Comprehensive drug interaction analysis with severity classification",
                    input_schema={"type": "object", "properties": {"medications": {"type": "array"}}},
                    output_schema={"type": "object", "properties": {"interactions": {"type": "array"}}},
                    supported_formats=["FHIR_R4", "RxNorm"],
                    requirements=["Drug interaction database", "Clinical pharmacology"]
                ),
                AgentCapability(
                    name="ALLERGY_SCREENING",
                    description="Allergy screening with cross-reactivity detection",
                    input_schema={"type": "object", "properties": {"allergies": {"type": "array"}}},
                    output_schema={"type": "object", "properties": {"alerts": {"type": "array"}}},
                    supported_formats=["FHIR_R4", "SNOMED_CT"],
                    requirements=["Allergy database", "Cross-reactivity mappings"]
                ),
                AgentCapability(
                    name="CLINICAL_GUIDELINES",
                    description="Evidence-based clinical guideline recommendations",
                    input_schema={"type": "object", "properties": {"clinical_context": {"type": "string"}}},
                    output_schema={"type": "object", "properties": {"recommendations": {"type": "array"}}},
                    supported_formats=["FHIR_R4", "ClinicalDocument"],
                    requirements=["Clinical guidelines database", "Evidence levels"]
                ),
                AgentCapability(
                    name="LAB_INTERPRETATION",
                    description="Laboratory value interpretation with clinical alerts",
                    input_schema={"type": "object", "properties": {"lab_results": {"type": "array"}}},
                    output_schema={"type": "object", "properties": {"interpretations": {"type": "array"}}},
                    supported_formats=["FHIR_R4", "HL7_v2.5", "LOINC"],
                    requirements=["Reference ranges", "Clinical interpretation rules"]
                ),
                AgentCapability(
                    name="RISK_ASSESSMENT",
                    description="Clinical risk assessment and scoring",
                    input_schema={"type": "object", "properties": {"patient_data": {"type": "object"}}},
                    output_schema={"type": "object", "properties": {"risk_scores": {"type": "array"}}},
                    supported_formats=["FHIR_R4"],
                    requirements=["Risk calculation algorithms", "Clinical models"]
                ),
            ],
            config=config,
        )
        
        # Initialize clinical decision support systems
        self.cds_system = clinical_decision_support
        self.lab_interpreter = LabValueInterpreter()
        
        # Performance tracking
        self.analysis_count = 0
        self.alert_count = 0
        self.recommendation_count = 0
        
        # Cache for recent analyses
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(hours=1)
    
    async def _on_start(self) -> None:
        """Hook called when the agent starts."""
        logger.info("Enhanced Clinical Decision Support Agent starting...")
        # Initialize any required resources
    
    async def _on_stop(self) -> None:
        """Hook called when the agent stops."""
        logger.info("Enhanced Clinical Decision Support Agent stopping...")
        # Clean up resources
    
    async def comprehensive_clinical_analysis(
        self,
        request: ClinicalAnalysisRequest
    ) -> Dict[str, Any]:
        """
        Perform comprehensive clinical decision support analysis.
        
        Args:
            request: Clinical analysis request
            
        Returns:
            Comprehensive analysis results
        """
        start_time = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = f"{request.patient_id}_{hash(str(request.dict()))}"
            if cache_key in self.analysis_cache:
                cached_result = self.analysis_cache[cache_key]
                if datetime.utcnow() - cached_result["timestamp"] < self.cache_ttl:
                    logger.info(f"Returning cached analysis for patient {request.patient_id}")
                    return cached_result["result"]
            
            # Perform comprehensive analysis
            results = await self.cds_system.comprehensive_analysis(
                patient_data=request.patient_data,
                medications=request.medications,
                allergies=request.allergies,
                clinical_contexts=request.clinical_contexts,
                patient_id=request.patient_id
            )
            
            # Add lab interpretation if requested and lab results provided
            if request.include_lab_interpretation and request.lab_results:
                lab_analysis = await self._interpret_lab_results(
                    request.lab_results,
                    request.patient_id,
                    request.patient_data.get("conditions", []),
                    request.patient_data.get("age"),
                    request.patient_data.get("sex")
                )
                results["lab_interpretations"] = lab_analysis["interpretations"]
                results["lab_alerts"] = lab_analysis["alerts"]
            
            # Add enhanced summary
            results["analysis_metadata"] = {
                "analysis_timestamp": start_time.isoformat(),
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "agent_version": self.version,
                "analysis_id": f"cds_{request.patient_id}_{start_time.timestamp()}"
            }
            
            # Update metrics
            self.analysis_count += 1
            self.alert_count += results["summary"]["total_alerts"]
            self.recommendation_count += results["summary"]["recommendations_count"]
            
            # Cache results
            self.analysis_cache[cache_key] = {
                "result": results,
                "timestamp": datetime.utcnow()
            }
            
            logger.info(
                f"Comprehensive clinical analysis completed for patient {request.patient_id}",
                total_alerts=results["summary"]["total_alerts"],
                critical_alerts=results["summary"]["critical_alerts"],
                processing_time_ms=results["analysis_metadata"]["processing_time_ms"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive clinical analysis: {e}")
            raise
    
    async def check_drug_interactions(
        self,
        request: DrugInteractionRequest
    ) -> Dict[str, Any]:
        """
        Check for drug interactions.
        
        Args:
            request: Drug interaction request
            
        Returns:
            Drug interaction analysis results
        """
        medications = request.medications.copy()
        if request.new_medication:
            medications.append(request.new_medication)
        
        alerts = await self.cds_system.drug_checker.check_interactions(
            medications, request.patient_id
        )
        
        # Focus on interactions involving the new medication if specified
        if request.new_medication:
            new_med_alerts = [
                alert for alert in alerts
                if request.new_medication.lower() in alert.triggered_by.lower()
            ]
            return {
                "patient_id": request.patient_id,
                "new_medication": request.new_medication,
                "interaction_count": len(new_med_alerts),
                "interactions": [self._alert_to_dict(alert) for alert in new_med_alerts],
                "recommendations": self._generate_interaction_recommendations(new_med_alerts)
            }
        
        return {
            "patient_id": request.patient_id,
            "medications": request.medications,
            "interaction_count": len(alerts),
            "interactions": [self._alert_to_dict(alert) for alert in alerts],
            "recommendations": self._generate_interaction_recommendations(alerts)
        }
    
    async def screen_allergies(
        self,
        request: AllergyScreeningRequest
    ) -> Dict[str, Any]:
        """
        Screen for allergy contraindications.
        
        Args:
            request: Allergy screening request
            
        Returns:
            Allergy screening results
        """
        alerts = await self.cds_system.allergy_screener.screen_allergies(
            request.patient_allergies,
            request.proposed_medications,
            request.patient_id
        )
        
        return {
            "patient_id": request.patient_id,
            "patient_allergies": request.patient_allergies,
            "proposed_medications": request.proposed_medications,
            "allergy_alert_count": len(alerts),
            "allergy_alerts": [self._alert_to_dict(alert) for alert in alerts],
            "safe_medications": self._identify_safe_medications(
                request.proposed_medications, alerts
            ),
            "recommendations": self._generate_allergy_recommendations(alerts)
        }
    
    async def interpret_lab_results(
        self,
        request: LabInterpretationRequest
    ) -> Dict[str, Any]:
        """
        Interpret laboratory results.
        
        Args:
            request: Lab interpretation request
            
        Returns:
            Lab interpretation results
        """
        return await self._interpret_lab_results(
            request.lab_results,
            request.patient_id,
            request.patient_conditions,
            request.patient_age,
            request.patient_sex
        )
    
    async def _interpret_lab_results(
        self,
        lab_results: List[Dict[str, Any]],
        patient_id: str,
        patient_conditions: List[str] = None,
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None
    ) -> Dict[str, Any]:
        """Internal method to interpret lab results."""
        interpretations = []
        
        for lab_data in lab_results:
            # Convert to LabResult object
            lab_result = LabResult(
                test_name=lab_data["test_name"],
                value=float(lab_data["value"]),
                unit=lab_data.get("unit", ""),
                reference_range=lab_data.get("reference_range", ""),
                collected_date=datetime.fromisoformat(lab_data.get("collected_date", datetime.utcnow().isoformat())),
                result_date=datetime.fromisoformat(lab_data.get("result_date", datetime.utcnow().isoformat())),
                patient_id=patient_id,
                previous_value=lab_data.get("previous_value")
            )
            
            # Interpret the lab value
            interpretation = await self.lab_interpreter.interpret_lab_value(
                lab_result, patient_conditions, patient_age, patient_sex
            )
            interpretations.append(interpretation)
        
        # Generate alerts for abnormal values
        alerts = await self.lab_interpreter.generate_lab_alerts(
            interpretations, patient_id
        )
        
        return {
            "patient_id": patient_id,
            "interpretation_count": len(interpretations),
            "interpretations": [self._interpretation_to_dict(interp) for interp in interpretations],
            "lab_alert_count": len(alerts),
            "alerts": [self._alert_to_dict(alert) for alert in alerts],
            "critical_values": [
                interp for interp in interpretations
                if interp.status.value in ["critical_low", "critical_high"]
            ],
            "abnormal_values": [
                interp for interp in interpretations  
                if interp.status.value in ["low", "high", "critical_low", "critical_high"]
            ]
        }
    
    def _alert_to_dict(self, alert: ClinicalAlert) -> Dict[str, Any]:
        """Convert ClinicalAlert to dictionary."""
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
    
    def _interpretation_to_dict(self, interpretation: LabInterpretation) -> Dict[str, Any]:
        """Convert LabInterpretation to dictionary."""
        return {
            "test_name": interpretation.test_name,
            "current_value": interpretation.current_value,
            "status": interpretation.status.value,
            "clinical_significance": interpretation.clinical_significance,
            "recommendations": interpretation.recommendations,
            "monitoring_frequency": interpretation.monitoring_frequency,
            "follow_up_tests": interpretation.follow_up_tests,
            "trend_analysis": interpretation.trend_analysis
        }
    
    def _generate_interaction_recommendations(
        self,
        alerts: List[ClinicalAlert]
    ) -> List[str]:
        """Generate overall recommendations for drug interactions."""
        recommendations = []
        
        critical_interactions = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        high_interactions = [a for a in alerts if a.severity == AlertSeverity.HIGH]
        
        if critical_interactions:
            recommendations.append(
                f"URGENT: {len(critical_interactions)} critical drug interaction(s) detected. "
                "Immediate medication review required."
            )
        
        if high_interactions:
            recommendations.append(
                f"WARNING: {len(high_interactions)} significant drug interaction(s) detected. "
                "Close monitoring recommended."
            )
        
        if alerts:
            recommendations.append(
                "Review all drug interactions with clinical pharmacist or physician."
            )
            recommendations.append(
                "Consider alternative medications or dose adjustments as clinically appropriate."
            )
        
        return recommendations
    
    def _generate_allergy_recommendations(
        self,
        alerts: List[ClinicalAlert]
    ) -> List[str]:
        """Generate overall recommendations for allergy alerts."""
        recommendations = []
        
        if alerts:
            critical_allergies = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
            
            if critical_allergies:
                recommendations.append(
                    f"CONTRAINDICATED: {len(critical_allergies)} medication(s) are contraindicated "
                    "due to known allergies. Do not administer."
                )
            
            recommendations.extend([
                "Verify all patient allergies and review allergy history.",
                "Consider allergy consultation for complex cases.",
                "Update patient allergy list and ensure proper documentation.",
                "Educate patient about allergic reactions and alternative medications."
            ])
        
        return recommendations
    
    def _identify_safe_medications(
        self,
        proposed_medications: List[str],
        alerts: List[ClinicalAlert]
    ) -> List[str]:
        """Identify medications that are safe (no allergy alerts)."""
        alerted_medications = set()
        
        for alert in alerts:
            # Extract medication name from triggered_by field
            triggered_med = alert.triggered_by.lower()
            alerted_medications.add(triggered_med)
        
        safe_medications = []
        for med in proposed_medications:
            if med.lower() not in alerted_medications:
                safe_medications.append(med)
        
        return safe_medications
    
    async def get_patient_alert_summary(
        self,
        patient_id: str,
        severity_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get summary of active alerts for a patient.
        
        Args:
            patient_id: Patient identifier
            severity_filter: Optional severity filter
            
        Returns:
            Patient alert summary
        """
        severity_enum = None
        if severity_filter:
            try:
                severity_enum = AlertSeverity(severity_filter.lower())
            except ValueError:
                pass
        
        alerts = await self.cds_system.get_patient_alerts(
            patient_id, severity_enum
        )
        
        # Group alerts by type
        alert_groups = {}
        for alert in alerts:
            alert_type = alert["alert_type"]
            if alert_type not in alert_groups:
                alert_groups[alert_type] = []
            alert_groups[alert_type].append(alert)
        
        return {
            "patient_id": patient_id,
            "total_alerts": len(alerts),
            "alert_groups": alert_groups,
            "severity_distribution": self._get_severity_distribution(alerts),
            "most_recent_alert": max(alerts, key=lambda x: x["created_at"]) if alerts else None,
            "summary_timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_severity_distribution(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of alerts by severity."""
        distribution = {}
        for alert in alerts:
            severity = alert["severity"]
            distribution[severity] = distribution.get(severity, 0) + 1
        return distribution
    
    async def execute_task(self, task: TaskRequest) -> TaskResponse:
        """
        Execute a clinical decision support task.
        
        Args:
            task: Task request
            
        Returns:
            Task response with results
        """
        task_type = task.task_type.lower()
        
        try:
            if task_type == "comprehensive_analysis":
                request = ClinicalAnalysisRequest(**task.parameters)
                result = await self.comprehensive_clinical_analysis(request)
                
                return TaskResponse(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status="completed",
                    result=result,
                    metadata={"analysis_type": "comprehensive", "patient_id": request.patient_id}
                )
            
            elif task_type == "drug_interactions":
                request = DrugInteractionRequest(**task.parameters)
                result = await self.check_drug_interactions(request)
                
                return TaskResponse(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status="completed",
                    result=result,
                    metadata={"analysis_type": "drug_interactions", "patient_id": request.patient_id}
                )
            
            elif task_type == "allergy_screening":
                request = AllergyScreeningRequest(**task.parameters)
                result = await self.screen_allergies(request)
                
                return TaskResponse(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status="completed",
                    result=result,
                    metadata={"analysis_type": "allergy_screening", "patient_id": request.patient_id}
                )
            
            elif task_type == "lab_interpretation":
                request = LabInterpretationRequest(**task.parameters)
                result = await self.interpret_lab_results(request)
                
                return TaskResponse(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status="completed",
                    result=result,
                    metadata={"analysis_type": "lab_interpretation", "patient_id": request.patient_id}
                )
            
            elif task_type == "patient_alerts":
                patient_id = task.parameters.get("patient_id")
                severity_filter = task.parameters.get("severity_filter")
                result = await self.get_patient_alert_summary(patient_id, severity_filter)
                
                return TaskResponse(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status="completed",
                    result=result,
                    metadata={"analysis_type": "patient_alerts", "patient_id": patient_id}
                )
            
            else:
                return TaskResponse(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status="failed",
                    error=f"Unknown task type: {task_type}",
                    metadata={
                        "supported_tasks": [
                            "comprehensive_analysis", "drug_interactions", 
                            "allergy_screening", "lab_interpretation", "patient_alerts"
                        ]
                    }
                )
                
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return TaskResponse(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status="failed",
                error=str(e),
                metadata={"task_type": task_type}
            )
    
    async def get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        system_stats = await self.cds_system.get_system_statistics()
        
        return {
            "agent_metrics": {
                "analyses_performed": self.analysis_count,
                "alerts_generated": self.alert_count,
                "recommendations_made": self.recommendation_count,
                "cache_size": len(self.analysis_cache),
                "uptime_hours": (datetime.utcnow() - self._start_time).total_seconds() / 3600 if hasattr(self, '_start_time') else 0
            },
            "system_statistics": system_stats,
            "capabilities": [cap.name for cap in self.capabilities],
            "version": self.version,
            "last_updated": datetime.utcnow().isoformat()
        }


# Maintain backwards compatibility
ClinicalDecisionSupportAgent = EnhancedClinicalDecisionSupportAgent