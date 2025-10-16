# Enhanced Clinical Decision Support Agent

## Overview

The Enhanced Clinical Decision Support Agent provides comprehensive clinical intelligence including drug interaction checking, allergy screening, evidence-based recommendations, laboratory value interpretation, and clinical risk assessment.

## Features

### 🔬 Drug Interaction Checking
- Comprehensive drug-drug interaction analysis
- Severity classification (minor, moderate, major, contraindicated)
- Mechanism-based alerts with clinical recommendations
- Support for polypharmacy scenarios

### 🛡️ Allergy Screening
- Cross-reactivity detection for drug allergies
- Alternative medication recommendations
- Severity-based contraindication alerts
- Comprehensive allergy database

### 📋 Clinical Guidelines
- Evidence-based recommendations from major medical societies
- ADA diabetes management guidelines
- AHA/ACC cardiovascular guidelines
- Condition-specific care protocols

### 🧪 Laboratory Interpretation
- Automated lab value interpretation with reference ranges
- Trend analysis and clinical significance assessment
- Critical value alerts and monitoring recommendations
- Support for 20+ common laboratory tests

### ⚡ Risk Assessment
- Cardiovascular risk calculation
- Bleeding risk assessment (HAS-BLED)
- Clinical risk stratification
- Personalized risk management recommendations

## Quick Start

### Basic Usage

```python
from vita_agents.agents.enhanced_clinical_decision_agent import (
    EnhancedClinicalDecisionSupportAgent,
    ClinicalAnalysisRequest
)

# Initialize the agent
agent = EnhancedClinicalDecisionSupportAgent()
await agent._on_start()

# Prepare patient data
patient_data = {
    "age": 65,
    "sex": "male",
    "conditions": ["diabetes", "hypertension"],
    "weight": 80.0
}

medications = ["warfarin", "aspirin", "metformin", "lisinopril"]
allergies = ["penicillin"]
lab_results = [
    {
        "test_name": "glucose",
        "value": 150.0,
        "unit": "mg/dL",
        "reference_range": "70-100",
        "collected_date": "2024-01-15T10:00:00Z",
        "result_date": "2024-01-15T11:00:00Z"
    }
]

# Create comprehensive analysis request
request = ClinicalAnalysisRequest(
    patient_id="patient-123",
    patient_data=patient_data,
    medications=medications,
    allergies=allergies,
    lab_results=lab_results,
    clinical_contexts=["diabetes", "cardiovascular"]
)

# Perform analysis
result = await agent.comprehensive_clinical_analysis(request)

# Access results
print(f"Total alerts: {result['summary']['total_alerts']}")
print(f"Drug interactions: {result['summary']['drug_interaction_count']}")
print(f"Clinical recommendations: {result['summary']['recommendations_count']}")
```

### Drug Interaction Checking

```python
from vita_agents.agents.enhanced_clinical_decision_agent import DrugInteractionRequest

# Check interactions for current medications
request = DrugInteractionRequest(
    patient_id="patient-123",
    medications=["warfarin", "aspirin", "simvastatin"]
)

result = await agent.check_drug_interactions(request)

for interaction in result["interactions"]:
    print(f"⚠️ {interaction['title']}")
    print(f"Severity: {interaction['severity']}")
    print(f"Description: {interaction['description']}")
    print(f"Recommendation: {interaction['recommendation']}")
    print("---")
```

### Check New Medication

```python
# Check if new medication interacts with current medications
request = DrugInteractionRequest(
    patient_id="patient-123",
    medications=["warfarin", "metformin"],
    new_medication="aspirin"
)

result = await agent.check_drug_interactions(request)

if result["interaction_count"] > 0:
    print(f"⚠️ {result['new_medication']} has {result['interaction_count']} interaction(s)")
    for recommendation in result["recommendations"]:
        print(f"📋 {recommendation}")
```

### Allergy Screening

```python
from vita_agents.agents.enhanced_clinical_decision_agent import AllergyScreeningRequest

request = AllergyScreeningRequest(
    patient_id="patient-123",
    patient_allergies=["penicillin", "sulfa"],
    proposed_medications=["amoxicillin", "sulfamethoxazole", "ciprofloxacin"]
)

result = await agent.screen_allergies(request)

print(f"🛡️ Safe medications: {result['safe_medications']}")

for alert in result["allergy_alerts"]:
    print(f"🚨 {alert['title']}")
    print(f"Severity: {alert['severity']}")
    print(f"Recommendation: {alert['recommendation']}")
```

### Laboratory Interpretation

```python
from vita_agents.agents.enhanced_clinical_decision_agent import LabInterpretationRequest

lab_results = [
    {
        "test_name": "hemoglobin_a1c",
        "value": 8.5,
        "unit": "%",
        "reference_range": "<7.0",
        "collected_date": "2024-01-15T10:00:00Z",
        "result_date": "2024-01-15T11:00:00Z"
    },
    {
        "test_name": "ldl_cholesterol",
        "value": 160.0,
        "unit": "mg/dL",
        "reference_range": "<100",
        "collected_date": "2024-01-15T10:00:00Z",
        "result_date": "2024-01-15T11:00:00Z"
    }
]

request = LabInterpretationRequest(
    patient_id="patient-123",
    lab_results=lab_results,
    patient_conditions=["diabetes", "hyperlipidemia"],
    patient_age=65,
    patient_sex="male"
)

result = await agent.interpret_lab_results(request)

for interpretation in result["interpretations"]:
    print(f"🧪 {interpretation['test_name']}: {interpretation['current_value']}")
    print(f"Status: {interpretation['status']}")
    print(f"Clinical significance: {interpretation['clinical_significance']}")
    print(f"Recommendations: {interpretation['recommendations']}")
    print("---")
```

## Integration Examples

### With EHR Systems

```python
class EHRClinicalDecisionWorkflow:
    """Integration workflow with EHR systems."""
    
    def __init__(self):
        self.cds_agent = EnhancedClinicalDecisionSupportAgent()
        self.ehr_connector = EHRConnector()  # Your EHR connector
    
    async def analyze_patient_orders(self, patient_id: str, new_orders: List[str]):
        """Analyze new orders against patient data."""
        
        # Fetch patient data from EHR
        patient_data = await self.ehr_connector.get_patient(patient_id)
        current_medications = await self.ehr_connector.get_medications(patient_id)
        allergies = await self.ehr_connector.get_allergies(patient_id)
        recent_labs = await self.ehr_connector.get_recent_labs(patient_id)
        
        # Prepare analysis request
        request = ClinicalAnalysisRequest(
            patient_id=patient_id,
            patient_data=patient_data,
            medications=current_medications + new_orders,
            allergies=allergies,
            lab_results=recent_labs,
            clinical_contexts=patient_data.get("conditions", [])
        )
        
        # Perform clinical decision support
        analysis = await self.cds_agent.comprehensive_clinical_analysis(request)
        
        # Process alerts and recommendations
        critical_alerts = [
            alert for alert in analysis["drug_interactions"] + analysis["allergy_alerts"]
            if alert["severity"] in ["critical", "high"]
        ]
        
        if critical_alerts:
            # Block order entry and notify physician
            await self.notify_physician(patient_id, critical_alerts)
            return {"status": "blocked", "alerts": critical_alerts}
        
        # Provide recommendations
        return {
            "status": "approved_with_recommendations",
            "analysis": analysis,
            "recommendations": analysis["clinical_guidelines"]
        }
    
    async def notify_physician(self, patient_id: str, alerts: List[Dict]):
        """Send critical alerts to physician."""
        message = f"Critical clinical alerts for patient {patient_id}:\n"
        for alert in alerts:
            message += f"• {alert['title']}: {alert['description']}\n"
        
        # Send notification through EHR system
        await self.ehr_connector.send_alert(patient_id, message)
```

### Real-time Monitoring

```python
class ClinicalMonitoringService:
    """Real-time clinical monitoring with CDS."""
    
    def __init__(self):
        self.cds_agent = EnhancedClinicalDecisionSupportAgent()
        self.monitoring_rules = self._load_monitoring_rules()
    
    async def monitor_patient_continuously(self, patient_id: str):
        """Continuous monitoring with periodic analysis."""
        
        while True:
            try:
                # Get latest patient data
                patient_data = await self.get_current_patient_data(patient_id)
                
                # Check if analysis is needed
                if self._should_analyze(patient_data):
                    analysis = await self._perform_monitoring_analysis(patient_id, patient_data)
                    
                    # Process critical alerts
                    await self._handle_monitoring_alerts(patient_id, analysis)
                
                # Wait before next check
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Monitoring error for patient {patient_id}: {e}")
                await asyncio.sleep(60)  # Retry in 1 minute
    
    async def _perform_monitoring_analysis(self, patient_id: str, patient_data: Dict):
        """Perform focused monitoring analysis."""
        
        request = ClinicalAnalysisRequest(
            patient_id=patient_id,
            patient_data=patient_data,
            medications=patient_data.get("current_medications", []),
            allergies=patient_data.get("allergies", []),
            lab_results=patient_data.get("recent_labs", []),
            clinical_contexts=patient_data.get("conditions", [])
        )
        
        return await self.cds_agent.comprehensive_clinical_analysis(request)
    
    def _should_analyze(self, patient_data: Dict) -> bool:
        """Determine if patient needs analysis."""
        
        # Check for new medications
        if patient_data.get("new_medications_since_last_check"):
            return True
        
        # Check for new lab results
        if patient_data.get("new_labs_since_last_check"):
            return True
        
        # Check for vital sign changes
        vitals = patient_data.get("vital_signs", {})
        if self._vital_signs_concerning(vitals):
            return True
        
        return False
    
    async def _handle_monitoring_alerts(self, patient_id: str, analysis: Dict):
        """Handle alerts from monitoring analysis."""
        
        critical_alerts = [
            alert for alert in analysis.get("drug_interactions", []) + 
                                analysis.get("allergy_alerts", []) +
                                analysis.get("lab_alerts", [])
            if alert.get("severity") == "critical"
        ]
        
        if critical_alerts:
            # Immediate notification
            await self.send_immediate_alert(patient_id, critical_alerts)
        
        high_alerts = [
            alert for alert in analysis.get("drug_interactions", []) + 
                                analysis.get("allergy_alerts", []) +
                                analysis.get("lab_alerts", [])
            if alert.get("severity") == "high"
        ]
        
        if high_alerts:
            # Schedule physician review
            await self.schedule_physician_review(patient_id, high_alerts)
```

### Task-based Integration

```python
from vita_agents.core.agent import TaskRequest

class ClinicalTaskProcessor:
    """Process clinical decision support tasks."""
    
    def __init__(self):
        self.cds_agent = EnhancedClinicalDecisionSupportAgent()
    
    async def process_clinical_task(self, task_data: Dict) -> Dict:
        """Process different types of clinical tasks."""
        
        task_type = task_data["type"]
        
        if task_type == "medication_review":
            return await self._process_medication_review(task_data)
        
        elif task_type == "lab_review":
            return await self._process_lab_review(task_data)
        
        elif task_type == "allergy_check":
            return await self._process_allergy_check(task_data)
        
        elif task_type == "comprehensive_assessment":
            return await self._process_comprehensive_assessment(task_data)
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _process_medication_review(self, task_data: Dict) -> Dict:
        """Process medication review task."""
        
        task = TaskRequest(
            task_id=task_data["task_id"],
            task_type="drug_interactions",
            parameters={
                "patient_id": task_data["patient_id"],
                "medications": task_data["medications"]
            }
        )
        
        response = await self.cds_agent.execute_task(task)
        
        return {
            "task_id": task_data["task_id"],
            "status": response.status,
            "result": response.result,
            "recommendations": response.result.get("recommendations", []) if response.result else []
        }
    
    async def _process_comprehensive_assessment(self, task_data: Dict) -> Dict:
        """Process comprehensive clinical assessment."""
        
        task = TaskRequest(
            task_id=task_data["task_id"],
            task_type="comprehensive_analysis",
            parameters=task_data["parameters"]
        )
        
        response = await self.cds_agent.execute_task(task)
        
        if response.status == "completed":
            # Extract key findings
            result = response.result
            key_findings = {
                "critical_interactions": len([
                    i for i in result.get("drug_interactions", [])
                    if i.get("severity") == "critical"
                ]),
                "allergy_contraindications": len([
                    a for a in result.get("allergy_alerts", [])
                    if a.get("severity") == "critical"
                ]),
                "abnormal_labs": len(result.get("abnormal_values", [])),
                "clinical_recommendations": len(result.get("clinical_guidelines", []))
            }
            
            return {
                "task_id": task_data["task_id"],
                "status": "completed",
                "key_findings": key_findings,
                "full_analysis": result
            }
        
        else:
            return {
                "task_id": task_data["task_id"],
                "status": "failed",
                "error": response.error
            }
```

## Configuration

### Agent Configuration

```python
config = {
    "cache_ttl_hours": 1,
    "max_cache_size": 1000,
    "enable_detailed_logging": True,
    "interaction_database_version": "2024.1",
    "allergy_database_version": "2024.1",
    "guidelines_version": "2024.1"
}

agent = EnhancedClinicalDecisionSupportAgent(config=config)
```

### Custom Alert Thresholds

```python
# Customize alert thresholds in clinical decision support system
agent.cds_system.configure_alert_thresholds({
    "drug_interaction_min_severity": "moderate",
    "allergy_cross_reactivity_threshold": 0.7,
    "lab_critical_value_multiplier": 2.0,
    "risk_assessment_high_threshold": 20.0
})
```

## Performance Considerations

### Caching Strategy
- Analysis results are cached for 1 hour by default
- Cache key includes patient ID and request parameters hash
- Automatic cache cleanup and size management

### Async Processing
- All major operations are asynchronous
- Parallel processing for multiple patients
- Non-blocking alert generation

### Resource Management
- Memory-efficient data structures
- Lazy loading of clinical databases
- Configurable resource limits

## Error Handling

The agent provides comprehensive error handling:

```python
try:
    result = await agent.comprehensive_clinical_analysis(request)
except ValueError as e:
    # Invalid request parameters
    logger.error(f"Invalid request: {e}")
except TimeoutError as e:
    # Analysis timeout
    logger.error(f"Analysis timeout: {e}")
except Exception as e:
    # Unexpected error
    logger.error(f"Unexpected error: {e}")
```

## Monitoring and Metrics

```python
# Get agent performance metrics
metrics = await agent.get_agent_metrics()

print(f"Analyses performed: {metrics['agent_metrics']['analyses_performed']}")
print(f"Alerts generated: {metrics['agent_metrics']['alerts_generated']}")
print(f"Cache hit rate: {metrics['system_statistics']['cache_hit_rate']}")
print(f"Average processing time: {metrics['system_statistics']['avg_processing_time_ms']}ms")
```

## Contributing

When extending the Enhanced Clinical Decision Support Agent:

1. **Add new clinical algorithms** to the `clinical_decision_support` package
2. **Update test cases** for new functionality
3. **Document clinical evidence** and validation data
4. **Follow medical coding standards** (ICD-10, SNOMED CT, LOINC)
5. **Ensure HIPAA compliance** for all patient data handling

## Medical Disclaimer

This clinical decision support system is intended to assist healthcare professionals and should not replace clinical judgment. All recommendations should be validated against current medical literature and institutional protocols.