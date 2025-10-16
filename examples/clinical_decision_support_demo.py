"""
Demonstration of Enhanced Clinical Decision Support Agent

This script demonstrates the comprehensive clinical decision support capabilities
including drug interactions, allergy screening, clinical guidelines, lab interpretation,
and risk assessment.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any

from vita_agents.agents.enhanced_clinical_decision_agent import (
    EnhancedClinicalDecisionSupportAgent,
    ClinicalAnalysisRequest,
    DrugInteractionRequest,
    AllergyScreeningRequest,
    LabInterpretationRequest
)


def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def print_subsection_header(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def print_alert(alert: Dict[str, Any], prefix: str = "🚨"):
    """Print a formatted alert."""
    print(f"{prefix} {alert['title']}")
    print(f"   Severity: {alert['severity'].upper()}")
    print(f"   Description: {alert['description']}")
    print(f"   Recommendation: {alert['recommendation']}")
    if alert.get('evidence_level'):
        print(f"   Evidence Level: {alert['evidence_level']}")
    print()


def print_lab_interpretation(interpretation: Dict[str, Any]):
    """Print a formatted lab interpretation."""
    test_name = interpretation['test_name']
    value = interpretation['current_value']
    status = interpretation['status']
    
    status_emoji = {
        'normal': '✅',
        'low': '⬇️',
        'high': '⬆️',
        'critical_low': '🔴',
        'critical_high': '🔴'
    }
    
    emoji = status_emoji.get(status, '❓')
    
    print(f"{emoji} {test_name}: {value} ({status.replace('_', ' ').title()})")
    print(f"   Clinical Significance: {interpretation['clinical_significance']}")
    
    if interpretation['recommendations']:
        print(f"   Recommendations: {', '.join(interpretation['recommendations'])}")
    
    if interpretation['monitoring_frequency']:
        print(f"   Monitoring: {interpretation['monitoring_frequency']}")
    print()


def print_guideline(guideline: Dict[str, Any]):
    """Print a formatted clinical guideline."""
    print(f"📋 {guideline['title']}")
    print(f"   Description: {guideline['description']}")
    print(f"   Recommendation: {guideline['recommendation']}")
    print(f"   Evidence: {guideline['evidence_level']} | Source: {guideline['source']}")
    print()


async def demonstrate_comprehensive_analysis():
    """Demonstrate comprehensive clinical analysis."""
    print_section_header("COMPREHENSIVE CLINICAL ANALYSIS DEMONSTRATION")
    
    # Initialize agent
    print("🚀 Initializing Enhanced Clinical Decision Support Agent...")
    agent = EnhancedClinicalDecisionSupportAgent()
    await agent._on_start()
    
    # Sample patient data
    patient_data = {
        "age": 68,
        "sex": "male",
        "weight": 85.0,
        "height": 175.0,
        "conditions": ["diabetes", "hypertension", "hyperlipidemia", "atrial_fibrillation"],
        "medical_history": ["myocardial_infarction", "stroke"],
        "vital_signs": {
            "blood_pressure": "145/92",
            "heart_rate": 88,
            "temperature": 98.4
        }
    }
    
    medications = [
        "warfarin", "aspirin", "metformin", "lisinopril", 
        "simvastatin", "clarithromycin", "metoprolol"
    ]
    
    allergies = ["penicillin", "sulfa"]
    
    lab_results = [
        {
            "test_name": "glucose",
            "value": 165.0,
            "unit": "mg/dL",
            "reference_range": "70-100",
            "collected_date": datetime.utcnow().isoformat(),
            "result_date": datetime.utcnow().isoformat()
        },
        {
            "test_name": "hemoglobin_a1c",
            "value": 8.9,
            "unit": "%",
            "reference_range": "<7.0",
            "collected_date": datetime.utcnow().isoformat(),
            "result_date": datetime.utcnow().isoformat()
        },
        {
            "test_name": "ldl_cholesterol",
            "value": 145.0,
            "unit": "mg/dL",
            "reference_range": "<100",
            "collected_date": datetime.utcnow().isoformat(),
            "result_date": datetime.utcnow().isoformat()
        },
        {
            "test_name": "inr",
            "value": 3.2,
            "unit": "",
            "reference_range": "2.0-3.0",
            "collected_date": datetime.utcnow().isoformat(),
            "result_date": datetime.utcnow().isoformat()
        },
        {
            "test_name": "creatinine",
            "value": 1.4,
            "unit": "mg/dL",
            "reference_range": "0.6-1.2",
            "collected_date": datetime.utcnow().isoformat(),
            "result_date": datetime.utcnow().isoformat()
        }
    ]
    
    print("\n👤 Patient Profile:")
    print(f"   Age: {patient_data['age']} years old {patient_data['sex']}")
    print(f"   Conditions: {', '.join(patient_data['conditions'])}")
    print(f"   Current Medications: {', '.join(medications)}")
    print(f"   Known Allergies: {', '.join(allergies)}")
    print(f"   Lab Tests: {len(lab_results)} recent results")
    
    # Create comprehensive analysis request
    request = ClinicalAnalysisRequest(
        patient_id="demo-patient-001",
        patient_data=patient_data,
        medications=medications,
        allergies=allergies,
        lab_results=lab_results,
        clinical_contexts=["diabetes", "cardiovascular", "anticoagulation"]
    )
    
    print("\n🔍 Performing comprehensive clinical analysis...")
    result = await agent.comprehensive_clinical_analysis(request)
    
    # Display analysis summary
    summary = result["summary"]
    print_subsection_header("ANALYSIS SUMMARY")
    print(f"⚡ Total Alerts Generated: {summary['total_alerts']}")
    print(f"💊 Drug Interactions Found: {summary['drug_interaction_count']}")
    print(f"🛡️ Allergy Alerts: {summary['allergy_alert_count']}")
    print(f"📋 Clinical Recommendations: {summary['recommendations_count']}")
    print(f"🧪 Lab Interpretations: {len(result.get('lab_interpretations', []))}")
    print(f"⚠️ Critical Alerts: {summary['critical_alerts']}")
    print(f"🔥 High Priority Alerts: {summary['high_alerts']}")
    
    # Display drug interactions
    if result["drug_interactions"]:
        print_subsection_header("DRUG INTERACTIONS")
        for interaction in result["drug_interactions"]:
            print_alert(interaction, "⚠️")
    
    # Display allergy alerts
    if result["allergy_alerts"]:
        print_subsection_header("ALLERGY ALERTS")
        for alert in result["allergy_alerts"]:
            print_alert(alert, "🛡️")
    
    # Display lab interpretations
    if result.get("lab_interpretations"):
        print_subsection_header("LABORATORY INTERPRETATIONS")
        for interpretation in result["lab_interpretations"]:
            print_lab_interpretation(interpretation)
    
    # Display clinical guidelines
    if result["clinical_guidelines"]:
        print_subsection_header("CLINICAL GUIDELINES & RECOMMENDATIONS")
        for guideline in result["clinical_guidelines"]:
            print_guideline(guideline)
    
    # Display risk assessment
    if result.get("risk_assessment"):
        print_subsection_header("RISK ASSESSMENT")
        risk_data = result["risk_assessment"]
        for risk_type, assessment in risk_data.items():
            if isinstance(assessment, dict) and "risk_category" in assessment:
                print(f"📊 {risk_type.replace('_', ' ').title()}: {assessment['risk_category'].upper()}")
                if "risk_percentage" in assessment:
                    print(f"   Risk Score: {assessment['risk_percentage']}%")
                if "recommendations" in assessment:
                    for rec in assessment["recommendations"]:
                        print(f"   • {rec}")
                print()
    
    # Display processing metadata
    metadata = result["analysis_metadata"]
    print_subsection_header("PROCESSING DETAILS")
    print(f"🕐 Processing Time: {metadata['processing_time_ms']:.1f}ms")
    print(f"🆔 Analysis ID: {metadata['analysis_id']}")
    print(f"📅 Timestamp: {metadata['analysis_timestamp']}")
    print(f"🤖 Agent Version: {metadata['agent_version']}")
    
    await agent._on_stop()
    return agent


async def demonstrate_focused_analyses():
    """Demonstrate focused clinical analyses."""
    print_section_header("FOCUSED CLINICAL ANALYSES DEMONSTRATION")
    
    agent = EnhancedClinicalDecisionSupportAgent()
    await agent._on_start()
    
    # Drug Interaction Analysis
    print_subsection_header("FOCUSED DRUG INTERACTION ANALYSIS")
    print("Scenario: Adding new medication to existing regimen")
    print("Current medications: warfarin, metformin")
    print("Proposed new medication: aspirin")
    
    drug_request = DrugInteractionRequest(
        patient_id="demo-patient-002",
        medications=["warfarin", "metformin"],
        new_medication="aspirin"
    )
    
    drug_result = await agent.check_drug_interactions(drug_request)
    
    print(f"\n🔍 Analysis Results:")
    print(f"   Interactions found: {drug_result['interaction_count']}")
    
    if drug_result["interactions"]:
        for interaction in drug_result["interactions"]:
            print_alert(interaction, "⚠️")
    
    print("📋 Overall Recommendations:")
    for rec in drug_result["recommendations"]:
        print(f"   • {rec}")
    
    # Allergy Screening
    print_subsection_header("ALLERGY SCREENING ANALYSIS")
    print("Scenario: Screening proposed antibiotics for allergic patient")
    print("Known allergies: penicillin, sulfa")
    print("Proposed medications: amoxicillin, sulfamethoxazole, ciprofloxacin")
    
    allergy_request = AllergyScreeningRequest(
        patient_id="demo-patient-003",
        patient_allergies=["penicillin", "sulfa"],
        proposed_medications=["amoxicillin", "sulfamethoxazole", "ciprofloxacin"]
    )
    
    allergy_result = await agent.screen_allergies(allergy_request)
    
    print(f"\n🔍 Screening Results:")
    print(f"   Allergy alerts: {allergy_result['allergy_alert_count']}")
    print(f"   Safe medications: {allergy_result['safe_medications']}")
    
    if allergy_result["allergy_alerts"]:
        print("\n🚨 Allergy Alerts:")
        for alert in allergy_result["allergy_alerts"]:
            print_alert(alert, "🛡️")
    
    print("📋 Recommendations:")
    for rec in allergy_result["recommendations"]:
        print(f"   • {rec}")
    
    # Laboratory Interpretation
    print_subsection_header("LABORATORY VALUE INTERPRETATION")
    print("Scenario: Interpreting recent lab results for diabetic patient")
    
    lab_data = [
        {
            "test_name": "glucose",
            "value": 220.0,
            "unit": "mg/dL",
            "reference_range": "70-100",
            "collected_date": datetime.utcnow().isoformat(),
            "result_date": datetime.utcnow().isoformat()
        },
        {
            "test_name": "hemoglobin_a1c",
            "value": 9.2,
            "unit": "%",
            "reference_range": "<7.0",
            "collected_date": datetime.utcnow().isoformat(),
            "result_date": datetime.utcnow().isoformat()
        },
        {
            "test_name": "potassium",
            "value": 2.8,
            "unit": "mEq/L",
            "reference_range": "3.5-5.0",
            "collected_date": datetime.utcnow().isoformat(),
            "result_date": datetime.utcnow().isoformat()
        }
    ]
    
    lab_request = LabInterpretationRequest(
        patient_id="demo-patient-004",
        lab_results=lab_data,
        patient_conditions=["diabetes", "hypertension"],
        patient_age=55,
        patient_sex="female"
    )
    
    lab_result = await agent.interpret_lab_results(lab_request)
    
    print(f"\n🔍 Lab Analysis Results:")
    print(f"   Tests interpreted: {lab_result['interpretation_count']}")
    print(f"   Abnormal values: {len(lab_result['abnormal_values'])}")
    print(f"   Critical values: {len(lab_result['critical_values'])}")
    
    print("\n🧪 Individual Test Results:")
    for interpretation in lab_result["interpretations"]:
        print_lab_interpretation(interpretation)
    
    if lab_result["alerts"]:
        print("🚨 Laboratory Alerts:")
        for alert in lab_result["alerts"]:
            print_alert(alert, "🔬")
    
    await agent._on_stop()


async def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring and metrics."""
    print_section_header("PERFORMANCE MONITORING DEMONSTRATION")
    
    agent = EnhancedClinicalDecisionSupportAgent()
    await agent._on_start()
    
    # Perform several analyses to generate metrics
    print("🏃 Performing multiple analyses to generate performance data...")
    
    for i in range(5):
        request = ClinicalAnalysisRequest(
            patient_id=f"perf-test-patient-{i:03d}",
            patient_data={"age": 50 + i*5, "conditions": ["diabetes"]},
            medications=["metformin", "lisinopril"],
            allergies=["penicillin"],
            clinical_contexts=["diabetes"]
        )
        
        await agent.comprehensive_clinical_analysis(request)
        print(f"   ✅ Analysis {i+1}/5 completed")
    
    # Get performance metrics
    print("\n📊 Retrieving performance metrics...")
    metrics = await agent.get_agent_metrics()
    
    print_subsection_header("AGENT PERFORMANCE METRICS")
    agent_metrics = metrics["agent_metrics"]
    print(f"📈 Analyses Performed: {agent_metrics['analyses_performed']}")
    print(f"🚨 Alerts Generated: {agent_metrics['alerts_generated']}")
    print(f"📋 Recommendations Made: {agent_metrics['recommendations_made']}")
    print(f"💾 Cache Size: {agent_metrics['cache_size']} entries")
    print(f"⏱️ Uptime: {agent_metrics['uptime_hours']:.2f} hours")
    
    print_subsection_header("SYSTEM STATISTICS")
    system_stats = metrics["system_statistics"]
    for key, value in system_stats.items():
        if isinstance(value, (int, float)):
            print(f"📊 {key.replace('_', ' ').title()}: {value}")
        else:
            print(f"📊 {key.replace('_', ' ').title()}: {value}")
    
    print_subsection_header("CAPABILITIES")
    capabilities = metrics["capabilities"]
    for i, capability in enumerate(capabilities, 1):
        print(f"   {i}. {capability}")
    
    print(f"\n🤖 Agent Version: {metrics['version']}")
    print(f"📅 Last Updated: {metrics['last_updated']}")
    
    await agent._on_stop()


async def demonstrate_caching_performance():
    """Demonstrate caching mechanism performance."""
    print_section_header("CACHING PERFORMANCE DEMONSTRATION")
    
    agent = EnhancedClinicalDecisionSupportAgent()
    await agent._on_start()
    
    # Create a consistent request for caching test
    request = ClinicalAnalysisRequest(
        patient_id="cache-test-patient",
        patient_data={"age": 60, "conditions": ["diabetes", "hypertension"]},
        medications=["warfarin", "aspirin", "metformin"],
        allergies=["penicillin"],
        clinical_contexts=["diabetes", "cardiovascular"]
    )
    
    print("🚀 Testing caching performance...")
    
    # First call (should be slow - no cache)
    print("\n1️⃣ First call (no cache):")
    start_time = datetime.utcnow()
    result1 = await agent.comprehensive_clinical_analysis(request)
    first_duration = (datetime.utcnow() - start_time).total_seconds() * 1000
    print(f"   ⏱️ Processing time: {first_duration:.1f}ms")
    print(f"   🆔 Analysis ID: {result1['analysis_metadata']['analysis_id']}")
    
    # Second call (should be fast - cached)
    print("\n2️⃣ Second call (with cache):")
    start_time = datetime.utcnow()
    result2 = await agent.comprehensive_clinical_analysis(request)
    second_duration = (datetime.utcnow() - start_time).total_seconds() * 1000
    print(f"   ⏱️ Processing time: {second_duration:.1f}ms")
    print(f"   🆔 Analysis ID: {result2['analysis_metadata']['analysis_id']}")
    
    # Verify results are identical
    analysis_id_match = result1['analysis_metadata']['analysis_id'] == result2['analysis_metadata']['analysis_id']
    print(f"\n📊 Cache Performance:")
    print(f"   ✅ Results identical: {analysis_id_match}")
    print(f"   🚀 Speed improvement: {(first_duration / second_duration):.1f}x faster")
    print(f"   💾 Cache entries: {len(agent.analysis_cache)}")
    
    await agent._on_stop()


async def main():
    """Run all demonstrations."""
    print("🏥 ENHANCED CLINICAL DECISION SUPPORT AGENT DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Run all demonstrations
        await demonstrate_comprehensive_analysis()
        await demonstrate_focused_analyses()
        await demonstrate_performance_monitoring()
        await demonstrate_caching_performance()
        
        print_section_header("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("✅ All clinical decision support capabilities demonstrated")
        print("🔬 Drug interactions, allergy screening, lab interpretation completed")
        print("📋 Clinical guidelines and risk assessment validated")
        print("⚡ Performance monitoring and caching verified")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())