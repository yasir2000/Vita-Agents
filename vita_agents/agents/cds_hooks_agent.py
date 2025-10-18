"""
Clinical Decision Support (CDS) Hooks implementation for Vita Agents.
Provides comprehensive CDS services including hooks, CQL engine, and quality measures.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
import structlog
from pydantic import BaseModel, Field
import uuid
import httpx

from vita_agents.core.agent import HealthcareAgent, AgentCapability, TaskRequest
from vita_agents.core.config import get_settings


logger = structlog.get_logger(__name__)


class CDSCard(BaseModel):
    """CDS Hooks card response."""
    
    summary: str
    detail: Optional[str] = None
    indicator: str  # info, warning, critical
    source: Dict[str, Any]
    suggestions: List[Dict[str, Any]] = []
    selectionBehavior: Optional[str] = None
    overrideReasons: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []


class CDSHook(BaseModel):
    """CDS Hook definition."""
    
    hook: str
    title: str
    description: str
    id: str
    prefetch: Optional[Dict[str, str]] = {}


class CDSRequest(BaseModel):
    """CDS Hooks request."""
    
    hook: str
    hookInstance: str
    fhirServer: Optional[str] = None
    fhirAuthorization: Optional[Dict[str, Any]] = None
    context: Dict[str, Any]
    prefetch: Optional[Dict[str, Any]] = {}


class CDSResponse(BaseModel):
    """CDS Hooks response."""
    
    cards: List[CDSCard] = []
    systemActions: List[Dict[str, Any]] = []


class CQLExpression(BaseModel):
    """Clinical Quality Language expression."""
    
    name: str
    expression: str
    result_type: str
    dependencies: List[str] = []


class CQLLibrary(BaseModel):
    """CQL Library definition."""
    
    name: str
    version: str
    using: List[str]
    include: List[str] = []
    expressions: List[CQLExpression] = []


class QualityMeasure(BaseModel):
    """Quality measure definition."""
    
    id: str
    title: str
    description: str
    population: Dict[str, str]
    measure_type: str  # proportion, ratio, continuous-variable
    scoring: str
    improvement_notation: str


class CDSHooksAgent(HealthcareAgent):
    """
    Clinical Decision Support (CDS) Hooks Agent.
    
    Capabilities:
    - Implement CDS Hooks protocol
    - Execute Clinical Quality Language (CQL)
    - Calculate quality measures
    - Provide clinical decision support cards
    - Integrate with FHIR servers for patient data
    - Support multiple hook types (patient-view, medication-prescribe, order-review)
    """
    
    def __init__(
        self,
        agent_id: str = "cds-hooks-agent",
        name: str = "CDS Hooks Agent",
        description: str = "Clinical Decision Support using CDS Hooks and CQL",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None,
    ):
        capabilities = [
            AgentCapability(
                name="execute_cds_hook",
                description="Execute CDS Hook and return clinical decision support cards",
                input_schema={
                    "type": "object",
                    "properties": {
                        "hook": {"type": "string"},
                        "context": {"type": "object"},
                        "prefetch": {"type": "object"},
                        "fhir_server": {"type": "string"}
                    },
                    "required": ["hook", "context"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "cards": {"type": "array"},
                        "system_actions": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="execute_cql",
                description="Execute Clinical Quality Language expressions",
                input_schema={
                    "type": "object",
                    "properties": {
                        "cql_expression": {"type": "string"},
                        "patient_context": {"type": "object"},
                        "library": {"type": "string"}
                    },
                    "required": ["cql_expression"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "result": {"type": "object"},
                        "execution_details": {"type": "object"}
                    }
                }
            ),
            AgentCapability(
                name="calculate_quality_measure",
                description="Calculate quality measures using CQL",
                input_schema={
                    "type": "object",
                    "properties": {
                        "measure_id": {"type": "string"},
                        "patient_population": {"type": "array"},
                        "measurement_period": {"type": "object"}
                    },
                    "required": ["measure_id"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "measure_report": {"type": "object"},
                        "population_results": {"type": "object"}
                    }
                }
            ),
            AgentCapability(
                name="validate_cql_library",
                description="Validate CQL library syntax and semantics",
                input_schema={
                    "type": "object",
                    "properties": {
                        "cql_library": {"type": "string"},
                        "library_name": {"type": "string"}
                    },
                    "required": ["cql_library"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "validation_result": {"type": "object"},
                        "errors": {"type": "array"},
                        "warnings": {"type": "array"}
                    }
                }
            )
        ]
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            version=version,
            capabilities=capabilities,
            config=config or {}
        )
        
        # CDS Hooks configuration
        self.supported_hooks = self._initialize_supported_hooks()
        self.cql_libraries = self._initialize_cql_libraries()
        self.quality_measures = self._initialize_quality_measures()
        
        # Register task handlers
        self.register_task_handler("execute_cds_hook", self._execute_cds_hook)
        self.register_task_handler("execute_cql", self._execute_cql)
        self.register_task_handler("calculate_quality_measure", self._calculate_quality_measure)
        self.register_task_handler("validate_cql_library", self._validate_cql_library)
    
    def _initialize_supported_hooks(self) -> Dict[str, CDSHook]:
        """Initialize supported CDS Hooks."""
        return {
            "patient-view": CDSHook(
                hook="patient-view",
                title="Patient View",
                description="Hook fired when user is viewing a patient record",
                id="patient-view-cds",
                prefetch={
                    "patient": "Patient/{{context.patientId}}",
                    "conditions": "Condition?patient={{context.patientId}}",
                    "medications": "MedicationStatement?patient={{context.patientId}}"
                }
            ),
            "medication-prescribe": CDSHook(
                hook="medication-prescribe",
                title="Medication Prescribe",
                description="Hook fired when prescribing medications",
                id="medication-prescribe-cds",
                prefetch={
                    "patient": "Patient/{{context.patientId}}",
                    "medications": "MedicationRequest?patient={{context.patientId}}&status=active"
                }
            ),
            "order-review": CDSHook(
                hook="order-review",
                title="Order Review",
                description="Hook fired when reviewing orders",
                id="order-review-cds",
                prefetch={
                    "patient": "Patient/{{context.patientId}}",
                    "orders": "ServiceRequest?patient={{context.patientId}}&status=active"
                }
            )
        }
    
    def _initialize_cql_libraries(self) -> Dict[str, CQLLibrary]:
        """Initialize CQL libraries."""
        return {
            "diabetes_management": CQLLibrary(
                name="DiabetesManagement",
                version="1.2.0",
                using=["FHIR", "QDM"],
                expressions=[
                    CQLExpression(
                        name="HasDiabetes",
                        expression="exists([Condition: \"Diabetes mellitus\"])",
                        result_type="Boolean"
                    ),
                    CQLExpression(
                        name="RecentHbA1c",
                        expression="[Observation: \"Hemoglobin A1c\"] O where O.effective during Interval[Today() - 6 months, Today()]",
                        result_type="List<Observation>"
                    )
                ]
            ),
            "cardiovascular_risk": CQLLibrary(
                name="CardiovascularRisk",
                version="2.1.0",
                using=["FHIR"],
                expressions=[
                    CQLExpression(
                        name="HasHypertension",
                        expression="exists([Condition: \"Essential hypertension\"])",
                        result_type="Boolean"
                    ),
                    CQLExpression(
                        name="CalculateFraminghamScore",
                        expression="// Complex risk calculation logic",
                        result_type="Decimal"
                    )
                ]
            )
        }
    
    def _initialize_quality_measures(self) -> Dict[str, QualityMeasure]:
        """Initialize quality measures."""
        return {
            "diabetes_hba1c": QualityMeasure(
                id="CMS122v10",
                title="Diabetes: Hemoglobin A1c (HbA1c) Poor Control (>9%)",
                description="Percentage of patients 18-75 years of age with diabetes who had hemoglobin A1c > 9.0% during the measurement period",
                population={
                    "initial": "PatientsWith Diabetes",
                    "denominator": "Adults 18-75 with diabetes",
                    "numerator": "Patients with HbA1c > 9%"
                },
                measure_type="proportion",
                scoring="proportion",
                improvement_notation="decrease"
            ),
            "colorectal_screening": QualityMeasure(
                id="CMS130v10",
                title="Colorectal Cancer Screening",
                description="Percentage of adults 50-75 years of age who had appropriate screening for colorectal cancer",
                population={
                    "initial": "Adults 50-75 years",
                    "denominator": "Eligible adults",
                    "numerator": "Adults with appropriate screening"
                },
                measure_type="proportion",
                scoring="proportion",
                improvement_notation="increase"
            )
        }
    
    async def _on_start(self) -> None:
        """Initialize CDS Hooks agent."""
        self.logger.info("Starting CDS Hooks agent", 
                        hooks_count=len(self.supported_hooks),
                        libraries_count=len(self.cql_libraries))
        
        # Initialize HTTP client for FHIR server communication
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        self.logger.info("CDS Hooks agent initialized")
    
    async def _on_stop(self) -> None:
        """Clean up CDS Hooks agent."""
        if hasattr(self, 'http_client'):
            await self.http_client.aclose()
        self.logger.info("CDS Hooks agent stopped")
    
    async def _execute_cds_hook(self, task: TaskRequest) -> Dict[str, Any]:
        """Execute CDS Hook and return clinical decision support cards."""
        try:
            hook = task.parameters.get("hook")
            context = task.parameters.get("context", {})
            prefetch = task.parameters.get("prefetch", {})
            fhir_server = task.parameters.get("fhir_server")
            
            if not hook:
                raise ValueError("Hook type is required")
            
            if hook not in self.supported_hooks:
                raise ValueError(f"Unsupported hook: {hook}")
            
            self.audit_log_action(
                action="execute_cds_hook",
                data_type="CDS",
                details={
                    "hook": hook,
                    "patient_id": context.get("patientId"),
                    "task_id": task.id
                }
            )
            
            # Execute hook-specific logic
            cards = []
            system_actions = []
            
            if hook == "patient-view":
                cards = await self._execute_patient_view_hook(context, prefetch, fhir_server)
            elif hook == "medication-prescribe":
                cards = await self._execute_medication_prescribe_hook(context, prefetch, fhir_server)
            elif hook == "order-review":
                cards = await self._execute_order_review_hook(context, prefetch, fhir_server)
            
            response = CDSResponse(cards=cards, systemActions=system_actions)
            
            return {
                "cds_response": response.dict(),
                "cards_count": len(cards),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("CDS Hook execution failed", error=str(e), task_id=task.id)
            raise
    
    async def _execute_patient_view_hook(self, context: Dict[str, Any], prefetch: Dict[str, Any], fhir_server: Optional[str]) -> List[CDSCard]:
        """Execute patient-view hook."""
        cards = []
        patient_id = context.get("patientId")
        
        if not patient_id:
            return cards
        
        # Get patient data
        patient_data = await self._get_patient_data(patient_id, fhir_server, prefetch)
        
        # Check for diabetes management
        diabetes_card = await self._check_diabetes_management(patient_data)
        if diabetes_card:
            cards.append(diabetes_card)
        
        # Check for preventive care
        preventive_card = await self._check_preventive_care(patient_data)
        if preventive_card:
            cards.append(preventive_card)
        
        # Check for drug allergies
        allergy_card = await self._check_allergy_alerts(patient_data)
        if allergy_card:
            cards.append(allergy_card)
        
        return cards
    
    async def _execute_medication_prescribe_hook(self, context: Dict[str, Any], prefetch: Dict[str, Any], fhir_server: Optional[str]) -> List[CDSCard]:
        """Execute medication-prescribe hook."""
        cards = []
        patient_id = context.get("patientId")
        medications = context.get("medications", [])
        
        if not patient_id or not medications:
            return cards
        
        # Get patient data
        patient_data = await self._get_patient_data(patient_id, fhir_server, prefetch)
        
        # Check for drug-drug interactions
        for medication in medications:
            interaction_card = await self._check_drug_interactions(medication, patient_data)
            if interaction_card:
                cards.append(interaction_card)
        
        # Check for drug-allergy conflicts
        for medication in medications:
            allergy_card = await self._check_drug_allergy_conflicts(medication, patient_data)
            if allergy_card:
                cards.append(allergy_card)
        
        # Check dosage appropriateness
        for medication in medications:
            dosage_card = await self._check_dosage_appropriateness(medication, patient_data)
            if dosage_card:
                cards.append(dosage_card)
        
        return cards
    
    async def _execute_order_review_hook(self, context: Dict[str, Any], prefetch: Dict[str, Any], fhir_server: Optional[str]) -> List[CDSCard]:
        """Execute order-review hook."""
        cards = []
        patient_id = context.get("patientId")
        orders = context.get("orders", [])
        
        if not patient_id or not orders:
            return cards
        
        # Get patient data
        patient_data = await self._get_patient_data(patient_id, fhir_server, prefetch)
        
        # Check order appropriateness
        for order in orders:
            appropriateness_card = await self._check_order_appropriateness(order, patient_data)
            if appropriateness_card:
                cards.append(appropriateness_card)
        
        # Check for duplicate orders
        duplicate_card = await self._check_duplicate_orders(orders, patient_data)
        if duplicate_card:
            cards.append(duplicate_card)
        
        return cards
    
    async def _execute_cql(self, task: TaskRequest) -> Dict[str, Any]:
        """Execute Clinical Quality Language expressions."""
        try:
            cql_expression = task.parameters.get("cql_expression")
            patient_context = task.parameters.get("patient_context", {})
            library = task.parameters.get("library", "")
            
            if not cql_expression:
                raise ValueError("CQL expression is required")
            
            self.audit_log_action(
                action="execute_cql",
                data_type="CQL",
                details={
                    "expression": cql_expression[:100],  # First 100 chars
                    "library": library,
                    "task_id": task.id
                }
            )
            
            # Execute CQL expression
            result = await self._evaluate_cql_expression(cql_expression, patient_context, library)
            
            execution_details = {
                "expression": cql_expression,
                "library": library,
                "execution_time": datetime.utcnow().isoformat(),
                "patient_context_keys": list(patient_context.keys())
            }
            
            return {
                "result": result,
                "execution_details": execution_details,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("CQL execution failed", error=str(e), task_id=task.id)
            raise
    
    async def _calculate_quality_measure(self, task: TaskRequest) -> Dict[str, Any]:
        """Calculate quality measures using CQL."""
        try:
            measure_id = task.parameters.get("measure_id")
            patient_population = task.parameters.get("patient_population", [])
            measurement_period = task.parameters.get("measurement_period", {})
            
            if not measure_id:
                raise ValueError("Measure ID is required")
            
            if measure_id not in self.quality_measures:
                raise ValueError(f"Unknown quality measure: {measure_id}")
            
            self.audit_log_action(
                action="calculate_quality_measure",
                data_type="Quality Measure",
                details={
                    "measure_id": measure_id,
                    "population_size": len(patient_population),
                    "task_id": task.id
                }
            )
            
            measure = self.quality_measures[measure_id]
            
            # Calculate measure for population
            measure_results = await self._calculate_measure_for_population(
                measure, patient_population, measurement_period
            )
            
            # Generate measure report
            measure_report = self._generate_measure_report(measure, measure_results, measurement_period)
            
            return {
                "measure_report": measure_report,
                "population_results": measure_results,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Quality measure calculation failed", error=str(e), task_id=task.id)
            raise
    
    async def _validate_cql_library(self, task: TaskRequest) -> Dict[str, Any]:
        """Validate CQL library syntax and semantics."""
        try:
            cql_library = task.parameters.get("cql_library")
            library_name = task.parameters.get("library_name", "")
            
            if not cql_library:
                raise ValueError("CQL library is required")
            
            self.audit_log_action(
                action="validate_cql_library",
                data_type="CQL Library",
                details={
                    "library_name": library_name,
                    "library_size": len(cql_library),
                    "task_id": task.id
                }
            )
            
            # Perform CQL validation
            validation_result = await self._perform_cql_validation(cql_library)
            
            return {
                "validation_result": validation_result,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("CQL library validation failed", error=str(e), task_id=task.id)
            raise
    
    async def _get_patient_data(self, patient_id: str, fhir_server: Optional[str], prefetch: Dict[str, Any]) -> Dict[str, Any]:
        """Get patient data from FHIR server or prefetch."""
        patient_data = {
            "patient": {},
            "conditions": [],
            "medications": [],
            "observations": [],
            "allergies": []
        }
        
        # Use prefetch data if available
        if prefetch:
            patient_data.update(prefetch)
            return patient_data
        
        # Otherwise fetch from FHIR server
        if fhir_server:
            try:
                # Fetch patient
                patient_response = await self.http_client.get(f"{fhir_server}/Patient/{patient_id}")
                if patient_response.status_code == 200:
                    patient_data["patient"] = patient_response.json()
                
                # Fetch conditions
                conditions_response = await self.http_client.get(f"{fhir_server}/Condition?patient={patient_id}")
                if conditions_response.status_code == 200:
                    bundle = conditions_response.json()
                    patient_data["conditions"] = [entry["resource"] for entry in bundle.get("entry", [])]
                
                # Fetch medications
                medications_response = await self.http_client.get(f"{fhir_server}/MedicationStatement?patient={patient_id}")
                if medications_response.status_code == 200:
                    bundle = medications_response.json()
                    patient_data["medications"] = [entry["resource"] for entry in bundle.get("entry", [])]
                
            except Exception as e:
                self.logger.warning("Failed to fetch patient data from FHIR server", error=str(e))
        
        return patient_data
    
    async def _check_diabetes_management(self, patient_data: Dict[str, Any]) -> Optional[CDSCard]:
        """Check diabetes management recommendations."""
        # Check if patient has diabetes
        has_diabetes = False
        for condition in patient_data.get("conditions", []):
            if "diabetes" in condition.get("code", {}).get("text", "").lower():
                has_diabetes = True
                break
        
        if not has_diabetes:
            return None
        
        # Check for recent HbA1c
        recent_hba1c = False
        for observation in patient_data.get("observations", []):
            if "hemoglobin a1c" in observation.get("code", {}).get("text", "").lower():
                # Check if observation is within last 6 months
                recent_hba1c = True
                break
        
        if not recent_hba1c:
            return CDSCard(
                summary="Diabetes: HbA1c testing due",
                detail="Patient with diabetes should have HbA1c tested every 6 months",
                indicator="warning",
                source={
                    "label": "Diabetes Management Guidelines",
                    "url": "https://care.diabetesjournals.org/content/diabetes-care"
                },
                suggestions=[
                    {
                        "label": "Order HbA1c test",
                        "actions": [
                            {
                                "type": "create",
                                "description": "Create HbA1c order",
                                "resource": {
                                    "resourceType": "ServiceRequest",
                                    "status": "draft",
                                    "code": {
                                        "coding": [{
                                            "system": "http://loinc.org",
                                            "code": "4548-4",
                                            "display": "Hemoglobin A1c"
                                        }]
                                    }
                                }
                            }
                        ]
                    }
                ]
            )
        
        return None
    
    async def _check_preventive_care(self, patient_data: Dict[str, Any]) -> Optional[CDSCard]:
        """Check preventive care recommendations."""
        patient = patient_data.get("patient", {})
        birth_date = patient.get("birthDate")
        
        if not birth_date:
            return None
        
        # Calculate age (simplified)
        try:
            from datetime import datetime
            birth_year = int(birth_date.split("-")[0])
            current_year = datetime.now().year
            age = current_year - birth_year
        except:
            return None
        
        # Check for colorectal screening (age 50-75)
        if 50 <= age <= 75:
            return CDSCard(
                summary="Preventive Care: Colorectal cancer screening due",
                detail="Adults aged 50-75 should receive regular colorectal cancer screening",
                indicator="info",
                source={
                    "label": "USPSTF Guidelines",
                    "url": "https://www.uspreventiveservicestaskforce.org/"
                }
            )
        
        return None
    
    async def _check_allergy_alerts(self, patient_data: Dict[str, Any]) -> Optional[CDSCard]:
        """Check for allergy alerts."""
        allergies = patient_data.get("allergies", [])
        
        if not allergies:
            return None
        
        # Check for serious allergies
        serious_allergies = []
        for allergy in allergies:
            if allergy.get("criticality") == "high":
                serious_allergies.append(allergy)
        
        if serious_allergies:
            allergy_names = [allergy.get("code", {}).get("text", "Unknown") for allergy in serious_allergies]
            return CDSCard(
                summary="Critical allergies documented",
                detail=f"Patient has documented allergies to: {', '.join(allergy_names)}",
                indicator="critical",
                source={
                    "label": "Patient Allergy Record",
                    "url": "#"
                }
            )
        
        return None
    
    async def _check_drug_interactions(self, medication: Dict[str, Any], patient_data: Dict[str, Any]) -> Optional[CDSCard]:
        """Check for drug-drug interactions."""
        # Simplified drug interaction checking
        current_medications = patient_data.get("medications", [])
        
        # Example: Check for warfarin + aspirin interaction
        new_med_name = medication.get("medicationCodeableConcept", {}).get("text", "").lower()
        
        if "warfarin" in new_med_name:
            for current_med in current_medications:
                current_med_name = current_med.get("medicationCodeableConcept", {}).get("text", "").lower()
                if "aspirin" in current_med_name:
                    return CDSCard(
                        summary="Drug interaction: Warfarin + Aspirin",
                        detail="Moderate interaction risk. Monitor INR more frequently and watch for bleeding signs.",
                        indicator="warning",
                        source={
                            "label": "Drug Interaction Database",
                            "url": "https://www.drugs.com/drug_interactions.html"
                        },
                        suggestions=[
                            {
                                "label": "Increase INR monitoring",
                                "actions": [
                                    {
                                        "type": "create",
                                        "description": "Schedule INR monitoring",
                                        "resource": {
                                            "resourceType": "ServiceRequest",
                                            "status": "draft",
                                            "code": {
                                                "coding": [{
                                                    "system": "http://loinc.org",
                                                    "code": "5902-2",
                                                    "display": "Prothrombin time"
                                                }]
                                            }
                                        }
                                    }
                                ]
                            }
                        ]
                    )
        
        return None
    
    async def _check_drug_allergy_conflicts(self, medication: Dict[str, Any], patient_data: Dict[str, Any]) -> Optional[CDSCard]:
        """Check for drug-allergy conflicts."""
        allergies = patient_data.get("allergies", [])
        medication_name = medication.get("medicationCodeableConcept", {}).get("text", "").lower()
        
        for allergy in allergies:
            allergy_name = allergy.get("code", {}).get("text", "").lower()
            
            # Example: Check for penicillin allergy vs penicillin prescription
            if "penicillin" in allergy_name and "penicillin" in medication_name:
                return CDSCard(
                    summary="Drug allergy conflict: Penicillin",
                    detail="Patient has documented penicillin allergy. Consider alternative antibiotic.",
                    indicator="critical",
                    source={
                        "label": "Patient Allergy Record",
                        "url": "#"
                    },
                    suggestions=[
                        {
                            "label": "Consider alternative antibiotic",
                            "actions": [
                                {
                                    "type": "delete",
                                    "description": "Remove penicillin prescription"
                                }
                            ]
                        }
                    ]
                )
        
        return None
    
    async def _check_dosage_appropriateness(self, medication: Dict[str, Any], patient_data: Dict[str, Any]) -> Optional[CDSCard]:
        """Check medication dosage appropriateness."""
        # Simplified dosage checking
        dosage = medication.get("dosage", [])
        patient = patient_data.get("patient", {})
        
        # Example: Check for elderly patients
        birth_date = patient.get("birthDate")
        if birth_date:
            try:
                from datetime import datetime
                birth_year = int(birth_date.split("-")[0])
                current_year = datetime.now().year
                age = current_year - birth_year
                
                if age >= 65:
                    return CDSCard(
                        summary="Geriatric dosing consideration",
                        detail="Consider dose adjustment for elderly patient (age 65+)",
                        indicator="info",
                        source={
                            "label": "Geriatric Dosing Guidelines",
                            "url": "https://www.beerscriteriaupdate.org/"
                        }
                    )
            except:
                pass
        
        return None
    
    async def _check_order_appropriateness(self, order: Dict[str, Any], patient_data: Dict[str, Any]) -> Optional[CDSCard]:
        """Check order appropriateness."""
        # Example: Check for imaging with contrast in patients with kidney disease
        order_code = order.get("code", {}).get("text", "").lower()
        
        if "contrast" in order_code:
            conditions = patient_data.get("conditions", [])
            for condition in conditions:
                condition_text = condition.get("code", {}).get("text", "").lower()
                if "kidney" in condition_text or "renal" in condition_text:
                    return CDSCard(
                        summary="Contrast study in kidney disease",
                        detail="Patient has kidney disease. Consider alternative imaging without contrast or verify creatinine levels.",
                        indicator="warning",
                        source={
                            "label": "Nephrology Guidelines",
                            "url": "https://www.kidney.org/"
                        }
                    )
        
        return None
    
    async def _check_duplicate_orders(self, orders: List[Dict[str, Any]], patient_data: Dict[str, Any]) -> Optional[CDSCard]:
        """Check for duplicate orders."""
        order_codes = []
        for order in orders:
            code = order.get("code", {}).get("text", "")
            if code in order_codes:
                return CDSCard(
                    summary="Duplicate order detected",
                    detail=f"Duplicate order for: {code}",
                    indicator="warning",
                    source={
                        "label": "Order Management System",
                        "url": "#"
                    }
                )
            order_codes.append(code)
        
        return None
    
    async def _evaluate_cql_expression(self, expression: str, patient_context: Dict[str, Any], library: str) -> Dict[str, Any]:
        """Evaluate CQL expression."""
        # Simplified CQL evaluation - in production would use proper CQL engine
        
        # Example evaluations
        if "HasDiabetes" in expression:
            # Check if patient has diabetes
            conditions = patient_context.get("conditions", [])
            has_diabetes = any("diabetes" in condition.get("code", {}).get("text", "").lower() 
                             for condition in conditions)
            return {"result": has_diabetes, "type": "Boolean"}
        
        elif "RecentHbA1c" in expression:
            # Find recent HbA1c observations
            observations = patient_context.get("observations", [])
            recent_hba1c = [obs for obs in observations 
                           if "hemoglobin a1c" in obs.get("code", {}).get("text", "").lower()]
            return {"result": recent_hba1c, "type": "List<Observation>"}
        
        else:
            # Default response
            return {"result": None, "type": "Unknown", "message": "Expression evaluation not implemented"}
    
    async def _calculate_measure_for_population(self, measure: QualityMeasure, population: List[str], period: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality measure for patient population."""
        # Simplified measure calculation
        results = {
            "initial_population": len(population),
            "denominator": 0,
            "numerator": 0,
            "denominator_exclusions": 0,
            "denominator_exceptions": 0
        }
        
        if measure.id == "CMS122v10":  # Diabetes HbA1c measure
            # Simulate diabetes population calculation
            results["denominator"] = int(len(population) * 0.15)  # 15% with diabetes
            results["numerator"] = int(results["denominator"] * 0.25)  # 25% with poor control
        
        elif measure.id == "CMS130v10":  # Colorectal screening
            # Simulate screening population calculation
            results["denominator"] = int(len(population) * 0.30)  # 30% eligible for screening
            results["numerator"] = int(results["denominator"] * 0.75)  # 75% up to date
        
        return results
    
    def _generate_measure_report(self, measure: QualityMeasure, results: Dict[str, Any], period: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality measure report."""
        report = {
            "resourceType": "MeasureReport",
            "id": str(uuid.uuid4()),
            "status": "complete",
            "type": "summary",
            "measure": measure.id,
            "period": period,
            "group": [
                {
                    "population": [
                        {
                            "code": {
                                "coding": [{
                                    "system": "http://terminology.hl7.org/CodeSystem/measure-population",
                                    "code": "initial-population"
                                }]
                            },
                            "count": results["initial_population"]
                        },
                        {
                            "code": {
                                "coding": [{
                                    "system": "http://terminology.hl7.org/CodeSystem/measure-population",
                                    "code": "denominator"
                                }]
                            },
                            "count": results["denominator"]
                        },
                        {
                            "code": {
                                "coding": [{
                                    "system": "http://terminology.hl7.org/CodeSystem/measure-population",
                                    "code": "numerator"
                                }]
                            },
                            "count": results["numerator"]
                        }
                    ]
                }
            ]
        }
        
        # Calculate score
        if results["denominator"] > 0:
            score = results["numerator"] / results["denominator"]
            report["group"][0]["measureScore"] = {
                "value": round(score, 4),
                "unit": "proportion"
            }
        
        return report
    
    async def _perform_cql_validation(self, cql_library: str) -> Dict[str, Any]:
        """Perform CQL library validation."""
        # Simplified CQL validation
        validation_result = {
            "valid": True,
            "syntax_valid": True,
            "semantics_valid": True,
            "dependencies_resolved": True,
            "errors": [],
            "warnings": []
        }
        
        # Basic syntax checks
        if "library" not in cql_library.lower():
            validation_result["valid"] = False
            validation_result["syntax_valid"] = False
            validation_result["errors"].append("Library declaration missing")
        
        if "using" not in cql_library.lower():
            validation_result["warnings"].append("Data model not specified")
        
        return {
            "validation_result": validation_result,
            "library_info": {
                "estimated_size": len(cql_library),
                "estimated_expressions": cql_library.count("define"),
                "validation_time": datetime.utcnow().isoformat()
            }
        }