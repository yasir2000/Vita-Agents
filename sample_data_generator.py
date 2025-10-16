"""
Sample Healthcare Data Generator
Creates realistic patient data and medical scenarios for testing and demonstration
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import uuid

@dataclass
class Patient:
    id: str
    name: str
    age: int
    gender: str
    email: str
    phone: str
    address: str
    emergency_contact: str
    medical_record_number: str
    date_of_birth: str
    insurance_info: Dict[str, str]
    allergies: List[str]
    current_medications: List[Dict[str, str]]
    medical_history: List[str]
    vital_signs: Dict[str, Any]
    last_visit: str
    next_appointment: str
    status: str
    primary_physician: str

@dataclass
class ClinicalNote:
    id: str
    patient_id: str
    date: str
    provider: str
    note_type: str
    chief_complaint: str
    history_present_illness: str
    physical_exam: str
    assessment: str
    plan: str
    vitals: Dict[str, Any]
    
@dataclass
class LabResult:
    id: str
    patient_id: str
    test_name: str
    result_value: str
    normal_range: str
    units: str
    status: str
    date_collected: str
    date_reported: str
    provider: str

@dataclass
class MedicalScenario:
    id: str
    title: str
    description: str
    patient: Patient
    clinical_notes: List[ClinicalNote]
    lab_results: List[LabResult]
    diagnosis: str
    treatment_plan: str
    learning_objectives: List[str]
    difficulty_level: str

class SampleDataGenerator:
    def __init__(self):
        self.first_names = [
            "Emily", "Michael", "Sarah", "David", "Jessica", "Christopher", "Ashley", "Matthew",
            "Amanda", "Joshua", "Jennifer", "Andrew", "Elizabeth", "Daniel", "Stephanie", "Joseph",
            "Lauren", "Ryan", "Nicole", "Brandon", "Samantha", "Tyler", "Rachel", "Kevin",
            "Maria", "John", "Lisa", "James", "Michelle", "Robert", "Amy", "William"
        ]
        
        self.last_names = [
            "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez",
            "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor",
            "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez",
            "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright"
        ]
        
        self.medications = [
            {"name": "Lisinopril", "dosage": "10mg", "frequency": "Daily", "indication": "Hypertension"},
            {"name": "Metformin", "dosage": "500mg", "frequency": "Twice daily", "indication": "Diabetes"},
            {"name": "Atorvastatin", "dosage": "20mg", "frequency": "Daily", "indication": "High cholesterol"},
            {"name": "Amlodipine", "dosage": "5mg", "frequency": "Daily", "indication": "Hypertension"},
            {"name": "Levothyroxine", "dosage": "75mcg", "frequency": "Daily", "indication": "Hypothyroidism"},
            {"name": "Aspirin", "dosage": "81mg", "frequency": "Daily", "indication": "Cardioprotection"},
            {"name": "Omeprazole", "dosage": "20mg", "frequency": "Daily", "indication": "GERD"},
            {"name": "Warfarin", "dosage": "5mg", "frequency": "Daily", "indication": "Anticoagulation"}
        ]
        
        self.allergies = [
            "Penicillin", "Sulfa drugs", "Latex", "Shellfish", "Peanuts", "Eggs",
            "NSAIDs", "Iodine", "Aspirin", "Codeine", "Morphine"
        ]
        
        self.medical_conditions = [
            "Hypertension", "Type 2 Diabetes", "Hyperlipidemia", "Asthma", "COPD",
            "Coronary Artery Disease", "Atrial Fibrillation", "Hypothyroidism",
            "Arthritis", "Depression", "Anxiety", "Chronic Kidney Disease"
        ]
        
        self.physicians = [
            "Dr. Sarah Chen, MD", "Dr. Michael Rodriguez, MD", "Dr. Emily Thompson, MD",
            "Dr. David Kim, MD", "Dr. Lisa Anderson, MD", "Dr. James Wilson, MD",
            "Dr. Maria Garcia, MD", "Dr. Robert Johnson, MD"
        ]
    
    def generate_patient(self) -> Patient:
        """Generate a realistic patient"""
        patient_id = str(uuid.uuid4())
        first_name = random.choice(self.first_names)
        last_name = random.choice(self.last_names)
        age = random.randint(18, 85)
        gender = random.choice(["Male", "Female"])
        
        # Generate date of birth based on age
        today = datetime.now()
        birth_year = today.year - age
        dob = datetime(birth_year, random.randint(1, 12), random.randint(1, 28))
        
        # Generate realistic vital signs based on age and conditions
        vitals = self._generate_vitals(age)
        
        # Generate allergies (0-3 allergies)
        patient_allergies = random.sample(self.allergies, random.randint(0, 3))
        
        # Generate medications (0-5 medications)
        patient_medications = random.sample(self.medications, random.randint(0, 5))
        
        # Generate medical history
        medical_history = random.sample(self.medical_conditions, random.randint(0, 4))
        
        return Patient(
            id=patient_id,
            name=f"{first_name} {last_name}",
            age=age,
            gender=gender,
            email=f"{first_name.lower()}.{last_name.lower()}@email.com",
            phone=f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}",
            address=f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Elm', 'Pine', 'Cedar'])} St, City, ST {random.randint(10000, 99999)}",
            emergency_contact=f"{random.choice(self.first_names)} {random.choice(self.last_names)} - ({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}",
            medical_record_number=f"MRN{random.randint(100000, 999999)}",
            date_of_birth=dob.strftime("%Y-%m-%d"),
            insurance_info={
                "provider": random.choice(["Blue Cross", "Aetna", "Cigna", "UnitedHealth", "Kaiser"]),
                "policy_number": f"POL{random.randint(100000, 999999)}",
                "group_number": f"GRP{random.randint(1000, 9999)}"
            },
            allergies=patient_allergies,
            current_medications=patient_medications,
            medical_history=medical_history,
            vital_signs=vitals,
            last_visit=(today - timedelta(days=random.randint(1, 180))).strftime("%Y-%m-%d"),
            next_appointment=(today + timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d"),
            status=random.choice(["Active", "Scheduled", "Follow-up needed"]),
            primary_physician=random.choice(self.physicians)
        )
    
    def _generate_vitals(self, age: int) -> Dict[str, Any]:
        """Generate realistic vital signs based on age"""
        # Base values with age-related adjustments
        base_systolic = 110 + (age * 0.3)
        base_diastolic = 70 + (age * 0.15)
        
        return {
            "blood_pressure": f"{int(base_systolic + random.randint(-10, 20))}/{int(base_diastolic + random.randint(-5, 15))}",
            "heart_rate": random.randint(60, 100),
            "respiratory_rate": random.randint(12, 20),
            "temperature": round(random.uniform(97.8, 99.2), 1),
            "oxygen_saturation": random.randint(95, 100),
            "weight": round(random.uniform(120, 250), 1),
            "height": f"{random.randint(5, 6)}'{random.randint(0, 11)}\"",
            "bmi": round(random.uniform(18.5, 35.0), 1)
        }
    
    def generate_clinical_scenarios(self, num_scenarios: int = 10) -> List[MedicalScenario]:
        """Generate comprehensive medical scenarios"""
        scenarios = []
        
        scenario_templates = [
            {
                "title": "Acute Chest Pain Evaluation",
                "description": "65-year-old patient presents to ED with chest pain",
                "diagnosis": "Acute Coronary Syndrome vs. Gastroesophageal Reflux",
                "difficulty": "Intermediate"
            },
            {
                "title": "Diabetic Ketoacidosis Management",
                "description": "Type 1 diabetic with nausea, vomiting, and altered mental status",
                "diagnosis": "Diabetic Ketoacidosis",
                "difficulty": "Advanced"
            },
            {
                "title": "Hypertensive Crisis",
                "description": "Patient with severe hypertension and end-organ damage",
                "diagnosis": "Hypertensive Emergency",
                "difficulty": "Advanced"
            },
            {
                "title": "Community-Acquired Pneumonia",
                "description": "Elderly patient with fever, cough, and dyspnea",
                "diagnosis": "Community-Acquired Pneumonia",
                "difficulty": "Beginner"
            },
            {
                "title": "Medication Reconciliation",
                "description": "Polypharmacy patient with potential drug interactions",
                "diagnosis": "Polypharmacy with Drug Interactions",
                "difficulty": "Intermediate"
            }
        ]
        
        for i in range(num_scenarios):
            template = random.choice(scenario_templates)
            patient = self.generate_patient()
            
            # Generate clinical notes
            clinical_notes = self._generate_clinical_notes(patient, template)
            
            # Generate lab results
            lab_results = self._generate_lab_results(patient, template["diagnosis"])
            
            scenario = MedicalScenario(
                id=str(uuid.uuid4()),
                title=f"{template['title']} - Case {i+1}",
                description=template["description"],
                patient=patient,
                clinical_notes=clinical_notes,
                lab_results=lab_results,
                diagnosis=template["diagnosis"],
                treatment_plan=self._generate_treatment_plan(template["diagnosis"]),
                learning_objectives=[
                    f"Evaluate {template['title'].lower()}",
                    "Apply clinical reasoning skills",
                    "Develop appropriate treatment plan",
                    "Consider differential diagnoses"
                ],
                difficulty_level=template["difficulty"]
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_clinical_notes(self, patient: Patient, template: Dict) -> List[ClinicalNote]:
        """Generate clinical notes for a scenario"""
        notes = []
        
        # Initial visit note
        note = ClinicalNote(
            id=str(uuid.uuid4()),
            patient_id=patient.id,
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            provider=patient.primary_physician,
            note_type="Initial Assessment",
            chief_complaint=self._generate_chief_complaint(template["title"]),
            history_present_illness=self._generate_hpi(template["title"]),
            physical_exam=self._generate_physical_exam(),
            assessment=template["diagnosis"],
            plan=self._generate_initial_plan(template["diagnosis"]),
            vitals=patient.vital_signs
        )
        notes.append(note)
        
        return notes
    
    def _generate_chief_complaint(self, scenario_type: str) -> str:
        """Generate chief complaint based on scenario"""
        complaints = {
            "Acute Chest Pain": "Chest pain for 2 hours",
            "Diabetic Ketoacidosis": "Nausea, vomiting, and fatigue for 2 days",
            "Hypertensive Crisis": "Severe headache and blurred vision",
            "Community-Acquired Pneumonia": "Cough, fever, and shortness of breath",
            "Medication Reconciliation": "Routine medication review"
        }
        
        for key in complaints:
            if key in scenario_type:
                return complaints[key]
        
        return "General medical evaluation"
    
    def _generate_hpi(self, scenario_type: str) -> str:
        """Generate history of present illness"""
        hpi_templates = {
            "Acute Chest Pain": "Patient reports sudden onset of substernal chest pain rated 8/10, radiating to left arm. Associated with diaphoresis and nausea. No previous episodes.",
            "Diabetic Ketoacidosis": "Patient with known Type 1 diabetes reports 2 days of nausea, vomiting, and increasing fatigue. Last insulin dose unclear.",
            "Hypertensive Crisis": "Patient reports severe headache onset this morning with associated blurred vision and dizziness. No recent medication changes.",
            "Community-Acquired Pneumonia": "Progressive cough with purulent sputum, fever to 101.5°F, and increasing dyspnea over past 3 days.",
            "Medication Reconciliation": "Patient reports taking multiple medications but unsure of exact regimen. Requests medication review."
        }
        
        for key in hpi_templates:
            if key in scenario_type:
                return hpi_templates[key]
        
        return "Patient presents for routine evaluation."
    
    def _generate_physical_exam(self) -> str:
        """Generate physical examination findings"""
        return "Vital signs as documented. General appearance: alert and oriented. HEENT: normocephalic, atraumatic. Cardiovascular: regular rate and rhythm. Pulmonary: clear to auscultation bilaterally. Abdomen: soft, non-tender. Neurologic: grossly intact."
    
    def _generate_initial_plan(self, diagnosis: str) -> str:
        """Generate initial treatment plan"""
        plans = {
            "Acute Coronary Syndrome": "ECG, cardiac enzymes, chest X-ray. Aspirin, nitroglycerin PRN. Cardiology consult.",
            "Diabetic Ketoacidosis": "IV fluids, insulin protocol, electrolyte monitoring. Endocrinology consult.",
            "Hypertensive Emergency": "Antihypertensive therapy, neurologic monitoring. Target gradual BP reduction.",
            "Community-Acquired Pneumonia": "Chest X-ray, blood cultures. Antibiotic therapy per guidelines.",
            "Polypharmacy": "Complete medication reconciliation, drug interaction screening, patient education."
        }
        
        for key in plans:
            if key in diagnosis:
                return plans[key]
        
        return "Continue current management, follow up as needed."
    
    def _generate_lab_results(self, patient: Patient, diagnosis: str) -> List[LabResult]:
        """Generate relevant lab results"""
        labs = []
        base_date = datetime.now() - timedelta(hours=2)
        
        # Basic metabolic panel
        labs.extend([
            LabResult(
                id=str(uuid.uuid4()),
                patient_id=patient.id,
                test_name="Glucose",
                result_value=str(random.randint(80, 200)),
                normal_range="70-100",
                units="mg/dL",
                status="Final",
                date_collected=base_date.strftime("%Y-%m-%d %H:%M"),
                date_reported=(base_date + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
                provider=patient.primary_physician
            ),
            LabResult(
                id=str(uuid.uuid4()),
                patient_id=patient.id,
                test_name="Sodium",
                result_value=str(random.randint(135, 145)),
                normal_range="136-145",
                units="mEq/L",
                status="Final",
                date_collected=base_date.strftime("%Y-%m-%d %H:%M"),
                date_reported=(base_date + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
                provider=patient.primary_physician
            ),
            LabResult(
                id=str(uuid.uuid4()),
                patient_id=patient.id,
                test_name="Creatinine",
                result_value=str(round(random.uniform(0.8, 1.5), 1)),
                normal_range="0.7-1.3",
                units="mg/dL",
                status="Final",
                date_collected=base_date.strftime("%Y-%m-%d %H:%M"),
                date_reported=(base_date + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
                provider=patient.primary_physician
            )
        ])
        
        # Diagnosis-specific labs
        if "Coronary" in diagnosis:
            labs.append(LabResult(
                id=str(uuid.uuid4()),
                patient_id=patient.id,
                test_name="Troponin I",
                result_value=str(round(random.uniform(0.01, 2.5), 2)),
                normal_range="<0.04",
                units="ng/mL",
                status="Final",
                date_collected=base_date.strftime("%Y-%m-%d %H:%M"),
                date_reported=(base_date + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
                provider=patient.primary_physician
            ))
        
        if "Diabetic" in diagnosis:
            labs.extend([
                LabResult(
                    id=str(uuid.uuid4()),
                    patient_id=patient.id,
                    test_name="HbA1c",
                    result_value=str(round(random.uniform(7.0, 12.0), 1)),
                    normal_range="<7.0",
                    units="%",
                    status="Final",
                    date_collected=base_date.strftime("%Y-%m-%d %H:%M"),
                    date_reported=(base_date + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
                    provider=patient.primary_physician
                ),
                LabResult(
                    id=str(uuid.uuid4()),
                    patient_id=patient.id,
                    test_name="Ketones",
                    result_value="Positive",
                    normal_range="Negative",
                    units="",
                    status="Final",
                    date_collected=base_date.strftime("%Y-%m-%d %H:%M"),
                    date_reported=(base_date + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
                    provider=patient.primary_physician
                )
            ])
        
        return labs
    
    def _generate_treatment_plan(self, diagnosis: str) -> str:
        """Generate comprehensive treatment plan"""
        plans = {
            "Acute Coronary Syndrome": """
            Immediate:
            - Dual antiplatelet therapy (aspirin + clopidogrel)
            - Anticoagulation with heparin
            - Beta-blocker therapy
            - Statin therapy
            
            Monitoring:
            - Serial ECGs and cardiac enzymes
            - Continuous cardiac monitoring
            - Blood pressure monitoring
            
            Follow-up:
            - Cardiology consultation
            - Echocardiogram
            - Cardiac rehabilitation referral
            """,
            
            "Diabetic Ketoacidosis": """
            Immediate:
            - IV fluid resuscitation with normal saline
            - Insulin infusion protocol
            - Electrolyte replacement (K+, PO4, Mg)
            - Glucose monitoring every hour
            
            Monitoring:
            - Arterial blood gases
            - Basic metabolic panel every 2-4 hours
            - Neurologic status
            
            Follow-up:
            - Endocrinology consultation
            - Diabetes education
            - Insulin regimen optimization
            """,
            
            "Community-Acquired Pneumonia": """
            Immediate:
            - Empirical antibiotic therapy
            - Oxygen therapy if hypoxic
            - Bronchodilators if indicated
            
            Monitoring:
            - Oxygen saturation
            - Respiratory status
            - Temperature curve
            
            Follow-up:
            - Chest X-ray in 48-72 hours
            - Complete antibiotic course
            - Primary care follow-up
            """
        }
        
        for key in plans:
            if key in diagnosis:
                return plans[key]
        
        return "Standard care plan to be developed based on clinical assessment."
    
    def save_sample_data(self, filename: str = "sample_healthcare_data.json"):
        """Generate and save comprehensive sample data"""
        # Generate patients
        patients = [self.generate_patient() for _ in range(50)]
        
        # Generate scenarios
        scenarios = self.generate_clinical_scenarios(10)
        
        # Compile all data
        sample_data = {
            "patients": [asdict(patient) for patient in patients],
            "scenarios": [asdict(scenario) for scenario in scenarios],
            "metadata": {
                "generated_date": datetime.now().isoformat(),
                "total_patients": len(patients),
                "total_scenarios": len(scenarios),
                "data_version": "1.0"
            }
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        return sample_data

# Generate sample data when run directly
if __name__ == "__main__":
    generator = SampleDataGenerator()
    data = generator.save_sample_data()
    print(f"Generated {len(data['patients'])} patients and {len(data['scenarios'])} scenarios")
    print("Sample data saved to sample_healthcare_data.json")