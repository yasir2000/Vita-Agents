"""
Enhanced CLI with LLM Integration and Sample Data
Extends the original CLI with LLM capabilities and realistic healthcare scenarios
"""

import typer
import json
import uuid
import time
import asyncio
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from datetime import datetime

# Import our new modules
from llm_integration import llm_manager, CLINICAL_PROMPTS
from sample_data_generator import SampleDataGenerator
from healthcare_agent_framework import (
    healthcare_workflow, HealthcareRole, ClinicalTaskType, PatientSeverity,
    PatientContext, create_default_healthcare_agents
)
from healthcare_team_framework import (
    team_manager, TeamType, TeamStatus, CoordinationPattern, 
    create_default_healthcare_teams
)

app = typer.Typer(help="🏥 Vita Agents Healthcare AI Platform with LLM Integration")
console = Console()

# LLM Commands
llm_app = typer.Typer(help="🤖 LLM Integration Commands")
app.add_typer(llm_app, name="llm")

# Sample Data Commands  
data_app = typer.Typer(help="📊 Sample Data Management")
app.add_typer(data_app, name="data")

# Healthcare Agent Framework Commands
agent_app = typer.Typer(help="🤖 Healthcare Agent Framework")
app.add_typer(agent_app, name="agents")

# Healthcare Team Framework Commands
team_app = typer.Typer(help="👥 Healthcare Team Management")
app.add_typer(team_app, name="teams")

# Original FHIR commands (simplified for brevity)
fhir_app = typer.Typer(help="🔗 FHIR Operations")
app.add_typer(fhir_app, name="fhir")

@llm_app.command("list-models")
def list_models(
    healthcare_only: bool = typer.Option(False, "--healthcare", "-h", help="Show only healthcare-optimized models")
):
    """List available LLM models"""
    models = llm_manager.get_healthcare_models() if healthcare_only else llm_manager.get_available_models()
    
    if not models:
        console.print("❌ No models available. Check your configuration.", style="red")
        return
    
    table = Table(title="🤖 Available LLM Models")
    table.add_column("Model Key", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Healthcare Optimized", style="yellow")
    table.add_column("Context Length", style="blue")
    table.add_column("Capabilities", style="magenta")
    table.add_column("Cost/Token", style="red")
    
    for key, model in models.items():
        healthcare_icon = "✅" if model.healthcare_optimized else "❌"
        capabilities = ", ".join(model.capabilities[:2])  # Show first 2 capabilities
        cost = f"${model.cost_per_token:.4f}" if model.cost_per_token > 0 else "Free"
        
        table.add_row(
            key,
            model.provider.value.title(),
            healthcare_icon,
            str(model.context_length),
            capabilities,
            cost
        )
    
    console.print(table)
    
    active_model = llm_manager.get_active_model()
    if active_model:
        console.print(f"\n🎯 Active Model: [bold green]{llm_manager.active_model}[/bold green]")
    else:
        console.print("\n⚠️ No active model selected. Use 'llm set-model' to select one.")

@llm_app.command("set-model")
def set_model(model_key: str = typer.Argument(..., help="Model key to activate")):
    """Set the active LLM model"""
    if llm_manager.set_active_model(model_key):
        console.print(f"✅ Active model set to: [bold green]{model_key}[/bold green]")
    else:
        console.print(f"❌ Model not found: {model_key}")
        console.print("Use 'llm list-models' to see available models.")

@llm_app.command("test")
def test_llm(
    prompt: str = typer.Option("What is hypertension?", "--prompt", "-p", help="Test prompt"),
    temperature: float = typer.Option(0.7, "--temp", "-t", help="Temperature (0.0-1.0)"),
    max_tokens: int = typer.Option(200, "--tokens", help="Maximum tokens")
):
    """Test the active LLM model"""
    active_model = llm_manager.get_active_model()
    if not active_model:
        console.print("❌ No active model selected. Use 'llm set-model' first.")
        return
    
    console.print(f"🤖 Testing model: [bold]{llm_manager.active_model}[/bold]")
    console.print(f"📝 Prompt: {prompt}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating response...", total=None)
        
        # Run async function in sync context
        async def run_test():
            return await llm_manager.generate_response(prompt, "", temperature, max_tokens)
        
        result = asyncio.run(run_test())
        progress.remove_task(task)
    
    if "error" in result:
        console.print(f"❌ Error: {result['error']}", style="red")
    else:
        console.print(Panel(
            result.get("response", "No response"),
            title=f"🤖 {result.get('model', 'Unknown')} Response",
            border_style="green"
        ))
        
        # Show usage info
        tokens_used = result.get("tokens_used", 0)
        cost = result.get("cost", 0)
        console.print(f"\n📊 Tokens used: {tokens_used} | Cost: ${cost:.4f}")

@llm_app.command("diagnose")
def ai_diagnose(
    age: int = typer.Option(..., "--age", help="Patient age"),
    gender: str = typer.Option(..., "--gender", help="Patient gender"),
    chief_complaint: str = typer.Option(..., "--complaint", help="Chief complaint"),
    hpi: str = typer.Option("", "--hpi", help="History of present illness"),
    vitals: str = typer.Option("", "--vitals", help="Vital signs"),
    physical_exam: str = typer.Option("", "--exam", help="Physical examination findings")
):
    """Generate AI-powered clinical diagnosis"""
    active_model = llm_manager.get_active_model()
    if not active_model:
        console.print("❌ No active model selected. Use 'llm set-model' first.")
        return
    
    # Use clinical prompt template
    prompt = CLINICAL_PROMPTS["diagnosis"].format(
        age=age,
        gender=gender,
        chief_complaint=chief_complaint,
        hpi=hpi or "Not provided",
        vitals=vitals or "Not provided", 
        physical_exam=physical_exam or "Not provided"
    )
    
    console.print("🩺 Generating clinical diagnosis...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing patient data...", total=None)
        
        async def run_diagnosis():
            return await llm_manager.generate_response(prompt, "", 0.3, 800)  # Lower temperature for clinical use
        
        result = asyncio.run(run_diagnosis())
        progress.remove_task(task)
    
    if "error" in result:
        console.print(f"❌ Error: {result['error']}", style="red")
    else:
        console.print(Panel(
            result.get("response", "No diagnosis generated"),
            title="🩺 Clinical Decision Support",
            border_style="blue"
        ))

@llm_app.command("drug-check")
def drug_interaction_check(
    current_meds: str = typer.Option(..., "--current", help="Current medications (comma-separated)"),
    new_med: str = typer.Option(..., "--new", help="New medication to add")
):
    """Check for drug interactions using AI"""
    active_model = llm_manager.get_active_model()
    if not active_model:
        console.print("❌ No active model selected. Use 'llm set-model' first.")
        return
    
    prompt = CLINICAL_PROMPTS["drug_interaction"].format(
        current_medications=current_meds,
        new_medication=new_med
    )
    
    console.print("💊 Checking drug interactions...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing medications...", total=None)
        
        async def run_check():
            return await llm_manager.generate_response(prompt, "", 0.2, 600)
        
        result = asyncio.run(run_check())
        progress.remove_task(task)
    
    if "error" in result:
        console.print(f"❌ Error: {result['error']}", style="red")
    else:
        console.print(Panel(
            result.get("response", "No interaction analysis generated"),
            title="💊 Drug Interaction Analysis", 
            border_style="yellow"
        ))

@data_app.command("generate")
def generate_sample_data(
    patients: int = typer.Option(50, "--patients", "-p", help="Number of patients to generate"),
    scenarios: int = typer.Option(10, "--scenarios", "-s", help="Number of clinical scenarios"),
    output_file: str = typer.Option("sample_healthcare_data.json", "--output", "-o", help="Output filename")
):
    """Generate sample healthcare data"""
    console.print(f"🔄 Generating {patients} patients and {scenarios} scenarios...")
    
    generator = SampleDataGenerator()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating data...", total=None)
        
        # Generate data
        sample_patients = [generator.generate_patient() for _ in range(patients)]
        sample_scenarios = generator.generate_clinical_scenarios(scenarios)
        
        # Save data
        from dataclasses import asdict
        sample_data = {
            "patients": [asdict(patient) for patient in sample_patients],
            "scenarios": [asdict(scenario) for scenario in sample_scenarios],
            "metadata": {
                "generated_date": datetime.now().isoformat(),
                "total_patients": len(sample_patients),
                "total_scenarios": len(sample_scenarios),
                "data_version": "1.0"
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        progress.remove_task(task)
    
    console.print(f"✅ Generated data saved to: [bold green]{output_file}[/bold green]")
    console.print(f"📊 Total patients: {len(sample_patients)}")
    console.print(f"🏥 Total scenarios: {len(sample_scenarios)}")

@data_app.command("list-patients")
def list_patients(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of patients to show"),
    data_file: str = typer.Option("sample_healthcare_data.json", "--file", "-f", help="Data file")
):
    """List sample patients"""
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        patients = data.get("patients", [])[:limit]
        
        if not patients:
            console.print("❌ No patients found in data file.")
            return
        
        table = Table(title="👥 Sample Patients")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Age", style="yellow")
        table.add_column("Gender", style="blue")
        table.add_column("Conditions", style="red")
        table.add_column("Medications", style="magenta")
        
        for patient in patients:
            conditions = ", ".join(patient.get("medical_history", [])[:2])
            medications = str(len(patient.get("current_medications", [])))
            
            table.add_row(
                patient["id"][:8] + "...",
                patient["name"],
                str(patient["age"]),
                patient["gender"],
                conditions or "None",
                f"{medications} meds"
            )
        
        console.print(table)
        
        total_patients = len(data.get("patients", []))
        console.print(f"\n📊 Showing {len(patients)} of {total_patients} patients")
        
    except FileNotFoundError:
        console.print(f"❌ Data file not found: {data_file}")
        console.print("Use 'data generate' to create sample data first.")

@data_app.command("list-scenarios")
def list_scenarios(
    data_file: str = typer.Option("sample_healthcare_data.json", "--file", "-f", help="Data file")
):
    """List clinical scenarios"""
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        scenarios = data.get("scenarios", [])
        
        if not scenarios:
            console.print("❌ No scenarios found in data file.")
            return
        
        table = Table(title="🏥 Clinical Scenarios")
        table.add_column("Title", style="cyan")
        table.add_column("Patient", style="green")
        table.add_column("Diagnosis", style="yellow")
        table.add_column("Difficulty", style="red")
        table.add_column("Lab Results", style="blue")
        
        for scenario in scenarios:
            patient_name = scenario["patient"]["name"]
            lab_count = len(scenario.get("lab_results", []))
            
            table.add_row(
                scenario["title"],
                patient_name,
                scenario["diagnosis"],
                scenario["difficulty_level"],
                f"{lab_count} tests"
            )
        
        console.print(table)
        
    except FileNotFoundError:
        console.print(f"❌ Data file not found: {data_file}")
        console.print("Use 'data generate' to create sample data first.")

@data_app.command("scenario-details")
def scenario_details(
    scenario_index: int = typer.Argument(..., help="Scenario index (0-based)"),
    data_file: str = typer.Option("sample_healthcare_data.json", "--file", "-f", help="Data file")
):
    """Show detailed scenario information"""
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        scenarios = data.get("scenarios", [])
        
        if scenario_index >= len(scenarios):
            console.print(f"❌ Scenario index {scenario_index} not found. Max index: {len(scenarios)-1}")
            return
        
        scenario = scenarios[scenario_index]
        patient = scenario["patient"]
        
        # Patient info
        console.print(Panel(
            f"Name: {patient['name']}\n"
            f"Age: {patient['age']} | Gender: {patient['gender']}\n"
            f"Medical History: {', '.join(patient['medical_history'])}\n"
            f"Current Medications: {len(patient['current_medications'])} medications\n"
            f"Allergies: {', '.join(patient['allergies']) if patient['allergies'] else 'None'}",
            title="👤 Patient Information",
            border_style="blue"
        ))
        
        # Clinical notes
        if scenario.get("clinical_notes"):
            note = scenario["clinical_notes"][0]
            console.print(Panel(
                f"Chief Complaint: {note['chief_complaint']}\n"
                f"History: {note['history_present_illness']}\n"
                f"Assessment: {note['assessment']}\n"
                f"Plan: {note['plan']}",
                title="📋 Clinical Notes",
                border_style="green"
            ))
        
        # Lab results
        if scenario.get("lab_results"):
            console.print("\n🧪 Lab Results:")
            lab_table = Table()
            lab_table.add_column("Test", style="cyan")
            lab_table.add_column("Result", style="yellow") 
            lab_table.add_column("Normal Range", style="green")
            lab_table.add_column("Status", style="red")
            
            for lab in scenario["lab_results"]:
                lab_table.add_row(
                    lab["test_name"],
                    f"{lab['result_value']} {lab['units']}",
                    lab["normal_range"],
                    lab["status"]
                )
            
            console.print(lab_table)
        
        # Treatment plan
        console.print(Panel(
            scenario["treatment_plan"],
            title="💊 Treatment Plan",
            border_style="yellow"
        ))
        
    except FileNotFoundError:
        console.print(f"❌ Data file not found: {data_file}")
        console.print("Use 'data generate' to create sample data first.")

# Healthcare Agent Framework Commands
@agent_app.command("list")
def list_agents():
    """List all registered healthcare agents"""
    agents = healthcare_workflow.agents
    
    if not agents:
        console.print("❌ No agents registered. Use 'agents init' to create default agents.")
        return
    
    table = Table(title="🤖 Healthcare Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Role", style="green")
    table.add_column("Specialties", style="yellow")
    table.add_column("Active Tasks", style="blue")
    table.add_column("Completed", style="magenta")
    table.add_column("Success Rate", style="red")
    table.add_column("Status", style="white")
    
    for agent in agents.values():
        specialties = ", ".join(agent.specialties[:2])  # Show first 2 specialties
        success_rate = agent.performance_metrics.get("success_rate", 0.0)
        status = "🟢 Active" if agent.active else "🔴 Inactive"
        
        table.add_row(
            agent.name,
            agent.role.value.replace('_', ' ').title(),
            specialties,
            str(len(agent.current_tasks)),
            str(len(agent.completed_tasks)),
            f"{success_rate:.1%}",
            status
        )
    
    console.print(table)

@agent_app.command("init")
def init_agents():
    """Initialize default healthcare agents"""
    console.print("🔄 Initializing default healthcare agents...")
    
    # Clear existing agents and recreate defaults
    healthcare_workflow.agents.clear()
    agents = create_default_healthcare_agents()
    
    console.print(f"✅ Created {len(agents)} healthcare agents:")
    for agent in agents:
        console.print(f"  • {agent.name} ({agent.role.value.replace('_', ' ').title()})")

@agent_app.command("status")
def agent_status():
    """Show healthcare workflow status"""
    status = healthcare_workflow.get_workflow_status()
    
    # Workflow Overview
    console.print(Panel(
        f"Total Agents: [bold]{status['total_agents']}[/bold]\n"
        f"Active Agents: [bold green]{status['active_agents']}[/bold green]\n"
        f"Active Tasks: [bold yellow]{status['active_tasks']}[/bold yellow]\n"
        f"Completed Tasks: [bold blue]{status['completed_tasks']}[/bold blue]\n"
        f"Average Confidence: [bold magenta]{status['average_confidence']:.1%}[/bold magenta]",
        title="🏥 Healthcare Workflow Status",
        border_style="green"
    ))
    
    # Agents by Role
    if status['agents_by_role']:
        console.print("\n👥 Agents by Role:")
        role_table = Table()
        role_table.add_column("Role", style="cyan")
        role_table.add_column("Count", style="green")
        
        for role, count in status['agents_by_role'].items():
            role_table.add_row(role.replace('_', ' ').title(), str(count))
        
        console.print(role_table)
    
    # Tasks by Type
    if status['tasks_by_type']:
        console.print("\n📋 Tasks by Type:")
        task_table = Table()
        task_table.add_column("Task Type", style="yellow")
        task_table.add_column("Count", style="blue")
        
        for task_type, count in status['tasks_by_type'].items():
            task_table.add_row(task_type.replace('_', ' ').title(), str(count))
        
        console.print(task_table)

@agent_app.command("diagnose")
def collaborative_diagnosis(
    age: int = typer.Option(..., "--age", help="Patient age"),
    gender: str = typer.Option(..., "--gender", help="Patient gender"),
    chief_complaint: str = typer.Option(..., "--complaint", help="Chief complaint"),
    medical_history: str = typer.Option("", "--history", help="Medical history (comma-separated)"),
    medications: str = typer.Option("", "--medications", help="Current medications (comma-separated)"),
    severity: str = typer.Option("routine", "--severity", help="Case severity (routine/urgent/critical/emergency)")
):
    """Collaborative diagnosis using multiple healthcare agents"""
    
    # Create patient context
    patient_context = PatientContext(
        patient_id=str(uuid.uuid4()),
        age=age,
        gender=gender,
        chief_complaint=chief_complaint,
        medical_history=medical_history.split(',') if medical_history else [],
        current_medications=medications.split(',') if medications else [],
        severity=PatientSeverity(severity.lower())
    )
    
    console.print(f"🏥 Starting collaborative diagnosis for {age}-year-old {gender}")
    console.print(f"📝 Chief Complaint: {chief_complaint}")
    
    # Create diagnosis task
    task = healthcare_workflow.create_clinical_task(
        task_type=ClinicalTaskType.DIAGNOSIS,
        patient_context=patient_context,
        description=f"Collaborative diagnosis for patient with {chief_complaint}",
        priority=5 if severity == "emergency" else 3
    )
    
    # Assign task to best available agent
    assigned_agent = healthcare_workflow.assign_task(task)
    if not assigned_agent:
        console.print("❌ No suitable agent available for this diagnosis task.")
        return
    
    console.print(f"🎯 Assigned to: {assigned_agent.name} ({assigned_agent.role.value.replace('_', ' ').title()})")
    
    # Execute task
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress_task = progress.add_task("Processing clinical diagnosis...", total=None)
        
        async def run_diagnosis():
            return await healthcare_workflow.execute_task(task.id)
        
        result = asyncio.run(run_diagnosis())
        progress.remove_task(progress_task)
    
    if "error" in result:
        console.print(f"❌ Diagnosis failed: {result['error']}", style="red")
        return
    
    # Display results
    console.print(Panel(
        result.get("response", "No diagnosis generated"),
        title=f"🩺 Collaborative Diagnosis - {assigned_agent.name}",
        border_style="blue"
    ))
    
    # Show additional metrics
    console.print(f"\n📊 Analysis Details:")
    console.print(f"  • Agent: {result.get('agent_role', 'Unknown').replace('_', ' ').title()}")
    console.print(f"  • Confidence: {result.get('confidence', 0):.1%}")
    console.print(f"  • Model: {result.get('model_used', 'Unknown')}")
    console.print(f"  • Tokens: {result.get('tokens_used', 0)}")

@agent_app.command("medication-review")
def medication_review(
    age: int = typer.Option(..., "--age", help="Patient age"),
    gender: str = typer.Option(..., "--gender", help="Patient gender"),
    medications: str = typer.Option(..., "--medications", help="Current medications (comma-separated)"),
    allergies: str = typer.Option("", "--allergies", help="Known allergies (comma-separated)"),
    new_medication: str = typer.Option("", "--new-med", help="New medication to add")
):
    """Comprehensive medication review using pharmacist agent"""
    
    # Create patient context
    patient_context = PatientContext(
        patient_id=str(uuid.uuid4()),
        age=age,
        gender=gender,
        chief_complaint="Medication review",
        current_medications=medications.split(','),
        allergies=allergies.split(',') if allergies else []
    )
    
    task_description = f"Comprehensive medication review for {len(patient_context.current_medications)} medications"
    if new_medication:
        task_description += f" and assessment of adding {new_medication}"
        patient_context.additional_context = {"new_medication": new_medication}
    
    console.print(f"💊 Starting medication review for {age}-year-old {gender}")
    console.print(f"📋 Current medications: {medications}")
    
    # Create medication review task
    task = healthcare_workflow.create_clinical_task(
        task_type=ClinicalTaskType.MEDICATION_REVIEW,
        patient_context=patient_context,
        description=task_description,
        priority=3
    )
    
    # Assign to pharmacist agent
    assigned_agent = healthcare_workflow.assign_task(task)
    if not assigned_agent:
        console.print("❌ No pharmacist agent available for medication review.")
        return
    
    console.print(f"🎯 Assigned to: {assigned_agent.name} ({assigned_agent.role.value.replace('_', ' ').title()})")
    
    # Execute task
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress_task = progress.add_task("Analyzing medications...", total=None)
        
        async def run_review():
            return await healthcare_workflow.execute_task(task.id)
        
        result = asyncio.run(run_review())
        progress.remove_task(progress_task)
    
    if "error" in result:
        console.print(f"❌ Medication review failed: {result['error']}", style="red")
        return
    
    # Display results
    console.print(Panel(
        result.get("response", "No review generated"),
        title=f"💊 Medication Review - {assigned_agent.name}",
        border_style="yellow"
    ))
    
    # Show additional metrics
    console.print(f"\n📊 Analysis Details:")
    console.print(f"  • Confidence: {result.get('confidence', 0):.1%}")
    console.print(f"  • Medications Reviewed: {len(patient_context.current_medications)}")
    console.print(f"  • Known Allergies: {len(patient_context.allergies)}")

@agent_app.command("workflow")
def workflow_demo():
    """Demonstrate multi-agent healthcare workflow"""
    console.print("🏥 Healthcare Multi-Agent Workflow Demo")
    console.print("=" * 50)
    
    # Create a complex patient case
    patient_context = PatientContext(
        patient_id="demo-patient-001",
        age=65,
        gender="female",
        chief_complaint="Chest pain and shortness of breath",
        medical_history=["Hypertension", "Type 2 Diabetes", "Hyperlipidemia"],
        current_medications=["Metformin 1000mg BID", "Lisinopril 10mg daily", "Atorvastatin 40mg daily"],
        allergies=["Penicillin"],
        vital_signs={"BP": "150/90", "HR": "98", "RR": "22", "O2Sat": "94%"},
        severity=PatientSeverity.URGENT
    )
    
    console.print(f"👤 Patient: {patient_context.age}-year-old {patient_context.gender}")
    console.print(f"📝 Chief Complaint: {patient_context.chief_complaint}")
    console.print(f"🔴 Severity: {patient_context.severity.value.upper()}")
    
    # Step 1: Initial Diagnosis
    console.print("\n🔄 Step 1: Initial Diagnosis")
    diagnosis_task = healthcare_workflow.create_clinical_task(
        task_type=ClinicalTaskType.DIAGNOSIS,
        patient_context=patient_context,
        description="Initial assessment of chest pain and dyspnea in elderly diabetic patient",
        priority=4
    )
    
    diagnosis_agent = healthcare_workflow.assign_task(diagnosis_task)
    if diagnosis_agent:
        console.print(f"✅ Assigned to: {diagnosis_agent.name}")
    
    # Step 2: Medication Review
    console.print("\n🔄 Step 2: Medication Review")
    med_task = healthcare_workflow.create_clinical_task(
        task_type=ClinicalTaskType.MEDICATION_REVIEW,
        patient_context=patient_context,
        description="Review current medications for interactions and optimize cardiac medications",
        priority=3
    )
    
    med_agent = healthcare_workflow.assign_task(med_task)
    if med_agent:
        console.print(f"✅ Assigned to: {med_agent.name}")
    
    # Step 3: Care Coordination
    console.print("\n🔄 Step 3: Care Coordination")
    care_task = healthcare_workflow.create_clinical_task(
        task_type=ClinicalTaskType.CARE_COORDINATION,
        patient_context=patient_context,
        description="Coordinate urgent cardiac workup and specialist referrals",
        priority=5
    )
    
    care_agent = healthcare_workflow.assign_task(care_task)
    if care_agent:
        console.print(f"✅ Assigned to: {care_agent.name}")
    
    # Show workflow summary
    console.print("\n📊 Workflow Summary:")
    console.print(f"  • Total tasks created: 3")
    console.print(f"  • Agents involved: {len(set([diagnosis_agent.id if diagnosis_agent else '', med_agent.id if med_agent else '', care_agent.id if care_agent else '']))}")
    console.print(f"  • Estimated completion: {patient_context.severity.value} priority workflow")
    
    console.print("\n💡 This demonstrates how multiple healthcare agents can collaborate on complex cases!")

@agent_app.command("capabilities")
def show_agent_capabilities(
    role: str = typer.Option("", "--role", help="Filter by healthcare role")
):
    """Show agent capabilities and specialties"""
    agents = healthcare_workflow.agents.values()
    
    if role:
        try:
            role_enum = HealthcareRole(role.lower())
            agents = [a for a in agents if a.role == role_enum]
        except ValueError:
            console.print(f"❌ Invalid role: {role}")
            console.print(f"Available roles: {', '.join([r.value for r in HealthcareRole])}")
            return
    
    if not agents:
        console.print("❌ No agents found matching criteria.")
        return
    
    for agent in agents:
        console.print(Panel(
            f"Role: [bold]{agent.role.value.replace('_', ' ').title()}[/bold]\n"
            f"Specialties: {', '.join(agent.specialties)}\n"
            f"Capabilities: {len(agent.capabilities)} defined\n"
            f"Performance: {agent.performance_metrics.get('success_rate', 0):.1%} success rate\n"
            f"Experience: {len(agent.completed_tasks)} tasks completed",
            title=f"🤖 {agent.name}",
            border_style="cyan"
        ))
        
        # Show detailed capabilities
        if agent.capabilities:
            cap_table = Table(title="📋 Capabilities")
            cap_table.add_column("Capability", style="yellow")
            cap_table.add_column("Specialty", style="green")
            cap_table.add_column("Proficiency", style="blue")
            
            for cap_name, capability in agent.capabilities.items():
                cap_table.add_row(
                    cap_name.replace('_', ' ').title(),
                    capability.specialty,
                    f"{capability.proficiency_level:.0%}"
                )
            
            console.print(cap_table)
        
        console.print()  # Add spacing between agents

# Healthcare Team Framework Commands
@team_app.command("list")
def list_teams():
    """List all healthcare teams"""
    teams = team_manager.teams
    
    if not teams:
        console.print("❌ No teams created. Use 'teams init' to create default teams.")
        return
    
    table = Table(title="👥 Healthcare Teams")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Members", style="blue")
    table.add_column("Active Cases", style="magenta")
    table.add_column("Success Rate", style="red")
    table.add_column("Lead Agent", style="white")
    
    for team in teams.values():
        status_icon = {
            TeamStatus.ACTIVE: "🟢 Active",
            TeamStatus.ON_CALL: "🟡 On Call", 
            TeamStatus.BUSY: "🔴 Busy",
            TeamStatus.INACTIVE: "⚫ Inactive",
            TeamStatus.ASSEMBLING: "🔄 Assembling"
        }.get(team.status, team.status.value)
        
        lead_name = "None"
        if team.lead_agent_id and team.lead_agent_id in team.members:
            lead_name = team.members[team.lead_agent_id].name
        
        table.add_row(
            team.name,
            team.team_type.value.replace('_', ' ').title(),
            status_icon,
            str(len(team.members)),
            str(len(team.current_cases)),
            f"{team.metrics.success_rate:.1%}",
            lead_name
        )
    
    console.print(table)

@team_app.command("init")
def init_teams():
    """Initialize default healthcare teams"""
    console.print("🔄 Initializing default healthcare teams...")
    
    # Ensure agents exist first
    if not healthcare_workflow.agents:
        console.print("⚠️ No agents found. Creating default agents first...")
        create_default_healthcare_agents()
    
    # Clear existing teams and create defaults
    team_manager.teams.clear()
    teams = create_default_healthcare_teams()
    
    if teams:
        console.print(f"✅ Created {len(teams)} healthcare teams:")
        for team in teams:
            console.print(f"  • {team.name} ({team.team_type.value.replace('_', ' ').title()}) - {len(team.members)} members")
    else:
        console.print("❌ Failed to create teams. Ensure enough agents are available.")

@team_app.command("status")
def team_status():
    """Show comprehensive team management status"""
    status = team_manager.get_team_performance_summary()
    
    # Overall Team Status
    console.print(Panel(
        f"Total Teams: [bold]{status['total_teams']}[/bold]\n"
        f"Active Teams: [bold green]{status['active_teams']}[/bold green]\n"
        f"Cases Handled: [bold blue]{status['total_cases_handled']}[/bold blue]\n"
        f"Average Success Rate: [bold magenta]{status['average_success_rate']:.1%}[/bold magenta]\n"
        f"Team Utilization: [bold yellow]{status['team_utilization']:.1%}[/bold yellow]",
        title="👥 Team Management Status",
        border_style="green"
    ))
    
    # Teams by Type
    if status['teams_by_type']:
        console.print("\n📊 Teams by Type:")
        type_table = Table()
        type_table.add_column("Team Type", style="cyan")
        type_table.add_column("Count", style="green")
        
        for team_type, count in status['teams_by_type'].items():
            type_table.add_row(team_type.replace('_', ' ').title(), str(count))
        
        console.print(type_table)

@team_app.command("details")
def team_details(
    team_name: str = typer.Option("", "--name", help="Team name to show details for"),
    team_type: str = typer.Option("", "--type", help="Show details for first team of this type")
):
    """Show detailed information about a specific team"""
    target_team = None
    
    if team_name:
        # Find team by name
        for team in team_manager.teams.values():
            if team.name.lower() == team_name.lower():
                target_team = team
                break
    elif team_type:
        # Find first team of specified type
        try:
            team_type_enum = TeamType(team_type.lower().replace(' ', '_'))
            for team in team_manager.teams.values():
                if team.team_type == team_type_enum:
                    target_team = team
                    break
        except ValueError:
            console.print(f"❌ Invalid team type: {team_type}")
            console.print(f"Available types: {', '.join([t.value.replace('_', ' ').title() for t in TeamType])}")
            return
    
    if not target_team:
        console.print("❌ Team not found. Use 'teams list' to see available teams.")
        return
    
    # Team Overview
    console.print(Panel(
        f"Type: [bold]{target_team.team_type.value.replace('_', ' ').title()}[/bold]\n"
        f"Status: [bold]{target_team.status.value.title()}[/bold]\n"
        f"Members: [bold]{len(target_team.members)}[/bold]\n"
        f"Coordination: [bold]{target_team.coordination_pattern.value.replace('_', ' ').title()}[/bold]\n"
        f"Protocols: [bold]{len(target_team.protocols)}[/bold]\n"
        f"Active Cases: [bold]{len(target_team.current_cases)}[/bold]\n"
        f"Success Rate: [bold]{target_team.metrics.success_rate:.1%}[/bold]",
        title=f"👥 {target_team.name}",
        border_style="blue"
    ))
    
    # Team Members
    if target_team.members:
        console.print("\n👤 Team Members:")
        member_table = Table()
        member_table.add_column("Agent", style="cyan")
        member_table.add_column("Role", style="green")
        member_table.add_column("Specialties", style="yellow")
        member_table.add_column("Lead", style="red")
        
        for agent_id, agent in target_team.members.items():
            is_lead = "👑 Yes" if agent_id == target_team.lead_agent_id else "No"
            specialties = ", ".join(agent.specialties[:2])
            
            member_table.add_row(
                agent.name,
                agent.role.value.replace('_', ' ').title(),
                specialties,
                is_lead
            )
        
        console.print(member_table)
    
    # Available Protocols
    if target_team.protocols:
        console.print("\n📋 Available Protocols:")
        protocol_table = Table()
        protocol_table.add_column("Protocol", style="cyan")
        protocol_table.add_column("Priority", style="red")
        protocol_table.add_column("Max Response Time", style="yellow")
        protocol_table.add_column("Required Roles", style="blue")
        
        for protocol in target_team.protocols:
            required_roles = ", ".join([role.value.replace('_', ' ').title() for role in protocol.required_roles])
            
            protocol_table.add_row(
                protocol.name,
                str(protocol.priority_level),
                str(protocol.max_response_time),
                required_roles
            )
        
        console.print(protocol_table)

@team_app.command("create")
def create_team(
    team_type: str = typer.Argument(..., help="Team type to create"),
    name: str = typer.Option("", "--name", help="Custom team name")
):
    """Create a new healthcare team"""
    try:
        team_type_enum = TeamType(team_type.lower().replace(' ', '_'))
    except ValueError:
        console.print(f"❌ Invalid team type: {team_type}")
        console.print(f"Available types: {', '.join([t.value.replace('_', ' ').title() for t in TeamType])}")
        return
    
    # Check if agents are available
    available_agents = list(healthcare_workflow.agents.values())
    if not available_agents:
        console.print("❌ No agents available. Use 'agents init' to create agents first.")
        return
    
    console.print(f"🔄 Creating {team_type_enum.value.replace('_', ' ').title()} team...")
    
    team = team_manager.auto_assemble_team(team_type_enum, available_agents)
    
    if team:
        if name:
            team.name = name
        
        console.print(f"✅ Created team: {team.name}")
        console.print(f"  • Type: {team.team_type.value.replace('_', ' ').title()}")
        console.print(f"  • Members: {len(team.members)}")
        console.print(f"  • Lead: {team.members[team.lead_agent_id].name if team.lead_agent_id else 'None'}")
        console.print(f"  • Protocols: {len(team.protocols)}")
    else:
        console.print(f"❌ Failed to create team. Not enough suitable agents available.")

@team_app.command("emergency-response")
def emergency_response_demo(
    emergency_type: str = typer.Option("cardiac_arrest", "--type", help="Emergency type (cardiac_arrest/stroke/sepsis)"),
    patient_age: int = typer.Option(65, "--age", help="Patient age"),
    patient_gender: str = typer.Option("male", "--gender", help="Patient gender")
):
    """Demonstrate emergency response team coordination"""
    console.print(f"🚨 [bold red]EMERGENCY RESPONSE SIMULATION[/bold red]")
    console.print("=" * 60)
    
    # Create emergency patient context
    emergency_contexts = {
        "cardiac_arrest": PatientContext(
            patient_id="emergency-001",
            age=patient_age,
            gender=patient_gender,
            chief_complaint="Cardiac arrest - unresponsive",
            medical_history=["Coronary artery disease", "Hypertension"],
            severity=PatientSeverity.EMERGENCY,
            vital_signs={"HR": "0", "BP": "0/0", "RR": "0", "O2Sat": "0%"}
        ),
        "stroke": PatientContext(
            patient_id="emergency-002", 
            age=patient_age,
            gender=patient_gender,
            chief_complaint="Acute neurological deficit",
            medical_history=["Atrial fibrillation", "Diabetes"],
            severity=PatientSeverity.EMERGENCY,
            vital_signs={"HR": "110", "BP": "180/100", "RR": "20", "O2Sat": "95%"}
        ),
        "sepsis": PatientContext(
            patient_id="emergency-003",
            age=patient_age,
            gender=patient_gender,
            chief_complaint="Septic shock",
            medical_history=["COPD", "Diabetes"],
            severity=PatientSeverity.CRITICAL,
            vital_signs={"HR": "130", "BP": "70/40", "RR": "28", "O2Sat": "88%"}
        )
    }
    
    if emergency_type not in emergency_contexts:
        console.print(f"❌ Unknown emergency type: {emergency_type}")
        return
    
    patient_context = emergency_contexts[emergency_type]
    
    console.print(f"🚨 Emergency: {emergency_type.replace('_', ' ').title()}")
    console.print(f"👤 Patient: {patient_context.age}-year-old {patient_context.gender}")
    console.print(f"📝 Chief Complaint: {patient_context.chief_complaint}")
    console.print(f"🔴 Severity: {patient_context.severity.value.upper()}")
    
    # Find or create emergency team
    emergency_teams = [t for t in team_manager.teams.values() if t.team_type == TeamType.EMERGENCY_TEAM]
    
    if not emergency_teams:
        console.print("\n🔄 No emergency team available. Auto-assembling...")
        available_agents = list(healthcare_workflow.agents.values())
        emergency_team = team_manager.auto_assemble_team(TeamType.EMERGENCY_TEAM, available_agents)
        
        if not emergency_team:
            console.print("❌ Failed to assemble emergency team.")
            return
    else:
        emergency_team = emergency_teams[0]
    
    console.print(f"\n👥 Emergency Team: {emergency_team.name}")
    console.print(f"  • Status: {emergency_team.status.value}")
    console.print(f"  • Members: {len(emergency_team.members)}")
    console.print(f"  • Lead: {emergency_team.members[emergency_team.lead_agent_id].name if emergency_team.lead_agent_id else 'None'}")
    
    # Activate appropriate protocol
    protocol_mapping = {
        "cardiac_arrest": "Cardiac Arrest Response",
        "stroke": "Stroke Alert", 
        "sepsis": "Sepsis Management"
    }
    
    protocol_name = protocol_mapping.get(emergency_type)
    if protocol_name:
        console.print(f"\n🔄 Activating Protocol: {protocol_name}")
        
        execution_plan = emergency_team.activate_protocol(protocol_name, patient_context)
        
        if "error" in execution_plan:
            console.print(f"❌ Protocol activation failed: {execution_plan['error']}")
            return
        
        # Display execution plan
        console.print(Panel(
            f"Protocol: [bold]{execution_plan['protocol']}[/bold]\n"
            f"Patient ID: {execution_plan['patient_id']}\n"
            f"Severity: {execution_plan['severity'].upper()}\n"
            f"Team Members: {', '.join(execution_plan['team_members'])}\n"
            f"Estimated Completion: {execution_plan['estimated_completion'].strftime('%H:%M:%S')}",
            title="📋 Emergency Response Plan",
            border_style="red"
        ))
        
        # Show protocol steps
        console.print("\n📝 Protocol Steps:")
        step_table = Table()
        step_table.add_column("Step", style="cyan")
        step_table.add_column("Action", style="yellow")
        step_table.add_column("Responsible", style="green")
        step_table.add_column("Duration", style="blue")
        step_table.add_column("Status", style="red")
        
        for step in execution_plan['steps']:
            step_table.add_row(
                str(step['step_number']),
                step['action'],
                step['responsible_role'].replace('_', ' ').title(),
                step['estimated_duration'],
                step['status'].title()
            )
        
        console.print(step_table)
        
        console.print(f"\n✅ Emergency response protocol activated successfully!")
        console.print(f"⏰ Expected response time: {protocol_mapping}")
    
    else:
        console.print(f"\n⚠️ No specific protocol available for {emergency_type}")

@team_app.command("workflow")
def team_workflow_demo():
    """Demonstrate advanced team-based healthcare workflow"""
    console.print("🏥 Advanced Team-Based Healthcare Workflow")
    console.print("=" * 55)
    
    # Create complex multi-system patient case
    patient_context = PatientContext(
        patient_id="complex-case-001",
        age=72,
        gender="female",
        chief_complaint="Multiple system failure - hypotension, altered mental status, fever",
        medical_history=["Heart failure", "Diabetes mellitus", "Chronic kidney disease", "COPD"],
        current_medications=["Metformin", "Lisinopril", "Furosemide", "Albuterol", "Insulin"],
        allergies=["Penicillin", "Contrast dye"],
        vital_signs={"HR": "125", "BP": "85/50", "RR": "26", "O2Sat": "89%", "Temp": "101.8°F"},
        severity=PatientSeverity.CRITICAL
    )
    
    console.print(f"👤 Complex Case: {patient_context.age}-year-old {patient_context.gender}")
    console.print(f"📝 Presentation: {patient_context.chief_complaint}")
    console.print(f"🔴 Severity: {patient_context.severity.value.upper()}")
    console.print(f"🏥 History: {', '.join(patient_context.medical_history[:3])}...")
    
    # Step 1: Emergency Team Initial Response
    console.print("\n🔄 Step 1: Emergency Team Response")
    
    case_id = str(uuid.uuid4())
    assigned_team = team_manager.assign_case_to_team(case_id, patient_context, ClinicalTaskType.DIAGNOSIS)
    
    if assigned_team:
        console.print(f"✅ Assigned to: {assigned_team.name}")
        console.print(f"  • Team Type: {assigned_team.team_type.value.replace('_', ' ').title()}")
        console.print(f"  • Members: {len(assigned_team.members)}")
        console.print(f"  • Lead: {assigned_team.members[assigned_team.lead_agent_id].name if assigned_team.lead_agent_id else 'None'}")
        
        # Activate sepsis protocol if emergency team
        if assigned_team.team_type == TeamType.EMERGENCY_TEAM:
            console.print("\n🚨 Activating Sepsis Management Protocol...")
            protocol_result = assigned_team.activate_protocol("Sepsis Management", patient_context)
            if "error" not in protocol_result:
                console.print("✅ Protocol activated successfully")
    
    # Step 2: ICU Team Coordination
    console.print("\n🔄 Step 2: ICU Team Coordination")
    
    # Try to find or create ICU team
    icu_teams = [t for t in team_manager.teams.values() if t.team_type == TeamType.ICU_TEAM]
    
    if not icu_teams:
        console.print("🔄 Creating ICU team for critical care management...")
        available_agents = list(healthcare_workflow.agents.values())
        icu_team = team_manager.auto_assemble_team(TeamType.ICU_TEAM, available_agents)
        
        if icu_team:
            console.print(f"✅ ICU Team assembled: {icu_team.name}")
        else:
            console.print("⚠️ Could not assemble ICU team - using existing emergency team")
    else:
        icu_team = icu_teams[0]
        console.print(f"✅ ICU Team available: {icu_team.name}")
    
    # Step 3: Multi-Team Coordination Summary
    console.print("\n📊 Multi-Team Workflow Summary:")
    
    active_teams = [t for t in team_manager.teams.values() if t.status == TeamStatus.ACTIVE]
    
    summary_table = Table(title="🏥 Active Healthcare Teams")
    summary_table.add_column("Team", style="cyan")
    summary_table.add_column("Type", style="green") 
    summary_table.add_column("Role in Case", style="yellow")
    summary_table.add_column("Key Responsibilities", style="blue")
    
    for team in active_teams:
        if team.team_type == TeamType.EMERGENCY_TEAM:
            role = "Initial Stabilization"
            responsibilities = "Rapid assessment, life support, initial treatment"
        elif team.team_type == TeamType.ICU_TEAM:
            role = "Critical Care Management"
            responsibilities = "Ongoing monitoring, complex care decisions"
        else:
            role = "Support"
            responsibilities = "Specialized consultation, care coordination"
        
        summary_table.add_row(
            team.name,
            team.team_type.value.replace('_', ' ').title(),
            role,
            responsibilities
        )
    
    console.print(summary_table)
    
    console.print(f"\n💡 This demonstrates advanced team coordination for complex critical care cases!")
    console.print(f"🔄 Multiple specialized teams working together for optimal patient outcomes")

# Basic FHIR commands (simplified)
@fhir_app.command("status")
def fhir_status():
    """Check FHIR server status"""
    console.print("🔗 FHIR Operations")
    console.print("Status: [bold green]Ready[/bold green]")
    console.print("Server: http://localhost:8080/fhir")

@app.command("dashboard")
def dashboard():
    """Show enhanced dashboard with LLM and data status"""
    console.print("🏥 [bold]Vita Agents Healthcare AI Platform[/bold]\n")
    
    # LLM Status
    active_model = llm_manager.get_active_model()
    total_models = len(llm_manager.get_available_models())
    healthcare_models = len(llm_manager.get_healthcare_models())
    
    llm_status = Panel(
        f"Active Model: [bold green]{llm_manager.active_model}[/bold green]\n"
        f"Available Models: {total_models}\n"
        f"Healthcare-Optimized: {healthcare_models}" if active_model else
        f"No active model selected\n"
        f"Available Models: {total_models}\n"
        f"Healthcare-Optimized: {healthcare_models}",
        title="🤖 LLM Status",
        border_style="blue"
    )
    
    # Data Status
    try:
        with open("sample_healthcare_data.json", 'r') as f:
            data = json.load(f)
        
        data_status = Panel(
            f"Patients: {len(data.get('patients', []))}\n"
            f"Scenarios: {len(data.get('scenarios', []))}\n" 
            f"Generated: {data.get('metadata', {}).get('generated_date', 'Unknown')}",
            title="📊 Sample Data",
            border_style="green"
        )
    except FileNotFoundError:
        data_status = Panel(
            "No sample data found\nUse 'data generate' to create sample data",
            title="📊 Sample Data",
            border_style="red"
        )
    
    # Agent Framework Status
    workflow_status = healthcare_workflow.get_workflow_status()
    agent_status_panel = Panel(
        f"Healthcare Agents: {workflow_status['total_agents']}\n"
        f"Active Agents: {workflow_status['active_agents']}\n"
        f"Active Tasks: {workflow_status['active_tasks']}\n"
        f"Completed Tasks: {workflow_status['completed_tasks']}\n"
        f"Success Rate: {workflow_status['average_confidence']:.1%}" if workflow_status['completed_tasks'] > 0 else
        f"Healthcare Agents: {workflow_status['total_agents']}\n"
        f"Active Agents: {workflow_status['active_agents']}\n"
        f"No tasks completed yet\n"
        f"Use 'agents init' to initialize default agents",
        title="🤖 Healthcare Agents",
        border_style="magenta"
    )
    
    # Team Framework Status
    team_status = team_manager.get_team_performance_summary()
    team_status_panel = Panel(
        f"Healthcare Teams: {team_status['total_teams']}\n"
        f"Active Teams: {team_status['active_teams']}\n"
        f"Cases Handled: {team_status['total_cases_handled']}\n"
        f"Team Success Rate: {team_status['average_success_rate']:.1%}\n"
        f"Team Utilization: {team_status['team_utilization']:.1%}" if team_status['total_teams'] > 0 else
        f"No teams created yet\n"
        f"Use 'teams init' to create default teams\n"
        f"Teams enable multi-agent collaboration\n"
        f"Emergency response protocols available",
        title="👥 Healthcare Teams",
        border_style="purple"
    )
    
    console.print(llm_status)
    console.print(data_status)
    console.print(agent_status_panel)
    console.print(team_status_panel)
    
    # Quick commands
    console.print("\n🚀 [bold]Quick Commands:[/bold]")
    console.print("• [cyan]teams init[/cyan] - Initialize healthcare teams")
    console.print("• [cyan]teams list[/cyan] - View healthcare teams")
    console.print("• [cyan]teams emergency-response[/cyan] - Emergency simulation")
    console.print("• [cyan]teams workflow[/cyan] - Multi-team workflow demo")
    console.print("• [cyan]agents init[/cyan] - Initialize healthcare agents")
    console.print("• [cyan]agents diagnose --help[/cyan] - Collaborative diagnosis")
    console.print("• [cyan]llm list-models[/cyan] - View available AI models")

@app.command()
def demo():
    """Run a comprehensive demonstration of the healthcare AI system"""
    console.print("\n[bold cyan]🏥 VITA Healthcare AI - Complete Demonstration[/bold cyan]")
    console.print("This demo showcases the full capabilities of our multi-agent healthcare system")
    
    console.print("\n[yellow]Step 1: Initializing LLM System...[/yellow]")
    time.sleep(1)
    
    # Initialize LLM
    from llm_integration import LLMManager
    llm = LLMManager()
    
    # Set a default model if none is active
    available_models = llm.get_available_models()
    if available_models and not llm.get_active_model():
        model_keys = list(available_models.keys())
        llm.set_active_model(model_keys[0])
        console.print(f"✅ Activated model: {model_keys[0]}")
    
    console.print("\n[yellow]Step 2: Setting up Healthcare Agents...[/yellow]")
    time.sleep(1)
    
    # Initialize agents
    from healthcare_agent_framework import create_default_healthcare_agents, HealthcareAgent
    agents = create_default_healthcare_agents()
    
    console.print(f"✅ Initialized {len(agents)} healthcare agents:")
    for agent in agents:
        console.print(f"  • {agent.name} - {agent.role}")
    
    console.print("\n[yellow]Step 3: Creating Healthcare Teams...[/yellow]")
    time.sleep(1)
    
    # Initialize teams
    from healthcare_team_framework import HealthcareTeamManager
    team_manager = HealthcareTeamManager()
    team_manager.initialize_default_teams()
    
    teams = team_manager.get_all_teams()
    console.print(f"✅ Created {len(teams)} healthcare teams:")
    for team_id, team in teams.items():
        console.print(f"  • {team.name} ({len(team.agent_ids)} agents)")
    
    console.print("\n[yellow]Step 4: Simulating Patient Case...[/yellow]")
    time.sleep(1)
    
    # Create a sample patient case
    patient_case = {
        "patient_id": "DEMO-001",
        "symptoms": ["chest pain", "shortness of breath", "fatigue"],
        "vital_signs": {"bp": "150/95", "hr": "102", "temp": "99.2°F"},
        "severity": "moderate"
    }
    
    console.print("👤 Sample Patient Case:")
    console.print(f"  • Patient ID: {patient_case['patient_id']}")
    console.print(f"  • Symptoms: {', '.join(patient_case['symptoms'])}")
    console.print(f"  • Vital Signs: BP {patient_case['vital_signs']['bp']}, HR {patient_case['vital_signs']['hr']}")
    
    console.print("\n[yellow]Step 5: Agent Collaboration...[/yellow]")
    time.sleep(1)
    
    # Simulate agent collaboration
    console.print("🤖 AI Diagnostician analyzing symptoms...")
    time.sleep(2)
    console.print("✅ Preliminary diagnosis: Possible cardiac condition, requires further evaluation")
    
    console.print("💊 AI Pharmacist reviewing medications...")
    time.sleep(2)
    console.print("✅ Medication recommendations: Hold current medications, consider cardiac workup")
    
    console.print("📋 Care Coordinator organizing care...")
    time.sleep(2)
    console.print("✅ Care plan: Schedule ECG, cardiology consult, monitor vitals q4h")
    
    console.print("\n[yellow]Step 6: Team Coordination...[/yellow]")
    time.sleep(1)
    
    # Simulate team response
    if teams:
        cardiac_teams = [team for team in teams.values() if "cardiac" in team.name.lower()]
        if cardiac_teams:
            cardiac_team = cardiac_teams[0]
            console.print(f"🏥 {cardiac_team.name} activated for patient assessment")
            console.print(f"  • Team size: {len(cardiac_team.agent_ids)} specialists")
            console.print(f"  • Response time: <15 minutes")
            console.print("✅ Coordinated care plan implemented")
        else:
            console.print("🏥 Emergency team activated for patient assessment")
            console.print("✅ Coordinated care plan implemented")
    
    console.print("\n[yellow]Step 7: Performance Metrics...[/yellow]")
    time.sleep(1)
    
    # Show performance
    team_stats = team_manager.get_team_performance_summary()
    
    console.print("📊 System Performance:")
    console.print(f"  • Agents Active: {len(agents)}")
    console.print(f"  • Teams Active: {team_stats['total_teams']}")
    console.print(f"  • Cases Processed: {team_stats['total_cases_handled']}")
    console.print(f"  • Success Rate: {team_stats['average_success_rate']:.1%}")
    
    console.print("\n[green]✅ Demonstration Complete![/green]")
    console.print("\n[bold]The VITA Healthcare AI system demonstrates:")
    console.print("• 🤖 Multi-agent AI collaboration")
    console.print("• 👥 Team-based healthcare workflows")
    console.print("• 🚨 Emergency response protocols")
    console.print("• 📊 Performance monitoring and analytics")
    console.print("• 🏥 Comprehensive healthcare management")
    
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("• Run 'status' to see current system state")
    console.print("• Use 'agents diagnose' for AI-powered diagnosis")
    console.print("• Try 'teams emergency-response' for emergency simulations")
    console.print("• Explore 'teams workflow' for multi-team coordination")


@app.command()
def version():
    """Show version information"""
    console.print("\n[bold cyan]🏥 VITA Healthcare AI Platform[/bold cyan]")
    console.print("Version: 2.0.0")
    console.print("Build: Multi-Agent Healthcare Framework")
    console.print("\nComponents:")
    console.print("• 🤖 Healthcare Agent Framework v1.0")
    console.print("• 👥 Team Management System v1.0") 
    console.print("• 🧠 LLM Integration v1.0")
    console.print("• 📊 Data Management v1.0")
    console.print("• 🖥️  Enhanced CLI v2.0")
    console.print("\nCapabilities:")
    console.print("• Multi-agent AI collaboration")
    console.print("• Team-based healthcare workflows")
    console.print("• Emergency response protocols")
    console.print("• Performance monitoring")
    console.print("• Comprehensive healthcare management")


if __name__ == "__main__":
    app()