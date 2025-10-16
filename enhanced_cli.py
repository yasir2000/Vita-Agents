"""
Enhanced CLI with LLM Integration and Sample Data
Extends the original CLI with LLM capabilities and realistic healthcare scenarios
"""

import typer
import json
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

app = typer.Typer(help="🏥 Vita Agents Healthcare AI Platform with LLM Integration")
console = Console()

# LLM Commands
llm_app = typer.Typer(help="🤖 LLM Integration Commands")
app.add_typer(llm_app, name="llm")

# Sample Data Commands  
data_app = typer.Typer(help="📊 Sample Data Management")
app.add_typer(data_app, name="data")

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
    
    console.print(llm_status)
    console.print(data_status)
    
    # Quick commands
    console.print("\n🚀 [bold]Quick Commands:[/bold]")
    console.print("• [cyan]llm list-models[/cyan] - View available AI models")
    console.print("• [cyan]llm test[/cyan] - Test active AI model")
    console.print("• [cyan]data generate[/cyan] - Create sample patient data")
    console.print("• [cyan]data list-patients[/cyan] - View sample patients")
    console.print("• [cyan]llm diagnose --help[/cyan] - AI diagnosis help")

if __name__ == "__main__":
    app()