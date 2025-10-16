"""
Command-line interface for Vita Agents.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich import print as rprint
import yaml

from vita_agents.core.orchestrator import AgentOrchestrator, get_orchestrator
from vita_agents.core.config import get_settings, load_config
from vita_agents.agents import FHIRAgent, HL7Agent, EHRAgent
from vita_agents.core.agent import TaskRequest, WorkflowDefinition, WorkflowStep


app = typer.Typer(help="Vita Agents - Multi-Agent AI Framework for Healthcare Interoperability")
console = Console()

# Global state
orchestrator: Optional[AgentOrchestrator] = None


@app.callback()
def main(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """Vita Agents CLI - Healthcare AI Multi-Agent Framework."""
    global orchestrator
    
    if config_file and config_file.exists():
        settings = load_config(config_file)
    else:
        settings = get_settings()
    
    if debug:
        settings.debug = True
        settings.monitoring.log_level = "DEBUG"
    
    orchestrator = get_orchestrator()


@app.command()
def start(
    agents: Optional[List[str]] = typer.Option(None, "--agent", "-a", help="Specific agents to start"),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run as daemon"),
):
    """Start the agent orchestrator and agents."""
    async def _start():
        try:
            console.print("[green]Starting Vita Agents orchestrator...[/green]")
            
            # Register agent types
            orchestrator.register_agent_type("fhir", FHIRAgent)
            orchestrator.register_agent_type("hl7", HL7Agent)
            orchestrator.register_agent_type("ehr", EHRAgent)
            
            # Create and register agents
            if not agents:
                agents_to_create = ["fhir", "hl7", "ehr"]
            else:
                agents_to_create = agents
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Creating agents...", total=len(agents_to_create))
                
                for agent_type in agents_to_create:
                    progress.update(task, description=f"Creating {agent_type} agent...")
                    await orchestrator.create_agent(agent_type)
                    progress.advance(task)
            
            # Start orchestrator
            await orchestrator.start()
            
            console.print("[green]✓ Orchestrator and agents started successfully![/green]")
            
            # Show status
            status = orchestrator.get_agent_status()
            _display_agent_status(status)
            
            if daemon:
                console.print("[yellow]Running in daemon mode. Press Ctrl+C to stop.[/yellow]")
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Shutting down...[/yellow]")
                    await orchestrator.stop()
            
        except Exception as e:
            console.print(f"[red]Error starting orchestrator: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(_start())


@app.command()
def stop():
    """Stop the agent orchestrator."""
    async def _stop():
        if orchestrator:
            console.print("[yellow]Stopping orchestrator...[/yellow]")
            await orchestrator.stop()
            console.print("[green]✓ Orchestrator stopped[/green]")
        else:
            console.print("[red]No orchestrator instance found[/red]")
    
    asyncio.run(_stop())


@app.command()
def status():
    """Show status of agents and workflows."""
    if not orchestrator:
        console.print("[red]Orchestrator not started. Run 'vita-agents start' first.[/red]")
        return
    
    # Agent status
    agent_status = orchestrator.get_agent_status()
    _display_agent_status(agent_status)
    
    # Workflow status
    workflow_status = orchestrator.get_workflow_status()
    _display_workflow_status(workflow_status)


@app.command()
def agents():
    """List and manage agents."""
    if not orchestrator:
        console.print("[red]Orchestrator not started. Run 'vita-agents start' first.[/red]")
        return
    
    status = orchestrator.get_agent_status()
    _display_agent_status(status)


@app.command()
def task(
    agent_id: str = typer.Argument(..., help="Agent ID to send task to"),
    task_type: str = typer.Argument(..., help="Type of task to execute"),
    parameters_file: Optional[Path] = typer.Option(None, "--params", "-p", help="JSON file with task parameters"),
    parameters: Optional[str] = typer.Option(None, "--json", help="JSON string with task parameters"),
):
    """Send a task to a specific agent."""
    async def _send_task():
        if not orchestrator:
            console.print("[red]Orchestrator not started. Run 'vita-agents start' first.[/red]")
            return
        
        # Parse parameters
        task_params = {}
        if parameters_file and parameters_file.exists():
            with open(parameters_file) as f:
                task_params = json.load(f)
        elif parameters:
            task_params = json.loads(parameters)
        
        # Create task request
        task_request = TaskRequest(
            task_type=task_type,
            parameters=task_params
        )
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task_progress = progress.add_task(f"Executing {task_type} on {agent_id}...", total=1)
                
                # Send task
                result = await orchestrator.send_task_to_agent(agent_id, task_request)
                progress.advance(task_progress)
            
            # Display result
            console.print(f"[green]✓ Task completed successfully[/green]")
            console.print("\n[bold]Result:[/bold]")
            
            result_json = json.dumps(result.dict(), indent=2)
            syntax = Syntax(result_json, "json", theme="monokai", line_numbers=True)
            console.print(syntax)
            
        except Exception as e:
            console.print(f"[red]Task execution failed: {e}[/red]")
    
    asyncio.run(_send_task())


@app.command()
def workflow(
    workflow_file: Path = typer.Argument(..., help="Workflow definition file (YAML/JSON)"),
    input_file: Optional[Path] = typer.Option(None, "--input", "-i", help="Input data file"),
    input_data: Optional[str] = typer.Option(None, "--data", help="Input data as JSON string"),
):
    """Execute a workflow."""
    async def _execute_workflow():
        if not orchestrator:
            console.print("[red]Orchestrator not started. Run 'vita-agents start' first.[/red]")
            return
        
        # Load workflow definition
        try:
            with open(workflow_file) as f:
                if workflow_file.suffix.lower() in ['.yaml', '.yml']:
                    workflow_data = yaml.safe_load(f)
                else:
                    workflow_data = json.load(f)
            
            workflow_def = WorkflowDefinition(**workflow_data)
            
        except Exception as e:
            console.print(f"[red]Failed to load workflow: {e}[/red]")
            return
        
        # Parse input data
        workflow_input = {}
        if input_file and input_file.exists():
            with open(input_file) as f:
                workflow_input = json.load(f)
        elif input_data:
            workflow_input = json.loads(input_data)
        
        try:
            # Register workflow
            orchestrator.register_workflow(workflow_def)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task_progress = progress.add_task(f"Executing workflow '{workflow_def.name}'...", total=1)
                
                # Execute workflow
                execution = await orchestrator.execute_workflow(
                    workflow_def.id, 
                    workflow_input
                )
                
                # Wait for completion (simplified)
                await asyncio.sleep(2)  # Give it time to process
                progress.advance(task_progress)
            
            # Show execution status
            console.print(f"[green]✓ Workflow '{workflow_def.name}' started[/green]")
            console.print(f"[blue]Execution ID: {execution.id}[/blue]")
            
            # Display execution details
            execution_json = json.dumps(execution.dict(), indent=2, default=str)
            syntax = Syntax(execution_json, "json", theme="monokai", line_numbers=True)
            console.print(syntax)
            
        except Exception as e:
            console.print(f"[red]Workflow execution failed: {e}[/red]")
    
    asyncio.run(_execute_workflow())


@app.command()
def validate(
    file_path: Path = typer.Argument(..., help="File to validate"),
    data_type: str = typer.Option("auto", "--type", "-t", help="Data type (fhir, hl7, auto)"),
    version: Optional[str] = typer.Option(None, "--version", "-v", help="Standard version"),
):
    """Validate healthcare data files."""
    async def _validate():
        if not orchestrator:
            console.print("[red]Orchestrator not started. Run 'vita-agents start' first.[/red]")
            return
        
        try:
            with open(file_path) as f:
                content = f.read()
            
            # Auto-detect data type
            if data_type == "auto":
                if file_path.suffix.lower() == ".json":
                    try:
                        data = json.loads(content)
                        if "resourceType" in data:
                            data_type = "fhir"
                    except:
                        pass
                elif content.startswith("MSH|"):
                    data_type = "hl7"
            
            # Validate based on type
            if data_type == "fhir":
                agent_id = "fhir-agent"
                task_type = "validate_fhir_resource"
                parameters = {
                    "resource": json.loads(content) if content.startswith("{") else content,
                    "version": version or "R4"
                }
            elif data_type == "hl7":
                agent_id = "hl7-agent"
                task_type = "validate_hl7_message"
                parameters = {
                    "message": content,
                    "version": version or "2.8"
                }
            else:
                console.print(f"[red]Unsupported data type: {data_type}[/red]")
                return
            
            # Create and send validation task
            task_request = TaskRequest(
                task_type=task_type,
                parameters=parameters
            )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task_progress = progress.add_task(f"Validating {data_type.upper()} data...", total=1)
                
                result = await orchestrator.send_task_to_agent(agent_id, task_request)
                progress.advance(task_progress)
            
            # Display validation results
            validation_result = result.result.get("validation_result", {})
            
            if validation_result.get("is_valid", False):
                console.print(f"[green]✓ {data_type.upper()} data is valid[/green]")
            else:
                console.print(f"[red]✗ {data_type.upper()} data is invalid[/red]")
            
            # Show errors and warnings
            errors = validation_result.get("errors", [])
            warnings = validation_result.get("warnings", [])
            
            if errors:
                console.print("\n[red]Errors:[/red]")
                for error in errors:
                    console.print(f"  • {error}")
            
            if warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"  • {warning}")
            
        except Exception as e:
            console.print(f"[red]Validation failed: {e}[/red]")
    
    asyncio.run(_validate())


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    generate: bool = typer.Option(False, "--generate", help="Generate example configuration"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for generated config"),
):
    """Manage configuration."""
    if show:
        settings = get_settings()
        config_dict = settings.dict()
        
        console.print("[bold]Current Configuration:[/bold]")
        config_json = json.dumps(config_dict, indent=2, default=str)
        syntax = Syntax(config_json, "json", theme="monokai", line_numbers=True)
        console.print(syntax)
    
    elif generate:
        example_config = {
            "environment": "development",
            "debug": False,
            "database": {
                "url": "postgresql://vita_user:vita_password@localhost:5432/vita_agents"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            },
            "agents": {
                "max_concurrent_agents": 10,
                "workflow_engine": "crewai"
            },
            "healthcare": {
                "fhir_version": "R4",
                "hl7_version": "2.8"
            },
            "security": {
                "hipaa_compliance": True,
                "audit_log_enabled": True
            }
        }
        
        if output:
            with open(output, 'w') as f:
                yaml.dump(example_config, f, default_flow_style=False)
            console.print(f"[green]✓ Example configuration written to {output}[/green]")
        else:
            console.print("[bold]Example Configuration:[/bold]")
            config_yaml = yaml.dump(example_config, default_flow_style=False)
            syntax = Syntax(config_yaml, "yaml", theme="monokai", line_numbers=True)
            console.print(syntax)


def _display_agent_status(status: Dict[str, Any]):
    """Display agent status in a table."""
    table = Table(title="Agent Status")
    table.add_column("Agent ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Tasks Completed", style="blue")
    table.add_column("Tasks Failed", style="red")
    table.add_column("Avg Time (ms)", style="magenta")
    
    for agent_info in status.get("agents", []):
        metrics = agent_info.get("metrics", {})
        table.add_row(
            agent_info.get("agent_id", "unknown"),
            agent_info.get("name", "unknown"),
            agent_info.get("status", "unknown"),
            str(metrics.get("tasks_completed", 0)),
            str(metrics.get("tasks_failed", 0)),
            f"{metrics.get('average_execution_time_ms', 0):.1f}"
        )
    
    console.print(table)
    
    # Summary
    total_agents = status.get("total_agents", 0)
    active_agents = status.get("active_agents", 0)
    
    summary = Panel(
        f"[green]Active Agents:[/green] {active_agents}/{total_agents}\n"
        f"[blue]Total Load:[/blue] {status.get('total_load', 0)}",
        title="Summary",
        border_style="blue"
    )
    console.print(summary)


def _display_workflow_status(status: Dict[str, Any]):
    """Display workflow status."""
    executions = status.get("executions", [])
    
    if not executions:
        console.print("[yellow]No workflow executions found[/yellow]")
        return
    
    table = Table(title="Workflow Executions")
    table.add_column("Execution ID", style="cyan")
    table.add_column("Workflow ID", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Started", style="blue")
    table.add_column("Completed", style="magenta")
    
    for execution in executions:
        table.add_row(
            execution.get("id", "unknown")[:8],
            execution.get("workflow_id", "unknown"),
            execution.get("status", "unknown"),
            execution.get("started_at", "")[:19],
            execution.get("completed_at", "")[:19] if execution.get("completed_at") else "N/A"
        )
    
    console.print(table)


if __name__ == "__main__":
    app()