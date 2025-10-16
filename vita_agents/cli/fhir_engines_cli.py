"""
Command Line Interface for Enhanced FHIR Agent
Provides easy management of FHIR engines and operations
"""

import asyncio
import json
import sys
from typing import Dict, Any, List, Optional
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax

from vita_agents.agents.enhanced_fhir_agent import EnhancedFHIRAgent
from vita_agents.fhir_engines.open_source_clients import (
    FHIREngineManager, FHIRServerConfiguration, FHIREngineType,
    get_server_template, list_server_templates
)
from vita_agents.config.fhir_engines_config import (
    get_config_for_environment, get_custom_server_config,
    get_performance_test_config, CUSTOM_SERVER_CONFIGS
)
from vita_agents.core.agent import TaskRequest

console = Console()


class FHIREnginesCLI:
    """CLI interface for FHIR engines management"""
    
    def __init__(self):
        self.agent = None
        self.engine_manager = FHIREngineManager()
    
    async def initialize_agent(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Enhanced FHIR Agent"""
        if not self.agent:
            self.agent = EnhancedFHIRAgent(config=config)
            await self.agent.start()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.agent:
            await self.agent.stop()
        await self.engine_manager.close_all_connections()


# CLI Commands
@click.group()
@click.option('--environment', '-e', default='development', 
              help='Environment (development, testing, production)')
@click.option('--config-file', '-c', help='Custom configuration file path')
@click.pass_context
def cli(ctx, environment, config_file):
    """Enhanced FHIR Agent CLI - Manage multiple open source FHIR engines"""
    ctx.ensure_object(dict)
    ctx.obj['environment'] = environment
    ctx.obj['config_file'] = config_file
    ctx.obj['cli'] = FHIREnginesCLI()


@cli.command()
@click.pass_context
def list_templates(ctx):
    """List available FHIR server templates"""
    templates = list_server_templates()
    
    if not templates:
        console.print("[yellow]No server templates available[/yellow]")
        return
    
    table = Table(title="Available FHIR Server Templates")
    table.add_column("Template ID", style="cyan")
    table.add_column("Engine Type", style="magenta")
    table.add_column("Name", style="green")
    table.add_column("Base URL", style="blue")
    table.add_column("FHIR Version", style="yellow")
    
    for template_id in templates:
        template = get_server_template(template_id)
        if template:
            table.add_row(
                template_id,
                template.engine_type.value,
                template.name,
                template.base_url,
                template.fhir_version.value
            )
    
    console.print(table)


@cli.command()
@click.argument('template_id')
@click.pass_context
def show_template(ctx, template_id):
    """Show details of a specific FHIR server template"""
    template = get_server_template(template_id)
    
    if not template:
        console.print(f"[red]Template '{template_id}' not found[/red]")
        return
    
    # Convert to dictionary for display
    template_dict = {
        "server_id": template.server_id,
        "name": template.name,
        "engine_type": template.engine_type.value,
        "base_url": template.base_url,
        "fhir_version": template.fhir_version.value,
        "authentication": template.authentication.dict() if template.authentication else None,
        "headers": template.headers,
        "description": template.description
    }
    
    # Pretty print JSON
    json_str = json.dumps(template_dict, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
    
    panel = Panel(syntax, title=f"Template: {template_id}", expand=False)
    console.print(panel)


@cli.command()
@click.argument('template_id')
@click.option('--timeout', default=30, help='Connection timeout in seconds')
@click.pass_context
def connect(ctx, template_id, timeout):
    """Connect to a FHIR engine using a template"""
    
    async def _connect():
        cli_obj = ctx.obj['cli']
        config = get_config_for_environment(ctx.obj['environment'])
        await cli_obj.initialize_agent(config)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Connecting to {template_id}...", total=None)
            
            # Create task request
            task_request = TaskRequest(
                task_id=f"connect_{template_id}",
                agent_id="enhanced-fhir-agent",
                task_type="connect_fhir_engine",
                parameters={
                    "template_name": template_id
                }
            )
            
            # Execute connection
            response = await cli_obj.agent.process_task(task_request)
            progress.update(task, completed=100)
        
        if response.success:
            console.print(f"[green]✓ Successfully connected to {template_id}[/green]")
            
            # Show connection details
            if response.data:
                table = Table(title="Connection Details")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="white")
                
                for key, value in response.data.items():
                    if key != "capabilities":  # Skip large capabilities object
                        table.add_row(key, str(value))
                
                console.print(table)
        else:
            console.print(f"[red]✗ Failed to connect to {template_id}[/red]")
            console.print(f"[red]Error: {response.error_message}[/red]")
        
        await cli_obj.cleanup()
    
    asyncio.run(_connect())


@cli.command()
@click.pass_context
def list_engines(ctx):
    """List all connected FHIR engines"""
    
    async def _list_engines():
        cli_obj = ctx.obj['cli']
        config = get_config_for_environment(ctx.obj['environment'])
        await cli_obj.initialize_agent(config)
        
        task_request = TaskRequest(
            task_id="list_engines",
            agent_id="enhanced-fhir-agent",
            task_type="list_engines",
            parameters={}
        )
        
        response = await cli_obj.agent.process_task(task_request)
        
        if response.success and response.data:
            connected_servers = response.data.get("connected_servers", [])
            
            if not connected_servers:
                console.print("[yellow]No FHIR engines currently connected[/yellow]")
            else:
                table = Table(title="Connected FHIR Engines")
                table.add_column("Server ID", style="cyan")
                table.add_column("Engine Type", style="magenta")
                table.add_column("Base URL", style="blue")
                table.add_column("FHIR Version", style="yellow")
                table.add_column("Status", style="green")
                
                for server in connected_servers:
                    table.add_row(
                        server.get("server_id", "N/A"),
                        server.get("engine_type", "N/A"),
                        server.get("base_url", "N/A"),
                        server.get("fhir_version", "N/A"),
                        "Connected"
                    )
                
                console.print(table)
                console.print(f"\n[green]Total connected engines: {len(connected_servers)}[/green]")
        else:
            console.print("[red]Failed to list engines[/red]")
            if response.error_message:
                console.print(f"[red]Error: {response.error_message}[/red]")
        
        await cli_obj.cleanup()
    
    asyncio.run(_list_engines())


@cli.command()
@click.argument('resource_type')
@click.option('--parameters', '-p', help='Search parameters as JSON string')
@click.option('--engines', '-e', help='Comma-separated list of engine IDs')
@click.option('--count', default=10, help='Number of resources to return')
@click.pass_context
def search(ctx, resource_type, parameters, engines, count):
    """Search for resources across multiple FHIR engines"""
    
    async def _search():
        cli_obj = ctx.obj['cli']
        config = get_config_for_environment(ctx.obj['environment'])
        await cli_obj.initialize_agent(config)
        
        # Parse parameters
        search_params = {}
        if parameters:
            try:
                search_params = json.loads(parameters)
            except json.JSONDecodeError:
                console.print("[red]Invalid JSON in parameters[/red]")
                return
        
        # Parse engines list
        engine_list = None
        if engines:
            engine_list = [e.strip() for e in engines.split(',')]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Searching {resource_type}...", total=None)
            
            task_request = TaskRequest(
                task_id=f"search_{resource_type}",
                agent_id="enhanced-fhir-agent",
                task_type="multi_engine_search",
                parameters={
                    "resource_type": resource_type,
                    "search_parameters": search_params,
                    "count": count,
                    "engines": engine_list
                }
            )
            
            response = await cli_obj.agent.process_task(task_request)
            progress.update(task, completed=100)
        
        if response.success and response.data:
            console.print(f"[green]✓ Search completed[/green]")
            
            # Show summary
            data = response.data
            table = Table(title="Search Results Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Resource Type", data.get("resource_type", "N/A"))
            table.add_row("Total Engines", str(data.get("total_engines", 0)))
            table.add_row("Successful Engines", str(data.get("successful_engines", 0)))
            table.add_row("Total Resources Found", str(data.get("total_resources_found", 0)))
            table.add_row("Execution Time (ms)", str(data.get("execution_time_ms", 0)))
            
            console.print(table)
            
            # Show per-engine results
            results = data.get("results", {})
            if results:
                console.print("\n[bold]Per-Engine Results:[/bold]")
                
                for engine_id, result in results.items():
                    success_icon = "✓" if result.get("success") else "✗"
                    color = "green" if result.get("success") else "red"
                    
                    console.print(f"[{color}]{success_icon} {engine_id}[/{color}]")
                    
                    if result.get("success") and result.get("data"):
                        resource_count = 0
                        if "entry" in result["data"]:
                            resource_count = len(result["data"]["entry"])
                        console.print(f"  Resources found: {resource_count}")
                        console.print(f"  Response time: {result.get('execution_time_ms', 0)}ms")
                    elif result.get("error_message"):
                        console.print(f"  Error: {result['error_message']}")
        else:
            console.print("[red]✗ Search failed[/red]")
            if response.error_message:
                console.print(f"[red]Error: {response.error_message}[/red]")
        
        await cli_obj.cleanup()
    
    asyncio.run(_search())


@cli.command()
@click.option('--load-type', default='light_load', 
              type=click.Choice(['light_load', 'medium_load', 'heavy_load']),
              help='Performance test load type')
@click.pass_context
def performance_test(ctx, load_type):
    """Run performance tests across FHIR engines"""
    
    async def _performance_test():
        cli_obj = ctx.obj['cli']
        config = get_config_for_environment(ctx.obj['environment'])
        await cli_obj.initialize_agent(config)
        
        test_config = get_performance_test_config(load_type)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running {load_type} performance test...", total=None)
            
            task_request = TaskRequest(
                task_id="performance_test",
                agent_id="enhanced-fhir-agent",
                task_type="engine_performance_analysis",
                parameters={
                    "operation_type": "search",
                    "resource_type": "Patient",
                    "sample_size": test_config["sample_size"]
                }
            )
            
            response = await cli_obj.agent.process_task(task_request)
            progress.update(task, completed=100)
        
        if response.success and response.data:
            console.print(f"[green]✓ Performance test completed[/green]")
            
            # Show results
            data = response.data
            performance_results = data.get("performance_results", {})
            
            if performance_results:
                table = Table(title="Performance Test Results")
                table.add_column("Engine", style="cyan")
                table.add_column("Avg Response (ms)", style="yellow")
                table.add_column("Min Response (ms)", style="green")
                table.add_column("Max Response (ms)", style="red")
                table.add_column("Success Rate", style="blue")
                
                for engine_id, metrics in performance_results.items():
                    success_rate = (metrics.get("success_count", 0) / 
                                  test_config["sample_size"] * 100) if test_config["sample_size"] > 0 else 0
                    
                    table.add_row(
                        engine_id,
                        f"{metrics.get('avg_response_time', 0):.2f}",
                        f"{metrics.get('min_response_time', 0):.2f}",
                        f"{metrics.get('max_response_time', 0):.2f}",
                        f"{success_rate:.1f}%"
                    )
                
                console.print(table)
                
                # Show recommendations
                recommendations = data.get("recommendations", [])
                if recommendations:
                    console.print("\n[bold]Recommendations:[/bold]")
                    for rec in recommendations:
                        console.print(f"• {rec}")
                
                fastest_engine = data.get("fastest_engine")
                if fastest_engine:
                    console.print(f"\n[green]🏆 Fastest engine: {fastest_engine}[/green]")
        else:
            console.print("[red]✗ Performance test failed[/red]")
            if response.error_message:
                console.print(f"[red]Error: {response.error_message}[/red]")
        
        await cli_obj.cleanup()
    
    asyncio.run(_performance_test())


@cli.command()
@click.argument('resource_file', type=click.File('r'))
@click.option('--engines', '-e', help='Comma-separated list of engine IDs')
@click.pass_context
def validate(ctx, resource_file, engines):
    """Validate a FHIR resource across multiple engines"""
    
    async def _validate():
        cli_obj = ctx.obj['cli']
        config = get_config_for_environment(ctx.obj['environment'])
        await cli_obj.initialize_agent(config)
        
        try:
            resource = json.load(resource_file)
        except json.JSONDecodeError:
            console.print("[red]Invalid JSON in resource file[/red]")
            return
        
        resource_type = resource.get("resourceType")
        if not resource_type:
            console.print("[red]Resource must have a resourceType[/red]")
            return
        
        # Parse engines list
        engine_list = None
        if engines:
            engine_list = [e.strip() for e in engines.split(',')]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),  
            console=console,
        ) as progress:
            task = progress.add_task(f"Validating {resource_type}...", total=None)
            
            task_request = TaskRequest(
                task_id="validate_resource",
                agent_id="enhanced-fhir-agent",
                task_type="validate_across_engines",
                parameters={
                    "resource": resource,
                    "resource_type": resource_type,
                    "engines": engine_list
                }
            )
            
            response = await cli_obj.agent.process_task(task_request)
            progress.update(task, completed=100)
        
        if response.success and response.data:
            data = response.data
            console.print(f"[green]✓ Validation completed[/green]")
            
            # Show summary
            consensus_valid = data.get("consensus_valid", False)
            if consensus_valid:
                console.print("[green]🎉 Resource is valid across all engines![/green]")
            else:
                console.print("[yellow]⚠️  Resource validation differs between engines[/yellow]")
            
            # Show per-engine results
            validation_results = data.get("validation_results", {})
            if validation_results:
                table = Table(title="Validation Results")
                table.add_column("Engine", style="cyan")
                table.add_column("Valid", style="green")
                table.add_column("Response Time (ms)", style="yellow")
                table.add_column("Issues", style="red")
                
                for engine_id, result in validation_results.items():
                    valid_icon = "✓" if result.get("valid") else "✗"
                    valid_color = "green" if result.get("valid") else "red"
                    
                    issues_count = 0
                    if result.get("operation_outcome") and "issue" in result["operation_outcome"]:
                        issues_count = len(result["operation_outcome"]["issue"])
                    
                    table.add_row(
                        engine_id,
                        f"[{valid_color}]{valid_icon}[/{valid_color}]",
                        str(result.get("execution_time_ms", 0)),
                        str(issues_count) if issues_count > 0 else "-"
                    )
                
                console.print(table)
            
            # Show engine differences
            differences = data.get("engine_differences", [])
            if differences:
                console.print("\n[bold]Validation Differences:[/bold]")
                for diff in differences:
                    engine = diff.get("engine", "Unknown")
                    issues = diff.get("issues", [])
                    console.print(f"\n[yellow]{engine}:[/yellow]")
                    for issue in issues:
                        severity = issue.get("severity", "unknown")
                        code = issue.get("code", "unknown")
                        diagnostics = issue.get("diagnostics", "No details")
                        console.print(f"  • [{severity.upper()}] {code}: {diagnostics}")
        else:
            console.print("[red]✗ Validation failed[/red]")
            if response.error_message:
                console.print(f"[red]Error: {response.error_message}[/red]")
        
        await cli_obj.cleanup()
    
    asyncio.run(_validate())


@cli.command()
@click.pass_context
def config(ctx):
    """Show current configuration"""
    config = get_config_for_environment(ctx.obj['environment'])
    
    # Pretty print configuration
    json_str = json.dumps(config, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
    
    panel = Panel(
        syntax, 
        title=f"Configuration ({ctx.obj['environment']})", 
        expand=False
    )
    console.print(panel)


@cli.command()
@click.pass_context  
def info(ctx):
    """Show information about the Enhanced FHIR Agent"""
    
    info_text = """
[bold cyan]Enhanced FHIR Agent[/bold cyan]
Version: 2.0.0

[bold]Supported FHIR Engines:[/bold]
• HAPI FHIR Server (hapifhir.io)
• IBM FHIR Server (github.com/IBM/FHIR)
• Firely .NET SDK (fire.ly)
• Medplum FHIR Server (medplum.com)  
• Spark FHIR Server (github.com/FirelyTeam/spark)
• LinuxForHealth FHIR Server
• Aidbox FHIR Platform
• And more...

[bold]Key Features:[/bold]
• Multi-engine operations (parallel execution)
• Performance analysis and benchmarking
• Cross-engine validation and migration
• SMART on FHIR and OAuth2 support
• Comprehensive CLI management tools

[bold]Supported FHIR Versions:[/bold]
• DSTU2, STU3, R4, R5

[bold]Authentication Types:[/bold]
• None (open servers)
• Basic Authentication
• OAuth2 / Bearer Token
• SMART on FHIR

[bold]Available Commands:[/bold]
Use 'fhir-engines --help' to see all available commands.
    """
    
    panel = Panel(info_text, title="Enhanced FHIR Agent Information", expand=False)
    console.print(panel)


if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)