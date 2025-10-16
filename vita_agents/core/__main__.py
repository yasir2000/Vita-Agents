"""
Main entry point for running the orchestrator directly.
Supports: python -m vita_agents.orchestrator
"""

import asyncio
import sys
from typing import Optional
import typer
from rich.console import Console
from rich import print as rprint

from vita_agents.core.orchestrator import AgentOrchestrator
from vita_agents.core.config import load_config, get_settings
from vita_agents.agents import (
    FHIRAgent, 
    HL7Agent, 
    EHRAgent, 
    ClinicalDecisionSupportAgent,
    DataHarmonizationAgent,
    ComplianceSecurityAgent,
    NaturalLanguageProcessingAgent
)


console = Console()


def create_default_agents(settings, database=None) -> list:
    """Create default set of agents."""
    # Mock database for now
    if database is None:
        database = {}
    
    agents = [
        FHIRAgent("fhir-agent-1", settings, database),
        HL7Agent("hl7-agent-1", settings, database),
        EHRAgent("ehr-agent-1", settings, database),
        ClinicalDecisionSupportAgent("clinical-agent-1", settings, database),
        DataHarmonizationAgent("harmonization-agent-1", settings, database),
        ComplianceSecurityAgent("compliance-agent-1", settings, database),
        NaturalLanguageProcessingAgent("nlp-agent-1", settings, database)
    ]
    
    return agents


async def run_orchestrator(
    config_file: Optional[str] = None,
    port: int = 8000,
    host: str = "0.0.0.0"
):
    """Run the agent orchestrator."""
    try:
        # Load configuration
        if config_file:
            settings = load_config(config_file)
        else:
            settings = get_settings()
        
        rprint("[bold green]Starting Vita Agents Orchestrator...[/bold green]")
        rprint(f"Configuration loaded from: {config_file or 'environment'}")
        
        # Create orchestrator
        orchestrator = AgentOrchestrator(settings)
        
        # Create and register default agents
        agents = create_default_agents(settings)
        
        for agent in agents:
            await orchestrator.register_agent(agent)
            rprint(f"✓ Registered agent: {agent.agent_id} ({agent.agent_type})")
        
        # Start orchestrator
        await orchestrator.start()
        rprint(f"[bold green]Orchestrator started with {len(agents)} agents[/bold green]")
        
        # Start API server (if available)
        try:
            import uvicorn
            from vita_agents.api.main import app
            
            rprint(f"[bold blue]Starting API server on {host}:{port}[/bold blue]")
            
            # Run API server
            config = uvicorn.Config(
                app=app,
                host=host,
                port=port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except ImportError:
            rprint("[yellow]API server not available (uvicorn not installed)[/yellow]")
            rprint("[yellow]Running orchestrator in CLI mode only[/yellow]")
            
            # Keep orchestrator running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                rprint("\n[yellow]Shutting down orchestrator...[/yellow]")
        
    except Exception as e:
        rprint(f"[bold red]Error starting orchestrator: {e}[/bold red]")
        sys.exit(1)
    finally:
        # Cleanup
        if 'orchestrator' in locals():
            await orchestrator.stop()
            rprint("[green]Orchestrator stopped[/green]")


def main():
    """Main entry point for CLI."""
    app = typer.Typer(help="Vita Agents Orchestrator")
    
    @app.command()
    def start(
        config: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
        port: int = typer.Option(8000, "--port", "-p", help="API server port"),
        host: str = typer.Option("0.0.0.0", "--host", "-h", help="API server host")
    ):
        """Start the orchestrator with agents."""
        asyncio.run(run_orchestrator(config, port, host))
    
    @app.command()
    def version():
        """Show version information."""
        rprint("Vita Agents v1.0.0")
        rprint("Multi-Agent AI Framework for Healthcare Interoperability")
    
    app()


if __name__ == "__main__":
    main()