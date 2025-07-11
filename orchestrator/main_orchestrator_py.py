#!/usr/bin/env python3
"""
Main entry point for the ML Processor MCP Orchestrator.
This serves as the interface that Claude will interact with.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import logging
from datetime import datetime

import structlog
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from orchestrator.workflow_orchestrator import WorkflowOrchestrator
from shared.schemas import WorkflowContext, WorkflowState


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
console = Console()


class MLProcessorCLI:
    """Command-line interface for the ML Processor"""
    
    def __init__(self, config_path: str = "config/mcp_config.json"):
        self.config_path = config_path
        self.config = {}
        self.orchestrator: Optional[WorkflowOrchestrator] = None
        
    def load_config(self) -> None:
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info("Configuration loaded", config_path=self.config_path)
        except FileNotFoundError:
            logger.error("Configuration file not found", path=self.config_path)
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in configuration", error=str(e))
            sys.exit(1)
    
    async def initialize(self) -> None:
        """Initialize the orchestrator"""
        self.load_config()
        
        rprint("[bold blue]🤖 Initializing ML Processor MCP System[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            init_task = progress.add_task("Connecting to MCP servers...", total=None)
            
            self.orchestrator = WorkflowOrchestrator(self.config)
            await self.orchestrator.initialize()
            
            progress.update(init_task, description="✅ System initialized successfully!")
        
        rprint("[bold green]System ready to process ML workflows![/bold green]")
    
    async def process_workflow(self, user_input: Dict[str, Any]) -> WorkflowContext:
        """Process a complete ML workflow"""
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not initialized")
        
        rprint(f"\n[bold cyan]🚀 Starting ML Workflow[/bold cyan]")
        rprint(f"📊 Problem: {user_input.get('prompt', 'Not specified')}")
        rprint(f"📁 Data file: {user_input.get('filepath', 'Not specified')}")
        
        # Start workflow with progress tracking
        with Progress(console=console) as progress:
            workflow_task = progress.add_task("Processing workflow...", total=100)
            
            # Start the workflow
            workflow_context = await self.orchestrator.start_workflow(user_input)
            
            # Monitor progress
            while workflow_context.current_state not in [WorkflowState.COMPLETED, WorkflowState.ERROR]:
                status = await self.orchestrator.get_workflow_status()
                progress.update(
                    workflow_task, 
                    completed=status['progress_percentage'],
                    description=f"[{status['current_state']}] {status['elapsed_time']}"
                )
                await asyncio.sleep(1)
            
            progress.update(workflow_task, completed=100, description="✅ Workflow completed!")
        
        return workflow_context
    
    def display_workflow_result(self, context: WorkflowContext) -> None:
        """Display workflow results in a nice format"""
        if context.current_state == WorkflowState.ERROR:
            rprint("[bold red]❌ Workflow failed with errors:[/bold red]")
            for error in context.errors:
                rprint(f"  • {error}")
            return
        
        rprint("\n[bold green]✅ Workflow completed successfully![/bold green]")
        
        # Create results table
        table = Table(title="ML Workflow Results", show_header=True, header_style="bold magenta")
        table.add_column("Stage", style="cyan")
        table.add_column("Result", style="green")
        table.add_column("Details", style="white")
        
        # Problem Classification
        if context.problem_classification:
            table.add_row(
                "Problem Type",
                context.problem_classification.problem_type,
                f"Confidence: {context.problem_classification.confidence:.2f}"
            )
        
        # Data Info
        ingestion_result = context.user_inputs.get('ingestion_result', {})
        if ingestion_result:
            data_summary = ingestion_result.get('data_summary', {})
            basic_info = data_summary.get('basic_info', {})
            shape = basic_info.get('shape', [0, 0])
            table.add_row(
                "Data Loaded", 
                f"{shape[0]} rows, {shape[1]} columns",
                f"Missing: {data_summary.get('missing_data', {}).get('missing_percentage', 0):.1f}%"
            )
        
        # Model Training
        training_result = context.user_inputs.get('training_result', {})
        if training_result:
            training_info = training_result.get('training_result', {})
            metrics = training_info.get('metrics', {})
            table.add_row(
                "Model Trained",
                training_info.get('model_type', 'Unknown'),
                f"Score: {metrics.get('primary_score', 'N/A')}"
            )
        
        # Export
        export_result = context.user_inputs.get('export_result', {})
        if export_result:
            export_package = export_result.get('export_package', {})
            table.add_row(
                "Export Package",
                "Generated",
                f"Path: {export_package.get('export_path', 'N/A')}"
            )
        
        console.print(table)
        
        # Display timing information
        rprint(f"\n⏱️  Total time: {self._format_duration(context.timestamps)}")
    
    def _format_duration(self, timestamps: Dict[str, datetime]) -> str:
        """Format workflow duration"""
        if 'start' not in timestamps or 'end' not in timestamps:
            return "Unknown"
        
        duration = timestamps['end'] - timestamps['start']
        total_seconds = duration.total_seconds()
        
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(seconds)}s"
    
    async def interactive_mode(self) -> None:
        """Interactive mode for testing"""
        rprint("[bold yellow]🔬 Interactive Mode - Enter ML problems to process[/bold yellow]")
        rprint("Type 'quit' to exit\n")
        
        while True:
            try:
                # Get user input
                prompt = console.input("[bold cyan]Describe your ML problem: [/bold cyan]")
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                filepath = console.input("[bold cyan]Path to your data file: [/bold cyan]")
                if not filepath or not os.path.exists(filepath):
                    rprint("[red]File not found. Please provide a valid file path.[/red]")
                    continue
                
                # Process workflow
                user_input = {
                    'prompt': prompt,
                    'filepath': filepath
                }
                
                context = await self.process_workflow(user_input)
                self.display_workflow_result(context)
                
                rprint("\n" + "="*60 + "\n")
                
            except KeyboardInterrupt:
                rprint("\n[yellow]Interrupted by user[/yellow]")
                break
            except Exception as e:
                logger.error("Error in interactive mode", error=str(e))
                rprint(f"[red]Error: {e}[/red]")
    
    async def process_single_request(self, prompt: str, filepath: str) -> None:
        """Process a single ML request"""
        user_input = {
            'prompt': prompt,
            'filepath': filepath
        }
        
        context = await self.process_workflow(user_input)
        self.display_workflow_result(context)
    
    async def show_status(self) -> None:
        """Show system status"""
        if not self.orchestrator:
            rprint("[red]Orchestrator not initialized[/red]")
            return
        
        status = await self.orchestrator.get_workflow_status()
        
        table = Table(title="System Status", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="white")
        
        # Server status
        for server_name in self.orchestrator.server_registry:
            server_info = self.orchestrator.server_registry[server_name]
            table.add_row(
                server_name,
                "Connected" if server_info['status'] == 'connected' else "Disconnected",
                f"Tools: {len(server_info['tools'])}, Resources: {len(server_info['resources'])}"
            )
        
        # Current workflow
        if status['status'] != 'no_active_workflow':
            table.add_row(
                "Active Workflow",
                status['current_state'],
                f"Progress: {status['progress_percentage']:.1f}%"
            )
        
        console.print(table)
    
    async def shutdown(self) -> None:
        """Shutdown the system"""
        if self.orchestrator:
            await self.orchestrator.shutdown()
        rprint("[bold yellow]👋 ML Processor shutdown complete[/bold yellow]")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ML Processor MCP System")
    parser.add_argument("--config", "-c", help="Configuration file path", 
                       default="config/mcp_config.json")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--prompt", "-p", help="ML problem description")
    parser.add_argument("--filepath", "-f", help="Path to data file")
    parser.add_argument("--status", "-s", action="store_true", 
                       help="Show system status")
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = MLProcessorCLI(args.config)
    
    try:
        # Initialize system
        await cli.initialize()
        
        if args.status:
            await cli.show_status()
        elif args.interactive:
            await cli.interactive_mode()
        elif args.prompt and args.filepath:
            await cli.process_single_request(args.prompt, args.filepath)
        else:
            rprint("[yellow]No action specified. Use --help for options.[/yellow]")
            rprint("[cyan]Tip: Use --interactive for interactive mode[/cyan]")
    
    except KeyboardInterrupt:
        rprint("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        logger.error("Fatal error", error=str(e))
        rprint(f"[red]Fatal error: {e}[/red]")
    finally:
        await cli.shutdown()


def claude_interface(prompt: str, filepath: str) -> Dict[str, Any]:
    """
    Main interface function for Claude to call.
    This is the primary entry point when Claude uses this system.
    """
    async def _run_workflow():
        cli = MLProcessorCLI()
        await cli.initialize()
        
        user_input = {
            'prompt': prompt,
            'filepath': filepath
        }
        
        context = await cli.process_workflow(user_input)
        await cli.shutdown()
        
        return context.dict()
    
    # Run the async workflow
    return asyncio.run(_run_workflow())


if __name__ == "__main__":
    asyncio.run(main())