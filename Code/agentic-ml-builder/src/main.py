"""
=============================================================================
AGENTIC ML BUILDER - COMPLETE MAIN APPLICATION
=============================================================================
This is the main entry point that orchestrates all agents and components.

Directory structure:
agentic-ml-builder/
├── src/
│   ├── main.py                      ← YOU ARE HERE (Entry point)
│   ├── integrated_ml_builder.py     ← Core orchestrator
│   ├── agents/                      ← All agent implementations
│   ├── rag/                         ← RAG system
│   ├── mcp/                         ← MCP server
│   └── utils/                       ← Utilities
├── requirements.txt
├── config.yaml
└── .env
=============================================================================
"""

# =============================================================================
# FILE 1: src/main.py (Entry Point - Run this!)
# =============================================================================
 
import asyncio
import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from int_ml_builder import IntMLBuilder
from utils.file_utils import setup_logging

# Load environment variables
load_dotenv("env")

console = Console()

def print_banner():
    """Print application banner"""
    banner = """
    ============================================================
                                                               
        AGENTIC ML BUILDER v1.0.0                       
                                                               
       AI-Powered ML & MLOps Scaffolding Generator            
       Powered by OpenAI GPT-4o, RAG, MCP & Azure AI Foundry 
                                                               
    ============================================================
    """
    console.print(banner, style="cyan bold")


async def main():
    """
    Main execution function
    
    This function:
    1. Parses command-line arguments
    2. Sets up logging
    3. Initializes the IntegratedMLBuilder
    4. Executes the ML project generation workflow
    5. Optionally validates the generated code
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Agentic ML Builder - Generate complete ML projects from JSON specs"
    )
    parser.add_argument(
        "--input", 
        required=True, 
        help="Input JSON specification file (e.g., input/scene_dataset.json)"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Output directory for generated project"
    )
    parser.add_argument(
        "--mode", 
        default="local", 
        choices=["local", "azure"], 
        help="Execution mode: local or azure"
    )
    parser.add_argument(
        "--validate", 
        action="store_true", 
        help="Run validation after generation"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    # Print banner
    print_banner()
    
    # Display configuration
    console.print("\n[bold yellow]Configuration:[/bold yellow]")
    console.print(f"  [INPUT] Input:    {args.input}")
    console.print(f"  [OUTPUT] Output:   {args.output}")
    console.print(f"  [CONFIG] Mode:     {args.mode}")
    console.print(f"  [OK] Validate: {args.validate}")
    console.print()
    
    try:
        # Initialize the IntegratedMLBuilder
        # This is the main orchestrator that manages all agents
        logger.info("Initializing Agentic ML Builder...")
        console.print("[bold cyan]Initializing ML Builder...[/bold cyan]")
        
        builder = IntMLBuilder(
            input_path=args.input,
            output_path=args.output,
            mode=args.mode
        )
        
        # Execute the workflow
        # This runs all agents in sequence: Spec -> Scaffold -> Test -> MLOps
        console.print("\n[bold cyan]Executing ML project generation workflow...[/bold cyan]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Generating ML project...", total=None)
            
            # THIS IS THE MAIN CALL - Execute the complete workflow
            result = await builder.execute()
            
            progress.update(task, completed=True)
        
        # Check results
        if result["status"] == "success":
            console.print("\n[bold green][OK] ML Project generated successfully![/bold green]")
            console.print(f"\n[bold]Project Details:[/bold]")
            console.print(f"  [PROJECT] Project Name: {result['project_name']}")
            console.print(f"  [FOLDER] Location:     {result['output_path']}")
            
            # Optional validation
            if args.validate:
                console.print("\n[bold cyan]Running validation...[/bold cyan]")
                validation_result = await builder.validate()
                
                if validation_result['status'] == 'passed':
                    console.print("[bold green][OK] Validation passed![/bold green]")
                else:
                    console.print("[bold red][ERROR] Validation failed[/bold red]")
                    console.print(f"Details: {validation_result.get('results', {})}")
            
            # Show next steps
            console.print("\n[bold yellow]Next Steps:[/bold yellow]")
            console.print(f"  1. cd {result['output_path']}")
            console.print(f"  2. pip install -r requirements.txt")
            console.print(f"  3. python train.py")
            console.print()
            
        else:
            console.print(f"\n[bold red][ERROR] Generation failed:[/bold red] {result.get('error')}")
            logger.error(f"Generation failed: {result.get('error')}")
            sys.exit(1)
            
    except FileNotFoundError as e:
        console.print(f"\n[bold red][ERROR] File not found:[/bold red] {e}")
        logger.error(f"File not found: {e}")
        sys.exit(1)
        
    except Exception as e:
        console.print(f"\n[bold red][ERROR] Error:[/bold red] {str(e)}")
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    """
    HOW TO RUN THIS:
    
    1. Basic usage (Scene dataset):
       python src/main.py --input input/scene_dataset.json --output output
    
    2. With validation:
       python src/main.py --input input/scene_dataset.json --output output --validate
    
    3. Azure mode:
       python src/main.py --input input/scene_dataset.json --output output --mode azure
    
    4. Debug mode:
       python src/main.py --input input/scene_dataset.json --output output --debug
    """
    asyncio.run(main())


# =============================================================================
# FILE 2: src/integrated_ml_builder.py (Core Orchestrator)
# =============================================================================

"""
IntegratedMLBuilder - Core Orchestrator
This class manages the complete workflow and coordinates all agents.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import json

from agents.orchestration_agent import OrchestrationAgent
from agents.spec_agent import SpecAgent
from agents.scaffold_agent import ScaffoldAgent
from tests.test_agents import TestAgent
from agents.validator_agent import ValidatorAgent
from agents.mlops_agent import MLOpsAgent
from rag.rag_system import RAGSystem
from mcp1.mcp_server import MCPServe
from utils.file_utils import read_json, write_file

logger = logging.getLogger(__name__)


class IntegratedMLBuilder:
    """
    Main orchestrator for ML project generation
    
    This class:
    1. Initializes all agents and systems
    2. Executes the complete workflow
    3. Coordinates inter-agent communication
    4. Handles file I/O
    5. Manages validation
    """
    
    def __init__(self, input_path: str, output_path: str, mode: str = "local"):
        """
        Initialize the ML Builder
        
        Args:
            input_path: Path to input JSON specification
            output_path: Output directory for generated project
            mode: 'local' or 'azure' execution mode
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.mode = mode
        
        logger.info("=" * 70)
        logger.info("INITIALIZING AGENTIC ML BUILDER")
        logger.info("=" * 70)
        
        # Initialize RAG system (retrieves ML best practices)
        logger.info("Initializing RAG system...")
        self.rag_system = RAGSystem()
        
        # Initialize MCP server (inter-agent communication)
        logger.info("Initializing MCP server...")
        self.mcp_server = MCPServer()
        
        # Initialize all agents
        logger.info("Initializing agents...")
        self.orchestration_agent = OrchestrationAgent()
        self.spec_agent = SpecAgent(self.rag_system, self.mcp_server)
        self.scaffold_agent = ScaffoldAgent(self.rag_system, self.mcp_server)
        self.test_agent = TestAgent(self.rag_system, self.mcp_server)
        self.validator_agent = ValidatorAgent(self.mcp_server)
        self.mlops_agent = MLOpsAgent(self.rag_system, self.mcp_server)
        
        logger.info("✓ All components initialized successfully")
    
    async def execute(self) -> Dict[str, Any]:
        """
        Execute the complete ML project generation workflow
        
        Workflow Steps:
        1. Read input specification (JSON)
        2. Analyze with SpecAgent
        3. Generate code with ScaffoldAgent
        4. Generate tests with TestAgent
        5. Generate MLOps configs with MLOpsAgent
        6. Write all files to output
        7. Validate workflow with OrchestrationAgent
        
        Returns:
            Dict with status and project details
        """
        try:
            logger.info("=" * 70)
            logger.info("EXECUTING ML PROJECT GENERATION WORKFLOW")
            logger.info("=" * 70)
            
            # STEP 1: Read input specification
            logger.info("\n[STEP 1/7] Reading input specification...")
            spec = read_json(self.input_path)
            logger.info(f"✓ Loaded specification for: {spec.get('project_name', 'Unknown')}")
            
            # STEP 2: Analyze specification
            logger.info("\n[STEP 2/7] Analyzing specification with SpecAgent...")
            analyzed_spec = await self.spec_agent.analyze(spec)
            logger.info(f"✓ Analysis complete")
            logger.info(f"  - Task: {analyzed_spec.get('ml_task')}")
            logger.info(f"  - Models: {analyzed_spec.get('recommended_models')}")
            logger.info(f"  - Complexity: {analyzed_spec.get('estimated_complexity')}")
            
            # STEP 3: Generate ML scaffold
            logger.info("\n[STEP 3/7] Generating ML scaffold with ScaffoldAgent...")
            scaffold = await self.scaffold_agent.generate(analyzed_spec)
            logger.info(f"✓ Generated {len(scaffold.get('files', {}))} scaffold files")
            for filename in scaffold.get('files', {}).keys():
                logger.info(f"  - {filename}")
            
            # STEP 4: Generate tests
            logger.info("\n[STEP 4/7] Generating tests with TestAgent...")
            tests = await self.test_agent.generate(analyzed_spec, scaffold)
            logger.info(f"✓ Generated {len(tests.get('files', {}))} test files")
            for filename in tests.get('files', {}).keys():
                logger.info(f"  - {filename}")
            
            # STEP 5: Generate MLOps configuration
            logger.info("\n[STEP 5/7] Generating MLOps configuration with MLOpsAgent...")
            mlops_config = await self.mlops_agent.generate(analyzed_spec, scaffold)
            logger.info(f"✓ Generated {len(mlops_config.get('files', {}))} MLOps files")
            for filename in mlops_config.get('files', {}).keys():
                logger.info(f"  - {filename}")
            
            # STEP 6: Write output files
            logger.info("\n[STEP 6/7] Writing output files...")
            output_path = await self._write_output(
                analyzed_spec, scaffold, tests, mlops_config
            )
            logger.info(f"✓ All files written to: {output_path}")
            
            # STEP 7: Orchestration validation
            logger.info("\n[STEP 7/7] Validating workflow with OrchestrationAgent...")
            validation = await self.orchestration_agent.validate_workflow(
                spec, scaffold, tests, mlops_config
            )
            if validation.get('valid'):
                logger.info("✓ Workflow validation passed")
            else:
                logger.warning("⚠ Workflow validation had issues")
            
            logger.info("\n" + "=" * 70)
            logger.info("WORKFLOW COMPLETE")
            logger.info("=" * 70)
            
            return {
                "status": "success",
                "output_path": str(output_path),
                "project_name": analyzed_spec.get("project_name"),
                "files_generated": {
                    "scaffold": len(scaffold.get('files', {})),
                    "tests": len(tests.get('files', {})),
                    "mlops": len(mlops_config.get('files', {}))
                }
            }
            
        except Exception as e:
            logger.error(f"Execution failed: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def validate(self) -> Dict[str, Any]:
        """
        Validate the generated project
        
        Returns:
            Dict with validation results
        """
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING VALIDATION")
        logger.info("=" * 70)
        
        result = await self.validator_agent.validate(self.output_path)
        
        logger.info("\nValidation Results:")
        for component, result_data in result.get('results', {}).items():
            status = "✓" if result_data.get('passed') else "✗"
            logger.info(f"  {status} {component}: {'PASSED' if result_data.get('passed') else 'FAILED'}")
        
        return result
    
    async def _write_output(
        self, 
        spec: Dict, 
        scaffold: Dict, 
        tests: Dict, 
        mlops: Dict
    ) -> Path:
        """
        Write all generated files to output directory
        
        Args:
            spec: Analyzed specification
            scaffold: Generated scaffold files
            tests: Generated test files
            mlops: Generated MLOps files
            
        Returns:
            Path to output directory
        """
        # Create project directory
        project_name = spec.get("project_name", "ml_project")
        output_dir = self.output_path / project_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating project structure in: {output_dir}")
        
        # Write scaffold files (train.py, model.py, etc.)
        for filename, content in scaffold.get("files", {}).items():
            filepath = output_dir / filename
            write_file(filepath, content)
            logger.debug(f"  Wrote: {filename}")
        
        # Write test files
        tests_dir = output_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        for filename, content in tests.get("files", {}).items():
            if filename.startswith("../"):
                # Handle files that go in parent directory (like .github/workflows)
                filepath = output_dir / filename.replace("../", "")
                filepath.parent.mkdir(parents=True, exist_ok=True)
                write_file(filepath, content)
            else:
                write_file(tests_dir / filename, content)
            logger.debug(f"  Wrote: {filename}")
        
        # Write MLOps files
        for filename, content in mlops.get("files", {}).items():
            filepath = output_dir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            write_file(filepath, content)
            logger.debug(f"  Wrote: {filename}")
        
        # Write project metadata
        metadata = {
            "specification": spec,
            "generated_files": list(scaffold.get("files", {}).keys()),
            "test_files": list(tests.get("files", {}).keys()),
            "mlops_files": list(mlops.get("files", {}).keys()),
            "generation_mode": self.mode
        }
        write_file(
            output_dir / "project_metadata.json", 
            json.dumps(metadata, indent=2)
        )
        
        logger.info(f"✓ Project structure created successfully")
        
        return output_dir


# =============================================================================
# HOW TO USE THIS APPLICATION
# =============================================================================
"""
COMPLETE USAGE GUIDE:

1. SETUP (First time only):
   
   # Create virtual environment
   python -m venv venv
   
   # Activate (Windows PowerShell)
   .\\venv\\Scripts\\Activate.ps1
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Create .env file with your OpenAI API key
   echo "OPENAI_API_KEY=sk-your-key-here" > .env


2. RUN THE APPLICATION:

   # Basic usage - Generate ML project for Scene dataset
   python src/main.py --input input/scene_dataset.json --output output
   
   # With validation
   python src/main.py --input input/scene_dataset.json --output output --validate
   
   # For Lyrics dataset
   python src/main.py --input input/lyrics_dataset.json --output output
   
   # For Birds dataset
   python src/main.py --input input/birds_dataset.json --output output
   
   # Azure mode (requires Azure credentials)
   python src/main.py --input input/scene_dataset.json --output output --mode azure
   
   # Debug mode (verbose logging)
   python src/main.py --input input/scene_dataset.json --output output --debug


3. WHAT HAPPENS:

   The application will:
   ✓ Read your JSON specification
   ✓ Analyze requirements with AI
   ✓ Generate complete ML project code
   ✓ Create test suite
   ✓ Generate CI/CD configuration
   ✓ Create MLOps deployment configs
   ✓ Write everything to output directory
   ✓ (Optional) Validate the generated code


4. OUTPUT STRUCTURE:

   output/
   └── your_project_name/
       ├── train.py              # Training script
       ├── model.py              # Model definition
       ├── data_loader.py        # Data loading
       ├── config.py             # Configuration
       ├── requirements.txt      # Dependencies
       ├── README.md             # Documentation
       ├── Dockerfile            # Container
       ├── tests/                # Test suite
       │   ├── test_model.py
       │   ├── test_dataloader.py
       │   └── test_training.py
       ├── .github/
       │   └── workflows/
       │       └── ml_pipeline.yml  # CI/CD
       ├── azure_ml_config.yml   # Azure ML config
       └── project_metadata.json # Metadata


5. NEXT STEPS:

   cd output/your_project_name
   pip install -r requirements.txt
   python train.py


6. CUSTOMIZATION:

   - Edit input JSON specs to change project requirements
   - Modify config.yaml for system settings
   - Add custom templates to RAG system
   - Extend agents in src/agents/


That's it! The main.py file is your entry point.
Just run: python src/main.py --input <your-spec.json> --output output
"""
