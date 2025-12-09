"""Integrated ML Builder - Orchestrates all agents"""
import logging
from pathlib import Path
from typing import Dict, Any
import json

# Import all specialized agents
from agents.orchestration_agent import OrchestrationAgent
from agents.spec_agent import SpecAgent
from agents.scaffold_agent import ScaffoldAgent
from agents.test_agent import TestAgent
from agents.validator_agent import ValidatorAgent
from agents.mlops_agent import MLOpsAgent

# Import RAG system for context retrieval
from rag.rag_system import RAGSystem

# Import MCP server for external tool integration
from mcp1.mcp_server import MCPServe

# Import file utilities for I/O operations
from utils.file_utils import read_json, write_file

logger = logging.getLogger(__name__)

class IntMLBuilder:
    """
    Integrated ML Builder - Main orchestrator for the agentic ML generation system
    
    This class coordinates all specialized agents to transform a JSON specification
    into a complete, production-ready ML project with:
    - Project structure and code scaffolding
    - Comprehensive test suites
    - MLOps configuration (Docker, Azure ML, etc.)
    - Documentation and README files
    
    Workflow:
    1. Read and parse JSON specification
    2. Analyze specification with SpecAgent
    3. Generate ML project scaffold with ScaffoldAgent
    4. Generate test suite with TestAgent
    5. Generate MLOps configuration with MLOpsAgent
    6. Write all files to output directory
    7. Validate workflow completion with OrchestrationAgent
    """
    
    def __init__(self, input_path: str, output_path: str, mode: str = "local"):
        """
        Initialize the IntegratedMLBuilder with paths and components
        
        Args:
            input_path (str): Path to input JSON specification file
            output_path (str): Base directory for generated output files
            mode (str): Deployment mode - "local", "azure", or "foundry" (default: "local")
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.mode = mode
        
        logger.info("Initializing components...")
        
        # Initialize RAG system for retrieving ML best practices and templates
        self.rag_system = RAGSystem()
        
        # Initialize MCP server for external tool integration
        self.mcp_server = MCPServe()
        
        # Initialize all specialized agents
        # OrchestrationAgent: Coordinates workflow and validates outputs
        self.orchestration_agent = OrchestrationAgent()
        
        # SpecAgent: Analyzes and enhances project specifications
        self.spec_agent = SpecAgent(self.rag_system, self.mcp_server)
        
        # ScaffoldAgent: Generates project structure and code files
        self.scaffold_agent = ScaffoldAgent(self.rag_system, self.mcp_server)
        
        # TestAgent: Generates comprehensive test suites
        self.test_agent = TestAgent(self.rag_system, self.mcp_server)
        
        # ValidatorAgent: Validates generated code and configuration
        self.validator_agent = ValidatorAgent(self.mcp_server)
        
        # MLOpsAgent: Generates MLOps and deployment configuration
        self.mlops_agent = MLOpsAgent(self.rag_system, self.mcp_server)
    
    async def execute(self) -> Dict[str, Any]:
        """
        Execute the complete ML project generation workflow
        
        This method orchestrates the entire pipeline:
        1. Reads the input JSON specification
        2. Analyzes and enhances the specification
        3. Generates ML project scaffold
        4. Generates comprehensive tests
        5. Generates MLOps configuration
        6. Writes all files to disk
        7. Validates the complete workflow
        
        Returns:
            Dict[str, Any]: Result dictionary with:
                - status: "success" or "error"
                - output_path: Path to generated project
                - project_name: Name of generated project
                - error: (if status is "error") Error message
        """
        try:
            # Step 1: Read the JSON specification file
            logger.info("Step 1/7: Reading specification...")
            spec = read_json(self.input_path)
            
            # Step 2: Analyze specification and extract ML requirements
            # SpecAgent uses RAG to provide context and enhance the spec
            logger.info("Step 2/7: Analyzing specification...")
            analyzed_spec = await self.spec_agent.analyze(spec)
            
            # Step 3: Generate ML project scaffold
            # ScaffoldAgent creates training scripts, model files, data loaders, etc.
            logger.info("Step 3/7: Generating ML scaffold...")
            scaffold = await self.scaffold_agent.generate(analyzed_spec)
            
            # Step 4: Generate comprehensive test suite
            # TestAgent creates unit tests, integration tests, and test fixtures
            logger.info("Step 4/7: Generating tests...")
            tests = await self.test_agent.generate(analyzed_spec, scaffold)
            
            # Step 5: Generate MLOps and deployment configuration
            # MLOpsAgent creates Docker, Azure ML configs, monitoring setup, etc.
            logger.info("Step 5/7: Generating MLOps configuration...")
            mlops_config = await self.mlops_agent.generate(analyzed_spec, scaffold)
            
            # Step 6: Write all generated files to output directory
            logger.info("Step 6/7: Writing output files...")
            output_path = await self._write_output(analyzed_spec, scaffold, tests, mlops_config)
            
            # Step 7: Validate the complete workflow and generated artifacts
            logger.info("Step 7/7: Orchestration validation...")
            await self.orchestration_agent.validate_workflow(spec, scaffold, tests, mlops_config)
            
            # Return success with project details
            return {
                "status": "success",
                "output_path": str(output_path),
                "project_name": analyzed_spec.get("project_name")
            }
        except Exception as e:
            # Log and return error information
            logger.error(f"Execution failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def validate(self) -> Dict[str, Any]:
        """
        Validate the generated project output
        
        Runs comprehensive validation checks on generated code including:
        - Python syntax validation
        - Code quality checks
        - Configuration validation
        - Dockerfile validation (if Docker available)
        
        Returns:
            Dict[str, Any]: Validation results with status and details
        """
        return await self.validator_agent.validate(self.output_path)
    
    async def _write_output(self, spec: Dict, scaffold: Dict, tests: Dict, mlops: Dict) -> Path:
        """
        Write all generated files to the output directory
        
        Organizes generated content into proper directory structure:
        - Project root: Python code files (train.py, model.py, etc.)
        - tests/: Test files (test_model.py, test_dataloader.py, etc.)
        - Root level: Configuration and deployment files
        
        Args:
            spec (Dict): Enhanced specification from SpecAgent
            scaffold (Dict): Generated project files from ScaffoldAgent
            tests (Dict): Generated test files from TestAgent
            mlops (Dict): Generated MLOps files from MLOpsAgent
        
        Returns:
            Path: Path to the created project directory
        """
        # Create project directory with project name from specification
        output_dir = self.output_path / spec.get("project_name", "ml_project")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write scaffold files (training scripts, model definitions, etc.)
        # These are the core ML project files
        for filename, content in scaffold.get("files", {}).items():
            write_file(output_dir / filename, content)
        
        # Create tests directory and write test files
        tests_dir = output_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        for filename, content in tests.get("files", {}).items():
            # Handle relative paths in test files (e.g., conftest.py at root level)
            if filename.startswith("../"):
                filepath = output_dir / filename.replace("../", "")
                filepath.parent.mkdir(parents=True, exist_ok=True)
                write_file(filepath, content)
            else:
                write_file(tests_dir / filename, content)
        
        # Write MLOps and deployment configuration files
        # These include Dockerfile, Azure ML configs, monitoring setup, etc.
        for filename, content in mlops.get("files", {}).items():
            write_file(output_dir / filename, content)
        
        # Create project metadata file documenting all generated files
        # This serves as a manifest of the generated project structure
        metadata = {
            "specification": spec,
            "generated_files": list(scaffold.get("files", {}).keys()),
            "test_files": list(tests.get("files", {}).keys()),
            "mlops_files": list(mlops.get("files", {}).keys())
        }
        write_file(output_dir / "project_metadata.json", json.dumps(metadata, indent=2))
        
        return output_dir