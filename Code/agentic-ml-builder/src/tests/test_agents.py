"""
test_agents.py

Test suite for Agentic ML Builder.

Contains unit and integration tests for the project's agent components
and the end-to-end ML project generation pipeline, including:
- SpecAgent
- ScaffoldAgent
- ValidatorAgent
- MLOpsAgent
- OrchestrationAgent
- RAGSystem
- MCP server integrations

This file validates core agent behaviors and integration scenarios used
during development and CI.
"""
import pytest
import asyncio
from pathlib import Path
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.spec_agent import SpecAgent
from agents.scaffold_agent import ScaffoldAgent
#from tests.test_agents import TestAgent
from agents.validator_agent import ValidatorAgent
from agents.mlops_agent import MLOpsAgent
from agents.orchestration_agent import OrchestrationAgent
from rag.rag_system import RAGSystem
from mcp1.mcp_server import MCPServe
#from int_ml_builder import IntMLBuilder
from utils.file_utils import read_json, write_json


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def rag_system():
    """Create RAG system instance"""
    return RAGSystem(collection_name="test_ml_templates")


@pytest.fixture
def mcp_server():
    """Create MCP server instance"""
    return MCPServe(port=8081)


@pytest.fixture
def sample_spec():
    """Sample project specification"""
    return {
        "project_name": "test_classifier",
        "purpose": "Test classification project",
        "data_type": "tabular",
        "ml_task": "classification",
        "target_environment": "local",
        "models": ["mlp", "random_forest"]
    }


@pytest.fixture
def temp_project_dir(tmp_path):
    """Create temporary project directory"""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    return project_dir


# ============================================================================
# SpecAgent Tests
# ============================================================================

class TestSpecAgent:
    """Tests for SpecAgent"""
    
    @pytest.mark.asyncio
    async def test_spec_agent_initialization(self, rag_system, mcp_server):
        """Test SpecAgent initialization"""
        agent = SpecAgent(rag_system, mcp_server)
        assert agent is not None
        assert agent.rag_system == rag_system
        assert agent.mcp_server == mcp_server
    
    @pytest.mark.asyncio
    async def test_analyze_specification(self, rag_system, mcp_server, sample_spec):
        """Test specification analysis"""
        agent = SpecAgent(rag_system, mcp_server)
        
        # Mock OpenAI response for testing
        # In real test, you'd mock the OpenAI client
        analyzed = await agent.analyze(sample_spec)
        
        assert "original_spec" in analyzed
        assert analyzed["original_spec"] == sample_spec


# ============================================================================
# ScaffoldAgent Tests
# ============================================================================

class TestScaffoldAgent:
    """Tests for ScaffoldAgent"""
    
    @pytest.mark.asyncio
    async def test_scaffold_agent_initialization(self, rag_system, mcp_server):
        """Test ScaffoldAgent initialization"""
        agent = ScaffoldAgent(rag_system, mcp_server)
        assert agent is not None
    
    @pytest.mark.asyncio
    async def test_generate_scaffold_structure(self, rag_system, mcp_server, sample_spec):
        """Test scaffold generation returns correct structure"""
        agent = ScaffoldAgent(rag_system, mcp_server)
        
        analyzed_spec = {**sample_spec, "ml_task": "classification"}
        scaffold = await agent.generate(analyzed_spec)
        
        assert "files" in scaffold
        assert "spec" in scaffold
        
        # Check required files
        required_files = ["train.py", "model.py", "data_loader.py", 
                         "config.py", "requirements.txt", "README.md"]
        for file in required_files:
            assert file in scaffold["files"]
    
    def test_generated_code_not_empty(self, rag_system, mcp_server):
        """Test that generated code files are not empty"""
        agent = ScaffoldAgent(rag_system, mcp_server)
        # Test individual generation methods if needed


# ============================================================================
# TestAgent Tests
# ============================================================================

class TestAgent:
    """Tests for TestAgent"""
    
    @pytest.mark.asyncio
    async def test_test_agent_initialization(self, rag_system, mcp_server):
        """Test TestAgent initialization"""
        agent = TestAgent(rag_system, mcp_server)
        assert agent is not None
    
    @pytest.mark.asyncio
    async def test_generate_tests_structure(self, rag_system, mcp_server, sample_spec):
        """Test test generation returns correct structure"""
        agent = TestAgent(rag_system, mcp_server)
        
        scaffold = {"files": {"train.py": "# training code"}}
        tests = await agent.generate(sample_spec, scaffold)
        
        assert "files" in tests
        
        # Check test files
        test_files = ["test_model.py", "test_dataloader.py", 
                     "test_training.py", "conftest.py"]
        for test_file in test_files:
            assert test_file in tests["files"]


# ============================================================================
# ValidatorAgent Tests
# ============================================================================

class TestValidatorAgent:
    """Tests for ValidatorAgent"""
    
    @pytest.mark.asyncio
    async def test_validator_initialization(self, mcp_server):
        """Test ValidatorAgent initialization"""
        agent = ValidatorAgent(mcp_server)
        assert agent is not None
    
    @pytest.mark.asyncio
    async def test_syntax_validation(self, mcp_server, temp_project_dir):
        """Test Python syntax validation"""
        agent = ValidatorAgent(mcp_server)
        
        # Create valid Python file
        (temp_project_dir / "valid.py").write_text("print('hello')")
        
        result = await agent._check_syntax(temp_project_dir)
        
        assert result["passed"] == True
        assert result["files_checked"] == 1
    
    @pytest.mark.asyncio
    async def test_syntax_validation_with_error(self, mcp_server, temp_project_dir):
        """Test syntax validation catches errors"""
        agent = ValidatorAgent(mcp_server)
        
        # Create invalid Python file
        (temp_project_dir / "invalid.py").write_text("print('hello'")
        
        result = await agent._check_syntax(temp_project_dir)
        
        assert result["passed"] == False
        assert len(result["errors"]) > 0


# ============================================================================
# MLOpsAgent Tests
# ============================================================================

class TestMLOpsAgent:
    """Tests for MLOpsAgent"""
    
    @pytest.mark.asyncio
    async def test_mlops_agent_initialization(self, rag_system, mcp_server):
        """Test MLOpsAgent initialization"""
        agent = MLOpsAgent(rag_system, mcp_server)
        assert agent is not None
    
    @pytest.mark.asyncio
    async def test_generate_mlops_config(self, rag_system, mcp_server, sample_spec):
        """Test MLOps configuration generation"""
        agent = MLOpsAgent(rag_system, mcp_server)
        
        scaffold = {"files": {"train.py": "# code"}}
        mlops_config = await agent.generate(sample_spec, scaffold)
        
        assert "files" in mlops_config
        
        # Check MLOps files
        mlops_files = ["azure_ml_config.yml", "deployment_config.json"]
        for file in mlops_files:
            assert file in mlops_config["files"]


# ============================================================================
# OrchestrationAgent Tests
# ============================================================================

class TestOrchestrationAgent:
    """Tests for OrchestrationAgent"""
    
    @pytest.mark.asyncio
    async def test_orchestration_initialization(self):
        """Test OrchestrationAgent initialization"""
        agent = OrchestrationAgent()
        assert agent is not None
        assert agent.workflow_state == {}
    
    @pytest.mark.asyncio
    async def test_validate_workflow(self, sample_spec):
        """Test workflow validation"""
        agent = OrchestrationAgent()
        
        scaffold = {
            "files": {
                "train.py": "# code",
                "model.py": "# code",
                "data_loader.py": "# code"
            }
        }
        
        tests = {
            "files": {
                "test_model.py": "# test"
            }
        }
        
        mlops = {
            "files": {
                "azure_ml_config.yml": "# config"
            }
        }
        
        result = await agent.validate_workflow(sample_spec, scaffold, tests, mlops)
        
        assert "valid" in result
        assert "validations" in result


# ============================================================================
# RAGSystem Tests
# ============================================================================

class TestRAGSystem:
    """Tests for RAG System"""
    
    @pytest.mark.asyncio
    async def test_rag_initialization(self):
        """Test RAG system initialization"""
        rag = RAGSystem(collection_name="test_rag")
        assert rag is not None
        assert rag.collection is not None
    
    @pytest.mark.asyncio
    async def test_rag_query(self):
        """Test RAG query functionality"""
        rag = RAGSystem(collection_name="test_rag_query")
        
        context = await rag.query("PyTorch training loop")
        
        assert isinstance(context, str)
        assert len(context) > 0
    
    def test_add_template(self):
        """Test adding template to RAG"""
        rag = RAGSystem(collection_name="test_rag_add")
        
        rag.add_template(
            "custom_template",
            "Custom ML template content",
            "custom"
        )
        
        categories = rag.get_all_categories()
        assert "custom" in categories


# ============================================================================
# MCPServer Tests
# ============================================================================

class TestMCPServer:
    """Tests for MCP Server"""
    
    @pytest.mark.asyncio
    async def test_mcp_initialization(self):
        """Test MCP server initialization"""
        mcp = MCPServer(port=8082)
        assert mcp is not None
        assert mcp.port == 8082
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_context(self):
        """Test context storage and retrieval"""
        mcp = MCPServer()
        
        await mcp.store_context("test_key", {"data": "test_value"})
        result = await mcp.retrieve_context("test_key")
        
        assert result == {"data": "test_value"}
    
    @pytest.mark.asyncio
    async def test_template_registration(self):
        """Test template registration"""
        mcp = MCPServer()
        
        await mcp.register_template("test_template", "template content")
        templates = await mcp.list_templates()
        
        assert "test_template" in templates
    
    @pytest.mark.asyncio
    async def test_code_validation(self):
        """Test code validation"""
        mcp = MCPServer()
        
        # Valid code
        result = await mcp.validate_code("print('hello')", "python")
        assert result["valid"] == True
        
        # Invalid code
        result = await mcp.validate_code("print('hello'", "python")
        assert result["valid"] == False


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self, tmp_path):
        """Test complete ML project generation pipeline"""
        # Create input specification
        input_file = tmp_path / "test_input.json"
        spec = {
            "project_name": "integration_test",
            "purpose": "Integration test project",
            "data_type": "tabular",
            "ml_task": "classification",
            "target_environment": "local",
            "models": ["mlp"]
        }
        write_json(input_file, spec)
        
        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Run builder
        #builder = IntMLBuilder(
         #   input_path=str(input_file),
          #  output_path=str(output_dir),
           # mode="local"
        #)
        
        # Note: This would require mocking OpenAI API
        # For real testing, use mock responses
        # result = await builder.execute()
        # assert result["status"] == "success"


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions"""
    
    def test_read_write_json(self, tmp_path):
        """Test JSON read/write"""
        from utils.file_utils import read_json, write_json
        
        test_file = tmp_path / "test.json"
        test_data = {"key": "value", "number": 42}
        
        write_json(test_file, test_data)
        read_data = read_json(test_file)
        
        assert read_data == test_data
    
    def test_write_file(self, tmp_path):
        """Test file writing"""
        from utils.file_utils import write_file, read_file
        
        test_file = tmp_path / "test.txt"
        content = "Test content"
        
        write_file(test_file, content)
        read_content = read_file(test_file)
        
        assert read_content == content


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests"""
    
    @pytest.mark.asyncio
    async def test_rag_query_performance(self):
        """Test RAG query performance"""
        import time
        
        rag = RAGSystem(collection_name="perf_test")
        
        start = time.time()
        await rag.query("PyTorch training")
        duration = time.time() - start
        
        # Should complete in reasonable time
        assert duration < 5.0  # 5 seconds


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
