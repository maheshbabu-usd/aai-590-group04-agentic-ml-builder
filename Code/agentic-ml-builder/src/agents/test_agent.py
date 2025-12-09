"""
Test Agent - Generates unit and integration tests for ML projects
"""
import logging
from typing import Dict, Any
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

class TestAgent:
    """Generates comprehensive test suites for ML projects"""
    
    def __init__(self, rag_system, mcp_server):
        self.rag_system = rag_system
        self.mcp_server = mcp_server
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def generate(self, analyzed_spec: Dict[str, Any], scaffold: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test suite for the ML project"""
        logger.info("Generating test suite...")
        
        files = {}
        
        # Generate test files
        files["test_model.py"] = await self._generate_model_tests(analyzed_spec, scaffold)
        files["test_dataloader.py"] = await self._generate_dataloader_tests(analyzed_spec)
        files["test_training.py"] = await self._generate_training_tests(analyzed_spec)
        files["conftest.py"] = await self._generate_conftest(analyzed_spec)
        files["pytest.ini"] = self._generate_pytest_config()
        
        logger.info(f"Generated {len(files)} test files")
        return {"files": files, "spec": analyzed_spec}
    
    async def _generate_model_tests(self, spec: Dict, scaffold: Dict) -> str:
        """Generate unit tests for model"""
        context = await self.rag_system.query("PyTorch model testing best practices")
        
        prompt = f"""Generate comprehensive unit tests for the ML model in this project:

Task: {spec.get('ml_task')}
Models: {spec.get('recommended_models')}
Input Dimensions: {spec.get('input_dimensions')}
Output Dimensions: {spec.get('output_dimensions')}

Context: {context}

Requirements:
1. Test model initialization
2. Test forward pass with correct input shapes
3. Test output shapes
4. Test gradient flow
5. Test model in eval mode
6. Test model parameter updates
7. Test batch processing (different batch sizes)
8. Test with different input types (CPU, GPU if available)
9. Test state dict save/load
10. Use pytest fixtures for reusable components

Generate COMPLETE, WORKING Python code with pytest syntax. Include all imports and full implementations."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert ML testing engineer. Generate complete, comprehensive test code with no placeholders."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=3000
        )
        
        return response.choices[0].message.content
    
    async def _generate_dataloader_tests(self, spec: Dict) -> str:
        """Generate tests for data loading"""
        context = await self.rag_system.query("PyTorch DataLoader testing patterns")
        
        prompt = f"""Generate comprehensive unit tests for the data loader:

Data Type: {spec.get('data_type')}
Data Modality: {spec.get('data_modality')}
Batch Size: {spec.get('batch_size', 32)}

Context: {context}

Requirements:
1. Test data loading initialization
2. Test batch shapes and formats
3. Test data augmentation (if applicable)
4. Test data normalization
5. Test batch iteration
6. Test dataset length
7. Test handling of edge cases (empty batches, etc.)
8. Test multi-worker loading
9. Test shuffle functionality
10. Use pytest fixtures and parametrization

Generate COMPLETE, WORKING Python code. Include all imports and full implementations."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert ML testing engineer. Generate complete, comprehensive test code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2500
        )
        
        return response.choices[0].message.content
    
    async def _generate_training_tests(self, spec: Dict) -> str:
        """Generate integration tests for training pipeline"""
        context = await self.rag_system.query("PyTorch training pipeline testing")
        
        prompt = f"""Generate integration tests for the training pipeline:

Task: {spec.get('ml_task')}
Target Environment: {spec.get('target_environment')}

Context: {context}

Requirements:
1. Test training loop initialization
2. Test loss decreases over iterations (with mock data)
3. Test model weights are updated
4. Test checkpoint saving
5. Test checkpoint loading
6. Test validation loop
7. Test metrics computation
8. Test early stopping mechanism
9. Test optimizer state management
10. Test end-to-end training (single epoch with small dataset)

Generate COMPLETE, WORKING Python code using pytest. Include all imports and full implementations."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert ML testing engineer. Generate complete, comprehensive test code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=3000
        )
        
        return response.choices[0].message.content
    
    async def _generate_conftest(self, spec: Dict) -> str:
        """Generate pytest configuration and fixtures"""
        prompt = f"""Generate a pytest conftest.py file with reusable fixtures for ML testing:

Project Type: {spec.get('ml_task')}

Include:
1. device fixture (returns appropriate device - CPU or GPU)
2. sample_data fixture (creates mock training data)
3. model fixture (creates instance of the model)
4. dataloader fixture (creates sample dataloader)
5. optimizer fixture (creates optimizer)
6. criterion fixture (creates loss function)
7. tmp_model_dir fixture (creates temporary directory for model artifacts)

Generate COMPLETE, WORKING Python code. Include all imports and full implementations."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert ML engineer. Generate complete, comprehensive conftest code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def _generate_pytest_config(self) -> str:
        """Generate pytest configuration file"""
        return """[pytest]
# pytest configuration

# Test discovery patterns
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*

# Output options
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings

# Markers
markers =
    unit: unit tests
    integration: integration tests
    slow: slow running tests
    gpu: tests requiring GPU
    asyncio: async tests

# Asyncio mode
asyncio_mode = auto

# Timeout for tests (in seconds)
timeout = 300

# Minimum Python version
minversion = 6.0
"""
