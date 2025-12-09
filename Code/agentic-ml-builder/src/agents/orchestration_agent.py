
# ============================================================================
# orchestration_agent.py
# ============================================================================
"""
Orchestration Agent - Manages workflow orchestration
"""
import logging
from typing import Dict, Any
import asyncio

logger = logging.getLogger(__name__)

class OrchestrationAgent:
    """Orchestrates the workflow between agents"""
    
    def __init__(self):
        self.workflow_state = {}
    
    async def validate_workflow(
        self, 
        spec: Dict, 
        scaffold: Dict, 
        tests: Dict, 
        mlops: Dict
    ) -> Dict[str, Any]:
        """Validate the complete workflow"""
        logger.info("Validating workflow orchestration...")
        
        validations = []
        
        # Validate spec completeness
        validations.append(await self._validate_spec(spec))
        
        # Validate scaffold consistency
        validations.append(await self._validate_scaffold(scaffold))
        
        # Validate test coverage
        validations.append(await self._validate_tests(tests))
        
        # Validate MLOps config
        validations.append(await self._validate_mlops(mlops))
        
        all_valid = all(v["valid"] for v in validations)
        
        return {
            "valid": all_valid,
            "validations": validations
        }
    
    async def _validate_spec(self, spec: Dict) -> Dict:
        """Validate specification"""
        required_fields = ['project_name', 'ml_task', 'data_type']
        missing = [f for f in required_fields if f not in spec]
        
        return {
            "component": "specification",
            "valid": len(missing) == 0,
            "missing_fields": missing
        }
    
    async def _validate_scaffold(self, scaffold: Dict) -> Dict:
        """Validate scaffold"""
        required_files = ['train.py', 'model.py', 'data_loader.py']
        files = scaffold.get('files', {})
        missing = [f for f in required_files if f not in files]
        
        return {
            "component": "scaffold",
            "valid": len(missing) == 0,
            "missing_files": missing,
            "files_generated": len(files)
        }
    
    async def _validate_tests(self, tests: Dict) -> Dict:
        """Validate tests"""
        files = tests.get('files', {})
        has_tests = any('test_' in f for f in files.keys())
        
        return {
            "component": "tests",
            "valid": has_tests,
            "test_files": len([f for f in files if 'test_' in f])
        }
    
    async def _validate_mlops(self, mlops: Dict) -> Dict:
        """Validate MLOps configuration"""
        files = mlops.get('files', {})
        
        return {
            "component": "mlops",
            "valid": len(files) > 0,
            "config_files": len(files)
        }
