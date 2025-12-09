# ============================================================================
# validator_agent.py
# ============================================================================
"""
Validator Agent - Validates generated code
"""
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any
import docker
import ast

logger = logging.getLogger(__name__)

class ValidatorAgent:
    """Validates generated ML project code"""
    
    def __init__(self, mcp_server):
        self.mcp_server = mcp_server
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized")
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
            self.docker_client = None
    
    async def validate(self, project_path: Path) -> Dict[str, Any]:
        """Run comprehensive validation"""
        logger.info("Running validation...")
        
        results = {
            "syntax": await self._check_syntax(project_path),
            "lint": await self._run_lint(project_path),
            "tests": await self._run_tests(project_path),
            "imports": await self._check_imports(project_path),
            "docker": await self._validate_docker(project_path)
        }
        
        all_passed = all(r.get("passed", False) for r in results.values() if r)
        
        return {
            "status": "passed" if all_passed else "failed",
            "results": results
        }
    
    async def _check_syntax(self, project_path: Path) -> Dict:
        """Check Python syntax"""
        try:
            python_files = list(project_path.rglob("*.py"))
            errors = []
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        ast.parse(f.read())
                except SyntaxError as e:
                    errors.append(f"{py_file}: Line {e.lineno}: {e.msg}")
            
            return {
                "passed": len(errors) == 0,
                "files_checked": len(python_files),
                "errors": errors
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def _run_lint(self, project_path: Path) -> Dict:
        """Run flake8 linting"""
        try:
            result = subprocess.run(
                ["flake8", str(project_path), 
                 "--max-line-length=127", 
                 "--extend-ignore=E203,W503"],
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "passed": result.returncode == 0,
                "output": result.stdout,
                "warnings": result.stderr
            }
        except FileNotFoundError:
            return {"passed": False, "error": "flake8 not installed"}
        except Exception as e:
            logger.error(f"Lint failed: {e}")
            return {"passed": False, "error": str(e)}
    
    async def _run_tests(self, project_path: Path) -> Dict:
        """Run pytest"""
        try:
            tests_dir = project_path / "tests"
            if not tests_dir.exists():
                return {"passed": False, "error": "No tests directory found"}
            
            result = subprocess.run(
                ["pytest", str(tests_dir), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(project_path)
            )
            return {
                "passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr
            }
        except FileNotFoundError:
            return {"passed": False, "error": "pytest not installed"}
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def _check_imports(self, project_path: Path) -> Dict:
        """Check if all imports are available"""
        try:
            python_files = list(project_path.rglob("*.py"))
            missing_imports = []
            
            for py_file in python_files:
                try:
                    result = subprocess.run(
                        ["python", "-c", f"import ast; ast.parse(open('{py_file}').read())"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode != 0:
                        missing_imports.append(str(py_file))
                except Exception:
                    pass
            
            return {
                "passed": len(missing_imports) == 0,
                "files_checked": len(python_files),
                "issues": missing_imports
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    async def _validate_docker(self, project_path: Path) -> Dict:
        """Validate in Docker container"""
        if not self.docker_client:
            return {"passed": False, "error": "Docker not available"}
        
        try:
            dockerfile = project_path / "Dockerfile"
            if not dockerfile.exists():
                return {"passed": False, "error": "No Dockerfile found"}
            
            # Build image
            image, logs = self.docker_client.images.build(
                path=str(project_path),
                tag="ml-project-validation:latest",
                rm=True
            )
            
            # Run basic test
            container = self.docker_client.containers.run(
                "ml-project-validation:latest",
                "python -c 'import torch; print(torch.__version__)'",
                remove=True,
                detach=False
            )
            
            return {"passed": True, "image_id": image.id}
        except Exception as e:
            logger.error(f"Docker validation failed: {e}")
            return {"passed": False, "error": str(e)}