# ============================================================================
# file_utils.py
# ============================================================================
"""
Utilities for reading and writing common file formats used by the project.

Module responsibilities
- Provide simple helpers to read/write JSON and YAML files.
- Provide small convenience helpers to read/write arbitrary text files.
- Provide a basic `setup_logging` helper that configures a file + console logger.

The helpers use `pathlib.Path` for path handling and will create parent
directories as needed when writing files.

Usage example
```
from utils.file_utils import read_yaml, write_json, setup_logging

setup_logging('DEBUG')
cfg = read_yaml(Path('config/experiment.yaml'))
write_json(Path('out/metrics.json'), {'accuracy': 0.9})
```
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict
import yaml

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "ml_builder.log"),
            logging.StreamHandler()
        ]
    )

def read_json(filepath: Path) -> Dict:
    """Read JSON file"""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def write_json(filepath: Path, data: Dict):
    """Write JSON file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def read_yaml(filepath: Path) -> Dict:
    """Read YAML file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def write_yaml(filepath: Path, data: Dict):
    """Write YAML file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)

def write_file(filepath: Path, content: str):
    """Write file with content"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def read_file(filepath: Path) -> str:
    """Read file content"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

