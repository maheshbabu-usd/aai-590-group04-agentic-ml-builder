"""
Scaffold Agent - Generates ML project code scaffolds
"""
import logging
from typing import Dict, Any
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

class ScaffoldAgent:
    """Generates ML project code scaffolding"""
    
    def __init__(self, rag_system, mcp_server):
        self.rag_system = rag_system
        self.mcp_server = mcp_server
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def generate(self, analyzed_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete ML project scaffold"""
        logger.info("Generating ML scaffold...")
        
        files = {}
        
        # Generate each component
        files["train.py"] = await self._generate_training_script(analyzed_spec)
        files["model.py"] = await self._generate_model_file(analyzed_spec)
        files["data_loader.py"] = await self._generate_dataloader(analyzed_spec)
        files["config.py"] = await self._generate_config(analyzed_spec)
        files["requirements.txt"] = await self._generate_requirements(analyzed_spec)
        files["README.md"] = await self._generate_readme(analyzed_spec)
        files["Dockerfile"] = await self._generate_dockerfile(analyzed_spec)
        files[".gitignore"] = self._generate_gitignore()
        
        logger.info(f"Generated {len(files)} scaffold files")
        return {"files": files, "spec": analyzed_spec}
    
    async def _generate_training_script(self, spec: Dict) -> str:
        """Generate training script"""
        context = await self.rag_system.query("PyTorch training loop best practices")
        
        prompt = f"""Generate a complete, production-ready PyTorch training script for:

Task: {spec.get('ml_task')}
Models: {spec.get('recommended_models')}
Metrics: {spec.get('evaluation_metrics')}
Data Modality: {spec.get('data_modality')}

Context: {context}

Requirements:
1. Import all necessary libraries (torch, numpy, etc.)
2. Complete training loop with:
   - Epoch iteration
   - Batch processing
   - Loss computation
   - Backpropagation
   - Optimizer step
3. Validation loop with:
   - Evaluation mode
   - Metric computation
   - No gradient tracking
4. Model checkpointing (save best model)
5. TensorBoard logging
6. Command-line arguments (learning rate, epochs, batch size)
7. Early stopping
8. Proper error handling
9. Progress bars (tqdm)
10. Device management (CPU/GPU)

Generate COMPLETE, WORKING Python code with NO placeholders. Include all imports and full implementations."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert ML engineer. Generate complete, production-ready code with no placeholders or TODO comments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        code = response.choices[0].message.content
        # Remove markdown code blocks if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        return code
    
    async def _generate_model_file(self, spec: Dict) -> str:
        """Generate model definition"""
        models = spec.get('recommended_models', [])
        model_type = models[0] if models else "neural network"
        
        prompt = f"""Generate a complete PyTorch model definition for {model_type}.

Task: {spec.get('ml_task')}
Data modality: {spec.get('data_modality')}

Requirements:
1. Complete model class inheriting from nn.Module
2. __init__ method with all layers
3. forward method with complete forward pass
4. Proper initialization (Xavier, He, etc.)
5. Activation functions
6. Dropout for regularization
7. Batch normalization if appropriate
8. Output layer matching task requirements
9. Docstrings explaining architecture
10. Type hints

For classification: output logits (no softmax in forward)
For regression: single output value

Generate COMPLETE, WORKING code with NO placeholders."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert ML engineer specializing in neural network architectures."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=3000
        )
        
        code = response.choices[0].message.content
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        return code
    
    async def _generate_dataloader(self, spec: Dict) -> str:
        """Generate data loader"""
        prompt = f"""Generate a complete PyTorch DataLoader implementation for:

Data modality: {spec.get('data_modality')}
Preprocessing: {spec.get('required_preprocessing')}
Task: {spec.get('ml_task')}

Requirements:
1. Complete Dataset class inheriting from torch.utils.data.Dataset
2. __init__, __len__, and __getitem__ methods
3. Data loading from files/URLs
4. Appropriate transformations for {spec.get('data_modality')} data
5. Train/validation split functionality
6. DataLoader creation with proper workers
7. Data augmentation for training set
8. Normalization
9. Error handling for missing data
10. Type hints and docstrings

Generate COMPLETE, WORKING code. Include helper functions if needed."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in data engineering for ML."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=3000
        )
        
        code = response.choices[0].message.content
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        return code
    
    async def _generate_config(self, spec: Dict) -> str:
        """Generate configuration file"""
        return f'''"""Configuration for {spec.get('project_name')}"""
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    """Training and model configuration"""
    
    # Model
    model_type: str = "{spec.get('recommended_models', ['mlp'])[0]}"
    input_dim: int = 128
    hidden_dims: List[int] = None
    output_dim: int = 10
    dropout: float = 0.5
    
    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    
    # Data
    train_split: float = 0.8
    val_split: float = 0.2
    num_workers: int = 4
    
    # Metrics
    metrics: List[str] = None
    
    # Paths
    data_path: str = "./data"
    model_save_path: str = "./models"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    
    # Device
    device: str = "cuda"  # Will auto-detect
    seed: int = 42
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]
        if self.metrics is None:
            self.metrics = {spec.get('evaluation_metrics', ['accuracy'])}

config = Config()
'''
    
    async def _generate_requirements(self, spec: Dict) -> str:
        """Generate requirements.txt"""
        base_requirements = """# Core ML Framework
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1

# Data Science
numpy==2.1.3
pandas==2.2.3
scikit-learn==1.5.2

# Visualization
matplotlib==3.9.2
seaborn==0.13.2
tensorboard==2.18.0

# Utilities
tqdm==4.67.1
pyyaml==6.0.2

# Testing
pytest==8.3.4
"""
        
        # Add task-specific requirements
        if spec.get('data_modality') == 'text':
            base_requirements += """
# NLP
transformers==4.46.3
tokenizers==0.20.3
"""
        elif spec.get('data_modality') == 'image':
            base_requirements += """
# Computer Vision
opencv-python==4.10.0.84
albumentations==1.4.20
"""
        elif spec.get('data_modality') == 'audio':
            base_requirements += """
# Audio Processing
librosa==0.10.2
soundfile==0.12.1
"""
        
        return base_requirements
    
    async def _generate_readme(self, spec: Dict) -> str:
        """Generate README"""
        return f"""# {spec.get('project_name')}

## Project Description

**ML Task**: {spec.get('ml_task')}  
**Data Modality**: {spec.get('data_modality')}  
**Complexity**: {spec.get('estimated_complexity')}

## Dataset Information

- **Source**: {spec.get('dataset_info', {}).get('source', 'N/A')}
- **Size**: {spec.get('dataset_info', {}).get('size_estimate', 'N/A')}
- **Features**: {spec.get('dataset_info', {}).get('features', 'N/A')}

## Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional but recommended)

### Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\\venv\\Scripts\\Activate.ps1

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Basic training
python train.py

# With custom parameters
python train.py --epochs 50 --batch-size 64 --lr 0.001
```

### Configuration

Edit `config.py` to modify:
- Model architecture
- Training hyperparameters
- Data paths
- Evaluation metrics

## Model Architecture

**Recommended Models**: {', '.join(spec.get('recommended_models', []))}

## Evaluation Metrics

{chr(10).join(['- ' + m for m in spec.get('evaluation_metrics', [])])}

## Project Structure

```
{spec.get('project_name')}/
├── train.py              # Training script
├── model.py              # Model definition
├── data_loader.py        # Data loading
├── config.py             # Configuration
├── requirements.txt      # Dependencies
├── README.md            # This file
├── data/                # Dataset directory
├── models/              # Saved models
├── logs/                # TensorBoard logs
└── checkpoints/         # Training checkpoints
```

## Results

Training results will be saved to:
- Models: `./models/`
- Logs: `./logs/` (view with TensorBoard)
- Checkpoints: `./checkpoints/`

### View Training Progress

```bash
tensorboard --logdir logs
```

## Deployment

### Local Inference

```python
import torch
from model import Model

# Load model
model = Model()
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    output = model(input_data)
```

### Azure Deployment

See deployment documentation for Azure ML deployment instructions.

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config.py
- Enable gradient checkpointing

### Training Too Slow
- Increase num_workers in DataLoader
- Use mixed precision training
- Check GPU utilization

## License

MIT License

## Generated by

Agentic ML Builder v1.0.0
"""
    
    async def _generate_dockerfile(self, spec: Dict) -> str:
        """Generate Dockerfile"""
        return f"""# Dockerfile for {spec.get('project_name')}
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models logs checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose TensorBoard port
EXPOSE 6006

# Default command
CMD ["python", "train.py"]
"""
    
    def _generate_gitignore(self) -> str:
        """Generate .gitignore"""
        return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# ML specific
data/
*.pth
*.pt
*.ckpt
models/
checkpoints/
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
output/
temp/
"""
