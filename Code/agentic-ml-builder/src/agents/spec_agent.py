"""
Spec Agent - Analyzes ML project specifications
"""
import logging
from typing import Dict, Any
from openai import OpenAI
import os
import json

logger = logging.getLogger(__name__)

class SpecAgent:
    """Analyzes and processes ML project specifications"""
    
    def __init__(self, rag_system, mcp_server):
        self.rag_system = rag_system
        self.mcp_server = mcp_server
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def analyze(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze project specification and extract requirements"""
        logger.info("Analyzing project specification...")
        
        # Get relevant context from RAG
        context = await self.rag_system.query(
            f"ML project setup for {spec.get('data_type', 'general')} data"
        )
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(spec, context)
        
        # Call OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert ML project specification analyzer. Provide detailed, structured analysis of ML requirements."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        analysis = response.choices[0].message.content
        
        # Parse and enhance the specification
        analyzed_spec = json.loads(analysis)
        analyzed_spec["original_spec"] = spec
        
        logger.info(f"Analysis complete: {analyzed_spec.get('project_name')}")
        return analyzed_spec
    
    def _build_analysis_prompt(self, spec: Dict, context: str) -> str:
        """Build prompt for specification analysis"""
        return f"""Analyze the following ML project specification and provide a structured analysis.

Project Specification:
{json.dumps(spec, indent=2)}

Relevant Context from Best Practices:
{context}

Provide a JSON response with the following structure:
{{
  "project_name": "suggested_project_name",
  "ml_task": "classification|regression|clustering|etc",
  "data_modality": "tabular|image|text|audio|video",
  "recommended_models": ["model1", "model2", "model3"],
  "required_preprocessing": ["preprocessing_step1", "preprocessing_step2"],
  "evaluation_metrics": ["metric1", "metric2"],
  "deployment_target": "local|azure|both",
  "estimated_complexity": "low|medium|high",
  "dataset_info": {{
    "source": "dataset source",
    "size_estimate": "small|medium|large",
    "features": "description of features"
  }},
  "infrastructure_requirements": {{
    "compute": "cpu|gpu|tpu",
    "memory_gb": estimated_memory,
    "storage_gb": estimated_storage
  }},
  "recommended_frameworks": ["pytorch", "tensorflow", "etc"],
  "estimated_training_time": "time estimate",
  "best_practices": ["practice1", "practice2"]
}}

Respond ONLY with valid JSON. No additional text."""
