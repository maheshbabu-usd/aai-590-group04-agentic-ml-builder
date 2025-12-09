"""
File: scaffold_agent.py

Module that implements the ScaffoldAgent class for generating complete,
runnable ML pipeline code. The agent uses specifications and template
references to produce production-ready code adapted to the user's
specific requirements.


"""

import os
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


class ScaffoldAgent:
    """
    Generates complete, runnable ML pipeline code from specifications and templates.
    
    Uses LLM to synthesize user specifications with reference templates to produce
    adapted, production-ready code that matches the exact requirements including
    dataset, model architecture, and preprocessing logic.
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, spec: dict, templates: List[Dict]) -> str:
        """
        Generates the final solution code based on spec and templates.
        """
        # Format retrieval context
        context_str = "\n\n".join([f"--- Template: {t['filename']} ---\n{t['content']}" for t in templates])
        
        system_prompt = """
        You are an expert Machine Learning Engineer.
        Your task is to generate a complete, runnable Python script (or set of scripts) based on the user's specification and the provided reference templates.
        
        Rules:
        1. CRITICAL: You MUST use the dataset URL and details provided in the 'User Specification'. Do NOT use the dataset from the templates if they differ.
        2. Use the provided templates as a structural guide, BUT ADAPT the logic (e.g., loss functions, model architecture, preprocessing) to fully match the 'User Specification'.
           - Example: If the spec is 'multi-label' but the template is 'multi-class', change the loss to BCEWithLogitsLoss and activation to Sigmoid.
        3. The code should be clean, commented, and ready to run.
        4. Include a main execution block.
        5. Return ONLY the code, inside markdown code blocks ```python ... ```.
        """

        user_prompt = f"""
        User Specification:
        {spec}

        Reference Templates:
        {context_str}

        Generate the solution code now.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            content = response.choices[0].message.content
            # Simple extraction of code block
            if "```python" in content:
                content = content.split("```python")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            return content
        except Exception as e:
            return f"# Error generating code: {str(e)}"

if __name__ == "__main__":
    # Test
    agent = ScaffoldAgent()
    spec = {"task_type": "classification", "modality": "tabular", "framework": "sklearn"}
    print(agent.generate(spec, []))
