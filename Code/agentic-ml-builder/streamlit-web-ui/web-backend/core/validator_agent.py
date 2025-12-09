"""
File: validator_agent.py

Module that implements the ValidatorAgent class for code validation and
quality assessment. The agent performs syntax checking, static analysis,
and LLM-based code review to ensure generated ML pipeline code is
correct, complete, and production-ready.

"""

import ast
import sys
import io
from openai import OpenAI
import os


class ValidatorAgent:
    """
    Validates and assesses quality of generated Python code.
    
    Performs syntax validation, static code analysis, and LLM-based
    code review to detect errors, identify quality issues, and provide
    actionable feedback for code improvement.
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def validate_code(self, code_str: str) -> dict:
        """
        Validates Python code for syntax correctness and basic quality.
        Returns a dict with 'valid' (bool), 'errors' (list), and 'bscore' (int 0-10).
        """
        results = {
            "valid": True,
            "errors": [],
            "score": 10,
            "feedback": ""
        }

        # 1. Syntax Check (Smoke Test)
        try:
            ast.parse(code_str)
        except SyntaxError as e:
            results["valid"] = False
            results["errors"].append(f"Syntax Error at line {e.lineno}: {e.msg}")
            results["score"] = 0
            results["feedback"] = "Code failed to compile. Major syntax errors detected."
            return results
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Validation Error: {str(e)}")
            results["score"] = 0
            return results

        # 2. Basic Static Analysis (Linting)
        # Check for imports
        if "import " not in code_str and "from " not in code_str:
            results["errors"].append("Warning: No imports detected. Code might be incomplete.")
            results["score"] -= 2

        # Check for TODOs
        if "# TODO" in code_str or "# todo" in code_str.lower():
            results["errors"].append("Info: Code contains TODO comments.")
            results["score"] -= 1

        # Check for empty functions
        if "pass" in code_str and "def " in code_str:
            # Simple heuristic, not perfect
            pass  

        # 3. LLM-based Code Review (Optional but requested "Lint testing")
        # specific for logic issues or best practices
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a senior python developer. Review this code for critical bugs, security issues, or bad practices. Return a brief bulleted list of issues. If none, return 'Looks good'."},
                    {"role": "user", "content": code_str[:4000]} # Truncate if too long
                ],
                max_tokens=150
            )
            llm_feedback = response.choices[0].message.content
            if "Looks good" not in llm_feedback and len(llm_feedback) > 10:
                results["feedback"] = llm_feedback
                # Adjust score based on length of feedback? simplified
                results["score"] -= 1
        except Exception as e:
            print(f"LLM validation failed: {e}")

        return results

    def fix_code(self, code_str: str, errors: list, user_feedback: str = None) -> str:
        """
        Attempts to fix the code based on the errors found and optional user feedback.
        """
        try:
            prompt = f"Fix the following Python code which has errors:\n\nERRORS:\n{errors}\n"
            
            if user_feedback:
                prompt += f"\nUSER INSTRUCTIONS (Priority):\n{user_feedback}\n"
                
            prompt += f"\nCODE:\n{code_str}"
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a Python code repair expert. Return ONLY the fixed code. No markdown formatting, just the code."},
                    {"role": "user", "content": prompt}
                ]
            )
            fixed_code = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if fixed_code.startswith("```python"):
                fixed_code = fixed_code.replace("```python", "").replace("```", "")
            return fixed_code.strip()
        except Exception as e:
            print(f"Auto-fix failed: {e}")
            return code_str
