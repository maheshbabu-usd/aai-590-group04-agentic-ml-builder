#!/usr/bin/env python3
"""Direct test of the full analyze() flow without Streamlit."""

import os
import sys
sys.path.insert(0, '/Users/ashutosh_nayak/.gemini/antigravity/scratch/agentic_ml_platform')

os.environ.setdefault('OPENAI_API_KEY', open('/Users/ashutosh_nayak/.gemini/antigravity/scratch/agentic_ml_platform/.env').read().split('=')[1].strip())

from core.spec_agent import SpecAgent

print("="*60)
print("TESTING FULL FLOW: 'dog breed classification'")
print("="*60 +"\n")

agent = SpecAgent()
result = agent.analyze('dog breed classification')

print("\n" + "="*60)
print("FINAL RESULT:")
print("="*60)
print(f"Dataset Name: {result.get('data_set_description', {}).get('name')}")
print(f"Source: {result.get('data_set_description', {}).get('source')}")
print(f"URL: {result.get('data_set_description', {}).get('url')}")
print("="*60)
