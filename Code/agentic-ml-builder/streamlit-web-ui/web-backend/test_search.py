#!/usr/bin/env python3
"""Test script to verify HuggingFace and Kaggle search functionality."""

import os
import sys
sys.path.insert(0, '/Users/ashutosh_nayak/.gemini/antigravity/scratch/agentic_ml_platform')

print("Testing HuggingFace API...")
try:
    from huggingface_hub import list_datasets
    
    print("Searching for 'dog' datasets...")
    datasets_list = list(list_datasets(search="dog", limit=5))
    
    print(f"✓ Found {len(datasets_list)} datasets:")
    for ds in datasets_list:
        print(f"  - {ds.id}")
except Exception as e:
    print(f"✗ HuggingFace FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50 + "\n")

print("Testing Kaggle/DuckDuckGo search...")
try:
    from duckduckgo_search import DDGS
    
    search_query = "dog breed dataset site:kaggle.com/datasets"
    print(f"Query: {search_query}")
    
    with DDGS() as ddgs:
        results = list(ddgs.text(search_query, max_results=3))
    
    print(f"✓ Found {len(results)} results:")
    for r in results:
        if 'kaggle.com/datasets/' in r['href']:
            print(f"  - {r['title']}: {r['href']}")
except Exception as e:
    print(f"✗ Kaggle search FAILED: {e}")
    import traceback
    traceback.print_exc()
