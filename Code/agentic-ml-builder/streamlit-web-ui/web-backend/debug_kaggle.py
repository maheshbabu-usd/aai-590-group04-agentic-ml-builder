
from duckduckgo_search import DDGS
import json

def test_search(query, backend='api'):
    print(f"Testing query: '{query}' with backend='{backend}'")
    try:
        with DDGS() as ddgs:
            # Note: library might not support 'backend' arg in text(), it's usually init or text() param
            # Checking docs usage: ddgs.text(keywords, backend='api')
            results = list(ddgs.text(query, max_results=5, backend=backend))
        print(f"Found {len(results)} results")
        for r in results:
            print(f"  - {r.get('title')} ({r.get('href')})")
    except Exception as e:
        print(f"Error: {e}")

print("--- Test 7: backend='html' ---")
test_search("site:kaggle.com/datasets plant classification", backend='html')

print("--- Test 8: backend='lite' ---")
test_search("site:kaggle.com/datasets plant classification", backend='lite')
