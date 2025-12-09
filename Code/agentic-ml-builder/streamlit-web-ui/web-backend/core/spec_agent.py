"""
File: spec_agent.py

Module that implements the SpecAgent class for specification generation and
dataset discovery. The agent uses LLM-driven analysis to understand user
requirements and discovers relevant datasets from HuggingFace, OpenML,
and other sources to create comprehensive ML pipeline specifications.

"""

import json
import os
import warnings
from openai import OpenAI
from dotenv import load_dotenv
import openml
from datasets import load_dataset_builder
from huggingface_hub import list_datasets

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()


class SpecAgent:
    """
    Generates ML pipeline specifications and discovers relevant datasets.
    
    Uses LLM analysis to understand user requirements and searches multiple
    dataset sources (HuggingFace, OpenML) to find appropriate datasets that
    match the specified task type, data modality, and domain requirements.
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _search_huggingface(self, keywords: list) -> list:
        """Search HuggingFace datasets."""
        try:
            print("=" * 50)
            print("🔍 SEARCHING HUGGINGFACE...")
            print(f"Keywords: {keywords}")
            results = []
            
            # Search HuggingFace datasets
            search_query = " ".join(keywords[:2])  # Use first 2 keywords
            print(f"HF Search Query: '{search_query}'")
            
            datasets_list = list(list_datasets(search=search_query, limit=10))
            print(f"HF API returned {len(datasets_list)} datasets")
            
            for ds in datasets_list[:5]:
                try:
                    dataset_id = ds.id
                    print(f"  - {dataset_id}")
                    results.append({
                        'source': 'huggingface',
                        'id': dataset_id,
                        'name': dataset_id,
                        'description': getattr(ds, 'description', 'HuggingFace dataset') or 'No description',
                        'url': f"https://huggingface.co/datasets/{dataset_id}",
                        'tags': getattr(ds, 'tags', [])
                    })
                except Exception as e:
                    print(f"  Error processing HF dataset: {e}")
                    continue
            
            print(f"✓ Found {len(results)} HuggingFace datasets")
            print("=" * 50)
            return results
        except Exception as e:
            print(f"❌ HuggingFace search FAILED: {e}")
            import traceback
            traceback.print_exc()
            print("=" * 50)
            return []
    
    def _search_kaggle(self, keywords: list) -> list:
        """Search Kaggle datasets via web scraping."""
        try:
            print("=" * 50)
            print("🔍 SEARCHING KAGGLE...")
            print(f"Keywords: {keywords}")
            from duckduckgo_search import DDGS
            
            results = []
            search_query = f"{' '.join(keywords)} dataset site:kaggle.com/datasets"
            print(f"Kaggle Search Query: '{search_query}'")
            
            with DDGS() as ddgs:
                search_results = list(ddgs.text(search_query, max_results=5))
            
            print(f"DuckDuckGo returned {len(search_results)} results")
            
            for r in search_results:
                if 'kaggle.com/datasets/' in r['href']:
                    print(f"  - {r['title']}")
                    results.append({
                        'source': 'kaggle',
                        'id': r['href'].split('/datasets/')[-1],
                        'name': r['title'],
                        'description': r['body'],
                        'url': r['href'],
                        'tags': keywords
                    })
            
            print(f"✓ Found {len(results)} Kaggle datasets")
            
            # Fallback: If 0 results, provide a manual search link
            if len(results) == 0:
                print("⚠️ DuckDuckGo search failed. Adding manual search link.")
                query_url = f"https://www.kaggle.com/search?q={'+'.join(keywords)}"
                results.append({
                    'source': 'kaggle',
                    'id': 'manual_search',
                    'name': 'Search on Kaggle.com',
                    'description': f'Automated access via DuckDuckGo failed. Click to browse datasets for: {", ".join(keywords)}',
                    'url': query_url,
                    'tags': keywords
                })
                
            print("=" * 50)
            return results
        except Exception as e:
            print(f"❌ Kaggle search FAILED: {e}")
            import traceback
            traceback.print_exc()
            print("=" * 50)
            return []

    def _search_openml(self, keywords: list) -> list:
        """Searches OpenML for real datasets matching the query."""
        try:
            print("Searching OpenML...")
            
            # Get all datasets
            datasets = openml.datasets.list_datasets(output_format='dataframe')
            
            # Filter by keywords in name
            def matches_keywords(name):
                name_lower = str(name).lower()
                return any(term.lower() in name_lower for term in keywords)
            
            filtered = datasets[datasets['name'].apply(matches_keywords)]
            
            # If no keyword matches, fall back to size-filtered datasets
            if len(filtered) == 0:
                print(f"⚠️  No OpenML datasets with keywords: {keywords}")
                filtered = datasets[(datasets['NumberOfInstances'] > 100) & (datasets['NumberOfInstances'] < 1000000)].head(10)
            
            results = []
            print(f"Fetching OpenML details for {min(len(filtered), 5)} datasets...")
            
            for idx, (_, row) in enumerate(filtered.head(5).iterrows()):
                try:
                    dataset_id = row['did']
                    dataset_name = str(row.get('name', 'unknown'))
                    
                    # Fetch full dataset to get description
                    ds = openml.datasets.get_dataset(dataset_id, download_data=False)
                    description = ds.description or "No description"
                    if len(description) > 200:
                        description = description[:200] + "..."
                    
                    results.append({
                        'source': 'openml',
                        'id': str(dataset_id),
                        'name': dataset_name,
                        'description': description,
                        'url': ds.url,
                        'format': ds.format or 'ARFF',
                        'default_target': ds.default_target_attribute or 'unknown',
                        'licence': ds.licence or 'Public',
                        'citation': ds.citation or 'OpenML dataset'
                    })
                except Exception as e:
                    print(f"Skipping OpenML dataset {dataset_id}: {e}")
                    continue
            
            print(f"Found {len(results)} OpenML datasets")
            return results
                
        except Exception as e:
            print(f"OpenML search error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def analyze(self, user_query: str) -> dict:
        """
        Analyzes the user query and finds the best dataset across multiple sources.
        """
        # Extract keywords
        print(f"Analyzing: {user_query}")
        keyword_response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract 1-3 simple, broad domain keywords for finding a dataset. EXCLUDE words like 'dataset', 'classification', 'project', 'model', 'python', 'pytorch'. Example: for 'dog breed classification', return 'dog, breed'. For 'customer churn', return 'customer churn'."},
                {"role": "user", "content": user_query}
            ]
        )
        keywords = [k.strip() for k in keyword_response.choices[0].message.content.strip().split(',')]
        print(f"Keywords: {keywords}")
        
        # Search all sources
        all_results = []
        
        # 1. HuggingFace (best for images, audio, text)
        hf_results = self._search_huggingface(keywords)
        all_results.extend(hf_results)
        
        # 2. Kaggle (diverse datasets)
        kaggle_results = self._search_kaggle(keywords)
        all_results.extend(kaggle_results)
        
        # 3. OpenML (tabular data)
        openml_results = self._search_openml(keywords)
        all_results.extend(openml_results)
        
        print("\n" + "=" * 50)
        print(f"📊 TOTAL RESULTS: {len(all_results)}")
        print(f"  - HuggingFace: {len(hf_results)}")
        print(f"  - Kaggle: {len(kaggle_results)}")
        print(f"  - OpenML: {len(openml_results)}")
        print("=" * 50 + "\n")
        
        if not all_results:
            return {"error": "No datasets found across any source (OpenML, HuggingFace, Kaggle)"}
        
        # Let GPT select the best dataset
        candidates_text = "\n\n".join([
            f"Source: {r['source'].upper()}\nName: {r['name']}\nURL: {r['url']}\nDescription: {r['description'][:150]}..."
            for r in all_results[:15]  # Top 15 across all sources
        ])
        
        selection_prompt = f"""
        User Request: "{user_query}"
        
        Available Datasets from Multiple Sources:
        {candidates_text}
        
        Which dataset is MOST relevant? 
        Return the exact dataset NAME (not URL or ID).
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at dataset selection. Return only the dataset name."},
                {"role": "user", "content": selection_prompt}
            ],
            temperature=0
        )
        
        selected_name = response.choices[0].message.content.strip()
        print(f"✓ GPT selected: {selected_name}")
        
        # Find the selected dataset
        selected = None
        for r in all_results:
            if selected_name.lower() in r['name'].lower() or r['name'].lower() in selected_name.lower():
                selected = r
                break
        
        if not selected:
            selected = all_results[0]  # Fallback to first
        
        print(f"✓ Final choice: {selected['source'].upper()} - {selected['name']}")
        
        # Build response in OpenML format
        return {
            "data_set_description": {
                "id": str(selected.get('id', 'unknown')),
                "name": selected['name'],
                "version": "1",
                "description": selected['description'],
                "description_version": "1",
                "format": selected.get('format', 'Various'),
                "upload_date": selected.get('upload_date', '2024'),
                "licence": selected.get('licence', 'See source'),
                "url": selected['url'],
                "parquet_url": selected['url'],
                "file_id": str(selected.get('id', 'unknown')),
                "default_target_attribute": selected.get('default_target', 'target'),
                "citation": selected.get('citation', f"Dataset from {selected['source']}"),
                "tag": self._extract_tags(user_query),
                "visibility": "public",
                "status": "active",
                "processing_date": "2024",
                "md5_checksum": "check_source",
                "source": selected['source']  # Added source info
            }
        }
    
    def _extract_tags(self, query: str) -> list:
        """Use LLM to extract relevant tags."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Extract ML-related tags from this query. Return JSON array of strings."},
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result.get('tags', ['machine_learning'])
        except:
            return ['machine_learning']

if __name__ == "__main__":
    agent = SpecAgent()
    print(agent.analyze("Build a customer churn prediction model using Random Forest on tabular data."))
