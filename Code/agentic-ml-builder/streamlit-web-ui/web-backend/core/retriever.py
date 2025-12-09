"""
File: retriever.py

Module that implements the TemplateRetriever class for semantic search
and retrieval of code templates. Uses FAISS indexing with OpenAI embeddings
to efficiently retrieve relevant templates based on user requests.

"""

import os
import faiss
import numpy as np
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


class TemplateRetriever:
    """
    Retrieves relevant code templates using semantic search with FAISS indexing.
    
    Loads templates from a directory, generates embeddings using OpenAI's
    text-embedding model, and provides similarity-based retrieval to find
    relevant templates matching user requests.
    """
    def __init__(self, template_dir: str = "data/templates"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.template_dir = template_dir
        self.templates: List[Dict] = []
        self.index = None
        self._load_and_index()

    def _get_embedding(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding, dtype='float32')

    def _load_and_index(self):
        """Loads templates from directory and builds FAISS index."""
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir, exist_ok=True)
            return

        documents = []
        embeddings = []

        for filename in os.listdir(self.template_dir):
            if filename.endswith(".py"):
                path = os.path.join(self.template_dir, filename)
                with open(path, "r") as f:
                    content = f.read()
                
                # Simple metadata extraction (filename as tag)
                self.templates.append({"filename": filename, "content": content})
                # Use filename + first 100 chars as embedding context
                context = f"{filename}: {content[:200]}"
                documents.append(context)
                embeddings.append(self._get_embedding(context))

        if embeddings:
            dimension = len(embeddings[0])
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.vstack(embeddings))
            print(f"Indexed {len(self.templates)} templates.")

    def retrieve(self, query: str, k: int = 2) -> List[Dict]:
        """Retrieves top-k relevant templates."""
        if not self.index or self.index.ntotal == 0:
            return []

        query_vec = self._get_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.templates):
                results.append(self.templates[idx])
        
        return results

if __name__ == "__main__":
    retriever = TemplateRetriever()
    # Mocking data add if empty for testing
    if not retriever.templates:
        print("No templates found in dir. Please add .py files to data/templates")
