# ============================================================================
# File: rag_system.py
# ============================================================================
# Description:
#   Implements a Retrieval-Augmented Generation (RAG) system for managing,
#   searching, and retrieving machine learning templates and best practices.
#   The system uses ChromaDB for vector storage with Sentence Transformers
#   for embeddings and OpenAI's API for semantic search and augmentation.
#
#
# Dependencies:
#   - chromadb: Vector database for embeddings and retrieval
#   - sentence-transformers: Embedding model for text vectorization
#   - openai: API client for GPT-based augmentation
#   - logging: Python standard library for logging
#
# Key Classes:
#   - RAGSystem: Main class for RAG operations including template management
#                and semantic search capabilities
#
# ============================================================================
"""
RAG System for ML templates and best practices
"""
import logging
from typing import List, Dict
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

OPENAI_API_KEY="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"

logger = logging.getLogger(__name__)

class RAGSystem:
    """Retrieval-Augmented Generation system for ML templates"""
    
    def __init__(self, collection_name: str = "ml_templates"):
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        #self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(collection_name)
            self._initialize_templates()
            logger.info(f"Created new collection: {collection_name}")
    
    def _initialize_templates(self):
        """Initialize with ML templates and best practices"""
        templates = [
            {
                "id": "pytorch_training_loop",
                "content": """PyTorch Training Loop Best Practices:
                1. Set model to training mode: model.train()
                2. Zero gradients: optimizer.zero_grad()
                3. Forward pass: outputs = model(inputs)
                4. Compute loss: loss = criterion(outputs, targets)
                5. Backward pass: loss.backward()
                6. Update weights: optimizer.step()
                7. Track metrics and log progress
                8. Use tqdm for progress bars
                9. Move data to correct device (CPU/GPU)
                10. Handle gradient accumulation for large batches""",
                "category": "training"
            },
            {
                "id": "pytorch_validation",
                "content": """PyTorch Validation Best Practices:
                1. Set model to eval mode: model.eval()
                2. Disable gradient computation: with torch.no_grad()
                3. Iterate through validation data
                4. Compute metrics without updating weights
                5. Track best model based on validation metric
                6. Return average loss and metrics""",
                "category": "validation"
            },
            {
                "id": "model_checkpointing",
                "content": """Model Checkpointing:
                1. Save model state_dict, not entire model
                2. Include optimizer state for resuming training
                3. Save epoch number and best metric
                4. Use torch.save() with dictionary
                5. Keep only best N checkpoints to save space
                6. Include timestamp in checkpoint filename""",
                "category": "checkpointing"
            },
            {
                "id": "data_loading",
                "content": """Data Loading Best Practices:
                1. Use Dataset class from torch.utils.data
                2. Implement __init__, __len__, __getitem__
                3. Apply transforms in __getitem__
                4. Use DataLoader with num_workers for parallel loading
                5. Pin memory for GPU training: pin_memory=True
                6. Shuffle training data, not validation
                7. Use appropriate batch size for your GPU memory""",
                "category": "data"
            },
            {
                "id": "azure_ml_deployment",
                "content": """Azure ML Deployment:
                1. Create environment with dependencies
                2. Define command job with inputs/outputs
                3. Specify compute target (CPU/GPU cluster)
                4. Use MLflow for experiment tracking
                5. Register model after training
                6. Create endpoint for inference
                7. Monitor performance and costs""",
                "category": "deployment"
            },
            {
                "id": "model_architecture",
                "content": """Neural Network Architecture:
                1. Inherit from nn.Module
                2. Define layers in __init__
                3. Implement forward method
                4. Use appropriate activation functions
                5. Add dropout for regularization
                6. Use batch normalization for stability
                7. Initialize weights properly (Xavier, He, etc.)
                8. Output logits for classification, no softmax in forward""",
                "category": "architecture"
            },
            {
                "id": "hyperparameter_tuning",
                "content": """Hyperparameter Tuning:
                1. Start with learning rate (most important)
                2. Tune batch size based on memory
                3. Adjust number of layers and units
                4. Tune dropout rate for regularization
                5. Experiment with optimizers (Adam, SGD, AdamW)
                6. Use learning rate scheduling
                7. Track all experiments with MLflow""",
                "category": "tuning"
            },
            {
                "id": "testing_ml_code",
                "content": """Testing ML Code:
                1. Test model initialization
                2. Test forward pass with known input shapes
                3. Test loss computation
                4. Test overfitting on small batch (sanity check)
                5. Test gradient flow
                6. Test data loading and transformations
                7. Test model save/load
                8. Use pytest fixtures for reusable components""",
                "category": "testing"
            }
        ]
        
        for template in templates:
            self.collection.add(
                ids=[template["id"]],
                documents=[template["content"]],
                metadatas=[{"category": template["category"]}]
            )
        
        logger.info(f"Initialized {len(templates)} templates")
    
    async def query(self, query_text: str, top_k: int = 3) -> str:
        """Query RAG system for relevant templates"""
        try:
            # Query collection
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k
            )
            
            if results and results['documents']:
                # Combine retrieved documents
                context = "\n\n".join(results['documents'][0])
                return context
            
            return "No relevant templates found."
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return ""
    
    def add_template(self, template_id: str, content: str, category: str):
        """Add new template to collection"""
        self.collection.add(
            ids=[template_id],
            documents=[content],
            metadatas=[{"category": category}]
        )
        logger.info(f"Added template: {template_id}")
    
    def get_all_categories(self) -> List[str]:
        """Get all template categories"""
        results = self.collection.get()
        if results and 'metadatas' in results:
            categories = set(m.get('category') for m in results['metadatas'] if m)
            return list(categories)
        return []


