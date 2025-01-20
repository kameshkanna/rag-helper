import numpy as np
from typing import List, Optional, Dict
import pickle
import logging
from dataclasses import dataclass

@dataclass
class ModelInfo:
    type: str
    dim: int
    description: str
    requires_api: bool = False
    provider: Optional[str] = None

class EasyEmbeddings:
    """A simple interface for generating and managing embeddings from various models."""
    
    # Available models configuration
    MODELS = {
        # OpenAI models
        "text-embedding-ada-002": ModelInfo(
            type="openai",
            dim=1536,
            description="OpenAI's latest embedding model, high quality for most tasks",
            requires_api=True,
            provider="openai"
        ),
        "text-embedding-3-small": ModelInfo(
            type="openai",
            dim=1536,
            description="OpenAI's newer small embedding model, good balance of quality/cost",
            requires_api=True,
            provider="openai"
        ),
        "text-embedding-3-large": ModelInfo(
            type="openai",
            dim=3072,
            description="OpenAI's newer large embedding model, highest quality",
            requires_api=True,
            provider="openai"
        ),

        # Cohere models
        "embed-english-v3.0": ModelInfo(
            type="cohere",
            dim=1024,
            description="Cohere's English-optimized embedding model",
            requires_api=True,
            provider="cohere"
        ),
        "embed-multilingual-v3.0": ModelInfo(
            type="cohere",
            dim=1024,
            description="Cohere's multilingual embedding model",
            requires_api=True,
            provider="cohere"
        ),

        # Sentence Transformers models
        "all-minilm-l6-v2": ModelInfo(
            type="sentence-transformers",
            dim=384,
            description="Lightweight general purpose model, good balance of speed/quality",
            requires_api=False
        ),
        "all-mpnet-base-v2": ModelInfo(
            type="sentence-transformers",
            dim=768,
            description="High quality general purpose model",
            requires_api=False
        ),
        "multi-qa-mpnet-base-dot-v1": ModelInfo(
            type="sentence-transformers",
            dim=768,
            description="Optimized for semantic search and QA",
            requires_api=False
        ),
        
        # HuggingFace models
        "bert-base-uncased": ModelInfo(
            type="transformers",
            dim=768,
            description="Classic BERT model, requires HF token for faster downloads",
            requires_api=True,
            provider="huggingface"
        ),
        "roberta-base": ModelInfo(
            type="transformers",
            dim=768,
            description="RoBERTa model, requires HF token for faster downloads",
            requires_api=True,
            provider="huggingface"
        ),
        
        # BGE models
        "BAAI/bge-large-en": ModelInfo(
            type="bge",
            dim=1024,
            description="Large BGE model for high quality embeddings",
            requires_api=False
        ),
        "BAAI/bge-base-en": ModelInfo(
            type="bge",
            dim=768,
            description="Base BGE model, good performance",
            requires_api=False
        )
    }
    
    @classmethod
    def list_models(cls, include_api_required: bool = True) -> Dict:
        """
        List all available models with their details.
        
        Args:
            include_api_required: If True, includes models that require API keys
        """
        models = {}
        for name, info in cls.MODELS.items():
            if not include_api_required and info.requires_api:
                continue
            models[name] = {
                "dimension": info.dim,
                "description": info.description,
                "requires_api": info.requires_api,
                "provider": info.provider if info.requires_api else None
            }
        return models
    
    def __init__(
        self, 
        model_name: str = "all-minilm-l6-v2",
        api_keys: Optional[Dict[str, str]] = None,
        use_auth_token: bool = False
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the model to use (use list_models() to see options)
            api_keys: Dictionary of API keys for different providers
                     Format: {"openai": "sk-...", "cohere": "...", "huggingface": "..."}
            use_auth_token: Whether to use HuggingFace auth token for downloading models
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Use list_models() to see available options.")
        
        self.model_name = model_name
        self.model_info = self.MODELS[model_name]
        self.api_keys = api_keys or {}
        self.use_auth_token = use_auth_token
        
        # Check API key requirements
        if self.model_info.requires_api:
            provider = self.model_info.provider
            if provider not in self.api_keys:
                raise ValueError(
                    f"API key required for {provider} models. "
                    f"Please provide it in the api_keys dictionary."
                )
        
        self.model = None
        self._setup_model()
    
    def _setup_model(self):
        """Set up the selected model."""
        try:
            if self.model_info.type == "sentence-transformers":
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                
            elif self.model_info.type == "transformers":
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                auth_token = self.api_keys.get("huggingface") if self.use_auth_token else None
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_auth_token=auth_token
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    use_auth_token=auth_token
                )
                
            elif self.model_info.type == "openai":
                import openai
                openai.api_key = self.api_keys["openai"]
                self.model = openai
                
            elif self.model_info.type == "cohere":
                import cohere
                self.model = cohere.Client(self.api_keys["cohere"])
                
            elif self.model_info.type == "bge":
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_info.type}")
                
        except ImportError as e:
            raise ImportError(f"Please install the required package for {self.model_info.type}. Error: {str(e)}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            raise ValueError("Input texts cannot be empty")
            
        try:
            if self.model_info.type == "sentence-transformers" or self.model_info.type == "bge":
                embeddings = self.model.encode(texts)
                
            elif self.model_info.type == "transformers":
                import torch
                
                inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                
            elif self.model_info.type == "openai":
                embeddings = []
                for text in texts:
                    response = self.model.Embedding.create(
                        model=self.model_name,
                        input=text
                    )
                    embeddings.append(response['data'][0]['embedding'])
                embeddings = np.array(embeddings)
                
            elif self.model_info.type == "cohere":
                response = self.model.embed(
                    texts=texts,
                    model=self.model_name
                )
                embeddings = np.array(response.embeddings)
                
            return embeddings
            
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from a file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
