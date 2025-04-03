# rag-service/src/utils/embeddings.py
import os
import logging
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Check if Ollama is available (optional)
try:
    from langchain_community.embeddings import OllamaEmbeddings
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama embeddings not available, using mock embeddings")

# Import fake embeddings for fallback
from langchain_community.embeddings.fake import FakeEmbeddings

class MockEmbeddings:
    """Mock embeddings for development and testing."""
    
    def __init__(self, size=1536):
        self.size = size
        self.rng = np.random.RandomState(42)  # For consistent embeddings
        self.embedding_cache = {}
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for documents."""
        return [self.embed_query(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Generate a mock embedding for a query."""
        # Use cache for consistent results
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Create a deterministic embedding based on text content
        hash_val = sum(ord(c) for c in text)
        self.rng.seed(hash_val)
        
        embedding = self.rng.randn(self.size).astype(float)
        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        # Cache and return
        self.embedding_cache[text] = embedding.tolist()
        return embedding.tolist()

def get_embeddings_model(config: Dict[str, Any]) -> Any:
    """Get the embedding model based on configuration."""
    provider = config.get("provider", "mock")
    
    # Use mock embeddings for testing
    if provider == "mock":
        logger.info("Using mock embeddings")
        return MockEmbeddings(size=1536)
    
    # Try to use Ollama if available
    if provider == "ollama" and OLLAMA_AVAILABLE:
        model_name = config.get("model_name", "mxbai-embed-large")
        base_url = config.get("base_url", "http://localhost:11434")
        
        try:
            embeddings = OllamaEmbeddings(
                model=model_name,
                base_url=base_url,
            )
            
            # Test embeddings
            embeddings.embed_query("Test embedding")
            logger.info(f"Successfully initialized {model_name} embeddings via Ollama")
            return embeddings
        except Exception as e:
            logger.warning(f"Error initializing Ollama embeddings: {e}")
    
    # Default fallback to mock embeddings
    logger.info("Falling back to mock embeddings")
    return MockEmbeddings(size=1536)