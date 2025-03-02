"""
Embedding models for binary instructions.
"""

from .base import BaseEmbedding
from .word2vec import Word2VecEmbedding
from .palmtree import PalmTreeEmbedding
from .bert_assembly import BERTAssemblyEmbedding
from .graph_embedding import GraphEmbedding
from .baseline_embeddings import TFIDFEmbedding, OneHotEmbedding, NGramEmbedding

# Register available embedding models
EMBEDDINGS = {
    'word2vec': Word2VecEmbedding,
    'palmtree': PalmTreeEmbedding,
    'bert': BERTAssemblyEmbedding,
    'graph': GraphEmbedding,
    'tfidf': TFIDFEmbedding,
    'onehot': OneHotEmbedding,
    'ngram': NGramEmbedding
}

def get_embedding_model(name, **kwargs):
    """
    Get an embedding model by name.
    
    Args:
        name: Name of the embedding model
        **kwargs: Additional parameters for the model
        
    Returns:
        BaseEmbedding: The embedding model
    """
    if name not in EMBEDDINGS:
        raise ValueError(f"Unknown embedding model: {name}. Available models: {list(EMBEDDINGS.keys())}")
    
    return EMBEDDINGS[name](**kwargs)