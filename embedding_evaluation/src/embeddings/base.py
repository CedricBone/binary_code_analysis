"""
Base class for instruction embedding models.
"""

import os
import pickle
from abc import ABC, abstractmethod
import numpy as np

class BaseEmbedding(ABC):
    """Base class for all instruction embedding models."""
    
    def __init__(self, embedding_dim=100, **kwargs):
        """
        Initialize the embedding model.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            **kwargs: Additional model-specific parameters
        """
        self.embedding_dim = embedding_dim
        self.model = None
        self.instruction_vocab = None
    
    @abstractmethod
    def fit(self, instructions, **kwargs):
        """
        Train the embedding model on the given instructions.
        
        Args:
            instructions: List of instructions or instruction sequences
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def transform(self, instructions):
        """
        Transform instructions into embedding vectors.
        
        Args:
            instructions: List of instructions or instruction sequences
            
        Returns:
            numpy.ndarray: Embedding vectors for the instructions
        """
        pass
    
    def fit_transform(self, instructions, **kwargs):
        """
        Train the model and return embeddings for the instructions.
        
        Args:
            instructions: List of instructions or instruction sequences
            **kwargs: Additional training parameters
            
        Returns:
            numpy.ndarray: Embedding vectors for the instructions
        """
        self.fit(instructions, **kwargs)
        return self.transform(instructions)
    
    def save(self, path):
        """
        Save the embedding model to disk.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'embedding_dim': self.embedding_dim,
                'instruction_vocab': self.instruction_vocab
            }, f)
    
    @classmethod
    def load(cls, path):
        """
        Load an embedding model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            BaseEmbedding: Loaded embedding model
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(embedding_dim=data['embedding_dim'])
        instance.model = data['model']
        instance.instruction_vocab = data['instruction_vocab']
        return instance
    
    def instruction_similarity(self, instr1, instr2):
        """
        Calculate similarity between two instructions.
        
        Args:
            instr1: First instruction
            instr2: Second instruction
            
        Returns:
            float: Similarity score between 0 (not similar) and 1 (identical)
        """
        vec1 = self.transform([instr1])[0]
        vec2 = self.transform([instr2])[0]
        
        # Avoid division by zero
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def most_similar(self, instruction, n=10):
        """
        Find the most similar instructions to the given instruction.
        
        Args:
            instruction: The reference instruction
            n: Number of similar instructions to return
            
        Returns:
            list: List of (instruction, similarity) tuples
        """
        if self.instruction_vocab is None:
            raise ValueError("Model has no instruction vocabulary. Train the model first.")
        
        query_vec = self.transform([instruction])[0]
        all_vecs = self.transform(self.instruction_vocab)
        
        # Compute similarities
        similarities = []
        for i, vec in enumerate(all_vecs):
            sim = np.dot(vec, query_vec) / (
                np.linalg.norm(vec) * np.linalg.norm(query_vec)
            )
            similarities.append((self.instruction_vocab[i], sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n
        return similarities[:n]