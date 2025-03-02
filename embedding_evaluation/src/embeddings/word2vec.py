"""
Word2Vec-based embedding model for binary instructions.
"""

import numpy as np
from gensim.models import Word2Vec
from .base import BaseEmbedding

class Word2VecEmbedding(BaseEmbedding):
    """Word2Vec-based embedding for binary instructions."""
    
    def __init__(self, embedding_dim=100, window=5, min_count=1, workers=4, **kwargs):
        """
        Initialize the Word2Vec embedding model.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            window: Maximum distance between current and predicted instruction
            min_count: Minimum frequency of instructions to consider
            workers: Number of worker threads
            **kwargs: Additional Word2Vec parameters
        """
        super().__init__(embedding_dim=embedding_dim, **kwargs)
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.additional_params = kwargs
    
    def fit(self, instructions, **kwargs):
        """
        Train the Word2Vec model on the given instructions.
        
        Args:
            instructions: List of tokenized instruction sequences (list of lists)
            **kwargs: Additional training parameters
        """
        # Ensure instructions are tokenized (list of lists)
        if not all(isinstance(instr, list) for instr in instructions):
            raise ValueError("Instructions must be tokenized (list of lists)")
        
        # Store unique tokens in vocabulary
        all_tokens = []
        for instr_seq in instructions:
            all_tokens.extend(instr_seq)
        self.instruction_vocab = list(set(all_tokens))
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=instructions,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            **{**self.additional_params, **kwargs}
        )
        
        return self
    
    def transform(self, instructions):
        """
        Transform instructions into embedding vectors.
        
        Args:
            instructions: List of instructions or instruction sequences
            
        Returns:
            numpy.ndarray: Embedding vectors for the instructions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Handle both single instructions and sequences
        if all(isinstance(instr, list) for instr in instructions):
            # Instruction sequences - average the embeddings of each token
            embeddings = []
            for instr_seq in instructions:
                token_vecs = []
                for token in instr_seq:
                    if token in self.model.wv:
                        token_vecs.append(self.model.wv[token])
                    else:
                        # Handle OOV tokens
                        token_vecs.append(np.zeros(self.embedding_dim))
                
                if token_vecs:
                    embeddings.append(np.mean(token_vecs, axis=0))
                else:
                    embeddings.append(np.zeros(self.embedding_dim))
            
            return np.array(embeddings)
        else:
            # Single instructions
            return np.array([
                self.model.wv[instr] if instr in self.model.wv 
                else np.zeros(self.embedding_dim)
                for instr in instructions
            ])
    
    def instruction_similarity(self, instr1, instr2):
        """
        Calculate similarity between two instructions.
        
        Args:
            instr1: First instruction (string or token list)
            instr2: Second instruction (string or token list)
            
        Returns:
            float: Similarity score between 0 (not similar) and 1 (identical)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Handle instruction format
        if isinstance(instr1, list) and isinstance(instr2, list):
            # Token lists - convert to vectors
            vec1 = self.transform([instr1])[0]
            vec2 = self.transform([instr2])[0]
        else:
            # Single tokens
            vec1 = self.model.wv[instr1] if instr1 in self.model.wv else np.zeros(self.embedding_dim)
            vec2 = self.model.wv[instr2] if instr2 in self.model.wv else np.zeros(self.embedding_dim)
        
        # Cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)