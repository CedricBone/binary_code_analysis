"""
PalmTree embedding model for binary instructions.

This is an implementation based on the contextual embedding approach described in:
"PalmTree: Learning an Assembly Language Model for Instruction Embedding"
"""

import numpy as np
import random
from .base import BaseEmbedding

class PalmTreeEmbedding(BaseEmbedding):
    """
    PalmTree embeddings for binary instructions.
    
    PalmTree uses contextual information and instruction semantics to create
    more accurate embeddings for assembly instructions.
    """
    
    def __init__(self, embedding_dim=100, context_size=5, alpha=0.01, 
                 negative_samples=5, epochs=5, **kwargs):
        """
        Initialize the PalmTree embedding model.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            context_size: Size of instruction context window
            alpha: Learning rate
            negative_samples: Number of negative samples per positive sample
            epochs: Number of training epochs
            **kwargs: Additional model parameters
        """
        super().__init__(embedding_dim=embedding_dim, **kwargs)
        self.context_size = context_size
        self.alpha = alpha
        self.negative_samples = negative_samples
        self.epochs = epochs
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.token_embeddings = None
        self.context_embeddings = None
    
    def fit(self, instructions, **kwargs):
        """
        Train the PalmTree model on the given instructions.
        
        Args:
            instructions: List of tokenized instruction sequences (list of lists)
            **kwargs: Additional training parameters
        """
        # Build vocabulary
        unique_tokens = set()
        for instr_seq in instructions:
            unique_tokens.update(instr_seq)
        
        self.instruction_vocab = sorted(list(unique_tokens))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.instruction_vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        
        # Initialize embedding matrices
        vocab_size = len(self.instruction_vocab)
        self.token_embeddings = np.random.normal(
            scale=0.1, 
            size=(vocab_size, self.embedding_dim)
        )
        self.context_embeddings = np.random.normal(
            scale=0.1, 
            size=(vocab_size, self.embedding_dim)
        )
        
        # Training with simplified Skip-gram with negative sampling
        for epoch in range(self.epochs):
            for instr_seq in instructions:
                seq_indices = [self.token_to_idx[token] for token in instr_seq if token in self.token_to_idx]
                
                # Skip short sequences
                if len(seq_indices) < 2:
                    continue
                
                # For each token, update embeddings based on context
                for i, target_idx in enumerate(seq_indices):
                    # Define context window
                    context_start = max(0, i - self.context_size)
                    context_end = min(len(seq_indices), i + self.context_size + 1)
                    context_indices = seq_indices[context_start:i] + seq_indices[i+1:context_end]
                    
                    # Skip if no context
                    if not context_indices:
                        continue
                    
                    # Update embeddings for each context pair
                    for context_idx in context_indices:
                        # Positive sample update
                        self._update_embeddings(target_idx, context_idx, True)
                        
                        # Negative sample updates
                        for _ in range(self.negative_samples):
                            negative_idx = random.randint(0, vocab_size - 1)
                            while negative_idx in context_indices or negative_idx == target_idx:
                                negative_idx = random.randint(0, vocab_size - 1)
                            
                            self._update_embeddings(target_idx, negative_idx, False)
        
        # Store the model
        self.model = {
            'token_to_idx': self.token_to_idx,
            'idx_to_token': self.idx_to_token,
            'token_embeddings': self.token_embeddings,
            'context_embeddings': self.context_embeddings,
            'vocab_size': vocab_size
        }
        
        return self
    
    def _update_embeddings(self, target_idx, context_idx, is_positive):
        """
        Update embeddings for a target-context pair.
        
        Args:
            target_idx: Index of the target token
            context_idx: Index of the context token
            is_positive: Whether this is a positive sample
        """
        # Get current vectors
        target_vec = self.token_embeddings[target_idx]
        context_vec = self.context_embeddings[context_idx]
        
        # Compute prediction (sigmoid of dot product)
        dot_product = np.dot(target_vec, context_vec)
        prediction = 1.0 / (1.0 + np.exp(-dot_product))
        
        # Compute error
        label = 1.0 if is_positive else 0.0
        error = label - prediction
        
        # Update vectors
        gradient = error * self.alpha
        
        # Update embeddings
        self.token_embeddings[target_idx] += gradient * context_vec
        self.context_embeddings[context_idx] += gradient * target_vec
    
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
            # Instruction sequences
            embeddings = []
            for instr_seq in instructions:
                token_vecs = []
                for token in instr_seq:
                    if token in self.token_to_idx:
                        idx = self.token_to_idx[token]
                        token_vecs.append(self.token_embeddings[idx])
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
            embeddings = []
            for instr in instructions:
                if instr in self.token_to_idx:
                    idx = self.token_to_idx[instr]
                    embeddings.append(self.token_embeddings[idx])
                else:
                    embeddings.append(np.zeros(self.embedding_dim))
            
            return np.array(embeddings)