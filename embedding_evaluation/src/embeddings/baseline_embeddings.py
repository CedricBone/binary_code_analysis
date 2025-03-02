"""
Baseline embedding models for binary instructions.

This module implements simple baseline models for comparison:
- TF-IDF Embeddings
- One-Hot Encodings
- N-gram Embeddings
"""

import os
import numpy as np
import pickle
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from .base import BaseEmbedding

class TFIDFEmbedding(BaseEmbedding):
    """TF-IDF based embeddings for assembly instructions."""
    
    def __init__(self, max_features=100, **kwargs):
        """
        Initialize the TF-IDF embedding model.
        
        Args:
            max_features: Maximum number of features (vocabulary size)
            **kwargs: Additional parameters
        """
        super().__init__(embedding_dim=max_features, **kwargs)
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            analyzer='word',
            tokenizer=lambda x: x.split(),
            preprocessor=lambda x: x,
            token_pattern=None
        )
    
    def fit(self, instructions, **kwargs):
        """
        Train the TF-IDF model on the given instructions.
        
        Args:
            instructions: List of tokenized instruction sequences
            **kwargs: Additional training parameters
        """
        # Flatten instruction sequences to individual instructions
        flattened_instructions = []
        for instr_seq in instructions:
            flattened_instructions.extend(instr_seq)
        
        # Store vocabulary
        self.instruction_vocab = list(set(flattened_instructions))
        
        # Convert instructions to strings for TF-IDF
        instr_strings = [" ".join(instr.split()) if isinstance(instr, str) else " ".join(instr) 
                         for instr in flattened_instructions]
        
        # Fit vectorizer
        self.vectorizer.fit(instr_strings)
        
        # Store model
        self.model = self.vectorizer
        
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
            # Instruction sequences - average the embeddings of each instruction
            embeddings = []
            for instr_seq in instructions:
                instr_strings = [" ".join(instr.split()) if isinstance(instr, str) else " ".join(instr) 
                                for instr in instr_seq]
                
                if instr_strings:
                    seq_vectors = self.vectorizer.transform(instr_strings).toarray()
                    embeddings.append(np.mean(seq_vectors, axis=0))
                else:
                    embeddings.append(np.zeros(self.embedding_dim))
            
            return np.array(embeddings)
        else:
            # Single instructions
            instr_strings = [" ".join(instr.split()) if isinstance(instr, str) else " ".join(instr) 
                            for instr in instructions]
            return self.vectorizer.transform(instr_strings).toarray()
    
    def save(self, path):
        """
        Save the embedding model to disk.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'embedding_dim': self.embedding_dim,
                'max_features': self.max_features,
                'instruction_vocab': self.instruction_vocab
            }, f)
    
    @classmethod
    def load(cls, path):
        """
        Load an embedding model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            TFIDFEmbedding: Loaded embedding model
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(max_features=data['max_features'])
        instance.vectorizer = data['vectorizer']
        instance.embedding_dim = data['embedding_dim']
        instance.instruction_vocab = data['instruction_vocab']
        instance.model = instance.vectorizer
        
        return instance


class OneHotEmbedding(BaseEmbedding):
    """One-hot encodings for assembly instructions."""
    
    def __init__(self, max_vocab_size=1000, **kwargs):
        """
        Initialize the one-hot embedding model.
        
        Args:
            max_vocab_size: Maximum vocabulary size
            **kwargs: Additional parameters
        """
        super().__init__(embedding_dim=max_vocab_size, **kwargs)
        self.max_vocab_size = max_vocab_size
        self.token_to_idx = {}
    
    def fit(self, instructions, **kwargs):
        """
        Train the one-hot embedding model on the given instructions.
        
        Args:
            instructions: List of tokenized instruction sequences
            **kwargs: Additional training parameters
        """
        # Flatten instruction sequences to individual instructions
        flattened_instructions = []
        for instr_seq in instructions:
            flattened_instructions.extend(instr_seq)
        
        # Count instruction frequencies
        instr_counts = Counter(flattened_instructions)
        
        # Take the most common instructions up to max_vocab_size
        self.instruction_vocab = [instr for instr, _ in instr_counts.most_common(self.max_vocab_size)]
        
        # Create token to index mapping
        self.token_to_idx = {token: idx for idx, token in enumerate(self.instruction_vocab)}
        
        # Set embedding dimension to actual vocabulary size
        self.embedding_dim = len(self.instruction_vocab)
        
        # Store model
        self.model = self.token_to_idx
        
        return self
    
    def transform(self, instructions):
        """
        Transform instructions into one-hot vectors.
        
        Args:
            instructions: List of instructions or instruction sequences
            
        Returns:
            numpy.ndarray: One-hot vectors for the instructions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Handle both single instructions and sequences
        if all(isinstance(instr, list) for instr in instructions):
            # Instruction sequences - represent as sum of one-hot vectors
            embeddings = []
            for instr_seq in instructions:
                seq_vector = np.zeros(self.embedding_dim)
                for instr in instr_seq:
                    if instr in self.token_to_idx:
                        idx = self.token_to_idx[instr]
                        seq_vector[idx] = 1
                embeddings.append(seq_vector)
            
            return np.array(embeddings)
        else:
            # Single instructions
            embeddings = np.zeros((len(instructions), self.embedding_dim))
            for i, instr in enumerate(instructions):
                if instr in self.token_to_idx:
                    idx = self.token_to_idx[instr]
                    embeddings[i, idx] = 1
            
            return embeddings
    
    def save(self, path):
        """
        Save the embedding model to disk.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'token_to_idx': self.token_to_idx,
                'embedding_dim': self.embedding_dim,
                'max_vocab_size': self.max_vocab_size,
                'instruction_vocab': self.instruction_vocab
            }, f)
    
    @classmethod
    def load(cls, path):
        """
        Load an embedding model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            OneHotEmbedding: Loaded embedding model
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(max_vocab_size=data['max_vocab_size'])
        instance.token_to_idx = data['token_to_idx']
        instance.embedding_dim = data['embedding_dim']
        instance.instruction_vocab = data['instruction_vocab']
        instance.model = instance.token_to_idx
        
        return instance


class NGramEmbedding(BaseEmbedding):
    """N-gram based embeddings for assembly instructions."""
    
    def __init__(self, embedding_dim=100, ngram_range=(1, 3), **kwargs):
        """
        Initialize the N-gram embedding model.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            ngram_range: Range of n-grams to consider
            **kwargs: Additional parameters
        """
        super().__init__(embedding_dim=embedding_dim, **kwargs)
        self.ngram_range = ngram_range
        self.vectorizer = CountVectorizer(
            max_features=embedding_dim,
            ngram_range=ngram_range,
            analyzer='char',
            lowercase=False
        )
    
    def fit(self, instructions, **kwargs):
        """
        Train the N-gram model on the given instructions.
        
        Args:
            instructions: List of tokenized instruction sequences
            **kwargs: Additional training parameters
        """
        # Flatten instruction sequences to individual instructions
        flattened_instructions = []
        for instr_seq in instructions:
            flattened_instructions.extend(instr_seq)
        
        # Store vocabulary
        self.instruction_vocab = list(set(flattened_instructions))
        
        # Fit vectorizer
        self.vectorizer.fit(flattened_instructions)
        
        # Store model
        self.model = self.vectorizer
        
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
            # Instruction sequences - average the embeddings of each instruction
            embeddings = []
            for instr_seq in instructions:
                if instr_seq:
                    seq_vectors = self.vectorizer.transform(instr_seq).toarray()
                    embeddings.append(np.mean(seq_vectors, axis=0))
                else:
                    embeddings.append(np.zeros(self.embedding_dim))
            
            return np.array(embeddings)
        else:
            # Single instructions
            return self.vectorizer.transform(instructions).toarray()
    
    def save(self, path):
        """
        Save the embedding model to disk.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'embedding_dim': self.embedding_dim,
                'ngram_range': self.ngram_range,
                'instruction_vocab': self.instruction_vocab
            }, f)
    
    @classmethod
    def load(cls, path):
        """
        Load an embedding model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            NGramEmbedding: Loaded embedding model
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            embedding_dim=data['embedding_dim'],
            ngram_range=data['ngram_range']
        )
        instance.vectorizer = data['vectorizer']
        instance.instruction_vocab = data['instruction_vocab']
        instance.model = instance.vectorizer
        
        return instance