"""
BERT-based embedding model for binary instructions.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
from .base import BaseEmbedding

class AssemblyDataset(Dataset):
    """Dataset for assembly instructions."""
    
    def __init__(self, instructions, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            instructions: List of instructions or instruction sequences
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.instructions = instructions
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.instructions)
    
    def __getitem__(self, idx):
        # Convert instruction or sequence to string
        if isinstance(self.instructions[idx], list):
            text = " ".join(self.instructions[idx])
        else:
            text = self.instructions[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

class BERTAssemblyEmbedding(BaseEmbedding):
    """BERT-based contextual embeddings for assembly code."""
    
    def __init__(self, embedding_dim=768, max_seq_length=128, model_name="bert-base-uncased", 
                 batch_size=32, device=None, **kwargs):
        """
        Initialize the BERT embedding model.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            max_seq_length: Maximum sequence length
            model_name: Pretrained BERT model to use
            batch_size: Batch size for processing
            device: Device to use (cpu or cuda)
            **kwargs: Additional model parameters
        """
        super().__init__(embedding_dim=embedding_dim, **kwargs)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.model_name = model_name
        
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        self.bert_model.to(self.device)
        
        # Add special tokens for assembly
        special_tokens = {
            'additional_special_tokens': [
                '[REG]', '[IMM]', '[MEM]', '[LABEL]',
                # Common registers
                'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp',
                'rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp',
                # Common operations
                'mov', 'push', 'pop', 'call', 'ret', 'jmp', 'add', 'sub'
            ]
        }
        
        # Add special tokens and resize embeddings
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens)
        if num_added_tokens > 0:
            self.bert_model.resize_token_embeddings(len(self.tokenizer))
            
    def fit(self, instructions, epochs=3, learning_rate=5e-5, **kwargs):
        """
        Fine-tune the BERT model on assembly instructions.
        
        Args:
            instructions: List of tokenized instruction sequences
            epochs: Number of training epochs
            learning_rate: Learning rate for fine-tuning
            **kwargs: Additional training parameters
        """
        # Prepare dataset and dataloader
        dataset = AssemblyDataset(instructions, self.tokenizer, self.max_seq_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Set up the optimizer
        optimizer = AdamW(self.bert_model.parameters(), lr=learning_rate)
        
        # Set model to training mode
        self.bert_model.train()
        
        # Store vocabulary
        self.instruction_vocab = []
        for instr_seq in instructions:
            if isinstance(instr_seq, list):
                self.instruction_vocab.extend(instr_seq)
            else:
                self.instruction_vocab.append(instr_seq)
        self.instruction_vocab = list(set(self.instruction_vocab))
        
        # Fine-tuning loop
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Forward pass with masked language modeling
                self.bert_model.zero_grad()
                outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Get the hidden states
                hidden_states = outputs.hidden_states[-1]
                
                # Compute masked language modeling loss
                # For simplicity, we're just using the model's internal loss
                # A more sophisticated approach would use a dedicated MLM head
                
                # Compute mean pooling loss as a simple approach
                pooled_output = outputs.pooler_output
                reconstruction_loss = torch.nn.functional.mse_loss(
                    pooled_output, 
                    hidden_states[:, 0]  # Compare to CLS token embedding
                )
                
                # Backward pass and optimization
                reconstruction_loss.backward()
                optimizer.step()
                
                total_loss += reconstruction_loss.item()
                progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})
        
        # Set model to evaluation mode
        self.bert_model.eval()
        
        # Store the model
        self.model = self.bert_model
        
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
        
        # Prepare dataset and dataloader
        dataset = AssemblyDataset(instructions, self.tokenizer, self.max_seq_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        # Generate embeddings
        embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Use the [CLS] token embedding as the sequence embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def save(self, path):
        """
        Save the embedding model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model and tokenizer
        model_path = os.path.join(os.path.dirname(path), "bert_model")
        tokenizer_path = os.path.join(os.path.dirname(path), "bert_tokenizer")
        
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save additional parameters
        config = {
            "embedding_dim": self.embedding_dim,
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size,
            "model_name": self.model_name,
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "instruction_vocab": self.instruction_vocab
        }
        
        torch.save(config, path)
    
    @classmethod
    def load(cls, path):
        """
        Load an embedding model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            BERTAssemblyEmbedding: Loaded embedding model
        """
        # Load config
        config = torch.load(path, map_location=torch.device("cpu"))
        
        # Create instance
        instance = cls(
            embedding_dim=config["embedding_dim"],
            max_seq_length=config["max_seq_length"],
            batch_size=config["batch_size"],
            model_name=config["model_name"]
        )
        
        # Load model and tokenizer
        instance.model = BertModel.from_pretrained(config["model_path"])
        instance.tokenizer = BertTokenizer.from_pretrained(config["tokenizer_path"])
        instance.model.to(instance.device)
        
        # Set vocabulary
        instance.instruction_vocab = config["instruction_vocab"]
        
        return instance