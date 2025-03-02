"""
Graph-based embedding model for binary instructions.

This implements a model inspired by Inst2Vec and other graph-based
approaches for binary code representation.
"""

import os
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
from collections import defaultdict

from .base import BaseEmbedding

class GNN(nn.Module):
    """Graph Neural Network for instruction embeddings."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the GNN model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden representations
            output_dim: Dimension of output embeddings
        """
        super(GNN, self).__init__()
        
        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        # Linear layers for final embedding
        self.lin1 = nn.Linear(output_dim, output_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features
            edge_index: Graph edge indices
            batch: Batch assignments for nodes
            
        Returns:
            torch.Tensor: Graph embeddings
        """
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third GCN layer
        x = self.conv3(x, edge_index)
        
        # Global pooling to get graph-level embeddings
        x = global_mean_pool(x, batch)
        
        # Final linear layer
        x = self.lin1(x)
        
        return x

class GraphEmbedding(BaseEmbedding):
    """Graph neural network based embeddings for assembly."""
    
    def __init__(self, embedding_dim=100, hidden_dim=128, initial_dim=64, 
                 batch_size=32, epochs=50, learning_rate=0.001, device=None, **kwargs):
        """
        Initialize the graph embedding model.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            hidden_dim: Dimension of hidden layers in GNN
            initial_dim: Dimension of initial node features
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            device: Device to use (cpu or cuda)
            **kwargs: Additional model parameters
        """
        super().__init__(embedding_dim=embedding_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.initial_dim = initial_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize token mapping
        self.token_to_idx = {}
        self.idx_to_token = {}
        
        # Initialize GNN model
        self.gnn = None
        
        # Store vocabulary and node features
        self.node_features = None
    
    def _tokenize_instruction(self, instruction):
        """
        Tokenize an instruction into components.
        
        Args:
            instruction: Instruction string
            
        Returns:
            list: List of instruction components
        """
        # Split instruction into opcode and operands
        parts = instruction.split(None, 1)
        opcode = parts[0]
        
        tokens = [opcode]
        
        # Process operands if present
        if len(parts) > 1:
            operands = parts[1].split(',')
            for operand in operands:
                tokens.append(operand.strip())
        
        return tokens
    
    def _build_instruction_graph(self, instruction_sequences):
        """
        Build a graph representation of instruction sequences.
        
        Args:
            instruction_sequences: List of instruction sequences
            
        Returns:
            list: List of graph data objects
        """
        graphs = []
        
        # Create vocabulary of unique tokens
        all_tokens = set()
        for sequence in instruction_sequences:
            for instruction in sequence:
                tokens = self._tokenize_instruction(instruction)
                all_tokens.update(tokens)
        
        # Create token mapping
        self.token_to_idx = {token: i for i, token in enumerate(sorted(all_tokens))}
        self.idx_to_token = {i: token for token, i in self.token_to_idx.items()}
        
        # Initialize node features
        self.node_features = nn.Embedding(len(self.token_to_idx), self.initial_dim)
        
        # Build a graph for each instruction sequence
        for sequence in instruction_sequences:
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add nodes for each instruction
            for i, instruction in enumerate(sequence):
                G.add_node(i, instruction=instruction)
            
            # Add control flow edges
            for i in range(len(sequence) - 1):
                instr = sequence[i].lower()
                
                # Check if instruction is a jump or branch
                if any(op in instr for op in ['jmp', 'je', 'jne', 'jg', 'jl', 'jge', 'jle', 'ja', 'jb']):
                    # For now, simplistically add an edge to a random node
                    # In a real implementation, you'd parse the jump target
                    target = min(i + 2, len(sequence) - 1)
                    G.add_edge(i, target, type='control')
                else:
                    # Sequential control flow
                    G.add_edge(i, i + 1, type='control')
            
            # Extract features and edges for PyTorch Geometric
            x = []
            for i in range(len(sequence)):
                # Get tokens for this instruction
                tokens = self._tokenize_instruction(sequence[i])
                
                # For each instruction, average the embeddings of its tokens
                token_indices = [self.token_to_idx[token] for token in tokens]
                token_features = self.node_features(torch.tensor(token_indices)).mean(dim=0)
                
                x.append(token_features)
            
            # Convert node features to tensor
            x = torch.stack(x)
            
            # Extract edges
            edge_index = []
            for u, v in G.edges():
                edge_index.append([u, v])
            
            # Convert to PyTorch Geometric Data object
            if edge_index:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                data = Data(x=x, edge_index=edge_index)
                graphs.append(data)
        
        return graphs
    
    def fit(self, instructions, **kwargs):
        """
        Train the graph embedding model on the given instructions.
        
        Args:
            instructions: List of tokenized instruction sequences
            **kwargs: Additional training parameters
        """
        # Ensure instructions are tokenized (list of lists)
        if not all(isinstance(instr, list) for instr in instructions):
            raise ValueError("Instructions must be tokenized (list of lists)")
        
        # Store vocabulary
        self.instruction_vocab = []
        for instr_seq in instructions:
            self.instruction_vocab.extend(instr_seq)
        self.instruction_vocab = list(set(self.instruction_vocab))
        
        # Build instruction graphs
        graphs = self._build_instruction_graph(instructions)
        
        # Create dataloader
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=True)
        
        # Initialize GNN model
        self.gnn = GNN(self.initial_dim, self.hidden_dim, self.embedding_dim).to(self.device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            list(self.node_features.parameters()) + list(self.gnn.parameters()), 
            lr=self.learning_rate
        )
        
        # Training loop
        self.gnn.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                batch = batch.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                embeddings = self.gnn(batch.x, batch.edge_index, batch.batch)
                
                # Compute contrastive loss
                # For simplicity, we use a triplet loss where sequential instructions are positive pairs
                # and random instructions are negative
                loss = self._contrastive_loss(embeddings, batch.batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")
        
        # Set model to evaluation mode
        self.gnn.eval()
        
        # Store the model
        self.model = {
            'gnn': self.gnn,
            'node_features': self.node_features,
            'token_to_idx': self.token_to_idx,
            'idx_to_token': self.idx_to_token
        }
        
        return self
    
    def _contrastive_loss(self, embeddings, batch_indices):
        """
        Compute contrastive loss for graph embeddings.
        
        Args:
            embeddings: Graph embeddings
            batch_indices: Batch indices for each node
            
        Returns:
            torch.Tensor: Contrastive loss
        """
        # Group embeddings by batch
        batch_dict = defaultdict(list)
        for i, batch_idx in enumerate(batch_indices.cpu().numpy()):
            batch_dict[batch_idx].append(i)
        
        loss = 0
        count = 0
        
        for batch_idx, indices in batch_dict.items():
            if len(indices) > 1:
                # For each embedding in this batch
                for i in range(len(indices)):
                    # Positive sample: next instruction in sequence
                    pos_idx = (i + 1) % len(indices)
                    
                    # Negative sample: random instruction from different batch
                    neg_batch = batch_idx
                    while neg_batch == batch_idx:
                        neg_batch = np.random.choice(list(batch_dict.keys()))
                    neg_idx = np.random.choice(batch_dict[neg_batch])
                    
                    # Compute triplet loss
                    anchor = embeddings[indices[i]]
                    positive = embeddings[indices[pos_idx]]
                    negative = embeddings[neg_idx]
                    
                    pos_dist = F.pairwise_distance(anchor.unsqueeze(0), positive.unsqueeze(0))
                    neg_dist = F.pairwise_distance(anchor.unsqueeze(0), negative.unsqueeze(0))
                    
                    # Triplet loss with margin
                    triplet_loss = F.relu(pos_dist - neg_dist + 0.5)
                    loss += triplet_loss
                    count += 1
        
        return loss / max(1, count)
    
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
            # Instruction sequences - build graphs and get embeddings
            graphs = self._build_instruction_graph(instructions)
            
            # Create dataloader
            loader = DataLoader(graphs, batch_size=self.batch_size)
            
            # Generate embeddings
            embeddings = []
            self.gnn.eval()
            
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    batch_embeddings = self.gnn(batch.x, batch.edge_index, batch.batch)
                    embeddings.extend(batch_embeddings.cpu().numpy())
            
            return np.array(embeddings)
        else:
            # Single instructions - treat each as a one-instruction sequence
            sequence_embeddings = self.transform([[instr] for instr in instructions])
            return sequence_embeddings
    
    def save(self, path):
        """
        Save the embedding model to disk.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model components
        model_path = os.path.join(os.path.dirname(path), "graph_model.pt")
        features_path = os.path.join(os.path.dirname(path), "node_features.pt")
        
        torch.save(self.gnn.state_dict(), model_path)
        torch.save(self.node_features.state_dict(), features_path)
        
        # Save additional data
        data = {
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'initial_dim': self.initial_dim,
            'token_to_idx': self.token_to_idx,
            'idx_to_token': self.idx_to_token,
            'instruction_vocab': self.instruction_vocab,
            'model_path': model_path,
            'features_path': features_path
        }
        
        torch.save(data, path)
    
    @classmethod
    def load(cls, path):
        """
        Load an embedding model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            GraphEmbedding: Loaded embedding model
        """
        # Load data
        data = torch.load(path, map_location=torch.device("cpu"))
        
        # Create instance
        instance = cls(
            embedding_dim=data['embedding_dim'],
            hidden_dim=data['hidden_dim'],
            initial_dim=data['initial_dim']
        )
        
        # Set attributes
        instance.token_to_idx = data['token_to_idx']
        instance.idx_to_token = data['idx_to_token']
        instance.instruction_vocab = data['instruction_vocab']
        
        # Load node features
        instance.node_features = nn.Embedding(len(instance.token_to_idx), instance.initial_dim)
        instance.node_features.load_state_dict(torch.load(data['features_path'], map_location=torch.device("cpu")))
        
        # Initialize and load GNN
        instance.gnn = GNN(instance.initial_dim, instance.hidden_dim, instance.embedding_dim)
        instance.gnn.load_state_dict(torch.load(data['model_path'], map_location=torch.device("cpu")))
        
        # Move to device
        instance.node_features.to(instance.device)
        instance.gnn.to(instance.device)
        
        # Store model
        instance.model = {
            'gnn': instance.gnn,
            'node_features': instance.node_features,
            'token_to_idx': instance.token_to_idx,
            'idx_to_token': instance.idx_to_token
        }
        
        return instance