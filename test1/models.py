"""
Model architectures for binary similarity and description generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any


class InstructionEmbedding(nn.Module):
    """Embedding layer for assembly instructions"""
    
    def __init__(self,
                 opcode_vocab_size: int,
                 operand_vocab_size: int,
                 embedding_dim: int,
                 dropout: float = 0.1):
        """
        Initialize the embedding layer
        
        Args:
            opcode_vocab_size: Size of opcode vocabulary
            operand_vocab_size: Size of operand vocabulary
            embedding_dim: Dimension of embeddings
            dropout: Dropout rate
        """
        super().__init__()
        self.opcode_embedding = nn.Embedding(opcode_vocab_size, embedding_dim)
        self.operand_embedding = nn.Embedding(operand_vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                opcode_ids: torch.Tensor, 
                operand_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed opcode and operands
        
        Args:
            opcode_ids: Tensor of opcode IDs [batch_size, seq_len]
            operand_ids: Tensor of operand IDs [batch_size, seq_len, max_operands]
            
        Returns:
            Embedded instructions [batch_size, seq_len, embedding_dim]
        """
        # Embed opcodes [batch_size, seq_len, embedding_dim]
        opcode_embeddings = self.opcode_embedding(opcode_ids)
        
        # Embed operands [batch_size, seq_len, max_operands, embedding_dim]
        operand_embeddings = self.operand_embedding(operand_ids)
        
        # Average operand embeddings for each instruction
        # [batch_size, seq_len, embedding_dim]
        # Skip padding tokens (zeros)
        operand_mask = (operand_ids != 0).float().unsqueeze(-1)
        operand_sum = torch.sum(operand_embeddings * operand_mask, dim=2)
        operand_count = torch.sum(operand_mask, dim=2) + 1e-10  # Avoid division by zero
        operand_avg = operand_sum / operand_count
        
        # Combine opcode and operand embeddings
        # [batch_size, seq_len, embedding_dim]
        instruction_embeddings = opcode_embeddings + operand_avg
        
        return self.dropout(instruction_embeddings)


class SAFEEncoder(nn.Module):
    """
    Self-Attentive Function Embedding (SAFE) encoder
    - Bi-directional LSTM with self-attention
    """
    
    def __init__(self,
                 opcode_vocab_size: int,
                 operand_vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 attention_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize the SAFE encoder
        
        Args:
            opcode_vocab_size: Size of opcode vocabulary
            operand_vocab_size: Size of operand vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of LSTM hidden states
            num_layers: Number of LSTM layers
            attention_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding = InstructionEmbedding(
            opcode_vocab_size, operand_vocab_size, embedding_dim, dropout
        )
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Bidirectional, so we split the hidden size
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout)
        
        # Output layer for function embedding
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        
    def forward(self, 
                opcode_ids: torch.Tensor, 
                operand_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a function
        
        Args:
            opcode_ids: Tensor of opcode IDs [batch_size, seq_len]
            operand_ids: Tensor of operand IDs [batch_size, seq_len, max_operands]
            
        Returns:
            Tuple of (function embedding, attention weights)
            - function embedding: [batch_size, hidden_dim]
            - attention_weights: [batch_size, seq_len]
        """
        # Embed instructions [batch_size, seq_len, embedding_dim]
        instruction_embeddings = self.embedding(opcode_ids, operand_ids)
        
        # Create padding mask (1 for real tokens, 0 for padding)
        padding_mask = (opcode_ids != 0).float()
        
        # Pass through LSTM [batch_size, seq_len, hidden_dim]
        lstm_output, _ = self.lstm(instruction_embeddings)
        
        # Apply self-attention
        # Transpose for attention: [seq_len, batch_size, hidden_dim]
        lstm_output = lstm_output.transpose(0, 1)
        
        # Self-attention
        attn_output, attn_weights = self.attention(
            lstm_output, lstm_output, lstm_output,
            key_padding_mask=(padding_mask == 0)  # Flip mask for PyTorch attention
        )
        
        # Transpose back: [batch_size, seq_len, hidden_dim]
        attn_output = attn_output.transpose(0, 1)
        
        # Apply padding mask to get valid token representations
        masked_output = attn_output * padding_mask.unsqueeze(-1)
        
        # Average pooling over sequence
        sum_embeddings = torch.sum(masked_output, dim=1)
        seq_lengths = torch.sum(padding_mask, dim=1, keepdim=True) + 1e-10
        function_embedding = sum_embeddings / seq_lengths
        
        # Final projection
        function_embedding = self.fc(function_embedding)
        
        return function_embedding, attn_weights.mean(dim=1)


class GNNLayer(nn.Module):
    """
    Graph Neural Network Layer for CFG processing
    Inspired by Structure2Vec (S2V) used in Gemini
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int):
        """
        Initialize GNN layer
        
        Args:
            feature_dim: Dimension of input node features
            hidden_dim: Dimension of output node embeddings
        """
        super().__init__()
        
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        
    def forward(self, 
                node_features: torch.Tensor, 
                node_embeddings: torch.Tensor, 
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Update node embeddings
        
        Args:
            node_features: Node features [batch_size, num_nodes, feature_dim]
            node_embeddings: Current node embeddings [batch_size, num_nodes, hidden_dim]
            adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Updated node embeddings [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, _ = node_features.size()
        
        # Project node features
        feature_proj = self.feature_proj(node_features)
        
        # Aggregate neighbor embeddings using adjacency matrix
        # [batch_size, num_nodes, hidden_dim]
        neighbor_embeddings = torch.bmm(adj_matrix, node_embeddings)
        neighbor_proj = self.neighbor_proj(neighbor_embeddings)
        
        # Combine feature projection and neighbor projection
        combined = feature_proj + neighbor_proj
        combined = F.relu(combined)
        
        # Update node embeddings
        updated_embeddings = []
        for i in range(batch_size):
            # [num_nodes, hidden_dim]
            updated = self.update(
                combined[i],
                node_embeddings[i]
            )
            updated_embeddings.append(updated)
            
        # Stack to get [batch_size, num_nodes, hidden_dim]
        return torch.stack(updated_embeddings)


class GeminiEncoder(nn.Module):
    """
    Graph-based encoder for binary functions
    Inspired by Gemini's Structure2Vec approach
    """
    
    def __init__(self,
                 node_feature_dim: int,
                 hidden_dim: int,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """
        Initialize Gemini encoder
        
        Args:
            node_feature_dim: Dimension of node features
            hidden_dim: Dimension of node embeddings
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Initial node embedding layer
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        
        # Stack of GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(node_feature_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        
    def forward(self, 
                node_features: torch.Tensor, 
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Encode a function's CFG
        
        Args:
            node_features: Node features [batch_size, num_nodes, feature_dim]
            adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Function embedding [batch_size, hidden_dim]
        """
        batch_size, num_nodes, _ = node_features.size()
        
        # Initial node embeddings
        node_embeddings = self.node_embedding(node_features)
        node_embeddings = F.relu(node_embeddings)
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            node_embeddings = gnn_layer(node_features, node_embeddings, adj_matrix)
            node_embeddings = self.dropout(node_embeddings)
        
        # Average pooling over nodes to get graph embedding
        graph_embedding = torch.mean(node_embeddings, dim=1)
        
        # Final projection
        function_embedding = self.fc(graph_embedding)
        
        return function_embedding


class HybridEncoder(nn.Module):
    """
    Hybrid encoder combining sequence and graph-based approaches
    Similar to OrderMatters approach
    """
    
    def __init__(self,
                 opcode_vocab_size: int,
                 operand_vocab_size: int,
                 embedding_dim: int,
                 node_feature_dim: int,
                 hidden_dim: int,
                 num_lstm_layers: int = 2,
                 num_gnn_layers: int = 3,
                 attention_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize hybrid encoder
        
        Args:
            opcode_vocab_size: Size of opcode vocabulary
            operand_vocab_size: Size of operand vocabulary
            embedding_dim: Dimension of embeddings
            node_feature_dim: Dimension of CFG node features
            hidden_dim: Dimension of hidden states
            num_lstm_layers: Number of LSTM layers
            num_gnn_layers: Number of GNN layers
            attention_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Sequence-based encoder
        self.safe_encoder = SAFEEncoder(
            opcode_vocab_size, operand_vocab_size, embedding_dim, hidden_dim,
            num_lstm_layers, attention_heads, dropout
        )
        
        # Graph-based encoder
        self.gemini_encoder = GeminiEncoder(
            node_feature_dim, hidden_dim, num_gnn_layers, dropout
        )
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        
    def forward(self, 
                opcode_ids: torch.Tensor, 
                operand_ids: torch.Tensor, 
                node_features: torch.Tensor, 
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Encode a function using both sequence and graph
        
        Args:
            opcode_ids: Tensor of opcode IDs [batch_size, seq_len]
            operand_ids: Tensor of operand IDs [batch_size, seq_len, max_operands]
            node_features: Node features [batch_size, num_nodes, feature_dim]
            adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Function embedding [batch_size, hidden_dim]
        """
        # Get sequence-based embedding
        seq_embedding, _ = self.safe_encoder(opcode_ids, operand_ids)
        
        # Get graph-based embedding
        graph_embedding = self.gemini_encoder(node_features, adj_matrix)
        
        # Concatenate embeddings
        combined = torch.cat([seq_embedding, graph_embedding], dim=1)
        
        # Fuse embeddings
        function_embedding = self.fusion(combined)
        function_embedding = F.relu(function_embedding)
        
        return self.dropout(function_embedding)


class SiameseSimilarityModel(nn.Module):
    """
    Siamese network for binary similarity detection
    Can use any encoder (SAFE, Gemini, or Hybrid)
    """
    
    def __init__(self, encoder: nn.Module, hidden_dim: int, margin: float = 0.5):
        """
        Initialize Siamese network
        
        Args:
            encoder: Function encoder module
            hidden_dim: Dimension of function embeddings
            margin: Margin for contrastive loss
        """
        super().__init__()
        
        self.encoder = encoder
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.margin = margin
        
    def encode_function(self, func_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode a single function
        
        Args:
            func_data: Dictionary of function data tensors
            
        Returns:
            Function embedding [batch_size, hidden_dim]
        """
        # Check which encoder we're using
        if isinstance(self.encoder, SAFEEncoder):
            embedding, _ = self.encoder(
                func_data['func_opcode_ids'],
                func_data['func_operand_ids']
            )
        
        elif isinstance(self.encoder, GeminiEncoder):
            embedding = self.encoder(
                func_data['func_node_features'],
                func_data['func_adj_matrix']
            )
            
        elif isinstance(self.encoder, HybridEncoder):
            embedding = self.encoder(
                func_data['func_opcode_ids'],
                func_data['func_operand_ids'],
                func_data['func_node_features'],
                func_data['func_adj_matrix']
            )
            
        else:
            raise ValueError(f"Unsupported encoder: {type(self.encoder)}")
            
        # Final projection
        return self.projection(embedding)
    
    def forward(self, 
                func1_data: Dict[str, torch.Tensor], 
                func2_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            func1_data: Dictionary of function 1 data tensors
            func2_data: Dictionary of function 2 data tensors
            
        Returns:
            Tuple of (similarity score, function embeddings)
            - similarity: [batch_size]
            - embeddings: Tuple of function embeddings
        """
        # Encode both functions
        func1_embedding = self.encode_function(func1_data)
        func2_embedding = self.encode_function(func2_data)
        
        # Normalize embeddings
        func1_embedding = F.normalize(func1_embedding, p=2, dim=1)
        func2_embedding = F.normalize(func2_embedding, p=2, dim=1)
        
        # Compute similarity (cosine)
        similarity = torch.sum(func1_embedding * func2_embedding, dim=1)
        
        return similarity, (func1_embedding, func2_embedding)
    
    def compute_loss(self, 
                    similarity: torch.Tensor, 
                    labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            similarity: Cosine similarity scores [batch_size]
            labels: Binary labels (1 for similar, 0 for dissimilar) [batch_size]
            
        Returns:
            Loss value
        """
        # Convert similarity from [-1, 1] to [0, 1]
        similarity = (similarity + 1) / 2
        
        # Contrastive loss
        pos_loss = labels * (1 - similarity) ** 2
        neg_loss = (1 - labels) * torch.clamp(similarity - self.margin, min=0) ** 2
        
        return torch.mean(pos_loss + neg_loss)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for binary functions
    Similar to jTrans approach
    """
    
    def __init__(self,
                 opcode_vocab_size: int,
                 operand_vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_length: int = 200):
        """
        Initialize transformer encoder
        
        Args:
            opcode_vocab_size: Size of opcode vocabulary
            operand_vocab_size: Size of operand vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of transformer hidden states
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        self.embedding = InstructionEmbedding(
            opcode_vocab_size, operand_vocab_size, embedding_dim, dropout
        )
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_seq_length, embedding_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layer
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        
    def forward(self, 
                opcode_ids: torch.Tensor, 
                operand_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a function
        
        Args:
            opcode_ids: Tensor of opcode IDs [batch_size, seq_len]
            operand_ids: Tensor of operand IDs [batch_size, seq_len, max_operands]
            
        Returns:
            Tuple of (function embedding, sequence outputs)
            - function embedding: [batch_size, hidden_dim]
            - sequence outputs: [batch_size, seq_len, embedding_dim]
        """
        # Create padding mask (1 for padding, 0 for real tokens)
        padding_mask = (opcode_ids == 0)
        
        # Embed instructions [batch_size, seq_len, embedding_dim]
        x = self.embedding(opcode_ids, operand_ids)
        
        # Add positional encodings
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Pass through transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Average pooling over sequence (excluding padding)
        # Create a mask for valid positions (1 for real tokens, 0 for padding)
        valid_mask = (padding_mask == 0).float().unsqueeze(-1)
        
        # Apply mask and sum
        masked_x = x * valid_mask
        sum_embeddings = torch.sum(masked_x, dim=1)
        
        # Get valid sequence lengths for averaging
        valid_lengths = torch.sum(valid_mask, dim=1) + 1e-10  # Avoid division by zero
        
        # Average
        pooled = sum_embeddings / valid_lengths
        
        # Final projection
        function_embedding = self.fc(pooled)
        
        return function_embedding, x


class DescriptionGenerator(nn.Module):
    """
    Model for generating descriptions from binary code
    Based on transformer encoder-decoder architecture
    """
    
    def __init__(self,
                 opcode_vocab_size: int,
                 operand_vocab_size: int,
                 desc_vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 encoder_layers: int = 6,
                 decoder_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_length: int = 200,
                 max_desc_length: int = 30,
                 encoder_type: str = 'transformer'):
        """
        Initialize description generator
        
        Args:
            opcode_vocab_size: Size of opcode vocabulary
            operand_vocab_size: Size of operand vocabulary
            desc_vocab_size: Size of description vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden states
            encoder_layers: Number of encoder layers
            decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            max_desc_length: Maximum description length
            encoder_type: Type of encoder ('transformer' or 'safe')
        """
        super().__init__()
        
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.max_desc_length = max_desc_length
        self.desc_vocab_size = desc_vocab_size
        
        # Encoder
        if encoder_type == 'transformer':
            self.encoder = TransformerEncoder(
                opcode_vocab_size, operand_vocab_size, embedding_dim, hidden_dim,
                encoder_layers, num_heads, dropout, max_seq_length
            )
        elif encoder_type == 'safe':
            self.encoder = SAFEEncoder(
                opcode_vocab_size, operand_vocab_size, embedding_dim, hidden_dim,
                encoder_layers // 2, num_heads, dropout
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        # Description embedding
        self.desc_embedding = nn.Embedding(desc_vocab_size, embedding_dim)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_desc_length, embedding_dim)
        )
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim, desc_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                opcode_ids: torch.Tensor, 
                operand_ids: torch.Tensor, 
                desc_ids: torch.Tensor, 
                teacher_forcing: bool = True) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            opcode_ids: Tensor of opcode IDs [batch_size, seq_len]
            operand_ids: Tensor of operand IDs [batch_size, seq_len, max_operands]
            desc_ids: Tensor of description IDs [batch_size, desc_len]
            teacher_forcing: Whether to use teacher forcing
            
        Returns:
            Output logits [batch_size, desc_len, vocab_size]
        """
        batch_size = opcode_ids.size(0)
        
        # Encode function
        if self.encoder_type == 'transformer':
            _, encoder_outputs = self.encoder(opcode_ids, operand_ids)
        else:  # 'safe'
            func_embedding, _ = self.encoder(opcode_ids, operand_ids)
            # Expand to sequence for attention
            encoder_outputs = func_embedding.unsqueeze(1).expand(-1, opcode_ids.size(1), -1)
        
        # Create padding mask for encoder outputs
        encoder_padding_mask = (opcode_ids == 0)
        
        # For training with teacher forcing
        if teacher_forcing:
            # Shift decoder input (remove last token, prepend start token)
            decoder_input = desc_ids[:, :-1]
            
            # Embed description tokens
            # [batch_size, desc_len-1, embedding_dim]
            tgt_embeddings = self.desc_embedding(decoder_input)
            
            # Add positional encodings
            seq_len = tgt_embeddings.size(1)
            tgt_embeddings = tgt_embeddings + self.pos_embedding[:, :seq_len, :]
            
            # Create causal mask for decoder
            # [desc_len-1, desc_len-1]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=tgt_embeddings.device) * float('-inf'),
                diagonal=1
            )
            
            # Create padding mask for decoder input
            tgt_padding_mask = (decoder_input == 0)
            
            # Pass through decoder
            # [batch_size, desc_len-1, embedding_dim]
            decoder_outputs = self.decoder(
                tgt_embeddings, 
                encoder_outputs,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=encoder_padding_mask
            )
            
            # Project to vocabulary
            # [batch_size, desc_len-1, vocab_size]
            logits = self.output_layer(decoder_outputs)
            
            return logits
            
        else:
            # For inference (autoregressive generation)
            # Start with batch of start tokens
            # [batch_size, 1]
            current_ids = torch.full((batch_size, 1), 2, # <START> token id
                                     device=opcode_ids.device, 
                                     dtype=torch.long)
            
            # Store all logits
            all_logits = []
            
            # Generate tokens one by one
            for i in range(self.max_desc_length - 1):
                # Embed current tokens
                # [batch_size, curr_len, embedding_dim]
                tgt_embeddings = self.desc_embedding(current_ids)
                
                # Add positional encodings
                seq_len = tgt_embeddings.size(1)
                tgt_embeddings = tgt_embeddings + self.pos_embedding[:, :seq_len, :]
                
                # Create causal mask for decoder
                # [curr_len, curr_len]
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=tgt_embeddings.device) * float('-inf'),
                    diagonal=1
                )
                
                # Create padding mask for decoder input (none for inference)
                tgt_padding_mask = None
                
                # Pass through decoder
                # [batch_size, curr_len, embedding_dim]
                decoder_outputs = self.decoder(
                    tgt_embeddings, 
                    encoder_outputs,
                    tgt_mask=causal_mask,
                    memory_key_padding_mask=encoder_padding_mask
                )
                
                # Get logits for the last token
                # [batch_size, vocab_size]
                logits = self.output_layer(decoder_outputs[:, -1, :])
                all_logits.append(logits)
                
                # Sample next token
                # [batch_size, 1]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Add to current tokens
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Stop if all sequences have generated <END> token
                if (next_token == 3).all():  # <END> token id
                    break
            
            # Stack all logits
            # [batch_size, generated_len, vocab_size]
            return torch.stack(all_logits, dim=1)
    
    def generate(self, 
                opcode_ids: torch.Tensor, 
                operand_ids: torch.Tensor, 
                max_length: int = None,
                beam_size: int = 1) -> List[List[int]]:
        """
        Generate descriptions
        
        Args:
            opcode_ids: Tensor of opcode IDs [batch_size, seq_len]
            operand_ids: Tensor of operand IDs [batch_size, seq_len, max_operands]
            max_length: Maximum generation length (defaults to self.max_desc_length)
            beam_size: Beam size for beam search (1 = greedy)
            
        Returns:
            List of generated token ID sequences
        """
        if max_length is None:
            max_length = self.max_desc_length
            
        batch_size = opcode_ids.size(0)
        
        # Encode function
        if self.encoder_type == 'transformer':
            _, encoder_outputs = self.encoder(opcode_ids, operand_ids)
        else:  # 'safe'
            func_embedding, _ = self.encoder(opcode_ids, operand_ids)
            # Expand to sequence for attention
            encoder_outputs = func_embedding.unsqueeze(1).expand(-1, opcode_ids.size(1), -1)
        
        # Create padding mask for encoder outputs
        encoder_padding_mask = (opcode_ids == 0)
        
        # If beam_size is 1, use greedy search
        if beam_size == 1:
            return self._greedy_search(
                encoder_outputs, encoder_padding_mask, batch_size, max_length
            )
        else:
            return self._beam_search(
                encoder_outputs, encoder_padding_mask, batch_size, max_length, beam_size
            )
    
    def _greedy_search(self, 
                      encoder_outputs: torch.Tensor, 
                      encoder_padding_mask: torch.Tensor, 
                      batch_size: int, 
                      max_length: int) -> List[List[int]]:
        """
        Greedy search for text generation
        
        Args:
            encoder_outputs: Encoder outputs [batch_size, seq_len, hidden_dim]
            encoder_padding_mask: Padding mask for encoder [batch_size, seq_len]
            batch_size: Batch size
            max_length: Maximum generation length
            
        Returns:
            List of generated token ID sequences
        """
        device = encoder_outputs.device
        
        # Start with batch of start tokens
        # [batch_size, 1]
        current_ids = torch.full((batch_size, 1), 2, # <START> token id
                                device=device, 
                                dtype=torch.long)
        
        # Keep track of finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens one by one
        for i in range(max_length - 1):
            # Stop if all sequences are finished
            if finished.all():
                break
                
            # Embed current tokens
            # [batch_size, curr_len, embedding_dim]
            tgt_embeddings = self.desc_embedding(current_ids)
            
            # Add positional encodings
            seq_len = tgt_embeddings.size(1)
            tgt_embeddings = tgt_embeddings + self.pos_embedding[:, :seq_len, :]
            
            # Create causal mask for decoder
            # [curr_len, curr_len]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                diagonal=1
            )
            
            # Pass through decoder
            # [batch_size, curr_len, embedding_dim]
            decoder_outputs = self.decoder(
                tgt_embeddings, 
                encoder_outputs,
                tgt_mask=causal_mask,
                memory_key_padding_mask=encoder_padding_mask
            )
            
            # Get logits for the last token
            # [batch_size, vocab_size]
            logits = self.output_layer(decoder_outputs[:, -1, :])
            
            # Sample next token (greedy)
            # [batch_size, 1]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Update finished sequences
            finished = finished | (next_token.squeeze(-1) == 3)  # <END> token id
            
            # Add to current tokens
            current_ids = torch.cat([current_ids, next_token], dim=1)
        
        # Convert to list
        result = []
        for ids in current_ids.tolist():
            # Remove <START> token and everything after <END> token
            if 3 in ids:  # <END> token id
                end_idx = ids.index(3)
                result.append(ids[1:end_idx+1])  # Include <END> token
            else:
                result.append(ids[1:])  # Remove <START> token
                
        return result
    
    def _beam_search(self, 
                    encoder_outputs: torch.Tensor, 
                    encoder_padding_mask: torch.Tensor, 
                    batch_size: int, 
                    max_length: int, 
                    beam_size: int) -> List[List[int]]:
        """
        Beam search for text generation
        
        Args:
            encoder_outputs: Encoder outputs [batch_size, seq_len, hidden_dim]
            encoder_padding_mask: Padding mask for encoder [batch_size, seq_len]
            batch_size: Batch size
            max_length: Maximum generation length
            beam_size: Beam size
            
        Returns:
            List of generated token ID sequences
        """
        device = encoder_outputs.device
        
        # Expand encoder outputs for beam search
        # [batch_size * beam_size, seq_len, hidden_dim]
        encoder_outputs = encoder_outputs.repeat_interleave(beam_size, dim=0)
        
        # Expand encoder padding mask
        # [batch_size * beam_size, seq_len]
        encoder_padding_mask = encoder_padding_mask.repeat_interleave(beam_size, dim=0)
        
        # Initialize beams for each sequence in batch
        all_beams = []
        
        for batch_idx in range(batch_size):
            # Start with a single sequence with <START> token
            # Each beam is a tuple of (token_ids, score)
            beams = [(torch.tensor([2], device=device), 0.0)]  # <START> token id
            
            # Keep track of finished beams
            finished_beams = []
            
            # Generate tokens one by one
            for i in range(max_length - 1):
                # Stop if we have enough finished beams
                if len(finished_beams) >= beam_size:
                    break
                    
                # Collect all candidates from all beams
                all_candidates = []
                
                # For each active beam
                for beam_idx, (token_ids, score) in enumerate(beams):
                    # Stop if this beam has an <END> token
                    if token_ids[-1] == 3:  # <END> token id
                        finished_beams.append((token_ids, score))
                        continue
                        
                    # Forward pass to get next token probabilities
                    # Embed current tokens
                    # [1, curr_len, embedding_dim]
                    tgt_embeddings = self.desc_embedding(token_ids.unsqueeze(0))
                    
                    # Add positional encodings
                    seq_len = tgt_embeddings.size(1)
                    tgt_embeddings = tgt_embeddings + self.pos_embedding[:, :seq_len, :]
                    
                    # Create causal mask for decoder
                    # [curr_len, curr_len]
                    causal_mask = torch.triu(
                        torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                        diagonal=1
                    )
                    
                    # Get encoder outputs for this beam
                    batch_encoder_outputs = encoder_outputs[batch_idx*beam_size + beam_idx].unsqueeze(0)
                    batch_encoder_padding_mask = encoder_padding_mask[batch_idx*beam_size + beam_idx].unsqueeze(0)
                    
                    # Pass through decoder
                    # [1, curr_len, embedding_dim]
                    decoder_outputs = self.decoder(
                        tgt_embeddings, 
                        batch_encoder_outputs,
                        tgt_mask=causal_mask,
                        memory_key_padding_mask=batch_encoder_padding_mask
                    )
                    
                    # Get logits for the last token
                    # [1, vocab_size]
                    logits = self.output_layer(decoder_outputs[:, -1, :])
                    
                    # Convert to log probabilities
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Get top-k next tokens
                    topk_log_probs, topk_tokens = log_probs.topk(beam_size)
                    
                    # Add candidates
                    for k in range(beam_size):
                        next_token = topk_tokens[0, k].item()
                        next_score = score + topk_log_probs[0, k].item()
                        next_ids = torch.cat([token_ids, torch.tensor([next_token], device=device)])
                        
                        all_candidates.append((next_ids, next_score))
                
                # If no candidates (all beams finished), break
                if not all_candidates:
                    break
                    
                # Sort candidates by score
                all_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # Get top-k candidates as new beams
                beams = all_candidates[:beam_size]
            
            # If not enough finished beams, add unfinished ones
            while len(finished_beams) < beam_size and beams:
                finished_beams.append(beams.pop(0))
                
            # Sort finished beams by score
            finished_beams.sort(key=lambda x: x[1], reverse=True)
            
            # Get best beam
            all_beams.append(finished_beams[0][0])
        
        # Convert to list
        result = []
        for ids in all_beams:
            # Remove <START> token and everything after <END> token
            ids = ids.tolist()
            if 3 in ids:  # <END> token id
                end_idx = ids.index(3)
                result.append(ids[1:end_idx+1])  # Include <END> token
            else:
                result.append(ids[1:])  # Remove <START> token
                
        return result
