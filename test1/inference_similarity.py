"""
Inference script for binary similarity detection
"""
import os
import argparse
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

from preprocessing import BinaryPreprocessor
from models import (
    SAFEEncoder, 
    GeminiEncoder, 
    HybridEncoder,
    SiameseSimilarityModel,
    TransformerEncoder
)
from utils import load_checkpoint, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Binary similarity inference")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default=None, 
                        choices=["safe", "gemini", "hybrid", "transformer"],
                        help="Model type (if not specified in checkpoint)")
    parser.add_argument("--vocab_dir", type=str, required=True, help="Directory containing vocabularies")
    
    # Data arguments
    parser.add_argument("--func1_path", type=str, required=True, help="Path to first function file")
    parser.add_argument("--func2_path", type=str, required=True, help="Path to second function file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_seq_length", type=int, default=200, help="Maximum sequence length")
    
    # Runtime arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device (cuda or cpu)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    return parser.parse_args()

def load_model(args, preprocessor):
    """Load model from checkpoint"""
    
    device = torch.device(args.device)
    logger = logging.getLogger()
    
    # Get vocabulary sizes
    opcode_vocab_size = len(preprocessor.opcode_vocab)
    operand_vocab_size = len(preprocessor.operand_vocab)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.model_path, map_location=args.device)
    model_args = checkpoint.get("args", {})
    
    # Determine model type
    model_type = args.model_type or model_args.get("model_type", "safe")
    logger.info(f"Using model type: {model_type}")
    
    # Get model hyperparameters
    embedding_dim = model_args.get("embedding_dim", 256)
    hidden_dim = model_args.get("hidden_dim", 512)
    num_layers = model_args.get("num_layers", 2)
    attention_heads = model_args.get("attention_heads", 8)
    dropout = model_args.get("dropout", 0.1)
    margin = model_args.get("margin", 0.5)
    
    # Initialize model based on type
    if model_type == "safe":
        encoder = SAFEEncoder(
            opcode_vocab_size=opcode_vocab_size,
            operand_vocab_size=operand_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            attention_heads=attention_heads,
            dropout=dropout
        )
    elif model_type == "gemini":
        encoder = GeminiEncoder(
            node_feature_dim=5,  # Number of features per node
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type == "hybrid":
        encoder = HybridEncoder(
            opcode_vocab_size=opcode_vocab_size,
            operand_vocab_size=operand_vocab_size,
            embedding_dim=embedding_dim,
            node_feature_dim=5,  # Number of features per node
            hidden_dim=hidden_dim,
            num_lstm_layers=num_layers,
            num_gnn_layers=num_layers,
            attention_heads=attention_heads,
            dropout=dropout
        )
    elif model_type == "transformer":
        encoder = TransformerEncoder(
            opcode_vocab_size=opcode_vocab_size,
            operand_vocab_size=operand_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=attention_heads,
            dropout=dropout,
            max_seq_length=args.max_seq_length
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model = SiameseSimilarityModel(
        encoder=encoder,
        hidden_dim=hidden_dim,
        margin=margin
    )
    
    # Load model state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, model_type

def load_functions(func1_path, func2_path):
    """Load functions from files"""
    
    # Read function files
    with open(func1_path, "r") as f1, open(func2_path, "r") as f2:
        func1 = f1.read().strip().split("\n")
        func2 = f2.read().strip().split("\n")
    
    return func1, func2

def process_function(func, preprocessor, device):
    """Process a function for input to the model"""
    
    # Tokenize function
    func_tokens = preprocessor.tokenize_function(func)
    
    # Convert to tensors
    func_opcode_ids = torch.tensor([func_tokens['opcode_ids']], dtype=torch.long).to(device)
    
    # Convert operand lists to padded tensor
    max_operands = max(len(ops) for ops in func_tokens['operand_ids_list'])
    func_operand_ids = torch.zeros((1, len(func_tokens['operand_ids_list']), max_operands), dtype=torch.long)
    for i, ops in enumerate(func_tokens['operand_ids_list']):
        if ops:
            func_operand_ids[0, i, :len(ops)] = torch.tensor(ops, dtype=torch.long)
    
    func_operand_ids = func_operand_ids.to(device)
    
    # Prepare function data
    func_data = {
        'func_opcode_ids': func_opcode_ids,
        'func_operand_ids': func_operand_ids
    }
    
    # Add graph data if available
    if 'cfg' in func_tokens:
        # Process CFG (simplified for inference)
        func_cfg = func_tokens['cfg']
        func_nodes = list(func_cfg.nodes)
        func_node_features = np.zeros((1, len(func_nodes), 5), dtype=np.float32)
        
        for i, node in enumerate(func_nodes):
            if 'features' in func_cfg.nodes[node]:
                func_node_features[0, i] = func_cfg.nodes[node]['features']
        
        import networkx as nx
        func_adj_matrix = nx.to_numpy_array(func_cfg, nodelist=func_nodes)
        func_adj_matrix = func_adj_matrix.reshape(1, func_adj_matrix.shape[0], func_adj_matrix.shape[1])
        
        # Convert to tensors
        func_node_features = torch.tensor(func_node_features, dtype=torch.float).to(device)
        func_adj_matrix = torch.tensor(func_adj_matrix, dtype=torch.float).to(device)
        
        func_data['func_node_features'] = func_node_features
        func_data['func_adj_matrix'] = func_adj_matrix
    
    return func_data

def compute_similarity(model, func1_data, func2_data):
    """Compute similarity between two functions"""
    
    with torch.no_grad():
        similarity, embeddings = model(func1_data, func2_data)
        
        # Convert to float
        similarity = similarity.cpu().item()
        
        # Convert to probability (0 to 1)
        similarity = (similarity + 1) / 2
    
    return similarity, embeddings

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(level=logging.INFO if args.verbose else logging.WARNING)
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Initialize preprocessor
    logger.info("Initializing preprocessor...")
    preprocessor = BinaryPreprocessor(
        max_seq_length=args.max_seq_length,
        normalize_registers=True,
        normalize_addresses=True
    )
    
    # Load vocabularies
    logger.info(f"Loading vocabularies from {args.vocab_dir}...")
    preprocessor.load_vocabularies(args.vocab_dir)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    model, model_type = load_model(args, preprocessor)
    
    # Set preprocessor to use CFG if needed
    preprocessor.use_cfg = model_type in ["gemini", "hybrid"]
    
    # Load functions
    logger.info(f"Loading functions from {args.func1_path} and {args.func2_path}...")
    func1, func2 = load_functions(args.func1_path, args.func2_path)
    
    # Process functions
    logger.info("Processing functions...")
    func1_data = process_function(func1, preprocessor, device)
    func2_data = process_function(func2, preprocessor, device)
    
    # Compute similarity
    logger.info("Computing similarity...")
    similarity, _ = compute_similarity(model, func1_data, func2_data)
    
    # Print result
    print(f"Similarity: {similarity:.4f}")
    
    # Interpret result
    if similarity > 0.8:
        print("Interpretation: Functions are very similar")
    elif similarity > 0.5:
        print("Interpretation: Functions have moderate similarity")
    else:
        print("Interpretation: Functions are not similar")

if __name__ == "__main__":
    main()
