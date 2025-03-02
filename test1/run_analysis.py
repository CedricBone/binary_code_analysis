"""
Main script to demonstrate binary code analysis
- Similarity detection
- Description generation
"""
import os
import argparse
import torch
import logging
import json
from typing import Dict, List, Tuple, Optional, Any

from preprocessing import BinaryPreprocessor
from models import (
    SAFEEncoder, 
    GeminiEncoder, 
    HybridEncoder,
    SiameseSimilarityModel,
    DescriptionGenerator,
    TransformerEncoder
)
from utils import (
    load_checkpoint, 
    setup_logging, 
    build_idx_to_token, 
    idx_to_tokens, 
    filter_special_tokens
)

def parse_args():
    parser = argparse.ArgumentParser(description="Binary code analysis")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["similarity", "description", "both"],
                        help="Analysis mode")
    
    # Model arguments
    parser.add_argument("--similarity_model", type=str, default=None, 
                        help="Path to similarity model checkpoint")
    parser.add_argument("--description_model", type=str, default=None, 
                        help="Path to description model checkpoint")
    parser.add_argument("--vocab_dir", type=str, required=True, 
                        help="Directory containing vocabularies")
    
    # Data arguments
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing input files")
    parser.add_argument("--output_file", type=str, default="analysis_results.json", 
                        help="Output file path")
    parser.add_argument("--max_seq_length", type=int, default=200, 
                        help="Maximum sequence length")
    parser.add_argument("--max_desc_length", type=int, default=30, 
                        help="Maximum description length")
    parser.add_argument("--beam_size", type=int, default=3, 
                        help="Beam size for description generation")
    
    # Runtime arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device (cuda or cpu)")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose output")
    
    return parser.parse_args()

def load_similarity_model(model_path, preprocessor, device):
    """Load similarity model from checkpoint"""
    
    logger = logging.getLogger()
    
    # Get vocabulary sizes
    opcode_vocab_size = len(preprocessor.opcode_vocab)
    operand_vocab_size = len(preprocessor.operand_vocab)
    
    # Load checkpoint
    checkpoint = load_checkpoint(model_path, map_location=device)
    model_args = checkpoint.get("args", {})
    
    # Determine model type
    model_type = model_args.get("model_type", "safe")
    logger.info(f"Using similarity model type: {model_type}")
    
    # Get model hyperparameters
    embedding_dim = model_args.get("embedding_dim", 256)
    hidden_dim = model_args.get("hidden_dim", 512)
    num_layers = model_args.get("num_layers", 2)
    attention_heads = model_args.get("attention_heads", 8)
    dropout = model_args.get("dropout", 0.1)
    margin = model_args.get("margin", 0.5)
    
    # Initialize encoder based on type
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
            max_seq_length=preprocessor.max_seq_length
        )
    else:
        raise ValueError(f"Unsupported similarity model type: {model_type}")
    
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

def load_description_model(model_path, preprocessor, device):
    """Load description model from checkpoint"""
    
    logger = logging.getLogger()
    
    # Get vocabulary sizes
    opcode_vocab_size = len(preprocessor.opcode_vocab)
    operand_vocab_size = len(preprocessor.operand_vocab)
    desc_vocab_size = len(preprocessor.description_vocab)
    
    # Load checkpoint
    checkpoint = load_checkpoint(model_path, map_location=device)
    model_args = checkpoint.get("args", {})
    
    # Determine encoder type
    encoder_type = model_args.get("encoder_type", "transformer")
    logger.info(f"Using description model encoder type: {encoder_type}")
    
    # Get model hyperparameters
    embedding_dim = model_args.get("embedding_dim", 256)
    hidden_dim = model_args.get("hidden_dim", 512)
    encoder_layers = model_args.get("encoder_layers", 6)
    decoder_layers = model_args.get("decoder_layers", 6)
    num_heads = model_args.get("num_heads", 8)
    dropout = model_args.get("dropout", 0.1)
    
    # Initialize model
    model = DescriptionGenerator(
        opcode_vocab_size=opcode_vocab_size,
        operand_vocab_size=operand_vocab_size,
        desc_vocab_size=desc_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_seq_length=preprocessor.max_seq_length,
        max_desc_length=preprocessor.max_description_length,
        encoder_type=encoder_type
    )
    
    # Load model state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, encoder_type

def load_functions(input_dir):
    """Load all functions from input directory"""
    
    functions = {}
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".asm") or filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r") as f:
                functions[filename] = f.read().strip().split("\n")
    
    return functions

def process_function(func, preprocessor, device):
    """Process a function for input to the models"""
    
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
    
    return func_data, func_opcode_ids, func_operand_ids

def compute_similarity_matrix(functions, preprocessor, model, device):
    """Compute similarity matrix for all function pairs"""
    
    logger = logging.getLogger()
    
    # Initialize similarity matrix
    function_names = list(functions.keys())
    n_functions = len(function_names)
    similarity_matrix = np.zeros((n_functions, n_functions))
    
    # Process all functions
    processed_functions = {}
    for i, func_name in enumerate(function_names):
        func = functions[func_name]
        func_data, _, _ = process_function(func, preprocessor, device)
        processed_functions[func_name] = func_data
    
    # Compute similarity for all pairs
    logger.info("Computing similarity matrix...")
    for i in range(n_functions):
        func1_name = function_names[i]
        func1_data = processed_functions[func1_name]
        
        for j in range(i, n_functions):
            func2_name = function_names[j]
            func2_data = processed_functions[func2_name]
            
            # Skip identical functions
            if i == j:
                similarity_matrix[i, j] = 1.0
                continue
            
            # Compute similarity
            with torch.no_grad():
                similarity, _ = model(func1_data, func2_data)
                
                # Convert to float
                similarity = similarity.cpu().item()
                
                # Convert to probability (0 to 1)
                similarity = (similarity + 1) / 2
            
            # Store similarity
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    return similarity_matrix, function_names

def generate_descriptions(functions, preprocessor, model, device, beam_size):
    """Generate descriptions for all functions"""
    
    logger = logging.getLogger()
    idx_to_token_map = build_idx_to_token(preprocessor.description_vocab)
    
    descriptions = {}
    
    logger.info("Generating descriptions...")
    for func_name, func in functions.items():
        # Process function
        _, func_opcode_ids, func_operand_ids = process_function(func, preprocessor, device)
        
        # Generate description
        with torch.no_grad():
            generated_ids = model.generate(
                opcode_ids=func_opcode_ids,
                operand_ids=func_operand_ids,
                beam_size=beam_size
            )
        
        # Convert to tokens and filter special tokens
        generated_tokens = idx_to_tokens(generated_ids[0], idx_to_token_map)
        generated_tokens = filter_special_tokens(generated_tokens)
        
        # Join tokens to form description
        description = " ".join(generated_tokens)
        
        # Store description
        descriptions[func_name] = description
    
    return descriptions

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
        max_description_length=args.max_desc_length,
        normalize_registers=True,
        normalize_addresses=True
    )
    
    # Load vocabularies
    logger.info(f"Loading vocabularies from {args.vocab_dir}...")
    try:
        preprocessor.load_vocabularies(args.vocab_dir)
    except Exception as e:
        logger.error(f"Failed to load vocabularies: {e}")
        logger.info("Using default vocabularies...")
    
    # Load functions
    logger.info(f"Loading functions from {args.input_dir}...")
    functions = load_functions(args.input_dir)
    logger.info(f"Loaded {len(functions)} functions")
    
    if len(functions) == 0:
        logger.error(f"No functions found in {args.input_dir}")
        return
    
    results = {}
    
    # Similarity analysis
    if args.mode in ["similarity", "both"]:
        if args.similarity_model is None:
            logger.error("Similarity model path not specified")
            if args.mode == "similarity":
                return
        else:
            # Load similarity model
            logger.info(f"Loading similarity model from {args.similarity_model}...")
            similarity_model, model_type = load_similarity_model(
                args.similarity_model, 
                preprocessor, 
                device
            )
            
            # Set preprocessor to use CFG if needed
            preprocessor.use_cfg = model_type in ["gemini", "hybrid"]
            
            # Compute similarity matrix
            similarity_matrix, function_names = compute_similarity_matrix(
                functions, 
                preprocessor, 
                similarity_model, 
                device
            )
            
            # Store results
            results["similarity"] = {
                "matrix": similarity_matrix.tolist(),
                "function_names": function_names
            }
            
            # Find similar function pairs (above threshold)
            similar_pairs = []
            for i in range(len(function_names)):
                for j in range(i+1, len(function_names)):
                    if similarity_matrix[i, j] > 0.8:  # Arbitrary threshold
                        similar_pairs.append({
                            "func1": function_names[i],
                            "func2": function_names[j],
                            "similarity": similarity_matrix[i, j]
                        })
            
            results["similarity"]["similar_pairs"] = similar_pairs
            
            logger.info(f"Found {len(similar_pairs)} similar function pairs")
    
    # Description generation
    if args.mode in ["description", "both"]:
        if args.description_model is None:
            logger.error("Description model path not specified")
            if args.mode == "description":
                return
        else:
            # Load description model
            logger.info(f"Loading description model from {args.description_model}...")
            description_model, encoder_type = load_description_model(
                args.description_model, 
                preprocessor, 
                device
            )
            
            # Set preprocessor to use CFG if needed
            preprocessor.use_cfg = encoder_type in ["gemini", "hybrid"]
            
            # Generate descriptions
            descriptions = generate_descriptions(
                functions, 
                preprocessor, 
                description_model, 
                device, 
                args.beam_size
            )
            
            # Store results
            results["descriptions"] = descriptions
            
            logger.info(f"Generated descriptions for {len(descriptions)} functions")
    
    # Save results
    logger.info(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("Analysis completed!")
    print(f"Results saved to {args.output_file}")
    
    if "similarity" in results:
        print(f"Similar function pairs found: {len(results['similarity']['similar_pairs'])}")
        for pair in results['similarity']['similar_pairs'][:3]:  # Show top 3
            print(f"  {pair['func1']} and {pair['func2']}: {pair['similarity']:.4f}")
        if len(results['similarity']['similar_pairs']) > 3:
            print(f"  ... and {len(results['similarity']['similar_pairs']) - 3} more")
    
    if "descriptions" in results:
        print(f"Generated descriptions for {len(results['descriptions'])} functions")
        for i, (func_name, desc) in enumerate(list(results['descriptions'].items())[:3]):  # Show top 3
            print(f"  {func_name}: \"{desc}\"")
        if len(results['descriptions']) > 3:
            print(f"  ... and {len(results['descriptions']) - 3} more")

if __name__ == "__main__":
    main()
