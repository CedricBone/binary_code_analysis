"""
Inference script for binary code description generation
"""
import os
import argparse
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

from preprocessing import BinaryPreprocessor
from models import DescriptionGenerator
from utils import (
    load_checkpoint, 
    setup_logging, 
    build_idx_to_token, 
    idx_to_tokens, 
    filter_special_tokens
)

def parse_args():
    parser = argparse.ArgumentParser(description="Binary code description generation inference")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--encoder_type", type=str, default=None, 
                        choices=["transformer", "safe"],
                        help="Encoder type (if not specified in checkpoint)")
    parser.add_argument("--vocab_dir", type=str, required=True, help="Directory containing vocabularies")
    
    # Data arguments
    parser.add_argument("--func_path", type=str, required=True, help="Path to function file")
    parser.add_argument("--max_seq_length", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--max_desc_length", type=int, default=30, help="Maximum description length")
    parser.add_argument("--beam_size", type=int, default=3, help="Beam size for generation")
    
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
    desc_vocab_size = len(preprocessor.description_vocab)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.model_path, map_location=args.device)
    model_args = checkpoint.get("args", {})
    
    # Determine encoder type
    encoder_type = args.encoder_type or model_args.get("encoder_type", "transformer")
    logger.info(f"Using encoder type: {encoder_type}")
    
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
        max_seq_length=args.max_seq_length,
        max_desc_length=args.max_desc_length,
        encoder_type=encoder_type
    )
    
    # Load model state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, encoder_type

def load_function(func_path):
    """Load function from file"""
    
    # Read function file
    with open(func_path, "r") as f:
        func = f.read().strip().split("\n")
    
    return func

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
    
    return func_opcode_ids, func_operand_ids, func_tokens

def generate_description(model, func_opcode_ids, func_operand_ids, beam_size):
    """Generate description for a function"""
    
    with torch.no_grad():
        # Generate description
        generated_ids = model.generate(
            opcode_ids=func_opcode_ids,
            operand_ids=func_operand_ids,
            beam_size=beam_size
        )
    
    return generated_ids[0]  # Return first (only) batch item

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
    preprocessor.load_vocabularies(args.vocab_dir)
    
    # Build index to token mapping for description vocabulary
    idx_to_token_map = build_idx_to_token(preprocessor.description_vocab)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    model, encoder_type = load_model(args, preprocessor)
    
    # Set preprocessor to use CFG if needed
    preprocessor.use_cfg = encoder_type in ["gemini", "hybrid"]
    
    # Load function
    logger.info(f"Loading function from {args.func_path}...")
    func = load_function(args.func_path)
    
    # Process function
    logger.info("Processing function...")
    func_opcode_ids, func_operand_ids, func_tokens = process_function(func, preprocessor, device)
    
    # Generate description
    logger.info("Generating description...")
    generated_ids = generate_description(model, func_opcode_ids, func_operand_ids, args.beam_size)
    
    # Convert to tokens and filter special tokens
    generated_tokens = idx_to_tokens(generated_ids, idx_to_token_map)
    generated_tokens = filter_special_tokens(generated_tokens)
    
    # Join tokens to form description
    description = " ".join(generated_tokens)
    
    # Print result
    print("Generated description:")
    print(description)
    
    # Print function snippets (first few lines for context)
    print("\nFunction snippet:")
    for i, line in enumerate(func[:10]):
        print(line)
    if len(func) > 10:
        print(f"... ({len(func) - 10} more lines)")

if __name__ == "__main__":
    main()
