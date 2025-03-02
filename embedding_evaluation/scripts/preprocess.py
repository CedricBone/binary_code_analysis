#!/usr/bin/env python
"""
Script for preprocessing binary files.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from tqdm import tqdm

# Add src to path for importing
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.data import BinaryPreprocessor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess binary files for instruction embedding evaluation"
    )
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="Directory containing binary files"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/processed",
        help="Directory to save processed data"
    )
    
    parser.add_argument(
        "--objdump_path", 
        type=str, 
        default="objdump",
        help="Path to objdump executable"
    )
    
    return parser.parse_args()

def setup_logging():
    """Set up logging."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("preprocess")

def main():
    """Main function for preprocessing binary files."""
    args = parse_args()
    logger = setup_logging()
    
    logger.info("Starting binary preprocessing")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = BinaryPreprocessor(objdump_path=args.objdump_path)
    
    # Process binary files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"binary_data_{timestamp}.json")
    
    try:
        logger.info(f"Processing binary files from {args.input_dir}")
        binary_files = []
        
        # Find all binary files
        for root, _, files in os.walk(args.input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip non-binary files
                if not os.path.isfile(file_path) or file.endswith(('.py', '.txt', '.md', '.json')):
                    continue
                
                binary_files.append(file_path)
        
        logger.info(f"Found {len(binary_files)} binary files")
        
        # Process each file
        all_sequences = []
        for file_path in tqdm(binary_files, desc="Processing files"):
            try:
                sequences = preprocessor.process_binary_file(file_path)
                all_sequences.extend(sequences)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Save processed data
        if all_sequences:
            logger.info(f"Saving {len(all_sequences)} processed sequences to {output_path}")
            preprocessor.save_processed_data(all_sequences, output_path)
        else:
            logger.warning("No sequences were processed")
        
        logger.info("Preprocessing complete")
    
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()