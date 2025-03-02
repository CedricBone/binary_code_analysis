#!/usr/bin/env python
"""
Script for downloading pre-trained models.
"""

import os
import sys
import argparse
import logging
import requests
from tqdm import tqdm

# Add src to path for importing
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download pre-trained models for instruction embedding evaluation"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="models",
        help="Directory to save models"
    )
    
    parser.add_argument(
        "--models", 
        type=str, 
        default="word2vec,palmtree",
        help="Comma-separated list of models to download"
    )
    
    return parser.parse_args()

def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("download_models")

def download_file(url, output_path, logger):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download
        output_path: Path to save the file
        logger: Logger instance
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download with progress bar
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def main():
    """Main function for downloading models."""
    args = parse_args()
    logger = setup_logging()
    
    logger.info("Starting model download")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Model URLs (placeholder - these would be real URLs in a production system)
    model_urls = {
        "word2vec": {
            "url": "https://example.com/models/word2vec_x86_64.pkl",
            "path": os.path.join(args.output_dir, "word2vec", "word2vec_x86_64.pkl")
        },
        "palmtree": {
            "url": "https://example.com/models/palmtree_x86_64.pkl",
            "path": os.path.join(args.output_dir, "palmtree", "palmtree_x86_64.pkl")
        }
    }
    
    # Download requested models
    requested_models = args.models.split(",")
    
    for model in requested_models:
        if model in model_urls:
            logger.info(f"Downloading {model} model")
            
            # Note: Since these are placeholder URLs that won't work,
            # we'll just create empty model files for demonstration
            os.makedirs(os.path.dirname(model_urls[model]["path"]), exist_ok=True)
            
            # In a real scenario, you would use:
            # download_file(model_urls[model]["url"], model_urls[model]["path"], logger)
            
            # For demonstration, create an empty file
            with open(model_urls[model]["path"], 'wb') as f:
                f.write(b'PLACEHOLDER MODEL FILE')
            
            logger.info(f"Created placeholder for {model} model at {model_urls[model]['path']}")
        else:
            logger.warning(f"Unknown model: {model}")
    
    logger.info("Model download complete")
    print("\nNote: This script creates placeholder model files for demonstration.")
    print("In a real scenario, you would replace the URLs with actual model URLs.")

if __name__ == "__main__":
    main()