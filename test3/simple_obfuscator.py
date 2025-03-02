#!/usr/bin/env python3
"""
Simple obfuscator that uses different compilation options to create binary variants.
"""
import os
import subprocess
import argparse
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleObfuscator:
    """Creates binary variants using different compilation settings."""
    
    def __init__(self):
        """Initialize the obfuscator."""
        self.verify_dependencies()
    
    def verify_dependencies(self):
        """Verify that all required tools are installed."""
        try:
            subprocess.run(["gcc", "--version"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.error("GCC not found. Please install GCC to use this tool.")
            exit(1)
    
    def generate_variants(self, source_file: str, output_dir: str) -> Dict[str, str]:
        """
        Generate binary variants using different compilation options.
        
        Args:
            source_file: Path to C source file
            output_dir: Directory to store generated binaries
            
        Returns:
            Dictionary mapping variant names to output binary paths
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # Base filename without extension
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        
        # Compile original with O0 as baseline
        original_bin = os.path.join(output_dir, f"{base_name}_original")
        subprocess.run(
            ["gcc", source_file, "-o", original_bin, "-O0"],
            check=True
        )
        results["original"] = original_bin
        
        # Different optimization levels
        for opt_level in ["O1", "O2", "O3", "Os"]:
            output_bin = os.path.join(output_dir, f"{base_name}_{opt_level}")
            subprocess.run(
                ["gcc", source_file, "-o", output_bin, f"-{opt_level}"],
                check=True
            )
            results[opt_level] = output_bin
        
        # With debug information
        debug_bin = os.path.join(output_dir, f"{base_name}_debug")
        subprocess.run(
            ["gcc", source_file, "-o", debug_bin, "-O0", "-g"],
            check=True
        )
        results["debug"] = debug_bin
        
        # With position-independent code
        pic_bin = os.path.join(output_dir, f"{base_name}_PIC")
        subprocess.run(
            ["gcc", source_file, "-o", pic_bin, "-O0", "-fPIC"],
            check=True
        )
        results["PIC"] = pic_bin
        
        # With stack protection
        stack_bin = os.path.join(output_dir, f"{base_name}_stack")
        subprocess.run(
            ["gcc", source_file, "-o", stack_bin, "-O0", "-fstack-protector-all"],
            check=True
        )
        results["stack"] = stack_bin
        
        # Different machine architecture (32-bit if possible)
        try:
            m32_bin = os.path.join(output_dir, f"{base_name}_m32")
            subprocess.run(
                ["gcc", source_file, "-o", m32_bin, "-O0", "-m32"],
                check=True
            )
            results["m32"] = m32_bin
        except subprocess.CalledProcessError:
            logger.warning("Couldn't compile 32-bit binary, skipping this variant")
        
        # With symbol stripping (reduces debug information)
        strip_bin = os.path.join(output_dir, f"{base_name}_stripped")
        subprocess.run(
            ["gcc", source_file, "-o", strip_bin, "-O0"],
            check=True
        )
        subprocess.run(
            ["strip", strip_bin],
            check=True
        )
        results["stripped"] = strip_bin
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Generate binary variants using different compilation options")
    parser.add_argument("--input", "-i", required=True, help="Input C source file")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for binaries")
    
    args = parser.parse_args()
    
    obfuscator = SimpleObfuscator()
    results = obfuscator.generate_variants(args.input, args.output_dir)
    
    logger.info("Generated binary variants:")
    for variant, path in results.items():
        logger.info(f"  {variant}: {path}")

if __name__ == "__main__":
    main()