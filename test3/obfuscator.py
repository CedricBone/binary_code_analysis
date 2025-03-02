#!/usr/bin/env python3
"""
Obfuscator for binary executables using various techniques.
"""
import os
import subprocess
import argparse
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinaryObfuscator:
    """Applies different obfuscation techniques to compiled binaries."""
    
    def __init__(self, tigress_path: str = "/opt/tigress/tigress"):
        """
        Initialize the obfuscator.
        
        Args:
            tigress_path: Path to Tigress C obfuscator
        """
        self.tigress_path = tigress_path
        self.verify_dependencies()
    
    def verify_dependencies(self):
        """Verify that all required tools are installed."""
        # Check if Tigress is available
        if not os.path.exists(self.tigress_path):
            logger.warning(f"Tigress not found at {self.tigress_path}. Source code obfuscation will be limited.")
        
        # Check if OLLVM is available
        try:
            subprocess.run(["which", "clang-11"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.warning("OLLVM/clang-11 not found. Binary obfuscation capabilities will be limited.")
    
    def obfuscate_source(self, source_file: str, output_dir: str, techniques: List[str]) -> Dict[str, str]:
        """
        Apply source-level obfuscation techniques and compile the results.
        
        Args:
            source_file: Path to C source file
            output_dir: Directory to store obfuscated files
            techniques: List of obfuscation techniques to apply
            
        Returns:
            Dictionary mapping technique names to output binary paths
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # Base filename without extension
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        
        # Compile original without obfuscation as baseline
        baseline_bin = os.path.join(output_dir, f"{base_name}_original")
        subprocess.run(
            ["gcc", source_file, "-o", baseline_bin, "-O0"],
            check=True
        )
        results["original"] = baseline_bin
        
        for technique in techniques:
            output_name = f"{base_name}_{technique}"
            output_c = os.path.join(output_dir, f"{output_name}.c")
            output_bin = os.path.join(output_dir, output_name)
            
            if technique == "flatten":
                # Apply control flow flattening using Tigress
                cmd = [
                    self.tigress_path,
                    "--Verbosity=1",
                    f"--out={output_c}",
                    f"--Transform=Flatten",
                    f"--Functions=*",
                    f"--FlattenDispatch=switch",
                    source_file
                ]
                subprocess.run(cmd, check=True)
                
            elif technique == "virtualize":
                # Apply virtualization obfuscation using Tigress
                cmd = [
                    self.tigress_path,
                    "--Verbosity=1",
                    f"--out={output_c}",
                    f"--Transform=Virtualize",
                    f"--Functions=*",
                    f"--VirtualizeDispatch=switch",
                    source_file
                ]
                subprocess.run(cmd, check=True)
                
            elif technique == "encodeliterals":
                # Apply literal encoding
                cmd = [
                    self.tigress_path,
                    "--Verbosity=1",
                    f"--out={output_c}",
                    f"--Transform=EncodeLiterals",
                    f"--Functions=*",
                    source_file
                ]
                subprocess.run(cmd, check=True)
                
            elif technique == "ollvm_bcf":
                # OLLVM control flow obfuscation
                output_bin = os.path.join(output_dir, output_name)
                subprocess.run([
                    "clang-11", "-mllvm", "-bcf", "-mllvm", "-bcf_prob=80", 
                    "-mllvm", "-bcf_loop=1", source_file, "-o", output_bin
                ], check=True)
                results[technique] = output_bin
                continue  # Skip the standard compilation below
                
            elif technique == "ollvm_sub":
                # OLLVM instruction substitution
                output_bin = os.path.join(output_dir, output_name)
                subprocess.run([
                    "clang-11", "-mllvm", "-sub", "-mllvm", "-sub_loop=1", 
                    source_file, "-o", output_bin
                ], check=True)
                results[technique] = output_bin
                continue  # Skip the standard compilation below
                
            elif technique == "ollvm_fla":
                # OLLVM control flow flattening
                output_bin = os.path.join(output_dir, output_name)
                subprocess.run([
                    "clang-11", "-mllvm", "-fla", 
                    source_file, "-o", output_bin
                ], check=True)
                results[technique] = output_bin
                continue  # Skip the standard compilation below
            
            # Compile the obfuscated source
            subprocess.run(["gcc", output_c, "-o", output_bin, "-O0"], check=True)
            results[technique] = output_bin
            
        return results

    def obfuscate_binary(self, binary_path: str, output_dir: str, techniques: List[str]) -> Dict[str, str]:
        """
        Apply binary-level obfuscation techniques.
        
        Args:
            binary_path: Path to original binary
            output_dir: Directory to store obfuscated files
            techniques: List of obfuscation techniques to apply
            
        Returns:
            Dictionary mapping technique names to output binary paths
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {"original": binary_path}
        
        for technique in techniques:
            output_name = f"{os.path.basename(binary_path)}_{technique}"
            output_path = os.path.join(output_dir, output_name)
            
            if technique == "strip":
                # Remove symbols from binary
                subprocess.run(["strip", "-o", output_path, binary_path], check=True)
                results[technique] = output_path
                
            # Add other binary transformation techniques here
                
        return results

def main():
    parser = argparse.ArgumentParser(description="Obfuscate binaries using various techniques")
    parser.add_argument("--input", "-i", required=True, help="Input C source file or binary")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for obfuscated files")
    parser.add_argument("--techniques", "-t", nargs="+", default=["flatten", "virtualize", "encodeliterals"],
                        help="Obfuscation techniques to apply")
    parser.add_argument("--binary", "-b", action="store_true", help="Input is a binary file, not source code")
    parser.add_argument("--tigress-path", default="/opt/tigress/tigress", help="Path to Tigress C obfuscator")
    
    args = parser.parse_args()
    
    obfuscator = BinaryObfuscator(tigress_path=args.tigress_path)
    
    if args.binary:
        results = obfuscator.obfuscate_binary(args.input, args.output_dir, args.techniques)
    else:
        results = obfuscator.obfuscate_source(args.input, args.output_dir, args.techniques)
    
    logger.info("Obfuscated files generated:")
    for technique, path in results.items():
        logger.info(f"  {technique}: {path}")

if __name__ == "__main__":
    main()