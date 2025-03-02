#!/usr/bin/env python3
"""
Extract features from binary executables for similarity analysis.
"""
import os
import subprocess
import json
import argparse
import logging
from typing import Dict, List, Any
import r2pipe  # Radare2 Python interface

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinaryFeatureExtractor:
    """Extracts features from binary executables for similarity detection models."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.verify_dependencies()
    
    def verify_dependencies(self):
        """Verify that all required tools are installed."""
        try:
            subprocess.run(["radare2", "-v"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.error("Radare2 not found. Please install radare2 to use this feature extractor.")
            exit(1)
    
    def extract_features(self, binary_path: str) -> Dict[str, Any]:
        """
        Extract various features from the binary executable.
        
        Args:
            binary_path: Path to the binary executable
            
        Returns:
            Dictionary of extracted features
        """
        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Binary file not found: {binary_path}")
        
        features = {
            "binary_path": binary_path,
            "functions": self._extract_function_features(binary_path),
            "binary_metadata": self._extract_binary_metadata(binary_path)
        }
        
        return features
    
    def _extract_function_features(self, binary_path: str) -> List[Dict[str, Any]]:
        """
        Extract features for each function in the binary.
        
        Args:
            binary_path: Path to the binary executable
            
        Returns:
            List of function features
        """
        # Open binary in radare2
        r2 = r2pipe.open(binary_path)
        r2.cmd('aaa')  # Analyze all
        
        # Get list of functions
        functions_json = r2.cmd('aflj')
        if not functions_json:
            logger.warning(f"No functions detected in {binary_path}")
            return []
            
        functions = json.loads(functions_json)
        if not functions:
            return []
        
        function_features = []
        
        for func in functions:
            # Skip library functions
            if "is_lib" in func and func["is_lib"]:
                continue
                
            # Basic function info
            func_info = {
                "name": func.get("name", "unknown"),
                "offset": func.get("offset", 0),
                "size": func.get("size", 0),
                "basic_blocks": [],
                "instructions": [],
                "callgraph": [],
                "strings": [],
                "constants": []
            }
            
            # Get basic blocks
            r2.cmd(f's {func["offset"]}')  # Seek to function
            bbs_json = r2.cmd('afbj')
            if bbs_json:
                bbs = json.loads(bbs_json)
                func_info["basic_blocks"] = [{
                    "addr": bb.get("addr", 0),
                    "size": bb.get("size", 0),
                    "jump": bb.get("jump", 0),
                    "fail": bb.get("fail", 0),
                    "ninstr": bb.get("ninstr", 0)
                } for bb in bbs]
            
            # Get instructions
            r2.cmd(f's {func["offset"]}')
            pdj = r2.cmd(f'pdfj')
            if pdj:
                pd = json.loads(pdj)
                if "ops" in pd:
                    func_info["instructions"] = [{
                        "offset": op.get("offset", 0),
                        "type": op.get("type", ""),
                        "disasm": op.get("disasm", ""),
                        "opcode": op.get("opcode", ""),
                        "bytes": op.get("bytes", "")
                    } for op in pd["ops"]]
            
            # Get function calls (callgraph)
            calls_json = r2.cmd(f'afxj')
            if calls_json:
                calls = json.loads(calls_json)
                if calls:
                    func_info["callgraph"] = [{
                        "from": call.get("from", 0),
                        "to": call.get("to", 0),
                        "type": call.get("type", "")
                    } for call in calls]
            
            function_features.append(func_info)
        
        r2.quit()
        return function_features
    
    def _extract_binary_metadata(self, binary_path: str) -> Dict[str, Any]:
        """
        Extract metadata about the binary executable.
        
        Args:
            binary_path: Path to the binary executable
            
        Returns:
            Dictionary of binary metadata
        """
        r2 = r2pipe.open(binary_path)
        
        # Get file info
        info_json = r2.cmd('ij')
        info = json.loads(info_json)
        
        # Get sections
        sections_json = r2.cmd('iSj')
        sections = json.loads(sections_json) if sections_json else []
        
        # Get imports
        imports_json = r2.cmd('iij')
        imports = json.loads(imports_json) if imports_json else []
        
        # Get exports
        exports_json = r2.cmd('iEj')
        exports = json.loads(exports_json) if exports_json else []
        
        metadata = {
            "arch": info.get("bin", {}).get("arch", "unknown"),
            "bits": info.get("bin", {}).get("bits", 0),
            "os": info.get("bin", {}).get("os", "unknown"),
            "size": info.get("core", {}).get("size", 0),
            "format": info.get("core", {}).get("format", "unknown"),
            "sections": [{
                "name": section.get("name", "unknown"),
                "size": section.get("size", 0),
                "vaddr": section.get("vaddr", 0),
                "perm": section.get("perm", "")
            } for section in sections],
            "imports": [{
                "name": imp.get("name", "unknown"),
                "plt": imp.get("plt", 0),
                "ordinal": imp.get("ordinal", 0)
            } for imp in imports],
            "exports": [{
                "name": exp.get("name", "unknown"),
                "size": exp.get("size", 0),
                "vaddr": exp.get("vaddr", 0)
            } for exp in exports]
        }
        
        r2.quit()
        return metadata
    
    def save_features(self, features: Dict[str, Any], output_path: str):
        """
        Save extracted features to a JSON file.
        
        Args:
            features: Dictionary of extracted features
            output_path: Path to save the features JSON file
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(features, f, indent=2)
        
        logger.info(f"Features saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract features from binary executables")
    parser.add_argument("--input", "-i", required=True, help="Input binary file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file for extracted features")
    
    args = parser.parse_args()
    
    extractor = BinaryFeatureExtractor()
    features = extractor.extract_features(args.input)
    extractor.save_features(features, args.output)

if __name__ == "__main__":
    main()