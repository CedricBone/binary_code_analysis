"""
Cross-architecture binary analysis utilities.

This module provides tools for processing and comparing binaries
across different CPU architectures.
"""

import os
import re
import json
import logging
import subprocess
import tempfile
from collections import defaultdict
from tqdm import tqdm

from .preprocessing import BinaryPreprocessor

logger = logging.getLogger(__name__)

class CrossArchitectureProcessor:
    """Processor for cross-architecture binary analysis."""
    
    # Common architecture-specific objdump paths
    OBJDUMP_PATHS = {
        'x86_64': 'objdump',
        'arm': 'arm-linux-gnueabi-objdump',
        'arm64': 'aarch64-linux-gnu-objdump',
        'mips': 'mips-linux-gnu-objdump',
        'riscv': 'riscv64-linux-gnu-objdump'
    }
    
    # Common architecture-specific compiler paths
    COMPILER_PATHS = {
        'x86_64': {'gcc': 'gcc', 'clang': 'clang'},
        'arm': {'gcc': 'arm-linux-gnueabi-gcc', 'clang': 'clang --target=arm-linux-gnueabi'},
        'arm64': {'gcc': 'aarch64-linux-gnu-gcc', 'clang': 'clang --target=aarch64-linux-gnu'},
        'mips': {'gcc': 'mips-linux-gnu-gcc', 'clang': 'clang --target=mips-linux-gnu'},
        'riscv': {'gcc': 'riscv64-linux-gnu-gcc', 'clang': 'clang --target=riscv64-linux-gnu'}
    }
    
    # Semantic instruction mappings across architectures
    SEMANTIC_MAPPINGS = {
        # Assignment/Move instructions
        ('x86_64', 'mov', 'arm64', 'mov'): True,
        ('x86_64', 'mov', 'arm', 'mov'): True,
        ('x86_64', 'mov', 'mips', 'move'): True,
        
        # Addition instructions
        ('x86_64', 'add', 'arm64', 'add'): True,
        ('x86_64', 'add', 'arm', 'add'): True,
        ('x86_64', 'add', 'mips', 'addu'): True,
        
        # Subtraction instructions
        ('x86_64', 'sub', 'arm64', 'sub'): True,
        ('x86_64', 'sub', 'arm', 'sub'): True,
        ('x86_64', 'sub', 'mips', 'subu'): True,
        
        # Jump instructions
        ('x86_64', 'jmp', 'arm64', 'b'): True,
        ('x86_64', 'jmp', 'arm', 'b'): True,
        ('x86_64', 'jmp', 'mips', 'j'): True,
        
        # Branch if equal
        ('x86_64', 'je', 'arm64', 'b.eq'): True,
        ('x86_64', 'je', 'arm', 'beq'): True,
        ('x86_64', 'je', 'mips', 'beq'): True,
        
        # Branch if not equal
        ('x86_64', 'jne', 'arm64', 'b.ne'): True,
        ('x86_64', 'jne', 'arm', 'bne'): True,
        ('x86_64', 'jne', 'mips', 'bne'): True,
        
        # Function calls
        ('x86_64', 'call', 'arm64', 'bl'): True,
        ('x86_64', 'call', 'arm', 'bl'): True,
        ('x86_64', 'call', 'mips', 'jal'): True,
        
        # Return
        ('x86_64', 'ret', 'arm64', 'ret'): True,
        ('x86_64', 'ret', 'arm', 'bx lr'): True,
        ('x86_64', 'ret', 'mips', 'jr ra'): True,
        
        # Push (approximate matches)
        ('x86_64', 'push', 'arm64', 'str'): True,
        ('x86_64', 'push', 'arm', 'push'): True,
        ('x86_64', 'push', 'mips', 'sw'): True,
        
        # Pop (approximate matches)
        ('x86_64', 'pop', 'arm64', 'ldr'): True,
        ('x86_64', 'pop', 'arm', 'pop'): True,
        ('x86_64', 'pop', 'mips', 'lw'): True,
    }
    
    def __init__(self, output_dir="data/cross_arch", cache_dir="data/cache"):
        """
        Initialize the cross-architecture processor.
        
        Args:
            output_dir: Directory to store processed data
            cache_dir: Directory to cache compiled binaries
        """
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "binaries"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "functions"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "mappings"), exist_ok=True)
        
        # Initialize preprocessors for each architecture
        self.preprocessors = {}
        for arch, objdump_path in self.OBJDUMP_PATHS.items():
            if self._check_tool_exists(objdump_path):
                self.preprocessors[arch] = BinaryPreprocessor(objdump_path=objdump_path)
            else:
                logger.warning(f"Objdump for {arch} not found: {objdump_path}")
    
    def _check_tool_exists(self, tool_path):
        """
        Check if a tool exists in PATH.
        
        Args:
            tool_path: Path or name of the tool
            
        Returns:
            bool: True if tool exists, False otherwise
        """
        # Extract command name (handle cases like "clang --target=arm")
        command = tool_path.split()[0]
        
        try:
            subprocess.run(["which", command], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def get_available_architectures(self):
        """
        Get list of available architectures.
        
        Returns:
            list: List of available architecture names
        """
        return list(self.preprocessors.keys())
    
    def compile_across_architectures(self, source_path, architectures=None, compiler="gcc", opt_level="-O0"):
        """
        Compile a source file for multiple architectures.
        
        Args:
            source_path: Path to the source file
            architectures: List of target architectures (or None for all available)
            compiler: Compiler to use (gcc or clang)
            opt_level: Optimization level
            
        Returns:
            dict: Dictionary mapping architecture to compiled binary path
        """
        if architectures is None:
            architectures = self.get_available_architectures()
        
        if compiler not in ["gcc", "clang"]:
            raise ValueError(f"Unsupported compiler: {compiler}")
        
        # Get base file name without extension
        base_name = os.path.basename(source_path).split(".")[0]
        
        # Compile for each architecture
        binary_paths = {}
        
        for arch in architectures:
            if arch not in self.COMPILER_PATHS:
                logger.warning(f"No compiler configuration for {arch}")
                continue
            
            if arch not in self.preprocessors:
                logger.warning(f"No preprocessor available for {arch}")
                continue
            
            # Get compiler command
            compiler_cmd = self.COMPILER_PATHS[arch][compiler]
            
            # Generate binary name
            binary_name = f"{base_name}_{arch}_{compiler}_{opt_level.replace('-', '')}"
            binary_path = os.path.join(self.output_dir, "binaries", binary_name)
            
            try:
                # Compile
                cmd_parts = compiler_cmd.split() + [opt_level, source_path, "-o", binary_path]
                subprocess.run(cmd_parts, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Check if compilation succeeded
                if os.path.exists(binary_path):
                    binary_paths[arch] = binary_path
                    logger.info(f"Compiled for {arch}: {binary_path}")
                else:
                    logger.warning(f"Compilation failed for {arch}")
            
            except subprocess.CalledProcessError as e:
                logger.error(f"Compilation error for {arch}: {e}")
        
        return binary_paths
    
    def process_binaries(self, binary_paths):
        """
        Process binaries for multiple architectures.
        
        Args:
            binary_paths: Dictionary mapping architecture to binary path
            
        Returns:
            dict: Dictionary mapping architecture to function data
        """
        function_data = {}
        
        for arch, binary_path in binary_paths.items():
            logger.info(f"Processing {arch} binary: {binary_path}")
            
            try:
                # Disassemble binary
                functions = self.preprocessors[arch].disassemble_binary(binary_path)
                
                # Store function data
                binary_name = os.path.basename(binary_path)
                function_data[arch] = {
                    'binary_name': binary_name,
                    'functions': functions
                }
                
                # Save to disk
                output_path = os.path.join(self.output_dir, "functions", f"{binary_name}.json")
                with open(output_path, 'w') as f:
                    # Convert to serializable format
                    serializable = {
                        'binary_name': binary_name,
                        'architecture': arch,
                        'functions': {name: instrs for name, instrs in functions.items()}
                    }
                    
                    json.dump(serializable, f, indent=2)
            
            except Exception as e:
                logger.error(f"Error processing {binary_path}: {e}")
        
        return function_data
    
    def generate_instruction_mappings(self, function_data):
        """
        Generate cross-architecture instruction mappings.
        
        Args:
            function_data: Dictionary mapping architecture to function data
            
        Returns:
            dict: Dictionary with instruction mappings across architectures
        """
        logger.info("Generating cross-architecture instruction mappings")
        
        # Extract opcodes from each architecture
        arch_opcodes = {}
        
        for arch, data in function_data.items():
            opcodes = set()
            
            for func_name, instructions in data['functions'].items():
                for instr in instructions:
                    parts = instr.split()
                    if parts:
                        opcodes.add(parts[0])
            
            arch_opcodes[arch] = list(opcodes)
        
        # Create mappings from predefined semantic mappings
        instruction_pairs = []
        labels = []
        
        # Add predefined mappings
        for (arch1, opcode1, arch2, opcode2), is_equivalent in self.SEMANTIC_MAPPINGS.items():
            if arch1 in arch_opcodes and arch2 in arch_opcodes:
                if opcode1 in arch_opcodes[arch1] and opcode2 in arch_opcodes[arch2]:
                    instruction_pairs.append((f"{arch1}:{opcode1}", f"{arch2}:{opcode2}"))
                    labels.append(is_equivalent)
        
        # Add some non-equivalent pairs
        for arch1, opcodes1 in arch_opcodes.items():
            for arch2, opcodes2 in arch_opcodes.items():
                if arch1 != arch2:
                    # Select a few random opcodes
                    import random
                    random.seed(42)  # For reproducibility
                    
                    sample1 = random.sample(opcodes1, min(5, len(opcodes1)))
                    sample2 = random.sample(opcodes2, min(5, len(opcodes2)))
                    
                    for opcode1 in sample1:
                        for opcode2 in sample2:
                            key = ((arch1, opcode1, arch2, opcode2))
                            if key not in self.SEMANTIC_MAPPINGS:
                                # Add as non-equivalent (this is a heuristic)
                                instruction_pairs.append((f"{arch1}:{opcode1}", f"{arch2}:{opcode2}"))
                                labels.append(False)
        
        # Save instruction mappings
        output_path = os.path.join(self.output_dir, "mappings", "instruction_mappings.json")
        mapping_data = {
            'instruction_pairs': instruction_pairs,
            'labels': labels
        }
        
        with open(output_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        return mapping_data
    
    def generate_block_mappings(self, function_data):
        """
        Generate cross-architecture block mappings.
        
        Args:
            function_data: Dictionary mapping architecture to function data
            
        Returns:
            dict: Dictionary with block mappings across architectures
        """
        logger.info("Generating cross-architecture block mappings")
        
        # Group functions by name across architectures
        functions_by_name = defaultdict(dict)
        
        for arch, data in function_data.items():
            for func_name, instructions in data['functions'].items():
                functions_by_name[func_name][arch] = instructions
        
        # Find functions available in multiple architectures
        block_pairs = []
        labels = []
        
        # Process each function available in at least two architectures
        for func_name, arch_instructions in functions_by_name.items():
            if len(arch_instructions) < 2:
                continue
            
            # Generate blocks for each architecture
            arch_blocks = {}
            
            for arch, instructions in arch_instructions.items():
                # Generate blocks (sequences of 3-5 instructions)
                blocks = []
                
                for i in range(len(instructions)):
                    # Generate blocks of different sizes
                    for size in range(3, 6):
                        if i + size <= len(instructions):
                            blocks.append(instructions[i:i+size])
                
                arch_blocks[arch] = blocks
            
            # Generate equivalent block pairs across architectures
            architectures = list(arch_blocks.keys())
            
            for i in range(len(architectures)):
                for j in range(i+1, len(architectures)):
                    arch1, arch2 = architectures[i], architectures[j]
                    
                    # Match blocks by position (simplistic approach)
                    # A more sophisticated approach would use sequence alignment or semantic analysis
                    
                    # Use the first few blocks from each architecture
                    num_blocks = min(3, min(len(arch_blocks[arch1]), len(arch_blocks[arch2])))
                    
                    for k in range(num_blocks):
                        # Blocks at same position are likely semantically equivalent
                        block1 = arch_blocks[arch1][k]
                        block2 = arch_blocks[arch2][k]
                        
                        # Add as equivalent
                        block_pairs.append((
                            [f"{arch1}:{instr}" for instr in block1],
                            [f"{arch2}:{instr}" for instr in block2]
                        ))
                        labels.append(True)
                        
                        # Add some non-equivalent pairs
                        if k + 1 < len(arch_blocks[arch1]) and k + 1 < len(arch_blocks[arch2]):
                            # Blocks at different positions are likely not equivalent
                            block1 = arch_blocks[arch1][k]
                            block2 = arch_blocks[arch2][k+1]
                            
                            block_pairs.append((
                                [f"{arch1}:{instr}" for instr in block1],
                                [f"{arch2}:{instr}" for instr in block2]
                            ))
                            labels.append(False)
        
        # Save block mappings
        output_path = os.path.join(self.output_dir, "mappings", "block_mappings.json")
        mapping_data = {
            'block_pairs': block_pairs,
            'labels': labels
        }
        
        with open(output_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        return mapping_data
    
    def process_source_directory(self, source_dir, architectures=None, compiler="gcc", opt_level="-O0"):
        """
        Process a directory of source files across architectures.
        
        Args:
            source_dir: Directory containing source files
            architectures: List of target architectures
            compiler: Compiler to use
            opt_level: Optimization level
            
        Returns:
            dict: Dictionary with instruction and block mappings
        """
        # Find source files
        source_files = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith((".c", ".cpp", ".cc")):
                    source_files.append(os.path.join(root, file))
        
        if not source_files:
            logger.warning(f"No source files found in {source_dir}")
            return {}
        
        # Limit the number of files to process
        if len(source_files) > 10:
            logger.info(f"Found {len(source_files)} source files, limiting to 10")
            source_files = source_files[:10]
        
        # Process each source file
        all_function_data = {}
        
        for source_file in tqdm(source_files, desc="Processing source files"):
            try:
                # Compile for different architectures
                binary_paths = self.compile_across_architectures(
                    source_file, architectures, compiler, opt_level
                )
                
                if not binary_paths:
                    logger.warning(f"No binaries compiled for {source_file}")
                    continue
                
                # Process binaries
                function_data = self.process_binaries(binary_paths)
                
                # Merge function data
                for arch, data in function_data.items():
                    if arch not in all_function_data:
                        all_function_data[arch] = {
                            'binary_name': data['binary_name'],
                            'functions': {}
                        }
                    
                    all_function_data[arch]['functions'].update(data['functions'])
            
            except Exception as e:
                logger.error(f"Error processing {source_file}: {e}")
        
        # Generate mappings
        instruction_mappings = self.generate_instruction_mappings(all_function_data)
        block_mappings = self.generate_block_mappings(all_function_data)
        
        return {
            'instruction_mappings': instruction_mappings,
            'block_mappings': block_mappings
        }
    
    def load_cross_architecture_data(self, mapping_type=None):
        """
        Load cross-architecture mapping data.
        
        Args:
            mapping_type: Type of mapping to load (instruction or block)
            
        Returns:
            dict: Dictionary with mapping data
        """
        if mapping_type == "instruction" or mapping_type is None:
            instruction_path = os.path.join(self.output_dir, "mappings", "instruction_mappings.json")
            if os.path.exists(instruction_path):
                with open(instruction_path, 'r') as f:
                    instruction_data = json.load(f)
            else:
                instruction_data = None
        else:
            instruction_data = None
        
        if mapping_type == "block" or mapping_type is None:
            block_path = os.path.join(self.output_dir, "mappings", "block_mappings.json")
            if os.path.exists(block_path):
                with open(block_path, 'r') as f:
                    block_data = json.load(f)
            else:
                block_data = None
        else:
            block_data = None
        
        result = {}
        
        if mapping_type == "instruction":
            return instruction_data
        elif mapping_type == "block":
            return block_data
        else:
            if instruction_data:
                result['instruction'] = instruction_data
            if block_data:
                result['block'] = block_data
            
            return result