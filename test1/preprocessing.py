"""
Binary code preprocessing module
- Disassembly
- Tokenization
- Graph extraction (CFG)
- Feature extraction
"""
import os
import re
import subprocess
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union

class BinaryPreprocessor:
    def __init__(self, 
                 max_seq_length: int = 200, 
                 max_description_length: int = 30,
                 use_cfg: bool = True,
                 normalize_registers: bool = True,
                 normalize_addresses: bool = True):
        """
        Initialize binary preprocessor
        
        Args:
            max_seq_length: Maximum sequence length for instructions
            max_description_length: Maximum sequence length for descriptions
            use_cfg: Whether to extract Control Flow Graph
            normalize_registers: Whether to normalize register names
            normalize_addresses: Whether to normalize memory addresses
        """
        self.max_seq_length = max_seq_length
        self.max_description_length = max_description_length
        self.use_cfg = use_cfg
        self.normalize_registers = normalize_registers
        self.normalize_addresses = normalize_addresses
        
        # Token vocabularies
        self.opcode_vocab = {"<PAD>": 0, "<UNK>": 1}
        self.operand_vocab = {"<PAD>": 0, "<UNK>": 1, "<REG>": 2, "<ADDR>": 3, "<NUM>": 4}
        self.description_vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        
        # Register normalization mapping
        self.register_map = {
            'eax': '<REG1>', 'ebx': '<REG2>', 'ecx': '<REG3>', 'edx': '<REG4>',
            'esi': '<REG5>', 'edi': '<REG6>', 'ebp': '<REG7>', 'esp': '<REG8>',
            'rax': '<REG1>', 'rbx': '<REG2>', 'rcx': '<REG3>', 'rdx': '<REG4>',
            'rsi': '<REG5>', 'rdi': '<REG6>', 'rbp': '<REG7>', 'rsp': '<REG8>'
        }
        
    def disassemble_binary(self, binary_path: str) -> Dict[str, List[str]]:
        """
        Disassemble a binary file into assembly instructions organized by function
        
        Args:
            binary_path: Path to the binary file
            
        Returns:
            Dict mapping function names to lists of assembly instructions
        """
        # This is a simplified version - in practice, use a library like r2pipe (Radare2)
        # or angr to disassemble binaries properly
        try:
            # Example using objdump (would be replaced with proper disassembler in production)
            result = subprocess.run(
                ['objdump', '-d', binary_path], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"Failed to disassemble {binary_path}: {result.stderr}")
                
            disassembly = result.stdout
            functions = {}
            current_function = None
            current_instructions = []
            
            # Simple parsing of objdump output - this is just an example
            # Real implementation would use a proper disassembler API
            for line in disassembly.splitlines():
                if line.endswith('>:'):  # Function start
                    if current_function and current_instructions:
                        functions[current_function] = current_instructions
                    
                    # Extract function name
                    match = re.search(r'<(.+?)>:', line)
                    if match:
                        current_function = match.group(1)
                        current_instructions = []
                
                elif current_function and '\t' in line:  # Instruction line
                    instruction = line.split('\t')[-1].strip()
                    if instruction:
                        current_instructions.append(instruction)
                        
            # Add the last function
            if current_function and current_instructions:
                functions[current_function] = current_instructions
                
            return functions
            
        except Exception as e:
            print(f"Error disassembling {binary_path}: {e}")
            return {}
    
    def normalize_instruction(self, instruction: str) -> Tuple[str, List[str]]:
        """
        Normalize an assembly instruction by separating opcode from operands
        and normalizing registers and addresses
        
        Args:
            instruction: Assembly instruction string
            
        Returns:
            Tuple of (opcode, list of normalized operands)
        """
        # Split instruction into opcode and operands
        parts = instruction.strip().split(None, 1)
        opcode = parts[0].lower()
        
        operands = []
        if len(parts) > 1:
            # Split operands by comma
            raw_operands = parts[1].split(',')
            for op in raw_operands:
                op = op.strip()
                
                # Normalize registers
                if self.normalize_registers:
                    for reg, replacement in self.register_map.items():
                        if reg in op.lower():
                            op = op.lower().replace(reg, replacement)
                
                # Normalize memory addresses
                if self.normalize_addresses and re.search(r'0x[0-9a-f]+', op):
                    op = re.sub(r'0x[0-9a-f]+', '<ADDR>', op)
                
                # Normalize numeric constants
                if re.search(r'\b\d+\b', op) and '<NUM>' not in op:
                    op = re.sub(r'\b\d+\b', '<NUM>', op)
                
                operands.append(op)
        
        return opcode, operands
    
    def build_cfg(self, function_instructions: List[str]) -> nx.DiGraph:
        """
        Build a Control Flow Graph (CFG) from a list of assembly instructions
        
        Args:
            function_instructions: List of assembly instructions
            
        Returns:
            NetworkX DiGraph representing the CFG
        """
        graph = nx.DiGraph()
        
        # This is a simplified CFG construction
        # A real implementation would need to handle jump target resolution
        # and basic block identification properly
        
        basic_blocks = {}
        current_block = []
        current_block_id = 0
        
        for i, instruction in enumerate(function_instructions):
            # Add instruction to current block
            instr_parts = instruction.lower().split()
            opcode = instr_parts[0] if instr_parts else ""
            
            current_block.append(instruction)
            
            # Check if this instruction ends a basic block
            ends_block = False
            next_block_id = None
            
            # Control flow instructions that end a basic block
            if opcode in ('ret', 'jmp', 'call') or opcode.startswith('j'):
                ends_block = True
                
                # For jumps, identify the target if possible
                if opcode.startswith('j') and len(instr_parts) > 1:
                    target = instr_parts[1]
                    if target.startswith('0x'):
                        # In a real implementation, map address to basic block ID
                        pass
            
            # Last instruction of function
            if i == len(function_instructions) - 1:
                ends_block = True
                
            # If block ends, store it and start a new one
            if ends_block:
                if current_block:
                    basic_blocks[current_block_id] = current_block
                    
                    # Add node to CFG
                    block_features = self.extract_block_features(current_block)
                    graph.add_node(current_block_id, features=block_features)
                    
                    # For sequential flow, add edge to next block
                    if opcode not in ('ret', 'jmp') and i < len(function_instructions) - 1:
                        graph.add_edge(current_block_id, current_block_id + 1)
                        
                    # For jumps, add edge to target if known
                    if next_block_id is not None:
                        graph.add_edge(current_block_id, next_block_id)
                    
                current_block = []
                current_block_id += 1
        
        return graph
    
    def extract_block_features(self, instructions: List[str]) -> np.ndarray:
        """
        Extract features from a basic block
        
        Args:
            instructions: List of instructions in the basic block
            
        Returns:
            Feature vector for the block
        """
        # This is a simplified feature extraction
        # Real implementation would have more sophisticated features
        
        # Example features:
        # - Number of instructions
        # - Number of arithmetic operations
        # - Number of memory operations
        # - Number of control flow instructions
        # - Number of calls
        
        num_instructions = len(instructions)
        arithmetic_ops = sum(1 for i in instructions if any(op in i.split()[0].lower() for op in 
                                                          ['add', 'sub', 'mul', 'div', 'inc', 'dec']))
        memory_ops = sum(1 for i in instructions if any(op in i.split()[0].lower() for op in 
                                                      ['mov', 'lea', 'push', 'pop', 'store', 'load']))
        control_flow = sum(1 for i in instructions if any(op in i.split()[0].lower() for op in 
                                                       ['jmp', 'je', 'jne', 'jz', 'jg', 'jl']))
        calls = sum(1 for i in instructions if 'call' in i.split()[0].lower())
        
        return np.array([num_instructions, arithmetic_ops, memory_ops, control_flow, calls], 
                        dtype=np.float32)
    
    def tokenize_function(self, instructions: List[str]) -> Dict:
        """
        Tokenize a function's instructions into a format suitable for neural models
        
        Args:
            instructions: List of assembly instructions
            
        Returns:
            Dictionary with:
                - token_ids: List of token IDs
                - opcode_ids: List of opcode IDs
                - operand_ids: List of lists of operand IDs
                - cfg: Control Flow Graph (if enabled)
        """
        opcode_ids = []
        operand_ids_list = []
        
        for instruction in instructions[:self.max_seq_length]:
            opcode, operands = self.normalize_instruction(instruction)
            
            # Update vocabulary
            if opcode not in self.opcode_vocab:
                self.opcode_vocab[opcode] = len(self.opcode_vocab)
            
            opcode_id = self.opcode_vocab[opcode]
            opcode_ids.append(opcode_id)
            
            # Process operands
            current_operand_ids = []
            for operand in operands:
                if operand not in self.operand_vocab:
                    self.operand_vocab[operand] = len(self.operand_vocab)
                
                operand_id = self.operand_vocab[operand]
                current_operand_ids.append(operand_id)
                
            operand_ids_list.append(current_operand_ids)
        
        # Pad sequences to max length
        opcode_ids = opcode_ids + [self.opcode_vocab["<PAD>"]] * (self.max_seq_length - len(opcode_ids))
        operand_ids_list = operand_ids_list + [[self.operand_vocab["<PAD>"]]] * (self.max_seq_length - len(operand_ids_list))
        
        result = {
            "opcode_ids": opcode_ids,
            "operand_ids_list": operand_ids_list,
        }
        
        # Add CFG if enabled
        if self.use_cfg:
            result["cfg"] = self.build_cfg(instructions)
            
        return result
    
    def tokenize_description(self, description: str) -> List[int]:
        """
        Tokenize a function description
        
        Args:
            description: Function description text
            
        Returns:
            List of token IDs
        """
        # Simple word-level tokenization - in practice, use a better tokenizer like BPE
        tokens = description.lower().split()
        token_ids = [self.description_vocab["<START>"]]
        
        for token in tokens[:self.max_description_length-2]:  # -2 for START/END tokens
            if token not in self.description_vocab:
                self.description_vocab[token] = len(self.description_vocab)
            
            token_id = self.description_vocab[token]
            token_ids.append(token_id)
            
        token_ids.append(self.description_vocab["<END>"])
        
        # Pad to max length
        token_ids = token_ids + [self.description_vocab["<PAD>"]] * (self.max_description_length - len(token_ids))
        
        return token_ids
    
    def save_vocabularies(self, output_dir: str):
        """
        Save vocabularies to files
        
        Args:
            output_dir: Directory to save vocabularies
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save opcode vocabulary
        with open(os.path.join(output_dir, "opcode_vocab.txt"), "w") as f:
            for token, idx in sorted(self.opcode_vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{idx}\n")
                
        # Save operand vocabulary
        with open(os.path.join(output_dir, "operand_vocab.txt"), "w") as f:
            for token, idx in sorted(self.operand_vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{idx}\n")
                
        # Save description vocabulary
        with open(os.path.join(output_dir, "description_vocab.txt"), "w") as f:
            for token, idx in sorted(self.description_vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{idx}\n")
    
    def load_vocabularies(self, input_dir: str):
        """
        Load vocabularies from files
        
        Args:
            input_dir: Directory containing vocabulary files
        """
        # Load opcode vocabulary
        self.opcode_vocab = {}
        try:
            with open(os.path.join(input_dir, "opcode_vocab.txt"), "r") as f:
                for line in f:
                    token, idx = line.strip().split("\t")
                    self.opcode_vocab[token] = int(idx)
        except FileNotFoundError:
            print(f"Opcode vocabulary file not found in {input_dir}")
            
        # Load operand vocabulary
        self.operand_vocab = {}
        try:
            with open(os.path.join(input_dir, "operand_vocab.txt"), "r") as f:
                for line in f:
                    token, idx = line.strip().split("\t")
                    self.operand_vocab[token] = int(idx)
        except FileNotFoundError:
            print(f"Operand vocabulary file not found in {input_dir}")
            
        # Load description vocabulary
        self.description_vocab = {}
        try:
            with open(os.path.join(input_dir, "description_vocab.txt"), "r") as f:
                for line in f:
                    token, idx = line.strip().split("\t")
                    self.description_vocab[token] = int(idx)
        except FileNotFoundError:
            print(f"Description vocabulary file not found in {input_dir}")
