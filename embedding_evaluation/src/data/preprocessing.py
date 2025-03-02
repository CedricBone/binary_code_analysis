"""
Binary code preprocessing utilities.
"""

import os
import re
import subprocess
import tempfile
from collections import defaultdict

class BinaryPreprocessor:
    """Preprocessor for binary files."""
    
    def __init__(self, objdump_path="objdump"):
        """
        Initialize the binary preprocessor.
        
        Args:
            objdump_path: Path to objdump executable
        """
        self.objdump_path = objdump_path
    
    def disassemble_binary(self, binary_path):
        """
        Disassemble a binary file.
        
        Args:
            binary_path: Path to the binary file
            
        Returns:
            dict: Dictionary of function name to list of instructions
        """
        try:
            # Run objdump to disassemble the binary
            result = subprocess.run(
                [self.objdump_path, '-d', '-M', 'intel', binary_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Process the output
            return self._parse_objdump_output(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error disassembling binary: {e}")
            return {}
    
    def _parse_objdump_output(self, output):
        """
        Parse objdump output.
        
        Args:
            output: objdump output
            
        Returns:
            dict: Dictionary of function name to list of instructions
        """
        functions = {}
        current_function = None
        instruction_list = []
        
        # Regular expressions for parsing
        function_regex = re.compile(r'^([0-9a-f]+) <(.+)>:$')
        instruction_regex = re.compile(r'^\s+[0-9a-f]+:\s+[0-9a-f]+\s+(.+)$')
        
        for line in output.split('\n'):
            # Check if this is the start of a new function
            function_match = function_regex.match(line)
            if function_match:
                # Save previous function if exists
                if current_function and instruction_list:
                    functions[current_function] = instruction_list
                
                # Start new function
                current_function = function_match.group(2)
                instruction_list = []
                continue
            
            # Check if this is an instruction
            instruction_match = instruction_regex.match(line)
            if instruction_match and current_function:
                instr = instruction_match.group(1).strip()
                instruction_list.append(instr)
        
        # Save the last function
        if current_function and instruction_list:
            functions[current_function] = instruction_list
        
        return functions
    
    def tokenize_instruction(self, instruction):
        """
        Tokenize an assembly instruction.
        
        Args:
            instruction: Assembly instruction string
            
        Returns:
            list: List of tokens
        """
        # Split instruction into opcode and operands
        parts = instruction.split(None, 1)
        opcode = parts[0]
        
        tokens = [opcode]
        
        # Process operands if present
        if len(parts) > 1:
            operands = parts[1].split(',')
            for operand in operands:
                operand = operand.strip()
                # Extract register names, memory references, and immediate values
                if re.match(r'^\[.+\]$', operand):  # Memory reference
                    tokens.append('MEM')
                    # Extract components within the memory reference
                    ref_content = operand[1:-1].strip()
                    ref_parts = re.findall(r'[+-]?\s*\w+', ref_content)
                    tokens.extend([p.strip() for p in ref_parts])
                elif re.match(r'^[re]?[abcd]x$|^[re]?[sd]i$|^[re]?[bs]p$|^r\d+$', operand):  # Register
                    tokens.append(operand)
                elif re.match(r'^0x[0-9a-f]+$|^[0-9]+$', operand):  # Immediate value
                    tokens.append('IMM')
                else:
                    tokens.append(operand)
        
        return tokens
    
    def preprocess_functions(self, functions):
        """
        Preprocess the extracted functions.
        
        Args:
            functions: Dictionary of function name to list of instructions
            
        Returns:
            list: List of tokenized instruction sequences
        """
        tokenized_sequences = []
        
        for func_name, instructions in functions.items():
            # Tokenize each instruction
            tokenized_func = [self.tokenize_instruction(instr) for instr in instructions]
            
            # Add to sequences
            tokenized_sequences.append({
                'function': func_name,
                'tokens': tokenized_func,
                'raw_instructions': instructions
            })
        
        return tokenized_sequences
    
    def process_binary_file(self, binary_path):
        """
        Process a binary file.
        
        Args:
            binary_path: Path to the binary file
            
        Returns:
            list: List of tokenized instruction sequences
        """
        functions = self.disassemble_binary(binary_path)
        return self.preprocess_functions(functions)
    
    def process_directory(self, directory_path, output_path=None):
        """
        Process all binary files in a directory.
        
        Args:
            directory_path: Path to the directory
            output_path: Path to save processed data
            
        Returns:
            list: List of tokenized instruction sequences
        """
        all_sequences = []
        
        # Process each file
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip non-binary files
                if not os.path.isfile(file_path) or file.endswith(('.py', '.txt', '.md', '.json')):
                    continue
                
                try:
                    print(f"Processing {file_path}...")
                    sequences = self.process_binary_file(file_path)
                    all_sequences.extend(sequences)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        # Save processed data if output path is provided
        if output_path and all_sequences:
            self.save_processed_data(all_sequences, output_path)
        
        return all_sequences
    
    def save_processed_data(self, sequences, output_path):
        """
        Save processed data to disk.
        
        Args:
            sequences: List of tokenized instruction sequences
            output_path: Path to save processed data
        """
        import json
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to serializable format
        serializable_data = []
        for seq in sequences:
            serializable_seq = {
                'function': seq['function'],
                'tokens': [' '.join(tokens) for tokens in seq['tokens']],
                'raw_instructions': seq['raw_instructions']
            }
            serializable_data.append(serializable_seq)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    @staticmethod
    def load_processed_data(input_path):
        """
        Load processed data from disk.
        
        Args:
            input_path: Path to load processed data from
            
        Returns:
            list: List of tokenized instruction sequences
        """
        import json
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Convert back to original format
        sequences = []
        for seq in data:
            restored_seq = {
                'function': seq['function'],
                'tokens': [tokens.split() for tokens in seq['tokens']],
                'raw_instructions': seq['raw_instructions']
            }
            sequences.append(restored_seq)
        
        return sequences