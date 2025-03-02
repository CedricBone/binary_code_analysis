"""
Synthetic data generator for evaluation tasks.
"""

import random
import numpy as np
from collections import defaultdict

class SyntheticDataGenerator:
    """Generator for synthetic binary code data."""
    
    # Common x86-64 instructions
    COMMON_INSTRUCTIONS = [
        "mov", "push", "pop", "call", "ret", "add", "sub", "cmp", 
        "jmp", "je", "jne", "jg", "jl", "test", "and", "or", "xor",
        "lea", "inc", "dec"
    ]
    
    # Common registers
    REGISTERS = [
        "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", 
        "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp"
    ]
    
    # Known instruction synonyms (functionally equivalent)
    INSTRUCTION_SYNONYMS = {
        ("push rbp", "sub rsp, 8"): True,  # Both reserve stack space
        ("add rax, 0", "nop"): True,       # Both do nothing
        ("mov rax, rax", "nop"): True,     # Both do nothing
        ("sub rax, 0", "nop"): True,       # Both do nothing
        ("xor rax, 0", "nop"): True,       # Both do nothing
        ("inc rax", "add rax, 1"): True,   # Both increment by 1
        ("dec rax", "sub rax, 1"): True,   # Both decrement by 1
        ("xor rax, rax", "mov rax, 0"): True,  # Both set to 0
        ("test rax, rax", "cmp rax, 0"): True,  # Both test if 0
        ("shl rax, 1", "add rax, rax"): True,  # Both multiply by 2
        ("mov [rsp], rax", "push rax"): True,  # Both put rax on stack
    }
    
    # Common instruction patterns (for blocks)
    BLOCK_PATTERNS = [
        ["push rbp", "mov rbp, rsp", "sub rsp, 16"],  # Function prologue
        ["mov rsp, rbp", "pop rbp", "ret"],           # Function epilogue
        ["mov rax, rbx", "add rax, rcx", "mov [rdx], rax"],  # Simple computation
        ["cmp rax, 0", "je LABEL"],     # Conditional jump
        ["mov rax, [rbx]", "add rbx, 8", "dec rcx", "jnz LOOP"],  # Simple loop
    ]
    
    # Known semantically equivalent blocks
    EQUIVALENT_BLOCKS = [
        (
            ["push rbp", "mov rbp, rsp", "sub rsp, 16"],  # Standard prologue
            ["push rbp", "mov rbp, rsp", "sub rsp, 8", "sub rsp, 8"]  # Split stack allocation
        ),
        (
            ["mov eax, 0"],  # Set to 0 directly
            ["xor eax, eax"]  # Clear register via XOR
        ),
        (
            ["cmp eax, ebx", "je label"],  # Compare and jump if equal
            ["sub eax, ebx", "jz label"]  # Subtract and jump if zero
        ),
        (
            ["inc eax", "dec ebx"],  # Increment and decrement
            ["add eax, 1", "sub ebx, 1"]  # Add/subtract immediate
        ),
        (
            ["mov eax, [ebx+8]", "add eax, ecx"],  # Load and add
            ["lea eax, [ebx+8]", "mov eax, [eax]", "add eax, ecx"]  # Use lea first
        )
    ]
    
    # Dead code patterns
    DEAD_CODE_PATTERNS = [
        "mov rax, rax",     # Self-move
        "add rax, 0",       # Add zero
        "sub rax, 0",       # Subtract zero
        "xor rax, 0",       # XOR with zero
        "push rax\npop rax",  # Push-pop same register
        "inc rax\ndec rax",   # Inc-dec same register
        "shl rax, 0",       # Shift by zero
        "and rax, 0xffffffff",  # AND with all ones (32-bit)
        "or rax, 0",         # OR with zero
    ]
    
    def __init__(self, seed=None):
        """
        Initialize the synthetic data generator.
        
        Args:
            seed: Random seed
        """
        self.rng = random.Random(seed)
        np.random.seed(seed)
    
    def generate_instruction(self):
        """
        Generate a random instruction.
        
        Returns:
            str: Generated instruction
        """
        opcode = self.rng.choice(self.COMMON_INSTRUCTIONS)
        
        # Determine operand count based on opcode
        if opcode in ["ret", "nop"]:
            operands = []
        elif opcode in ["push", "pop", "inc", "dec", "jmp", "je", "jne", "jg", "jl", "call"]:
            operands = [self._generate_operand()]
        else:
            operands = [self._generate_operand(), self._generate_operand()]
        
        # Build instruction
        if operands:
            return f"{opcode} {', '.join(operands)}"
        else:
            return opcode
    
    def _generate_operand(self):
        """
        Generate a random operand.
        
        Returns:
            str: Generated operand
        """
        operand_type = self.rng.choice(["register", "immediate", "memory"])
        
        if operand_type == "register":
            return self.rng.choice(self.REGISTERS)
        elif operand_type == "immediate":
            return str(self.rng.randint(0, 100))
        else:  # memory
            base_reg = self.rng.choice(self.REGISTERS)
            if self.rng.random() < 0.5:
                # Simple memory reference
                return f"[{base_reg}]"
            else:
                # Memory reference with offset
                offset = self.rng.randint(0, 128)
                return f"[{base_reg}+{offset}]"
    
    def generate_instruction_sequence(self, length):
        """
        Generate a random instruction sequence.
        
        Args:
            length: Number of instructions
            
        Returns:
            list: List of instructions
        """
        return [self.generate_instruction() for _ in range(length)]
    
    def generate_block_with_dead_code(self, length, num_dead):
        """
        Generate a block with dead code.
        
        Args:
            length: Block length
            num_dead: Number of dead instructions
            
        Returns:
            tuple: (Block, list of dead instruction indices)
        """
        if num_dead > length:
            raise ValueError("Number of dead instructions cannot exceed block length")
        
        # Generate base block
        block = self.generate_instruction_sequence(length - num_dead)
        
        # Insert dead code
        dead_indices = []
        for _ in range(num_dead):
            idx = self.rng.randint(0, len(block))
            dead_code = self.rng.choice(self.DEAD_CODE_PATTERNS)
            block.insert(idx, dead_code)
            dead_indices.append(idx)
        
        return block, dead_indices
    
    def generate_instruction_synonym_data(self, num_samples):
        """
        Generate data for instruction synonym detection.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            dict: Dictionary with 'instruction_pairs' and 'labels'
        """
        instruction_pairs = []
        labels = []
        
        # Include known synonyms
        for pair, is_synonym in self.INSTRUCTION_SYNONYMS.items():
            instruction_pairs.append(pair)
            labels.append(is_synonym)
        
        # Generate random instruction pairs
        while len(instruction_pairs) < num_samples:
            if self.rng.random() < 0.5:
                # Generate likely non-synonym
                instr1 = self.generate_instruction()
                instr2 = self.generate_instruction()
                
                # Ensure they're different instructions
                while instr1 == instr2:
                    instr2 = self.generate_instruction()
                
                pair = (instr1, instr2)
                
                # Check if they're known synonyms
                if pair in self.INSTRUCTION_SYNONYMS or (instr2, instr1) in self.INSTRUCTION_SYNONYMS:
                    continue
                
                instruction_pairs.append(pair)
                labels.append(False)
            else:
                # Take a known synonym and modify it slightly
                base_pair = self.rng.choice(list(self.INSTRUCTION_SYNONYMS.keys()))
                is_synonym = self.INSTRUCTION_SYNONYMS[base_pair]
                
                if self.rng.random() < 0.7:
                    # Maintain synonym relationship but change registers
                    instr1, instr2 = base_pair
                    reg1 = self.rng.choice(self.REGISTERS)
                    reg2 = reg1 if "push" in instr1 or "pop" in instr1 else self.rng.choice(self.REGISTERS)
                    
                    instr1 = instr1.replace("rax", reg1).replace("rbx", reg2)
                    instr2 = instr2.replace("rax", reg1).replace("rbx", reg2)
                    
                    instruction_pairs.append((instr1, instr2))
                    labels.append(is_synonym)
                else:
                    # Use original pair
                    instruction_pairs.append(base_pair)
                    labels.append(is_synonym)
        
        return {
            'instruction_pairs': instruction_pairs,
            'labels': labels
        }
    
    def generate_semantic_block_data(self, num_samples, max_block_length=5):
        """
        Generate data for semantic block equivalence detection.
        
        Args:
            num_samples: Number of samples to generate
            max_block_length: Maximum block length
            
        Returns:
            dict: Dictionary with 'block_pairs' and 'labels'
        """
        block_pairs = []
        labels = []
        
        # Include known equivalent blocks
        for block1, block2 in self.EQUIVALENT_BLOCKS:
            block_pairs.append((block1, block2))
            labels.append(True)
        
        # Generate additional samples
        while len(block_pairs) < num_samples:
            if self.rng.random() < 0.4:
                # Generate likely non-equivalent blocks
                length1 = self.rng.randint(2, max_block_length)
                length2 = self.rng.randint(2, max_block_length)
                
                block1 = self.generate_instruction_sequence(length1)
                block2 = self.generate_instruction_sequence(length2)
                
                block_pairs.append((block1, block2))
                labels.append(False)
            elif self.rng.random() < 0.7:
                # Create a slightly modified version of the same block
                length = self.rng.randint(2, max_block_length)
                block1 = self.generate_instruction_sequence(length)
                
                # Choose a random instruction to replace with a synonym
                if length > 0:
                    idx = self.rng.randint(0, length - 1)
                    instr = block1[idx]
                    
                    # Find a synonym
                    found_synonym = False
                    for (i1, i2), is_syn in self.INSTRUCTION_SYNONYMS.items():
                        if is_syn and (i1 in instr or i2 in instr):
                            # Replace with synonym
                            block2 = block1.copy()
                            if i1 in instr:
                                block2[idx] = instr.replace(i1, i2)
                            else:
                                block2[idx] = instr.replace(i2, i1)
                            found_synonym = True
                            break
                    
                    if found_synonym:
                        block_pairs.append((block1, block2))
                        labels.append(True)
                    else:
                        # Just duplicate the block as fallback
                        block_pairs.append((block1, block1))
                        labels.append(True)
                else:
                    # Edge case: empty block
                    block_pairs.append((block1, block1))
                    labels.append(True)
            else:
                # Use one of the patterns with slight modifications
                pattern = self.rng.choice(self.BLOCK_PATTERNS)
                block1 = pattern.copy()
                
                if self.rng.random() < 0.5:
                    # Identical blocks
                    block2 = block1.copy()
                    is_equivalent = True
                else:
                    # Modify registers but keep same semantic meaning
                    block2 = []
                    for instr in block1:
                        modified = instr
                        for i, reg in enumerate(self.REGISTERS):
                            if reg in instr:
                                # Swap register with another one consistently
                                new_reg = self.REGISTERS[(i + 1) % len(self.REGISTERS)]
                                modified = modified.replace(reg, new_reg)
                        block2.append(modified)
                    
                    is_equivalent = False  # Changing registers usually changes semantics
                
                block_pairs.append((block1, block2))
                labels.append(is_equivalent)
        
        return {
            'block_pairs': block_pairs,
            'labels': labels
        }
    
    def generate_dead_code_data(self, num_samples, max_block_length=10):
        """
        Generate data for dead code detection.
        
        Args:
            num_samples: Number of samples to generate
            max_block_length: Maximum block length
            
        Returns:
            dict: Dictionary with 'code_blocks', 'target_indices', and 'labels'
        """
        code_blocks = []
        target_indices = []
        labels = []
        
        for _ in range(num_samples):
            # Determine if this sample should have dead code
            has_dead_code = self.rng.random() < 0.5
            
            if has_dead_code:
                # Generate block with dead code
                block_length = self.rng.randint(3, max_block_length)
                num_dead = self.rng.randint(1, min(3, block_length - 1))
                
                block, dead_indices = self.generate_block_with_dead_code(block_length, num_dead)
                
                # Choose one of the dead indices to target
                target_idx = self.rng.choice(dead_indices)
                
                code_blocks.append(block)
                target_indices.append(target_idx)
                labels.append(True)  # This is dead code
            else:
                # Generate block without dead code
                block_length = self.rng.randint(3, max_block_length)
                block = self.generate_instruction_sequence(block_length)
                
                # Choose a random instruction to target
                target_idx = self.rng.randint(0, block_length - 1)
                
                code_blocks.append(block)
                target_indices.append(target_idx)
                labels.append(False)  # This is not dead code
        
        return {
            'code_blocks': code_blocks,
            'target_indices': target_indices,
            'labels': labels
        }
    
    def generate_all_task_data(self, num_samples_per_task):
        """
        Generate data for all evaluation tasks.
        
        Args:
            num_samples_per_task: Number of samples per task
            
        Returns:
            dict: Dictionary with data for each task
        """
        return {
            'synonym': self.generate_instruction_synonym_data(num_samples_per_task),
            'block': self.generate_semantic_block_data(num_samples_per_task),
            'dead_code': self.generate_dead_code_data(num_samples_per_task)
        }