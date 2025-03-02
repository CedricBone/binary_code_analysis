"""
Utilities for processing real binary files and extracting ground truth.
"""

import os
import re
import json
import subprocess
import tempfile
import shutil
import hashlib
import logging
from tqdm import tqdm
from collections import defaultdict
import networkx as nx
from pathlib import Path
import requests
import tarfile
import zipfile

from .preprocessing import BinaryPreprocessor

logger = logging.getLogger(__name__)

class RealBinaryLoader:
    """Loader for real binary data with ground truth generation."""
    
    def __init__(self, output_dir="data/real", cache_dir="data/cache", 
                 use_cache=True, objdump_path="objdump"):
        """
        Initialize the real binary loader.
        
        Args:
            output_dir: Directory to store processed data
            cache_dir: Directory to cache downloaded binaries
            use_cache: Whether to use cached data
            objdump_path: Path to the objdump executable
        """
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.preprocessor = BinaryPreprocessor(objdump_path=objdump_path)
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "functions"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "ground_truth"), exist_ok=True)
    
    def download_from_github(self, repo_url, branch="master"):
        """
        Download binaries from a GitHub repository.
        
        Args:
            repo_url: GitHub repository URL
            branch: Repository branch to download
            
        Returns:
            str: Path to downloaded files
        """
        # Extract owner and repo name
        match = re.match(r"https://github.com/([^/]+)/([^/]+)", repo_url)
        if not match:
            raise ValueError(f"Invalid GitHub URL: {repo_url}")
        
        owner, repo = match.groups()
        
        # Create cache directory for this repo
        repo_cache_dir = os.path.join(self.cache_dir, f"{owner}_{repo}")
        os.makedirs(repo_cache_dir, exist_ok=True)
        
        # Check if already downloaded
        if self.use_cache and os.path.exists(os.path.join(repo_cache_dir, "downloaded")):
            logger.info(f"Using cached download for {repo_url}")
            return repo_cache_dir
        
        # Download archive
        archive_url = f"https://github.com/{owner}/{repo}/archive/{branch}.zip"
        logger.info(f"Downloading {archive_url}")
        
        response = requests.get(archive_url, stream=True)
        response.raise_for_status()
        
        archive_path = os.path.join(repo_cache_dir, f"{repo}.zip")
        with open(archive_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading"):
                f.write(chunk)
        
        # Extract archive
        logger.info(f"Extracting {archive_path}")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(repo_cache_dir)
        
        # Mark as downloaded
        with open(os.path.join(repo_cache_dir, "downloaded"), 'w') as f:
            f.write(f"Downloaded from {repo_url} branch {branch}")
        
        return repo_cache_dir
    
    def compile_with_multiple_settings(self, source_dir, output_dir=None, 
                                      compilers=None, opt_levels=None):
        """
        Compile source code with multiple compilers and optimization levels.
        
        Args:
            source_dir: Directory containing source code
            output_dir: Directory to store compiled binaries
            compilers: List of compilers to use
            opt_levels: List of optimization levels
            
        Returns:
            list: Paths to compiled binaries
        """
        if output_dir is None:
            output_dir = os.path.join(self.cache_dir, "compiled")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if compilers is None:
            compilers = ["gcc", "clang"]
        
        if opt_levels is None:
            opt_levels = ["-O0", "-O1", "-O2", "-O3"]
        
        # Check for source files
        source_files = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith((".c", ".cpp", ".cc")):
                    source_files.append(os.path.join(root, file))
        
        if not source_files:
            logger.warning(f"No source files found in {source_dir}")
            return []
        
        # Compile each source file with each compiler and optimization level
        binaries = []
        
        for source_file in tqdm(source_files[:100], desc="Compiling source files"):  # Limit to 100 files
            file_name = os.path.basename(source_file).split(".")[0]
            
            for compiler in compilers:
                for opt_level in opt_levels:
                    try:
                        # Check if compiler is available
                        if shutil.which(compiler) is None:
                            logger.warning(f"{compiler} not found, skipping")
                            continue
                        
                        # Generate binary name
                        binary_name = f"{file_name}_{compiler}_{opt_level.replace('-', '')}"
                        binary_path = os.path.join(output_dir, binary_name)
                        
                        # Compile
                        cmd = [compiler, opt_level, source_file, "-o", binary_path]
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        
                        # Add to list if compilation successful
                        if os.path.exists(binary_path):
                            binaries.append(binary_path)
                    
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Compilation failed for {source_file} with {compiler} {opt_level}: {e}")
        
        logger.info(f"Compiled {len(binaries)} binaries")
        return binaries
    
    def extract_functions_from_binaries(self, binary_paths):
        """
        Extract functions from binaries for analysis.
        
        Args:
            binary_paths: List of paths to binary files
            
        Returns:
            dict: Dictionary of functions by binary
        """
        functions_by_binary = {}
        
        for binary_path in tqdm(binary_paths, desc="Extracting functions"):
            try:
                # Process binary file
                functions = self.preprocessor.disassemble_binary(binary_path)
                
                # Store functions
                binary_name = os.path.basename(binary_path)
                functions_by_binary[binary_name] = functions
                
                # Save to disk
                output_path = os.path.join(self.output_dir, "functions", f"{binary_name}.json")
                with open(output_path, 'w') as f:
                    # Convert to serializable format
                    serializable = {}
                    for func_name, instructions in functions.items():
                        serializable[func_name] = instructions
                    
                    json.dump(serializable, f, indent=2)
            
            except Exception as e:
                logger.error(f"Error processing {binary_path}: {e}")
        
        return functions_by_binary
    
    def find_instruction_synonyms(self, functions_by_binary):
        """
        Find instruction synonyms by comparing across optimization levels.
        
        Args:
            functions_by_binary: Dictionary of functions by binary
            
        Returns:
            list: List of instruction synonym pairs with labels
        """
        logger.info("Finding instruction synonyms")
        
        # Group binaries by base name (to compare same source with different opts)
        binary_groups = defaultdict(list)
        for binary_name in functions_by_binary:
            # Extract base name (without compiler and opt level)
            base_name = '_'.join(binary_name.split('_')[:-2])
            binary_groups[base_name].append(binary_name)
        
        # Find synonyms by analyzing instruction patterns
        instruction_pairs = []
        labels = []
        
        # Dictionary to track already found synonyms
        synonym_dict = {}
        
        # Known synonyms from analysis
        known_synonyms = {
            "xor eax, eax": "mov eax, 0",
            "xor rax, rax": "mov rax, 0",
            "add rax, 1": "inc rax",
            "sub rax, 1": "dec rax",
            "add rbx, 0": "nop",
            "jmp .+5": "nop",
        }
        
        # Add known synonyms first
        for instr1, instr2 in known_synonyms.items():
            if (instr1, instr2) not in synonym_dict and (instr2, instr1) not in synonym_dict:
                instruction_pairs.append((instr1, instr2))
                labels.append(True)
                synonym_dict[(instr1, instr2)] = True
        
        # Find synonyms across optimization levels
        for base_name, binaries in binary_groups.items():
            if len(binaries) < 2:
                continue
            
            # Group by function name
            functions_across_binaries = defaultdict(list)
            for binary in binaries:
                for func_name, instructions in functions_by_binary[binary].items():
                    functions_across_binaries[func_name].append((binary, instructions))
            
            # Look for patterns indicating synonyms
            for func_name, func_variants in functions_across_binaries.items():
                if len(func_variants) < 2:
                    continue
                
                # Compare each pair of variants
                for i in range(len(func_variants)):
                    for j in range(i + 1, len(func_variants)):
                        binary_i, instrs_i = func_variants[i]
                        binary_j, instrs_j = func_variants[j]
                        
                        # Skip if identical
                        if instrs_i == instrs_j:
                            continue
                        
                        # Extract optimization levels
                        opt_i = binary_i.split('_')[-1]
                        opt_j = binary_j.split('_')[-1]
                        
                        # Focus on comparisons between O0 and higher opt levels
                        if not (opt_i == 'O0' or opt_j == 'O0'):
                            continue
                        
                        # Find potential instruction replacements
                        self._find_synonyms_in_instruction_lists(
                            instrs_i, instrs_j, instruction_pairs, labels, synonym_dict
                        )
        
        # Generate some non-synonyms as well
        self._generate_non_synonyms(instruction_pairs, labels, synonym_dict)
        
        # Save to disk
        output_path = os.path.join(self.output_dir, "ground_truth", "instruction_synonyms.json")
        with open(output_path, 'w') as f:
            json.dump({
                'instruction_pairs': instruction_pairs,
                'labels': labels
            }, f, indent=2)
        
        return instruction_pairs, labels
    
    def _find_synonyms_in_instruction_lists(self, instrs_i, instrs_j, instruction_pairs, labels, synonym_dict):
        """
        Find potential synonyms by comparing two instruction lists.
        
        Args:
            instrs_i: First instruction list
            instrs_j: Second instruction list
            instruction_pairs: List to store instruction pairs
            labels: List to store labels
            synonym_dict: Dictionary to track found synonyms
        """
        # Focus on short sequences that differ by one instruction
        for idx_i in range(len(instrs_i) - 2):
            for idx_j in range(len(instrs_j) - 2):
                # Check if surrounding context is identical
                if (instrs_i[idx_i] == instrs_j[idx_j] and 
                    instrs_i[idx_i + 2] == instrs_j[idx_j + 2] and
                    instrs_i[idx_i + 1] != instrs_j[idx_j + 1]):
                    
                    # Potential synonym found
                    instr1 = instrs_i[idx_i + 1]
                    instr2 = instrs_j[idx_j + 1]
                    
                    if (instr1, instr2) not in synonym_dict and (instr2, instr1) not in synonym_dict:
                        # Filter out unlikely synonyms
                        if self._are_potential_synonyms(instr1, instr2):
                            instruction_pairs.append((instr1, instr2))
                            labels.append(True)
                            synonym_dict[(instr1, instr2)] = True
                            
                            # Only collect a limited number of synonyms
                            if len(instruction_pairs) >= 1000:
                                return
    
    def _are_potential_synonyms(self, instr1, instr2):
        """
        Check if two instructions are potential synonyms.
        
        Args:
            instr1: First instruction
            instr2: Second instruction
            
        Returns:
            bool: True if potentially synonyms, False otherwise
        """
        # Extract opcodes
        opcode1 = instr1.split()[0]
        opcode2 = instr2.split()[0]
        
        # Common opcode pairs that could be synonyms
        synonym_opcodes = {
            'mov': ['lea', 'xor'],
            'add': ['inc', 'lea'],
            'sub': ['dec'],
            'jmp': ['je', 'jne'],
            'test': ['cmp'],
            'and': ['test'],
            'xor': ['mov', 'sub'],
            'push': ['mov', 'sub'],
            'pop': ['mov', 'add'],
        }
        
        # Check if opcodes match known synonym pairs
        if opcode1 in synonym_opcodes and opcode2 in synonym_opcodes.get(opcode1, []):
            return True
        if opcode2 in synonym_opcodes and opcode1 in synonym_opcodes.get(opcode2, []):
            return True
        
        # Check for nop equivalents
        nop_patterns = ['add .*, 0', 'sub .*, 0', 'xor .*, 0', 'mov .*, .*']
        if opcode1 == 'nop' or opcode2 == 'nop':
            for pattern in nop_patterns:
                if (re.match(pattern, instr1) or re.match(pattern, instr2)):
                    return True
        
        return False
    
    def _generate_non_synonyms(self, instruction_pairs, labels, synonym_dict, num_non_synonyms=None):
        """
        Generate non-synonym pairs for balanced dataset.
        
        Args:
            instruction_pairs: List of instruction pairs
            labels: List of labels
            synonym_dict: Dictionary of known synonyms
            num_non_synonyms: Number of non-synonyms to generate
        """
        # Count existing synonyms
        num_synonyms = sum(labels)
        
        if num_non_synonyms is None:
            # Generate approximately equal number of non-synonyms
            num_non_synonyms = num_synonyms
        
        # Collect all unique instructions
        all_instructions = set()
        for instr1, instr2 in instruction_pairs:
            all_instructions.add(instr1)
            all_instructions.add(instr2)
        
        all_instructions = list(all_instructions)
        
        # Generate non-synonym pairs
        import random
        random.seed(42)  # For reproducibility
        
        non_synonyms_added = 0
        attempts = 0
        
        while non_synonyms_added < num_non_synonyms and attempts < 10000:
            instr1 = random.choice(all_instructions)
            instr2 = random.choice(all_instructions)
            
            attempts += 1
            
            # Skip if identical or already known
            if (instr1 == instr2 or 
                (instr1, instr2) in synonym_dict or 
                (instr2, instr1) in synonym_dict):
                continue
            
            # Skip if likely to be synonyms
            if self._are_potential_synonyms(instr1, instr2):
                continue
            
            # Add as non-synonym
            instruction_pairs.append((instr1, instr2))
            labels.append(False)
            synonym_dict[(instr1, instr2)] = False
            non_synonyms_added += 1
    
    def find_block_equivalence(self, functions_by_binary):
        """
        Find semantically equivalent blocks across binaries.
        
        Args:
            functions_by_binary: Dictionary of functions by binary
            
        Returns:
            list: List of block pairs with labels
        """
        logger.info("Finding semantically equivalent blocks")
        
        # Group binaries by base name
        binary_groups = defaultdict(list)
        for binary_name in functions_by_binary:
            base_name = '_'.join(binary_name.split('_')[:-2])
            binary_groups[base_name].append(binary_name)
        
        # Find semantically equivalent blocks
        block_pairs = []
        labels = []
        
        # Track already found pairs
        block_pair_dict = {}
        
        # Process each group
        for base_name, binaries in binary_groups.items():
            if len(binaries) < 2:
                continue
            
            # Group by function name
            functions_across_binaries = defaultdict(list)
            for binary in binaries:
                for func_name, instructions in functions_by_binary[binary].items():
                    functions_across_binaries[func_name].append((binary, instructions))
            
            # Process each function
            for func_name, func_variants in functions_across_binaries.items():
                if len(func_variants) < 2:
                    continue
                
                # Compare each pair of variants
                for i in range(len(func_variants)):
                    for j in range(i + 1, len(func_variants)):
                        binary_i, instrs_i = func_variants[i]
                        binary_j, instrs_j = func_variants[j]
                        
                        # Extract optimization levels
                        opt_i = binary_i.split('_')[-1]
                        opt_j = binary_j.split('_')[-1]
                        
                        # Focus on comparisons between different opt levels
                        if opt_i == opt_j:
                            continue
                        
                        # Find equivalent blocks
                        self._find_equivalent_blocks(
                            instrs_i, instrs_j, block_pairs, labels, block_pair_dict
                        )
                        
                        # Limit the number of pairs
                        if len(block_pairs) >= 1000:
                            break
                    
                    if len(block_pairs) >= 1000:
                        break
                
                if len(block_pairs) >= 1000:
                    break
        
        # Generate non-equivalent blocks
        self._generate_non_equivalent_blocks(block_pairs, labels, block_pair_dict)
        
        # Save to disk
        output_path = os.path.join(self.output_dir, "ground_truth", "block_equivalence.json")
        with open(output_path, 'w') as f:
            json.dump({
                'block_pairs': block_pairs,
                'labels': labels
            }, f, indent=2)
        
        return block_pairs, labels
    
    def _find_equivalent_blocks(self, instrs_i, instrs_j, block_pairs, labels, block_pair_dict, block_size=3):
        """
        Find semantically equivalent blocks in two instruction lists.
        
        Args:
            instrs_i: First instruction list
            instrs_j: Second instruction list
            block_pairs: List to store block pairs
            labels: List to store labels
            block_pair_dict: Dictionary to track found pairs
            block_size: Size of blocks to compare
        """
        # Define a function to compute block hash (for fast comparison)
        def get_block_hash(block):
            # Normalize instructions (remove specific addresses, registers)
            normalized = []
            for instr in block:
                # Keep only opcodes and general structure
                parts = instr.split()
                if not parts:
                    continue
                opcode = parts[0]
                normalized.append(opcode)
            
            return hashlib.md5(' '.join(normalized).encode()).hexdigest()
        
        # Group blocks by hash
        block_hashes_i = {}
        block_hashes_j = {}
        
        # Generate blocks from instructions
        for idx_i in range(len(instrs_i) - block_size + 1):
            block_i = instrs_i[idx_i:idx_i + block_size]
            block_hash = get_block_hash(block_i)
            block_hashes_i[block_hash] = block_i
        
        for idx_j in range(len(instrs_j) - block_size + 1):
            block_j = instrs_j[idx_j:idx_j + block_size]
            block_hash = get_block_hash(block_j)
            block_hashes_j[block_hash] = block_j
        
        # Find equivalent blocks by hash
        for hash_val in block_hashes_i:
            if hash_val in block_hashes_j:
                block_i = block_hashes_i[hash_val]
                block_j = block_hashes_j[hash_val]
                
                # Skip if identical
                if block_i == block_j:
                    continue
                
                # Skip if already found
                pair_key = (tuple(block_i), tuple(block_j))
                if pair_key in block_pair_dict:
                    continue
                
                # Add as equivalent
                block_pairs.append((block_i, block_j))
                labels.append(True)
                block_pair_dict[pair_key] = True
    
    def _generate_non_equivalent_blocks(self, block_pairs, labels, block_pair_dict, num_non_equivalent=None):
        """
        Generate non-equivalent block pairs for balanced dataset.
        
        Args:
            block_pairs: List of block pairs
            labels: List of labels
            block_pair_dict: Dictionary of known pairs
            num_non_equivalent: Number of non-equivalent pairs to generate
        """
        # Count existing equivalent pairs
        num_equivalent = sum(labels)
        
        if num_non_equivalent is None:
            # Generate approximately equal number
            num_non_equivalent = num_equivalent
        
        # Collect all unique blocks
        all_blocks = []
        for block1, block2 in block_pairs:
            if tuple(block1) not in all_blocks:
                all_blocks.append(tuple(block1))
            if tuple(block2) not in all_blocks:
                all_blocks.append(tuple(block2))
        
        # Generate non-equivalent pairs
        import random
        random.seed(42)  # For reproducibility
        
        non_equivalent_added = 0
        attempts = 0
        
        while non_equivalent_added < num_non_equivalent and attempts < 10000:
            block1 = random.choice(all_blocks)
            block2 = random.choice(all_blocks)
            
            attempts += 1
            
            # Skip if identical or already known
            if (block1 == block2 or 
                (block1, block2) in block_pair_dict or 
                (block2, block1) in block_pair_dict):
                continue
            
            # Add as non-equivalent
            block_pairs.append((list(block1), list(block2)))
            labels.append(False)
            block_pair_dict[(block1, block2)] = False
            non_equivalent_added += 1
    
    def find_dead_code(self, functions_by_binary):
        """
        Find dead code by comparing optimization levels.
        
        Args:
            functions_by_binary: Dictionary of functions by binary
            
        Returns:
            dict: Dictionary with code blocks, target indices, and labels
        """
        logger.info("Finding dead code")
        
        # Group binaries by base name and compiler
        binary_groups = defaultdict(list)
        for binary_name in functions_by_binary:
            # Extract compiler and base name
            parts = binary_name.split('_')
            if len(parts) < 3:
                continue
            
            compiler = parts[-2]
            base_name = '_'.join(parts[:-2])
            key = f"{base_name}_{compiler}"
            binary_groups[key].append(binary_name)
        
        # Find dead code by comparing optimization levels
        code_blocks = []
        target_indices = []
        labels = []
        
        # Process each group
        for group_key, binaries in binary_groups.items():
            # Sort by optimization level
            binaries.sort(key=lambda x: x.split('_')[-1])
            
            # Need at least O0 and a higher optimization level
            if len(binaries) < 2 or not any(b.endswith('O0') for b in binaries):
                continue
            
            # Find O0 and highest available optimization level
            o0_binary = next(b for b in binaries if b.endswith('O0'))
            highest_opt_binary = binaries[-1]
            
            # Group by function name
            o0_functions = functions_by_binary[o0_binary]
            highest_functions = functions_by_binary[highest_opt_binary]
            
            # Find common functions
            common_funcs = set(o0_functions.keys()) & set(highest_functions.keys())
            
            for func_name in common_funcs:
                o0_instrs = o0_functions[func_name]
                high_instrs = highest_functions[func_name]
                
                # Skip if identical
                if o0_instrs == high_instrs:
                    continue
                
                # Find dead code in O0 by comparing with higher optimization
                self._find_dead_code_in_function(
                    o0_instrs, high_instrs, code_blocks, target_indices, labels
                )
                
                # Limit the number of samples
                if len(code_blocks) >= 1000:
                    break
            
            if len(code_blocks) >= 1000:
                break
        
        # Generate some non-dead code examples for balance
        self._generate_non_dead_code(code_blocks, target_indices, labels)
        
        # Save to disk
        output_path = os.path.join(self.output_dir, "ground_truth", "dead_code.json")
        with open(output_path, 'w') as f:
            json.dump({
                'code_blocks': code_blocks,
                'target_indices': target_indices,
                'labels': labels
            }, f, indent=2)
        
        return {
            'code_blocks': code_blocks,
            'target_indices': target_indices,
            'labels': labels
        }
    
    def _find_dead_code_in_function(self, o0_instrs, high_instrs, code_blocks, target_indices, labels):
        """
        Find dead code in a function by comparing optimization levels.
        
        Args:
            o0_instrs: Instructions from O0 optimization
            high_instrs: Instructions from higher optimization
            code_blocks: List to store code blocks
            target_indices: List to store target indices
            labels: List to store labels
        """
        # Extract opcodes for easier comparison
        o0_opcodes = [instr.split()[0] if instr.split() else "" for instr in o0_instrs]
        high_opcodes = [instr.split()[0] if instr.split() else "" for instr in high_instrs]
        
        # Find longest common subsequence to align instructions
        lcs_indices = self._longest_common_subsequence_indices(o0_opcodes, high_opcodes)
        
        if not lcs_indices:
            return
        
        # Map O0 indices to high opt indices
        o0_to_high = {i: j for i, j in lcs_indices}
        
        # Instructions in O0 but not in higher opt are potentially dead code
        for i in range(len(o0_instrs)):
            # Skip if this instruction is part of the common subsequence
            if i in o0_to_high:
                continue
            
            # Extract context (a few instructions before and after)
            start_idx = max(0, i - 2)
            end_idx = min(len(o0_instrs), i + 3)
            block = o0_instrs[start_idx:end_idx]
            
            # Determine the index of the target instruction in the block
            target_idx = i - start_idx
            
            # Skip obvious cases
            instruction = o0_instrs[i]
            if self._is_obviously_not_dead_code(instruction):
                continue
            
            # Add to dataset
            code_blocks.append(block)
            target_indices.append(target_idx)
            labels.append(True)  # This is dead code
    
    def _is_obviously_not_dead_code(self, instruction):
        """
        Check if an instruction is obviously not dead code.
        
        Args:
            instruction: Instruction to check
            
        Returns:
            bool: True if obviously not dead code, False otherwise
        """
        # Instructions that are unlikely to be dead code
        if not instruction or not instruction.split():
            return True
        
        opcode = instruction.split()[0]
        
        # Control flow instructions are usually not dead
        control_flow = ['jmp', 'je', 'jne', 'jg', 'jl', 'jge', 'jle', 'ja', 'jb', 'ret', 'call']
        if opcode in control_flow:
            return True
        
        return False
    
    def _longest_common_subsequence_indices(self, seq1, seq2):
        """
        Find indices of longest common subsequence.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            list: List of (i, j) index pairs
        """
        # Create LCS table
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        # Backtrack to find indices
        indices = []
        i, j = m, n
        
        while i > 0 and j > 0:
            if seq1[i - 1] == seq2[j - 1]:
                indices.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        
        return indices[::-1]  # Reverse to get in order
    
    def _generate_non_dead_code(self, code_blocks, target_indices, labels, num_non_dead=None):
        """
        Generate non-dead code examples for balanced dataset.
        
        Args:
            code_blocks: List of code blocks
            target_indices: List of target indices
            labels: List of labels
            num_non_dead: Number of non-dead examples to generate
        """
        # Count existing dead code examples
        num_dead = sum(labels)
        
        if num_non_dead is None:
            # Generate approximately equal number
            num_non_dead = num_dead
        
        # Use some of the same blocks but different target indices
        import random
        random.seed(42)  # For reproducibility
        
        non_dead_added = 0
        
        while non_dead_added < num_non_dead and non_dead_added < len(code_blocks):
            # Select a random block
            idx = random.randint(0, len(code_blocks) - 1)
            block = code_blocks[idx]
            current_target = target_indices[idx]
            
            # Select a different instruction as target
            new_target = current_target
            while new_target == current_target:
                new_target = random.randint(0, len(block) - 1)
            
            # Select a different instruction that's likely not dead code
            instruction = block[new_target]
            if self._is_obviously_not_dead_code(instruction):
                code_blocks.append(block)
                target_indices.append(new_target)
                labels.append(False)  # Not dead code
                non_dead_added += 1
    
    def process_and_generate_ground_truth(self, repo_urls=None, source_dirs=None):
        """
        Process binaries and generate ground truth for evaluation tasks.
        
        Args:
            repo_urls: List of GitHub repository URLs
            source_dirs: List of source code directories
            
        Returns:
            dict: Dictionary with ground truth data for each task
        """
        # Use default test repositories if none provided
        if repo_urls is None:
            repo_urls = [
                "https://github.com/antirez/redis",
                "https://github.com/sqlite/sqlite",
                "https://github.com/tmux/tmux"
            ]
        
        # Download repositories
        binary_paths = []
        
        for repo_url in repo_urls:
            try:
                repo_dir = self.download_from_github(repo_url)
                logger.info(f"Downloaded {repo_url} to {repo_dir}")
                
                # Find source directories
                source_dirs_found = []
                for root, dirs, files in os.walk(repo_dir):
                    if any(file.endswith((".c", ".cpp", ".cc")) for file in files):
                        source_dirs_found.append(root)
                
                # Compile source code
                for source_dir in source_dirs_found[:3]:  # Limit to 3 directories
                    compiled_paths = self.compile_with_multiple_settings(source_dir)
                    binary_paths.extend(compiled_paths)
                    
                    # Limit the number of binaries
                    if len(binary_paths) >= 100:
                        break
                
                if len(binary_paths) >= 100:
                    break
            
            except Exception as e:
                logger.error(f"Error processing {repo_url}: {e}")
        
        # Add custom source directories if provided
        if source_dirs:
            for source_dir in source_dirs:
                try:
                    compiled_paths = self.compile_with_multiple_settings(source_dir)
                    binary_paths.extend(compiled_paths)
                except Exception as e:
                    logger.error(f"Error processing {source_dir}: {e}")
        
        logger.info(f"Found {len(binary_paths)} binary files")
        
        # Extract functions from binaries
        functions_by_binary = self.extract_functions_from_binaries(binary_paths)
        
        # Generate ground truth for each task
        instruction_pairs, synonym_labels = self.find_instruction_synonyms(functions_by_binary)
        block_pairs, block_labels = self.find_block_equivalence(functions_by_binary)
        dead_code_data = self.find_dead_code(functions_by_binary)
        
        return {
            'instruction_synonyms': {
                'instruction_pairs': instruction_pairs,
                'labels': synonym_labels
            },
            'block_equivalence': {
                'block_pairs': block_pairs,
                'labels': block_labels
            },
            'dead_code': dead_code_data
        }
    
    def load_ground_truth(self, task=None):
        """
        Load ground truth data for evaluation tasks.
        
        Args:
            task: Task name to load (synonym, block, or dead_code)
            
        Returns:
            dict: Dictionary with ground truth data
        """
        ground_truth_dir = os.path.join(self.output_dir, "ground_truth")
        
        if task == "synonym" or task is None:
            synonym_path = os.path.join(ground_truth_dir, "instruction_synonyms.json")
            if os.path.exists(synonym_path):
                with open(synonym_path, 'r') as f:
                    synonym_data = json.load(f)
            else:
                synonym_data = None
        else:
            synonym_data = None
        
        if task == "block" or task is None:
            block_path = os.path.join(ground_truth_dir, "block_equivalence.json")
            if os.path.exists(block_path):
                with open(block_path, 'r') as f:
                    block_data = json.load(f)
            else:
                block_data = None
        else:
            block_data = None
        
        if task == "dead_code" or task is None:
            dead_code_path = os.path.join(ground_truth_dir, "dead_code.json")
            if os.path.exists(dead_code_path):
                with open(dead_code_path, 'r') as f:
                    dead_code_data = json.load(f)
            else:
                dead_code_data = None
        else:
            dead_code_data = None
        
        result = {}
        
        if task == "synonym":
            return synonym_data
        elif task == "block":
            return block_data
        elif task == "dead_code":
            return dead_code_data
        else:
            if synonym_data:
                result['synonym'] = synonym_data
            if block_data:
                result['block'] = block_data
            if dead_code_data:
                result['dead_code'] = dead_code_data
            
            return result