#!/usr/bin/env python3
"""
Extract functions from compiled binaries using objdump.
"""

import os
import re
import glob
import json
import argparse
import random
import subprocess
from multiprocessing import Pool
from pathlib import Path
import config
from utils import (
    logger, create_directories, get_build_path, run_command, 
    get_disassembler_command, parse_function, normalize_instruction,
    save_json, load_json
)

def find_binaries(build_dir):
    """Find all binary files in the build directory"""
    binary_extensions = ['.o', '.so', '.a', '', '.bin', '.elf']
    binaries = []
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(build_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Skip directories and non-binary files
            if os.path.isdir(file_path):
                continue
                
            # Check if it's a binary file
            extension = os.path.splitext(file)[1]
            if extension in binary_extensions:
                # Run 'file' command to check if it's a binary
                try:
                    file_output = subprocess.check_output(['file', file_path]).decode('utf-8')
                    if 'ELF' in file_output and 'executable' in file_output:
                        binaries.append(file_path)
                except subprocess.CalledProcessError:
                    continue
    
    return binaries

def disassemble_binary(binary_path, architecture):
    """Disassemble a binary using the appropriate disassembler"""
    output_path = binary_path + ".dis"
    
    # Check if already disassembled
    if os.path.exists(output_path):
        logger.debug(f"Binary already disassembled: {output_path}")
        with open(output_path, 'r') as f:
            return f.read()
    
    # Get disassembler command
    command = get_disassembler_command(architecture, binary_path, output_path)
    
    try:
        # Run disassembler
        run_command(command)
        
        # Read disassembled output
        with open(output_path, 'r') as f:
            disassembly = f.read()
        
        return disassembly
    except Exception as e:
        logger.error(f"Error disassembling {binary_path}: {e}")
        return ""

def extract_functions_from_disassembly(disassembly, architecture):
    """Extract functions from disassembled output"""
    # Dictionary to store functions (name -> instructions)
    functions = {}
    
    # Define regex patterns based on architecture
    if architecture == "x86_64":
        function_pattern = r'^[0-9a-f]+ <(.+?)>:$'
    elif architecture == "arm":
        function_pattern = r'^[0-9a-f]+ <(.+?)>:$'
    elif architecture == "mips":
        function_pattern = r'^[0-9a-f]+ <(.+?)>:$'
    elif architecture == "powerpc":
        function_pattern = r'^[0-9a-f]+ <(.+?)>:$'
    else:
        logger.error(f"Unsupported architecture: {architecture}")
        return {}
    
    # Split disassembly into functions
    current_function = None
    current_content = []
    
    for line in disassembly.split('\n'):
        # Check if line starts a new function
        match = re.match(function_pattern, line)
        if match:
            # Save previous function if exists
            if current_function and current_content:
                functions[current_function] = '\n'.join(current_content)
            
            # Start new function
            current_function = match.group(1)
            current_content = []
        elif current_function:
            # Add line to current function
            current_content.append(line)
    
    # Add the last function
    if current_function and current_content:
        functions[current_function] = '\n'.join(current_content)
    
    return functions

def process_function(function_name, function_content, architecture):
    """Process a function and return its normalized instructions"""
    # Parse the function into instructions
    instructions = parse_function(function_content, architecture)
    
    # Skip if too small or too large
    if len(instructions) < config.MIN_FUNCTION_SIZE or len(instructions) > config.MAX_FUNCTION_SIZE:
        return None
    
    # Normalize instructions
    normalized_instructions = [normalize_instruction(instr, architecture) for instr in instructions]
    
    # Return the result
    return {
        "name": function_name,
        "instructions": normalized_instructions,
        "raw": function_content
    }

def extract_functions_from_binary(args):
    """Extract functions from a binary file"""
    project, architecture, compiler, opt_level, binary_path = args
    
    logger.info(f"Extracting functions from {os.path.basename(binary_path)} for {architecture} with {compiler} -{opt_level}")
    
    try:
        # Disassemble binary
        disassembly = disassemble_binary(binary_path, architecture)
        if not disassembly:
            logger.warning(f"Empty disassembly for {binary_path}")
            return []
        
        # Extract functions
        function_dict = extract_functions_from_disassembly(disassembly, architecture)
        logger.info(f"Found {len(function_dict)} functions in {os.path.basename(binary_path)}")
        
        # Process functions
        processed_functions = []
        for func_name, func_content in function_dict.items():
            # Skip compiler-generated functions
            if (func_name.startswith('.') or func_name.startswith('_')
                or 'helper' in func_name.lower() or 'stub' in func_name.lower()):
                continue
            
            # Process function
            processed = process_function(func_name, func_content, architecture)
            if processed:
                processed_functions.append(processed)
        
        logger.info(f"Processed {len(processed_functions)} functions from {os.path.basename(binary_path)}")
        
        # Return the processed functions
        return processed_functions
    
    except Exception as e:
        logger.error(f"Error extracting functions from {binary_path}: {e}")
        return []

def extract_functions_from_project(args):
    """Extract functions from a project"""
    project, architecture, compiler, opt_level = args
    
    try:
        # Get build directory
        build_dir = get_build_path(project, architecture, compiler, opt_level)
        if not os.path.exists(build_dir):
            logger.warning(f"Build directory does not exist: {build_dir}")
            return []
        
        # Find binaries
        binaries = find_binaries(build_dir)
        logger.info(f"Found {len(binaries)} binaries in {build_dir}")
        
        # Take a subset of binaries if there are too many
        if len(binaries) > 10:
            binaries = random.sample(binaries, 10)
            logger.info(f"Selected 10 random binaries for processing")
        
        # Extract functions from each binary
        all_functions = []
        for binary in binaries:
            # Extract functions
            functions = extract_functions_from_binary((project, architecture, compiler, opt_level, binary))
            all_functions.extend(functions)
        
        # Select a subset of functions if there are too many
        if len(all_functions) > config.FUNCTIONS_PER_BINARY:
            all_functions = random.sample(all_functions, config.FUNCTIONS_PER_BINARY)
        
        # Save functions to file
        project_name = f"{project['name']}-{project['version']}"
        output_path = os.path.join(
            config.FUNCTION_DIR,
            project_name,
            architecture,
            compiler,
            opt_level,
            "functions.json"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save functions to file
        save_json(all_functions, output_path)
        
        logger.info(f"Saved {len(all_functions)} functions to {output_path}")
        return all_functions
    
    except Exception as e:
        logger.error(f"Error extracting functions from project {project['name']}: {e}")
        return []

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Extract functions from compiled binaries')
    parser.add_argument('--parallel', type=int, default=4, help='Number of parallel extractions')
    parser.add_argument('--project', type=str, default=None, help='Specific project to extract functions from')
    parser.add_argument('--arch', type=str, default=None, help='Specific architecture to extract functions from')
    parser.add_argument('--compiler', type=str, default=None, help='Specific compiler to extract functions from')
    parser.add_argument('--opt-level', type=str, default=None, help='Specific optimization level to extract functions from')
    
    args = parser.parse_args()
    
    # Create necessary directories
    create_directories()
    
    # Prepare extraction configurations
    configs = []
    
    # Filter projects based on command line arguments
    projects = [p for p in config.PROJECTS if args.project is None or p['name'] == args.project]
    if args.project and not projects:
        logger.error(f"Project {args.project} not found")
        return
    
    # Filter architectures
    architectures = [a for a in config.ARCHITECTURES.keys() if args.arch is None or a == args.arch]
    if args.arch and args.arch not in config.ARCHITECTURES:
        logger.error(f"Architecture {args.arch} not supported")
        return
    
    # Filter compilers
    compilers = [c for c in config.COMPILERS.keys() if args.compiler is None or c == args.compiler]
    if args.compiler and args.compiler not in config.COMPILERS:
        logger.error(f"Compiler {args.compiler} not supported")
        return
    
    # Filter optimization levels
    opt_levels = [o for o in config.OPTIMIZATION_LEVELS if args.opt_level is None or o == args.opt_level]
    if args.opt_level and args.opt_level not in config.OPTIMIZATION_LEVELS:
        logger.error(f"Optimization level {args.opt_level} not supported")
        return
    
    # Generate all combinations
    for project in projects:
        for architecture in architectures:
            for compiler in compilers:
                for opt_level in opt_levels:
                    configs.append((project, architecture, compiler, opt_level))
    
    logger.info(f"Total configurations to extract functions from: {len(configs)}")
    
    # Extract functions in parallel
    with Pool(processes=args.parallel) as pool:
        results = list(pool.imap_unordered(extract_functions_from_project, configs))
    
    # Report results
    total_functions = sum(len(r) for r in results)
    logger.info(f"Extraction complete. Total functions extracted: {total_functions}")

if __name__ == "__main__":
    main()