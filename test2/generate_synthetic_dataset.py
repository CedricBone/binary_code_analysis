#!/usr/bin/env python3
"""
Generate a complete synthetic dataset for the experiment.
This allows quick testing of the pipeline without compilation issues.
"""

import os
import json
import argparse
import random
import numpy as np
from pathlib import Path
import config
from utils import logger, create_directories, set_random_seed

def generate_function_with_optimization_level(func_id, architecture, opt_level):
    """Generate a synthetic function with characteristics of a specific optimization level"""
    
    # Dictionary to store architecture-specific settings
    arch_settings = {
        "x86_64": {
            "registers": ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rsp", "rbp", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"],
            "prologue": ["push rbp", "mov rbp, rsp", "sub rsp, {stack_frame}"],
            "epilogue": ["leave", "ret"],
            "instr_templates": {
                "mem": ["mov {reg1}, [{reg2}+{offset}]", "mov [{reg1}+{offset}], {reg2}"],
                "arith": ["add {reg1}, {reg2}", "sub {reg1}, {reg2}", "imul {reg1}, {reg2}"],
                "logic": ["and {reg1}, {reg2}", "or {reg1}, {reg2}", "xor {reg1}, {reg2}"],
                "shift": ["shl {reg1}, {imm}", "shr {reg1}, {imm}"],
                "cflow": ["cmp {reg1}, {reg2}", "je label_{label}", "jne label_{label}", "jmp label_{label}"]
            }
        },
        "arm": {
            "registers": ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "sp", "lr", "pc"],
            "prologue": ["push {r4, r5, r6, r7, lr}", "sub sp, sp, #{stack_frame}"],
            "epilogue": ["add sp, sp, #{stack_frame}", "pop {r4, r5, r6, r7, pc}"],
            "instr_templates": {
                "mem": ["ldr {reg1}, [{reg2}, #{offset}]", "str {reg1}, [{reg2}, #{offset}]"],
                "arith": ["add {reg1}, {reg2}, {reg3}", "sub {reg1}, {reg2}, {reg3}", "mul {reg1}, {reg2}, {reg3}"],
                "logic": ["and {reg1}, {reg2}, {reg3}", "orr {reg1}, {reg2}, {reg3}", "eor {reg1}, {reg2}, {reg3}"],
                "shift": ["lsl {reg1}, {reg2}, #{imm}", "lsr {reg1}, {reg2}, #{imm}"],
                "cflow": ["cmp {reg1}, {reg2}", "beq label_{label}", "bne label_{label}", "b label_{label}"]
            }
        },
        "mips": {
            "registers": ["$zero", "$at", "$v0", "$v1", "$a0", "$a1", "$a2", "$a3", "$t0", "$t1", "$t2", "$t3", "$s0", "$s1", "$ra", "$sp"],
            "prologue": ["addiu $sp, $sp, -{stack_frame}", "sw $ra, {stack_frame}-4($sp)", "sw $fp, {stack_frame}-8($sp)"],
            "epilogue": ["lw $ra, {stack_frame}-4($sp)", "lw $fp, {stack_frame}-8($sp)", "addiu $sp, $sp, {stack_frame}", "jr $ra"],
            "instr_templates": {
                "mem": ["lw {reg1}, {offset}({reg2})", "sw {reg1}, {offset}({reg2})"],
                "arith": ["addu {reg1}, {reg2}, {reg3}", "subu {reg1}, {reg2}, {reg3}", "mul {reg1}, {reg2}, {reg3}"],
                "logic": ["and {reg1}, {reg2}, {reg3}", "or {reg1}, {reg2}, {reg3}", "xor {reg1}, {reg2}, {reg3}"],
                "shift": ["sll {reg1}, {reg2}, {imm}", "srl {reg1}, {reg2}, {imm}"],
                "cflow": ["beq {reg1}, {reg2}, label_{label}", "bne {reg1}, {reg2}, label_{label}", "j label_{label}"]
            }
        }
    }
    
    # Default to x86_64 if architecture not found
    if architecture not in arch_settings:
        architecture = "x86_64"
    
    settings = arch_settings[architecture]
    
    # Function name based on ID
    func_name = f"func_{func_id:04d}"
    
    # Optimization level characteristics
    opt_characteristics = {
        "O0": {
            "min_length": 30, 
            "max_length": 50,
            "instr_distribution": {"mem": 0.3, "arith": 0.3, "logic": 0.2, "shift": 0.1, "cflow": 0.1},
            "max_labels": 6,
            "redundant_code": True
        },
        "O1": {
            "min_length": 25, 
            "max_length": 40,
            "instr_distribution": {"mem": 0.25, "arith": 0.3, "logic": 0.2, "shift": 0.15, "cflow": 0.1},
            "max_labels": 5,
            "redundant_code": False
        },
        "O2": {
            "min_length": 20, 
            "max_length": 35,
            "instr_distribution": {"mem": 0.2, "arith": 0.35, "logic": 0.2, "shift": 0.15, "cflow": 0.1},
            "max_labels": 4,
            "redundant_code": False
        },
        "O3": {
            "min_length": 15, 
            "max_length": 30,
            "instr_distribution": {"mem": 0.15, "arith": 0.4, "logic": 0.2, "shift": 0.15, "cflow": 0.1},
            "max_labels": 3,
            "redundant_code": False
        }
    }
    
    characteristics = opt_characteristics[opt_level]
    
    # Generate random body length
    body_length = random.randint(characteristics["min_length"], characteristics["max_length"])
    
    # Generate stack frame size (multiple of 16)
    stack_frame = random.randint(2, 8) * 16
    
    # Generate random labels
    num_labels = random.randint(1, characteristics["max_labels"])
    labels = [f"{random.randint(1, 999)}" for _ in range(num_labels)]
    
    # Process prologue - insert stack frame size
    prologue = [p.replace("{stack_frame}", str(stack_frame)) for p in settings["prologue"]]
    
    # Process epilogue - insert stack frame size
    epilogue = [e.replace("{stack_frame}", str(stack_frame)) for e in settings["epilogue"]]
    
    # Generate instructions based on optimization characteristics
    body = []
    for _ in range(body_length):
        # Select instruction type based on distribution
        instr_type = random.choices(
            list(characteristics["instr_distribution"].keys()),
            weights=list(characteristics["instr_distribution"].values()),
            k=1
        )[0]
        
        # Select an instruction template of that type
        template = random.choice(settings["instr_templates"][instr_type])
        
        # Fill in template with appropriate values
        instruction = template
        if "{reg1}" in instruction:
            instruction = instruction.replace("{reg1}", random.choice(settings["registers"]))
        if "{reg2}" in instruction:
            instruction = instruction.replace("{reg2}", random.choice(settings["registers"]))
        if "{reg3}" in instruction:
            instruction = instruction.replace("{reg3}", random.choice(settings["registers"]))
        if "{offset}" in instruction:
            instruction = instruction.replace("{offset}", str(random.randint(0, 128)))
        if "{imm}" in instruction:
            instruction = instruction.replace("{imm}", str(random.randint(1, 16)))
        if "{label}" in instruction:
            instruction = instruction.replace("{label}", random.choice(labels))
        
        body.append(instruction)
    
    # Add label definitions to the body
    label_definitions = []
    for label in labels:
        label_definitions.append(f"label_{label}:")
    
    # Randomly insert label definitions throughout the body
    for label_def in label_definitions:
        insert_pos = random.randint(0, len(body))
        body.insert(insert_pos, label_def)
    
    # Add redundant code if O0
    if characteristics["redundant_code"]:
        redundant_code = [
            f"mov {random.choice(settings['registers'])}, {random.choice(settings['registers'])}",
            f"push {random.choice(settings['registers'])}",
            f"pop {random.choice(settings['registers'])}"
        ]
        
        # Insert 3-5 redundant instructions
        for _ in range(random.randint(3, 5)):
            insert_pos = random.randint(0, len(body))
            body.insert(insert_pos, random.choice(redundant_code))
    
    # Combine prologue, body, and epilogue
    instructions = prologue + body + epilogue
    
    # Create function object
    function = {
        "name": func_name,
        "instructions": instructions,
        "raw": "\n".join(instructions),
        "synthetic": True,
        "architecture": architecture,
        "optimization": opt_level
    }
    
    return function

def create_synthetic_dataset(output_dir, num_functions=200, architectures=None, compilers=None, opt_levels=None):
    """Create a complete synthetic dataset for the experiment"""
    if architectures is None:
        architectures = list(config.ARCHITECTURES.keys())
    
    if compilers is None:
        compilers = list(config.COMPILERS.keys())
    
    if opt_levels is None:
        opt_levels = config.OPTIMIZATION_LEVELS
    
    # Set random seed for reproducibility
    set_random_seed()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track all generated functions
    all_functions = {}
    
    # Generate functions for each architecture, compiler, and optimization level
    for architecture in architectures:
        for compiler in compilers:
            for opt_level in opt_levels:
                # Create a unique ID for this configuration
                config_id = f"{architecture}_{compiler}_{opt_level}"
                logger.info(f"Generating synthetic data for {config_id}")
                
                # Generate functions
                functions = []
                for i in range(num_functions):
                    function = generate_function_with_optimization_level(i, architecture, opt_level)
                    functions.append(function)
                
                # Save functions
                config_dir = os.path.join(output_dir, architecture, compiler, opt_level)
                os.makedirs(config_dir, exist_ok=True)
                
                with open(os.path.join(config_dir, "functions.json"), 'w') as f:
                    json.dump(functions, f, indent=2)
                
                # Store functions in memory
                all_functions[config_id] = functions
                
                logger.info(f"Generated {len(functions)} functions for {config_id}")
    
    # Create a metadata file with details about the dataset
    metadata = {
        "num_functions": num_functions,
        "architectures": architectures,
        "compilers": compilers,
        "optimization_levels": opt_levels,
        "configurations": list(all_functions.keys()),
        "total_functions": sum(len(funcs) for funcs in all_functions.values())
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Synthetic dataset created with {metadata['total_functions']} total functions")
    logger.info(f"Dataset saved to {output_dir}")
    
    return all_functions

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset for the experiment')
    parser.add_argument('--output-dir', type=str, default=config.FUNCTION_DIR,
                        help='Output directory for the synthetic dataset')
    parser.add_argument('--num-functions', type=int, default=200,
                        help='Number of functions to generate for each configuration')
    parser.add_argument('--architectures', type=str, default=None,
                        help='Comma-separated list of architectures to generate data for')
    parser.add_argument('--compilers', type=str, default=None,
                        help='Comma-separated list of compilers to generate data for')
    parser.add_argument('--opt-levels', type=str, default=None,
                        help='Comma-separated list of optimization levels to generate data for')
    
    args = parser.parse_args()
    
    # Parse arguments
    architectures = args.architectures.split(',') if args.architectures else None
    compilers = args.compilers.split(',') if args.compilers else None
    opt_levels = args.opt_levels.split(',') if args.opt_levels else None
    
    # Create directories
    create_directories()
    
    # Create synthetic dataset
    create_synthetic_dataset(
        args.output_dir,
        num_functions=args.num_functions,
        architectures=architectures,
        compilers=compilers,
        opt_levels=opt_levels
    )

if __name__ == "__main__":
    main()