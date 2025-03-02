"""
Utility functions for Compiler Optimization Impact Analysis Experiment.
"""

import os
import subprocess
import tarfile
import logging
import time
import requests
import zipfile
import shutil
import sys
import platform
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import config
import re
from pathlib import Path

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file = os.path.join(config.LOG_DIR, f"experiment_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Create logger
logger = setup_logging()

def set_random_seed(seed=config.RANDOM_SEED):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_directories():
    """Create necessary directories for the experiment"""
    directories = [
        config.BUILD_DIR,
        config.DOWNLOAD_DIR,
        config.FUNCTION_DIR,
        config.MODEL_DIR,
        config.RESULTS_DIR,
        config.LOG_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def check_command_exists(command):
    """Check if a command exists in the system"""
    try:
        subprocess.run(
            ["which", command],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError:
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []
    
    # Check compilers
    for compiler_name, compiler_info in config.COMPILERS.items():
        if not check_command_exists(compiler_info["command"]):
            missing_deps.append(compiler_info["command"])
    
    # Check cross-compilers and disassemblers for each architecture
    for arch_name, arch_info in config.ARCHITECTURES.items():
        if arch_name != "x86_64":  # Native architecture likely doesn't need cross-compiler
            if not check_command_exists(arch_info["disassembler"]):
                missing_deps.append(arch_info["disassembler"])
    
    # Check basic build tools
    for tool in ["make", "tar", "wget", "curl"]:
        if not check_command_exists(tool):
            missing_deps.append(tool)
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Please install these dependencies before running the experiment.")
        return False
    
    logger.info("All required dependencies are installed.")
    return True

def run_command(command, cwd=None, env=None):
    """Run a shell command and return its output"""
    logger.debug(f"Running command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env
        )
        return result.stdout.decode('utf-8', errors='replace').strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command}")
        logger.error(f"Error output: {e.stderr.decode('utf-8', errors='replace')}")
        raise

def download_file(url, output_path):
    """Download a file from URL to output_path with progress bar"""
    logger.info(f"Downloading {url} to {output_path}")
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(output_path):
        logger.info(f"File already exists: {output_path}")
        return output_path
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True) as progress_bar:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
    
    return output_path

def extract_archive(archive_path, output_dir):
    """Extract an archive file to the output directory"""
    logger.info(f"Extracting {archive_path} to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract based on file extension
    if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)
    elif archive_path.endswith('.tar.bz2'):
        with tarfile.open(archive_path, 'r:bz2') as tar:
            tar.extractall(path=output_dir)
    elif archive_path.endswith('.tar.xz'):
        with tarfile.open(archive_path, 'r:xz') as tar:
            tar.extractall(path=output_dir)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    else:
        logger.error(f"Unsupported archive format: {archive_path}")
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    # Return the directory containing the extracted files
    # Typically, archives contain a single top-level directory
    extracted_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    if len(extracted_dirs) == 1:
        return os.path.join(output_dir, extracted_dirs[0])
    else:
        return output_dir

def download_project(project):
    """Download and extract a project"""
    project_name = f"{project['name']}-{project['version']}"
    download_path = os.path.join(config.DOWNLOAD_DIR, os.path.basename(project['url']))
    extract_path = os.path.join(config.DOWNLOAD_DIR, project_name)
    
    # Check if already extracted
    if os.path.exists(extract_path):
        logger.info(f"Project already extracted: {extract_path}")
        return extract_path
    
    # Download
    download_file(project['url'], download_path)
    
    # Extract
    extracted_dir = extract_archive(download_path, config.DOWNLOAD_DIR)
    
    # Rename if needed to ensure consistency
    if os.path.basename(extracted_dir) != project_name:
        os.rename(extracted_dir, extract_path)
    
    return extract_path

def get_build_path(project, architecture, compiler, opt_level):
    """Get the build directory path for a specific configuration"""
    project_name = f"{project['name']}-{project['version']}"
    return os.path.join(
        config.BUILD_DIR,
        project_name,
        architecture,
        compiler,
        opt_level
    )

def get_compiler_command(compiler_name, architecture, opt_level):
    """Get the compiler command for a specific architecture"""
    compiler_info = config.COMPILERS[compiler_name]
    arch_info = config.ARCHITECTURES[architecture]
    
    # For x86_64, use native compiler
    if architecture == "x86_64":
        compiler_cmd = compiler_info["command"]
    else:
        # For cross-compilation, use appropriate triple
        if compiler_name == "gcc":
            compiler_cmd = compiler_info["cross_format"].format(triple=arch_info["triple"])
        else:  # clang uses -target instead
            compiler_cmd = compiler_info["command"]
    
    # Add architecture-specific flags
    if compiler_name == "gcc":
        arch_flags = arch_info["gcc_flags"]
    else:
        arch_flags = arch_info["clang_flags"]
    
    # Add optimization flags
    opt_flags = f"-{opt_level}"
    
    # Combine everything
    full_command = f"{compiler_cmd} {arch_flags} {opt_flags}"
    return full_command

def get_disassembler_command(architecture, binary_path, output_path):
    """Get the disassembler command for a specific architecture"""
    arch_info = config.ARCHITECTURES[architecture]
    disassembler = arch_info["disassembler"]
    flags = arch_info["disassembler_flags"]
    
    command = f"{disassembler} {flags} {binary_path} > {output_path}"
    return command

def parse_function(func_text, architecture):
    """Parse a function from disassembled text"""
    lines = func_text.strip().split('\n')
    instructions = []
    
    # Different parsing logic based on architecture
    if architecture == "x86_64":
        pattern = r'^\s*[0-9a-f]+:\s+([0-9a-f\s]+)\s+(.+)$'
        for line in lines:
            match = re.match(pattern, line)
            if match:
                byte_code = match.group(1).strip()
                instruction = match.group(2).strip()
                instructions.append(instruction)
    else:
        # Generic pattern for other architectures
        pattern = r'^\s*[0-9a-f]+:\s+([0-9a-f\s]+)\s+(.+)$'
        for line in lines:
            match = re.match(pattern, line)
            if match and not line.startswith("Disassembly"):
                instruction = match.group(2).strip()
                instructions.append(instruction)
    
    return instructions

def normalize_instruction(instruction, architecture):
    """Normalize an instruction for embedding"""
    # Remove comments (typically after # or ;)
    instruction = re.sub(r'[#;].*$', '', instruction)
    
    # Remove memory addresses
    instruction = re.sub(r'0x[0-9a-f]+', '0xADDR', instruction)
    
    # Replace immediate values with a placeholder
    instruction = re.sub(r'\s+[0-9]+', ' IMM', instruction)
    
    # Remove multiple spaces
    instruction = re.sub(r'\s+', ' ', instruction).strip()
    
    return instruction

def save_json(data, output_path):
    """Save data as JSON"""
    import json
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(input_path):
    """Load data from JSON"""
    import json
    with open(input_path, 'r') as f:
        return json.load(f)

def plot_heatmap(data, x_labels, y_labels, title, output_path):
    """Plot a heatmap and save it to output_path"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=x_labels,
        yticklabels=y_labels
    )
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()