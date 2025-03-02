#!/usr/bin/env python3
"""
Script to compile projects with different optimization levels and compilers
for multiple architectures.
"""

import os
import shutil
import argparse
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path
import config
from utils import logger, create_directories, download_project, run_command, get_build_path, get_compiler_command, check_dependencies

def prepare_build_environment(source_dir, build_dir):
    """Prepare the build environment by copying source to build directory"""
    # Create build directory
    os.makedirs(build_dir, exist_ok=True)
    
    # Copy source to build directory (if not already done)
    if not os.listdir(build_dir):
        logger.info(f"Copying source from {source_dir} to {build_dir}")
        # Using distutils to copy the directory tree
        shutil.copytree(source_dir, build_dir, dirs_exist_ok=True)
    
    return build_dir

def compile_configuration(args):
    """Compile a specific project with a specific configuration"""
    project, architecture, compiler_name, opt_level = args
    
    project_name = f"{project['name']}-{project['version']}"
    source_dir = download_project(project)
    build_dir = get_build_path(project, architecture, compiler_name, opt_level)
    
    logger.info(f"Compiling {project_name} for {architecture} with {compiler_name} -{opt_level}")
    
    try:
        # Prepare build environment
        build_src_dir = prepare_build_environment(source_dir, build_dir)
        
        # Get compiler command
        compiler_cmd = get_compiler_command(compiler_name, architecture, opt_level)
        
        # Set environment variables for compilation
        env = os.environ.copy()
        env["CC"] = compiler_cmd
        env["CFLAGS"] = f"-{opt_level}"
        
        # Run build command with a timeout (some builds might hang)
        build_command = project['build_command']
        logger.info(f"Running build command: {build_command}")
        
        # Some projects have specific configure flags
        if 'configure_flags' in project and project['configure_flags']:
            build_command = build_command.replace('./configure', f'./configure {project["configure_flags"]}')
        
        result = subprocess.run(
            build_command,
            shell=True,
            cwd=build_src_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Build failed for {project_name} - {architecture} - {compiler_name} - {opt_level}")
            logger.error(f"Error output: {result.stderr.decode('utf-8', errors='replace')}")
            return False
        
        logger.info(f"Successfully compiled {project_name} for {architecture} with {compiler_name} -{opt_level}")
        return True
    
    except subprocess.TimeoutExpired:
        logger.error(f"Build timed out for {project_name} - {architecture} - {compiler_name} - {opt_level}")
        return False
    
    except Exception as e:
        logger.error(f"Error compiling {project_name} for {architecture} with {compiler_name} -{opt_level}: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Compile projects with different configurations')
    parser.add_argument('--parallel', type=int, default=4, help='Number of parallel compilations')
    parser.add_argument('--project', type=str, default=None, help='Specific project to compile')
    parser.add_argument('--arch', type=str, default=None, help='Specific architecture to compile for')
    parser.add_argument('--compiler', type=str, default=None, help='Specific compiler to use')
    parser.add_argument('--opt-level', type=str, default=None, help='Specific optimization level to use')
    
    args = parser.parse_args()
    
    # Create necessary directories
    create_directories()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies. Aborting.")
        return
    
    # Prepare compilation configurations
    configs = []
    
    # Filter projects based on command line arguments
    projects = [p for p in config.PROJECTS if args.project is None or p['name'] == args.project]
    if not projects:
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
    
    logger.info(f"Total configurations to compile: {len(configs)}")
    
    # Compile configurations in parallel
    start_time = time.time()
    with Pool(processes=args.parallel) as pool:
        results = list(pool.imap_unordered(compile_configuration, configs))
    
    # Report results
    successful = results.count(True)
    failed = results.count(False)
    total_time = time.time() - start_time
    
    logger.info(f"Compilation complete. Time taken: {total_time:.2f} seconds")
    logger.info(f"Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logger.warning(f"Some compilations failed. Check logs for details.")

if __name__ == "__main__":
    main()