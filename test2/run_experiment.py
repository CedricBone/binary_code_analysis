#!/usr/bin/env python3
"""
Main script to run the Compiler Optimization Impact Analysis experiment.
This script coordinates the entire pipeline from data generation to analysis.
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"== {text}")
    print("=" * 80 + "\n")

def print_colored(text, color="blue"):
    """Print colored text to the console"""
    colors = {
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "end": "\033[0m"
    }
    
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def run_command(cmd, exit_on_error=True):
    """Run a shell command and handle errors"""
    print_colored(f"Running: {cmd}", "blue")
    
    try:
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Print output in real-time
        while True:
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()
            
            if stdout_line == '' and stderr_line == '' and process.poll() is not None:
                break
            
            if stdout_line:
                print(stdout_line.strip())
            if stderr_line:
                print_colored(stderr_line.strip(), "yellow")
        
        # Get return code
        return_code = process.poll()
        
        if return_code != 0:
            print_colored(f"Command failed with return code {return_code}", "red")
            if exit_on_error:
                sys.exit(return_code)
            return False
        
        return True
    
    except Exception as e:
        print_colored(f"Command execution failed: {e}", "red")
        if exit_on_error:
            sys.exit(1)
        return False

def check_file_exists(file_path):
    """Check if a file exists and print status"""
    if os.path.exists(file_path):
        print_colored(f"Found: {file_path}", "green")
        return True
    else:
        print_colored(f"File not found: {file_path}", "red")
        return False

def generate_synthetic_data(args):
    """Generate synthetic dataset for all architectures and optimization levels"""
    print_header("STEP 1: Generating Synthetic Dataset")
    
    cmd = f"python generate_synthetic_dataset.py --num-functions {args.num_functions}"
    
    if args.architectures:
        cmd += f" --architectures {args.architectures}"
    
    if args.compilers:
        cmd += f" --compilers {args.compilers}"
        
    if args.opt_levels:
        cmd += f" --opt-levels {args.opt_levels}"
    
    return run_command(cmd)

def train_model(source_arch, source_compiler, source_opt, args):
    """Train a model on a specific configuration"""
    print_header(f"STEP 2: Training Model - {source_arch}/{source_compiler}/{source_opt}")
    
    cmd = f"python train_model.py --source-arch {source_arch} --source-compiler {source_compiler} --source-opt {source_opt}"
    
    # Add additional parameters
    cmd += f" --epochs {args.epochs} --batch-size {args.batch_size} --use-synthetic-data"
    
    return run_command(cmd)

def test_model(source_arch, source_compiler, source_opt, args):
    """Test a model on all configurations"""
    print_header(f"STEP 3: Testing Model - {source_arch}/{source_compiler}/{source_opt}")
    
    cmd = f"python test_model.py --source-arch {source_arch} --source-compiler {source_compiler} --source-opt {source_opt}"
    
    cmd += f" --parallel {args.parallel} --use-synthetic-data"
    
    if args.target_arch:
        cmd += f" --target-arch {args.target_arch}"
    
    if args.target_compiler:
        cmd += f" --target-compiler {args.target_compiler}"
        
    if args.target_opt:
        cmd += f" --target-opt {args.target_opt}"
    
    return run_command(cmd)

def analyze_results(source_arch, source_compiler, source_opt):
    """Analyze results for a specific model"""
    print_header(f"STEP 4: Analyzing Results - {source_arch}/{source_compiler}/{source_opt}")
    
    cmd = f"python analyze_results.py --source-arch {source_arch} --source-compiler {source_compiler} --source-opt {source_opt}"
    
    return run_command(cmd)

def analyze_transfer(args):
    """Run transfer analysis across all models"""
    print_header("STEP 5: Analyzing Optimization Transfer")
    
    # Run transfer analysis for each architecture/compiler pair
    architectures = args.architectures.split(',') if args.architectures else ["x86_64"]
    compilers = args.compilers.split(',') if args.compilers else ["gcc"]
    
    success = True
    for arch in architectures:
        for compiler in compilers:
            cmd = f"python analyze_transfer.py --architecture {arch} --compiler {compiler}"
            success = success and run_command(cmd, exit_on_error=False)
    
    return success

def generate_report():
    """Generate final report combining all results"""
    print_header("STEP 6: Generating Final Report")
    
    # Check if we have transfer analysis results
    transfer_dir = "results/transfer_analysis"
    if not os.path.exists(transfer_dir):
        print_colored("Transfer analysis results not found. Skipping report generation.", "yellow")
        return False
    
    # Check for study findings document
    findings_path = os.path.join(transfer_dir, "study_findings.md")
    if not os.path.exists(findings_path):
        print_colored("Study findings document not found. Skipping report generation.", "yellow")
        return False
    
    # Create final report directory
    report_dir = "results/final_report"
    os.makedirs(report_dir, exist_ok=True)
    
    # Create report file
    report_path = os.path.join(report_dir, "optimization_impact_report.md")
    
    with open(report_path, 'w') as report_file, open(findings_path, 'r') as findings_file:
        # Write report header
        report_file.write("# Compiler Optimization Impact Analysis Report\n\n")
        report_file.write("## Executive Summary\n\n")
        report_file.write("This report presents the findings of a study on how compiler optimization levels\n")
        report_file.write("affect binary code similarity detection across different architectures and compilers.\n\n")
        
        # Add timestamp
        report_file.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Add study findings
        report_file.write("## Key Findings\n\n")
        findings_content = findings_file.read()
        report_file.write(findings_content)
        
        # Add references to visualizations
        report_file.write("\n\n## Visualizations\n\n")
        report_file.write("The following visualizations provide detailed insights into optimization transfer patterns:\n\n")
        
        # List visualizations
        for root, _, files in os.walk(transfer_dir):
            for file in files:
                if file.endswith('.png'):
                    rel_path = os.path.join(os.path.relpath(root, "results"), file)
                    name = os.path.splitext(file)[0].replace('_', ' ').title()
                    report_file.write(f"- [{name}](../{rel_path})\n")
    
    print_colored(f"Final report generated: {report_path}", "green")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run the Compiler Optimization Impact Analysis experiment')
    
    # Data generation parameters
    parser.add_argument('--num-functions', type=int, default=200,
                        help='Number of functions to generate for each configuration')
    parser.add_argument('--architectures', type=str, default=None,
                        help='Comma-separated list of architectures to include')
    parser.add_argument('--compilers', type=str, default=None,
                        help='Comma-separated list of compilers to include')
    parser.add_argument('--opt-levels', type=str, default=None,
                        help='Comma-separated list of optimization levels to include')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    
    # Testing parameters
    parser.add_argument('--parallel', type=int, default=4,
                        help='Number of parallel tests')
    parser.add_argument('--target-arch', type=str, default=None,
                        help='Specific target architecture to test')
    parser.add_argument('--target-compiler', type=str, default=None,
                        help='Specific target compiler to test')
    parser.add_argument('--target-opt', type=str, default=None,
                        help='Specific target optimization level to test')
    
    # Execution control
    parser.add_argument('--skip-data-generation', action='store_true',
                        help='Skip synthetic data generation step')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training step')
    parser.add_argument('--skip-testing', action='store_true',
                        help='Skip model testing step')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip result analysis step')
    parser.add_argument('--skip-transfer-analysis', action='store_true',
                        help='Skip transfer analysis step')
    parser.add_argument('--opt-levels-to-run', type=str, default="O0,O1,O2,O3",
                        help='Comma-separated list of optimization levels to run experiments for')
    
    args = parser.parse_args()
    
    # Set default architectures and compilers if not specified
    if not args.architectures:
        args.architectures = "x86_64,arm,mips,powerpc"
    
    if not args.compilers:
        args.compilers = "gcc,clang"
    
    if not args.opt_levels:
        args.opt_levels = "O0,O1,O2,O3"
    
    # Parse optimization levels to run experiments for
    opt_levels_to_run = args.opt_levels_to_run.split(',')
    
    # Record start time
    start_time = time.time()
    
    # Step 1: Generate synthetic data
    if not args.skip_data_generation:
        generate_synthetic_data(args)
    else:
        print_colored("Skipping synthetic data generation", "yellow")
    
    # Source configuration
    source_arch = args.architectures.split(',')[0]
    source_compiler = args.compilers.split(',')[0]
    
    # Step 2-4: For each optimization level, train, test, and analyze
    for opt_level in opt_levels_to_run:
        if not args.skip_training:
            train_model(source_arch, source_compiler, opt_level, args)
        else:
            print_colored(f"Skipping training for {source_arch}/{source_compiler}/{opt_level}", "yellow")
        
        if not args.skip_testing:
            test_model(source_arch, source_compiler, opt_level, args)
        else:
            print_colored(f"Skipping testing for {source_arch}/{source_compiler}/{opt_level}", "yellow")
        
        if not args.skip_analysis:
            analyze_results(source_arch, source_compiler, opt_level)
        else:
            print_colored(f"Skipping analysis for {source_arch}/{source_compiler}/{opt_level}", "yellow")
    
    # Step 5: Analyze transfer learning across optimization levels
    if not args.skip_transfer_analysis:
        analyze_transfer(args)
    else:
        print_colored("Skipping transfer analysis", "yellow")
    
    # Step 6: Generate final report
    generate_report()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print_header("Experiment Completed!")
    print_colored(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s", "green")
    print_colored("Results can be found in the 'results' directory", "green")
    print_colored("Final report: results/final_report/optimization_impact_report.md", "green")

if __name__ == "__main__":
    main()