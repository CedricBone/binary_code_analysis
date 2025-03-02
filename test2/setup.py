#!/usr/bin/env python3
"""
Setup script for the Compiler Optimization Impact Analysis experiment.
Creates necessary directories and checks dependencies.
"""

import os
import sys
import subprocess
import platform
import argparse

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

def check_python_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        "tensorflow", "sklearn", "matplotlib", "seaborn", 
        "pandas", "requests", "tqdm", "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print_colored(f"Missing Python packages: {', '.join(missing_packages)}", "yellow")
        
        # Ask user if they want to install missing packages
        response = input("Do you want to install the missing packages? (y/n): ")
        if response.lower() == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
                print_colored("Successfully installed missing packages", "green")
            except subprocess.CalledProcessError:
                print_colored("Failed to install packages. Please install them manually.", "red")
                return False
        else:
            print_colored("Please install the missing packages before continuing.", "yellow")
            return False
    
    return True

def check_system_dependencies():
    """Check if required system tools are installed"""
    required_tools = ["gcc", "objdump"]
    
    missing_tools = []
    for tool in required_tools:
        try:
            subprocess.check_call(["which", tool], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            missing_tools.append(tool)
    
    if missing_tools:
        print_colored(f"Missing system tools: {', '.join(missing_tools)}", "red")
        print_colored("Please install them before continuing.", "yellow")
        return False
    
    return True

def create_directories():
    """Create necessary directories for the experiment"""
    directories = [
        "build", "downloads", "functions", "models", "results", "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print_colored(f"Created directory: {directory}", "green")

def main():
    parser = argparse.ArgumentParser(description='Setup the experiment environment')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency checks')
    
    args = parser.parse_args()
    
    print_colored("Setting up Compiler Optimization Impact Analysis experiment...", "blue")
    
    if not args.skip_checks:
        # Check Python dependencies
        print_colored("Checking Python dependencies...", "blue")
        if not check_python_dependencies():
            return
        
        # Check system dependencies
        print_colored("Checking system dependencies...", "blue")
        if not check_system_dependencies():
            return
    
    # Create directories
    print_colored("Creating necessary directories...", "blue")
    create_directories()
    
    # Check platform
    print_colored(f"Detected platform: {platform.machine()}", "blue")
    
    print_colored("Setup complete!", "green")
    print_colored("\nYou can now run the experiment with:", "blue")
    print_colored("python run_experiment.py", "green")

if __name__ == "__main__":
    main()