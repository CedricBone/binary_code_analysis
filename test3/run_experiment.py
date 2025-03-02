#!/usr/bin/env python3
"""
Main script to run the complete robustness assessment experiment.
"""
import os
import argparse
import logging
import subprocess
import glob
import shutil
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Runs the complete robustness assessment experiment."""
    
    def __init__(self, base_dir: str):
        """
        Initialize the experiment runner.
        
        Args:
            base_dir: Base directory for the experiment
        """
        self.base_dir = os.path.abspath(base_dir)
        self.scripts_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create directory structure
        self.source_dir = os.path.join(self.base_dir, "source")
        self.binary_dir = os.path.join(self.base_dir, "binaries")
        self.features_dir = os.path.join(self.base_dir, "features")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.results_dir = os.path.join(self.base_dir, "results")
        
        for directory in [self.source_dir, self.binary_dir, self.features_dir, 
                         self.models_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def setup_sample_programs(self):
        """Set up sample C programs for testing."""
        # Sample programs
        samples = {
            "factorial": '''
                #include <stdio.h>
                
                int factorial(int n) {
                    if (n <= 1) return 1;
                    return n * factorial(n - 1);
                }
                
                int main(int argc, char **argv) {
                    int i;
                    for (i = 1; i <= 10; i++) {
                        printf("Factorial of %d is %d\\n", i, factorial(i));
                    }
                    return 0;
                }
            ''',
            
            "fibonacci": '''
                #include <stdio.h>
                
                int fibonacci(int n) {
                    if (n <= 0) return 0;
                    if (n == 1) return 1;
                    return fibonacci(n-1) + fibonacci(n-2);
                }
                
                int main(int argc, char **argv) {
                    int i;
                    for (i = 0; i < 15; i++) {
                        printf("Fibonacci of %d is %d\\n", i, fibonacci(i));
                    }
                    return 0;
                }
            ''',
            
            "sort": '''
                #include <stdio.h>
                #include <stdlib.h>
                
                void bubble_sort(int arr[], int n) {
                    int i, j, temp;
                    for (i = 0; i < n-1; i++) {
                        for (j = 0; j < n-i-1; j++) {
                            if (arr[j] > arr[j+1]) {
                                temp = arr[j];
                                arr[j] = arr[j+1];
                                arr[j+1] = temp;
                            }
                        }
                    }
                }
                
                void print_array(int arr[], int size) {
                    int i;
                    for (i = 0; i < size; i++)
                        printf("%d ", arr[i]);
                    printf("\\n");
                }
                
                int main() {
                    int arr[] = {64, 34, 25, 12, 22, 11, 90};
                    int n = sizeof(arr)/sizeof(arr[0]);
                    
                    printf("Original array: ");
                    print_array(arr, n);
                    
                    bubble_sort(arr, n);
                    
                    printf("Sorted array: ");
                    print_array(arr, n);
                    return 0;
                }
            ''',
            
            "prime": '''
                #include <stdio.h>
                #include <stdbool.h>
                
                bool is_prime(int n) {
                    if (n <= 1) return false;
                    if (n <= 3) return true;
                    
                    if (n % 2 == 0 || n % 3 == 0) return false;
                    
                    for (int i = 5; i * i <= n; i += 6) {
                        if (n % i == 0 || n % (i + 2) == 0)
                            return false;
                    }
                    
                    return true;
                }
                
                int main() {
                    int count = 0;
                    printf("Prime numbers between 1 and 100: ");
                    
                    for (int i = 1; i <= 100; i++) {
                        if (is_prime(i)) {
                            printf("%d ", i);
                            count++;
                        }
                    }
                    
                    printf("\\nTotal: %d prime numbers\\n", count);
                    return 0;
                }
            '''
        }
        
        # Create source files
        for name, code in samples.items():
            source_path = os.path.join(self.source_dir, f"{name}.c")
            with open(source_path, 'w') as f:
                f.write(code)
            logger.info(f"Created sample program: {source_path}")
    
    def generate_binaries(self, obfuscation_techniques: List[str]):
        """
        Generate obfuscated binaries from source code.
        
        Args:
            obfuscation_techniques: List of obfuscation techniques to apply
        """
        logger.info("Generating binaries with various obfuscation techniques...")
        
        obfuscator_script = os.path.join(self.scripts_dir, "obfuscator.py")
        
        # Process each source file
        for source_file in glob.glob(os.path.join(self.source_dir, "*.c")):
            program_name = os.path.splitext(os.path.basename(source_file))[0]
            output_dir = os.path.join(self.binary_dir, program_name)
            
            # Run obfuscator
            cmd = [
                "python3", obfuscator_script,
                "--input", source_file,
                "--output-dir", output_dir,
                "--techniques"
            ] + obfuscation_techniques
            
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
    
    def extract_features(self):
        """Extract features from all generated binaries."""
        logger.info("Extracting features from binaries...")
        
        feature_extractor_script = os.path.join(self.scripts_dir, "feature_extractor.py")
        
        # Process each binary
        for binary_dir in glob.glob(os.path.join(self.binary_dir, "*")):
            program_name = os.path.basename(binary_dir)
            
            for binary_file in glob.glob(os.path.join(binary_dir, "*")):
                if os.path.isfile(binary_file) and os.access(binary_file, os.X_OK):
                    # Get technique name from binary filename
                    binary_name = os.path.basename(binary_file)
                    technique = binary_name.replace(program_name + "_", "")
                    
                    # Extract features
                    output_file = os.path.join(self.features_dir, f"{program_name}_{technique}.json")
                    
                    cmd = [
                        "python3", feature_extractor_script,
                        "--input", binary_file,
                        "--output", output_file
                    ]
                    
                    logger.info(f"Running: {' '.join(cmd)}")
                    try:
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Error extracting features from {binary_file}: {e}")
    
    def train_models(self):
        """Train binary similarity models."""
        logger.info("Training binary similarity models...")
        
        model_manager_script = os.path.join(self.scripts_dir, "model_manager.py")
        
        cmd = [
            "python3", model_manager_script,
            "--train",
            "--features-dir", self.features_dir,
            "--model-dir", self.models_dir
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    
    def evaluate_robustness(self):
        """Evaluate the robustness of trained models against obfuscation."""
        logger.info("Evaluating model robustness...")
        
        evaluator_script = os.path.join(self.scripts_dir, "evaluator.py")
        
        cmd = [
            "python3", evaluator_script,
            "--features-dir", self.features_dir,
            "--models-dir", self.models_dir,
            "--output-dir", self.results_dir,
            "--baseline", "original"
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    
    def run_experiment(self, obfuscation_techniques: List[str], skip_steps: List[str] = None):
        """
        Run the complete experiment workflow.
        
        Args:
            obfuscation_techniques: List of obfuscation techniques to apply
            skip_steps: List of steps to skip (setup, binaries, features, models, evaluate)
        """
        if skip_steps is None:
            skip_steps = []
            
        logger.info("Starting robustness assessment experiment...")
        
        # Setup sample programs
        if "setup" not in skip_steps:
            self.setup_sample_programs()
        else:
            logger.info("Skipping setup step")
        
        # Generate binaries with obfuscation
        if "binaries" not in skip_steps:
            self.generate_binaries(obfuscation_techniques)
        else:
            logger.info("Skipping binary generation step")
        
        # Extract features
        if "features" not in skip_steps:
            self.extract_features()
        else:
            logger.info("Skipping feature extraction step")
        
        # Train models
        if "models" not in skip_steps:
            self.train_models()
        else:
            logger.info("Skipping model training step")
        
        # Evaluate robustness
        if "evaluate" not in skip_steps:
            self.evaluate_robustness()
        else:
            logger.info("Skipping robustness evaluation step")
        
        logger.info(f"Experiment completed! Results are available in {self.results_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run the complete robustness assessment experiment")
    parser.add_argument("--base-dir", default="./experiment", help="Base directory for the experiment")
    parser.add_argument("--techniques", nargs="+", 
                        default=["flatten", "virtualize", "encodeliterals", "ollvm_sub", "ollvm_bcf", "ollvm_fla"],
                        help="Obfuscation techniques to apply")
    parser.add_argument("--skip", nargs="+", choices=["setup", "binaries", "features", "models", "evaluate"],
                        help="Steps to skip")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.base_dir)
    runner.run_experiment(args.techniques, args.skip)

if __name__ == "__main__":
    main()