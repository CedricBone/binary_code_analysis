#!/bin/bash

# Run the Compiler Optimization Impact Analysis Experiment

# Set the number of parallel processes
PARALLEL=4

# Configuration - modify these values as needed
SOURCE_ARCH="x86_64"
SOURCE_COMPILER="gcc"
SOURCE_OPT="O2"
EPOCHS=10
BATCH_SIZE=64

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print section header
print_header() {
    echo -e "\n${GREEN}======================================================${NC}"
    echo -e "${GREEN}== $1${NC}"
    echo -e "${GREEN}======================================================${NC}\n"
}

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed.${NC}"
        return 1
    fi
    return 0
}

# Check requirements
check_requirements() {
    echo -e "${YELLOW}Checking requirements...${NC}"
    
    # Check Python and required packages
    check_command python3 || { echo -e "${RED}Python 3 is required.${NC}"; exit 1; }
    
    # Check if required Python packages are installed
    python3 -c "import tensorflow, numpy, pandas, matplotlib, seaborn, sklearn" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}Required Python packages not found. Installing...${NC}"
        pip install tensorflow scikit-learn matplotlib seaborn pandas requests tqdm
    fi
    
    # Check other required commands
    check_command make || { echo -e "${YELLOW}Warning: 'make' not found. May be required for compilation.${NC}"; }
    check_command objdump || { echo -e "${YELLOW}Warning: 'objdump' not found. May be required for function extraction.${NC}"; }
    
    echo -e "${GREEN}All requirements checked.${NC}"
}

# Make scripts executable
make_scripts_executable() {
    echo -e "${YELLOW}Making scripts executable...${NC}"
    chmod +x *.py
    echo -e "${GREEN}Scripts are now executable.${NC}"
}

# 1. Compile projects
compile_projects() {
    print_header "Step 1: Compiling projects"
    
    echo -e "${YELLOW}This step may take some time depending on the number of projects and architectures.${NC}"
    echo -e "${YELLOW}You can monitor progress in the logs directory.${NC}"
    
    python3 compile_projects.py --parallel $PARALLEL
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Warning: Some compilations may have failed. Continuing with available binaries.${NC}"
    else
        echo -e "${GREEN}Compilation completed successfully.${NC}"
    fi
}

# 2. Extract functions
extract_functions() {
    print_header "Step 2: Extracting functions"
    
    echo -e "${YELLOW}Extracting functions from compiled binaries...${NC}"
    
    python3 extract_functions.py --parallel $PARALLEL
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Warning: Function extraction encountered some errors. Continuing with extracted functions.${NC}"
    else
        echo -e "${GREEN}Function extraction completed successfully.${NC}"
    fi
}

# 3. Train model
train_model() {
    print_header "Step 3: Training model"
    
    echo -e "${YELLOW}Training model on $SOURCE_ARCH with $SOURCE_COMPILER -$SOURCE_OPT...${NC}"
    
    python3 train_model.py \
        --source-arch $SOURCE_ARCH \
        --source-compiler $SOURCE_COMPILER \
        --source-opt $SOURCE_OPT \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Model training failed.${NC}"
        exit 1
    else
        echo -e "${GREEN}Model training completed successfully.${NC}"
    fi
}

# 4. Test model
test_model() {
    print_header "Step 4: Testing model"
    
    echo -e "${YELLOW}Testing model on all architectures, compilers, and optimization levels...${NC}"
    
    python3 test_model.py \
        --source-arch $SOURCE_ARCH \
        --source-compiler $SOURCE_COMPILER \
        --source-opt $SOURCE_OPT \
        --parallel $PARALLEL
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Warning: Model testing encountered some errors. Continuing with available results.${NC}"
    else
        echo -e "${GREEN}Model testing completed successfully.${NC}"
    fi
}

# 5. Analyze results
analyze_results() {
    print_header "Step 5: Analyzing results"
    
    echo -e "${YELLOW}Analyzing results and generating visualizations...${NC}"
    
    python3 analyze_results.py \
        --source-arch $SOURCE_ARCH \
        --source-compiler $SOURCE_COMPILER \
        --source-opt $SOURCE_OPT
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Warning: Results analysis encountered some errors.${NC}"
    else
        echo -e "${GREEN}Results analysis completed successfully.${NC}"
    fi
}

# Main execution flow
main() {
    print_header "Compiler Optimization Impact Analysis Experiment"
    
    echo -e "${YELLOW}Configuration:${NC}"
    echo -e "  Source architecture: ${GREEN}$SOURCE_ARCH${NC}"
    echo -e "  Source compiler: ${GREEN}$SOURCE_COMPILER${NC}"
    echo -e "  Source optimization: ${GREEN}$SOURCE_OPT${NC}"
    echo -e "  Parallel processes: ${GREEN}$PARALLEL${NC}"
    echo -e "  Training epochs: ${GREEN}$EPOCHS${NC}"
    
    # Ask for confirmation
    echo -e "\n${YELLOW}Do you want to continue with this configuration? (y/n)${NC}"
    read -r confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo -e "${RED}Experiment aborted.${NC}"
        exit 0
    fi
    
    # Start the experiment
    check_requirements
    make_scripts_executable
    
    # Record start time
    start_time=$(date +%s)
    
    # Run each step
    compile_projects
    extract_functions
    train_model
    test_model
    analyze_results
    
    # Calculate and display elapsed time
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    hours=$((elapsed_time / 3600))
    minutes=$(( (elapsed_time % 3600) / 60 ))
    seconds=$((elapsed_time % 60))
    
    print_header "Experiment completed!"
    echo -e "${GREEN}Total time: ${hours}h ${minutes}m ${seconds}s${NC}"
    echo -e "${GREEN}Results are available in the 'results' directory.${NC}"
    
    # Open results directory if on a desktop system
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "results/${SOURCE_ARCH}_${SOURCE_COMPILER}_${SOURCE_OPT}"
    elif [[ "$OSTYPE" == "linux-gnu"* && -n "$DISPLAY" ]]; then
        xdg-open "results/${SOURCE_ARCH}_${SOURCE_COMPILER}_${SOURCE_OPT}" &>/dev/null
    elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        start "results/${SOURCE_ARCH}_${SOURCE_COMPILER}_${SOURCE_OPT}"
    fi
}

# Run the main function
main