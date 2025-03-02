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
    
    # Check critical dependencies
    for cmd in gcc g++ make objdump; do
        if ! check_command $cmd; then
            echo -e "${RED}Critical dependency $cmd is missing. Please install it before continuing.${NC}"
            exit 1
        fi
    done
    
    # Check for cross-architecture tools - warn but don't fail
    missing=0
    for tool in arm-linux-gnueabi-objdump mips-linux-gnu-objdump powerpc64-linux-gnu-objdump; do
        if ! check_command $tool; then
            echo -e "${YELLOW}Warning: $tool not found. Cross-architecture analysis will be limited.${NC}"
            missing=1
        fi
    done
    
    if [ $missing -eq 1 ]; then
        echo -e "${YELLOW}Some cross-architecture tools are missing. You can install them with:${NC}"
        echo -e "${YELLOW}sudo apt install binutils-arm-linux-gnueabi binutils-mips-linux-gnu binutils-powerpc-linux-gnu${NC}"
        echo -e "${YELLOW}Continuing with x86_64 only...${NC}"
    else
        echo -e "${GREEN}All cross-architecture tools are available.${NC}"
    fi
    
    echo -e "${GREEN}All core requirements checked.${NC}"
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
    
    # If cross-arch tools are missing, only compile for x86_64
    if ! command -v arm-linux-gnueabi-objdump &> /dev/null; then
        echo -e "${YELLOW}Cross-architecture tools missing, compiling only for x86_64...${NC}"
        python3 compile_projects.py --parallel $PARALLEL --arch x86_64
    else
        python3 compile_projects.py --parallel $PARALLEL
    fi
    
    RESULT=$?
    if [ $RESULT -ne 0 ]; then
        echo -e "${RED}Error: Compilation failed. Check logs for details.${NC}"
        # Continue anyway to see how far we can get
    else
        echo -e "${GREEN}Compilation completed successfully.${NC}"
    fi
}

# 2. Extract functions
extract_functions() {
    print_header "Step 2: Extracting functions"
    
    echo -e "${YELLOW}Extracting functions from compiled binaries...${NC}"
    
    # If cross-arch tools are missing, only extract for x86_64
    if ! command -v arm-linux-gnueabi-objdump &> /dev/null; then
        echo -e "${YELLOW}Cross-architecture tools missing, extracting only from x86_64...${NC}"
        python3 extract_functions.py --parallel $PARALLEL --arch x86_64
    else
        python3 extract_functions.py --parallel $PARALLEL
    fi
    
    RESULT=$?
    if [ $RESULT -ne 0 ]; then
        echo -e "${RED}Error: Function extraction failed. Check logs for details.${NC}"
        # Continue anyway, use synthetic data if needed
    else
        echo -e "${GREEN}Function extraction completed successfully.${NC}"
    fi
}

# 3. Train model
train_model() {
    print_header "Step 3: Training model"
    
    echo -e "${YELLOW}Training model on $SOURCE_ARCH with $SOURCE_COMPILER -$SOURCE_OPT...${NC}"
    
    # Try to generate synthetic data if no real data is available
    python3 train_model.py \
        --source-arch $SOURCE_ARCH \
        --source-compiler $SOURCE_COMPILER \
        --source-opt $SOURCE_OPT \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --use-synthetic-data
    
    RESULT=$?
    if [ $RESULT -ne 0 ]; then
        echo -e "${RED}Error: Model training failed.${NC}"
        echo -e "${RED}Cannot continue with testing and analysis.${NC}"
        exit 1
    else
        echo -e "${GREEN}Model training completed successfully.${NC}"
    fi
}

# 4. Test model
test_model() {
    print_header "Step 4: Testing model"
    
    echo -e "${YELLOW}Testing model on all architectures, compilers, and optimization levels...${NC}"
    
    # If cross-arch tools are missing, only test on x86_64
    if ! command -v arm-linux-gnueabi-objdump &> /dev/null; then
        echo -e "${YELLOW}Cross-architecture tools missing, testing only on x86_64...${NC}"
        python3 test_model.py \
            --source-arch $SOURCE_ARCH \
            --source-compiler $SOURCE_COMPILER \
            --source-opt $SOURCE_OPT \
            --target-arch x86_64 \
            --parallel $PARALLEL \
            --use-synthetic-data
    else
        python3 test_model.py \
            --source-arch $SOURCE_ARCH \
            --source-compiler $SOURCE_COMPILER \
            --source-opt $SOURCE_OPT \
            --parallel $PARALLEL \
            --use-synthetic-data
    fi
    
    RESULT=$?
    if [ $RESULT -ne 0 ]; then
        echo -e "${RED}Error: Model testing failed. Check logs for details.${NC}"
        echo -e "${RED}Cannot continue with analysis.${NC}"
        exit 1
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
    
    RESULT=$?
    if [ $RESULT -ne 0 ]; then
        echo -e "${RED}Error: Results analysis failed. Check logs for details.${NC}"
        exit 1
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