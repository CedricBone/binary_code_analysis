"""
Configuration file for Compiler Optimization Impact Analysis Experiment.
Contains all the settings and parameters needed for the experiment.
"""

# Projects to compile
PROJECTS = [
    {
        "name": "openssl",
        "version": "1.1.1p",
        "url": "https://www.openssl.org/source/openssl-1.1.1p.tar.gz",
        "build_command": "./config && make",
        "configure_flags": "",
        "functions_of_interest": ["AES_encrypt", "SHA256_Update", "RSA_verify"]
    },
    {
        "name": "coreutils",
        "version": "9.0",
        "url": "https://ftp.gnu.org/gnu/coreutils/coreutils-9.0.tar.xz",
        "build_command": "./configure && make",
        "configure_flags": "--disable-nls",
        "functions_of_interest": ["sort_buffer", "do_ls", "copy_file"]
    },
    {
        "name": "binutils",
        "version": "2.35.2",
        "url": "https://ftp.gnu.org/gnu/binutils/binutils-2.35.2.tar.xz",
        "build_command": "./configure && make",
        "configure_flags": "--disable-werror",
        "functions_of_interest": ["disassemble_fn", "bfd_map_over_sections", "objdump_print_value"]
    },
    {
        "name": "libgcrypt",
        "version": "1.10.1",
        "url": "https://gnupg.org/ftp/gcrypt/libgcrypt/libgcrypt-1.10.1.tar.bz2",
        "build_command": "./configure && make",
        "configure_flags": "--disable-doc",
        "functions_of_interest": ["gcry_md_hash_buffer", "gcry_cipher_encrypt", "gcry_pk_encrypt"]
    }
]

# Target architectures
ARCHITECTURES = {
    "x86_64": {
        "gcc_flags": "-m64",
        "clang_flags": "-m64",
        "triple": "x86_64-linux-gnu",
        "disassembler": "objdump",
        "disassembler_flags": "-d -M intel",
    },
    "arm": {
        "gcc_flags": "-march=armv8-a",
        "clang_flags": "-target arm-linux-gnueabi",
        "triple": "arm-linux-gnueabi",
        "disassembler": "arm-linux-gnueabi-objdump",
        "disassembler_flags": "-d",
    },
    "mips": {
        "gcc_flags": "-march=mips32r2",
        "clang_flags": "-target mips-linux-gnu",
        "triple": "mips-linux-gnu",
        "disassembler": "mips-linux-gnu-objdump",
        "disassembler_flags": "-d",
    },
    "powerpc": {
        "gcc_flags": "-mcpu=powerpc64",
        "clang_flags": "-target powerpc64-linux-gnu",
        "triple": "powerpc64-linux-gnu",
        "disassembler": "powerpc64-linux-gnu-objdump",
        "disassembler_flags": "-d",
    }
}

# Compilers to use
COMPILERS = {
    "gcc": {
        "command": "gcc",
        "cross_format": "{triple}-gcc",
        "version": "10.0.0"  # Minimum version
    },
    "clang": {
        "command": "clang",
        "cross_format": "clang",  # clang uses -target flag instead
        "version": "12.0.0"  # Minimum version
    }
}

# Optimization levels
OPTIMIZATION_LEVELS = ["O0", "O1", "O2", "O3"]

# Output directories
BUILD_DIR = "build"                # Directory for compiled binaries
DOWNLOAD_DIR = "downloads"         # Directory for downloaded source
FUNCTION_DIR = "functions"         # Directory for extracted functions
MODEL_DIR = "models"               # Directory for trained models
RESULTS_DIR = "results"            # Directory for results
LOG_DIR = "logs"                   # Directory for logs

# Model parameters
EMBEDDING_DIM = 200               # Dimension of instruction embeddings
SEQUENCE_LENGTH = 150             # Maximum sequence length for instructions
BATCH_SIZE = 64                   # Batch size for training
EPOCHS = 10                       # Number of epochs for training
LEARNING_RATE = 0.001             # Learning rate for training
VALIDATION_SPLIT = 0.2            # Validation split ratio
NUM_CLASSES = 2                   # Number of classes (similar/dissimilar)

# Function extraction parameters
MIN_FUNCTION_SIZE = 5            # Minimum number of instructions in a function
MAX_FUNCTION_SIZE = 500          # Maximum number of instructions in a function
FUNCTIONS_PER_BINARY = 200       # Number of functions to extract from each binary
MAX_FUNCTIONS_TOTAL = 10000      # Maximum total functions to use for training

# Dataset generation parameters
SIMILAR_PAIRS_RATIO = 0.5        # Ratio of similar pairs in the dataset
NUM_TEST_PAIRS = 2000            # Number of function pairs to use for testing

# Random seed for reproducibility
RANDOM_SEED = 42