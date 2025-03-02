"""
Simplified configuration for the initial phase of the experiment.
This focuses on x86_64 architecture with better error handling.
"""

# Reduced set of projects that are more likely to compile successfully
SIMPLIFIED_PROJECTS = [
    {
        "name": "coreutils",
        "version": "9.0",
        "url": "https://ftp.gnu.org/gnu/coreutils/coreutils-9.0.tar.xz",
        "build_command": "./configure --disable-nls && make",
        "configure_flags": "--disable-nls",
        "functions_of_interest": ["sort_buffer", "do_ls", "copy_file"]
    },
    {
        "name": "libgcrypt",
        "version": "1.10.1",
        "url": "https://gnupg.org/ftp/gcrypt/libgcrypt/libgcrypt-1.10.1.tar.bz2",
        "build_command": "./configure --disable-doc && make",
        "configure_flags": "--disable-doc",
        "functions_of_interest": ["gcry_md_hash_buffer", "gcry_cipher_encrypt", "gcry_pk_encrypt"]
    }
]

# Focus on x86_64 architecture initially
SIMPLIFIED_ARCHITECTURES = {
    "x86_64": {
        "gcc_flags": "-m64",
        "clang_flags": "-m64",
        "triple": "x86_64-linux-gnu",
        "disassembler": "objdump",
        "disassembler_flags": "-d -M intel",
    }
}

# Use this function to get the appropriate project list based on environment
def get_projects(use_simplified=False):
    """Get project configuration based on environment"""
    if use_simplified:
        return SIMPLIFIED_PROJECTS
    else:
        # Import from the main config
        from config import PROJECTS
        return PROJECTS

# Use this function to get the appropriate architecture list
def get_architectures(arch_mode="x86_64_only"):
    """Get architecture configuration based on mode"""
    from config import ARCHITECTURES
    
    if arch_mode == "x86_64_only":
        return {"x86_64": ARCHITECTURES["x86_64"]}
    elif arch_mode == "x86_64_arm":
        return {k: ARCHITECTURES[k] for k in ["x86_64", "arm"] if k in ARCHITECTURES}
    elif arch_mode == "x86_64_mips":
        return {k: ARCHITECTURES[k] for k in ["x86_64", "mips"] if k in ARCHITECTURES}
    else:
        return ARCHITECTURES

# Function to modify compile_projects.py to use this simplified config
def patch_compile_script():
    """
    Code to dynamically update compile_projects.py to use simplified config
    This can be added at the top of the script
    """
    import os
    arch_mode = os.environ.get("ARCH_MODE", "x86_64_only")
    use_simplified = os.environ.get("USE_SIMPLIFIED", "True").lower() == "true"
    
    # Import necessary functions from simplified_config
    from simplified_config import get_projects, get_architectures
    
    # Override the global variables in config
    import config
    if use_simplified:
        config.PROJECTS = get_projects(use_simplified=True)
    
    config.ARCHITECTURES = get_architectures(arch_mode)