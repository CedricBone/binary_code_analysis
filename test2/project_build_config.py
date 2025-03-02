"""Project-specific build configurations for different architectures."""

PROJECT_CONFIGS = {
    "openssl": {
        "x86_64": {
            "gcc": "./config no-asm && make",
            "clang": "./config no-asm CC=clang && make"
        },
        "arm": {
            "gcc": "./Configure linux-armv4 no-asm && make",
            "clang": "./Configure linux-armv4 no-asm CC=clang && make"  
        },
        "mips": {
            "gcc": "./Configure linux-mips32 no-asm && make",
            "clang": "./Configure linux-mips32 no-asm CC=clang && make"
        }
    },
    "coreutils": {
        "x86_64": {
            "gcc": "./configure --disable-nls && make",
            "clang": "./configure --disable-nls CC=clang && make"
        },
        "arm": {
            "gcc": "./configure --disable-nls --host=arm-linux-gnueabi && make",
            "clang": "./configure --disable-nls CC='clang -target arm-linux-gnueabi' && make"
        }
    }
}

def get_build_command(project_name, architecture, compiler):
    """Get the appropriate build command for a specific configuration."""
    project = PROJECT_CONFIGS.get(project_name, {})
    arch_config = project.get(architecture, {})
    
    # Return project-specific command if available
    if compiler in arch_config:
        return arch_config[compiler]
    
    # Default to generic command
    if compiler == "gcc":
        return f"./configure && make"
    elif compiler == "clang":
        return f"./configure CC=clang && make"
    else:
        return "./configure && make"