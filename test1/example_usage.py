"""
Example usage of the binary code analysis system
"""
import os
import argparse
import shutil
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Example usage of binary code analysis system")
    
    parser.add_argument("--demo", type=str, default="train", 
                        choices=["train", "inference", "full"],
                        help="Demo mode")
    parser.add_argument("--data_dir", type=str, default="./sample_data", 
                        help="Directory for sample data")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Output directory")
    
    return parser.parse_args()

def setup_sample_data(data_dir):
    """Create sample data for demonstration"""
    
    print("Setting up sample data...")
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "training", "similar_pairs"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "training", "dissimilar_pairs"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "training", "samples"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "inference"), exist_ok=True)
    
    # Sample assembly functions
    
    # SHA-256 implementation - remove leading newline
    sha256_func = """.globl sha256_transform
.type sha256_transform, @function
sha256_transform:
    push    %rbp
    mov     %rsp, %rbp
    push    %rbx
    push    %r12
    push    %r13
    push    %r14
    push    %r15
    sub     $0x110, %rsp
    
    # Load message block into W array
    mov     %rsi, %rax
    mov     0x0(%rax), %r8
    bswap   %r8
    mov     %r8, 0x0(%rsp)
    # ... more loading and byte swapping
    
    # Initialize working variables
    mov     0x0(%rdi), %eax    # a = h0
    mov     0x4(%rdi), %ebx    # b = h1
    mov     0x8(%rdi), %ecx    # c = h2
    mov     0xc(%rdi), %edx    # d = h3
    mov     0x10(%rdi), %r8d   # e = h4
    mov     0x14(%rdi), %r9d   # f = h5
    mov     0x18(%rdi), %r10d  # g = h6
    mov     0x1c(%rdi), %r11d  # h = h7
    
    # Main loop - 64 rounds
    xor     %r14d, %r14d
sha256_loop:
    # Calculate temp1
    mov     %r11d, %r13d         # temp1 = h
    mov     %r8d, %r12d          # e
    ror     $6, %r12d
    xor     %r8d, %r12d
    ror     $5, %r12d
    xor     %r8d, %r12d
    ror     $2, %r12d            # S1 = ROR(e,6) ^ ROR(e,11) ^ ROR(e,25)
    add     %r12d, %r13d         # temp1 += S1
    mov     %r8d, %r12d          # e
    mov     %r9d, %r15d          # f
    xor     %r15d, %r12d
    and     %r10d, %r12d
    xor     %r10d, %r15d
    and     %r8d, %r15d
    xor     %r15d, %r12d         # ch = (e & f) ^ ((~e) & g)
    add     %r12d, %r13d         # temp1 += ch
    movl    k_table(,%r14,4), %r15d
    add     %r15d, %r13d         # temp1 += k[i]
    movl    (%rsp,%r14,4), %r15d
    add     %r15d, %r13d         # temp1 += w[i]
    
    # Calculate temp2
    mov     %eax, %r12d          # a
    ror     $2, %r12d
    xor     %eax, %r12d
    ror     $11, %r12d
    xor     %eax, %r12d
    ror     $9, %r12d            # S0 = ROR(a,2) ^ ROR(a,13) ^ ROR(a,22)
    mov     %r12d, %r15d         # temp2 = S0
    mov     %eax, %r12d          # a
    mov     %ebx, %esi           # b
    and     %esi, %r12d          # a & b
    add     %r12d, %r15d         # temp2 += (a & b)
    mov     %eax, %r12d          # a
    mov     %ecx, %esi           # c
    and     %esi, %r12d          # a & c
    add     %r12d, %r15d         # temp2 += (a & c)
    mov     %ebx, %r12d          # b
    and     %ecx, %r12d          # b & c
    add     %r12d, %r15d         # temp2 += (b & c) = maj
    
    # Update working variables
    mov     %r10d, %r11d         # h = g
    mov     %r9d, %r10d          # g = f
    mov     %r8d, %r9d           # f = e
    mov     %edx, %r8d           # e = d + temp1
    add     %r13d, %r8d
    mov     %ecx, %edx           # d = c
    mov     %ebx, %ecx           # c = b
    mov     %eax, %ebx           # b = a
    add     %r13d, %r15d         # a = temp1 + temp2
    mov     %r15d, %eax
    
    # Increment counter and continue loop
    inc     %r14d
    cmp     $64, %r14d
    jl      sha256_loop
    
    # Add the compressed chunk to the current hash value
    mov     0x0(%rdi), %r12d
    add     %eax, %r12d
    mov     %r12d, 0x0(%rdi)
    mov     0x4(%rdi), %r12d
    add     %ebx, %r12d
    mov     %r12d, 0x4(%rdi)
    mov     0x8(%rdi), %r12d
    add     %ecx, %r12d
    mov     %r12d, 0x8(%rdi)
    mov     0xc(%rdi), %r12d
    add     %edx, %r12d
    mov     %r12d, 0xc(%rdi)
    mov     0x10(%rdi), %r12d
    add     %r8d, %r12d
    mov     %r12d, 0x10(%rdi)
    mov     0x14(%rdi), %r12d
    add     %r9d, %r12d
    mov     %r12d, 0x14(%rdi)
    mov     0x18(%rdi), %r12d
    add     %r10d, %r12d
    mov     %r12d, 0x18(%rdi)
    mov     0x1c(%rdi), %r12d
    add     %r11d, %r12d
    mov     %r12d, 0x1c(%rdi)
    
    # Cleanup and return
    add     $0x110, %rsp
    pop     %r15
    pop     %r14
    pop     %r13
    pop     %r12
    pop     %rbx
    pop     %rbp
    ret
"""
    
    # Modified SHA-256 with different register usage but same algorithm
    sha256_variant = """.globl sha256_process
.type sha256_process, @function
sha256_process:
    push    %rbp
    mov     %rsp, %rbp
    push    %rbx
    push    %r12
    push    %r13
    push    %r14
    push    %r15
    sub     $0x120, %rsp
    
    # Load message block into W array (different loading method)
    mov     %rsi, %rcx
    mov     0x0(%rcx), %r9
    bswap   %r9
    mov     %r9, 0x0(%rsp)
    # ... more loading and byte swapping
    
    # Initialize working variables (using different register mapping)
    mov     0x0(%rdi), %r8d     # a = h0
    mov     0x4(%rdi), %r9d     # b = h1
    mov     0x8(%rdi), %r10d    # c = h2
    mov     0xc(%rdi), %r11d    # d = h3
    mov     0x10(%rdi), %eax    # e = h4
    mov     0x14(%rdi), %ebx    # f = h5
    mov     0x18(%rdi), %ecx    # g = h6
    mov     0x1c(%rdi), %edx    # h = h7
    
    # Main loop - 64 rounds (same algorithm, different registers)
    xor     %r14d, %r14d
sha256_main_loop:
    # Calculate temp1 (different registers but same algorithm)
    mov     %edx, %r13d          # temp1 = h
    mov     %eax, %r12d          # e
    ror     $6, %r12d
    xor     %eax, %r12d
    ror     $5, %r12d
    xor     %eax, %r12d
    ror     $2, %r12d            # S1 = ROR(e,6) ^ ROR(e,11) ^ ROR(e,25)
    add     %r12d, %r13d         # temp1 += S1
    mov     %eax, %r12d          # e
    mov     %ebx, %r15d          # f
    xor     %r15d, %r12d
    and     %ecx, %r12d
    xor     %ecx, %r15d
    and     %eax, %r15d
    xor     %r15d, %r12d         # ch = (e & f) ^ ((~e) & g)
    add     %r12d, %r13d         # temp1 += ch
    movl    k_values(,%r14,4), %r15d
    add     %r15d, %r13d         # temp1 += k[i]
    movl    (%rsp,%r14,4), %r15d
    add     %r15d, %r13d         # temp1 += w[i]
    
    # Calculate temp2 (different registers but same algorithm)
    mov     %r8d, %r12d          # a
    ror     $2, %r12d
    xor     %r8d, %r12d
    ror     $11, %r12d
    xor     %r8d, %r12d
    ror     $9, %r12d            # S0 = ROR(a,2) ^ ROR(a,13) ^ ROR(a,22)
    mov     %r12d, %r15d         # temp2 = S0
    mov     %r8d, %r12d          # a
    mov     %r9d, %esi           # b
    and     %esi, %r12d          # a & b
    add     %r12d, %r15d         # temp2 += (a & b)
    mov     %r8d, %r12d          # a
    mov     %r10d, %esi          # c
    and     %esi, %r12d          # a & c
    add     %r12d, %r15d         # temp2 += (a & c)
    mov     %r9d, %r12d          # b
    and     %r10d, %r12d         # b & c
    add     %r12d, %r15d         # temp2 += (b & c) = maj
    
    # Update working variables (different register mapping but same algorithm)
    mov     %ecx, %edx           # h = g
    mov     %ebx, %ecx           # g = f
    mov     %eax, %ebx           # f = e
    mov     %r11d, %eax          # e = d + temp1
    add     %r13d, %eax
    mov     %r10d, %r11d         # d = c
    mov     %r9d, %r10d          # c = b
    mov     %r8d, %r9d           # b = a
    add     %r13d, %r15d         # a = temp1 + temp2
    mov     %r15d, %r8d
    
    # Increment counter and continue loop
    inc     %r14d
    cmp     $64, %r14d
    jl      sha256_main_loop
    
    # Add the compressed chunk to the current hash value
    mov     0x0(%rdi), %r12d
    add     %r8d, %r12d
    mov     %r12d, 0x0(%rdi)
    mov     0x4(%rdi), %r12d
    add     %r9d, %r12d
    mov     %r12d, 0x4(%rdi)
    # ... continue with other hash values
    
    # Cleanup and return
    add     $0x120, %rsp
    pop     %r15
    pop     %r14
    pop     %r13
    pop     %r12
    pop     %rbx
    pop     %rbp
    ret
"""
    
    # Completely different function - AES encryption
    aes_func = """.globl aes_encrypt_block
.type aes_encrypt_block, @function
aes_encrypt_block:
    push    %rbp
    mov     %rsp, %rbp
    push    %rbx
    push    %r12
    push    %r13
    push    %r14
    push    %r15
    sub     $0x60, %rsp
    
    # Parameters:
    # rdi = state (16 bytes, modified in-place)
    # rsi = key schedule (44 words for AES-128)
    
    # Load state into registers
    movdqu  (%rdi), %xmm0
    
    # Add first round key (pre-whitening)
    movdqu  (%rsi), %xmm1
    pxor    %xmm1, %xmm0
    
    # 9 rounds of:
    # - SubBytes (S-box substitution)
    # - ShiftRows (cyclic shift of rows)
    # - MixColumns (matrix multiplication in GF(2^8))
    # - AddRoundKey
    
    # Round 1
    lea     sbox_table(%rip), %r8
    aesenc  16(%rsi), %xmm0
    
    # Round 2
    aesenc  32(%rsi), %xmm0
    
    # Round 3
    aesenc  48(%rsi), %xmm0
    
    # Round 4
    aesenc  64(%rsi), %xmm0
    
    # Round 5
    aesenc  80(%rsi), %xmm0
    
    # Round 6
    aesenc  96(%rsi), %xmm0
    
    # Round 7
    aesenc  112(%rsi), %xmm0
    
    # Round 8
    aesenc  128(%rsi), %xmm0
    
    # Round 9
    aesenc  144(%rsi), %xmm0
    
    # Final round (no MixColumns)
    aesenclast 160(%rsi), %xmm0
    
    # Store result back to memory
    movdqu  %xmm0, (%rdi)
    
    # Cleanup and return
    add     $0x60, %rsp
    pop     %r15
    pop     %r14
    pop     %r13
    pop     %r12
    pop     %rbx
    pop     %rbp
    ret
"""
    
    # Binary search implementation
    binary_search_func = """.globl binary_search
.type binary_search, @function
binary_search:
    # Parameters:
    # rdi = array pointer
    # rsi = array size
    # rdx = target value to find
    
    push    %rbp
    mov     %rsp, %rbp
    
    # Initialize bounds
    mov     $0, %rax        # left = 0
    mov     %rsi, %rcx      # right = size
    sub     $1, %rcx
    
binary_search_loop:
    # Check if left > right
    cmp     %rcx, %rax
    jg      not_found
    
    # Calculate mid = (left + right) / 2
    mov     %rax, %r8
    add     %rcx, %r8
    shr     $1, %r8
    
    # Compare array[mid] with target
    mov     (%rdi,%r8,4), %r9d    # Assuming 4-byte elements
    cmp     %edx, %r9d
    je      found           # Equal - found it
    jl      search_right    # Less than - search right half
    
    # Greater than - search left half
    mov     %r8, %rcx       # right = mid - 1
    sub     $1, %rcx
    jmp     binary_search_loop
    
search_right:
    mov     %r8, %rax       # left = mid + 1
    add     $1, %rax
    jmp     binary_search_loop
    
found:
    mov     %r8, %rax       # Return mid index
    jmp     done
    
not_found:
    mov     $-1, %rax       # Return -1 if not found
    
done:
    pop     %rbp
    ret
"""
    
    # Binary search variant (different variable ordering, same algorithm)
    binary_search_variant = """.globl binary_search_int
.type binary_search_int, @function
binary_search_int:
    # Parameters:
    # rdi = int array pointer
    # rsi = array length
    # rdx = value to search for
    
    push    %rbp
    mov     %rsp, %rbp
    
    # Initialize search boundaries
    xor     %r8, %r8        # low = 0
    mov     %rsi, %r9       # high = length
    dec     %r9             # high = length - 1
    
search_loop:
    # Check termination condition
    cmp     %r9, %r8
    jg      value_not_found
    
    # Find middle index: mid = (low + high) / 2
    mov     %r8, %r10
    add     %r9, %r10
    shr     $1, %r10
    
    # Compare array[mid] with target
    mov     (%rdi,%r10,4), %eax   # Get array[mid] (4-byte ints)
    cmp     %edx, %eax
    je      value_found     # If equal, we found it
    jg      search_lower    # If greater, search lower half
    
    # Search upper half (array[mid] < target)
    lea     1(%r10), %r8    # low = mid + 1
    jmp     search_loop
    
search_lower:
    dec     %r10            # high = mid - 1
    mov     %r10, %r9
    jmp     search_loop
    
value_found:
    mov     %r10, %rax      # Return position where found
    jmp     search_done
    
value_not_found:
    mov     $-1, %rax       # Return -1 if not found
    
search_done:
    pop     %rbp
    ret
"""
    
    # A simple description of SHA-256 function
    sha256_desc = "Implements the SHA-256 transform function that processes a single 64-byte block of data. The function updates the hash state in-place by performing the SHA-256 compression function with 64 rounds of processing."
    
    # A simple description of AES function
    aes_desc = "Performs AES-128 encryption on a single 16-byte block using the provided key schedule. Uses hardware AES instructions to implement the SubBytes, ShiftRows, MixColumns, and AddRoundKey operations across 10 rounds of encryption."
    
    # A simple description of binary search function
    binary_search_desc = "Implements binary search algorithm to find a target value in a sorted array of integers. Returns the index of the target if found, or -1 if not present in the array."
    
    # Save functions to files
    # Training data
    # Similar pairs
    os.makedirs(os.path.join(data_dir, "training", "similar_pairs", "pair1"), exist_ok=True)
    with open(os.path.join(data_dir, "training", "similar_pairs", "pair1", "func1.asm"), "w") as f:
        f.write(sha256_func)
    with open(os.path.join(data_dir, "training", "similar_pairs", "pair1", "func2.asm"), "w") as f:
        f.write(sha256_variant)
    
    os.makedirs(os.path.join(data_dir, "training", "similar_pairs", "pair2"), exist_ok=True)
    with open(os.path.join(data_dir, "training", "similar_pairs", "pair2", "func1.asm"), "w") as f:
        f.write(binary_search_func)
    with open(os.path.join(data_dir, "training", "similar_pairs", "pair2", "func2.asm"), "w") as f:
        f.write(binary_search_variant)
    
    # Dissimilar pairs
    os.makedirs(os.path.join(data_dir, "training", "dissimilar_pairs", "pair1"), exist_ok=True)
    with open(os.path.join(data_dir, "training", "dissimilar_pairs", "pair1", "func1.asm"), "w") as f:
        f.write(sha256_func)
    with open(os.path.join(data_dir, "training", "dissimilar_pairs", "pair1", "func2.asm"), "w") as f:
        f.write(aes_func)
    
    os.makedirs(os.path.join(data_dir, "training", "dissimilar_pairs", "pair2"), exist_ok=True)
    with open(os.path.join(data_dir, "training", "dissimilar_pairs", "pair2", "func1.asm"), "w") as f:
        f.write(binary_search_func)
    with open(os.path.join(data_dir, "training", "dissimilar_pairs", "pair2", "func2.asm"), "w") as f:
        f.write(aes_func)
    
    # Description samples
    os.makedirs(os.path.join(data_dir, "training", "samples", "sample1"), exist_ok=True)
    with open(os.path.join(data_dir, "training", "samples", "sample1", "func.asm"), "w") as f:
        f.write(sha256_func)
    with open(os.path.join(data_dir, "training", "samples", "sample1", "desc.txt"), "w") as f:
        f.write(sha256_desc)
    
    os.makedirs(os.path.join(data_dir, "training", "samples", "sample2"), exist_ok=True)
    with open(os.path.join(data_dir, "training", "samples", "sample2", "func.asm"), "w") as f:
        f.write(aes_func)
    with open(os.path.join(data_dir, "training", "samples", "sample2", "desc.txt"), "w") as f:
        f.write(aes_desc)
    
    os.makedirs(os.path.join(data_dir, "training", "samples", "sample3"), exist_ok=True)
    with open(os.path.join(data_dir, "training", "samples", "sample3", "func.asm"), "w") as f:
        f.write(binary_search_func)
    with open(os.path.join(data_dir, "training", "samples", "sample3", "desc.txt"), "w") as f:
        f.write(binary_search_desc)
    
    # Inference data
    with open(os.path.join(data_dir, "inference", "sha256.asm"), "w") as f:
        f.write(sha256_func)
    with open(os.path.join(data_dir, "inference", "sha256_variant.asm"), "w") as f:
        f.write(sha256_variant)
    with open(os.path.join(data_dir, "inference", "aes.asm"), "w") as f:
        f.write(aes_func)
    with open(os.path.join(data_dir, "inference", "binary_search.asm"), "w") as f:
        f.write(binary_search_func)
    
    print(f"Sample data created in {data_dir}")

def train_similarity_model(data_dir, output_dir):
    """Train a similarity model on sample data"""
    
    print("Training similarity model...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run training script
    cmd = [
        sys.executable, "train_similarity.py",
        "--train_data", os.path.join(data_dir, "training"),
        "--val_data", os.path.join(data_dir, "training"),  # Use same data for simplicity
        "--output_dir", os.path.join(output_dir, "similarity"),
        "--model_type", "safe",
        "--embedding_dim", "128",
        "--hidden_dim", "256",
        "--num_layers", "2",
        "--epochs", "3",
        "--batch_size", "2",
        "--max_seq_length", "200"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Similarity model trained and saved to {os.path.join(output_dir, 'similarity')}")
    except subprocess.CalledProcessError as e:
        print(f"Error training similarity model: {e}")
        print("Check if the implementation of preprocessing.py has been fixed.")

def train_description_model(data_dir, output_dir):
    """Train a description generation model on sample data"""
    
    print("Training description generation model...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run training script
    cmd = [
        sys.executable, "train_description.py",
        "--train_data", os.path.join(data_dir, "training"),
        "--val_data", os.path.join(data_dir, "training"),  # Use same data for simplicity
        "--output_dir", os.path.join(output_dir, "description"),
        "--encoder_type", "transformer",
        "--embedding_dim", "128",
        "--hidden_dim", "256",
        "--encoder_layers", "3",
        "--decoder_layers", "3",
        "--epochs", "3",
        "--batch_size", "2",
        "--max_seq_length", "200",
        "--max_desc_length", "30"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Description model trained and saved to {os.path.join(output_dir, 'description')}")
    except subprocess.CalledProcessError as e:
        print(f"Error training description model: {e}")
        print("Check if the implementation of preprocessing.py has been fixed.")

def run_inference(data_dir, output_dir):
    """Run inference on sample data"""
    
    print("Running inference...")
    
    # Similarity inference
    print("\nRunning similarity inference...")
    cmd = [
        sys.executable, "inference_similarity.py",
        "--model_path", os.path.join(output_dir, "similarity", "best_model.pt"),
        "--vocab_dir", os.path.join(output_dir, "similarity"),
        "--func1_path", os.path.join(data_dir, "inference", "sha256.asm"),
        "--func2_path", os.path.join(data_dir, "inference", "sha256_variant.asm"),
        "--verbose"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running similarity inference: {e}")
    
    # Description generation inference
    print("\nRunning description generation inference...")
    cmd = [
        sys.executable, "inference_description.py",
        "--model_path", os.path.join(output_dir, "description", "best_model.pt"),
        "--vocab_dir", os.path.join(output_dir, "description"),
        "--func_path", os.path.join(data_dir, "inference", "binary_search.asm"),
        "--beam_size", "2",
        "--verbose"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running description inference: {e}")
    
    # Full analysis
    print("\nRunning full analysis...")
    cmd = [
        sys.executable, "run_analysis.py",
        "--mode", "both",
        "--similarity_model", os.path.join(output_dir, "similarity", "best_model.pt"),
        "--description_model", os.path.join(output_dir, "description", "best_model.pt"),
        "--vocab_dir", os.path.join(output_dir, "similarity"),
        "--input_dir", os.path.join(data_dir, "inference"),
        "--output_file", os.path.join(output_dir, "analysis_results.json"),
        "--verbose"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\nAnalysis results saved to {os.path.join(output_dir, 'analysis_results.json')}")
    except subprocess.CalledProcessError as e:
        print(f"Error running full analysis: {e}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup sample data
    setup_sample_data(args.data_dir)
    
    # Run demo based on mode
    if args.demo == "train":
        train_similarity_model(args.data_dir, args.output_dir)
        train_description_model(args.data_dir, args.output_dir)
    elif args.demo == "inference":
        # For inference demo, we need trained models
        if not os.path.exists(os.path.join(args.output_dir, "similarity", "best_model.pt")):
            print("Error: Trained similarity model not found. Please run training first.")
            return
        if not os.path.exists(os.path.join(args.output_dir, "description", "best_model.pt")):
            print("Error: Trained description model not found. Please run training first.")
            return
        
        run_inference(args.data_dir, args.output_dir)
    elif args.demo == "full":
        train_similarity_model(args.data_dir, args.output_dir)
        train_description_model(args.data_dir, args.output_dir)
        run_inference(args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()