.globl aes_encrypt_block
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
