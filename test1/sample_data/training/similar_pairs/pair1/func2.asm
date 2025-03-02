.globl sha256_process
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
