.globl sha256_transform
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
