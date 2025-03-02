.globl binary_search
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
