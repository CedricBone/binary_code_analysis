.globl binary_search_int
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
