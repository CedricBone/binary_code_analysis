
                #include <stdio.h>
                
                int fibonacci(int n) {
                    if (n <= 0) return 0;
                    if (n == 1) return 1;
                    return fibonacci(n-1) + fibonacci(n-2);
                }
                
                int main(int argc, char **argv) {
                    int i;
                    for (i = 0; i < 15; i++) {
                        printf("Fibonacci of %d is %d\n", i, fibonacci(i));
                    }
                    return 0;
                }
            