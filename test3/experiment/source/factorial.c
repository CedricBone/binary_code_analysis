
                #include <stdio.h>
                
                int factorial(int n) {
                    if (n <= 1) return 1;
                    return n * factorial(n - 1);
                }
                
                int main(int argc, char **argv) {
                    int i;
                    for (i = 1; i <= 10; i++) {
                        printf("Factorial of %d is %d\n", i, factorial(i));
                    }
                    return 0;
                }
            