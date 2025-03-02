
                #include <stdio.h>
                #include <stdbool.h>
                
                bool is_prime(int n) {
                    if (n <= 1) return false;
                    if (n <= 3) return true;
                    
                    if (n % 2 == 0 || n % 3 == 0) return false;
                    
                    for (int i = 5; i * i <= n; i += 6) {
                        if (n % i == 0 || n % (i + 2) == 0)
                            return false;
                    }
                    
                    return true;
                }
                
                int main() {
                    int count = 0;
                    printf("Prime numbers between 1 and 100: ");
                    
                    for (int i = 1; i <= 100; i++) {
                        if (is_prime(i)) {
                            printf("%d ", i);
                            count++;
                        }
                    }
                    
                    printf("\nTotal: %d prime numbers\n", count);
                    return 0;
                }
            