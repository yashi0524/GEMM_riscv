#include <stdio.h>
#include <stdlib.h>

//#if __riscv_v_intrinsic >= 1000000
#include <riscv_vector.h>
//#endif /* __riscv_v_intrinsic */

//#include "utils.h"
#include "platform.h"

// Matrix dimensions
// by default M=N=4, user can use -DM to set M=N=K, or use -DM -DN -DK to set seprately
#ifndef M
#define M 4
#endif 

#ifndef N
#define N M
#endif

#ifndef K
#define K M
#endif

unsigned long long cycle_count, inst_count;
unsigned long long hpmcounter[32] = {0};


void scalar_dgemm(int , int , int , 
                  double , const double *, int ,
                  const double *, int ,
                  double , double *, int ) __attribute__((noinline));

/**
 * Scalar DGEMM: C = alpha*(A*B) + beta*C
 * Optimized with i-k-j loop order for better cache locality.
 */
void scalar_dgemm(int m, int n, int k, 
                  double alpha, const double *A, int lda,
                  const double *B, int ldb,
                  double beta, double *C, int ldc)
{
    for (int i = 0; i < m; ++i) {
        // Step 1: Scale existing C by beta
        for (int j = 0; j < n; ++j) {
            C[i * ldc + j] *= beta;
        }

        // Step 2: Accumulate alpha * A * B
        for (int l = 0; l < k; ++l) {
            double temp_a = alpha * A[i * lda + l];
            for (int j = 0; j < n; ++j) {
                C[i * ldc + j] += temp_a * B[l * ldb + j];
            }
        }
    }
}

void print_matrix(const char *name, double *mat, int rows, int cols) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.1f ", mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Static allocation for simplicity in embedded/sim environment
    double A[M * K] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 8.0, 7.0, 6.0,
        5.0, 4.0, 3.0, 2.0
    };

    double B[K * N] = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    };

    double C[M * N] = {0}; // Initialize C with zeros

    double alpha = 1.0;
    double beta = 0.0;
  
    printf("misa = 0x%016lX\n", READ_CSR(misa));

    printf("Starting Scalar DGEMM...\n\n");

    //test vector intrinsic
    {
        size_t vl = __riscv_vsetvl_e64m1(16);
        printf("vl = 0x%08X\n", vl);
        //vint32m1_t __riscv_vadd_vv_i32m1(vint32m1_t vs2, vint32m1_t vs1, size_t vl);
        vint64m1_t vs1, vs2;
        vint64m1_t vd = __riscv_vadd_vv_i64m1( vs2, vs1, vl);
    }

    cycle_count = READ_CSR(mcycle);
    inst_count = READ_CSR(minstret);
    WRITE_CSR(61, mhpmevent3);
    WRITE_CSR(64, mhpmevent4);
    WRITE_CSR(65, mhpmevent5);

    scalar_dgemm(M, N, K, alpha, A, K, B, N, beta, C, N);

    cycle_count = READ_CSR(mcycle) - cycle_count;
    inst_count = READ_CSR(minstret) - inst_count;
    hpmcounter[3] = READ_CSR(mhpmcounter3);
    hpmcounter[4] = READ_CSR(mhpmcounter4);
    hpmcounter[5] = READ_CSR(mhpmcounter5);

#if 0    
    print_matrix("A", A, M, K);
    print_matrix("B", B, K, N);
    print_matrix("C (Result)", C, M, N);
#endif

    printf("counter: mcycle = %llu\n", cycle_count);
    printf("counter: minstret = %llu\n", inst_count);
    printf("hpmcounter[3]: Vector = %llu\n", hpmcounter[3]);
    printf("hpmcounter[4]: VectorLoad = %llu\n", hpmcounter[4]);
    printf("hpmcounter[5]: VectorStore = %llu\n", hpmcounter[5]);

    return 0;
}