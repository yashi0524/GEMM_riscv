#include <stdio.h>
#include <stdlib.h>

//#if __riscv_v_intrinsic >= 1000000
#include <riscv_vector.h>
//#endif /* __riscv_v_intrinsic */

//#include "utils.h"
#include "platform.h"

// Data type — override at build time with -Dtarget_float=<type>
// e.g. make gemm TARGET_FLOAT=__bf16
#ifndef target_float
#define target_float double
#endif

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

// Static allocation for simplicity in embedded/sim environment
target_float A[M * K] __attribute__((aligned(64))) = {
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 8.0, 7.0, 6.0,
    5.0, 4.0, 3.0, 2.0
};

target_float B[K * N] __attribute__((aligned(64))) = {
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0
};

target_float C[M * N] __attribute__((aligned(64))) = {0};
target_float C_ref[M * N] __attribute__((aligned(64))) = {0};  // snapshot of scalar_gemm's result, for opt_gemm correctness check


void scalar_gemm(int , int , int ,
                 target_float , const target_float *, int ,
                 const target_float *, int ,
                 target_float , target_float *, int ) __attribute__((noinline));

/**
 * Scalar GEMM: C = alpha*(A*B) + beta*C
 * Optimized with i-k-j loop order for better cache locality.
 */
void scalar_gemm(int m, int n, int k,
                 target_float alpha, const target_float *A, int lda,
                 const target_float *B, int ldb,
                 target_float beta, target_float *C, int ldc)
{
    for (int i = 0; i < m; ++i) {
        // Step 1: Scale existing C by beta
        for (int j = 0; j < n; ++j) {
            C[i * ldc + j] *= beta;
        }

        // Step 2: Accumulate alpha * A * B
        for (int l = 0; l < k; ++l) {
            target_float temp_a = alpha * A[i * lda + l];
            for (int j = 0; j < n; ++j) {
                C[i * ldc + j] += temp_a * B[l * ldb + j];
            }
        }
    }
}

void opt_gemm(int , int , int ,
             target_float , const target_float *, int ,
             const target_float *, int ,
             target_float , target_float *, int ) __attribute__((noinline));

#ifndef OPT_GEMM_UNROLL
#define OPT_GEMM_UNROLL 8
#endif

/**
 * Optimized GEMM: C = alpha*(A*B) + beta*C
 * Same i-k-j loop order and math as scalar_gemm, but hand-vectorized with
 * explicit RVV intrinsics instead of relying on auto-vectorization, and the
 * l-reduction is split across OPT_GEMM_UNROLL independent accumulator
 * chains (vc0..vc7), summed together after the loop — the same
 * dependency-chain-breaking technique fmacc.c/fmacc_fp16.c use for their
 * peak-compute unroll. Each vcN is a real vector-register SSA value kept
 * live across its share of l, so C is touched only twice per row (read
 * once for the beta-scale, written once at the end) instead of once per
 * (row, l) — a register-resident accumulator neither scalar_gemm nor two
 * earlier restructuring attempts achieved (see git history/PR notes):
 * clang's per-innermost-loop auto-vectorizer never hoists an accumulator
 * across an *enclosing* scalar loop regardless of aliasing hints, so those
 * versions re-load/re-store C (or a stand-in array) through memory on every
 * l iteration. A single (non-unrolled) hand-vectorized accumulator fixes
 * that memory traffic but leaves a serial vc->vc dependency chain across
 * all of l — the same throughput floor fmacc's serial version hit on
 * MinorCPU's dual-issue pipeline; unrolling into independent chains here
 * targets that same bottleneck.
 *
 * k need not be a multiple of OPT_GEMM_UNROLL; a scalar-l tail loop folds
 * any remainder into vc0 after the unrolled main loop.
 *
 * fp16 vs. fp64 intrinsics are selected via #ifdef __riscv_zvfh, which this
 * project's build always enables exactly when target_float=_Float16 (see
 * Makefile's fmacc_fp16 target and sweep_config.json's fp16 "march") — so
 * this single function correctly covers both dtypes this repo builds.
 */
void opt_gemm(int m, int n, int k,
             target_float alpha, const target_float *A, int lda,
             const target_float *B, int ldb,
             target_float beta, target_float *C, int ldc)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ) {
#if defined(__riscv_zvfh)
            size_t vl = __riscv_vsetvl_e16m1(n - j);
            vfloat16m1_t vc0 = __riscv_vle16_v_f16m1(&C[i * ldc + j], vl);
            vc0 = __riscv_vfmul_vf_f16m1(vc0, beta, vl);
            vfloat16m1_t vc1 = __riscv_vfmv_v_f_f16m1(0, vl);
            vfloat16m1_t vc2 = __riscv_vfmv_v_f_f16m1(0, vl);
            vfloat16m1_t vc3 = __riscv_vfmv_v_f_f16m1(0, vl);
            vfloat16m1_t vc4 = __riscv_vfmv_v_f_f16m1(0, vl);
            vfloat16m1_t vc5 = __riscv_vfmv_v_f_f16m1(0, vl);
            vfloat16m1_t vc6 = __riscv_vfmv_v_f_f16m1(0, vl);
            vfloat16m1_t vc7 = __riscv_vfmv_v_f_f16m1(0, vl);

            int l = 0;
            for (; l + OPT_GEMM_UNROLL <= k; l += OPT_GEMM_UNROLL) {
                target_float a0 = alpha * A[i * lda + l + 0];
                target_float a1 = alpha * A[i * lda + l + 1];
                target_float a2 = alpha * A[i * lda + l + 2];
                target_float a3 = alpha * A[i * lda + l + 3];
                target_float a4 = alpha * A[i * lda + l + 4];
                target_float a5 = alpha * A[i * lda + l + 5];
                target_float a6 = alpha * A[i * lda + l + 6];
                target_float a7 = alpha * A[i * lda + l + 7];

                vfloat16m1_t b0 = __riscv_vle16_v_f16m1(&B[(l + 0) * ldb + j], vl);
                vfloat16m1_t b1 = __riscv_vle16_v_f16m1(&B[(l + 1) * ldb + j], vl);
                vfloat16m1_t b2 = __riscv_vle16_v_f16m1(&B[(l + 2) * ldb + j], vl);
                vfloat16m1_t b3 = __riscv_vle16_v_f16m1(&B[(l + 3) * ldb + j], vl);
                vfloat16m1_t b4 = __riscv_vle16_v_f16m1(&B[(l + 4) * ldb + j], vl);
                vfloat16m1_t b5 = __riscv_vle16_v_f16m1(&B[(l + 5) * ldb + j], vl);
                vfloat16m1_t b6 = __riscv_vle16_v_f16m1(&B[(l + 6) * ldb + j], vl);
                vfloat16m1_t b7 = __riscv_vle16_v_f16m1(&B[(l + 7) * ldb + j], vl);

                vc0 = __riscv_vfmacc_vf_f16m1(vc0, a0, b0, vl);
                vc1 = __riscv_vfmacc_vf_f16m1(vc1, a1, b1, vl);
                vc2 = __riscv_vfmacc_vf_f16m1(vc2, a2, b2, vl);
                vc3 = __riscv_vfmacc_vf_f16m1(vc3, a3, b3, vl);
                vc4 = __riscv_vfmacc_vf_f16m1(vc4, a4, b4, vl);
                vc5 = __riscv_vfmacc_vf_f16m1(vc5, a5, b5, vl);
                vc6 = __riscv_vfmacc_vf_f16m1(vc6, a6, b6, vl);
                vc7 = __riscv_vfmacc_vf_f16m1(vc7, a7, b7, vl);
            }
            for (; l < k; ++l) {
                target_float a = alpha * A[i * lda + l];
                vfloat16m1_t b = __riscv_vle16_v_f16m1(&B[l * ldb + j], vl);
                vc0 = __riscv_vfmacc_vf_f16m1(vc0, a, b, vl);
            }

            vfloat16m1_t vc01 = __riscv_vfadd_vv_f16m1(vc0, vc1, vl);
            vfloat16m1_t vc23 = __riscv_vfadd_vv_f16m1(vc2, vc3, vl);
            vfloat16m1_t vc45 = __riscv_vfadd_vv_f16m1(vc4, vc5, vl);
            vfloat16m1_t vc67 = __riscv_vfadd_vv_f16m1(vc6, vc7, vl);
            vfloat16m1_t vc0123 = __riscv_vfadd_vv_f16m1(vc01, vc23, vl);
            vfloat16m1_t vc4567 = __riscv_vfadd_vv_f16m1(vc45, vc67, vl);
            vfloat16m1_t vc = __riscv_vfadd_vv_f16m1(vc0123, vc4567, vl);

            __riscv_vse16_v_f16m1(&C[i * ldc + j], vc, vl);
#else
            size_t vl = __riscv_vsetvl_e64m1(n - j);
            vfloat64m1_t vc0 = __riscv_vle64_v_f64m1(&C[i * ldc + j], vl);
            vc0 = __riscv_vfmul_vf_f64m1(vc0, beta, vl);
            vfloat64m1_t vc1 = __riscv_vfmv_v_f_f64m1(0, vl);
            vfloat64m1_t vc2 = __riscv_vfmv_v_f_f64m1(0, vl);
            vfloat64m1_t vc3 = __riscv_vfmv_v_f_f64m1(0, vl);
            vfloat64m1_t vc4 = __riscv_vfmv_v_f_f64m1(0, vl);
            vfloat64m1_t vc5 = __riscv_vfmv_v_f_f64m1(0, vl);
            vfloat64m1_t vc6 = __riscv_vfmv_v_f_f64m1(0, vl);
            vfloat64m1_t vc7 = __riscv_vfmv_v_f_f64m1(0, vl);

            int l = 0;
            for (; l + OPT_GEMM_UNROLL <= k; l += OPT_GEMM_UNROLL) {
                target_float a0 = alpha * A[i * lda + l + 0];
                target_float a1 = alpha * A[i * lda + l + 1];
                target_float a2 = alpha * A[i * lda + l + 2];
                target_float a3 = alpha * A[i * lda + l + 3];
                target_float a4 = alpha * A[i * lda + l + 4];
                target_float a5 = alpha * A[i * lda + l + 5];
                target_float a6 = alpha * A[i * lda + l + 6];
                target_float a7 = alpha * A[i * lda + l + 7];

                vfloat64m1_t b0 = __riscv_vle64_v_f64m1(&B[(l + 0) * ldb + j], vl);
                vfloat64m1_t b1 = __riscv_vle64_v_f64m1(&B[(l + 1) * ldb + j], vl);
                vfloat64m1_t b2 = __riscv_vle64_v_f64m1(&B[(l + 2) * ldb + j], vl);
                vfloat64m1_t b3 = __riscv_vle64_v_f64m1(&B[(l + 3) * ldb + j], vl);
                vfloat64m1_t b4 = __riscv_vle64_v_f64m1(&B[(l + 4) * ldb + j], vl);
                vfloat64m1_t b5 = __riscv_vle64_v_f64m1(&B[(l + 5) * ldb + j], vl);
                vfloat64m1_t b6 = __riscv_vle64_v_f64m1(&B[(l + 6) * ldb + j], vl);
                vfloat64m1_t b7 = __riscv_vle64_v_f64m1(&B[(l + 7) * ldb + j], vl);

                vc0 = __riscv_vfmacc_vf_f64m1(vc0, a0, b0, vl);
                vc1 = __riscv_vfmacc_vf_f64m1(vc1, a1, b1, vl);
                vc2 = __riscv_vfmacc_vf_f64m1(vc2, a2, b2, vl);
                vc3 = __riscv_vfmacc_vf_f64m1(vc3, a3, b3, vl);
                vc4 = __riscv_vfmacc_vf_f64m1(vc4, a4, b4, vl);
                vc5 = __riscv_vfmacc_vf_f64m1(vc5, a5, b5, vl);
                vc6 = __riscv_vfmacc_vf_f64m1(vc6, a6, b6, vl);
                vc7 = __riscv_vfmacc_vf_f64m1(vc7, a7, b7, vl);
            }
            for (; l < k; ++l) {
                target_float a = alpha * A[i * lda + l];
                vfloat64m1_t b = __riscv_vle64_v_f64m1(&B[l * ldb + j], vl);
                vc0 = __riscv_vfmacc_vf_f64m1(vc0, a, b, vl);
            }

            vfloat64m1_t vc01 = __riscv_vfadd_vv_f64m1(vc0, vc1, vl);
            vfloat64m1_t vc23 = __riscv_vfadd_vv_f64m1(vc2, vc3, vl);
            vfloat64m1_t vc45 = __riscv_vfadd_vv_f64m1(vc4, vc5, vl);
            vfloat64m1_t vc67 = __riscv_vfadd_vv_f64m1(vc6, vc7, vl);
            vfloat64m1_t vc0123 = __riscv_vfadd_vv_f64m1(vc01, vc23, vl);
            vfloat64m1_t vc4567 = __riscv_vfadd_vv_f64m1(vc45, vc67, vl);
            vfloat64m1_t vc = __riscv_vfadd_vv_f64m1(vc0123, vc4567, vl);

            __riscv_vse64_v_f64m1(&C[i * ldc + j], vc, vl);
#endif
            j += vl;
        }
    }
}

void print_matrix(const char *name, target_float *mat, int rows, int cols) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.1f ", (double)mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {

    target_float alpha = 1.0;
    target_float beta = 0.0;

    printf("misa = 0x%016lX\n", READ_CSR(misa));

    //test vector intrinsic
    {
        size_t vl = __riscv_vsetvl_e64m1(16);
        printf("vl = 0x%08X\n", vl);
        //vint32m1_t __riscv_vadd_vv_i32m1(vint32m1_t vs2, vint32m1_t vs1, size_t vl);
        vint64m1_t vs1, vs2;
        vint64m1_t vd = __riscv_vadd_vv_i64m1( vs2, vs1, vl);
    }

    WRITE_CSR(61, mhpmevent3);
    WRITE_CSR(64, mhpmevent4);
    WRITE_CSR(65, mhpmevent5);

    printf("Starting Scalar GEMM...\n\n");

    cycle_count   = READ_CSR(mcycle);
    inst_count    = READ_CSR(minstret);
    hpmcounter[3] = READ_CSR(mhpmcounter3);
    hpmcounter[4] = READ_CSR(mhpmcounter4);
    hpmcounter[5] = READ_CSR(mhpmcounter5);

    scalar_gemm(M, N, K, alpha, A, K, B, N, beta, C, N);

    cycle_count   = READ_CSR(mcycle)       - cycle_count;
    inst_count    = READ_CSR(minstret)     - inst_count;
    hpmcounter[3] = READ_CSR(mhpmcounter3) - hpmcounter[3];
    hpmcounter[4] = READ_CSR(mhpmcounter4) - hpmcounter[4];
    hpmcounter[5] = READ_CSR(mhpmcounter5) - hpmcounter[5];

    printf("counter: mcycle = %llu\n", cycle_count);
    printf("counter: minstret = %llu\n", inst_count);
    printf("hpmcounter[3]: Vector = %llu\n", hpmcounter[3]);
    printf("hpmcounter[4]: VectorLoad = %llu\n", hpmcounter[4]);
    printf("hpmcounter[5]: VectorStore = %llu\n", hpmcounter[5]);

    for (int idx = 0; idx < M * N; ++idx) C_ref[idx] = C[idx];

    printf("\nStarting Optimized GEMM (opt_gemm)...\n\n");

    cycle_count   = READ_CSR(mcycle);
    inst_count    = READ_CSR(minstret);
    hpmcounter[3] = READ_CSR(mhpmcounter3);
    hpmcounter[4] = READ_CSR(mhpmcounter4);
    hpmcounter[5] = READ_CSR(mhpmcounter5);

    opt_gemm(M, N, K, alpha, A, K, B, N, beta, C, N);

    cycle_count   = READ_CSR(mcycle)       - cycle_count;
    inst_count    = READ_CSR(minstret)     - inst_count;
    hpmcounter[3] = READ_CSR(mhpmcounter3) - hpmcounter[3];
    hpmcounter[4] = READ_CSR(mhpmcounter4) - hpmcounter[4];
    hpmcounter[5] = READ_CSR(mhpmcounter5) - hpmcounter[5];

#if 0
    print_matrix("A", A, M, K);
    print_matrix("B", B, K, N);
    print_matrix("C (Result)", C, M, N);
#endif

    {
        /* Unrolled opt_gemm sums OPT_GEMM_UNROLL partial accumulators in a
         * different order than scalar_gemm's strictly sequential l-loop, so
         * results can differ in the last bit or two (FP addition isn't
         * associative) — compare with a relative tolerance, not bit-exact. */
        double max_rel_diff = 0.0;
        for (int idx = 0; idx < M * N; ++idx) {
            double ref  = (double)C_ref[idx];
            double diff = (double)C[idx] - ref;
            if (diff < 0) diff = -diff;
            double denom = ref < 0 ? -ref : ref;
            if (denom < 1.0) denom = 1.0;
            double rel_diff = diff / denom;
            if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;
        }
        printf("opt_gemm vs scalar_gemm: max relative |C - C_ref| = %g (%s, tol 1e-2)\n",
               max_rel_diff, max_rel_diff < 1e-2 ? "PASS" : "FAIL");
    }

    printf("counter: mcycle = %llu\n", cycle_count);
    printf("counter: minstret = %llu\n", inst_count);
    printf("hpmcounter[3]: Vector = %llu\n", hpmcounter[3]);
    printf("hpmcounter[4]: VectorLoad = %llu\n", hpmcounter[4]);
    printf("hpmcounter[5]: VectorStore = %llu\n", hpmcounter[5]);

    return 0;
}
