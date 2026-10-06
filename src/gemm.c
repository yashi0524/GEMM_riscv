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
            size_t vl = (size_t)(n - j) >= __riscv_vsetvlmax_e16m1() ? __riscv_vsetvlmax_e16m1() : __riscv_vsetvl_e16m1(n - j);
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
            size_t vl = (size_t)(n - j) >= __riscv_vsetvlmax_e64m1() ? __riscv_vsetvlmax_e64m1() : __riscv_vsetvl_e64m1(n - j);
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

void opt_gemm_blocked(int , int , int ,
                      target_float , const target_float *, int ,
                      const target_float *, int ,
                      target_float , target_float *, int ) __attribute__((noinline));

#if M != 16
#error "opt_gemm_blocked hardcodes a 16-row block (M=16); rebuild with -DM=16 or generalize this function first"
#endif

/**
 * Row-blocked GEMM: C = alpha*(A*B) + beta*C
 * opt_gemm still reloads B[l,:] once per (row, l) -- redundant M=16x across
 * rows, since B doesn't depend on i at all. This version blocks the entire
 * M=16 rows into one tile: for each j-block, B[l,:] is loaded exactly once
 * per l and reused across all 16 rows' accumulators in that same
 * l-iteration, cutting B's reload factor from 16x down to 1x (matching the
 * theoretical minimum: B is loaded exactly K times per j-block, period).
 *
 * This also sidesteps the need for opt_gemm's separate OPT_GEMM_UNROLL
 * dependency-chain fix: the 16 independent per-row accumulators (vc0..vc15)
 * are updated once per l-iteration, so the pipeline already has 16-way
 * row-level ILP to hide FMA latency, without unrolling l at all.
 *
 * Register budget: 16 accumulators + 1 shared B register (+ scalar A
 * broadcasts, which live in scalar not vector registers) = 17 of RVV's 32
 * architectural vector registers -- comfortable. Combining this with
 * opt_gemm's k-unroll would need 16 x OPT_GEMM_UNROLL accumulators, which
 * doesn't fit for any unroll factor > 1, hence the two techniques are
 * exercised as separate functions rather than stacked.
 *
 * Hardcodes M=16 (this benchmark's fixed row count, like the A/B/C array
 * sizes above) rather than looping generically over row-blocks of M --
 * matches this file's existing convention of baking in the compile-time
 * M/N/K macros rather than handling arbitrary runtime shapes.
 *
 * fp16 vs. fp64 intrinsics are selected via #ifdef __riscv_zvfh, same as
 * opt_gemm.
 */
void opt_gemm_blocked(int m, int n, int k,
                      target_float alpha, const target_float *A, int lda,
                      const target_float *B, int ldb,
                      target_float beta, target_float *C, int ldc)
{
    for (int j = 0; j < n; ) {
#if defined(__riscv_zvfh)
        size_t vl = (size_t)(n - j) >= __riscv_vsetvlmax_e16m1() ? __riscv_vsetvlmax_e16m1() : __riscv_vsetvl_e16m1(n - j);
            vfloat16m1_t vc0 = __riscv_vfmul_vf_f16m1(__riscv_vle16_v_f16m1(&C[ 0 * ldc + j], vl), beta, vl);
            vfloat16m1_t vc1 = __riscv_vfmul_vf_f16m1(__riscv_vle16_v_f16m1(&C[ 1 * ldc + j], vl), beta, vl);
            vfloat16m1_t vc2 = __riscv_vfmul_vf_f16m1(__riscv_vle16_v_f16m1(&C[ 2 * ldc + j], vl), beta, vl);
            vfloat16m1_t vc3 = __riscv_vfmul_vf_f16m1(__riscv_vle16_v_f16m1(&C[ 3 * ldc + j], vl), beta, vl);
            vfloat16m1_t vc4 = __riscv_vfmul_vf_f16m1(__riscv_vle16_v_f16m1(&C[ 4 * ldc + j], vl), beta, vl);
            vfloat16m1_t vc5 = __riscv_vfmul_vf_f16m1(__riscv_vle16_v_f16m1(&C[ 5 * ldc + j], vl), beta, vl);
            vfloat16m1_t vc6 = __riscv_vfmul_vf_f16m1(__riscv_vle16_v_f16m1(&C[ 6 * ldc + j], vl), beta, vl);
            vfloat16m1_t vc7 = __riscv_vfmul_vf_f16m1(__riscv_vle16_v_f16m1(&C[ 7 * ldc + j], vl), beta, vl);
            vfloat16m1_t vc8 = __riscv_vfmul_vf_f16m1(__riscv_vle16_v_f16m1(&C[ 8 * ldc + j], vl), beta, vl);
            vfloat16m1_t vc9 = __riscv_vfmul_vf_f16m1(__riscv_vle16_v_f16m1(&C[ 9 * ldc + j], vl), beta, vl);
            vfloat16m1_t vc10 = __riscv_vfmul_vf_f16m1(__riscv_vle16_v_f16m1(&C[10 * ldc + j], vl), beta, vl);
            vfloat16m1_t vc11 = __riscv_vfmul_vf_f16m1(__riscv_vle16_v_f16m1(&C[11 * ldc + j], vl), beta, vl);
            vfloat16m1_t vc12 = __riscv_vfmul_vf_f16m1(__riscv_vle16_v_f16m1(&C[12 * ldc + j], vl), beta, vl);
            vfloat16m1_t vc13 = __riscv_vfmul_vf_f16m1(__riscv_vle16_v_f16m1(&C[13 * ldc + j], vl), beta, vl);
            vfloat16m1_t vc14 = __riscv_vfmul_vf_f16m1(__riscv_vle16_v_f16m1(&C[14 * ldc + j], vl), beta, vl);
            vfloat16m1_t vc15 = __riscv_vfmul_vf_f16m1(__riscv_vle16_v_f16m1(&C[15 * ldc + j], vl), beta, vl);

            for (int l = 0; l < k; ++l) {
                vfloat16m1_t vb = __riscv_vle16_v_f16m1(&B[l * ldb + j], vl);
                target_float a0 = alpha * A[ 0 * lda + l];
                target_float a1 = alpha * A[ 1 * lda + l];
                target_float a2 = alpha * A[ 2 * lda + l];
                target_float a3 = alpha * A[ 3 * lda + l];
                target_float a4 = alpha * A[ 4 * lda + l];
                target_float a5 = alpha * A[ 5 * lda + l];
                target_float a6 = alpha * A[ 6 * lda + l];
                target_float a7 = alpha * A[ 7 * lda + l];
                target_float a8 = alpha * A[ 8 * lda + l];
                target_float a9 = alpha * A[ 9 * lda + l];
                target_float a10 = alpha * A[10 * lda + l];
                target_float a11 = alpha * A[11 * lda + l];
                target_float a12 = alpha * A[12 * lda + l];
                target_float a13 = alpha * A[13 * lda + l];
                target_float a14 = alpha * A[14 * lda + l];
                target_float a15 = alpha * A[15 * lda + l];
                vc0 = __riscv_vfmacc_vf_f16m1(vc0, a0, vb, vl);
                vc1 = __riscv_vfmacc_vf_f16m1(vc1, a1, vb, vl);
                vc2 = __riscv_vfmacc_vf_f16m1(vc2, a2, vb, vl);
                vc3 = __riscv_vfmacc_vf_f16m1(vc3, a3, vb, vl);
                vc4 = __riscv_vfmacc_vf_f16m1(vc4, a4, vb, vl);
                vc5 = __riscv_vfmacc_vf_f16m1(vc5, a5, vb, vl);
                vc6 = __riscv_vfmacc_vf_f16m1(vc6, a6, vb, vl);
                vc7 = __riscv_vfmacc_vf_f16m1(vc7, a7, vb, vl);
                vc8 = __riscv_vfmacc_vf_f16m1(vc8, a8, vb, vl);
                vc9 = __riscv_vfmacc_vf_f16m1(vc9, a9, vb, vl);
                vc10 = __riscv_vfmacc_vf_f16m1(vc10, a10, vb, vl);
                vc11 = __riscv_vfmacc_vf_f16m1(vc11, a11, vb, vl);
                vc12 = __riscv_vfmacc_vf_f16m1(vc12, a12, vb, vl);
                vc13 = __riscv_vfmacc_vf_f16m1(vc13, a13, vb, vl);
                vc14 = __riscv_vfmacc_vf_f16m1(vc14, a14, vb, vl);
                vc15 = __riscv_vfmacc_vf_f16m1(vc15, a15, vb, vl);
            }

            __riscv_vse16_v_f16m1(&C[ 0 * ldc + j], vc0, vl);
            __riscv_vse16_v_f16m1(&C[ 1 * ldc + j], vc1, vl);
            __riscv_vse16_v_f16m1(&C[ 2 * ldc + j], vc2, vl);
            __riscv_vse16_v_f16m1(&C[ 3 * ldc + j], vc3, vl);
            __riscv_vse16_v_f16m1(&C[ 4 * ldc + j], vc4, vl);
            __riscv_vse16_v_f16m1(&C[ 5 * ldc + j], vc5, vl);
            __riscv_vse16_v_f16m1(&C[ 6 * ldc + j], vc6, vl);
            __riscv_vse16_v_f16m1(&C[ 7 * ldc + j], vc7, vl);
            __riscv_vse16_v_f16m1(&C[ 8 * ldc + j], vc8, vl);
            __riscv_vse16_v_f16m1(&C[ 9 * ldc + j], vc9, vl);
            __riscv_vse16_v_f16m1(&C[10 * ldc + j], vc10, vl);
            __riscv_vse16_v_f16m1(&C[11 * ldc + j], vc11, vl);
            __riscv_vse16_v_f16m1(&C[12 * ldc + j], vc12, vl);
            __riscv_vse16_v_f16m1(&C[13 * ldc + j], vc13, vl);
            __riscv_vse16_v_f16m1(&C[14 * ldc + j], vc14, vl);
            __riscv_vse16_v_f16m1(&C[15 * ldc + j], vc15, vl);
#else
        size_t vl = (size_t)(n - j) >= __riscv_vsetvlmax_e64m1() ? __riscv_vsetvlmax_e64m1() : __riscv_vsetvl_e64m1(n - j);
            vfloat64m1_t vc0 = __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&C[ 0 * ldc + j], vl), beta, vl);
            vfloat64m1_t vc1 = __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&C[ 1 * ldc + j], vl), beta, vl);
            vfloat64m1_t vc2 = __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&C[ 2 * ldc + j], vl), beta, vl);
            vfloat64m1_t vc3 = __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&C[ 3 * ldc + j], vl), beta, vl);
            vfloat64m1_t vc4 = __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&C[ 4 * ldc + j], vl), beta, vl);
            vfloat64m1_t vc5 = __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&C[ 5 * ldc + j], vl), beta, vl);
            vfloat64m1_t vc6 = __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&C[ 6 * ldc + j], vl), beta, vl);
            vfloat64m1_t vc7 = __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&C[ 7 * ldc + j], vl), beta, vl);
            vfloat64m1_t vc8 = __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&C[ 8 * ldc + j], vl), beta, vl);
            vfloat64m1_t vc9 = __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&C[ 9 * ldc + j], vl), beta, vl);
            vfloat64m1_t vc10 = __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&C[10 * ldc + j], vl), beta, vl);
            vfloat64m1_t vc11 = __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&C[11 * ldc + j], vl), beta, vl);
            vfloat64m1_t vc12 = __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&C[12 * ldc + j], vl), beta, vl);
            vfloat64m1_t vc13 = __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&C[13 * ldc + j], vl), beta, vl);
            vfloat64m1_t vc14 = __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&C[14 * ldc + j], vl), beta, vl);
            vfloat64m1_t vc15 = __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&C[15 * ldc + j], vl), beta, vl);

            for (int l = 0; l < k; ++l) {
                vfloat64m1_t vb = __riscv_vle64_v_f64m1(&B[l * ldb + j], vl);
                target_float a0 = alpha * A[ 0 * lda + l];
                target_float a1 = alpha * A[ 1 * lda + l];
                target_float a2 = alpha * A[ 2 * lda + l];
                target_float a3 = alpha * A[ 3 * lda + l];
                target_float a4 = alpha * A[ 4 * lda + l];
                target_float a5 = alpha * A[ 5 * lda + l];
                target_float a6 = alpha * A[ 6 * lda + l];
                target_float a7 = alpha * A[ 7 * lda + l];
                target_float a8 = alpha * A[ 8 * lda + l];
                target_float a9 = alpha * A[ 9 * lda + l];
                target_float a10 = alpha * A[10 * lda + l];
                target_float a11 = alpha * A[11 * lda + l];
                target_float a12 = alpha * A[12 * lda + l];
                target_float a13 = alpha * A[13 * lda + l];
                target_float a14 = alpha * A[14 * lda + l];
                target_float a15 = alpha * A[15 * lda + l];
                vc0 = __riscv_vfmacc_vf_f64m1(vc0, a0, vb, vl);
                vc1 = __riscv_vfmacc_vf_f64m1(vc1, a1, vb, vl);
                vc2 = __riscv_vfmacc_vf_f64m1(vc2, a2, vb, vl);
                vc3 = __riscv_vfmacc_vf_f64m1(vc3, a3, vb, vl);
                vc4 = __riscv_vfmacc_vf_f64m1(vc4, a4, vb, vl);
                vc5 = __riscv_vfmacc_vf_f64m1(vc5, a5, vb, vl);
                vc6 = __riscv_vfmacc_vf_f64m1(vc6, a6, vb, vl);
                vc7 = __riscv_vfmacc_vf_f64m1(vc7, a7, vb, vl);
                vc8 = __riscv_vfmacc_vf_f64m1(vc8, a8, vb, vl);
                vc9 = __riscv_vfmacc_vf_f64m1(vc9, a9, vb, vl);
                vc10 = __riscv_vfmacc_vf_f64m1(vc10, a10, vb, vl);
                vc11 = __riscv_vfmacc_vf_f64m1(vc11, a11, vb, vl);
                vc12 = __riscv_vfmacc_vf_f64m1(vc12, a12, vb, vl);
                vc13 = __riscv_vfmacc_vf_f64m1(vc13, a13, vb, vl);
                vc14 = __riscv_vfmacc_vf_f64m1(vc14, a14, vb, vl);
                vc15 = __riscv_vfmacc_vf_f64m1(vc15, a15, vb, vl);
            }

            __riscv_vse64_v_f64m1(&C[ 0 * ldc + j], vc0, vl);
            __riscv_vse64_v_f64m1(&C[ 1 * ldc + j], vc1, vl);
            __riscv_vse64_v_f64m1(&C[ 2 * ldc + j], vc2, vl);
            __riscv_vse64_v_f64m1(&C[ 3 * ldc + j], vc3, vl);
            __riscv_vse64_v_f64m1(&C[ 4 * ldc + j], vc4, vl);
            __riscv_vse64_v_f64m1(&C[ 5 * ldc + j], vc5, vl);
            __riscv_vse64_v_f64m1(&C[ 6 * ldc + j], vc6, vl);
            __riscv_vse64_v_f64m1(&C[ 7 * ldc + j], vc7, vl);
            __riscv_vse64_v_f64m1(&C[ 8 * ldc + j], vc8, vl);
            __riscv_vse64_v_f64m1(&C[ 9 * ldc + j], vc9, vl);
            __riscv_vse64_v_f64m1(&C[10 * ldc + j], vc10, vl);
            __riscv_vse64_v_f64m1(&C[11 * ldc + j], vc11, vl);
            __riscv_vse64_v_f64m1(&C[12 * ldc + j], vc12, vl);
            __riscv_vse64_v_f64m1(&C[13 * ldc + j], vc13, vl);
            __riscv_vse64_v_f64m1(&C[14 * ldc + j], vc14, vl);
            __riscv_vse64_v_f64m1(&C[15 * ldc + j], vc15, vl);
#endif
        j += vl;
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

/* Compares the current C against the C_ref snapshot taken right after
 * scalar_gemm ran; prints a relative-tolerance PASS/FAIL verdict. */
void report_correctness(const char *label) {
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
    printf("%s vs scalar_gemm: max relative |C - C_ref| = %g (%s, tol 1e-2)\n",
           label, max_rel_diff, max_rel_diff < 1e-2 ? "PASS" : "FAIL");
}

#ifdef ROOFLINE_KERNEL
/* ROOFLINE_KERNEL (gem5 only: the m5 pseudo-ops below are illegal
 * instructions on whisper / real hardware): 1 = scalar_gemm, 2 = opt_gemm,
 * 3 = opt_gemm_blocked. main() runs only that kernel and brackets each phase
 * with m5_reset_stats / m5_dump_stats, so stats.txt holds one section per
 * phase, in this order:
 *   dump 1  cold call   - A/B/C evicted first, so D-cache fills = the
 *                         kernel's DRAM read traffic
 *   dump 2  evict       - pushes C's dirty lines out: D-cache writebacks =
 *                         the kernel's (deferred) DRAM write traffic
 *   dump 3  warm call   - after one untimed warm-up call
 * (gem5 appends a 4th section at exit; ignore it.) The startup code touches
 * A/B/C (.data copy / .bss zero), so without the eviction they would already
 * be cached. See doc/gemm_analysis.md, "How to generate the roofline". */
#if ROOFLINE_KERNEL == 1
#define RL_KERNEL scalar_gemm
#define RL_NAME   "scalar_gemm"
#elif ROOFLINE_KERNEL == 2
#define RL_KERNEL opt_gemm
#define RL_NAME   "opt_gemm"
#elif ROOFLINE_KERNEL == 3
#define RL_KERNEL opt_gemm_blocked
#define RL_NAME   "opt_gemm_blocked"
#else
#error "ROOFLINE_KERNEL must be 1, 2 or 3"
#endif

/* gem5 RISC-V m5op: .word 0x7b | (func << 25), args in a0 (delay), a1 (period) */
#define M5OP(func) __asm__ volatile("li a0, 0\n\tli a1, 0\n\t.word %0" \
                                    :: "i"(0x7b | ((func) << 25)) : "a0", "a1", "memory")
#define m5_reset_stats() M5OP(0x40)
#define m5_dump_stats()  M5OP(0x41)

/* 2x the 64 KiB 4-way D-cache: reading it once evicts every older line */
#define EVICT_BYTES (128 * 1024)
static unsigned char evict_buf[EVICT_BYTES] __attribute__((aligned(64)));
static volatile unsigned long evict_sink;

static void evict_dcache(void) {
    /* volatile: evict_buf is never written, so plain loads fold to 0 */
    const volatile unsigned char *p = evict_buf;
    unsigned long sum = 0;
    for (int i = 0; i < EVICT_BYTES; i += 64) sum += p[i];
    evict_sink = sum;
}

/* printf only after the last dump: its buffer writes would otherwise show
 * up as dirty-line writebacks in the evict phase */
static unsigned long long rl_cycles[2], rl_insts[2];

static void rl_call(int idx, target_float alpha, target_float beta) {
    m5_reset_stats();
    cycle_count = READ_CSR(mcycle);
    inst_count  = READ_CSR(minstret);
    RL_KERNEL(M, N, K, alpha, A, K, B, N, beta, C, N);
    cycle_count = READ_CSR(mcycle)   - cycle_count;
    inst_count  = READ_CSR(minstret) - inst_count;
    m5_dump_stats();
    rl_cycles[idx] = cycle_count;
    rl_insts[idx]  = inst_count;
}

static int roofline_main(void) {
    target_float alpha = 1.0;
    target_float beta  = 0.0;

    evict_dcache();
    rl_call(0, alpha, beta);               /* dump 1: cold */

    m5_reset_stats();
    evict_dcache();
    m5_dump_stats();                       /* dump 2 */

    RL_KERNEL(M, N, K, alpha, A, K, B, N, beta, C, N);   /* warm-up */
    rl_call(1, alpha, beta);               /* dump 3: warm */

    printf("roofline %s cold: mcycle = %llu minstret = %llu\n", RL_NAME, rl_cycles[0], rl_insts[0]);
    printf("roofline %s warm: mcycle = %llu minstret = %llu\n", RL_NAME, rl_cycles[1], rl_insts[1]);

    /* correctness: kernel result -> C_ref, scalar reference -> C */
    for (int idx = 0; idx < M * N; ++idx) C_ref[idx] = C[idx];
    scalar_gemm(M, N, K, alpha, A, K, B, N, beta, C, N);
    report_correctness(RL_NAME " (reference = scalar_gemm)");
    return 0;
}
#endif /* ROOFLINE_KERNEL */

/* WARMUP_RUNS: untimed calls of each kernel before its measured call, so the
 * measurement sees warm I-/D-caches and branch predictor instead of
 * first-touch costs. Every kernel fully overwrites C (beta = 0), so the
 * extra calls don't change the result. Default 0 = original cold behavior. */
#ifndef WARMUP_RUNS
#define WARMUP_RUNS 0
#endif

int main() {
#ifdef ROOFLINE_KERNEL
    return roofline_main();
#endif

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

    for (int w = 0; w < WARMUP_RUNS; ++w)
        scalar_gemm(M, N, K, alpha, A, K, B, N, beta, C, N);

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

    for (int w = 0; w < WARMUP_RUNS; ++w)
        opt_gemm(M, N, K, alpha, A, K, B, N, beta, C, N);

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

    /* Unrolled/blocked variants sum partial accumulators in a different
     * order than scalar_gemm's strictly sequential l-loop, so results can
     * differ in the last bit or two (FP addition isn't associative) —
     * compare with a relative tolerance, not bit-exact. */
    report_correctness("opt_gemm");

    printf("counter: mcycle = %llu\n", cycle_count);
    printf("counter: minstret = %llu\n", inst_count);
    printf("hpmcounter[3]: Vector = %llu\n", hpmcounter[3]);
    printf("hpmcounter[4]: VectorLoad = %llu\n", hpmcounter[4]);
    printf("hpmcounter[5]: VectorStore = %llu\n", hpmcounter[5]);

    printf("\nStarting Row-Blocked GEMM (opt_gemm_blocked)...\n\n");

    for (int w = 0; w < WARMUP_RUNS; ++w)
        opt_gemm_blocked(M, N, K, alpha, A, K, B, N, beta, C, N);

    cycle_count   = READ_CSR(mcycle);
    inst_count    = READ_CSR(minstret);
    hpmcounter[3] = READ_CSR(mhpmcounter3);
    hpmcounter[4] = READ_CSR(mhpmcounter4);
    hpmcounter[5] = READ_CSR(mhpmcounter5);

    opt_gemm_blocked(M, N, K, alpha, A, K, B, N, beta, C, N);

    cycle_count   = READ_CSR(mcycle)       - cycle_count;
    inst_count    = READ_CSR(minstret)     - inst_count;
    hpmcounter[3] = READ_CSR(mhpmcounter3) - hpmcounter[3];
    hpmcounter[4] = READ_CSR(mhpmcounter4) - hpmcounter[4];
    hpmcounter[5] = READ_CSR(mhpmcounter5) - hpmcounter[5];

    report_correctness("opt_gemm_blocked");

    printf("counter: mcycle = %llu\n", cycle_count);
    printf("counter: minstret = %llu\n", inst_count);
    printf("hpmcounter[3]: Vector = %llu\n", hpmcounter[3]);
    printf("hpmcounter[4]: VectorLoad = %llu\n", hpmcounter[4]);
    printf("hpmcounter[5]: VectorStore = %llu\n", hpmcounter[5]);

    return 0;
}
