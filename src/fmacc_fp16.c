#include <stdio.h>
#include <riscv_vector.h>
#include "platform.h"

#ifndef ITERS
#define ITERS 10000
#endif

/* Unrolled x16: 16 independent accumulators break the vc->vc dependency
 * chain, so the pipeline can overlap FMAs instead of serializing them.
 * ITERS need not be a multiple of 16; total_ops (below) tracks the real
 * count actually executed (this loop overshoots to a multiple of 16). */
static void fmacc_fp16_unroll_16(void)
{
    /* Hardware-max vl for LMUL=1 SEW=16: VLEN=512 -> vl=32 */
    size_t vl = __riscv_vsetvlmax_e16m1();
    printf("--- FMACC FP16 unroll x16 ---\n");
    printf("vl = %zu\n", vl);
    printf("ITERS = %d\n\n", ITERS);

    /* va=vb=1.0, vcN=0.0 -> each vcN grows by 1.0 per iteration */
    vfloat16m1_t va  = __riscv_vfmv_v_f_f16m1(1.0, vl);
    vfloat16m1_t vb  = __riscv_vfmv_v_f_f16m1(1.0, vl);
    vfloat16m1_t vc0  = __riscv_vfmv_v_f_f16m1(0.0, vl);
    vfloat16m1_t vc1  = __riscv_vfmv_v_f_f16m1(0.0, vl);
    vfloat16m1_t vc2  = __riscv_vfmv_v_f_f16m1(0.0, vl);
    vfloat16m1_t vc3  = __riscv_vfmv_v_f_f16m1(0.0, vl);
    vfloat16m1_t vc4  = __riscv_vfmv_v_f_f16m1(0.0, vl);
    vfloat16m1_t vc5  = __riscv_vfmv_v_f_f16m1(0.0, vl);
    vfloat16m1_t vc6  = __riscv_vfmv_v_f_f16m1(0.0, vl);
    vfloat16m1_t vc7  = __riscv_vfmv_v_f_f16m1(0.0, vl);
    vfloat16m1_t vc8  = __riscv_vfmv_v_f_f16m1(0.0, vl);
    vfloat16m1_t vc9  = __riscv_vfmv_v_f_f16m1(0.0, vl);
    vfloat16m1_t vc10 = __riscv_vfmv_v_f_f16m1(0.0, vl);
    vfloat16m1_t vc11 = __riscv_vfmv_v_f_f16m1(0.0, vl);
    vfloat16m1_t vc12 = __riscv_vfmv_v_f_f16m1(0.0, vl);
    vfloat16m1_t vc13 = __riscv_vfmv_v_f_f16m1(0.0, vl);
    vfloat16m1_t vc14 = __riscv_vfmv_v_f_f16m1(0.0, vl);
    vfloat16m1_t vc15 = __riscv_vfmv_v_f_f16m1(0.0, vl);

    WRITE_CSR(61, mhpmevent3);  /* Vector compute (event 61) */

    unsigned long long t_cycle = READ_CSR(mcycle);
    unsigned long long t_inst  = READ_CSR(minstret);
    unsigned long long t_vec   = READ_CSR(mhpmcounter3);

    for (int i = 0; i < ITERS; i += 16) {
        vc0  = __riscv_vfmacc_vv_f16m1(vc0,  va, vb, vl);
        vc1  = __riscv_vfmacc_vv_f16m1(vc1,  va, vb, vl);
        vc2  = __riscv_vfmacc_vv_f16m1(vc2,  va, vb, vl);
        vc3  = __riscv_vfmacc_vv_f16m1(vc3,  va, vb, vl);
        vc4  = __riscv_vfmacc_vv_f16m1(vc4,  va, vb, vl);
        vc5  = __riscv_vfmacc_vv_f16m1(vc5,  va, vb, vl);
        vc6  = __riscv_vfmacc_vv_f16m1(vc6,  va, vb, vl);
        vc7  = __riscv_vfmacc_vv_f16m1(vc7,  va, vb, vl);
        vc8  = __riscv_vfmacc_vv_f16m1(vc8,  va, vb, vl);
        vc9  = __riscv_vfmacc_vv_f16m1(vc9,  va, vb, vl);
        vc10 = __riscv_vfmacc_vv_f16m1(vc10, va, vb, vl);
        vc11 = __riscv_vfmacc_vv_f16m1(vc11, va, vb, vl);
        vc12 = __riscv_vfmacc_vv_f16m1(vc12, va, vb, vl);
        vc13 = __riscv_vfmacc_vv_f16m1(vc13, va, vb, vl);
        vc14 = __riscv_vfmacc_vv_f16m1(vc14, va, vb, vl);
        vc15 = __riscv_vfmacc_vv_f16m1(vc15, va, vb, vl);
    }

    t_cycle = READ_CSR(mcycle)       - t_cycle;
    t_inst  = READ_CSR(minstret)     - t_inst;
    t_vec   = READ_CSR(mhpmcounter3) - t_vec;

    /* Combine accumulators outside the timed region, via a pairwise tree
     * (not a linear chain): FP16 has only 10 mantissa bits, so a linear sum
     * (625, 1250, 1875, 2500, 3125, ...) hits odd partial sums that aren't
     * exactly representable once the running total exceeds 2048, rounding
     * away from total_ops. A tree keeps every partial sum a power-of-two
     * multiple of 625 (625/1250/2500/5000/10000), each exactly representable
     * in FP16 at its magnitude, so the final result is exact.
     * Prevent dead-code elimination; result[0] should equal total_ops. */
    long total_ops = ((ITERS + 15) / 16) * 16;  /* actual FMAs executed */
    vfloat16m1_t vc01   = __riscv_vfadd_vv_f16m1(vc0,  vc1,  vl);
    vfloat16m1_t vc23   = __riscv_vfadd_vv_f16m1(vc2,  vc3,  vl);
    vfloat16m1_t vc45   = __riscv_vfadd_vv_f16m1(vc4,  vc5,  vl);
    vfloat16m1_t vc67   = __riscv_vfadd_vv_f16m1(vc6,  vc7,  vl);
    vfloat16m1_t vc89   = __riscv_vfadd_vv_f16m1(vc8,  vc9,  vl);
    vfloat16m1_t vc1011 = __riscv_vfadd_vv_f16m1(vc10, vc11, vl);
    vfloat16m1_t vc1213 = __riscv_vfadd_vv_f16m1(vc12, vc13, vl);
    vfloat16m1_t vc1415 = __riscv_vfadd_vv_f16m1(vc14, vc15, vl);

    vfloat16m1_t vc0123     = __riscv_vfadd_vv_f16m1(vc01,   vc23,   vl);
    vfloat16m1_t vc4567     = __riscv_vfadd_vv_f16m1(vc45,   vc67,   vl);
    vfloat16m1_t vc891011   = __riscv_vfadd_vv_f16m1(vc89,   vc1011, vl);
    vfloat16m1_t vc12131415 = __riscv_vfadd_vv_f16m1(vc1213, vc1415, vl);

    vfloat16m1_t vc01234567     = __riscv_vfadd_vv_f16m1(vc0123,   vc4567,     vl);
    vfloat16m1_t vc89101112131415 = __riscv_vfadd_vv_f16m1(vc891011, vc12131415, vl);

    vfloat16m1_t vc = __riscv_vfadd_vv_f16m1(vc01234567, vc89101112131415, vl);

    _Float16 result_arr[32];
    __riscv_vse16_v_f16m1(result_arr, vc, vl);
    printf("total_ops = %ld\n", total_ops);
    printf("result[0] = %.1f\n\n", (double)result_arr[0]);

    printf("counter: mcycle = %llu\n",       t_cycle);
    printf("counter: minstret = %llu\n",     t_inst);
    printf("hpmcounter[3]: Vector = %llu\n", t_vec);
}

/* Serial x1: a single accumulator makes every FMA depend on the result of
 * the previous one, so the pipeline cannot overlap iterations. This measures
 * back-to-back FMA latency, in contrast with fmacc_fp16_unroll_16 which
 * measures throughput across 16 independent chains. */
static void fmacc_fp16_serial(void)
{
    size_t vl = __riscv_vsetvlmax_e16m1();
    printf("--- FMACC FP16 serial x1 ---\n");
    printf("vl = %zu\n", vl);
    printf("ITERS = %d\n\n", ITERS);

    /* va=vb=1.0, vc=0.0 -> vc grows by 1.0 per iteration */
    vfloat16m1_t va = __riscv_vfmv_v_f_f16m1(1.0, vl);
    vfloat16m1_t vb = __riscv_vfmv_v_f_f16m1(1.0, vl);
    vfloat16m1_t vc = __riscv_vfmv_v_f_f16m1(0.0, vl);

    WRITE_CSR(61, mhpmevent3);  /* Vector compute (event 61) */

    unsigned long long t_cycle = READ_CSR(mcycle);
    unsigned long long t_inst  = READ_CSR(minstret);
    unsigned long long t_vec   = READ_CSR(mhpmcounter3);

    for (int i = 0; i < ITERS; i++) {
        vc = __riscv_vfmacc_vv_f16m1(vc, va, vb, vl);
    }

    t_cycle = READ_CSR(mcycle)       - t_cycle;
    t_inst  = READ_CSR(minstret)     - t_inst;
    t_vec   = READ_CSR(mhpmcounter3) - t_vec;

    /* Prevent dead-code elimination; result[0] approaches total_ops but,
     * unlike the tree-combined unroll_16 result, loses precision past 2048
     * since this is a genuine linear FP16 accumulation of 1.0's. */
    long total_ops = ITERS;
    _Float16 result_arr[32];
    __riscv_vse16_v_f16m1(result_arr, vc, vl);
    printf("total_ops = %ld\n", total_ops);
    printf("result[0] = %.1f\n\n", (double)result_arr[0]);

    printf("counter: mcycle = %llu\n",       t_cycle);
    printf("counter: minstret = %llu\n",     t_inst);
    printf("hpmcounter[3]: Vector = %llu\n", t_vec);
}

int main(void)
{
    printf("misa = 0x%016lX\n", READ_CSR(misa));
    printf("Starting FMACC FP16 micro-benchmark...\n\n");

    fmacc_fp16_unroll_16();
    fmacc_fp16_serial();

    return 0;
}
