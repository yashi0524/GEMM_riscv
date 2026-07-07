#include <stdio.h>
#include <riscv_vector.h>
#include "platform.h"

#ifndef ITERS
#define ITERS 10000
#endif

int main(void)
{
    printf("misa = 0x%016lX\n", READ_CSR(misa));
    printf("Starting FMACC micro-benchmark...\n\n");

    /* Hardware-max vl for LMUL=1 SEW=64: VLEN=512 -> vl=8 */
    size_t vl = __riscv_vsetvlmax_e64m1();
    printf("vl = %zu\n", vl);
    printf("ITERS = %d\n\n", ITERS);

    /* va=vb=1.0, vcN=0.0 -> each vcN grows by 1.0 per iteration */
    vfloat64m1_t va  = __riscv_vfmv_v_f_f64m1(1.0, vl);
    vfloat64m1_t vb  = __riscv_vfmv_v_f_f64m1(1.0, vl);
    vfloat64m1_t vc0  = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vc1  = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vc2  = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vc3  = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vc4  = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vc5  = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vc6  = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vc7  = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vc8  = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vc9  = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vc10 = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vc11 = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vc12 = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vc13 = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vc14 = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vc15 = __riscv_vfmv_v_f_f64m1(0.0, vl);

    WRITE_CSR(61, mhpmevent3);  /* Vector compute (event 61) */

    unsigned long long t_cycle = READ_CSR(mcycle);
    unsigned long long t_inst  = READ_CSR(minstret);
    unsigned long long t_vec   = READ_CSR(mhpmcounter3);

    /* Unrolled x16: 16 independent accumulators break the vc->vc dependency
     * chain, so the pipeline can overlap FMAs instead of serializing them.
     * ITERS need not be a multiple of 16; total_ops (below) tracks the real
     * count actually executed (this loop overshoots to a multiple of 16). */
    for (int i = 0; i < ITERS; i += 16) {
        vc0  = __riscv_vfmacc_vv_f64m1(vc0,  va, vb, vl);
        vc1  = __riscv_vfmacc_vv_f64m1(vc1,  va, vb, vl);
        vc2  = __riscv_vfmacc_vv_f64m1(vc2,  va, vb, vl);
        vc3  = __riscv_vfmacc_vv_f64m1(vc3,  va, vb, vl);
        vc4  = __riscv_vfmacc_vv_f64m1(vc4,  va, vb, vl);
        vc5  = __riscv_vfmacc_vv_f64m1(vc5,  va, vb, vl);
        vc6  = __riscv_vfmacc_vv_f64m1(vc6,  va, vb, vl);
        vc7  = __riscv_vfmacc_vv_f64m1(vc7,  va, vb, vl);
        vc8  = __riscv_vfmacc_vv_f64m1(vc8,  va, vb, vl);
        vc9  = __riscv_vfmacc_vv_f64m1(vc9,  va, vb, vl);
        vc10 = __riscv_vfmacc_vv_f64m1(vc10, va, vb, vl);
        vc11 = __riscv_vfmacc_vv_f64m1(vc11, va, vb, vl);
        vc12 = __riscv_vfmacc_vv_f64m1(vc12, va, vb, vl);
        vc13 = __riscv_vfmacc_vv_f64m1(vc13, va, vb, vl);
        vc14 = __riscv_vfmacc_vv_f64m1(vc14, va, vb, vl);
        vc15 = __riscv_vfmacc_vv_f64m1(vc15, va, vb, vl);
    }

    t_cycle = READ_CSR(mcycle)       - t_cycle;
    t_inst  = READ_CSR(minstret)     - t_inst;
    t_vec   = READ_CSR(mhpmcounter3) - t_vec;

    /* Combine accumulators outside the timed region.
     * Prevent dead-code elimination; result[0] should equal total_ops. */
    long total_ops = ((ITERS + 15) / 16) * 16;  /* actual FMAs executed */
    vfloat64m1_t vc = __riscv_vfadd_vv_f64m1(vc0, vc1, vl);
    vc = __riscv_vfadd_vv_f64m1(vc, vc2, vl);
    vc = __riscv_vfadd_vv_f64m1(vc, vc3, vl);
    vc = __riscv_vfadd_vv_f64m1(vc, vc4, vl);
    vc = __riscv_vfadd_vv_f64m1(vc, vc5, vl);
    vc = __riscv_vfadd_vv_f64m1(vc, vc6, vl);
    vc = __riscv_vfadd_vv_f64m1(vc, vc7, vl);
    vc = __riscv_vfadd_vv_f64m1(vc, vc8, vl);
    vc = __riscv_vfadd_vv_f64m1(vc, vc9, vl);
    vc = __riscv_vfadd_vv_f64m1(vc, vc10, vl);
    vc = __riscv_vfadd_vv_f64m1(vc, vc11, vl);
    vc = __riscv_vfadd_vv_f64m1(vc, vc12, vl);
    vc = __riscv_vfadd_vv_f64m1(vc, vc13, vl);
    vc = __riscv_vfadd_vv_f64m1(vc, vc14, vl);
    vc = __riscv_vfadd_vv_f64m1(vc, vc15, vl);

    double result_arr[8];
    __riscv_vse64_v_f64m1(result_arr, vc, vl);
    printf("total_ops = %ld\n", total_ops);
    printf("result[0] = %.1f\n\n", result_arr[0]);

    printf("counter: mcycle = %llu\n",       t_cycle);
    printf("counter: minstret = %llu\n",     t_inst);
    printf("hpmcounter[3]: Vector = %llu\n", t_vec);

    return 0;
}
