#include <stdio.h>
#include <riscv_vector.h>
#include "platform.h"

#ifndef ITERS
#define ITERS 10000
#endif

int main(void)
{
    printf("misa = 0x%016lX\n", READ_CSR(misa));
    printf("Starting FMADD micro-benchmark...\n\n");

    /* Hardware-max vl for LMUL=1 SEW=64: VLEN=512 -> vl=8 */
    size_t vl = __riscv_vsetvlmax_e64m1();
    printf("vl = %zu\n", vl);
    printf("ITERS = %d\n\n", ITERS);

    /* va=vb=1.0, vc=0.0 -> vc grows by 1.0 per iteration; result[0] == ITERS */
    vfloat64m1_t va = __riscv_vfmv_v_f_f64m1(1.0, vl);
    vfloat64m1_t vb = __riscv_vfmv_v_f_f64m1(1.0, vl);
    vfloat64m1_t vc = __riscv_vfmv_v_f_f64m1(0.0, vl);

    WRITE_CSR(61, mhpmevent3);  /* Vector compute (event 61) */

    unsigned long long t_cycle = READ_CSR(mcycle);
    unsigned long long t_inst  = READ_CSR(minstret);
    unsigned long long t_vec   = READ_CSR(mhpmcounter3);

    for (int i = 0; i < ITERS; ++i)
        vc = __riscv_vfmacc_vv_f64m1(vc, va, vb, vl);  /* vc = va*vb + vc */

    t_cycle = READ_CSR(mcycle)       - t_cycle;
    t_inst  = READ_CSR(minstret)     - t_inst;
    t_vec   = READ_CSR(mhpmcounter3) - t_vec;

    /* Prevent dead-code elimination; result[0] should equal (double)ITERS */
    double result_arr[8];
    __riscv_vse64_v_f64m1(result_arr, vc, vl);
    printf("result[0] = %.1f\n\n", result_arr[0]);

    printf("counter: mcycle = %llu\n",       t_cycle);
    printf("counter: minstret = %llu\n",     t_inst);
    printf("hpmcounter[3]: Vector = %llu\n", t_vec);

    return 0;
}
