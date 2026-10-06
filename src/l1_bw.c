#include <stdio.h>
#include <riscv_vector.h>
#include "platform.h"

/* L1 vector-load bandwidth probe, for the L1 ceiling of the hierarchical
 * roofline (doc/gemm_analysis.md, "How to generate the roofline").
 * Streams unit-stride vle16 loads over a buffer that fits in the 64 KiB L1
 * D-cache, after one untimed warm-up pass, and reports bytes/cycle.
 *
 * Eight independent accumulators (vadd, not FP) consume the loads so the
 * compiler can't drop them, without adding a dependency chain long enough
 * to bound throughput ahead of the loads themselves. */

#ifndef BUF_BYTES
#define BUF_BYTES (16 * 1024)      /* L1-resident: 1/4 of the D-cache */
#endif
#ifndef PASSES
#define PASSES 64
#endif

static unsigned short buf[BUF_BYTES / 2] __attribute__((aligned(64)));

/* One pass over buf: 8 independent vle16 + vadd per step. RVV types can't
 * live in arrays, so the accumulators are 8 locals threaded through a macro. */
#define LOAD_PASS()                                                              \
    for (const unsigned short *p = buf; p < buf + BUF_BYTES / 2; p += 8 * vl) { \
        a0 = __riscv_vadd_vv_u16m1(a0, __riscv_vle16_v_u16m1(p + 0 * vl, vl), vl); \
        a1 = __riscv_vadd_vv_u16m1(a1, __riscv_vle16_v_u16m1(p + 1 * vl, vl), vl); \
        a2 = __riscv_vadd_vv_u16m1(a2, __riscv_vle16_v_u16m1(p + 2 * vl, vl), vl); \
        a3 = __riscv_vadd_vv_u16m1(a3, __riscv_vle16_v_u16m1(p + 3 * vl, vl), vl); \
        a4 = __riscv_vadd_vv_u16m1(a4, __riscv_vle16_v_u16m1(p + 4 * vl, vl), vl); \
        a5 = __riscv_vadd_vv_u16m1(a5, __riscv_vle16_v_u16m1(p + 5 * vl, vl), vl); \
        a6 = __riscv_vadd_vv_u16m1(a6, __riscv_vle16_v_u16m1(p + 6 * vl, vl), vl); \
        a7 = __riscv_vadd_vv_u16m1(a7, __riscv_vle16_v_u16m1(p + 7 * vl, vl), vl); \
    }

int main(void)
{
    /* Hardware-max vl for LMUL=1 SEW=16: VLEN=512 -> vl=32 (64 B per load) */
    size_t vl = __riscv_vsetvlmax_e16m1();
    vuint16m1_t a0 = __riscv_vmv_v_x_u16m1(0, vl), a1 = __riscv_vmv_v_x_u16m1(1, vl);
    vuint16m1_t a2 = __riscv_vmv_v_x_u16m1(2, vl), a3 = __riscv_vmv_v_x_u16m1(3, vl);
    vuint16m1_t a4 = __riscv_vmv_v_x_u16m1(4, vl), a5 = __riscv_vmv_v_x_u16m1(5, vl);
    vuint16m1_t a6 = __riscv_vmv_v_x_u16m1(6, vl), a7 = __riscv_vmv_v_x_u16m1(7, vl);
    for (int i = 0; i < BUF_BYTES / 2; ++i) buf[i] = (unsigned short)i;

    printf("--- L1 vector-load bandwidth ---\n");
    printf("vl = %zu, BUF_BYTES = %d, PASSES = %d\n\n", vl, BUF_BYTES, PASSES);

    LOAD_PASS();                                         /* warm-up */

    unsigned long long t_cycle = READ_CSR(mcycle);
    unsigned long long t_inst  = READ_CSR(minstret);
    for (int pass = 0; pass < PASSES; ++pass)
        LOAD_PASS();
    t_cycle = READ_CSR(mcycle)   - t_cycle;
    t_inst  = READ_CSR(minstret) - t_inst;

    unsigned long long bytes = (unsigned long long)BUF_BYTES * PASSES;
    vuint16m1_t sum = __riscv_vadd_vv_u16m1(__riscv_vadd_vv_u16m1(__riscv_vadd_vv_u16m1(a0, a1, vl),
                                                                  __riscv_vadd_vv_u16m1(a2, a3, vl), vl),
                                            __riscv_vadd_vv_u16m1(__riscv_vadd_vv_u16m1(a4, a5, vl),
                                                                  __riscv_vadd_vv_u16m1(a6, a7, vl), vl), vl);

    printf("counter: mcycle = %llu\n", t_cycle);
    printf("counter: minstret = %llu\n", t_inst);
    printf("bytes loaded = %llu\n", bytes);
    printf("L1 load bandwidth = %llu.%02llu bytes/cycle\n",
           bytes / t_cycle, (bytes * 100 / t_cycle) % 100);
    printf("checksum = %u\n", (unsigned)__riscv_vmv_x_s_u16m1_u16(sum));
    return 0;
}
