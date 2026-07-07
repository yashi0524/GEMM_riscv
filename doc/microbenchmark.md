# FMACC micro-benchmark

`src/fmacc.c` is a peak-compute probe: a tight loop of `vfmacc` (fused
multiply-accumulate) vector ops with no memory traffic, used to establish the
compute ceiling for the gemm roofline analysis in [../README.md](../README.md).

Fixed setup: `va = vb = 1.0`, each accumulator starts at `0.0`, `vl = 8`,
`FP64`, `VLEN = 512`. Every `vfmacc` adds exactly `1.0` per lane, so the
combined result must equal the total op count — this is checked on every run
(`result[0] == total_ops`) as a correctness guard.

## Why unroll it

The initial version used a single accumulator:

```c
for (int i = 0; i < ITERS; ++i)
    vc = __riscv_vfmacc_vv_f64m1(vc, va, vb, vl);  /* vc = va*vb + vc */
```

This is a strict serial dependency chain — every `vfmacc` must wait for the
previous one's result. On gem5 MinorCPU (in-order, dual-issue) this chain
left the pipeline with only one stream of work, so it stalled waiting on it
regardless of functional-unit latency. We confirmed this directly: overriding
`MinorDefaultFloatSimdFU`'s `opLat` (6 → 2) and even `MinorDefaultIntFU`'s
`opLat` (3 → 1) simultaneously changed the measured cycle count by less than
0.02% — the serial chain's throughput floor comes from the pipeline's
dual-issue width, not from how fast any single functional unit computes.

The fix: split the accumulation across N independent registers
(`vc0..vcN-1`), each fed by its own `vfmacc` inside the same loop body, then
sum them together after the timed region. With no dependency between `vc0`
and `vc1`, the pipeline can have both in flight at once instead of stalling
between them.

```c
for (int i = 0; i < ITERS; i += N) {
    vc0 = __riscv_vfmacc_vv_f64m1(vc0, va, vb, vl);
    vc1 = __riscv_vfmacc_vv_f64m1(vc1, va, vb, vl);
    /* ... vc2 .. vcN-1 */
}
```

When `ITERS` isn't a multiple of `N` (true for `ITERS=10000, N=12`), the loop
overshoots to the next multiple of `N` — the benchmark tracks this as
`total_ops` rather than assuming it equals `ITERS`, so the correctness check
and cycle-count math stay honest.

## Results (gem5 MinorCPU, ITERS=10000, vl=8, FP64)

| unroll | accumulators | whisper mcycle | gem5 mcycle | total_ops | cycles/vfmacc | peak compute | vs serial | vs previous |
|---|---|---|---|---|---|---|---|---|
| serial | 1 | 30,004 | 60,054 | 10,000 | 6.00 | 2.66 GFLOP/s | 1.00× | — |
| x4 | vc0..vc3 | 15,009 | 22,566 | 10,000 | 2.26 | 7.09 GFLOP/s | 2.66× | 2.66× |
| x8 | vc0..vc7 | 13,763 | 14,108 | 10,000 | 1.41 | 11.34 GFLOP/s | 4.26× | 1.60× |
| x12 | vc0..vc11 | 11,693 | 10,391 | 10,008 | 1.04 | 15.41 GFLOP/s | 5.79× | 1.36× |
| **x16** | vc0..vc15 | 11,896 | 10,170 | 10,000 | **1.02** | **15.73 GFLOP/s** | **5.91×** | 1.02× |

(`peak compute = (vl × 2 FLOP) / (cycles/vfmacc / 1 GHz)`; theoretical max at
vl=8, 1 GHz is `8 × 2 × 1 GHz = 16.0 GFLOP/s`.)

whisper's `mcycle` is a functional (non-timing) count and roughly tracks
retired-instruction count; only gem5 MinorCPU's `mcycle` reflects real
pipeline timing and is what the GFLOP/s figures above are derived from.

IPC (`minstret/mcycle`, gem5) climbs with each unroll step:

```
serial   0.50
x4       0.67
x8       0.98
x12      1.13
x16      1.17
```

approaching but never reaching MinorCPU's dual-issue front-end ceiling of
IPC=2 (`executeIssueLimit=2`, `decodeInputWidth=2`). Gains shrink fast as
unrolling adds more: 2.66× (x1→x4), 1.60× (x4→x8), 1.36× (x8→x12), only
1.02× (x12→x16). By x16 the benchmark has essentially saturated the 2-wide
in-order pipeline's issue bandwidth for this instruction mix — reaching
15.73 GFLOP/s, 98.3% of the 16.0 GFLOP/s theoretical max. Further unrolling
would buy almost nothing without a wider-issue (or out-of-order) core.

## Takeaway

Only load/store-bound kernels like `gemm` benefited from MinorCPU's
pipelining before this investigation; a compute-bound kernel like `fmacc`
needed its single dependency chain broken by hand (via unrolling) before it
could benefit too. The x16 result (15.73 GFLOP/s) is the peak-compute
ceiling used in the [gemm roofline analysis](../README.md).
