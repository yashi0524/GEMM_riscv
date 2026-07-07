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

## Why not 32 GFLOP/s? (dual-issue width vs. functional-unit count)

If MinorCPU issues 2 instructions/cycle, shouldn't peak compute be
`vl × 2 FLOP × 2 issue × 1 GHz = 32 GFLOP/s` instead of 16? No — dual-issue
width and functional-unit count are different things. `MinorDefaultFUPool`
(`src/cpu/minor/BaseMinorCPU.py`) has **two** `MinorDefaultIntFU` instances
but only **one** `MinorDefaultFloatSimdFU` instance:

```python
class MinorDefaultFUPool(MinorFUPool):
    funcUnits = [
        MinorDefaultIntFU(), MinorDefaultIntFU(),   # 2 int units
        MinorDefaultIntMulFU(), MinorDefaultIntDivFU(),
        MinorDefaultFloatSimdFU(),                   # only 1 float/SIMD unit
        MinorDefaultPredFU(), MinorDefaultMemFU(), MinorDefaultMiscFU(),
    ]
```

`executeIssueLimit=2` lets the pipeline issue 2 instructions per cycle
*total*, but each instruction needs a distinct FU slot to issue into — with
only one FloatSimd FU, at most one `vfmacc` can issue per cycle no matter how
wide the front end is. The second issue slot each cycle goes to something
else (e.g. the loop's `addi`/branch, via an IntFU). That's exactly the 1/cycle
ceiling the x16 result approached (16.0 GFLOP/s theoretical, 15.73 achieved).

To test this, we added a diagnostic knob to
`sim_config/gem5_riscv_demo_riscv_baremetal_semihost_minor.py`
(`MINOR_FLOAT_FU_COUNT`, default `1` — no change to any other benchmark's
results) that duplicates `MinorDefaultFloatSimdFU` in the pool:

```
MINOR_FLOAT_FU_COUNT=1 (default): gem5 mcycle = 10,170 → 1.02 cycles/vfmacc → 15.73 GFLOP/s
MINOR_FLOAT_FU_COUNT=2:            gem5 mcycle =  9,534 → 0.95 cycles/vfmacc → 16.78 GFLOP/s
```

A second FloatSimd FU helped, but only **1.07×** — nowhere near the 2×
(→32 GFLOP/s) a naive "more FUs = proportionally more throughput" model would
predict. With 16 `vfmacc` + ~2 loop-control instructions per unrolled block,
`executeIssueLimit=2` caps total issue at 18/2=9 cycles/block best case
(0.56 cycles/vfmacc, ~28 GFLOP/s) even with 2 FloatSimd FUs available —
we're still short of that too, so the front end (fetch/decode width, not
just FU count) is the next limiter. Reaching 32 GFLOP/s would need both more
FloatSimd FUs *and* a wider issue/decode front end, not one or the other.

## FP16: shorter elements, same instruction, more FLOPs/cycle

`src/fmacc_fp16.c` is a copy of the x16 version with the element type switched
to FP16 (`vfloat16m1_t`, `__riscv_vfmacc_vv_f16m1`). `VLEN` is unchanged at
512 bits, but a 16-bit element means `vl = __riscv_vsetvlmax_e16m1() = 32`
(vs. `vl = 8` for FP64) — 4× more elements per vector register, hence 4×
more FLOPs per `vfmacc`. Built via `make fmacc_fp16` (or
`make test/fmacc_fp16_riscv`), which needs the `zvfh` extension: the
Makefile bakes `-march=rv64gcv_zvfh` into that target's `BENCH_EXTRA_FLAGS`
(a second `-march=` on the command line wins over the base `rv64gcv`, so no
caller-side changes were needed — this is the same pattern
`sweep_fp16.py`/`compare_shapes_fp16.py` use, just baked into the Makefile
target instead of passed by the caller).

**Precision pitfall**: the original linear combine chain (`vc0+vc1`,
`+vc2`, `+vc3`, ...) gave `result[0] = 9992.0` instead of `10000.0`. FP16 has
only 10 mantissa bits, so integers above 2048 aren't all exactly
representable (ULP=2 in `[2048,4096)`, ULP=4 in `[4096,8192)`, etc.) — the
linear chain's partial sums (625, 1250, **1875**, 2500, **3125**, ...) hit
odd values in ranges where only even (or multiple-of-4) values are exact,
rounding away from the true sum. Switching to a pairwise **tree** reduction
(`vc0+vc1`, `vc2+vc3`, ... → `vc01+vc23`, ... → final) keeps every partial
sum a clean power-of-two multiple of 625 (625 → 1250 → 2500 → 5000 → 10000),
each exactly representable in FP16 at its own magnitude — `result[0]` came
back exact. (This combine step is outside the timed region either way, so it
never affected the cycle-count measurements — only the correctness check.)

Results (gem5 MinorCPU, ITERS=10000, x16 unroll, `MINOR_FLOAT_FU_COUNT=1`
default):

| dtype | vl | gem5 mcycle | total_ops | cycles/vfmacc | peak compute | theoretical max | efficiency |
|---|---|---|---|---|---|---|---|
| FP64 | 8 | 10,170 | 10,000 | 1.02 | 15.73 GFLOP/s | 16.0 GFLOP/s | 98.3% |
| **FP16** | **32** | **10,219** | **10,000** | **1.02** | **62.63 GFLOP/s** | **64.0 GFLOP/s** | **97.9%** |

`cycles/vfmacc` is essentially identical between the two (1.02 either way) —
the same 1-`vfmacc`/cycle ceiling from the single FloatSimd FU applies
regardless of element width, since gem5's `MinorFU` model charges a fixed
`opLat` per *instruction*, not per element. So a wider `vl` is "free" in this
model: 4× the elements per instruction gives ~4× the GFLOP/s (62.63 vs 15.73)
for the same cycle count, with no separate accounting for narrower ALU
lanes or internal element-serialization a real vector unit's datapath width
might impose. Take the exact 4× scaling as a property of this timing model,
not a guarantee that real FP16 vector hardware is always 4× FP64 hardware.

## Takeaway

Only load/store-bound kernels like `gemm` benefited from MinorCPU's
pipelining before this investigation; a compute-bound kernel like `fmacc`
needed its single dependency chain broken by hand (via unrolling) before it
could benefit too — and even then, its ceiling is set by the *count* of
float/SIMD functional units, not just the pipeline's issue width. The x16,
1-FU, FP64 result (15.73 GFLOP/s) is the peak-compute ceiling used in the
[gemm roofline analysis](../README.md), since it reflects MinorCPU's actual
default configuration and the `gemm` kernel's own dtype; the 2-FU and FP16
results are diagnostic data points, not changes to that default.
