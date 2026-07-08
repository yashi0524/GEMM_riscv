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

## O3CPU: what changes with out-of-order execution

`sim_config/gem5_riscv_demo_riscv_baremetal_semihost_o3.py` swaps in
`RiscvO3CPU` (same system/cache setup otherwise). It needs no whisper
changes either — whisper has no CPU-microarchitecture model at all, so it's
identical regardless of which gem5 CPU config is used.

Results (gem5, same binaries as above, ITERS=10000):

| workload | MinorCPU mcycle | O3CPU mcycle | speedup |
|---|---|---|---|
| gemm (M=N=K=16) | 9,547 | 2,863 | 3.34× |
| fmacc serial (1 accumulator) | 60,054 | 10,062 | 5.97× |
| fmacc x16 (FP64) | 10,170 | 3,202 | 3.18× |
| fmacc_fp16 x16 | 10,219 | 3,198 | 3.20× |

`gemm`'s 3.34× is a fair architectural comparison: O3's out-of-order
scheduling (32-entry load/store queues, 8-wide fetch/decode/issue/commit,
large ROB) overlaps independent loads far more aggressively than Minor's
narrow 2-wide in-order pipeline, while both use the *same* cache model
(identical `Cache()` params) — so the speedup reflects real scheduling
capability, not a cache/memory timing difference.

**`fmacc`'s numbers need a caveat, though.** The serial version — a single,
totally unbroken dependency chain, the same code that stalled MinorCPU
regardless of FU opLat — runs at 1.006 cycles/vfmacc under O3CPU, i.e.
almost as fast as MinorCPU's fully-unrolled x16 (1.02 cycles/vfmacc). That's
suspiciously good for a true RAW dependency chain, so we checked the
instantiated config directly:

```
FUList[3]/opList[1]  FloatMultAcc      opLat=5   (scalar FP FMA)
FUList[5]/opList[20] SimdFloatMultAcc  opLat=1   (vfmacc — this is what we use)
```

O3's stock `SIMD_Unit` FU pool (`cpu/o3/FuncUnitConfig.py`) doesn't override
`opLat` for `SimdFloatMultAcc`, so it falls back to `OpDesc`'s default of
**1 cycle** — 5-6× more optimistic than Minor's deliberately-configured
`MinorDefaultFloatSimdFU` (`opLat=6`) or even O3's own scalar `FloatMultAcc`
(`opLat=5`). This is a gap in gem5's *default* O3 config for RVV.

**Fixing it took two tries.** The first attempt set `system.cpu.fuPool =
CustomFUPool()` directly — this ran without error and even showed up in
`config.json`, but had *zero* effect on timing (opLat=5/6 gave the exact
same cycle counts as the unmodified default). Reason: `fuPool` isn't
actually a declared `BaseO3CPU` parameter at all — gem5 silently accepts
attribute assignments of SimObject values under any name and records them as
orphan child nodes in the config tree, even when nothing in the C++ model
ever reads that attribute. The functional-unit pool O3 actually uses lives
on each `IQUnit` inside `system.cpu.instQueues` (`cpu/o3/IQUnit.py`:
`fuPool = Param.FUPool(DefaultFUPool(), ...)`), so the fix was:

```python
system.cpu.instQueues = [IQUnit(fuPool=CustomFUPool())]
```

Confirmed via `config.json` afterward: exactly one `fuPool` reference now,
showing the overridden value. (`sim_config/..._o3.py` has the full
`CustomFUPool`/`CustomSIMDUnit` definitions, gated behind the
`O3_SIMD_FMA_OPLAT` env var, default `1` — unchanged stock behavior unless
overridden.)

**Results with the fix, `O3_SIMD_FMA_OPLAT` matched to Minor's `opLat=6`:**

| workload | Minor (opLat=6) | O3 (opLat=1, stock) | O3 (opLat=6, matched) | O3 vs Minor, matched |
|---|---|---|---|---|
| fmacc serial | 60,054 | 10,062 | **60,042** | **1.00×** |
| fmacc x16 | 10,170 | 3,202 | **3,822** | **2.66×** |

With latency actually matched, the serial chain runs at essentially
*identical* speed on both CPU models (60,042 vs 60,054) — exactly what
physics predicts: out-of-order execution cannot parallelize a genuine RAW
dependency chain: there's no independent work to reorder around, so OoO
buys nothing. The stock 5.97× "speedup" reported earlier was purely the
opLat=1-vs-6 modeling gap, not a real OoO effect.

The x16 case is different: even with opLat matched at 6, O3 is still
**2.66× faster than Minor** (3,822 vs 10,170) — this is a genuine,
verified OoO advantage, coming from O3's 4 `SIMD_Unit` instances (vs
Minor's 1) actually running 4 of the 16 independent accumulator chains in
parallel, something a latency parameter alone can't fake.

One more data point along the way: at `opLat=5`, x16 measured identically
to `opLat=1` (3,202 both) — 16 independent chains over 4 ports (4 chains/
port) is enough instruction-level parallelism to fully hide up to ~5 cycles
of latency, but not quite enough to fully hide 6 (3,822, a real increase).
That crossover is itself informative: it's the point where available ILP
stops being "more than enough" and starts being "the binding constraint."

## Takeaway

Only load/store-bound kernels like `gemm` benefited from MinorCPU's
pipelining before this investigation; a compute-bound kernel like `fmacc`
needed its single dependency chain broken by hand (via unrolling) before it
could benefit too — and even then, its ceiling is set by the *count* of
float/SIMD functional units, not just the pipeline's issue width. The x16,
1-FU, FP64 result (15.73 GFLOP/s) is the peak-compute ceiling used in the
[gemm roofline analysis](../README.md), since it reflects MinorCPU's actual
default configuration and the `gemm` kernel's own dtype; the 2-FU, FP16, and
O3CPU results are diagnostic data points, not changes to that default.

The O3CPU investigation is a double lesson: (1) always check a timing
model's *actual instantiated* functional-unit latencies before trusting a
cross-CPU-model comparison — gem5's stock O3 config quietly assumes 1-cycle
vector FMA, which inflated the naive comparison 5-6×; and (2) verify a
config override actually reaches the model that reads it — an assignment
that "runs fine and shows up in config.json" is not proof it did anything,
since gem5 will silently accept SimObject attributes under names that
aren't real parameters. With both fixed, the fair result stands: `gemm`'s
OoO speedup (3.34×) and `fmacc` x16's (2.66×) are real, driven by wider
memory scheduling and more execution ports respectively; `fmacc` serial's
apparent 5.97× speedup was not real at all.
