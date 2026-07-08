# GEMM sweep roofline analysis (FP64/FP16 × MinorCPU/O3CPU)

Data from `test/sweep.py` driven by `test/config/sweep_config.json` (all four
variants enabled: `fp64_minor`, `fp64_o3`, `fp16_minor`, `fp16_o3`), plotted
against the compute-roof ceilings measured in
[microbenchmark.md](microbenchmark.md)'s `fmacc`/`fmacc_fp16` x16-unroll
micro-benchmark — not derived from this sweep's own data.

Shape: `M=16 K=16` for both dtypes (`shape.m`/`shape.k` in the config); `N` is
derived per dtype as `source_A_width_bits / data_format` (1024/64=16 for FP64,
1024/16=64 for FP16) — so FP64 runs a square `16×16×16` kernel (identical to
the single-config kernel documented in [../README.md](../README.md)), while
FP16's wider `N=64` keeps the two dtypes' input-array footprint identical:
with `source_A_width_bits=1024` shared, `il=1` memory traffic (`Q`) comes out
to the same 102,400 bytes for both dtypes. Memory peak = 12.8 GB/s
(DDR3-1600 8×8).

An interactive log-log version of this chart (with hover tooltips) is
available as a Claude artifact; this file is the durable, in-repo copy of the
same data.

## Compute-roof ceilings used

| Config | Compute roof | Source |
|---|---|---|
| FP64 MinorCPU | 15.73 GFLOP/s | microbenchmark.md's designated default ceiling (x16 unroll, 98.3% of 16.0 GFLOP/s theoretical, vl=8, 1 FloatSimd FU, 1 GHz) |
| FP16 MinorCPU | 62.63 GFLOP/s | same benchmark, vl=32 (97.9% of 64.0 GFLOP/s theoretical) |
| FP64 O3CPU | 49.97 GFLOP/s | derived from microbenchmark.md's O3CPU mcycle=3,202 (10,000 total_ops, stock O3 functional-unit pool — `O3_SIMD_FMA_OPLAT` unset, defaults to 1 — matching this sweep's O3 config exactly) |
| FP16 O3CPU | 200.1 GFLOP/s | derived from microbenchmark.md's O3CPU mcycle=3,198 (same stock config) |

**Caveat on the O3 ceilings:** microbenchmark.md flags the stock O3 config as
carrying a known modeling gap — gem5's default O3 functional-unit pool
assumes a 1-cycle vector FMA (`opLat=1`) vs. MinorCPU's deliberately
configured 6-cycle latency, inflating the stock O3 ceiling ~5–6× relative to
a latency-matched comparison. With `opLat` matched to Minor's 6, the doc
measures a *fair* FP64 O3 ceiling of **41.86 GFLOP/s** (a genuine 2.66× OoO
advantage over Minor's 15.73, from 4 SIMD ports vs Minor's 1) — no matched
FP16 figure exists in the doc, so it isn't used here. Either way, the sweep's
GEMM points are so deep in the memory-bound region (AI 0.080–0.320, ridge
points at 3.9/15.6 FLOP/B) that this ceiling choice doesn't change the
qualitative conclusion below.

## Full sweep results

`Attainable` is the memory-bound roofline value at this row's AI —
`AI × 12.8 GB/s` — i.e. the slope-line ceiling that actually applies here,
since every row's AI sits far left of its ridge point (see below). `mem%` is
achieved GFLOP/s against *that* ceiling, and `roof%` is achieved GFLOP/s
against the compute-roof ceiling from the table above; `roof% = mem% ×
(AI / ridge_point)`, i.e. `roof%` is just `mem%` rescaled by how far below
the ridge point this kernel's AI sits.

| Config | core | w  | il | mcycle  | AI (FLOP/B) | GFLOP/s | Attainable | mem%   | Compute roof | roof% |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| FP64 | minor | 4  | 1 | 16,582  | 0.080 | 0.4940 | 1.024 | 48.24%  | 15.73  | 3.14% |
| FP64 | minor | 4  | 2 | 13,718  | 0.080 | 0.5972 | 1.024 | 58.32%  | 15.73  | 3.80% |
| FP64 | minor | 4  | 4 | 11,184  | 0.080 | 0.7325 | 1.024 | 71.53%  | 15.73  | 4.66% |
| FP64 | minor | 8  | 1 | 11,258  | 0.080 | 0.7277 | 1.024 | 71.06%  | 15.73  | 4.63% |
| FP64 | minor | 8  | 2 | 9,547   | 0.080 | 0.8581 | 1.024 | 83.80%  | 15.73  | 5.46% |
| FP64 | minor | 8  | 4 | 56,457  | N/A*  | 0.1451 | N/A*  | N/A*    | 15.73  | 0.92% |
| FP64 | o3    | 4  | 1 | 6,415   | 0.080 | 1.2770 | 1.024 | 124.71% | 49.97  | 2.56% |
| FP64 | o3    | 4  | 2 | 4,285   | 0.080 | 1.9118 | 1.024 | 186.70% | 49.97  | 3.83% |
| FP64 | o3    | 4  | 4 | 3,921   | 0.080 | 2.0893 | 1.024 | 204.03% | 49.97  | 4.18% |
| FP64 | o3    | 8  | 1 | 4,333   | 0.080 | 1.8906 | 1.024 | 184.63% | 49.97  | 3.78% |
| FP64 | o3    | 8  | 2 | 2,863   | 0.080 | 2.8613 | 1.024 | 279.42% | 49.97  | 5.73% |
| FP64 | o3    | 8  | 4 | 23,265  | N/A*  | 0.3521 | N/A*  | N/A*    | 49.97  | 0.70% |
| FP16 | minor | 32 | 1 | 21,126  | 0.320 | 1.5511 | 4.096 | 37.87%  | 62.63  | 2.48% |
| FP16 | minor | 32 | 2 | 41,595  | 0.195 | 0.7878 | 2.496 | 31.56%  | 62.63  | 1.26% |
| FP16 | minor | 32 | 4 | 357,290 | N/A*  | 0.0917 | N/A*  | N/A*    | 62.63  | 0.15% |
| FP16 | o3    | 32 | 1 | 5,899   | 0.320 | 5.5548 | 4.096 | 135.61% | 200.1  | 2.78% |
| FP16 | o3    | 32 | 2 | 16,452  | 0.195 | 1.9917 | 2.496 | 79.79%  | 200.1  | 1.00% |
| FP16 | o3    | 32 | 4 | 221,897 | N/A*  | 0.1477 | N/A*  | N/A*    | 200.1  | 0.07% |

\* `il=4` rows compiled to scalar code (no vector load/store instructions
retired), so arithmetic intensity is undefined on this vector-AI axis;
GFLOP/s and roof-% are still shown for reference, but `Attainable`/`mem%`
have no defined value without a vector AI.

**`mem% > 100%` on every O3CPU row** is not a measurement error: this
kernel's working set (~6–16 KB of A/B/C) fits entirely in the 64 KB L1
cache, so O3CPU's real sustained bandwidth is well above the 12.8 GB/s
DRAM-peak figure that `Attainable` assumes — meaning O3 isn't actually
bandwidth-bound on this kernel at all, unlike MinorCPU where `mem%` stays
under 100% and roughly tracks classic roofline behavior.

FP64's `minor, w=8, il=2` row (mcycle=9,547) is the exact same kernel
documented in [../README.md](../README.md)'s single-config roofline
walkthrough — both use the identical `M=N=K=16` FP64 shape, so the numbers
cross-validate.

## Key finding

Every measured GEMM point sits far left of all four ridge points (where each
ceiling meets the memory-bound diagonal `GFLOP/s = 12.8 × AI`) — this
`M=K=16` problem (FP64 `N=16`, FP16 `N=64`, same 1024-bit input-row budget
for both) is deep in the memory-bound regime on every CPU model and dtype
tested, reaching only **0.1–5.7%** of its own compute roof. At this problem
size, GEMM is limited by load/store issue and dependency stalls, not by FMA
throughput — consistent with microbenchmark.md's conclusion that only
compute-bound kernels like `fmacc` (not `gemm`) approach the compute
roofline.
