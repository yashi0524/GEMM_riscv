# O3CPU pipeline stall analysis: `opt_gemm` vs `opt_gemm_blocked`

Cycle-level breakdown of where O3CPU spends its time in the two hand-vectorized
GEMM kernels, from gem5's `O3PipeView` trace. Companion to
[gemm_analysis.md](gemm_analysis.md) (roofline sweep) and
[microbenchmark.md](microbenchmark.md) (FU-latency / peak-compute probes).

## Setup

- Run: `gem5.opt --debug-flags=O3PipeView --debug-file=pipeview.trace -d test/m5out_pipeview sim_config/gem5_riscv_demo_riscv_baremetal_semihost_o3.py test/gemm_riscv`
  (2026-10-01, gem5 25.1.0.0, log in `test/pipeview_run.log`)
- Kernel: FP64, `M=N=K=16`, VLEN=512 (`vl=8`), each kernel called **once**
- Trace: `test/m5out_pipeview/pipeview.trace` (24 MB, 106,420 micro-op records)
- Kernel PC ranges (from `llvm-nm-18 -S gemm_riscv`):
  `opt_gemm` 0x10474–0x1073e, `opt_gemm_blocked` 0x1073e–0x10b54
- **FU config caveat:** the main tables below use the stock O3 FU pool —
  `config.ini` shows `SimdFloatMultAcc opLat=1` (`O3_SIMD_FMA_OPLAT` unset).
  See [microbenchmark.md](microbenchmark.md) for why that is optimistic.
  A second capture with a realistic `opLat=6` is analyzed in
  [Re-capture with FMA opLat=6](#re-capture-with-fma-oplat6) — the
  conclusions hold.

Reproduce with:

```
script/analyze_o3_pipeview.py test/m5out_pipeview/pipeview.trace test/gemm_riscv
```

(kernel PC ranges are read from the binary via `llvm-nm-18 -S`; pass kernel
names as extra args to analyze others).

Method: each trace record gives per-micro-op fetch / decode / rename /
dispatch / issue / complete / retire ticks (1 cycle = 1000 ticks); `retire=0`
marks a squashed micro-op. Records were filtered by PC range per kernel.
Zero-commit cycles were attributed to the state of the ROB head (oldest
uncommitted, eventually-committed instruction) in that cycle. Fetch gaps were
classified as *cold I-line* (first fetch from a 64 B line never touched
before), *after squash* (a squashed micro-op was fetched inside the gap), or
*other* (taken-branch fetch breaks, etc.).

## Summary

| | opt_gemm | opt_gemm_blocked |
|---|---:|---:|
| span in trace (mcycle) | 1,712 (1,790) | 1,544 (1,625) |
| committed micro-ops | 4,569 | 2,626 |
| IPC | 2.67 | 1.70 |
| **zero-commit cycles** | **737 (43%)** | **882 (57%)** |
| squash events / squashed micro-ops | 22 / 490 | 18 / 499 |

The commit histogram is bimodal — almost every cycle commits either 0 or the
full 8 micro-ops (opt: 737 cycles at 0, 293 at 8; blocked: 882 at 0, 194 at 8).
The core is not throughput-limited by execution; it is waiting.

### Per-stage mean latency (committed micro-ops, cycles)

| | fetch→decode | decode→rename | rename→dispatch | dispatch→issue | issue→complete | complete→retire |
|---|---:|---:|---:|---:|---:|---:|
| opt | 1.0 | 1.0 | 2.0 | 1.8 | 1.4 | 5.4 |
| blocked | 1.0 | 1.0 | 2.0 | 1.7 | 1.6 | 5.5 |

### Per-class breakdown

`opt_gemm`:

| class | n | dispatch→issue | issue→complete | complete→retire |
|---|---:|---:|---:|---:|
| int/other | 2,164 | 1.5 | 1.7 | 5.2 |
| scalar load | 574 | 0.2 | 1.0 | 7.2 |
| vload | 544 | 3.0 | 1.0 | 5.2 |
| vfmacc | 512 | 3.7 | 1.0 | 4.4 |
| other vector | 480 | 1.9 | 1.0 | 4.9 |
| branch | 167 | 0.7 | 1.0 | 7.2 |
| vstore | 128 | 1.7 | 1.0 | 5.9 |

`opt_gemm_blocked`:

| class | n | dispatch→issue | issue→complete | complete→retire |
|---|---:|---:|---:|---:|
| int/other | 1,320 | 1.8 | 2.2 | 4.8 |
| scalar load | 589 | 0.8 | 1.0 | 6.8 |
| vfmacc | 512 | 2.0 | 1.0 | 6.1 |
| vstore | 68 | 1.6 | 1.0 | 3.8 |
| vload | 64 | 2.8 | 1.0 | 4.5 |
| branch | 41 | 1.2 | 1.0 | 6.3 |
| other vector | 32 | 4.9 | 1.0 | 4.4 |

Blocking cuts vector loads 544 → 64 (B row reused across the 16-row block),
as intended. `vfmacc` execute time is 1 cycle in both — an artifact of the
stock `opLat=1`.

## Zero-commit cycles: what the ROB head was doing

| ROB-head state | opt | blocked |
|---|---:|---:|
| not fetched yet (frontend) | 366 | 605 |
| in decode/rename (frontend) | 69 | 105 |
| **frontend total** | **435** | **710** |
| head complete, awaiting commit (behind `fld`→`fmul.d` chain) | 271 | 134 |
| head executing | 31 | 38 |

Both kernels are **frontend-bound**; blocked dramatically so (80% of its
zero-commit cycles).

## Root causes

### 1. Cold I-cache misses (dominant for blocked)

| | cold I-lines | fetch-gap cycles on cold lines | after squash | other |
|---|---:|---:|---:|---:|
| opt | 7 | 358 (21% of span) | 241 | 341 |
| blocked | 15 | **680 (44% of span)** | 177 | 260 |

Each kernel is executed once, so every 64 B code line is a first-touch miss.
Whole-run stats: 320 I-cache misses, 17,690,000 ticks miss latency →
~55 cycles/miss. `opt_gemm_blocked` is 1,046 B of fully-unrolled code
(16 `fld` + 16 `fmul.d` + 16 `vfmacc` per inner iteration), so it pays twice
the cold-miss cost of `opt_gemm`.

Rough steady-state estimate with cold-miss gaps removed: opt ≈ 1,354 cycles,
blocked ≈ 864 cycles → blocked would be **~1.57× faster** than opt, versus the
measured 1.10× (1,790 vs 1,625 mcycle). This is the "small-problem fixed cost"
noted in [gemm_analysis.md](gemm_analysis.md)'s key finding, and it explains
why blocking gains so little on O3 there (9.42% → 9.66% of compute roof).
A warm-cache measurement (call each kernel twice, measure the second call)
would confirm it.

### 2. Squashes, about half right after `vsetvli`

Last committed instruction before each squashed run:

| kernel | count | PC | instruction |
|---|---:|---|---|
| opt | 6 | 0x1064a | `vsetvli zero, s7, e64, m1, ta, ma` |
| opt | 4 | 0x106b0 | `bgeu a2, s9, -172` (loop branch) |
| opt | 3 | 0x105c4 | `vsetvli s7, a4, e64, m1, ta, ma` |
| opt | 2 | 0x106b4 | `bge s8, a2, -284` |
| opt | 2 | 0x105bc | `bge s6, a1, -76` |
| blocked | 7 | 0x10a2c | `vsetvli zero, a0, e64, m1, ta, ma` |
| blocked | 6 | 0x10b30 | `bne t4, s3, -260` (inner-loop back-edge) |
| blocked | 2 | 0x108de | `vsetvli a0, a0, e64, m1, ta, ma` |

In `opt_gemm_blocked`, `0x10a2c` is the inner loop's back-edge target, so the
compiler re-executes `vsetvli` every iteration even though `vl` is
loop-invariant. Squash refetch cost: 241 cycles (opt), 177 cycles (blocked).
The exact gem5 mechanism that makes `vsetvli` trigger a squash has not been
verified in the gem5 source yet.

### 3. Scalar `alpha` scaling on the critical path

Each A element is loaded and scaled by `alpha` before the `vfmacc` consumes it
(`fld` → `fmul.d` (4 cycles) → `vfmacc`). Example ROB-head stall in `opt_gemm`:

```
sn     pc       F      R      D      I      C      Ret
47982  0x10616  28638  28640  28642  28642  28643  28649  fld    fa2, -8(a4)
47983  0x1061a  28639  28641  28643  28645  28649  28651  fmul_d fa5, fa5, fa0   ← ROB head
47984  0x1061e  28639  28641  28643  28646  28650  28652  fmul_d fa4, fa4, fa0
47985  0x10622  28639  28641  28643  28646  28650  28652  fmul_d fa3, fa3, fa0
47986  0x10626  28639  28641  28643  28647  28651  28653  fmul_d fa2, fa2, fa0
```

`0x1061a` alone accounts for 81 zero-commit cycles in opt. Blocked has 16 such
`fmul.d` per inner iteration (`0x10a4c`–`0x10ae8`); they top its
dispatch→issue wait list (4–5 cycles each).

## Re-capture with FMA opLat=6

- Run: `O3_SIMD_FMA_OPLAT=6 gem5.opt --debug-flags=O3PipeView ... -d test/m5out_pipeview_oplat6 ...`
  (2026-10-02, log `test/pipeview_oplat6_run.log`, full analyzer output
  `test/pipeview_oplat6_analysis.txt`)
- Verified: `config.ini` shows `SimdFloatMultAcc opLat=1 → 6`, and every
  `vfmacc_vf_micro` / `vfmacc_vv_micro` in the trace now has issue→complete =
  6 cycles (was 1). Other vector FP ops (`vfadd`, `vfmul`, `vfmv`) stay at 1;
  scalar `fmul.d` stays at 4.

**Kernel cycle counts do not change at all:**

| kernel | mcycle, opLat=1 | mcycle, opLat=6 |
|---|---:|---:|
| scalar_gemm | 2,882 | 2,896 |
| opt_gemm | 1,790 | **1,790** |
| opt_gemm_blocked | 1,625 | **1,625** |

Trace span (1,712 / 1,544), squash events (22 / 18), and every fetch-gap
number (cold I-line 358 / 680, after-squash 241 / 177) are also identical.
Five extra cycles on all 1,024 `vfmacc` micro-ops are fully absorbed: the
back end finishes waiting on them in the time the frontend spends refilling.
This is the strongest confirmation that both kernels are frontend-bound.

What does move is *which* instruction blocks commit when the frontend is not
the cause:

| | opt, opLat=1 | opt, opLat=6 | blocked, opLat=1 | blocked, opLat=6 |
|---|---:|---:|---:|---:|
| zero-commit cycles | 737 | 696 | 882 | 856 |
| frontend (not fetched + decode/rename) | 435 | 423 | 710 | 697 |
| head complete, awaiting commit | 271 | 243 | 134 | 123 |
| head executing | 31 | 30 | 38 | 36 |
| vfmacc issue→complete | 1.0 | 6.0 | 1.0 | 6.0 |
| vfmacc complete→retire | 4.4 | 2.3 | 6.1 | 2.6 |

- **opt:** the top commit-blocking head changes from `fmul_d` at `0x1061a`
  (81 cycles) to `vfmacc` at `0x1068a` (72 cycles). The 8→1 reduction tree
  appears in the dispatch→issue wait list (`vfadd` at `0x105ac` / `0x105b0`
  wait 9.6 / 10.6 cycles, and the following `vse64` waits 10.6), because it
  sits behind the last FMA of each chain. Mean complete→retire for all
  classes rises (e.g. branch 7.2 → 11.8): finished instructions now queue
  behind a 6-cycle FMA at the ROB head.
- **blocked:** almost no change. Each of the 16 accumulators is updated only
  once per inner iteration, so the RAW distance on `v8`–`v23` is a whole
  iteration (69 instructions), far more than 6 cycles. The `fmul.d` scaling
  still tops the dispatch→issue wait list. The expected `v24` fan-out stall
  doesn't appear: all 16 FMAs read `v24`, but they don't depend on each other.

## Recommendations

1. **Hoist `alpha` out of the inner loop** — accumulate `A·B` unscaled and
   apply `C = alpha·acc + beta·C` once per output row. Removes 8 (opt) /
   16 (blocked) `fmul.d` per inner iteration and the 4-cycle FP multiply from
   the critical path. At opLat=6, opt's commit blocker becomes the `vfmacc`
   itself, so this mainly helps by cutting instruction count (frontend
   pressure) and helps blocked more than opt.
2. **Hoist `vsetvli` out of the blocked inner loop** (`vl` is invariant) —
   should remove most of the `vsetvli`-adjacent squashes.
3. **Measure warm** — call each kernel twice and record the second call's
   mcycle, to separate steady-state throughput from cold I-cache cost.
4. ~~Re-capture with `O3_SIMD_FMA_OPLAT=6`~~ — done (see above): zero
   cycle impact; FMA latency is not the bottleneck on O3 for this problem
   size. Use opLat=6 for future O3 captures anyway, so that the code changes
   above are measured against realistic FMA latency.
