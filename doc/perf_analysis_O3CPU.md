# O3CPU pipeline stall analysis: `opt_gemm` vs `opt_gemm_blocked`

Cycle-level breakdown of where O3CPU spends its time in the two hand-vectorized
GEMM kernels, from gem5's `O3PipeView` trace. Companion to
[gemm_analysis.md](gemm_analysis.md) (roofline sweep) and
[microbenchmark.md](microbenchmark.md) (FU-latency / peak-compute probes).

## Setup

- Run: `gem5.opt --debug-flags=O3PipeView --debug-file=pipeview.trace -d test/m5out_pipeview sim_config/gem5_riscv_demo_riscv_baremetal_semihost_o3.py test/gemm_riscv`
  (2026-10-01, gem5 25.1.0.0, log in `test/pipeview_run.log`)
- Kernel: FP64, `M=N=K=16`, VLEN=512 (`vl=8`), each kernel called **once**
  (cold) unless noted — see [Warm-cache measurement](#warm-cache-measurement)
- Source: the original `src/gemm.c` (commit `1921399`). Later sections change
  the source (`vsetvlmax`, `WARMUP_RUNS`) and the simulated machine
  (I-cache prefetcher); each section states what it ran.
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
The [warm-cache measurement](#warm-cache-measurement) confirms it: 1.64×.

### 2. Squashes, about half right after `vsetvli`

> **Correction (warm data):** most of these squashes come from a cold branch
> predictor, not from `vsetvli` itself. With warm predictor/caches there are
> only 4–6 squash events per call, 1–2 of them after a `vsetvli` — see
> [Warm-cache measurement](#warm-cache-measurement).

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

## `vsetvli` out of the inner loops (`vsetvlmax`)

The source already calls `vsetvl` once per column tile, outside the `l` loop.
The in-loop `vsetvli` instructions are **compiler-inserted** (LLVM's vsetvli
insertion doesn't recognize the vl as unchanged across the back-edge):

| kernel | PC (original build) | executions | role |
|---|---|---:|---|
| opt | `0x105c4` `vsetvli s7, a4` | 32 | source `vsetvl(n - j)`, once per tile — keep |
| opt | `0x1064a` `vsetvli zero, s7` | 64 | compiler-inserted inside the unrolled `l` loop — redundant |
| blocked | `0x108de` `vsetvli a0, a0` | 2 | source `vsetvl(n - j)` — keep |
| blocked | `0x10a2c` `vsetvli zero, a0` | 32 | compiler-inserted at the inner-loop back-edge target — redundant |
| blocked | `0x10874` `vsetvli zero, a0` | 2 | compiler-inserted on loop exit — redundant, minor |

Change (`src/gemm.c:129/185/291/362`, all four FP16/FP64 × opt/blocked sites):

```c
size_t vl = (size_t)(n - j) >= __riscv_vsetvlmax_e64m1() ? __riscv_vsetvlmax_e64m1()
                                                         : __riscv_vsetvl_e64m1(n - j);
```

Full tiles use `vsetvlmax` (a known constant); only a partial last tile uses
`vsetvl(n - j)`. Verified with `llvm-objdump-18`: no `vsetvli` left inside
either inner loop. All results below: O3, `O3_SIMD_FMA_OPLAT=6`, correctness
PASS.

### Cold, compressed (default build): net loss from code alignment

| | opt before | opt after | blocked before | blocked after |
|---|---:|---:|---:|---:|
| mcycle | 1,790 | 1,865 (+4%) | 1,625 | 1,784 (+10%) |
| squash events / squashed micro-ops | 22 / 490 | 23 / 480 | 18 / 499 | 13 / 384 |
| "other" fetch-gap cycles | 341 | 545 | 260 | 546 |

The extra compare shifted the code by 2 bytes, leaving 4-byte instructions
across 64 B line boundaries inside the hot loops (blocked: 0 → 5, at
`0x10a3e`, `0x10a7e`, `0x10abe`, `0x10afe`, `0x10b3e`; opt: 2 → 3). O3 fetches
from a one-line buffer (`fetchBufferSize=64`), so each straddling
instruction costs an extra fetch (~1–2 cycles) every iteration.

### Cold, no compressed instructions (`MARCH=rv64imafdv_zicsr_zifencei`)

All instructions are 4 B and 4 B-aligned, so none can straddle a line:

| | opt before | opt after | blocked before | blocked after |
|---|---:|---:|---:|---:|
| mcycle | 2,015 | **1,909 (−5.3%)** | 1,813 | 1,810 (≈0) |
| squash events / squashed micro-ops | 21 / 576 | 24 / 575 | 20 / 640 | 13 / 387 |
| cold I-line fetch-gap cycles | 615 (12 lines) | 608 (12 lines) | 991 (21 lines) | 1,081 (22 lines) |

Blocked's ~90-cycle squash saving is cancelled by one extra cold I-line.
Disabling compression itself costs ~10% (more code → more cold lines), so
it is only for like-for-like comparison, not a default.

## Warm-cache measurement

`make gemm M=16 WARMUP=1` builds with `-DWARMUP_RUNS=1`: each kernel is
called once untimed before its measured call (default `WARMUP=0` = cold, as
before). `script/analyze_o3_pipeview.py --last-call` restricts the analysis
to the measured call. O3, `O3_SIMD_FMA_OPLAT=6`, all PASS.

| build | opt mcycle | blocked mcycle | blocked vs opt |
|---|---:|---:|---:|
| original, compressed | 1,215 | 739 | 1.64× |
| vlmax, compressed | 1,306 (+7%) | 879 (+19%) | 1.49× |
| original, no compressed | 1,291 | 705 | 1.83× |
| vlmax, no compressed | **1,168 (−10%)** | **694 (−2%)** | 1.68× |

| | opt orig, no-C | opt vlmax, no-C | blocked orig, no-C | blocked vlmax, no-C |
|---|---:|---:|---:|---:|
| IPC | 3.75 | 4.20 | 4.20 | 4.23 |
| squash events / squashed micro-ops | 5 / 151 | 4 / 134 | 5 / 161 | 6 / 178 |
| cold I-line fetch-gap cycles | 32 | 39 | 21 | 22 |
| head complete, awaiting commit | 185 | 102 | — | — |

- Cold I-line cost drops from 21–62% of span to 2–4%; blocked's true
  advantage over opt is 1.64–1.83× (cold: 1.10×).
- Squashes fall from 18–24 to 4–6 per call: the cold-run squashes were mostly
  a cold branch predictor (see correction in root cause 2).
- `vsetvlmax` helps opt by 10% warm: its in-loop `vsetvli` sat mid-body
  after the `fmul.d` group; removing it cuts awaiting-commit cycles
  185 → 102. In blocked it was 1 cheap instruction of 69 (−2%).
- With compression, the line-straddle penalty is a steady-state cost:
  blocked "other" fetch gaps 166 → 291, +19%.

## I-cache prefetcher (simulated-machine change)

The stock system has no L2 and no prefetcher: an I-cache miss costs ~55
cycles (DRAM access ≈24 of them, the rest L1 + crossbar), paid serially
while fetch walks straight-line unrolled code. An L2 would not help here —
the kernel code is first-touch, so L2 would be cold too.

`O3_ICACHE_PF_DEGREE=N` (default 0 = none) attaches a
`TaggedPrefetcher(degree=N)` to the I-cache in
`sim_config/gem5_riscv_demo_riscv_baremetal_semihost_o3.py`. Original
source, compressed, `O3_SIMD_FMA_OPLAT=6`, all PASS:

| degree | scalar | opt | blocked | I-misses | avg miss (cyc) | pf issued / useful / late |
|---|---:|---:|---:|---:|---:|---|
| 0 | 2,896 | 1,790 | 1,625 | 320 | 55.3 | – |
| 1 | 2,762 | 1,523 (−15%) | 1,253 (−23%) | 220 | 48.1 | 262 / 76 / 53 |
| 2 | 2,700 | 1,494 (−17%) | 1,194 (−27%) | 200 | 49.2 | 406 / 89 / 175 |
| 4 | 2,660 | 1,456 (−19%) | 1,104 (−32%) | 185 | 51.4 | 551 / 112 / 295 |
| 8 | 2,619 | **1,374 (−23%)** | **1,045 (−36%)** | 155 | 57.6 | 804 / 130 / 535 |
| warm, d=0 and d=4 | 2,514 | 1,215 | 739 | | | |

- Blocked (long straight-line code) benefits most; cold blocked-vs-opt goes
  1.10× → 1.31×. Warm results are identical with and without the
  prefetcher, so it costs nothing in steady state.
- Diminishing returns: most prefetches are late (535/804 at degree 8) —
  the I-cache has only 4 MSHRs, and a tagged prefetcher only triggers on a
  miss or a prefetched-line hit, so each function's first line still
  misses in full. ~300 cycles of cold cost remain for blocked (1,045 vs 739).
- This changes the modelled machine; numbers with the prefetcher are not
  comparable to earlier results (or MinorCPU) unless those are re-run on the
  same config.

## Status and TODO

Done:

- [x] Re-capture with `O3_SIMD_FMA_OPLAT=6` — zero cycle impact; use opLat=6
  for all future O3 captures.
- [x] Warm-cache measurement (`WARMUP=1`, `--last-call`).
- [x] Remove compiler-inserted in-loop `vsetvli` via `vsetvlmax` — **now in
  `src/gemm.c`**. Net win warm without compression (opt −10%, blocked −2%),
  but a **net loss in the default compressed build** (cold +4% / +10%, warm
  +7% / +19%) until the loop-alignment item below is done.
- [x] I-cache `TaggedPrefetcher` knob (`O3_ICACHE_PF_DEGREE`) and degree sweep.

TODO:

- [ ] **Fix hot-loop line straddling in the compressed build** — try
  `-mllvm -align-loops=64` (or check loop placement with `llvm-objdump-18`
  after each build) and re-measure `vsetvlmax` cold and warm. Decide whether
  to keep `vsetvlmax` in the default build based on that.
- [ ] **Hoist `alpha` out of the inner loop** — accumulate `A·B` unscaled and
  apply `C = alpha·acc + beta·C` once per output row. Removes 8 (opt) /
  16 (blocked) `fmul.d` per inner iteration. At opLat=6 opt's commit blocker
  is the `vfmacc` itself, so this mainly cuts instruction count / frontend
  pressure; expect more benefit for blocked.
- [ ] **I-cache MSHRs** — re-run the prefetcher sweep with I-cache `mshrs`
  4 → 16 (degree 8) to see if the late prefetches become useful.
- [ ] **Confirm the `vsetvli` squash mechanism** in gem5's RISC-V O3 source
  (1–2 squashes per call remain after a `vsetvli` even when warm).
- [ ] **Re-run the roofline sweep** ([gemm_analysis.md](gemm_analysis.md))
  warm and/or with the prefetcher, so the O3 roof-% numbers reflect
  steady-state performance rather than first-touch cost.
