# GEMM Performance Analysis on RISC-V Vector (RVV)

## Summary

This repo measures GEMM kernel performance on RISC-V's Vector extension
(RVV) across CPU models (gem5 TimingSimpleCPU / MinorCPU / O3CPU, plus
whisper for functional event counts), two datatypes (FP64, FP16), and two
kernel implementations (`scalar_gemm`, the original auto-vectorized triple
loop; `opt_gemm`, a hand-vectorized rewrite). The core question is a
roofline one: how close does each configuration get to its hardware's
compute ceiling, and what's actually stopping it?

> **Note:** the Brain Float series (BF16/BF8) is not covered by this sweep.
> whisper's ISA string already includes `zfbfmin`/`zvfbfmin`/`zvfbfwma`, and
> the Makefile has a `__bf16` build-flag comment (`rv64gcv_zvfbfmin1p0_zvfbfwma1p0`),
> but gem5 doesn't support decoding these RVV BF16 extension instructions in
> this environment — the O3 `CustomFUPool` config lists `SimdBf16*` op
> classes, but that alone doesn't imply the ISA decoder accepts the
> instructions. Without gem5 timing coverage there'd be no cycle-accurate
> half of the comparison, so BF16/BF8 were left out rather than tested
> functionally-only.

**Key findings:**
- At this problem size (`M=N=K=16` FP64 / `M=16,N=64,K=16` FP16), GEMM is
  **deeply memory-bound** on every CPU model tested — every measured point
  lands far left of its compute-roof ridge point (only 0.1–9.4% of peak
  compute reached; see the roofline chart below).
- **MinorCPU → O3CPU** gives a genuine 2–3× cycle-count speedup even for
  this memory-bound kernel, purely from better overlap of independent
  loads — not from faster compute.
- **`opt_gemm`** (register-resident accumulator + `fmacc`-style 8-way
  unroll, mirroring the peak-compute micro-benchmark's own unroll
  technique) roughly **doubles `roof%`** on every dtype/core combination by
  fixing redundant memory traffic and breaking a serial dependency chain —
  but is still far from compute-bound; a redundant reload of `B` across
  rows (not yet fixed) is the next limiter.

![GEMM roofline: FP64/FP16 × MinorCPU/O3CPU vs. fmacc peak compute](doc/gemm_roofline.svg)

See [doc/microbenchmark.md](doc/microbenchmark.md) for the FMACC peak-compute
micro-benchmark (unroll progression, IPC analysis, and why functional-unit
latency isn't the bottleneck) that backs the peak-compute ceiling used below.

See [doc/gemm_analysis.md](doc/gemm_analysis.md) for the broader `test/sweep.py`
roofline analysis across FP64/FP16 × MinorCPU/O3CPU (the data behind the
chart above), plotted against the FMACC peak-compute ceilings. Its FP64
sweep uses this same `M=N=K=16` kernel shape, so the two documents'
FP64/MinorCPU numbers cross-validate each other exactly. That doc also has
the full `scalar_gemm` vs. `opt_gemm` comparison summarized above.

## test environment
    simulator : gem5 MinorCPU (in-order, pipelined). Alternate configs also
                available: TimingSimpleCPU
                (sim_config/gem5_riscv_demo_riscv_baremetal_semihost.py) and
                O3CPU, out-of-order
                (sim_config/gem5_riscv_demo_riscv_baremetal_semihost_o3.py —
                see doc/microbenchmark.md for a comparison and an important
                caveat about its default vector-FMA latency).
    ISA       : RV64GCV  (VLEN=512 bits, ELEN=64 bits)
    clock     : 1 GHz
    cache     : 64 KB L1 I-cache + 64 KB L1 D-cache (4-way, 2-cycle latency)
    memory    : DDR3-1600 8x8
    OS        : bare-metal M-mode (semihosting I/O)
    toolchain : xpack-riscv-none-elf-gcc-13.2.0 / clang-18
    flags     : --target=riscv64-unknown-elf -march=rv64gcv -O3
                -mllvm -force-vector-width=8

## kernel
    operation : C = alpha*A*B + beta*C  (scalar DGEMM, i-k-j loop order)
    M=N=K     : 16
    dtype     : double (float64)
    alpha=1.0, beta=0.0

    Compiled with -force-vector-width=8: clang vectorizes the inner j-loop
    with vl=8, sew=64 (8 × FP64 = 64 bytes per vector op), matching VLEN=512.

## simulation results  (kernel-only CSR deltas)
    note: mcycle/minstret are READ_CSR deltas taken before and after
          scalar_gemm(). hpmcounterN values from whisper only (see run_log.txt).

    --- whisper (functional, VLEN=512) ---
    mcycle        =   8,796
    minstret      =   8,797
    Vector        =   1,072   (vector compute instructions, event 61)
    VectorLoad    =   1,056   (vector load instructions,   event 64)
    VectorStore   =     544   (vector store instructions,  event 65)

    --- gem5 MinorCPU (in-order pipelined, VLEN=512, 64KB L1 I/D cache) ---
    mcycle        =   9,547
    minstret      =   8,797
    IPC           =   0.921   (8,797 / 9,547)
    CPI           =   1.085   (9,547 / 8,797)

    --- FMACC micro-benchmark (peak compute probe, ITERS=10000, vl=8, FP64) ---
    Full unroll progression (serial → x4 → x8 → x12 → x16), IPC analysis, and
    the functional-unit-latency investigation are in doc/microbenchmark.md.
    Summary: unrolling into independent accumulators broke a serial
    dependency chain that MinorCPU's dual-issue pipeline couldn't otherwise
    fill; the best result (x16, 16 independent accumulators) reached
    gem5 mcycle = 10,170 for total_ops = 10,000 → 1.02 cycles/vfmacc →
    15.73 GFLOP/s (98.3% of the 16.0 GFLOP/s theoretical max), used below as
    the compute ceiling for the gemm roofline analysis.

## roofline analysis  (gemm 16×16 FP64 kernel, gem5 MinorCPU)

    See doc/gemm_analysis.md for the general roofline methodology (FLOPs, AI,
    ridge point, hardware ceilings sourced from doc/microbenchmark.md) and
    its extension to FP64/FP16 across MinorCPU/O3CPU via test/sweep.py. This
    section keeps only what's unique to this specific M=N=K=16 kernel: its
    exact memory-traffic breakdown, and a CPU-model comparison
    (TimingSimpleCPU vs MinorCPU) the sweep-based analysis doesn't cover.

    --- FLOPs & memory traffic (derived from whisper VectorLoad/VectorStore) ---
    FLOPs = 2 × M × N × K = 2 × 16 × 16 × 16 = 8,192 FLOP
    vector element width : 8 × FP64 = 64 bytes per vector instruction (vl=8, VLEN=512)
    bytes loaded  = 1,056 vec loads  × 64 B = 67,584 B
    bytes stored  =   544 vec stores × 64 B = 34,816 B
    total Q       = 102,400 B  →  AI = 8,192 / 102,400 = 0.080 FLOP/B

    VectorLoad breakdown:
      scale step (beta=0)  :  16 rows × 2 j-blocks =   32 vle64
      accum load C         : 256 (i,k) × 2 j-blocks =  512 vle64
      accum load B         : 256 (i,k) × 2 j-blocks =  512 vle64
      total                :                           1,056  ✓

    VectorStore breakdown:
      scale step (beta=0)  :  16 rows × 2 j-blocks =   32 vse64
      accum store C        : 256 (i,k) × 2 j-blocks =  512 vse64
      total                :                             544  ✓

    --- observed performance (gem5 MinorCPU, VLEN=512, 64KB L1 cache) ---
    T_kernel         = 9,547 cycles / 1 GHz           =   9.55 μs
    achieved FLOP/s  = 8,192 FLOP  / 9.55 μs          = 858.1 MFLOP/s
    achieved BW      = 102,400 B   / 9.55 μs          = 10,727 MB/s = 10.73 GB/s
    efficiency vs attainable BW ceiling (AI × 12.8 GB/s peak = 1.024 GFLOP/s)
      = 858.1 / 1,024 = 83.8 %  (= BW utilization: 10.73 / 12.8 GB/s)

    --- vs TimingSimpleCPU baseline (same VLEN=512, cache config) ---
    mcycle        : 23,641 → 9,547    (2.48× speedup)
    FLOP/s        : 346.5  → 858.1 MFLOP/s
    efficiency    : 33.8%  → 83.8%
    avg cyc/load  : 22.4   → 9.04 cycles  (pipelining hides most stall latency)

    --- bottleneck: L1 cache access latency, partially hidden by pipelining ---
    All three 16×16 FP64 matrices (~6 KB) fit in the 64 KB L1 D-cache.
    Unlike TimingSimpleCPU (which fully stalls the pipeline on every load until
    the L1 responds), MinorCPU overlaps fetch/decode/issue across independent
    loads, hiding most of the per-access latency.

      avg cycles/load = 9,547 / 1,056 ≈ 9.04 cycles  (64-byte wide load)
      L1 cache hit latency (config): tag_latency=2 + data_latency=2 = 4 cycles
      avg cycles/vfmacc (from FMACC bench): 6.0 (serial) → 1.02 (x16 unrolled)
      — see doc/microbenchmark.md

    Switching CPU model from TimingSimpleCPU to MinorCPU cuts kernel cycles
    2.48× (23,641 → 9,547) purely by overlapping load latency instead of
    stalling on each access — the algorithm, data footprint, and VLEN are
    unchanged. The remaining gap from 100% BW utilization (83.8% achieved)
    reflects residual pipeline stalls (avg 9.04 cycles/load vs the 4-cycle raw
    L1 hit latency); an OOO core or deeper MSHR/issue width would close more
    of it. fmacc needed the same principle applied by hand (breaking its
    single dependency chain via unrolling) to see any benefit from Minor's
    pipelining at all — see doc/microbenchmark.md for that investigation.

## notes
    1. gem5 hpmcounterN are NOT valid event counts on either CPU model:
       neither TimingSimpleCPU nor MinorCPU implements HPM performance events;
       the counters accumulate raw cycles from simulation start. Use whisper
       hpmcounterN for real event counts.

    2. Whisper VLEN=512 (bytes_per_vec=64) and gem5 VLEN=512 (vlen=512) match.
       -force-vector-width=8 directly instructs clang to use vl=8 (8 × FP64 =
       64 bytes per vector op), consistent with VLEN=512.

    3. Switching gem5's CPU model from TimingSimpleCPU to MinorCPU (same
       VLEN=512, same 64KB L1 cache config) cuts gemm kernel cycles 2.48×
       (23,641 → 9,547) by overlapping L1 load latency across the pipeline
       instead of stalling fully on every access. This is a CPU-model effect,
       independent of the earlier VLEN=256→512 width study; the two are not
       directly comparable since that study predates this switch. Further
       gains beyond MinorCPU's 83.8% BW efficiency would require OOO execution
       or more MSHRs/issue width to hide the residual per-load stall.

    misa = 0x800000000034112D  →  RV64 I M A F D C V
