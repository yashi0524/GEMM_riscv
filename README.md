# dgemm_riscv

## test environment
    simulator : gem5 MinorCPU (in-order, pipelined; TimingSimpleCPU config also
                available at sim_config/gem5_riscv_demo_riscv_baremetal_semihost.py)
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
    whisper:  mcycle = 30,004   minstret = 30,004   Vector = 10,001
    gem5:     mcycle = 60,054   minstret = 30,004
    → 6.0 cycles/vfmacc (gem5); measured peak compute = 2.66 GFLOP/s

    note: fmacc's accumulate chain (vc = vc + va*vb, same register every
    iteration) is a strict serial dependency — MinorCPU's pipelining cannot
    hide this, so cycles/vfmacc is essentially identical to TimingSimpleCPU
    (6.0 vs 6.0). Only load/store-bound kernels like gemm benefit from Minor's
    pipeline overlap (see roofline section below).

## roofline analysis  (gemm 16×16 FP64 kernel)

    --- FLOPs ---
    FLOPs = 2 × M × N × K = 2 × 16 × 16 × 16 = 8,192 FLOP

    --- memory traffic (derived from whisper VectorLoad/VectorStore) ---
    vector element width : 8 × FP64 = 64 bytes per vector instruction (vl=8, VLEN=512)
    bytes loaded  = 1,056 vec loads  × 64 B = 67,584 B
    bytes stored  =   544 vec stores × 64 B = 34,816 B
    total Q       = 102,400 B

    note: Q is identical to the VLEN=256 run — same data is accessed, just in
    half as many (but twice as wide) vector instructions.

    VectorLoad breakdown:
      scale step (beta=0)  :  16 rows × 2 j-blocks =   32 vle64
      accum load C         : 256 (i,k) × 2 j-blocks =  512 vle64
      accum load B         : 256 (i,k) × 2 j-blocks =  512 vle64
      total                :                           1,056  ✓

    VectorStore breakdown:
      scale step (beta=0)  :  16 rows × 2 j-blocks =   32 vse64
      accum store C        : 256 (i,k) × 2 j-blocks =  512 vse64
      total                :                             544  ✓

    --- arithmetic intensity ---
    AI = 8,192 FLOP / 102,400 B = 0.080 FLOP/B
    (AI is VLEN-invariant: same algorithm, same data footprint)

    --- hardware ceilings ---
    peak compute (measured, FMACC micro-bench, gem5 MinorCPU):
      ITERS=10000 vfmacc, vl=8, FP64 → FLOPs = 160,000
      gem5 mcycle = 60,054 (loop-only measurement)  →  6.0 cycles/vfmacc
      peak compute  = (8 × 2 FLOP) / (6.0 cycles / 1 GHz)  = 2.66 GFLOP/s
      (theoretical max = 8 × 2 × 1 GHz = 16.0 GFLOP/s; MinorCPU's default
       float/SIMD functional unit has a 6-cycle op latency, and fmacc's
       accumulate chain is fully serial — one op per opLat cycles, regardless
       of pipelining. TimingSimpleCPU measures the same 6.0 cycles/vfmacc,
       confirming this is an FU-latency limit, not an issue-width limit)
    peak memory BW (DDR3-1600 8x8) = 1600 MT/s × 8 B          = 12.8 GB/s

    --- roofline ---
    ridge point  = 2.66 GFLOP/s / 12.8 GB/s = 0.208 FLOP/B
    kernel AI (0.080) < ridge (0.208)  →  MEMORY BOUND

    attainable perf = AI × peak_BW = 0.080 × 12.8 GB/s = 1.024 GFLOP/s

    --- observed performance (gem5 MinorCPU, VLEN=512, 64KB L1 cache) ---
    T_kernel         = 9,547 cycles / 1 GHz           =   9.55 μs
    achieved FLOP/s  = 8,192 FLOP  / 9.55 μs          = 858.1 MFLOP/s
    achieved BW      = 102,400 B   / 9.55 μs          = 10,727 MB/s = 10.73 GB/s

    --- efficiency ---
    vs attainable BW ceiling : 858.1 MFLOP/s / 1,024 MFLOP/s = 83.8 %
    BW utilization            :  10.73 GB/s  /  12.8 GB/s     = 83.8 %

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
      avg cycles/vfmacc (from FMACC bench) ≈ 6.0 cycles  (FU latency, not
      hidden — fmacc's dependency chain leaves the pipeline nothing to overlap)

    Switching CPU model from TimingSimpleCPU to MinorCPU cuts kernel cycles
    2.48× (23,641 → 9,547) purely by overlapping load latency instead of
    stalling on each access — the algorithm, data footprint, and VLEN are
    unchanged. The remaining gap from 100% BW utilization (83.8% achieved)
    reflects residual pipeline stalls (avg 9.04 cycles/load vs the 4-cycle raw
    L1 hit latency); an OOO core or deeper MSHR/issue width would close more
    of it. fmacc throughput does not improve with this switch, since its
    serial accumulate chain is bound by FU latency, not issue overlap.

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
