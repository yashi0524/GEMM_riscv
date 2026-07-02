# dgemm_riscv

## test environment
    simulator : gem5 TimingSimpleCPU
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
          scalar_dgemm(). hpmcounterN values from whisper only (see run_log.txt).

    --- whisper (functional, VLEN=512) ---
    mcycle        =   8,795
    minstret      =   8,796
    Vector        =   1,072   (vector compute instructions, event 61)
    VectorLoad    =   1,056   (vector load instructions,   event 64)
    VectorStore   =     544   (vector store instructions,  event 65)

    --- gem5 TimingSimpleCPU (cycle-accurate, VLEN=512, 64KB L1 I/D cache) ---
    mcycle        =  26,924
    minstret      =   8,796
    IPC           =   0.327   (8,796 / 26,924)
    CPI           =   3.06    (26,924 / 8,796)

    --- FMADD micro-benchmark (peak compute probe, ITERS=10000, vl=8, FP64) ---
    whisper:  mcycle = 30,004   minstret = 30,004   Vector = 10,001
    gem5:     mcycle = 60,016   minstret = 30,004
    → 4.0 cycles/vfmadd (gem5); measured peak compute = 4.0 GFLOP/s

## roofline analysis  (dgemm 16×16 FP64 kernel)

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
    peak compute (measured, FMADD micro-bench, gem5):
      ITERS=10000 vfmadd, vl=8, FP64 → FLOPs = 160,000
      gem5 mcycle = 60,016;  scalar instr ≈ 20,003 × 1 cycle = 20,003 cycles
      vfmadd cycles = 60,016 − 20,003 = 40,013  →  4.0 cycles/vfmadd
      peak compute  = (8 × 2 FLOP) / (4 cycles / 1 GHz)  = 4.0 GFLOP/s
      (theoretical max = 8 × 2 × 1 GHz = 16.0 GFLOP/s; TimingSimpleCPU
       issues one instruction per cycle but vfmadd has 4-cycle execute latency)
    peak memory BW (DDR3-1600 8x8) = 1600 MT/s × 8 B          = 12.8 GB/s

    --- roofline ---
    ridge point  = 4.0 GFLOP/s / 12.8 GB/s = 0.3125 FLOP/B
    kernel AI (0.080) < ridge (0.3125)  →  MEMORY BOUND

    attainable perf = AI × peak_BW = 0.080 × 12.8 GB/s = 1.024 GFLOP/s

    --- observed performance (gem5, VLEN=512, 64KB L1 cache) ---
    T_kernel         = 26,924 cycles / 1 GHz          =  26.9 μs
    achieved FLOP/s  = 8,192 FLOP  / 26.9 μs          = 304.3 MFLOP/s
    achieved BW      = 102,400 B   / 26.9 μs           = 3,806 MB/s = 3.81 GB/s

    --- efficiency ---
    vs attainable BW ceiling : 304.3 MFLOP/s / 1,024 MFLOP/s = 29.7 %
    BW utilization            :   3.81 GB/s  /  12.8 GB/s     = 29.8 %

    --- vs VLEN=256 baseline (same cache config) ---
    mcycle        : 35,611 → 26,924   (1.32× speedup)
    FLOP/s        : 230.0  → 304.3 MFLOP/s
    efficiency    : 22.5%  → 29.7%
    avg cyc/load  : 16.9   → 25.5 cycles  (wider loads, half as many)

    --- bottleneck: L1 cache stall latency (in-order, stall-on-access CPU) ---
    All three 16×16 FP64 matrices (~6 KB) fit in the 64 KB L1 D-cache.
    TimingSimpleCPU stalls the pipeline on every load until the L1 responds.

      avg cycles/load = 26,924 / 1,056 ≈ 25.5 cycles  (64-byte wide load)
      L1 cache hit latency (config): tag_latency=2 + data_latency=2 = 4 cycles
      avg cycles/vfmadd (from FMADD bench) ≈ 4.0 cycles

    Per-load cycle count is higher than VLEN=256 (25.5 vs 16.9) because 64-byte
    loads span a full cache line and incur more internal pipeline steps. However,
    total cycles are lower (26,924 vs 35,611) because there are half as many
    load instructions. The remaining gap from attainable (29.7% efficiency)
    reflects sequential issue overhead — an OOO core would overlap load latency.

## notes
    1. gem5 hpmcounterN are NOT valid event counts: TimingSimpleCPU does not
       implement HPM performance events; the counters accumulate raw cycles
       from simulation start. Use whisper hpmcounterN for event counts.

    2. Whisper VLEN=512 (bytes_per_vec=64) and gem5 VLEN=512 (vlen=512) match.
       -force-vector-width=8 directly instructs clang to use vl=8 (8 × FP64 =
       64 bytes per vector op), consistent with VLEN=512.

    3. The 64 KB L1 caches deliver a 25× cycle reduction vs the no-cache
       baseline (891,693 → 35,611 cycles at VLEN=256). Widening to VLEN=512
       adds a further 1.32× on top, reaching 26,924 cycles. Further gains
       would require OOO execution or software pipelining to hide L1-hit stalls.

    misa = 0x800000000034112D  →  RV64 I M A F D C V
