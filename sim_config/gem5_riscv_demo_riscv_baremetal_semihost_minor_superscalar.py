# bare_metal_riscv_minor_superscalar.py — MinorCPU customized into an
# in-order, dual-issue (2-wide superscalar) core with a separate vector pipe. Same
# system / cache / memory setup as the MinorCPU config, so results compare
# directly against it.
#
# Design point: an in-order dual-issue scalar pipeline plus a decoupled
# vector unit whose datapath (DLEN) is narrower than VLEN — the common shape
# of in-order RVV 1.0 application cores. What changes vs. stock MinorCPU:
#
#   front end  stock Minor predicts branches in Fetch2 with one I-cache line
#              fetch in flight (~6-cycle bubble per taken branch). Real
#              in-order superscalar cores predict at fetch (BTB looked up
#              with the fetch address). Approximated here with 2 fetches in
#              flight and a 1-cycle I-cache, which brings the taken-branch
#              cost to ~2.8 cycles (doc/microbenchmark.md).
#   vector     vector ops get their own pipe, separate from scalar FP, so a
#              scalar FP op and a vector op can dual-issue. The pipe is
#              occupied VLEN/DLEN cycles per vector register (issueLat):
#              gem5 cracks an LMUL>1 op into one micro-op per register, so
#              an LMUL=m op occupies it m * VLEN/DLEN cycles.
#   vsetvli    SimdConfig executes on the integer ALUs (scalar pipeline),
#              not in the vector pipe.
#
# Not modelled: fetch-stage branch prediction proper (still Fetch2), lane-
# specific pairing rules, a vector instruction queue / chaining, and
# DLEN-limited vector load/store bandwidth (vector memory ops still go
# through the single Mem FU at one per cycle). See doc/microbenchmark.md.
#
# Knobs (environment, defaults in brackets):
#   VPU_DLEN=256       vector datapath width in bits [256]; VLEN is fixed at
#                      512 below, so issueLat = 512/256 = 2
#   VPU_OPLAT=6        vector op latency [6]; dependent vector ops issue
#                      every opLat-2 cycles (srcRegsRelativeLats=[2])
#   SCALAR_FP_OPLAT=6  scalar FP op latency [6]
import os
import m5
from m5.objects import *

VLEN = 512
DLEN = int(os.environ.get("VPU_DLEN", "256"))
assert VLEN % DLEN == 0 and DLEN <= VLEN, "VPU_DLEN must divide VLEN=512"
VPU_ISSUE_LAT = VLEN // DLEN
VPU_OPLAT = int(os.environ.get("VPU_OPLAT", "6"))
SCALAR_FP_OPLAT = int(os.environ.get("SCALAR_FP_OPLAT", "6"))

# Split MinorDefaultFloatSimdFU's op classes into scalar FP and vector.
SCALAR_FP_OPS = [
    "FloatAdd", "FloatCmp", "FloatCvt", "FloatMisc", "FloatMult",
    "FloatMultAcc", "FloatDiv", "FloatSqrt", "Bf16Cvt",
]
VECTOR_OPS = [
    "SimdAdd", "SimdAddAcc", "SimdAlu", "SimdCmp", "SimdCvt", "SimdMisc",
    "SimdMult", "SimdMultAcc", "SimdMatMultAcc", "SimdShift", "SimdShiftAcc",
    "SimdDiv", "SimdSqrt", "SimdFloatAdd", "SimdFloatAlu", "SimdFloatCmp",
    "SimdFloatCvt", "SimdFloatDiv", "SimdFloatMisc", "SimdFloatMult",
    "SimdFloatMultAcc", "SimdFloatMatMultAcc", "SimdFloatSqrt",
    "SimdReduceAdd", "SimdReduceAlu", "SimdReduceCmp", "SimdFloatReduceAdd",
    "SimdFloatReduceCmp", "SimdAes", "SimdAesMix", "SimdSha1Hash",
    "SimdSha1Hash2", "SimdSha256Hash", "SimdSha256Hash2", "SimdShaSigma2",
    "SimdShaSigma3", "SimdSha3", "SimdSm4e", "SimdCrc", "Matrix", "MatrixMov",
    "MatrixOP", "SimdExt", "SimdFloatExt", "SimdDotProd", "SimdBf16Add",
    "SimdBf16Cmp", "SimdBf16Cvt", "SimdBf16DotProd", "SimdBf16MatMultAcc",
    "SimdBf16Mult", "SimdBf16MultAcc",
]


class IntAluConfigFU(MinorDefaultIntFU):
    """Integer ALU that also executes vsetvli (SimdConfig)."""
    opClasses = minorMakeOpClassSet(["IntAlu", "SimdConfig"])


class ScalarFpFU(MinorFU):
    opClasses = minorMakeOpClassSet(SCALAR_FP_OPS)
    timings = [MinorFUTiming(description="ScalarFp", srcRegsRelativeLats=[2])]
    opLat = SCALAR_FP_OPLAT


class VectorFU(MinorFU):
    opClasses = minorMakeOpClassSet(VECTOR_OPS)
    timings = [MinorFUTiming(description="Vector", srcRegsRelativeLats=[2])]
    opLat = VPU_OPLAT
    issueLat = VPU_ISSUE_LAT


class InOrderDualIssueFUPool(MinorFUPool):
    funcUnits = [
        IntAluConfigFU(),
        IntAluConfigFU(),
        MinorDefaultIntMulFU(),
        MinorDefaultIntDivFU(),
        ScalarFpFU(),
        VectorFU(),
        MinorDefaultPredFU(),
        MinorDefaultMemFU(),
        MinorDefaultMiscFU(),
    ]


# --- System ---
system = System()
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = "1GHz"
system.clk_domain.voltage_domain = VoltageDomain()
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("512MB")]
system.m5ops_base = 0x10010000   #enables m5ops pseudo-inst decoding

# --- CPU: in-order, 2-wide decode / issue / commit (Minor defaults) ---
system.cpu = RiscvMinorCPU()
system.cpu.executeFuncUnits = InOrderDualIssueFUPool()
system.cpu.fetch1FetchLimit = 2
system.cpu.isa = RiscvISA(vlen=VLEN, elen=64)

# --- Memory bus ---
system.membus = SystemXBar()

# --- L1 caches (64 kB each, 4-way); 1-cycle I-cache for the faster front end ---
system.cpu.icache = Cache(
    size="64kB",
    assoc=4,
    tag_latency=1,
    data_latency=1,
    response_latency=1,
    mshrs=4,
    tgts_per_mshr=20,
)
system.cpu.dcache = Cache(
    size="64kB",
    assoc=4,
    tag_latency=2,
    data_latency=2,
    response_latency=2,
    mshrs=4,
    tgts_per_mshr=20,
)

# --- Connect CPU → L1 caches → membus ---
system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.icache.mem_side = system.membus.cpu_side_ports
system.cpu.dcache.cpu_side = system.cpu.dcache_port
system.cpu.dcache.mem_side = system.membus.cpu_side_ports

# --- Interrupt controller (no interrupt bus wiring needed for RISC-V) ---
system.cpu.createInterruptController()

# --- Memory controller ---
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# --- System port ---
system.system_port = system.membus.cpu_side_ports

# --- Bare-metal workload (M-mode, no BBL/Linux) ---
system.workload = RiscvBareMetal()
system.workload.bootloader = sys.argv[1]  # ELF entry must be at 0x80000000
system.workload.wait_for_remote_gdb = False

# Enable RISC-V semihosting — output goes directly to gem5's stdout
system.workload.semihosting = RiscvSemihosting()

system.cpu.createThreads()

# --- Instantiate & run ---
root = Root(full_system=True, system=system)
m5.instantiate()

print(f"Starting bare-metal RISC-V M-mode simulation (MinorCPU superscalar, "
      f"VLEN={VLEN} DLEN={DLEN})...")
exit_event = m5.simulate()
print(f"Exit @ tick {m5.curTick()}: {exit_event.getCause()}")
