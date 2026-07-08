# bare_metal_riscv_o3.py — same system as the TimingSimpleCPU/MinorCPU
# configs, but using O3CPU (out-of-order) instead.
import os
import m5
from m5.objects import *

# --- SimdFloatMultAcc (vfmacc) opLat override (diagnostic knob) ---
# O3's stock SIMD_Unit doesn't set an explicit opLat for SimdFloatMultAcc,
# so it silently defaults to OpDesc's opLat=1 — 5-6x more optimistic than
# Minor's tuned MinorDefaultFloatSimdFU (opLat=6) or even O3's own scalar
# FloatMultAcc (opLat=5). Override this to get an apples-to-apples timing
# comparison against Minor/O3-scalar instead of the (probably unrealistic)
# stock default.
#   O3_SIMD_FMA_OPLAT=5 gem5.opt ... this_config.py <binary>
SIMD_FMA_OPLAT = int(os.environ.get("O3_SIMD_FMA_OPLAT", "1"))

class CustomSIMDUnit(SIMD_Unit):
    opList = [
        OpDesc(opClass="SimdAdd"),
        OpDesc(opClass="SimdAddAcc"),
        OpDesc(opClass="SimdAlu"),
        OpDesc(opClass="SimdCmp"),
        OpDesc(opClass="SimdCvt"),
        OpDesc(opClass="SimdMisc"),
        OpDesc(opClass="SimdMult"),
        OpDesc(opClass="SimdMultAcc"),
        OpDesc(opClass="SimdMatMultAcc"),
        OpDesc(opClass="SimdShift"),
        OpDesc(opClass="SimdShiftAcc"),
        OpDesc(opClass="SimdDiv"),
        OpDesc(opClass="SimdSqrt"),
        OpDesc(opClass="SimdFloatAdd"),
        OpDesc(opClass="SimdFloatAlu"),
        OpDesc(opClass="SimdFloatCmp"),
        OpDesc(opClass="SimdFloatCvt"),
        OpDesc(opClass="SimdFloatDiv"),
        OpDesc(opClass="SimdFloatMisc"),
        OpDesc(opClass="SimdFloatMult"),
        OpDesc(opClass="SimdFloatMultAcc", opLat=SIMD_FMA_OPLAT),  # was: default (1)
        OpDesc(opClass="SimdFloatMatMultAcc"),
        OpDesc(opClass="SimdFloatSqrt"),
        OpDesc(opClass="SimdReduceAdd"),
        OpDesc(opClass="SimdReduceAlu"),
        OpDesc(opClass="SimdReduceCmp"),
        OpDesc(opClass="SimdFloatReduceAdd"),
        OpDesc(opClass="SimdFloatReduceCmp"),
        OpDesc(opClass="SimdExt"),
        OpDesc(opClass="SimdFloatExt"),
        OpDesc(opClass="SimdConfig"),
        OpDesc(opClass="SimdDotProd"),
        OpDesc(opClass="SimdAes"),
        OpDesc(opClass="SimdAesMix"),
        OpDesc(opClass="SimdSha1Hash"),
        OpDesc(opClass="SimdSha1Hash2"),
        OpDesc(opClass="SimdSha256Hash"),
        OpDesc(opClass="SimdSha256Hash2"),
        OpDesc(opClass="SimdShaSigma2"),
        OpDesc(opClass="SimdShaSigma3"),
        OpDesc(opClass="SimdSha3"),
        OpDesc(opClass="SimdSm4e"),
        OpDesc(opClass="SimdCrc"),
        OpDesc(opClass="SimdBf16Add"),
        OpDesc(opClass="SimdBf16Cmp"),
        OpDesc(opClass="SimdBf16Cvt"),
        OpDesc(opClass="SimdBf16DotProd"),
        OpDesc(opClass="SimdBf16MatMultAcc"),
        OpDesc(opClass="SimdBf16Mult"),
        OpDesc(opClass="SimdBf16MultAcc"),
    ]
    count = 4

class CustomFUPool(DefaultFUPool):
    FUList = [
        IntALU(),
        IntMultDiv(),
        FP_ALU(),
        FP_MultDiv(),
        ReadPort(),
        CustomSIMDUnit(),
        Matrix_Unit(),
        System_Unit(),
        PredALU(),
        WritePort(),
        RdWrPort(),
    ]

# --- System ---
system = System()
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = "1GHz"
system.clk_domain.voltage_domain = VoltageDomain()
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("512MB")]
system.m5ops_base = 0x10010000   #enables m5ops pseudo-inst decoding

# --- CPU ---
system.cpu = RiscvO3CPU()
# NOTE: "fuPool" is not itself a real BaseO3CPU param (gem5 silently accepts
# unknown SimObject-valued attributes as orphan child nodes, so a typo'd or
# misplaced assignment like system.cpu.fuPool = ... appears in config.json
# but is never read by the model). The functional-unit pool actually used
# lives on each IQUnit in the instQueues vector.
system.cpu.instQueues = [IQUnit(fuPool=CustomFUPool())]
system.cpu.isa = RiscvISA(vlen=512, elen=64)

# --- Memory bus ---
system.membus = SystemXBar()

# --- L1 caches (64 kB each, 4-way) ---
system.cpu.icache = Cache(
    size="64kB",
    assoc=4,
    tag_latency=2,
    data_latency=2,
    response_latency=2,
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

# Halt CPU at tick 0 and wait for GDB to connect before running
#system.workload.wait_for_remote_gdb = True
system.workload.wait_for_remote_gdb = False

# Enable RISC-V semihosting — output goes directly to gem5's stdout
system.workload.semihosting = RiscvSemihosting()

system.cpu.createThreads()

# --- Instantiate & run ---
root = Root(full_system=True, system=system)
m5.instantiate()

print("Starting bare-metal RISC-V M-mode simulation (O3CPU)...")
exit_event = m5.simulate()
print(f"Exit @ tick {m5.curTick()}: {exit_event.getCause()}")
