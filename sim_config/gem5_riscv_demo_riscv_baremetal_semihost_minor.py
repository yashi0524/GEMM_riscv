# bare_metal_riscv_minor.py — same system as the TimingSimpleCPU config,
# but using MinorCPU (in-order, pipelined) instead of TimingSimpleCPU.
import os
import m5
from m5.objects import *

# --- Float/SIMD functional-unit count override (diagnostic knob) ---
# MinorDefaultFUPool has exactly 1 FloatSimd FU by default (vs 2 IntFUs) —
# that's why a dependency-free vfmacc stream saturates at 1/cycle even with
# MinorCPU's dual-issue width. MINOR_FLOAT_FU_COUNT=2 adds a 2nd instance to
# test whether 2 vfmaccs/cycle (32 GFLOP/s) becomes achievable.
#   MINOR_FLOAT_FU_COUNT=2 gem5.opt ... this_config.py <binary>
FLOAT_FU_COUNT = int(os.environ.get("MINOR_FLOAT_FU_COUNT", "1"))

# --- Front-end knobs (diagnostic) ---
# With 2 FloatSimd FUs the fmacc loop becomes fetch-bound: Minor's default
# fetch1FetchLimit=1 allows one I-cache line fetch in flight, the I-cache
# hit costs tag+data+response = 2+2+2 cycles, and every loop-back taken
# branch makes Fetch2 redirect Fetch1 to a fresh line. See
# doc/microbenchmark.md, "Why not 32 GFLOP/s?".
#   MINOR_FETCH_LIMIT=4    line fetches in flight (default 1 = stock)
#   MINOR_ICACHE_LAT=1     I-cache tag/data/response latency, each
#                          (default unset = 2/2/2 below)
FETCH_LIMIT = int(os.environ.get("MINOR_FETCH_LIMIT", "1"))
ICACHE_LAT = os.environ.get("MINOR_ICACHE_LAT")

class CustomMinorFUPool(MinorDefaultFUPool):
    funcUnits = [
        MinorDefaultIntFU(),
        MinorDefaultIntFU(),
        MinorDefaultIntMulFU(),
        MinorDefaultIntDivFU(),
    ] + [MinorDefaultFloatSimdFU() for _ in range(FLOAT_FU_COUNT)] + [
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

# --- CPU ---
system.cpu = RiscvMinorCPU()
system.cpu.executeFuncUnits = CustomMinorFUPool()
system.cpu.fetch1FetchLimit = FETCH_LIMIT
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

if ICACHE_LAT:
    system.cpu.icache.tag_latency = int(ICACHE_LAT)
    system.cpu.icache.data_latency = int(ICACHE_LAT)
    system.cpu.icache.response_latency = int(ICACHE_LAT)

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

print("Starting bare-metal RISC-V M-mode simulation (MinorCPU)...")
exit_event = m5.simulate()
print(f"Exit @ tick {m5.curTick()}: {exit_event.getCause()}")
