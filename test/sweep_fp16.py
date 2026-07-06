#!/usr/bin/env python3
"""FP16 flag sweep: -force-vector-width in {32} x -force-vector-interleave in {1,2,4}
   Row data size fixed at 1024 bits: M = 1024/64 = 16 (FP64 ref), M = 1024/16 = 64 (FP16).
   Reference row: FP64 best (w=8, il=2).

Usage:
  python3 sweep_fp16.py          -- run the full sweep
  python3 sweep_fp16.py clean    -- remove all sweep_fp16 output logs and all *_m5out dirs
"""

import os
import sys
import shutil
import subprocess

pattern_root       = "/home/ajno5/work/2_pattern/gemm"
pattern_script     = f"{pattern_root}/script"
test_dir           = f"{pattern_root}/test"
sim_config_gem5    = f"{pattern_root}/sim_config/gem5_riscv_demo_riscv_baremetal_semihost.py"
sim_config_whisper = f"{pattern_root}/sim_config/whisper_rv64gcv_config.json"

sys.path.append(pattern_script)
from python_riscv_sim import riscv_sim
from python_log_parser import sim_log_parser

# Row data size fixed at 1024 bits: M = 1024 / (bits per element)
ROW_BITS = 1024

# FP16 sweep parameters
MARCH_FP16        = "rv64gcv_zvfh"
TARGET_FLOAT_FP16 = "_Float16"
SIZEOF_FP16       = 2
M_FP16            = ROW_BITS // (SIZEOF_FP16 * 8)   # 1024/16 = 64
FLOPS_FP16        = 2 * M_FP16 * M_FP16 * M_FP16     # 524,288

WIDTHS_FP16  = [32]      # VLEN=512 / SEW=16
INTERLEAVE   = [1, 2, 4]

# FP64 reference (best from double sweep: w=8, il=2)
MARCH_FP64        = "rv64gcv"
TARGET_FLOAT_FP64 = "double"
SIZEOF_FP64       = 8
M_FP64            = ROW_BITS // (SIZEOF_FP64 * 8)   # 1024/64 = 16
FLOPS_FP64        = 2 * M_FP64 * M_FP64 * M_FP64     # 8,192
W_FP64_REF        = 8
IL_FP64_REF       = 2

binary = f"{test_dir}/gemm_riscv"

def sweep_tags():
    return [f"fp16_w{w}_il{il}" for w in WIDTHS_FP16 for il in INTERLEAVE]

def clean():
    removed = []
    for tag in sweep_tags() + ["fp64_ref"]:
        for f in (f"{test_dir}/sweep_{tag}_whisper.txt",
                  f"{test_dir}/sweep_{tag}_gem5.txt"):
            if os.path.exists(f):
                os.remove(f)
                removed.append(f)
    for entry in os.listdir(test_dir):
        if entry.endswith("_m5out"):
            d = os.path.join(test_dir, entry)
            if os.path.isdir(d):
                shutil.rmtree(d)
                removed.append(d)
    if removed:
        print(f"Removed {len(removed)} file(s)/dir(s):")
        for r in removed:
            print(f"  {r}")
    else:
        print("Nothing to clean.")

if len(sys.argv) > 1 and sys.argv[1] == "clean":
    clean()
    sys.exit(0)

def get(lst, name):
    for d in lst:
        if d["name"] == name:
            return int(d["value"])
    return None

def run_one(tag, march, target_float, sizeof_elem, m, flops, w, il):
    bench_extra   = f"-DM={m} -Dtarget_float={target_float} -mllvm -force-vector-width={w} -mllvm -force-vector-interleave={il}"
    whisper_log   = f"{test_dir}/sweep_{tag}_whisper.txt"
    gem5_log      = f"{test_dir}/sweep_{tag}_gem5.txt"
    m5out_dir     = f"{test_dir}/sweep_{tag}_m5out"

    print(f"\n{'='*60}")
    print(f"  {target_float}  width={w}  interleave={il}")
    print(f"{'='*60}")

    # Build
    for ext in ("", "_flags"):
        try:
            os.remove(f"{binary}{ext}")
        except FileNotFoundError:
            pass
    subprocess.run(
        ["make", "test/gemm_riscv",
         f"MARCH={march}",
         f"BENCH_EXTRA_FLAGS={bench_extra}"],
        cwd=pattern_root, check=False
    )

    # Whisper
    sim_w = riscv_sim()
    sim_w.root         = pattern_root
    sim_w.test_dir     = test_dir
    sim_w.simulator    = "whisper"
    sim_w.test_pattern = binary
    sim_w.sim_config   = sim_config_whisper
    sim_w.logfile      = whisper_log
    sim_w.run_pattern()

    # gem5
    sim_g = riscv_sim()
    sim_g.root         = pattern_root
    sim_g.test_dir     = test_dir
    sim_g.simulator    = "gem5"
    sim_g.test_pattern = binary
    sim_g.sim_config   = sim_config_gem5
    sim_g.logfile      = gem5_log
    sim_g.m5out_dir    = m5out_dir
    sim_g.run_pattern()

    # Parse
    parser = sim_log_parser()
    parser.log_parse(whisper_log, "whisper")
    parser.log_parse(gem5_log,    "gem5")

    mcycle    = get(parser.result["counter"], "mcycle")
    minstret  = get(parser.result["counter"], "minstret")
    vec_load  = get(parser.result["hpm"],     "VectorLoad")
    vec_store = get(parser.result["hpm"],     "VectorStore")

    bytes_per_vec = w * sizeof_elem
    total_vec_mem = (vec_load or 0) + (vec_store or 0)
    bytes_q       = total_vec_mem * bytes_per_vec if total_vec_mem else None
    ai            = flops / bytes_q               if bytes_q        else None
    gflops        = flops / (mcycle / 1e9) / 1e9 if mcycle         else None
    cyc_load      = mcycle / vec_load             if vec_load       else None

    return {
        "dtype": target_float, "m": m, "w": w, "il": il,
        "mcycle": mcycle, "minstret": minstret,
        "VecLoad": vec_load, "VecStore": vec_store,
        "bytes_Q": bytes_q, "AI": ai,
        "cyc_load": cyc_load,
        "MFLOP/s": gflops * 1000 if gflops else None,
    }

results = []

# FP64 reference
results.append(run_one("fp64_ref", MARCH_FP64, TARGET_FLOAT_FP64, SIZEOF_FP64,
                        M_FP64, FLOPS_FP64, W_FP64_REF, IL_FP64_REF))

# FP16 sweep
for w in WIDTHS_FP16:
    for il in INTERLEAVE:
        tag = f"fp16_w{w}_il{il}"
        results.append(run_one(tag, MARCH_FP16, TARGET_FLOAT_FP16, SIZEOF_FP16,
                                M_FP16, FLOPS_FP16, w, il))

# Summary table
def fmt(v, spec, fallback="  N/A"):
    return format(v, spec) if v is not None else fallback

HDR = (f"{'dtype':>10} {'m':>3} {'w':>4} {'il':>3} | "
       f"{'mcycle':>8} {'minstret':>9} | "
       f"{'VecLoad':>8} {'VecSto':>7} | "
       f"{'Q(B)':>8} {'AI':>6} | "
       f"{'cyc/ld':>7} {'MFLOP/s':>9}")
SEP = "-" * len(HDR)
print(f"\n{SEP}\n{HDR}\n{SEP}")
for i, r in enumerate(results):
    if i == 1:
        print(SEP)   # separator between fp64 ref and fp16 sweep
    print(f"{r['dtype']:>10} {r['m']:>3} {r['w']:>4} {r['il']:>3} | "
          f"{fmt(r['mcycle'],   '>8,'):>8} {fmt(r['minstret'],  '>9,'):>9} | "
          f"{fmt(r['VecLoad'],  '>8,'):>8} {fmt(r['VecStore'],  '>7,'):>7} | "
          f"{fmt(r['bytes_Q'],  '>8,'):>8} {fmt(r['AI'],        '>6.3f'):>6} | "
          f"{fmt(r['cyc_load'], '>7.1f'):>7} {fmt(r['MFLOP/s'], '>9.1f'):>9}")
print(SEP)
