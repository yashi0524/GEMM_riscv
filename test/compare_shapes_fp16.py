#!/usr/bin/env python3
"""FP16 shape comparison: (M=K=16, N=32) vs (M=N=K=32), fixed w=32, il=2.

Usage:
  python3 compare_shapes_fp16.py          -- run both shapes
  python3 compare_shapes_fp16.py clean    -- remove output logs and m5out dirs
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

MARCH_FP16        = "rv64gcv_zvfh"
TARGET_FLOAT_FP16 = "_Float16"
SIZEOF_FP16       = 2
W  = 32
IL = 2

SHAPES = [
    ("MK16_N32", 16, 32, 16),   # tag, M, N, K
    ("M=N=K=32", 32, 32, 32),
]

binary = f"{test_dir}/gemm_riscv"

def sweep_tags():
    return [f"shape_{tag}_w{W}_il{IL}" for tag, *_ in SHAPES]

def clean():
    removed = []
    for tag in sweep_tags():
        for f in (f"{test_dir}/{tag}_whisper.txt",
                  f"{test_dir}/{tag}_gem5.txt"):
            if os.path.exists(f):
                os.remove(f)
                removed.append(f)
    for entry in os.listdir(test_dir):
        if entry.endswith("_m5out") and entry.startswith("shape_"):
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

def run_one(tag, m, n, k, w, il):
    flops = 2 * m * n * k
    bench_extra = (f"-DM={m} -DN={n} -DK={k} -Dtarget_float={TARGET_FLOAT_FP16} "
                    f"-mllvm -force-vector-width={w} -mllvm -force-vector-interleave={il}")
    full_tag      = f"shape_{tag}_w{w}_il{il}"
    whisper_log   = f"{test_dir}/{full_tag}_whisper.txt"
    gem5_log      = f"{test_dir}/{full_tag}_gem5.txt"
    m5out_dir     = f"{test_dir}/{full_tag}_m5out"

    print(f"\n{'='*60}")
    print(f"  {tag}  M={m} N={n} K={k}  width={w}  interleave={il}")
    print(f"{'='*60}")

    for ext in ("", "_flags"):
        try:
            os.remove(f"{binary}{ext}")
        except FileNotFoundError:
            pass
    subprocess.run(
        ["make", "test/gemm_riscv",
         f"MARCH={MARCH_FP16}",
         f"BENCH_EXTRA_FLAGS={bench_extra}"],
        cwd=pattern_root, check=False
    )

    sim_w = riscv_sim()
    sim_w.root         = pattern_root
    sim_w.test_dir     = test_dir
    sim_w.simulator    = "whisper"
    sim_w.test_pattern = binary
    sim_w.sim_config   = sim_config_whisper
    sim_w.logfile      = whisper_log
    sim_w.run_pattern()

    sim_g = riscv_sim()
    sim_g.root         = pattern_root
    sim_g.test_dir     = test_dir
    sim_g.simulator    = "gem5"
    sim_g.test_pattern = binary
    sim_g.sim_config   = sim_config_gem5
    sim_g.logfile      = gem5_log
    sim_g.m5out_dir    = m5out_dir
    sim_g.run_pattern()

    parser = sim_log_parser()
    parser.log_parse(whisper_log, "whisper")
    parser.log_parse(gem5_log,    "gem5")

    mcycle    = get(parser.result["counter"], "mcycle")
    minstret  = get(parser.result["counter"], "minstret")
    vec_load  = get(parser.result["hpm"],     "VectorLoad")
    vec_store = get(parser.result["hpm"],     "VectorStore")

    bytes_per_vec = w * SIZEOF_FP16
    total_vec_mem = (vec_load or 0) + (vec_store or 0)
    bytes_q       = total_vec_mem * bytes_per_vec if total_vec_mem else None
    ai            = flops / bytes_q               if bytes_q        else None
    gflops        = flops / (mcycle / 1e9) / 1e9 if mcycle         else None
    cyc_load      = mcycle / vec_load             if vec_load       else None

    return {
        "shape": tag, "m": m, "n": n, "k": k, "w": w, "il": il,
        "flops": flops,
        "mcycle": mcycle, "minstret": minstret,
        "VecLoad": vec_load, "VecStore": vec_store,
        "bytes_Q": bytes_q, "AI": ai,
        "cyc_load": cyc_load,
        "MFLOP/s": gflops * 1000 if gflops else None,
    }

results = []
for tag, m, n, k in SHAPES:
    results.append(run_one(tag, m, n, k, W, IL))

def fmt(v, spec, fallback="  N/A"):
    return format(v, spec) if v is not None else fallback

HDR = (f"{'shape':>10} {'m':>3} {'n':>3} {'k':>3} {'w':>4} {'il':>3} | "
       f"{'FLOPs':>8} {'mcycle':>8} {'minstret':>9} | "
       f"{'VecLoad':>8} {'VecSto':>7} | "
       f"{'Q(B)':>8} {'AI':>6} | "
       f"{'cyc/ld':>7} {'MFLOP/s':>9}")
SEP = "-" * len(HDR)
print(f"\n{SEP}\n{HDR}\n{SEP}")
for r in results:
    print(f"{r['shape']:>10} {r['m']:>3} {r['n']:>3} {r['k']:>3} {r['w']:>4} {r['il']:>3} | "
          f"{fmt(r['flops'],    '>8,'):>8} "
          f"{fmt(r['mcycle'],   '>8,'):>8} {fmt(r['minstret'],  '>9,'):>9} | "
          f"{fmt(r['VecLoad'],  '>8,'):>8} {fmt(r['VecStore'],  '>7,'):>7} | "
          f"{fmt(r['bytes_Q'],  '>8,'):>8} {fmt(r['AI'],        '>6.3f'):>6} | "
          f"{fmt(r['cyc_load'], '>7.1f'):>7} {fmt(r['MFLOP/s'], '>9.1f'):>9}")
print(SEP)

if len(results) == 2 and results[0]["mcycle"] and results[1]["mcycle"]:
    r0, r1 = results
    print(f"\nSpeedup ({r1['shape']} vs {r0['shape']}):")
    print(f"  mcycle ratio   : {r1['mcycle']/r0['mcycle']:.3f}x  ({r1['mcycle']:,} / {r0['mcycle']:,})")
    print(f"  FLOPs ratio    : {r1['flops']/r0['flops']:.3f}x")
    if r0['MFLOP/s'] and r1['MFLOP/s']:
        print(f"  MFLOP/s ratio  : {r1['MFLOP/s']/r0['MFLOP/s']:.3f}x  ({r1['MFLOP/s']:.1f} vs {r0['MFLOP/s']:.1f})")
