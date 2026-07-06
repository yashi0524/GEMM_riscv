#!/usr/bin/env python3
"""Flag sweep: -force-vector-width in {4,8} x -force-vector-interleave in {1,2,4}

Usage:
  python3 sweep.py          -- run the full sweep
  python3 sweep.py clean    -- remove all sweep output logs and m5out dirs
"""

import os
import sys
import shutil
import subprocess

pattern_root       = "/home/ajno5/work/2_pattern/gemm"
pattern_script     = f"{pattern_root}/script"
test_dir           = f"{pattern_root}/test"
sim_config_gem5    = f"{pattern_root}/sim_config/gem5_riscv_demo_riscv_baremetal_semihost_minor.py"
sim_config_whisper = f"{pattern_root}/sim_config/whisper_rv64gcv_config.json"

sys.path.append(pattern_script)
from python_riscv_sim import riscv_sim
from python_log_parser import sim_log_parser

TARGET_FLOAT = "double"
M, N, K   = 8, 16, 8        # N = (VLEN/8/sizeof(double)) * interleave = (512/64)*2
FLOPS     = 2 * M * N * K
PEAK_BW   = 12.8             # GB/s  (DDR3-1600 8x8)

WIDTHS     = [4, 8]
INTERLEAVE = [1, 2, 4]

binary = f"{test_dir}/gemm_riscv"

def sweep_tags():
    return [f"w{w}_il{il}" for w in WIDTHS for il in INTERLEAVE]

def clean():
    removed = []
    for tag in sweep_tags():
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

results = []

for w in WIDTHS:
    for il in INTERLEAVE:
        tag         = f"w{w}_il{il}"
        bench_extra = (f"-DM={M} -DN={N} -DK={K} -Dtarget_float={TARGET_FLOAT} "
                        f"-mllvm -force-vector-width={w} -mllvm -force-vector-interleave={il}")
        whisper_log = f"{test_dir}/sweep_{tag}_whisper.txt"
        gem5_log    = f"{test_dir}/sweep_{tag}_gem5.txt"
        m5out_dir   = f"{test_dir}/sweep_{tag}_m5out"

        print(f"\n{'='*60}")
        print(f"  width={w}  interleave={il}  (M={M} N={N} K={K} {TARGET_FLOAT})")
        print(f"{'='*60}")

        # Build — delete binary first so make always rebuilds
        for ext in ("", "_flags"):
            try:
                os.remove(f"{binary}{ext}")
            except FileNotFoundError:
                pass
        subprocess.run(
            ["make", "test/gemm_riscv", f"BENCH_EXTRA_FLAGS={bench_extra}"],
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

        bytes_per_vec = w * 8                           # vl * sizeof(double)
        total_vec_mem = (vec_load or 0) + (vec_store or 0)
        bytes_q       = total_vec_mem * bytes_per_vec if total_vec_mem else None
        ai            = FLOPS / bytes_q               if bytes_q        else None
        gflops        = FLOPS / (mcycle / 1e9) / 1e9 if mcycle         else None
        cyc_load      = mcycle / vec_load             if vec_load       else None
        attain        = ai * PEAK_BW                  if ai             else None
        eff           = (gflops / attain) * 100       if gflops and attain else None

        results.append({
            "width": w, "interleave": il,
            "mcycle": mcycle, "minstret": minstret,
            "VecLoad": vec_load, "VecStore": vec_store,
            "bytes_Q": bytes_q, "AI": ai,
            "cyc_load": cyc_load,
            "MFLOP/s": gflops * 1000,
            "eff_%": eff,
        })

# Summary table
HDR  = f"{'w':>4} {'il':>4} | {'mcycle':>8} {'minstret':>9} | {'VecLoad':>8} {'VecSto':>7} | {'Q(B)':>8} {'AI':>6} | {'cyc/ld':>7} {'MFLOP/s':>9} {'eff%':>6}"
SEP  = "-" * len(HDR)
def fmt(v, fmt_str, fallback="  N/A"):
    return format(v, fmt_str) if v is not None else fallback

print(f"\n{SEP}\n{HDR}\n{SEP}")
for r in results:
    print(f"{r['width']:>4} {r['interleave']:>4} | "
          f"{fmt(r['mcycle'],  '>8,'):>8} {fmt(r['minstret'], '>9,'):>9} | "
          f"{fmt(r['VecLoad'], '>8,'):>8} {fmt(r['VecStore'], '>7,'):>7} | "
          f"{fmt(r['bytes_Q'], '>8,'):>8} {fmt(r['AI'],       '>6.3f'):>6} | "
          f"{fmt(r['cyc_load'],'>7.1f'):>7} {fmt(r['MFLOP/s'],'>9.1f'):>9} {fmt(r['eff_%'],'>6.1f'):>6}")
print(SEP)
