#!/usr/bin/env python3
"""Multi-variant flag sweep: -force-vector-width x -force-vector-interleave,
   per variant, driven by test/config/sweep_config.json.
   Each variant (e.g. "fp64_o3", "fp64_minor", "fp16_o3", "fp16_minor") gets
   its own march/target_float/widths/interleave/shape/gem5_core/test_enable/
   peak_gflops. A "_template" entry holds defaults shared across variants;
   each variant config is template-then-variant shallow-merged, so a variant
   only needs to list the fields where it differs from (or is missing from)
   the template. By default the "*_o3" variants are enabled and "*_minor"
   disabled, so the two dtypes can be swept side by side without also
   doubling runs per core.
   N normally derives as source_A_width_bits / data_format (data_format =
   the element width in bits, e.g. 64 for fp64, 16 for fp16); "shape" gives
   M/K explicitly (or both default to N for a square shape if "shape" is
   omitted), and an explicit "n" inside "shape" overrides the derived value
   — the default template sets shape M=N=K=16 for every variant this way.
   gem5_core selects the gem5 CPU model per variant: "timing" (RiscvTimingSimpleCPU),
   "minor" (RiscvMinorCPU), or "o3" (RiscvO3CPU).
   test_enable: true/false toggles whether a variant's sweep runs at all.
   peak_gflops: the compute-roof ceiling for this variant, taken from the
   fmacc/fmacc_fp16 x16-unroll peak-compute results in doc/microbenchmark.md
   (not derived from this sweep) — used for the table's "roof%" column.

Usage:
  python3 sweep.py          -- run the full sweep
  python3 sweep.py clean    -- remove all sweep output logs and all *_m5out dirs
"""

import os
import sys
import json
import shutil
import subprocess

pattern_root       = "/home/ajno5/work/2_pattern/gemm"
pattern_script     = f"{pattern_root}/script"
test_dir           = f"{pattern_root}/test"
sweep_config_path  = f"{test_dir}/config/sweep_config.json"
sim_config_whisper = f"{pattern_root}/sim_config/whisper_rv64gcv_config.json"

GEM5_CORE_CONFIGS = {
    "timing": f"{pattern_root}/sim_config/gem5_riscv_demo_riscv_baremetal_semihost.py",
    "minor":  f"{pattern_root}/sim_config/gem5_riscv_demo_riscv_baremetal_semihost_minor.py",
    "o3":     f"{pattern_root}/sim_config/gem5_riscv_demo_riscv_baremetal_semihost_o3.py",
}

sys.path.append(pattern_script)
from python_riscv_sim import riscv_sim
from python_log_parser import sim_log_parser

PEAK_BW  = 12.8   # GB/s  (DDR3-1600 8x8)

SIZEOF_BY_TARGET_FLOAT = {"double": 8, "_Float16": 2}

TEMPLATE_KEY = "_template"

def load_sweep_cfg(path):
    raw      = json.load(open(path))
    template = raw.get(TEMPLATE_KEY, {})
    return {
        variant: {**template, **variant_cfg}
        for variant, variant_cfg in raw.items() if variant != TEMPLATE_KEY
    }

sweep_cfg = load_sweep_cfg(sweep_config_path)
VARIANTS  = [v for v in sweep_cfg if sweep_cfg[v].get("test_enable", True)]

def variant_params(variant):
    cfg         = sweep_cfg[variant]
    sizeof      = SIZEOF_BY_TARGET_FLOAT[cfg["target_float"]]
    data_format = sizeof * 8                                       # element width in bits
    n           = cfg["source_A_width_bits"] // data_format        # n = source_A_width_bits / data_format
    if "shape" in cfg:
        m, k = cfg["shape"]["m"], cfg["shape"]["k"]
        n     = cfg["shape"].get("n", n)                           # explicit n overrides the derived value
    else:
        m = k = n                                                  # square fallback
    flops = 2 * m * n * k
    return {
        "march":        cfg["march"],
        "target_float": cfg["target_float"],
        "sizeof":       sizeof,
        "m": m, "n": n, "k": k,
        "flops":        flops,
        "widths":       cfg["widths"],
        "interleave":   cfg["interleave"],
        "gem5_core":    cfg["gem5_core"],
        "sim_config_gem5": GEM5_CORE_CONFIGS[cfg["gem5_core"]],
        "peak_gflops":  cfg["peak_gflops"],
    }

binary = f"{test_dir}/gemm_riscv"

def sweep_tags():
    tags = []
    for variant in VARIANTS:
        p = variant_params(variant)
        tags += [f"{variant}_w{w}_il{il}" for w in p["widths"] for il in p["interleave"]]
    return tags

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

def run_one(tag, march, target_float, sizeof_elem, m, n, k, flops, w, il, sim_config_gem5, gem5_core, peak_gflops):
    bench_extra   = (f"-DM={m} -DN={n} -DK={k} -Dtarget_float={target_float} "
                      f"-mllvm -force-vector-width={w} -mllvm -force-vector-interleave={il}")
    whisper_log   = f"{test_dir}/sweep_{tag}_whisper.txt"
    gem5_log      = f"{test_dir}/sweep_{tag}_gem5.txt"
    m5out_dir     = f"{test_dir}/sweep_{tag}_m5out"

    print(f"\n{'='*60}")
    print(f"  {target_float}  core={gem5_core}  width={w}  interleave={il}  (M={m} N={n} K={k})")
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
    attain        = ai * PEAK_BW                  if ai             else None
    mem_eff       = (gflops / attain) * 100       if gflops and attain else None
    roof_eff      = (gflops / peak_gflops) * 100  if gflops         else None

    return {
        "dtype": target_float, "core": gem5_core, "m": m, "n": n, "k": k, "w": w, "il": il,
        "mcycle": mcycle, "minstret": minstret,
        "VecLoad": vec_load, "VecStore": vec_store,
        "bytes_Q": bytes_q, "AI": ai,
        "cyc_load": cyc_load,
        "MFLOP/s": gflops * 1000 if gflops else None,
        "mem_%": mem_eff,
        "roof_%": roof_eff,
    }

# Merged sweep across all enabled variants, driven by sweep_config.json
results        = []
group_start_at = []   # result index where each variant group begins
for variant in VARIANTS:
    p = variant_params(variant)
    group_start_at.append(len(results))
    for w in p["widths"]:
        for il in p["interleave"]:
            tag = f"{variant}_w{w}_il{il}"
            results.append(run_one(tag, p["march"], p["target_float"], p["sizeof"],
                                    p["m"], p["n"], p["k"], p["flops"], w, il,
                                    p["sim_config_gem5"], p["gem5_core"], p["peak_gflops"]))

# Summary table
def fmt(v, spec, fallback="  N/A"):
    return format(v, spec) if v is not None else fallback

HDR = (f"{'dtype':>10} {'core':>6} {'m':>3} {'n':>3} {'k':>3} {'w':>4} {'il':>3} | "
       f"{'mcycle':>8} {'minstret':>9} | "
       f"{'VecLoad':>8} {'VecSto':>7} | "
       f"{'Q(B)':>8} {'AI':>6} | "
       f"{'cyc/ld':>7} {'MFLOP/s':>9} {'mem%':>6} {'roof%':>6}")
SEP = "-" * len(HDR)
print(f"\n{SEP}\n{HDR}\n{SEP}")
for i, r in enumerate(results):
    if i in group_start_at and i != 0:
        print(SEP)   # separator between variant groups
    print(f"{r['dtype']:>10} {r['core']:>6} {r['m']:>3} {r['n']:>3} {r['k']:>3} {r['w']:>4} {r['il']:>3} | "
          f"{fmt(r['mcycle'],   '>8,'):>8} {fmt(r['minstret'],  '>9,'):>9} | "
          f"{fmt(r['VecLoad'],  '>8,'):>8} {fmt(r['VecStore'],  '>7,'):>7} | "
          f"{fmt(r['bytes_Q'],  '>8,'):>8} {fmt(r['AI'],        '>6.3f'):>6} | "
          f"{fmt(r['cyc_load'], '>7.1f'):>7} {fmt(r['MFLOP/s'], '>9.1f'):>9} {fmt(r['mem_%'], '>6.1f'):>6} {fmt(r['roof_%'], '>6.1f'):>6}")
print(SEP)
