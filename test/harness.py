"""
Usage:
  python3 harness.py          -- build + run the enabled benchmarks
  python3 harness.py clean    -- remove all m5out dirs under test/
"""

import os
import sys
import json
import shutil

pattern_root    = "/home/ajno5/work/2_pattern/gemm"
pattern_script  = f"{pattern_root}/script"
test_dir        = f"{pattern_root}/test"
test_config_path = f"{test_dir}/config/test_config.json"
sim_config_gem5    = f"{pattern_root}/sim_config/gem5_riscv_demo_riscv_baremetal_semihost.py"
sim_config_whisper = f"{pattern_root}/sim_config/whisper_rv64gcv_config.json"

sys.path.append(f"{pattern_script}/")
from python_riscv_sim import riscv_sim
from python_log_parser import sim_log_parser

# --- Simulator toggles ---
whisper_test = True
gem5_test    = True
log_parse    = True

# --- Benchmark registry ---
# Each entry drives: build → whisper → gem5 → parse.
# make_vars feeds make on the command line (overrides Makefile defaults).
# To add a new benchmark: drop src/<name>.c, add a dict entry here.
test_cfg = json.load(open(test_config_path))

BENCHES = [
    {
        "enabled":   True,
        "name":      "gemm",
        "make_vars": {"M": test_cfg["matrix"]["M"]},
        "whisper_log": f"{test_dir}/whisper_run_log.txt",
        "gem5_log":    f"{test_dir}/gem5_run_log.txt",
        "output":      f"{test_dir}/output.txt",
        "m5out":       f"{test_dir}/m5out",
    },
    {
        "enabled":   False,
        "name":      "fmacc",
        "make_vars": {"ITERS": test_cfg["fmacc"]["ITERS"]},
        "whisper_log": f"{test_dir}/fmacc_whisper_run_log.txt",
        "gem5_log":    f"{test_dir}/fmacc_gem5_run_log.txt",
        "output":      f"{test_dir}/fmacc_output.txt",
        "m5out":       f"{test_dir}/fmacc_m5out",
    },
]

def clean():
    removed = []
    for entry in os.listdir(test_dir):
        if entry == "m5out" or entry.endswith("_m5out"):
            d = os.path.join(test_dir, entry)
            if os.path.isdir(d):
                shutil.rmtree(d)
                removed.append(d)
    if removed:
        print(f"Removed {len(removed)} dir(s):")
        for r in removed:
            print(f"  {r}")
    else:
        print("Nothing to clean.")

if len(sys.argv) > 1 and sys.argv[1] == "clean":
    clean()
    sys.exit(0)

# --- Run ---
for bench in [b for b in BENCHES if b["enabled"]]:
    name   = bench["name"]
    binary = f"{test_dir}/{name}_riscv"

    # Build once — shared by both simulators
    builder = riscv_sim()
    builder.test_dir = test_dir
    builder.build_bench(name, bench["make_vars"])

    if whisper_test:
        sim_w = riscv_sim()
        sim_w.root         = pattern_root
        sim_w.test_dir     = test_dir
        sim_w.simulator    = "whisper"
        sim_w.test_pattern = binary
        sim_w.sim_config   = sim_config_whisper
        sim_w.logfile      = bench["whisper_log"]
        sim_w.run_pattern()

    if gem5_test:
        shutil.rmtree(bench["m5out"], ignore_errors=True)
        sim_g = riscv_sim()
        sim_g.root         = pattern_root
        sim_g.test_dir     = test_dir
        sim_g.simulator    = "gem5"
        sim_g.test_pattern = binary
        sim_g.sim_config   = sim_config_gem5
        sim_g.logfile      = bench["gem5_log"]
        sim_g.m5out_dir    = bench["m5out"]
        sim_g.run_pattern()

    if log_parse:
        parser = sim_log_parser()
        parser.log_parse(bench["whisper_log"], "whisper")
        parser.dump_result(bench["output"], "whisper")
        parser.log_parse(bench["gem5_log"], "gem5")
        parser.dump_result(bench["output"], "gem5", mode="a")
        parser.print_output(bench["output"])
