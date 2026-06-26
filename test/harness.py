import sys
import json
import subprocess

#test_cfg = json.load(open("./config/test_config.json"))
#print(test_cfg)

pattern_root = "/home/ajno5/work/2_pattern/dgemm"

pattern_script = f"{pattern_root}/script"

test_dir = f"{pattern_root}/test"
test_config = f"{pattern_root}/test/config/test_config.json"
sim_config = f"{pattern_root}/sim_config/gem5_riscv_demo_riscv_baremetal_semihost.py"

sim_config_w = f"{pattern_root}/sim_config/whisper_rv64gcv_config.json"

sys.path.append(f"{pattern_script}/")

from python_riscv_sim import riscv_sim
from python_log_parser import sim_log_parser

whisper_test = True
#whisper_test = False

gem5_test = True
#gem5_test = False

log_parse = True
#log_parse = False

if whisper_test :
    sim_w = riscv_sim()

    sim_w.root = pattern_root
    sim_w.test_dir = test_dir

    sim_w.simulator = "whisper" 
    sim_w.test_pattern = f"{sim_w.test_dir}/dgemm_riscv"
    sim_w.test_config = test_config
    sim_w.sim_config = sim_config_w
    sim_w.logfile = f"{sim_w.test_dir}/whisper_run_log.txt"

    sim_w.build_pattern()
    sim_w.run_pattern()

if gem5_test :
    sim_g = riscv_sim()

    sim_g.root = pattern_root
    sim_g.test_dir = test_dir

    sim_g.simulator = "gem5" 
    sim_g.test_pattern = f"{sim_g.test_dir}/dgemm_riscv"
    sim_g.test_config = test_config
    sim_g.sim_config = sim_config
    sim_g.logfile = f"{sim_g.test_dir}/gem5_run_log.txt"

    sim_g.build_pattern()
    sim_g.dump_flags()

    sim_g.run_pattern()

if log_parse:
    output_file = f"{test_dir}/output.txt"

    parser = sim_log_parser()

    parser.log_parse(f"{test_dir}/whisper_run_log.txt", "whisper")
    parser.dump_result(output_file, "whisper")

    parser.log_parse(f"{test_dir}/gem5_run_log.txt", "gem5")
    parser.dump_result(output_file, "gem5", mode ="a")

    parser.print_output(output_file)

