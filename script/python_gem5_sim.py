#!/usr/bin/env python3
import sys
import subprocess
import re
import json

class gem5_sim:

    def __init__(self):
        print("init")
        if len(sys.argv) < 2:
            print("Usage: script.py <test_pattern> [log_file] [gdb]")
            sys.exit(1)

        self.test_pattern = sys.argv[1]
        self.logfile = sys.argv[2] if len(sys.argv) > 2 else "./gem5_run_log.txt"
        self.mode = sys.argv[3] if len(sys.argv) > 3 else ""

    def build_pattern(self):
        config=json.load(open("/home/ajno5/work/2_pattern/dgemm/config/pattern_config.json"))

        compile_option= " ".join([
            f"M={config['matrix']['M']}"
        ])

        cmd=f"make clean && make {compile_option} && make dis"
        subprocess.run(cmd, shell=True)

    def run_pattern(self):
        config_file = "/home/ajno5/work/2_pattern/dgemm/config/gem5_riscv_demo_riscv_baremetal_semihost.py"

        postfix = f" 2>&1 | tee {self.logfile}"
        print(f"postfix={postfix}")

        print(f"test_pattern={self.test_pattern}")
        print(f"configuration file={config_file}")

        if self.mode == "gdb":
            # todo
            print("Running with debugger.")

            #cmd = [
            #    "gdb-multiarch", test_pattern,
            #    "--ex", f"target remote | whisper --configfile {config_file} {' '.join(options)} --gdb {test_pattern}"
            #]
            #cmd = f" gdb-multiarch {self.test_pattern} --ex ' target remote | whisper --configfile {config_file} {' '.join(options)} --gdb {self.test_pattern}'"

        else:
            print(f"Running normally with: {self.test_pattern}")
            cmd = f"gem5.opt {config_file} {self.test_pattern} {postfix}"
            print(f"cmd = {cmd}")

        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    root_path="/home/ajno5/work/2_pattern/dgemm"

    sim = gem5_sim()
    #sim.logfile = "test_log.txt"
    sim.build_pattern()
    #sim.run_pattern()
