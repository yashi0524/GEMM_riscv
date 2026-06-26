#!/usr/bin/env python3
import os
import sys
import subprocess
import re
import json

class riscv_sim:

    def __init__(self):
        print("init")

        self.root = " "
        self.test_dir = " "
        self.simulator = " "        
        self.test_pattern = " "
        self.test_config = " "
        self.sim_config = " "
        self.logfile = " "
        self.mode = " "


    def parse_argv(self):    
        print("parse sys argv")
        if len(sys.argv) < 3:
            print("Usage: script.py <simulator> <test_pattern> [test_config] [sim_config] [log_file] [gdb]")
            sys.exit(1)

        self.root = os.getcwd()
        #print(root_tmp)

        self.test_dir = f"{self.root}/test"

        self.simulator = sys.argv[1]

        self.test_pattern = sys.argv[2]

        self.test_config = sys.argv[3] if len(sys.argv) > 3 else f"{self.test_dir}/config/test_config.json"
        #print(self.test_config)


        match self.simulator:
            case "gem5":
               self.sim_config = sys.argv[4] if len(sys.argv) > 4 else f"{self.root}/sim_config/gem5_riscv_demo_riscv_baremetal_semihost.py"
               self.logfile = sys.argv[5] if len(sys.argv) > 5 else f"{self.test_dir}/gem5_run_log.txt"

            case "whisper":
                self.sim_config = sys.argv[4] if len(sys.argv) > 4 else f"{self.root}/sim_config/whisper_rv64gcv_config.json"
                self.logfile = sys.argv[5] if len(sys.argv) > 5 else f"{self.test_dir}/whisper_run_log.txt"               

        self.mode = sys.argv[6] if len(sys.argv) > 6 else ""

    def build_pattern(self):
        test_config=json.load(open(self.test_config))
        print(f"build pattern with {self.test_config}")

        compile_option= " ".join([
            f"M={test_config['matrix']['M']}"
        ])

        #cmd=f"make clean && make {compile_option} && make dis"
        cmd=f"make clean && make {compile_option}"
        subprocess.run(cmd, shell=True)

    def dump_flags(self):
        postfix = f" 2>&1 | tee {self.logfile}"
        cmd = f"make dump_flags {postfix}"
        subprocess.run(cmd, shell=True)


    def run_pattern(self):
        postfix = f" 2>&1 | tee {self.logfile}"
        print(f"postfix={postfix}")

        print(f"test_pattern={self.test_pattern}")
        print(f"configuration file={self.sim_config}")

        match self.simulator:
            case "gem5":
                sim_bin = "gem5.opt"
                options = ""
                options += (f" -d {self.test_dir}/m5out {self.sim_config} ")

            case "whisper":
                sim_bin = "whisper"
                options = ""
                options += f" --configfile {self.sim_config}"
                options += f" --semihosting"
                options += f" --counters"

        if self.mode == "gdb":
            # todo
            print("Running with debugger. not done yet")
            return 
            #cmd = [
            #    "gdb-multiarch", test_pattern,
            #    "--ex", f"target remote | whisper --configfile {sim_config} {' '.join(options)} --gdb {test_pattern}"
            #]
            #cmd = f" gdb-multiarch {self.test_pattern} --ex ' target remote | whisper --configfile {sim_config} {' '.join(options)} --gdb {self.test_pattern}'"

        else:

            print(f"Running normally with: {self.test_pattern}")
            cmd = f"{sim_bin} {options} {self.test_pattern} {postfix}"
            print(f"cmd = {cmd}")

        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    root_path="/home/ajno5/work/2_pattern/dgemm"

    sim = riscv_sim()
    sim.parse_argv()
    #sim.logfile = "test_log.txt"
    sim.build_pattern()
    #sim.dump_flags()
    sim.run_pattern()
