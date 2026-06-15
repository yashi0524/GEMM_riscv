#!/usr/bin/env python3
import sys
import subprocess
import re

class whisper_sim:

    def __init__(self):
        print("init")
        if len(sys.argv) < 2:
            print("Usage: script.py <test_pattern> [log_file] [gdb]")
            sys.exit(1)

        self.test_pattern = sys.argv[1]
        self.logfile = sys.argv[2] if len(sys.argv) > 2 else "./whisper_run_log.txt"
        self.mode = sys.argv[3] if len(sys.argv) > 3 else ""

        #gen re_pattern for hpmcounters
        #self.hpm_pattern = re.compile(r'hpmcounter\[(?P<hpm>[0-9]+)]:\s*(?P<name>[a-fA-F]+)\s*=\s*(?P<value>[0-9]+)')
        self.hpm_pattern = re.compile(r'hpmcounter\[(?P<idx>[0-9]+)]:\s*(?P<name>[a-zA-Z]+)\s*=\s*(?P<value>[0-9]+)')
        self.result = { "hpm":[] }

    def run_pattern(self):
        config_file = "/home/ajno5/work/2_pattern/dgemm/config/whisper_rv64gcv_config.json"
        #options = ["--counters", "--semihosting"]
        options = ["--semihosting"]
        options.append("--counters")

        print(f"options={options}")

        postfix = f" 2>&1 | tee {self.logfile}"
        print(f"postfix={postfix}")

        print(f"test_pattern={self.test_pattern}")
        print(f"configuration file={config_file}")

        if self.mode == "gdb":
            print("Running with debugger.")

            #cmd = [
            #    "gdb-multiarch", test_pattern,
            #    "--ex", f"target remote | whisper --configfile {config_file} {' '.join(options)} --gdb {test_pattern}"
            #]
            cmd = f" gdb-multiarch {self.test_pattern} --ex ' target remote | whisper --configfile {config_file} {' '.join(options)} --gdb {self.test_pattern}'"

        else:
            print(f"Running normally with: {self.test_pattern}")
            cmd = f"whisper --configfile {config_file} {' '.join(options)} {self.test_pattern} {postfix}"
            print(f"cmd = {cmd}")

        subprocess.run(cmd, shell=True)

    def log_parse( self):
        with open( self.logfile, "r") as log:
            for line_num, line in enumerate(log,1):
                line = line.rstrip()
                hpm_match = self.hpm_pattern.search(line)
                if hpm_match: 
                    self.result["hpm"].append({
                        "idx": hpm_match.group("idx"),
                        "name": hpm_match.group("name"),
                        "value": hpm_match.group("value")
                                               })
                    continue

                    #print(f"line = {line}")

    def print_result(self):
        for res in self.result["hpm"]:
            print(f"hmpcounter[{res['idx']}] : {res['name']} = {res['value']}")

if __name__ == "__main__":
    
    sim = whisper_sim()
    #sim.logfile = "test_log.txt"
    sim.run_pattern()
