#!/usr/bin/env python3
import sys
import re


class sim_log_parser:
    def __init__(self):
        print("logger init")
        # re.gen
        #gen re_pattern for hpmcounters
        self.hpm_pattern = re.compile(r'hpmcounter\[(?P<idx>[0-9]+)]:\s*(?P<name>[a-zA-Z]+)\s*=\s*(?P<value>[0-9]+)')
        self.counter_pattern = re.compile(r'counter:\s*(?P<name>[a-zA-Z]+)\s*=\s*(?P<value>[0-9]+)')
        self.result = { "counter":[], "hpm":[] }

    def log_parse(self,logfile,sim_select):

        if sim_select == "whisper":
            target_pattern = self.hpm_pattern
            target_list = "hpm"
        else:
            target_pattern = self.counter_pattern
            target_list = "counter"

        with open(logfile, "r") as log:
            for line_num, line in enumerate(log,1):
                line = line.rstrip()
                matched_pattern = target_pattern.search(line)
                if matched_pattern: 
                    self.result[target_list].append({
                        "name": matched_pattern.group("name"),
                        "value": matched_pattern.group("value")
                                               })
                    continue

                    #print(f"line = {line}")

    def print_result(self):
        for res in self.result["counter"]:
            print(f"counter: {res['name']} = {res['value']}")

    def dump_result(self, output_file, sim_select, mode = "w"):
        if sim_select == "whisper":
            target_list = "hpm"
        else:
            target_list = "counter"

        with open(output_file, mode) as output:
            output.write(f"=== {sim_select} result ===\n")
            for item in self.result[target_list]:
                output.write(f"{item['name']}: {item['value']}\n")

    def print_output(self, output_file):
        with open(output_file, "r") as output:
            for line in output:
                print(line)


if __name__ == "__main__":
    output_file = "./output.txt"

    parser = sim_log_parser()
    parser.log_parse("gem5_run_log.txt", "gem5")
    parser.dump_result(output_file, "gem5")
    
    parser.log_parse("whisper_run_log.txt", "whisper")
    parser.dump_result(output_file, "whisper", mode ="a")

    parser.print_output(output_file)

   