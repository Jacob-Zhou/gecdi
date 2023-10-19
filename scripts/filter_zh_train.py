# -*- coding: utf-8 -*-

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--input-file', help='input file')
    parser.add_argument('--output-file', help='output file')

    args, unknown = parser.parse_known_args()

    output = open(args.output_file, "w")
    broken_output = open(f"{args.output_file}.broken", "w")
    source_line = ""
    correct_line = ""

    with open(args.input_file, "r") as lines:
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                assert source_line != "" and correct_line != ""
                if 0 < len(correct_line) < 1.5 * len(source_line):
                    output.write(f"S\t{source_line}\n")
                    output.write(f"T\t{correct_line}\n\n")
                else:
                    broken_output.write(f"S\t{source_line}\n")
                    broken_output.write(f"T\t{correct_line}\n\n")
                source_line = ""
                correct_line = ""
            else:
                line = line + '\t'
                type, line = line.split("\t", maxsplit=1)
                line = line.strip()
                if type == 'T':
                    if len(line) > 0:
                        correct_line = line
                elif type == 'S':
                    source_line = line
                else:
                    raise ValueError(f"Unknown type: {type}")
        else:
            if source_line != "" and correct_line != "":
                if len(correct_line) < 1.5 * len(source_line):
                    output.write(f"S\t{source_line}\n")
                    output.write(f"T\t{correct_line}\n\n")
                else:
                    broken_output.write(f"S\t{source_line}\n")
                    broken_output.write(f"T\t{correct_line}\n\n")
    output.close()