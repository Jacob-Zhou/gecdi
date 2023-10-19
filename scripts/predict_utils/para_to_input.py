# -*- coding: utf-8 -*-

import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--input-file', help='input file')

    args, unknown = parser.parse_known_args()

    with open(args.input_file, "r") as lines:
        for line in lines:
            line = line.strip()
            if len(line) > 0 and line[0] == "S":
                _, sent = line.split("\t", 1)
                sent = sent.rstrip("\n")
                print(sent)
