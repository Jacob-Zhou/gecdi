# -*- coding: utf-8 -*-

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fix tokenization issues')
    parser.add_argument('input_file', nargs='+', help='input file')
    args, unknown = parser.parse_known_args()

    for line in open(args.input_file[0], 'r'):
        line = line.strip()
        line = line.replace("(", '-LRB-').replace(")", '-RRB-')
        print(f"(TOP (Sent (_ {line})))")
        print(f"(TOP (Sent (_ {line})))\n")

