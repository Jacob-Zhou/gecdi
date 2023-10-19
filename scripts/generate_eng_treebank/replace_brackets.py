# -*- coding: utf-8 -*-

import argparse
import nltk
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--input-file', help='input file')
    parser.add_argument('--output-file', help='output file')

    args, unknown = parser.parse_known_args()

    output = open(args.output_file, "w")
    for i, line in enumerate(tqdm(open(args.input_file, "r"))):
        line = line.replace('(', '-LRB-').replace(')', '-RRB-')
        output.write(line)
    output.close()
