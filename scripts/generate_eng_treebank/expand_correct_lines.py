# -*- coding: utf-8 -*-

import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--input-file', help='input file')
    parser.add_argument('--match-file', help='match file')
    parser.add_argument('--output-file', help='output file')

    args, unknown = parser.parse_known_args()

    sentence_index = -1
    line = ''

    output_file = open(f"{args.output_file}", "w")

    with open(args.input_file, "r") as input_lines, open(args.match_file, "r") as match_lines:
        for match in tqdm(match_lines):
            cur_index, _ = match.split()
            while int(cur_index) != sentence_index:
                line = next(input_lines)
                sentence_index += 1
            output_file.write(line)
    
    output_file.close()
