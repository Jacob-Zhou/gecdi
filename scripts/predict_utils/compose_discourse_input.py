# -*- coding: utf-8 -*-

import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--input-file', help='input file')
    parser.add_argument('--index-file', help='input file')
    parser.add_argument('--output-file', help='output file')

    args, unknown = parser.parse_known_args()

    output_file = open(f"{args.output_file}", "w")

    discourse_index = 0
    target_discourse_buff = ""
    with open(args.input_file, "r") as lines, open(args.index_file, "r") as index:
        for line in lines:
            line = line.strip()
            cur_index, _, end = next(index).split('\t')
            end = end.strip()
            if int(cur_index) == discourse_index:
                target_discourse_buff += line
            else:
                discourse_index = int(cur_index)
                output_file.write(f"{target_discourse_buff}\n")
                target_discourse_buff = line
            if end != 'P':
                target_discourse_buff = target_discourse_buff[:-1] + end
        else:
            if (target_discourse_buff) != '':
                output_file.write(f"{target_discourse_buff}\n")

    output_file.close()

