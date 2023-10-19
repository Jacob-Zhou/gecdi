# -*- coding: utf-8 -*-

import argparse
import re
from transformers import AutoTokenizer
import os
import sys

sys.path.append(os.getcwd())
from gec.parser import Seq2SeqParser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--input-file', help='input file')
    parser.add_argument('--output-file', help='output file')
    parser.add_argument('--path', help='input file')
    parser.add_argument('--max_len', '-m', type=int, help='max length')
    parser.add_argument('--style',
                        '-s',
                        default="NLPCC",
                        choices=["NLPCC", "MuCGEC"],
                        help='seed for generating random numbers')

    args, unknown = parser.parse_known_args()
    tokenize_func = Seq2SeqParser.load(args.path).SRC.tokenize

    if args.style == "MuCGEC":
        rePERIOD = re.compile(r'(?<=，|,|。|!|！|\?|？)(?!”)')
    else:
        rePERIOD = re.compile(r'(?<=，|,)')

    index_file = open(f"{args.output_file}.index", "w")
    para_file = open(f"{args.output_file}", "w")
    input_file = open(f"{args.output_file}.input", "w")

    max_len = args.max_len
    discourse_index = 0
    with open(args.input_file, "r") as lines:
        for line in lines:
            line = line.strip()
            line = re.split(rePERIOD, line)
            if line[-1] == '':
                line = line[:-1]
            idx = 0
            buff = ''
            for s in line:
                # if longer than max lenght than split it
                if len(tokenize_func(buff + s)) >= max_len and buff != '':
                    index_file.write(
                        f"{discourse_index}\t{idx}\t{buff[-1] if buff.endswith((',', '，')) else 'P'}\n"
                    )
                    input_file.write(f"{buff}\n")
                    para_file.write(f"S\t{buff}\nT\n\n")
                    idx += 1
                    buff = s
                else:
                    buff += s
                # if not end with comma split it!
                if not buff.endswith((',', '，')) and args.style == "MuCGEC":
                    index_file.write(f"{discourse_index}\t{idx}\tP\n")
                    input_file.write(f"{buff}\n")
                    para_file.write(f"S\t{buff}\nT\n\n")
                    idx += 1
                    buff = ''
            if buff != '':
                index_file.write(f"{discourse_index}\t{idx}\tP\n")
                input_file.write(f"{buff}\n")
                para_file.write(f"S\t{buff}\nT\n\n")
            discourse_index += 1

    index_file.close()
    input_file.close()
    para_file.close()
