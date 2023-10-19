# -*- coding: utf-8 -*-
"""Simple regular expressions to fix tokenization issues for CoNLL.

Usage:
$ python3 retokenize.py [model_predictions_file] > [retokenized_predictions_file]
"""
import fileinput
import re

import argparse
from pkunlp import Segmentor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fix tokenization issues')
    parser.add_argument('--style',
                        '-s',
                        default="NLPCC",
                        choices=["NLPCC", "MuCGEC"],
                        help='seed for generating random numbers')
    parser.add_argument('input_file', nargs='+', help='input file')
    args, unknown = parser.parse_known_args()

    if args.style == "NLPCC":
        segmentor = Segmentor("scripts/predict_utils/pkunlp-features/segment.feat", "scripts/predict_utils/pkunlp-features/segment.dic")
        for line in open(args.input_file[0], 'r'):
            line = segmentor.seg_string("".join(line.split()))
            print(" ".join(line))
    else:
        raise NotImplementedError
