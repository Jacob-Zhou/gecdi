# -*- coding: utf-8 -*-
"""Simple regular expressions to fix tokenization issues for CoNLL.

Usage:
$ python3 retokenize.py [model_predictions_file] > [retokenized_predictions_file]
"""
import fileinput
import re

import argparse
import random

import spacy
en = spacy.load("en_core_web_sm")

retokenization_rules = [
    # Remove extra space around single quotes, hyphens, and slashes.
    (" ' (.*?) ' ", " '\\1' "),
    (" ?- ?", "-"),
    (" / ", "/"),
    # Ensure there are spaces around parentheses and brackets.
    (r"([\]\[\(\){}<>])", " \\1 "),
    (r"\s+", " "),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fix tokenization issues')
    parser.add_argument('--style',
                        '-s',
                        default="CoNLL",
                        choices=["CoNLL", "BEA"],
                        help='seed for generating random numbers')
    parser.add_argument('input_file', nargs='+', help='input file')
    args, unknown = parser.parse_known_args()

    if args.style == "CoNLL":
        for line in open(args.input_file[0], 'r'):
            line = re.sub(" '\s?((?:m )|(?:ve )|(?:ll )|(?:s )|(?:d ))",
                          "'\\1", line)
            line = " ".join([t.text for t in en.tokenizer(line)])
            for rule in retokenization_rules:
                line = re.sub(rule[0], rule[1], line)
            print(line.strip())
    else:
        assert args.style == "BEA"
        for line in open(args.input_file[0], 'r'):
            line = re.sub(" '\s?((?:m )|(?:ve )|(?:ll )|(?:s )|(?:d ))",
                          "'\\1", line)
            line = " ".join([t.text for t in en.tokenizer(line)])
            # in spaCy v1.9.0 and the en_core_web_sm-1.2.0 model
            # 80% -> 80%, but in newest ver. 2.3.9', 80% -> 80 %
            # haven't -> haven't, but in newest ver. 2.3.9', haven't -> have n't
            line = re.sub("(?<=\d)\s+%", "%", line)
            line = re.sub("((?:have)|(?:has)) n't", "\\1n't", line)
            line = re.sub("^-", "- ", line)
            line = re.sub(r"\s+", " ", line)
            print(line.strip())
