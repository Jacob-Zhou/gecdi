# -*- coding: utf-8 -*-

import argparse
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer

def tokenize(word, tokenizer):
    return tokenizer.basic_tokenizer.tokenize(word)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--input-file', help='input file')
    parser.add_argument('--output-file', help='output file')
    parser.add_argument('--tokenizer', help='input file')

    args, unknown = parser.parse_known_args()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False, local_files_only=False)

    output = open(args.output_file, "w")
    for i, line in enumerate(tqdm(open(args.input_file, "r"))):
        line = tokenize(line, tokenizer)
        line = " ".join(line).replace('(', '-LRB-').replace(')', '-RRB-').split()
        tree = nltk.Tree('TOP', [nltk.Tree('S', [nltk.Tree("Char", [token]) for token in line])])
        output.write(tree.pformat(1000000000)+"\n")
    output.close()
