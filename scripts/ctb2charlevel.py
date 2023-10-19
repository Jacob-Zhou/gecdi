# -*- coding: utf-8 -*-

import argparse
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer

def tokenize(word, tokenizer):
    return tokenizer.basic_tokenizer.tokenize(word)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--tree-file', help='input file')
    parser.add_argument('--output-file', help='output file')
    parser.add_argument('--tokenizer', help='input file')

    args, unknown = parser.parse_known_args()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False, local_files_only=False)

    output = open(args.output_file, "w")
    for i, tree in enumerate(tqdm(open(args.tree_file, "r"))):
        tree = nltk.Tree.fromstring(tree)
        subtrees = [subtree for subtree in tree.subtrees() if not isinstance(subtree[0], nltk.Tree)]
        for subtree in subtrees:
            tag = subtree.label()
            subtree[:] = [nltk.Tree('Char', [char]) for char in tokenize(subtree.leaves()[0], tokenizer)]
        output.write(tree.pformat(1000000) + "\n")

    output.close()
