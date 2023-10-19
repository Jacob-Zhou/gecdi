# -*- coding: utf-8 -*-

import argparse
import nltk
from tqdm import tqdm
from functools import lru_cache


@lru_cache(maxsize=1024)
def tree_fromstring(seq):
    return nltk.Tree.fromstring(seq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--input-file', help='input file')
    parser.add_argument('--output-file', help='output file')

    args, unknown = parser.parse_known_args()

    output = open(args.output_file, "w")
    legal = 0
    for i, line in enumerate(tqdm(open(args.input_file, "r"))):
        src, tgt = line.split('\t')
        src_tree = tree_fromstring(src)
        tgt_tree = tree_fromstring(tgt)
        if len(src_tree.leaves()) < 1 or len(tgt_tree.leaves()) < 1:
            continue
        if (tgt_tree.label() in {'TOP', ''} and len(tgt_tree) == 1
                and isinstance(tgt_tree[0], nltk.Tree)):
            output.write(f"{src.strip()}\n{tgt.strip()}\n\n")
            legal += 1
    print(legal)
    output.close()
