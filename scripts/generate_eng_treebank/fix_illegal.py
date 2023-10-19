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
    legal = 0
    for i, line in enumerate(tqdm(open(args.input_file, "r"))):
        src, tgt = line.split('\t')
        tgt_tree = nltk.Tree.fromstring(tgt)
        if tgt_tree.label() != 'TOP':
            tgt_tree = nltk.Tree('TOP', [tgt_tree])
            output.write(f"{src.strip()}\n{tgt_tree.pformat(100000000000)}\n\n")
        elif len(tgt_tree) != 1 or isinstance(tgt_tree[0], nltk.Tree):
            tgt_tree[:] = nltk.Tree('FRAG', tgt_tree)
            output.write(f"{src.strip()}\n{tgt_tree.pformat(100000000000)}\n\n")
        else:
            output.write(f"{src.strip()}\n{tgt.strip()}\n\n")
    output.close()
