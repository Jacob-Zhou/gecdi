# -*- coding: utf-8 -*-

import argparse
from gec.parser import Seq2SeqParser


def convert(file, fout, fin, fpath, max_len=64):
    count, sentence = 0, []
    tokenize_func = Seq2SeqParser.load(fpath).SRC.tokenize
    with open(file) as f, open(fout, 'w') as fout, open(fin) as fin:
        src_lines = [line.rstrip("\n") for line in fin]
        tgt_lines = []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                tgt_lines.append((sentence[1]+'\t').split('\t')[1])
                sentence = []
            else:
                sentence.append(line)
        count = 0
        for line in src_lines:
            if len(tokenize_func(line)) >= max_len:
                fout.write(line + "\n")
            else:
                fout.write(tgt_lines[count] + "\n")
                count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Output fils in line with m2scorer eval format.'
    )
    parser.add_argument('--path', '-p', help='path to the model file')
    parser.add_argument('--input', '-i', help='path to the input file')
    parser.add_argument('--hyp', help='path to the predicted file')
    parser.add_argument('--fout', '-o', help='path to output file')
    parser.add_argument('--max_len', '-m', help='max length')
    args = parser.parse_args()
    convert(args.hyp, args.fout, args.input, args.path, int(args.max_len))
