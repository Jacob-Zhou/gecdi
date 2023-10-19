# -*- coding: utf-8 -*-

import argparse
from tqdm import tqdm

# def tokenize(sentence):
#     pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--parallel-file', help='input file')
    parser.add_argument('--collapse', action="store_true", help='input file')
    parser.add_argument('--output-file', help='output file')

    args, unknown = parser.parse_known_args()

    output = open(args.output_file, "w")
    output_i = open(args.output_file+".input", "w")
    for i, line in enumerate(tqdm(open(args.parallel_file, "r"))):
        line = line.strip()
        sents = line.split("\t")
        if args.collapse:
            output.write(f"S\t{sents[0]}\n")
            output_i.write(sents[0]+"\n")
        if len(sents) > 1:
            for sent in sents[1:]:
                if sent.strip() == '':
                    sent = sents[0]
                if args.collapse:
                    output.write(f"T\t{sent}\n")
                else:
                    output_i.write(sents[0]+"\n")
                    output.write(f"S\t{sents[0]}\n")
                    output.write(f"T\t{sent}\n\n")
            else:
                if args.collapse:
                    output.write(f"\n")
        else:
            if not args.collapse:
                output_i.write(sents[0]+"\n")
                output.write(f"S\t{sents[0]}\n")
            output.write(f"T\t{sents[0]}\n\n")

    output.close()
