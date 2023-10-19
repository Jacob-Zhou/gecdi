# -*- coding: utf-8 -*-
"""Simple regular expressions to fix tokenization issues for CoNLL.

Usage:
$ python3 retokenize.py [model_predictions_file] > [retokenized_predictions_file]
"""
from collections import defaultdict
from functools import partial
import re
import argparse
import errant
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fix tokenization issues')
    parser.add_argument('--gold',
                        help='seed for generating random numbers')
    parser.add_argument('--src',
                        help='seed for generating random numbers')
    parser.add_argument('pred_file', nargs='+', help='input file')
    args, unknown = parser.parse_known_args()

    open_fn = partial(open, mode='r')
    rePENALTY = re.compile(r"penalty-(\d+\.\d+)\.")

    annotator = errant.load("en")

    same_as_original = defaultdict(int)
    same_as_previous = defaultdict(int)
    same_as_source = defaultdict(int)
    n_edit = defaultdict(int)

    for gold_line, src_line, *pred_file_lines in tqdm(zip(*map(open_fn, (args.gold, args.src, *args.pred_file)))):
        # original = pred_file_lines[0]
        # previous = original
        # original_parsed = annotator.parse(src_line.strip())
        # for name, line in zip(args.pred_file[1:], pred_file_lines[1:]):
        #     penalty = rePENALTY.search(name)
        #     pred_parsed = annotator.parse(line.strip())
        #     edits = annotator.annotate(original_parsed, pred_parsed)
        #     if penalty:
        #         if line != original:
        #             same_as_original[float(penalty.group(1))] += 1
        #         if line != previous:
        #             same_as_previous[float(penalty.group(1))] += 1
        #         if line != src_line:
        #             same_as_source[float(penalty.group(1))] += 1
        #         n_edit[float(penalty.group(1))] += len(edits)

        #     previous = line

        if len(set(pred_file_lines)) == 1:
            continue
        print(f"{'gold':>8s}: {gold_line.strip()}")
        print(f"{'source':>8s}: {src_line.strip()}")
        for name, line in zip(args.pred_file, pred_file_lines):
            penalty = rePENALTY.search(name)
            if penalty:
                print(f"{float(penalty.group(1)):>8.2f}: {line.strip()}")
            else:
                print(f"{'w/o syn':>8s}: {line.strip()}")
        print()
    # for k, v in sorted(same_as_original.items(), key=lambda x: x[0]):
    #     print(f"{k:>8.2f}: {v}")
    # print()
    # for k, v in sorted(same_as_source.items(), key=lambda x: x[0]):
    #     print(f"{k:>8.2f}: {v}")
    # print()
    # for k, v in sorted(same_as_previous.items(), key=lambda x: x[0]):
    #     print(f"{k:>8.2f}: {v}")
    # print()
    # for k, v in sorted(n_edit.items(), key=lambda x: x[0]):
    #     print(f"{k:>8.2f}: {v}")
