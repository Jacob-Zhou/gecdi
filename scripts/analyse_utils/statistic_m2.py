# -*- coding: utf-8 -*-

import argparse
from collections import Counter, defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--m2scorer-file', help='m2scorer file')

    args, unknown = parser.parse_known_args()

    cur_annotator = 0

    golden_actions = {}
    correct_actions = {}
    missed_actions = {}
    unneeded_actions = {}

    all_golden_types = Counter()
    all_correct_types = Counter()
    all_missed_types = Counter()
    all_unneeded_types = Counter()

    with open(args.m2scorer_file, "r") as lines:
        for line in lines:
            if line.startswith(">> Chosen Annotator for line"):
                annotator = int(line.strip().split()[-1])
                all_correct_types.update([a[-1] for a in correct_actions[annotator]])
                all_golden_types.update([a[-1] for a in golden_actions[annotator]])
                all_missed_types.update([a[-1] for a in missed_actions[annotator]])
                all_unneeded_types.update([f"'{a[2]}' -> '{a[3]}'" for a in unneeded_actions[annotator]])
                golden_actions = {}
                correct_actions = {}
                missed_actions = {}
                unneeded_actions = {}
            elif line.startswith(">> Annotator:"):
                cur_annotator = int(line.strip().split()[-1])
            elif line.startswith("CORRECT EDITS : "):
                correct_actions[cur_annotator] = eval(line.split(": ", 1)[-1])
            elif line.startswith("GOLD EDITS    : "):
                golden_actions[cur_annotator] = eval(line.split(": ", 1)[-1])
            elif line.startswith("MISSED EDITS  : "):
                missed_actions[cur_annotator] = eval(line.split(": ", 1)[-1])
            elif line.startswith("UNEEDED EDITS : "):
                unneeded_actions[cur_annotator] = eval(line.split(": ", 1)[-1])

    print("Correct Error Type:")
    for k, v in all_correct_types.most_common():
        print(f"{k:>8}: {v:>4d} {v/all_golden_types[k]:7.2%}")

    print()
    print("Missed Error Type:")
    for k, v in all_missed_types.most_common():
        print(f"{k:>8}: {v:>4d} {v/all_golden_types[k]:7.2%}")

    print()
    print(f"Unneeded Error Type: {len(list(all_unneeded_types.elements()))}")
    for k, v in all_unneeded_types.most_common():
        print(f"{v:4>d} {k}")
