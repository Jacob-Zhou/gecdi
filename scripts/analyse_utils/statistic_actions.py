# -*- coding: utf-8 -*-

import argparse
from collections import Counter
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--m2-file', help='match file')

    args, unknown = parser.parse_known_args()
    action_counter = Counter()

    for line in tqdm(open(args.m2_file, "r")):
        if len(line.strip()) > 0 and line.startswith("A"):
            action = line.split('|||')[1]
            action_counter.update([action])

    # print(action_counter)
    for k in sorted(action_counter):
        type, *tier = k.split(":")
        print(f"{type},{' '.join(tier)},{action_counter[k]}")