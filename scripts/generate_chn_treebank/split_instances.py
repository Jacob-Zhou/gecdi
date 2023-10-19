# -*- coding: utf-8 -*-

import argparse
import random

def equal(sent_a, sent_b):
    sent_a = "".join(sent_a.split())
    sent_b = "".join(sent_b.split())
    return sent_a == sent_b

def similar(sent_a, sent_b):
    sent_a = "".join(sent_a.split())
    sent_b = "".join(sent_b.split())
    return abs(len(sent_a) - len(sent_b)) < 15

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--input-file', help='input file')
    parser.add_argument('--output-file', help='output file')
    parser.add_argument('--seed',
                        '-s',
                        default=1,
                        type=int,
                        help='seed for generating random numbers')

    args, unknown = parser.parse_known_args()

    random.seed(args.seed)

    splits = ["train", "dev"]

    correct_index = {s: 0 for s in splits}
    error_index = {s: 0 for s in splits}

    correct_lines = []
    source_lines = []
    error_lines = []

    correct_file = {
        s: open(f"{args.output_file}.p_{s}.correct", "w")
        for s in splits
    }
    sourcet_file = {
        s: open(f"{args.output_file}.p_{s}.source", "w")
        for s in splits
    }
    error_file = {
        s: open(f"{args.output_file}.p_{s}.error", "w")
        for s in splits
    }
    match_file = {
        s: open(f"{args.output_file}.p_{s}.match", "w")
        for s in splits
    }

    with open(args.input_file, "r") as lines:
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                random_num = random.random()
                s = "train"
                if 0 <= random_num < 0.9:
                    s = "train"
                else:
                    s = "dev"
                assert len(correct_lines) <= 1
                if len(correct_lines) == 0:
                    assert len(source_lines) == 1
                    correct_line = source_lines[0]
                else:
                    correct_line = correct_lines[0]
                correct_file[s].write(correct_line + '\n')
                sourcet_file[s].write(source_lines[0] + '\n')
                for error_line in set(error_lines):
                    # make sure error sentence is different from the correct one, and not too wrong.
                    if not equal(correct_line, error_line) and similar(correct_line, error_line):
                        error_file[s].write(error_line + '\n')
                        match_file[s].write(
                            f"{correct_index[s]}\t{error_index[s]}\n")
                        error_index[s] += 1
                correct_index[s] += 1
                source_lines = []
                correct_lines = []
                error_lines = []
            else:
                line = line + '\t'
                type, line = line.split("\t", maxsplit=1)
                line = line.strip()
                if type == 'T':
                    if len(line) > 0:
                        correct_lines.append(line)
                elif type == 'S':
                    source_lines.append(line)
                else:
                    error_lines.append(line)
    for s in splits:
        correct_file[s].close()
        sourcet_file[s].close()
        error_file[s].close()
        match_file[s].close()
