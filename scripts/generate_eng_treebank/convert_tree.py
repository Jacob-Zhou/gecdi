# -*- coding: utf-8 -*-

import argparse
from collections import Counter
import nltk
from tqdm import tqdm
import errant
from functools import lru_cache
from multiprocessing.pool import Pool


def append_error(node, error):
    if node.label() == "CORRECT":
        node.set_label(error)
    elif "MISS-L" in node.label() and "MISS-L" in error:
        pass
    else:
        node.set_label(node.label() + "::" + error)


def convert_tree(tree, actions):
    original_tree = tree
    n_words = len(tree.leaves())
    tree = tree.copy(True)
    tree.collapse_unary(joinChar='~')
    unmodified = False

    for action in actions:
        start_pos, end_pos = action.o_start, action.o_end
        if action.type == 'noop':
            # target sentences has zero error, then ignore it.
            unmodified = True
            break
        error_type, error_tier = action.type.split(':', 1)

        if error_type == 'M':
            # need to delete the token between (start_pos, end_pos)
            ## we first mark the tokens need to delete
            assert end_pos == start_pos
            if end_pos != n_words:
                append_error(tree[0][start_pos], f"MISS-L:{error_tier}")
        elif error_type == 'U':
            # need to insert the token to start_pos, assert "start_pos == end_pos"
            # the parent of the insert token is the smallest span that i < start_pos < j
            assert end_pos - start_pos == 1
            append_error(tree[0][start_pos], f"RED:{error_tier}")
        elif error_type == 'R':
            # replace tokens between (start_pos, end_pos) with another tokens
            assert end_pos - start_pos >= 1
            # if "word order" happened and the "target_word_num" is bigger than one, then we mask word order error to their lowest parent
            # the word reorder error
            for i in range(end_pos - start_pos):
                append_error(tree[0][start_pos + i], f"SUB:{error_tier}")
        else:
            # unrecognized errors
            print(error_type)

    if unmodified:
        return None
    else:
        return tree


@lru_cache(maxsize=512)
def annotator_parse_cached(seq):
    return annotator.parse(seq)


@lru_cache(maxsize=512)
def tokenize(seq):
    return seq.strip().split()


def create_tree(seq):
    return nltk.Tree("TOP", [
        nltk.Tree("S", [
            nltk.Tree("CORRECT",
                      [word.replace('(', '-LRB-').replace(')', '-RRB-')])
            for word in seq
        ])
    ])


def process_lines(inputs):
    source, target = inputs
    source = tokenize(source)
    target = tokenize(target)
    tree = create_tree(source)
    if len(target) < 2:
        return tree
    if source == target:
        return tree
    source = annotator_parse_cached(" ".join(source))
    # the same target may repeat multi times
    target = annotator_parse_cached(" ".join(target))
    edits = annotator.annotate(source, target, merging="all-split")
    return convert_tree(tree, edits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--source-file', help='source file')
    parser.add_argument('--target-file', help='target tree file')
    parser.add_argument('--output-file', help='output file')
    parser.add_argument('--tokenizer', help='tokenizer')
    parser.add_argument('--processes',
                        type=int,
                        default=8,
                        help="numbers of processes")
    parser.add_argument('--chunksize',
                        type=int,
                        default=64,
                        help="chunksize of imap")

    args, unknown = parser.parse_known_args()
    action_counter = Counter()
    annotator = errant.load("en")

    output = open(args.output_file, "w")
    source_lines = open(args.source_file, "r")
    target_lines = open(args.target_file, "r")
    with Pool(args.processes) as pool:
        for converted_tree in tqdm(
                pool.imap(process_lines, zip(source_lines, target_lines),
                          args.chunksize)):
            if converted_tree is not None:
                output.write(converted_tree.pformat(1000000000))
                output.write("\n")

    output.close()
