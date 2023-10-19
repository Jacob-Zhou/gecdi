# -*- coding: utf-8 -*-

import argparse
from functools import lru_cache
from typing import Iterable, Tuple
import nltk
from tqdm import tqdm


def factorize(
    tree: nltk.Tree,
    allow_label=["RED:", "MISS-L", "SUB:"],
) -> Iterable[Tuple]:
    r"""
    Factorizes the tree into a sequence traversed in post-order.

    Args:
        tree (nltk.tree.Tree):
            The tree to be factorized.
        delete_labels (Optional[Set[str]]):
            A set of labels to be ignored. This is used for evaluation.
            If it is a pre-terminal label, delete the word along with the brackets.
            If it is a non-terminal label, just delete the brackets (don't delete children).
            In `EVALB`_, the default set is:
            {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}
            Default: ``None``.
        equal_labels (Optional[Dict[str, str]]):
            The key-val pairs in the dict are considered equivalent (non-directional). This is used for evaluation.
            The default dict defined in `EVALB`_ is: {'ADVP': 'PRT'}
            Default: ``None``.

    Returns:
        The sequence of the factorized tree.

    Examples:
        >>> from supar.utils import Tree
        >>> tree = nltk.Tree.fromstring('''
                                        (TOP
                                            (S
                                            (NP (_ She))
                                            (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
                                            (_ .)))
                                        ''')
        >>> Tree.factorize(tree)
        [(0, 1, 'NP'), (3, 4, 'NP'), (2, 4, 'VP'), (2, 4, 'S'), (1, 4, 'VP'), (0, 5, 'S'), (0, 5, 'TOP')]
        >>> Tree.factorize(tree, delete_labels={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''})
        [(0, 1, 'NP'), (3, 4, 'NP'), (2, 4, 'VP'), (2, 4, 'S'), (1, 4, 'VP'), (0, 5, 'S')]

    .. _EVALB:
        https://nlp.cs.nyu.edu/evalb/
    """

    def track(tree, i):
        label = tree.label()
        if not any([etype in label for etype in allow_label]):
            label = None
        if len(tree) == 1 and not isinstance(tree[0], nltk.Tree):
            return i + 1, []
        j, spans = i, []
        for child in tree:
            j, s = track(child, j)
            spans += s
        if label is not None and j > i:
            spans = spans + [(i, j, label)]
        return j, spans
    return track(tree, 0)[1]

@lru_cache(maxsize=10000)
def get_error_spans(tree: str):
    tree = nltk.Tree.fromstring(tree)
    return factorize(tree, allow_label=["RED", "MISS-L", "SUB"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--input-file', help='input file')
    parser.add_argument('--output-file', help='output file')

    args, unknown = parser.parse_known_args()

    output = open(args.output_file, "w")
    lines = []

    with open(args.input_file, "r") as input_lines:
        for line in tqdm(input_lines):
            line = line.strip()
            if len(line) == 0:
                assert len(lines) == 2
                error_spans = get_error_spans(lines[0])
                if len(error_spans) > 0:
                    output.write(f"S\t{lines[0]}\n")
                    output.write(f"T\t{lines[1]}\n\n")
                lines = []
            else:
                lines.append(line.strip())
        else:
            if len(lines) == 2:
                error_spans = get_error_spans(lines[0])
                if len(error_spans) > 0:
                    output.write(f"S\t{lines[0]}\n")
                    output.write(f"T\t{lines[1]}\n\n")
    output.close()