# -*- coding: utf-8 -*-

import argparse
import nltk
from nltk.tree import TreePrettyPrinter

NUL = '<nul>'

def isroot(node):
    return node == tree[0]

def isterminal(node):
    return len(node) == 1 and not isinstance(node[0], nltk.Tree)

def last_leaf(node):
    pos = ()
    while True:
        pos += (len(node) - 1,)
        node = node[-1]
        if isterminal(node):
            return node, pos

def parent(position):
    return tree[position[:-1]]

def grand(position):
    return tree[position[:-2]]

def detach(tree):
    last, last_pos = last_leaf(tree)
    siblings = parent(last_pos)[:-1]

    if len(siblings) > 0:
        last_subtree = last
        last_subtree_siblings = siblings
        parent_label = NUL
    else:
        last_subtree, last_pos = parent(last_pos), last_pos[:-1]
        last_subtree_siblings = [] if isroot(last_subtree) else parent(last_pos)[:-1]
        parent_label = last_subtree.label()

    target_pos, new_label, last_tree = 0, NUL, tree
    if isroot(last_subtree):
        last_tree = None
    elif len(last_subtree_siblings) == 1 and not isterminal(last_subtree_siblings[0]):
        new_label = parent(last_pos).label()
        target = last_subtree_siblings[0]
        last_grand = grand(last_pos)
        if last_grand is None:
            last_tree = target
        else:
            last_grand[-1] = target
        target_pos = len(last_pos) - 2
    else:
        target = parent(last_pos)
        target.pop()
        target_pos = len(last_pos) - 2
    action = target_pos, parent_label, new_label
    return action, last_tree


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--tree-file', help='tree file')

    args, unknown = parser.parse_known_args()

    for line in open(args.tree_file):
        tree = nltk.Tree.fromstring(line)
        print(TreePrettyPrinter(tree, None, ()).text(unicodelines=True))
        print()
        print("-"*50+"\n")

        if len(tree) > 1:
            tree[:] = [nltk.Tree('*', tree)]
        tree.collapse_unary(joinChar='::')
        if len(tree) == 1 and not isinstance(tree[0][0], nltk.Tree):
            tree[0] = nltk.Tree('*', [tree[0]])

        partial_trees = []
        while tree is not None:
            partial_trees.append(tree.copy(True))
            _, tree = detach(tree)

        for tree in reversed(partial_trees):
            print(TreePrettyPrinter(tree, None, ()).text(unicodelines=True))
            print()
        print("="*50+"\n")