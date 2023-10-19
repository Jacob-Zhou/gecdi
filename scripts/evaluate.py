# -*- coding: utf-8 -*-

from __future__ import annotations
from collections import Counter

import sys
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set,
                    Tuple, Union)

import nltk

class Metric(object):

    def __init__(self, reverse: Optional[bool] = None, eps: float = 1e-12) -> Metric:
        super().__init__()

        self.n = 0.0
        self.count = 0.0
        self.total_loss = 0.0
        self.reverse = reverse
        self.eps = eps

    def __lt__(self, other: Metric) -> bool:
        if not hasattr(self, 'score'):
            return True
        if not hasattr(other, 'score'):
            return False
        return (self.score < other.score) if not self.reverse else (self.score > other.score)

    def __le__(self, other: Metric) -> bool:
        if not hasattr(self, 'score'):
            return True
        if not hasattr(other, 'score'):
            return False
        return (self.score <= other.score) if not self.reverse else (self.score >= other.score)

    def __gt__(self, other: Metric) -> bool:
        if not hasattr(self, 'score'):
            return False
        if not hasattr(other, 'score'):
            return True
        return (self.score > other.score) if not self.reverse else (self.score < other.score)

    def __ge__(self, other: Metric) -> bool:
        if not hasattr(self, 'score'):
            return False
        if not hasattr(other, 'score'):
            return True
        return (self.score >= other.score) if not self.reverse else (self.score <= other.score)

    def __add__(self, other: Metric) -> Metric:
        return other

    @property
    def score(self):
        raise AttributeError

    @property
    def loss(self):
        return self.total_loss / (self.count + self.eps)


class SpanMetric(Metric):

    def __init__(
        self,
        loss: Optional[float] = None,
        preds: Optional[List[List[Tuple]]] = None,
        golds: Optional[List[List[Tuple]]] = None,
        reverse: bool = False,
        eps: float = 1e-12
    ) -> SpanMetric:
        super().__init__(reverse=reverse, eps=eps)

        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.n_tr = 0.0
        self.n_fr = 0.0
        self.n_e = 0.0
        self.n_c = 0.0
        self.utp = 0.0
        self.ltp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.labeled = True
        self.show_sent_recall = False

        if loss is not None:
            self(loss, preds, golds)

    def __repr__(self):
        # s = f"loss: {self.loss:.4f} - "
        if self.show_sent_recall:
            s = f"ErrorSents: {self.n_e:5.0f} CorrentSents: {self.n_c:5.0f} TR: {self.tr:7.2%} FR: {self.fr:7.2%} "
            return s
        s = f"gold: {self.gold:5.0f} pred: {self.pred:5.0f} "
        if self.labeled:
            s += f"LP: {self.lp:7.2%} LR: {self.lr:7.2%} LF: {self.lf:7.2%}"
        else:
            s += f"UP: {self.up:7.2%} UR: {self.ur:7.2%} UF: {self.uf:7.2%} "
        return s

    def __call__(
        self,
        loss: float,
        preds: List[List[Tuple]],
        golds: List[List[Tuple]]
    ) -> SpanMetric:
        self.n += len(preds)
        self.count += 1
        self.total_loss += float(loss)
        for pred, gold in zip(preds, golds):
            upred, ugold = Counter([tuple(span[:-1]) for span in pred]), Counter([tuple(span[:-1]) for span in gold])
            lpred, lgold = Counter([tuple(span) for span in pred]), Counter([tuple(span) for span in gold])
            utp, ltp = list((upred & ugold).elements()), list((lpred & lgold).elements())
            self.n_ucm += len(utp) == len(pred) == len(gold)
            self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.n_tr += ((len(gold) > 0) and (len(pred) > 0))
            self.n_fr += ((len(gold) == 0) and (len(pred) > 0))
            self.n_e += (len(gold) > 0)
            self.n_c += (len(gold) == 0)
            self.utp += len(utp)
            self.ltp += len(ltp)
            self.pred += len(pred)
            self.gold += len(gold)
        return self

    def __add__(self, other: SpanMetric) -> SpanMetric:
        metric = SpanMetric(eps=self.eps)
        metric.n = self.n + other.n
        metric.count = self.count + other.count
        metric.total_loss = self.total_loss + other.total_loss
        metric.n_ucm = self.n_ucm + other.n_ucm
        metric.n_lcm = self.n_lcm + other.n_lcm
        metric.n_tr = self.n_tr + other.n_tr
        metric.n_fr = self.n_fr + other.n_fr
        metric.n_e = self.n_e + other.n_e
        metric.n_c = self.n_c + other.n_c
        metric.utp = self.utp + other.utp
        metric.ltp = self.ltp + other.ltp
        metric.pred = self.pred + other.pred
        metric.gold = self.gold + other.gold
        metric.reverse = self.reverse or other.reverse
        return metric

    @property
    def score(self):
        return self.lf

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def tr(self):
        return self.n_tr / (self.n_e + self.eps)

    @property
    def fr(self):
        return self.n_fr / (self.n_c + self.eps)

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def lp(self):
        return self.ltp / (self.pred + self.eps)

    @property
    def lr(self):
        return self.ltp / (self.gold + self.eps)

    @property
    def lf(self):
        return 2 * self.ltp / (self.pred + self.gold + self.eps)

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

if __name__ == '__main__':
    args = sys.argv
    if len(args) < 3:
        print("Usage: python -m scripts.evaluate pred gold")
        exit()
    print(args)
    metrics = []

    gold = [
        factorize(nltk.Tree.fromstring(g))
        for g in open(args[-1], 'r')
    ]
    pred = [
        factorize(nltk.Tree.fromstring(p))
        for p in open(args[1], 'r')
    ]
    metric = SpanMetric(0, pred, gold)

    metric.show_sent_recall = True
    print(f"{'Sent Recall':>15}: ", end="")
    print(metric)
    metric.show_sent_recall = False

    print(f"{'Overall':>15}: ", end="")
    print(metric)

    print(f"{'Overall:U':>15}: ", end="")
    metric.labeled = False
    print(metric)

    for etype in ["SUB:ADJ", "SUB:ADJ:FORM", "SUB:ADV", "SUB:CONJ", "SUB:CONTR", "SUB:DET", "SUB:MORPH", "SUB:NOUN", "SUB:NOUN:INFL", "SUB:NOUN:NUM", "SUB:NOUN:POSS", 
                  "SUB:ORTH", "SUB:OTHER", "SUB:PART", "SUB:PREP", "SUB:PRON", "SUB:PUNCT", "SUB:SPELL", "SUB:VERB", "SUB:VERB:FORM", "SUB:VERB:INFL", "SUB:VERB:SVA", 
                  "SUB:VERB:TENSE", "SUB:WO", "MISS-L", "RED:ADJ", "RED:ADV", "RED:CONJ", "RED:DET", "RED:NOUN", "RED:NOUN:POSS", "RED:OTHER", "RED:PART", "RED:PREP", 
                  "RED:PRON", "RED:PUNCT", "RED:VERB", "RED:VERB:FORM", "RED:VERB:TENSE"]:
        gold = [
            factorize(nltk.Tree.fromstring(g), [etype])
            for g in open(args[-1], 'r')
        ]
        pred = [
            factorize(nltk.Tree.fromstring(p), [etype])
            for p in open(args[1], 'r')
        ]
        print(f"{etype:>15}: ", end="")
        print(SpanMetric(0, pred, gold))
