# -*- coding: utf-8 -*-

from __future__ import annotations
from collections import Counter

import math
import os
import subprocess
import tempfile
from typing import List, Optional, Tuple

import torch
from supar.utils.metric import Metric


class PerplexityMetric(Metric):
    def __init__(self,
                 loss: Optional[float] = None,
                 preds: Optional[torch.Tensor] = None,
                 golds: Optional[torch.Tensor] = None,
                 mask: Optional[torch.BoolTensor] = None,
                 reverse: bool = True,
                 eps: float = 1e-12) -> PerplexityMetric:
        super().__init__(reverse=reverse, eps=eps)

        self.n_tokens = 0.

        self.tp = 0.0
        self.pred = 0.0
        self.gold = 0.0

        self.total_loss = 0.

        if loss is not None:
            self(loss, preds, golds, mask)

    def __repr__(self):
        s = f"loss: {self.loss:.4f} PPL: {self.ppl:.4f}"
        if self.tp > 0:
            s += f" - TGT: P: {self.p:6.2%} R: {self.r:6.2%} F0.5: {self.f:6.2%}"
        return s

    def __call__(self, loss: float, preds: Tuple[List, torch.Tensor],
                 golds: Tuple[List, torch.Tensor],
                 mask: torch.BoolTensor) -> PerplexityMetric:
        n_tokens = mask.sum().item()
        self.n += len(mask)
        self.count += 1
        self.n_tokens += n_tokens
        self.total_loss += float(loss) * n_tokens

        if preds is not None:
            with tempfile.TemporaryDirectory() as t:
                fsrc, fpred, fgold = os.path.join(t, 'src'), os.path.join(
                    t, 'pred'), os.path.join(t, 'gold')
                pred_m2, gold_m2 = os.path.join(t, 'pred.m2'), os.path.join(
                    t, 'gold.m2')
                with open(fsrc, 'w') as fs, open(fpred, 'w') as f:
                    for s, i in preds:
                        fs.write(s + '\n')
                        f.write(i + '\n')
                with open(fgold, 'w') as f:
                    for _, i in golds:
                        f.write(i + '\n')
                subprocess.check_output([
                    'errant_parallel', '-orig', f'{fsrc}', '-cor', f'{fpred}',
                    '-out', f'{pred_m2}'
                ])
                subprocess.check_output([
                    'errant_parallel', '-orig', f'{fsrc}', '-cor', f'{fgold}',
                    '-out', f'{gold_m2}'
                ])
                out = subprocess.check_output(
                    [
                        'errant_compare', '-hyp', f'{pred_m2}', '-ref',
                        f'{gold_m2}'
                    ],
                    stderr=subprocess.STDOUT).decode()
                tp, fp, fn = (int(i) for i in out.split('\n')[3].split()[:3])
                self.tp += tp
                self.pred += tp + fp
                self.gold += tp + fn
        return self

    def __add__(self, other: PerplexityMetric) -> PerplexityMetric:
        metric = PerplexityMetric(eps=self.eps)
        metric.n = self.n + other.n
        metric.count = self.count + other.count
        metric.n_tokens = self.n_tokens + other.n_tokens
        metric.total_loss = self.total_loss + other.total_loss

        metric.tp = self.tp + other.tp
        metric.pred = self.pred + other.pred
        metric.gold = self.gold + other.gold
        metric.reverse = self.reverse or other.reverse
        return metric

    @property
    def score(self):
        return self.f if self.f > 0 else self.ppl

    @property
    def loss(self):
        return self.total_loss / self.n_tokens

    @property
    def ppl(self):
        return math.pow(2, (self.loss / math.log(2)))

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return (1 + 0.5**2) * self.p * self.r / (0.5**2 * self.p + self.r +
                                                 self.eps)


class SpanMetric(Metric):
    def __init__(self,
                 loss: Optional[float] = None,
                 preds: Optional[List[List[Tuple]]] = None,
                 golds: Optional[List[List[Tuple]]] = None,
                 reverse: bool = False,
                 beta: Optional[float] = 1.,
                 eps: float = 1e-12) -> SpanMetric:
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
        self.beta = beta

        if loss is not None:
            self(loss, preds, golds)

    def __repr__(self):
        s = f"ErrorSents: {self.n_e:6.0f} CorrectSents: {self.n_c:6.0f} TR: {self.tr:7.2%} FR: {self.fr:7.2%} "
        # s += f"GoldSpans: {self.gold:6.0f} PredSpans: {self.pred:6.0f} "
        s += f"UP: {self.up:7.2%} UR: {self.ur:7.2%} UF{'' if self.beta == 1.0 else self.beta}: {self.uf:7.2%} "
        s += f"LP: {self.lp:7.2%} LR: {self.lr:7.2%} LF{'' if self.beta == 1.0 else self.beta}: {self.lf:7.2%}"
        return s

    def __call__(self, loss: float, preds: List[List[Tuple]],
                 golds: List[List[Tuple]]) -> SpanMetric:
        self.n += len(preds)
        self.count += 1
        self.total_loss += float(loss)
        for pred, gold in zip(preds, golds):
            upred, ugold = Counter([tuple(span[:-1])
                                    for span in pred]), Counter(
                                        [tuple(span[:-1]) for span in gold])
            lpred, lgold = Counter([tuple(span) for span in pred
                                    ]), Counter([tuple(span) for span in gold])
            utp, ltp = list((upred & ugold).elements()), list(
                (lpred & lgold).elements())
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
        metric = SpanMetric(eps=self.eps, beta=self.beta)
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
        return (1 + self.beta**2) * self.utp / (self.pred +
                                                (self.beta**2) * self.gold +
                                                self.eps)

    @property
    def lp(self):
        return self.ltp / (self.pred + self.eps)

    @property
    def lr(self):
        return self.ltp / (self.gold + self.eps)

    @property
    def lf(self):
        return (1 + self.beta**2) * self.ltp / (self.pred +
                                                (self.beta**2) * self.gold +
                                                self.eps)
