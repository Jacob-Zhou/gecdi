# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Iterable, Tuple

import nltk


def map_token_ids(vocab_0, vocab_1, equal_labels=None):
    """
    Map token ids from vocab_0 to vocab_1

    Args:
        vocab_0 (dict): vocab_0
        vocab_1 (dict): vocab_1
        equal_labels (dict): equal_labels
    """
    if equal_labels is None:
        equal_labels = {}
    return [(i, vocab_1[equal_labels.get(k, k)]) for k, i in vocab_0.items()
            if k in vocab_1]
