# -*- coding: utf-8 -*-

from __future__ import annotations
import copy

import os
import shutil
import tempfile
from contextlib import contextmanager
from io import StringIO
from typing import Iterable, List, Optional, Union

import nltk
import pathos.multiprocessing as mp
import torch
import torch.distributed as dist
import supar
from supar.utils.common import NUL
from supar.utils.fn import binarize, debinarize, pad
from supar.utils.logging import progress_bar
from supar.utils.parallel import gather, is_master
from supar.utils.tokenizer import Tokenizer
from supar.utils.transform import Sentence, Transform


class Field(supar.utils.Field):
    r"""
    Defines a datatype together with instructions for converting to :class:`~torch.Tensor`.
    :class:`Field` models common text processing datatypes that can be represented by tensors.
    It holds a :class:`~supar.utils.vocab.Vocab` object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The :class:`Field` object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method.

    Args:
        name (str):
            The name of the field.
        pad_token (str):
            The string token used as padding. Default: ``None``.
        unk_token (str):
            The string token used to represent OOV words. Default: ``None``.
        bos_token (str):
            A token that will be prepended to every example using this field, or ``None`` for no `bos_token`.
            Default: ``None``.
        eos_token (str):
            A token that will be appended to every example using this field, or ``None`` for no `eos_token`.
        lower (bool):
            Whether to lowercase the text in this field. Default: ``False``.
        use_vocab (bool):
            Whether to use a :class:`~supar.utils.vocab.Vocab` object.
            If ``False``, the data in this field should already be numerical.
            Default: ``True``.
        tokenize (function):
            The function used to tokenize strings using this field into sequential examples. Default: ``None``.
        fn (function):
            The function used for preprocessing the examples. Default: ``None``.
    """

    def __init__(self, *args, **kwargs):
        self.padding_side = kwargs.pop('padding_side') if 'padding_side' in kwargs else 'right'
        super().__init__(*args, **kwargs)

    def compose(self, batch: Iterable[torch.Tensor]) -> torch.Tensor:
        r"""
        Composes a batch of sequences into a padded tensor.

        Args:
            batch (Iterable[~torch.Tensor]):
                A list of tensors.

        Returns:
            A padded tensor converted to proper device.
        """

        return pad(batch, self.pad_index, padding_side=self.padding_side).to(self.device, non_blocking=True)


class Text(Transform):

    fields = ['SRC', 'TGT']

    def __init__(
        self,
        SRC: Optional[Union[Field, Iterable[Field]]] = None,
        TGT: Optional[Union[Field, Iterable[Field]]] = None
    ) -> Text:
        super().__init__()

        self.SRC = SRC
        self.TGT = TGT

    @property
    def src(self):
        return self.SRC,

    @property
    def tgt(self):
        return self.TGT,

    def load(
        self,
        data: Union[str, Iterable],
        lang: Optional[str] = None,
        **kwargs
    ) -> Iterable[TextSentence]:
        r"""
        Loads the data in Text-X format.
        Also supports for loading data from Text-U file with comments and non-integer IDs.

        Args:
            data (str or Iterable):
                A filename or a list of instances.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.

        Returns:
            A list of :class:`TextSentence` instances.
        """

        if lang is not None:
            tokenizer = Tokenizer(lang)
        if isinstance(data, str) and os.path.exists(data):
            f = open(data)
            if data.endswith('.txt'):
                lines = (i
                         for s in f
                         if len(s) > 1
                         for i in StringIO((s.split() if lang is None else tokenizer(s)) + '\n'))
            else:
                lines = f
        else:
            if lang is not None:
                data = [tokenizer(s) for s in ([data] if isinstance(data, str) else data)]
            else:
                data = [data] if isinstance(data[0], str) else data
            lines = (i for s in data for i in StringIO(s + '\n'))

        index, sentence = 0, []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                sentence = TextSentence(self, sentence, index)
                yield sentence
                index += 1
                sentence = []
            else:
                sentence.append(line)


class TextSentence(Sentence):

    def __init__(self, transform: Text, lines: List[str], index: Optional[int] = None) -> TextSentence:
        super().__init__(transform, index)

        self.cands = [(line+'\t').split('\t')[1] for line in lines[1:]]
        self.values = [lines[0].split('\t')[1], self.cands[0]]

    def __repr__(self):
        self.cands = self.values[1] if isinstance(self.values[1], list) else [self.values[1]]
        lines = ['S\t' + self.values[0]]
        lines.extend(['T\t' + i for i in self.cands])
        return '\n'.join(lines) + '\n'

class Tree(Transform):

    fields = ['SRC', 'TGT', 'SRCERROR', 'TGTERROR']

    def __init__(
        self,
        SRC: Optional[Union[Field, Iterable[Field]]] = None,
        TGT: Optional[Union[Field, Iterable[Field]]] = None,
        SRCERROR: Optional[Union[Field, Iterable[Field]]] = None,
        TGTERROR: Optional[Union[Field, Iterable[Field]]] = None,
        **kwargs
    ) -> Tree:
        super().__init__()
        self.error_schema = kwargs.pop('error_schema') if 'error_schema' in kwargs else 'last'
        self.fine_error_type = kwargs.pop('fine_error_type') if 'fine_error_type' in kwargs else False

        self.SRC = SRC
        self.TGT = TGT
        self.SRCERROR = SRCERROR
        self.TGTERROR = TGTERROR

    @property
    def src(self):
        return self.SRC, self.TGT

    @property
    def tgt(self):
        return self.SRCERROR, self.TGTERROR

    def load(
        self,
        data: Union[str, Iterable],
        lang: Optional[str] = None,
        **kwargs
    ) -> Iterable[TextSentence]:
        r"""
        Loads the data in Text-X format.
        Also supports for loading data from Text-U file with comments and non-integer IDs.

        Args:
            data (Union[str, Iterable]):
                A filename or a list of instances.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.

        Returns:
            A list of :class:`TextSentence` instances.
        """

        if lang is not None:
            tokenizer = Tokenizer(lang)
        if isinstance(data, str) and os.path.exists(data):
            f = open(data)
            if data.endswith('.txt'):
                lines = (i
                         for s in f
                         if len(s) > 1
                         for i in StringIO((s.split() if lang is None else tokenizer(s)) + '\n'))
            else:
                lines = f
        else:
            if lang is not None:
                data = [tokenizer(s) for s in ([data] if isinstance(data, str) else data)]
            else:
                data = [data] if isinstance(data[0], str) else data
            lines = (i for s in data for i in StringIO(s + '\n'))

        def consume(lines, chunksize=10000):
            index, sentence, chunk = 0, [], []
            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    chunk.append((sentence, index))
                    if len(chunk) == chunksize:
                        yield chunk
                        chunk = []
                    index += 1
                    sentence = []
                else:
                    sentence.append(line)
            if len(chunk) > 0:
                yield chunk

        @contextmanager
        def cache(lines):
            global global_transform
            global_transform = self
            ftemp = tempfile.mkdtemp()
            fbin = os.path.join(ftemp, 'data')
            try:
                yield ((chunk, f"{fbin}.{i}") for i, chunk in enumerate(consume(lines))), fbin
            finally:
                if dist.is_initialized() and not is_master():
                    dist.barrier()
                del global_transform
                shutil.rmtree(ftemp)

        with cache(lines) as (chunks, fbin):
            if is_master():
                def process(chunk, fb):
                    sentences = [TreeSentence(global_transform, *s) for s in progress_bar(chunk)]
                    sentences = [s for s in sentences if s.vaild]
                    return binarize({'sentences': sentences}, fb)[0]
                with mp.Pool(32) as pool:
                    results = [pool.apply_async(process, (chunk, fb)) for chunk, fb in chunks]
                    binarize((r.get() for r in results), fbin, merge=True)
            if dist.is_initialized() and not is_master():
                fbin = gather(fbin)[0]
                dist.barrier()
            for s in debinarize(fbin, meta=True)['sentences']:
                yield debinarize(fbin, s)


class TreeSentence(Sentence):

    def __init__(self, transform: Text, lines: List[str], index: Optional[int] = None) -> TextSentence:
        super().__init__(transform, index)
        self.cands = [self.get_tree(line) for line in lines[1:]]
        src_tree, tgt_tree = self.get_tree(lines[0]), self.cands[0]
        if len(tgt_tree.leaves()) == 0:
            tgt_tree = copy.deepcopy(src_tree)
        src_leaves, tgt_leaves = [], []
        for pos in sorted(src_tree.treepositions('leaves'), reverse=True):
            terminal = src_tree[pos[:-1]]
            label, token = terminal.label(), terminal[0].replace('-LRB-', '(').replace('-RRB-', ')')
            if not transform.fine_error_type:
                label = "::".join({l.split(":")[0] for l in label.split('::')})
            src_leaves.insert(0, token)
            subtokens = transform.SRC.tokenize(token)
            if len(subtokens) == 1:
                src_tree[pos[:-2]][pos[-2]:pos[-2]+1] = [nltk.Tree(label, subtokens)]
            else:
                labels = [label] * len(subtokens)
                src_tree[pos[:-2]][pos[-2]:pos[-2]+1] = [nltk.Tree(l, [st]) for l, st in zip(labels, subtokens)]
        for pos in sorted(tgt_tree.treepositions('leaves'), reverse=True):
            terminal = tgt_tree[pos[:-1]]
            label, token = terminal.label(), terminal[0].replace('-LRB-', '(').replace('-RRB-', ')')
            if not transform.fine_error_type:
                label = "::".join({l.split(":")[0] for l in label.split('::')})
            tgt_leaves.insert(0, token)
            subtokens = transform.TGT.tokenize(token)
            if len(subtokens) == 1:
                tgt_tree[pos[:-2]][pos[-2]:pos[-2]+1] = [nltk.Tree(label, subtokens)]
            else:
                if transform.error_schema == 'last':
                    labels = ['CORRECT'] * (len(subtokens) - 1) + [label]
                elif transform.error_schema == 'first':
                    labels = [label] + ['CORRECT'] * (len(subtokens) - 1)
                elif transform.error_schema == 'all':
                    labels = [label] * len(subtokens)
                elif transform.error_schema == 'partial-last':
                    labels = [NUL if label != 'CORRECT' else 'CORRECT'] * (len(subtokens) - 1) + [label]
                else:
                    raise ValueError(f'Unknown error schema: {transform.error_schema}')
                tgt_tree[pos[:-2]][pos[-2]:pos[-2]+1] = [nltk.Tree(l, [st]) for l, st in zip(labels, subtokens)]
        src_leaves, tgt_leaves = ' '.join(src_leaves), ' '.join(tgt_leaves)

        self.vaild = True

        # the root node must have a unary chain
        if len(src_tree) > 1:
            src_tree[:] = [nltk.Tree('', src_tree)]
        if len(src_tree.leaves()) > 0:
            src_errors = list(zip(*src_tree.pos()))[-1]
        else:
            self.vaild = False
            return
        if len(tgt_tree) > 1:
            tgt_tree[:] = [nltk.Tree('', tgt_tree)]
        if len(tgt_tree.leaves()) > 0:
            tgt_errors = list(zip(*tgt_tree.pos()))[-1]
        else:
            self.vaild = False
            return

        self.tgt_tree = tgt_tree

        self.values = [src_leaves, tgt_leaves, src_errors, tgt_errors]

    def get_tree(self, text):
        if text.startswith("(") or len(text) == 0:
            return nltk.Tree.fromstring(text)
        elif text[0].lower() in {'t', 's'}:
            text = (text + '\t').split('\t')[1]
            text = text.replace("(", '-LRB-').replace(")", '-RRB-')
            return nltk.Tree('TOP', [nltk.Tree('S', [nltk.Tree('_', [word]) for word in text.split()])])
        else:
            text = text.replace("(", '-LRB-').replace(")", '-RRB-')
            return nltk.Tree('TOP', [nltk.Tree('S', [nltk.Tree('_', [word]) for word in text.split()])])

    def __repr__(self):
        cands = self.values[1] if isinstance(self.values[1], list) else [self.values[1]]
        lines = ['S\t' + self.values[0]]
        lines.extend(['T\t' + i for i in cands])
        return '\n'.join(lines) + '\n'
