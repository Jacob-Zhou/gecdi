# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import nltk
import os
import functools
import signal

import numpy as np

from collections import Counter, namedtuple
from tqdm import tqdm
from functools import lru_cache
from multiprocessing.pool import Pool
from transformers import AutoTokenizer


def timeout(sec):
    """
    timeout decorator
    :param sec: function raise TimeoutError after ? seconds
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):

            def _handle_timeout(signum, frame):
                err_msg = f'Function {func.__name__} timed out after {sec} seconds'
                raise TimeoutError(err_msg)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(sec)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapped_func

    return decorator


##################################################
#   Copy from ChERRANT
##################################################

import Levenshtein
from typing import List, Tuple, Dict
from pypinyin import pinyin, Style
from string import punctuation
from itertools import groupby
from char_smi import CharFuncs

REAL_PATH = os.path.split(os.path.realpath(__file__))[0]
char_smi = CharFuncs('scripts/generate_chn_treebank/data/char_meta.txt')
chinese_punct = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"
english_punct = punctuation
punct = chinese_punct + english_punct
Correction = namedtuple(
    "Correction",
    ["o_start", "o_end", "o_str", "c_start", "c_end", "c_str", "type"],
)


def check_all_chinese(word):
    """
    判断一个单词是否全部由中文组成
    :param word:
    :return:
    """
    return all(['\u4e00' <= ch <= '\u9fff' for ch in word])


def check_spell_error(src_span: str,
                      tgt_span: str,
                      threshold: float = 0.8) -> bool:
    if len(src_span) != len(tgt_span):
        return False
    src_chars = [ch for ch in src_span]
    tgt_chars = [ch for ch in tgt_span]
    if sorted(src_chars) == sorted(tgt_chars):  # 词内部字符异位
        return True
    for src_char, tgt_char in zip(src_chars, tgt_chars):
        if src_char != tgt_char:
            if src_char not in char_smi.data or tgt_char not in char_smi.data:
                return False
            v_sim = char_smi.shape_similarity(src_char, tgt_char)
            p_sim = char_smi.pronunciation_similarity(src_char, tgt_char)
            if v_sim + p_sim < threshold and not (set(
                    pinyin(src_char, style=Style.NORMAL, heteronym=True)[0]
            ) & set(pinyin(tgt_char, style=Style.NORMAL, heteronym=True)[0])):
                return False
    return True


def read_cilin():
    """
    Cilin 詞林 is a thesaurus with semantic information
    """
    # TODO -- fix this path
    lines = open("scripts/generate_chn_treebank/data/cilin.txt",
                 "r",
                 encoding="gbk").read().strip().split("\n")
    semantic_dict = {}
    semantic_classes = {}
    for line in lines:
        code, *words = line.split(" ")
        for word in words:
            semantic_dict[word] = code
        # make reverse dict
        if code in semantic_classes:
            semantic_classes[code] += words
        else:
            semantic_classes[code] = words
    return semantic_dict, semantic_classes


def read_confusion():
    confusion_dict = {}
    with open("scripts/generate_chn_treebank/data/confusion_dict.txt",
              "r",
              encoding="utf-8") as f:
        for line in f:
            li = line.rstrip('\n').split(" ")
            confusion_dict[li[0]] = li[1:]
    return confusion_dict


class Alignment:
    """
    对齐错误句子和正确句子，
    使用编辑距离算法抽取编辑操作
    """

    def __init__(
        self,
        semantic_dict: Dict,
        confusion_dict: Dict,
        granularity: str = "word",
    ) -> None:
        """
        构造函数
        :param semantic_dict: 语义词典（大词林）
        :param confusion_dict: 字符混淆集
        """
        self.insertion_cost = 1
        self.deletion_cost = 1
        self.semantic_dict = semantic_dict
        self.confusion_dict = confusion_dict
        # Because we use character level tokenization, this doesn't currently use POS
        self._open_pos = {}  # 如果是词级别，还可以利用词性是否相同来计算cost
        self.granularity = granularity  # word-level or character-level
        self.align_seqs = []

    def __call__(self, src: List[Tuple], tgt: List[Tuple]):
        cost_matrix, oper_matrix = self.align(src[::-1], tgt[::-1])
        align_seq = self.get_cheapest_align_seq(oper_matrix)
        o_l, c_l = len(src), len(tgt)
        align_seq = [[(op, o_l - o_e, o_l - o_s, c_l - c_e, c_l - c_s)
                      for op, o_s, o_e, c_s, c_e in a[::-1]]
                     for a in align_seq]
        return align_seq

    def _get_semantic_class(self, word):
        """
        NOTE: Based on the paper:
        Improved-Edit-Distance Kernel for Chinese Relation Extraction
        获取每个词语的语义类别（基于大词林，有三个级别）
        """
        if word in self.semantic_dict:
            code = self.semantic_dict[word]
            high, mid, low = code[0], code[1], code[2:4]
            return high, mid, low
        else:  # unknown
            return None

    @staticmethod
    def _get_class_diff(a_class, b_class):
        """
        d == 3 for equivalent semantics
        d == 0 for completely different semantics
        根据大词林的信息，计算两个词的语义类别的差距
        """
        d = sum([a == b for a, b in zip(a_class, b_class)])
        return d

    def _get_semantic_cost(self, a, b):
        """
        计算基于语义信息的替换操作cost
        :param a: 单词a的语义类别
        :param b: 单词b的语义类别
        :return: 替换编辑代价
        """
        a_class = self._get_semantic_class(a)
        b_class = self._get_semantic_class(b)
        # unknown class, default to 1
        if a_class is None or b_class is None:
            return 4
        elif a_class == b_class:
            return 0
        else:
            return 2 * (3 - self._get_class_diff(a_class, b_class))

    def _get_pos_cost(self, a_pos, b_pos):
        """
        计算基于词性信息的编辑距离cost
        :param a_pos: 单词a的词性
        :param b_pos: 单词b的词性
        :return: 替换编辑代价
        """
        if a_pos == b_pos:
            return 0
        elif a_pos in self._open_pos and b_pos in self._open_pos:
            return 0.25
        else:
            return 0.499

    def _get_char_cost(self, a, b, pinyin_a, pinyin_b):
        """
        NOTE: This is a replacement of ERRANTS lemma cost for Chinese
        计算基于字符相似度的编辑距离cost
        """
        if not (check_all_chinese(a) and check_all_chinese(b)):
            return 0.5
        if len(a) > len(b):
            a, b = b, a
            pinyin_a, pinyin_b = pinyin_b, pinyin_a
        if a == b:
            return 0
        else:
            return self._get_spell_cost(a, b, pinyin_a, pinyin_b)

    def _get_spell_cost(self, a, b, pinyin_a, pinyin_b):
        """
        计算两个单词拼写相似度，分别由字形相似度和字音相似度组成
        :param a: 单词a
        :param b: 单词b，且单词a的长度小于等于b
        :param pinyin_a: 单词a的拼音
        :param pinyin_b: 单词b的拼音
        :return: 替换操作cost
        """
        count = 0
        for i in range(len(a)):
            for j in range(len(b)):
                if a[i] == b[j] or (set(pinyin_a) & set(pinyin_b)) or (
                        b[j] in self.confusion_dict.keys()
                        and a[i] in self.confusion_dict[b[j]]) or (
                            a[i] in self.confusion_dict.keys()
                            and b[j] in self.confusion_dict[a[i]]):
                    count += 1
                    break
        return (len(a) - count) / (len(a) * 2)

    def get_sub_cost(self, a_seg, b_seg):
        """
        Calculate the substitution cost between words a and b
        计算两个单词替换操作的编辑cost，最大为2，等于一次删除和一次添加
        """
        if a_seg[0] == b_seg[0]:
            return 0

        if self.granularity == "word":  # 词级别可以额外利用词性信息
            semantic_cost = self._get_semantic_cost(a_seg[0], b_seg[0]) / 6.0
            pos_cost = self._get_pos_cost(a_seg[1], b_seg[1])
            char_cost = self._get_char_cost(a_seg[0], b_seg[0], a_seg[2],
                                            b_seg[2])
            return semantic_cost + pos_cost + char_cost
        else:  # 字级别只能利用字义信息（从大词林中获取）和字面相似度信息
            semantic_cost = self._get_semantic_cost(a_seg[0], b_seg[0]) / 6.0
            if a_seg[0] in punct and b_seg[0] in punct:
                pos_cost = 0.0
            elif a_seg[0] not in punct and b_seg[0] not in punct:
                pos_cost = 0.25
            else:
                pos_cost = 0.499
            # char_cost = self._get_char_cost(a_seg[0], b_seg[0], a_seg[2], b_seg[2])
            # return semantic_cost + char_cost + pos_cost
            return pos_cost + 1

    # @timeout(1)
    def align(self, src: List[Tuple], tgt: List[Tuple]):
        """
        Based on ERRANT's alignment
        基于改进的动态规划算法，为原句子的每个字打上编辑标签，以便使它能够成功转换为目标句子。
        编辑操作类别：
        1) M：Match，即KEEP，即当前字保持不变
        2) D：Delete，删除，即当前字需要被删除
        3) I：Insert，插入，即当前字需要被插入
        4) T：Transposition，移位操作，即涉及到词序问题
        """
        cost_matrix = np.zeros((len(src) + 1, len(tgt) + 1))  # 编辑cost矩阵
        oper_matrix = np.full((len(src) + 1, len(tgt) + 1), "O",
                              dtype=object)  # 操作矩阵
        # Fill in the edges
        for i in range(1, len(src) + 1):
            cost_matrix[i][0] = cost_matrix[i - 1][0] + 1
            oper_matrix[i][0] = ["D"]
        for j in range(1, len(tgt) + 1):
            cost_matrix[0][j] = cost_matrix[0][j - 1] + 1
            oper_matrix[0][j] = ["I"]

        # Loop through the cost matrix
        for i in range(len(src)):
            for j in range(len(tgt)):
                # Matches
                if src[i][0] == tgt[j][0]:  # 如果两个字相等，则匹配成功（Match），编辑距离为0
                    cost_matrix[i +
                                1][j +
                                   1] = cost_matrix[i][j]  # distance punish
                    oper_matrix[i + 1][j + 1] = ["M"]
                # Non-matches
                else:
                    del_cost = cost_matrix[i][
                        j + 1] + self.deletion_cost  # 由删除动作得到的总cost
                    ins_cost = cost_matrix[
                        i + 1][j] + self.insertion_cost  # 由插入动作得到的总cost
                    sub_cost = cost_matrix[i][j] + self.get_sub_cost(
                        src[i], tgt[j])  # 由替换动作得到的总cost
                    # Calculate transposition cost
                    # 计算移位操作的总cost
                    trans_cost = float("inf")
                    k = 1
                    while (i - k >= 0 and j - k >= 0
                           and cost_matrix[i - k + 1][j - k + 1] !=
                           cost_matrix[i - k][j - k]):
                        p1 = sorted([a[0] for a in src][i - k:i + 1])
                        p2 = sorted([b[0] for b in tgt][j - k:j + 1])
                        if p1 == p2:
                            trans_cost = cost_matrix[i - k][j - k] + k
                            break
                        k += 1

                    costs = [trans_cost, sub_cost, ins_cost, del_cost]
                    ind = costs.index(min(costs))
                    cost_matrix[i + 1][j + 1] = costs[ind]
                    for idx, cost in enumerate(costs):
                        if cost == costs[ind]:
                            if idx == 0:
                                if oper_matrix[i + 1][j + 1] == "O":
                                    oper_matrix[i + 1][j +
                                                       1] = ["T" + str(k + 1)]
                                else:
                                    oper_matrix[i + 1][j +
                                                       1].append("T" +
                                                                 str(k + 1))
                            elif idx == 1:
                                if oper_matrix[i + 1][j + 1] == "O":
                                    oper_matrix[i + 1][j + 1] = ["S"]
                                else:
                                    oper_matrix[i + 1][j + 1].append("S")
                            elif idx == 2:
                                if oper_matrix[i + 1][j + 1] == "O":
                                    oper_matrix[i + 1][j + 1] = ["I"]
                                else:
                                    oper_matrix[i + 1][j + 1].append("I")
                            else:
                                if oper_matrix[i + 1][j + 1] == "O":
                                    oper_matrix[i + 1][j + 1] = ["D"]
                                else:
                                    oper_matrix[i + 1][j + 1].append("D")
        return cost_matrix, oper_matrix

    def _dfs(self, i, j, align_seq_now, oper_matrix, strategy="all"):
        """
        深度优先遍历，获取最小编辑距离相同的所有序列
        """
        if i + j == 0:
            self.align_seqs.append(align_seq_now)
        else:
            ops = oper_matrix[i][j]  # 可以类比成搜索一棵树从根结点到叶子结点的所有路径
            if strategy != "all": ops = ops[:1]
            for op in ops:
                if op in {"M", "S"}:
                    self._dfs(i - 1, j - 1,
                              align_seq_now + [(op, i - 1, i, j - 1, j)],
                              oper_matrix, strategy)
                elif op == "D":
                    self._dfs(i - 1, j, align_seq_now + [(op, i - 1, i, j, j)],
                              oper_matrix, strategy)
                elif op == "I":
                    self._dfs(i, j - 1, align_seq_now + [(op, i, i, j - 1, j)],
                              oper_matrix, strategy)
                else:
                    k = int(op[1:])
                    self._dfs(i - k, j - k,
                              align_seq_now + [(op, i - k, i, j - k, j)],
                              oper_matrix, strategy)

    def get_cheapest_align_seq(self, oper_matrix):
        """
        回溯获得编辑距离最小的编辑序列
        """
        self.align_seqs = []
        i = oper_matrix.shape[0] - 1
        j = oper_matrix.shape[1] - 1
        if abs(i - j) > 10:
            self._dfs(i, j, [], oper_matrix, "first")
        else:
            self._dfs(i, j, [], oper_matrix, "all")
        final_align_seqs = [seq[::-1] for seq in self.align_seqs]
        return final_align_seqs


class Classifier:

    def __call__(self, src, tgt, edits):
        """
        为编辑操作划分错误类型
        :param src: 错误句子信息
        :param tgt: 正确句子信息
        :param edits: 编辑操作
        :param verbose: 是否打印信息
        :return: 划分完错误类型后的编辑操作
        """
        results = []
        src_tokens = [x[0] for x in src]
        tgt_tokens = [x[0] for x in tgt]
        for edit in edits:
            error_type = edit[0]
            src_span = " ".join(src_tokens[edit[1]:edit[2]])
            tgt_span = " ".join(tgt_tokens[edit[3]:edit[4]])
            # print(tgt_span)
            cor = None
            if error_type[0] == "T":
                # cor = Correction("W", tgt_span, (edit[1], edit[2]))
                cor = Correction(edit[1], edit[2], src_span, edit[3], edit[4],
                                 tgt_span, "R:WO")
            elif error_type[0] == "D":
                # cor = Correction("R", "-NONE-", (edit[1], edit[2]))
                cor = Correction(edit[1], edit[2], src_span, edit[3], edit[4],
                                 "", "U")
            elif error_type[0] == "I":
                # cor = Correction("M", tgt_span, (edit[1], edit[2]))
                cor = Correction(edit[1], edit[2], src_span, edit[3], edit[4],
                                 tgt_span, "M")
            elif error_type[0] == "S":
                if check_spell_error(src_span.replace(" ", ""),
                                     tgt_span.replace(" ", "")):
                    # cor = Correction("S:SPELL", tgt_span, (edit[1], edit[2]))
                    cor = Correction(edit[1], edit[2], src_span, edit[3],
                                     edit[4], tgt_span, "R:SPELL")
                else:
                    # cor = Correction("S", tgt_span, (edit[1], edit[2]))
                    cor = Correction(edit[1], edit[2], src_span, edit[3],
                                     edit[4], tgt_span, "R")
            results.append(cor)
        return results


class Annotator:

    def __init__(self,
                 align: Alignment,
                 merger: Merger,
                 classifier: Classifier,
                 granularity: str = "word",
                 strategy: str = "first"):
        self.align = align
        self.merger = merger
        self.classifier = classifier
        self.granularity = granularity
        self.strategy = strategy

    @classmethod
    def create_default(cls,
                       granularity: str = "word",
                       strategy: str = "first"):
        """
        Default parameters used in the paper
        """
        semantic_dict, semantic_class = read_cilin()
        confusion_dict = read_confusion()
        align = Alignment(semantic_dict, confusion_dict, granularity)
        merger = Merger(granularity)
        classifier = Classifier()
        return cls(align, merger, classifier, granularity, strategy)

    def __call__(self,
                 src: List[Tuple],
                 tgt: List[Tuple],
                 annotator_id: int = 0,
                 verbose: bool = False):
        """
        Align sentences and annotate them with error type information
        """
        align_obj = self.align(src, tgt)[0]
        edits = self.merger(align_obj, src, tgt, verbose)
        cors = self.classifier(src, tgt, edits)
        return cors


class Merger:
    """
    合并编辑操作，从Token-Level转换为Span-Level
    """

    def __init__(self, granularity: str = "word", merge: bool = False):
        chinese_punct = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟–—‘'‛“”„‟…‧."
        self.punctuation = punctuation + chinese_punct
        self.not_merge_token = [punct for punct in self.punctuation]
        self.granularity = granularity
        self.merge = merge

    @staticmethod
    def _merge_edits(seq, tag="X"):
        if seq:
            return [(tag, seq[0][1], seq[-1][2], seq[0][3], seq[-1][4])]
        else:
            return seq

    @staticmethod
    def _check_revolve(span_a, span_b):
        span_a = span_a + span_a
        return span_b in span_a

    def _process_seq(self, seq, src_tokens, tgt_tokens):
        if len(seq) <= 1:
            return seq

        ops = [op[0] for op in seq]

        if set(ops) == {"M"}:
            return self._merge_edits(seq, "M")
        else:
            return seq

    def __call__(self, align_obj, src: List, tgt: List, verbose: bool = False):
        """
        Based on ERRANT's merge, adapted for Chinese
        """
        src_tokens = [x[0] for x in src]
        tgt_tokens = [x[0] for x in tgt]
        edits = []
        # Split alignment into groups of M, T and rest. (T has a number after it)
        # Todo 一旦插入、删除、替换的对象中含有标点，那么不与其它编辑合并
        # Todo 缺失成分标签也不与其它编辑合并
        for op, group in groupby(
                align_obj,
                lambda x: x[0][0] if x[0][0] in {"M", "T"} else False,
        ):
            group = list(group)
            # T is always split TODO: Evaluate this
            if op == "T":
                for seq in group:
                    edits.append(seq)
            # Process D, I and S subsequence
            else:
                # Turn the processed sequence into edits
                processed = self._process_seq(group, src_tokens, tgt_tokens)
                for seq in processed:
                    edits.append(seq)

        filtered_edits = []
        i = 0
        while i < len(edits):
            e1 = edits[i][0][0]

            if i < len(edits) - 2:
                e2 = edits[i + 1][0][0]
                e3 = edits[i + 2][0][0]

                # Find "S M S" patterns
                # Ex:
                #   S     M     S
                # 冬阴功  对  外国人
                # 外国人  对  冬阴功
                if e1 == "S" and e2 == "M" and e3 == "S":
                    w1 = "".join(src_tokens[edits[i][1]:edits[i][2]])
                    w2 = "".join(tgt_tokens[edits[i][3]:edits[i][4]])
                    w3 = "".join(src_tokens[edits[i + 2][1]:edits[i + 2][2]])
                    w4 = "".join(tgt_tokens[edits[i + 2][3]:edits[i + 2][4]])
                    if min([len(w1), len(w2), len(w3), len(w4)]) == 1:
                        if w1 == w4 and w2 == w3:
                            group = [edits[i], edits[i + 1], edits[i + 2]]
                            processed = self._merge_edits(
                                group,
                                "T" + str(edits[i + 2][2] - edits[i][1]))
                            for seq in processed:
                                filtered_edits.append(seq)
                            i += 3
                        else:
                            filtered_edits.append(edits[i])
                            i += 1
                    else:
                        if Levenshtein.distance(
                                w1, w4) <= 1 and Levenshtein.distance(w2,
                                                                      w3) <= 1:
                            group = [edits[i], edits[i + 1], edits[i + 2]]
                            processed = self._merge_edits(
                                group,
                                "T" + str(edits[i + 2][2] - edits[i][1]))
                            for seq in processed:
                                filtered_edits.append(seq)
                            i += 3
                        else:
                            filtered_edits.append(edits[i])
                            i += 1
                # Find "D M I" or "I M D" patterns
                # Ex:
                #   D        M              I
                # 旅游 去   陌生 的   地方
                #      去   陌生 的   地方  旅游
                elif (e1 == "D" and (e2 == "M" or e2.startswith("T"))
                      and e3 == "I") or (e1 == "I" and
                                         (e2 == "M" or e2.startswith("T"))
                                         and e3 == "D"):
                    if e1 == "D":
                        delete_token = src_tokens[edits[i][1]:edits[i][2]]
                        insert_token = tgt_tokens[edits[i + 2][3]:edits[i +
                                                                        2][4]]
                    else:
                        delete_token = src_tokens[edits[i + 2][1]:edits[i +
                                                                        2][2]]
                        insert_token = tgt_tokens[edits[i][3]:edits[i][4]]
                    a, b = "".join(delete_token), "".join(insert_token)
                    if len(a) < len(b):
                        a, b = b, a
                    if a not in self.punctuation and b not in self.punctuation and len(
                            a) - len(b) <= 1:
                        if len(b) == 1:
                            if a == b:
                                group = [edits[i], edits[i + 1], edits[i + 2]]
                                processed = self._merge_edits(
                                    group,
                                    "T" + str(edits[i + 2][2] - edits[i][1]))
                                for seq in processed:
                                    filtered_edits.append(seq)
                                i += 3
                            else:
                                filtered_edits.append(edits[i])
                                i += 1
                        else:
                            if Levenshtein.distance(a, b) <= 1 or (
                                    len(a) == len(b)
                                    and self._check_revolve(a, b)):
                                group = [edits[i], edits[i + 1], edits[i + 2]]
                                processed = self._merge_edits(
                                    group,
                                    "T" + str(edits[i + 2][2] - edits[i][1]))
                                for seq in processed:
                                    filtered_edits.append(seq)
                                i += 3
                            else:
                                filtered_edits.append(edits[i])
                                i += 1
                    else:
                        filtered_edits.append(edits[i])
                        i += 1
                else:
                    if e1 != "M":
                        filtered_edits.append(edits[i])
                    i += 1
            else:
                if e1 != "M":
                    filtered_edits.append(edits[i])
                i += 1
        # In rare cases with word-level tokenization, the following error can occur:
        # M     D   S       M
        # 有    時  住      上層
        # 有        時住    上層
        # Which results in S: 時住 --> 時住
        # We need to filter this case out
        second_filter = []
        for edit in filtered_edits:  # 避免因为分词错误导致的mismatch现象
            span1 = "".join(src_tokens[edit[1]:edit[2]])
            span2 = "".join(tgt_tokens[edit[3]:edit[4]])

            if span1 != span2:
                if edit[0] == "S":
                    b = True
                    # In rare cases with word-level tokenization, the following error can occur:
                    # S       I     I       M
                    # 负责任               老师
                    # 负     责任   的     老师
                    # Which results in S: 负责任 --> 负 责任 的
                    # We need to convert this edit to I: --> 的

                    # 首部有重叠
                    common_str = ""
                    tmp_new_start_1 = edit[1]
                    for i in range(edit[1], edit[2]):
                        if not span2.startswith(common_str + src_tokens[i]):
                            break
                        common_str += src_tokens[i]
                        tmp_new_start_1 = i + 1
                    new_start_1, new_start_2 = edit[1], edit[3]
                    if common_str:
                        tmp_str = ""
                        for i in range(edit[3], edit[4]):
                            tmp_str += tgt_tokens[i]
                            if tmp_str == common_str:
                                new_start_1, new_start_2 = tmp_new_start_1, i + 1
                                b = False
                                break
                            elif len(tmp_str) > len(common_str):
                                break
                    # 尾部有重叠
                    common_str = ""
                    new_end_1, new_end_2 = edit[2], edit[4]
                    tmp_new_end_1 = edit[2]
                    for i in reversed(range(new_start_1, edit[2])):
                        if not span2.endswith(src_tokens[i] + common_str):
                            break
                        common_str = src_tokens[i] + common_str
                        tmp_new_end_1 = i
                    if common_str:
                        tmp_str = ""
                        for i in reversed(range(new_start_2, edit[4])):
                            tmp_str = tgt_tokens[i] + tmp_str
                            if tmp_str == common_str:
                                new_end_1, new_end_2 = tmp_new_end_1, i
                                b = False
                                break
                            elif len(tmp_str) > len(common_str):
                                break
                    if b:
                        second_filter.append(edit)
                    else:
                        if new_start_1 == new_end_1:
                            new_edit = ("I", new_start_1, new_end_1,
                                        new_start_2, new_end_2)
                        elif new_start_2 == new_end_2:
                            new_edit = ("D", new_start_1, new_end_1,
                                        new_start_2, new_end_2)
                        else:
                            new_edit = ("S", new_start_1, new_end_1,
                                        new_start_2, new_end_2)
                        second_filter.append(new_edit)
                else:
                    second_filter.append(edit)
        return second_filter


##################################################
#   Copy from ChERRANT
##################################################


def factorize(tree, insert_to_lowest_parent=True):
    span2node = {}
    n_words = len(tree.leaves())
    if insert_to_lowest_parent:
        insert_points = {0: (0, (0, n_words))}
    else:
        insert_points = {}

    def track(tree, i):
        label = tree.label()
        if len(tree) == 1 and not isinstance(tree[0], nltk.Tree):
            span2node[(i, i + 1)] = tree
            return i + 1, []
        j, spans = i, []
        span = (i, i + len(tree.leaves()))
        for k, child in enumerate(tree, 1):
            if (not insert_to_lowest_parent):
                insert_points[j] = (k - 1, span)
            j, s = track(child, j)
            if insert_to_lowest_parent and j not in insert_points:
                insert_points[j] = (k, span)
            spans += s
        if insert_to_lowest_parent:
            # remove right
            del insert_points[j]
        elif j == n_words and j not in insert_points:
            insert_points[j] = (len(tree), span)
        if j > i:
            spans = spans + [(i, j, label)]
            span2node[(i, j)] = tree
        return j, spans

    spans = track(tree, 0)[1]
    if insert_to_lowest_parent:
        insert_points[n_words] = (len(tree), (0, n_words))
    return spans, span2node, insert_points


def check_conflict(spans, cand):
    lj, rj = cand
    for span in spans:
        li, ri = span
        if (li < lj < ri or li < rj < ri) and (li - lj) * (ri - rj) > 0:
            return True
    return False


def isterminal(node):
    return len(node) == 1 and not isinstance(node[0], nltk.Tree)


def clear_tree(tree, remove_tier=False, uncollapse_unary=True):
    tree = tree.copy(True)

    def track(tree):
        if not isinstance(tree, nltk.Tree):
            return
        if remove_tier:
            tree.set_label(tree.label().split(":")[0])
        for child in tree:
            track(child)
        # remove DELETED nodes and subtrees having zero child
        is_unary = len(tree) == 1
        tree[:] = [
            child for child in tree if (not isinstance(child, nltk.Tree) or (
                (not child.label().startswith('DEL')) and len(child) > 0))
        ]
        # remove the collapse unary introduced by delete
        if not is_unary and len(tree) == 1:
            child_label = tree[0].label()
            tree[:] = tree[0]
            tree.set_label(child_label)
        # uncollapse unary
        if len(tree) > 0:
            sublabels = tree.label().split('~')
            if len(sublabels) > 1:
                tree.set_label(sublabels[-1])
                for sublabel in reversed(sublabels[:-1]):
                    tree[:] = [tree.copy(True)]
                    tree.set_label(sublabel)

    track(tree)

    return tree


def append_error(node, error):
    if node.label() == "CORRECT":
        node.set_label(error)
    elif "MISS-L" in node.label() and "MISS-L" in error:
        pass
    else:
        node.set_label(node.label() + "::" + error)


def convert_tree(tree, actions, incremantal=True, scheme_zhang=False):
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
        error_type = action.type[0]
        error_tier = action.type.split(":")[1] if len(
            action.type) > 1 else "OTHER"

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
    seq = seq.split("\t")
    return [(char, "unk", pinyin(char, style=Style.NORMAL, heteronym=True)[0])
            for char in seq]


@lru_cache(maxsize=512)
def tokenize(seq):
    return tokenizer.basic_tokenizer.tokenize(seq.strip())


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
    if len(target) < 2 or len(source) < 2:
        return tree
    if source == target:
        return tree

    source = annotator_parse_cached("\t".join(source))
    target = annotator_parse_cached("\t".join(target))

    try:
        edits = annotator(source, target)
    except TimeoutError:
        return tree
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
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                                  use_fast=False,
                                                  local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                                  use_fast=False,
                                                  local_files_only=False)

    action_counter = Counter()

    annotator = Annotator.create_default("char", "all")

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

    # for inputs in tqdm(zip(source_lines, target_lines)):
    #     converted_tree = process_lines(inputs)
    #     if converted_tree is not None:
    #         output.write(converted_tree.pformat(1000000000))
    #         output.write("\n")

    output.close()
