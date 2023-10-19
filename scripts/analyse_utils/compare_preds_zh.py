from __future__ import annotations

import readkeys
import os
from collections import defaultdict, namedtuple
from functools import lru_cache, partial
import re
import argparse
import errant
import torch
from tqdm import tqdm

##################################################
#   Copy from ChERRANT
##################################################

import Levenshtein
import numpy as np
from ltp import LTP
from typing import List, Tuple, Dict
from pypinyin import pinyin, Style, lazy_pinyin
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
    [
        "o_start",
        "o_end",
        "o_str",
        "c_start",
        "c_end",
        "c_str",
        "type"
    ],
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
            if v_sim + p_sim < threshold and not (
                    set(pinyin(src_char, style=Style.NORMAL, heteronym=True)[0]) & set(pinyin(tgt_char, style=Style.NORMAL, heteronym=True)[0])):
                return False
    return True

def read_cilin():
    """
    Cilin 詞林 is a thesaurus with semantic information
    """
    # TODO -- fix this path
    lines = open("scripts/generate_chn_treebank/data/cilin.txt", "r",
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
    with open("scripts/generate_chn_treebank/data/confusion_dict.txt", "r", encoding="utf-8") as f:
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
        
    def __call__(self,
                 src: List[Tuple],
                 tgt: List[Tuple]):
        cost_matrix, oper_matrix = self.align(src[::-1], tgt[::-1])
        align_seq = self.get_cheapest_align_seq(oper_matrix)
        o_l, c_l = len(src), len(tgt)
        align_seq = [[(op, o_l - o_e, o_l - o_s, c_l - c_e, c_l - c_s) for op, o_s, o_e, c_s, c_e in a[::-1]] for a in align_seq]
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
                if a[i] == b[j] or (set(pinyin_a) & set(pinyin_b)) or (b[j] in self.confusion_dict.keys() and a[i] in self.confusion_dict[b[j]]) or (a[i] in self.confusion_dict.keys() and b[j] in self.confusion_dict[a[i]]):
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
            char_cost = self._get_char_cost(a_seg[0], b_seg[0], a_seg[2], b_seg[2])
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
    def align(self,
              src: List[Tuple],
              tgt: List[Tuple]):
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
        oper_matrix = np.full(
            (len(src) + 1, len(tgt) + 1), "O", dtype=object
        )  # 操作矩阵
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
                    cost_matrix[i + 1][j + 1] = cost_matrix[i][j] # distance punish
                    oper_matrix[i + 1][j + 1] = ["M"]
                # Non-matches
                else:
                    del_cost = cost_matrix[i][j + 1] + self.deletion_cost  # 由删除动作得到的总cost
                    ins_cost = cost_matrix[i + 1][j] + self.insertion_cost  # 由插入动作得到的总cost
                    sub_cost = cost_matrix[i][j] + self.get_sub_cost(
                        src[i], tgt[j]
                    )  # 由替换动作得到的总cost
                    # Calculate transposition cost
                    # 计算移位操作的总cost
                    trans_cost = float("inf")
                    k = 1
                    while (
                            i - k >= 0
                            and j - k >= 0
                            and cost_matrix[i - k + 1][j - k + 1]
                            != cost_matrix[i - k][j - k]
                    ):
                        p1 = sorted([a[0] for a in src][i - k: i + 1])
                        p2 = sorted([b[0] for b in tgt][j - k: j + 1])
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
                                    oper_matrix[i + 1][j + 1] = ["T" + str(k + 1)]
                                else:
                                    oper_matrix[i + 1][j + 1].append("T" + str(k + 1))
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
                    self._dfs(i - 1, j - 1, align_seq_now + [(op, i - 1, i, j - 1, j)], oper_matrix, strategy)
                elif op == "D":
                    self._dfs(i - 1, j, align_seq_now + [(op, i - 1, i, j, j)], oper_matrix, strategy)
                elif op == "I":
                    self._dfs(i, j - 1, align_seq_now + [(op, i, i, j - 1, j)], oper_matrix, strategy)
                else:
                    k = int(op[1:])
                    self._dfs(i - k, j - k, align_seq_now + [(op, i - k, i, j - k, j)], oper_matrix, strategy)

    def get_cheapest_align_seq(self, oper_matrix):
        """
        回溯获得编辑距离最小的编辑序列
        """
        self.align_seqs = []
        i = oper_matrix.shape[0] - 1
        j = oper_matrix.shape[1] - 1
        if abs(i - j) > 10:
            self._dfs(i, j , [], oper_matrix, "first")
        else:
            self._dfs(i, j , [], oper_matrix, "all")
        final_align_seqs = [seq[::-1] for seq in self.align_seqs]
        return final_align_seqs

class Classifier:

    def __call__(self,
                 src,
                 tgt,
                 edits):
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
            src_span = " ".join(src_tokens[edit[1]: edit[2]])
            tgt_span = " ".join(tgt_tokens[edit[3]: edit[4]])
            # print(tgt_span)
            cor = None
            if error_type[0] == "T":
                # cor = Correction("W", tgt_span, (edit[1], edit[2]))
                cor = Correction(edit[1], edit[2], src_span, edit[3], edit[4], tgt_span, "R:WO")
            elif error_type[0] == "D":
                # cor = Correction("R", "-NONE-", (edit[1], edit[2]))
                cor = Correction(edit[1], edit[2], src_span, edit[3], edit[4], "", "U:OTHER")
            elif error_type[0] == "I":
                # cor = Correction("M", tgt_span, (edit[1], edit[2]))
                cor = Correction(edit[1], edit[2], src_span, edit[3], edit[4], tgt_span, "M:OTHER")
            elif error_type[0] == "S":
                if check_spell_error(src_span.replace(" ", ""), tgt_span.replace(" ", "")):
                    # cor = Correction("S:SPELL", tgt_span, (edit[1], edit[2]))
                    cor = Correction(edit[1], edit[2], src_span, edit[3], edit[4], tgt_span, "R:SPELL")
                else:
                    # cor = Correction("S", tgt_span, (edit[1], edit[2]))
                    cor = Correction(edit[1], edit[2], src_span, edit[3], edit[4], tgt_span, "R:OTHER")
            results.append(cor)
        return results

class Tokenizer:
    """
    分词器
    """

    def __init__(self,
                 granularity: str = "word",
                 segmented: bool = False,
                 ) -> None:
        """
        构造函数
        :param mode: 分词模式，可选级别：字级别（char）、词级别（word）
        """
        self.ltp = None 
        if granularity == "word":
            self.ltp = LTP()
            self.ltp.add_words(words=["[缺失成分]"])
        self.segmented = segmented
        self.granularity = granularity
        if self.granularity == "word":
            self.tokenizer = self.split_word
        elif self.granularity == "char":
            self.tokenizer = self.split_char
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        return "{:s}\nMode:{:s}\n}".format(str(self.__class__.__name__), self.mode)

    def __call__(self,
                 input_strings: List[str]
                 ) -> List:
        """
        分词函数
        :param input_strings: 需要分词的字符串列表
        :return: 分词后的结果列表，由元组组成，元组为(token,pos_tag,pinyin)的形式
        """
        if not self.segmented:
            input_strings = ["".join(s.split(" ")) for s in input_strings]
        results = self.tokenizer(input_strings)
        return results

    def split_char(self, input_strings: List[str]) -> List:
        """
        分字函数
        :param input_strings: 需要分字的字符串
        :return: 分字结果
        """
        results = []
        for input_string in input_strings:
            if not self.segmented:  # 如果没有被分字，就按照每个字符隔开（不考虑英文标点的特殊处理，也不考虑BPE），否则遵循原分字结果
                segment_string = " ".join([char for char in input_string])
            else:
                segment_string = input_string
                # print(segment_string)
            segment_string = segment_string.replace("[ 缺 失 成 分 ]", "[缺失成分]").split(" ")  # 缺失成分当成一个单独的token
            results.append([(char, "unk", pinyin(char, style=Style.NORMAL, heteronym=True)[0]) for char in segment_string])
        return results

    def split_word(self, input_strings: List[str]) -> List:
        """
        分词函数
        :param input_strings: 需要分词的字符串
        :return: 分词结果
        """
        if self.segmented:
            input_strings = [input_string.split(" ") for input_string in input_strings]
            outputs = self.ltp.pipeline(input_strings, tasks=["pos"])
            seg = input_strings
            pos = outputs.pos
        else:
            outputs = self.ltp.pipeline(input_strings, tasks=["cws", "pos"])
            seg = outputs.cws
            pos = outputs.pos
        result = []
        for s, p in zip(seg, pos):
            pinyin = [lazy_pinyin(word) for word in s]
            result.append(list(zip(s, p, pinyin)))
        return result

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
    def create_default(cls, granularity: str = "word", strategy: str = "first"):
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

    def __init__(self,
                 granularity: str = "word",
                 merge: bool = False):
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

    def __call__(self,
                 align_obj,
                 src: List,
                 tgt: List,
                 verbose: bool = False):
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
            lambda x: x[0][0] if x[0][0] in {"M", "T"}  else False,
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
                    w1 = "".join(src_tokens[edits[i][1]: edits[i][2]])
                    w2 = "".join(tgt_tokens[edits[i][3]: edits[i][4]])
                    w3 = "".join(src_tokens[edits[i + 2][1]: edits[i + 2][2]])
                    w4 = "".join(tgt_tokens[edits[i + 2][3]: edits[i + 2][4]])
                    if min([len(w1), len(w2), len(w3), len(w4)]) == 1:
                        if w1 == w4 and w2 == w3:
                            group = [edits[i], edits[i + 1], edits[i + 2]]
                            processed = self._merge_edits(group, "T" + str(edits[i+2][2] - edits[i][1]))
                            for seq in processed:
                                filtered_edits.append(seq)
                            i += 3
                        else:
                            filtered_edits.append(edits[i])
                            i += 1
                    else:
                        if Levenshtein.distance(w1, w4) <= 1 and Levenshtein.distance(w2, w3) <= 1:
                            group = [edits[i], edits[i + 1], edits[i + 2]]
                            processed = self._merge_edits(group, "T" + str(edits[i + 2][2] - edits[i][1]))
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
                elif (e1 == "D" and (e2 == "M" or e2.startswith("T")) and e3 == "I") or (e1 == "I" and (e2 == "M" or e2.startswith("T")) and e3 == "D"):
                    if e1 == "D":
                        delete_token = src_tokens[edits[i][1]: edits[i][2]]
                        insert_token = tgt_tokens[edits[i + 2][3]: edits[i + 2][4]]
                    else:
                        delete_token = src_tokens[edits[i + 2][1]: edits[i + 2][2]]
                        insert_token = tgt_tokens[edits[i][3]: edits[i][4]]
                    a, b = "".join(delete_token), "".join(insert_token)
                    if len(a) < len(b):
                        a, b = b, a
                    if a not in self.punctuation and b not in self.punctuation and len(a) - len(b) <= 1:
                        if len(b) == 1:
                            if a == b:
                                group = [edits[i], edits[i + 1], edits[i + 2]]
                                processed = self._merge_edits(group, "T" + str(edits[i+2][2] - edits[i][1]))
                                for seq in processed:
                                    filtered_edits.append(seq)
                                i += 3
                            else:
                                filtered_edits.append(edits[i])
                                i += 1
                        else:
                            if Levenshtein.distance(a, b) <= 1 or (len(a) == len(b) and self._check_revolve(a, b)):
                                group = [edits[i], edits[i + 1], edits[i + 2]]
                                processed = self._merge_edits(group, "T" + str(edits[i + 2][2] - edits[i][1]))
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
            span1 = "".join(src_tokens[edit[1] : edit[2]])
            span2 = "".join(tgt_tokens[edit[3] : edit[4]])

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
                            new_edit = ("I", new_start_1, new_end_1, new_start_2, new_end_2)
                        elif new_start_2 == new_end_2:
                            new_edit = ("D", new_start_1, new_end_1, new_start_2, new_end_2)
                        else:
                            new_edit = ("S", new_start_1, new_end_1, new_start_2, new_end_2)
                        second_filter.append(new_edit)
                else:
                    second_filter.append(edit)
        return second_filter

##################################################
#   Copy from ChERRANT
##################################################



line_raw = {}

LINE_WIDTH = 180
VLINE = "-"*LINE_WIDTH

display_buffer = {
        "key_tips": "<A>: BACKWARD  <D>: FOREWARD  <Space>: FAST-FOREWARD  <q>: EXIT",
        "cur_action": "",
        "line_idx": "",
        "lines": []
}

def get_actions():
    key = readkeys.getkey()
    return {
        "d": "BACKWARD",
        "\x1b[D": "BACKWARD",
        "a": "FOREWARD",
        "\x1b[C": "FOREWARD",
        " ": "FAST-FOREWARD",
        "\x1b[1;2C": "FAST-FOREWARD",
        "\x1b[1;2D": "FAST-BACKWARD",
        "x": "SWITCH-WARP",
        "\r": "ENTER",
        "q": "EXIT",
        "\x1b": "EXIT"
    }.get(key, key)

def update_buffer(action, sent_idx, iswarp):
    display_buffer["cur_action"] = action
    display_buffer["line_idx"] = sent_idx
    display_buffer["iswarp"] = iswarp
    line = line_raw[sent_idx]
    display_buffer["lines"] = []
    display_buffer["lines"].append(VLINE)
    display_buffer["lines"].extend(warp_line(f"{'src:':<8s}{'Gi':^3s}{'TP':^5s}{'FP':^5s}{'FN':^5s}{'SentF0.5':^8s}{'F0.5':^8s} ", line['src'], iswarp))
    display_buffer["lines"].append(VLINE)
    for i, (g, e) in enumerate(line['golds']):
        formatted_line = format_edits(g, e)
        display_buffer["lines"].extend(warp_line(f"gold-{i:<37d} ", formatted_line, iswarp))
    display_buffer["lines"].append(VLINE)
    ref_tp, ref_fp, ref_fn, ref_sf, ref_f, ref_gold = line['preds'][0][2]
    for n, p, m, e, *_ in line['preds']:
        best_tp, best_fp, best_fn, best_sf, best_f, best_gold = m
        better_prefix = f"\033[1;32m"
        worse_prefix = f"\033[1;31m"
        reset_prefix = "\033[0m"
        display_line = f"{n:<8s}"
        display_line += f"{best_gold:^3d}"
        display_line += f"{better_prefix if best_tp > ref_tp else worse_prefix}{reset_prefix if best_tp == ref_tp else ''}{best_tp:^5d}\033[0m"
        display_line += f"{better_prefix if best_fp < ref_fp else worse_prefix}{reset_prefix if best_fp == ref_fp else ''}{best_fp:^5d}\033[0m"
        display_line += f"{better_prefix if best_fn < ref_fn else worse_prefix}{reset_prefix if best_fn == ref_fn else ''}{best_fn:^5d}\033[0m"
        display_line += f"{better_prefix if best_sf > ref_sf else worse_prefix}{reset_prefix if best_sf == ref_sf else ''}{best_sf:^8.2%}\033[0m"
        display_line += f"{better_prefix if best_f > ref_f else worse_prefix}{reset_prefix if best_f == ref_f else ''}{best_f:^8.2%}\033[0m "
        formatted_line = format_edits(p, e)
        display_buffer["lines"].extend(warp_line(display_line, formatted_line, iswarp))
    display_buffer["lines"].append(VLINE)
    # display_buffer["lines"].extend([p[-2] for p in line['preds']])

def warp_line(prefix, line, iswarp):
    if iswarp:
        buff = []
        offset = 0
        while offset < len(line):
            buff.append(prefix + line[offset:offset+LINE_WIDTH-43])
            prefix = " "*43
            offset += LINE_WIDTH-43
        return buff
    else:
        return [prefix + f"{line}"]

def print_buffer():
    os.system("clear")
    print(display_buffer["key_tips"])
    print(VLINE)
    print(f"{display_buffer['line_idx']:<8d}{display_buffer['cur_action']:<15}WARP:{'ON' if display_buffer['iswarp'] else 'OFF'}")
    for line in display_buffer["lines"]:
        print(line)

def decorate_tokens(tokens, type, iscorrect):
    prefix = ""
    if type == "M":
        prefix = f"\033[{'1' if iscorrect else '4'};32m"
    elif type == "U":
        prefix = f"\033[{'1' if iscorrect else '3'};37;41m"
    else:
        prefix = f"\033[{'1' if iscorrect else '4'};34m"
    suffix = "\033[0m"
    return [f"{prefix}{token}{suffix}" for token in tokens]


def format_edits(line, edits):
    tokens = line.split()
    inserted = []
    for edit in edits:
        if isinstance(edit, tuple) and not isinstance(edit, Correction):
            edit, iscorrect = edit
        else:
            iscorrect = True
        start_pos, end_pos = edit.c_start, edit.c_end
        error_type, _ = edit.type.split(':', 1)
        refined_start_pos = start_pos + sum([w for i, w in inserted if i <= start_pos])
        refined_end_pos = end_pos + sum([w for i, w in inserted if i <= end_pos])
        if error_type == 'U' and start_pos != end_pos:
            error_type = 'R'
        if error_type == 'U':
            assert start_pos == end_pos, f"{edit.to_m2()}, {line}"
            inserted_tokens = edit.o_str.split(" ")
            tokens[refined_start_pos:refined_start_pos] = decorate_tokens(inserted_tokens, error_type, iscorrect)
            inserted.append((start_pos, len(inserted_tokens)))
        else:
            tokens[refined_start_pos:refined_end_pos] = decorate_tokens(tokens[refined_start_pos:refined_end_pos], error_type, iscorrect)
    return " ".join(tokens)

@lru_cache(maxsize=10000)
def annotator_parse_cached(seq):
    return tokenizer([seq])[0]

@lru_cache(maxsize=10000)
def annotate_cached(source, target):
    if source == target:
        return []
    source = annotator_parse_cached(source)
    target = annotator_parse_cached(target)
    return annotator(source, target)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix tokenization issues')
    parser.add_argument('--gold',
                        help='gold file')
    parser.add_argument('pred_file', nargs='+', help='input file')
    args, unknown = parser.parse_known_args()

    beta = 0.5

    open_fn = partial(open, mode='r')
    rePENALTY = re.compile(r"penalty-(\d+\.\d+)\.")
    sent_idx = 0
    sentence = []
    tokenizer = Tokenizer("word", "nlpcc18" in args.gold)
    annotator = Annotator.create_default("word", "all")

    for line in tqdm(open_fn(args.gold)):
        line = line.strip()
        if len(line) == 0:
            if len(sentence) == 2 and sentence[-1].strip() == "T":
                sentence[-1] = sentence[0]
            src = (sentence[0]+'\t').split('\t')[1]
            golds = [(s+'\t').split('\t')[1] for s in sentence[1:]]
            line_raw[sent_idx] = {
                "src": " ".join([t[0] for t in annotator_parse_cached(src)]),
                "golds": [" ".join([t[0] for t in annotator_parse_cached(gold)]) for gold in golds]
            }
            line_raw[sent_idx]['golds'] = [(gold, 
                                            [] if gold == line_raw[sent_idx]['src'] else annotate_cached(line_raw[sent_idx]["src"], gold)) 
                                           for gold in line_raw[sent_idx]['golds']]
            sentence = []
            sent_idx += 1
        else:
            sentence.append(line)
    for sent_idx, pred_file_lines in tqdm(enumerate(zip(*map(open_fn, args.pred_file)))):
        line_raw[sent_idx].update(
            {"preds": []}
        )
        for i, (name, line) in enumerate(zip(args.pred_file, pred_file_lines)):
            penalty = rePENALTY.search(name)
            line = line.strip()
            line = " ".join([t[0] for t in annotator_parse_cached(line)])

            if penalty:
                name = f"{float(penalty.group(1)):.2f}"
            else:
                name = 'w/o syn'
            edit = annotate_cached(line_raw[sent_idx]["src"], line)
            ref_edit = annotate_cached(line_raw[sent_idx]["preds"][0][1], line) if i > 0 else []
            best_tp, best_fp, best_fn, best_sf, best_f, best_gold = 0, 0, 0, -1, -1, 0
            best_edit = []
            for gold_idx, (gold, gold_edit) in enumerate(line_raw[sent_idx]["golds"]):
                tp = 0
                new_edit = []
                gold_action = {(e.o_start, e.o_end, e.c_str) for e in gold_edit}
                for e in edit:
                    action = (e.o_start, e.o_end, e.c_str)
                    iscorrect = False
                    if action in gold_action:
                        tp += 1
                        iscorrect = True
                    new_edit.append((e, iscorrect))
                fp = len(edit) - tp
                fn = len(gold_edit) - tp

                p = float(tp)/(tp+fp) if fp else 1.0
                r = float(tp)/(tp+fn) if fn else 1.0
                sf = float((1+(beta**2))*p*r)/(((beta**2)*p)+r) if p+r else 0.0

                gp = float(tp+1)/(tp+fp+1)
                gr = float(tp+1)/(tp+fn+1)
                f = float((1+(beta**2))*gp*gr)/(((beta**2)*gp)+gr)

                if (f > best_f) or \
                (f == best_f and tp > best_tp) or \
                (f == best_f and tp == best_tp and fp < best_fp) or \
                (f == best_f and tp == best_tp and fp == best_fp and fn < best_fn):
                    best_tp, best_fp, best_fn = tp, fp, fn
                    best_f, best_sf, best_gold = f, sf, gold_idx
                    best_edit = new_edit
            line_raw[sent_idx]["preds"].append((name, line, (best_tp, best_fp, best_fn, best_sf, best_f, best_gold), best_edit, ref_edit))

    sent_idx = 0
    iswarp = False
    update_buffer("NUL", sent_idx, iswarp)
    print_buffer()
    index_buffer = ''
    while (action := get_actions()) != 'EXIT':
        if action == "FOREWARD":
            if sent_idx + 1 < len(line_raw):
                sent_idx += 1
        elif action == "FAST-FOREWARD":
            while sent_idx + 1 < len(line_raw) and len({line for _, line, *_ in line_raw[sent_idx + 1]["preds"]}) == 1:
                sent_idx += 1
            else:
                sent_idx += 1
            if sent_idx >= len(line_raw):
                sent_idx = len(line_raw) - 1
        elif action == "FAST-BACKWARD":
            while sent_idx > 0 and len({line for _, line, *_ in line_raw[sent_idx - 1]["preds"]}) == 1:
                sent_idx -= 1
            else:
                sent_idx -= 1
            if sent_idx < 0:
                sent_idx = 0
        elif action == "BACKWARD":
            if sent_idx > 0:
                sent_idx -= 1
        elif str.isdigit(action):
            index_buffer += action
            action = ":" + index_buffer
        elif action == "ENTER":
            if len(index_buffer) > 0:
                target_sent_idx = int(index_buffer)
                index_buffer = ''
            if target_sent_idx in line_raw:
                sent_idx = target_sent_idx
                action = "JUMP"
            else:
                action = "INVAILD-INDEX"
        else:
            continue
        # elif action == "SWITCH-WARP":
            # iswarp = iswarp ^ True
        update_buffer(action, sent_idx, iswarp)
        print_buffer()
        
