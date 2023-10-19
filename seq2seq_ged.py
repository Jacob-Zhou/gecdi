# -*- coding: utf-8 -*-

import argparse

from gec.parser import Seq2SeqDetector
from supar.cmds.cmd import init


def main():
    parser = argparse.ArgumentParser(description='Create Seq2Seq GEC Parser.')
    parser.set_defaults(Parser=Seq2SeqDetector)
    parser.add_argument('--checkpoint-path', help='whether to evaluate tgt')
    parser.add_argument('--bin-path', help='whether to evaluate tgt')
    parser.add_argument('--eval-tgt', action='store_true', help='whether to evaluate tgt')
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--checkpoint', action='store_true', help='whether to load a checkpoint to restore training')
    subparser.add_argument('--gec-init', action='store_true', help='whether to load a checkpoint to restore training')
    subparser.add_argument('--error-schema', default='partial-last', help='path to train file')
    subparser.add_argument('--max-len', type=int, default=1024, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--chunk-size', default=1000, type=int, help='size of chunk')
    subparser.add_argument('--train', default='data/clang8.train', help='path to train file')
    subparser.add_argument('--dev', default='data/bea19.dev', help='path to dev file')
    subparser.add_argument('--test', default=None, help='path to test file')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/conll14.test', help='path to dataset')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/conll14.test', help='path to dataset')
    subparser.add_argument('--pred', default='pred.txt', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    init(parser)


if __name__ == "__main__":
    main()
