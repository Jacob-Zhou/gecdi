# -*- coding: utf-8 -*-

import argparse

from gec.parser import Seq2seqIntervenedParser
from supar.cmds.cmd import init


def main():
    parser = argparse.ArgumentParser(description='Create Seq2Seq GEC Parser.')
    parser.set_defaults(Parser=Seq2seqIntervenedParser)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # predict
    subparser = subparsers.add_parser(
        'predict', help='Use a trained parser to make predictions.')
    parser.add_argument('--gec-path', help='whether to evaluate tgt')
    parser.add_argument('--ged-path', help='whether to evaluate tgt')
    parser.add_argument('--language', default=None, help='language')
    parser.add_argument('--lm-path',
                        default="gpt2",
                        help='whether to evaluate tgt')
    parser.add_argument('--lm-alpha',
                        default=0.0,
                        type=float,
                        help='whether to evaluate tgt')
    parser.add_argument('--lm-beta',
                        default=0.0,
                        type=float,
                        help='whether to evaluate tgt')
    parser.add_argument('--ged-topk',
                        default=24,
                        type=int,
                        help='whether to evaluate tgt')
    parser.add_argument('--ged-alpha',
                        default=0.5,
                        type=float,
                        help='whether to evaluate tgt')
    parser.add_argument('--ged-beta',
                        default=0.0,
                        type=float,
                        help='whether to evaluate tgt')
    subparser.add_argument('--buckets',
                           default=8,
                           type=int,
                           help='max num of buckets to use')
    subparser.add_argument('--data',
                           default='data/conll14.test',
                           help='path to dataset')
    subparser.add_argument('--pred',
                           default='pred.txt',
                           help='path to predicted result')
    subparser.add_argument('--prob',
                           action='store_true',
                           help='whether to output probs')
    init(parser)


if __name__ == "__main__":
    main()
