# -*- coding: utf-8 -*-

import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--input-file', help='input file')
    parser.add_argument('--index-file', help='input file')
    parser.add_argument('--output-file', help='output file')

    args, unknown = parser.parse_known_args()

    output_file = open(f"{args.output_file}", "w")

    discourse_index = 0
    source_discourse_buff = ""
    target_discourse_buff = ""
    source_sentence_buff = ""
    target_sentence_buff = ""
    with open(args.input_file, "r") as lines, open(args.index_file, "r") as index:
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                cur_index, _ = next(index).split('\t')
                if int(cur_index) == discourse_index:
                    source_discourse_buff += source_sentence_buff
                    target_discourse_buff += target_sentence_buff
                else:
                    discourse_index = int(cur_index)
                    output_file.write(f"S\t{source_discourse_buff}\nT\t{target_discourse_buff}\n\n")
                    source_discourse_buff = source_sentence_buff
                    target_discourse_buff = target_sentence_buff
                source_sentence_buff = ''
                target_sentence_buff = ''
            else:
                line += '\t'
                type, sent = line.split("\t", 1)
                if type == "T" and sent == '':
                    sent = source_sentence_buff
                if type == "S":
                    source_sentence_buff = sent.strip()
                else:
                    # only keep the best sentence
                    if target_sentence_buff == '':
                        target_sentence_buff = sent.strip()
        else:
            if (source_discourse_buff + target_discourse_buff) != '':
                output_file.write(f"S\t{source_discourse_buff}\nT\t{target_discourse_buff}\n")
        output_file.write("\n")

    output_file.close()

