# -*- coding: utf-8 -*-

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--gold-file', help='input file')
    parser.add_argument('--pred-file', help='input file')
    parser.add_argument('--output-file', help='output file')

    args, unknown = parser.parse_known_args()

    source_lines = []
    source_pred_lines = []
    target_lines = []
    predict_lines = []

    output_file = open(f"{args.output_file}", "w")

    with open(args.gold_file, "r") as gold_lines, open(args.pred_file, "r") as pred_lines:
        for line in gold_lines:
            line = line.strip()
            if len(line) == 0:
                while True:
                    try:
                        pred_line = next(pred_lines).strip()
                    except:
                        break
                    if len(pred_line) == 0:
                        break
                    else:
                        pred_line = pred_line + '\t'
                        type, pred_line = pred_line.split("\t", maxsplit=1)
                        pred_line = pred_line.strip()
                        if type == 'T':
                            predict_lines.append(pred_line)
                        else:
                            source_pred_lines.append(pred_line)
                assert len(source_lines) == len(source_pred_lines) == 1, f"{source_lines}\n{source_pred_lines}"
                assert source_lines[0] == source_pred_lines[0], f"{source_lines[0]}\n{source_pred_lines[0]}"
                source_line = source_lines[0]
                for pred_line in predict_lines:
                    output_file.write(f"P\t{pred_line}\n")
                output_file.write(f"S\t{source_line}\n")
                for target_line in target_lines:
                    output_file.write(f"T\t{target_line}\n")
                output_file.write(f"\n")
                source_lines = []
                target_lines = []
                predict_lines = []
                source_pred_lines = []
            else:
                line = line + '\t'
                type, line = line.split("\t", maxsplit=1)
                line = line.strip()
                if type == 'T':
                    target_lines.append(line)
                else:
                    source_lines.append(line)
    output_file.close()
