import readkeys
import os
from collections import defaultdict
from functools import lru_cache, partial
import re
import argparse
import errant
from tqdm import tqdm

line_raw = {}

LINE_WIDTH = 180
VLINE = "-" * LINE_WIDTH

display_buffer = {
    "key_tips":
    "<A>: BACKWARD  <D>: FOREWARD  <Space>: FAST-FOREWARD  <q>: EXIT",
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
    display_buffer["lines"].extend(
        warp_line(
            f"{'src:':<15s}{'Gi':^3s}{'TP':^5s}{'FP':^5s}{'FN':^5s}{'SentF0.5':^8s}{'F0.5':^8s} ",
            line['src'], iswarp))
    display_buffer["lines"].append(VLINE)
    for i, (g, e) in enumerate(line['golds']):
        formatted_line = format_edits(g, e)
        display_buffer["lines"].extend(
            warp_line(f"gold-{i:<44d} ", formatted_line, iswarp))
    display_buffer["lines"].append(VLINE)
    ref_tp, ref_fp, ref_fn, ref_sf, ref_f, ref_gold = line['preds'][0][2]
    for n, p, m, e, *_ in line['preds']:
        best_tp, best_fp, best_fn, best_sf, best_f, best_gold = m
        better_prefix = f"\033[1;32m"
        worse_prefix = f"\033[1;31m"
        reset_prefix = "\033[0m"
        display_line = f"{n:<15s}"
        display_line += f"{best_gold:^3d}"
        display_line += f"{better_prefix if best_tp > ref_tp else worse_prefix}{reset_prefix if best_tp == ref_tp else ''}{best_tp:^5d}\033[0m"
        display_line += f"{better_prefix if best_fp < ref_fp else worse_prefix}{reset_prefix if best_fp == ref_fp else ''}{best_fp:^5d}\033[0m"
        display_line += f"{better_prefix if best_fn < ref_fn else worse_prefix}{reset_prefix if best_fn == ref_fn else ''}{best_fn:^5d}\033[0m"
        display_line += f"{better_prefix if best_sf > ref_sf else worse_prefix}{reset_prefix if best_sf == ref_sf else ''}{best_sf:^8.2%}\033[0m"
        display_line += f"{better_prefix if best_f > ref_f else worse_prefix}{reset_prefix if best_f == ref_f else ''}{best_f:^8.2%}\033[0m "
        formatted_line = format_edits(p, e)
        display_buffer["lines"].extend(
            warp_line(display_line, formatted_line, iswarp))
    display_buffer["lines"].append(VLINE)


def warp_line(prefix, line, iswarp):
    if iswarp:
        buff = []
        offset = 0
        while offset < len(line):
            buff.append(prefix + line[offset:offset + LINE_WIDTH - 43])
            prefix = " " * 43
            offset += LINE_WIDTH - 43
        return buff
    else:
        return [prefix + f"{line}"]


def print_buffer():
    os.system("clear")
    print(display_buffer["key_tips"])
    print(VLINE)
    print(
        f"{display_buffer['line_idx']:<8d}{display_buffer['cur_action']:<15}WARP:{'ON' if display_buffer['iswarp'] else 'OFF'}"
    )
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
        if isinstance(edit, tuple):
            edit, iscorrect = edit
        else:
            iscorrect = True
        start_pos, end_pos = edit.c_start, edit.c_end
        error_type, _ = edit.type.split(':', 1)
        refined_start_pos = start_pos + sum(
            [w for i, w in inserted if i <= start_pos])
        refined_end_pos = end_pos + sum(
            [w for i, w in inserted if i <= end_pos])
        if error_type == 'U' and start_pos != end_pos:
            error_type = 'R'
        if error_type == 'U':
            assert start_pos == end_pos, f"{edit.to_m2()}, {line}"
            inserted_tokens = [tok.text for tok in edit.o_toks]
            tokens[refined_start_pos:refined_start_pos] = decorate_tokens(
                inserted_tokens, error_type, iscorrect)
            inserted.append((start_pos, len(inserted_tokens)))
        else:
            tokens[refined_start_pos:refined_end_pos] = decorate_tokens(
                tokens[refined_start_pos:refined_end_pos], error_type,
                iscorrect)
    return " ".join(tokens)


@lru_cache(maxsize=10000)
def annotator_parse_cached(seq):
    return annotator.parse(seq)


@lru_cache(maxsize=10000)
def annotate_cached(source, target):
    if source == target:
        return []
    source = annotator_parse_cached(source)
    target = annotator_parse_cached(target)
    return annotator.annotate(source, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix tokenization issues')
    parser.add_argument('--gold', help='seed for generating random numbers')
    parser.add_argument('pred_file', nargs='+', help='input file')
    args, unknown = parser.parse_known_args()

    beta = 0.5

    def open_fn(file_name):
        return open(file_name.split(":")[-1], mode='r')

    sent_idx = 0
    sentence = []
    annotator = errant.load("en")

    for line in tqdm(open_fn(args.gold)):
        line = line.strip()
        if len(line) == 0:
            if len(sentence) == 2 and sentence[-1].strip() == "T":
                sentence[-1] = sentence[0]
            line_raw[sent_idx] = {
                "src": (sentence[0] + '\t').split('\t')[1],
                "golds": [(s + '\t').split('\t')[1] for s in sentence[1:]]
            }
            line_raw[sent_idx]['golds'] = [
                (gold, [] if gold == line_raw[sent_idx]['src'] else
                 annotate_cached(line_raw[sent_idx]["src"], gold))
                for gold in line_raw[sent_idx]['golds']
            ]
            sentence = []
            sent_idx += 1
        else:
            sentence.append(line)
    for sent_idx, pred_file_lines in tqdm(
            enumerate(zip(*map(open_fn, args.pred_file)))):
        line_raw[sent_idx].update({"preds": []})
        for i, (name, line) in enumerate(zip(args.pred_file, pred_file_lines)):
            name = name.split(":")[0]
            edit = annotate_cached(line_raw[sent_idx]["src"], line)
            ref_edit = annotate_cached(line_raw[sent_idx]["preds"][0][1],
                                       line) if i > 0 else []
            best_tp, best_fp, best_fn, best_sf, best_f, best_gold = 0, 0, 0, -1, -1, 0
            best_edit = []
            for gold_idx, (gold, gold_edit) in enumerate(
                    line_raw[sent_idx]["golds"]):
                tp = 0
                new_edit = []
                gold_action = {(e.o_start, e.o_end,
                                " ".join([tok.text for tok in e.c_toks]))
                               for e in gold_edit}
                for e in edit:
                    action = (e.o_start, e.o_end,
                              " ".join([tok.text for tok in e.c_toks]))
                    iscorrect = False
                    if action in gold_action:
                        tp += 1
                        iscorrect = True
                    new_edit.append((e, iscorrect))
                fp = len(edit) - tp
                fn = len(gold_edit) - tp

                p = float(tp) / (tp + fp) if fp else 1.0
                r = float(tp) / (tp + fn) if fn else 1.0
                sf = float((1 + (beta**2)) * p * r) / ((
                    (beta**2) * p) + r) if p + r else 0.0

                gp = float(tp + 1) / (tp + fp + 1)
                gr = float(tp + 1) / (tp + fn + 1)
                f = float((1 + (beta**2)) * gp * gr) / (((beta**2) * gp) + gr)

                if (f > best_f) or \
                (f == best_f and tp > best_tp) or \
                (f == best_f and tp == best_tp and fp < best_fp) or \
                (f == best_f and tp == best_tp and fp == best_fp and fn < best_fn):
                    best_tp, best_fp, best_fn = tp, fp, fn
                    best_f, best_sf, best_gold = f, sf, gold_idx
                    best_edit = new_edit
            line_raw[sent_idx]["preds"].append(
                (name, line, (best_tp, best_fp, best_fn, best_sf, best_f,
                              best_gold), best_edit, ref_edit))

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
            while sent_idx + 1 < len(line_raw) and len(
                {line
                 for _, line, *_ in line_raw[sent_idx + 1]["preds"]}) == 1:
                sent_idx += 1
            else:
                sent_idx += 1
            if sent_idx >= len(line_raw):
                sent_idx = len(line_raw) - 1
        elif action == "FAST-BACKWARD":
            while sent_idx > 0 and len(
                {line
                 for _, line, *_ in line_raw[sent_idx - 1]["preds"]}) == 1:
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
