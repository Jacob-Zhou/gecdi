# -*- coding: utf-8 -*-

import argparse
from tqdm import tqdm


def simplify_edits(edits):
    edit_dict = {}
    for edit in edits:
        edit = edit.split("|||")
        span = edit[0][2:].split()  # [2:] ignore the leading "A "
        start = int(span[0])
        end = int(span[1])
        cat = edit[1]
        cor = edit[2]
        id = edit[-1]
        # Save the useful info as a list
        proc_edit = [start, end, cat, cor]
        # Save the proc_edit inside the edit_dict using coder id
        if id in edit_dict.keys():
            edit_dict[id].append(proc_edit)
        else:
            edit_dict[id] = [proc_edit]
    return edit_dict


def get_cor_and_edits(orig, edits):
    # Copy orig; we will apply edits to it to make cor
    cor = orig.split()
    offset = 0
    # Sort the edits by offsets before processing them
    edits = sorted(edits, key=lambda e: (e[0], e[1]))
    # Loop through edits: [o_start, o_end, cat, cor_str]
    for edit in edits:
        o_start = edit[0]
        o_end = edit[1]
        cat = edit[2]
        cor_toks = edit[3].split()
        cor_toks = cor_toks if cor_toks != ["-NONE-"] else []
        # Apply the edits
        cor[o_start+offset:o_end+offset] = cor_toks
        # Get the cor token start and end offsets in cor
        c_start = o_start+offset
        c_end = c_start+len(cor_toks)
        # Keep track of how this affects orig edit offsets
        offset = offset-(o_end-o_start)+len(cor_toks)
    return " ".join(cor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split instances.')
    parser.add_argument('--m2-file', help='input file')
    parser.add_argument('--output-file', help='output file')

    args, unknown = parser.parse_known_args()
    m2_block = []

    output = open(args.output_file, "w")
    for i, line in enumerate(tqdm(open(args.m2_file, "r"))):
        line = line.strip()
        if line:
            m2_block.append(line)
        else:
            output.write(f"S\t{m2_block[0][2:]}\n")
            edit_dict = simplify_edits(m2_block[1:])
            if len(edit_dict) == 0:
                output.write(f"T\n")
            for id, raw_edits in sorted(edit_dict.items()):
                if raw_edits[0][2] == "noop":
                    output.write(f"T\t{m2_block[0][2:]}\n")
                    continue
                cor = get_cor_and_edits(m2_block[0][2:], raw_edits)
                output.write(f"T\t{cor}\n")
            output.write("\n")
            m2_block = []
    output.close()
