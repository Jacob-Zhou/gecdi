'''
file format:

S	I want to play with children and see their simle all day .
T	 I want to play with children and see their lives all day .
T	 I want to play with children and see their faces all day .
T	 I want to play with children and see their simle all day .
T	 I want to play with children and see their children all day .
T	 I want to play with children and see their smiles all day .
T	 I want to play with children and see their smile all day .
T	 I want to play with children and see their similes all day .
T	 I want to play with children and see them all day .
T	 I want to play with children and see their simile all day .
T	 I want to play with children and see their souls all day .
T	 I want to play with children and see their face all day .
T	 I want to play with children and see their skin all day .
'''

# Path: split.py
# Split the Ts into different files

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input file")
    args = parser.parse_args()
    source = ""
    tgt_id = 0
    tgt_file = []
    with open(args.input, "r") as f:
        for line in f:
            if line.startswith("S"):
                # Source, restore for later use
                source = line.strip()
                tgt_id = 0
            elif line.startswith("T"):
                # Target
                if len(tgt_file) < tgt_id + 1:
                    tgt_file.append(open(f"{args.input}.{tgt_id:0>3d}", "w"))
                target = line.strip()
                tgt_file[tgt_id].write(f"{source}\n{target}\n\n")
                tgt_id += 1
