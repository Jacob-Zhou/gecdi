from Levenshtein import distance

'''
    S\t<source text>\n
    T\t<target text>\n\n
'''


# for dataset in ["data/clang8.train", "data/error_coded.train", "data/wi_locness.train", "data/cgec/lang8.train", "data/cgec/hsk.train"]:
for dataset in ["data/cgec/lang8.train", "data/cgec/hsk.train", "data/cgec/lang8_5xhsk.yuezhang.train"]:
    source_len = []
    n_edit = []
    with open(dataset) as f:
        lines = f.readlines()
        source = lines[0::3]
        target = lines[1::3]
        # import pdb
        # pdb.set_trace()
        for s, t in zip(source, target):
            s = s.strip().split("\t")[1]
            t = t.strip().split("\t")[1]
            # source_len.append(len(s.split()))
            # n_edit.append(distance(s.split(), t.split()))
            source_len.append(len(s))
            n_edit.append(distance(s, t))

    # Output average statistics
    print("Dataset: ", dataset)
    # sentence level
    print("Number of sentences: ", len(source_len))
    print("Number of correct sentences: ", len([x for x in n_edit if x == 0]))
    print(f"Ratio of correct sentences: {len([x for x in n_edit if x == 0])/len(n_edit):.2%}")
    print("Average source length: ", sum(source_len)/len(source_len))
    print("Average number of edits: ", sum(n_edit)/len(n_edit))
    print("Average number of edits per token: ", sum(n_edit)/sum(source_len))
    print()
