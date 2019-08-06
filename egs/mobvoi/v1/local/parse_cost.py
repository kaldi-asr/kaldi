#!/usr/bin/env python

# Copyright 2018 Johns Hopkins University (author: Yiming Wang)
# Apache 2.0

""" This script prepares the speech commands data into kaldi format.
"""


import argparse
import os
import sys
from collections import OrderedDict
import glob

def main():
    parser = argparse.ArgumentParser(description="""Parse cost files.""")
    parser.add_argument('wake_word_cost_file', type=str,
                        help='wake word cost file')
    parser.add_argument('non_wake_word_cost_file', type=str,
                        help='non-wake word cost file')
 
    args = parser.parse_args()

    with open(args.wake_word_cost_file, 'r') as f1, \
        open(args.non_wake_word_cost_file, 'r') as f2:
        lines_wake_word, lines_non_wake_word = f1.readlines(), f2.readlines()

    one_entry = []
    wake_word_cost = OrderedDict()
    # each entry is one line of utt-id either followed by a cost line and an empty line:
    # <utt-id>
    # 0 <cost>
    #
    # or only followed by an empty line:
    # <utt-id>
    #
    for i, line in enumerate(lines_wake_word):
        if line.strip() == '':
            assert len(one_entry) == 1 or len(one_entry) == 2
            assert not one_entry[0] in wake_word_cost
            if len(one_entry) == 2:
                assert one_entry[1].strip().split()[0] == "0"
                wake_word_cost[one_entry[0]] = float(one_entry[1].strip().split()[1])
            else:
                wake_word_cost[one_entry[0]] = None
            one_entry = []
        else:
            one_entry.append(line.strip())

    one_entry = []
    non_wake_word_cost = OrderedDict()
    for i, line in enumerate(lines_non_wake_word):
        if line.strip() == '':
            assert len(one_entry) == 1 or len(one_entry) == 2
            assert not one_entry[0] in non_wake_word_cost
            if len(one_entry) == 2:
                assert one_entry[1].strip().split()[0] == "0"
                non_wake_word_cost[one_entry[0]] = float(one_entry[1].strip().split()[1])
            else:
                non_wake_word_cost[one_entry[0]] = None
            one_entry = []
        else:
            one_entry.append(line.strip())

    assert len(wake_word_cost) == len(non_wake_word_cost)
    cost = OrderedDict()
    # each entry in the output looks like:
    # <utt-id> <wake-word-cost> <non-wake-word-cost>
    for k, v in wake_word_cost.items():
        assert k in non_wake_word_cost
        v2 = non_wake_word_cost[k]
        assert not (v is None and v2 is None)
        cost[k] = [v if v is not None else 9999.0, v2 if v2 is not None else 9999.0]
        print("{} {:.3f} {:.3f}".format(k, cost[k][0], cost[k][1]))

if __name__ == "__main__":
    main()
