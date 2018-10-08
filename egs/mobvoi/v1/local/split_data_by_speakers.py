#!/usr/bin/env python

# Copyright 2018 Johns Hopkins University (author: Yiming Wang)
# Apache 2.0

""" This script prepares the speech commands data into kaldi format.
"""


import argparse
import os
import sys
import random

def main():
    parser = argparse.ArgumentParser(description="""Split out training set from text1 and text2 with proportion specified by --train-proportion.""")
    parser.add_argument('text1', type=str,
                        default='data/hixiaowen_text', help='text1')
    parser.add_argument('text2', type=str,
                        default='data/freetext_text', help='text2')
    parser.add_argument('outfile', type=str,
                        default='data/train/text', help='train text')
    parser.add_argument('--train-proportion', type=float, dest='train_proportion',
                        default=0.8, help='train proportion')

    args = parser.parse_args()

    spk_to_text1 = {}
    with open(args.text1, 'r', encoding='utf-8') as fin1:
        lines = fin1.readlines()
        counts1 = len(lines)
        for line in lines:
            utt_id, _ = line.split(' ', 1)
            spk_id, _ = utt_id.split('-', 1)
            if spk_id not in spk_to_text1:
                spk_to_text1[spk_id] = [line]
            else:
                spk_to_text1[spk_id].append(line)
    spk1 = set(spk_to_text1.keys())

    spk_to_text2 = {}
    with open(args.text2, 'r', encoding='utf-8') as fin2:
        lines = fin2.readlines()
        counts2 = len(lines)
        for line in lines:
            utt_id, _ = line.split(' ', 1)
            spk_id, _ = utt_id.split('-', 1)
            if spk_id not in spk_to_text2:
                spk_to_text2[spk_id] = [line]
            else:
                spk_to_text2[spk_id].append(line)
    spk2 = set(spk_to_text2.keys())

    overlap_spk = sorted(list(spk1.intersection(spk2)))
    random.seed(0)
    random.shuffle(overlap_spk)
    tot_counts = counts1 + counts2
    train_counts = int(args.train_proportion * tot_counts)
    counter = 0
    i = 0
    with open(args.outfile, 'w', encoding='utf-8') as fout: 
        for spk in overlap_spk:
            incr_counts = len(spk_to_text1[spk]) + len(spk_to_text2[spk])
            if counter >= train_counts:
                break
            else:
                counter += incr_counts
                for line in spk_to_text1[spk]:
                    fout.write(line)
                for line in spk_to_text2[spk]:
                    fout.write(line)


if __name__ == "__main__":
    main()
