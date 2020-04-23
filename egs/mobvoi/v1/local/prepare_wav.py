#!/usr/bin/env python3

# Copyright 2018-2020  Yiming Wang
#           2018-2020  Daniel Povey
# Apache 2.0

""" This script prepares the Mobvoi data into kaldi format.
"""


import argparse
import os
import sys
import glob

def main():
    parser = argparse.ArgumentParser(description="""Generates {train|dev|eval}_wav.scp files.""")
    parser.add_argument('dir', type=str,
                        default='data',
                        help='path to the directory containing downloaded dataset')
    args = parser.parse_args()

    assert os.path.isdir(args.dir)
    with open(os.path.join(args.dir, "train", "text"), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        train_set = set([os.path.splitext(os.path.split(line.strip().split()[0])[1])[0] for line in lines])
        assert len(train_set) > 0
    with open(os.path.join(args.dir, "dev", "text"), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        dev_set = set([os.path.splitext(os.path.split(line.strip().split()[0])[1])[0] for line in lines])
        assert len(dev_set) > 0
    with open(os.path.join(args.dir, "eval", "text"), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        eval_set = set([os.path.splitext(os.path.split(line.strip().split()[0])[1])[0] for line in lines])
        assert len(eval_set) > 0
    assert len(train_set.intersection(dev_set)) == 0
    assert len(train_set.intersection(eval_set)) == 0
    assert len(dev_set.intersection(eval_set)) == 0

    train_wav_scp = open(os.path.join(args.dir, "train", "wav.scp"), 'w', encoding='utf-8')
    dev_wav_scp = open(os.path.join(args.dir, "dev", "wav.scp"), 'w', encoding='utf-8')
    eval_wav_scp = open(os.path.join(args.dir, "eval", "wav.scp"), 'w', encoding='utf-8')

    # Look through all the subfolders to find audio samples
    wav_files = {}
    search_path = os.path.join(args.dir, '**', '*.wav')
    for wav_path in glob.glob(search_path, recursive=True):
        _, basename = os.path.split(wav_path)
        utt_id = os.path.splitext(basename)[0]
        extended_wav_path = "sox " + os.path.abspath(wav_path) + " -t wav - |"
        if not utt_id in wav_files:
            wav_files[utt_id] = extended_wav_path
    for utt_id in train_set:
        train_wav_scp.write(utt_id + " " + wav_files[utt_id] + "\n")
    for utt_id in dev_set:
        dev_wav_scp.write(utt_id + " " + wav_files[utt_id] + "\n")
    for utt_id in eval_set:
        eval_wav_scp.write(utt_id + " " + wav_files[utt_id] + "\n")

    train_wav_scp.close()
    dev_wav_scp.close()
    eval_wav_scp.close()

if __name__ == "__main__":
    main()
