#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Johns Hopkins University (author: Yiming Wang)
# Apache 2.0

""" This script prepares the speech commands data into kaldi format.
"""


import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="""Wake word detection based on cost and threshold.""")
    parser.add_argument('cost_file', type=str,
                        help='cost file')
    parser.add_argument('--thres', type=float, default=0.0,
                        help='threshold of differnce between wake word/non wake word cost below '
                        'which wake word is considered detected.')
    parser.add_argument('--wake-word', type=str, default='嗨小问',
                        help='wake word')
    args = parser.parse_args()

    with open(args.cost_file, 'r') as f:
        for line in f:
            utt_id, wake_word_cost, non_wake_word_cost = line.strip().split()
            if float(non_wake_word_cost) - float(wake_word_cost) > args.thres:
                sys.stdout.buffer.write("{} {}\n".format(utt_id, args.wake_word).encode('utf-8'))
            else:
                sys.stdout.buffer.write("{} \n".format(utt_id).encode('utf-8'))

if __name__ == "__main__":
    main()
