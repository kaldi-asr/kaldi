#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Johns Hopkins University (author: Yiming Wang)
# Apache 2.0

""" This script prepares the speech commands data into kaldi format.
"""


import argparse
import os
import io
import sys

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        """This script requires matplotlib.
        Please install it to generate plots.
        If you are on a cluster where you do not have admin rights you could
        try using virtualenv.""")
    
def main():
    parser = argparse.ArgumentParser(description="""Computes metrics for evalutuon.""")
    parser.add_argument('cost_file', type=str, help='cost file')
    parser.add_argument('text_file', type=str, help='text file')
    parser.add_argument('--wake-word', type=str, default='嗨小问',
                        help='wake word')

    args = parser.parse_args()

    cost = {}
    with open(args.cost_file) as f:
        for line in f:
            assert line.strip().split()[0] not in cost
            cost[line.strip().split()[0]] = float(line.strip().split()[1])

    text = []
    with open(args.text_file) as f:
        for line in f:
            text.append([line.strip().split()[0], line.strip().split()[1]])

    score = [-cost[entry[0]] for entry in text]

    colors = ['red' if args.wake_word in entry[1] else 'blue' for entry in text]

    fig = plt.figure()
    plt.scatter(score, range(1, len(score)+1), s=3, c=colors, alpha=0.5)
    plt.axvline(x=0.0)
    plt.xlim(-1000, 1000)
    plt.xlabel('Score')
    plt.ylabel('Index')
    fig.suptitle("Scatter")
    figfile_name = os.path.join(os.path.dirname(args.cost_file), 'scatter.pdf')
    plt.savefig(figfile_name, bbox_inches='tight')
    print("Saved scatter plot as " + figfile_name)

if __name__ == "__main__":
    main()
