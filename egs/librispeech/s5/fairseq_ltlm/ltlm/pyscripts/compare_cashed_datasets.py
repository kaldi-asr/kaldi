# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import argparse
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

from ltlm.datasets import LatsDataSet
from ltlm.Tokenizer import WordTokenizer
from ltlm.pyutils.logging_utils import setup_logger

logger = logging.getLogger(__name__)


def plot_hist(x, save_to, bins=20, title='<unknown>'):
    plt.figure(None, (15,10))
    plt.hist(x, bins=bins, alpha=0.75)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Number')
    plt.savefig(save_to)


if __name__ == "__main__":
    setup_logger(stream=sys.stderr)
    parser = argparse.ArgumentParser()
    WordTokenizer.add_args(parser)
    LatsDataSet.add_args(parser, prefix='first_', add_scale_opts=False)
    LatsDataSet.add_args(parser, prefix='second_', add_scale_opts=False)
    parser.add_argument('--hist_dir', type=str, default=None, help="Directory for saving histograms")
    parser.add_argument('--hist_bins', type=int, default=50, help="Histograms number of beam")
    parser.add_argument('--round', type=int, default=4, help="Report float round")
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compare_clipped', action='store_true', help='Compare only clipped lattices')

    args = parser.parse_args()
    tokenizer = WordTokenizer.build_from_args(args)
    d1 = LatsDataSet.build_from_args(args, tokenizer, prefix='first_')
    d2 = LatsDataSet.build_from_args(args, tokenizer, prefix='second_')

    if args.hist_dir is not None:
        os.makedirs(args.hist_dir, exist_ok=True)

    report = d1.compare(d2, progress_bar=(not args.no_progress_bar), normalize=False, compare_clipped=args.compare_clipped)
    for k, v in report.items():
        if k == 'utts' or len(v) == 0 :
            continue
        if args.hist_dir is not None:
            plot_hist(v, os.path.join(args.hist_dir, k.replace(' ', '_') + ".png"), bins=args.hist_bins, title=k)
        v_np = np.fromiter(map(abs,v), dtype=np.float)
        min_i = np.argmin(v_np)
        max_i = np.argmax(v_np)
        print(f"{k} = {round(sum(v)/len(v), args.round)}. abs min = {v[min_i]} ({report['utts'][min_i]}). abs max = {v[max_i]} ({report['utts'][max_i]}).")
