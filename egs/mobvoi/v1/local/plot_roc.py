#!/usr/bin/env python

# Copyright 2018 Johns Hopkins University (author: Yiming Wang)
# Apache 2.0

""" This script prepares the speech commands data into kaldi format.
"""


import argparse
import os
import io
import sys
import codecs
import re

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
    parser.add_argument('comparison_path', type=str, nargs='+',
                        help='paths to result file')

    args = parser.parse_args()
    if (args.comparison_path is not None and len(args.comparison_path) > 6):
        raise Exception(
            """max 6 comparison paths can be specified.
            If you want to compare with more comparison_path, you would have to
            carefully tune the plot_colors variable which specified colors used
            for plotting.""")

    g_plot_colors = ['red', 'blue', 'green', 'black', 'magenta', 'yellow', 'cyan']

    pattern = r"precision: (\d+\.\d*)  recall: (\d+\.\d*)  FPR: (\d+\.\d*)  total: \d+"
    prog = re.compile(pattern)
    
    fig = plt.figure()
    pr_plots = []
    roc_plots = []
    for index, path in enumerate(args.comparison_path):
        if index == 0:
            savedir = os.path.dirname(path)

        color_val = g_plot_colors[index]
        with open(path, 'r') as f:
            lines = f.readlines()

        precision = []
        recall = []
        FPR = []
        for line in lines:
            m = prog.match(line)
            if m:
                precision.append(m.group(1))
                recall.append(m.group(2))
                FPR.append(m.group(3))

        sorted_index = sorted(range(len(recall)),key=recall.__getitem__)
        precision = [precision[i] for i in sorted_index]
        recall = [recall[i] for i in sorted_index]
        FPR = [FPR[i] for i in sorted_index]

        color_val = g_plot_colors[index]
        pr_plot_handle, = plt.plot(recall, precision, color=color_val,
            linestyle="-", label="{}".format(os.path.dirname(path))
        )
        pr_plots.append(pr_plot_handle)
        #roc_plot_handle, = plt.plot(FPR, recall, color=color_val,
        #    linestyle="--", label="{}".format(os.path.dirname(path))
        #)
        #roc_plots.append(roc_plot_handle)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    lgd = plt.legend(handles=pr_plots, loc='lower center',
            bbox_to_anchor=(0.5, -0.2 + len(args.comparison_path) * -0.1),
            ncol=1, borderaxespad=0.)
    plt.grid(True)
    fig.suptitle("Precision-Recall curve")
    figfile_name = os.path.join(savedir, 'pr.pdf')
    plt.savefig(figfile_name, bbox_extra_artists=(lgd,), bbox_inches='tight')

    print("Saved PR curves as " + \
        os.path.dirname(figfile_name))

if __name__ == "__main__":
    main()
