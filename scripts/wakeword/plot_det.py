#!/usr/bin/env python3

# Copyright 2018-2020  Yiming Wang
# Apache 2.0

""" This script plots the DET curves
"""


import argparse
import os
import io
import sys
import codecs
import re

try:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        """This script requires matplotlib.
        Please install it to generate plots.
        If you are on a cluster where you do not have admin rights you could
        try using virtualenv.""")
    

def main():
    parser = argparse.ArgumentParser(description="""Computes metrics for evalutuon.""")
    parser.add_argument("comparison_path", type=str, nargs="+",
                        help="paths to result file, and each line in the file should have the format specfied in pattern variable below.")

    args = parser.parse_args()
    if (args.comparison_path is not None and len(args.comparison_path) > 6):
        raise Exception(
            """max 6 comparison paths can be specified.
            If you want to compare with more comparison_path, you would have to
            carefully tune the plot_colors variable which specified colors used
            for plotting.""")

    g_plot_colors = ["red", "blue", "green", "black", "magenta", "yellow", "cyan"]

    pattern = r"precision: (\d+\.\d*)  recall: (\d+\.\d*)  FPR: (\d+\.\d*)  FNR: (\d+\.\d*)  FP per hour: (\d+\.\d*)  total: \d+"
    prog = re.compile(pattern)
    
    fig = plt.figure()
    det_plots = []
    for index, path in enumerate(args.comparison_path):
        if index == 0:
            savedir = os.path.dirname(path)

        color_val = g_plot_colors[index]
        with open(path, "r") as f:
            lines = f.readlines()

        precision = []
        recall = []
        FPR = []
        FNR = []
        FP_per_hour = []
        for line in lines:
            m = prog.match(line)
            if m:
                precision.append(float(m.group(1)))
                recall.append(float(m.group(2)))
                FPR.append(float(m.group(3)))
                FNR.append(float(m.group(4)))
                FP_per_hour.append(float(m.group(5)))

        sorted_index = sorted(range(len(FP_per_hour)), key=FP_per_hour.__getitem__)
        FPR = [FPR[i] * 100 for i in sorted_index]
        FNR = [FNR[i] * 100 for i in sorted_index]
        FP_per_hour = [float(FP_per_hour[i]) for i in sorted_index]

        color_val = g_plot_colors[index]
        plt.xlim(0.0, 2.0)
        plt.ylim(0.0, 2.0)
        det_plot_handle, = plt.plot(FP_per_hour, FNR, color=color_val,
            linestyle="--", label="{}".format(os.path.dirname(path)), linewidth=2.0,
        )
        det_plots.append(det_plot_handle)

    plt.xlabel("False Alarms per hour")
    plt.ylabel("False Rejection Rate (%)")
    lgd = plt.legend(handles=det_plots, loc="lower center",
            bbox_to_anchor=(0.5, -0.2 + len(args.comparison_path) * -0.1),
            ncol=1, borderaxespad=0.)
    plt.grid(True)
    fig.suptitle("DET curve")
    figfile_name = os.path.join(savedir, "det.pdf")
    plt.savefig(figfile_name, bbox_extra_artists=(lgd,), bbox_inches="tight")

    print("Saved DET curves as " + figfile_name)

if __name__ == "__main__":
    main()
