#!/usr/bin/python3
# This script visualize RTTM file with matplotlib
# "rttm: <type> <file-id> <channel-id> <begin-time> <duration> <NA> <NA> <speaker> <conf>"

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as pl
from matplotlib import collections  as mc
import sys


def main():
    # rttm_filename = sys.argv[1]
    # uttname = sys.argv[2]
    rttm_filename = "/Users/ashisharora/Desktop/opensat_oct/check2"
    uttname = "0001_20190110_200708_part1_AB_xxxxx"
    with open(rttm_filename, 'r') as fh:
        content = fh.readlines()

    start_time_list = []
    end_time_list = []
    speaker_list = []
    spkname_dict = {}
    num_spk = 0
    num_seg = 0
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        if line_split[1] != uttname:
            continue
        num_seg += 1
        start_time = float(line_split[3])
        end_time = start_time + float(line_split[4])
        speaker = line_split[7]
        start_time_list.append(start_time)
        end_time_list.append(end_time)
        if speaker not in spkname_dict:
            num_spk += 1
            spkname_dict[speaker] = num_spk
        speaker_list.append(spkname_dict[speaker])
    print("Utterance {}: {} speakers {} segments".format(uttname, num_spk, num_seg))

    c = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    seg_list = []
    color_list = []
    for i in range(num_seg):
        seg_list.append([(start_time_list[i], speaker_list[i]), (end_time_list[i], speaker_list[i])])
        color_list.append(c[speaker_list[i] % 10])
    lc = mc.LineCollection(seg_list, colors=color_list, linewidths=2)
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    fig.savefig("tmp.png")
    return 0

if __name__ == "__main__":
    main()
