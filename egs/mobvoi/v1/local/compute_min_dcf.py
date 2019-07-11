#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2019  Yiming Wang
# Apache 2.0

from __future__ import print_function
from operator import itemgetter
import sys, argparse, os
import numpy as np

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
 

def GetArgs():
    parser = argparse.ArgumentParser(description="Compute the minimum "
        "detection cost function along with the threshold at which it occurs. "
        "Usage: sid/compute_min_dcf.py [options...] <trials-file> "
        "<scores-file> "
        "E.g., sid/compute_min_dcf.py --wake-word '嗨小问' "
        "data/test/text exp/tdnn_1a/decode_test/scoring/score.txt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--wake-word', type=str, dest = "wake_word", default = '嗨小问', help="wake word")
    parser.add_argument('--duration', type=float, dest='duration', default=0.0)
    parser.add_argument("trials_filename",
        help="Input trials file, with columns of the form "
        "<utt-id> <wake-word> or <other-word>")
    parser.add_argument("scores_filename", type=str, nargs='+',
        help="List of input scores file (larger means more confidence for wake word), with columns of the form "
        "<utt-id> <score>")
    sys.stderr.write(' '.join(sys.argv) + "\n")
    args = parser.parse_args()
    args = CheckArgs(args)
    return args

def CheckArgs(args):
    if args.duration < 0.0:
        raise Exception("""duration must be >= 0.0""")
    if (args.scores_filename is not None and len(args.scores_filename) > 6):
        raise Exception(
            """max 6 scores filenames can be specified.
            If you want to compare with more, you would have to
            carefully tune the plot_colors variable which specified colors used
            for plotting.""")
    return args

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly rejected scores
      # less than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      fps = [fprs_norm - x for x in fprs]
      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, fps, thresholds

def PlotRoc(fnrs_list, fprs_list, color_val_list, name_list, savedir):
    assert len(fnrs_list) == len(fprs_list) and \
        len(fnrs_list) == len(color_val_list) and len(fnrs_list) == len(name_list)
    fig = plt.figure()
    roc_plots = []
    for i in range(len(fnrs_list)):
        fnrs = fnrs_list[i]
        fprs = fprs_list[i]
        color_val = color_val_list[i]
        name = name_list[i]
        roc_plot_handle, = plt.plot([fpr * 100 for fpr in fprs],
            [fnr * 100 for fnr in fnrs], color=color_val,
            linestyle="--", label="{}".format(name)
        )
        roc_plots.append(roc_plot_handle)

    plt.xlabel('False Alarms (%)')
    plt.ylabel('False Rejects (%)')
    plt.xlim((0, 20))
    plt.ylim((0, 20))
    lgd = plt.legend(handles=roc_plots, loc='lower center',
        bbox_to_anchor=(0.5, -0.2 + len(fnrs_list) * -0.1),
        ncol=1, borderaxespad=0.)
    plt.grid(True)
    fig.suptitle("ROC curve")
    figfile_name = os.path.join(savedir, 'roc.pdf')
    plt.savefig(figfile_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    print("Saved ROC curves as " + figfile_name)

def PlotRoc2(fnrs_list, fps_list, color_val_list, name_list, duration, savedir):
    assert len(fnrs_list) == len(fps_list) and \
        len(fnrs_list) == len(color_val_list) and len(fnrs_list) == len(name_list)
    fig = plt.figure()
    roc_plots = []
    for i in range(len(fnrs_list)):
        fnrs = fnrs_list[i]
        fps = fps_list[i]
        color_val = color_val_list[i]
        name = name_list[i]
        roc_plot_handle, = plt.plot([fp * 3600 / duration for fp in fps],
            [fnr * 100 for fnr in fnrs], color=color_val,
            linestyle="--", label="{}".format(name)
        )
        roc_plots.append(roc_plot_handle)

    plt.xlabel('False Alarm per Hour')
    plt.ylabel('False Rejects (%)')
    plt.xlim((0, 5))
    plt.xticks(np.arange(0, 5, step=0.5))
    plt.ylim((0, 2))
    plt.yticks(np.arange(0, 2, step=0.1))
    lgd = plt.legend(handles=roc_plots, loc='lower center',
        bbox_to_anchor=(0.5, -0.2 + len(fnrs_list) * -0.1),
        ncol=1, borderaxespad=0.)
    plt.grid(True)
    fig.suptitle("ROC curve")
    figfile_name = os.path.join(savedir, 'roc2.pdf')
    plt.savefig(figfile_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    print("Saved ROC2 curves as " + figfile_name)

def main():
    args = GetArgs()
    g_plot_colors = ['red', 'blue', 'green', 'black', 'magenta', 'yellow']
    trials_file = open(args.trials_filename, 'r', encoding='utf-8').readlines()

    trials = {}
    for line in trials_file:
        if len(line.rstrip().split()) == 2:
            utt_id, target = line.rstrip().split()
        else:
            assert len(line.rstrip().split()) == 1
            utt_id = line.rstrip().split()[0]
            target = ""
        trials[utt_id] = target

    fnrs_list, fprs_list, fps_list, color_val_list, name_list = [], [], [], [], []
    savedir = os.path.dirname(args.scores_filename[0])
    for index, path in enumerate(args.scores_filename):
        scores = []
        labels = []
        scores_file = open(path, 'r', encoding='utf-8').readlines()
        for line in scores_file:
            utt_id, score = line.rstrip().split()
            if utt_id in trials:
                scores.append(float(score))
                if trials[utt_id] == args.wake_word:
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                raise Exception("Missing entry for " + utt_id
                    + " " + path)

        fnrs, fprs, fps, thresholds = ComputeErrorRates(scores, labels)
        fnrs_list.append(fnrs)
        fprs_list.append(fprs)
        fps_list.append(fps)
        color_val_list = g_plot_colors[:len(args.scores_filename)]
        name_list.append(os.path.dirname(path))

    PlotRoc(fnrs_list, fprs_list, color_val_list, name_list, savedir)
    if args.duration > 0.0:
        PlotRoc2(fnrs_list, fps_list, color_val_list, name_list, args.duration,
            savedir)

if __name__ == "__main__":
  main()
