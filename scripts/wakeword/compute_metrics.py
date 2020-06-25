#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018-2020  Yiming Wang
# Apache 2.0

""" This script computes several metrics for wake word detection.
"""


import argparse
import os
import io
import sys
import codecs

def main():
    parser = argparse.ArgumentParser(description="""Computes metrics for evalutuon.""")
    parser.add_argument('ref', type=str,
                        default='ref.txt',
                        help='path to the reference')
    parser.add_argument('hyp', type=str,
                        default='hyp.txt',
                        help='path to the hypothesis')
    parser.add_argument('--wake-word', type=str, dest='wake_word', default='嗨小问',
                        help='wake word')
    parser.add_argument('--duration', type=float, dest='duration', default=0.0)
    args = parser.parse_args()

    f = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8') if args.ref == "-" else codecs.open(args.ref, 'r', encoding='utf-8')
    lines = f.readlines()
    ref = [line.strip().split(None, 1) if len(line.strip().split(None, 1)) == 2 else [line.strip().split(None, 1)[0], ''] for line in lines]
    f.close()

    f = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8') if args.hyp == "-" else codecs.open(args.hyp, 'r', encoding='utf-8')
    lines = f.readlines()
    hyp = {}
    for line in lines:
        hyp[line.strip().split(None, 1)[0]] = line.strip().split(None, 1)[1] if len(line.strip().split(None, 1)) == 2 else ""
    f.close()

    if len(ref) != len(hyp):
        print("The lengths of reference and hypothesis do not match. ref: {} vs hyp: {}.".format(len(ref), len(hyp)), file=sys.stderr)
    TP = TN = FP = FN = 0.0
    for i in range(len(ref)):
        if ref[i][0] not in hyp:
            print("reference {} does not exist in hypothesis.".format(ref[i][0]), file=sys.stderr)
            continue
        if ref[i][1] == args.wake_word:
            if args.wake_word in hyp[ref[i][0]]:
                TP += 1.
            else:
                FN += 1.
        else:
            if args.wake_word in hyp[ref[i][0]]:
                FP += 1.
            else:
                TN += 1.
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    false_positive_rate = FP / (FP + TN) if FP + TN > 0 else 0.0
    false_negative_rate = FN / (FN + TP) if FN + TP > 0 else 0.0
    false_alarms_per_hour = FP / (args.duration / 3600) if args.duration > 0.0 else 0.0

    print("precision: %.5f  recall: %.5f  FPR: %.5f  FNR: %.5f  FP per hour: %.5f  total: %d" % (precision, recall, false_positive_rate, false_negative_rate, false_alarms_per_hour, TP+TN+FP+FN))

if __name__ == "__main__":
    main()
