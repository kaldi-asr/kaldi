#! /usr/bin/env python

# Copyright 2016  Vimal Manohar
# Apache 2.0

"""This script makes a simple unigram FST for decoding for segmentation."""

from __future__ import print_function
import argparse, math

def get_args():
    parser = argparse.ArgumentParser("""Make a simple unigram FST for
decoding for segmentation purpose.""")

    parser.add_argument("--word2prior-map", type=str, required=True,
                        help = "A file with priors for different words")
    parser.add_argument("--end-probability", type=float, default=0.01,
                        help = "Ending probability")

    args = parser.get_args()

    return args


def read_map(map_file):
    out_map = {}
    sum_prob = 0
    for line in open(map_file):
        parts = line.strip().split()
        if len(parts) == 0:
            continue
        if len(parts) != 2:
            raise Exception("Invalid line {0} in {1}".format(line.strip(), map_file))

        if parts[0] in out_map:
            raise Exception("Duplicate entry of {0} in {1}".format(parts[0], map_file))

        prob = float(parts[1])
        out_map[parts[0]] = prob

        sum_prob += prob

    return (out_map, sum_prob)


def main():
    args = get_args()

    word2prior, sum_prob = read_map(args.word2prior_map)
    sum_prob += args.end_probability

    for w,p in word2prior.iteritems():
        print ("0 0 {word} {word} {log_p}".format(
            word = w, log_p = -math.log(p / sum_prob)))
    print ("0 {log_p}".format(
        word = w, log_p = -math.log(args.end_probability / sum_prob)))


if __name__ == '__main__':
    main()
