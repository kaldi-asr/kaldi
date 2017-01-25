#! /usr/bin/env python

# Copyright 2016    Vimal Manohar
# Apache 2.0

from __future__ import print_function
import argparse, random

def GetArgs():
    parser = argparse.ArgumentParser(description="""This script converts a wav.scp
into noise-set-paramters that can be passed to steps/data/reverberate_data_dir.py.""")

    parser.add_argument("wav_scp", type=str,
                        help = "The input wav.scp")
    parser.add_argument("noise_list", type=str,
                        help = "File to write the output noise-set-parameters")

    args = parser.parse_args()

    return args

def Main():
    args = GetArgs()

    noise_list = open(args.noise_list, 'w')

    for line in open(args.wav_scp):
        parts = line.strip().split()

        print ('''--noise-id {reco} --noise-type point-source \
--bg-fg-type foreground "{wav}"'''.format(
                reco = parts[0],
                wav = " ".join(parts[1:])), file = noise_list)

    noise_list.close()

if __name__ == '__main__':
    Main()

