#!/usr/bin/env python

from __future__ import print_function
import sys
import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description=""" """,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # feat options
    parser.add_argument("--bad-utt-info-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--max-wer", type=float, default=20)

    print(' '.join(sys.argv))
    args = parser.parse_args()

    return args

def GetSortedWers(utt_info_file):
    utt_wer = []
    for line in open(utt_info_file, 'r'):
        parts = line.split()
        utt = parts[0]
        wer = float(parts[1])/float(parts[2])*100
        utt_wer.append([utt, wer])

    utt_wer_sorted = sorted(utt_wer, key = lambda k : k[1])
    try:
        import numpy as np
        bins = list(range(0,105,5))
        bins.append(sys.float_info.max)

        hist, bin_edges = np.histogram([x[1] for x in utt_wer_sorted],
                                       bins = bins)
        num_utts = len(utt_wer)
        string = ''
        for i in range(len(hist)):
            string += '[{0}, {1}] {2}\n'.format(bin_edges[i], bin_edges[i+1], float(hist[i])/num_utts * 100)
        logger.info("The histogram is \n {0}".format(string))
    except ImportError:
        pass

    return utt_wer_sorted

def Main():
    args = GetArgs()
    utt_wer_sorted = GetSortedWers(args.bad_utt_info_file)
    out_file = open(args.output_file, 'w')
    logger.info("Writing output to file : {0}.".format(args.output_file))

    for row in utt_wer_sorted:
        if row[1] <= args.max_wer:
            out_file.write('{0} {1}\n'.format(row[0], row[1]))
    out_file.close()
if __name__ == "__main__":
    Main()
