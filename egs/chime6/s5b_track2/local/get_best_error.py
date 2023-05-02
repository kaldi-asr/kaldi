#! /usr/bin/env python3
# Copyright   2019   Ashish Arora
# Apache 2.0.
"""This script finds best matching of reference and hypothesis speakers.
  For the best matching speakers,it provides the WER for the reference session
  (eg:S02) and hypothesis recording (eg: S02_U02)"""

import itertools
import numpy as np
import argparse
from munkres import Munkres

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script finds best matching of reference and hypothesis speakers.
  For the best matching it provides the WER""")
    parser.add_argument("WER_dir", type=str,
                        help="path of WER files")
    parser.add_argument("recording_id", type=str,
                        help="recording_id name")
    parser.add_argument("num_speakers", type=str,
                        help="number of speakers in ref")
    args = parser.parse_args()
    return args


def get_results(filename):
    with open(filename) as f:
        first_line = f.readline()
        parts = first_line.strip().split(',')
        total_words = parts[0].split()[-1]
        ins = parts[1].split()[0]
        deletions = parts[2].split()[0]
        sub = parts[3].split()[0]
        return total_words, ins, deletions, sub


def get_min_wer(recording_id, num_speakers, WER_dir):
    best_wer_file = WER_dir + '/' + 'best_wer' + '_' + recording_id
    best_wer_writer = open(best_wer_file, 'w')
    m = Munkres()
    total_error_mat = [0] * num_speakers
    all_errors_mat = [0] * num_speakers
    for i in range(num_speakers):
        total_error_mat[i] = [0] * num_speakers
        all_errors_mat[i] = [0] * num_speakers
    for i in range(1, num_speakers+1):
        for j in range(1, num_speakers+1):
            filename = '/wer_' + recording_id + '_' + 'r' + str(i)+ 'h' + str(j)
            filename = WER_dir + filename
            total_words, ins, deletions, sub = get_results(filename)
            ins = int(ins)
            deletions = int(deletions)
            sub = int(sub)
            total_error = ins + deletions + sub
            total_error_mat[i-1][j-1]=total_error
            all_errors_mat[i-1][j-1]= (total_words, total_error, ins, deletions, sub)

    indexes = m.compute(total_error_mat)
    total_errors=total_words=total_ins=total_del=total_sub=0
    spk_order = '('
    for row, column in indexes:
        words, errs, ins, deletions, sub = all_errors_mat[row][column]
        total_errors += int(errs)
        total_words += int(words)
        total_ins += int(ins)
        total_del += int(deletions)
        total_sub += int(sub)
        spk_order = spk_order + str(column+1) + ', '
    spk_order = spk_order + ')' 
    text = "Best error: (#T #E #I #D #S) " + str(total_words)+ ', '+str(total_errors)+ ', '+str(total_ins)+ ', '+str(total_del)+ ', '+str(total_sub)
    best_wer_writer.write(" recording_id: "+ recording_id + ' ')
    best_wer_writer.write(' best hypothesis speaker order: ' + spk_order + ' ')
    best_wer_writer.write(text+ '\n')
    best_wer_writer.close()


def main():
    args = get_args()
    get_min_wer(args.recording_id, int(args.num_speakers), args.WER_dir)


if __name__ == '__main__':
    main()
