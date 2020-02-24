#!/usr/bin/env python3
# Copyright   2019   Ashish Arora
# Apache 2.0.

import argparse
import sys, os
import string

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script splits a kaldi text file
        into per_speaker per_session text files""")
    parser.add_argument("wer_dir_path", type=str,
                        help="path of directory containing wer files")
    parser.add_argument("output_dir_path", type=str,
                        help="path of the directory containing per speaker output files")
    args = parser.parse_args()
    return args

def get_results(filename):
    with open(filename) as f:
        first_line = f.readline()
        parts = first_line.strip().split(',')
        total_words = parts[0].split()[-1]
        ins = parts[1].split()[0]
        deletion = parts[2].split()[0]
        sub = parts[3].split()[0]
        return int(total_words), int(ins), int(deletion), int(sub)

def main():
    args = get_args()
    recodingid_error_dict={}
    min_wer_per_recording = os.path.join(args.wer_dir_path, 'all.txt')
    for line in open(min_wer_per_recording, 'r', encoding='utf8'):
        toks = line.strip().split()
        recordingid = toks[1]
        total_words = toks[-5][:-1]
        total_errors = toks[-4][:-1]
        total_ins = toks[-3][:-1]
        total_del = toks[-2][:-1]
        total_sub = toks[-1]
        recodingid_error_dict[recordingid]=(total_words, total_errors, total_ins, total_del, total_sub)
    
    recording_spkorder_file = os.path.join(args.output_dir_path, 'recordinid_spkorder')
    for line in open(recording_spkorder_file, 'r', encoding='utf8'):
        parts = line.strip().split(':')
        recordingid = parts[0]
        spkorder = parts[1]
        spkorder_list=spkorder.split('_')
        num_speakers=len(spkorder_list)
        total_errors=total_words=total_ins=total_del=total_sub=0    
        for i in range(1, num_speakers+1):
            filename = 'wer_' + recordingid + '_' + 'r' + str(i)+ 'h' + str(spkorder_list[i-1])
            wer_filename = os.path.join(args.wer_dir_path, filename)
            words, ins, deletion, sub = get_results(wer_filename)
            total_words += words
            total_ins += ins
            total_del += deletion
            total_sub += sub
            total_errors += ins + deletion + sub
        assert int(total_words) == int(recodingid_error_dict[recordingid][0]), "Total words mismatch"
        assert int(total_errors) == int(recodingid_error_dict[recordingid][1]), "Total errors mismatch"
        assert int(total_ins) == int(recodingid_error_dict[recordingid][2]), "Total insertions mismatch"
        assert int(total_del) == int(recodingid_error_dict[recordingid][3]), "Total deletions mismatch"
        assert int(total_sub) == int(recodingid_error_dict[recordingid][4]), "Total substitutions mismatch"


if __name__ == '__main__':
    main()
