#!/usr/bin/env python3

# Copyright 2018-2020  Yiming Wang
#           2018-2020  Daniel Povey
# Apache 2.0

""" This script prepares the SNIPS data into kaldi format.
"""


import argparse
import os
import sys
import json

def main():
    parser = argparse.ArgumentParser(description="""Prepare data.""")
    parser.add_argument('path', type=str,
                        help='path to the json file')
    parser.add_argument('out_dir', type=str,
                        help='out dir')
    parser.add_argument('--wake-word', type=str, default='HeySnips',
                        help='wake word transcript')
    parser.add_argument('--non-wake-word', type=str, default='FREETEXT',
                        help='non-wake word transcript')
    args = parser.parse_args()

    with open(args.path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        utt_id, spk_id, wav_file, label = [], [], [], []
        for entry in data:
            utt_id.append(entry['id'])
            spk_id.append(entry['worker_id'])
            wav_file.append(entry['audio_file_path'])
            label.append(entry['is_hotword'])

    dir_prefix = os.path.dirname(os.path.abspath(args.path))
    with open(os.path.join(args.out_dir, 'wav.scp'), 'w', encoding='utf-8') as f_wav, \
        open(os.path.join(args.out_dir, 'text'), 'w', encoding='utf-8') as f_text, \
        open(os.path.join(args.out_dir, 'utt2spk'), 'w', encoding='utf-8') as f_utt2spk:
        count = 0
        for utt, spk, wav, l in zip(utt_id, spk_id, wav_file, label):
            f_wav.write(spk + '-' + utt + ' ' + os.path.join(dir_prefix, wav) + '\n')
            f_text.write(spk + '-' + utt + ' ' + (args.wake_word if l == 1 else args.non_wake_word) + '\n')
            f_utt2spk.write(spk + '-' + utt + ' ' + spk + '\n')
            if l == 1:
                count += 1
        print(str(count) + '/' + str(len(label)) +' of utterances are wake word.')

if __name__ == "__main__":
    main()
