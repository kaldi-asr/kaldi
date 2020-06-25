#!/usr/bin/env python3

# Copyright 2018-2020  Yiming Wang
#           2018-2020  Daniel Povey
# Apache 2.0

""" This script prepares the Mobvoi data into kaldi format.
"""


import argparse
import os
import sys
import json

def main():
    parser = argparse.ArgumentParser(description="""Prepare data.""")
    parser.add_argument("wav_dir", type=str,
                        help="dir containing all the wav files")
    parser.add_argument("path", type=str,
                        help="path to the json file")
    parser.add_argument("out_dir", type=str,
                        help="out dir")
    parser.add_argument("--non-wake-word", type=str, default="FREETEXT",
                        help="non-wake word transcript")
    args = parser.parse_args()

    assert args.non_wake_word != "HiXiaowen" and args.non_wake_word != "NihaoWenwen"
    with open(args.path, "r", encoding="utf-8") as f:
        data = json.load(f)
        utt_id, spk_id, wav_file, label = [], [], [], []
        for entry in data:
            utt_id.append(entry["utt_id"])
            spk_id.append(entry["speaker_id"])
            label.append(entry["keyword_id"])

    abs_dir = os.path.abspath(args.wav_dir)
    with open(os.path.join(args.out_dir, "wav.scp"), "w", encoding="utf-8") as f_wav, \
        open(os.path.join(args.out_dir, "text"), "w", encoding="utf-8") as f_text, \
        open(os.path.join(args.out_dir, 'utt2spk'), 'w', encoding="utf-8") as f_utt2spk:
        for utt, spk, l in zip(utt_id, spk_id, label):
            if spk is None:
                spk = utt  # deal with None speaker
            f_wav.write(spk + "-" + utt + " " + os.path.join(abs_dir, utt + ".wav") + "\n")
            if l == 0:
                text = "HiXiaowen"
            elif l == 1:
                text = "NihaoWenwen"
            else:
                assert l == -1
                text = args.non_wake_word
            f_text.write(spk + "-" + utt + " " + text + "\n")
            f_utt2spk.write(spk + "-" + utt + " " + spk + "\n")

if __name__ == "__main__":
    main()
