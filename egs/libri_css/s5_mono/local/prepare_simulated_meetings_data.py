#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright  2020  Johns Hopkins University (Author: Desh Raj)
# Copyright  2020  University of Stuttgart (Author: Pavel Denisov)
# Apache 2.0

import argparse
import glob
import json
import os
from progressbar import ProgressBar

def write_dict_to_file(utt2data, file_path):
    f = open(file_path, "w")
    for utt in utt2data.keys():
        f.write("{} {}\n".format(utt, utt2data[utt]))
    f.close()
    return


def main(args):
    os.makedirs(args.tgtpath, exist_ok=True)
    pbar = ProgressBar()

    # Collect transcriptions from LibriSpeech data
    transcriptions = {}
    trans_pattern = os.path.join(args.txtpath, "{}*".format(args.type), "*", "*", "*.trans.txt")
    for trans in glob.glob(trans_pattern):
        with open(trans, "r") as f:
            for line in f:
                index = line.index(" ")
                utt_id = line[:index]
                transcription = line[index + 1 :].strip()
                transcriptions[utt_id] = transcription

    # Dictionary to store all info that we will write to files after
    # reading all files.
    reco2wav = {}  # for wav.scp
    reco2segments = {}  # for segments
    utt2spk = {}  # for utt2spk
    utt2text = {}  # for text
    utt2clean = {} # Path to the corresponding clean Librispeech utterance

    with open(os.path.join(args.wavpath, "mixlog.json"), "r") as f:
        mixlog = json.load(f)

    for recording in pbar(mixlog):
        wav_name = recording["output"].split(os.path.sep)[-1]
        reco = "libricss_sim_{}".format(wav_name[:-4])
        wav_path = os.path.join(args.wavpath, "wav", wav_name)
        reco2wav[reco] = "sox {} -r 16k -t wav -c 1 -b 16 -e signed - |".format(
            wav_path
        )
        reco2segments[reco] = []
        for utt in recording["inputs"]:
            orig_segment_id = utt["path"].split(os.path.sep)[-1][:-4]
            utt_id = "{}_{}".format(orig_segment_id, reco)
            start = round(utt["offset"], 2)
            end = round(utt["offset"] + utt["length_in_seconds"], 2)
            reco2segments[reco].append([utt_id, start, end])
            utt2spk[utt_id] = utt["speaker_id"]
            utt2text[utt_id] = transcriptions[orig_segment_id]
            utt2clean[utt_id] = "sox {} -r 16k -t wav -c 1 -b 16 -e signed - |".format(
                utt["path"]
            )

    # Write all dictionaries to respective files
    write_dict_to_file(reco2wav, os.path.join(args.tgtpath, "wav.scp"))
    write_dict_to_file(utt2spk, os.path.join(args.tgtpath, "utt2spk"))
    write_dict_to_file(utt2text, os.path.join(args.tgtpath, "text"))
    write_dict_to_file(utt2clean, os.path.join(args.tgtpath, "wav_clean.scp"))

    f = open(os.path.join(args.tgtpath, "segments"), "w")
    for reco in reco2segments.keys():
        segments = reco2segments[reco]
        for segment in segments:
            f.write("{} {} {} {}\n".format(segment[0], reco, segment[1], segment[2]))
    f.close()


def make_argparse():
    parser = argparse.ArgumentParser(
        description="Prepare simulated meetings LibriCSS data into Kaldi format."
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--txtpath",
        metavar="<path>",
        required=True,
        help="LibiSpeech subset path for transcriptions.",
    )
    parser.add_argument(
        "--wavpath",
        metavar="<path>",
        required=True,
        help="Simulated meetings LibriCSS data path for WAV.",
    )
    parser.add_argument(
        "--tgtpath", metavar="<path>", required=True, help="Destination path."
    )
    parser.add_argument(
        "--type", required=True, help="train/dev/test"
    )
    
    return parser


if __name__ == "__main__":
    parser = make_argparse()
    args = parser.parse_args()
    main(args)
