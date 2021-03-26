#!/usr/bin/env python3
# Copyright 2015   David Snyder
#           2019   Phani Sankar Nidadavolu
# Apache 2.0.
#
# This file is meant to be invoked by make_musan.sh.

import os, sys, argparse
sys.path.append("steps/data/")
sys.path.insert(0, 'steps/')
import libs.common as common_lib

def get_args():
    parser = argparse.ArgumentParser(description="Create MUSAN corpus",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use-vocals", type=str,
                        dest='use_vocals', default=True,
                        action=common_lib.StrToBoolAction,
                        choices=["true", "false"],
                        help='use vocals from the music corpus')
    parser.add_argument('--sampling-rate', type=int, default=16000,
                        help="Sampling rate of the source data. If a positive integer is specified with this option, "
                        "the MUSAN corpus will be resampled to the rate of the source data."
                        "Original MUSAN corpus is sampled at 16KHz. Defaults to 16000 Hz")
    parser.add_argument("in_dir", help="Input data directory")
    parser.add_argument("out_dir", help="Output data directory")

    print(' '.join(sys.argv))
    args = parser.parse_args()
    args = check_args(args)

    return args

def check_args(args):
    if not os.path.exists(args.in_dir):
        raise Exception('input dir {0} does not exist'.format(args.in_dir))
    if not os.path.exists(args.out_dir):
        print("Preparing {0}/musan...".format(args.out_dir))
        os.makedirs(args.out_dir)

    return args

def process_music_annotations(path):
    utt2spk = {}
    utt2vocals = {}
    lines = open(path, 'r').readlines()
    for line in lines:
        utt, genres, vocals, musician = line.rstrip().split()[:4]
        # For this application, the musican ID isn't important
        utt2spk[utt] = utt
        utt2vocals[utt] = vocals == "Y"
    return utt2spk, utt2vocals

def prepare_music(root_dir, use_vocals, sampling_rate):
    utt2vocals = {}
    utt2spk = {}
    utt2wav = {}
    num_good_files = 0
    num_bad_files = 0
    music_dir = os.path.join(root_dir, "music")
    for root, dirs, files in os.walk(music_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".wav"):
                utt = str(file).replace(".wav", "")
                utt2wav[utt] = file_path
            elif str(file) == "ANNOTATIONS":
                utt2spk_part, utt2vocals_part = process_music_annotations(file_path)
                utt2spk.update(utt2spk_part)
                utt2vocals.update(utt2vocals_part)

    utt2spk_str = ""
    utt2wav_str = ""
    for utt in utt2vocals:
        if utt in utt2wav:
            if use_vocals or not utt2vocals[utt]:
                utt2spk_str = utt2spk_str + utt + " " + utt2spk[utt] + "\n"
                if sampling_rate == 16000:
                    utt2wav_str = utt2wav_str + utt + " " + utt2wav[utt] + "\n"
                else:
                    utt2wav_str = utt2wav_str + utt + " sox -t wav " + utt2wav[utt] + " -r" \
                                    " {fs} -t wav - |\n".format(fs=sampling_rate)
            num_good_files += 1
        else:
            print("Missing file {}".format(utt))
            num_bad_files += 1
    print("In music directory, processed {} files; {} had missing wav data".format(
                                                    num_good_files, num_bad_files))
    return utt2spk_str, utt2wav_str


def prepare_speech(root_dir, sampling_rate):
    utt2spk = {}
    utt2wav = {}
    num_good_files = 0
    num_bad_files = 0
    speech_dir = os.path.join(root_dir, "speech")
    for root, dirs, files in os.walk(speech_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".wav"):
                utt = str(file).replace(".wav", "")
                utt2wav[utt] = file_path
                utt2spk[utt] = utt

    utt2spk_str = ""
    utt2wav_str = ""
    for utt in utt2spk:
        if utt in utt2wav:
            utt2spk_str = utt2spk_str + utt + " " + utt2spk[utt] + "\n"
            if sampling_rate == 16000:
                utt2wav_str = utt2wav_str + utt + " " + utt2wav[utt] + "\n"
            else:
                utt2wav_str = utt2wav_str + utt + " sox -t wav " + utt2wav[utt] + " -r" \
                                    " {fs} -t wav - |\n".format(fs=sampling_rate)
            num_good_files += 1
        else:
            print("Missing file {}".format(utt))
            num_bad_files += 1
    print("In speech directory, processed {} files; {} had missing wav data".format(
                                                    num_good_files, num_bad_files))
    return utt2spk_str, utt2wav_str


def prepare_noise(root_dir, sampling_rate):
    utt2spk = {}
    utt2wav = {}
    num_good_files = 0
    num_bad_files = 0
    noise_dir = os.path.join(root_dir, "noise")
    for root, dirs, files in os.walk(noise_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".wav"):
                utt = str(file).replace(".wav", "")
                utt2wav[utt] = file_path
                utt2spk[utt] = utt

    utt2spk_str = ""
    utt2wav_str = ""
    for utt in utt2spk:
        if utt in utt2wav:
            utt2spk_str = utt2spk_str + utt + " " + utt2spk[utt] + "\n"
            if sampling_rate == 16000:
                utt2wav_str = utt2wav_str + utt + " " + utt2wav[utt] + "\n"
            else:
                utt2wav_str = utt2wav_str + utt + " sox -t wav " + utt2wav[utt] + " -r" \
                                    " {fs} -t wav - |\n".format(fs=sampling_rate)
            num_good_files += 1
        else:
            print("Missing file {}".format(utt))
            num_bad_files += 1
    print("In noise directory, processed {} files; {} had missing wav data".format(
                                    num_good_files, num_bad_files))
    return utt2spk_str, utt2wav_str


def main():
    args = get_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    use_vocals = args.use_vocals
    sampling_rate = args.sampling_rate

    utt2spk_music, utt2wav_music = prepare_music(in_dir, use_vocals, sampling_rate)
    utt2spk_speech, utt2wav_speech = prepare_speech(in_dir, sampling_rate)
    utt2spk_noise, utt2wav_noise = prepare_noise(in_dir, sampling_rate)

    utt2spk = utt2spk_speech + utt2spk_music + utt2spk_noise
    utt2wav = utt2wav_speech + utt2wav_music + utt2wav_noise
    wav_fi = open(os.path.join(out_dir, "wav.scp"), 'w')
    wav_fi.write(utt2wav)
    utt2spk_fi = open(os.path.join(out_dir, "utt2spk"), 'w')
    utt2spk_fi.write(utt2spk)


if __name__=="__main__":
    main()
