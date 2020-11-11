#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
import scipy.io.wavfile as siw
import math
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        """Extract noises from the corpus based on the non-speech regions.
        e.g. {} /export/corpora4/CHiME5/audio/train/ \\
                /export/corpora4/CHiME5/transcriptions/train/ \\
                /export/b05/zhiqiw/noise/""".format(sys.argv[0]))

    parser.add_argument("--segment-length", default=20)
    parser.add_argument("out_dir", help="Output directory to write noise files. e.g. /export/b05/zhiqiw/noise/")

    args = parser.parse_args()
    return args


def Get_time(conf, tag, fs):
    for line in conf:
        line = line.strip().split(' ')
        st = float(line[0].strip())
        start_time = int(st*fs)
        ed = float(line[1].strip())
        end_time = int(ed*fs)
        tag[start_time:end_time] = 0
    return tag


def write_noise(out_dir, seg, audio, sig, tag, fs, cnt):
    sig_noise = sig[np.nonzero(tag)]
    for i in range(math.floor(len(sig_noise)/(seg*fs))):
        siw.write(out_dir +'/noise'+str(cnt)+'.wav', fs, sig_noise[i*seg*fs:(i+1)*seg*fs])
        cnt += 1
    return cnt


def main():
    args = get_args()

    audio_dir = "/export/c02/aarora8/kaldi2/egs/opensat2020/s5b_aug/data/train/wav_files2"
    trans_dir = "/export/c02/aarora8/kaldi2/egs/opensat2020/s5b_aug/data/train/time_stamp"
    audio_list = "data/local/audio_list"
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    wav_list = open(audio_list).readlines()

    cnt = 1
    for i, audio in enumerate(wav_list):
        audio = audio.strip()
        fs, sig = siw.read(audio_dir + "/" + audio + '.wav')
        tag = np.ones(len(sig))
        time_stamp = open(trans_dir + "/" + audio).readlines()
        tag = Get_time(time_stamp, tag, fs)
        cnt = write_noise(args.out_dir, args.segment_length, audio, sig, tag, fs, cnt)

if __name__ == '__main__':
    main()
