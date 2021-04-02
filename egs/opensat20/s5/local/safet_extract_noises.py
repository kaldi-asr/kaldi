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
    parser.add_argument("audio_dir", help="""Location of the CHiME5 Audio files. e.g. /export/corpora4/CHiME5/audio/train/""")
    parser.add_argument("trans_dir", help="""Location of the CHiME5 Transcriptions. e.g. /export/corpora4/CHiME5/transcriptions/train/""")
    parser.add_argument("audio_list", help="""List of ids of the CHiME5 recordings from which noise is extracted. e.g. local/distant_audio_list""")
    parser.add_argument("out_dir", help="Output directory to write noise files. e.g. /export/b05/zhiqiw/noise/")

    args = parser.parse_args()
    return args

""" 
audio_dir = "/export/c02/aarora8/kaldi2/egs/opensat2020/s5b_aug/data/train/wav_files2"
trans_dir = "/export/c02/aarora8/kaldi2/egs/opensat2020/s5b_aug/data/train/time_stamp"
audio_list = "data/local/audio_list"
"""

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
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    print(args.audio_list)
    print(args.audio_dir)
    print(args.trans_dir)
    print(args.out_dir)
    wav_list = open(args.audio_list).readlines()

    cnt = 1
    for i, audio in enumerate(wav_list):
        audio = audio.strip()
        fs, sig = siw.read(args.audio_dir + "/" + audio + '.wav')
        tag = np.ones(len(sig))
        time_stamp = open(args.trans_dir + "/" + audio).readlines()
        tag = Get_time(time_stamp, tag, fs)
        cnt = write_noise(args.out_dir, args.segment_length, audio, sig, tag, fs, cnt)

if __name__ == '__main__':
    main()
