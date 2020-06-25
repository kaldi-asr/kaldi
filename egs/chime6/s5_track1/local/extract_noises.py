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


def Trans_time(time, fs):
    units = time.split(':')
    time_second = float(units[0]) * 3600 + float(units[1]) * 60 + float(units[2])
    return int(time_second*fs)


# remove mic dependency for CHiME-6
def Get_time(conf, tag, fs):
    for i in conf:
        st = Trans_time(i['start_time'], fs)
        ed = Trans_time(i['end_time'], fs)
        tag[st:ed] = 0
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

    wav_list = open(args.audio_list).readlines()

    cnt = 1
    for i, audio in enumerate(wav_list):
        parts = audio.strip().split('.')
        if len(parts) == 2:
            # Assuming distant mic with name like S03_U01.CH1
            session, mic = parts[0].split('_')
            channel = parts[1]
            base_name = session + "_" + mic + "." + channel
        else:
            # Assuming close talk mic with name like S03_P09
            session, mic = audio.strip().split('_')
            base_name = session + "_" + mic
        fs, sig = siw.read(args.audio_dir + "/" + base_name + '.wav')
        tag = np.ones(len(sig))
        if i == 0 or session != session_p:
            with open(args.trans_dir + "/" + session + '.json') as f:
                conf = json.load(f)
        tag = Get_time(conf, tag, fs)
        cnt = write_noise(args.out_dir, args.segment_length, audio, sig, tag, fs, cnt)
        session_p = session


if __name__ == '__main__':
    main()
