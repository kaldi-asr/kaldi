#!/usr/bin/env python3
# Copyright 2020 Xuechen LIU
# Apache 2.0.
#
# prepare HIMIA data
import os
import sys

def main():
    srcdir = sys.argv[1]
    tardir = sys.argv[2]
    os.makedirs(tardir, exist_ok=True)
    
    if 'train' in srcdir:
        src_wavscp_file = srcdir + '/SPEECHDATA/train.scp'
        wavdir = srcdir + '/SPEECHDATA'
    elif 'dev' in srcdir:
        src_wavscp_file = srcdir + '/SPEECHDATA/dev.scp'
        wavdir = srcdir + '/SPEECHDATA'
    else:
        src_wavscp_file = srcdir + '/wav.scp'
        wavdir = srcdir + '/wav'
    
    src_wavscp = open(src_wavscp_file, 'r')
    tar_wavscp = open(tardir + '/wav.scp', 'w')
    tar_utt2spk = open(tardir + '/utt2spk', 'w')

    for line in src_wavscp.readlines():
        wav_path = wavdir + '/' + line.rstrip()
        wav_name = os.path.splitext(line)[0].split('/')[-1].replace('_', '-')
        spk = wav_name.split('-')[0]
        utt = wav_name
        tar_wavscp.write(utt + ' ' + wav_path + '\n')
        tar_utt2spk.write(utt + ' ' + spk + '\n')
    
    src_wavscp.close()
    tar_wavscp.close()
    tar_utt2spk.close()


if __name__ == "__main__":
    main()
