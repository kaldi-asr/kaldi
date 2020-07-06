#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright  2020  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

import argparse, os, glob, tqdm, zipfile, pathlib

def write_dict_to_file(utt2data, file_path):
    f = open(file_path, 'w')
    for utt in utt2data.keys():
        f.write('{} {}\n'.format(utt, utt2data[utt]))
    f.close()
    return

def main(args):
    os.makedirs(args.tgtpath, exist_ok=True)

    # Dictionary to store all info that we will write to files after 
    # reading all files.
    reco2wav = {} # for wav.scp
    reco2segments = {} # for segments
    utt2spk = {} # for utt2spk
    utt2text = {} # for text

    # First we create reco2wav from the separated wav files
    wavs = os.listdir(args.wav_path)
    for wav in wavs:
        path = os.path.join(args.wav_path, wav)
        _,_,olr,_,sil_max,sessid,_,_,stream = pathlib.Path(path).stem.split('_')
        cond = "OV{}".format(int(float(olr)))
        if (float(olr) == 0):
            if (sil_max == '0.5'):
                cond = "0S"
            else:
                cond = "0L"
        wav_name = "{}_CH0_{}_{}".format(sessid, cond, stream) # session0_CH0_0L_1
        reco2wav[wav_name] = path
        if (args.volume != 1):
            reco2wav[wav_name] = "sox -v {} -t wav {} -t wav - |".format(args.volume, path) 

    
    # Now we get other info from the original LibriCSS corpus dir
    conditions = ('0L','0S','OV10','OV20','OV30','OV40')   
    for cond in tqdm.tqdm(conditions):
        meeting = glob.glob(os.path.join(args.srcpath, cond, 'overlap*'))
        for meet in meeting:
            segments = []
            _,_,_,_,_,sessid,_ = os.path.basename(meet).split('_')
            reco_id = "{}_CH0_{}".format(sessid, cond) # session0_CH0_0L
            with open(os.path.join(os.path.abspath(meet), 'transcription', 'meeting_info.txt'), 'r') as f:
                next(f)
                for line in f:
                    start,end,spkid,_,text = line.strip().split(maxsplit=4)
                    start = float("{:.2f}".format(float(start)))
                    end = float("{:.2f}".format(float(end)))
                    utt_id = "{}_{}_{}_{}".format(spkid,reco_id,"{:.0f}".format(100*start).zfill(6),
                        "{:.0f}".format(100*end).zfill(6)) # 6930_Session0_CH1_0L_000853_002463
                    utt2spk[utt_id] = spkid
                    utt2text[utt_id] = text
                    segments.append((utt_id, start, end))
            
            reco2segments[reco_id] = segments
    
    # Write all dictionaries to respective files
    write_dict_to_file(reco2wav, os.path.join(args.tgtpath, 'wav.scp'))
    write_dict_to_file(utt2spk, os.path.join(args.tgtpath, 'utt2spk'))
    write_dict_to_file(utt2text, os.path.join(args.tgtpath, 'text'))

    f = open(os.path.join(args.tgtpath, 'segments'), 'w')
    for reco in reco2segments.keys():
        segments = reco2segments[reco]
        for segment in segments:
            f.write('{} {} {} {}\n'.format(segment[0], reco, segment[1], segment[2]))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reorganize LibriCSS data into Kaldi format.'
        ' Additionally, use separated wav files.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--srcpath', metavar='<path>', required=True, 
                        help='Original LibriCSS data path.')
    parser.add_argument('--wav-path', metavar='<path>', required=True, 
                        help='Path to directory containing separated wavs.')
    parser.add_argument('--tgtpath', metavar='<path>', required=True, 
                        help='Destination path.')
    parser.add_argument('--volume', default=1, type=float, help='sox -v option')

    args = parser.parse_args()
    main(args)
