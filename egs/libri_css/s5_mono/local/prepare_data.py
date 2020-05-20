#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright  2020  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

import argparse, os, glob, tqdm, zipfile
import soundfile as sf

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

    # Create a directory to store channel-separated wav files
    wav_dir = os.path.join(args.tgtpath,'wavs')
    os.makedirs(wav_dir, exist_ok=True)

    conditions = ('0L','0S','OV10','OV20','OV30','OV40')
    all_lines=[]
    for cond in tqdm.tqdm(conditions):
        meeting = glob.glob(os.path.join(args.srcpath, cond, 'overlap*'))
        for meet in meeting:
            # Extract the signals of the selected microphones. 
            meeting_name = os.path.basename(meet)
            _,_,_,_,_,sessid,olr = meeting_name.split('_')

            wav_path = os.path.join(os.path.abspath(meet), 'record', 'raw_recording.wav')
            s, f = sf.read(wav_path)
            for mic in args.mics:
                reco_id = "LC_{}_CH{}_{}".format(sessid, mic, cond) # LC_Session0_CH1_0L
                new_wav_path = os.path.join(wav_dir, reco_id+'.wav')
                sf.write(new_wav_path, s[:, mic], f)
                reco2wav[reco_id] = os.path.abspath(new_wav_path)
            
            segments = []
            with open(os.path.join(os.path.abspath(meet), 'transcription', 'meeting_info.txt'), 'r') as f:
                next(f)
                for line in f:
                    start,end,spkid,_,text = line.strip().split(maxsplit=4)
                    start = float("{:.2f}".format(float(start)))
                    end = float("{:.2f}".format(float(end)))
                    utt_id = "{}_{}_{}_{}".format(reco_id,spkid,"{:.0f}".format(100*start).zfill(6),
                        "{:.0f}".format(100*end).zfill(6)) # LC_Session0_CH1_0L_6930_000853_002463
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



def make_argparse():
    parser = argparse.ArgumentParser(description='Reorganize LibriCSS data into Kaldi format.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--srcpath', metavar='<path>', required=True, 
                        help='Original LibriCSS data path.')
    parser.add_argument('--tgtpath', metavar='<path>', required=True, 
                        help='Destination path.')
    parser.add_argument('--mics', type=int, metavar='<#mics>', nargs='+', default=[0, 1, 2, 3, 4, 5, 6], 
                        help='Microphone indices.')

    return parser



if __name__ == '__main__':
    parser = make_argparse()
    args = parser.parse_args()
    main(args)
