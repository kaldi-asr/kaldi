#!/usr/bin/env python3
"""
 Copyright 2020 Johns Hopkins University  (Author: Desh Raj)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

 Prepare AMI mix-headset data. We use the RTTMs and SAD labels from the
 "only_words" category of BUT's AMI setup:
 https://github.com/BUTSpeechFIT/AMI-diarization-setup
 
 For more details about AMI splits and references used in other literature,
 please refer to Section 4 of this paper: https://arxiv.org/abs/2012.14952
"""

import sys
import os
import argparse
import subprocess

import pandas as pd

def find_audios(wav_path, file_list):
    # Get all wav file names from audio directory
    command = 'find %s -name "*Mix-Headset.wav"' % (wav_path)
    wavs = subprocess.check_output(command, shell=True).decode('utf-8').splitlines()
    keys = [ os.path.splitext(os.path.basename(wav))[0] for wav in wavs ]
    data = {'key': keys, 'file_path': wavs}
    df_wav = pd.DataFrame(data)

    # Filter list to keep only those in annotations (for the specific data split)
    file_names_str = "|".join(file_list)
    df_wav = df_wav.loc[df_wav['key'].str.contains(file_names_str)].sort_values('key')
    return df_wav

def write_wav(df_wav, output_path, bin_wav=True):
    with open(output_path + '/wav.scp', 'w') as f:
        for key,file_path in zip(df_wav['key'], df_wav['file_path']):
            key = key.split('.')[0]
            if bin_wav:
                f.write('%s sox %s -t wav - remix 1 | \n' % (key, file_path))
            else:
                f.write('%s %s\n' % (key, file_path))
                
def write_segments(sad_labels_dir, output_path):
    with open(output_path + '/segments', 'w') as f:
        for sad_file in os.listdir(sad_labels_dir):
            lab_path = os.path.join(sad_labels_dir, sad_file)
            file_id = sad_file.split('.')[0]
            with open(lab_path, 'r') as f_lab:
                for line in f_lab:
                    parts = line.strip().split()
                    start = float(parts[0])
                    end = float(parts[1])
                    seg_id = f'{file_id}_{100*start:06.0f}_{100*end:06.0f}'
                    f.write(f'{seg_id} {file_id} {start} {end}\n')
                    

def make_diar_data(meetings, wav_path, output_path, sad_labels_dir=None):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print('get file list')
    file_list = []
    with open(meetings, 'r') as f:
        for line in f:
            file_list.append(line.strip())

    print('read audios')
    df_wav = find_audios(wav_path, file_list)
    
    print('make wav.scp')
    write_wav(df_wav, output_path)
    
    if sad_labels_dir:
        print('make segments')
        write_segments(sad_labels_dir, output_path)


if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Prepare AMI dataset for diarization')

    parser.add_argument('meetings', help="Path to file containing list of meetings")
    parser.add_argument('wav_path', help="Path to AMI corpus dir")
    parser.add_argument('output_path', help="Path to generate data directory")
    parser.add_argument('--sad-labels-dir', help="Path to SAD labels", default=None)
    args=parser.parse_args()
    
    make_diar_data(**vars(args))
