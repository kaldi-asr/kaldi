#!/usr/bin/env python3
"""
 Copyright 2020 Johns Hopkins University  (Author: Desh Raj)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

 Prepare AMI mix-headset data with RTTM from the annotations
 provided with the corpus. You first need to run local/ami_text_prep.sh
 to generate the data/local/annotations directory which is
 required to run this script.
"""

import sys
import os
import argparse
import subprocess
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

class Segment:
    def __init__(self, parts):
        self.reco_id = parts[0]
        self.spk_id = parts[2]
        self.start_time = float(parts[3])
        self.end_time = float(parts[4])
        self.dur = self.end_time - self.start_time
        self.text = parts[5]

def groupby(iterable, keyfunc):
    """Wrapper around ``itertools.groupby`` which sorts data first."""
    iterable = sorted(iterable, key=keyfunc)
    for key, group in itertools.groupby(iterable, keyfunc):
        yield key, group

def read_annotations(file_path):
    segments = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            segments.append(Segment(parts))
    return segments
    

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


def write_output(segments, out_path, min_length):
    reco_and_spk_to_segs = defaultdict(list,
        {uid : list(g) for uid, g in groupby(segments, lambda x: (x.reco_id,x.spk_id))})
    rttm_str = "SPEAKER {0} 1 {1:7.3f} {2:7.3f} <NA> <NA> {3} <NA> <NA>\n"
    with open(out_path+'/rttm.annotation','w') as rttm_writer:
        for uid in sorted(reco_and_spk_to_segs):
            segs = sorted(reco_and_spk_to_segs[uid], key=lambda x: x.start_time)
            reco_id, spk_id = uid

            for seg in segs:
                if seg.dur >= min_length:
                    rttm_writer.write(rttm_str.format(reco_id, seg.start_time, seg.dur, spk_id))

def make_diar_data(annotations, wav_path, output_path, min_length):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print ('read annotations to get segments')
    segments = read_annotations(annotations)

    reco_to_segs = defaultdict(list,
        {reco_id : list(g) for reco_id, g in groupby(segments, lambda x: x.reco_id)})
    file_list = list(reco_to_segs.keys())

    print('read audios')
    df_wav = find_audios(wav_path, file_list)
    
    print('make wav.scp')
    write_wav(df_wav, output_path)

    print('write annotation rttm')
    write_output(segments, output_path, min_length)


if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Prepare AMI dataset for diarization')

    parser.add_argument('annotations', help="Path to annotations file")
    parser.add_argument('wav_path', help="Path to AMI corpus dir")
    parser.add_argument('output_path', help="Path to generate data directory")
    parser.add_argument('--min-length', default=0.025, type=float, help="minimum length of segments to create")
    args=parser.parse_args()
    
    make_diar_data(**vars(args))
