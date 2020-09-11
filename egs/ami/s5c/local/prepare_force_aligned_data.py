#!/usr/bin/env python3
"""
 Copyright 2020 Johns Hopkins University  (Author: Desh Raj)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

 Prepare AMI mix-headset data with RTTM from the forced alignments
 provided with the corpus.
"""

import sys
import os
import argparse
import time
import logging
import subprocess
import re
import itertools
from collections import defaultdict
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

class Segment:
    def __init__(self, reco_id, spk_id, start_time, end_time, text):
        self.reco_id = reco_id
        self.spk_id = spk_id
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.duration = self.end_time - self.start_time
        self.text = text

def groupby(iterable, keyfunc):
    """Wrapper around ``itertools.groupby`` which sorts data first."""
    iterable = sorted(iterable, key=keyfunc)
    for key, group in itertools.groupby(iterable, keyfunc):
        yield key, group

def read_annotations(annotations):
    segments = []
    with open(annotations, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(maxsplit=5)
            if (len(parts) < 6):
                continue
            seg = Segment(parts[0], parts[2], parts[3], parts[4], parts[5])
            segments.append(seg)
    return segments

def find_audios(wav_path):
    
    command = 'find %s -name "*Mix-Headset.wav"' % (wav_path)
    wavs = subprocess.check_output(command, shell=True).decode('utf-8').splitlines()
    keys = [ os.path.splitext(os.path.basename(wav))[0] for wav in wavs ]
    data = {'key': keys, 'file_path': wavs}
    df_wav = pd.DataFrame(data)
    return df_wav


def filter_wavs(df_wav, file_names):
    file_names_str = "|".join(file_names)
    df_wav = df_wav.loc[df_wav['key'].str.contains(file_names_str)].sort_values('key')
    return df_wav


def write_wav(df_wav, output_path, bin_wav=True):

    with open(output_path + '/wav.scp', 'w') as f:
        for key,file_path in zip(df_wav['key'], df_wav['file_path']):
            if bin_wav:
                f.write('%s sox %s -t wav - remix 1 | \n' % (key, file_path))
            else:
                f.write('%s %s\n' % (key, file_path))


def write_segments(df_wav, output_path, reco2segs):
    with open(output_path + '/segments', 'w') as segments_writer, open(output_path + '/utt2spk', 'w') as utt2spk_writer, \
        open(output_path + '/text', 'w') as text_writer:
        for key in df_wav['key']:
            reco_id = key.split('.')[0]
            segs = reco2segs[reco_id]
            for seg in segs:
                st = int(seg.start_time * 100)
                end = int(seg.end_time * 100)
                seg_id = "{0}_{1}-{2:06d}-{3:06d}".format(seg.spk_id, key, st, end)
                segments_writer.write("{0} {1} {2} {3}\n".format(seg_id, key, seg.start_time, seg.end_time))
                utt2spk_writer.write("{0} {1}\n".format(seg_id, seg.spk_id))
                text_writer.write("{0} {1}\n".format(seg_id, seg.text))


def make_diar_data(wav_path, output_path, annotations):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print('read annotations')
    segments = read_annotations(annotations)
    reco2segs = defaultdict(list,
        {reco_id : list(g) for reco_id, g in groupby(segments, lambda x: x.reco_id)})
    file_list = reco2segs.keys()

    print('read audios')
    df_wav = find_audios(wav_path)
    
    print('make wav.scp')
    df_wav = filter_wavs(df_wav, file_list)
    write_wav(df_wav, output_path)

    print('making segments, text, and utt2spk files')
    write_segments(df_wav, output_path, reco2segs)


if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Prepare AMI dataset for diarization')

    parser.add_argument('wav_path', help="Path to AMI corpus dir")
    parser.add_argument('output_path', help="Path to generate data directory")
    parser.add_argument('annotations', help="Path to annotations file")

    args=parser.parse_args()
    
    make_diar_data(**vars(args))
