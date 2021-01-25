#!/usr/bin/env python3
"""
Copyright 2020 Johns Hopkins University  (Author: Desh Raj)
Copyright 2020 ARL  (Author: John Morgan)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

Prepare rats_sad data for training a speech activity detection system.
Input comes from the annotations provided with the corpus. 
Output is written to an RTTM file.
"""

import sys
import os
import argparse
import subprocess
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

class Segment:
    """
    Field 5 (SAD segment label) can have one of the following values:
    S  : speech segment
    NS : non-speech segment
    NT : "button-off" segment (not transmitted according to PTT log)
    RX : "button-off" segment (not transmitted according to RMS scan)

    Filenames:
    transceiver files:   {iiiii}_{jjjjj}_{lng}_{c}.{type}
    where:
    {iiiii} is a 5-digit source audio identifier
    {jjjjj} is a 5-digit transmission session identifier
    {lng} is one of:
        alv: Levantine Arabic
        eng: American English
        fas: Farsi (Persian)
        pus: Pashto
        urd: Urdu
    {c}    is one of: A B C D E F G H
    """
    def __init__(self, fields):
        self.partition = fields[0]
        self.reco_id = fields[1]
        self.start_time = float(fields[2])
        self.end_time = float(fields[3])
        self.dur = self.end_time - self.start_time
        self.sad_label = fields[4]
        self.sad_provenance = fields[5]
        # self.speaker_id = fields[6]
        # self.sid_provenance = fields[7]
        # self.language_id = fields[8]
        self.language_id = fields[6]
        # self.lid_provenance = fields[9]
        self.lid_provenance = fields[7]
        # self.transcript = fields[10]
        # self.transcript_provenance = fields[11]

        rec_info = self.reco_id.split('_')
        if len(rec_info) == 3:
            src_audio_id = rec_info[0]
            lng = rec_info[1]
            src = rec_info[2]
            self.spk_id = src_audio_id
        elif len(rec_info) == 4:
            src_audio_id = rec_info[0]
            transmission_session_id = rec_info[1]
            lng = rec_info[2]
            channel = rec_info[3]
            self.spk_id = src_audio_id


def groupby(iterable, keyfunc):
    """Wrapper around ``itertools.groupby`` which sorts data first."""
    iterable = sorted(iterable, key=keyfunc)
    for key, group in itertools.groupby(iterable, keyfunc):
        yield key, group

def read_annotations(file_path):
    segments = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            fields = line.strip().split()
            segments.append(Segment(fields))
    return segments


def find_audios(wav_path, file_list):
    # Get all .flac file names from audio directory
    wav_path = Path(wav_path)
    wavs_glob = wav_path.rglob('*.flac')
    wavs = [ w for w in wavs_glob ]
    keys = [ Path(wf).stem for wf in wavs ]
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
    # write 5 places after the decimal point
    rttm_str = "SPEAKER {0} 1 {1:7.5f} {2:7.5f} <NA> <NA> {3} <NA> <NA>\n"
    with open(out_path+'/rttm.annotation','w') as rttm_writer:
        for uid in sorted(reco_and_spk_to_segs):
            segs = sorted(reco_and_spk_to_segs[uid], key=lambda x: x.start_time)
            reco_id, spk_id = uid

            for seg in segs:
                # skip the non-speech segments
                if seg.sad_label == 'NS':
                    continue
                elif seg.sad_label == 'NT':
                    continue
                elif seg.sad_label == 'RX':
                    continue
                elif seg.dur >= min_length:
                    rttm_writer.write(rttm_str.format(reco_id, seg.start_time, seg.dur, spk_id))
                else:
                    print('Bad segment', seg)


def make_sad_data(annotations, wav_path, output_path, min_length):
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
        description='Prepare rats_sad corpus for Speech Activity Detection')
    parser.add_argument('annotations', help="Output path to annotations file")
    parser.add_argument('wav_path', help="Path to source rats_sad corpus audio files directory")
    parser.add_argument('output_path', help="Path to data directory")
    parser.add_argument('--min-length', default=0.0001, type=float, help="minimum length of segments to create")
    args=parser.parse_args()
    make_sad_data(**vars(args))
