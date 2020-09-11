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

class Word:
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

def read_split(split_file):
    files = {}
    with open(split_file, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            files[parts[0]] = parts[1].split()
    return files

def find_audios(wav_path, file_list):
    
    command = 'find %s -name "*Mix-Headset.wav"' % (wav_path)
    wavs = subprocess.check_output(command, shell=True).decode('utf-8').splitlines()
    keys = [ os.path.splitext(os.path.basename(wav))[0] for wav in wavs ]
    data = {'key': keys, 'file_path': wavs}
    df_wav = pd.DataFrame(data)
    return df_wav

def rttm_is_sorted_by_tbeg(rttm):
    tbeg=rttm['tbeg'].values
    file_id=rttm['file_id'].values
    return np.all(np.logical_or(tbeg[1:]-tbeg[:-1]>=0,
                                file_id[1:] != file_id[:-1]))

def sort_rttm(rttm):
    return rttm.sort_values(by=['file_id','tbeg'])

def read_rttm(rttm_path, uem_path, file_list, sep=' ', rttm_suffix=''):

    rttm = pd.read_csv(rttm_path, sep=sep, header=None,
                       names=['segment_type','file_id','chnl','tbeg','tdur',
                              'ortho','stype','name','conf','slat'],
                       dtype={'name': np.str, 'chnl': np.int, 'tbeg': np.float, 'tdur': np.float})
    #remove empty lines:
    rttm['ortho'] = '<NA>'
    rttm['stype'] = '<NA>'
    if not rttm_is_sorted_by_tbeg(rttm):
        print('RTTM %s not properly sorted, sorting it' % (rttm_path))
        rttm = sort_rttm(rttm)

    # filter rttm based on file list
    file_names_str = "|".join(file_list)
    rttm = rttm.loc[rttm['file_id'].str.contains(file_names_str)].sort_values('file_id')

    #cross with uem
    uem = pd.read_csv(uem_path, sep=' ', header=None,
                      names=['file_id','chnl','file_tbeg','file_tend'],
                      dtype={'chnl': np.int, 'file_tbeg': np.float, 'file_tend': np.float})

    # filter uem based on available wav files
    uem = uem.loc[uem['file_id'].str.contains(file_names_str)].sort_values('file_id')
    
    rttm_uem = pd.merge(left=rttm, right=uem, on=['file_id', 'chnl'])

    assert rttm_uem.shape[0] == rttm.shape[0]

    index_fix=(rttm_uem['tbeg'] < rttm_uem['file_tend']) & (rttm_uem['tbeg'] + rttm_uem['tdur']> rttm_uem['file_tend'])
    if np.sum(index_fix) > 0:
        print('fixing %d segments with exceding file duration' % (np.sum(index_fix)))
        print(rttm_uem[index_fix])
        rttm_uem.loc[index_fix, 'tdur'] = 0 # rttm_uem[index_fix].file_tend - rttm_uem[index_fix].tbeg
              
    index_keep=(rttm_uem['tbeg'] < rttm_uem['file_tend']) 
    n_rm = rttm.shape[0] - np.sum(index_keep)
    if n_rm > 0:
        print('removing %d segments outside file duration' % (n_rm))
        print(rttm_uem[~index_keep])
        rttm_uem = rttm_uem[index_keep]

    rttm = rttm_uem.drop(columns=['file_tbeg', 'file_tend'])
    
    return rttm


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


def get_spkid_from_segment_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root.find('./segment').attrib['participant']

def read_words_from_xml(xml_file, reco_id, spk_id):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    word_tags = root.findall('./w')
    words = []
    for word_tag in word_tags:
        word = Word(reco_id, spk_id, word_tag.attrib['starttime'], word_tag.attrib['endtime'], word_tag.text)
        words.append(word)
    return words


def get_word_segments(df_wav, xml_dir):
    command = 'find %s -name "*segments.xml"' % (xml_dir)
    segments_xml_files = subprocess.check_output(command, shell=True).decode('utf-8').splitlines()
    command = 'find %s -name "*words.xml"' % (xml_dir)
    words_xml_files = subprocess.check_output(command, shell=True).decode('utf-8').splitlines()

    words = []
    for file_id in df_wav['key']:
        file_id_trunc = file_id.split('.')[0]
        file_segment_xml_files = [xml for xml in segments_xml_files if file_id_trunc in xml]
        file_words_xml_files = [xml for xml in words_xml_files if file_id_trunc in xml]
        for part in ['A','B','C','D']:
            cur_segments_file = [xml for xml in file_segment_xml_files if "{}.segments".format(part) in xml]
            cur_words_file = [xml for xml in file_words_xml_files if "{}.words".format(part) in xml]
            if len(cur_segments_file) == 0 or len(cur_words_file) == 0:
                continue
            assert (len(cur_segments_file) == 1 and len(cur_words_file) == 1)
            spk_id = get_spkid_from_segment_xml(cur_segments_file[0])
            words += read_words_from_xml(cur_words_file[0], file_id, spk_id)

    return words


def get_annotation_segments(df_wav, xml_dir):
    command = 'find %s -name "*segments.xml"' % (xml_dir)
    segments_xml_files = subprocess.check_output(command, shell=True).decode('utf-8').splitlines()
    command = 'find %s -name "*words.xml"' % (xml_dir)
    words_xml_files = subprocess.check_output(command, shell=True).decode('utf-8').splitlines()

    words = []
    for file_id in df_wav['key']:
        file_id_trunc = file_id.split('.')[0]
        file_segment_xml_files = [xml for xml in segments_xml_files if file_id_trunc in xml]
        file_words_xml_files = [xml for xml in words_xml_files if file_id_trunc in xml]
        for part in ['A','B','C','D']:
            cur_segments_file = [xml for xml in file_segment_xml_files if "{}.segments".format(part) in xml]
            cur_words_file = [xml for xml in file_words_xml_files if "{}.words".format(part) in xml]
            if len(cur_segments_file) == 0 or len(cur_words_file) == 0:
                continue
            assert (len(cur_segments_file) == 1 and len(cur_words_file) == 1)
            spk_id = get_spkid_from_segment_xml(cur_segments_file[0])
            words += read_words_from_xml(cur_words_file[0], file_id, spk_id)

    return words


def write_rttm(segments, out_path, min_length):
    reco_and_spk_to_segs = defaultdict(list,
        {uid : list(g) for uid, g in groupby(segments, lambda x: (x.reco_id,x.spk_id))})
    rttm_str = "SPEAKER {0} 1 {1:7.3f} {2:7.3f} <NA> <NA> {3} <NA> <NA>\n"
    segments_str = "{0} {1} {2:7.3f} {3:7.3f}\n"
    with open(out_path+'/rttm.annotation','w') as rttm_writer, open(out_path+'/segments','w') as segments_writer, \
        open(out_path+'/utt2spk','w') as utt2spk_writer:
        for uid in sorted(reco_and_spk_to_segs):
            segs = sorted(reco_and_spk_to_segs[uid], key=lambda x: x.start_time)
            reco_id, spk_id = uid
            cur_start = segs[0].start_time
            cur_end = segs[0].end_time
            for seg in segs[1:]:
                # First flush existing seg if it satisfies minimum length
                if (cur_end - cur_start >= min_length):
                    st = int(cur_start * 100)
                    end = int(cur_end * 100)
                    seg_id = "{0}_{1}-{2:06d}-{3:06d}".format(reco_id, spk_id, st, end)
                    rttm_writer.write(rttm_str.format(reco_id, cur_start, cur_end-cur_start, spk_id))
                    segments_writer.write(segments_str.format(seg_id, reco_id, cur_start, cur_end))
                    utt2spk_writer.write("{} {}\n".format(seg_id, seg_id))
                    cur_start = seg.start_time
                    cur_end = seg.end_time
                else:
                    cur_end = seg.end_time
            
            # Flush last remaining segment
            if (cur_end - cur_start >= min_length):
                st = int(cur_start * 100)
                end = int(cur_end * 100)
                seg_id = "{0}_{1}-{2:06d}-{3:06d}".format(reco_id, spk_id, st, end)
                rttm_writer.write(rttm_str.format(reco_id, cur_start, cur_end-cur_start, spk_id))
                segments_writer.write(segments_str.format(seg_id, reco_id, cur_start, cur_end))
                utt2spk_writer.write("{} {}\n".format(seg_id, seg_id))


def write_rttm_forced(segments, out_path, min_length):
    reco_and_spk_to_segs = defaultdict(list,
        {uid : list(g) for uid, g in groupby(segments, lambda x: (x.reco_id,x.spk_id))})
    rttm_str = "SPEAKER {0} 1 {1:7.3f} {2:7.3f} <NA> <NA> {3} <NA> <NA>\n"
    segments_str = "{0} {1} {2:7.3f} {3:7.3f}\n"
    with open(out_path+'/rttm.forced','w') as rttm_writer:
        for uid in sorted(reco_and_spk_to_segs):
            segs = sorted(reco_and_spk_to_segs[uid], key=lambda x: x.start_time)
            reco_id, spk_id = uid
            cur_start = segs[0].start_time
            cur_end = segs[0].end_time
            for seg in segs[1:]:
                # First flush existing seg if it satisfies minimum length
                if (cur_end - cur_start >= min_length):
                    rttm_writer.write(rttm_str.format(reco_id, cur_start, cur_end-cur_start, spk_id))
                    cur_start = seg.start_time
                    cur_end = seg.end_time
                else:
                    cur_end = seg.end_time
            
            # Flush last remaining segment
            if (cur_end - cur_start >= min_length):
                rttm_writer.write(rttm_str.format(reco_id, cur_start, cur_end-cur_start, spk_id))

def write_rttm_spk(df_vad, output_path):
    file_path = output_path + '/rttm.annotation'
    df_vad[['segment_type', 'file_id', 'chnl',
            'tbeg','tdur','ortho', 'stype',
            'name', 'conf', 'slat']].to_csv(
                file_path, sep=' ', float_format='%.3f',
                index=False, header=False)

def make_diar_data(wav_path, output_path, partition, split_file, min_length, rttm_path, uem_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print ('read official split file')
    file_list = read_split(split_file)[partition]

    print('read rttm')
    rttm = read_rttm(rttm_path, uem_path, file_list)

    print('read audios')
    df_wav = find_audios(wav_path, file_list)
    
    print('make wav.scp')
    df_wav = filter_wavs(df_wav, file_list)
    write_wav(df_wav, output_path)

    print('write annotation rttm')
    write_rttm_spk(rttm, output_path)

    print('getting words and alignments for the files')
    segments = get_word_segments(df_wav, os.path.join(wav_path, "ami_public_auto_1.5.1/ASR/ASR_AS_CTM_v1.0_feb07"))
    print('write force-aligned rttm')
    write_rttm_forced(segments, output_path, min_length)




if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Prepare AMI dataset for diarization')

    parser.add_argument('wav_path', help="Path to AMI corpus dir")
    parser.add_argument('output_path', help="Path to generate data directory")
    parser.add_argument('partition', choices=['train', 'dev', 'test'])
    parser.add_argument('split_file', help="Path to file containing train, dev, test splits")
    parser.add_argument('--min-length', default=0.5, type=float, help="minimum length of segments to create")
    parser.add_argument('--rttm-path', dest='rttm_path', required=True)
    parser.add_argument('--uem-path', dest='uem_path', required=True)
    args=parser.parse_args()
    
    make_diar_data(**vars(args))
