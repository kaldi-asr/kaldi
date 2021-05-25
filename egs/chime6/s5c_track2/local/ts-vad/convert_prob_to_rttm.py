#!/usr/bin/env python3

# Copyright  2020  Yuri Khokhlov, Ivan Medennikov (STC-innovations Ltd)
# Apache 2.0.

"""This script converts TS-VAD output probabilities to a NIST RTTM file.

The segments file format is:
<segment-id> <recording-id> <start-time> <end-time>
The labels file format is:
<segment-id> <speaker-id>

The output RTTM format is:
<type> <file> <chnl> <tbeg> \
        <tdur> <ortho> <stype> <name> <conf> <slat>
where:
<type> = "SPEAKER"
<file> = <recording-id>
<chnl> = "0"
<tbeg> = start time of segment
<tdur> = duration of segment
<ortho> = "<NA>"
<stype> = "<NA>"
<name> = <speaker-id>
<conf> = "<NA>"
<slat> = "<NA>"
"""


import os
import argparse
import regex as re
import numpy as np
from scipy import signal, ndimage
from kaldiio import ReadHelper


class Segment:
    def __init__(self, begin, end, label):
        self.begin = begin
        self.end = end
        self.label = label

    def length(self):
        return self.end - self.begin


class VadProbSet:
    def __init__(self, vad_rspec, reg_exp):
        data = dict()
        prev = -1
        with ReadHelper(vad_rspec) as reader:
            for utid, prob in reader:
                result = reg_exp.match(utid)
                assert result is not None, 'Wrong utterance ID format: \"{}\"'.format(utid)
                sess_indx = result.group(1)
                spkr = result.group(2)

                result = reg_exp.match(sess_indx)
                assert result is not None, 'Wrong utterance ID format: \"{}\"'.format(sess_indx)
                sess = result.group(1)
                indx = int(result.group(2))

                sess = sess + '-' + spkr

                if sess not in data.keys():
                    assert indx == 1
                    prev = -1
                    data[sess] = list()
                assert indx >= prev
                data[sess].append(prob)
                prev = indx
            reader.close()
        print('  loaded {} sessions'.format(len(data)))
        print('  combining fragments')
        self.data = dict()
        for sess, items in data.items():
            self.data[sess] = np.hstack(items)

    def apply_filter(self, window, threshold, threshold_first):
        for sess in self.data.keys():
            if threshold_first:
                self.data[sess] = np.vectorize(lambda value: 1.0 if value > threshold else 0.0)(self.data[sess]).astype(dtype=np.int32)
                if window > 1:
                    self.data[sess] = signal.medfilt(self.data[sess], window).astype(dtype=np.int32)
            else:
                if window > 1:
                    self.data[sess] = signal.medfilt(self.data[sess], window)
                self.data[sess] = np.vectorize(lambda value: 1.0 if value > threshold else 0.0)(self.data[sess]).astype(dtype=np.int32)

    def convert(self, frame_shift,  min_silence, min_speech, out_rttm):
        min_silence = int(round(min_silence / frame_shift))
        min_speech = int(round(min_speech / frame_shift))
        with open(out_rttm, 'wt', encoding='utf-8') as wstream:
            for sess, prob in self.data.items():
                print('  session: {}  num_frames: {}  duration: {:.2f} hrs'.format(sess, len(prob), len(prob) * frame_shift / 60 / 60))
                segments = list()
                for i, label in enumerate(prob):
                    if (len(segments) == 0) or (segments[-1].label != label):
                        segments.append(Segment(i, i + 1, label))
                    else:
                        segments[-1].end += 1
                if (min_silence > 0) or (min_speech > 0):
                    items = segments
                    segments = list()
                    for segm in items:
                        if len(segments) == 0:
                            segments.append(segm)
                        elif segm.label == segments[-1].label:
                            segments[-1].end = segm.end
                        else:
                            min_length = min_silence if segm.label == 0 else min_speech
                            if segm.length() < min_length:
                                segments[-1].end = segm.end
                            else:
                                segments.append(segm)
                for segm in segments:
                    if segm.label == 1:
                        begin = frame_shift * segm.begin
                        length = frame_shift * segm.length()
                        result = reg_exp.match(sess)
                        assert result is not None, 'Wrong format: \"{}\"'.format(sess)
                        utid = result.group(1)
                        spk = result.group(2)
                        wstream.write('SPEAKER {} 1 {:7.3f} {:7.3f} <NA> <NA> {} <NA> <NA>\n'.format(utid, begin, length, spk))
        wstream.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: convert_prob_to_wa.py <vad-rspec> <rttm>')
    parser.add_argument("--frame_shift", "-s", type=float, default=0.010)
    parser.add_argument("--reg_exp", "-x", type=str, default=r'^(\S+)-(\d+)$')
    parser.add_argument("--window", "-w", type=int, default=1)
    parser.add_argument("--threshold", "-t", type=float, default=0.5)
    parser.add_argument("--threshold_first", "-r", action="store_true")
    parser.add_argument("--min_silence", "-k", type=float, default=0.0)
    parser.add_argument("--min_speech", "-m", type=float, default=0.0)
    parser.add_argument('vad_rspec', type=str)
    parser.add_argument('out_rttm', type=str)
    args = parser.parse_args()

    print('Options:')
    print('  Frame shift in sec:  {}'.format(args.frame_shift))
    print('  Utterance ID regexp: {}'.format(args.reg_exp))
    print('  Med. filter window:  {}'.format(args.window))
    print('  Prob. threshold:     {}'.format(args.threshold))
    print('  Apply thresh. first: {}'.format(args.threshold_first))
    print('  Min silence length:  {}'.format(args.min_silence))
    print('  Min speech length:   {}'.format(args.min_speech))
    print('  VAD rspec:           {}'.format(args.vad_rspec))
    print('  Output rttm file:    {}'.format(args.out_rttm))

    reg_exp = re.compile(args.reg_exp)

    parent = os.path.dirname(os.path.abspath(args.out_rttm))
    if not os.path.exists(parent):
        os.makedirs(parent)

    print('Loading VAD probabilities')
    vad_prob = VadProbSet(args.vad_rspec, reg_exp)

    print('Applying filtering')
    vad_prob.apply_filter(args.window, args.threshold, args.threshold_first)

    print('Writing rttm')
    vad_prob.convert(args.frame_shift, args.min_silence, args.min_speech, args.out_rttm)
