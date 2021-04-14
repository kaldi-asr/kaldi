#!/usr/bin/env python3

# Copyright  2020  Yuri Khokhlov, Ivan Medennikov (STC-innovations Ltd)
# Apache 2.0.

"""This script converts JSON to VAD alignment.
Single-speaker speech is treated as 1, whereas both silence and overlapped speech as 0.
This can be used for excluding overlapping regions from i-vectors estimation"""

import os
import json
import datetime
import argparse
import numpy as np
from pathlib import Path
from kaldiio import WriteHelper

def time_to_seconds(time):
    parts = time.split(':')
    return datetime.timedelta(hours=float(parts[0]), minutes=float(parts[1]), seconds=float(parts[2])).total_seconds()

class Segment:
    def __init__(self, begin, end, label):
        self.begin = begin
        self.end = end
        self.label = label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: make_json_align.py <json-path> <ali-wspec>')
    parser.add_argument("--frequency", "-f", type=int, default=16000)
    parser.add_argument("--frame_len", "-l", type=float, default=0.025)
    parser.add_argument("--frame_shift", "-s", type=float, default=0.010)
    parser.add_argument('json_path', type=str)
    parser.add_argument('ali_wspec', type=str)
    args = parser.parse_args()

    frame_len = int(round(args.frame_len * args.frequency))
    frame_shift = int(round(args.frame_shift * args.frequency))

    print('Options:')
    print('  Sampling frequency:  {}'.format(args.frequency))
    print('  Frame length in sec: {}  ({})'.format(args.frame_len, frame_len))
    print('  Frame shift in sec:  {}  ({})'.format(args.frame_shift, frame_shift))
    print('  Path to the source JSON:   {}'.format(args.json_path))
    print('  Alignment write specifier: {}'.format(args.ali_wspec))

    json_path = Path(args.json_path)
    assert os.path.isfile(args.json_path), 'File does not exist {}'.format(args.json_path)

    print('Loading file {}'.format(json_path))
    with open(str(json_path)) as stream:
        data = json.load(stream)
        stream.close()
    print('  loaded {} segments'.format(len(data)))

    print('Building alignment')
    duration = 0.0
    for reco in data:
        duration = max(time_to_seconds(reco['end_time']), duration)
    print('  session duration {:.2f} hrs'.format(duration / 60 / 60))
    total_frames = (int(duration * args.frequency) - frame_len) // frame_shift + 1
    print('  total number of frames {}'.format(total_frames))
    alignment = np.zeros(total_frames, dtype=np.int32)
    for reco in data:
        start_time = time_to_seconds(reco['start_time'])
        end_time = time_to_seconds(reco['end_time'])
        num_frames = (int((end_time - start_time) * args.frequency) - frame_len) // frame_shift + 1
        start_frame = int(start_time * args.frequency) // frame_shift
        end_frame = min(start_frame + num_frames, total_frames)
        alignment[start_frame: end_frame] += 1
    value = args.frame_shift * np.count_nonzero(alignment == 0)
    print('  out of segments: {:.2f} hrs'.format(value / 60 / 60))
    value = args.frame_shift * np.count_nonzero(alignment == 1)
    print('  single speaker: {:.2f} hrs'.format(value / 60 / 60))
    value = args.frame_shift * np.count_nonzero(alignment > 1)
    print('  overlapped speech: {:.2f} hrs'.format(value / 60 / 60))

    alignment = np.vectorize(lambda nspk: 0 if nspk > 1 else nspk)(alignment)

    print('Writing alignment to {}'.format(args.ali_wspec))
    with WriteHelper(args.ali_wspec) as writer:
        writer(data[0]['session_id'], alignment)
        writer.close()
