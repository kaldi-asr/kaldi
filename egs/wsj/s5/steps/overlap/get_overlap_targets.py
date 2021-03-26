#!/usr/bin/env python

# Copyright 2020  Desh Raj (Johns Hopkins University)
# Apache 2.0

# This script prepares targets for whole recordings for training
# an overlap detector system. It just takes as input a data dir
# where it is assumed that MFCC features have already been
# extracted. It also takes an overlap RTTM file containing
# "single" and "overlap" segments, ideally generated using the
# get_overlap_segments.py script. It uses these segments to 
# obtain per-frame targets for the recordings in the format:
# [ silence single overlap ]

from __future__ import division

import argparse
import logging
import numpy as np
import subprocess
import sys
import itertools
from collections import defaultdict

sys.path.insert(0, 'steps')
import libs.common as common_lib


def get_args():
    parser = argparse.ArgumentParser(
        description="""This script prepares targets for whole recordings for training
            an overlap detector system. It just takes as input a data dir
            where it is assumed that MFCC features have already been
            extracted. It also takes an overlap RTTM file containing
            "single" and "overlap" segments, ideally generated using the
            get_overlap_segments.py script. It uses these segments to 
            obtain per-frame targets for the recordings in the format:
            [ silence single overlap ]
        """)

    parser.add_argument("--frame-shift", type=float, default=0.01,
                        help="Frame shift value in seconds")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Value between 0 and 1. Amount of label smoothing to apply"
                        "to get soft labels instead of one-hot labels")
    parser.add_argument("reco2num_frames", type=str,
                        help="""The number of frames per reco
                        is used to determine the num-rows of the output matrix
                        """)
    parser.add_argument("overlap_rttm", type=str,
                        help="Input RTTM file containing single and overlap segments")
    parser.add_argument("out_targets_ark", type=str,
                        help="""Output archive to which the
                        recording-level matrix will be written in text
                        format""")

    args = parser.parse_args()

    if args.frame_shift < 0.0001 or args.frame_shift > 1:
        raise ValueError("--frame-shift should be in [0.0001, 1]; got {0}"
                         "".format(args.frame_shift))
    return args

class Segment:
    """Stores all information about a segment"""
    reco_id = ''
    spk_id = ''
    start_time = 0
    dur = 0
    end_time = 0

    def __init__(self, reco_id, start_time, dur = None, end_time = None, label = None):
        self.reco_id = reco_id
        self.start_time = start_time
        if (dur is None):
            self.end_time = end_time
            self.dur = end_time - start_time
        else:
            self.dur = dur
            self.end_time = start_time + dur
        self.label = label

def groupby(iterable, keyfunc):
    """Wrapper around ``itertools.groupby`` which sorts data first."""
    iterable = sorted(iterable, key=keyfunc)
    for key, group in itertools.groupby(iterable, keyfunc):
        yield key, group

def run(args):
    # Get all reco to num_frames, which will be used to decide the number of
    # rows of matrix
    reco2num_frames = {}
    with common_lib.smart_open(args.reco2num_frames) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                raise ValueError("Could not parse line {0}".format(line))
            reco2num_frames[parts[0]] = int(parts[1])

    # We read all segments and store as a list of objects
    segments = []
    with common_lib.smart_open(args.overlap_rttm) as f:
        for line in f.readlines():
            parts = line.strip().split()
            segments.append(Segment(parts[1], float(parts[3]), dur=float(parts[4]), label=parts[7]))

    # We group the segment list into a dictionary indexed by reco_id
    reco2segs = defaultdict(list,
        {reco_id : list(g) for reco_id, g in groupby(segments, lambda x: x.reco_id)})

    # Now, for each reco, create a matrix of shape num_frames x 3 and fill in using
    # the segments information for that reco
    reco2targets = {}
    for reco_id in reco2num_frames:
        segs = sorted(reco2segs[reco_id], key=lambda x: x.start_time)

        target_val = 1 - args.label_smoothing
        other_val = args.label_smoothing / 2
        silence_vec = np.array([target_val,other_val,other_val], dtype=np.float)
        single_vec = np.array([other_val,target_val,other_val], dtype=np.float)
        overlap_vec = np.array([other_val,other_val,target_val], dtype=np.float)
        num_targets = [0,0,0]

        # The default target (if not single or overlap) is silence
        targets_mat = np.tile(silence_vec, (reco2num_frames[reco_id],1))

        # Now iterate over all segments of the recording and assign targets
        for seg in segs:
            start_frame = int(seg.start_time / args.frame_shift)
            end_frame = min(int(seg.end_time / args.frame_shift), reco2num_frames[reco_id])
            num_frames = end_frame - start_frame
            if (num_frames <= 0):
                continue
            if (seg.label == "overlap"):
                targets_mat[start_frame:end_frame] = np.tile(overlap_vec, (num_frames,1))
                num_targets[2] += end_frame - start_frame
            else:
                targets_mat[start_frame:end_frame] = np.tile(single_vec, (num_frames,1))
                num_targets[1] += end_frame - start_frame

        num_targets[0] = reco2num_frames[reco_id] - sum(num_targets)
        # print ("{}: {}".format(reco_id, num_targets))
        reco2targets[reco_id] = targets_mat

    with common_lib.smart_open(args.out_targets_ark, 'w') as f:
        for reco_id in sorted(reco2targets.keys()):
            common_lib.write_matrix_ascii(f, reco2targets[reco_id].tolist(), key=reco_id)

def main():
    args = get_args()
    try:
        run(args)
    except Exception:
        raise

if __name__ == "__main__":
    main()