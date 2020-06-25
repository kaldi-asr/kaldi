#!/usr/bin/env python

# Copyright 2017  Vimal Manohar
# Apache 2.0

"""
This script gets targets for the whole recording
by adding 'default_targets' vector read from file specified by
--default-targets option for the out-of-segments regions and
zeros for all other frames. See steps/segmentation/lats_to_targets.sh
for details about the targets matrix.
By default, the 'default_targets' would be [ 1 0 0 ], which means all
the out-of-segment regions are assumed as silence. But depending, on
the application and data, this could be [ 0 0 0 ] or [ 0 0 1 ] or
something with fractional weights.
"""
from __future__ import division

import argparse
import logging
import numpy as np
import subprocess
import sys

sys.path.insert(0, 'steps')
import libs.common as common_lib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    parser = argparse.ArgumentParser(
        description="""This script gets targets for the whole recording
        by adding 'default_targets' vector read from file specified by
        --default-targets option for the out-of-segments regions and
        zeros for all other frames. See steps/segmentation/lats_to_targets.sh
        for details about the targets matrix.
        By default, the 'default_targets' would be [ 1 0 0 ], which means all
        the out-of-segment regions are assumed as silence. But depending, on
        the application and data, this could be [ 0 0 0 ] or [ 0 0 1 ] or
        something with fractional weights.
        """)

    parser.add_argument("--frame-shift", type=float, default=0.01,
                        help="Frame shift value in seconds")
    parser.add_argument("--default-targets", type=str, default=None,
                        action=common_lib.NullstrToNoneAction,
                        help="Vector of default targets for out-of-segments "
                        "region")
    parser.add_argument("--length-tolerance", type=int, default=2,
                        help="Tolerate length mismatches of this many frames")
    parser.add_argument("--verbose", type=int, default=0, choices=[0,1,2],
                        help="Verbose level")

    parser.add_argument("--reco2num-frames", type=str, required=True,
                        action=common_lib.NullstrToNoneAction,
                        help="""The number of frames per reco
                        is used to determine the num-rows of the output matrix
                        """)
    parser.add_argument("reco2utt", type=str,
                        help="""reco2utt file.
                        The format is <reco> <utt-1> <utt-2> ... <utt-N>""")
    parser.add_argument("segments", type=str,
                        help="Input kaldi segments file")
    parser.add_argument("out_targets_ark", type=str,
                        help="""Output archive to which the
                        recording-level matrix will be written in text
                        format""")

    args = parser.parse_args()

    if args.frame_shift < 0.0001 or args.frame_shift > 1:
        raise ValueError("--frame-shift should be in [0.0001, 1]; got {0}"
                         "".format(args.frame_shift))

    if args.verbose >= 2:
        logger.setLevel(logging.DEBUG)
        handler.setLevel(logging.DEBUG)

    return args


def run(args):
    reco2utt = {}
    with common_lib.smart_open(args.reco2utt) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                raise ValueError("Could not parse line {0}".format(line))
            reco2utt[parts[0]] = parts[1:]

    reco2num_frames = {}
    with common_lib.smart_open(args.reco2num_frames) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                raise ValueError("Could not parse line {0}".format(line))
            if parts[0] not in reco2utt:
                continue
            reco2num_frames[parts[0]] = int(parts[1])

    segments = {}
    with common_lib.smart_open(args.segments) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) not in [4, 5]:
                raise ValueError("Could not parse line {0}".format(line))
            utt = parts[0]
            reco = parts[1]
            if reco not in reco2utt:
                continue
            start_time = float(parts[2])
            end_time = float(parts[3])
            segments[utt] = [reco, start_time, end_time]

    num_utt_err = 0
    num_utt = 0
    num_reco = 0

    if args.default_targets is not None:
        default_targets = np.matrix(common_lib.read_matrix_ascii(args.default_targets))
    else:
        default_targets = np.matrix([[1, 0, 0]])
    assert (np.shape(default_targets)[0] == 1
            and np.shape(default_targets)[1] == 3)

    with common_lib.smart_open(args.out_targets_ark, 'w') as f:
        for reco, utts in reco2utt.items():
            reco_mat = np.repeat(default_targets, reco2num_frames[reco],
                                 axis=0)
            utts.sort(key=lambda x: segments[x][1])   # sort on start time
            for i, utt in enumerate(utts):
                if utt not in segments:
                    num_utt_err += 1
                    continue
                segment = segments[utt]

                start_frame = int(segment[1] / args.frame_shift)
                end_frame = int(segment[2] / args.frame_shift)
                num_frames = end_frame - start_frame

                if end_frame > reco2num_frames[reco]:
                    end_frame = reco2num_frames[reco]
                    num_frames = end_frame - start_frame

                reco_mat[start_frame:end_frame] = np.zeros([num_frames, 3])
                num_utt += 1

            if reco_mat.shape[0] > 0:
                common_lib.write_matrix_ascii(f, reco_mat.tolist(),
                                              key=reco)
                num_reco += 1

    logger.info("Got default out-of-segment targets for {num_reco} recordings "
                "containing {num_utt} in-segment regions; "
                "failed to account {num_utt_err} utterances"
                "".format(num_reco=num_reco, num_utt=num_utt,
                          num_utt_err=num_utt_err))

    if num_utt == 0 or num_utt_err > num_utt // 2 or num_reco == 0:
        raise RuntimeError


def main():
    args = get_args()
    try:
        run(args)
    except Exception:
        raise


if __name__ == "__main__":
    main()
