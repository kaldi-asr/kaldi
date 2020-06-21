#! /usr/bin/env python3
# Copyright   2020   Desh Raj
# Apache 2.0.
"""This script takes an RTTM file and spits out all
the overlapping segments in an RTTM-like file"""

import argparse, os
import itertools
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script takes an RTTM file and spits out all
                    the overlapping segments in an RTTM-like file""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--label", type=str, default="overlap",
                        help="Label for the segments")
    parser.add_argument("input_rttm", type=str,
                        help="path of input rttm file")
    parser.add_argument("output_rttm", type=str,
                        help="path of output rttm file")
    args = parser.parse_args()
    return args

class Segment:
    """Stores all information about a segment"""
    reco_id = ''
    spk_id = ''
    start_time = 0
    dur = 0
    end_time = 0

    def __init__(self, reco_id, start_time, dur = None, end_time = None, spk_id = None):
        self.reco_id = reco_id
        self.start_time = start_time
        if (dur is None):
            self.end_time = end_time
            self.dur = end_time - start_time
        else:
            self.dur = dur
            self.end_time = start_time + dur
        self.spk_id = spk_id

def groupby(iterable, keyfunc):
    """Wrapper around ``itertools.groupby`` which sorts data first."""
    iterable = sorted(iterable, key=keyfunc)
    for key, group in itertools.groupby(iterable, keyfunc):
        yield key, group

def find_overlapping_segments(segs):
    reco_id = segs[0].reco_id
    tokens = []
    for seg in segs:
        tokens.append(("BEG", seg.start_time))
        tokens.append(("END", seg.end_time))
    sorted_tokens = sorted(tokens, key=lambda x: x[1])
    
    overlap_segs = []
    spkr_count = 0
    ovl_begin = 0
    ovl_end = 0
    for token in sorted_tokens:
        if (token[0] == "BEG"):
            spkr_count +=1
            if (spkr_count == 2):
                ovl_begin = token[1]
        else:
            spkr_count -= 1
            if (spkr_count == 1):
                ovl_end = token[1]
                overlap_segs.append(Segment(reco_id, ovl_begin, end_time=ovl_end))
    
    return overlap_segs

def main():
    args = get_args()

    # First we read all segments and story as a list of objects
    segments = []
    with open(args.input_rttm, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            segments.append(Segment(parts[1], float(parts[3]), dur=float(parts[4]), spk_id=parts[7]))

    # We group the segment list into a dictionary indexed by reco_id
    reco2segs = defaultdict(list,
        {reco_id : list(g) for reco_id, g in groupby(segments, lambda x: x.reco_id)})

    overlap_segs = []
    for reco_id in reco2segs.keys():
        segs = reco2segs[reco_id]
        overlap_segs.extend(find_overlapping_segments(segs))
    
    rttm_str = "SPEAKER {0} 1 {1:7.3f} {2:7.3f} <NA> <NA> {3} <NA> <NA>\n"
    with open(args.output_rttm, 'w') as f:
        for seg in overlap_segs:
            f.write(rttm_str.format(seg.reco_id, seg.start_time, seg.dur, args.label))


if __name__ == '__main__':
    main()
