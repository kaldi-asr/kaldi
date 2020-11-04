#! /usr/bin/env python3
# Copyright   2020   Desh Raj
# Apache 2.0.
"""This script takes an input RTTM and creates a segments file from it.
Overlapping segments are retained only once.
"""

import argparse, os
import itertools
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script filters an RTTM in several ways.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_rttm", type=str,
                        help="path of input rttm file")
    args = parser.parse_args()
    return args

class Segment:
    """Stores all information about a segment"""

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

def find_extended_segments(segs):
    reco_id = segs[0].reco_id
    tokens = []
    for seg in segs:
        tokens.append(("BEG", seg.start_time))
        tokens.append(("END", seg.end_time))
    sorted_tokens = sorted(tokens, key=lambda x: x[1])

    extended_segs = []
    spkr_count = 0
    for token in sorted_tokens:
        if (token[0] == "BEG"):
            spkr_count +=1
            if (spkr_count == 1):
                seg_begin = token[1]
        else:
            spkr_count -= 1
            if (spkr_count == 0):
                seg_end = token[1]
                extended_segs.append(Segment(reco_id, seg_begin, end_time=seg_end, spk_id="speech"))
    
    return extended_segs

def main():
    args = get_args()

    # First we read all segments and store as a list of objects
    segments = []
    with open(args.input_rttm, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            segments.append(Segment(parts[1], float(parts[3]), dur=float(parts[4]), spk_id=parts[7]))

    # We group the segment list into a dictionary indexed by reco_id
    reco2segs = defaultdict(list,
        {reco_id : list(g) for reco_id, g in groupby(segments, lambda x: x.reco_id)})

    all_segs = []
    for reco_id in reco2segs.keys():
        segs = reco2segs[reco_id]
        all_segs += find_extended_segments(segs)
    
    for seg in sorted(all_segs, key=lambda x: (x.reco_id, x.start_time)):
        segment_id = '%s-%07d-%07d' % (seg.reco_id, int(seg.start_time*100), int(seg.end_time*100))
        if (seg.dur > 0):
            print('%s %s %.2f %.2f' % (segment_id, seg.reco_id, seg.start_time, seg.end_time))


if __name__ == '__main__':
    main()
