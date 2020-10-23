#! /usr/bin/env python3
# Copyright   2020   Desh Raj
# Apache 2.0.
"""This script takes an input RTTM and transforms it in a 
particular way: all overlapping segments are re-labeled 
as "overlap". This is useful for 2 cases: 
1. By retaining just the overlap segments (grep overlap),
the resulting RTTM can be used to train an overlap
detector.
2. By retaining just the non-overlap segments (grep -v overlap),
the resulting file can be used to obtain (fairly) clean 
speaker embeddings from the single-speaker regions of the
recording.
The output is written to stdout.
"""

import argparse, os
import itertools
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script filters an RTTM in several ways.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--label", type=str, default="overlap",
                        help="Label for the overlap segments")
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

def find_overlapping_segments(segs, label):
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
                overlap_segs.append(Segment(reco_id, ovl_begin, end_time=ovl_end, spk_id=label))
    
    return overlap_segs

def find_single_speaker_segments(segs):
    reco_id = segs[0].reco_id
    tokens = []
    for seg in segs:
        tokens.append(("BEG", seg.start_time, seg.spk_id))
        tokens.append(("END", seg.end_time, seg.spk_id))
    sorted_tokens = sorted(tokens, key=lambda x: x[1])
    
    single_speaker_segs = []
    running_spkrs = set()
    for token in sorted_tokens:
        if (token[0] == "BEG"):
            running_spkrs.add(token[2])
            if (len(running_spkrs) == 1):
                seg_begin = token[1]
                cur_spkr = token[2]
            elif (len(running_spkrs) == 2):
                single_speaker_segs.append(Segment(reco_id, seg_begin, end_time=token[1], spk_id=cur_spkr))
        elif (token[0] == "END"):
            try:
                running_spkrs.remove(token[2])
            except:
                Warning ("Speaker not found")
            if (len(running_spkrs) == 1):
                seg_begin = token[1]
                cur_spkr = list(running_spkrs)[0]
            elif (len(running_spkrs) == 0):
                single_speaker_segs.append(Segment(reco_id, seg_begin, end_time=token[1], spk_id=cur_spkr))
    
    return single_speaker_segs

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

    overlap_segs = []
    for reco_id in reco2segs.keys():
        segs = reco2segs[reco_id]
        overlap_segs.extend(find_overlapping_segments(segs, args.label))

    single_speaker_segs = []
    for reco_id in reco2segs.keys():
        segs = reco2segs[reco_id]
        single_speaker_segs.extend(find_single_speaker_segments(segs))
    final_segs = sorted(overlap_segs + single_speaker_segs, key = lambda x: (x.reco_id, x.start_time))
    
    rttm_str = "SPEAKER {0} 1 {1:7.3f} {2:7.3f} <NA> <NA> {3} <NA> <NA>"
    for seg in final_segs:
        if (seg.dur > 0):
            print(rttm_str.format(seg.reco_id, seg.start_time, seg.dur, seg.spk_id))


if __name__ == '__main__':
    main()
