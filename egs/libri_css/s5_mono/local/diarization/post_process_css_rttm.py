#! /usr/bin/env python3
# Copyright   2020   Desh Raj
# Apache 2.0.
"""This script takes an RTTM file and removes same-speaker segments
which may be present at the same time across streams. This is meant
to be used as a post-processing step after performing clustering-based
diarization on top of separated streams of audio. The idea is to
eliminate false alarms caused by leakage, since the separation
method may not be perfect."""

import argparse, os
import itertools
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script takes an RTTM file and removes same-speaker segments
                which may be present at the same time across streams. This is meant
                to be used as a post-processing step after performing clustering-based
                diarization on top of separated streams of audio. The idea is to
                eliminate false alarms caused by leakage, since the separation
                method may not be perfect.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_rttm", type=str,
                        help="path of input rttm file")
    parser.add_argument("output_rttm", type=str,
                        help="path of output rttm file")
    args = parser.parse_args()
    return args

class Segment:
    def __init__(self, parts):
        self.reco_id = '_'.join(parts[1].split('_')[:-1])
        self.stream = int(parts[1].split('_')[-1])
        self.start_time = float(parts[3])
        self.duration = float(parts[4])
        self.end_time = self.start_time + self.duration
        self.label = int(parts[7])


def main():
    args = get_args()

    # First we read all segments and store as a list of objects
    segments = []
    with open(args.input_rttm,'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            segments.append(Segment(parts))

    groupfn = lambda x: (x.reco_id,x.label)
    sort(segments, key=groupfn)
    # We group the segment list into a dictionary indexed by (reco_id, spk_id)
    reco_and_spk_to_segs = defaultdict(list,
        {uid : list(g) for uid, g in itertools.groupby(segments, groupfn)})

    reco_and_spk_to_final_segs = {}
    for uid in reco_and_spk_to_segs.keys():
        reco_id, spk_id = uid
        segs = reco_and_spk_to_segs[uid]
        tokens = []
        for seg in segs:
            tokens.append(('BEG',seg.start_time,seg.stream))
            tokens.append(('END',seg.end_time,seg.stream))
        tokens.sort(key=lambda x:x[1])
        
        # Remove segments which lie completely inside another segment
        running_segs = {}
        new_segs = [] # (start_time, end_time, stream)
        for token in tokens:
            if token[0] == 'BEG':
                running_segs[token[2]] = token[1]
            else:
                seg_start = running_segs[token[2]]
                seg_end = token[1]
                seg_stream = token[2]
                new_seg = (seg_start, seg_end, seg_stream)
                del running_segs[token[2]]

                # if this segment was the only running segment, then append
                if len(running_segs) == 0:
                    new_segs.append(new_seg)
                    continue
                
                # if any running segment started before this one, it means, this
                # segment is totally enclosed within the other, so we don't add it
                if not any(i < new_seg[0] for i in running_segs.values()):
                    new_segs.append(new_seg)
        
        new_segs.sort(key=lambda x: x[0])
        num_segs = len(new_segs)
        # Now we have partially overlapping segments. We divide the overlapping
        # portion equally.
        final_segs = [] # (start_time, end_time, stream)
        for i in range(num_segs):
            seg = new_segs[i]
            # If it is last segment in recording or last contiguous segment, add it to new_segs
            if (i == num_segs-1 or seg[1] <= new_segs[i+1][0]):
                final_segs.append(seg)
            # Otherwise split overlapping interval between current and next segment
            else:
                avg = (new_segs[i+1][0] + seg[1]) / 2
                final_segs.append((seg[0], avg, seg[2]))
                if not (avg < new_segs[i+1][1]):
                    print (reco_id, spk_id, seg, new_segs[i+1])
                new_segs[i+1] = (avg, new_segs[i+1][1], new_segs[i+1][2])
                new_segs[i+1:].sort(key=lambda x: x[0])
        reco_and_spk_to_final_segs[(reco_id, spk_id)] = final_segs
    
    rttm_str = "SPEAKER {0} 1 {1:7.3f} {2:7.3f} <NA> <NA> {3} <NA> <NA>\n"
    with open(args.output_rttm, 'w') as f:
        for (reco_id, spk_id) in sorted(reco_and_spk_to_final_segs):
            segs = reco_and_spk_to_final_segs[(reco_id, spk_id)]
            for seg in segs:
                utt_id = "{}_{}".format(reco_id, seg[2])
                dur = seg[1] - seg[0]
                if dur > 0.025:
                    f.write(rttm_str.format(utt_id, seg[0], dur, spk_id))

if __name__ == '__main__':
    main()
