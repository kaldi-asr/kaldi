#! /usr/bin/env python
# Copyright   2020  Desh Raj (Johns Hopkins University)
# Apache 2.0.

"""This script uses forced alignments of the Librispeech data
to generate an RTTM file for the simulated LibriCSS data. 
This "new" RTTM file can be used to then obtain new segments
and utt2spk files. We do this because the original Librispeech
utterances often have long silences (which are still considered
speech). These can be harmful when used for training systems
for diarization or overlap detection, etc. The alignment file
must have the following format:
116-288045-0000 1 0.530 0.160 AS"""

import argparse
import itertools
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script uses forced alignments of the Librispeech data
                    to generate an RTTM file for the simulated LibriCSS data. 
                    This "new" RTTM file can be used to then obtain new segments
                    and utt2spk files. We do this because the original Librispeech
                    utterances often have long silences (which are still considered
                    speech). These can be harmful when used for training systems
                    for diarization or overlap detection, etc.""")
    parser.add_argument("--max-pause", type=float, default=0.5,
                        help="Maximum pause between words in a segment")
    parser.add_argument("--extend-time", type=float, default=0,
                        help="Extend segments by this duration on each end")

    parser.add_argument("ctm_file", type=str,
                        help="""Input CTM file.
                        The format of the CTM file is
                        <segment-id> <channel-id> <begin-time> """
                        """<duration> <word>""")
    parser.add_argument("utt2spk", type=str,
                        help="Input utt2spk file")
    parser.add_argument("segments", type=str,
                        help="Input segments file")

    args = parser.parse_args()

    return args

class Word:
    def __init__(self, parts):
        self.utt_id = parts[0]
        self.start_time = float(parts[2])
        self.duration = float(parts[3])
        self.end_time = self.start_time + self.duration
        self.text = parts[4]

class Segment:
    def __init__(self, parts, spkid):
        self.utt_reco_id = parts[0]
        self.reco_id = parts[1]
        self.spk_id = spkid
        self.start_time = float(parts[2])
        self.end_time = float(parts[3])
        self.utt_id = self.utt_reco_id.split('_')[0]

def groupby(iterable, keyfunc):
    """Wrapper around ``itertools.groupby`` which sorts data first."""
    iterable = sorted(iterable, key=keyfunc)
    for key, group in itertools.groupby(iterable, keyfunc):
        yield key, group

def main():
    args = get_args()

    # Read the CTM file and store as a list of Word objects
    ctm_words=[]
    with open(args.ctm_file, 'r') as f:
        for line in f:
            ctm_words.append(Word(line.strip().split()))
    
    # Group the list into a dictionary indexed by reco id
    utt_to_words = defaultdict(list,
        {uid : list(g) for uid, g in groupby(ctm_words, lambda x: x.utt_id)})

    segments = []
    with open(args.segments, 'r') as f1, open(args.utt2spk, 'r') as f2:
        for line1, line2 in zip(f1, f2):
            parts = line1.strip().split()
            spkid = line2.strip().split()[1]
            segments.append(Segment(parts, spkid))

    new_segments = [] # [(reco_id, start_time, end_time, spkid)]
    reco_to_segs = defaultdict(list,
        {uid : list(g) for uid, g in groupby(segments, lambda x: x.reco_id)})

    for reco_id in sorted(reco_to_segs):
        segs = sorted(reco_to_segs[reco_id], key=lambda x: x.start_time)

        for seg in segs:
            subsegment_idx = 0
            words = sorted(utt_to_words[seg.utt_id], key=lambda x: x.start_time)
            cur_start = words[0].start_time
            cur_end = words[0].end_time
            for word in words[1:]:
                if (word.start_time > cur_end + args.max_pause):
                    # flush current subsegment
                    subsegment_start = max(0, seg.start_time + cur_start - args.extend_time)
                    subsegment_end = min(seg.start_time + cur_end + args.extend_time, seg.end_time)
                    utt_id = "{}-{}_{}".format(seg.utt_id, subsegment_idx, reco_id)
                    new_segments.append((reco_id, subsegment_start, subsegment_end, seg.spk_id))

                    # start new segment and text
                    cur_start = word.start_time
                    cur_end = word.end_time
                    subsegment_idx += 1

                else:
                    # extend running segment and text
                    cur_end = word.end_time

            # flush last remaining segment
            subsegment_start = max(0, seg.start_time + cur_start - args.extend_time)
            subsegment_end = min(seg.start_time + cur_end + args.extend_time, seg.end_time)
            utt_id = "{}_{}".format(seg.spk_id, reco_id)
            new_segments.append((reco_id,  subsegment_start, subsegment_end, seg.spk_id))

    rttm_str = "SPEAKER {0} 1 {1:7.3f} {2:7.3f} <NA> <NA> {3} <NA> <NA>"
    for segment in sorted(new_segments, key=lambda x: (x[0], x[1])):
        print(rttm_str.format(segment[0], segment[1], segment[2] - segment[1], segment[3]))

if __name__ == '__main__':
    main()
