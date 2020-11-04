#! /usr/bin/env python
# Copyright   2020  Desh Raj (Johns Hopkins University)
# Apache 2.0.

"""This script uses forced alignments of the AMI training data
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

    args = parser.parse_args()

    return args

class Word:
    def __init__(self, parts):
        seg_id = parts[0]
        spk_reco_id, seg_start, _ = seg_id.rsplit('-', 2)
        self.spk_id, self.reco_id = spk_reco_id.split('_')
        self.start_time = float(seg_start)/100 + float(parts[2])
        self.duration = float(parts[3])
        self.end_time = self.start_time + self.duration
        self.text = parts[4]

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
    reco_and_spk_to_words = defaultdict(list,
        {reco_id : list(g) for reco_id, g in groupby(ctm_words, lambda x: (x.reco_id,x.spk_id))})

    new_segments = [] # [(reco_id, start_time, end_time, spkid)]

    for uid in sorted(reco_and_spk_to_words):
        reco_id, spk_id = uid
        words = sorted(reco_and_spk_to_words[uid], key=lambda x: x.start_time)

        cur_start = words[0].start_time
        cur_end = words[0].end_time
        for word in words[1:]:
            if (word.start_time > cur_end + args.max_pause):
                # flush current segment
                segment_start = max(0, cur_start - args.extend_time)
                segment_end = cur_end + args.extend_time
                new_segments.append((reco_id, segment_start, segment_end, spk_id))

                # start new segment and text
                cur_start = word.start_time
                cur_end = word.end_time

            else:
                # extend running segment and text
                cur_end = word.end_time

        # flush last remaining segment
        segment_start = max(0, cur_start - args.extend_time)
        segment_end = cur_end + args.extend_time
        new_segments.append((reco_id, segment_start, segment_end, spk_id))

    rttm_str = "SPEAKER {0} 1 {1:7.3f} {2:7.3f} <NA> <NA> {3} <NA> <NA>"
    for segment in sorted(new_segments, key=lambda x: (x[0], x[1])):
        print(rttm_str.format(segment[0], segment[1], segment[2] - segment[1], segment[3]))

if __name__ == '__main__':
    main()
