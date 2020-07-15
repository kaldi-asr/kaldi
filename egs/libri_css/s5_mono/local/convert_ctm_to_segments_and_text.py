#! /usr/bin/env python
# Copyright   2020  Desh Raj (Johns Hopkins University)
# Apache 2.0.

"""This script converts a CTM file to a segments file and a 
Kaldi text file. It is meant to be used in recipes where
ASR is performed before diarization."""

import argparse
import itertools
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script converts a CTM file to a segments file and a 
                    Kaldi text file. It is meant to be used in recipes where
                    ASR is performed before diarization.""")
    parser.add_argument("--max-pause", type=float, default=0.25,
                        help="Maximum pause between words in a segment")
    parser.add_argument("--extend-time", type=float, default=0,
                        help="Extend segments by this duration on each end")
    parser.add_argument("--min-segment-length", type=float, default=0.025,
                        help="minimum length for a segment")
    parser.add_argument("--max-segment-length", type=float, default=10,
                        help="maximum length for a segment")
    parser.add_argument("--min-cf", type=float, default=0.5,
                        help="minimum confidence of word to include in text")

    parser.add_argument("ctm_file", type=str,
                        help="""Input CTM file.
                        The format of the CTM file is
                        <segment-id> <channel-id> <begin-time> """
                        """<duration> <word> <conf>""")
    parser.add_argument("segments", type=str,
                        help="Output segments file")
    parser.add_argument("text", type=str,
                        help="Output text file")

    args = parser.parse_args()

    return args

class Word:
    def __init__(self, parts):
        subparts = parts[0].split('_')
        self.reco_id = '_'.join(subparts[:-2])
        seg_start = float(subparts[-2])/100
        self.start_time = seg_start + float(parts[2])
        self.duration = float(parts[3])
        self.end_time = self.start_time + self.duration
        self.text = parts[4]
        self.conf = float(parts[5])

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
        for line in f.readlines():
            ctm_words.append(Word(line.strip().split()))
    
    # Group the list into a dictionary indexed by reco id
    reco_to_words = defaultdict(list,
        {uid : list(g) for uid, g in groupby(ctm_words, lambda x: x.reco_id)})

    segments = []   # [(reco_id, utt_id, start_time, end_time)]
    text = []       # [(utt_id, text)]
    for reco_id in sorted(reco_to_words):
        words = sorted(reco_to_words[reco_id], key=lambda x: x.start_time)

        cur_start = words[0].start_time
        cur_end = words[0].end_time
        cur_text = [words[0].text] if words[0].conf >= args.min_cf else []
        for word in words[1:]:
            if (word.start_time > cur_end + args.max_pause) \
                or (word.end_time > cur_start + args.max_segment_length):
                # flush current segment and text
                cur_start = max(0, cur_start - args.extend_time)
                cur_end = cur_end + args.extend_time
                utt_id = "{}_{}_{}".format(reco_id, "{:.0f}".format(100*cur_start).zfill(6),
                    "{:.0f}".format(100*cur_end).zfill(6))
                if (cur_end - cur_start >= args.min_segment_length):
                    segments.append((reco_id, utt_id, cur_start, cur_end))
                    text.append((utt_id, " ".join(cur_text)))

                # start new segment and text
                cur_start = word.start_time
                cur_end = word.end_time
                cur_text = [word.text] if word.conf >= args.min_cf else []

            else:
                # extend running segment and text
                cur_end = word.end_time
                if word.conf >= args.min_cf:
                    cur_text.append(word.text)

        # flush last remaining segment and text
        cur_start = max(0, cur_start - args.extend_time)
        cur_end = cur_end + args.extend_time
        utt_id = "{}_{}_{}".format(reco_id, "{:.0f}".format(100*cur_start).zfill(6),
            "{:.0f}".format(100*cur_end).zfill(6))
        if (cur_end - cur_start >= args.min_segment_length):
            segments.append((reco_id, utt_id, cur_start, cur_end))
            text.append((utt_id, " ".join(cur_text)))        

    text_writer = open(args.text, 'w')
    segments_writer = open(args.segments, 'w')

    for utt in text:
        text_writer.write("{}\n".format(' '.join(utt)))

    for utt in segments:
        segments_writer.write("{} {} {} {}\n".format(utt[1], utt[0], utt[2], utt[3]))

if __name__ == '__main__':
    main()
