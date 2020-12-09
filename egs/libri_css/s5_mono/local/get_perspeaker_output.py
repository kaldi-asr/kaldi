#! /usr/bin/env python3
# Copyright   2020   Desh Raj
# Apache 2.0.
"""This script splits a kaldi output (text) file
  into per_speaker output (text) file"""

import argparse, os
import itertools
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script splits a kaldi text file
        into per_speaker text files""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--affix", type=str,
                        help="Append in front of output file")
    parser.add_argument("--multi-stream", dest='multi_stream', action='store_true',
                        default=False,
                        help="Score with multiple decoding streams e.g. CSS")
    parser.add_argument("input_text", type=str,
                        help="path of text file")
    parser.add_argument("input_utt2spk", type=str,
                        help="path of utt2spk file")
    parser.add_argument("output_dir", type=str,
                        help="Output path for per_session per_speaker reference files")
    args = parser.parse_args()
    return args

class Utterance:
    """Stores all information about an utterance"""
    reco_id = ''
    spk_id = ''
    text = ''
    start_time = 0
    end_time = 0

    def __init__(self, uttid, spkid, text, multi_stream):
        parts = uttid.strip().split('_')
        self.reco_id = '_'.join(parts[1:4])
        if not multi_stream:
            self.start_time = float(parts[4])/100
            self.end_time = float(parts[5])/100
        else:
            self.start_time = float(parts[5])/100
            self.end_time = float(parts[6])/100            
        self.spk_id = spkid
        self.text = text

def main():
    args = get_args()
    utt2spk = {}
    utt_list = []

    # First we read the utt2spk file and create a mapping
    for line in open(args.input_utt2spk):
        uttid, spkid = line.strip().split()
        utt2spk[uttid] = spkid

    # Next we read the input text file and create a list of
    # Utterance class objects
    for line in open(args.input_text):
        parts = line.strip().split(maxsplit=1)
        uttid = parts[0]
        text = "" if len(parts) == 1 else parts[1]
        utterance = Utterance(uttid, utt2spk[uttid], text, args.multi_stream)
        utt_list.append(utterance)

    groupfn = lambda x: (x.reco_id, x.spk_id)
    sort(utt_list, key=groupfn)
    # We group the utterance list into a dictionary indexed by (reco_id, spk_id)
    reco_spk_to_utts = defaultdict(list,
        {uid : list(g) for uid, g in itertools.groupby(utt_list, groupfn)})
    
    # Now for each (reco_id, spk_id) pair, we write the concatenated text to an
    # output  (we assign speaker ids 1,2,3,..)
    for i, uid in enumerate(sorted(reco_spk_to_utts.keys())):
        reco_id = reco_spk_to_utts[uid][0].reco_id
        output_file = os.path.join(args.output_dir, '{}_{}_{}_comb'.format(args.affix, i, reco_id))
        output_writer = open(output_file, 'w')
        utterances = reco_spk_to_utts[uid]

        # We sort all utterances by start time and concatenate.
        sorted_utterances = sorted(utterances, key=lambda x: x.start_time)
        combined_text = ' '.join([utt.text for utt in sorted_utterances])

        output_writer.write("{} {}".format(reco_id, combined_text))
        output_writer.close()

if __name__ == '__main__':
    main()
