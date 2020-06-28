#!/usr/bin/env python3
# Copyright   2020   Desh Raj
# Apache 2.0.

# This script gets per speaker reference and hypothesis files and
# writes them to disk for scoring. Its purpose is the same as the
# get_perspeaker_output.py script, but it is meant to be used for
# the "ASR before diarization" recipes.

import sys, io, argparse, os
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
import math
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser(
        description="""This script computes concatenated minimum-permutation WERs for
                    the pipeline where ASR is performed before diarization. Since
                    each utterance is assigned a single speaker and there is no
                    lmwt tuning required, this script is simpler and faster than
                    the scoring scripts for the "ASR after diarization" case. As
                    input, it takes the "hyp_text" file which was generated from
                    the CTM, "labels" file which is the output of the
                    diarizer, and the "text" file (reference).""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ref_text", type=str,
                        help="path of reference text file")
    parser.add_argument("hyp_text", type=str,
                        help="path of hypothesis text file")   
    parser.add_argument("labels", type=str,
                        help="path of diarization output file")
    parser.add_argument("out_dir", type=str,
                        help="path of output directory where we store combined text")
    args = parser.parse_args()
    return args

class Utterance:
    def __init__(self, reco_id, spk_id, start_time, end_time, text):
        self.reco_id = reco_id
        self.spk_id = spk_id
        self.start_time = start_time
        self.end_time = end_time
        self.text = text

def get_reference_list(ref_text):
    utts = []
    with open(ref_text, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(maxsplit=1)
            subparts = parts[0].split('_')
            reco_id = '_'.join(subparts[1:4])
            start_time = float(subparts[4])/100
            end_time = float(subparts[5])/100
            utt = Utterance(reco_id, int(subparts[0]), start_time, end_time, parts[1])
            utts.append(utt)
    return utts

def get_hypothesis_list(hyp_text, labels):
    label_dict = {}
    with open(labels, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            utt_id = '-'.join(parts[0].split('-')[:-2])
            label_dict[utt_id] = int(parts[1])
    
    utts = []
    with open(hyp_text, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(maxsplit=1)
            if (len(parts) == 1):
                parts.append("")
            subparts = parts[0].split('-')
            reco_id = '_'.join(subparts[0].split('_')[:-1])
            start_time = float(subparts[1])/100
            end_time = float(subparts[2])/100
            if parts[0] in label_dict: # Segments smaller than 0.5s were not included in diarization
                utt = Utterance(reco_id, label_dict[parts[0]], start_time, end_time, parts[1])
            utts.append(utt)
    return utts

# Helper function to group the list by ref/hyp ids
def groupby(iterable, keyfunc):
    """Wrapper around ``itertools.groupby`` which sorts data first."""
    iterable = sorted(iterable, key=keyfunc)
    for key, group in itertools.groupby(iterable, keyfunc):
        yield key, group

def main():
    args = get_args()

    ref_utts = get_reference_list(args.ref_text)
    hyp_utts = get_hypothesis_list(args.hyp_text, args.labels)

    # Create mappings from reco_id to utterances for both reference and hypothesis
    ref_reco_to_utts = defaultdict(list,
        {reco_id : list(g) for reco_id, g in groupby(ref_utts, lambda x: x.reco_id)})

    hyp_reco_to_utts = defaultdict(list,
        {reco_id : list(g) for reco_id, g in groupby(hyp_utts, lambda x: x.reco_id)})

    # Now compute cpWER for each reco_id
    for reco_id in ref_reco_to_utts:
        print ("Writing per-speaker output files for recording: {}".format(reco_id))
        reco_ref_utts = ref_reco_to_utts[reco_id]
        reco_hyp_utts = hyp_reco_to_utts[reco_id]

        # Create mapping from spk_id to utterances
        ref_spk_to_utts = defaultdict(list,
            {spk_id : list(g) for spk_id, g in groupby(reco_ref_utts, lambda x: x.spk_id)})
        hyp_spk_to_utts = defaultdict(list,
            {spk_id : list(g) for spk_id, g in groupby(reco_hyp_utts, lambda x: x.spk_id)})

        # Combine reference utterances of each speaker sorted by time
        ref_utts_comb = []
        for spk_id in sorted(ref_spk_to_utts):
            utts = sorted(ref_spk_to_utts[spk_id], key=lambda x: x.start_time)
            ref_utts_comb.append(" ".join([utt.text for utt in utts]))

        # Do the same for hypothesis utterances
        hyp_utts_comb = []
        for spk_id in sorted(hyp_spk_to_utts):
            utts = sorted(hyp_spk_to_utts[spk_id], key=lambda x: x.start_time)
            hyp_utts_comb.append(" ".join([utt.text for utt in utts]))

        for i, ref_utt in enumerate(ref_utts_comb):
            out_file = os.path.join(args.out_dir,"ref_{}_{}_comb".format(i, reco_id))
            with open(out_file, 'w') as f:
                f.write("{} {}".format(reco_id, ref_utt))

        for i, hyp_utt in enumerate(hyp_utts_comb):
            out_file = os.path.join(args.out_dir,"hyp_{}_{}_comb".format(i, reco_id))
            with open(out_file, 'w') as f:
                f.write("{} {}".format(reco_id, hyp_utt))    

    return

if __name__=="__main__":
    main()