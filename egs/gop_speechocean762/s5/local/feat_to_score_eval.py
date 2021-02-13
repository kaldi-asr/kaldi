# Copyright 2021  Xiaomi Corporation (Author: Junbo Zhang, Yongqing Wang)
# Apache 2.0

# This script does phone-level pronunciation scoring by GOP-based features.

import sys
import argparse
import pickle
import kaldi_io
import numpy as np
from utils import round_score


def get_args():
    parser = argparse.ArgumentParser(
        description='Phone-level scoring.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', help='Input the model file')
    parser.add_argument('feature_scp',
                        help='Input gop-based feature file, in Kaldi scp')
    parser.add_argument('output', help='Output the predicted file')
    sys.stderr.write(' '.join(sys.argv) + "\n")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.model, 'rb') as f:
        model_of = pickle.load(f)

    feats_for_phone = {}
    idxs_for_phone = {}
    for ph_key, feat in kaldi_io.read_vec_flt_scp(args.feature_scp):
        ph = int(feat[0])
        feats_for_phone.setdefault(ph, []).append(feat[1:])
        idxs_for_phone.setdefault(ph, []).append(ph_key)

    with open(args.output, 'wt') as f:
        for ph in feats_for_phone:
            feats = np.array(feats_for_phone[ph])
            scores = model_of[ph].predict(feats)
            for ph_key, score in zip(idxs_for_phone[ph], list(scores)):
                score = round_score(score, 1)
                f.write(f'{ph_key}\t{score:.1f}\t{ph}\n')


if __name__ == "__main__":
    main()
