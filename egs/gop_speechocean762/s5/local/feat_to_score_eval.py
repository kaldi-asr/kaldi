# Copyright 2021  Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

# This script do phone-level pronunciation scoring by GOP-based features.

import sys
import argparse
import pickle
import kaldi_io
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

    with open(args.output, 'wt') as f:
        for ph_key, feat in kaldi_io.read_vec_flt_scp(args.feature_scp):
            ph = int(feat[0])
            feat = feat[1:].reshape(1, -1)
            score = model_of[ph].predict(feat).reshape(1)[0]
            score = round_score(score, 1)
            f.write(f'{ph_key}\t{score:.1f}\t{ph}\n')


if __name__ == "__main__":
    main()
