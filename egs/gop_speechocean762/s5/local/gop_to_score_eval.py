# Copyright 2021  Xiaomi Corporation (Author: Junbo Zhang, Yongqing Wang)
# Apache 2.0

# This script does phone-level pronunciation scoring by GOP values.

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
    parser.add_argument('gop_scp', help='Input gop file, in Kaldi scp')
    parser.add_argument('output', help='Output the predicted file')
    sys.stderr.write(' '.join(sys.argv) + "\n")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.model, 'rb') as f:
        model_of = pickle.load(f)

    with open(args.output, 'wt') as f:
        for key, gops in kaldi_io.read_post_scp(args.gop_scp):
            for i, [(ph, gop)] in enumerate(gops):
                ph_key = f'{key}.{i}'
                c, b = model_of[ph]
                score = b + c[0] + c[1] * gop + c[2] * gop * gop
                score = round_score(score, 1)
                f.write(f'{ph_key}\t{score:.1f}\t{ph}\n')


if __name__ == "__main__":
    main()
