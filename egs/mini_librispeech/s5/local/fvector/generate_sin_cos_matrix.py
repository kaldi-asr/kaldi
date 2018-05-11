#!/usr/bin/env/python

from __future__ import print_function
import argparse
import logging
import os
import pprint
import shutil
import sys
import traceback

sys.path.insert(0, 'steps')
import libs.common as common_lib

def get_args():
    parser = argparse.ArgumentParser(description="Generate sine_transform.mat "
            "and cosine_transform.mat for frequency domain raw waveform setup.",
            epilog="Called by local/fvector/run_fvector.sh")
    parser.add_argument("--feat-dim", type=int, required=True,
            help="The dimension of input.")
    parser.add_argument("--add-bias", type=str,
            help="If true, add a column for fft matrix.",
            default=True, choices=["True","False"])
    parser.add_argument("--half-range", type=str,
            help="If true, generate half fft matrix.",
            default=True, choices=["True","False"])
    parser.add_argument("--dir", type=str, required=True,
            help="The output directory.")

    print(' '.join(sys.argv), file=sys.stderr)
    print(sys.argv, file=sys.stderr)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    
    feat_dim = args.feat_dim
    num_fft_bins = (2**(args.feat_dim-1).bit_length())
    add_bias = args.add_bias
    half_range = args.half_range

    common_lib.write_sin_cos_transform_matrix(feat_dim, num_fft_bins,
            "{0}/configs/cos_transform.mat".format(args.dir),                   
            compute_cosine=True, add_bias=add_bias, half_range=half_range)
    common_lib.write_sin_cos_transform_matrix(feat_dim, num_fft_bins,
            "{0}/configs/sin_transform.mat".format(args.dir),
            compute_cosine=False, add_bias=add_bias, half_range=half_range)

if __name__ == "__main__":
    main()
