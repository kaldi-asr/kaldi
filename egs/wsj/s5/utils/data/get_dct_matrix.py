#!/usr/bin/env python

# Copyright 2016  Vimal Manohar
# Apache 2.0

from __future__ import print_function
import argparse
import sys

sys.path.insert(0, 'steps')
import libs.common as common_lib


def compute_dct_matrix(K, N, cepstral_lifter=0):
    matrix = np.zeros((K, N))
    # normalizer for X_0
    normalizer = math.sqrt(1.0 / N);
    for j in range(0, N):
        matrix[0, j] = normalizer;
    # normalizer for other elements
    normalizer = math.sqrt(2.0 / N);
    for k in range(1, K):
      for n in range(0, N):
        matrix[k, n] = normalizer * math.cos(math.pi/N * (n + 0.5) * k);

    if cepstral_lifter != 0:
        lifter_coeffs = common_lib.compute_lifter_coeffs(cepstral_lifter, K)
        for k in range(0, K):
            matrix[k, :] = matrix[k, :] * lifter_coeffs[k];

    return matrix


def get_args():
    parser = argparse.ArgumentParser(description="Write DCT/IDCT matrix")
    parser.add_argument("--cepstral-lifter", type=float, default=22.0,
                        help="""Here we need the scaling factor on cepstra in
                        the production of MFCC to cancel out the effect of
                        lifter, e.g. 22.0""")
    parser.add_argument("--num-ceps", type=int,
                        default=13,
                        help="Number of cepstral dimensions")
    parser.add_argument("--num-filters", type=int,
                        default=23,
                        help="Number of mel filters")
    parser.add_argument("--get-idct-matrix", type=str, default=False,
                        choices=["true","false"],
                        help="Get IDCT matrix instead of DCT matrix")
    parser.add_argument("--add-zero-column", type=str, default=True,
                        choices=["true","false"],
                        action=common_lib.StrToBoolAction,
                        help="""Add a column to convert the matrix from a linear
                        transform to affine transform""")
    parser.add_argument("out_file", type=argparse.FileType('w'),
                        help="Output file")

    args = parser.parse_args()

    if args.num_ceps > args.num_filters:
        raise Exception("num-ceps must not be larger than num-filters")

    return args


def main():
    args = get_args()

    if args.get_idct_matrix:
        matrix = common_lib.compute_idct_matrix(
            args.num_ceps, args.num_filters, args.cepstral_lifter)

        if args.add_zero_column:
            matrix = np.append(matrix, np.zeros((args.num_filters,1)), 1)
    else:
        matrix = compute_dct_matrix(args.num_ceps, args.num_filters,
                                    args.cepstral_lifter)
        if args.add_zero_column:
            matrix = np.append(matrix, np.zeros((args.num_ceps,1)), 1)

    print('[ ', file=args.out_file)
    np.savetxt(args.out_file, matrix, fmt='%.6e')
    print(' ]', file=args.out_file)


if __name__ == "__main__":
    main()
