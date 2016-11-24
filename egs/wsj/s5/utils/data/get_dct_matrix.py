#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import os, argparse, sys, math, warnings

import numpy as np

def ComputeLifterCoeffs(Q, dim):
    coeffs = np.zeros((dim))
    for i in range(0, dim):
        coeffs[i] = 1.0 + 0.5 * Q * math.sin(math.pi * i / Q);

    return coeffs

def ComputeIDctMatrix(K, N, cepstral_lifter=0):
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
        lifter_coeffs = ComputeLifterCoeffs(cepstral_lifter, K)
        for k in range(0, K):
            matrix[k, :] = matrix[k, :] / lifter_coeffs[k];

    return matrix.T

def ComputeDctMatrix(K, N, cepstral_lifter=0):
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
        lifter_coeffs = ComputeLifterCoeffs(cepstral_lifter, K)
        for k in range(0, K):
            matrix[k, :] = matrix[k, :] * lifter_coeffs[k];

    return matrix

def GetArgs():
    parser = argparse.ArgumentParser(description="Write DCT/IDCT matrix")
    parser.add_argument("--cepstral-lifter", type=float,
                        help="Here we need the scaling factor on cepstra in the production of MFCC"
                        "to cancel out the effect of lifter, e.g. 22.0", default=22.0)
    parser.add_argument("--num-ceps", type=int,
                        default=13,
                        help="Number of cepstral dimensions")
    parser.add_argument("--num-filters", type=int,
                        default=23,
                        help="Number of mel filters")
    parser.add_argument("--get-idct-matrix", type=str, default="false",
                        choices=["true","false"],
                        help="Get IDCT matrix instead of DCT matrix")
    parser.add_argument("--add-zero-column", type=str, default="true",
                        choices=["true","false"],
                        help="Add a column to convert the matrix from a linear transform to affine transform")
    parser.add_argument("out_file", type=str,
                        help="Output file")

    args = parser.parse_args()

    return args

def CheckArgs(args):
    if args.num_ceps > args.num_filters:
        raise Exception("num-ceps must not be larger than num-filters")

    args.out_file_handle = open(args.out_file, 'w')

    return args

def Main():
    args = GetArgs()
    args = CheckArgs(args)

    if args.get_idct_matrix == "false":
        matrix = ComputeDctMatrix(args.num_ceps, args.num_filters,
                                  args.cepstral_lifter)
        if args.add_zero_column == "true":
            matrix = np.append(matrix, np.zeros((args.num_ceps,1)), 1)
    else:
        matrix = ComputeIDctMatrix(args.num_ceps, args.num_filters,
                                   args.cepstral_lifter)

        if args.add_zero_column == "true":
            matrix = np.append(matrix, np.zeros((args.num_filters,1)), 1)

    print('[ ', file=args.out_file_handle)
    np.savetxt(args.out_file_handle, matrix, fmt='%.6e')
    print(' ]', file=args.out_file_handle)

if __name__ == "__main__":
    Main()

