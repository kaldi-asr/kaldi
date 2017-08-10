#!/usr/bin/env python

# Copyright 2017  Vimal Manohar
# Apache 2.0

"""
This script reads a Kaldi text archive of matrices from 'targets_in_ark' (e.g.
'-' for standard input), modifies them by subsampling them, and writes the
modified archive to 'targets_out_ark'.
This form of 'subsampling' is similar to taking every n'th frame (specifically:
every n'th row), except that we average over blocks of size 'n' instead of
taking every n'th element.
Thus, this script is similar to the binary 'subsample-feats' except that
it subsamples by averaging.
"""

import argparse
import logging
import numpy as np
import sys

sys.path.insert(0, 'steps')
import libs.common as common_lib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    parser = argparse.ArgumentParser(
        description="""
This script reads a Kaldi text archive of matrices from 'targets_in_ark' (e.g.
'-' for standard input), modifies them by subsampling them, and writes the
modified archive to 'targets_out_ark'.
This form of 'subsampling' is similar to taking every n'th frame (specifically:
every n'th row), except that we average over blocks of size 'n' instead of
taking every n'th element.
Thus, this script is similar to the binary 'subsample-feats' except that
it subsamples by averaging.""")

    parser.add_argument("--subsampling-factor", type=int, default=1,
                        help="The sampling rate is scaled by this factor")
    parser.add_argument("--verbose", type=int, default=0, choices=[0,1,2],
                        help="Verbose level")

    parser.add_argument("targets_in_ark", type=argparse.FileType('r'),
                        help="Input targets archive")
    parser.add_argument("targets_out_ark", type=argparse.FileType('w'),
                        help="Output targets archive")

    args = parser.parse_args()

    if args.subsampling_factor < 1:
        raise ValueError("Invalid --subsampling-factor value {0}".format(
                            args.subsampling_factor))

    if args.verbose >= 2:
        logger.setLevel(logging.DEBUG)
        handler.setLevel(logging.DEBUG)

    return args


def run(args):
    num_utts = 0
    for key, mat in common_lib.read_mat_ark(args.targets_in_ark):
        mat = np.matrix(mat)
        if args.subsampling_factor > 0:
            num_indexes = ((mat.shape[0] + args.subsampling_factor - 1)
                            / args.subsampling_factor)

        out_mat = np.zeros([num_indexes, mat.shape[1]])
        i = 0
        for k in range(int(args.subsampling_factor / 2.0),
                       mat.shape[0], args.subsampling_factor):
            st = int(k - float(args.subsampling_factor) / 2.0)
            end = int(k + float(args.subsampling_factor) / 2.0)

            if st < 0:
                st = 0
            if end > mat.shape[0]:
                end = mat.shape[0]

            try:
                out_mat[i, :] = np.sum(mat[st:end, :], axis=0) / float(end - st)
            except IndexError:
                logger.error("mat.shape = {0}, st = {1}, end = {2}"
                             "".format(mat.shape, st, end))
                raise
            assert i == k / args.subsampling_factor
            i += 1

        common_lib.write_matrix_ascii(args.targets_out_ark, out_mat, key=key)
        num_utts += 1
    args.targets_in_ark.close()
    args.targets_out_ark.close()

    logger.info("Sub-sampled {num_utts} target matrices"
                "".format(num_utts=num_utts))


def main():
    args = get_args()
    try:
        run(args)
    except Exception as e:
        logger.error("Script failed; traceback = ", exc_info=True)
        raise SystemExit(1)
    finally:
        for f in [args.targets_in_ark, args.targets_out_ark]:
            if f is not None:
                f.close()


if __name__ == "__main__":
    main()
