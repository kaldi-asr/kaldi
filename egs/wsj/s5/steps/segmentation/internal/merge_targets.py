#!/usr/bin/env python

# Copyright 2017  Vimal Manohar
# Apache 2.0

"""
This script merges targets created from multiple sources (systems) into
single targets matrices.

Usage: merge_targets.py [options] <pasted-targets> <out-targets>
 e.g.: paste-feats scp:targets1.scp scp:targets2.scp ark,t:- | merge_targets.py --dim=3 - - | copy-feats ark,t:- ark:-

<pasted-targets> is matrix archive with matrices corresponding to
targets from multiple sources appended together using paste-feats.
The column dimension is num-sources * dim, which dim is specified by --dim
option.
"""

from __future__ import print_function
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
    This script merges targets created from multiple sources (systems) into
    single targets matrices.
    Usage: merge_targets.py [options] <pasted-targets> <out-targets>
     e.g.: paste-feats scp:targets1.scp scp:targets2.scp ark,t:- | merge_targets.py --dim=3 - - | copy-feats ark,t:- ark:-
    """,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--weights", type=str, default="",
                        help="A comma-separated list of weights corresponding "
                        "to each targets source being combined.")
    parser.add_argument("--dim", type=int, default=3,
                        help="Number of columns corresponding to each "
                        "target matrix")
    parser.add_argument("--remove-mismatch-frames", type=str, default=False,
                        choices=["true", "false"],
                        action=common_lib.StrToBoolAction,
                        help="If true, the mismatch frames are removed by "
                        "setting targets to 0 in the following cases:\n"
                        "a) If none of the sources have a column with value "
                        "> 0.5\n"
                        "b) If two sources have columns with value > 0.5, but "
                        "they occur at different indexes e.g. silence prob is "
                        "> 0.5 for the targets from alignment, and speech prob "
                        "> 0.5 for the targets from decoding.")

    parser.add_argument("pasted_targets", type=str,
                        help="Input target matrices with columns appended "
                        "together using paste-feats. Its column dimension is "
                        "num-sources * dim, which dim is specified by --dim "
                        "option.")
    parser.add_argument("out_targets", type=str,
                        help="Output target matrices")

    args = parser.parse_args()

    if args.weights != "":
        args.weights = [float(x) for x in args.weights.split(",")]
        weights_sum = sum(args.weights)
        args.weights = [x / weights_sum for x in args.weights]
    else:
        args.weights = None

    return args


def run(args):
    num_done = 0

    with common_lib.smart_open(args.pasted_targets) as targets_reader, \
            common_lib.smart_open(args.out_targets, 'w') as targets_writer:
        for key, mat in common_lib.read_mat_ark(targets_reader):
            mat = np.matrix(mat)
            if mat.shape[1] % args.dim != 0:
                raise RuntimeError(
                    "For utterance {utt} in {f}, num-columns {nc} "
                    "is not a multiple of dim {dim}"
                    "".format(utt=key, f=args.pasted_targets.name,
                              nc=mat.shape[1], dim=args.dim))
            num_sources = mat.shape[1] / args.dim

            out_mat = np.matrix(np.zeros([mat.shape[0], args.dim]))

            if args.remove_mismatch_frames:
                for n in range(mat.shape[0]):
                    if np.amax(mat[n, :]) > 0.5:
                        # We're confident of the one of the sources.
                        max_idx = np.argmax(mat[n, :])
                        max_val = mat[n, max_idx]

                        best_class = max_idx % args.dim
                        min_val = min([mat[n, i * args.dim + best_class]
                                       for i in range(num_sources)])

                        if min_val < 0.5:
                            out_mat[n, :] = np.zeros([1, args.dim])
                        else:
                            for i in range(num_sources):
                                out_mat[n, :] += (
                                    mat[n, (i * args.dim) : ((i+1) * args.dim)]
                                    * (1.0 if args.weights is None
                                       else args.weights[i]))
                    else:
                        # We're not confident of an index from any of the
                        # sources.
                        out_mat[n, :] = np.zeros([1, args.dim])
            else:
                for i in range(num_sources):
                    out_mat += (
                        mat[:, (i * args.dim) : ((i+1) * args.dim)]
                        * (1.0 if args.weights is None else args.weights[i]))

            common_lib.write_matrix_ascii(targets_writer, out_mat.tolist(),
                                          key=key)
            num_done += 1

    logger.info("Merged {num_done} target matrices"
                "".format(num_done=num_done))

    if num_done == 0:
        raise RuntimeError


def main():
    args = get_args()
    try:
        run(args)
    except Exception:
        raise


if __name__ == '__main__':
    main()
