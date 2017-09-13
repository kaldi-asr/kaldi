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
                        "to each targets source being combined. "
                        "Weights will be normalized internally to sum-to-one.")
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


def should_remove_frame(row, dim):
    """Returns True if the frame needs to be removed.

    Input:
        row -- a list of values (of dimension num-sources x dim) corresponding
               to the targets for one of the frames
        dim -- Usually 3. The number of sources can be computed as the
               len(row) / dim.

    The frame is determined to be removed in the following cases:
        1) None of the values > 0.5.
        2) More than one source has best value >= 0.5, but at different
           indexes in the source.
    e.g. [ 1 0 0 0.6 0 0.4 0 0 0 ]   # kept because 1 and 0.6 are both > 0.5
                                     # at the same class namely 0
                                     # source[0] = [ 1 0 0 ]
                                     # source[1] = [ 0.6 0 0.4 ]
                                     # source[2] = [ 0 0 0 ]
    e.g. [ 0 0 0 0.4 0 0.6 1 0 0 ]   # removed because source[1] has best value
                                     # 0.6 > 0.5 at class 2 and source[2] has
                                     # best value 1 > 0.5 at class 0.
                                     # source[0] = [ 0 0 0 ]
                                     # source[1] = [ 0.4 0 0.6 ]
                                     # source[2] = [ 0 0 0 ]
    """
    assert len(row) % dim == 0
    num_sources = len(row) / dim

    max_idx = np.argmax(row)
    max_val = row[max_idx]

    if max_val < 0.5:
        # All the values < 0.5. So we are not confident of any sources.
        # Remove frame.
        return True

    best_source = max_idx / dim
    best_class = max_idx % dim

    confident_in_source = []  # List of length num_sources
                              # Element 'i' is 1,
                              # if the best value for the source 'i' is > 0.5
    best_values_for_source = []  # Element 'i' is a pair (value, class),
                                 # where 'class' is argmax over the scores
                                 # corresponding to the source 'i' and
                                 # 'value' is the corresponding score.
    for source_idx in range(num_sources):
        idx = np.argmax(row[(source_idx * dim):
                            ((source_idx+1) * dim)])
        val = row[source_idx * dim + idx]
        confident_in_source.append(bool(val > 0.5))
        best_values_for_source.append((val, idx))

    if sum(confident_in_source) == 1:
        # We are confident in only one source. Keep frame.
        return False

    for source_idx in range(num_sources):
        if source_idx == best_source:
            assert confident_in_source[source_idx]
            continue
        if not confident_in_source[source_idx]:
            continue
        else:
            # We are confident in a source other than the 'best_source'.
            # If it's index is different from the 'best_class', then it is
            # a mismatch and the frame must be removed.
            val, idx = best_values_for_source[source_idx]
            assert val > 0.5
            if idx != best_class:
                return True
    return False


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
                    if should_remove_frame(mat[n, :].getA()[0], args.dim):
                        out_mat[n, :] = np.zeros([1, args.dim])
                    else:
                        for i in range(num_sources):
                            out_mat[n, :] += (
                                mat[n, (i * args.dim) : ((i+1) * args.dim)]
                                * (1.0 if args.weights is None
                                   else args.weights[i]))
            else:
                # Just interpolate the targets
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
