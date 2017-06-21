#! /usr/bin/env python

# Copyright 2017  Vimal Manohar
# Apache 2.0

from __future__ import print_function
import argparse
import logging
import os
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
    This script merges targets dirs created from multiple sources (systems) into
    single targets matrices.
  Usage: steps/segmentation/merge_targets_dirs.py <data> <targets-1> <targets-2> ... <merged-targets>
   e.g.: steps/segmentation/merge_targets_dirs.py --weights 1.0,0.5 \
    data/train_whole \
    exp/segmentation1a/tri3b_train_whole_sup_targets_sub3 \
    exp/segmentation1a/tri3b_train_whole_targets_sub3 \
    exp/segmentation1a/tri3b_train_whole_combined_targets_sub3""",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--nj", type=int, default=4,
                        help="Number of jobs to run parallel")
    parser.add_argument("--cmd", type=str, default="run.pl",
                        help="Command to run jobs")
    parser.add_argument("--weights", type=str, default="",
                        help="A comma-separated list of weights corresponding "
                        "to each targets source being combined.")
    parser.add_argument("--remove-mismatch-frames", type=str, default="false",
                        choices=["true", "false"],
                        help="Remove mismatch frames by setting targets to 0")

    parser.add_argument("data", type=str, help="Data directory")
    parser.add_argument("targets_dirs", type=str, nargs='+',
                        help="Targets directories")
    parser.add_argument("dir", type=str, help="Output targets dir")

    args = parser.parse_args()

    return args


def read_frame_subsampling_factor(dir_, expected_factor=None):
    frame_subsampling_factor = 1
    if os.path.exists("{0}/frame_subsampling_factor".format(dir_)):
        frame_subsampling_factor = int(
            open("{0}/frame_subsampling_factor".format(dir_)).readline())

    if (expected_factor is not None
            and expected_factor != frame_subsampling_factor):
        raise TypeError("frame_subsampling_factor in {dir} ({fsf}) differs from "
                        "expected value {v}".format(
                            dir=dir_, fsf=frame_subsampling_factor,
                            v=expected_factor))
    return frame_subsampling_factor


def run(args):
    sdata = common_lib.split_data(args.data, args.nj, per_utt=True)

    frame_subsampling_factor = read_frame_subsampling_factor(
        args.targets_dirs[0])

    logger.info("Expected frame-subsampling-factor is {0}".format(
        frame_subsampling_factor))

    targets_rspecifiers = []
    for targets_dir in args.targets_dirs:
        read_frame_subsampling_factor(targets_dir,
                                      expected_factor=frame_subsampling_factor)

        targets_rspecifiers.append(
            '"ark,s,cs:utils/filter_scp.pl {sdata}/JOB/utt2spk '
            '{targets_dir}/targets.scp | copy-feats scp:- ark:- |"'
            ''.format(sdata=sdata, targets_dir=targets_dir))

    common_lib.execute_command(
        """{cmd} JOB=1:{nj} {dir}/log/merge_targets.JOB.log \
            paste-feats {targets} ark,t:- \| \
            steps/segmentation/internal/merge_targets.py \
                --weights="{weights}" --remove-mismatch-frames={rmf} - - \| \
            copy-feats ark,t:- ark,scp:{fdir}/targets.JOB.ark,"""
        """{fdir}/targets.JOB.scp""".format(
            cmd=args.cmd, nj=args.nj, weights=args.weights, dir=args.dir,
            rmf=args.remove_mismatch_frames,
            targets=' '.join(targets_rspecifiers),
            fdir=os.path.realpath(args.dir)))

    with open("{dir}/targets.scp".format(dir=args.dir), 'w') as f:
        for n in range(1, args.nj + 1):
            for line in open("{dir}/targets.{n}.scp".format(dir=args.dir, n=n)):
                print (line.strip(), file=f)

    if frame_subsampling_factor != 1:
        with open("{dir}/frame_subsampling_factor".format(dir=args.dir), 'w') as f:
            print ("{0}".format(frame_subsampling_factor), file=f)

    common_lib.execute_command(
        "steps/segmentation/validate_targets_dir.sh {dir} {data}"
        "".format(dir=args.dir, data=args.data))


def main():
    args = get_args()

    try:
        run(args)
    except Exception:
        raise


if __name__ == '__main__':
    main()
