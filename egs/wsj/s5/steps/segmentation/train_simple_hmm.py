#! /usr/bin/env python

# Copyright 2016  Vimal Manohar
# Apache 2.0.

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
    """Parse command-line arguments"""

    parser = argparse.ArgumentParser(
        """Train a simple HMM model starting from HMM topology.""")

    # Alignment options
    parser.add_argument("--align.transition-scale", dest='transition_scale',
                        type=float, default=10.0,
                        help="""Transition-probability scale [relative to
                        acoustics]""")
    parser.add_argument("--align.self-loop-scale", dest='self_loop_scale',
                        type=float, default=1.0,
                        help="""Scale on self-loop versus non-self-loop log
                        probs [relative to acoustics]""")
    parser.add_argument("--align.beam", dest='beam',
                        type=float, default=6,
                        help="""Decoding beam used in alignment""")

    # Training options
    parser.add_argument("--training.num-iters", dest='num_iters',
                        type=int, default=30,
                        help="""Number of iterations of training""")
    parser.add_argument("--training.use-soft-counts", dest='use_soft_counts',
                        type=str, action=common_lib.StrToBoolAction,
                        choices=["true", "false"], default=False,
                        help="""Use soft counts (posteriors) instead of
                        alignments""")

    # General options
    parser.add_argument("--scp2ark-cmd", type=str,
                        default="copy-int-vector scp:- ark:- |",
                        help="The command used to convert scp from stdin to "
                        "write archive to stdout")
    parser.add_argument("--cmd", dest='command', type=str,
                        default="run.pl",
                        help="Command used to run jobs")
    parser.add_argument("--stage", type=int, default=-10,
                        help="""Stage to run training from""")

    parser.add_argument("--data", type=str, required=True,
                        help="Data directory; primarily used for splitting")

    labels_group = parser.add_mutually_exclusive_group(required=True)
    labels_group.add_argument("--labels-scp", type=str,
                              help="Input labels that must be convert to alignment "
                              "of class-ids using --scp2ark-cmd")
    labels_group.add_argument("--labels-rspecifier", type=str,
                              help="Input labels rspecifier")

    parser.add_argument("--lang", type=str, required=True,
                        help="The language directory containing the "
                        "HMM Topology file topo")
    parser.add_argument("--loglikes-dir", type=str, required=True,
                        help="Directory containing the log-likelihoods")
    parser.add_argument("--dir", type=str, required=True,
                        help="Directory where the intermediate and final "
                        "models will be written")

    args = parser.parse_args()

    if args.use_soft_counts:
        raise NotImplementedError("--use-soft-counts not supported yet!")

    return args


def check_files(args):
    """Check files required for this script"""

    files = ("{lang}/topo {data}/utt2spk "
             "{loglikes_dir}/log_likes.1.gz {loglikes_dir}/num_jobs "
             "".format(lang=args.lang, data=args.data,
                       loglikes_dir=args.loglikes_dir).split())

    if args.labels_scp is not None:
        files.append(args.labels_scp)

    for f in files:
        if not os.path.exists(f):
            logger.error("Could not find file %s", f)
            raise RuntimeError


def run(args):
    """The function that does it all"""

    check_files(args)

    if args.stage <= -2:
        logger.info("Initializing simple HMM model")
        common_lib.run_kaldi_command(
            """{cmd} {dir}/log/init.log simple-hmm-init {lang}/topo """
            """  {dir}/0.mdl""".format(cmd=args.command, dir=args.dir,
                                       lang=args.lang))

    num_jobs = common_lib.get_number_of_jobs(args.loglikes_dir)
    split_data = common_lib.split_data(args.data, num_jobs)

    if args.labels_rspecifier is not None:
        labels_rspecifier = args.labels_rspecifier
    else:
        labels_rspecifier = ("ark:utils/filter_scp.pl {sdata}/JOB/utt2spk "
                             "{labels_scp} | {scp2ark_cmd}".format(
                                 sdata=split_data, labels_scp=args.labels_scp,
                                 scp2ark_cmd=args.scp2ark_cmd))

    if args.stage <= -1:
        logger.info("Compiling training graphs")
        common_lib.run_kaldi_command(
            """{cmd} JOB=1:{nj} {dir}/log/compile_graphs.JOB.log """
            """  compile-train-simple-hmm-graphs {dir}/0.mdl """
            """    "{labels_rspecifier}" """
            """    "ark:| gzip -c > {dir}/fsts.JOB.gz" """.format(
                cmd=args.command, nj=num_jobs,
                dir=args.dir, lang=args.lang,
                labels_rspecifier=labels_rspecifier))

    scale_opts = ("--transition-scale={tscale} --self-loop-scale={loop_scale}"
                  "".format(tscale=args.transition_scale,
                            loop_scale=args.self_loop_scale))

    for iter_ in range(0, args.num_iters):
        if args.stage > iter_:
            continue

        logger.info("Training iteration %d", iter_)

        common_lib.run_kaldi_command(
            """{cmd} JOB=1:{nj} {dir}/log/align.{iter}.JOB.log """
            """  simple-hmm-align-compiled {scale_opts} """
            """    --beam={beam} --retry-beam={retry_beam} {dir}/{iter}.mdl """
            """    "ark:gunzip -c {dir}/fsts.JOB.gz |" """
            """    "ark:gunzip -c {loglikes_dir}/log_likes.JOB.gz |" """
            """    ark:- \| """
            """  simple-hmm-acc-stats-ali {dir}/{iter}.mdl ark:- """
            """    {dir}/{iter}.JOB.acc""".format(
                cmd=args.command, nj=num_jobs, dir=args.dir, iter=iter_,
                scale_opts=scale_opts, beam=args.beam,
                retry_beam=args.beam * 4, loglikes_dir=args.loglikes_dir))

        common_lib.run_kaldi_command(
            """{cmd} {dir}/log/update.{iter}.log """
            """  simple-hmm-est {dir}/{iter}.mdl """
            """    "vector-sum {dir}/{iter}.*.acc - |" """
            """    {dir}/{new_iter}.mdl""".format(
                cmd=args.command, dir=args.dir, iter=iter_,
                new_iter=iter_ + 1))

        common_lib.run_kaldi_command(
            "rm {dir}/{iter}.*.acc".format(dir=args.dir, iter=iter_))
    # end train loop

    common_lib.force_symlink("{0}.mdl".format(args.num_iters),
                             "{0}/final.mdl".format(args.dir))

    logger.info("Done training simple HMM in %s/final.mdl", args.dir)


def main():
    try:
        args = get_args()
        run(args)
    except Exception:
        logger.error("Failed training models")
        raise


if __name__ == '__main__':
    main()
