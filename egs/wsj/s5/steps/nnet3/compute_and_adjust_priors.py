#!/usr/bin/env python

# Copyright 2016    Pegah Ghahremani.
# Apache 2.0.

""" This script is based on steps/nnet3/tdnn/train.sh
"""

import argparse
import logging
import os
import pprint
import shutil
import sys
import traceback

sys.path.insert(0, 'steps')
import libs.nnet3.train.common as common_train_lib
import libs.common as common_lib
import libs.nnet3.train.frame_level_objf as train_lib
import libs.nnet3.report.log_parse as nnet3_log_parse


logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting compute and adjusting posteriors (train_dnn.py)')


def get_args():
    """ Get args from stdin.

    We add compulsary arguments as named arguments for readability

    The common options are defined in the object
    libs.nnet3.train.common.CommonParser.parser.
    See steps/libs/nnet3/train/common.py
    """

    parser = argparse.ArgumentParser(
        description="""Compute the posteriors using model posteriors computed
            on example data. It re-adjusts the prior using computed average
            posterior.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
        #parents=[common_train_lib.CommonParser().parser])

    # options for average posterior computation
    parser.add_argument("--post-process.prior-subset-size", type=int,
                        dest='prior_subset_size', default=20000,
                        help="Number of samples for computing priors")
    parser.add_argument("--post-process.num-jobs-compute-prior", type=int,
                        dest='num_jobs_compute_prior', default=10,
                        help="The prior computation jobs are single "
                        "threaded and run on the CPU")
    parser.add_argument("--post-process.output-name", type=str,
                       dest='output_name',
                       default='output',
                       help='The average posteriors computed and adjusted '
                            'for this output node.')
    parser.add_argument("--post-process.model", type=str,
                       action=common_lib.NullstrToNoneAction,
                       dest='model',
                       default=None,
                       help='The model name in dir used to compute average posteriors'
                            ' w.r.t its output node.')
    parser.add_argument("--post-process.readjust-model", type=str, 
                        dest='readjusted_model',
                        default="final",
                        help="the model name used to store readjusted model.")
    parser.add_argument("--post-process.readjust-priors", type=str, 
                        action=common_lib.StrToBoolAction,
                        dest='readjust_priors',
                        default=True, choices=["true", "false"],
                        help='If true, the prior re-adjusted using computed '
                             'averarage posterior.')
    # extra egs option
    parser.add_argument("--egs.use-multitask-egs", type=str, 
                        action=common_lib.StrToBoolAction,
                        dest='use_multitask_egs',
                        default=False, choices=["true", "false"],
                        help='If true, it is assumed there are different examples '
                          'correspond to multiple tasks in output layer.')
    # General options
    parser.add_argument("--reporting.email", dest="email",
                         type=str, default=None,
                         action=common_lib.NullstrToNoneAction,
                         help=""" Email-id to report about the progress
                         of the experiment.  NOTE: It assumes the
                         machine on which the script is being run can
                         send emails from command line via. mail
                         program. The Kaldi mailing list will not
                         support this feature.  It might require local
                         expertise to setup. """)
    parser.add_argument("--background-polling-time",
                         dest="background_polling_time",
                         type=float, default=60,
                         help="""Polling frequency in seconds at which
                         the background process handler checks for
                         errors in the processes.""")    
    parser.add_argument("--use-gpu", type=str,
                        action=common_lib.StrToBoolAction,
                        choices=["true", "false"],
                        help="Use GPU for training", default=True)
    parser.add_argument("--cmd", type=str, dest="command",
                        action=common_lib.NullstrToNoneAction,
                        help="""Specifies the script to launch jobs.
                        e.g. queue.pl for launching on SGE cluster
                              run.pl for launching on local machine
                        """, default="queue.pl")
    parser.add_argument("--egs.dir", type=str, dest='egs_dir',
                             default=None,
                             action=common_lib.NullstrToNoneAction,
                             help="""Directory with egs. If specified this
                             directory will be used rather than extracting
                             egs""")
    parser.add_argument("--ali-dir", type=str, dest='ali_dir',
                             default=None,
                             action=common_lib.NullstrToNoneAction,
                             help="""Directory with alignment. If specified the
                             transition model in this
                             directory will be used to create acoustic model.""")
    parser.add_argument("--dir", type=str, required=True,
                        help="Directory to store the models and "
                        "all other files.")

    print(' '.join(sys.argv))
    print(sys.argv)

    args = parser.parse_args()

    [args, run_opts] = process_args(args)

    return [args, run_opts]

def process_args(args):
    """ Process the options got from get_args()
    """

    if (not os.path.exists(args.dir)):
        raise Exception("This scripts expects {0} to exist.")

    # set the options corresponding to args.use_gpu
    run_opts = common_train_lib.RunOpts()
    if args.use_gpu:
        if not common_lib.check_if_cuda_compiled():
            logger.warning(
                """You are running with one thread but you have not compiled
                   for CUDA.  You may be running a setup optimized for GPUs.
                   If you have GPUs and have nvcc installed, go to src/ and do
                   ./configure; make""")

        run_opts.train_queue_opt = "--gpu 1"
        run_opts.parallel_train_opts = ""
        run_opts.combine_queue_opt = "--gpu 1"
        run_opts.prior_gpu_opt = "--use-gpu=yes"
        run_opts.prior_queue_opt = "--gpu 1"

    else:
        logger.warning("Without using a GPU this will be very slow. "
                       "nnet3 does not yet support multiple threads.")

        run_opts.train_queue_opt = ""
        run_opts.parallel_train_opts = "--use-gpu=no"
        run_opts.combine_queue_opt = ""
        run_opts.prior_gpu_opt = "--use-gpu=no"
        run_opts.prior_queue_opt = ""

    run_opts.command = args.command
    run_opts.num_jobs_compute_prior = args.num_jobs_compute_prior

    return [args, run_opts]


def compute_and_adjust_priors(args, run_opts, background_process_handler):
    """ compute the average posteriors using example and model and 
        re-adjusting the prior in the model for target output.
    """
    if args.egs_dir is None:
      egs_dir = '{0}/egs'.format(args.dir)
    else:
      egs_dir = args.egs_dir
    
    left_context = int(open('{0}/info/left_context'.format(
                            egs_dir)).readline())
    right_context = int(open('{0}/info/right_context'.format(
                            egs_dir)).readline()) 
    num_archives = int(open('{0}/info/num_archives'.format(
                            egs_dir)).readline())
    logger.info("Getting average posterior for purposes of "
                "adjusting the priors for output {output}.".format(output=args.output_name))

    avg_post_vec_file = train_lib.common.compute_average_posterior(
        dir=args.dir, iter=args.output_name, egs_dir=egs_dir, model=args.model,
        num_archives=num_archives,
        left_context=left_context, right_context=right_context,
        prior_subset_size=args.prior_subset_size, run_opts=run_opts,
        output_name=args.output_name,
        use_multitask_egs=args.use_multitask_egs)

    if args.readjust_priors: 
        logger.info("Re-adjusting priors based on computed posteriors")
        if args.model is not None:
            if args.ali_dir is None:
                init_model = "{dir}/{init_mdl}.raw".format(dir=args.dir, init_mdl=args.model)
            else:
                train_lib.acoustic_model.prepare_acoustic_model(dir=args.dir,
                                                                alidir=args.ali_dir,
                                                                run_opts=run_opts,
                                                                model=args.model)
                init_model = "{dir}/{model}.mdl".format(dir=args.dir, model=args.model)

        else:
          if args.ali_dir is None:
              # If ali_dir is not specified, it assumes the model contains transition model.
              init_model = "{dir}/combined.mdl".format(dir=args.dir)
          else:
              train_lib.acoustic_model.prepare_acoustic_model(dir=args.dir,
                                                              alidir=args.ali_dir,
                                                              run_opts=run_opts,
                                                              model="final")
              init_model = "{dir}/final.mdl".format(dir=args.dir)

        final_model = "{dir}/{target_mdl}.mdl".format(dir=args.dir, target_mdl=args.readjusted_model)
        train_lib.common.adjust_am_priors(args.dir, init_model,
                                          avg_post_vec_file, final_model,
                                          run_opts)
def main():
    [args, run_opts] = get_args()
    try:
        background_process_handler = common_lib.BackgroundProcessHandler(
            polling_time=args.background_polling_time)
        compute_and_adjust_priors(args, run_opts, background_process_handler)
        background_process_handler.ensure_processes_are_done()
    except Exception as e:
        if args.email is not None:
            message = ("Training session for experiment {dir} "
                       "died due to an error.".format(dir=args.dir))
            common_lib.send_mail(message, message, args.email)
        traceback.print_exc()
        background_process_handler.stop()
        raise e


if __name__ == "__main__":
    main()
