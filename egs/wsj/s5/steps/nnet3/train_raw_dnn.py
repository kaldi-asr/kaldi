#!/usr/bin/env python

# Copyright 2016    Vijayaditya Peddinti.
#           2016    Vimal Manohar
# Apache 2.0.

""" This script is similar to steps/nnet3/train_dnn.py but trains a
raw neural network instead of an acoustic model.
"""

from __future__ import print_function
import argparse
import logging
import pprint
import os
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
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting raw DNN trainer (train_raw_dnn.py)')


def get_args():
    """ Get args from stdin.

    The common options are defined in the object
    libs.nnet3.train.common.CommonParser.parser.
    See steps/libs/nnet3/train/common.py
    """

    parser = argparse.ArgumentParser(
        description="""Trains a feed forward raw DNN (without transition model)
        using frame-level objectives like cross-entropy and mean-squared-error.
        DNNs include simple DNNs, TDNNs and CNNs.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve',
        parents=[common_train_lib.CommonParser(include_chunk_context=False).parser])

    # egs extraction options
    parser.add_argument("--egs.frames-per-eg", type=int, dest='frames_per_eg',
                        default=8,
                        help="Number of output labels per example")
    parser.add_argument("--image.augmentation-opts", type=str,
                        dest='image_augmentation_opts',
                        default=None,
                        help="Image augmentation options")

    # trainer options
    parser.add_argument("--trainer.prior-subset-size", type=int,
                        dest='prior_subset_size', default=20000,
                        help="Number of samples for computing priors")
    parser.add_argument("--trainer.num-jobs-compute-prior", type=int,
                        dest='num_jobs_compute_prior', default=10,
                        help="The prior computation jobs are single "
                        "threaded and run on the CPU")

    # Parameters for the optimization
    parser.add_argument("--trainer.optimization.minibatch-size",
                        type=str, dest='minibatch_size', default='512',
                        help="""Size of the minibatch used in SGD training
                        (argument to nnet3-merge-egs); may be a more general
                        rule as accepted by the --minibatch-size option of
                        nnet3-merge-egs; run that program without args to see
                        the format.""")
    parser.add_argument("--compute-average-posteriors",
                        type=str, action=common_lib.StrToBoolAction,
                        choices=["true", "false"], default=False,
                        help="""If true, then the average output of the
                        network is computed and dumped as post.final.vec""")

    # General options
    parser.add_argument("--nj", type=int, default=4,
                        help="Number of parallel jobs")
    parser.add_argument("--use-dense-targets", type=str,
                        action=common_lib.StrToBoolAction,
                        default=True, choices=["true", "false"],
                        help="Train neural network using dense targets")
    parser.add_argument("--feat-dir", type=str, required=False,
                        help="Directory with features used for training "
                        "the neural network.")
    parser.add_argument("--targets-scp", type=str, required=False,
                        help="Targets for training neural network.")
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

    if args.frames_per_eg < 1:
        raise Exception("--egs.frames-per-eg should have a minimum value of 1")

    if not common_train_lib.validate_minibatch_size_str(args.minibatch_size):
        raise Exception("--trainer.optimization.minibatch-size has an invalid value")

    if (not os.path.exists(args.dir)
            or not os.path.exists(args.dir+"/configs")):
        raise Exception("This scripts expects {0} to exist and have a configs "
                        "directory which is the output of "
                        "make_configs.py script")

    # set the options corresponding to args.use_gpu
    run_opts = common_train_lib.RunOpts()
    if args.use_gpu in ["true", "false"]:
        args.use_gpu = ("yes" if args.use_gpu == "true" else "no")
    if args.use_gpu in ["yes", "wait"]:
        if not common_lib.check_if_cuda_compiled():
            logger.warning(
                """You are running with one thread but you have not compiled
                   for CUDA.  You may be running a setup optimized for GPUs.
                   If you have GPUs and have nvcc installed, go to src/ and do
                   ./configure; make""")

        run_opts.train_queue_opt = "--gpu 1"
        run_opts.parallel_train_opts = "--use-gpu={}".format(args.use_gpu)
        run_opts.combine_gpu_opt = "--use-gpu={}".format(args.use_gpu)
        run_opts.combine_queue_opt = "--gpu 1"
        run_opts.prior_gpu_opt = "--use-gpu={}".format(args.use_gpu)
        run_opts.prior_queue_opt = "--gpu 1"

    else:
        logger.warning("Without using a GPU this will be very slow. "
                       "nnet3 does not yet support multiple threads.")

        run_opts.train_queue_opt = ""
        run_opts.parallel_train_opts = "--use-gpu=no"
        run_opts.combine_gpu_opt = "--use-gpu=no"
        run_opts.combine_queue_opt = ""
        run_opts.prior_gpu_opt = "--use-gpu=no"
        run_opts.prior_queue_opt = ""

    run_opts.command = args.command
    run_opts.egs_command = (args.egs_command
                            if args.egs_command is not None else
                            args.command)
    run_opts.num_jobs_compute_prior = args.num_jobs_compute_prior

    return [args, run_opts]


def train(args, run_opts):
    """ The main function for training.

    Args:
        args: a Namespace object with the required parameters
            obtained from the function process_args()
        run_opts: RunOpts object obtained from the process_args()
    """

    arg_string = pprint.pformat(vars(args))
    logger.info("Arguments for the experiment\n{0}".format(arg_string))

    # Set some variables.

    # note, feat_dim gets set to 0 if args.feat_dir is unset (None).
    feat_dim = common_lib.get_feat_dim(args.feat_dir)
    ivector_dim = common_lib.get_ivector_dim(args.online_ivector_dir)
    ivector_id = common_lib.get_ivector_extractor_id(args.online_ivector_dir)

    config_dir = '{0}/configs'.format(args.dir)
    var_file = '{0}/vars'.format(config_dir)

    variables = common_train_lib.parse_generic_config_vars_file(var_file)

    # Set some variables.
    try:
        model_left_context = variables['model_left_context']
        model_right_context = variables['model_right_context']

    except KeyError as e:
        raise Exception("KeyError {0}: Variables need to be defined in "
                        "{1}".format(str(e), '{0}/configs'.format(args.dir)))

    left_context = model_left_context
    right_context = model_right_context


    # Initialize as "raw" nnet, prior to training the LDA-like preconditioning
    # matrix.  This first config just does any initial splicing that we do;
    # we do this as it's a convenient way to get the stats for the 'lda-like'
    # transform.
    if (args.stage <= -5) and os.path.exists(args.dir+"/configs/init.config"):
        logger.info("Initializing the network for computing the LDA stats")
        common_lib.execute_command(
            """{command} {dir}/log/nnet_init.log \
                    nnet3-init --srand=-2 {dir}/configs/init.config \
                    {dir}/init.raw""".format(command=run_opts.command,
                                             dir=args.dir))

    default_egs_dir = '{0}/egs'.format(args.dir)
    if (args.stage <= -4) and args.egs_dir is None:
        if args.targets_scp is None or args.feat_dir is None:
            raise Exception("If you don't supply the --egs-dir option, the "
                            "--targets-scp and --feat-dir options are required.")

        logger.info("Generating egs")

        if args.use_dense_targets:
            target_type = "dense"
            try:
                num_targets = int(variables['num_targets'])
                if (common_lib.get_feat_dim_from_scp(args.targets_scp)
                        != num_targets):
                    raise Exception("Mismatch between num-targets provided to "
                                    "script vs configs")
            except KeyError as e:
                num_targets = -1
        else:
            target_type = "sparse"
            try:
                num_targets = int(variables['num_targets'])
            except KeyError as e:
                raise Exception("KeyError {0}: Variables need to be defined "
                                "in {1}".format(
                                    str(e), '{0}/configs'.format(args.dir)))

        train_lib.raw_model.generate_egs_using_targets(
            data=args.feat_dir, targets_scp=args.targets_scp,
            egs_dir=default_egs_dir,
            left_context=left_context, right_context=right_context,
            run_opts=run_opts,
            frames_per_eg_str=str(args.frames_per_eg),
            srand=args.srand,
            egs_opts=args.egs_opts,
            cmvn_opts=args.cmvn_opts,
            online_ivector_dir=args.online_ivector_dir,
            samples_per_iter=args.samples_per_iter,
            stage=args.egs_stage,
            target_type=target_type,
            num_targets=num_targets)

    if args.egs_dir is None:
        egs_dir = default_egs_dir
    else:
        egs_dir = args.egs_dir

    [egs_left_context, egs_right_context,
     frames_per_eg_str, num_archives] = (
         common_train_lib.verify_egs_dir(egs_dir, feat_dim,
                                         ivector_dim, ivector_id,
                                         left_context, right_context))
    assert str(args.frames_per_eg) == frames_per_eg_str

    if args.num_jobs_final > num_archives:
        raise Exception('num_jobs_final cannot exceed the number of archives '
                        'in the egs directory')

    # copy the properties of the egs to dir for
    # use during decoding
    common_train_lib.copy_egs_properties_to_exp_dir(egs_dir, args.dir)

    if args.stage <= -3 and os.path.exists(args.dir+"/configs/init.config"):
        logger.info('Computing the preconditioning matrix for input features')

        train_lib.common.compute_preconditioning_matrix(
            args.dir, egs_dir, num_archives, run_opts,
            max_lda_jobs=args.max_lda_jobs,
            rand_prune=args.rand_prune)

    if args.stage <= -1:
        logger.info("Preparing the initial network.")
        common_train_lib.prepare_initial_network(args.dir, run_opts)

    # set num_iters so that as close as possible, we process the data
    # $num_epochs times, i.e. $num_iters*$avg_num_jobs) ==
    # $num_epochs*$num_archives, where
    # avg_num_jobs=(num_jobs_initial+num_jobs_final)/2.
    num_archives_expanded = num_archives * args.frames_per_eg
    num_archives_to_process = int(args.num_epochs * num_archives_expanded)
    num_archives_processed = 0
    num_iters = ((num_archives_to_process * 2)
                 / (args.num_jobs_initial + args.num_jobs_final))

    # If do_final_combination is True, compute the set of models_to_combine.
    # Otherwise, models_to_combine will be none.
    if args.do_final_combination:
        models_to_combine = common_train_lib.get_model_combine_iters(
            num_iters, args.num_epochs,
            num_archives_expanded, args.max_models_combine,
            args.num_jobs_final)
    else:
        models_to_combine = None

    if os.path.exists('{0}/valid_diagnostic.scp'.format(egs_dir)):
        if os.path.exists('{0}/valid_diagnostic.egs'.format(egs_dir)):
            raise Exception('both {0}/valid_diagnostic.egs and '
                            '{0}/valid_diagnostic.scp exist.'
                            'This script expects only one of them to exist.'
                            ''.format(egs_dir))
        use_multitask_egs = True
    else:
        if not os.path.exists('{0}/valid_diagnostic.egs'.format(egs_dir)):
            raise Exception('neither {0}/valid_diagnostic.egs nor '
                            '{0}/valid_diagnostic.scp exist.'
                            'This script expects one of them.'
                            ''.format(egs_dir))
        use_multitask_egs = False

    logger.info("Training will run for {0} epochs = "
                "{1} iterations".format(args.num_epochs, num_iters))

    for iter in range(num_iters):
        if (args.exit_stage is not None) and (iter == args.exit_stage):
            logger.info("Exiting early due to --exit-stage {0}".format(iter))
            return
        current_num_jobs = int(0.5 + args.num_jobs_initial
                               + (args.num_jobs_final - args.num_jobs_initial)
                               * float(iter) / num_iters)

        if args.stage <= iter:
            lrate = common_train_lib.get_learning_rate(iter, current_num_jobs,
                                                       num_iters,
                                                       num_archives_processed,
                                                       num_archives_to_process,
                                                       args.initial_effective_lrate,
                                                       args.final_effective_lrate)

            shrinkage_value = 1.0 - (args.proportional_shrink * lrate)
            if shrinkage_value <= 0.5:
                raise Exception("proportional-shrink={0} is too large, it gives "
                                "shrink-value={1}".format(args.proportional_shrink,
                                                          shrinkage_value))

            percent = num_archives_processed * 100.0 / num_archives_to_process
            epoch = (num_archives_processed * args.num_epochs
                     / num_archives_to_process)
            shrink_info_str = ''
            if shrinkage_value != 1.0:
                shrink_info_str = 'shrink: {0:0.5f}'.format(shrinkage_value)
            logger.info("Iter: {0}/{1}    "
                        "Epoch: {2:0.2f}/{3:0.1f} ({4:0.1f}% complete)    "
                        "lr: {5:0.6f}    {6}".format(iter, num_iters - 1,
                                                     epoch, args.num_epochs,
                                                     percent,
                                                     lrate, shrink_info_str))

            train_lib.common.train_one_iteration(
                dir=args.dir,
                iter=iter,
                srand=args.srand,
                egs_dir=egs_dir,
                num_jobs=current_num_jobs,
                num_archives_processed=num_archives_processed,
                num_archives=num_archives,
                learning_rate=lrate,
                dropout_edit_string=common_train_lib.get_dropout_edit_string(
                    args.dropout_schedule,
                    float(num_archives_processed) / num_archives_to_process,
                    iter),
                train_opts=' '.join(args.train_opts),
                minibatch_size_str=args.minibatch_size,
                frames_per_eg=args.frames_per_eg,
                momentum=args.momentum,
                max_param_change=args.max_param_change,
                shrinkage_value=shrinkage_value,
                shuffle_buffer_size=args.shuffle_buffer_size,
                run_opts=run_opts,
                get_raw_nnet_from_am=False,
                image_augmentation_opts=args.image_augmentation_opts,
                use_multitask_egs=use_multitask_egs,
                backstitch_training_scale=args.backstitch_training_scale,
                backstitch_training_interval=args.backstitch_training_interval)

            if args.cleanup:
                # do a clean up everything but the last 2 models, under certain
                # conditions
                common_train_lib.remove_model(
                    args.dir, iter-2, num_iters, models_to_combine,
                    args.preserve_model_interval,
                    get_raw_nnet_from_am=False)

            if args.email is not None:
                reporting_iter_interval = num_iters * args.reporting_interval
                if iter % reporting_iter_interval == 0:
                    # lets do some reporting
                    [report, times, data] = (
                        nnet3_log_parse.generate_acc_logprob_report(args.dir))
                    message = report
                    subject = ("Update : Expt {dir} : "
                               "Iter {iter}".format(dir=args.dir, iter=iter))
                    common_lib.send_mail(message, subject, args.email)

        num_archives_processed = num_archives_processed + current_num_jobs

    if args.stage <= num_iters:
        if args.do_final_combination:
            logger.info("Doing final combination to produce final.raw")
            train_lib.common.combine_models(
                dir=args.dir, num_iters=num_iters,
                models_to_combine=models_to_combine, egs_dir=egs_dir,
                minibatch_size_str=args.minibatch_size, run_opts=run_opts,
                get_raw_nnet_from_am=False,
                max_objective_evaluations=args.max_objective_evaluations,
                use_multitask_egs=use_multitask_egs)
        else:
            common_lib.force_symlink("{0}.raw".format(num_iters),
                                     "{0}/final.raw".format(args.dir))

    if args.compute_average_posteriors and args.stage <= num_iters + 1:
        logger.info("Getting average posterior for output-node 'output'.")
        train_lib.common.compute_average_posterior(
            dir=args.dir, iter='final', egs_dir=egs_dir,
            num_archives=num_archives,
            prior_subset_size=args.prior_subset_size, run_opts=run_opts,
            get_raw_nnet_from_am=False)

    if args.cleanup:
        logger.info("Cleaning up the experiment directory "
                    "{0}".format(args.dir))
        remove_egs = args.remove_egs
        if args.egs_dir is not None:
            # this egs_dir was not created by this experiment so we will not
            # delete it
            remove_egs = False

        common_train_lib.clean_nnet_dir(
            nnet_dir=args.dir, num_iters=num_iters, egs_dir=egs_dir,
            preserve_model_interval=args.preserve_model_interval,
            remove_egs=remove_egs,
            get_raw_nnet_from_am=False)

    # do some reporting
    outputs_list = common_train_lib.get_outputs_list("{0}/final.raw".format(
        args.dir), get_raw_nnet_from_am=False)
    if 'output' in outputs_list:
        [report, times, data] = nnet3_log_parse.generate_acc_logprob_report(
            args.dir)
        if args.email is not None:
            common_lib.send_mail(report, "Update : Expt {0} : "
                                         "complete".format(args.dir),
                                 args.email)

        with open("{dir}/accuracy.{output_name}.report".format(dir=args.dir,
                                                               output_name="output"),
                  "w") as f:
            f.write(report)

    common_lib.execute_command("steps/info/nnet3_dir_info.pl "
                               "{0}".format(args.dir))


def main():
    [args, run_opts] = get_args()
    try:
        train(args, run_opts)
        common_lib.wait_for_background_commands()
    except BaseException as e:
        # look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if args.email is not None:
            message = ("Training session for experiment {dir} "
                       "died due to an error.".format(dir=args.dir))
            common_lib.send_mail(message, message, args.email)
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
