#!/usr/bin/env python

# Copyright 2016 Vijayaditya Peddinti.
#           2016 Vimal Manohar
# Apache 2.0.

# this script is based on steps/nnet3/lstm/train.sh

import subprocess
import argparse
import sys
import pprint
import logging
import imp
import traceback
from nnet3_train_lib import *

nnet3_log_parse = imp.load_source('nlp', 'steps/nnet3/report/nnet3_log_parse_lib.py')
train_lib = imp.load_source('rtl', 'steps/nnet3/libs/rnn_train_lib.py')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting RNN trainer (train_raw_rnn.py)')


def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="""
    Trains an RNN acoustic model using the cross-entropy objective.
    RNNs include LSTMs, BLSTMs and GRUs.
    RNN acoustic model training differs from feed-forward DNN training
    in the following ways
        1. RNN acoustic models train on output chunks rather than individual
           outputs
        2. The training includes additional stage of shrinkage, where
           the parameters of the model are scaled when the derivative averages
           at the non-linearities are below a threshold.
        3. RNNs can also be trained with state preservation training
    """,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    conflict_handler = 'resolve')

    train_lib.AddCommonTrainArgs(parser)

    # egs extraction options
    parser.add_argument("--egs.chunk-width", type=int, dest='chunk_width',
                        default = 20,
                        help="""Number of output labels in the sequence
                        used to train an LSTM.
                        Caution: if you double this you should halve
                        --trainer.samples-per-iter.""")
    parser.add_argument("--egs.chunk-left-context", type=int, dest='chunk_left_context',
                        default = 40,
                        help="""Number of left steps used in the estimation of LSTM
                        state before prediction of the first label""")
    parser.add_argument("--egs.chunk-right-context", type=int, dest='chunk_right_context',
                        default = 0,
                        help="""Number of right steps used in the estimation of BLSTM
                        state before prediction of the first label""")
    parser.add_argument("--trainer.samples-per-iter", type=int, dest='samples_per_iter',
                        default=20000,
                        help="""This is really the number of egs in each
                        archive.  Each eg has 'chunk_width' frames in it--
                        for chunk_width=20, this value (20k) is equivalent
                        to the 400k number that we use as a default in
                        regular DNN training.""")

    # Parameters for the optimization
    parser.add_argument("--trainer.optimization.momentum", type=float, dest='momentum',
                        default = 0.5,
                        help="""Momentum used in update computation.
                        Note: we implemented it in such a way that
                        it doesn't increase the effective learning rate.""")
    parser.add_argument("--trainer.optimization.shrink-value", type=float, dest='shrink_value',
                        default = 0.99,
                        help="Scaling factor used for scaling the parameter matrices when the derivative averages are below the shrink-threshold at the non-linearities")
    parser.add_argument("--trainer.optimization.shrink-threshold", type=float, dest='shrink_threshold',
                        default = 0.15,
                        help="If the derivative averages are below this threshold we scale the parameter matrices with the shrink-value. It is less than 0.25 for sigmoid non-linearities.")
    parser.add_argument("--trainer.optimization.cv-minibatch-size", type=int, dest='cv_minibatch_size',
            default = 256,
            help="Size of the minibatch to be used in diagnostic jobs (use smaller value for BLSTMs to control memory usage)")



    # RNN specific trainer options
    parser.add_argument("--trainer.rnn.num-chunk-per-minibatch", type=int, dest='num_chunk_per_minibatch',
                        default=100,
                        help="Number of sequences to be processed in parallel every minibatch" )
    parser.add_argument("--trainer.rnn.num-bptt-steps", type=int, dest='num_bptt_steps',
                        default=None,
                        help="The number of time steps to back-propagate from the last label in the chunk. By default it is same as the chunk-width." )

    # General options
    parser.add_argument("--nj", type=int, default=4,
                        help="Number of parallel jobs")

    parser.add_argument("--use-dense-targets", type=str, action=StrToBoolAction,
                       default = True, choices = ["true", "false"],
                       help="Train neural network using dense targets")
    parser.add_argument("--feat-dir", type=str, required = True,
                        help="Directory with features used for training the neural network.")
    parser.add_argument("--targets-scp", type=str, required = True,
                        help="Target for training neural network.")
    parser.add_argument("--dir", type=str, required = True,
                        help="Directory to store the models and all other files.")

    print(' '.join(sys.argv))

    args = parser.parse_args()

    [args, run_opts] = ProcessArgs(args)

    return [args, run_opts]

def ProcessArgs(args):
    # process the options
    if args.chunk_width < 1:
        raise Exception("--egs.chunk-width should have a minimum value of 1")

    if args.chunk_left_context < 0:
        raise Exception("--egs.chunk-left-context should be positive")

    if args.chunk_right_context < 0:
        raise Exception("--egs.chunk-right-context should be positive")

    if (not os.path.exists(args.dir)) or (not os.path.exists(args.dir+"/configs")):
        raise Exception("""This scripts expects {0} to exist and have a configs
        directory which is the output of make_configs.py script""")

    # set the options corresponding to args.use_gpu
    run_opts = train_lib.RunOpts()
    if args.use_gpu:
        if not CheckIfCudaCompiled():
            logger.warning("""
    You are running with one thread but you have not compiled
    for CUDA.  You may be running a setup optimized for GPUs.  If you have
    GPUs and have nvcc installed, go to src/ and do ./configure; make""")

        run_opts.train_queue_opt = "--gpu 1"
        run_opts.parallel_train_opts = ""
        run_opts.combine_queue_opt = "--gpu 1"
        run_opts.prior_gpu_opt = "--use-gpu=yes"
        run_opts.prior_queue_opt = "--gpu 1"

    else:
        logger.warning("""
    Without using a GPU this will be very slow.  nnet3 does not yet support multiple threads.""")

        run_opts.train_queue_opt = ""
        run_opts.parallel_train_opts = "--use-gpu=no"
        run_opts.combine_queue_opt = ""
        run_opts.prior_gpu_opt = "--use-gpu=no"
        run_opts.prior_queue_opt = ""

    run_opts.command = args.command
    run_opts.egs_command = args.egs_command if args.egs_command is not None else args.command
    run_opts.num_jobs_compute_prior = args.num_jobs_compute_prior

    return [args, run_opts]

# args is a Namespace with the required parameters
def Train(args, run_opts):
    arg_string = pprint.pformat(vars(args))
    logger.info("Arguments for the experiment\n{0}".format(arg_string))

    # Set some variables.
    feat_dim = GetFeatDim(args.feat_dir)
    ivector_dim = GetIvectorDim(args.online_ivector_dir)

    # split the training data into parts for individual jobs
    SplitData(args.feat_dir, args.nj)

    config_dir = '{0}/configs'.format(args.dir)
    var_file = '{0}/vars'.format(config_dir)

    variables = ParseModelConfigGenericVarsFile(var_file)

    # Set some variables.

    try:
        model_left_context = variables['model_left_context']
        model_right_context = variables['model_right_context']
        num_hidden_layers = variables['num_hidden_layers']
        num_targets = int(variables['num_targets'])
        add_lda = StrToBool(variables['add_lda'])
        include_log_softmax = StrToBool(variables['include_log_softmax'])
        objective_type = variables['objective_type']
    except KeyError as e:
        raise Exception("KeyError {0}: Variables need to be defined in {1}".format(
            str(e), '{0}/configs'.format(args.dir)))

    left_context = args.chunk_left_context + model_left_context
    right_context = args.chunk_right_context + model_right_context

    # Initialize as "raw" nnet, prior to training the LDA-like preconditioning
    # matrix.  This first config just does any initial splicing that we do;
    # we do this as it's a convenient way to get the stats for the 'lda-like'
    # transform.

    if args.use_dense_targets:
        if GetFeatDimFromScp(args.targets_scp) != num_targets:
            raise Exception("Mismatch between num-targets provided to "
                            "script vs configs")

    if (args.stage <= -4):
        logger.info("Initializing a basic network")
        RunKaldiCommand("""
{command} {dir}/log/nnet_init.log \
    nnet3-init --srand=-2 {dir}/configs/init.config {dir}/init.raw
    """.format(command = run_opts.command,
               dir = args.dir))

    default_egs_dir = '{0}/egs'.format(args.dir)

    if args.use_dense_targets:
        target_type = "dense"
        compute_accuracy = False
    else:
        target_type = "sparse"
        compute_accuracy = True if objective_type == "linear" else False

    if (args.stage <= -3) and args.egs_dir is None:
        logger.info("Generating egs")

        GenerateEgsFromTargets(args.feat_dir, args.targets_scp, default_egs_dir,
                    left_context, right_context,
                    args.chunk_width + left_context,
                    args.chunk_width + right_context, run_opts,
                    frames_per_eg = args.chunk_width,
                    srand = args.srand,
                    egs_opts = args.egs_opts,
                    cmvn_opts = args.cmvn_opts,
                    online_ivector_dir = args.online_ivector_dir,
                    samples_per_iter = args.samples_per_iter,
                    transform_dir = args.transform_dir,
                    stage = args.egs_stage,
                    target_type = target_type,
                    num_targets = num_targets)

    if args.egs_dir is None:
        egs_dir = default_egs_dir
    else:
        egs_dir = args.egs_dir

    [egs_left_context, egs_right_context, frames_per_eg, num_archives] = VerifyEgsDir(egs_dir, feat_dim, ivector_dim, left_context, right_context)
    assert(args.chunk_width == frames_per_eg)

    if (args.num_jobs_final > num_archives):
        raise Exception('num_jobs_final cannot exceed the number of archives in the egs directory')

    # copy the properties of the egs to dir for
    # use during decoding
    CopyEgsPropertiesToExpDir(egs_dir, args.dir)

    if (add_lda and args.stage <= -2):
        logger.info('Computing the preconditioning matrix for input features')

        ComputePreconditioningMatrix(args.dir, egs_dir, num_archives, run_opts,
                                     max_lda_jobs = args.max_lda_jobs,
                                     rand_prune = args.rand_prune)


    if (args.stage <= -1):
        logger.info("Preparing the initial acoustic model.")
        PrepareInitialNetwork(args.dir, run_opts)


    # set num_iters so that as close as possible, we process the data $num_epochs
    # times, i.e. $num_iters*$avg_num_jobs) == $num_epochs*$num_archives,
    # where avg_num_jobs=(num_jobs_initial+num_jobs_final)/2.
    num_archives_to_process = args.num_epochs * num_archives
    num_archives_processed = 0
    num_iters=(num_archives_to_process * 2) / (args.num_jobs_initial + args.num_jobs_final)

    num_iters_combine = VerifyIterations(num_iters, args.num_epochs,
                                         num_hidden_layers, num_archives,
                                         args.max_models_combine, args.add_layers_period,
                                         args.num_jobs_final)

    learning_rate = lambda iter, current_num_jobs, num_archives_processed: GetLearningRate(iter, current_num_jobs, num_iters,
                                                                   num_archives_processed,
                                                                    num_archives_to_process,
                                                                    args.initial_effective_lrate,
                                                                    args.final_effective_lrate)
    if args.num_bptt_steps is None:
        num_bptt_steps = args.chunk_width
    else:
        num_bptt_steps = args.num_bptt_steps

    min_deriv_time = args.chunk_width - num_bptt_steps


    logger.info("Training will run for {0} epochs = {1} iterations".format(args.num_epochs, num_iters))
    for iter in range(num_iters):
        if (args.exit_stage is not None) and (iter == args.exit_stage):
            logger.info("Exiting early due to --exit-stage {0}".format(iter))
            return
        current_num_jobs = int(0.5 + args.num_jobs_initial + (args.num_jobs_final - args.num_jobs_initial) * float(iter) / num_iters)

        if args.stage <= iter:
            model_file = "{dir}/{iter}.raw".format(dir = args.dir, iter = iter)
            shrinkage_value = args.shrink_value if DoShrinkage(iter, model_file, "Lstm*", "SigmoidComponent", args.shrink_threshold, use_raw_nnet = True) else 1
            logger.info("On iteration {0}, learning rate is {1} and shrink value is {2}.".format(iter, learning_rate(iter, current_num_jobs, num_archives_processed), shrinkage_value))

            train_lib.TrainOneIteration(dir = args.dir,
                                        iter = iter,
                                        srand = args.srand,
                                        egs_dir = egs_dir,
                                        num_jobs = current_num_jobs,
                                        num_archives_processed = num_archives_processed,
                                        num_archives = num_archives,
                                        learning_rate = learning_rate(iter, current_num_jobs, num_archives_processed),
                                        shrinkage_value = shrinkage_value,
                                        num_chunk_per_minibatch = args.num_chunk_per_minibatch,
                                        num_hidden_layers = num_hidden_layers,
                                        add_layers_period = args.add_layers_period,
                                        left_context = left_context,
                                        right_context = right_context,
                                        min_deriv_time = min_deriv_time,
                                        momentum = args.momentum,
                                        max_param_change = args.max_param_change,
                                        shuffle_buffer_size = args.shuffle_buffer_size,
                                        cv_minibatch_size = args.cv_minibatch_size,
                                        run_opts = run_opts,
                                        compute_accuracy = compute_accuracy,
                                        use_raw_nnet = True)
            if args.cleanup:
                # do a clean up everythin but the last 2 models, under certain conditions
                RemoveModel(args.dir, iter-2, num_iters, num_iters_combine,
                            args.preserve_model_interval, use_raw_nnet = True)

            if args.email is not None:
                reporting_iter_interval = num_iters * args.reporting_interval
                if iter % reporting_iter_interval == 0:
                # lets do some reporting
                    [report, times, data] = nnet3_log_parse.GenerateAccuracyReport(args.dir)
                    message = report
                    subject = "Update : Expt {dir} : Iter {iter}".format(dir = args.dir, iter = iter)
                    SendMail(message, subject, args.email)

        num_archives_processed = num_archives_processed + current_num_jobs

    if args.stage <= num_iters:
        logger.info("Doing final combination to produce final.raw")
        CombineModels(args.dir, num_iters, num_iters_combine, egs_dir, run_opts,
                chunk_width = args.chunk_width, use_raw_nnet = True, compute_accuracy = compute_accuracy)

    if include_log_softmax and args.stage <= num_iters + 1:
        logger.info("Getting average posterior for purpose of using as priors to convert posteriors into likelihoods.")
        avg_post_vec_file = ComputeAveragePosterior(args.dir, 'final', egs_dir,
                                num_archives, args.prior_subset_size, run_opts, use_raw_nnet = True)

    if args.cleanup:
        logger.info("Cleaning up the experiment directory {0}".format(args.dir))
        remove_egs = args.remove_egs
        if args.egs_dir is not None:
            # this egs_dir was not created by this experiment so we will not
            # delete it
            remove_egs = False

        CleanNnetDir(args.dir, num_iters, egs_dir,
                     preserve_model_interval = args.preserve_model_interval,
                     remove_egs = remove_egs,
                     use_raw_nnet = True)

    # do some reporting
    [report, times, data] = nnet3_log_parse.GenerateAccuracyReport(args.dir)
    if args.email is not None:
        SendMail(report, "Update : Expt {0} : complete".format(args.dir), args.email)

    report_handle = open("{dir}/accuracy.report".format(dir = args.dir), "w")
    report_handle.write(report)
    report_handle.close()

    os.system("steps/info/nnet3_dir_info.pl " + args.dir)

def Main():
    [args, run_opts] = GetArgs()
    try:
        Train(args, run_opts)
    except Exception as e:
        if args.email is not None:
            message = "Training session for experiment {dir} died due to an error.".format(dir = args.dir)
            SendMail(message, message, args.email)
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    Main()
