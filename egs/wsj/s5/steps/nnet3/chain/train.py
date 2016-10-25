#!/usr/bin/env python


# Copyright 2016 Vijayaditya Peddinti.
# Apache 2.0.


# this script is based on steps/nnet3/chain/train.sh

import os
import subprocess
import argparse
import sys
import pprint
import logging
import imp
import traceback
import shutil
import math

common_train_lib = imp.load_source('ntl', 'steps/nnet3/libs/common_train_lib.py')
chain_lib = imp.load_source('ncl', 'steps/nnet3/libs/chain_train_lib.py')
nnet3_log_parse = imp.load_source('nlp', 'steps/nnet3/report/nnet3_log_parse_lib.py')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting chain model trainer (train.py)')


def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="""
    Trains RNN and DNN acoustic models using the 'chain' objective function.
    """,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    conflict_handler = 'resolve')

    common_train_lib.AddCommonTrainArgs(parser)

    # egs extraction options
    parser.add_argument("--egs.chunk-width", type=int, dest='chunk_width',
                        default = 150,
                        help="Number of output labels in each example. Caution: if you double this you should halve --trainer.samples-per-iter.")

    # chain options
    parser.add_argument("--chain.lm-opts", type=str, dest='lm_opts',
                        default = None, action = common_train_lib.NullstrToNoneAction,
                        help="options to be be passed to chain-est-phone-lm")
    parser.add_argument("--chain.l2-regularize", type=float, dest='l2_regularize',
                        default = 0.0,
                        help="Weight of regularization function which is the"
                        " l2-norm of the output of the network. It should be"
                        " used without the log-softmax layer for the outputs."
                        " As l2-norm of the log-softmax outputs can dominate"
                        " the objective function.")
    parser.add_argument("--chain.xent-regularize", type=float, dest='xent_regularize',
                        default = 0.0,
                        help="Weight of regularization function which is the"
                        " cross-entropy cost the outputs.")
    parser.add_argument("--chain.right-tolerance", type=int, dest='right_tolerance',
                        default = 5, help="")
    parser.add_argument("--chain.left-tolerance", type=int, dest='left_tolerance',
                        default = 5, help="")
    parser.add_argument("--chain.leaky-hmm-coefficient", type=float, dest='leaky_hmm_coefficient',
                        default = 0.00001, help="")
    parser.add_argument("--chain.apply-deriv-weights", type=str, dest='apply_deriv_weights',
                        default=True, action=common_train_lib.StrToBoolAction,
                        choices = ["true", "false"],
                        help="")
    parser.add_argument("--chain.truncate-deriv-weights", type=float, dest='truncate_deriv_weights',
                        default =0,
                        help="Can be used to set to zero the weights of derivs"
                        " from frames near the edges.  (counts subsampled frames)")
    parser.add_argument("--chain.frame-subsampling-factor", type=int,
                        dest='frame_subsampling_factor',
                        default = 3,
                        help="ratio of frames-per-second of features we train"
                        " on, to chain model's output")
    parser.add_argument("--chain.alignment-subsampling-factor", type=int,
                        dest='alignment_subsampling_factor',
                        default = 3,
                        help="ratio of frames-per-second of input alignments to"
                        " chain model's output")
    parser.add_argument("--chain.left-deriv-truncate", type=int,
                        dest='left_deriv_truncate',
                        default = None, help="")
    parser.add_argument("--chain.right-deriv-truncate", type=int,
                        dest='right_deriv_truncate',
                        default = None, help="")


    # trainer options
    parser.add_argument("--trainer.num-epochs", type=int, dest='num_epochs',
                        default = 10,
                        help="Number of epochs to train the model")
    parser.add_argument("--trainer.frames-per-iter", type=int, dest='frames_per_iter',
                        default=800000,
                        help ="Each iteration of training, see this many [input]"
                        " frames per job.  This option is passed to get_egs.sh."
                        " Aim for about a minute of training time")

    # Parameters for the optimization
    parser.add_argument("--trainer.optimization.initial-effective-lrate", type=float, dest='initial_effective_lrate',
                        default = 0.0002,
                        help="Learning rate used during the initial iteration")
    parser.add_argument("--trainer.optimization.final-effective-lrate", type=float, dest='final_effective_lrate',
                        default = 0.00002,
                        help="Learning rate used during the final iteration")
    parser.add_argument("--trainer.optimization.shrink-value", type=float, dest='shrink_value',
                        default = 1.0,
                        help="Scaling factor used for scaling the parameter"
                        " matrices when the derivative averages are below the"
                        " shrink-threshold at the non-linearities")
    parser.add_argument("--trainer.optimization.shrink-threshold", type=float, dest='shrink_threshold',
                        default = 0.15,
                        help="If the derivative averages are below this"
                        " threshold we scale the parameter matrices with the"
                        " shrink-value. It is less than 0.25 for sigmoid non-linearities.")
    parser.add_argument("--trainer.optimization.shrink-nonlinearity", type=str, dest='shrink_nonlinearity',
                        default = "SigmoidComponent", choices = ["TanhComponent", "SigmoidComponent"],
                        help="The non-linear component from which the"
                        " deriv-avg values are going to used to compute"
                        " mean-deriv-avg. The mean-deriv-avg is going to be"
                        " compared with shrink-threshold. Be careful to specify"
                        " a shrink-threshold which is dependent on the"
                        " shrink-nonlinearity type")

    # RNN specific trainer options
    parser.add_argument("--trainer.rnn.num-chunk-per-minibatch", type=int, dest='num_chunk_per_minibatch',
                        default=100,
                        help="Number of sequences to be processed in parallel every minibatch" )

    # General options
    parser.add_argument("--feat-dir", type=str, required = True,
                        help="Directory with features used for training the neural network.")
    parser.add_argument("--tree-dir", type=str, required = True,
                        help="Tree directory")
    parser.add_argument("--lat-dir", type=str, required = True,
                        help="Directory with alignments used for training the neural network.")
    parser.add_argument("--dir", type=str, required = True,
                        help="Directory to store the models and all other files.")

    print(' '.join(sys.argv))
    print(sys.argv)

    args = parser.parse_args()

    [args, run_opts] = ProcessArgs(args)

    return [args, run_opts]

def ProcessArgs(args):
    # process the options
    if args.chunk_width < 1:
        raise Exception("--egs.chunk-width should have a minimum value of 1")

    if args.chunk_left_context < 0:
        raise Exception("--egs.chunk-left-context should be non-negative")

    if args.chunk_right_context < 0:
        raise Exception("--egs.chunk-right-context should be non-negative")

    if (not os.path.exists(args.dir)) or (not os.path.exists(args.dir+"/configs")):
        raise Exception("""This scripts expects {0} to exist and have a configs
        directory which is the output of make_configs.py script""")

    if args.transform_dir is None:
        args.transform_dir = args.lat_dir
    # set the options corresponding to args.use_gpu
    run_opts = common_train_lib.RunOpts()
    if args.use_gpu:
        if not common_train_lib.CheckIfCudaCompiled():
            logger.warning("""
    You are running with one thread but you have not compiled
    for CUDA.  You may be running a setup optimized for GPUs.  If you have
    GPUs and have nvcc installed, go to src/ and do ./configure; make""")

        run_opts.train_queue_opt = "--gpu 1"
        run_opts.parallel_train_opts = ""
        run_opts.combine_queue_opt = "--gpu 1"

    else:
        logger.warning("""
    Without using a GPU this will be very slow.  nnet3 does not yet support multiple threads.""")

        run_opts.train_queue_opt = ""
        run_opts.parallel_train_opts = "--use-gpu=no"
        run_opts.combine_queue_opt = ""

    run_opts.command = args.command

    return [args, run_opts]

# args is a Namespace with the required parameters
def Train(args, run_opts):
    arg_string = pprint.pformat(vars(args))
    logger.info("Arguments for the experiment\n{0}".format(arg_string))

    # Check files
    CheckForRequiredFiles(args.feat_dir, args.tree_dir, args.lat_dir)

    # Set some variables.
    num_jobs = common_train_lib.GetNumberOfJobs(args.tree_dir)
    feat_dim = common_train_lib.GetFeatDim(args.feat_dir)
    ivector_dim = common_train_lib.GetIvectorDim(args.online_ivector_dir)

    # split the training data into parts for individual jobs
    # we will use the same number of jobs as that used for alignment
    common_train_lib.SplitData(args.feat_dir, num_jobs)
    shutil.copy('{0}/tree'.format(args.tree_dir), args.dir)
    f = open('{0}/num_jobs'.format(args.dir), 'w')
    f.write(str(num_jobs))
    f.close()

    config_dir = '{0}/configs'.format(args.dir)
    var_file = '{0}/vars'.format(config_dir)

    variables = common_train_lib.ParseGenericConfigVarsFile(var_file)

    # Set some variables.

    try:
        model_left_context = variables['model_left_context']
        model_right_context = variables['model_right_context']
        num_hidden_layers = variables['num_hidden_layers']
    except KeyError as e:
        raise Exception("KeyError {0}: Variables need to be defined in {1}".format(
            str(e), '{0}/configs'.format(args.dir)))

    left_context = args.chunk_left_context + model_left_context
    right_context = args.chunk_right_context + model_right_context

    # Initialize as "raw" nnet, prior to training the LDA-like preconditioning
    # matrix.  This first config just does any initial splicing that we do;
    # we do this as it's a convenient way to get the stats for the 'lda-like'
    # transform.
    if (args.stage <= -6):
        logger.info("Creating phone language-model")
        chain_lib.CreatePhoneLm(args.dir, args.tree_dir, run_opts, lm_opts = args.lm_opts)

    if (args.stage <= -5):
        logger.info("Creating denominator FST")
        chain_lib.CreateDenominatorFst(args.dir, args.tree_dir, run_opts)

    if (args.stage <= -4):
        logger.info("Initializing a basic network for estimating preconditioning matrix")
        common_train_lib.RunKaldiCommand("""
{command} {dir}/log/nnet_init.log \
    nnet3-init --srand=-2 {dir}/configs/init.config {dir}/init.raw
    """.format(command = run_opts.command,
               dir = args.dir))

    default_egs_dir = '{0}/egs'.format(args.dir)
    if (args.stage <= -3) and args.egs_dir is None:
        logger.info("Generating egs")
        # this is where get_egs.sh is called.
        chain_lib.GenerateChainEgs(args.dir, args.feat_dir, args.lat_dir, default_egs_dir,
                                   left_context + args.frame_subsampling_factor/2,
                                   right_context + args.frame_subsampling_factor/2,
                                   run_opts,
                                   left_tolerance = args.left_tolerance,
                                   right_tolerance = args.right_tolerance,
                                   frame_subsampling_factor = args.frame_subsampling_factor,
                                   alignment_subsampling_factor = args.alignment_subsampling_factor,
                                   frames_per_eg = args.chunk_width,
                                   egs_opts = args.egs_opts,
                                   cmvn_opts = args.cmvn_opts,
                                   online_ivector_dir = args.online_ivector_dir,
                                   frames_per_iter = args.frames_per_iter,
                                   srand = args.srand,
                                   transform_dir = args.transform_dir,
                                   stage = args.egs_stage)

    if args.egs_dir is None:
        egs_dir = default_egs_dir
    else:
        egs_dir = args.egs_dir

    [egs_left_context, egs_right_context,
     frames_per_eg, num_archives] = (
             common_train_lib.VerifyEgsDir(egs_dir, feat_dim, ivector_dim,
                                           left_context, right_context) )
    assert(args.chunk_width == frames_per_eg)
    num_archives_expanded = num_archives * args.frame_subsampling_factor

    if (args.num_jobs_final > num_archives_expanded):
        raise Exception('num_jobs_final cannot exceed the expanded number of archives')

    # copy the properties of the egs to dir for
    # use during decoding
    common_train_lib.CopyEgsPropertiesToExpDir(egs_dir, args.dir)

    if (args.stage <= -2):
        logger.info('Computing the preconditioning matrix for input features')

        chain_lib.ComputePreconditioningMatrix(args.dir, egs_dir, num_archives, run_opts,
                                               max_lda_jobs = args.max_lda_jobs,
                                               rand_prune = args.rand_prune)

    if (args.stage <= -1):
        logger.info("Preparing the initial acoustic model.")
        chain_lib.PrepareInitialAcousticModel(args.dir, run_opts)

    file_handle = open("{0}/frame_subsampling_factor".format(args.dir),"w")
    file_handle.write(str(args.frame_subsampling_factor))
    file_handle.close()

    # set num_iters so that as close as possible, we process the data $num_epochs
    # times, i.e. $num_iters*$avg_num_jobs) == $num_epochs*$num_archives,
    # where avg_num_jobs=(num_jobs_initial+num_jobs_final)/2.
    num_archives_to_process = args.num_epochs * num_archives_expanded
    num_archives_processed = 0
    num_iters=(num_archives_to_process * 2) / (args.num_jobs_initial + args.num_jobs_final)

    num_iters_combine = common_train_lib.VerifyIterations(
                                         num_iters, args.num_epochs,
                                         num_hidden_layers, num_archives_expanded,
                                         args.max_models_combine, args.add_layers_period,
                                         args.num_jobs_final)

    learning_rate = (lambda iter, current_num_jobs, num_archives_processed:
                        common_train_lib.GetLearningRate(
                                         iter, current_num_jobs, num_iters,
                                         num_archives_processed,
                                         num_archives_to_process,
                                         args.initial_effective_lrate,
                                         args.final_effective_lrate)
                    )

    logger.info("Training will run for {0} epochs = {1} iterations".format(args.num_epochs, num_iters))
    for iter in range(num_iters):
        if (args.exit_stage is not None) and (iter == args.exit_stage):
            logger.info("Exiting early due to --exit-stage {0}".format(iter))
            return
        current_num_jobs = int(0.5 + args.num_jobs_initial + (args.num_jobs_final - args.num_jobs_initial) * float(iter) / num_iters)

        if args.stage <= iter:
            model_file = "{dir}/{iter}.mdl".format(dir = args.dir, iter = iter)
            shrinkage_value = (args.shrink_value
                               if common_train_lib.DoShrinkage(iter, model_file,
                                                               args.shrink_nonlinearity,
                                                               args.shrink_threshold)
                               else 1
                               )
            logger.info("On iteration {0}, learning rate is {1} and shrink value is {2}.".format(iter, learning_rate(iter, current_num_jobs, num_archives_processed), shrinkage_value))

            TrainOneIteration(dir = args.dir, iter = iter, srand = args.srand,
                              egs_dir = egs_dir,
                              num_jobs = current_num_jobs,
                              num_archives_processsed =  num_archives_processed,
                              num_archives = num_archives,
                              learning_rate = learning_rate(iter, current_num_jobs, num_archives_processed),
                              shrinkage_value = shrinkage_value,
                              num_chunk_per_minibatch = args.num_chunk_per_minibatch,
                              num_hidden_layers = num_hidden_layers,
                              add_layers_period = args.add_layers_period,
                              apply_deriv_weights = args.apply_deriv_weights,
                              left_deriv_truncate = args.left_deriv_truncate,
                              right_deriv_truncate = args.right_deriv_truncate,
                              l2_regularize = args.l2_regularize,
                              xent_regularize = args.xent_regularize,
                              leaky_hmm_coefficient = args.leaky_hmm_coefficient,
                              momentum = args.momentum,
                              max_param_change = args.max_param_change,
                              shuffle_buffer_size = args.shuffle_buffer_size,
                              frame_subsampling_factor = args.frame_subsampling_factor,
                              truncate_deriv_weight = args.truncate_deriv_weights,
                              run_opts = run_opts)

            if args.cleanup:
                # do a clean up everythin but the last 2 models, under certain conditions
                common_train_lib.RemoveModel(
                                 args.dir, iter-2, num_iters, num_iters_combine,
                                 args.preserve_model_interval)

            if args.email is not None:
                reporting_iter_interval = num_iters * args.reporting_interval
                if iter % reporting_iter_interval == 0:
                # lets do some reporting
                    [report, times, data] = nnet3_log_parse.GenerateAccuracyReport(args.dir, key="log-probability")
                    message = report
                    subject = "Update : Expt {dir} : Iter {iter}".format(dir = args.dir, iter = iter)
                    common_train_lib.SendMail(message, subject, args.email)

        num_archives_processed = num_archives_processed + current_num_jobs

    if args.stage <= num_iters:
        logger.info("Doing final combination to produce final.mdl")
        chain_lib.CombineModels(args.dir, num_iters, num_iters_combine,
                                args.num_chunk_per_minibatch, egs_dir,
                                args.leaky_hmm_coefficient, args.l2_regularize,
                                args.xent_regularize, run_opts)

    if args.cleanup:
        logger.info("Cleaning up the experiment directory {0}".format(args.dir))
        remove_egs = args.remove_egs
        if args.egs_dir is not None:
            # this egs_dir was not created by this experiment so we will not
            # delete it
            remove_egs = False

        common_train_lib.CleanNnetDir(args.dir, num_iters, egs_dir,
                         preserve_model_interval = args.preserve_model_interval,
                         remove_egs = remove_egs)

    # do some reporting
    [report, times, data] = nnet3_log_parse.GenerateAccuracyReport(args.dir, "log-probability")
    if args.email is not None:
        common_train_lib.SendMail(report, "Update : Expt {0} : complete".format(args.dir), args.email)

    report_handle = open("{dir}/accuracy.report".format(dir = args.dir), "w")
    report_handle.write(report)
    report_handle.close()

    os.system("steps/info/chain_dir_info.pl " + args.dir)

def Main():
    [args, run_opts] = GetArgs()
    try:
        Train(args, run_opts)
    except Exception as e:
        if args.email is not None:
            message = "Training session for experiment {dir} died due to an error.".format(dir = args.dir)
            common_train_lib.SendMail(message, message, args.email)
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    Main()
