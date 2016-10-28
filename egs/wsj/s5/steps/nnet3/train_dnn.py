#!/usr/bin/env python


# Copyright 2016 Vijayaditya Peddinti.
#           2016 Vimal Manohar
# Apache 2.0.


# this script is based on steps/nnet3/tdnn/train.sh


import os
import subprocess
import argparse
import sys
import pprint
import logging
import imp
import traceback
import shutil

common_train_lib = imp.load_source('', 'steps/nnet3/libs/common_train_lib.py')
nnet3_log_parse = imp.load_source('', 'steps/nnet3/report/nnet3_log_parse_lib.py')
train_lib = imp.load_source('', 'steps/nnet3/libs/train_lib.py')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting DNN trainer (train_dnn.py)')


def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="""
    Trains a feed forward DNN acoustic model using the cross-entropy objective.
    DNNs include simple DNNs, TDNNs and CNNs.
    """,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    conflict_handler = 'resolve',
    parents=[common_train_lib.common_parser])
    # For common options defined in common_train_lib.common_parser,
    # see steps/nnet3/libs/common_train_lib.py

    # egs extraction options
    parser.add_argument("--egs.frames-per-eg", type=int, dest='frames_per_eg',
                        default = 8,
                        help="Number of output labels per example")

    parser.add_argument("--trainer.optimization.minibatch-size", type=float, dest='minibatch_size',
                        default = 512,
                        help="Size of the minibatch used to compute the gradient")

    parser.add_argument("--trainer.presoftmax-prior-scale-power", type=float, dest='presoftmax_prior_scale_power',
                        default=-0.25,
                        help="")

    # General options
    parser.add_argument("--feat-dir", type=str, required = True,
                        help="Directory with features used for training the neural network.")
    parser.add_argument("--lang", type=str, required = True,
                        help="Language directory")
    parser.add_argument("--ali-dir", type=str, required = True,
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
    if args.frames_per_eg < 1:
        raise Exception("--egs.frames-per-eg should have a minimum value of 1")

    if (not os.path.exists(args.dir)) or (not os.path.exists(args.dir+"/configs")):
        raise Exception("This scripts expects {0} to exist and have a configs"
        " directory which is the output of make_configs.py script")

    if args.transform_dir is None:
        args.transform_dir = args.ali_dir

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
    num_leaves = common_train_lib.GetNumberOfLeaves(args.ali_dir)
    num_jobs = common_train_lib.GetNumberOfJobs(args.ali_dir)
    feat_dim = common_train_lib.GetFeatDim(args.feat_dir)
    ivector_dim = common_train_lib.GetIvectorDim(args.online_ivector_dir)

    # split the training data into parts for individual jobs
    # we will use the same number of jobs as that used for alignment
    common_train_lib.SplitData(args.feat_dir, num_jobs)
    shutil.copy('{0}/tree'.format(args.ali_dir), args.dir)
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
        num_hidden_layers = variables['num_hidden_layers'] # this is really the number of times we add layers to the network for discriminative pretraining
    except KeyError as e:
        raise Exception("KeyError {0}: Variables need to be defined in {1}".format(
            str(e), '{0}/configs'.format(args.dir)))

    left_context = args.chunk_left_context + model_left_context
    right_context = args.chunk_right_context + model_right_context

    # Initialize as "raw" nnet, prior to training the LDA-like preconditioning
    # matrix.  This first config just does any initial splicing that we do;
    # we do this as it's a convenient way to get the stats for the 'lda-like'
    # transform.

    if (args.stage <= -5):
        logger.info("Initializing a basic network for estimating preconditioning matrix")
        common_train_lib.RunKaldiCommand("""
{command} {dir}/log/nnet_init.log \
    nnet3-init --srand=-2 {dir}/configs/init.config {dir}/init.raw
    """.format(command = run_opts.command,
               dir = args.dir))

    default_egs_dir = '{0}/egs'.format(args.dir)
    if (args.stage <= -4) and args.egs_dir is None:
        logger.info("Generating egs")

        train_lib.GenerateEgs(args.feat_dir, args.ali_dir, default_egs_dir,
                              left_context, right_context,
                              left_context, right_context, run_opts,
                              frames_per_eg = args.frames_per_eg,
                              srand = args.srand,
                              egs_opts = args.egs_opts,
                              cmvn_opts = args.cmvn_opts,
                              online_ivector_dir = args.online_ivector_dir,
                              samples_per_iter = args.samples_per_iter,
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
    assert(args.frames_per_eg == frames_per_eg)

    if (args.num_jobs_final > num_archives):
        raise Exception('num_jobs_final cannot exceed the number of archives in the egs directory')

    # copy the properties of the egs to dir for
    # use during decoding
    common_train_lib.CopyEgsPropertiesToExpDir(egs_dir, args.dir)

    if (args.stage <= -3):
        logger.info('Computing the preconditioning matrix for input features')

        train_lib.ComputePreconditioningMatrix(args.dir, egs_dir, num_archives, run_opts,
                                               max_lda_jobs = args.max_lda_jobs,
                                               rand_prune = args.rand_prune)

    if (args.stage <= -2):
        logger.info("Computing initial vector for FixedScaleComponent before"
                    " softmax, using priors^{prior_scale} and rescaling to"
                    " average 1".format(prior_scale = args.presoftmax_prior_scale_power))

        common_train_lib.ComputePresoftmaxPriorScale(
                args.dir, args.ali_dir, num_jobs, run_opts,
                presoftmax_prior_scale_power = args.presoftmax_prior_scale_power)


    if (args.stage <= -1):
        logger.info("Preparing the initial acoustic model.")
        train_lib.PrepareInitialAcousticModel(args.dir, args.ali_dir, run_opts)


    # set num_iters so that as close as possible, we process the data $num_epochs
    # times, i.e. $num_iters*$avg_num_jobs) == $num_epochs*$num_archives,
    # where avg_num_jobs=(num_jobs_initial+num_jobs_final)/2.
    num_archives_expanded = num_archives * args.frames_per_eg
    num_archives_to_process = args.num_epochs * num_archives_expanded
    num_archives_processed = 0
    num_iters=(num_archives_to_process * 2) / (args.num_jobs_initial + args.num_jobs_final)

    num_iters_combine = common_train_lib.VerifyIterations(
                                         num_iters, args.num_epochs,
                                         num_hidden_layers, num_archives_expanded,
                                         args.max_models_combine, args.add_layers_period,
                                         args.num_jobs_final)

    LearningRate = (lambda iter, current_num_jobs, num_archives_processed:
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

            logger.info("On iteration {0}, learning rate is {1}.".format(iter, LearningRate(iter, current_num_jobs, num_archives_processed)))

            train_lib.TrainOneIteration(
                      dir = args.dir,
                      iter = iter,
                      srand = args.srand,
                      egs_dir = egs_dir,
                      num_jobs = current_num_jobs,
                      num_archives_processed = num_archives_processed,
                      num_archives = num_archives,
                      learning_rate = LearningRate(iter, current_num_jobs, num_archives_processed),
                      minibatch_size = args.minibatch_size,
                      frames_per_eg = args.frames_per_eg,
                      num_hidden_layers = num_hidden_layers,
                      add_layers_period = args.add_layers_period,
                      left_context = left_context,
                      right_context = right_context,
                      momentum = args.momentum,
                      max_param_change = args.max_param_change,
                      shuffle_buffer_size = args.shuffle_buffer_size,
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
                    [report, times, data] = nnet3_log_parse.GenerateAccuracyReport(args.dir)
                    message = report
                    subject = "Update : Expt {dir} : Iter {iter}".format(dir = args.dir, iter = iter)
                    common_train_lib.SendMail(message, subject, args.email)

        num_archives_processed = num_archives_processed + current_num_jobs

    if args.stage <= num_iters:
        logger.info("Doing final combination to produce final.mdl")
        train_lib.CombineModels(args.dir, num_iters, num_iters_combine, egs_dir, run_opts)

    if args.stage <= num_iters + 1:
        logger.info("Getting average posterior for purposes of adjusting the priors.")
        avg_post_vec_file = train_lib.ComputeAveragePosterior(
                            args.dir, 'combined', egs_dir,
                            num_archives, args.prior_subset_size, run_opts)

        logger.info("Re-adjusting priors based on computed posteriors")
        combined_model = "{dir}/combined.mdl".format(dir = args.dir)
        final_model = "{dir}/final.mdl".format(dir = args.dir)
        train_lib.AdjustAmPriors(args.dir, combined_model, avg_post_vec_file,
                                 final_model, run_opts)

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
    [report, times, data] = nnet3_log_parse.GenerateAccuracyReport(args.dir)
    if args.email is not None:
        common_train_lib.SendMail(report, "Update : Expt {0} : complete".format(args.dir), args.email)

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
            common_train_lib.SendMail(message, message, args.email)
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    Main()
