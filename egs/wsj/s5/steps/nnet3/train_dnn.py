#!/usr/bin/env python


# Copyright 2016 Vijayaditya Peddinti.
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

nnet3_log_parse = imp.load_source('', 'steps/nnet3/report/nnet3_log_parse_lib.py')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting RNN trainer (train_rnn.py)')


def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="""
    Trains a feed forward DNN acoustic model using the cross-entropy objective.
    DNNs include simple DNNs, TDNNs and CNNs.
    """,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # feat options
    parser.add_argument("--feat.online-ivector-dir", type=str, dest='online_ivector_dir',
                        default = None, action = NullstrToNoneAction,
                        help="""directory with the ivectors extracted in
                        an online fashion.""")
    parser.add_argument("--feat.cmvn-opts", type=str, dest='cmvn_opts',
                        default = None, action = NullstrToNoneAction,
                        help="A string specifying '--norm-means' and '--norm-vars' values")

    # egs extraction options
    parser.add_argument("--egs.frames-per-eg", type=int, dest='frames_per_eg',
                        default = 8,
                        help="Number of output labels per example")
    parser.add_argument("--egs.transform_dir", type=str, dest='transform_dir',
                        default = None, action = NullstrToNoneAction,
                        help="""String to provide options directly to steps/nnet3/get_egs.sh script""")
    parser.add_argument("--egs.dir", type=str, dest='egs_dir',
                        default = None, action = NullstrToNoneAction,
                        help="""Directory with egs. If specified this directory
                        will be used rather than extracting egs""")
    parser.add_argument("--egs.stage", type=int, dest='egs_stage',
                        default = 0, help="Stage at which get_egs.sh should be restarted")
    parser.add_argument("--egs.opts", type=str, dest='egs_opts',
                        default = None, action = NullstrToNoneAction,
                        help="""String to provide options directly to steps/nnet3/get_egs.sh script""")

    # trainer options
    parser.add_argument("--trainer.num-epochs", type=int, dest='num_epochs',
                        default = 8,
                        help="Number of epochs to train the model")
    parser.add_argument("--trainer.prior-subset-size", type=int, dest='prior_subset_size',
                        default = 20000,
                        help="Number of samples for computing priors")
    parser.add_argument("--trainer.num-jobs-compute-prior", type=int, dest='num_jobs_compute_prior',
                        default = 10,
                        help="The prior computation jobs are single threaded and run on the CPU")
    parser.add_argument("--trainer.max-models-combine", type=int, dest='max_models_combine',
                        default = 20,
                        help="The maximum number of models used in the final model combination stage. These models will themselves be averages of iteration-number ranges")
    parser.add_argument("--trainer.shuffle-buffer-size", type=int, dest='shuffle_buffer_size',
                        default = 5000,
                        help="Controls randomization of the samples on each"
                        "iteration. If 0 or a large value the randomization is"
                        "complete, but this will consume memory and cause spikes"
                        "in disk I/O.  Smaller is easier on disk and memory but"
                        "less random.  It's not a huge deal though, as samples"
                        "are anyway randomized right at the start."
                        "(the point of this is to get data in different"
                        "minibatches on different iterations, since in the"
                        "preconditioning method, 2 samples in the same minibatch"
                        "can affect each others' gradients.")
    parser.add_argument("--trainer.add-layers-period", type=int, dest='add_layers_period',
                        default=2,
                        help="The number of iterations between adding layers"
                        "during layer-wise discriminative training.")
    parser.add_argument("--trainer.max-param-change", type=float, dest='max_param_change',
                        default=2.0,
                        help="The maximum change in parameters allowed per minibatch,"
                        "measured in Frobenius norm over the entire model")
    parser.add_argument("--trainer.samples-per-iter", type=int, dest='samples_per_iter',
                        default=400000,
                        help="This is really the number of egs in each archive.")
    parser.add_argument("--trainer.lda.rand-prune", type=float, dest='rand_prune',
                        default=4.0,
                        help="""Value used in preconditioning matrix estimation""")
    parser.add_argument("--trainer.lda.max-lda-jobs", type=float, dest='max_lda_jobs',
                        default=10,
                        help="""Max number of jobs used for LDA stats accumulation""")
    parser.add_argument("--trainer.presoftmax-prior-scale-power", type=float, dest='presoftmax_prior_scale_power',
                        default=-0.25,
                        help="")

    # Realignment parameters
    parser.add_argument("--trainer.realign.command", type=str, dest='realign_command',
                        default=None, action=NullstrToNoneAction,
                        help="""Command to be used with steps/nnet3/align.sh during realignment""")
    parser.add_argument("--trainer.realign.num-jobs", type=int, dest='realign_num_jobs',
                        default=30,
                        help="Number of jobs to use for realignment")
    parser.add_argument("--trainer.realign.times", type=str, dest='realign_times',
                        default=None, action=NullstrToNoneAction,
                        help="""A space seperated string of realignment
                        times. Values must be between 0 and 1
                        e.g. '0.1 0.2 0.3' """)

    parser.add_argument("--trainer.realign.use_gpu", type=str, dest='realign_use_gpu',
                        default=True, action=StrToBoolAction,
                        choices = ["true", "false"],
                        help="If true, gpu is used with steps/nnet3/align.sh")

    # Parameters for the optimization
    parser.add_argument("--trainer.optimization.minibatch-size", type=float, dest='minibatch_size',
                        default = 512,
                        help="Size of the minibatch used to compute the gradient")
    parser.add_argument("--trainer.optimization.initial-effective-lrate", type=float, dest='initial_effective_lrate',
                        default = 0.0003,
                        help="Learning rate used during the initial iteration")
    parser.add_argument("--trainer.optimization.final-effective-lrate", type=float, dest='final_effective_lrate',
                        default = 0.00003,
                        help="Learning rate used during the final iteration")
    parser.add_argument("--trainer.optimization.num-jobs-initial", type=int, dest='num_jobs_initial',
                        default = 1,
                        help="Number of neural net jobs to run in parallel at the start of training")
    parser.add_argument("--trainer.optimization.num-jobs-final", type=int, dest='num_jobs_final',
                        default = 8,
                        help="Number of neural net jobs to run in parallel at the end of training")
    parser.add_argument("--trainer.optimization.max-models-combine", type=int, dest='max_models_combine',
                        default = 20,
                        help = """ The is the maximum number of models we give to the
                                   final 'combine' stage, but these models will themselves
                                   be averages of iteration-number ranges. """)
    parser.add_argument("--trainer.optimization.momentum", type=float, dest='momentum',
                        default = 0.0,
                        help="""Momentum used in update computation.
                        Note: we implemented it in such a way that
                        it doesn't increase the effective learning rate.""")
    # General options
    parser.add_argument("--stage", type=int, default=-4,
                        help="Specifies the stage of the experiment to execution from")
    parser.add_argument("--exit-stage", type=int, default=None,
                        help="If specified, training exits before running this stage")
    parser.add_argument("--cmd", type=str, action = NullstrToNoneAction,
                        dest = "command",
                        help="""Specifies the script to launch jobs.
                        e.g. queue.pl for launching on SGE cluster
                             run.pl for launching on local machine
                        """, default = "queue.pl")
    parser.add_argument("--use-gpu", type=str, action = StrToBoolAction,
                        choices = ["true", "false"],
                        help="Use GPU for training", default=True)
    parser.add_argument("--cleanup", type=str, action = StrToBoolAction,
                        choices = ["true", "false"],
                        help="Clean up models after training", default=True)
    parser.add_argument("--cleanup.remove-egs", type=str, dest='remove_egs',
                        default = True, action = StrToBoolAction,
                        choices = ["true", "false"],
                        help="""If true, remove egs after experiment""")
    parser.add_argument("--cleanup.preserve-model-interval", dest = "preserve_model_interval",
                        type=int, default=100,
                        help="Determines iterations for which models will be preserved during cleanup. If iter % preserve_model_interval == 0 model will be preserved.")

    parser.add_argument("--reporting.email", dest = "email",
                        type=str, default=None, action = NullstrToNoneAction,
                        help=""" Email-id to report about the progress of the experiment.
                              NOTE: It assumes the machine on which the script is being run can send
                              emails from command line via. mail program. The
                              Kaldi mailing list will not support this feature.
                              It might require local expertise to setup. """)
    parser.add_argument("--reporting.interval", dest = "reporting_interval",
                        type=int, default=0.1,
                        help="Frequency with which reports have to be sent, measured in terms of fraction of iterations. If 0 and reporting mail has been specified then only failure notifications are sent")

    parser.add_argument("--feat-dir", type=str, required = True,
                        help="Directory with features used for training the neural network.")
    parser.add_argument("--lang", type=str, required = True,
                        help="Languade directory")
    parser.add_argument("--ali-dir", type=str, required = True,
                        help="Directory with alignments used for training the neural network.")
    parser.add_argument("--dir", type=str, required = True,
                        help="Directory to store the models and all other files.")

    print(' '.join(sys.argv))

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
    run_opts = RunOpts()
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

    if args.realign_use_gpu is True:
        run_opts.realign_use_gpu = True
        run_opts.realign_queue_opt = "--gpu 1"
    else:
        run_opts.realign_use_gpu = False
        run_opts.realign_queue_opt = ""

    if args.realign_command is None:
        run_opts.realign_command = args.command
    else:
        run_opts.realign_command = args.realign_command
    run_opts.realign_num_jobs = args.realign_num_jobs

    run_opts.command = args.command
    run_opts.num_jobs_compute_prior = args.num_jobs_compute_prior

    return [args, run_opts]

# a class to store run options
class RunOpts:
    def __init__(self):
        self.command = None
        self.train_queue_opt = None
        self.combine_queue_opt = None
        self.prior_gpu_opt = None
        self.prior_queue_opt = None
        self.parallel_train_opts = None
        self.realign_use_gpu = None

# this is the main method which differs between RNN and DNN training
def TrainNewModels(dir, iter, num_jobs, num_archives_processed, num_archives,
                   raw_model_string, egs_dir, frames_per_eg,
                   left_context, right_context,
                   momentum, max_param_change,
                   shuffle_buffer_size, minibatch_size,
                   run_opts):
      # We cannot easily use a single parallel SGE job to do the main training,
      # because the computation of which archive and which --frame option
      # to use for each job is a little complex, so we spawn each one separately.
      # this is no longer true for RNNs as we use do not use the --frame option
      # but we use the same script for consistency with FF-DNN code

    context_opts="--left-context={0} --right-context={1}".format(
                  left_context, right_context)
    processes = []
    for job in range(1,num_jobs+1):
        k = num_archives_processed + job - 1 # k is a zero-based index that we will derive
                                               # the other indexes from.
        archive_index = (k % num_archives) + 1 # work out the 1-based archive index.
        frame = (k / num_archives) % frames_per_eg
        process_handle = RunKaldiCommand("""
{command} {train_queue_opt} {dir}/log/train.{iter}.{job}.log \
  nnet3-train {parallel_train_opts} \
  --print-interval=10 --momentum={momentum} \
  --max-param-change={max_param_change} \
  "{raw_model}" \
  "ark,bg:nnet3-copy-egs --frame={frame} {context_opts} ark:{egs_dir}/egs.{archive_index}.ark ark:- | nnet3-shuffle-egs --buffer-size={shuffle_buffer_size} --srand={iter} ark:- ark:-| nnet3-merge-egs --minibatch-size={minibatch_size} --measure-output-frames=false --discard-partial-minibatches=true ark:- ark:- |" \
  {dir}/{next_iter}.{job}.raw
          """.format(command = run_opts.command,
                     train_queue_opt = run_opts.train_queue_opt,
                     dir = dir, iter = iter, next_iter = iter + 1, job = job,
                     parallel_train_opts = run_opts.parallel_train_opts,
                     frame = frame,
                     momentum = momentum, max_param_change = max_param_change,
                     raw_model = raw_model_string, context_opts = context_opts,
                     egs_dir = egs_dir, archive_index = archive_index,
                     shuffle_buffer_size = shuffle_buffer_size,
                     minibatch_size = minibatch_size),
          wait = False)

        processes.append(process_handle)

    all_success = True
    for process in processes:
        process.wait()
        [stdout_value, stderr_value] = process.communicate()
        print(stderr_value)
        if process.returncode != 0:
            all_success = False

    if not all_success:
        open('{0}/.error'.format(dir), 'w').close()
        raise Exception("There was error during training iteration {0}".format(iter))

def TrainOneIteration(dir, iter, egs_dir,
                      num_jobs, num_archives_processed, num_archives,
                      learning_rate, minibatch_size,
                      frames_per_eg, num_hidden_layers, add_layers_period,
                      left_context, right_context,
                      momentum, max_param_change, shuffle_buffer_size,
                      run_opts):



    # Set off jobs doing some diagnostics, in the background.
    # Use the egs dir from the previous iteration for the diagnostics
    logger.info("Training neural net (pass {0})".format(iter))

    ComputeTrainCvProbabilities(dir, iter, egs_dir, run_opts)

    if iter > 0:
        ComputeProgress(dir, iter, egs_dir, run_opts)

    if iter > 0 and (iter <= (num_hidden_layers-1) * add_layers_period) and (iter % add_layers_period == 0):

        do_average = False # if we've just mixed up, don't do averaging but take the
                           # best.
        cur_num_hidden_layers = 1 + iter / add_layers_period
        config_file = "{0}/configs/layer{1}.config".format(dir, cur_num_hidden_layers)
        raw_model_string = "nnet3-am-copy --raw=true --learning-rate={lr} {dir}/{iter}.mdl - | nnet3-init --srand={iter} - {config} - |".format(lr=learning_rate, dir=dir, iter=iter, config=config_file )
    else:
        do_average = True
        if iter == 0:
            do_average = False   # on iteration 0, pick the best, don't average.
        raw_model_string = "nnet3-am-copy --raw=true --learning-rate={0} {1}/{2}.mdl - |".format(learning_rate, dir, iter)

    if do_average:
      cur_minibatch_size = minibatch_size
      cur_max_param_change = max_param_change
    else:
      # on iteration zero or when we just added a layer, use a smaller minibatch
      # size (and we will later choose the output of just one of the jobs): the
      # model-averaging isn't always helpful when the model is changing too fast
      # (i.e. it can worsen the objective function), and the smaller minibatch
      # size will help to keep the update stable.
      cur_minibatch_size = minibatch_size / 2
      cur_max_param_change = float(max_param_change) / math.sqrt(2)

    try:
        os.remove("{0}/.error".format(dir))
    except OSError:
        pass

    TrainNewModels(dir, iter, num_jobs, num_archives_processed, num_archives,
                   raw_model_string, egs_dir, frames_per_eg,
                   left_context, right_context,
                   momentum, max_param_change,
                   shuffle_buffer_size, cur_minibatch_size,
                   run_opts)
    [models_to_average, best_model] = GetSuccessfulModels(num_jobs, '{0}/log/train.{1}.%.log'.format(dir,iter))
    nnets_list = []
    for n in models_to_average:
      nnets_list.append("{0}/{1}.{2}.raw".format(dir, iter + 1, n))

    if do_average:
        # average the output of the different jobs.
        RunKaldiCommand("""
{command} {dir}/log/average.{iter}.log \
nnet3-average {nnet_list} - \| \
nnet3-am-copy --set-raw-nnet=- {dir}/{iter}.mdl {dir}/{new_iter}.mdl
        """.format(command = run_opts.command,
                   dir = dir,
                   iter = iter,
                   nnet_list = " ".join(nnets_list),
                   new_iter = iter + 1))

    else:
        # choose the best model from different jobs
        RunKaldiCommand("""
{command} {dir}/log/select.{iter}.log \
    nnet3-am-copy --set-raw-nnet={dir}/{next_iter}.{best_model_index}.raw  {dir}/{iter}.mdl {dir}/{next_iter}.mdl
        """.format(command = run_opts.command,
                   dir = dir, iter = iter, next_iter = iter + 1,
                   best_model_index =  best_model))

    try:
        for i in range(1, num_jobs + 1):
            os.remove("{0}/{1}.{2}.raw".format(dir, iter + 1, i))
    except OSError:
        raise Exception("Error while trying to delete the raw models")

    new_model = "{0}/{1}.mdl".format(dir, iter + 1)

    if not os.path.isfile(new_model):
        raise Exception("Could not find {0}, at the end of iteration {1}".format(new_model, iter))
    elif os.stat(new_model).st_size == 0:
        raise Exception("{0} has size 0. Something went wrong in iteration {1}".format(new_model, iter))

# args is a Namespace with the required parameters
def Train(args, run_opts):
    arg_string = pprint.pformat(vars(args))
    logger.info("Arguments for the experiment\n{0}".format(arg_string))

    # Set some variables.
    num_leaves = GetNumberOfLeaves(args.ali_dir)
    num_jobs = GetNumberOfJobs(args.ali_dir)
    feat_dim = GetFeatDim(args.feat_dir)
    ivector_dim = GetIvectorDim(args.online_ivector_dir)

    # split the training data into parts for individual jobs
    # we will use the same number of jobs as that used for alignment
    SplitData(args.feat_dir, num_jobs)
    shutil.copy('{0}/tree'.format(args.ali_dir), args.dir)
    f = open('{0}/num_jobs'.format(args.dir), 'w')
    f.write(str(num_jobs))
    f.close()

    config_dir = '{0}/configs'.format(args.dir)
    var_file = '{0}/vars'.format(config_dir)

    [left_context, right_context, num_hidden_layers] = ParseModelConfigVarsFile(var_file)
    # Initialize as "raw" nnet, prior to training the LDA-like preconditioning
    # matrix.  This first config just does any initial splicing that we do;
    # we do this as it's a convenient way to get the stats for the 'lda-like'
    # transform.

    if (args.stage <= -5):
        logger.info("Initializing a basic network for estimating preconditioning matrix")
        RunKaldiCommand("""
{command} {dir}/log/nnet_init.log \
    nnet3-init --srand=-2 {dir}/configs/init.config {dir}/init.raw
    """.format(command = run_opts.command,
               dir = args.dir))

    default_egs_dir = '{0}/egs'.format(args.dir)
    if (args.stage <= -4) and args.egs_dir is None:
        logger.info("Generating egs")

        GenerateEgs(args.feat_dir, args.ali_dir, default_egs_dir,
                    left_context, right_context,
                    left_context, right_context, run_opts,
                    frames_per_eg = args.frames_per_eg,
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

    [egs_left_context, egs_right_context, frames_per_eg, num_archives] = VerifyEgsDir(egs_dir, feat_dim, ivector_dim, left_context, right_context)
    assert(args.frames_per_eg == frames_per_eg)

    if (args.num_jobs_final > num_archives):
        raise Exception('num_jobs_final cannot exceed the number of archives in the egs directory')

    # copy the properties of the egs to dir for
    # use during decoding
    CopyEgsPropertiesToExpDir(egs_dir, args.dir)

    if (args.stage <= -3):
        logger.info('Computing the preconditioning matrix for input features')

        ComputePreconditioningMatrix(args.dir, egs_dir, num_archives, run_opts,
                                     max_lda_jobs = args.max_lda_jobs,
                                     rand_prune = args.rand_prune)

    if (args.stage <= -2):
        logger.info("Computing initial vector for FixedScaleComponent before"
                    " softmax, using priors^{prior_scale} and rescaling to"
                    " average 1".format(prior_scale = args.presoftmax_prior_scale_power))

        ComputePresoftmaxPriorScale(args.dir, args.ali_dir, num_jobs, run_opts,
                                    presoftmax_prior_scale_power = args.presoftmax_prior_scale_power)


    if (args.stage <= -1):
        logger.info("Preparing the initial acoustic model.")
        PrepareInitialAcousticModel(args.dir, args.ali_dir, run_opts)


    # set num_iters so that as close as possible, we process the data $num_epochs
    # times, i.e. $num_iters*$avg_num_jobs) == $num_epochs*$num_archives,
    # where avg_num_jobs=(num_jobs_initial+num_jobs_final)/2.
    num_archives_expanded = num_archives * args.frames_per_eg
    num_archives_to_process = args.num_epochs * num_archives_expanded
    num_archives_processed = 0
    num_iters=(num_archives_to_process * 2) / (args.num_jobs_initial + args.num_jobs_final)

    num_iters_combine = VerifyIterations(num_iters, args.num_epochs,
                                         num_hidden_layers, num_archives_expanded,
                                         args.max_models_combine, args.add_layers_period,
                                         args.num_jobs_final)

    learning_rate = lambda iter, current_num_jobs, num_archives_processed: GetLearningRate(iter, current_num_jobs, num_iters,
                                                                   num_archives_processed,
                                                                    num_archives_to_process,
                                                                    args.initial_effective_lrate,
                                                                    args.final_effective_lrate)
    realign_iters = []
    if args.realign_times is not None:
        realign_iters = GetRealignIters(args.realign_times,
                                        num_iters,
                                        args.num_jobs_initial,
                                        args.num_jobs_final)
        print(realign_iters)
    # egs_dir will be updated if there is realignment
    cur_egs_dir=egs_dir

    logger.info("Training will run for {0} epochs = {1} iterations".format(args.num_epochs, num_iters))
    for iter in range(num_iters):
        if (args.exit_stage is not None) and (iter == args.exit_stage):
            logger.info("Exiting early due to --exit-stage {0}".format(iter))
            return
        current_num_jobs = int(0.5 + args.num_jobs_initial + (args.num_jobs_final - args.num_jobs_initial) * float(iter) / num_iters)

        if args.stage <= iter:
            if iter in realign_iters:
                logger.info("Re-aligning the data at iteration {0}".format(iter))
                prev_egs_dir=cur_egs_dir
                cur_egs_dir="{0}/egs_{1}".format(args.dir, "iter"+str(iter))
                new_ali_dir="{0}/ali_{1}".format(args.dir, "iter"+str(iter))
                Realign(args.dir, iter, args.feat_dir, args.lang,
                        prev_egs_dir, cur_egs_dir,
                        args.prior_subset_size, num_archives, run_opts,
                        transform_dir = args.transform_dir, online_ivector_dir = args.online_ivector_dir)
                if args.cleanup and args.egs_dir is None:
                    RemoveEgs(prev_egs_dir)
            model_file = "{dir}/{iter}.mdl".format(dir = args.dir, iter = iter)

            logger.info("On iteration {0}, learning rate is {1}.".format(iter, learning_rate(iter, current_num_jobs, num_archives_processed)))

            TrainOneIteration(args.dir, iter, egs_dir, current_num_jobs,
                              num_archives_processed, num_archives,
                              learning_rate(iter, current_num_jobs, num_archives_processed),
                              args.minibatch_size, args.frames_per_eg,
                              num_hidden_layers, args.add_layers_period,
                              left_context, right_context,
                              args.momentum, args.max_param_change,
                              args.shuffle_buffer_size, run_opts)
            if args.cleanup:
                # do a clean up everythin but the last 2 models, under certain conditions
                RemoveModel(args.dir, iter-2, num_iters, num_iters_combine,
                            args.preserve_model_interval)

            if args.email is not None:
                reporting_iter_interval = num_iters * args.reporting_interval
                if iter % reporting_iter_interval == 0:
                # lets do some reporting
                    [report, times, data] = nnet3_log_parse.GenerateAccuracyReport(args.dir)
                    message = report
                    subject = "Update : Expt {dir} : Iter {iter}".format(dir = args.dir, iter = iter)
                    sendMail(message, subject, args.email)

        num_archives_processed = num_archives_processed + current_num_jobs

    if args.stage <= num_iters:
        logger.info("Doing final combination to produce final.mdl")
        CombineModels(args.dir, num_iters, num_iters_combine, egs_dir, run_opts)

    if args.stage <= num_iters + 1:
        logger.info("Getting average posterior for purposes of adjusting the priors.")
        avg_post_vec_file = ComputeAveragePosterior(args.dir, 'combined', egs_dir,
                                num_archives, args.prior_subset_size, run_opts)

        logger.info("Re-adjusting priors based on computed posteriors")
        combined_model = "{dir}/combined.mdl".format(dir = args.dir)
        final_model = "{dir}/final.mdl".format(dir = args.dir)
        AdjustAmPriors(args.dir, combined_model, avg_post_vec_file, final_model, run_opts)

    if args.cleanup:
        logger.info("Cleaning up the experiment directory {0}".format(args.dir))
        remove_egs = args.remove_egs
        if args.egs_dir is not None:
            # this egs_dir was not created by this experiment so we will not
            # delete it
            remove_egs = False

        CleanNnetDir(args.dir, num_iters, cur_egs_dir,
                     preserve_model_interval = args.preserve_model_interval,
                     remove_egs = remove_egs)

    # do some reporting
    [report, times, data] = nnet3_log_parse.GenerateAccuracyReport(args.dir)
    if args.email is not None:
        SendMail(report, "Update : Expt {0} : complete".format(args.dir), args.email)

    report_handle = open("{dir}/accuracy.report".format(dir = args.dir), "w")
    report_handle.write(report)
    report_handle.close()

def Main():
    [args, run_opts] = GetArgs()
    try:
        Train(args, run_opts)
    except Exception as e:
        if args.email is not None:
            message = "Training session for experiment {dir} died due to an error.".format(dir = args.dir)
            sendMail(message, message, args.email)
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    Main()
