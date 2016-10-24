#!/usr/bin/env python

# Copyright 2016 Vijayaditya Peddinti.
#           2016 Vimal Manohar
# Apache 2.0.

# This is a module with methods which will be used by scripts for training of
# deep neural network acoustic model and raw model (i.e., generic neural
# network without transition model) with frame-level objectives.

import logging
import math
import imp

common_train_lib = imp.load_source('ntl', 'steps/nnet3/common_train_lib.py')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def AddCommonTrainArgs(parser):
    # feat options
    parser.add_argument("--feat.online-ivector-dir", type=str, dest='online_ivector_dir',
                        default = None, action = common_train_lib.NullstrToNoneAction,
                        help="""directory with the ivectors extracted in
                        an online fashion.""")
    parser.add_argument("--feat.cmvn-opts", type=str, dest='cmvn_opts',
                        default = None, action = common_train_lib.NullstrToNoneAction,
                        help="A string specifying '--norm-means' and '--norm-vars' values")

    # egs extraction options
    parser.add_argument("--egs.transform_dir", type=str, dest='transform_dir',
                        default = None, action = common_train_lib.NullstrToNoneAction,
                        help="""String to provide options directly to steps/nnet3/get_egs.sh script""")
    parser.add_argument("--egs.dir", type=str, dest='egs_dir',
                        default = None, action = common_train_lib.NullstrToNoneAction,
                        help="""Directory with egs. If specified this directory
                        will be used rather than extracting egs""")
    parser.add_argument("--egs.stage", type=int, dest='egs_stage',
                        default = 0, help="Stage at which get_egs.sh should be restarted")
    parser.add_argument("--egs.opts", type=str, dest='egs_opts',
                        default = None, action = common_train_lib.NullstrToNoneAction,
                        help="""String to provide options directly to steps/nnet3/get_egs.sh script""")

    # trainer options
    parser.add_argument("--trainer.srand", type=int, dest='srand',
                        default = 0,
                        help="Sets the random seed for model initialization and egs shuffling. "
                        "Warning: This random seed does not control all aspects of this experiment. "
                        "There might be other random seeds used in other stages of the experiment "
                        "like data preparation (e.g. volume perturbation).")
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
                        help=""" Controls randomization of the samples on each
                        iteration. If 0 or a large value the randomization is
                        complete, but this will consume memory and cause spikes
                        in disk I/O.  Smaller is easier on disk and memory but
                        less random.  It's not a huge deal though, as samples
                        are anyway randomized right at the start.
                        (the point of this is to get data in different
                        minibatches on different iterations, since in the
                        preconditioning method, 2 samples in the same minibatch
                        can affect each others' gradients.""")
    parser.add_argument("--trainer.add-layers-period", type=int, dest='add_layers_period',
                        default=2,
                        help="The number of iterations between adding layers"
                        "during layer-wise discriminative training.")
    parser.add_argument("--trainer.max-param-change", type=float, dest='max_param_change',
                        default=2.0,
                        help="""The maximum change in parameters allowed
                        per minibatch, measured in Frobenius norm over
                        the entire model""")
    parser.add_argument("--trainer.samples-per-iter", type=int, dest='samples_per_iter',
                        default=400000,
                        help="This is really the number of egs in each archive.")
    parser.add_argument("--trainer.lda.rand-prune", type=float, dest='rand_prune',
                        default=4.0,
                        help="""Value used in preconditioning matrix estimation""")
    parser.add_argument("--trainer.lda.max-lda-jobs", type=float, dest='max_lda_jobs',
                        default=10,
                        help="""Max number of jobs used for LDA stats accumulation""")

    # Parameters for the optimization
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
    parser.add_argument("--cmd", type=str, action = common_train_lib.NullstrToNoneAction,
                        dest = "command",
                        help="""Specifies the script to launch jobs.
                        e.g. queue.pl for launching on SGE cluster
                             run.pl for launching on local machine
                        """, default = "queue.pl")
    parser.add_argument("--egs.cmd", type=str, action = common_train_lib.NullstrToNoneAction,
                        dest = "egs_command",
                        help="""Script to launch egs jobs""", default = "queue.pl")
    parser.add_argument("--use-gpu", type=str, action = common_train_lib.StrToBoolAction,
                        choices = ["true", "false"],
                        help="Use GPU for training", default=True)
    parser.add_argument("--cleanup", type=str, action = common_train_lib.StrToBoolAction,
                        choices = ["true", "false"],
                        help="Clean up models after training", default=True)
    parser.add_argument("--cleanup.remove-egs", type=str, dest='remove_egs',
                        default = True, action = common_train_lib.StrToBoolAction,
                        choices = ["true", "false"],
                        help="""If true, remove egs after experiment""")
    parser.add_argument("--cleanup.preserve-model-interval", dest = "preserve_model_interval",
                        type=int, default=100,
                        help="Determines iterations for which models will be preserved during cleanup. If mod(iter,preserve_model_interval) == 0 model will be preserved.")

    parser.add_argument("--reporting.email", dest = "email",
                        type=str, default=None, action = common_train_lib.NullstrToNoneAction,
                        help=""" Email-id to report about the progress of the experiment.
                              NOTE: It assumes the machine on which the script is being run can send
                              emails from command line via. mail program. The
                              Kaldi mailing list will not support this feature.
                              It might require local expertise to setup. """)
    parser.add_argument("--reporting.interval", dest = "reporting_interval",
                        type=int, default=0.1,
                        help="Frequency with which reports have to be sent, measured in terms of fraction of iterations. If 0 and reporting mail has been specified then only failure notifications are sent")

# this is the main method which differs between RNN and DNN training
def TrainNewModels(dir, iter, srand, num_jobs,
                   num_archives_processed, num_archives,
                   raw_model_string, egs_dir, frames_per_eg,
                   left_context, right_context,
                   momentum, max_param_change,
                   shuffle_buffer_size, minibatch_size,
                   cache_read_opt, run_opts):
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

        cache_write_opt = ""
        if job == 1:
            # an option for writing cache (storing pairs of nnet-computations and
            # computation-requests) during training.
            cache_write_opt="--write-cache={dir}/cache.{iter}".format(dir=dir, iter=iter+1)

        process_handle = common_train_lib.RunKaldiCommand("""
{command} {train_queue_opt} {dir}/log/train.{iter}.{job}.log \
  nnet3-train {parallel_train_opts} {cache_read_opt} {cache_write_opt} \
  --print-interval=10 --momentum={momentum} \
  --max-param-change={max_param_change} \
  "{raw_model}" \
  "ark,bg:nnet3-copy-egs --frame={frame} {context_opts} ark:{egs_dir}/egs.{archive_index}.ark ark:- | nnet3-shuffle-egs --buffer-size={shuffle_buffer_size} --srand={srand} ark:- ark:-| nnet3-merge-egs --minibatch-size={minibatch_size} --measure-output-frames=false --discard-partial-minibatches=true ark:- ark:- |" \
  {dir}/{next_iter}.{job}.raw
          """.format(command = run_opts.command,
                     train_queue_opt = run_opts.train_queue_opt,
                     dir = dir, iter = iter, srand = iter + srand, next_iter = iter + 1, job = job,
                     parallel_train_opts = run_opts.parallel_train_opts,
                     cache_read_opt = cache_read_opt, cache_write_opt = cache_write_opt,
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

def TrainOneIteration(dir, iter, srand, egs_dir,
                      num_jobs, num_archives_processed, num_archives,
                      learning_rate, minibatch_size,
                      frames_per_eg, num_hidden_layers, add_layers_period,
                      left_context, right_context,
                      momentum, max_param_change, shuffle_buffer_size,
                      run_opts,
                      get_raw_nnet_from_am = True):


    # Set off jobs doing some diagnostics, in the background.
    # Use the egs dir from the previous iteration for the diagnostics
    logger.info("Training neural net (pass {0})".format(iter))

    # check if different iterations use the same random seed
    if os.path.exists('{0}/srand'.format(dir)):
        try:
            saved_srand = int(open('{0}/srand'.format(dir), 'r').readline().strip())
        except IOError, ValueError:
            raise Exception('Exception while reading the random seed for training')
        if srand != saved_srand:
            logger.warning("The random seed provided to this iteration (srand={0}) is different from the one saved last time (srand={1}). Using srand={0}.".format(srand, saved_srand))
    else:
        f = open('{0}/srand'.format(dir), 'w')
        f.write(str(srand))
        f.close()

    # Sets off some background jobs to compute train and
    # validation set objectives
    train_lib.ComputeTrainCvProbabilities(dir, iter, egs_dir, run_opts,
                                          get_raw_nnet_from_am = get_raw_nnet_from_am)

    if iter > 0:
        # Runs in the background
        train_lib.ComputeProgress(dir, iter, egs_dir, run_opts,
                                  get_raw_nnet_from_am = get_raw_nnet_from_am)

    # an option for writing cache (storing pairs of nnet-computations
    # and computation-requests) during training.
    cache_read_opt = ""
    if iter > 0 and (iter <= (num_hidden_layers-1) * add_layers_period) and (iter % add_layers_period == 0):

        do_average = False # if we've just added new hiden layer, don't do
                           # averaging but take the best.
        cur_num_hidden_layers = 1 + iter / add_layers_period
        config_file = "{0}/configs/layer{1}.config".format(dir, cur_num_hidden_layers)
        if get_raw_nnet_from_am:
            raw_model_string = "nnet3-am-copy --raw=true --learning-rate={lr} {dir}/{iter}.mdl - | nnet3-init --srand={srand} - {config} - |".format(lr=learning_rate, dir=dir, iter=iter, srand=iter + srand, config=config_file)
        else:
            raw_model_string = "nnet3-copy --learning-rate={lr} {dir}/{iter}.raw - | nnet3-init --srand={srand} - {config} - |".format(lr=learning_rate, dir=dir, iter=iter, srand=iter + srand, config=config_file)
    else:
        do_average = True
        if iter == 0:
            do_average = False   # on iteration 0, pick the best, don't average.
        else:
            cache_read_opt = "--read-cache={dir}/cache.{iter}".format(dir=dir, iter=iter)
        if get_raw_nnet_from_am:
            raw_model_string = "nnet3-am-copy --raw=true --learning-rate={0} {1}/{2}.mdl - |".format(learning_rate, dir, iter)
        else:
            raw_model_string = "nnet3-copy --learning-rate={lr} {dir}/{iter}.raw - |".format(lr = learning_rate, dir = dir, iter = iter)

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

    train_lib.TrainNewModels(dir, iter, srand, num_jobs, num_archives_processed, num_archives,
                             raw_model_string, egs_dir, frames_per_eg,
                             left_context, right_context,
                             momentum, max_param_change,
                             shuffle_buffer_size, cur_minibatch_size,
                             cache_read_opt, run_opts)
    [models_to_average, best_model] = common_train_lib.GetSuccessfulModels(num_jobs, '{0}/log/train.{1}.%.log'.format(dir,iter))
    nnets_list = []
    for n in models_to_average:
        nnets_list.append("{0}/{1}.{2}.raw".format(dir, iter + 1, n))

    if do_average:
        # average the output of the different jobs.
        common_train_lib.GetAverageNnetModel(
                        dir = dir, iter = iter,
                        nnets_list = " ".join(nnets_list),
                        run_opts = run_opts,
                        get_raw_nnet_from_am = get_raw_nnet_from_am)
    else:
        # choose the best model from different jobs
        common_train_lib.GetBestNnetModel(
                        dir = dir, iter = iter,
                        best_model_index = best_model,
                        run_opts = run_opts,
                        get_raw_nnet_from_am = get_raw_nnet_from_am)

    try:
        for i in range(1, num_jobs + 1):
            os.remove("{0}/{1}.{2}.raw".format(dir, iter + 1, i))
    except OSError:
        raise Exception("Error while trying to delete the raw models")

    if get_raw_nnet_from_am:
        new_model = "{0}/{1}.mdl".format(dir, iter + 1)
    else:
        new_model = "{0}/{1}.raw".format(dir, iter + 1)

    if not os.path.isfile(new_model):
        raise Exception("Could not find {0}, at the end of iteration {1}".format(new_model, iter))
    elif os.stat(new_model).st_size == 0:
        raise Exception("{0} has size 0. Something went wrong in iteration {1}".format(new_model, iter))
    if cache_read_opt and os.path.exists("{0}/cache.{1}".format(dir, iter)):
        os.remove("{0}/cache.{1}".format(dir, iter))

def GenerateEgs(data, alidir, egs_dir,
                left_context, right_context,
                valid_left_context, valid_right_context,
                run_opts, stage = 0,
                feat_type = 'raw', online_ivector_dir = None,
                samples_per_iter = 20000, frames_per_eg = 20, srand = 0,
                egs_opts = None, cmvn_opts = None, transform_dir = None):

    common_train_lib.RunKaldiCommand("""
steps/nnet3/get_egs.sh {egs_opts} \
  --cmd "{command}" \
  --cmvn-opts "{cmvn_opts}" \
  --feat-type {feat_type} \
  --transform-dir "{transform_dir}" \
  --online-ivector-dir "{ivector_dir}" \
  --left-context {left_context} --right-context {right_context} \
  --valid-left-context {valid_left_context} \
  --valid-right-context {valid_right_context} \
  --stage {stage} \
  --samples-per-iter {samples_per_iter} \
  --frames-per-eg {frames_per_eg} \
  --srand {srand} \
  {data} {alidir} {egs_dir}
      """.format(command = run_opts.command,
          cmvn_opts = cmvn_opts if cmvn_opts is not None else '',
          feat_type = feat_type,
          transform_dir = transform_dir if transform_dir is not None else '',
          ivector_dir = online_ivector_dir if online_ivector_dir is not None else '',
          left_context = left_context, right_context = right_context,
          valid_left_context = valid_left_context,
          valid_right_context = valid_right_context,
          stage = stage, samples_per_iter = samples_per_iter,
          frames_per_eg = frames_per_eg, srand = srand, data = data, alidir = alidir,
          egs_dir = egs_dir,
          egs_opts = egs_opts if egs_opts is not None else '' ))

# This method generates egs directly from an scp file of targets, instead of
# getting them from the alignments (as with the method GenerateEgs).
# The targets are in matrix format for target_type="dense" and in posterior
# format for target_type="sparse".
# If using sparse targets, num_targets must be explicity specified.
# If using dense targets, num_targets is computed by reading the feature matrix dimension.
def GenerateEgsUsingTargets(data, targets_scp, egs_dir,
                left_context, right_context,
                valid_left_context, valid_right_context,
                run_opts, stage = 0,
                feat_type = 'raw', online_ivector_dir = None,
                target_type = 'dense', num_targets = -1,
                samples_per_iter = 20000, frames_per_eg = 20, srand = 0,
                egs_opts = None, cmvn_opts = None, transform_dir = None):
    if target_type == 'dense':
        num_targets = common_train_lib.GetFeatDimFromScp(targets_scp)
    else:
        if num_targets == -1:
            raise Exception("--num-targets is required if target-type is dense")

    common_train_lib.RunKaldiCommand("""
steps/nnet3/get_egs_targets.sh {egs_opts} \
  --cmd "{command}" \
  --cmvn-opts "{cmvn_opts}" \
  --feat-type {feat_type} \
  --transform-dir "{transform_dir}" \
  --online-ivector-dir "{ivector_dir}" \
  --left-context {left_context} --right-context {right_context} \
  --valid-left-context {valid_left_context} \
  --valid-right-context {valid_right_context} \
  --stage {stage} \
  --samples-per-iter {samples_per_iter} \
  --frames-per-eg {frames_per_eg} \
  --srand {srand} \
  --target-type {target_type} \
  --num-targets {num_targets} \
  {data} {targets_scp} {egs_dir}
      """.format(command = run_opts.egs_command,
          cmvn_opts = cmvn_opts if cmvn_opts is not None else '',
          feat_type = feat_type,
          transform_dir = transform_dir if transform_dir is not None else '',
          ivector_dir = online_ivector_dir if online_ivector_dir is not None else '',
          left_context = left_context, right_context = right_context,
          valid_left_context = valid_left_context,
          valid_right_context = valid_right_context,
          stage = stage, samples_per_iter = samples_per_iter,
          frames_per_eg = frames_per_eg, srand = srand,
          num_targets = num_targets,
          data = data,
          targets_scp = targets_scp, target_type = target_type,
          egs_dir = egs_dir,
          egs_opts = egs_opts if egs_opts is not None else '' ))

def ComputePreconditioningMatrix(dir, egs_dir, num_lda_jobs, run_opts,
                                 max_lda_jobs = None, rand_prune = 4.0,
                                 lda_opts = None):
    if max_lda_jobs is not None:
        if num_lda_jobs > max_lda_jobs:
            num_lda_jobs = max_lda_jobs

    common_train_lib.RunKaldiCommand("""
{command} JOB=1:{num_lda_jobs} {dir}/log/get_lda_stats.JOB.log \
 nnet3-acc-lda-stats --rand-prune={rand_prune} \
    {dir}/init.raw "ark:{egs_dir}/egs.JOB.ark" {dir}/JOB.lda_stats""".format(
        command = run_opts.command,
        num_lda_jobs = num_lda_jobs,
        dir = dir,
        egs_dir = egs_dir,
        rand_prune = rand_prune))

    # the above command would have generated dir/{1..num_lda_jobs}.lda_stats
    lda_stat_files = map(lambda x: '{0}/{1}.lda_stats'.format(dir, x),
                         range(1, num_lda_jobs + 1))

    common_train_lib.RunKaldiCommand("""
{command} {dir}/log/sum_transform_stats.log \
    sum-lda-accs {dir}/lda_stats {lda_stat_files}""".format(
        command = run_opts.command,
        dir = dir, lda_stat_files = " ".join(lda_stat_files)))

    for file in lda_stat_files:
        try:
            os.remove(file)
        except OSError:
            raise Exception("There was error while trying to remove lda stat files.")
    # this computes a fixed affine transform computed in the way we described in
    # Appendix C.6 of http://arxiv.org/pdf/1410.7455v6.pdf; it's a scaled variant
    # of an LDA transform but without dimensionality reduction.

    common_train_lib.RunKaldiCommand("""
{command} {dir}/log/get_transform.log \
 nnet-get-feature-transform {lda_opts} {dir}/lda.mat {dir}/lda_stats
     """.format(command = run_opts.command,dir = dir,
                lda_opts = lda_opts if lda_opts is not None else ""))

    common_train_lib.ForceSymlink("../lda.mat", "{0}/configs/lda.mat".format(dir))


def PrepareInitialAcousticModel(dir, alidir, run_opts):
    """ Adds the first layer; this will also add in the lda.mat and
        presoftmax_prior_scale.vec. It will also prepare the acoustic model
        with the transition model."""

    common_train_lib.PrepareInitialNetwork(dir, run_opts)

  # Convert to .mdl, train the transitions, set the priors.
    common_train_lib.RunKaldiCommand("""
{command} {dir}/log/init_mdl.log \
    nnet3-am-init {alidir}/final.mdl {dir}/0.raw - \| \
    nnet3-am-train-transitions - "ark:gunzip -c {alidir}/ali.*.gz|" {dir}/0.mdl
        """.format(command = run_opts.command,
                   dir = dir, alidir = alidir))


def ComputeTrainCvProbabilities(dir, iter, egs_dir, run_opts, mb_size=256,
                                wait = False, get_raw_nnet_from_am = True):

    if get_raw_nnet_from_am:
        model = "nnet3-am-copy --raw=true {dir}/{iter}.mdl - |".format(dir = dir, iter = iter)
    else:
        model = "{dir}/{iter}.raw".format(dir = dir, iter = iter)

    common_train_lib.RunKaldiCommand("""
{command} {dir}/log/compute_prob_valid.{iter}.log \
  nnet3-compute-prob {compute_prob_opts} "{model}" \
        "ark,bg:nnet3-merge-egs --minibatch-size={mb_size} ark:{egs_dir}/valid_diagnostic.egs ark:- |"
    """.format(command = run_opts.command,
               dir = dir,
               iter = iter,
               mb_size = mb_size,
               model = model,
               compute_prob_opts = compute_prob_opts,
               egs_dir = egs_dir), wait = wait)

    common_train_lib.RunKaldiCommand("""
{command} {dir}/log/compute_prob_train.{iter}.log \
  nnet3-compute-prob {compute_prob_opts} "{model}" \
       "ark,bg:nnet3-merge-egs --minibatch-size={mb_size} ark:{egs_dir}/train_diagnostic.egs ark:- |"
    """.format(command = run_opts.command,
               dir = dir,
               iter = iter,
               mb_size = mb_size,
               model = model,
               compute_prob_opts = compute_prob_opts,
               egs_dir = egs_dir), wait = wait)

def ComputeProgress(dir, iter, egs_dir, run_opts, mb_size=256, wait=False,
                    get_raw_nnet_from_am = True):
    if get_raw_nnet_from_am:
        prev_model = "nnet3-am-copy --raw=true {dir}/{iter}.mdl - |".format(dir, iter - 1)
        model = "nnet3-am-copy --raw=true {dir}/{iter}.mdl - |".format(dir, iter)
    else:
        prev_model = '{0}/{1}.raw'.format(dir, iter - 1)
        model = '{0}/{1}.raw'.format(dir, iter)

    common_train_lib.RunKaldiCommand("""
{command} {dir}/log/progress.{iter}.log \
nnet3-info {model} '&&' \
nnet3-show-progress --use-gpu=no {prev_model} {model} \
"ark,bg:nnet3-merge-egs --minibatch-size={mb_size} ark:{egs_dir}/train_diagnostic.egs ark:-|"
    """.format(command = run_opts.command,
               dir = dir,
               iter = iter,
               model = model,
               mb_size = mb_size,
               prev_model = prev_model,
               egs_dir = egs_dir), wait = wait)

def CombineModels(dir, num_iters, num_iters_combine, egs_dir,
                  run_opts, chunk_width = None,
                  get_raw_nnet_from_am = True):
    # Now do combination.  In the nnet3 setup, the logic
    # for doing averaging of subsets of the models in the case where
    # there are too many models to reliably esetimate interpolation
    # factors (max_models_combine) is moved into the nnet3-combine
    raw_model_strings = []
    print num_iters_combine
    for iter in range(num_iters - num_iters_combine + 1, num_iters + 1):
      if get_raw_nnet_from_am:
          model_file = '{0}/{1}.mdl'.format(dir, iter)
          if not os.path.exists(model_file):
              raise Exception('Model file {0} missing'.format(model_file))
          raw_model_strings.append('"nnet3-am-copy --raw=true {0} -|"'.format(model_file))
      else:
          model_file = '{0}/{1}.raw'.format(dir, iter)
          if not os.path.exists(model_file):
              raise Exception('Model file {0} missing'.format(model_file))
          raw_model_strings.append(model_file)

    if chunk_width is not None:
        # this is an RNN model
        mbsize = int(1024.0/(chunk_width))
    else:
        mbsize = 1024

    if get_raw_nnet_from_am:
        out_model = "|nnet3-am-copy --set-raw-nnet=- {dir}/{num_iters}.mdl {dir}/combined.mdl".format(dir = dir, num_iters = num_iters)
    else:
        out_model = '{dir}/final.raw'.format(dir = dir)

    common_train_lib.RunKaldiCommand("""
{command} {combine_queue_opt} {dir}/log/combine.log \
nnet3-combine --num-iters=40 \
   --enforce-sum-to-one=true --enforce-positive-weights=true \
   --verbose=3 {raw_models} "ark,bg:nnet3-merge-egs --measure-output-frames=false --minibatch-size={mbsize} ark:{egs_dir}/combine.egs ark:-|" \
   {out_model}
   """.format(command = run_opts.command,
               combine_queue_opt = run_opts.combine_queue_opt,
               dir = dir, raw_models = " ".join(raw_model_strings),
               mbsize = mbsize,
               out_model = out_model,
               egs_dir = egs_dir))

    # Compute the probability of the final, combined model with
    # the same subset we used for the previous compute_probs, as the
    # different subsets will lead to different probs.
    if get_raw_nnet_from_am:
        train_lib.ComputeTrainCvProbabilities(dir, 'combined', egs_dir, run_opts, wait = False)
    else:
        train_lib.ComputeTrainCvProbabilities(dir, 'final', egs_dir, run_opts,
                                              wait = False, get_raw_nnet_from_am = False)

