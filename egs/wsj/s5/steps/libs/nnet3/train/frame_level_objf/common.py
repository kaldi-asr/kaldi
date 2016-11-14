

# Copyright 2016 Vijayaditya Peddinti.
#           2016 Vimal Manohar
# Apache 2.0.

""" This is a module with methods which will be used by scripts for training of
deep neural network acoustic model and raw model (i.e., generic neural
network without transition model) with frame-level objectives.
"""

import logging
import math
import imp
import os
import sys

sys.path.append("steps/libs")
import nnet3.train.common as common_train_lib
import common as common_lib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def TrainNewModels(dir, iter, srand, num_jobs,
                   num_archives_processed, num_archives,
                   raw_model_string, egs_dir,
                   left_context, right_context,
                   momentum, max_param_change,
                   shuffle_buffer_size, minibatch_size,
                   cache_read_opt, run_opts,
                   frames_per_eg=-1, min_deriv_time=None):
    """ Called from TrainOneIteration, this model does one iteration of training
    with 'num_jobs' jobs, and
    writes files like exp/tdnn_a/24.{1,2,3,..<num_jobs>}.raw

    We cannot easily use a single parallel SGE job to do the main training,
    because the computation of which archive and which --frame option
    to use for each job is a little complex, so we spawn each one separately.
    this is no longer true for RNNs as we use do not use the --frame option
    but we use the same script for consistency with FF-DNN code

    Args:
        frames_per_eg: The default value -1 implies chunk_level_training, which
            is particularly applicable to RNN training. If it is > 0, then it
            implies frame-level training, which is applicable for DNN training.
            If it is > 0, then each parallel SGE job created, a different frame
            numbered 0..frames_per_eg-1 is used.
        min_deriv_time: Applicable for RNN training. A default value of None
            implies a min_deriv_time of 0 is used. During RNN training, its value
            is set to chunk_width - num_bptt_steps in the training script.
    """

    chunk_level_training = False if frames_per_eg > 0 else True

    deriv_time_opts = (""
                       if min_deriv_time is None
                       else "--optimization.min-deriv-time={0}".format(min_deriv_time)
                       )

    context_opts = "--left-context={0} --right-context={1}".format(
        left_context, right_context)

    processes = []
    for job in range(1,num_jobs+1):
        k = num_archives_processed + job - 1 # k is a zero-based index that we will derive
                                               # the other indexes from.
        archive_index = (k % num_archives) + 1 # work out the 1-based archive index.

        if not chunk_level_training:
            frame = (k / num_archives) % frames_per_eg

        cache_write_opt = ""
        if job == 1:
            # an option for writing cache (storing pairs of nnet-computations and
            # computation-requests) during training.
            cache_write_opt="--write-cache={dir}/cache.{iter}".format(dir=dir, iter=iter+1)

        process_handle = common_lib.RunKaldiCommand("""
{command} {train_queue_opt} {dir}/log/train.{iter}.{job}.log \
    nnet3-train {parallel_train_opts} {cache_read_opt} {cache_write_opt} \
    --print-interval=10 --momentum={momentum} \
    --max-param-change={max_param_change} \
    {deriv_time_opts} "{raw_model}" \
    "ark,bg:nnet3-copy-egs {frame_opts} {context_opts} ark:{egs_dir}/egs.{archive_index}.ark ark:- |"""
    """nnet3-shuffle-egs --buffer-size={shuffle_buffer_size} --srand={srand} ark:- ark:-| """
    """nnet3-merge-egs --minibatch-size={minibatch_size} --measure-output-frames=false """
    """--discard-partial-minibatches=true ark:- ark:- |" \
    {dir}/{next_iter}.{job}.raw
    """.format(command=run_opts.command,
               train_queue_opt=run_opts.train_queue_opt,
               dir=dir, iter=iter, srand=iter + srand, next_iter=iter + 1,
               job=job, parallel_train_opts=run_opts.parallel_train_opts,
               cache_read_opt=cache_read_opt, cache_write_opt=cache_write_opt,
               frame_opts=""
                          if chunk_level_training
                          else "--frame={0}".format(frame),
               momentum=momentum, max_param_change=max_param_change,
               deriv_time_opts=deriv_time_opts,
               raw_model=raw_model_string, context_opts=context_opts,
               egs_dir=egs_dir, archive_index=archive_index,
               shuffle_buffer_size=shuffle_buffer_size,
               minibatch_size=minibatch_size),
            background_process_handler=background_process_handler)

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
                      num_hidden_layers, add_layers_period,
                      left_context, right_context,
                      momentum, max_param_change, shuffle_buffer_size,
                      run_opts,
                      cv_minibatch_size=256, frames_per_eg=-1,
                      min_deriv_time=None, shrinkage_value=1.0,
                      get_raw_nnet_from_am=True):
    """ Called from steps/nnet3/train_*.py scripts for one iteration of training

    Args:
        frames_per_eg: The default value -1 implies chunk_level_training, which
            is particularly applicable to RNN training. If it is > 0, then it
            implies frame-level training, which is applicable for DNN training.
            If it is > 0, then each parallel SGE job created, a different frame
            numbered 0..frames_per_eg-1 is used.
        min_deriv_time: Applicable for RNN training. A default value of None
            implies a min_deriv_time of 0 is used. During RNN training, its value
            is set to chunk_width - num_bptt_steps in the training script.
        shrinkage_value: If value is 1.0, no shrinkage is done; otherwise
            parameter values are scaled by this value.
        get_raw_nnet_from_am: If True, then the network is read and stored as
            acoustic model i.e. along with transition model e.g. 10.mdl
            as against a raw network e.g. 10.raw when the value is False.
    """


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
    ComputeTrainCvProbabilities(dir, iter, egs_dir, run_opts,
                                mb_size=cv_minibatch_size,
                                get_raw_nnet_from_am=get_raw_nnet_from_am)

    if iter > 0:
        # Runs in the background
        ComputeProgress(dir, iter, egs_dir, run_opts,
                        mb_size=cv_minibatch_size,
                        get_raw_nnet_from_am=get_raw_nnet_from_am)

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
            raw_model_string = "nnet3-copy --learning-rate={lr} {dir}/{iter}.raw - |".format(lr=learning_rate, dir=dir, iter=iter)

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

    TrainNewModels(dir, iter, srand, num_jobs,
                   num_archives_processed, num_archives,
                   raw_model_string, egs_dir,
                   left_context, right_context,
                   momentum, max_param_change,
                   shuffle_buffer_size, cur_minibatch_size,
                   cache_read_opt, run_opts,
                   frames_per_eg=frames_per_eg,
                   min_deriv_time=min_deriv_time)

    [models_to_average, best_model] = common_train_lib.GetSuccessfulModels(
            num_jobs, '{0}/log/train.{1}.%.log'.format(dir,iter))
    nnets_list = []
    for n in models_to_average:
        nnets_list.append("{0}/{1}.{2}.raw".format(dir, iter + 1, n))

    if do_average:
        # average the output of the different jobs.
        common_train_lib.GetAverageNnetModel(
            dir=dir, iter=iter,
            nnets_list=" ".join(nnets_list),
            run_opts=run_opts,
            get_raw_nnet_from_am=get_raw_nnet_from_am,
            shrink=shrinkage_value)

    else:
        # choose the best model from different jobs
        common_train_lib.GetBestNnetModel(
            dir=dir, iter=iter,
            best_model_index=best_model,
            run_opts=run_opts,
            get_raw_nnet_from_am=get_raw_nnet_from_am,
            shrink=shrinkage_value)

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


def ComputePreconditioningMatrix(dir, egs_dir, num_lda_jobs, run_opts,
                                 max_lda_jobs=None, rand_prune=4.0,
                                 lda_opts=None):
    if max_lda_jobs is not None:
        if num_lda_jobs > max_lda_jobs:
            num_lda_jobs = max_lda_jobs

    common_lib.RunKaldiCommand("""
{command} JOB=1:{num_lda_jobs} {dir}/log/get_lda_stats.JOB.log \
nnet3-acc-lda-stats --rand-prune={rand_prune} \
    {dir}/init.raw "ark:{egs_dir}/egs.JOB.ark" {dir}/JOB.lda_stats""".format(
        command=run_opts.command,
        num_lda_jobs=num_lda_jobs,
        dir=dir,
        egs_dir=egs_dir,
        rand_prune=rand_prune))

    # the above command would have generated dir/{1..num_lda_jobs}.lda_stats
    lda_stat_files = map(lambda x: '{0}/{1}.lda_stats'.format(dir, x),
                         range(1, num_lda_jobs + 1))

    common_lib.RunKaldiCommand("""
{command} {dir}/log/sum_transform_stats.log \
    sum-lda-accs {dir}/lda_stats {lda_stat_files}""".format(
        command=run_opts.command,
        dir=dir, lda_stat_files=" ".join(lda_stat_files)))

    for file in lda_stat_files:
        try:
            os.remove(file)
        except OSError:
            raise Exception("There was error while trying to remove lda stat files.")
    # this computes a fixed affine transform computed in the way we described in
    # Appendix C.6 of http://arxiv.org/pdf/1410.7455v6.pdf; it's a scaled variant
    # of an LDA transform but without dimensionality reduction.

    common_lib.RunKaldiCommand("""
{command} {dir}/log/get_transform.log \
    nnet-get-feature-transform {lda_opts} {dir}/lda.mat {dir}/lda_stats
    """.format(command=run_opts.command,dir=dir,
               lda_opts=lda_opts if lda_opts is not None else ""))

    common_lib.ForceSymlink("../lda.mat", "{0}/configs/lda.mat".format(dir))


def PrepareInitialAcousticModel(dir, alidir, run_opts):
    """ Adds the first layer; this will also add in the lda.mat and
        presoftmax_prior_scale.vec. It will also prepare the acoustic model
        with the transition model."""

    common_lib.PrepareInitialNetwork(dir, run_opts)

  # Convert to .mdl, train the transitions, set the priors.
    common_lib.RunKaldiCommand("""
{command} {dir}/log/init_mdl.log \
    nnet3-am-init {alidir}/final.mdl {dir}/0.raw - \| \
    nnet3-am-train-transitions - "ark:gunzip -c {alidir}/ali.*.gz|" {dir}/0.mdl
    """.format(command=run_opts.command,
               dir=dir, alidir=alidir))


def ComputeTrainCvProbabilities(dir, iter, egs_dir, run_opts,
                                mb_size=256, background_process_handler=None,
                                get_raw_nnet_from_am=True):

    if get_raw_nnet_from_am:
        model = "nnet3-am-copy --raw=true {dir}/{iter}.mdl - |".format(dir=dir, iter=iter)
    else:
        model = "{dir}/{iter}.raw".format(dir=dir, iter=iter)

    common_lib.RunKaldiCommand("""
{command} {dir}/log/compute_prob_valid.{iter}.log \
    nnet3-compute-prob "{model}" \
    "ark,bg:nnet3-merge-egs --minibatch-size={mb_size} ark:{egs_dir}/valid_diagnostic.egs ark:- |"
    """.format(command=run_opts.command,
               dir=dir,
               iter=iter,
               mb_size=mb_size,
               model=model,
               egs_dir=egs_dir),
            background_process_handler=background_process_handler)

    common_lib.RunKaldiCommand("""
{command} {dir}/log/compute_prob_train.{iter}.log \
    nnet3-compute-prob "{model}" \
    "ark,bg:nnet3-merge-egs --minibatch-size={mb_size} ark:{egs_dir}/train_diagnostic.egs ark:- |"
    """.format(command=run_opts.command,
               dir=dir,
               iter=iter,
               mb_size=mb_size,
               model=model,
               egs_dir=egs_dir),
            background_process_handler=background_process_handler)


def ComputeProgress(dir, iter, egs_dir, run_opts,
                    mb_size=256, background_process_handler=None,
                    get_raw_nnet_from_am=True):
    if get_raw_nnet_from_am:
        prev_model = "nnet3-am-copy --raw=true {0}/{1}.mdl - |".format(dir, iter - 1)
        model = "nnet3-am-copy --raw=true {0}/{1}.mdl - |".format(dir, iter)
    else:
        prev_model = '{0}/{1}.raw'.format(dir, iter - 1)
        model = '{0}/{1}.raw'.format(dir, iter)

    common_lib.RunKaldiCommand("""
{command} {dir}/log/progress.{iter}.log \
    nnet3-info {model} '&&' \
    nnet3-show-progress --use-gpu=no {prev_model} {model} \
    "ark,bg:nnet3-merge-egs --minibatch-size={mb_size} ark:{egs_dir}/train_diagnostic.egs ark:-|"
    """.format(command=run_opts.command,
               dir=dir,
               iter=iter,
               model=model,
               mb_size=mb_size,
               prev_model=prev_model,
               egs_dir=egs_dir),
            background_process_handler=background_process_handler)


def CombineModels(dir, num_iters, models_to_combine, egs_dir,
                  run_opts, background_process_handler=None,
                  chunk_width=None,
                  get_raw_nnet_from_am=True):
    """
    Now do combination.  In the nnet3 setup, the logic
    for doing averaging of subsets of the models in the case where
    there are too many models to reliably esetimate interpolation
    factors (max_models_combine) is moved into the nnet3-combine.
    """
    raw_model_strings = []
    print len(models_to_combine)

    models_to_combine.add(num_iters)

    for iter in models_to_combine:
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
        out_model = "| nnet3-am-copy --set-raw-nnet=- {dir}/{num_iters}.mdl {dir}/combined.mdl".format(dir=dir, num_iters=num_iters)
    else:
        out_model = '{dir}/final.raw'.format(dir=dir)

    common_lib.RunKaldiCommand("""
{command} {combine_queue_opt} {dir}/log/combine.log \
nnet3-combine --num-iters=40 \
   --enforce-sum-to-one=true --enforce-positive-weights=true \
   --verbose=3 {raw_models} "ark,bg:nnet3-merge-egs --measure-output-frames=false --minibatch-size={mbsize} ark:{egs_dir}/combine.egs ark:-|" \
   "{out_model}"
   """.format(command=run_opts.command,
               combine_queue_opt=run_opts.combine_queue_opt,
               dir=dir, raw_models=" ".join(raw_model_strings),
               mbsize=mbsize,
               out_model=out_model,
               egs_dir=egs_dir))

    # Compute the probability of the final, combined model with
    # the same subset we used for the previous compute_probs, as the
    # different subsets will lead to different probs.
    if get_raw_nnet_from_am:
        ComputeTrainCvProbabilities(dir, 'combined', egs_dir, run_opts,
                                    background_process_handler=background_process_handler)
    else:
        ComputeTrainCvProbabilities(dir, 'final', egs_dir, run_opts,
                                    background_process_handler=background_process_handler,
                                    get_raw_nnet_from_am=False)


