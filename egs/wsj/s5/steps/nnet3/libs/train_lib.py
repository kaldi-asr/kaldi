import subprocess
import logging
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


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

        process_handle = RunKaldiCommand("""
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
                      compute_accuracy = True,
                      use_raw_nnet = False):


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

    ComputeTrainCvProbabilities(dir, iter, egs_dir, run_opts, use_raw_nnet = use_raw_nnet, compute_accuracy = compute_accuracy)

    if iter > 0:
        ComputeProgress(dir, iter, egs_dir, run_opts, use_raw_nnet = use_raw_nnet)

    # an option for writing cache (storing pairs of nnet-computations
    # and computation-requests) during training.
    cache_read_opt = ""
    if iter > 0 and (iter <= (num_hidden_layers-1) * add_layers_period) and (iter % add_layers_period == 0):

        do_average = False # if we've just mixed up, don't do averaging but take the
                           # best.
        cur_num_hidden_layers = 1 + iter / add_layers_period
        config_file = "{0}/configs/layer{1}.config".format(dir, cur_num_hidden_layers)
        if use_raw_nnet:
            raw_model_string = "nnet3-copy --learning-rate={lr} {dir}/{iter}.raw - | nnet3-init --srand={srand} - {config} - |".format(lr=learning_rate, dir=dir, iter=iter, srand=iter + srand, config=config_file )
        else:
            raw_model_string = "nnet3-am-copy --raw=true --learning-rate={lr} {dir}/{iter}.mdl - | nnet3-init --srand={srand} - {config} - |".format(lr=learning_rate, dir=dir, iter=iter, srand=iter + srand, config=config_file )
    else:
        do_average = True
        if iter == 0:
            do_average = False   # on iteration 0, pick the best, don't average.
        else:
            cache_read_opt = "--read-cache={dir}/cache.{iter}".format(dir=dir, iter=iter)
        if use_raw_nnet:
            raw_model_string = "nnet3-copy --learning-rate={lr} {dir}/{iter}.raw - |".format(lr = learning_rate, dir = dir, iter = iter)
        else:
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

    TrainNewModels(dir, iter, srand, num_jobs, num_archives_processed, num_archives,
                   raw_model_string, egs_dir, frames_per_eg,
                   left_context, right_context,
                   momentum, max_param_change,
                   shuffle_buffer_size, cur_minibatch_size,
                   cache_read_opt, run_opts)
    [models_to_average, best_model] = GetSuccessfulModels(num_jobs, '{0}/log/train.{1}.%.log'.format(dir,iter))
    nnets_list = []
    for n in models_to_average:
        nnets_list.append("{0}/{1}.{2}.raw".format(dir, iter + 1, n))

    if do_average:
        # average the output of the different jobs.
        GetAverageNnetModel(dir = dir, iter = iter,
                            nnets_list = " ".join(nnets_list),
                            run_opts = run_opts,
                            use_raw_nnet = use_raw_nnet)
    else:
        # choose the best model from different jobs
        GetBestNnetModel(dir = dir, iter = iter,
                         best_model_index = best_model,
                         run_opts = run_opts,
                         use_raw_nnet = use_raw_nnet)

    try:
        for i in range(1, num_jobs + 1):
            os.remove("{0}/{1}.{2}.raw".format(dir, iter + 1, i))
    except OSError:
        raise Exception("Error while trying to delete the raw models")

    if use_raw_nnet:
        new_model = "{0}/{1}.raw".format(dir, iter + 1)
    else:
        new_model = "{0}/{1}.mdl".format(dir, iter + 1)

    if not os.path.isfile(new_model):
        raise Exception("Could not find {0}, at the end of iteration {1}".format(new_model, iter))
    elif os.stat(new_model).st_size == 0:
        raise Exception("{0} has size 0. Something went wrong in iteration {1}".format(new_model, iter))
    if cache_read_opt and os.path.exists("{0}/cache.{1}".format(dir, iter)):
        os.remove("{0}/cache.{1}".format(dir, iter))


