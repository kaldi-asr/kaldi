import subprocess
import logging
import math
import re
import time

train_lib = imp.load_source('', 'steps/nnet3/nnet3_train_lib.py')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def GenerateChainEgs(dir, data, lat_dir, egs_dir,
                    left_context, right_context,
                    run_opts, stage = 0,
                    valid_left_context = None, valid_right_context = None,
                    left_tolerance = None, right_tolerance = None,
                    frame_subsampling_factor = 3,
                    feat_type = 'raw', online_ivector_dir = None,
                    samples_per_iter = 20000, frames_per_eg = 20,
                    egs_opts = None, cmvn_opts = None, transform_dir = None):
        
    RunKaldiCommand("""
steps/nnet3/chain/get_egs.sh {egs_opts} \
  --cmd "{command}" \
  --cmvn-opts "{cmvn_opts}" \
  --feat-type {feat_type} \
  --transform-dir "{transform_dir}" \
  --online-ivector-dir "{ivector_dir}" \
  --left-context {left_context} --right-context {right_context} \
  --valid-left-context '{valid_left_context}' \
  --valid-right-context '{valid_right_context}' \
  --left-tolerance '{left_tolerance}' \
  --right-tolerance '{right_tolerance}' \
  --frame-subsampling-factor {frame_subsampling_factor} \
  --stage {stage} \
  --samples-per-iter {samples_per_iter} \
  --frames-per-eg {frames_per_eg} \
  {data} {dir} {lat_dir} {egs_dir}
      """.format(command = run_opts.command,
          cmvn_opts = cmvn_opts if cmvn_opts is not None else '',
          feat_type = feat_type,
          transform_dir = transform_dir if transform_dir is not None else '',
          ivector_dir = online_ivector_dir if online_ivector_dir is not None else '',
          left_context = left_context, right_context = right_context,
          valid_left_context = valid_left_context if valid_left_context is not None else '',
          valid_right_context = valid_right_context if valid_right_context is not None else '',
          left_tolerance = left_tolerance if left_tolerance is not None else '',
          right_tolerance = right_tolerance if right_tolerance is not None else '',
          frame_subsampling_factor = frame_subsampling_factor,
          stage = stage, samples_per_iter = samples_per_iter,
          frames_per_eg = frames_per_eg,
          data = data, lat_dir = lat_dir, dir = dir, egs_dir = egs_dir,
          egs_opts = egs_opts if egs_opts is not None else '' ))


# this function is exactly similar to the version in nnet3_train_lib.py
# except it uses egs files in place of cegs files
def ComputePreconditioningMatrix(dir, egs_dir, num_lda_jobs, run_opts,
                                 max_lda_jobs = None, rand_prune = 4.0,
                                 lda_opts = None):
    if max_lda_jobs is not None:
        if num_lda_jobs > max_lda_jobs:
            num_lda_jobs = max_lda_jobs


  # Write stats with the same format as stats for LDA.
    RunKaldiCommand("""
{command} JOB=1:{num_lda_jobs} {dir}/log/get_lda_stats.JOB.log \
 nnet3-chain-acc-lda-stats --rand-prune={rand_prune} \
    {dir}/init.raw "ark:{egs_dir}/cegs.JOB.ark" {dir}/JOB.lda_stats""".format(
        command = run_opts.command,
        num_lda_jobs = num_lda_jobs,
        dir = dir,
        egs_dir = egs_dir,
        rand_prune = rand_prune))

    # the above command would have generated dir/{1..num_lda_jobs}.lda_stats
    lda_stat_files = map(lambda x: '{0}/{1}.lda_stats'.format(dir, x),
                         range(1, num_lda_jobs + 1))
   
    RunKaldiCommand("""
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

    RunKaldiCommand("""
{command} {dir}/log/get_transform.log \
 nnet-get-feature-transform {lda_opts} {dir}/lda.mat {dir}/lda_stats
     """.format(command = run_opts.command,dir = dir,
                lda_opts = lda_opts if lda_opts is not None else ""))
            
    train_lib.ForceSymlink("../lda.mat", "{0}/configs/lda.mat".format(dir))


def PrepareInitialAcousticModel(dir, run_opts):
    """ Adds the first layer; this will also add in the lda.mat and
        presoftmax_prior_scale.vec. It will also prepare the acoustic model
        with the transition model."""

    RunKaldiCommand("""
{command} {dir}/log/add_first_layer.log \
   nnet3-init --srand=-1 {dir}/init.raw {dir}/configs/layer1.config {dir}/0.raw     """.format(command = run_opts.command,
               dir = dir))
    
    # The model-format for a 'chain' acoustic model is just the transition
    # model and then the raw nnet, so we can use 'cat' to create this, as
    # long as they have the same mode (binary or not binary).
    # We ensure that they have the same mode (even if someone changed the
    # script to make one or both of them text mode) by copying them both
    # before concatenating them.
    RunKaldiCommand("""
{command} {dir}/log/init_mdl.log \
    nnet3-am-init {dir}/0.trans_mdl {dir}/0.raw {dir}/0.mdl""".format(
                   command = run_opts.command, dir = dir))

def CombineModels(dir, num_iters, num_iters_combine, num_chunk_per_minibatch,
                  egs_dir, leaky_hmm_coefficient, l2_regularize, run_opts):
    # Now do combination.  In the nnet3 setup, the logic
    # for doing averaging of subsets of the models in the case where
    # there are too many models to reliably esetimate interpolation
    # factors (max_models_combine) is moved into the nnet3-combine
    raw_model_strings = []
    print num_iters_combine
    for iter in range(num_iters - num_iters_combine + 1, num_iters + 1):
      model_file = '{0}/{1}.mdl'.format(dir, iter)
      if not os.path.exists(model_file):
          raise Exception('Model file {0} missing'.format(model_file))
      raw_model_strings.append('"nnet3-am-copy --raw=true {0} -|"'.format(model_file))
    RunKaldiCommand("""
{command} {combine_queue_opt} {dir}/log/combine.log \
nnet3-chain-combine --num-iters=40 \
   --l2-regularize={l2} --leaky-hmm-coefficient={leaky} \
   --enforce-sum-to-one=true --enforce-positive-weights=true \
   --verbose=3 {dir}/den.fst {raw_models} "ark:nnet3-chain-merge-egs --minibatch-size={num_chunk_per_minibatch} ark:{egs_dir}/combine.cegs ark:-|" \
"|nnet3-am-copy --set-raw-nnet=- {dir}/{num_iters}.mdl {dir}/final.mdl" 
    """.format(command = run_opts.command,
               combine_queue_opt = run_opts.combine_queue_opt,
               l2 = l2_regularize, leaky = leaky_hmm_coefficient,
               dir = dir, raw_models = " ".join(raw_model_strings),
               num_chunk_per_minibatch = num_chunk_per_minibatch,
               num_iters = num_iters,
               egs_dir = egs_dir))

  # Compute the probability of the final, combined model with
  # the same subset we used for the previous compute_probs, as the
  # different subsets will lead to different probs.
    ComputeTrainCvProbabilities(dir, 'combined', egs_dir, run_opts, wait = False)



def ComputeTrainCvProbabilities(dir, iter, egs_dir, l2_regularize, xent_regularize,
                                leaky_hmm_coefficient, run_opts, wait = False):

    model = '{0}/{1}.mdl'.format(dir, iter)

    RunKaldiCommand("""
{command} {dir}/log/compute_prob_valid.{iter}.log \
  nnet3-chain-compute-prob --l2-regularize={l2} --leaky-hmm-coefficient={leaky} \
  --xent-regularize {xent_reg} \
  "nnet3-am-copy --raw=true {model} - |" {dir}/den.fst \
        "ark:nnet3-chain-merge-egs ark:{egs_dir}/valid_diagnostic.cegs ark:- |"
    """.format(command = run_opts.command,
               dir = dir, iter = iter, model = model,
               l2 = l2_regularize, leaky = leaky_hmm_coefficient,
               xent_reg = xent_regularize,
               egs_dir = egs_dir), wait = wait)

    RunKaldiCommand("""
{command} {dir}/log/compute_prob_train.{iter}.log \
  nnet3-chain-compute-prob --l2-regularize={l2} --leaky-hmm-coefficient={leaky} \
  --xent-regularize {xent_reg} \
  "nnet3-am-copy --raw=true {model} - |" {dir}/den.fst \
        "ark:nnet3-chain-merge-egs ark:{egs_dir}/train_diagnostic.cegs ark:- |"
    """.format(command = run_opts.command,
               dir = dir,
               iter = iter,
               model = model,
               egs_dir = egs_dir), wait = wait)

def ComputeProgress(dir, iter, run_opts, wait=False):

    prev_model = '{0}/{1}.mdl'.format(dir, iter - 1)
    model = '{0}/{1}.mdl'.format(dir, iter)
    RunKaldiCommand("""
{command} {dir}/log/progress.{iter}.log \
nnet3-am-info {model} '&&' \
nnet3-show-progress --use-gpu=no "nnet3-am-copy --raw=true {prev_model} - |" "nnet3-am-copy --raw=true {model} - |"
    """.format(command = run_opts.command,
               dir = dir,
               iter = iter,
               model = model,
               prev_model = prev_model), wait = wait)
