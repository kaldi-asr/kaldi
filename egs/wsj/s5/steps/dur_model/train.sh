#!/bin/bash

# Copyright 2015 Hossein Hadian
# Apache 2.0.
#
# This script initializes an nnet-duration model and trains it. There are two 
# new objects in this setup: [1] duration-model and [2] nnet-duration-model.
# The duration-model contains all the information and parameters required for
# phone duration modeling. This model is based on phone durations and the
# features for each phone are extracted from the left and right context of
# that phone. The features consist of phone identities (which are worked
# out according to roots.int) and binary features (worked out according
# to extra_questions.int) for each phone in the context, as well as the
# durations of the phones in the left context. The durations are represented
# in number of frames. Stage -2 initializes a duration-model.
# The nnet-duration-model (which is used for training) is a simple object which
# stores a duration-model object and a raw nnet3 object. It is initialized in
# stage 0.
# The training is performed using standard nnet3 programs, and training
# examples are created (by get_egs.sh) from the alignments in an alignement dir 
# (for eg., exp/tri3_ali/ali.*.gz). The examples are prepared at stage -1.


# Begin configuration section.
num_epochs=25                 # The total number of epochs of training.
                              # the number of iterations and the number of 
                              # epochs per iteration is worked out from this
                              # and from the number of the training examples
                              # such that each job lasts for ~5 minutes.

minibatch_size=512            # Size of the minibatch. This default is suitable for GPU-based training.

num_epochs_per_iter=0         # Number of epochs per each iteration of training.
                              # In this script, each iteration of training consists 
                              # of one GPU job (i.e. one call to nnet3-train) 
                              # with $num_epochs_per_iter number of epochs
                              # over all training examples.
                              # this is to make sure that each GPU job lasts for 
                              # a reasonable amount of time (i.e., ~5 minutes)
                              # When set to 0, it will be determined
                              # automatically according to num_examples_per_job

nnet_config=                  # config file for the raw nnet. If not set, it will be created automatically.
early_stop=true               # If true, early stopping will happen if the 
                              # validation-log-prob reaches a peak and a minimum
                              # number of epochs (i.e. $early_stop_min_epochs) has been completed.  
early_stop_min_epochs=5       # Minimum of number of epochs for early stopping
cmd=run.pl
use_gpu=true                  # If true, we train on GPU.
stage=-2
shuffle_buffer_size=5000      # The buffer size for shuffling nnet examples 
                              # please refer to steps/nnet3/train_tdnn.sh for more info on this
egs_opts=                     # The options to be passed to get_egs.sh
nnet_opts=                    # The options to be passed to make_nnet_config.sh

max_duration=50               # Max duration for the phone duration model. All
                              # durations higher than this value, will be mapped to this
                              # value for training. If it is set to 0, it will be 
                              # determined automatically according to the training alignments
left_context=4                # Left context size for the phone duration model
right_context=2               # Right context size for the phone duration model
min_count=5                   # If max_duration is 0, it will be worked out
                              # from $min_count. It is the max duration d such that
                              # all durations d-1, d-2, ... occur at least 
                              # $min_count in the training alignments
num_examples_per_job=40000000 # Number of examples that are given to nnet3-train
                              # in each GPU job (i.e. number of examples in each
                              # iteration; please refer to num_epochs_per_iter)
noise_magnitude=0.05          # The relative magnitude of noise to be added
                              # to duration values during training for better
                              # generalization: duration
estimate_logprob_num_examples=5000000       # Number of training examples to use for estimating
                                            # average logprobs to later subtract at test time
scorenorm_leftcontext=1
scorenorm_rightcontext=0
extra_questions_file=         # Use this option to set an arbitrary extra_questions.int file
                              # other than the default at lang/phones
roots_file=                   # Use this option to set an arbitrary roots_file.int
                              # other than the default at lang/phones
lognormal_objective=false     # true means that the objective function for the 
                              # nnet will be lognormal instead of cross-entropy
patience=3                    # If early stopping is activated, a non-improved 
                              # validation objective will be tolerated only
                              # $patience times 
always_perturb=false          # If noise_magnitude is not 0, always_perturb=true
                              # means that even if the noise (i.e. noise_mag * duration)
                              # is zero (eg., for durations < 1/noise_mag) a
                              # minimum noise of +-1 is added to duration values.
priors_k=1000

echo "$0 $@"  # Print the command line for logging
cmdline="$0 $@"

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [[ $# != 3 ]]; then
   echo "Usage: $0 [options] <phones-dir> <ali-dir> <duration-model-dir>"
   echo "e.g.: $0 data/lang/phones exp/mono_ali exp/mono/durmod"
   echo ""
   echo "Main options (for others, see top of script file):"
   echo "  --num-epochs <number>                    # max number of epochs for training"
   echo "  --minibatch-size <size>                  # minibatch size"
   echo "  --compute-prob-interval <int>            # interval for measuring accuracy (diagnostics)"
   echo "  --nnet-config <nnet3-conf-file>          # use this config for training"
   echo "Options related to initializing (stage 0):"
   echo "  --max-duration <duration-in-frames>      # max duration; if not set, it will be determined automatically"
   echo "  --left-context <size>                    # left phone context size"
   echo "  --right-context <size>                   # right phone context size"
   exit 1;
fi

phones_dir=$1
alidir=$2
dir=$3
transmodel=$alidir/final.mdl

durmodel=$dir/durmodel.mdl
mkdir -p $dir/log

echo $cmdline > $dir/log/cmd.log

if [ $stage -le -2 ]; then
  echo "$0: Initializing the duration model and nnet3..."

  [ -z $extra_questions_file ] && extra_questions_file=$phones_dir/extra_questions.int
  echo "$0: Questions-file used for binary features: $extra_questions_file"
  [ -z $roots_file ] && roots_file=$phones_dir/roots.int
  echo "$0: Roots-file used for phone-identity features: $roots_file"


  for f in $transmodel $roots_file $extra_questions_file; do
    [ ! -f $f ] && echo "$0: Required file for initializing not found: $f" && exit 1;
  done

  if [[ $max_duration == 0  && $lognormal_objective == false ]]; then
    echo "$0: Determining max-duration..."

    max_duration=`ali-to-phones --write-lengths $transmodel "ark:gunzip -c $alidir/ali.*.gz|" \
                  ark,t:- | awk -v min_count=$min_count -F';' \
                  '{ for(i=1; i<=NF; i++){ \
                       n=split($(i), a, " ");\
                       duration=a[n];\
                       counts[duration]++;\
                       if(duration>max_duration) max_duration=duration; \
                   }} END { for(d=3; d<=max_duration; d++) \
                              if(counts[d]<min_count){\
                                print d-1; exit 0;\
                              }\
                            print max_duration;\
                            exit 0;}'` || exit 1;

  fi
  if $lognormal_objective; then
    echo "$0: Using lognormal objective function for nnet output..."
    max_duration=0  # providing max_duration=0 to nnet3-durmodel-init will enable lognormal objective
  else
    echo "$0: Max duration: $max_duration"
  fi

  $cmd $dir/log/durmod_init.log \
       durmodel-init --left-context=$left_context \
                     --right-context=$right_context \
                     $roots_file $extra_questions_file \
                     $durmodel || exit 1;

  if [[ ! -z $nnet_config ]]; then
    echo "$0: Using provided config file for nnet: $nnet_config"
  else
    nnet_config=$dir/nnet.conf
    feat_dim=$(durmodel-info $durmodel 2>/dev/null | grep feature-dim | awk '{ print $2 }') || exit 1;
    if [[ $max_duration == 0 ]]; then
      output_dim=2
    else
      output_dim=$max_duration
    fi
    steps/dur_model/make_nnet_config.sh --lognormal-objective $lognormal_objective \
                                        $nnet_opts $feat_dim $output_dim >$nnet_config
    echo "$0: Wrote nnet config to "$nnet_config
  fi
  $cmd $dir/log/nnet_init.log \
       nnet3-init $nnet_config $dir/nnet.raw || exit 1;
  $cmd $dir/log/nnet_durmodel_init.log \
       nnet3-durmodel-init --max-duration=$max_duration $durmodel $dir/nnet.raw $dir/0.mdl || exit 1;
fi

if [ $stage -le -1 ]; then
  echo "$0: Preparing examples..."
  steps/dur_model/get_egs.sh --cmd "$cmd" --shuffle-buffer-size $shuffle_buffer_size $egs_opts $alidir $dir || exit 1;
fi


if [[ $stage == 0 ]]; then  # the user wants to just (re)start training...so update the nnet config again
  echo "$0: UPDATING NNET ..."
  if [[ ! -z $nnet_config ]]; then
    echo "$0: Updating nnet config before training using provided config file: $nnet_config"
  else
    nnet_config=$dir/nnet.conf
    feat_dim=$(durmodel-info $durmodel 2>/dev/null | grep feature-dim | awk '{ print $2 }') || exit 1;
    if $lognormal_objective; then
      output_dim=2
    else
      output_dim=$max_duration ## should be taken from 0.mdl
    fi
    steps/dur_model/make_nnet_config.sh --lognormal-objective $lognormal_objective \
                                        $nnet_opts $feat_dim $output_dim >$nnet_config
    echo "$0: Rewrote nnet config to "$nnet_config
  fi
  $cmd $dir/log/nnet_init.log \
       nnet3-init $nnet_config $dir/nnet.raw || exit 1;
  $cmd $dir/log/nnet_durmodel_set_rawnnet.log \
       nnet3-durmodel-copy --set-raw-nnet=$dir/nnet.raw $dir/0.mdl $dir/0.mdl || exit 1;
fi

[ ! -f $durmodel ] && echo "$0: Duration model file not found (have you completed stage -2?): $durmodel" && exit 1;
[ ! -f $dir/train.egs ] && echo "$0: Train examples file not found (have you completed stage -1?): $dir/train.egs" && exit 1;
[ ! -f $dir/val.egs ] && echo "$0: Validation examples file not found (have you completed stage -1?): $dir/val.egs" && exit 1;

if $use_gpu; then
  train_queue_opt="--gpu 1"
  parallel_train_opts="--use-gpu=yes"
  if ! cuda-compiled; then
    echo "$0: WARNING: you are running with one thread but you have not compiled"
    echo "   for CUDA.  You may be running a setup optimized for GPUs.  If you have"
    echo "   GPUs and have nvcc installed, go to src/ and do ./configure; make"
    exit 1
  fi
else
  parallel_train_opts="--use-gpu=no"
fi

if [[ $num_epochs_per_iter == 0 ]]; then
  num_examples=`cat $dir/num_examples` || num_examples=1
  num_epochs_per_iter=$[$num_examples_per_job/$num_examples]
  [[ $num_epochs_per_iter -le 0 ]] && num_epochs_per_iter=1
  [[ $num_epochs_per_iter -gt 8 ]] && num_epochs_per_iter=8
fi

num_iterations=$[$num_epochs/$num_epochs_per_iter]
[[ $[$num_epochs%$num_epochs_per_iter] != 0 ]] && num_iterations=$[$num_iterations+1]

[ $stage -lt 0 ] && stage=0
if [[ $stage -le $[$num_iterations-1] ]]; then
  echo "$0: Number of epochs per each iteration is $num_epochs_per_iter"
  echo "$0: Will train from iteration $stage through iteration $[$num_iterations-1] ..."
  [ ! -f $dir/$stage.mdl ] && echo "$0: Nnet-duration model file not found (you provided --stage=$stage): $dir/$stage.mdl" && exit 1;
fi

best_val_obj="-1000.0"
best_mdl=
base_patience=$patience

# the following take care of early stopping paramteres for resuming (i.e. when $stage > 0)
for iter in $(seq 0 $[$stage-1]); do
  next_mdl=$dir/$[$iter+1].mdl
  if $lognormal_objective; then
    likelihood_term="Overall obj.*"
  else
    likelihood_term="Overall log.*"
  fi
  val_obj=$(grep -o "$likelihood_term" $dir/log/train.$[$iter+1].log | awk '{print $6}')
  if [[ $(echo "$val_obj > $best_val_obj" | bc 2>/dev/null) == 1 ]]; then
    best_val_obj=$val_obj
    best_mdl=$next_mdl
    patience=$base_patience  # we got an improvement, reset the patience value
  else
    patience=$[$patience-1]
    echo "$0: Patience: $patience"
  fi
done

for iter in $(seq $stage $[$num_iterations-1]); do
  echo "Iteration: "$iter
  curr_mdl=$dir/$iter.mdl
  next_mdl=$dir/$[$iter+1].mdl

  train_egs="ark:for n in $(seq -s ' ' $num_epochs_per_iter); do cat $dir/train.egs; done |"
  $cmd $train_queue_opt $dir/log/train.$[$iter+1].log \
       nnet3-train $parallel_train_opts "nnet3-durmodel-copy --raw=true $curr_mdl -|" \
                   "ark:nnet3-durmodel-copy-egs --srand=$iter --always-perturb=$always_perturb --noise-magnitude=$noise_magnitude '$train_egs' ark:- | \
                   nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$iter ark:- ark:-| \
                   nnet3-merge-egs --minibatch-size=$minibatch_size ark:- ark:- |" \
                   $dir/nnet.raw \
       '&&' \
       nnet3-compute-prob $parallel_train_opts $dir/nnet.raw \
                          "ark:nnet3-merge-egs --minibatch-size=$minibatch_size ark:$dir/val.egs ark:- |" \
       && grep -o -H "Overall.*" $dir/log/train.$[$iter+1].log \
       || exit 1;

  $cmd $dir/log/durmod_set_raw_nnet.log \
       nnet3-durmodel-copy --set-raw-nnet=$dir/nnet.raw $curr_mdl $next_mdl

  if $early_stop; then
    if $lognormal_objective; then
      likelihood_term="Overall obj.*"
    else
      likelihood_term="Overall log.*"
    fi
    val_obj=$(grep -o "$likelihood_term" $dir/log/train.$[$iter+1].log | awk '{print $6}')
    if [[ $(echo "$val_obj > $best_val_obj" | bc 2>/dev/null) == 1 ]]; then
      best_val_obj=$val_obj
      best_mdl=$next_mdl
      patience=$base_patience  # we got an improvement, reset the patience value
    else
      if [[ $patience -le 0 ]]; then
        echo "$0: Early stopping...Best model is $best_mdl with objective $best_val_obj"
        break;
      fi
      patience=$[$patience-1]
      echo "$0: Patience: $patience"
    fi
  fi

  $cmd $dir/log/progress.$[$iter+1].log \
        nnet3-show-progress --use-gpu=no "nnet3-durmodel-copy --raw=true $curr_mdl - |" "nnet3-durmodel-copy --raw=true $next_mdl - |" \
        "ark:nnet3-merge-egs --minibatch-size=$minibatch_size ark:$dir/val.egs ark:- |" '&&' \
        nnet3-info "nnet3-durmodel-copy --raw=true $next_mdl - |" &
done # training loop

if [ ! -z $best_mdl ]; then
  ln -s -f $(basename $best_mdl) $dir/final_nnet_dur_model.mdl
  grep -o -H "Overall.*" $dir/log/train.$(basename $best_mdl '.mdl').log >$dir/log/final_objectives.log
elif [ ! -z $next_mdl ]; then
  ln -s -f $(basename $next_mdl) $dir/final_nnet_dur_model.mdl
  grep -o -H "Overall.*" $dir/log/train.$(basename $next_mdl '.mdl').log >$dir/log/final_objectives.log
fi

echo "$0: Estimating average logprobs over training data..."
### tmp ##
#$cmd $dir/log/estimate-avg-logprobs.log \
#     nnet3-durmodel-estimate-avg-logprobs --num-examples=$estimate_logprob_num_examples --binary=false \
#     --left-context=$scorenorm_leftcontext --right-context=$scorenorm_rightcontext \
#     $dir/final_nnet_dur_model.mdl $alidir/final.mdl \
#     "ark:gunzip -c $alidir/ali.*.gz |" $dir/avg_logprobs.data
## tmp ###

ali-to-phones --write-lengths $transmodel "ark:gunzip -c $alidir/ali.*.gz|" \
              ark,t:- | awk -v K=$priors_k -F';' \
              '{ for(i=1; i<=NF; i++){ \
                   n=split($(i), a, " ");\
                   duration=a[n];\
                   counts[duration]++;\
                   total_count++;\
                   if(duration>max_duration) max_duration=duration; \
               }} END { N = length(counts); \
                        printf "[ "; \
                        for(d=1; d<=max_duration; d++){ \
                          counts[d] = (counts[d] + K) / (total_count + N*K); \
                          printf counts[d]" "; \
                        } \
                        printf "]";}' >$dir/priors.vec || exit 1;

echo "$0: Done"

