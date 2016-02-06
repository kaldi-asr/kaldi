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

num_epochs_per_iter=1         # Number of epochs per each iteration of training.
                              # In this script, each iteration of training consists 
                              # of one GPU job (i.e. one call to nnet3-train) 
                              # with $num_epochs_per_iter number of epochs
                              # over all training examples.
                              # this is to make sure that each GPU job lasts for 
                              # a reasonable amount of time (i.e., ~5 minutes)

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

echo "$0 $@"  # Print the command line for logging

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

durmodel=$dir/durmodel.mdl
mkdir -p $dir/log

if [ $stage -le -2 ]; then
  echo "$0: Initializing the duration model..."

  transmodel=$alidir/final.mdl
  for f in $transmodel $phones_dir/roots.int $phones_dir/extra_questions.int; do
    [ ! -f $f ] && echo "$0: Required file for initializing not found: $f" && exit 1;
  done

  if [ $max_duration == 0 ]; then
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
  echo "$0: Max duration: $max_duration"

  $cmd $dir/log/durmod_init.log \
       durmodel-init --max-duration=$max_duration \
                     --left-context=$left_context \
                     --right-context=$right_context \
                     $phones_dir/roots.int $phones_dir/extra_questions.int \
                     $durmodel || exit 1;

fi

if [ $stage -le -1 ]; then
  echo "$0: Preparing examples..."
  steps/dur_model/get_egs.sh --cmd "$cmd" --shuffle-buffer-size $shuffle_buffer_size $egs_opts $alidir $dir || exit 1;
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

if [ $stage -le 0 ]; then  # do the nnet-related initializations only if stage<=0
  echo "$0: Initializing nnet3 model..."
  stage=0
  if [[ ! -z $nnet_config ]]; then
    echo "$0: Using provided config file for nnet: $nnet_config"
  else
    nnet_config=$dir/nnet.conf
    feat_dim=$(durmodel-info $durmodel 2>/dev/null | grep feature-dim | awk '{ print $2 }') || exit 1;
    output_dim=$(durmodel-info $durmodel 2>/dev/null | grep max-duration | awk '{ print $2 }') || exit 1;
    steps/dur_model/make_nnet_config.sh $nnet_opts $feat_dim $output_dim >$nnet_config
    echo "$0: Wrote nnet config to "$nnet_config
  fi
  $cmd $dir/log/nnet_init.log \
       nnet3-init $nnet_config $dir/nnet.raw || exit 1;
  $cmd $dir/log/nnet_durmodel_init.log \
       nnet3-durmodel-init $durmodel $dir/nnet.raw $dir/0.mdl || exit 1;
else
  [ ! -f $dir/$stage.mdl ] && echo "$0: Nnet-duration model file not found (you provided --stage=$stage): $dir/$stage.mdl" && exit 1;
fi

num_examples=`cat $dir/num_examples` || num_examples=1
num_epochs_per_iter=$[$num_examples_per_job/$num_examples]
[[ $num_epochs_per_iter -le 0 ]] && num_epochs_per_iter=1
[[ $num_epochs_per_iter -gt 8 ]] && num_epochs_per_iter=8
num_iterations=$[$num_epochs/$num_epochs_per_iter]
[[ $[$num_epochs%$num_epochs_per_iter] != 0 ]] && num_iterations=$[$num_iterations+1]

echo "$0: Number of epochs per each iteration is $num_epochs_per_iter"
echo "$0: Will train from iteration $stage through iteration $[$num_iterations-1] ..."

for iter in $(seq $stage $[$num_iterations-1]); do
  echo "Iteration: "$iter
  curr_mdl=$dir/$iter.mdl
  next_mdl=$dir/$[$iter+1].mdl

  train_egs="ark:for n in $(seq -s ' ' $num_epochs_per_iter); do cat $dir/train.egs; done |"
  $cmd $train_queue_opt $dir/log/train.$[$iter+1].log \
       nnet3-train $parallel_train_opts "nnet3-durmodel-copy --raw=true $curr_mdl -|" \
                   "ark:nnet3-durmodel-copy-egs --srand=$iter --noise-magnitude=$noise_magnitude '$train_egs' ark:- | \
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
    curr_val_logprob=$(grep -o "Overall log.*" $dir/log/train.$iter.log 2>/dev/null | awk '{print $6}')
    next_val_logprob=$(grep -o "Overall log.*" $dir/log/train.$[$iter+1].log | awk '{print $6}')
    num_epochs_until_now=$[$iter*$num_epochs_per_iter]
    if [[ $(echo "$next_val_logprob < $curr_val_logprob" | bc 2>/dev/null) == 1 && $num_epochs_until_now -ge $early_stop_min_epochs ]]; then
      echo "$0: Early stopping...Best model is $curr_mdl"
      rm $next_mdl
      break;
    fi
  fi

  $cmd $dir/log/progress.$[$iter+1].log \
        nnet3-show-progress --use-gpu=no "nnet3-durmodel-copy --raw=true $curr_mdl - |" "nnet3-durmodel-copy --raw=true $next_mdl - |" \
        "ark:nnet3-merge-egs --minibatch-size=$minibatch_size ark:$dir/val.egs ark:- |" '&&' \
        nnet3-info "nnet3-durmodel-copy --raw=true $next_mdl - |" &
done # training loop

if [ -f $next_mdl ]; then
  ln -s -f $(basename $next_mdl) $dir/final_nnet_dur_model.mdl
else
  ln -s -f $(basename $curr_mdl) $dir/final_nnet_dur_model.mdl
fi
wait
echo "$0: Done"

