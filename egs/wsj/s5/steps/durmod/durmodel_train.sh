#!/bin/bash

# Copyright 2015 Hossein Hadian

num_epochs=50
minibatch_size=512
train_history_size=5
compute_prob_interval=5
nnet_config=
cmd=run.pl
use_gpu=true    # if true, we run on GPU.
stage=0

egs_opts=
max_duration=0
left_context=4
right_context=2

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [[ $# != 1 ]] && [[ $# != 3 ]]; then
   echo "Usage: $0 [options] <duration-model-dir> ( <phones-dir> <ali-dir> )"
   echo "e.g.: $0 --stage 1 exp/mono/durmod"
   echo "e.g.: $0 --stage 0 exp/mono/durmod data/lang/phones exp/mono_ali"
   echo ""
   echo "Main options (for others, see top of script file):"
   echo "  --num-epochs <number>                       # max number of epochs for training"
   echo "  --minibatch-size <size>                     # minibatch size"
   echo "  --compute-prob-interval <int>               # interval for measuring accuracy (diagnostics)"
   echo "  --nnet-config <nnet3-conf-file>             # use this config for training"
   echo "Options related to initializing (stage 0):"
   echo "  --max-duration <duration-in-frames>         # max duration; if not set, it will be determined automatically"
   echo "  --left-context <size>                       # left phone context size"
   echo "  --right-context <size>                      # right phone context size"
   exit 1;
fi

dir=$1
if [[ $# == 3 ]]; then
  phones_dir=$2
  alidir=$3
else
    [ ! $stage -gt 0 ] && echo "$0: You can not run stage 0 unless you provide 3 arguments." && exit 1;
fi
durmodel=$dir/durmodel.mdl
mkdir -p $dir/log


if [ $stage -le 0 ]; then
  echo "$0: Initializing the duration model and preparing examples..."
  steps/durmod/durmodel_prepare_examples.sh --left-context $left_context \
                                            --right-context $right_context \
                                            --max-duration $max_duration \
                                            --cmd $cmd \
                                            $egs_opts \
                                            $phones_dir $alidir $dir || exit 1;
fi

if [ $stage -le 1 ]; then
  
  [ ! -f $durmodel ] && echo "$0: Duration model file not found (have you completed stage 0?): $durmodel" && exit 1;
  [ ! -f $dir/all.egs ] && echo "$0: Examples file not found (have you completed stage 0?): $dir/all.egs" && exit 1;
  
  if $use_gpu; then
    train_queue_opt="--gpu 1"
    if ! cuda-compiled; then
      echo "$0: WARNING: you are running with one thread but you have not compiled"
      echo "   for CUDA.  You may be running a setup optimized for GPUs.  If you have"
      echo "   GPUs and have nvcc installed, go to src/ and do ./configure; make"
      exit 1
    fi
  else
    parallel_train_opts="--use-gpu=no"
  fi

  if [[ ! -z $nnet_config ]]; then
    echo "$0: Using provided config file for nnet."
    $cmd $dir/log/nnet_init.log \
         nnet3-init $nnet_config $dir/0.raw || exit 1;
    $cmd $dir/log/durmod_set_raw_nnet.log \
         durmod-copy --set-raw-nnet=$dir/0.raw $durmodel $durmodel || exit 1;
  fi

  $cmd $dir/log/durmod_copy_raw.log \
       durmod-copy --raw=true $durmodel $dir/0.raw || exit 1;

  for epoch in $(seq 0 $[$num_epochs-1]); do
    echo "Epoch: "$epoch
    curr_nnet=$dir/$[$epoch%($train_history_size+1)].raw
    next_nnet=$dir/$[($epoch+1)%($train_history_size+1)].raw

    $cmd $train_queue_opt $dir/log/train_$epoch.log \
         nnet3-train $parallel_train_opts $curr_nnet \
              "ark:nnet3-shuffle-egs --srand=$epoch ark:$dir/all.egs ark:-| \
              nnet3-merge-egs --minibatch-size=$minibatch_size ark:- ark:- |" \
              $next_nnet || exit 1;
    
    grep Overall $dir/log/train_$epoch.log
    $cmd $dir/log/durmod_set_raw_nnet.log \
         durmod-copy --set-raw-nnet=$next_nnet $durmodel $durmodel

    if [[ $[$epoch%$compute_prob_interval] == 0 ]]; then
      $cmd $dir/log/compute_prob_$epoch.log \
         nnet3-compute-prob $next_nnet \
         "ark:nnet3-merge-egs --minibatch-size=$minibatch_size ark:$dir/all.egs ark:- |" &
    fi
  done # training loop
fi # stage 1
