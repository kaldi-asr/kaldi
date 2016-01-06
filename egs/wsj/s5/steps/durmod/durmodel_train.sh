#!/bin/bash

# Copyright 2015 Hossein Hadian

num_epochs=100
minibatch_size=512
train_history_size=10
compute_prob_interval=5
nnet_config=
cmd=run.pl

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
   echo "Usage: $0 [options] <duration-model-dir>"
   echo "e.g.: $0 exp/mono/durmod"
   echo ""
   echo "Main options (for others, see top of script file):"
   echo "  --num-epochs <number>                       # max number of epochs for training"
   echo "  --minibatch-size <size>                     # minibatch size"
   echo "  --compute-prob-interval <int>               # interval for measuring accuracy"
   echo "  --nnet-config <nnet3-conf-file>             # use this config for training"     
   exit 1;
fi

dir=$1
durmodel=$dir/durmodel.mdl
mkdir -p $dir/log
mkdir -p $dir/tmp


if [[ ! -z $nnet_config ]]; then
  echo "Using provided config file for nnet."
  nnet3-init $nnet_config $dir/0.raw || exit 1;
else
  durmod-copy --raw=true $durmodel $dir/0.raw || exit 1;
fi

for epoch in $(seq 0 $[$num_epochs-1]); do
  
  echo "Epoch: "$epoch
  curr_nnet=$dir/$[$epoch%($train_history_size+1)].raw
  next_nnet=$dir/$[($epoch+1)%($train_history_size+1)].raw
  
  $cmd $dir/log/train_$epoch.log \
       nnet3-train $curr_nnet \
       "ark:nnet3-shuffle-egs --srand=$epoch ark:$dir/all.egs ark:-| \
            nnet3-merge-egs --minibatch-size=$minibatch_size ark:- ark:- |" \
            $next_nnet || exit 1;
  
  cat $dir/log/train_$epoch.log | grep Overall
  durmod-copy --set-raw-nnet=$next_nnet $durmodel $dir/tmp/next_durmod.mdl
  mv -f $dir/tmp/next_durmod.mdl $durmodel

  if [[ $[$epoch%$compute_prob_interval] == 0 ]] && [[ $epoch != 0 ]]; then
    $cmd $dir/log/compute_prob_$epoch.log \
       nnet3-compute-prob $next_nnet \
       "ark:nnet3-merge-egs --minibatch-size=$minibatch_size ark:$dir/all.egs ark:- |"
    cat $dir/log/compute_prob_$epoch.log | grep accuracy
  fi
  
done
