#!/bin/bash

# Copyright 2012-2015  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

# Schedules epochs and controls learning rate during the neural network training

# Begin configuration.

# training options,
learn_rate=0.008
momentum=0
l1_penalty=0
l2_penalty=0

# data processing,
train_tool="nnet-train-frmshuff"
train_tool_opts="--minibatch-size=256 --randomizer-size=32768 --randomizer-seed=777"
feature_transform=

# learn rate scheduling,
max_iters=20
min_iters=0 # keep training, disable weight rejection, start learn-rate halving as usual,
keep_lr_iters=0 # fix learning rate for N initial epochs, disable weight rejection,
dropout_schedule= # dropout-rates for N initial epochs, for example: 0.1,0.1,0.1,0.1,0.1,0.0
start_halving_impr=0.01
end_halving_impr=0.001
halving_factor=0.5

# misc,
verbose=1
frame_weights=
utt_weights=

# End configuration.

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh;

. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 6 ]; then
   echo "Usage: $0 <mlp-init> <feats-tr> <feats-cv> <labels-tr> <labels-cv> <exp-dir>"
   echo " e.g.: $0 0.nnet scp:train.scp scp:cv.scp ark:labels_tr.ark ark:labels_cv.ark exp/dnn1"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi

mlp_init=$1
feats_tr=$2
feats_cv=$3
labels_tr=$4
labels_cv=$5
dir=$6

[ ! -d $dir ] && mkdir $dir
[ ! -d $dir/log ] && mkdir $dir/log
[ ! -d $dir/nnet ] && mkdir $dir/nnet

dropout_array=($(echo ${dropout_schedule} | tr ',' ' '))

# Skip training
[ -e $dir/final.nnet ] && echo "'$dir/final.nnet' exists, skipping training" && exit 0

##############################
# start training

# choose mlp to start with,
mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*}

# optionally resume training from the best epoch, using saved learning-rate,
[ -e $dir/.mlp_best ] && mlp_best=$(cat $dir/.mlp_best)
[ -e $dir/.learn_rate ] && learn_rate=$(cat $dir/.learn_rate)

# cross-validation on original network,
log=$dir/log/iter00.initial.log; hostname>$log
$train_tool --cross-validate=true --randomize=false --verbose=$verbose $train_tool_opts \
  ${feature_transform:+ --feature-transform=$feature_transform} \
  ${frame_weights:+ "--frame-weights=$frame_weights"} \
  ${utt_weights:+ "--utt-weights=$utt_weights"} \
  "$feats_cv" "$labels_cv" $mlp_best \
  2>> $log

loss=$(cat $dir/log/iter00.initial.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
loss_type=$(cat $dir/log/iter00.initial.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $5; }')
echo "CROSSVAL PRERUN AVG.LOSS $(printf "%.4f" $loss) $loss_type"

# resume lr-halving,
halving=0
[ -e $dir/.halving ] && halving=$(cat $dir/.halving)

# training,
for iter in $(seq -w $max_iters); do
  echo -n "ITERATION $iter: "
  mlp_next=$dir/nnet/${mlp_base}_iter${iter}

  # skip iteration (epoch) if already done,
  [ -e $dir/.done_iter$iter ] && echo -n "skipping... " && ls $mlp_next* && continue

  # set dropout-rate from the schedule,
  if [ -n ${dropout_array[$((${iter#0}-1))]-''} ]; then
    dropout_rate=${dropout_array[$((${iter#0}-1))]}
    nnet-copy --dropout-rate=$dropout_rate $mlp_best ${mlp_best}.dropout_rate${dropout_rate}
    mlp_best=${mlp_best}.dropout_rate${dropout_rate}
  fi

  # training,
  log=$dir/log/iter${iter}.tr.log; hostname>$log
  $train_tool --cross-validate=false --randomize=true --verbose=$verbose $train_tool_opts \
    --learn-rate=$learn_rate --momentum=$momentum \
    --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
    ${feature_transform:+ --feature-transform=$feature_transform} \
    ${frame_weights:+ "--frame-weights=$frame_weights"} \
    ${utt_weights:+ "--utt-weights=$utt_weights"} \
    "$feats_tr" "$labels_tr" $mlp_best $mlp_next \
    2>> $log || exit 1;

  tr_loss=$(cat $dir/log/iter${iter}.tr.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  echo -n "TRAIN AVG.LOSS $(printf "%.4f" $tr_loss), (lrate$(printf "%.6g" $learn_rate)), "

  # cross-validation,
  log=$dir/log/iter${iter}.cv.log; hostname>$log
  $train_tool --cross-validate=true --randomize=false --verbose=$verbose $train_tool_opts \
    ${feature_transform:+ --feature-transform=$feature_transform} \
    ${frame_weights:+ "--frame-weights=$frame_weights"} \
    ${utt_weights:+ "--utt-weights=$utt_weights"} \
    "$feats_cv" "$labels_cv" $mlp_next \
    2>>$log || exit 1;

  loss_new=$(cat $dir/log/iter${iter}.cv.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  echo -n "CROSSVAL AVG.LOSS $(printf "%.4f" $loss_new), "

  # accept or reject?
  loss_prev=$loss
  if [ 1 == $(awk "BEGIN{print($loss_new < $loss ? 1:0);}") -o $iter -le $keep_lr_iters -o $iter -le $min_iters ]; then
    # accepting: the loss was better, or we had fixed learn-rate, or we had fixed epoch-number,
    loss=$loss_new
    mlp_best=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)
    [ $iter -le $min_iters ] && mlp_best=${mlp_best}_min-iters-$min_iters
    [ $iter -le $keep_lr_iters ] && mlp_best=${mlp_best}_keep-lr-iters-$keep_lr_iters
    mv $mlp_next $mlp_best
    echo "nnet accepted ($(basename $mlp_best))"
    echo $mlp_best > $dir/.mlp_best
  else
    # rejecting,
    mlp_reject=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)_rejected
    mv $mlp_next $mlp_reject
    echo "nnet rejected ($(basename $mlp_reject))"
  fi

  # create .done file, the iteration (epoch) is completed,
  touch $dir/.done_iter$iter

  # continue with original learn-rate,
  [ $iter -le $keep_lr_iters ] && continue

  # stopping criterion,
  rel_impr=$(awk "BEGIN{print(($loss_prev-$loss)/$loss_prev);}")
  if [ 1 == $halving -a 1 == $(awk "BEGIN{print($rel_impr < $end_halving_impr ? 1:0);}") ]; then
    if [ $iter -le $min_iters ]; then
      echo we were supposed to finish, but we continue as min_iters : $min_iters
      continue
    fi
    echo finished, too small rel. improvement $rel_impr
    break
  fi

  # start learning-rate fade-out when improvement is low,
  if [ 1 == $(awk "BEGIN{print($rel_impr < $start_halving_impr ? 1:0);}") ]; then
    halving=1
    echo $halving >$dir/.halving
  fi

  # reduce the learning-rate,
  if [ 1 == $halving ]; then
    learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
    echo $learn_rate >$dir/.learn_rate
  fi
done

# select the best network,
if [ $mlp_best != $mlp_init ]; then
  mlp_final=${mlp_best}_final_
  ( cd $dir/nnet; ln -s $(basename $mlp_best) $(basename $mlp_final); )
  ( cd $dir; ln -s nnet/$(basename $mlp_final) final.nnet; )
  echo "$0: Succeeded training the Neural Network : '$dir/final.nnet'"
else
  echo "$0: Error training neural network..."
  exit 1
fi

