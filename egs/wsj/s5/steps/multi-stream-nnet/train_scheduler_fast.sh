#!/bin/bash

# Copyright 2015  Brno University of Technology (author: Sri Harish Mallidi)
# Apache 2.0

cmd=run.pl

# Schedules epochs and controls learning rate during the neural network training

# training options,
learn_rate=0.008
momentum=0
l1_penalty=0
l2_penalty=0

# data processing,
train_tool="nnet-train-frmshuff"
train_tool_opts="--minibatch-size=256 --randomizer-size=32768"
feature_transform=

# multi-stream options
iters_per_epoch=1
stream_combination_pvals=

# learn rate scheduling,
max_iters=20
min_iters=0 # keep training, disable weight rejection, start learn-rate halving as usual,
keep_lr_iters=0 # fix learning rate for N initial epochs, disable weight rejection,
start_halving_impr=0.01
end_halving_impr=0.001
halving_factor=0.5

# misc,
seed=777
verbose=1
frame_weights=
utt_weights=
 
# End configuration.

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh; 

. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 7 ]; then
   echo "Usage: $0 <mlp-init> <feats-tr> <feats-cv> <labels-tr> <labels-cv> <stream-indices> <exp-dir>"
   echo " e.g.: $0 0.nnet scp:train.scp scp:cv.scp ark:labels_tr.ark ark:labels_cv.ark 0:84:168:252:336:420 exp/dnn1"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi

mlp_init=$1
feats_tr=$2
feats_cv=$3
labels_tr=$4
labels_cv=$5
stream_indices=$6
dir=$7

[ ! -d $dir ] && mkdir $dir
[ ! -d $dir/log ] && mkdir $dir/log
[ ! -d $dir/nnet ] && mkdir $dir/nnet

# Skip training
[ -e $dir/final.nnet ] && echo "'$dir/final.nnet' exists, skipping training" && exit 0

##############################
# number of stream combinations
num_streams=`echo $stream_indices | awk -F ":" '{print NF-1}'`
all_stream_combn=`echo 2^$num_streams -1|bc`
stream_combns=`(for ((n=1;n<=$all_stream_combn; n++)); do echo $n; done) | tr -s '\n' ':' | rev | cut -c 2- | rev`

echo "####################################"
echo "Multi-stream options are as follows:"
echo "####################################"

echo "stream_indices: $stream_indices"
echo "number of streams: $num_streams"

get_dim_from=$feature_transform
num_fea=$(feat-to-dim "$feats_tr nnet-forward \"$get_dim_from\" ark:- ark:- |" -)
##############################
# start training

# choose mlp to start with,
mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*}

# optionally resume training from the best epoch, using saved learning-rate,
[ -e $dir/.mlp_best ] && mlp_best=$(cat $dir/.mlp_best)
[ -e $dir/.learn_rate ] && learn_rate=$(cat $dir/.learn_rate)

if [ ! -e $dir/.done_iter00.initial.cv ]; then
  log=$dir/log/iter00.initial.comb.${all_stream_combn}.log; hostname>$log
  $train_tool --cross-validate=true --randomize=false --verbose=$verbose $train_tool_opts \
    ${frame_weights:+ "--frame-weights=$frame_weights"} \
    ${utt_weights:+ "--utt-weights=$utt_weights"} \
    "$feats_cv nnet-forward $feature_transform ark:- ark:- | apply-feature-stream-mask --cross-validate=true --stream-combination=$all_stream_combn $stream_indices ark:- ark:- |" "$labels_cv" $mlp_best \
  2>>$log || exit 1;
  touch $dir/.done_iter00.initial.cv
fi

# cross-validation on original network,
log=$dir/log/iter00.initial.comb.${all_stream_combn}.log;

loss=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
loss_type=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $5; }')

echo "CROSSVAL PRERUN AVG.LOSS $(printf "%.4f" $loss) $loss_type"

# resume lr-halving,
halving=0
[ -e $dir/.halving ] && halving=$(cat $dir/.halving)

# training,
for iter in $(seq -w $max_iters); do
  echo -n "ITERATION $iter: "
  mlp_next=$dir/nnet/${mlp_base}_iter${iter}
  
  # skip iteration (epoch) if already done,
  [ -e $dir/.done_iter$iter ] && echo "skipping... " && ls $mlp_next* && continue 
  
  # training,
  this_mlp_inp=$mlp_best
  for ((ii=1; ii<=$iters_per_epoch; ii++)); do 
    this_mlp_out=${mlp_next}_epoch${ii}

    if [ -e $dir/.done_iter${iter}_epoch${ii}.tr ]; then
      echo "    skipping... "
      echo "    $this_mlp_out exists..."
      this_mlp_inp=$this_mlp_out      
      continue
    fi
    this_seed=$(($seed+${iter#0}+$ii))

    multistream_mask_transf_str="<NnetProto>\n <MultiStreamMaskComponent> \
                                 <InputDim> $num_fea <OutputDim> $num_fea \
                                 <StreamIndices> ${stream_indices}\n</NnetProto>"
    feature_transform_with_multistream="echo -e \"$multistream_mask_transf_str\" | \
                                       nnet-initialize - -| nnet-concat ${dir}/final.feature_transform - -|"
    
    log=$dir/log/iter${iter}_epoch${ii}.tr.log; hostname>$log;
    $train_tool $train_tool_opts \
      --feature-transform="$feature_transform_with_multistream" \
      --cross-validate=false --randomize=true --verbose=$verbose \
      --randomizer-seed=$this_seed \
      --learn-rate=$learn_rate --momentum=$momentum \
      --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
      ${frame_weights:+ "--frame-weights=$frame_weights"} \
      ${utt_weights:+ "--utt-weights=$utt_weights"} \
      "$feats_tr" "$labels_tr" $this_mlp_inp $this_mlp_out \
      2>>$log || exit 1;
    touch $dir/.done_iter${iter}_epoch${ii}.tr
    this_mlp_inp=$this_mlp_out
  done
  (cd $dir/log/; ln -sf iter${iter}_epoch$((${ii}-1)).tr.log iter${iter}.tr.log; cd $OLDPWD)
  (cd $dir/nnet/; ln -sf $(basename $this_mlp_out) $(basename $mlp_next); cd $OLDPWD)
  tr_loss=$(cat $dir/log/iter${iter}.tr.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  echo -n "TRAIN AVG.LOSS $(printf "%.4f" $tr_loss), (lrate$(printf "%.6g" $learn_rate)), "
  
  # cross-validation,
  if [ ! -e $dir/.done_iter${iter}.cv ]; then
    log=$dir/log/iter${iter}.cv.comb.${all_stream_combn}.log; hostname>$log
    $train_tool --cross-validate=true --randomize=false --verbose=$verbose $train_tool_opts \
      ${frame_weights:+ "--frame-weights=$frame_weights"} \
      ${utt_weights:+ "--utt-weights=$utt_weights"} \
      "$feats_cv nnet-forward $feature_transform ark:- ark:- | apply-feature-stream-mask --cross-validate=true --stream-combination=${all_stream_combn} $stream_indices ark:- ark:- |" "$labels_cv" $mlp_next \
    2>>$log || exit 1
    touch $dir/.done_iter${iter}.cv
  fi

  log=$dir/log/iter${iter}.cv.comb.${all_stream_combn}.log;

  loss_new=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  loss_type=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $5; }')
  echo -n "CROSSVAL AVG.LOSS $(printf "%.4f" $loss_new), "

  # accept or reject?
  loss_prev=$loss
  if [ 1 == $(bc <<< "$loss_new < $loss") -o $iter -le $keep_lr_iters -o $iter -le $min_iters ]; then
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
  rel_impr=$(bc <<< "scale=10; ($loss_prev-$loss)/$loss_prev")
  if [ 1 == $halving -a 1 == $(bc <<< "$rel_impr < $end_halving_impr") ]; then
    if [ $iter -le $min_iters ]; then
      echo we were supposed to finish, but we continue as min_iters : $min_iters
      continue
    fi
    echo finished, too small rel. improvement $rel_impr
    break
  fi

  # start learning-rate fade-out when improvement is low,
  if [ 1 == $(bc <<< "$rel_impr < $start_halving_impr") ]; then
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
  echo "Succeeded training the Neural Network : $dir/final.nnet"
else
  "Error training neural network..."
  exit 1
fi

