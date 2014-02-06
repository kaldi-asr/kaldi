#!/bin/bash

# Copyright 2012  Karel Vesely (Brno University of Technology)
# Apache 2.0

# Train neural network

# Begin configuration.

# training options
learn_rate=0.008
momentum=0
l1_penalty=0
l2_penalty=0
# data processing
bunch_size=256
cache_size=16384
seed=777
feature_transform=
# learn rate scheduling
max_iters=20
min_iters=
start_halving_inc=0.5
end_halving_inc=0.1
halving_factor=0.5
# misc.
verbose=1
# tool
train_tool="nnet-train-xent-hardlab-frmshuff"
 
# End configuration.

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh; 

. parse_options.sh || exit 1;

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

# Skip training
[ -e $dir/final.nnet ] && echo "'$dir/final.nnet' exists, skipping training" && exit 0

##############################
#start training

#choose mlp to start with
mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*}
#optionally resume training from the best epoch
[ -e $dir/.mlp_best ] && mlp_best=$(cat $dir/.mlp_best)
[ -e $dir/.learn_rate ] && learn_rate=$(cat $dir/.learn_rate)

#prerun cross-validation
$train_tool --cross-validate=true \
 --bunchsize=$bunch_size --cachesize=$cache_size --verbose=$verbose \
 ${feature_transform:+ --feature-transform=$feature_transform} \
 $mlp_best "$feats_cv" "$labels_cv" \
 2> $dir/log/prerun.log || exit 1;

acc=$(cat $dir/log/prerun.log | awk '/FRAME_ACCURACY/{ acc=$3; sub(/%/,"",acc); } END{print acc}')
xent=$(cat $dir/log/prerun.log | awk 'BEGIN{FS=":"} /err\/frm:/{ xent = $NF; } END{print xent}')
echo "CROSSVAL PRERUN ACCURACY $(printf "%.2f" $acc) (avg.xent$(printf "%.4f" $xent)), "

#resume lr-halving
halving=0
[ -e $dir/.halving ] && halving=$(cat $dir/.halving)
#training
for iter in $(seq -w $max_iters); do
  echo -n "ITERATION $iter: "
  mlp_next=$dir/nnet/${mlp_base}_iter${iter}
  
  #skip iteration if already done
  [ -e $dir/.done_iter$iter ] && echo -n "skipping... " && ls $mlp_next* && continue 
  
  #training
  $train_tool \
   --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
   --bunchsize=$bunch_size --cachesize=$cache_size --randomize=true --verbose=$verbose \
   ${feature_transform:+ --feature-transform=$feature_transform} \
   ${seed:+ --seed=$seed} \
   $mlp_best "$feats_tr" "$labels_tr" $mlp_next \
   2> $dir/log/iter$iter.log || exit 1; 

  tr_acc=$(cat $dir/log/iter$iter.log | awk '/FRAME_ACCURACY/{ acc=$3; sub(/%/,"",acc); } END{print acc}')
  tr_xent=$(cat $dir/log/iter$iter.log | awk 'BEGIN{FS=":"} /err\/frm:/{ xent = $NF; } END{print xent}')
  echo -n "TRAIN ACCURACY $(printf "%.2f" $tr_acc) (avg.xent$(printf "%.4f" $tr_xent),lrate$(printf "%.6g" $learn_rate)), "
  
  #cross-validation
  $train_tool --cross-validate=true \
   --bunchsize=$bunch_size --cachesize=$cache_size --verbose=$verbose \
   ${feature_transform:+ --feature-transform=$feature_transform} \
   $mlp_next "$feats_cv" "$labels_cv" \
   2>>$dir/log/iter$iter.log || exit 1;
  
  acc_new=$(cat $dir/log/iter$iter.log | awk '/FRAME_ACCURACY/{ acc=$3; sub(/%/,"",acc); } END{print acc}')
  xent_new=$(cat $dir/log/iter$iter.log | awk 'BEGIN{FS=":"} /err\/frm:/{ xent = $NF; } END{print xent}')
  echo -n "CROSSVAL ACCURACY $(printf "%.2f" $acc_new) (avg.xent$(printf "%.4f" $xent_new)), "

  #accept or reject new parameters (based no per-frame accuracy)
  acc_prev=$acc
  if [ "1" == "$(awk "BEGIN{print($acc_new>$acc);}")" ]; then
    acc=$acc_new
    mlp_best=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.2f" $tr_acc)_cv$(printf "%.2f" $acc_new)
    mv $mlp_next $mlp_best
    echo "nnet accepted ($(basename $mlp_best))"
    echo $mlp_best > $dir/.mlp_best 
  else
    mlp_reject=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.2f" $tr_acc)_cv$(printf "%.2f" $acc_new)_rejected
    mv $mlp_next $mlp_reject
    echo "nnet rejected ($(basename $mlp_reject))"
  fi

  #create .done file as a mark that iteration is over
  touch $dir/.done_iter$iter

  #stopping criterion
  if [[ "1" == "$halving" && "1" == "$(awk "BEGIN{print($acc < $acc_prev+$end_halving_inc)}")" ]]; then
    if [[ "$min_iters" != "" ]]; then
      if [ $min_iters -gt $iter ]; then
        echo we were supposed to finish, but we continue, min_iters : $min_iters
        continue
      fi
    fi
    echo finished, too small improvement $(awk "BEGIN{print($acc-$acc_prev)}")
    break
  fi

  #start annealing when improvement is low
  if [ "1" == "$(awk "BEGIN{print($acc < $acc_prev+$start_halving_inc)}")" ]; then
    halving=1
    echo $halving >$dir/.halving
  fi
  
  #do annealing
  if [ "1" == "$halving" ]; then
    learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
    echo $learn_rate >$dir/.learn_rate
  fi
done

#select the best network
if [ $mlp_best != $mlp_init ]; then 
  mlp_final=${mlp_best}_final_
  ( cd $dir/nnet; ln -s $(basename $mlp_best) $(basename $mlp_final); )
  ( cd $dir; ln -s nnet/$(basename $mlp_final) final.nnet; )
  echo "Succeeded training the Neural Network : $dir/final.nnet"
else
  "Error training neural network..."
  exit 1
fi




