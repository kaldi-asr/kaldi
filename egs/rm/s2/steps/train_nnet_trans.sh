#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation  Karel Vesely

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# To be run from ..
if [ -f path.sh ]; then . path.sh; fi

dir=$PWD/exp/nnet_trans
mkdir -p $dir/{log,nnet}

#use following features and alignments
cp exp/mono/train.scp exp/mono/cur.ali $dir
head -n 800 $dir/train.scp > $dir/train.scp.tr
tail -n 200 $dir/train.scp > $dir/train.scp.cv
feats="ark:add-deltas --print-args=false scp:$dir/train.scp ark:- |"
feats_tr="ark:add-deltas --print-args=false scp:$dir/train.scp.tr ark:- |"
feats_cv="ark:add-deltas --print-args=false scp:$dir/train.scp.cv ark:- |"
labels="ark:$dir/cur.ali"

#compute per utterance CMVN
cmvn="ark:$dir/cmvn.ark"
compute-cmvn-stats "$feats" $cmvn
feats_tr="$feats_tr apply-cmvn --print-args=false --norm-vars=true $cmvn ark:- ark:- |"
feats_cv="$feats_cv apply-cmvn --print-args=false --norm-vars=true $cmvn ark:- ark:- |"


#initialize the nnet
mlp_init=$dir/nnet.init
scripts/gen_mlp_init.py --dim=39:512:301 --gauss --negbias > $mlp_init

#global config for trainig
max_iters=20
start_halving_inc=0.5
end_halving_inc=0.1
lrate=0.001



nnet-train-xent-hardlab-perutt --cross-validate=true $mlp_init "$feats_cv" "$labels" &> $dir/log/prerun.log
if [ $? != 0 ]; then cat $dir/log/prerun.log; exit 1; fi
acc=$(cat $dir/log/prerun.log | grep Xent | tail -n 1 | cut -d'[' -f 2 | cut -d'%' -f 1)
echo CROSSVAL PRERUN ACCURACY $acc

mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*}
halving=0
for iter in $(seq -w $max_iters); do
  mlp_next=$dir/nnet/${mlp_base}_iter${iter}
  nnet-train-xent-hardlab-perutt --learn-rate=$lrate $mlp_best "$feats_tr" "$labels" $mlp_next &> $dir/log/iter$iter.log
  if [ $? != 0 ]; then cat $dir/log/iter$iter.log; exit 1; fi
  tr_acc=$(cat $dir/log/iter$iter.log | grep Xent | tail -n 1 | cut -d'[' -f 2 | cut -d'%' -f 1)
  echo TRAIN ITERATION $iter ACCURACY $tr_acc LRATE $lrate
  nnet-train-xent-hardlab-perutt --cross-validate=true $mlp_next "$feats_cv" "$labels" 1>>$dir/log/iter$iter.log 2>>$dir/log/iter$iter.log
  if [ $? != 0 ]; then cat $dir/log/iter$iter.log; exit 1; fi

  #accept or reject new parameters
  acc_new=$(cat $dir/log/iter$iter.log | grep Xent | tail -n 1 | cut -d'[' -f 2 | cut -d'%' -f 1)
  echo CROSSVAL ITERATION $iter ACCURACY $acc_new
  acc_prev=$acc
  if [ 1 == $(awk 'BEGIN{print('$acc_new' > '$acc')}') ]; then
    acc=$acc_new
    mlp_best=$dir/nnet/$mlp_base.iter${iter}_tr$(printf "%.5g" $tr_acc)_cv$(printf "%.5g" $acc_new)
    mv $mlp_next $mlp_best
    echo nnet $mlp_best accepted
  else
    mlp_reject=$dir/nnet/$mlp_base.iter${iter}_tr$(printf "%.5g" $tr_acc)_cv$(printf "%.5g" $acc_new)
    mv $mlp_next $mlp_reject
    echo nnet $mlp_reject rejected 
  fi

  #stopping criterion
  if [[ 1 == $halving && 1 == $(awk 'BEGIN{print('$acc' < '$acc_prev'+'$end_halving_inc')}') ]]; then
    echo finished, too small improvement $(awk 'BEGIN{print('$acc'-'$acc_prev')}')
    break
  fi

  #start annealing when improvement is low
  if [ 1 == $(awk 'BEGIN{print('$acc' < '$acc_prev'+'$start_halving_inc')}') ]; then
    halving=1
  fi
  
  #do annealing
  if [ 1 == $halving ]; then
    lrate=$(awk 'BEGIN{print('$lrate'*0.5)}')
  fi
done

if [ $mlp_best != $mlp_init ]; then 
  iter=$(echo $mlp_best | sed 's/^.*iter\([0-9][0-9]*\).*$/\1/')
fi
mlp_final=$dir/${mlp_base}_final_iter${iter:-0}_acc${acc}
cp $mlp_best $mlp_final
ln -s $mlp_final $dir/${mlp_base}_final

echo final network $mlp_final

