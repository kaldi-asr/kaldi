#!/bin/bash

# To be run from ..
if [ -f path.sh ]; then . path.sh; fi

dir=exp/nnet_tri2a_s2
mkdir -p $dir/{log,nnet}



###### SELECT FEATURES ######
cat data/train.scp | scripts/shuffle_list.pl > $dir/train.scp
head -n 3591 $dir/train.scp > $dir/train.scp.tr
tail -n 399 $dir/train.scp > $dir/train.scp.cv
feats="ark:add-deltas --print-args=false scp:$dir/train.scp ark:- |"
feats_tr="ark:add-deltas --print-args=false scp:$dir/train.scp.tr ark:- |"
feats_cv="ark:add-deltas --print-args=false scp:$dir/train.scp.cv ark:- |"

###### SELECT ALIGNMENTS ######
#choose directory with alignements
dir_ali=exp/tri2a
echo alignments: $dir_ali
#convert ali to pdf
ln -s $PWD/$dir_ali/cur.ali $dir
ali-to-pdf $dir_ali/final.mdl ark:$dir/cur.ali t,ark:$dir/cur.pdf
labels="ark:$dir/cur.pdf"
#count the class frames for division by prior
scripts/count_class_frames.awk $dir/cur.pdf $dir/cur.counts



###### NORMALIZE FEATURES ######
#compute per-utterance CMN
cmn="ark:$dir/cmn.ark"
compute-cmvn-stats "$feats" $cmn
feats="$feats apply-cmvn --print-args=false --norm-vars=false $cmn ark:- ark:- |"
feats_tr="$feats_tr apply-cmvn --print-args=false --norm-vars=false $cmn ark:- ark:- |"
feats_cv="$feats_cv apply-cmvn --print-args=false --norm-vars=false $cmn ark:- ark:- |"

#compute global CVN
cvn="ark:$dir/cvn.ark"
gcvn_spk2utt=$dir/globalcvn.spk2utt
{ echo -n "global "
  cat $dir/train.scp | cut -d " " -f 1 | tr '\n' ' '
} > $gcvn_spk2utt
compute-cmvn-stats --spk2utt=ark:${gcvn_spk2utt} "$feats" $cvn 

#add global CVN to feature extration
gcvn_utt2spk=$dir/globalcvn.utt2spk
cat $dir/train.scp | cut -d " " -f 1 | awk '{ print $0" global";}' > $gcvn_utt2spk
gcvn_utt2spk_opt="--utt2spk=ark:$gcvn_utt2spk"

feats="$feats apply-cmvn --print-args=false $gcvn_utt2spk_opt --norm-vars=true $cvn ark:- ark:- |"
feats_tr="$feats_tr apply-cmvn --print-args=false $gcvn_utt2spk_opt --norm-vars=true $cvn ark:- ark:- |"
feats_cv="$feats_cv apply-cmvn --print-args=false $gcvn_utt2spk_opt --norm-vars=true $cvn ark:- ark:- |"



###### INITIALIZE THE NNET ######
mlp_init=$dir/nnet.init
num_tgt=$(grep NUMPDFS $dir_ali/final.mdl | awk '{ print $4 }')
scripts/gen_mlp_init.py --dim=39:1024:${num_tgt} --gauss --negbias > $mlp_init



###### TRAIN ######
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
ln -s $PWD/$mlp_best $mlp_final

if [ -e $dir/final.nnet ]; then
  unlink $dir/final.nnet
fi
ln -s $PWD/$mlp_final $dir/final.nnet

echo final network $mlp_final

