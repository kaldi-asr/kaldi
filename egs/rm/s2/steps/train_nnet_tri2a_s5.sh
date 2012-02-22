#!/bin/bash

# To be run from ..
if [ -f path.sh ]; then . path.sh; fi

dir=exp/nnet_tri2a_s5
mkdir -p $dir/{log,nnet}


#TRAIN_TOOL=nnet-train-xent-hardlab-perutt
TRAIN_TOOL="nnet-train-xent-hardlab-frmshuff --bunchsize=64 "



###### SELECT ALIGNMENTS ######
#choose directory with alignments
dir_ali=exp/tri2a
echo alignments: $dir_ali
#convert ali to pdf
ln -s $PWD/$dir_ali/cur.ali $dir
ali-to-pdf $dir_ali/final.mdl ark:$dir/cur.ali t,ark:$dir/cur.pdf
labels="ark:$dir/cur.pdf"
#count the class frames for division by prior
scripts/count_class_frames.awk $dir/cur.pdf $dir/cur.counts



###### SELECT FEATURES ######
cat data/train_fbank.scp | scripts/shuffle_list.pl ${seed:-777} > $dir/train.scp
head -n 3591 $dir/train.scp > $dir/train.scp.tr
tail -n 399 $dir/train.scp > $dir/train.scp.cv

delta=0

feats="ark:add-deltas --delta-order=$delta --print-args=false scp:$dir/train.scp ark:- |"
feats_tr="ark:add-deltas --delta-order=$delta --print-args=false scp:$dir/train.scp.tr ark:- |"
feats_cv="ark:add-deltas --delta-order=$delta --print-args=false scp:$dir/train.scp.cv ark:- |"


###### NORMALIZE FEATURES ######
#compute per-speaker CMVN
cmvn="ark:$dir/cmvn.ark"
compute-cmvn-stats --spk2utt=ark:data_prep/train.spk2utt "$feats" $cmvn
feats="$feats apply-cmvn --print-args=false --norm-vars=true --utt2spk=ark:data_prep/train.utt2spk $cmvn ark:- ark:- |"
feats_tr="$feats_tr apply-cmvn --print-args=false --norm-vars=true --utt2spk=ark:data_prep/train.utt2spk $cmvn ark:- ark:- |"
feats_cv="$feats_cv apply-cmvn --print-args=false --norm-vars=true --utt2spk=ark:data_prep/train.utt2spk $cmvn ark:- ark:- |"

###### ADD SPLICING ######

splice_lr=15

feats="$feats splice-feats --print-args=false --left-context=$splice_lr --right-context=$splice_lr ark:- ark:- |"
feats_tr="$feats_tr splice-feats --print-args=false --left-context=$splice_lr --right-context=$splice_lr ark:- ark:- |"
feats_cv="$feats_cv splice-feats --print-args=false --left-context=$splice_lr --right-context=$splice_lr ark:- ark:- |"


###### ADD THE HAMM DCT TRANSFORM ######

dct_basis=16
transf=$dir/hamm_dct.mat

scripts/gen_hamm_mat.py --fea-dim=23 --splice=$splice_lr > $dir/hamming.mat
scripts/gen_dct_mat.py --fea-dim=23 --splice=$splice_lr --dct-basis=$dct_basis > $dir/dct.mat
compose-transforms --binary=false $dir/dct.mat $dir/hamming.mat $transf 

#this is too slow for the GPU
#feats="$feats transform-feats --print-args=false $transf ark:- ark:- |"
#feats_tr="$feats_tr transform-feats --print-args=false $transf ark:- ark:- |"
#feats_cv="$feats_cv transform-feats --print-args=false $transf ark:- ark:- |"

{
  echo "<biasedlinearity> 368 713"
  cat $transf
  echo -n ' [ '
  for i in $(seq 368); do echo -n '0 '; done
  echo ']'
} > $transf.net
feats="$feats nnet-forward --print-args=false --silent=true $transf.net ark:- ark:- |"
feats_tr="$feats_tr nnet-forward --print-args=false --silent=true $transf.net ark:- ark:- |"
feats_cv="$feats_cv nnet-forward --print-args=false --silent=true $transf.net ark:- ark:- |"

###### RENORMALIZE THE FEAUTERES ######
cmvn_g="$dir/cmvn_glob.mat"
[ -r $cmvn_g ] || compute-cmvn-stats --binary=false "$feats" $cmvn_g
feats="$feats apply-cmvn --print-args=false --norm-vars=true $cmvn_g ark:- ark:- |"
feats_tr="$feats_tr apply-cmvn --print-args=false --norm-vars=true $cmvn_g ark:- ark:- |"
feats_cv="$feats_cv apply-cmvn --print-args=false --norm-vars=true $cmvn_g ark:- ark:- |"


###### INITIALIZE THE NNET ######
mlp_init=$dir/nnet.init
num_tgt=$(gmm-copy --binary=false $dir_ali/final.mdl - | grep NUMPDFS | awk '{ print $4 }')
scripts/gen_mlp_init.py --dim=368:577:${num_tgt} --gauss --negbias --seed=777 > $mlp_init



###### TRAIN ######
#global config for trainig
max_iters=20
start_halving_inc=0.5
end_halving_inc=0.1
lrate=0.008



$TRAIN_TOOL --cross-validate=true $mlp_init "$feats_cv" "$labels" &> $dir/log/prerun.log
if [ $? != 0 ]; then cat $dir/log/prerun.log; exit 1; fi
acc=$(cat $dir/log/prerun.log | grep FRAME_ACCURACY | tail -n 1 | cut -d' ' -f 3 | cut -d'%' -f 1)
echo CROSSVAL PRERUN ACCURACY $acc

mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*}
halving=0
for iter in $(seq -w $max_iters); do
  mlp_next=$dir/nnet/${mlp_base}_iter${iter}
  $TRAIN_TOOL --learn-rate=$lrate $mlp_best "$feats_tr" "$labels" $mlp_next &> $dir/log/iter$iter.log
  if [ $? != 0 ]; then cat $dir/log/iter$iter.log; exit 1; fi
  tr_acc=$(cat $dir/log/iter$iter.log | grep FRAME_ACCURACY | tail -n 1 | cut -d' ' -f 3 | cut -d'%' -f 1)
  echo TRAIN ITERATION $iter ACCURACY $tr_acc LRATE $lrate
  $TRAIN_TOOL --cross-validate=true $mlp_next "$feats_cv" "$labels" 1>>$dir/log/iter$iter.log 2>>$dir/log/iter$iter.log
  if [ $? != 0 ]; then cat $dir/log/iter$iter.log; exit 1; fi

  #accept or reject new parameters
  acc_new=$(cat $dir/log/iter$iter.log | grep FRAME_ACCURACY | tail -n 1 | cut -d' ' -f 3 | cut -d'%' -f 1)
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

